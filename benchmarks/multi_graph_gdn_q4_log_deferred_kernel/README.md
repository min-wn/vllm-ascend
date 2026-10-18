# FP32 Q4 compact log with deferred accepted-state commit

This isolated benchmark extension separates the existing compact-log solve
from its dense accepted-prefix commit. Its purpose is to test whether the
pure AIV commit can overlap subsequent output-projection or MLP Cube work.
It is an optional structural experiment, with no claimed performance gain
or production cache integration.

```python
extension.solve(products, query, key, value, beta, gate, base,
                sequence, ids, accepted, output,
                old_logs, old_valid, new_logs, new_valid, cores=40,
                request_start=0, request_stop=-1)
extension.commit(base, sequence, ids, accepted, old_logs, old_valid,
                 new_logs, new_valid, cores=40,
                 request_start=0, request_stop=-1)
extension.export_candidates(base, old_logs, old_valid, candidates, cores=40)
```

The installed module is named `gdn_q4_log_deferred`. It does not replace
`gdn_q4_log`. Build it into a fresh directory after every source change:

```bash
python benchmarks/multi_graph_gdn_q4_log_deferred_kernel/load.py \
  --build-dir /workspace/multi-graph-results/q4-log-deferred-build-v1
```

## Phase contract

Before solve, the pure Cube products kernel computes initial products from
the current dense base. Solve reads these products and the old log bank,
corrects their projections, solves the four-token recurrence, and writes
attention outputs plus the new log bank. It never writes the dense base or
the old log/validity bank.

Commit applies the previous accepted prefix to the dense base using the old
bank, then publishes only each active head's new record and validity into the
canonical old bank. The new bank remains read-only. It uses the same `ApplyOldLog` method
as the exporter, including the identical FP32 multiply/FMA instructions and
their ordering. After commit, the base plus the canonical old bank represents
all four current candidate states. Export must run after commit completion.

Each generation needs exactly one solve and exactly one commit for each
active request. Commit is not idempotent; executing it twice applies the
current log as if it were the previous generation. No generation can start
until the preceding commit has finished. Public state changes are not
atomically published by the native extension. The caller must join all
commits before exposing a complete generation to another consumer. The two
bank pointers stay fixed across graph replays; there is no pointer swap.

Required stream dependencies are:

1. Finish all base-product reads before any commit can write that base.
2. Finish the corresponding solve before consuming attention output and
   before commit can publish over its old log bank or reuse scratch.
3. Allow output projection/MLP on the compute stream to overlap commit on a
   separate stream. Commit has no dependence on those dense outputs.
4. Join the commit stream before exporting candidates, exposing canonical
   state, starting the next generation, or recording the complete endpoint.

Same-stream calls satisfy these dependencies sequentially. Cross-stream
calls require explicit events/waits. Omitting commit from complete-endpoint
timing is invalid. Native entry points launch on the current NPU stream
and do not add host synchronizations or cross-stream dependencies.

## Layout and ownership

Base is FP32 `[Nrequest,24,128,128]`; its inner dimensions are contiguous
and optional page padding is retained. Each old/new log bank is contiguous
FP32 `[Nrequest,24,1056]`: four 128-element keys, four 128-element updates,
and four eight-float padded decay scalars. Validity is contiguous int32
`[Nrequest,24,8]`. Each head owns a complete 32-byte validity record to avoid
losing neighboring scalar stores on A2. Old and new bank address ranges must
be disjoint; disjoint views in one larger allocation are permitted. Mutable
outputs cannot overlap their read inputs.

Q/K are BF16 `[4B,8,128]`, V/output are BF16 `[4B,24,128]`, beta is BF16
`[4B,24]`, gate is FP32 `[4B,24]`, and products are FP32 `[B,24,128,8]`.
`sequence` is int32 `[B+1]` with the initial token offset followed by per-row
counts. Counts are zero or four. `accepted` is int32 `[B]`, and the four
int32 `ids` for an active request repeat its unique nonzero base slot.
Slot zero is null. As in the original prototype, malformed rows are skipped
and metadata validity/unique-slot ownership remain caller preconditions.

Both solve and commit accept the same optional half-open request range.
Range partitions must be disjoint and cover each active request exactly
once. Request slots must be distinct across ranges. `request_stop=-1`
selects the full fixed-capacity batch, up to 128 requests.

Inactive/null rows are untouched in both phases. Because only active records
are published into the canonical old bank, paused requests retain their
previous state without copying either whole log bank. Inactive records in
the scratch new bank are unspecified and must never be exported or promoted
directly. Bank-management costs and synchronization must be included in a
complete benchmark. Output zeroing is a caller responsibility.
Metadata and banks must remain alive and unchanged until their dependent
kernels finish; neither retries nor reassignment are integrated here.

## Scratch lifetime and verification

The fixed scratch allocation is 185472 bytes (181.125 KiB), below the
192 KiB A2 UB capacity. This first experiment retains the original buffers
in each phase; it does not claim an optimized commit-only scratch footprint.
The two-deep state-output queue preserves candidate DMA lifetimes.
`MTE3_MTE2` waits prevent scratch reuse while a previous copy-out or canonical
log publication still reads it. A `V_MTE2` wait also protects the old-log
scratch before loading the new record for publication and between tasks.

This directory only supplies native kernels and a loader. Qualification must
verify solve leaves base/old banks bitwise unchanged, commit matches the
previous exported accepted prefix exactly, published-bank export matches the
sequential compact implementation, and all checks hold across continued
dynamic steps, range partitions, inactive/null slots and page padding.
Multi-stream full-endpoint timing must include the commit join. Local
overlap or a shorter solve alone does not establish a split-versus-padding
improvement.
