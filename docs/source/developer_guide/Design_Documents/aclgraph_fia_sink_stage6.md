# FIA sink stage 6: producer-inclusive native policy comparison

Status: the bounded formal experiment is in progress. This document records
the mechanism, protocol and interpretation rules before the completed
evidence is analyzed. It does not report or predict a performance result.

## Question and comparison boundary

Stage four found an attention-execution opportunity for assigning complete
rows to cores, while the CPU plan construction and metadata transfer consumed
the observed saving. Stage five implemented that policy inside the native
AICPU metadata producer and qualified its replay behavior. Stage six asks
whether an executable native producer can expose a net benefit after all
required preparation, planning, replay and synchronization costs are included.

The two policies use the same qualified D256 attention consumer, physical
default buckets, resident private Q/K/V, causal mask and metadata ABI:

| Policy | Native plan producer |
| --- | --- |
| `upstream` | The pinned producer's original `SplitCore` policy, including its FD splitting and reduction schedule. |
| `whole` | The guarded stage-five producer, using native per-head costs and contiguous complete-row assignment with no FD reduction. |

Both controls capture one shared metadata producer followed by 16 sequential
attention occurrences inside one ACLGraph. They do not split the full model
or execute attention on multiple concurrent streams. There is no legacy FIA,
request-graph splitting, separate padding algorithm, or full-model control
in this experiment. A difference between these policies cannot establish
superiority over padding, split-parallel acceleration, or end-to-end speedup.

The native whole-row policy changes both work assignment and FD removal.
Its comparison with the upstream plan therefore measures their combined
effect, including any producer-cost difference. It does not isolate the
duration of the FD merge alone.

## Mechanism and artifact identity

For Q4 with 12 query heads and two KV heads, a complete
`(physical request, KV head)` contains 24 query/group positions and fits in
the existing `mBase=512` tile. The whole-row producer keeps all KV traversal
for that row on one core. Canonical endpoints, zero partial-result fields
and a zero FD count allow the unchanged consumer to write its final output
directly. The producer includes every physical dummy row in its final
endpoint and retains real zero KV lengths.

The implementation, guard, fallback and transactional publication are
documented in `aclgraph_fia_sink_stage5.md`. This stage does not introduce
an incremental plan cache, remove exact lengths, select batch-invariant
arithmetic, or add a new attention kernel. Floating-point reduction order
can differ between whole-row online softmax and FD partial merging, so
correctness is checked with the qualified FP32 reference tolerance.

Each fresh process verifies the configured AICPU dispatch name and its exact
library hash before loading Torch/native operators. The shared consumer
OpAPI hash is pinned independently:

| Artifact | SHA256 |
| --- | --- |
| Upstream AICPU library `libfia_v2_sink_aicpu_kernels.so` | `796983827bc6326a250009a2aacc810c44f65545ec2ae56dab517a5ec4e7111b` |
| Reviewed whole-row AICPU library `libthesis_stage5_whole_aicpu.so` | `1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90` |
| Common stage-two OpAPI | `2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40` |

The reports retain every experiment script hash, extension hash, copied
vendor inventory, loaded-symbol provenance and library mappings. A successful
compiler exit or zero base FD count is not sufficient policy evidence:
the whole-row control validates the complete 1024-word device metadata,
including endpoints and per-core FD arrays.

## Predeclared bounded protocol

The canonical trace digest remains
`0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`.
The runner selects the first original trace record at each default physical
token capacity 256, 400, 496 and 512. It verifies the archived default capture
catalog against the installed model configuration. No capture size or
request size is added for this policy. Model configuration is inspected;
model weights are not loaded.

The formal suite uses four process rounds with five repeats per selected
record and policy:

| Round | Fresh-process order |
| --- | --- |
| 0 | `upstream`, then `whole` |
| 1 | `whole`, then `upstream` |
| 2 | `upstream`, then `whole` |
| 3 | `whole`, then `upstream` |

Every process owns one policy and exits before the next policy starts.
Distinct vendor directories and new interpreters prevent switching a cached
native producer inside a live process. Device zero is selected explicitly,
`TASK_QUEUE_ENABLE=0` applies to both policies, and batch-invariant mode is
disabled. Parent launchers forward termination to their owned child process
groups and wait for cleanup before returning.

Each formal process owns 16 separate Q/K/V occurrences and captures four
graphs, one per selected bucket. Bucket graphs have independent graph pools,
so the replay rotation does not depend on a shared-pool capture-order
assumption. Q/K/V remain resident, and each bucket has independent exact-size
length/page-table storage and fixed-address outputs and metadata.

After capture and initial correctness qualification, each selected record
runs one complete warmup. Timed repeats rotate the record order by
`(round + repeat) % 4`. The formal design contains eight fresh-process reports,
20 timed observations per process, 160 observations in total, and 80 matched
`(round, capacity, repeat)` policy pairs. Repeats are nested within processes;
80 pairs are not 80 independent process trials.

The `--screen` gate is separate: one occurrence, capacity400, one process
round and one repeat. Screen evidence is not merged into the formal suite.

## Timed endpoint

`wall_us` starts immediately before the current record fills its persistent
pinned descriptors. It ends after the graph and final event join complete.
It includes host descriptor fill, required KV-length/page-table H2D,
the one native AICPU producer, all 16 private attention executions, graph
submission and the final synchronization. Fixed cumulative Q4 endpoints are
uploaded during setup in both policies and remain invariant for a bucket.

| Field | Meaning |
| --- | --- |
| `wall_us` | Complete declared attention-chain endpoint, including preparation and final join. |
| `preparation_enqueue_us` | CPU fill/enqueue interval; it does not measure completed H2D duration. |
| `submit_join_cpu_us` | Host submission interval, which can overlap device work. |
| `queue_span_us` | Events around replay on its stream, including submission gaps and dependency waits. |

The queue span is neither pure kernel time nor a complete independent H2D
measurement. Host submission and queue time overlap and must not be added
to reconstruct a latency. Tensor hashing, FP32 references, metadata readback,
qualification and report writes are outside the timed endpoint.

The workload excludes upstream QKV projection, KV writes, other model layers,
sampler, scheduler and request queues. It uses synthetic resident tensors
with real canonical request ownership, lengths and pages; it is not a
measurement of online request generation or continuous model state.

## Correctness and input matching

Before attention warmup/capture for every bucket, the process runs the native
metadata producer separately, preserves all 1024 words and their hash, and
checks the expected resource limits. The whole-row policy additionally
requires canonical advancing complete-head endpoints, complete physical
coverage, zero FD arrays and zero reduction counts. Unsupported fallback to
the upstream FD policy fails whole-row qualification rather than silently
changing the measured algorithm.

Qualification runs for every selected bucket before and after timing:

- Every private output must be finite and every physical KV0 padding output
  must be written as exact zero, without trimming or external clearing.
- Every occurrence is compared with an independent FP32 attention reference
  at live request rows zero, middle and last: 48 checks per bucket in a
  16-occurrence run. The established tolerance is `rtol=0.03`, `atol=0.003`.
- Query contents, exact Q/KV lengths, page routing and mask remain unchanged.
  Input, output and metadata addresses are compared with capture-time
  descriptors, including their shapes and strides.
- Complete input-byte hashes cover every private Q/K/V tensor before capture
  and after all timing/qualification. CPU hashing uses bounded chunks and
  includes every byte. The suite analysis matches these identities across
  independent policy processes; a common random seed alone is insufficient.
- Q/K/V storage intervals must be disjoint across private occurrences.
  Sharing the same occurrence's KV storage across bucket graphs is allowed.
- Pre/post complete output hashes are retained as repeatability diagnostics.
  Equal hashes are not required between policies, and unequal hashes alone
  do not imply a correctness failure; numerical acceptance uses FP32 tolerance.

The FP32 checks sample three rows per occurrence and bucket. They do not
qualify every output element, every possible length, the complete 128-step
trajectory, or a complete language model. The separate stage-five dynamic
and all-zero gates retain their own narrower correctness scopes.

## Reproduction and evidence

Use the frozen experiment scripts and their adjacent helper modules inside
the isolated experiment tree. Keep the source/vendor trees and earlier
results intact. Every suite or probe invocation requires a new output path.
The following is an example with a new run name, after sourcing the existing
Ascend profile:

```bash
experiment_root=/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e
tools_dir="$experiment_root/stage6-tools-FROZEN_RUN"

python "$tools_dir/aclgraph_fia_sink_stage6_suite.py" \
  --root "$experiment_root" \
  --upstream-vendor "$experiment_root/opp/packages/vendors/thesis_fia_stage2_transformer" \
  --whole-vendor "$experiment_root/stage5-no-fd/relinked-v3/opp/packages/vendors/materialized-baseline-v3" \
  --output-dir "$experiment_root/stage6-suite-new-run" \
  --trace "$tools_dir/structural-dynamic-trace-v1.json" \
  --selection "$tools_dir/q4-tile-full-replay-v8-queue0.rank0.json" \
  --rounds 4 --repeats 5 --seed 1703 --timeout 900
```

`FROZEN_RUN` denotes the already deployed tools directory whose hashes are
recorded in the evidence. Supply that existing directory, rather than
creating an unversioned helper set. Add `--screen` to perform only the
separate safety screen.

The suite invokes `aclgraph_fia_sink_stage6_probe.py` once per process with
`--policy upstream|whole`, `--trial ROUND` and the selected vendor. Its raw
report name is `stage6-producer-probe.json`. A direct probe uses the same
`--root`, `--opp-root`, `--output-dir`, `--trace`, `--selection`, `--seed` and
`--timeout` arguments, with optional `--screen`; formal mode uses 16
occurrences. The suite preserves each report's byte hash, process identity,
launch/finish time and relative path.

After copying the complete suite directory, retain its relative raw report
paths. Analyze it with the matching local sources and original trace evidence:

```bash
python benchmarks/aclgraph_fia_sink_stage6_analyze.py \
  --input benchmarks/results/multi_graph_structural/fia-sink-stage6-v1/stage6-suite.json \
  --output benchmarks/results/multi_graph_structural/fia-sink-stage6-analysis.json
```

The analyzer checks process order, sample completeness, pairing, script and
native provenance, canonical schedule, input identity, pre/post qualification
and timed endpoints before reporting an accepted comparison. It retains all
observations, including outliers; reductions use the difference of matched
aggregate times, not an average of per-sample percentage ratios. AB/BA order
reduces order imbalance but cannot eliminate between-process drift.

Expected repository evidence locations are:

- `benchmarks/results/multi_graph_structural/fia-sink-stage6-v1/stage6-suite.json`
  with all eight relative raw process reports and logs.
- `benchmarks/results/multi_graph_structural/fia-sink-stage6-analysis.json`
  for independent protocol qualification and paired statistics.

## Results and next gate

Completed formal evidence and its qualified analysis will supply this
section. No performance numbers are inferred from an in-progress process,
screen run, CPU cost proxy or stage-four oracle.

Regardless of the outcome, the next performance gate is the unchanged
128-step dynamic trajectory, with its default capture catalog and true
per-step metadata. A favorable four-record result does not complete that
gate. Full-model replay, a padding comparison and serving-level measurements
remain separate later stages with their own correctness and attribution
requirements.
