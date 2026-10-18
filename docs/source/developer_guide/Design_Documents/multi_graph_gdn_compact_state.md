# Compact Q4 GDN state: arithmetic and device qualification

## Decision and scope

The fused V2 device prototype preserves the scoped recurrence's outputs and all
four logical speculative states exactly, including mixed per-request accepted
counts and dynamic graph replay. It closes V1's copied-kernel tiling gap, but
the complete compact step remains 8.20%, 14.62% and 28.00% slower than the
installed recurrence for accepted1/2/4. Stop this representation as a latency
optimization route for the current workload. Do not integrate it into the full
model or present it as a split speedup. Its storage reduction remains a separate
property, without a demonstrated complete-model memory or latency benefit.

This work targets actual state traffic rather than another stream-order change.
Q4 recurrence currently stores four complete BF16 candidate matrices per
request and head. The proposal retains one BF16 base and the actual FP32 decay,
key and rank-one update operands for four tokens. A later accepted prefix can
reconstruct its candidate by replaying those updates in order. This can reduce
state storage and traffic, but adds work to the next step.

The representation is also available to an optimized padding implementation.
It is a general GDN operator change. Its effect on request-split parallelism
must eventually be measured against padding using the same state improvement.
Reducing bytes does not establish that the current kernel is bandwidth-bound.

## Arithmetic and state contract

For each token, the scoped installed kernel performs these operations:

```text
gamma = exp(gate)                         FP32
decayed = state * gamma                   FP32 multiplication
prediction = reduce(decayed * key)        installed reduction order
update = (value - prediction) * beta      FP32
state = FMA(update, key, decayed)          FP32 rank-one update
candidate[token] = BF16(state)
```

The Q4 loop retains FP32 state between tokens. Writing a BF16 candidate does
not feed that rounded candidate back into the next token. At the next decode
step, the selected candidate is loaded from BF16 into FP32.

The compact log therefore stores the actual exponentiated `gamma`, normalized
key and computed `update` as FP32. Reconstruction applies each gamma multiply
and FMA sequentially, then rounds to BF16 before the next Q4 step loads it.
Combining gamma products or evaluating one accumulated low-rank matrix update
would change floating-point operation order and is outside this qualification.

V1 uses separate reconstruction and recurrence kernels.
Reconstruction updates the BF16 base in place; the following recurrence reads
that base and emits new logs. It does not materialize four dense candidates in
the timed hot path. Any consumer requiring those candidates must pay the export
cost. Exporting four matrices is not a free cache view.

V2 fuses accepted-prefix reconstruction, the required BF16 step-boundary
rounding, one base write and the next Q4 recurrence in one kernel. It retains
the reconstructed matrix in UB rather than rereading it from global memory.
Old logs are loaded once per head before the same log bank is overwritten.
Queued log output batches the four records per V tile. These changes remove
V1's separate reconstruction launch, extra base round trip and per-token log
copy barriers; their measured result is recorded below.

Null row zero and inactive requests retain their base and logs. A paused request
can resume using its preserved log. The live operation skips inactive requests;
correctness export deliberately reconstructs every request, including inactive
ones, to compare all four logical candidates against dense storage.

## Reproducible CPU oracle

`benchmarks/multi_graph_gdn_compact_reference.py` provides:

- `dense_step`: four BF16 candidates and FP32 outputs.
- `emit_compact_step`: a BF16 base, FP32 gamma/key/update log and FP32 outputs.
- `restore_accepted`: a scalar or per-request prefix in 1-4, restored as BF16.
- `export_logical_states`: all four BF16 candidates with real reconstruction work.
- `qualify_closed_loop`: mixed prefixes, request reordering, pauses and resumption.

Its `CompactState` contains base `[B,H,V,K]`, gamma `[B,4,H]`, key `[B,4,H,K]`,
update `[B,4,H,V]` and a per-request validity flag. Log values are not rounded
to BF16. Returned storage does not alias source buffers.

The CPU oracle uses CPU reduction and an explicit FP32 FMA simulation via FP64
intermediates. It proves equivalence under that arithmetic; it does not certify
the device's `ExpMasked`, AddFold reduction or `MulAddDst`. The device probe
separately checks against the installed operator.

Focused CPU tests include a hand-computed scalar example, 128-dimensional
states, all four candidates, mixed accepted1/2/4 across continued steps,
stable request identity, inactive/null preservation, aliases, and rejection of
malformed or nonfinite state/log inputs. The earlier external research checked
56,623,104 candidate elements across three seeds and 16 steps. The repository
tests use smaller cases rather than repeating that entire experiment.

## Device v1 result

`gdn-compact-v1.json` records one 910B3 device in the existing CANN 9.1 / old
driver environment. This is a single-rank recurrence qualification, with the
TP2-local head dimensions: 8 key heads, 24 value heads, K=V=128, BF16 Q/K/V,
FP32 scalar gates and Q4. It does not execute a TP2 model or FIA.

The capacity is 100 requests / 400 tokens, with 98 live requests during timing.
State shapes include a null row:

| Buffer | Shape |
| --- | --- |
| Dense candidates | `[401,24,128,128]` BF16 |
| Compact base | `[101,24,128,128]` BF16 |
| Decay log | `[101,4,24,8]` FP32 |
| Key and update logs, each | `[101,4,24,128]` FP32 |

Decay uses an aligned eight-float record. Keys are replicated per value head
in this first prototype, even though the model uses grouped key/value heads.
Total allocated state/log bytes are 315,359,232 for dense and 89,668,608 for
compact base plus logs, a 71.57% storage reduction. Allocated capacity bytes
are distinct from active-request traffic below.

The device run records 20 completed checks: eight eager continued-state steps
with live batches 98/97/99/98 repeated and accepted1/2/4, followed by four
dynamic graph steps for each of three implementations. Changed inputs,
eager/replay outputs, complete state/log contents, all four candidates,
null/inactive preservation and finite values pass the exact comparisons.
Accepted counts are uniform across requests within each step; arbitrary
mixed per-request counts and request permutations are not established by v1.

There are 324 timing samples: three implementations, three accepted counts,
three rounds and 12 samples per round. Mode order reverses on alternate rounds;
each mode starts from the same saved state/log contents outside timing. Reported
values are the median of the three per-round medians:

| Accepted | Installed official device, us | Copied dense device, us | Compact device, us | Compact vs official |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1002.51 | 1467.62 | 1949.37 | 94.45% slower |
| 2 | 1002.19 | 1467.24 | 2119.45 | 111.48% slower |
| 4 | 1001.92 | 1467.30 | 2446.12 | 144.14% slower |

| Accepted | Installed official wall, us | Copied dense wall, us | Compact wall, us |
| --- | ---: | ---: | ---: |
| 1 | 1113.68 | 1583.95 | 2062.12 |
| 2 | 1114.37 | 1580.84 | 2230.93 |
| 4 | 1118.04 | 1583.32 | 2560.70 |

The three compact device round medians are 1949.07/1952.54/1949.37 us for
accepted1, 2119.45/2119.55/2116.15 us for accepted2, and
2450.12/2446.12/2430.49 us for accepted4. The negative result is present in
each round. Whole-sample medians differ slightly from the round-first statistic;
neither should be silently substituted when reproducing these tables.

Copied dense is approximately 46.4% slower than the installed kernel despite
exact arithmetic checks. Its explicit V tile, buffering and manual tiling
configuration are not performance-equivalent to the installed host dispatcher.
Both controls are needed: copied dense isolates the log-output modification;
the installed kernel is the fair performance baseline. Compact is also slower
than copied dense by 32.83%, 44.45% and 66.71% for accepted1/2/4.

## Device v1 kernel breakdown

The subsequent `gdn-compact-v1-breakdown.json` retains the complete compact
endpoint and adds isolated log-emission and reconstruction diagnostics. It
again passes 20 qualification checks and records 540 timing samples, with
the same three-round / 12-sample convention. Median of round medians:

| Accepted | Official device, us | Copied dense device, us | Emit only device, us | Reconstruct only device, us | Complete compact device, us |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1001.39 | 1466.07 | 1654.35 | 307.12 | 1952.18 |
| 2 | 1001.16 | 1467.03 | 1658.62 | 477.11 | 2121.59 |
| 4 | 1001.67 | 1466.27 | 1652.00 | 810.62 | 2456.11 |

| Accepted | Emit only wall, us | Reconstruct only wall, us | Complete compact wall, us |
| --- | ---: | ---: | ---: |
| 1 | 1772.06 | 418.81 | 2065.46 |
| 2 | 1771.29 | 586.97 | 2235.88 |
| 4 | 1766.00 | 920.00 | 2565.42 |

The emit-only path does not reconstruct the previous accepted prefix, so it
is not a valid replacement step endpoint. The reconstruction-only path does
not compute current outputs or the new update log. Their separate timings
must not be added and presented as an observed closed-loop latency; the
complete compact column measures that endpoint directly.

Log emission alone is approximately 186-192 us slower than copied dense,
despite removing four dense candidate writes. The source adds explicit
Vector-to-copy and copy-to-Vector synchronization while reusing update/key
buffers. This identifies log export and its synchronization as costs that
require optimization; it does not isolate a hardware cause without further
trace evidence. Reconstruction also grows from about 307 to 811 us as the
selected prefix grows. These results reject the assumption that reducing
logical bytes automatically reduces kernel latency. Fusing away the extra
launch and base read is insufficient unless the export and reconstruction
execution costs are also addressed.

## Device V2 fused result and route decision

`gdn-compact-v2-fused.json` records the fused implementation on the same
single-NPU 910B3 configuration and 100-request capacity, timing 98 active
requests. It is a recurrence qualification using the TP2-local head layout,
not a TP2 decoder measurement. The probe uses `--fused --mixed --diagnostics`
with 12 samples in each of three alternating-order rounds.

All 20 qualification checks pass exactly: eight continued eager steps and four
dynamic graph steps per implementation. Changed inputs, mixed per-request
accepted counts, active outputs, all four logical candidates, state/log
contents, inactive/null preservation, finite values and same-schedule
eager/replay comparisons pass. This extends V1's uniform-per-step acceptance
qualification. The small probe does not establish serving request permutations,
cache-slot reassignment, eviction/export integration or full-model correctness.

There are 540 timing samples across five modes, three accepted counts and
three rounds. The following values are the median of the three round medians:

| Accepted | Official device, us | Copied dense device, us | Emit only device, us | Complete fused compact device, us | Compact vs official |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 999.50 | 989.52 | 962.83 | 1081.47 | 8.20% slower |
| 2 | 999.16 | 989.68 | 962.81 | 1145.20 | 14.62% slower |
| 4 | 998.74 | 989.64 | 963.82 | 1278.40 | 28.00% slower |

| Accepted | Official wall, us | Copied dense wall, us | Complete fused compact wall, us | Separate reconstruction diagnostic device, us |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1112.28 | 1102.92 | 1196.28 | 310.08 |
| 2 | 1111.39 | 1105.70 | 1261.60 | 477.41 |
| 4 | 1112.57 | 1101.86 | 1394.09 | 810.99 |

Complete compact device round medians are 1081.47/1081.61/1081.23 us for
accepted1, 1145.20/1144.90/1145.36 us for accepted2, and
1279.49/1278.40/1277.49 us for accepted4. Official medians span
998.61-999.58 us across these nine groups. The negative result is consistent
across rounds, rather than an aggregate statistic hiding a successful round.
These short recurrence samples do not supply the full dynamic workload's
P95/P99 or end-to-end performance evidence.

The official host tiling calculation selects V tile 64 and output queue
depths 2/2 on the machine's 192 KiB UB. V1 used tile 16, requiring eight
V iterations instead of two. V2 matches that calculation, as documented in
`benchmarks/multi_graph_gdn_compact_state/tiling_provenance.md`. Its copied
dense control is about 1% faster than the installed operator, closing the
previous approximately 46% gap. The complete compact regression therefore
cannot be explained by leaving that known baseline mismatch unresolved.

Queued emission without previous-prefix reconstruction is only about
35-37 us faster than official. Fused accepted-prefix recovery adds about
119/182/315 us relative to that emission diagnostic for accepted1/2/4.
Those are differences between measured endpoints, not a profiler attribution
or the independently timed reconstruction latency. The accepted-prefix trend
and the recurrence formula identify a real extra computation/dependency cost;
the measurements do not prove the hardware is exclusively compute-bound.
Separate diagnostic latencies must not be added to manufacture a fused result.

The often rounded roughly 58% byte-saving hypothesis is insufficient for a
latency claim. With the actual padded gamma records, replicated keys and all
four old log records prefetched, V2's logical state-byte reduction is 54.84%.
Reading only the accepted prefix gives an idealized 54.84-56.78% budget below;
it is not V2's actual execution. These are logical bytes rather than measured
HBM transactions. The installed kernel
already buffers/prefetches state and overlaps copies; deleting its candidate
writes can save much less time than their byte count suggests. Reconstruction
also performs ordered full-matrix scale/FMA operations, preserves BF16
rounding, reads prior logs and introduces dependent work before the current
recurrence. The narrow emission saving is smaller than that observed cost
even after fusion. Further stream order or buffer-size tuning has no evidenced
budget for the required complete replay improvement.

The route fails its investment gate: no accepted1/2/4 complete recurrence
endpoint beats the installed operator, so the requested 5% complete replay
reduction is unproven and this implementation should not advance to model
integration. Reopening it would require a materially different recovery
algorithm that removes the full-matrix prefix work, or a separate demonstrated
memory-capacity objective. It should not consume further small tuning cycles
on the strength of logical bytes alone. Any later valid compact-cache benefit
must also be made available to padding before measuring split parallelism.

Persistent state allocation remains 79,429,632 bytes for the compact base and
10,238,976 bytes for the FP32 log bank, totaling 89,668,608 bytes versus dense
315,359,232 bytes: 225,690,624 bytes saved, or 71.57%. Logs add 10,238,976 bytes
over a single base; fusion uses one log bank, not parity copies. The kernel
adds 6400 bytes of queued log output and 4224 bytes of old log records per AIV
in UB. Outputs are unchanged. These figures are scoped tensor/UB allocations,
not allocator peak or full-model memory measurements. The qualification
process also allocates multiple control states and export/snapshot scratch;
that test-process footprint must not be reported as the serving memory cost.

The timed compact endpoint includes ordered recovery, base rounding/write,
current recurrence and new logs. Metadata preparation, convolution/gating,
FIA, TP communication, decoder layers, final hidden/residual and any required
dense export are outside this probe. Correctness export of four candidates is
outside timing and does not establish an export-inclusive speedup. No part of
the active goal's complete replay gate is satisfied by the diagnostic-only
emission saving.

## State-traffic and reconstruction budget

For 98 active requests and 24 heads, one BF16 state matrix set is 77,070,336
bytes. Dense execution reads one selected matrix set and writes four candidates:
385,351,680 bytes, or 385.35 decimal MB. This is a logical state-data model,
not measured HBM traffic; cache behavior and other operator traffic are excluded.

The V1 separate kernels read the old base, write its selected BF16
replacement, and read that replacement again in recurrence. They also write
four logs and read the previous accepted-prefix log. The table includes the
prototype's padded eight-float gamma and replicated keys:

| Accepted | V1 two-kernel bytes, MB | Dense state bytes saved | Ideal prefix-read fused bytes, MB | Ideal fused bytes saved | Actual V2 logical bytes, MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 243.63 | 36.78% | 166.56 | 56.78% | 174.01 |
| 2 | 246.11 | 36.13% | 169.04 | 56.13% | 174.01 |
| 4 | 251.08 | 34.84% | 174.01 | 54.84% | 174.01 |

V1 saves about 35-37% of logical state traffic. V2 retains the selected matrix
locally while preserving the BF16 rounding/load boundary, writes the new base
once, then proceeds with Q4. `LoadPreviousLogs` reads all four old records
before their in-place overwrite regardless of accepted count; thus its actual
logical state bytes equal the accepted4 budget for every count, saving 54.84%.
It does not retain an unrounded FP32 selected state across the step boundary.
The larger byte reduction does not translate into a latency improvement in V2.

Replaying accepted1/2/4 adds approximately 115.6/231.2/462.4 million scalar
operations for gamma scaling plus FMA over the matrices. This cost and native
buffer synchronization must be measured. Saving bytes does not demonstrate
that those operations can be hidden.

Exporting four dense candidates writes another 308.28 MB, reads the base and
logs, and performs ordered matrix updates. If dense export is required in a
consumer's hot path, include it in that consumer's endpoint. V1 and V2 recurrence
timing excludes export because the compact representation is the intended
persistent state; all candidates are exported separately for correctness.

The earlier approximately 981 us recurrence / 2709 us whole-GDN observations
give a research threshold: a 5% GDN improvement needs about 135.45 us net core
saving, including accepted-prefix reconstruction. A complete two-layer
approximately 4.8 ms endpoint needs roughly 240 us saving for 5%. These are
investment gates, not predictions, and neither gate is met by V1 or V2.

## Fixed dynamic workload for later integration

`benchmarks/multi_graph_dynamic_trace.py` predeclares a reproducible 128-step
workload with seed 20261018, uniform random batch counts 64-128, 128 stable
request IDs, shuffled batch positions and pauses/resumption. Initial prefix
lengths range from 512 to 1536. Subsequent sequence lengths advance only when
that request becomes active and consume its previously recorded accepted count
exactly once. The current four query tokens are included in the reported length.

The retained `structural-dynamic-trace-v1.json` has trace SHA256:

```text
0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524
```

The trace requires legal default graph capacities, changing live shapes and
contents, stable request state ownership, accepted counts 1-4, true FIA host
length updates and the existing replay/update order. Timed execution must
include preparation, all device work/joins, and hidden plus residual outputs.
This manifest is not a serving scheduler and has not been connected to the
engine or replayed through a full model. It must not be described as a completed
dynamic-serving performance result.

## Evidence and reproducibility

Raw v1 JSON/log artifacts are retained under the outer workspace's
`outputs/multi_graph_tp_attention_mlp/`. The report SHA256 is
`769bc0bfde89c193bda7a4ee07d01ae045eda5a261cd321bf7bcb02a0f00299f`;
the measured probe source SHA256 recorded in it is
`f0280aa5e592f377f276f37833da946b710e0a7ac68ea0e3a28dcc9c2b6a9f8b`.
The v1 report does not embed the extension binary/build fingerprint. Do not
substitute hashes of later edited kernel sources as provenance for that run.

The breakdown report SHA256 is
`ac3a99dcf3dc8b2e8ce3f7093da37c877834bb36fd6eb0cd1d3fc30ee4676f7f`.
It records probe SHA256
`ee75257435f5bff0bb096e764e0a9e104982748cb6380bb0e1c16cf7bddcc1cf`
and extension binary SHA256
`f3bed71d76edae420f03018f31d208009c0542fbf86f9af6094cef839c9eed40`.

The isolated build and native call contracts are documented in
`benchmarks/multi_graph_gdn_compact_state/README.md`. The copied vendor source
retains its CANN license and source attribution. No installed operator or driver
is replaced.

The V2 report SHA256 is:

```text
e1c5fe4ecad62852fea7470a58633c371ccc93fda05f7d697ae002c01bb91f0d
```

The measured probe SHA256 is:

```text
428df5f3fbdaf0e9212893d63e1edb8103db43d4904bc44966d3c0a11421068b
```

The extension `gdn_compact_state.so` SHA256 is:

```text
aa65d3f0afe8a0bb9666b78c742c8314a4d03db33f6e8f44839750e84f1cda61
```

Its Ascend C dependency `libgdn_compact_kernels.so` SHA256 is:

```text
e66ca66d20feeee9e7c8d4aaa73563b33bdf2788d96ee3ce45c57749d5b6fc7b
```

Both binaries, the measured probe, complete prototype sources, build log and
raw JSON/log are archived on `vllm-ascend-030` under
`/workspace/multi-graph-results/compact-state-v2-sources/`. The archive
`manifest.json` records each file's SHA256/size and confirms that the probe
and extension match the report. Its SHA256 is:

```text
dd65584b39d359372ae10c11a1823965760a4986864757ca6040ff91c234da5e
```

`LICENSE-CANN.txt` is byte-identical to the complete CANN license in the
installed wheel's `dist-info/licenses/LICENSES/` directory, with SHA256:

```text
89b1c114df62cfd183004f38d18d07f186436c3b37203dd69c3e84becf9d2282
```

The copied headers retain their Tianjin University copyright and CANN 2.0
notices; their source provenance and original hashes are in the prototype
README. The archive contains that full license. It is not an Apache-only
source bundle. The archive operation performs file copies and hashing only.

To reproduce in the existing sourced remote environment, build into a fresh
directory and use a new result path:

```bash
python benchmarks/multi_graph_gdn_compact_state/load.py \
  --build-dir /workspace/multi-graph-results/compact-state-build-NEW
python benchmarks/multi_graph_gdn_compact_probe.py \
  --extension /workspace/multi-graph-results/compact-state-build-NEW/gdn_compact_state.so \
  --output /workspace/multi-graph-results/tp-boundary/gdn-compact-NEW.json \
  --fused --mixed --diagnostics --repeats 12
```

The V1 breakdown, official-tiling correction and fused V2 checks complete the
planned causal screening. The next structural research should target another
measured critical path; this result supplies no complete-model or split versus
padding speedup claim.
