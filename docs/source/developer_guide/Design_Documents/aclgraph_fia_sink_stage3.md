# FIA sink stage 3: exposed planning and attention-chain budget

Status: paired measurement and budget analysis completed, 2026-10-19. This
stage measures a sequential attention-only chain, not the complete model or
online serving throughput. The evidence does not support planner maintenance
alone as the main source of a large complete-model improvement.
The [second stage](aclgraph_fia_sink_stage2.md) qualified a bounded D256
prototype; it did not establish a performance improvement.

## Fixed experiment

Use the unchanged 128-step Q4 trace with canonical SHA256
`0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`.
Installed `EngineArgs` defaults are resolved again without loading weights
or creating an executor. The required default token buckets are 256..512
in steps of 16. No additional graph size is supplied. The original first
step is B98, 392 live query tokens, default capacity 400.

Physical capacity rows all have Q4, including padding requests; padding KV
length is zero. Real cumulative Q endpoints, exact KV lengths and owner page
tables are refreshed each step. This matches the established full-model FIA
control rather than giving sink an additional zero-Q skipping advantage.
The maximum actual trace KV length is 1795. Page reservations come from the
whole fixed owner history and cover each request's permitted exact lengths.

Sixteen occurrences own sixteen independent resident synthetic Q/K/V sets,
corresponding to the target's local TP2 head geometry. Contents are fixed
within a run; owner membership, row order, lengths and page routing follow
the trace. All controls consume the same tensors. No QKV projection, recurrent
state, TP communication, sampler, accepted-token update or model weight is
included. The unchanged workload metadata does not make this a model replay.

| Mode | Captured execution | Interpretation |
| --- | --- | --- |
| `per_layer` | Sixteen sequences of metadata then attention | Per-layer planning baseline. |
| `shared` | One metadata producer then sixteen private attention calls | Strong shared-plan baseline; upstream sharing itself is not the proposed thesis contribution. |
| `attention_only` | Sixteen attention calls, with current exact metadata prepared before timing | Oracle diagnostic only; not a valid dynamic algorithm result. |
| `planner` | One metadata producer | Planning-path scale, with common preparation and launch overhead. |
| `legacy` | Sixteen FIA tasks and sixteen true host-length updates | Existing host-updated attention path, retaining required events and final joins. |

All graphs execute exclusively and use a single capture stream/pool. Outputs
and metadata remain owned. KV is allocated once per occurrence and reused by
all buckets/modes. The legacy control shares one maximum scratch per bucket
across sequential same-geometry occurrences, matching the existing framework;
it does not remove any update or share output tensors.

## Endpoints and attribution

Formal wall time starts before constructing current host descriptor lists
and tensors. It includes required H2D, replay, plan execution or real FIA
updates, dependencies and final completion. Legacy consumes host length lists,
so it uploads only its required page table, not unused device length arrays.
The sink modes upload their device query/KV lengths and page table. Synthetic
Q/K/V initialization and numerical checks remain outside formal timing.

Graph-external NPU events report a queue span. Host launch gaps and update
submission can lie between these events; the span is not pure kernel time.
A separate short profiler run follows the formal trials. Profiler overhead
is excluded from the trial measurements.

The paired `per_layer - shared` difference measures removal of fifteen plan
builds on these actual execution paths. `shared - attention_only` is an
oracle diagnostic affected by graph layout, dependencies and warm caches;
it is not automatically a pure planner duration. Incremental planning must
beat `shared`, and may save at most one exposed plan per step plus actual
attention execution improvements, less every added maintenance/merge cost.
Do not multiply one planner's cost by sixteen against the shared baseline.

The previous complete-model mean is 143474 us; a 5% improvement would need
7173.7 us net. Attention-only differences are budget evidence, not a measured
end-to-end improvement or proof that savings add linearly to that model run.

## Qualification and provenance

Before timing, each required default bucket and the final trace step compare
all sixteen private outputs across controls, with finite output and exact
zero padding. Three live rows from occurrence zero are independently checked
against FP32 attention. This is sampled correctness qualification, not all
128 continued model steps. After timing, the final step is checked again to
cover changed replay order and retained graph storage.

Native metadata resource bounds, library hashes, selected symbols, source
hashes, installed default catalog and peak memory are recorded. vLLM imports
the batch-invariant registration library even when that mode is disabled;
the final provenance check permits only its exact audited hash, confirms it
does not export the experimental sink symbols, and resolves those symbols
back to the isolated stage-two library. Batch-invariant execution is disabled.

Two preliminary attempts are preserved as failed evidence. One exposed a
length-view storage extent mismatch; fixed buckets now own exact-sized
length/page/query allocations. Another rejected the additional registration
library before its symbols were audited. A first full capture stopped at
the 48 GiB limit because the initial legacy helper duplicated workspace per
layer; the final helper restores the framework's existing scratch reuse.
These attempts are not included as successful timing trials.

## Result

The final run returned 0, with all 1920 expected samples: 128 steps, five modes
and three trials. All 85 graphs were captured. The installed default catalog
and dispatch were revalidated; the first step remains B98, 392 to 400 tokens.
No special graph size was introduced. The final allocated memory is
13,575,370,240 bytes, approximately 12.64 GiB. The legacy scratch is one
285,704,192-byte allocation per bucket, shared by its sixteen sequential
operations; each operation still owns its private output and real update.

The 18 pre-timing checks cover the first record for each of the 17 buckets and
the final trace step. All sixteen outputs per control agree within the stated
tolerance; padding is exactly zero and outputs are finite. The largest sampled
FP32 absolute error is 0.000317. Two final-step checks after timing also pass.
These checks do not establish all-step continued model correctness.

Times below are milliseconds per step, including the required preparation,
H2D, replay/update and final completion. Percentiles cover 384 observations
per mode, not percentiles of three trial means.

| Mode | Mean | P50 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: |
| `per_layer` | 6.241 | 6.258 | 7.515 | 7.796 |
| `shared` | 5.328 | 5.377 | 6.511 | 6.790 |
| `attention_only`, diagnostic | 5.264 | 5.326 | 6.456 | 6.733 |
| `planner`, including preparation/launch | 0.895 | 0.892 | 1.008 | 1.126 |
| `legacy`, sixteen true updates | 5.621 | 5.658 | 6.463 | 6.692 |

Removing fifteen repeated metadata builds saves 913.470 us on average, or
14.636% of the `per_layer` chain. All 384 matched observations improve. This
is evidence for the shared-plan baseline, not an independent incremental
algorithm contribution. Its three trial mean differences are 910.674,
940.427 and 889.307 us.

Against legacy, `shared` saves 293.567 us on average, or 5.222% of this chain.
Trial mean differences are 269.712, 177.830 and 433.159 us; 285 of 384 matched
observations improve. This is a mean improvement, not a uniform win. Shared
P95 and P99 are slightly worse than legacy, and larger capacities regress:

| Default capacity | Matched legacy minus shared wall | Reduction of legacy mean |
| --- | ---: | ---: |
| 256 | 562.967 us | 12.340% |
| 400 | 567.760 us | 9.773% |
| 480 | 10.079 us | 0.164% |
| 496 | -29.443 us | -0.473% |
| 512 | -93.992 us | -1.448% |

All seventeen capacities, their sample counts and matched differences are
retained in the analysis. Capacity 512 has only five wins among 27 pairs.
Its shared queue span is still 69.580 us shorter, but host preparation is
159.932 us longer than legacy. Capacity 496 likewise saves 119.354 us in
queue span while adding 157.132 us in preparation. The sink path constructs
and uploads Q and KV arrays in addition to the table; legacy consumes real
host lengths. The wall endpoint correctly retains this extra work.

## What the profiler establishes

A separate three-replay diagnostic at B98/capacity400 contains all fifteen
expected mode/repeat labels, with a hardware model-completion notification
per label. Each legacy replay contains sixteen CPU FIA update calls,
sixteen PCIe update DMA tasks and sixteen attention kernels. Each shared
replay contains one metadata task and sixteen attention kernels. Thus the
legacy comparison did not obtain its numbers by skipping real updates.

| Device quantity per replay | Shared sink | Legacy FIA |
| --- | ---: | ---: |
| Sum of sixteen attention kernel durations | 3997.600 us | 3569.238 us |
| First attention start to last attention end | 4043.354 us | 4963.165 us |
| Time between attention executions | 45.754 us | 1393.927 us |
| `MEM_WAIT_VALUE` inside that interval | 0 us | 1324.879 us |

The legacy kernel is actually faster: the adapted sink attention sum is
428.362 us, or 12.001%, longer. The shared path wins overall by removing
much of the legacy update supply/handshake delay. Legacy CPU update calls
sum to 2280.323 us under profiling; nested ACLNN durations and overlapping
device waits must not be added to this number.

One shared metadata producer takes 61.221 us, plus 1.127 us for its clearing
task. The formal `shared - attention_only` difference is 63.995 us, with
trial means 53.749, 80.826 and 57.409 us. The agreement of scale is useful,
but the oracle difference remains a diagnostic rather than a pure kernel
measurement. The `planner` wall mean of 894.650 us includes 575.767 us of
host preparation and additional launch/dependency overhead. It must not be
reported as 895 us of removable planner computation.

The profiler emitted a stop-in-RECORD warning, retained in the saved log and
analysis. Matching task counts and completion markers support this coarse
attribution; they do not prove all profiler records or hardware metrics are
complete. There are no AIC metrics or per-core execution measurements. The
profile therefore does not identify FD merging, core imbalance, bandwidth or
Cube/Vector utilization as the cause of the sink kernel's regression.

## Decision and next gate

Defer a native incremental metadata producer as the main performance effort
on this workload. Against `shared`, only one plan remains. Its approximately
64 us exposed diagnostic scale is small compared with the 7173.7 us needed
for a 5% reduction of the previous 143474 us complete-model reference.
The chain and complete-model endpoints differ: these are scale comparisons,
not predictions or absolute end-to-end bounds. This stage establishes neither
independent-algorithm speedup nor complete-model or serving improvement.

The next bounded experiments should address the measured costs:

1. Establish a stronger preparation baseline. Initialize each bucket's
   invariant cumulative Q4 array once, reuse pinned host descriptors, and
   refresh exact KV lengths and owner pages every step. Count host filling,
   every required copy and the final join. Keep the legacy control equally
   prepared. This tests the observed high-capacity preparation loss; it is
   not the proposed thesis contribution. Separate device length allocations
   must retain the qualified exact storage extents: a shared backing slab
   cannot be assumed safe after the stage-three storage-view failure.
2. Target the sink's measured attention execution cost. With identical D256
   arithmetic and exact lengths, compare the current full planner with a
   legal assignment of whole request/KV-head groups without FD splitting,
   then with a merge-aware selective split policy. Preserve the existing
   internal matrix subblocks. Rebuild valid native endpoints and reduction
   ownership; merely clearing FD metadata would lose or duplicate work.
   Existing sampled plans all use FD, with 7..14 groups, but that alone does
   not prove FD is the bottleneck. Count actual partials, reduction/barrier
   work, task coverage and complete occurrence time to test that hypothesis.
3. Implement incremental repair only if maintaining the improved execution
   assignment produces an additional measured benefit against its own full
   planner. A planner-only improvement cannot recover a slower attention
   kernel. Require positive complete attention endpoints, including the
   largest default capacities, before another complete-model integration.

No new native incremental planner or selective-split consumer is implemented
by this stage. The negative planner budget is an experimental result and
changes the priority of the proposal; it must not be hidden by crediting
upstream plan sharing or omitting preparation.

The next [stage-four experiment](aclgraph_fia_sink_stage4.md) keeps the same
consumer and tests a legal whole-row no-FD assignment. It finds a 292.9 us
oracle attention advantage across four real default capacities, while the CPU
plan plus copy remains 371.3 us slower than shared native execution. This
separates a promising execution-work target from an as-yet-unusable planner
implementation.

## Saved evidence and reproduction

The parent workspace retains original successful and failed reports under
`outputs/thesis_incremental_plan/stage3-evidence/`. Original final report
SHA256 is `967a7ca18380e1363273564963b0d05cbcb564ef9a7b1e09c384d74e816d2ee3`.
The repository saves a compact projection with all timing samples,
qualification checks, native provenance and a source/collection manifest:

- [`fia-sink-stage3-budget.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage3-budget.json)
- [`fia-sink-stage3-budget-analysis.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage3-budget-analysis.json)
- [`fia-sink-stage3-profile.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage3-profile.json)
- [`fia-sink-stage3-profile-analysis.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage3-profile-analysis.json)
- [`fia-sink-stage3-budget.log`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage3-budget.log)

Only redundant per-step exact descriptor arrays and owner-page arrays are
omitted from the projection. The pinned canonical trace and selection plus
`aclgraph_fia_sink_stage3_trace.load_trace_schedule()` reconstruct them.
The projection records the omission and original raw hash explicitly.

The standard-library analyzer independently reconstructs the expected
step/batch/default-capacity contract and rejects incomplete or duplicate
coverage, invalid timings and mismatched source evidence. Execution/protocol
qualification is reported separately from numeric coverage. It checks the
reported success code, sixteen occurrences and revalidated installed catalog;
it does not turn sampled operator correctness into model correctness.

```bash
python benchmarks/aclgraph_fia_sink_stage3_analyze.py \
  --input benchmarks/results/multi_graph_structural/fia-sink-stage3-budget.json \
  --output /tmp/fia-sink-stage3-analysis.json
```

CPU regression tests cover the independent trace/owner/capture contract,
timing evidence rejection and real-update ordering/shared-scratch behavior.
The remote qualification and profiler exercise the actual NPU path; CPU
fixtures are never presented as device performance evidence.

The 47 isolated CPU regression tests pass, as do scoped Ruff, formatting,
project forbidden-import and long-function checks. Required full
`bash format.sh ci` was attempted. Ruff and Markdown pass; the full command
still fails on existing spelling/import findings and missing Windows shell,
ShellCheck and Python launcher hooks. Its log is retained in the parent
workspace at `outputs/thesis_incremental_plan/format-ci-stage3.log`.
