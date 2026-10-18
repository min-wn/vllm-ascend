# Objective: identify gains caused by request splitting

## Scope amendment on 2026-10-18

The user stopped attention-weight replication and broadened the research to
dynamic-shape ACLGraph inference, whether or not request splitting is needed.
Keep the original TP weight partition. The [literature-backed research plan](aclgraph_dynamic_shape_research.md)
records that amendment and the next qualification gates. The original
split-specific objective below is retained for history; a general optimization
must not be misreported as proof that splitting itself won.

For the user's graduation-project requirement, the
[independent algorithm proposal](aclgraph_incremental_attention_thesis.md)
focuses on incremental attention-plan maintenance with dynamic membership
and selective KV-range splitting. Official device-length support and plan
sharing are baselines, not independent contributions or evidence of novelty.

## Goal and scope

Find and validate a structural optimization whose incremental benefit comes
from splitting requests into independently advancing execution units. The
target is a repeatable reduction of at least 5% in complete-model mean replay
latency over the fastest valid, equally optimized padding implementation, with
10% as the stretch target. Report tails and memory as well as the mean. A local
operator or decoder-layer gain does not complete this objective.

Use Qwen3.5-27B, tensor parallelism of two, and four query tokens per request
(Q4 means query length, not quantization), on the existing 910B hardware without
a driver upgrade or reboot. Keep the installed default capture-size catalog
and the immutable random dynamic trajectory in
`benchmarks/results/multi_graph_structural/structural-dynamic-trace-v1.json`:
SHA256 `0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`.
Do not add favorable capture capacities, select only favorable batches, replace
the trace with a fixed shape, or split attention heads to satisfy the goal.

## Causal definition of a splitting gain

For every candidate, describe the following before investing in a full-model
run:

1. Which real requests become ready and advance independently after splitting?
2. Which specific barrier, resource serialization, or completion dependency
   prevents that progress in the optimized padding execution?
3. Which dependency changes, and which interval on the complete replay critical
   path should therefore shrink?
4. What are the expected costs of extra launches, synchronization, state
   ownership, communication, interference, and pipeline fill/drain?

Request tiles, multiple streams, or producer/consumer overlap are mechanisms;
their presence alone does not establish a splitting gain. Separate the benefit
from doing less padded work from the benefit of independent request progress.
Both can be measured, but neither substitutes for the other in attribution.

A structural change can count when obtaining its benefit requires independently
ready request subsets, independently scheduled graphs or stages, or earlier
progress across an otherwise shared execution boundary. Identify that boundary
and verify the dependency experimentally. Do not require an optimization to be
impossible to express inside a large graph: if that graph implements the same
independent request execution, classify the execution structure rather than its
API name or number of graph objects.

## Public optimizations and portability control

Give padding every optimization that can operate without the proposed change
to independent request execution: compact state, accepted-state deferral,
layouts, fusion, efficient kernels, device-side metadata, and kernel-internal
request tiling or Cube/Vector producer/consumer overlap. Share the same math,
state representation, numerical settings, and correctness obligations.

For each new mechanism, explicitly test or establish by the actual call path
whether it can run over the full padded capacity inside an ordinary padding
graph. If so, include it in the padding baseline. A kernel-internal request
pipeline does not become exclusive to graph splitting because it was developed
in the splitting branch or uses request indices.

An untested portability claim remains unresolved; it is not evidence of
exclusive benefit. Conversely, do not declare every optimization public merely
because the entire split implementation could be copied into another wrapper.
If the copy introduces the independent request progress being investigated,
it has also copied the structural change. State exactly what execution
dependencies the portability control preserves or changes.

## Required comparison groups

Use three mandatory groups, with a fourth group when the candidate introduces
an operator, fusion, layout, or device pipeline that may be portable:

| Group | Execution and purpose |
| --- | --- |
| P: optimized padding | Fastest valid full-capacity padding graph with every applicable public optimization. This is the final performance baseline. |
| S: matched split serial/delayed | Same partitions, kernels, math, state, and ownership as the candidate, with the proposed independent progress disabled. This isolates partition/work changes and the cost of splitting. |
| X: split parallel | Same implementation as S with only the proposed readiness or concurrency change enabled. Compare against both S and P. |
| K: padding portability | Full-capacity padding using the new efficient kernel or device pipeline. If valid and faster, K becomes part of P; it cannot remain an excluded diagnostic. |

Keep workload, capacities, state history, update semantics, and timing scope
identical. A delayed variant must preserve task assignment and math; explain
which wait it inserts. If several structural changes are needed, add focused
ablations instead of assigning all of their combined gain to parallelism.

The X-versus-S difference identifies the effect of the enabled dependency
change. It proves a splitting-specific effect only after the K control and
execution audit show that optimized padding cannot obtain the same effect
while retaining its original dependency structure. X must also beat the
fastest P, not merely a deliberately delayed version of itself.

## Complete replay acceptance

Time each dynamic step from CPU preparation through completion, including H2D,
true FIA host sequence-length updates, all model layers, TP communication,
joins, hidden and residual processing, and speculative state maintenance. Do
not omit a required update or move work outside the timer for one group. Report
the boundary explicitly; a complete-model replay result is not automatically a
serving throughput result and does not include unmeasured draft execution.

Qualify continuous eager and captured/replayed execution over the entire fixed
trajectory. Check outputs, canonical and staging state, all four candidates,
accepted counts, paused/resumed owners, inactive/null requests, canaries, and
finite values. Document numerical tolerances and justify any deviation from an
exact comparison. Retain complete correctness even when performance is the
primary optimization objective.

Use repeated paired rounds and critical-rank wall time. Report all rounds,
mean and tail changes, memory, and failures. Require at least 5% repeatable mean
improvement against the strongest qualified padding control. Preserve source,
configuration, tests, and raw evidence, then commit and push them. Reaching a
local performance gate or exhausting a candidate does not complete the goal.

Prefer a small test that resolves a concrete dependency or portability question
before a full-model run. Reject or reclassify candidates whose gains disappear
when padding receives the same kernel. Public gains remain useful and should
strengthen the common baseline for the next structural candidate.

## Attribution audit of the current request-tile candidate

The current balanced AIC/AIV request-tile implementation already obtains its
ready pipeline through a **single invocation over the whole padded batch**:

- In [the backend tile dispatch](../../../../benchmarks/multi_graph_gdn_q4_log_backend.py#L463),
  both mixed delayed and ready modes enter the same branch. The
  [single `tile_module.solve` call](../../../../benchmarks/multi_graph_gdn_q4_log_backend.py#L488)
  passes the full buffers and only changes the streaming boolean at line 507;
  it does not supply a request subrange.
- The [binding defaults](../../../../benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/torch_binding.cpp#L187)
  are `request_start=0` and `request_stop=-1`.
  [The contract](../../../../benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/torch_binding.cpp#L56)
  replaces `-1` with the batch size derived from the accepted-count capacity.
  [The current-stream handler](../../../../benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/torch_binding.cpp#L97)
  submits one native solve launch for that range.
- [The native launch](../../../../benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp#L289)
  issues one `q4_tile_solve` kernel. Its
  [mixed-kernel entry](../../../../benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp#L259)
  selects `KERNEL_TYPE_MIX_AIC_1_2` and runs Cube production and Vector solve
  inside that kernel. [The producer](../../../../benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp#L91)
  publishes groups as they become ready, with at most two outstanding groups;
  [the consumer](../../../../benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/tile_pipeline_kernel.cpp#L239)
  waits per group in ready mode and waits before its loop in delayed mode.
- The older graph/stream request-half path is a
  [different backend branch](../../../../benchmarks/multi_graph_gdn_q4_log_backend.py#L509).
  The tile branch does not use that path. Its recurrence returns complete
  output for the following layer work, and it retains the same
  [deferred accepted-state commit](../../../../benchmarks/multi_graph_gdn_q4_log_backend.py#L532).

These call paths establish a useful kernel-internal producer/consumer
optimization. They do not establish that independent split graphs are required
for its benefit: the ready kernel can occupy the recurrence node of a full
padding graph without splitting that graph. Treat whole-batch ready execution
as K and therefore as an eligible public padding baseline. A later candidate
must add and measure a further structural change relative to this baseline.

The retained balanced recurrence result reports approximately 20.6% reduction
against its fastest older padding control; the complete real GDN layer reports
approximately 6.0% against its strongest delayed control. Those measurements
support investment in the kernel, not a claim of splitting-specific gain or
complete-model success. Keep the raw reports unchanged:

- [Balanced recurrence evidence](../../../../benchmarks/results/multi_graph_structural/gdn-q4-tile-solve-v3-balanced.json).
- [Complete real-layer evidence](../../../../benchmarks/results/multi_graph_structural/gdn-q4-tile-real-layer-v4.json).

The original raw real-layer summary selected an older padding variant and
reported 6.42%; including the faster mixed delayed control gives approximately
6.00%. Neither percentage resolves the portability issue. The
[complete-model attribution](../../../../benchmarks/multi_graph_gdn_q4_full_probe.py#L64)
therefore includes the ready variant among public padding modes and leaves
splitting-specific improvement unproven. Baseline membership follows execution
structure and portability, not which mode was initially called the candidate.
