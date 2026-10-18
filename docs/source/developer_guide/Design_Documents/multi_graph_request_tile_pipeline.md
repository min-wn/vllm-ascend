# Request-tile producer/consumer pipeline for compact Q4

## Attribution correction after full-bucket integration

The current ready-tile implementation is a **public padding optimization**.
`Q4LogBackend` calls `tile_module.solve` once for the complete legal bucket.
The producer and consumer exchange device flags inside that single mixed
kernel, and the complete model is captured as one graph. No independently
scheduled request graph is necessary to obtain the measured local benefit.
Consequently, the retained 20.55% recurrence and 6.00% strongest-control
real-layer improvements establish a kernel pipeline benefit, not a
split-specific advantage. Earlier investment language below describes the
candidate's original motivation and must not override this attribution.

Give this ready-tile kernel to padding before evaluating another request-split
candidate. A ready/delayed pair establishes the value of early tile consumption
within the kernel; it does not establish that multiple request graphs are
required. The [revised objective](multi_graph_split_optimization_objective.md)
requires both that counterfactual and a complete-model causal comparison.

## Objective and attribution

Only request-splitting gains count. Compact state, deferred accepted-state commit,
weight layout and any generic kernel fusion belong to the common padding
baseline. Keep full-capacity QKV/O/GateUp/Down projections and the installed
default capture catalog. This proposal changes the granularity of a real data
dependency inside GDN; it does not split attention-head graphs or add capacities.

The goal remains at least 5% repeatable complete-model mean reduction over an
equally optimized padding control, aiming at 10%, with all state/host updates,
communication, joins, hidden and residual included. The numbers below are
investment gates for a candidate, not predictions of model speedup.

## Measured bottleneck and dependency

Current request-range execution is:

```text
P(A) -> [S(A) || P(B)] -> S(B) -> gated norm -> [O/TP/MLP || accepted commit]
```

`P` computes base-state products. `S` corrects their old-log projections and
solves current outputs and candidate logs. No request needs another request's
products, but the separate kernels impose a half-bucket completion barrier.
The first request in A cannot start its solve when its own products are ready.
This is an execution dependency, not a mathematical dependency.

Retained measurements:

| Scope/control | Latency in microseconds |
| --- | ---: |
| Complete compact recurrence, full | 698.05 |
| Same recurrence, serial request halves | 726.31 |
| Same recurrence, parallel request halves | 666.66 |
| Complete real layer, common deferred padding | 2359.51 |
| Complete real layer, common deferred plus request halves | 2297.28 |
| Isolated preparation/products | 235.80 |
| Isolated solve | 333.68 |

Sources are `gdn-q4-log-pipeline-v3.json`,
`gdn-q4-log-real-layer-combined-v4.json` and `gdn-q4-log-deferred-v1.json`
under `benchmarks/results/multi_graph_structural/`. Isolated components were
measured separately: they cannot be added as an exact wall-time decomposition.
The true-layer split gain is 2.64%, below the goal.

## Structural change and cost budget

An A2 mixed AIC/AIV kernel publishes completed request/key-head tiles on the
device. Paired Vector consumers solve ready tiles while Cube produces later
requests. AIC and AIV are the existing resources; there are no extra streams of
small dense projections. Native task boundaries use the same real request IDs.

The existing producer task folds three value heads into M384/N8/K128 and writes
12 KiB of FP32 products. At B100 the full product buffer is 9.375 MiB. A later
two-slot, 20-producer handoff ring would occupy about 480 KiB. The first causal
prototype deliberately retains full products storage for both controls, so a
buffer change cannot be confused with overlap.

The existing A2 implementation in
`csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_spilit_fusion.h`
provides an installed-code reference for mixed AIC/paired-AIV task flags.
This is feasibility evidence only; build and runtime support for the new kernel
must still pass explicit tests. Avoid Vector Matmul service code, preserve
FP32 state and disabled HF32, and publish only after Cube stores are visible.

The theoretical removable interval is bounded by the smaller producer/consumer
stage, minus fill/drain and interference. Using the isolated values only as a
screen gives an upper budget near 235 us; actual removable time may be much
smaller. In the retained real-layer fixture, reaching 5% from the existing
parallel result requires another 55.75 us; reaching 10% requires 173.72 us.
Full-model requirements must be measured independently.

Explicit costs and hazards:

- Per-tile flags, fences, reads, acknowledgment and pipeline fill/drain.
- Two paired AIVs consuming three value heads can create a 2:1 task imbalance.
  Pairing producer tiles may balance six heads, at the cost of coarser readiness.
- Mixed-kernel scheduling or FP32 Cube resource use may slow either stage.
- UB usage must remain within the actual device budget.
- Every skipped/inactive/null tile must follow a matched control protocol.
- Reused flags must not overflow, and buffers must not be overwritten before
  both consumers finish. Continued graph replay must not inherit stale signals.

## Minimal causal experiment and rejection gates

1. Qualify FP32 product transport with AIC producer and paired AIV consumers.
   Check repeated eager/captured execution, dynamic input, inactive/null tasks,
   canaries and exact product identity. Do not treat transport throughput as a
   GDN recurrence speedup.
2. Reuse the actual solve body. Compare identical mixed-kernel arithmetic and
   storage with consumption delayed until each producer finishes its assigned
   product wave versus tile-ready consumption. This is a per-producer control,
   not a global device barrier. Also retain separate-kernel padding, serial request
   halves and current parallel halves. Generic fusion gain belongs to padding.
3. Count all preprocessing and deferred commit in a closed recurrence endpoint.
   Require at least roughly 60-100 us additional causal saving before investing
   in real-layer integration. Reject if imbalance or fences erase that budget.
4. Run the complete real layer with the same public optimizations, and then the
   immutable full-model dynamic trajectory only if the causal gate survives.
   Report three controls, paired repetitions, tails and memory.

If the padding implementation adopts the same request-tile readiness protocol,
it has adopted this request-splitting mechanism. Describe that explicitly as
internal request-task splitting; do not claim it validates two whole-model
graphs running concurrently. The final comparison must state exactly which
request dependency is partitioned and where the common baseline ends.

## Initial device qualification

The A2 mixed transport protocol compiled and ran on the existing driver/CANN.
Transport V1 passed 48 exact dynamic eager/replay/repeated-replay checks,
including read-only state and page canaries. With a trivial copy consumer,
per-tile handoff cost exceeded the available overlap: delayed 155.65 us versus
streamed 201.88 us. This was a protocol qualification, not a speedup claim.

Actual solve V2 then passed 144 closed-recurrence checks: eight continued
dynamic request/acceptance steps, six controls, eager and two replay executions.
Checks include complete canonical/staging state, all four exported candidates,
output, paused/null owners, finite values and page canaries. It saved 160.21 us
against the same mixed implementation's delayed consumer, but only 6.97 us
against the old two-half pipeline. The measured 2:1 paired-consumer imbalance
was therefore addressed before any real-layer expansion.

V3 publishes two unchanged M384 products together. Six value heads give each
paired AIV exactly three solves; readiness publications are halved. All math
and complete state work remain unchanged. It passed the same 144 device checks.

| V3 control, one paired run | Median device latency, us |
| --- | ---: |
| Original full padding products + deferred solve + commit | 836.81 |
| Separate products + same paired consumer + commit | 1110.99 |
| Mixed delayed consumer + commit | 835.98 |
| Mixed request-tile ready consumer + commit | 664.82 |
| Existing serial request halves + commit | 856.88 |
| Existing parallel request halves + commit | 764.55 |

This is 171.16 us overlap relative to the identical delayed mixed body,
171.99 us (20.55%) over the faster padding control, and 99.73 us (13.04%)
over the old request-half pipeline. Generic fusion alone is approximately flat
in this fixture; the per-request readiness mechanism supplies the improvement.
The slower same-consumer control is not substituted for the faster padding.

Raw reports:
[transport V1](../../../../benchmarks/results/multi_graph_structural/gdn-q4-tile-transport-v1.json),
[solve V2](../../../../benchmarks/results/multi_graph_structural/gdn-q4-tile-solve-v2.json),
[balanced solve V3](../../../../benchmarks/results/multi_graph_structural/gdn-q4-tile-solve-v3-balanced.json).
Native artifacts and exact staged sources remain under remote
`/workspace/multi-graph-results/q4-tile-build-v1-transport`,
`q4-tile-build-v2-solve` and `q4-tile-build-v3-balanced`.

These are a local causal gate, not a layer/model result. Preparation and commit
are included, but convolution, gated norm, projections, TP, FIA and complete
model metadata preparation are absent. Real-layer integration must retain the
public deferred commit and compare against the best equally optimized padding;
complete-model acceptance remains unproven.

## Complete real-layer qualification

The balanced library also passed the connected TP2 layer experiment. All ten
controls run the same real GDN layer, official convolution/gates/L2 norm,
projections, gated norm, TP reductions, residual and MLP. Accepted-state commit
remains the common deferred optimization, with its final join inside timing.
Both ranks passed 80 checks over eight continued steps: repeated eager,
eager/replay, complete state, inactive/null and candidate/cross-algorithm audits.

| Complete layer, critical-rank round medians | Round 1, us | Round 2, us | Round 3, us |
| --- | ---: | ---: | ---: |
| Common deferred padding | 2340.41 | 2349.94 | 2347.46 |
| Same paired consumer, separate kernels | 2607.11 | 2601.52 | 2621.42 |
| Mixed per-producer delayed wave | 2343.40 | 2333.54 | 2337.04 |
| Mixed request-tile readiness | 2196.82 | 2206.53 | 2185.73 |
| Previous request-half pipeline | 2295.71 | 2275.74 | 2267.31 |

The fastest median-of-round-medians public control is the mixed delayed wave,
2337.04 us. The ready candidate is 2196.82 us, **6.00% lower**. Paired gains
against that control are 6.26%, 5.44%, 6.47%. The same comparison establishes
the request-readiness increment without assigning generic fusion to splitting.
Relative to the earlier request-half pipeline the improvement is 3.47%.

The untouched [raw V4 report](../../../../benchmarks/results/multi_graph_structural/gdn-q4-tile-real-layer-v4.json)
was produced before the report selector included delayed wave as an eligible
public baseline. Its `round_medians_us` and raw samples are authoritative; its
original best-padding summary says 6.42% against the old deferred control.
The 6.00% figure here deliberately uses the stronger control. The current
selector and its CPU regression now include all three public controls.

This satisfies a real-layer investment gate, not the final goal. Full-model
capture and continuous dynamic replay are still in qualification. No complete
model throughput, tail-latency or >=5% whole-model result is claimed.
