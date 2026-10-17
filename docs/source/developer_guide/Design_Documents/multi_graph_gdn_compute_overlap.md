# TP GateUp and GDN state-core overlap

## Decision

The proposed Cube/Vector pair overlaps efficiently on 910B3. That local gain
does not survive the closed GDN tail: losing the existing communication/core
overlap and fragmenting GateUp make the candidate slower than whole-bucket
execution. Starting the core earlier does not fix the measured endpoint.
Do not expand this native-stream schedule into three layers on these results.

This experiment completes the compute-pair investigation proposed after
[request-ownership qualification](multi_graph_gdn_ownership_qualification.md).
It preserves TP2 weights and 24 local value heads. There is no complete-weight
replication, new capture size, head-based graph split or omitted state snapshot.

## Scope and controls

The actual Qwen3.5-27B layer 4 supplies the GDN core, O projection, Gemma-style
residual Norm, GateUp and SwiGLU. Q4, batch 98 and default capacities 400 and
200+200 are retained. A's already computed local O output and B's actual input
projection are frozen before the measured endpoint; their input source is
synthetic. All B state updates use the registered official core.

The pair endpoint contains half-bucket GateUp, B core and activation deferred
until B core finishes. Serial and overlap have identical arithmetic. The
closed endpoint starts by restoring A's local O tensor and submitting its
real in-place AllReduce. It finishes with 400 rows of intermediate activation,
400 rows of residual, B core output and every B candidate state.

The three main closed controls are:

```text
window_full:
  AR_A || core_B -> O_B/AR_B -> join
  -> whole post-Norm -> whole GateUp -> activation

window_split_serial:
  AR_A -> post-Norm_A -> core_B -> GateUp_A
  -> O_B/AR_B -> activation_A -> post-Norm_B/GateUp_B/activation_B -> join

window_overlap:
  AR_A -> post-Norm_A -> (core_B || GateUp_A)
  -> O_B/AR_B -> activation_A -> post-Norm_B/GateUp_B/activation_B -> join
```

An additional `window_early_core` starts B core with AR_A, then runs post-Norm_A
and GateUp_A after AR_A. Activation still waits for the core. This directly
tests whether short Norm work can make progress during the long Vector phase.
Every candidate restores A's O buffer, performs both reductions in A/B order,
and includes all waits, both Norm branches and joins. These are closed tail
measurements, not complete GDN, complete decoder or serving measurements.
The MLP Down and its output reduction are outside all tail variants.

## Formal device results

Each run has three alternating-order rounds and six samples per mode per round.
Samples take the slowest rank before computing round medians. Timing is outside
the graph; profiler capture occurs separately after formal timing. Units are us.

| Endpoint | Run v1 | Run v2 |
| --- | ---: | ---: |
| B core only | 657.53 | 663.33 |
| Half GateUp and activation | 233.97 | 240.06 |
| Whole GateUp, no activation | 317.41 | 323.04 |
| Two half GateUps and concatenation, no activation | 435.52 | 436.44 |
| Serial pair | 1056.05 | 1064.34 |
| Concurrent pair, then activation | 754.51 | 765.99 |
| Closed whole-bucket tail | 1364.98 | 1379.76 |
| Closed split serial tail | 1675.62 | 1691.70 |
| Closed split concurrent tail | 1472.62 | 1498.54 |
| Closed early-core tail | Not tested | 1537.08 |

The concurrent pair saves about 298-302 us. Against identical split serial
math, the closed concurrent tail saves about 193-203 us. Against whole-bucket
execution it is **7.9-8.6% slower**. The early-core variant is 11.4% slower than
whole-bucket in v2. This is a repeatable negative result for the complete tested
tail, despite a positive operator-pair result.

Half GateUp includes activation; whole/split GateUp diagnostics do not. Only
the latter two directly measure fragmentation: roughly 113-118 us extra.
Separate endpoint durations are not additive latency predictions.

## Trace findings

The v1 analysis checks both ranks and two profiler replays. It inventories
actual kernel intervals using union/intersection, excluding profiler aggregate
rows. Formal samples are recomputed independently from per-rank observations.

The pair has real recurrence/GateUp overlap of roughly 273-275 us. Relative to
serial, recurrence duration grows about 3.3-5.0%, GateUp_A about 15-16%, and
convolution about 40-43%. Native concurrency is real and retains a local gain.

The closed overlap starts the B-core envelope about 184-192 us later than the
whole-bucket control. The whole-bucket control overlaps AR_A with actual core
kernels by roughly 82-105 us; the candidate has no such overlap. This includes
convolution, gating and layout work, not just recurrence.

GateUp_B also grows from roughly 208-215 us in split serial to 345-346 us in the
concurrent tail, despite no remaining simultaneous Cube kernel at that point.
GateUp_A-to-B spacing is longer while the state kernel accesses memory. This
is consistent with a weight-cache/traffic effect, but no cache or bandwidth
counter proves that explanation. It must not be described as established HBM
saturation. The actual complete endpoint, rather than a sum of these effects,
determines the performance conclusion.

## Correctness and implementation

`benchmarks/multi_graph_tp_gdn_compute_overlap.py` constructs fresh Python phase
guards around stable projected tensor views. The official registered core owns
all convolution and SSM state handling. Left/null/inactive state rows remain
unchanged; all four speculative snapshots remain present. Same-schedule eager
and replay outputs and full states are strict. Serial/concurrent variants also
compare directly, so a reproducible stream-dependency error cannot pass merely
by matching itself.

Dynamic B98/B97/B99/B98 steps vary frozen input contents and accepted-token
counts while retaining storage and metadata addresses. Both ranks pass 44
checks in v1 and 52 in v2. All cross-shape comparisons pass the existing
numerical tolerance; maximum normalized RMSE is approximately 0.0044% in v1.
All modes execute B state updates from identical continued-state trajectories.

The shared timing helper now permits an optional untimed preparation callback
before each mode's warmup, used here to restore state consistently. Callers
without this callback retain their previous behavior.

## Reproduction and evidence

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 timeout 480 torchrun \
  --standalone --nnodes=1 --nproc-per-node=2 \
  benchmarks/multi_graph_tp_gdn_ownership.py \
  --compute-overlap-only --trace --rounds 3 --repeats 6 \
  --output /new/compute-overlap.json
```

Raw reports `tp2-27b-gdn-compute-overlap-v1` and `v2`, logs, both-rank traces,
tested source snapshots and `analyze_gdn_compute_overlap.py` are retained in
the parent workspace's `outputs/multi_graph_tp_attention_mlp/` and the remote
`/workspace/multi-graph-results/tp-boundary/` directory. The v1 helper SHA256 is
`11307ad57df3dbebc24cfda506d1608dbe104ad283d38fd72aa7bf16f740d81c`.
Snapshots are retained separately; v2 adds the early-core candidate.

The focused CPU suite passes 124 tests, including actual orchestration, event
ordering, callback failures and invalid modes. Scoped Ruff and codespell checks
pass. Full repository format checks retain the previously documented unrelated
findings and Windows tool gaps. No experimental schedule is enabled in the
production runner, and these results do not satisfy the active full-replay
performance objective.
