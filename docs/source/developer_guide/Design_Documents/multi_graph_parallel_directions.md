# Causal tests of branch-specific parallel execution

## Objective and interpretation

Find execution opportunities created by independent request branches, against
the same endpoint with an explicit barrier or whole-bucket control. A faster
generic GEMM or collective alone does not establish a split advantage.

All tests use the existing TP2 Qwen3.5-27B checkpoint on 910B3, BF16, batch 98,
Q4 and capacity 400. The 200 + 200 and 272 + 128 capacities are existing default
capture sizes. Reversing branch order preserves the request rows; no bespoke
size, attention-head split, or acceptance-rate optimization is introduced.

## Mechanisms and small experiments

### Preserve dense work and pipeline only the reduction-producing suffix

Keep full-capacity post-Norm, GateUp and activation, then compute Down A,
start its reduction, and compute Down B while that reduction runs. A barrier
control computes both Down projections before either reduction. Both retain
all output reductions, waits and output/residual joins. A whole-bucket control
uses one complete Down and one reduction.

This directly tests a new boundary: previous controls either split the whole
MLP or kept the whole MLP intact. It can avoid GateUp fragmentation, but cannot
recover overlap that required GateUp A to begin before O B completed. The
isolated experiment starts with already-reduced O; a win here requires later
integration before making a complete-layer claim.

The asymmetric 272 + 128 cases also reverse the order on identical request
rows. The hypothesis is that the first branch's reduction should be matched
to the second branch's compute duration. Equal arithmetic with different
critical paths isolates this scheduling effect.

### Release branch work before the whole-bucket barrier

Two microprobes test early execution of real normalization work:

- O A and O B reductions share the original communicator. Post-Norm A can
  overlap O B reduction; the complete GateUp and Down remain whole-bucket.
- After branch MLP/Down, the actual adjacent layer's input Norm A runs on a
  separate compute stream while B continues. A barrier control delays this
  identical Norm until Down B finishes.

The first starts with rank-local O projections already available and ends
after the reduced MLP output and residual. The second additionally includes
the next input Norm, but excludes the actual next GDN core. These are distinct
timed endpoints and cannot be compared directly to each other.

### Prioritize a branch's downstream collective

Compare O A, O B, Down A, Down B with O A, Down A, O B, Down B. Both ranks use
the same order, and both have the same real branch-shaped MLP and next-Norm
endpoint. The depth-priority order could release A earlier, but it also holds
O B behind A's MLP and can idle the communication stream. This is an explicit
negative-control candidate rather than an assumed win.

### Remove the root-stream serialization before the real GDN prefix

The existing complete two-layer overlap launches MLP A and MLP B on root
before issuing the next layer's local GDN A prefix. Thus it only overlaps
that prefix with Down B's reduction tail. An old separate-profiler sample
showed approximately 224 us between Down A reduction completion and MLP B
completion. This is an issue window, not a predicted recoverable latency.

`--gdn-wavefront` adds a third compute stream. It waits on Down A, executes
the real GDN A local prefix while root computes MLP B, and records completion.
Root waits for that completion and Down B before submitting any GDN reduction.
The original collective order is preserved on both ranks. Full GDN B, both
GDN MLPs, final reductions, state writes and hidden/residual joins remain in
the measured endpoint.

This is the substantial parallelism hypothesis: the next layer's branch is
ready but trapped behind unrelated root work. The initial same-run modes are padding,
split barrier, original split overlap, and the new independent-stream overlap.
Cube/HBM contention can consume the apparent window, so the last two must be
compared directly.

## Execution and correctness

The MLP-boundary experiments run through:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/multi_graph_tp_attention.py --pair 200 200 \
  --include-mlp --shared-qkv --shared-rope --parallel-probe \
  --rounds 3 --repeats 6 --output NEW.json
```

The complete two-layer experiment uses `--gdn-pipeline --gdn-wavefront
--numerical-assessment` instead of `--parallel-probe`. It retains FIA host
updates after replay submission. No FIA operation occurs inside the isolated
MLP-boundary probe's timed endpoint; its input-producing decoder decomposition
is checked against the official eager layer before probing.

Timers are external to graph capture and include all stream joins. Each sample
uses the slower rank, then round medians. Alternate rounds reverse case order.
Wall time includes dispatch through completion. Same-schedule eager/replay
and same-shape branch controls require bitwise equality. Cross-shape numerical
differences are diagnostic, with nonfinite values rejected. Two-layer state
checks retain dynamic batches and continued-state controls.

Only focused CPU tests and these targeted NPU runs are needed for this stage.
These experiments do not measure full-model generation or serving throughput.

## Initial NPU results

`tp2-27b-parallel-directions-v1.json` completed all eager/replay and same-shape
control checks, using three rounds of six samples. The following values are
device replay medians in microseconds, including complete family endpoints:

| Boundary experiment | Whole bucket | Split barrier | Split overlap |
| --- | ---: | ---: | ---: |
| Whole GateUp, Down 200 + 200 | 920.42 | 944.76 | 844.99 |
| Whole GateUp, Down 272 then 128 | 920.42 | 944.70 | 876.23 |
| Whole GateUp, Down 128 then 272, original row identity | 920.42 | 933.82 | 837.83 |
| Early Post-Norm, 200 + 200 | 1156.64 | 1217.08 | 1183.58 |
| Early Post-Norm, 272 + 128 | 1156.64 | 1238.91 | 1194.08 |

The Down 200 + 200 pipeline saves 99.77 us (10.56%) against its same-shape
barrier and 75.43 us (8.20%) against whole bucket. Its wall medians are
1044.09 us whole, 1069.82 us barrier and 965.99 us overlap: 7.48% below whole.
This is a causal positive result at the MLP boundary, with O already reduced.
It does not yet establish a complete-layer or complete-two-layer improvement.

Reversing the fixed asymmetric branches saves another 38.40 us device versus
the forward overlap schedule (4.38%). It leaves input rows and legal capacities
unchanged. This supports matching the trailing compute to the first reduction,
but not adding a new capture size. Both early Post-Norm cases recover part of
their barrier cost but remain slower than whole bucket.

The priority probe, whose endpoint includes actual next input Norm, measured
1178.55 us whole bucket, 1154.23 us FIFO, 1156.59 us FIFO barrier and 1421.87 us
depth priority. Delaying O B for Down A loses 267.64 us against FIFO. The
next-Norm-only early release saves merely 2.36 us in this run. Neither is a
large optimization route for this workload.

`tp2-27b-gdn-wavefront-v1.json` completed 209 checks per rank, including dynamic
variants and eight continued-state steps. Wavefront, old overlap, barrier and
identical eager execution agree bitwise in the same-shape controls, including
complete state and inactive/null invariants. Complete-two-layer medians:

| Schedule | Wall replay, us | Device replay, us |
| --- | ---: | ---: |
| Padding | 4831.19 | 4760.17 |
| Split barrier | 5217.90 | 5155.62 |
| Original split overlap | 5097.80 | 5030.04 |
| Independent GDN A compute stream | 5120.69 | 5055.91 |

The new stream loses 22.89 us wall against the existing overlap. All three
rounds show the same ordering. It remains 5.99% slower than padding. The old
issue window did not translate into net benefit. The extra-stream route is
not promoted based on this result; device-trace overlap and contention need
careful attribution before proposing more streams.

The positive Down microprobe is therefore integrated as `--down-pipeline`,
requiring `--gdn-pipeline --gdn-shared`. It compares padding, the existing split
MLP with whole GDN, and whole GateUp/activation with branch Down and whole GDN.
This keeps the actual two-layer endpoint. It has no integrated Down-barrier
mode: causal overlap attribution comes from the microprobe, while the integrated
comparison measures the total effect after including attention producers and
their joins.

`tp2-27b-down-pipeline-v1.json` completed 171 checks per rank. Its complete
two-layer medians were padding 4834.46 us wall / 4775.10 us device, old split
MLP with whole GDN 4802.24 / 4738.76, and shared GateUp with split Down and
whole GDN 4809.02 / 4745.79. The new candidate is 0.53% below padding in wall
time, but 6.77 us slower than the existing split candidate. All three round
medians are below padding, yet this is a small one-run result. The isolated
8.20% improvement does not survive as a large two-layer improvement because
the full GateUp must wait for both O branches. This is a demonstrated limit
of that boundary, rather than evidence of a large new win.

## Trace-directed resource pairing

The independent-stream traces prove that concurrency was present. Across both
ranks and two separate profiler replays, GDN A's first projection increased
from 153.28-158.24 us to 342.96-380.51 us, and the preceding Down B increased
from 107.84-109.26 us to 216.48-265.54 us. The two streams overlap Cube work
for approximately 285.5-341.25 us. These are separate profiler observations,
not the formal replay medians above.

The roughly 505 us GDN recurrent kernel is labelled `KERNEL_AIVEC`. In both
second replays it starts after Down B's GEMM finishes. Thus the attempted
wavefront pairs two competing Cube phases, leaving its long vector phase
outside the useful compute window.

This motivates a more selective scheduling experiment, `split_gdn_resource`,
included alongside the four old modes under `--gdn-wavefront`:

```text
GDN A stream: wait Down_A -> input projections -> vector core -> output
                                                 ||
root stream:         wait projection-complete -> MLP_B
```

The benchmark temporarily hooks the actual GDN's last input projection module
to record a dependency event: Z for the unpacked projection path, BA for the
packed path. The hook runs before subsequent vector reshapes and the state
core. Root waits on this event before MLP B; the official GDN forward, core
and state updates execute unchanged. The temporary hook is removed on success
or error, and existing hooks remain intact. The complete prefix is joined
before later collectives.

This deliberately delays B until A's Cube projections finish, aiming to pair
the vector recurrence with B's Cube MLP. Vector work, bandwidth and tail
projections can still contend, so this is a targeted hypothesis rather than
an assumption that different kernel labels guarantee free overlap. It retains
the complete two-layer output and state endpoint and all same-shape controls.

Two preliminary resource runs were interrupted while progress logging had
stopped after capture; neither produced a completed report. They do not prove
a deadlock or incompatibility. Full-state CPU checks on this configuration
can take substantial time. The final experiment keeps explicit warmup/capture
progress and verifies that the projection dependency is actually published.

`tp2-27b-gdn-resource-v3.json` completed all five schedules and 243 checks per
rank, including eight continued-state steps. All same-shape output/state,
inactive/null and eager/replay controls passed. Three-by-six complete endpoint:

| Same-run schedule | Wall replay, us | Device replay, us |
| --- | ---: | ---: |
| Padding | 4785.75 | 4724.43 |
| Split barrier | 5203.46 | 5137.22 |
| Original split overlap | 5078.49 | 5015.96 |
| Unrestricted independent stream | 5110.03 | 5043.01 |
| Projection-gated resource pairing | 5108.98 | 5047.91 |

Resource pairing loses 30.50 us wall against original split, and is 6.75%
slower than padding. Every round favors original split over the new pairing.
It is not promoted as a performance improvement.

Separate resource-v3 profiler replays establish why the revised dependency
still fails. GDN A's input projection returns to roughly 150 us with no overlap
against B GEMMs, and the vector recurrence really overlaps GateUp B. However,
GateUp B grows to 290-305 us versus the old 217-223 us, and B's SwiGLU becomes
roughly 350 us versus roughly 18 us while the 512 us GDN vector core executes.
After that core, GDN A's output projection competes with Down B: approximately
286-289 us versus 81-83 us for the GDN projection, and 186-188 us versus
107-109 us for Down B. The interference moved from input projections to the
activation and output tail. Different kernel resource labels alone do not
create useful independence.

The next design would need to keep B's activation outside A's vector core and
separate the two output GEMMs, or reduce kernel resource occupancy. That is a
new causal target, not a measured win. It must also account for the smaller
remaining overlap window; these trace durations cannot be added to predict
replay savings. No extra stream or scheduling policy is enabled in serving.

## Reproducibility and validation

Reports, logs, both-rank traces and exact tested source snapshots are stored in
the task's `outputs/multi_graph_tp_attention_mlp/` and the remote
`/workspace/multi-graph-results/tp-boundary/` directories. Separate trace
analysis scripts and JSON summaries identify intervals and source hashes.
Main source SHA256 for the completed resource-v3 run:
`c34dd71076804fd9a972ee2eb5b1d5971271c88bc08f99d2885f0b7e1e5f1d33`.
GDN helper SHA256:
`e43d539c65efb2f018331432ca7773a647d91e8cf2db10e3f3ba801bf5adaa54`.

The final focused CPU suite passed 91 tests. Scoped Ruff, codespell and
Markdown checks passed. Full `bash format.sh ci` was attempted; existing
resource-probe spelling findings, direct Triton imports in `device_length_decode.py`,
and Windows shell/tool gaps remain. See `parallel-directions-format-ci-final.log`.
This stage adds opt-in benchmark experiments, not a production scheduling
change or a demonstrated full-model speedup.

The subsequent [phase and work-balance experiments](multi_graph_parallel_phases.md)
test activation-safe cross-layer scheduling, intra-GDN branch interleaving,
shared dense GDN with split state/output work, concurrent state cores with
Vector budgets, and capacity-constrained KV work balancing. They record both
the real overlap recovered and the costs that prevent a substantial padding win.
