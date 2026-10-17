# FFN readiness-window feasibility gate

## Question and decision

Can the real attention completion gap pay for splitting FFN, or for splitting
Gate/Up+SwiGLU while retaining a shared Down projection?

The measured-stage release model does not support a sufficiently large benefit
for the current B98/default104 workload. Do not implement an AscendC fused FFN
or extend to two decoder layers on the strength of this candidate. This is a
negative engineering gate, not proof that every future fused kernel must fail.
No model backend or graph topology was changed by this experiment.

## Method

`benchmarks/multi_graph_ffn_readiness.py` reads the previous real-dependency
unrestricted block report and extracts both FIA completion times. The three
observed gaps are 129.5, 131.25, 132.25 us. These are completion gaps in the existing
schedule, not intrinsic request costs: host update ordering, resource scheduling,
and changing topology can change them. They are not direct FFN-input readiness
measurements, because O projection and normalization still precede FFN.

Measure Gate/Up+SwiGLU, Down, and complete FFN at 104, 56, 48 rows using actual
Qwen3-4B first-layer BF16 weights, NZ layout, and seeded random inputs. Each
captured operation passes original-ND eager comparison at rtol=0.02, atol=0.02.
Nine numerical checks pass per run. Three alternating-order rounds of 32 replays
per stage yield median wall times, with synchronization at sample boundaries.
Weights are hot. These are isolated operators, not first-block timings.

There is no documented sleep helper exposed in the installed torch.npu or
torch_npu module inventory. Rather than use a resource-consuming delay kernel,
this experiment uses an explicit analytical release model. It does not physically
replay the measured delay, simulate ongoing FIA interference, or claim a rigorous
hardware lower bound. Separately captured stage times include submission and
kernel-boundary costs which a future fused graph could change.

## Measured stages

Final independent run, microseconds:

| Rows | Gate/Up+SwiGLU | Down | Whole FFN |
| --- | ---: | ---: | ---: |
| 104 | 79.35 | 36.53 | 109.09 |
| 56 | 64.11 | 30.57 | 86.77 |
| 48 | 63.77 | 33.04 | 88.63 |

Initial run: whole FFN104 107.22 us, FFN56 87.65 us, FFN48 89.58 us.
Whole FFN is measured as one graph; it need not equal the sum of separately
measured stages. The two runs have the same qualitative conclusion.

## Release model

Let A be the early group, B the late group, and delta their input-release gap.
For the optimistic calculation, substitute the FIA completion gap for delta;
ignore O/norm, output assembly, metadata, and concurrent attention interference.
Evaluate both early-group orientations because stream IDs alone do not identify
request capacity. Keep capacities 56+48; no special capture sizes are introduced.

- Join then compute: `delta + FFN104`.
- Early whole FFN: `max(FFN_A, delta) + FFN_B`.
- Early Gate/Up, shared Down: `max(Gate_A, delta) + Gate_B + Down104`.

At delta 131.25 us and A56/B48, the final model predicts:

| Schedule | Completion from early release, us | Saving vs join, us |
| --- | ---: | ---: |
| Wait and run FFN104 | 240.34 | 0 |
| Early FFN56 then FFN48 | 219.88 | 20.46 |
| Early Gate56, Gate48, shared Down104 | 231.55 | 8.79 |

Reverse A48/B56 gives 22.32 us and 8.45 us savings respectively. Initial-run
A56/B48 predictions were 17.64 us and 9.98 us. These are model estimates, not
measured split-versus-padding improvements.

The key observation is that a 130 us release window does not imply 130 us saved:
when the early branch is fully hidden, the late small FFN still takes about89 us,
versus109 us for one large FFN. Once delta exceeds the early FFN duration,
increasing delta does not enlarge the predicted saving for this schedule.

## Gate and limits

A 5% improvement on the preceding approximately 1.125 ms padded first block would
require about 56 us saved, even before paying any remaining split overhead.
The present measured-stage model offers only about18-22 us for whole FFN,
or 8-10 us for the shared-Down candidate. These figures cannot be subtracted
from previous block timings as a prediction, but are too small to justify the
proposed large kernel rewrite based solely on release-window hiding.

This does not exclude a newly engineered FFN kernel that substantially improves
its own execution cost. Such a kernel must also optimize padding; its standalone
improvement is not a split-specific gain. Fusion could change stage timings and
reuse, so this calculation is not an impossibility theorem for tile scheduling.

No artificial-delay NPU experiment, fused FFN, shared-weight task scheduler, or
two-layer pipeline was implemented. No full model regression matrix was run.
The hardware qualification is limited to the nine isolated stage outputs.

## Reproduction and artifacts

```bash
ASCEND_RT_VISIBLE_DEVICES=0 timeout 180 python benchmarks/multi_graph_ffn_readiness.py \
  --trace-report /workspace/multi-graph-results/resource-probe/block-full.json \
  --output /workspace/multi-graph-results/ffn-readiness/new-stages.json
```

Outputs refuse overwrite. The report records source and input-report hashes.
Local evidence is in `outputs/multi_graph_ffn_readiness/`: the initial and final
stage reports, the exact input block report, and the tested benchmark source.
Remote results are in `/workspace/multi-graph-results/ffn-readiness/`.
