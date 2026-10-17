# GDN request ownership: hardware qualification

## Decision and scope

This follows the [larger-opportunity research](multi_graph_parallel_research_next.md).
The request-owned GDN core is executable on the current 910B3 software stack,
and ownership preserves the tested output and state exactly. Full-weight
GateUp is the main additional cost. The measured isolated costs do not yet
justify a claim of a large improvement over optimized TP padding.

These are Qwen3.5-27B TP2 probes on NPU 0/1, Q4, batch 98, default capacity
400 and owner capacities 200/200. Actual layer 4 weights and official GDN
kernels are used. The dense stage inputs are frozen synthetic inputs at each
projection boundary. No experiment in this record runs three connected GDN
layers, the complete decoder, or serving. No driver change is involved.

## Real matrix costs

`tp2-27b-gdn-ownership-cost-v2.json` measures ND and NZ for both strategies,
then chooses formats per stage and retimes the five projections as one graph.
Each sample takes the slowest rank before round medians. Formats are selected
from the same run; their choices can contain measurement noise.

| Isolated stage, best tested format | TP device, us | Request owner device, us | Change, us |
| --- | ---: | ---: | ---: |
| QKVZ | 164.46 | 196.02 | +31.56 |
| BA | 82.70 | 83.83 | +1.13 |
| O | 83.65 | 96.11 | +12.46 |
| GateUp | 316.83 | 475.07 | +158.24 |
| Down | 204.98 | 209.40 | +4.42 |
| Retimed mixed-format dense bundle | 820.94 | 1081.48 | +260.54 |

Do not substitute the isolated-stage sum for the observed bundle: allocation,
launch and execution behavior differ. The bundle still has five independent
inputs and omits recurrence, Norm and activation. It is not a layer forward.

The owner GateUp retains the per-rank theoretical FLOPs but changes from
`(M400,K5120,N17408)` to `(M200,K5120,N34816)`. Its weight bytes double and
its measured duration increases about 50%. This is a concrete bottleneck,
not evidence that the state core became slower. HBM counters were not collected,
so weight traffic is a plausible mechanism rather than a proven bandwidth
saturation diagnosis.

Both ranks passed 138 eager/replay/dynamic checks in this report. The same
captured buffers process B98, B97, B99 and B98 with changed contents. Component
sharding is checked separately on CPU. All ten stage/format cross-strategy
diagnostics pass the prior numerical tolerance. O and Down normalized RMSE
are about 0.286%; projection-shape differences are kept distinct from strict
same-schedule execution checks.

## Actual state-core qualification

`tp2-27b-gdn-owner-core-v1.json` compares the actual causal convolution,
gating, Q/K normalization, recurrence, gated Norm and required layout work:

| Core endpoint | TP | Request owner |
| --- | --- | --- |
| Tokens / value heads per rank | 400 / 24 | 200 / 48 |
| Conv state BF16 | `[401,6,5120]` | `[201,6,10240]` |
| SSM state BF16 | `[401,24,128,128]` | `[201,48,128,128]` |
| Device replay, us | 1154.13 | 1145.81 |
| Wall replay, us | 1290.25 | 1282.90 |

The manually separated TP path first passes an independent comparison against
the registered official GDN core, including the complete state. Owner inputs
then reconstruct the same frozen projections and actual Conv/A_log/dt weights
in canonical head order. Global TP configuration is unchanged.

Eight steps vary live batch, input contents and accepted-token counts. Each
strategy's eager/replay output and full state match strictly; null and inactive
state rows remain unchanged. Both ranks record 16 execution/state checks.
Canonical cross-strategy output, Conv and SSM diagnostics have **zero maximum
absolute difference in every step**. All four candidate state snapshots remain
materialized. The committed helper also makes those cross-layout equalities
hard assertions for future runs; the v1 report records their zero diagnostics.

Input projection and redistribution, output projection and collectives are
outside this endpoint. The roughly 8 us difference is too small to describe
as a substantial compute improvement. The important result is that full-head
request ownership is feasible without losing state correctness or core speed.

## Communication and packing

The dense-cost report measures six actual TP AllReduce calls at 1384.64 us
device time, owner input selection at 85.52 us, and two output AllGathers plus
canonical reordering at 307.87 us. Both hidden and residual are included.
Current NPU AllReduce mutates its input despite an upstream out-of-place
comment. The isolated timing uses six independent zero buffers so replay
remains stable without artificial reset copies. A nonzero, independent
rank-sum oracle is checked outside timing. The initial v1 attempt correctly
failed this mutation check and produced no usable timing.

`tp2-27b-gdn-owner-boundary-v1.json` separately compares complete boundaries:

| Boundary endpoint | Device replay, us |
| --- | ---: |
| Two output AllGathers and reordering | 322.79 |
| Pack + one AllGather + reorder, two output views | 299.20 |
| Same packed exit, both outputs made contiguous | 316.50 |
| Input selection + two-AllGather exit | 343.20 |
| Input selection + packed view exit | 323.17 |
| Input selection + packed contiguous exit | 332.45 |

Packing/reordering and contiguous copies are inside replay. View outputs have
row stride 10240; ordinary outputs have stride 5120. A consumer requiring
contiguous tensors cannot claim the view-only figure. The equivalent
contiguous roundtrip saves only 10.75 us device, with no wall-latency win in
this sample. It does not recover the GateUp penalty.

All seven boundary modes pass four dynamic variants against independent CPU
request/channel sentinels, including input/output addresses and input
nonmutation. Each rank records 56 hidden/residual checks. Ordinary graph
capture is used; no peer protocol or unsupported device-side HCCL is required.

## Targeted GateUp channel-group experiment

`tp2-27b-gdn-owner-mlp-v1.json` tests whether local intermediate-channel
partitioning improves the full-weight, small-M GateUp. For each TP/owner layout,
it compares one, two and four matching gate/up groups. All weights use NZ.
The local MLP includes the actual Gemma-style residual Norm, actual activation,
intermediate concatenation and one complete local Down. No output reduction
or owner gather is timed in this endpoint.

| Local connected MLP | One group, us | Two groups, us | Four groups, us |
| --- | ---: | ---: | ---: |
| TP | 613.61 | 635.60 | 704.93 |
| Request owner | 753.21 | 767.28 | 805.79 |

The owner MLP becomes 1.87% slower with two groups and 6.98% slower with four.
Both padding/TP and owner keep their best one-group implementation. An
independent bare-GateUp endpoint, including canonical packed-output copies,
also loses: owner takes 498.16, 568.35 and 599.91 us respectively. These
different endpoints are not added together. The tested grouping does not
recover the full-weight GateUp penalty.

Both ranks pass 60 eager/replay/dynamic/residual checks. All reported numerical
comparisons pass the previous tolerance, with maximum normalized RMSE about
0.2834%. Captures reuse fixed legal request capacities and changing input
contents; partitioning channels does not introduce a new capture size.

## What the costs do and do not establish

An isolated-cost screening calculation from the dense report is:

```text
six AR - entry - exit                         = 991.25 us
three times extra mixed-format dense work     = 781.62 us
remaining before core/elementwise changes     = 209.63 us
including the separately measured core change = 234.59 us
```

These are costs from separate endpoints and runs. They are **not measured
three-layer latency or a net speedup**. Actual Norm/activation changes,
inter-layer storage, the real three layers' weights and all joins still
matter. The connected boundary roundtrip is also not the sum of independently
timed entry and exit; use it only as another screening observation.

The larger initial communication budget has mostly been consumed by dense
shape costs. Extending this design immediately into a production runner is
not supported by a demonstrated 5% block win. The targeted GateUp/MLP experiment
also rejects the simple local channel-group remedy. Retain these qualifications
without representing ownership alone as a large optimization. A large future
gain would require removing the full-weight/small-M efficiency penalty,
retaining efficient TP dense computation under a different ownership protocol,
or qualifying the alternative cross-layer compute-phase pipeline. None of
those changes is established by the present probes.

The owner layout is a mixed TP/request-DP architecture. Equally optimized
padding may use the same ownership and local padded graphs. Attribute any
future benefit to that architecture, not exclusively to multiple graph
objects. State owners follow stable request IDs modulo two, and state must
not migrate when batch positions compact. Tests with prefix request IDs do
not implement a full serving ownership scheduler or qualify migration.

## Reproduction and evidence

Launch with the existing profile and installed workspace, using new output
paths and bounded execution time. For example:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 timeout 480 torchrun \
  --standalone --nnodes=1 --nproc-per-node=2 \
  benchmarks/multi_graph_tp_gdn_ownership.py \
  --skip-core --rounds 3 --repeats 6 --output /new/dense-cost.json
```

Use `--core-only`, `--boundary-only` or `--mlp-only` for the other endpoints. The scripts
refuse existing output paths. Input projection weights load from checkpoint
components rather than blindly concatenating packed TP shards. The existing
weight-only vLLM loader initializes the real process groups and modules;
engine KV allocation, full-model prefill and full-model capture do not run.

Raw JSON, logs, analysis and tested source snapshots are retained locally in
the parent workspace's `outputs/multi_graph_tp_attention_mlp/` and remotely
under `/workspace/multi-graph-results/tp-boundary/`. The CPU analysis script
`analyze_ownership_cost.py` keeps report hashes and labels every arithmetic
estimate as screening. Exact measured source SHA256 values:

- Dense v2 main:
  `741b40261a84aeb3047241bca2ee17eadb76f5a76c6d6b74d09362f57319a69b`.
- Core v1 helper:
  `ee8d183eaadf07a7e55b43e8f9feff094915a185e67e65858e36bc344902b5df`.
- MLP v1 helper:
  `1c65a5bec5b8a329a202e405674d524a0ea5d2232ae172bb017d1203c6aeeb40`.

The saved snapshots match their report hashes. Later additions strengthen
cross-layout state assertions and dynamic checks for mixed-format bundles;
they do not retroactively change the timing sources. No measured variant is
enabled in the production runner.

## Validation

The focused CPU suite passes 112 tests, covering component reconstruction,
stable owners, state pages, per-sample rank aggregation, actual boundary
operations and grouped-MLP mathematics. Scoped Ruff, Markdown and codespell
checks pass. Full `bash format.sh ci` was attempted; it retains the previously
recorded resource-probe spelling/direct Triton import findings and Windows
bash/shellcheck/python3 tool gaps. Its log is
`outputs/multi_graph_tp_attention_mlp/gdn-ownership-format-ci.log` in the parent
workspace. Completed NPU jobs leave no benchmark running.
