# Resource complementarity gate for request-split execution

## Goal

Before implementing ready-tile/shared-weight Dense scheduling, test whether
attention and Dense have useful device-resource complementarity on the current
A2. This is a deliberately optimistic feasibility probe, not a model speedup.
The next stage requires a benefit after real dependencies and FIA updates.

The design direction is inspired by task-level dependency execution in
[Mirage Persistent Kernel](https://arxiv.org/abs/2512.22219),
[Event Tensor](https://arxiv.org/abs/2604.13327), and heterogeneous resource
co-scheduling in [NanoFlow](https://arxiv.org/abs/2408.12757).
These GPU implementations are not drop-in Ascend backends.

## Probe contract

`benchmarks/multi_graph_resource_probe.py` uses the actual Qwen3-4B first-layer
O/gate/up/down BF16 weights, checked NZ format 29, and seeded random inputs.
O has shape 4096 -> 2560; FFN is 2560 -> 19456, SwiGLU, then 9728 -> 2560.
Dense capacity is 56 and FIA capacity is 48 with 42 live requests at KV512,
matching the lane counts of B98 under default capacity104 / split56+48.
Dense inputs are already available and independent of this FIA. There is no
claim that these inputs could be ready at that time in a real decoder.

For each operation pair, compare a serial dependency against a fork/join graph.
Both use the same tensors, weights, stream quotas, and math. Test unrestricted
streams and 10 Cube / 20 Vector per stream on the device's 20/40 limits.
A stream quota is not evidence of exclusive physical-core partitioning.

Fixed-length capture is a device-execution feasibility control. Updated graphs
retain task-update begin/end, FIA, ExternalEvent wait/reset/record on every
replay. Updated correctness checks lengths 512 -> 129 -> 511 -> 512 against
unrestricted eager FIA; Dense checks original ND eager at rtol=0.02, atol=0.02.
Only live FIA rows are checked; dummy outputs are not consumed. This is not
full hidden/residual model qualification.

Three alternating-order rounds of 20 replays are measured without profiler.
The wall measurement amortizes repeated replay and includes final device sync;
it excludes H2D, scheduling, cache writes and full-model dependencies. The
update path uses Python submission and is not a production host-cost estimate.
Repeated first-layer weights are hot; cold/multilayer behavior is unmeasured.

## Results

Initial qualified run, medians in microseconds:

| Pair and requested quota | Fixed serial | Fixed parallel | Updated serial | Updated parallel |
| --- | ---: | ---: | ---: | ---: |
| O, unrestricted | 108.22 | 110.37 | 380.79 | 380.24 |
| O, 10/10 | 159.69 | 137.20 | 382.08 | 381.36 |
| FFN, unrestricted | 235.07 | 264.33 | 384.51 | 379.20 |
| FFN, 10/10 | 353.94 | 186.01 | 395.93 | 386.70 |

Independent FFN repeat, half-core configuration measured first:

| Requested quota | Fixed serial | Fixed parallel | Updated serial | Updated parallel |
| --- | ---: | ---: | ---: | ---: |
| 10/10 | 360.55 | 189.66 | 405.50 | 398.32 |
| Unrestricted | 232.71 | 262.21 | 390.79 | 384.12 |

O fails the gate: limiting cores does not beat unrestricted serial execution.
FFN has an optimistic fixed-graph signal: 189.66 vs 232.71 us (18.5% lower).
It does not establish a benefit once updates are included: limited parallel
398.32 us is slower than unrestricted serial 390.79 us and unrestricted parallel
384.12 us. These are only short pilots, not confidence intervals.

## Trace diagnosis

Separate four-step FFN traces confirm distinct concurrent hardware streams.
In the unrestricted fixed parallel trace, gate/up MatMul grows from about
66 us (serial) to 190 us (parallel), explaining why apparent overlap is harmful.
With 10/10 quotas it is about 85 us in parallel. FIA still slows from about
128 us limited serial to 185 us limited parallel; quotas do not eliminate
interference. Kernel spans in these individual profiled steps differ from
unprofiled amortized timings and must not be treated as an independent speedup.

In the updated parallel traces, Dense executes before FIA becomes ready; the
FIA update path prevents the ideal simultaneous-start window. This motivates
checking the existing native-update real block before undertaking a scheduler.
The level-0 trace does not establish actual per-core occupancy or HBM saturation.

## Failed and excluded runs

The first pilot did not enable `allow_internal_format`, so requested NZ weights
were not actually NZ. It then failed with an invalid-memory AI Core error while
initializing a subsequent configuration. Its timing is excluded. The harness
now asserts format29, synchronizes input initialization before side-stream
warmup, and retains captured resources through the process lifetime. These
changes are not a proven diagnosis of the original device exception.

An exploratory FIA6/Dense14 configuration completed for O but did not finish
for FFN after more than two minutes. The owned process was terminated and the
five completed rows retained as `qualified-partial.json`; the unfinished FFN
configuration has no valid result and is excluded from the retained search.
The narrow final harness only searches unrestricted and 10/10 allocations.
The independent FFN repeat completed normally.

## Reproduction and evidence

```bash
ASCEND_RT_VISIBLE_DEVICES=0 timeout 180 python benchmarks/multi_graph_resource_probe.py \
  --output /workspace/multi-graph-results/resource-probe/new.json
ASCEND_RT_VISIBLE_DEVICES=0 timeout 180 python benchmarks/multi_graph_resource_probe.py \
  --focus-ffn --trace --output /workspace/multi-graph-results/resource-probe/repeat-new.json
```

Outputs refuse overwrite. Source SHA256 is recorded. Local artifacts are under
`outputs/multi_graph_resource_probe/`; remote artifacts are under
`/workspace/multi-graph-results/resource-probe/`.

## Real-dependency first-block follow-up

Because FFN passed the optimistic fixed-graph gate, run the existing real block
pipeline twice: per-lane10/10 and unrestricted, keeping native replay/update,
packed metadata, NZ, shared cache, full FIA updates, and the same topology.
This follows actual FIA -> O -> Norm -> FFN dependencies; it does not yet
implement a shared-weight tiled Dense scheduler. Both runs retain the original
ND hidden/residual checks, 22 checks per run, all passed.

Two rounds of 16 steps, dynamic B98/97/103/98, default104 and split56+48:

| Configuration | Padding ms | Split ms | Split slowdown |
| --- | ---: | ---: | ---: |
| Half-core per lane, dynamic | 1.126734 | 1.188194 | 5.45% |
| Unrestricted, dynamic | 1.124434 | 1.207739 | 7.41% |
| Half-core per lane, fixed | 1.116731 | 1.224527 | 9.65% |
| Unrestricted, fixed | 1.123484 | 1.205952 | 7.34% |

The dynamic half-core pilot is about 1.6% shorter than unrestricted split, but
the fixed case reverses that ordering. Do not promote it as a stable improvement.
Neither reaches the proposed 5% advantage over equally optimized padding.

Median separate profiled block observations:

| Split configuration | MatMul duration sum us | FIA span us | FIA/MatMul overlap us | Compute span us |
| --- | ---: | ---: | ---: | ---: |
| Half-core | 405.5 | 405.5 | 89.0 | 805.75 |
| Unrestricted | 552.0 | 312.25 | 130.5 | 774.0 |

MatMul interference falls, but FIA completion stretches by about93 us and the
profiled compute span grows. Duration sums are not additive wall costs because
kernels overlap. The traces and unprofiled small-pilot wall times must not be
combined into a precise causal decomposition.

A recipe for each block follow-up (new output path required):

```bash
ASCEND_RT_VISIBLE_DEVICES=0 VLLM_BATCH_INVARIANT=0 python benchmarks/multi_graph_bucket_block.py \
  --model /models/Qwen3-4B --nz --native-update --native-replay --static-task-plan \
  --shared-cache --packed-metadata --independent-updates --pipeline-ffn \
  --lane-cores 10 10 --rounds 2 --repeats 16 --trace \
  --output /workspace/multi-graph-results/resource-probe/new-block-half.json
```

Omit `--lane-cores 10 10` for the unrestricted comparison. The local helper
`outputs/multi_graph_resource_probe/run_resource_block.py` records the two
commands and subprocess timeout.

## Decision

Do not integrate limited-core FFN pipelines or start a broad core-count search.
The O candidate failed even the optimistic comparison; FFN has real resource
complementarity, but simple lane quotas exchange Dense interference for slower
attention. This run therefore does not qualify the ready-tile/shared-weight
scheduler for model integration, and no such scheduler was implemented.

The result does not disprove a custom scheduler: it would need to preserve
attention progress while sharing Dense work and weights across ready request
blocks, rather than statically halving both whole lanes. That is a separate
operator implementation with an unresolved synchronization/resource contract.
It cannot be justified by reporting the optimistic18.5% as an expected model gain.
