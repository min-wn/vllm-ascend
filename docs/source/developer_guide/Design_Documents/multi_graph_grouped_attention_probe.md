# Grouped device-length vector attention feasibility

## Question and scope

Can sharing paged KV loads among the four query heads of one KV head make
our device-length vector prototype competitive with FIA on A2?

This is operator-internal GQA reuse, not head-based graph splitting.
The probe uses Qwen3-4B attention descriptors: BF16, Q1, 32 query heads,
8 KV heads, dimension 128, block size 128. Timing uses 104 live requests
and KV512. It is not the dynamic B98/capacity104 split-versus-padding test.

## Implementation

`benchmarks/grouped_device_length_kernel.py` is an isolated, fixed-descriptor
experimental kernel. One program processes one request and one KV head.
It loads a 32-token tile of K/V and broadcasts it over four query heads;
FP32 vector reductions implement scores, online softmax and value accumulation.
It reads current lengths and pages from device memory without task update.
It neither replaces the selected backend nor changes graph capture sizes.
The launcher is deliberately specialized to these descriptors and is not a
validated general-purpose API.

`benchmarks/grouped_device_length_probe.py` checks captured-graph length shrink,
empty requests, page permutation and regrowth against CPU FP32 attention.
The tolerance remains rtol=0.02, atol=0.01. It also compares all 104 outputs
with FIA before timing. It reuses the established device-length microbenchmark
by explicitly specializing its source in memory, preserving the FIA update
and ExternalEvent sequence; source hashes accompany the result.

## Final qualified result

Three alternating-order rounds, 20 replays per sample, synchronized at sample
boundaries. Units are microseconds per replay, including host submission:

| Path | Median us |
| --- | ---: |
| Original per-query-head vector prototype | 2722.88 |
| Grouped vector prototype | 1334.20 |
| Original four-KV-partition vector prototype | 2753.68 |
| Fixed-length FIA | 205.31 |
| FIA with host update | 389.73 |

Grouped reuse reduced prototype latency by about 51%, but remained 3.4 times
slower than FIA with update and 6.5 times slower than fixed-length FIA.
The four device-metadata transitions passed; maximum absolute error was
0.003875. The final run also passed full-batch grouped-versus-FIA qualification and
records source hashes. Raw results are saved locally under
`outputs/multi_graph_grouped_attention/pilot.json`. The initial pilot gave
1337.82 us grouped and 392.26 us updated FIA, consistent with this result.

## Interpretation and stopping point

Stop expanding this vector implementation as an FIA replacement for this
workload. Removing update alone does not compensate for the remaining kernel
cost. This experiment demonstrates a useful optimization of our prototype,
not evidence of redundant fourfold KV traffic in official FIA. Source-level
load reuse also does not prove the corresponding reduction in physical HBM
traffic; no memory-counter profile was collected for this probe.

Vector reductions, broadcast intermediates and scheduling remain candidate
costs, not separately measured bottlenecks. A Cube-based or native AscendC
implementation would be a substantially different kernel engineering task.
It needs an operator-level feasibility result before model integration.
Even a competitive replacement must be applied to padding as well as split;
its generic improvement cannot establish a split-specific benefit.

Fixed-length FIA is only a lower bound for dynamic-length execution. These
isolated replay timings exclude metadata transfers, KV writes, model layers,
and scheduler work. They must not be multiplied by layer count or compared
directly with earlier first-block timings as if the measurement scopes matched.

## Reproduction

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/grouped_device_length_probe.py
```

The narrow probe writes
`/workspace/multi-graph-results/grouped-device-length/pilot.json`.
No full model test matrix is warranted by this negative feasibility result.
