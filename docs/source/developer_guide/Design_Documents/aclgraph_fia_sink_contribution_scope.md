# FIA sink contribution scope and related-work boundary

Status: scope audit, 2026-10-19. This document compares the pinned source,
existing research records and saved stage evidence. It adds no runtime or
NPU result and does not establish literature novelty.

## Reference and verification scope

The official baseline is [PR 15336][sink] at commit
`bec19b8dea1e538e64838f3064f7e91a3fa9ee72`. The retained local archive and
source-contract audit pin its identity. The current comparison re-read the
native metadata producer and consumer from that source and the Python
helper/attention builder directly from the retained archive.

Official source anchors, relative to that pinned tree:

| Source | Verifiable behavior |
| --- | --- |
| `vllm_ascend/ops/fia_v2_sink.py` | Clones exact device length inputs, calls the AICPU metadata operator using actual stream limits, and optionally copies the 1024-word result to stable builder storage. Attention consumes that metadata. |
| `vllm_ascend/attention/attention_v1.py` | The metadata builder owns `_fia_v2_metadata_buffer`; its compatible layer group shares that plan. The official wrapper describes planning before attention replay, so plan sharing is already upstream. |
| `csrc/attention/fused_infer_attention_score_v2_sink_metadata/op_kernel_aicpu/fused_infer_attention_score_v2_sink_metadata_aicpu.cpp` | `SplitCore` calls `CalcSplitInfo` and `CalcCostInfo`, searches a candidate core-count interval through `CalcSplitPlan`, keeps a lower-cost result and invokes `SplitFD` when needed. |
| `csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fia_kernel_nonquant_sink.h` | The consumer reads native endpoints, executes the prefix of used cores and conditionally calls `FlashDecode`. Zero FD is already representable. |

Thus "outside the official PR" means a separate code change. It does not
establish research priority, novelty or superiority. In particular, the
upstream planner already evaluates costs and legal partitions; our change
must not be described as the first cost-based attention scheduler.

## Mechanism and contribution matrix

| Item | Existing mechanism | Our concrete difference | Current evidence boundary |
| --- | --- | --- | --- |
| Device lengths and plan/run | Official AICPU planning consumes device lengths and a fixed `int32[1024]` ABI; compatible layers share metadata. | Reused infrastructure. Capturing one producer with sixteen consumers in our experiment changes the experimental placement, not the invention of plan/run sharing. | Strong baseline, not an independent novelty claim. |
| Native scheduling and FD | Official `SplitCore` searches native cost-based schedules and assigns FD partial/reduction work. Core ranges are prefix-connected; the ABI has no arbitrary request-ID task list. | Stage 5 changes the legal ownership atom and policy under strict geometry guards. | Compare against upstream shared planning, not only legacy host updates. |
| A2 D256 adaptation | The pinned host whitelist rejects D256; Stage 1 found the full K tile exceeds the existing slot. | Stage 2 uses D256-only smaller MM1/MM2 subblocks with capacity assertions, retaining the existing L1 layout. | Engineering adaptation qualified by the patched isolated package and bounded eager/graph gates; not general upstream D256 support. |
| Guarded whole-row producer | The consumer already accepts legal complete-head endpoints and no-FD output. | Stage 5 reuses native costs, partitions complete `(request, KV head)` atoms by contiguous minimax, constructs a fresh zero-FD candidate, validates it before publication and falls back to upstream on failure. | Independent native policy code; no first-ever whole-row/no-FD claim. |
| CPU incremental prototype | Request-local templates, exact tails, boundary repair, merge-cost checks and fallback are implemented in Python. | This is the proposed maintenance algorithm, distinct from the native whole-row rebuild. | CPU oracle/UT evidence only; no native incremental exporter, device repair or NPU speedup. |
| Stage 6 evaluation | Same consumer, four default buckets, private synthetic Q/K/V, one producer and sixteen sequential consumers. | Required preparation, native producer, replay and joins are included in a matched policy comparison. | Bounded attention-chain evidence, not a loaded model or split/padding comparison. |
| Stage 7 evaluation | Completed original 128-step descriptor trace, seventeen buckets and separate pre/post qualification. | Measures native policy under chronological membership/length/page changes, with both formal and independent audits passing. | Observed producer-inclusive attention-chain mean reduction of 1.788171%; one round and overall P99 regress. No loaded-model or split/padding comparison. |
| Stage 8 protocol | Existing TP2 real-model harness provides real QKV/cache/GDN state. | Planned `legacy_pad`/`upstream_sink_pad`/`whole_sink_pad` integration isolates adoption from native policy. | Protocol only; no implemented model integration or measured gain. |
| Current-consumer diagnostic | Two original records, six fresh processes and separate unprofiled/profile groups compare legacy, upstream and whole. | Real update task auditing is kept outside timing; hardware completion identifies consumer execution, producer and inter-attention gaps separately. | Bounded whole/upstream wall reductions of 9.418%/10.095%; whole attention kernels remain 7.675%/8.226% slower than legacy. The point result does not replace Stage 7 or establish a new scheduling algorithm. |

## Exact native-policy difference

Stage 5 retains upstream `CalcSplitInfo` and `CalcCostInfo`. It reads
`bN2CostOfEachBatch` and duplicates each request cost for its two KV heads.
For the qualified Q4/GQA6 geometry, each whole head has 24 query/group
positions and fits the retained M tile.

The policy selects at most twenty active cores, bounded by the number of
positive-cost atoms. Binary search plus greedy feasibility minimizes the
maximum contiguous segment cost under those integer weights. Reconstruction
creates useful segments and a final sentinel covering every physical dummy
row. A fresh `SplitResult` keeps all GS1/S2/FD/reduction fields zero;
strict endpoint validation precedes publication. All-zero KV uses one core.
Unsupported geometry or failed construction calls upstream `SplitCore`.

This is a full current-plan rebuild on every native producer invocation.
It does not reuse a previous plan, repair only changed owners or maintain
request generations. The CPU incremental prototype has those research goals,
but is not wired to the native ABI.

The native weights are inherited heuristics, not calibrated measured core
times. Contiguous minimax partitioning and no-FD attention are not claimed
as new general algorithms. The implemented contribution is the guarded
policy, its legal native export, fallback and target-specific evaluation.
The inherited integer maximum-segment objective is not itself new; the
complete-row constraint, fresh export and exact target guards are the
relevant implementation differences from the pinned search.

## Relationship to existing work

The following conclusions and original URLs come from the existing repository
research record. A focused primary-source recheck on 2026-10-19 confirmed the
FlashInfer paper, the pinned balanced-GQA README and the pinned official sink
wrapper. This is not a comprehensive new literature or priority search.

| Work | Previously verified mechanism | Consequence for our claim |
| --- | --- | --- |
| [FlashInfer, MLSys 2025][flashinfer] | Separate planning/specialized execution, fixed-grid workspace and compatible-layer reuse. | Plan/run and shared plans are prior design principles. |
| [FlashInfer attention API][flashinfer-api] | `plan()` prepares structures and `run()` consumes them; graph mode retains fixed buffers. | Cached metadata does not imply a new pointer-bound executor. |
| [FlashInfer balanced GQA example][cake] | Device planning splits long requests and packs short work in a specialized configuration. | Device scheduling is established; its recorded SM100/103 D128/GQA8/HND/page16 target does not qualify our A2 D256 route. |
| [FlexAttention for inference][flex] | Reuses causal/block structure during decoding. | Boundary reuse alone is not an independent incremental algorithm. |
| [TensorRT-LLM attention][trt] | Length/occupancy-based multi-block attention is established. | Length-based splitting alone is not novel. |
| [PyTorch CUDAGraph Trees][trees] and [vLLM graph design][vllm-graphs] | Captured paths need explicit pool/liveness/dispatch contracts. | Fixed buffers and default graph dispatch are infrastructure, not new scheduling theory. |

None of these citations establishes that our exact implementation is novel.
A publication-level comparison must still check closer variants of
whole-row policies, FD tradeoffs and incremental partition maintenance.

The [FlashInfer paper, Algorithm 1 and Sections 3.3-3.4][flashinfer-paper], is
particularly close: it forms KV chunks, orders them by length and assigns
them to CTA queues. Its CPU plan is refreshed per step and reused across
compatible layers; graph capture covers execution. Our present native policy
instead keeps complete ordered request/head atoms and exports contiguous
endpoints to the retained A2 consumer. This is a concrete implementation
difference, not evidence that either whole-row scheduling or plan reuse is new.

The pinned [balanced-GQA implementation][cake] is a closer device-scheduling
reference: one persistent launch reads device lengths, assigns work tickets
and merges split results. Its fixed SM100/SM103, D128, GQA8 and page16 route
differs from our A2/D256/GQA6/page128 qualification. We retain a separate AICPU
producer and the existing consumer, not its kernel-owned ticket scheduler.
These hardware and ABI differences motivate separate evaluation; they do
not by themselves establish an algorithmic novelty claim.

## Results that can be written now

The patched A2 D256 adaptation and Stage 5 native whole-row implementation
are completed engineering/algorithm artifacts. Their source, ABI checks,
isolation, fallback and bounded correctness gates can be described precisely.

Stage 6 records a producer-inclusive wall mean of 5.101027 ms upstream versus
4.812053 ms whole-row, a 0.288974 ms or 5.665% reduction over 80 matched
pairs. Seventy pairs improve. Three process rounds improve; one regresses
1.149%. Capacity256 empirical P99 increases from 4.210017 to 4.761551 ms
(13.101%) with only twenty samples per policy. These counterexamples remain
part of the result.

This is an observed isolated attention-chain mean reduction. Fresh-process
AB/BA blocks retain process/time drift, and repeated samples are not
independent process trials. Both complete-row assignment and FD removal
change together, so the result does not isolate FD merge duration.

Stage 7 completes all 3072 observations and 1536 matched pairs over the
chronological trace. Wall mean falls from 4996.977324 to 4907.622840 us,
an 89.354484 us or 1.788171% reduction, with 1283 wins and 253 losses.
Its four round reductions are 1.957489%, 2.083247%, -0.117017% and 3.269628%.
All seventeen bucket means improve, but capacity512 gains only 0.242842%.
Overall P99 regresses from 6550.212150 to 6631.752200 us; capacity256 P99
regresses from 4025.352650 to 5977.988800 us with 36 observations per policy.
The formal analyzer and independent audit both pass. Separate correctness
passes contain 2048 step checks, 2184 full payload checks and 6144 sampled
FP32 row comparisons, with maximum absolute error 0.0003158003091812134.
Timed samples are not individually reference-qualified. The result remains
a synthetic attention trace, not a full model, serving or split-over-padding
result, and its process blocks retain possible time/process drift.

The CPU incremental prototype may be discussed as a tested hypothesis with
merge-cost counterexamples and fallback diagnostics. It has no native NPU
implementation. Stage 8 must be implemented and qualified before any
real-model claim.
The existing CPU screen records that a high merge charge caused fallback on
126 of 127 later steps and that the Python incremental path was slower than
its full-plan control. These are negative prototype observations, not
measurements of a future native implementation.

## Claims not established

- First use of plan/run, device exact lengths, plan sharing, whole-row/no-FD
  scheduling or cost-based attention partitioning.
- Native incremental maintenance, selective incremental KV splitting or
  a measured speedup of the CPU prototype.
- Multi-stream or graph-split superiority over default padding.
- Full-model, sampler, draft-model or serving throughput gains from the
  native policy.
- Broad A2/FIA compatibility outside the explicitly qualified geometry.

## Remaining attribution gates

Before expanding Stage 8, the
[current-consumer diagnostic](aclgraph_fia_sink_consumer_diagnostic.md) localizes
a remaining 270/370 us hardware consumer gap over sixteen occurrences,
compared with only about 37 us of sink inter-attention gaps. This motivates
consumer-specific diagnostics rather than more planner or launch micro-tuning.
Its stronger two-record result also depends on replay context: same-record
inputs and plans match Stage 7, but the captured graph catalog and history
differ. Neither complete dynamic-trace robustness nor a particular allocation
cause is established.

The [four-process catalog bridge](aclgraph_fia_sink_catalog_bridge.md) preserves
the repeated two-record protocol while retaining two or seventeen graphs.
Its wall reductions are 6.086%/5.401% and 7.580%/8.316%, respectively, with
144 sampled FP32 rows passing. This does not support catalog count alone as
the explanation for Stage 7 attenuation. A single process per cell cannot
separate replay history, capture context and process/device drift. The bridge
adds experimental attribution evidence, not a new algorithm or a stable
dynamic-trace/full-model performance claim.

Stage 8's three-step real-model integration screen and critical-path headroom
measurement remain conditional. Establish whether optimized attention work is
exposed in the real TP2 model; the Stage 7 chain saving is not a promised
complete-forward saving. Only after a sound integration and useful headroom
should the experiment proceed to one continued 128-step model qualification
per control and later formal three-control measurement.
Retain true KV writes, all model state, critical-rank communication, producer,
preparation, output-copy and final-join costs.

The essential ablations are legacy padding versus upstream sink (adoption)
and upstream sink versus whole-row sink (native policy). Preparation versus
execution, per-bucket/transition behavior and tail regressions support those
comparisons; they cannot independently identify an FD-merge causal effect.
If an incremental device producer is later implemented, compare it with the
same full native policy and consumer, including repair/fallback costs.
A future split path must give padding the same best sink policy.

## Reproduction anchors

- [Research record](aclgraph_dynamic_shape_research.md)
- [Incremental proposal](aclgraph_incremental_attention_thesis.md)
- [Stage 1](aclgraph_fia_sink_stage1.md), [Stage 2](aclgraph_fia_sink_stage2.md),
  [Stage 3](aclgraph_fia_sink_stage3.md), [Stage 4](aclgraph_fia_sink_stage4.md)
- [Stage 5](aclgraph_fia_sink_stage5.md), [Stage 6](aclgraph_fia_sink_stage6.md),
  [Stage 7](aclgraph_fia_sink_stage7.md), [Stage 8 protocol](aclgraph_fia_sink_stage8_protocol.md)
- [Stage 5 native gate](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage5-native-gate.json)
- [Stage 6 raw bundle](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage6-v1/stage6-suite.json)
- [Stage 6 analysis](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage6-analysis.json)
- [Stage 7 raw bundle](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage7-v1/stage7-suite.json)
- [Stage 7 analysis](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage7-analysis.json)
- [Stage 7 independent audit](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage7-independent-audit.json)

[sink]: https://github.com/vllm-project/vllm-ascend/pull/15336
[flashinfer]: https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html
[flashinfer-paper]: https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf
[flashinfer-api]: https://docs.flashinfer.ai/api/attention.html
[cake]: https://github.com/flashinfer-ai/flashinfer/blob/759163ee4787a920c3c83998ecf6d1b61f327c3a/flashinfer/experimental/balanced_gqa_decode/README.md
[flex]: https://pytorch.org/blog/flexattention-for-inference/
[trt]: https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html
[trees]: https://docs.pytorch.org/docs/2.9/torch.compiler_cudagraph_trees.html
[vllm-graphs]: https://docs.vllm.ai/en/latest/design/cuda_graphs/
