# Larger opportunities for request-split parallel execution

## Research decision

Two larger targets warrant focused qualification: a pipeline across **actual
consecutive GDN layers**, and **request ownership across ranks inside a short
GDN block**. The first overlaps substantial Cube and Vector phases; the second
removes block-internal TP reductions by replicating only that block's weights.
Both change the dependency structure rather than adjusting stream priorities.
Neither has a new measured performance result.

Prioritize the short request-owned block for a larger structural gain: it
removes six collective dependencies and avoids writing a new device runtime.
Keep the compute-phase pipeline as the alternative that does not replicate
weights. Its single-layer budget is less certain after accounting for the
overlap it replaces.

For the phase pipeline, first qualify `GateUp_A || state_core_B`. Existing
schedules do not expose this window, but the candidate also replaces an
existing communication overlap and reintroduces fragmented GateUp cost.
For request ownership, first measure the new full-weight/half-row GEMM shapes
and block-boundary gathering cost before implementing state redistribution.

Device-side branch publication is a second track. A2 has an AscendC HCCL client
that can submit a reduction when a branch has actually finished. However, one
fused Down boundary alone has insufficient budget for a large improvement of
the complete two-layer endpoint. Treat it as a building block, with a compute
efficiency test before implementing communication.

GDN output/layout/normalization fusion is a useful producer for both tracks.
Its generic savings must also be offered to padding; it is not an independent
large split-specific improvement.

This record uses source inspection and retained traces as of 2026-10-17.
**No new NPU timing or runtime qualification was performed for this research.**
The reference is Qwen3.5-27B, TP2, 910B3, Q4, batch 98, default capacity 400
and legal request partitions. The measured endpoint contains actual
full-attention layer 3 and GDN layer 4 with synthetic upstream inputs; it is
not full-model or serving latency.

## Measured opportunity and investment threshold

The [phase experiments](multi_graph_parallel_phases.md) establish fragmented
dense GEMM costs and extra split preparation. More streams alone do not remove
either cost. The following inventory takes the union of device intervals,
avoiding double counting concurrent kernels. Each range covers the two ranks'
medians of two profiler replays, from GDN input MatMul to the final notification.
These are separate profiler observations, not formal replay measurements.

| GDN interval inventory | Whole GDN | Shared-dense core pipeline |
| --- | ---: | ---: |
| Elapsed interval, us | 2708-2710 | 2767-2770 |
| Any AI kernel active, us | 2186-2205 | 2439-2480 |
| Communication active without any AI kernel, us | 497-514 | 277-316 |

The pipeline removes about 200-224 us of communication-only time but adds about
253-276 us of AI activity. Whole-GDN MatMul occupies about 964-978 us and Vector
kernels about 1161-1165 us, including roughly 981 us of recurrence. Their
observed overlap is almost zero. This motivates compute overlap beyond the
remaining communication tails. Kernel labels do not prove independence:
MatMul, recurrence and communication share other resources.

The paired formal complete-two-layer run measured padding at 4826.772 us wall
and the shared-dense core pipeline at 4850.283 us. Beating that padding by 5%
requires about **264.850 us additional net saving** from this pipeline. Five
percent is an investment target, not a prediction. Improving the padding
baseline changes the target.

## Direction 1: substantial Cube and Vector phases across request branches

### A concrete untested window

`prepare_mlp()` in `benchmarks/multi_graph_tp_gdn.py` executes post-attention
Norm, GateUp and SwiGLU together. `finish_gdn_branches()` in
`benchmarks/multi_graph_tp_attention.py` calls all of it before starting B's
state core. The previous two-window schedule therefore overlaps only A's Down
and communication with that core. It cannot test GateUp/core overlap.

The proposed boundary is:

```text
B input projection is ready
A state -> O_A -> AR_A -> postNorm_A
                              |
                     +--------+--------+
                     |                 |
                 GateUp_A          state_core_B
                   Cube               Vector
                     |                 |
                     +------ join -----+
                              |
               O_B / AR_B, A activation / Down_A
                              |
              finish B MLP, all reductions and residuals
```

Finish post-Norm before the long state kernel; start SwiGLU after it. This
avoids the measured activation-starvation failure. Deliberately serialize
Cube GEMMs: simultaneous dense projections have already caused contention.
Publish O_B's reduction before Down_A's reduction, avoiding the measured losing
depth-first collective order.

The phase trace contains branch GateUp medians of roughly 438 and 293 us;
B's state/core-layout segment is roughly 629 us. These are measured split-shape
durations, not half the whole GEMM. They define a substantial candidate window
but do not prove durations under concurrency. An older trace's roughly 1.34x
GateUp inflation concerns the full-attention layer's GateUp_B against GDN_A,
with other Vector work also present. It is not a measurement of this pair.

This opportunity is not free. Delaying core_B until AR_A and post-Norm finish
also loses the existing `AR_A || core_B` window, approximately 140 us in one
trace. Two GateUp branches add roughly 268 us versus whole GateUp in the
measured decomposition. These observations are not additive latency estimates,
but they invalidate counting the entire hidden GateUp as a new net saving.
The pair test must include the current shared-dense core pipeline as a control,
not only an artificially serialized pair. A single-layer win is uncertain.

### The larger version: actual three-layer progression

Qwen3.5 contains runs of three GDN layers. After pair qualification, test actual
layers 4-6 with separate real weights, states and metadata. One request branch
can begin a later layer while another supplies complementary work in an
earlier layer. Request independence creates this scheduling freedom.

Keeping every GEMM at full M prevents that progression: every complete GateUp
or input projection is a rendezvous. More layers do not remove those repeated
barriers. A sustained pipeline must accept selected branch-shaped GEMMs or
implement a producer that accepts/publishes branch work internally. Include
the resulting fragmentation cost.

A CPU dependency model uses measured split-shape stage medians, two serial
request chains and a single collective issue sequence:

| Repeated GDN stages | Ordinary greedy schedule, us/layer | Delay core until other branch's post-Norm completes, us/layer |
| --- | ---: | ---: |
| 1 | 2725 | 2580 |
| 3 | 2630 | 2415 |

The three-stage model has about 10.9% opportunity relative to three 2709 us
whole-GDN intervals. **This is optimistic dependency screening, not measured
NPU performance or a full-model prediction.** It assumes fixed concurrent
durations, independent Cube/Vector/HCCL resources, no launch/rank-skew cost,
and repeated layers with equal stage durations. Some core/layout work may use
other resources. Previous measurements already disprove perfect independence
for several operator pairs. The model only prioritizes the next experiment.

### Qualification and stopping criteria

1. Separate post-Norm, GateUp and activation in the benchmark adapter. Compare
   the real `GateUp_A + core_B` pair serially and concurrently, with identical
   inputs, state resets, both ranks and every final wait. Exclude activation
   from this pair. Validate eager/replay and continued state, not just overlap.
2. Advance to three real layers if the paired endpoint saves approximately
   200 us or more with stable state correctness. This is a screening threshold
   to justify engineering effort, not a claimed complete-endpoint improvement.
3. Compare three-layer padding, serial split and phase-pipelined split in the
   same run. Include every projection, activation, reduction, layout change,
   join, hidden/residual output and required metadata operation. Require at
   least 5% repeatable net improvement over equally optimized padding before
   expanding to a longer segment.

If native operators lose the pair's budget to contention, do not implement
more layers around the same kernels. The deeper version requires cooperative
Vector scheduling: service short Norm/activation work between recurrent
request/head tasks, potentially inside one mixed kernel. This changes the
operator scheduler; runtime stream quotas alone have already failed. It needs
a separate resource, synchronization and state-correctness proof.

## Direction 2: branch production and device-side communication submission

### Verified A2 construction blocks

Installed CANN 9.1 headers provide `Hccl<HCCL_SERVER_TYPE_AICPU>` with AllReduce
preparation, `Commit`, `Query`, `Wait` and `Finalize`. Official documentation
includes A2 and BF16. Prepare before MatMul and commit only after its input is
complete. A custom producer can then compute B while A's reduction progresses.

Start with a bounded Down operator:

```text
prepare AR_A and AR_B in one communication session
finish every A output tile over N and K
ensure all producer cores' A writes are visible -> Commit AR_A
continue producing B
complete B writes -> Commit AR_B
Wait in preparation order; synchronize; Finalize
return complete endpoint
```

A per-tile MatMul callback cannot certify completion of all A columns and
producer cores. Explicit completion and memory ordering are essential.
Prepare/Commit must use a consistent configured AIC or AIV control path.

### Why one Down cannot be the large breakthrough

Whole GDN Down is roughly 221-231 us and its reduction 249-263 us in the traces.
Consider an illustrative model with total Down `D=225`, whole reduction `C=255`,
two serialized branch reductions `Ca=Cb=140` and A ready at `a=110`, in us:

```text
whole boundary = D + C = 480
branch boundary = max(D, a + Ca) + Cb = 390
saving = 90
```

Even the unphysical favorable case `a=0` leaves 365 us, saving 115 us. B's last
reduction remains a tail. New communication durations can change the model;
these figures are not a universal hardware bound. On the observed scale, one
Down cannot provide the roughly 265 us additional saving for a 5% two-layer
win. The existing pipeline already overlaps much of Down_A's reduction.

Early delivery also conflicts with weight reuse. Local Down weight
`[5120,8704]` contains about 89.1 MB of BF16 data. Sharing N/K tiles across A/B
can improve reuse but often finishes both branches near the same time.
Completing all A tiles first may reload weights for B and restore the small-M
penalty. One launch does not solve this. The 200-row boundary also crosses
common 64/128-row tiles, requiring explicit boundary handling.

Before integrating HCCL, measure A-ready time versus total custom Down time.
In this model, retaining roughly 90 us saving needs A ready by about 110 us
and custom compute complete by about 250 us, before other costs. If no tiling
simultaneously provides early delivery and efficient total compute, stop.

### Fair controls and current runtime limits

Compare library GEMM plus optimized full reduction; custom GEMM plus full
reduction; the same custom GEMM with two delayed branch commits; and the same
kernel with early branch commits. Also include the best supported fused-padding
path. This separates computation, communication fragmentation and early
publication. If padding adopts the same row-ready protocol, it implements the
same pipeline strategy; its benefit is not exclusive to multiple graphs.

Before real Down, qualify a bounded TP2 BF16 communication operator on the
current driver: eager, capture, repeated replay, changing contents at fixed
addresses, inactive rows, independent buffer canaries and complete session
cleanup. Measure initialization/finalization rather than treating all HCCL
duration as startup. No driver change is required by this plan.

The API boundaries matter:

- A2 uses the AICPU server. A2 does **not** support HCCL `Iterate()` for partial
  AllReduce consumption. Its documented Iterate use is a different,
  hardware-specific AlltoAll facility.
- The Prepare budget is limited per Hccl object/session, documented as no more
  than 63 calls. Installed code has a per-object counter; Finalize resets
  server turn counts for the next operator. Reserve control-message slots too.
  Do not build an unlimited 64-layer session from this interface.
- Equal contiguous branches may use one handle with `repeat=2`, fixed buffer
  stride and matching Commit/Wait counts. Unequal branches require valid
  separate handles/layouts. Wait order follows Prepare order.
- Do not initialize multiple active Hccl objects on one context. Qualify one
  bounded operator/session, successive operators and repeated graph replay.
- An ordinary graph event releases the entire kernel. Internal AR_A completion
  cannot release an existing Norm_A graph early. Cross-layer consumers require
  device-side readiness participation or a real completed kernel boundary.

This track becomes a larger opportunity only across multiple boundaries or
when it enables the compute pipeline in Direction 1.

## Direction 3: remove TP reductions inside a request-owned GDN block

This is a larger execution-strategy change, not exclusive to multiple captured
graphs. Across actual GDN layers 4-6, replicate just these layers' complete
weights on both ranks. Each rank processes half the requests with all heads:

```text
replicated hidden and residual at block entry
        |                         |
rank 0: request group A       rank 1: request group B
 full GDN+MLP, layer 4         full GDN+MLP, layer 4
 full GDN+MLP, layer 5         full GDN+MLP, layer 5
 full GDN+MLP, layer 6         full GDN+MLP, layer 6
        |                         |
        +-- gather hidden and residual --+
                      |
           resume original TP attention
```

Each owner computes a complete hidden vector, so O and Down no longer produce
partial sums. This deletes **six AllReduce dependencies** inside the three-layer
block and replaces them with the necessary block-exit gathers. The current
whole-GDN trace has about 500 us of exposed reduction time per layer, roughly
1.5 ms across three layers. This is a larger opportunity budget than the
90-115 us single-Down model. It is not a predicted saving: new GEMM dimensions,
boundary communication and ownership handling consume the budget.

The model configuration is H=5120, I=17408, 16 key heads, 48 value heads,
head dimension 128. With BF16 weights, a full MLP is about 534.8 MB and a full
GDN about 231.7 MB. Relative to TP2, replicating three such layers adds about
**1.071 GiB per rank**. Safetensors-header dtype accounting including auxiliary
parameters gives 1,149,788,592 additional bytes for three layers; including the
extra full-head null state slot gives about 1.073 GiB. This excludes activation,
graph/workspace pools, NZ duplication and communication buffers.
State ownership changes from all requests with half the
heads to half the requests with all heads, so its principal capacity is similar
for balanced owners. Null slots, alignment and actual state dtype still count.
Do not retain 401 state blocks with twice the heads on each owner: use the
owner's capacity, for example 201 blocks with 48 heads versus TP's 401 with
24 heads. Otherwise state memory silently doubles.

Per-rank theoretical GEMM work is approximately unchanged, but weight bytes
double and rows halve. For example, GateUp changes from TP2 M=400, K=5120,
N=17408 to owner M=200, K=5120, N=34816. Down changes from M=400, K=8704,
N=5120 to M=200, K=17408, N=5120. These shapes can lose enough weight reuse
to erase the communication saving. Test them directly; ordinary TP split-M
results do not establish full-weight owner performance.

Use a legal default capacity of 200 token rows per owner (50 Q4 requests), not
a new 49-request capture size. Batch 98 has 392 live tokens in two capacity-200
branches, with original request/state identities and inactive masking. No
special input values or acceptance assumptions are required.

The exact main matrix shapes, expressed as `(M,K,N)`, are:

| Projection | TP2, full capacity per rank | Request owner, half capacity per rank |
| --- | --- | --- |
| QKVZ | `(400,5120,8192)` | `(200,5120,16384)` |
| BA | `(400,5120,48)` | `(200,5120,96)` |
| O | `(400,3072,5120)` | `(200,6144,5120)` |
| GateUp | `(400,5120,17408)` | `(200,5120,34816)` |
| Down | `(400,8704,5120)` | `(200,17408,5120)` |

Each pair has the same `2*M*K*N` work per rank. This is a shape/cost identity,
not a performance prediction. Packed QKVZ/BA shards must be reconstructed in
the correct component/head order, not blindly concatenated. The recurrence
has the same number of live request/head tasks under a balanced partition;
the actual kernel tiling and state correctness still require qualification.

For a padded capacity 400 and H=5120, each BF16 hidden/residual endpoint is
4.096 MB globally. Two TP2 gathers send a total of about 4.096 MB per rank in
an ideal ring accounting, compared with about 24.576 MB/rank for six full
AllReduces. This is approximately a sixfold payload reduction; it does not
predict latency or count required packing/state migration. Pack both endpoints
into one collective only if its copy/workspace costs are included.

An illustrative go/no-go budget uses 1500 us deleted communication and a
projected three-layer TP baseline of `3*2709=8127` us, not a measured block.
If exit gathers/layout cost 400 us, a 5%
block improvement leaves about 694 us for all additional owner computation
and other overhead. That is about 231 us/layer, or roughly 24% of the observed
970 us/layer MatMul inventory. If boundary costs are 600 us, the remaining
budget falls to about 494 us. These are sensitivity scenarios, not measured
gather times. Measure the five shape pairs and boundary first.

Dynamic ownership needs an explicit contract. Initially place each request's
complete GDN state on one owner; retain that owner across replays. Joining
hidden/residual does not join or migrate SSM/conv state. Rebalance only with
accounted state transfer, or accept bounded imbalance and record its cost.
One existing request's four state slots across three layers are approximately
19.4 MiB. Converting already populated TP state to owner state needs about
485 MiB of remote head shards per rank at capacity, excluding protocol costs.
This is an initial transition, not work to repeat at each layer or replay.
Persistent owner-local state avoids that recurring transfer.

An arbitrary reshuffle each step could destroy the gain. A fixed owner can also
produce unequal active counts: evaluate legal owner buckets and state memory
on the maximum owner capacity, not an assumed perfect 49/49 split. Prefill, request
arrival/exit, rollback, prefix-state copy and return to TP layout must preserve
the same logical state. A short benchmark can hold state on its owners while
using ordinary TP only outside this block.

Qualify in three stages:

1. Compare actual TP and owner GEMM shapes, retaining real weight format and
   per-rank endpoints; measure two block-exit gathers for hidden and residual.
   This separates bandwidth/shape risk from state-runtime engineering.
2. Use three actual GDN layers with stable owners. Compare complete owner
   execution, optimized TP padding and existing split, including gathers and
   state correctness. Check owner-schedule eager/replay strictly and assess
   cross-TP numerical differences separately. An owner-local padded graph gets
   identical optimizations.
3. Only expand if complete block latency beats equally optimized TP padding
   by at least 5% with enough headroom for changing requests. Do not initially
   replicate all 48 GDN layers: additional weights, full-engine state dtype,
   KV cache and graph/workspace pools can exceed 64 GiB.

This changes TP to request-data parallelism inside one block. The opportunity
comes from request independence making reductions unnecessary there; it is
not communication overlap and not inherently a benefit of graph capture.
An equally capable padding implementation may adopt the same strategy. Report
that comparison honestly rather than labeling all hybrid-TP/DP gains as split
graph gains.

## Supporting work: a GDN state kernel that directly produces consumable output

Installed FLA AscendC recurrent code assigns request/head tasks to Vector
cores. All value chunks of a head are handled by the same core. Accumulate a
small BF16 head-output temporary, apply head-dimension-128 gated RMSNorm there,
and write the final O-input layout. No new cross-head reduction is needed.

Preserve the existing BF16 rounding point and all state-update/copy semantics.
Change output handling first. Speculative active-index gather/scatter cannot
be removed merely because it appears as a copy. A later tile-ready O consumer
requires a safe producer/consumer protocol.

The trace identifies roughly 74 us extra copy/slice/concat and 33-37 us extra
norm/gating duration in the split-core path, plus roughly 80 us of fragmented
O projection cost. These are work inventories, not additive latency savings.
Fusion removes only the operations in its scope. Offer the same gated-Norm
fusion to padding. This is useful producer engineering, unlikely to be a large
split advantage alone.

The state source also rules out a tempting false saving: Q4 already reuses
initial state in UB across its token loop. It still writes each verification
token's state for rollback. At batch 98, 24 local heads and 128 by 128 BF16
state, initial reads are about 77.1 MB/rank and four state snapshots about
308 MB/rank, excluding padding and other tensors. Omitting snapshots changes
the contract. These bytes can interfere with concurrent GEMM traffic, but no
hardware counters currently establish HBM bandwidth as the dominant bottleneck.

A more radical alternative was also inspected: the recurrence is scalar decay
plus a rank-one update, so a base state and bounded `{decay, key, delta}` log
could represent candidate states compactly. This changes the cache interface.
The next step must reconstruct the accepted prefix, preserve step-boundary
BF16 rounding and pay compaction/materialization costs. Padding can use it
too. It does not satisfy today's endpoint requiring four materialized states,
and moving that work past the timer is invalid. Treat it as separate general
GDN-cache research requiring closed-loop timing, not a claimed split gain.
The existing kernel already contains output double-buffering and next-chunk
prefetch mechanisms; their mere addition is not a new proposal.

## Lower-priority or rejected routes

- More native streams/two state kernels: already slower, including tested Vector
  quotas. Resource labels do not establish independent hardware partitions.
- Blindly switching to AIV communication: current HCCL docs recommend at least
  24 Vector cores for A2 AllReduce/ReduceScatter. This can consume the resource
  intended for overlapping GDN compute.
- Reusing latest peer-window MoE directly: the inspected BF16 dispatch/FFN/combine
  operator registers `ascend910_93`, with `ascend910b` commented out. New headers
  and weak symbols do not qualify this path on the current machine.
- More capture sizes or KV sorting: default legal-bucket packing has a small
  work budget, and the tested KV balancing implementation lost even before
  separately counted CPU planning overhead.
- Stacking the full-M shared-dense schedule across 64 layers: its branch
  rendezvous remains at every shared projection.

## Dynamic requests and correct attribution

These proposals depend on request independence, not fixed input values or
attention-head splitting. Live contents and sequence lengths change each
replay within existing legal capacities. Preserve global state IDs, Q4 token
order, inactive states and collective ordering. Update each live state once.

Keep true FIA host lengths updated every replay and retain replay-before-update
event ordering. Count all waits/joins, layout or ownership reconstruction,
hidden and residual outputs. Keep strict same-schedule eager/replay and
continued-state checks; assess cross-shape BF16 drift separately.

The objective is a win over **equally optimized padding**. Generic fusion,
lower precision, deferred required work or bespoke capture sizes do not prove
an advantage of request-split execution. Qualify the compute pair and the
request-owned block's matrix/boundary costs first, then implement the surviving
three-layer candidate. Custom producer work follows a justified bottleneck.
The unit of optimization changes from independent complete branch graphs to
cooperating request phases or request-owned layer blocks.

## Retained evidence and sources

Analysis files in the parent workspace's
`outputs/multi_graph_tp_attention_mlp/`:

- `research_pipeline_bounds.py` and `research_pipeline_bounds.json`: interval
  unions from both `tp2-27b-gdn-core-pipeline-v1` rank traces.
- `research_gdn_phase_rotation.py` and `research_gdn_phase_rotation.json`: CPU
  dependency screening using `gdn_phases_v1_trace_analysis.json`.
- Core-pipeline trace SHA256 values:
  `003b9adaebaee34fb69306c431335f0fc8246a8550b93e10e9771bd11e3cef81`
  and `342e0ef07560070e7c386b170a234f811698fcd2edba0e220603f74a82f86273`.
- Phase-analysis input SHA256:
  `a17105c9d872ac75e13468d9ccd25fdaee77b6c559d47dd70459fe895a8c33c8`.

`outputs/gdn_request_ownership_budget.py` and its JSON report in the parent
workspace reproduce the matrix FLOPs, checkpoint-weight bytes, owner state
capacity and ideal TP2 communication bytes without torch or NPU execution.
The script SHA256 is
`a3fd65873fd4aa63f18852993b836cb5b681a37071e1518a4266cc37761a12e0`;
the JSON SHA256 is
`3b36db343f04f68c3af5edafb9584538b24abfd57c3eeeda5632367c13eeedbc`.
The deployed model config SHA256 is
`f8d190c5b89c1521220f935d2567a587d6e291ed69066a45a106560b05a2174c`.
Header reads include BF16 Conv/hidden Norm/dt bias, FP32 A_log/gated Norm,
and the three layers' actual component dimensions. MatMul-only work is
459,866,112,000 FLOPs/rank for three layers in either strategy.

Official AscendC source/docs inspected at asc-devkit commit
`a5e1e7815d50e62f6f1814e0928a443234ce4cf6`:

- [AllReduce: A2, BF16, repeat and Prepare limit](https://gitcode.com/cann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/AllReduce.md).
- [Prepare before MatMul, commit after production](https://gitcode.com/cann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/HCCL_usage.md).
- [A2 server and control core](https://gitcode.com/cann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/HCCL_template_params.md).
- [Iterate hardware and collective restrictions](https://gitcode.com/cann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/adv_api/HCCL_communication/HCCL_Kernel/Iterate.md).

Installed evidence is in `ascendc/highlevel_api/lib/hccl/hccl.h`,
`asc/impl/adv_api/detail/hccl/common/hccl_aicpu_impl.h` and
`asc/impl/adv_api/detail/hccl/common/hccl_aicpu_def.h` under CANN 9.1. Public
header and implementation SHA256 values respectively:
`2b9513ed37717b62e7cc8a02c7fb25a57e0d6b4a01d4f40e0a7a93c04f2fab66`
and `e6271df41e9a1e19afbe55d13687544acd1b91697f1c8f2cd72dc8737467ebc6`.
The recurrent source is the installed
`fla_npu_transformer_impl/ascendc/recurrent_gated_delta_rule/recurrent_gated_delta_rule.h`.

For mixed-kernel synchronization, official CrossCoreSetFlag/CrossCoreWaitFlag
contracts document A2 modes 0/2 and warn of deadlock when concurrent waiting
kernels exceed device resources. Prefer a single mixed kernel's explicit
protocol. The latest full example requires CANN 9.2 and is not a ready-made
example for this 9.1 installation. The repository's A2-registered
`grouped_matmul_swiglu_quant_v2` mixed kernel is a code-organization reference,
not BF16 GDN performance evidence.

Latest HCCL documentation snapshot
`3868603b19fdd2fb55bd5695947177d40a6c24b0`:
[AIV resource/configuration contract](https://github.com/hicann/hccl/blob/3868603b19fdd2fb55bd5695947177d40a6c24b0/docs/zh/user_guide/hccl_env/HCCL_OP_EXPANSION_MODE.md)
(a pinned mirror of the official GitCode project).
Official vLLM-Ascend snapshot `22730f46da2d821dc3955f47d56d16284eb2de81`:
[peer-window implementation](https://github.com/vllm-project/vllm-ascend/blob/22730f46da2d821dc3955f47d56d16284eb2de81/csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/utils/hccl_shmem.hpp)
and [hardware registration](https://github.com/vllm-project/vllm-ascend/blob/22730f46da2d821dc3955f47d56d16284eb2de81/csrc/mc2/dispatch_ffn_combine_bf16/op_host/dispatch_ffn_combine_bf16_def.cpp).
Source availability is not qualification of the installed compiler, driver
and ACL runtime.

## Document validation

The following [hardware qualification](multi_graph_gdn_ownership_qualification.md)
tests real dense shapes, the official state core and complete owner boundaries.
It confirms state correctness but finds that GateUp consumes most of the
communication budget; it does not establish a three-layer speedup.

Scoped Markdown and codespell checks pass. Full `bash format.sh ci` was
attempted; it retains the previously known resource-probe spelling findings,
direct Triton imports in `device_length_decode.py`, and Windows bash/shellcheck/
python3 tool gaps. The log is
`outputs/multi_graph_tp_attention_mlp/parallel-research-next-format-ci.log` in
the parent workspace. No functional implementation changed in this update,
and no NPU performance claim is introduced.
