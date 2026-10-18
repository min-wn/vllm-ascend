# Stage Four: Whole-Row No-FD Metadata Audit

This is a read-only audit of pinned source `bec19b8dea1e538e64838f3064f7e91a3fa9ee72` and the stage-two transformation. No native code was modified and no device workload was launched for this audit.

## Finding

The existing attention consumer can execute a legal no-FD plan without recompiling its attention kernel. The plan must assign every complete `(request, KV head, query-group tile)` to one core, keep the existing inner M and S2 tiling, and rebuild all contiguous core endpoints. Clearing FD counts on an existing split plan is unsafe: output-routing decisions also depend on the endpoints, and partially split rows would still write partial results into reduction scratch while the final reducer is disabled.

For the qualified Q4/Q12/KV2 geometry, `g=6` and each request/head has only 24 query-group rows. The observed `mBase=512` gives exactly one query-group tile per `(request, KV head)`. Therefore a first no-FD diagnostic can restrict all core boundaries to the particularly simple `(bN2,0,0)` form, with `bN2=request_index*2+kv_head`. It does not require a new arbitrary task-list consumer.

This keeps the same attention kernel and its inner arithmetic/tiling, but changing from partial softmax plus FD merge to a full-row online softmax changes floating-point reduction order. Do not call this bitwise-identical arithmetic or a batch-invariant mode. Qualify against the same FP32 reference and declared tolerance.

## Source Map

All native paths in this table are relative to `outputs/thesis_incremental_plan/pinned_sink_source/source/`.

| Location | Role |
| --- | --- |
| `csrc/attention/fused_infer_attention_score_v2_sink_metadata/op_kernel_aicpu/fused_infer_attention_score_v2_sink_metadata_aicpu.cpp:42` | `Compute`: parse exact device lengths, select inner tiling, build split inputs, call `SplitCore`, serialize metadata. |
| Same producer file:77 | `GenMetaData`: AIC endpoints and FD arrays, AIV reduction endpoints, base metadata. |
| Same producer file:283 | `CalcInnerSizeGqa`: ordinary TND uses S2 size 512, clamped to maximum actual KV when shorter; batch-invariant changes tiling and is a distinct control. |
| Same producer file:363 | `CalcMBaseSizeGqa`: ordinary TND uses M base 512; batch-invariant uses 256. |
| Same producer file:401 | `CreateSplitInputGqa`: exact length vectors, sparse mode, and `supportFD=(keySinkNumber==0)`. |
| Same producer file:1190 | `GetS1SeqSize`/`GetS2SeqSize`: cumulative Q and independent paged KV length interpretation. |
| Same producer file:1258 | `CalcCost`: abstract `6*ceil(M/16)+10*ceil(S2/64)` cost. |
| Same producer file:1284 | `CalcS2Range`: exact causal/sparse block range for each query-group tile. |
| Same producer file:1345 | `CalcSplitInfo`: query-group and S2 tile counts, tails, all-empty flag. |
| Same producer file:1438 | `CalcS1GCache`: whole query-group tile cost and S2 range. |
| Same producer file:1560 | `CalcBatchCost`: sum whole query-group tile costs for one request/KV head. |
| Same producer file:1586 | `CalcCostInfo`: multiply per-request cost by KV-head count. |
| Same producer file:1612 | `UpdateCursor`: canonicalize row/head/request transitions. |
| Same producer file:1659 | `AssignByBatch`: assign complete request/KV-head workloads. |
| Same producer file:1699 | `AssignByRow`: assign complete query-group rows. |
| Same producer file:1754 | `AssignByBlock`: can cut within S2, creating FD work. |
| Same producer file:1827 | `IsNeedRecordFDInfo` and `RecordFDInfo`: record rows split across cores and partial counts. |
| Same producer file:1876 | `CalcSplitPlan`: batch, row, then optional S2-block assignment. |
| Same producer file:1949 | `SplitFD`: assign reduction work to vector cores. |
| Same producer file:1992 | `SplitCore`: candidate core-count search and FD postprocessing. |
| `csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fused_infer_attention_score_v2_sink_metadata.h:26` | Exact 1024-int32 allocation contract and offsets. |
| `csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fia_kernel_nonquant_sink.h:604` | `InitCalcParamsEach`: core zero starts at zero; every other core starts at the preceding core's endpoint. |
| Same consumer file:508 | Runtime `fdFlag` is `base.numOfFd > 0`. |
| Same consumer file:784 | `tndIsS2SplitCore` also depends on row endpoint identity and `headS2Split`/`tailS2Split`, independently of `fdFlag`. |
| Same consumer file:840 | `FlashDecode`: allocates reduction buffers, performs `SyncAll`, then vector reduction. |
| Same consumer file:904 | `ShouldDispatchTask`: stop at the exact triple endpoint. |
| Same consumer file:910 | `DealZeroActSeqLen`: output zero handling, including TND. |
| Same consumer file:968 | `UpdateAxisInfo`: inner S2, query-group, then bN2 iteration order. |
| Same consumer file:1063 | `GetTaskDealMode`: exact zero lengths produce `DEAL_ZERO`; sparse ranges are applied. |
| Same consumer file:1214 | Sparse first/last row handling: `s2Start=0` and `s2End=0` mean no cross-core partial row. |
| Same consumer file:1284 | `Process`: only the prefix `aiCoreIdx < usedCoreNum` runs attention; FD is separately gated by `fdFlag`. |
| `csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fia_block_vec_nonquant_sink.h:838` | `Bmm2ResCopyOut`: split rows write partial FD scratch; unsplit rows cast/copy final output. |

Stage-two transformation `work/vllm-ascend/benchmarks/aclgraph_fia_sink_stage2_patch.py:95` only repairs the metadata include path. Its D256 changes add the audited host geometry gate and Cube inner subblocks; it does not alter the metadata ABI, producer policy, FD serialization, endpoint consumer, or vector output routing.

## ABI Required For A Legal No-FD Plan

The device tensor is contiguous `int32[1024]`, 4096 bytes. Unsigned words are serialized into that storage. Start from a fully zeroed array to avoid stale FD state and inactive-core data.

| Region | Word offset and content |
| --- | --- |
| AIC slot `i` | `10*i+0`: enable; `+1`: exclusive bN2 endpoint; `+2`: exclusive query-group endpoint; `+3`: exclusive S2 endpoint. |
| AIC FD fields | `10*i+4..9`: FD group coordinates, split counts, partial scratch start and M reduction information. All zero for no-FD. |
| AIV slot `j` | `360+3*j+0`: enable; `+1..2`: FD reduction endpoints. Set enable for the first `2*usedCoreNum` slots and leave endpoints zero for no-FD. |
| Base | Starts at 576: `mBase,sInner,mFdBase,numFD,usedCoreNum,usedFDvec,maxS1,maxS2,actualQDims,actualKVDims`. |

Keep `mBase`, `sInner`, `mFdBase`, maximum actual lengths and both exact length-vector dimensions consistent with the producer and this step. Set `numFD=0`, `usedFDvec=0`; set used cores to the number of nonempty contiguous assignments, at least one. Cores must be a prefix; enable holes do not implement an arbitrary sparse schedule because the consumer dispatches by `usedCoreNum` and preceding-slot endpoint.

For target Q4, define sorted bN2 cuts `0 < e0 < e1 < ... < eK-1 = Bphysical*2`. For each used core write `(BN2_END,GS1_END,S2_END)=(ei,0,0)`. Core `i` then owns the contiguous interval from the previous cut to `ei`; each complete head processes all its real S2 blocks in order. Set all S2 split fields to zero. There must be no active core whose start and end are the same query-group row: the consumer interprets that as a split row even if `numFD` is zero.

The final endpoint must include physical dummy rows, not stop at `Blive*2`. A Q4/KV0 dummy is a real physical query segment that must be reached and zeroed. For an all-zero-KV batch, preserve the existing special case: one used core, endpoint `(Bphysical*2,0,0)`, no FD. For paused holes inside the batch, preserve those row positions and allow the assigned interval to traverse them.

## Concrete First Algorithm

1. Restrict the first implementation to BF16, TND, Q12/KV2, D256, page128, sparse3, no sink/RoPE, fixed Q4 physical segments. Reject other geometry instead of attempting partial support.
2. Derive `q_i=4`, exact `kv_i`, `g=6`. Preserve original `mBase` and S2 base. Current trace has `mBase=512`, S2 base512; do not hardcode those values for short/all-empty tests without matching `CalcInnerSizeGqa`.
3. Build one ordered atom per `(physical request, KV head)`. A zero-KV atom has zero cost but remains in the domain. For nonzero KV, compute the full query-group tile's cost using the existing aligned M/S2 coefficients and causal range. At Q4 all S2 blocks up to the last query's KV endpoint are covered; row softmax still applies the exact per-query causal mask.
4. Partition the ordered atoms into at most 20 contiguous segments. Binary-search the minimum feasible maximum segment cost using a greedy feasibility pass, then reconstruct cuts while ensuring every used segment contains at least one nonzero atom. Alternatively use exact prefix DP for this small CPU diagnostic. Include zero atoms in adjacent intervals; the last segment owns every trailing dummy row.
5. Emit canonical `(next_bN2,0,0)` endpoints. One atom owns all S2 tiles, so no partial scratch indices or FD arrays are needed. For generalized multiple query-group tiles, cuts may be `(bN2,next_gS1,0)` only at complete query-group boundaries, with terminal groups canonicalized to the next bN2. That generalization needs a separate coverage oracle; the first diagnostic need not implement it.
6. Validate complete coverage, endpoint monotonicity, final sentinel, prefix used cores, no split-row identity, resource limits, metadata dimensions, and no FD references before copying the plan to device.

The cost coefficients are an existing abstract heuristic, not calibrated time. A no-FD plan can improve synchronization and partial-buffer costs but can worsen the longest-core load. A policy comparison must account for that tradeoff rather than predict a guaranteed speedup.

## Implementation Options

### A. AICPU Producer-Only Policy

Add an explicit isolated compile-time policy around `Compute`/`SplitCore`, after `CreateSplitInputGqa`, or an independent `SplitCoreWholeRows` function using `CalcSplitInfo`, `CalcBatchCache`, `CalcS1GCache` and `GenMetaData`. Recompile only the metadata AICPU shared object and stage its existing `libfia_v2_sink_aicpu_kernels.so` alias in a separate vendor tree. The public Torch schema, metadata op API and already qualified attention object can remain byte-identical. Verify their hashes.

The existing `baseInfo.supportFD` gate at producer lines1918 and1932 suppresses block assignment and FD bookkeeping, but blindly setting it false is not a proof of a valid partitioner. `AssignByRow` and the candidate core-count search must still make forward progress, cannot leave identical endpoints for empty cores, and must preserve all-zero rows. In particular `AssignByRow` advances query-group indices internally, while the block/force path and cursor updates are coupled to normal progression; use a dedicated explicit whole-atom implementation or qualify that existing branch with an independent exhaustive coverage oracle.

Do not implement no-FD by setting `batchInvariant=true`: that also changes S2 from512 to256 and M base512 to256 in this target, confounding both arithmetic and kernel performance.

The metadata dispatch JSON selects an AICPU SO by fixed contract name. Compare isolated variants in fresh child processes so AICPU loader caching cannot silently select the old producer. A full infrastructure rebuild may be the simplest package operation, but no attention source change or attention recompilation is logically required for the policy.

### B. CPU Metadata Prototype Diagnostic

Build the same 1024-word ABI on CPU, copy into an owned fixed-address NPU `int32[1024]`, then replay the unchanged attention graph. This requires no native rebuild and is the fastest bounded causal check that no-FD endpoints can remove enough attention critical-path time to matter.

Use the exact producer base geometry, reproduced for the restricted target and validated against a native baseline. Reading native metadata back every timed step would introduce D2H synchronization and must be included if used; preferably audit baseline fields outside timing and reproduce them directly with checked formulas. Every timed CPU plan construction, CPU tensor serialization, metadata H2D copy, real length/page H2D copy, graph submission and final join belongs inside the wall endpoint. Report an attention-only kernel span separately. It is legitimate to share one device plan across 16 same-geometry private Q/K/V occurrences, matching the strong shared-plan control; do not copy16 plans merely to recreate the weaker per-layer baseline.

The graph can contain only the attention chain consuming the stable metadata tensor. This has one metadata copy before replay and no AICPU producer. It is a real CPU-produced plan algorithm if all construction/copy costs are included, and an oracle diagnostic if the plan is precomputed outside timing; label these separately.

## Bounded Qualification Before Performance

- Independently enumerate exact `(physical request,KV head,query-group tile,S2 tile)` ownership; no omissions, duplicates or cross-core row splits. Include all zero-KV/padding segments in the structural traversal domain even though they have no attention tiles.
- Require actual per-core load estimates and cuts for B98/400. The recorded baseline for this case has 20 used cores and12 FD groups of2 partials. There are196 useful complete head atoms, so a 20-core no-FD plan has enough coarse parallel work without special capture sizes.
- Compare no-FD output with the ordinary shared FD plan and independent FP32 on representative rows; check all16 private occurrences across controls and exact zero padding. Do not alter the original TP shards or input contents.
- Cover B1 (only2 useful head atoms), B4, B98/400, a short-KV value below512, KV127/128/129/511/512/513, KV4097, holes, all-zero, shorter-after-longer, and FD/no-FD plan alternation at fixed metadata address. B1 can legitimately use fewer than20 cores and may lose to FD.
- Poison unused metadata words, then ensure the emitted plan zeroes all1024 words consistently; poison outputs and verify native writes, especially padding. Verify K/V input sentinels stay unchanged.
- Graph capture at a short KV length followed by replay at the qualified maximum must retain sufficient workspace. No-FD skips FD scratch, but the unchanged kernel still has ordinary per-core workspaces and host-selected kernel geometry constraints.
- All metadata buffers, lengths, page tables, private Q/K/V, and output descriptors must retain exact storage extent as well as stable addresses. The stage-three prefix-view descriptor failure must not recur.
- Record actual hardware attention-task duration and replay envelope. FD is in the same fused attention task, so task count16 cannot alone prove FD elimination; verify `numFD=0`, zero split markers, and the rebuilt row endpoints.
- Run one occurrence first, then16 sequential private occurrences with the same capture bucket and matched preparation. Only after the bounded causal result warrants it run the immutable128-step trajectory. No full-model speedup is established by this audit.

## Decision Gate

A useful result is a demonstrated net reduction of the shared-plan attention chain after including the CPU producer/copy or device producer cost. If removing FD does not improve the measured chain, do not invest in an incremental native producer solely to save the approximately62us shared build. The experiment then falsifies FD as the main lever for this trace and redirects effort toward the independently measured D256 compute gap or input-preparation budget.
