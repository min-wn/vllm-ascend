# Complete Q4 hybrid-model replay with compact GDN ownership

## Scope and immutable inputs

Implement complete Qwen3.5-27B TP2 verification forward: all 64 decoders,
48 GDN layers, 16 FIA layers, residuals and final normalization. Q4 means
four query tokens per request. Padding receives the same compact FP32
recurrence and deferred commit as split. Local/layer gains do not establish
complete-model improvement. TP2 remains the priority.

Use [structural-dynamic-trace-v1.json](../../../../benchmarks/results/multi_graph_structural/structural-dynamic-trace-v1.json)
unchanged: SHA256 `0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`,
128 steps, 128 stable owners, random batch 64..128, shuffled request order,
pauses/resumption, fixed accepted prefixes, maximum length 1795. Verify its
digest before execution. Never regenerate it after observing performance.

## Inspected source contracts

| Primary source | Consequence |
| --- | --- |
| `benchmarks/multi_graph_tp_boundary_loader.py:69-102` | Official external launcher exposes the complete loaded model before cache allocation. |
| `benchmarks/multi_graph_tp_gdn.py:45-46` | Current fixture allocates raw four-candidate pages; do not reuse that allocation. |
| `vllm_ascend/ops/gdn.py:138-144,412-447` | Official convolution uses a real Tensor and selects candidate-table column zero. |
| `vllm_ascend/ops/gdn.py:564-578` | Spec-only recurrence receives `kv_cache[1]` without first reading Tensor shape/dtype or indexing it. |
| `vllm_ascend/ops/gdn.py:634-700` | Official prefill gathers/scatters selected real SSM rows; it needs a real Tensor. |
| `benchmarks/multi_graph_gdn_q4_log_backend.py:32-69,191-199` | Existing adapter retains dense initialization storage; introduce direct compact ownership. |
| `vllm_ascend/patch/worker/patch_qwen3_5.py:103-146` | Real decoder maintains attention, residual, normalization and MLP. |
| `vllm_ascend/attention/attention_v1.py:514-535,637-648` | FIA providers resolve host lengths by layer name; use `UpdatableGraph`. |
| `vllm_ascend/compilation/updatable_graph.py` | Registered providers/handles cover every captured attention occurrence. |

Installed upstream sources, under `/vllm-workspace/vllm/vllm`:

- `model_executor/layers/mamba/abstract.py:29-45,70-91`: standard binder
  unpacks dense raw pages; this is not a compact allocator.
- `model_executor/layers/mamba/mamba_utils.py:274-295`: official conv/SSM shapes.
- `model_executor/models/qwen3_next.py:688-745`, inherited by Qwen3.5:
  complete text forward traverses every layer and final normalization.
- `model_executor/models/qwen3_5.py:550-580`: real wrapper text-forward entry.
- `v1/worker/utils.py:591-664`: standard runner binding records raw allocations.
- `config/vllm.py:2151-2415`: runtime default capture-catalog calculation.

Installed `fla_npu` AscendC convolution source, relative to its package:
`opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/causal_conv1d/causal_conv1d.h:1260-1261`
reads speculative history at `accepted - 1`. Record inspected source hashes
and installed revisions with the eventual run; these anchors are not results.

## Explicit allocations and opaque cache handle

Build a benchmark cache manager immediately after weight loading. Do not call
the standard raw-page allocator, and do not retain stale dense candidate pools.
Allocate per GDN layer:

| Buffer | Shape/dtype |
| --- | --- |
| Real base | `[129,24,128,128]`, FP32 |
| Canonical and staging logs | Two `[129,24,1056]`, FP32 |
| Canonical and staging valid records | Two `[129,24,8]`, int32, full 32-byte records |
| Official decode conv | `[513,*gdn.get_state_shape()[0]]`, official BF16 dtype |
| Candidate-to-owner lookup | `[513]`, int32; zero null, quartet maps to owner `r+1` |

Use real loaded conv orientation; its logical per-slot bytes are 61440.
A direct-base constructor adopts initialized base without retaining a dense
source. Register 48 distinct compact owners, each with an opaque
`CompactSSMHandle`: no fake Tensor pointer/shape or implicit dense conversion.
Timed spec execution binds `gdn.kv_cache=(real_conv, registered_handle)`;
the scoped dispatcher resolves the exact handle to that layer's backend.
Unknown handles, unexpected Tensor fallbacks and contract mismatches raise.

For each FIA layer allocate independent ND BF16 K/V `[1293,128,2,256]`:
1292 fixed maximum-trace pages plus null page. Reserve stable owner pages once,
using each owner's maximum length. Bind both layer and implementation cache
references, following `multi_graph_tp_attention.py:284-291`. Never share K/V
contents across layers. Keep cache identity checks for every graph bucket.

Tracked [capacity evidence](../../../../benchmarks/results/multi_graph_structural/structural-hybrid-memory-budget-v1.json)
and direct compact arithmetic give ideal per-rank allocations:

| Item | GiB |
| --- | ---: |
| Ideal text weight shard | 25.048858 |
| All 48 FP32 bases | 9.070313 |
| All decode conv | 1.408997 |
| Two log banks | 1.169220 |
| Two valid banks | 0.008858 |
| All maximum FIA K/V | 5.050781 |
| Subtotal | 41.757026 |

Independent widest-bucket products for all 48 layers add 0.562500 GiB.
Casts, activations, workspaces, communication, NZ weight copies, runtime and
graph pools remain additional. Measure allocation at every phase; this
estimate does not prove fit. Avoid GPU correctness snapshots. Reuse a graph
pool only for mutually exclusive variants with legal buffer lifetimes.

## Coherent prefix initialization and transition

1. Bind every GDN to real `conv_init[129,*conv_shape]` and its final real
   FP32 base `[129,24,128,128]`. Zero states/null. No four-candidate SSM exists.
2. Generate deterministic per-owner token IDs of exactly the immutable initial
   prefix length; save seed/hash. Execute one owner at a time through the
   complete real model, including all TP reductions and residuals. This avoids
   materializing all 125001 prefix tokens as one forward.
3. Build the full per-layer metadata mapping. GDN uses official non-spec
   prefill with table `[[owner]]`, no draft-token metadata and no initial
   state on the first chunk. Official gather/scatter directly updates
   `base[owner]`. Later chunks, if used, need true initial-state flags.
4. FIA independently uses that owner's real page IDs, slot mapping, positions
   and prefix lengths. GDN owner IDs and FIA page IDs are separate because
   metadata lookup is by layer name. Keep official prefill operators active.
5. Raise the current loader's `max_model_len=512` and prefill-buffer limit to
   cover the fixed prefixes/1795 maximum; never truncate the trajectory.
6. After all prefixes finish, copy each `conv_init[r+1]` to decode quartet
   `1+4*r .. 4+4*r`, preserving orientation and three meaningful history
   positions. Copying all four slots is conservative. Release init-conv
   references layer by layer; base remains the same allocation.
7. Leave log validity false, install handles/48-backend registry and snapshot
   coherent prefix state. Prefix execution/capture are reported setup costs.

`multi_graph_dynamic_trace.make_trace` initializes `pending[r]=1`. Thus every
owner's first activation, even after step zero, already has
`accepted_from_previous=1`; official conv reads history offset zero. No trace
field changes or artificial acceptance choices are needed. Previously seen,
paused owners retain their true pending acceptance and must not be reset.

## Default dispatch, metadata and true FIA updates

Resolve/save the installed default capture catalog from a matching
graph-enabled config before switching to eager weight loading. The current
loader explicitly has eager mode/empty sizes, which cannot establish defaults.
The full Q4 trace needs up to 512 token rows; use only required entries from
the resolved default catalog. Never add special partition sizes.

Padding rounds `4*batch` to the smallest legal bucket. Preserve exact random
`request_ids` order: row position never identifies state. Inactive rows use
null GDN IDs/zero counts, zero FIA context lengths and `slot_mapping=-1`;
restore unspecified inactive outputs inside the timed path for both controls.
Only recurrence remaps candidate quartets to compact owners. Official conv
keeps original candidate IDs and accepted counts. All layers share generations.

For reported length `L`, write current FIA K/V at logical offsets `L-4..L-1`
through stable owner pages; use true host context length `L`. Previously
accepted progression overwrites rejected pending entries without resetting
paused state. Refresh fixed-address input/position/metadata buffers in place.

Split shapes must also come from the same catalog. Native request-range task
partitions inside a legal full graph do not create new capture shapes; report
this distinction. Assert all 16 FIA layer providers and every split occurrence
registered their tasks. Resolve/update true host query/context lengths for
all occurrences using `UpdatableGraph`, including update-stream waits; neither
updating one layer nor using the unimplemented legacy update API is valid.

## Deferred commit and complete measurement

Install a scoped recurrence dispatcher and real out-projection launch hook
for each GDN layer. Give concurrent metadata lanes independent scratch/events.
Order `products -> solve -> (projection/TP/MLP || commit)`; all base readers
and old-log readers finish before commit mutates their storage. Commit applies
the old accepted prefix, then publishes active staging records to fixed
canonical addresses. Inactive owners survive without whole-bank copies.
Join every pending commit before exposing the endpoint or next generation.
Commit runs exactly once per active owner/head; replay/retry is not idempotent.

Strong padding gets identical compact state and deferred commit overlap.
Additional split P/S-range overlap must beat this common control; a gain over
dense recurrence alone is not evidence of a parallel-specific improvement.

Time from before required CPU step preparation through complete device result
and every state join. Include host derivation, input/position preparation,
H2D copies, owner lookup, casts, output masking, all FIA updates, graph replay,
communication, waits and hidden/residual flow. Device-event timing is separate.
Load/prefix/capture/warmup and independent-trial restoration are setup costs.
Run continued complete trajectories; do not reset state before each step.
Repeat paired/rotated trials; retain raw samples, critical-rank mean,
P50/P95/P99, per-batch results and peak allocated/reserved memory.

Snapshot to CPU one layer/bounded tile at a time after commit join: all K/V,
conv, base, canonical logs/validity, generations/pending acceptance and any
persistent hidden/input state. Include staging banks for strict physical-state
comparison. Restore similarly; never retain GPU copies of all trajectories.
Export/hash candidates one layer at a time and release its roughly 774 MiB
export before another; cover full canonical bytes/shape/dtype/owner/generation.
Hash equality does not replace finite checks or numerical drift assessment.

## Required gates and unsupported paths

1. CPU contracts: direct ownership, registry, fixed-trace/first-use mapping,
   snapshots, complete metadata and explicit rejection paths.
2. TP2 allocation audit with 48 compact/16 FIA caches and no dense source;
   prove widest-bucket fit before capturing the complete required catalog.
3. Official coherent whole-model prefix, then one complete eager Q4 step;
   prove layer/provider invocation counts, populated caches and null safety.
4. Same-schedule repeated eager/replay exactness, continued random owners,
   every candidate, inactive/null/page canaries and finite checks. Observe
   final hidden/residual at the real normalization boundary. Assess
   cross-shape/algorithm drift separately from scheduling/state correctness.
5. Full bucket-dispatch/update/commit-join audit and paired fixed trajectories.
   Require repeatable >=5% mean complete-step reduction; target >=10%, report
   tails and memory. Local kernels/layers cannot satisfy this gate.

Reject mixed/non-spec decode, later prefill, stale generations, duplicate
concurrent owners, retries, connectors/offload, reassignment and foreign dense
consumers until explicitly implemented and costed. This manual benchmark owner
does not establish production KV-manager support. Native/layer qualification
alone proves none of the full-model gates; retain failures and keep the goal
active until the complete scope is demonstrated.
