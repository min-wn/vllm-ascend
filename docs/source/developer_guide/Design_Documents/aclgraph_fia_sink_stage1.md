# FIA sink stage 1: isolated build and compatibility gate

Status: stage-one investigation completed, 2026-10-18. The isolated native
build and D128 constant eager smoke pass. **The actual Q4/D256/GQA6 target
fails qualification:** the pinned upstream host route explicitly rejects
D256, and bypassing that check would exceed the kernel's K L1 slot size.
No graph, dynamic correctness or performance qualification is claimed.

The evidence summary is
[`fia-sink-stage1-qualification.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage1-qualification.json).
Raw build, smoke, source manifests and selected CANN logs are preserved in
the parent workspace's `outputs/thesis_incremental_plan/stage1-evidence/`
and the remote isolated root below.

## Source and isolation

The source is pinned to vllm-ascend PR 15336 commit
`bec19b8dea1e538e64838f3064f7e91a3fa9ee72`. The 15,437,997-byte source archive
has SHA256
`ae9570e8bfb601e225edee81d874f336f4252add2a2d080a41193be5462cb2b9`.
The independent remote root is
`/workspace/aclgraph-sink-stage1-bec19b8dea1e`; the native build runs under
`source/csrc`. It does not build from or install over the existing working
repository, installed Python extension, or system OPP.

```bash
cd /workspace/aclgraph-sink-stage1-bec19b8dea1e/source/csrc
MAX_JOBS=4 MAKEFLAGS=-j4 CMAKE_BUILD_PARALLEL_LEVEL=4 \
  bash build.sh --pkg \
    --ops='fused_infer_attention_score_v2_sink;fused_infer_attention_score_v2_sink_metadata' \
    --soc=ascend910b --vendor_name=thesis_fia_stage1 -j4
```

Only the two selected operators are requested. The complete pinned CANN
build infrastructure and the three sink source families are retained for
dependency completeness. Existing `json`, `makeself`, `abseil-cpp`,
`ascend_protobuf`, and `pkg` dependency caches are copied into the isolated
tree when available; their origin remains read-only. The copied dependency
inventory is recorded in `third-party-cache-copy.json`.

The inner make and CMake limits above correct serial nested compiler
orchestration. They change build scheduling, not operator arithmetic,
tiling keys, or runtime scheduling. The independent host extension uses
`MAX_JOBS=2`.

## Audited compatibility changes

1. In the metadata AICPU source, replace the invalid relative include
   `../../common/aicpu/cpu_context_util.h` with
   `../aicpu/cpu_context_util.h`. The destination is the operator-local
   header already shipped by the pinned commit; its SHA256 is
   `bb1f33fa2c373999b371102147c763d7196b69e226401873a371705fa52d2d54`.
   No header is copied or synthesized.
2. Compile the host bindings into dispatcher namespace
   `thesis_fia_sink_stage1`, with separate binding and opapi C++ namespaces.
   Both schemas come from the pinned registration fragment. The installed
   `_C_ascend` extension is not modified.
3. In the isolated `EXEC_NPU_CMD_V1` macro, assign the outer owning workspace
   tensor instead of declaring a shadowing local tensor, and capture that
   tensor by value in the queued `OpCommand` callback. This repairs workspace
   ownership for the experiment; it is not a measured performance change.

`source-patches.json`, `extension-source-manifest.json`, and
`extension-source.patch` record the native include change and transformed
host-binding source hashes. Four missing `const_def.h` references remain in
unreferenced legacy copy headers; no speculative substitution was made.

## Completed host checks

The independent extension compiled and loaded successfully. Dispatcher
inspection found both isolated schemas and their `PrivateUse1` and `Meta`
implementations, without executing an NPU operator:

- `thesis_fia_sink_stage1::npu_fused_infer_attention_score_v2_sink`
- `thesis_fia_sink_stage1::_npu_fused_infer_attention_score_v2_sink_metadata`

A separate host compile passed static assertions against the installed
torch_npu headers for descriptor size/alignment and the offsets of
`storage_sizes_`, `npu_format_`, `origin_format_`, and `npu_desc_`.
This covers the storage fields used by the shim. It does not establish
general private-ABI compatibility, stream/callback lifetime safety, or
graph execution correctness. Evidence: `extension-status.json` and
`storage-abi-check.json` with their source and installed-header hashes.

## Native and device results

| Gate | Final status and evidence |
| --- | --- |
| Native two-operator package build | Pass, return code 0. Final package is 19,392,188 bytes. |
| Isolated OPP staging | Pass. Native opapi, proto, tiling and named AICPU library/JSON are present under the isolated prefix. |
| Native ELF symbols | Pass. Both V3 attention and metadata entrypoint pairs resolve from the selected isolated opapi ELF. |
| Independent Torch extension | Pass. Both schemas have NPU and Meta registrations; used storage-field layout assertions compile. |
| Constant eager smoke: D128 GQA4 | Pass. BF16 output `[8,4,128]`, finite, maximum absolute error from 1 is 0. |
| Constant eager smoke: Q4/D256/GQA6 | Metadata call and synchronization complete; attention fails during host tiling. Target qualification fails. |

The generated package file SHA256 is
`72cc23bc8681b18a031e38397178bc12df3721e610e615567f7a5ed308f3ea09`.
The staged `libcust_opapi.so` SHA256 is
`db876e892f7256ff97ca549cd03d9972b37af4afd48fbe88094778830e76ef6e`.
The independently compiled Torch extension SHA256 is
`25a41e47473a3bc9c34a50c053766181c7a764ea496b93d4d3b70180b0ecf76b`.

The smoke exposes physical device 0 only: Ascend910B3, CANN 9.1.0,
torch 2.10.0+cpu, torch_npu 2.10.0.post4, current driver 24.1.rc3.
The installed stream reports 20 Cube and 40 Vector cores; these actual
limits are passed to metadata instead of the binding's 24/48 defaults.
Both cases use batch 1, KV length 516, page 128, TND query, paged ND KV,
noncausal attention, Q/K zeros and V ones. D128 uses Q8, 4 Q heads and
one KV head. D256 uses Q4, 12 Q heads and two KV heads.

Fresh-process checks verify `ASCEND_CUSTOM_OPP_PATH`, every required symbol
with `dladdr`, extension hashes, schemas and `/proc/self/maps`. Python startup
on this image prepends another vendor's OPP through a `.pth` hook. The smoke
records that value, then restores the selected isolated vendor before
importing Torch or loading native libraries. It does not edit the hook or
global configuration.

The executed smoke SHA256 is
`4420f2e62f6bef5b7d4d8d02bb652bae682d9ad6144bd84a97f3dda7d6c7e661`.
Its raw JSON keeps the successful D128 case and the D256 attention traceback.
The D256 metadata checkpoint is recoverable from the executed source: the
metadata call, synchronization and shape/nonzero checks all precede the
failing attention call. A later saved-tool improvement records these stages
explicitly and validates output-directory containment before creating it;
that helper revision has CPU checks but has not been rerun on the NPU.

## Why the D256 target fails

The compiled source files match the pinned source hashes. The failure chain
is specific and reproducible from that source:

1. [`CheckGqaDSupport`, lines 180-199][guard] allows `(Dqk,Drope,Dv)` values
   `(128,0,128)`, `(64,0,64)`, `(128,64,128)`, `(192,0,128)` and
   `(192,0,192)`. It does not allow `(256,0,256)`.
2. Ordinary BF16 GQA reaches this check through `CheckGqaConstrain` and
   `RouteToFia`. The [outer tiling function][route] returns `GRAPH_FAILED`
   when the route is rejected. The separate metadata operator accepting
   positive head dimensions does not establish attention-kernel support.
3. CANN logs for the failed case report `Failed to execute tiling`, followed
   by workspace-size and output-null errors. These are downstream symptoms
   of host tiling failure; they do not establish a device kernel crash or
   a driver incompatibility.

Adding D256 to that list is insufficient. The [consumer][consumer] uses
`FiaBlockCubeNonQuantGqa<FIAT>` for its matmul service. In the [Cube block][cube],
the actual `kpL1Tensor` slot and allocation use
`L1_SINK_KP_SIZE = 128 * 192` BF16 elements, or 48 KiB. A complete N block
of `CopyKToL1` at D256 requires `128 * 256` BF16 elements, or 64 KiB.
That is 16 KiB beyond one slot. The separately declared `L1_KP_SIZE` constant
does not size this actual allocation. The current host check prevents entry
to this unsupported configuration; no unsafe execution was attempted.

The earlier audit inferred that D256 could use `GENERAL_GQA` because the
downstream parser and template branch existed. **That inference was wrong:**
the earlier host route prevents reaching it, and the actual buffer contract
is too small. This report supersedes that eligibility statement.

Supporting the target requires an explicit kernel/tiling design: audit and
resize or repartition Q/K/V/P buffers, L1/UB/L0 usage, matmul tiles and pipeline
lifetimes together. Then validate numerical and memory safety before graph
capture. It is a separate implementation milestone, not a one-line whitelist
change or an incremental planner optimization.

## Reusable tools and stage boundary

`benchmarks/aclgraph_fia_sink_stage1_extension.py` generates the isolated
host sources and starts the extension build. It intentionally refuses to
overwrite a previous build record. The pinned source, include correction
and separate native build must already exist in the fixed remote root.

`benchmarks/aclgraph_fia_sink_stage1_smoke.py --help` documents the fresh-process
smoke arguments. The runtime vendor leaf is
`/workspace/aclgraph-sink-stage1-bec19b8dea1e/opp/packages/vendors/thesis_fia_stage1_transformer`.
The smoke's nonzero exit is a real target failure; do not reclassify it as
a passing combined test because D128 passed.

Scoped Ruff, compilation, forbidden-import and long-function checks pass;
the saved smoke's CPU helper checks pass. Required `bash format.sh ci` was
attempted: Ruff and Markdown pass, while the full check fails on existing
unrelated spelling/import findings and unavailable Windows shell/tool hooks.
The complete log and its hash are retained in the evidence summary.

The smoke checks constant inputs against analytically expected
outputs. A passing package build or constant eager smoke establishes only
that specific integration gate. It does not qualify varying sequence
lengths, causal masks, ragged batches, changing pages, inactive/recycled
requests, continued state, ACLGraph capture/replay, or performance.

No incremental native metadata producer or exporter is qualified in this
stage. The proposed scheduling algorithm has no new NPU speedup result.
Dynamic correctness and performance require separate controls against
legacy updated FIA and shared full-plan sink execution, including all
planning, copies, synchronization, scratch, and merge costs.

Stage one ends here. Resolve or explicitly change the unsupported target
kernel before starting target ACLGraph or incremental native-planner tests.
The existing legacy-FIA model path is retained; this stage does not extend
its correctness qualification.

[guard]: https://github.com/vllm-project/vllm-ascend/blob/bec19b8dea1e538e64838f3064f7e91a3fa9ee72/csrc/attention/fused_infer_attention_score_v2_sink/op_host/fused_infer_attention_score_v2_sink_tiling_v3.cpp#L180
[route]: https://github.com/vllm-project/vllm-ascend/blob/bec19b8dea1e538e64838f3064f7e91a3fa9ee72/csrc/attention/fused_infer_attention_score_v2_sink/op_host/fused_infer_attention_score_v2_sink_tiling.cpp#L32
[consumer]: https://github.com/vllm-project/vllm-ascend/blob/bec19b8dea1e538e64838f3064f7e91a3fa9ee72/csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fia_kernel_nonquant_sink.h#L115
[cube]: https://github.com/vllm-project/vllm-ascend/blob/bec19b8dea1e538e64838f3064f7e91a3fa9ee72/csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fia_block_cube_nonquant_gqa_sink.h#L345
