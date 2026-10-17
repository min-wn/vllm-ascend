# Conditional ACLGraph qualification on the current A2 machine

## Purpose and limits

This investigation changes the optimization question from arranging existing
streams to avoiding execution of unnecessary graph work. A conditional parent
graph could select an active request block or a prepared subgraph without a
Python replay decision for each block. Unselected branches must actually skip
their tasks, rather than run kernels whose output is masked afterward.

The first probe qualifies only a raw IF graph with tiny D2D-copy branches. It
does not establish performance improvement, attention support, HCCL safety,
cross-stream model execution, or compatibility with the vLLM graph allocator.
FIA still requires its real host sequence lengths and existing replay/update
protocol. Conditional graph execution cannot remove that operator contract.

## Reproducible source evidence

The official CANN runtime repository is
<https://gitcode.com/cann/runtime>. Its GitHub mirror was inspected at commit
`b5c8dd9a4022aba8964feb2daa2dc8973f7cb214`:

- [Conditional IF/WHILE/SWITCH example](https://github.com/hicann/runtime/blob/b5c8dd9a4022aba8964feb2daa2dc8973f7cb214/example/2_advanced_features/model_ri/3_cond_model/README.md).
- [Complete conditional example source](https://github.com/hicann/runtime/blob/b5c8dd9a4022aba8964feb2daa2dc8973f7cb214/example/2_advanced_features/model_ri/3_cond_model/main.cpp).
- [Runtime support gate](https://github.com/hicann/runtime/blob/b5c8dd9a4022aba8964feb2daa2dc8973f7cb214/src/runtime/feature/aclgraph/capture_model_utils.cc).
- [Condition default initialization](https://github.com/hicann/runtime/blob/b5c8dd9a4022aba8964feb2daa2dc8973f7cb214/src/runtime/feature/aclgraph/cond_handle.cc).
- [Task update restrictions](https://github.com/hicann/runtime/blob/b5c8dd9a4022aba8964feb2daa2dc8973f7cb214/docs/zh/dev_guide/04-03_task_update.md).

The example support table explicitly lists Atlas A2 and A3 as supported, and
Ascend 950PR/950DT as unsupported for that example. This is a source-level
product contract; it is not evidence that the installed 24.1.rc3 driver and
firmware support the required runtime features.

Read-only inspection on `vllm-ascend-030` found these declarations in
`/usr/local/Ascend/ascend-toolkit/latest/include/acl/acl_rt.h` and corresponding
exports in `libascendcl.so`. The toolkit resolves to CANN 9.1.0. The installed
torch-npu 2.10.0.post4 Python graph wrapper does not expose these conditional
APIs. The benchmark therefore uses a standalone `ctypes` binding and does not
load torch, a model, or the vLLM runtime.

## ABI and feature checks

The relevant C signatures are:

```c
aclError aclmdlRICondHandleCreate(aclmdlRI modelRI,
    uint32_t defaultLaunchValue, aclmdlRICondHandleFlag flag,
    aclmdlRICondHandle *handle);
aclError aclmdlRICondHandleGetCondPtr(aclmdlRICondHandle handle,
    uint64_t **ptr);
aclError aclmdlRIAddCondTask(aclmdlRICondTaskParams params,
    aclrtStream stream, uint32_t flags);
aclError aclmdlRICaptureToModelRIBegin(aclrtStream stream,
    aclmdlRI modelRI, aclmdlRICaptureMode mode);
```

`aclmdlRICondTaskParams` contains a pointer handle, a 32-bit condition type,
a 32-bit branch count, and a pointer to the array of child model handles.
Its size is 24 bytes on the current 64-bit ABI, with fields at offsets
0, 8, 12, and 16. IF is type zero, thread-local capture mode is one, and
`ACL_MODEL_RI_COND_HANDLE_ASSIGN_DEFAULT` is flag one.

`CheckCaptureModelSupportCondOp` checks software-SQ support and
`TS_FEATURE_ACLGRAPH_COND_OP`. The software-SQ check additionally requires the
chip feature, `TS_FEATURE_SOFTWARE_SQ_ENABLE`, and the driver capability
`FEATURE_TRSDRV_SQ_SUPPORT_DYNAMIC_BIND`. A missing header or symbol and an
unsupported device/firmware return are distinct findings. The stable public
device-capability enum does not expose this complete conditional feature gate;
the smallest graph creation probe tests the actual gate.

`ACL_ERROR_RT_FEATURE_NOT_SUPPORT` is 207000. The probe records the exact API,
return value and `aclGetRecentErrMsg`. It labels that result `unsupported`,
which is a completed capability investigation, not a passed graph execution
test. Only `qualified_raw_conditional_IF` indicates successful execution.

`aclmdlRITaskDisable` is not a general execution-skip control: its public
contract disables a task's update flag. It must not be used as evidence for
CUDA-style graph-node enable/disable semantics.

## Minimal execution protocol

`benchmarks/multi_graph_acl_conditional_probe.py` creates one parent model,
two captured children, and a separate execution stream. All allocations and
handles remain alive until cleanup.

Each child copies a distinct integer sentinel into its own destination and
into a shared output. Before every replay the probe resets both destinations
to distinct canaries and the common output to a third canary. Conditions
follow `0, 1, 1, 0, 1, 0`, exercising changes and repeated values on the same
graph. Both output destinations are read back after a bounded synchronization.
The selected branch must publish its sentinel; the unselected branch's canary
must remain unchanged. The common output must match the selected branch.

The `ASSIGN_DEFAULT` flag restores the condition default before model
execution. Writing a condition pointer only before `aclmdlRIExecuteAsync`
would therefore fail to prove device-driven selection. This probe instead
captures a parent D2D copy from a persistent `conditionInput` buffer into the
runtime-owned condition pointer before the IF task. Every replay writes the
input buffer, and readback checks the condition value as well as the branch
outputs. The model and condition pointer identities must remain stable.

All branch operations are D2D copies. These are deliberately simpler than an
aclnn operator and test task selection without compiler or custom-kernel
dependencies. A successful result would still require separate qualification
with actual compute kernels, private graph pools, external events and HCCL
collectives before use in model execution.

## Bounded run and retained evidence

The launcher starts one isolated worker with a 90-second process timeout.
Every stream synchronization uses a 5-second timeout. The worker writes its
result before cleanup; if cleanup hangs, the parent retains that result inside
its timeout report. The probe never calls a device reset, changes the driver,
or runs concurrently with another NPU experiment.

Example, to be run only by the experiment coordinator on the reserved device:

```bash
python benchmarks/multi_graph_acl_conditional_probe.py --device 0 \
  --sync-timeout-ms 5000 --timeout-seconds 90 \
  --output /workspace/multi-graph-results/tp-boundary/a2-conditional-v1.json
```

The JSON records API results, failure location, recent runtime diagnostics,
source hash, replay checks, stable identities and cleanup results. A nonzero
or timeout result must remain visible rather than being silently downgraded
to a functional fallback. Timing is intentionally excluded from this first
compatibility gate.

## CPU verification

`tests/ut/worker/multi_graph/test_acl_conditional_probe.py` checks the actual
64-bit ABI, IF branch mapping and canary expectations. A fake ACL executor
models capture, default-condition reset, the parent condition copy, and
selected-child execution. It verifies 12 replays and cleanup through the
actual worker function. Negative controls deliberately ignore the condition
copy or run both children, and must be rejected. Another control returns
207000 at condition-handle creation and requires a recorded unsupported
result with no claimed replays.

These CPU tests validate the probe and its oracle. They do not replace the
hardware feature gate.

## Result

The isolated device probe completed on the current 910B3, driver 24.1.rc3,
CANN 9.1.0 environment. `aclmdlRICondHandleCreate` returned 207000, and the
runtime's recent error was:

```text
rtModelCondHandleCreate execution failed, the feature is not supported.
```

The report status is `unsupported`, cleanup completed successfully, and no
conditional branch replay was qualified. The failure identifies the actual
runtime feature gate, but the public error does not identify which of its
driver, firmware or conditional-feature checks failed.

The retained remote artifact is
`/workspace/multi-graph-results/tp-boundary/a2-conditional-v1.json`; its local
copy is `outputs/multi_graph_tp_attention_mlp/a2-conditional-v1.json` relative
to the outer project workspace. Tested probe SHA256:
`0fd464cb958fbb00792fc8ddb51184cb619adbbe1143179de57e13c43c511168`.
The focused CPU suite passed 10 tests before the device result was recorded.

This excludes the official conditional-ACLGraph route on the present software
and driver combination. It does not establish permanent lack of support on
all A2 hardware or other driver/firmware versions. No driver update or
conditional model optimization is enabled by this investigation.
