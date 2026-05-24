# inplace serial offset graph on-demand lazy capture 修改报告

本文记录本次针对 `inplace_serial` split-1 offset ACL graph 的修改。

目标不是在初始化阶段预捕获 offset 图，而是在 split-1 进入普通
`self.model()` 执行路径之前，先判断对应 offset 图是否存在。若不存在，则进入
一个单独的 on-demand lazy capture 流程：先按普通 capture 路径做 eager warmup，
再做 ACL graph capture，capture 当次直接返回捕获输出，不在 capture 后立即
replay。

## 背景

之前的失败发生在 split-1 offset graph 首次 lazy capture 内：

```text
RuntimeError: copy_between_host_and_device_opapi ... aclrtMemcpy, error code is 107030
Not allow to synchronize captured-stream
When layout is TND and PA not enabled, keyT(256) and valueT(256) must be equal
to the last element of actualSeqenceLengthKV(9)
```

核心问题是：原来的 runtime lazy capture 直接在正常 `self.model()` 路径里触发，
没有单独 warmup，也容易把 capture、stream synchronize、真实 decode metadata
混在一起。它和启动阶段预捕获流程不一致。

## GPU 流程对照

GPU 初始化预捕获在 `vllm/v1/worker/gpu_model_runner.py` 的
`_capture_cudagraphs()` 中有 warmup：

```python
for _ in range(self.compilation_config.cudagraph_num_of_warmups):
    self._dummy_run(
        num_tokens,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
        force_attention=cudagraph_runtime_mode == CUDAGraphMode.FULL,
        ...
    )
self._dummy_run(
    num_tokens,
    cudagraph_runtime_mode=cudagraph_runtime_mode,
    is_graph_capturing=True,
    ...
)
```

但 GPU `CUDAGraphWrapper.__call__()` 中 runtime 缺图捕获是直接 capture：

```python
if entry.cudagraph is None:
    validate_cudagraph_capturing_enabled()
    cudagraph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cudagraph, pool=self.graph_pool):
        output = self.runnable(*args, **kwargs)
    entry.output = weak_ref_tensors(output)
    entry.cudagraph = cudagraph
    return output
```

因此可以明确区分：

- 预捕获路径：有 warmup。
- wrapper runtime lazy capture：直接 capture。

本次 NPU 修改选择让 split-1 offset graph 的 on-demand lazy capture 接近预捕获
形态，而不是继续依赖 wrapper 在正常 `self.model()` 路径内直接捕获。

## 修改方案

### 1. ACLGraphWrapper 增加图存在性查询

`ACLGraphWrapper` 新增 `has_graph(batch_descriptor, in_parallel_streams)`，供
model runner 在进入 `self.model()` 前查询具体 `BatchDescriptor` 是否已经有
真实 ACL graph。

这避免 split-1 offset 图缺失时直接进入普通 replay/capture 路径。

### 2. capture 当次返回 capture output，不立即 replay

`ACLGraphWrapper.__call__()` 的 capture 分支保持 GPU wrapper 语义：

```python
entry.output = weak_ref_tensors(output)
entry.aclgraph = aclgraph
return output
```

也就是说，首次 capture 当次使用 capture 过程中产生的输出；后续命中同一个
descriptor 时才走 replay。这样可以避免 capture 结束后立刻 replay，又立刻遇到
attention graph task update / stream 同步时序问题。

### 3. split-1 缺图时进入独立 on-demand capture 流程

`NPUModelRunner` 新增三段逻辑：

- `_has_aclgraph_for_context(context)`：透传到 `self.model.has_graph()`。
- `_needs_inplace_serial_offset_capture(metadata)`：判断当前 split 是否是
  `start_num_tokens > 0`、`CUDAGraphMode.FULL`、已授权 lazy capture、且图不存在。
- `_run_inplace_serial_offset_capture(...)`：单独执行 warmup + capture。

on-demand capture 流程：

1. 保存当前 `context.cudagraph_runtime_mode` 和 `context.capturing`。
2. 在 `self.stream_main` 上执行 `cudagraph_num_of_warmups` 次 eager forward：
   `context.cudagraph_runtime_mode = CUDAGraphMode.NONE`。
3. 恢复 runtime mode 为原始 FULL，执行一次 `self.model()` 触发 ACL graph capture。
4. 同步 stream，恢复 context 状态。
5. 校验对应 descriptor 的 ACL graph 已经创建。
6. 返回 capture 当次输出。

### 4. split loop 在进入普通 self.model 前先分流

`_run_split_batch_inplace_serial()` 的执行顺序调整为：

```python
for split:
    if _needs_inplace_serial_offset_capture(metadata):
        split_result = _run_inplace_serial_offset_capture(...)
    else:
        if offset FULL graph still missing:
            raise RuntimeError(...)
        with override_forward_context(metadata.context):
            split_result = self.model(...)
            if FULL:
                _update_attn_params_for_split_ubatch(...)
```

这样 split-1 首次缺图不会进入普通 `self.model()` 路径；普通路径只处理已有 graph
的 replay，或非 FULL/eager 路径。

## 详细 diff

以下是本次方案相关的核心 diff 摘录。仓库当前工作区包含之前 inplace split 多阶段
改动，完整 `git diff` 会包含大量既有内容；这里仅列出本次 on-demand lazy capture
相关修改。

### `vllm_ascend/compilation/acl_graph.py`

```diff
@@
 class ACLGraphWrapper:
@@
     def unwrap(self) -> Callable:
         # in case we need to access the original runnable.
         return self.runnable
 
+    def has_graph(self,
+                  batch_descriptor: Optional[BatchDescriptor],
+                  in_parallel_streams: bool = False) -> bool:
+        """Return whether a concrete ACL graph has already been captured."""
+        if batch_descriptor is None:
+            return False
+        entries = (self.concrete_aclgraph_entries2
+                   if in_parallel_streams else self.concrete_aclgraph_entries)
+        entry = entries.get(batch_descriptor)
+        return entry is not None and entry.aclgraph is not None
+
     def __call__(self, *args, **kwargs):
         forward_context = get_forward_context()
```

```diff
@@
             entry.output = weak_ref_tensors(output)
             entry.aclgraph = aclgraph
 
             compilation_counter.num_cudagraph_captured += 1
@@
            # important: we need to return the output, rather than
            # the weak ref of the output, so that pytorch can correctly
            # manage the memory during acl graph capture
             return output
```

当前实现中 capture 分支保持为通用 GPU/NPU capture 语义：写入
`entry.output` / `entry.aclgraph` 后直接 `return output`，没有单独的
capture 后立即 replay 分支。

### `vllm_ascend/worker/model_runner_v3.py`

```diff
@@
     def _merge_split_outputs(self, outputs: list[Any]) -> Any:
         ...
         return torch.cat(outputs, dim=0)
 
+    def _has_aclgraph_for_context(self, context: Any) -> bool:
+        has_graph = getattr(self.model, "has_graph", None)
+        if not callable(has_graph):
+            return False
+        return bool(
+            has_graph(
+                getattr(context, "batch_descriptor", None),
+                bool(getattr(context, "in_parallel_streams", False)),
+            ))
+
+    def _needs_inplace_serial_offset_capture(
+            self, metadata: AscendUbatchMetadata) -> bool:
+        context = metadata.context
+        batch_descriptor = getattr(context, "batch_descriptor", None)
+        if batch_descriptor is None:
+            return False
+        if getattr(context, "cudagraph_runtime_mode",
+                   CUDAGraphMode.NONE) != CUDAGraphMode.FULL:
+            return False
+        if int(getattr(batch_descriptor, "start_num_tokens", 0) or 0) <= 0:
+            return False
+        if not bool(getattr(context, "allow_inplace_lazy_capture", False)):
+            return False
+        return not self._has_aclgraph_for_context(context)
```

```diff
@@
+    def _run_inplace_serial_offset_capture(
+            self,
+            metadata: AscendUbatchMetadata,
+            split_slice: Any,
+            model_kwargs: dict[str, Any],
+    ) -> Any:
+        """Warm up and capture a missing inplace offset graph on demand.
+
+        The normal GPU/NPU preload path runs eager warmups before graph
+        capture.  Split-1 offset graphs are intentionally not pre-captured, so
+        this path reproduces that warmup-before-capture shape before the split
+        enters the regular replay branch.
+        """
+        context = metadata.context
+        batch_descriptor = getattr(context, "batch_descriptor", None)
+        warmups = int(
+            getattr(self.compilation_config, "cudagraph_num_of_warmups", 0)
+            or 0)
+        previous_mode = getattr(context, "cudagraph_runtime_mode",
+                                CUDAGraphMode.NONE)
+        previous_capturing = bool(getattr(context, "capturing", False))
+        step_id = _split_debug_step_from_runner(self)
+
+        if split_debug.is_enabled():
+            split_debug.log_event(
+                "inplace_lazy_capture_prepare",
+                {
+                    "batch_descriptor":
+                    split_debug.batch_descriptor_info(batch_descriptor),
+                    "warmups": warmups,
+                    "num_tokens": int(split_slice.num_tokens),
+                    "graph_num_tokens": int(split_slice.graph_num_tokens),
+                    "start_num_tokens":
+                    int(getattr(batch_descriptor, "start_num_tokens", 0)
+                        or 0),
+                },
+                step_id=step_id,
+            )
+
+        capture_result = None
+        try:
+            with torch.npu.stream(self.stream_main):
+                for warmup_idx in range(warmups):
+                    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
+                    context.capturing = False
+                    with override_forward_context(context):
+                        _ = self.model(
+                            input_ids=metadata.input_ids,
+                            positions=metadata.positions,
+                            inputs_embeds=metadata.inputs_embeds,
+                            intermediate_tensors=metadata.intermediate_tensors,
+                            **model_kwargs,
+                        )
+                    self.stream_main.synchronize()
+                    if split_debug.is_enabled():
+                        split_debug.log_event(
+                            "inplace_lazy_capture_warmup",
+                            {
+                                "warmup_idx": warmup_idx,
+                                "batch_descriptor":
+                                split_debug.batch_descriptor_info(
+                                    batch_descriptor),
+                            },
+                            step_id=step_id,
+                        )
+
+                context.cudagraph_runtime_mode = previous_mode
+                context.capturing = False
+                with override_forward_context(context):
+                    capture_result = self.model(
+                        input_ids=metadata.input_ids,
+                        positions=metadata.positions,
+                        inputs_embeds=metadata.inputs_embeds,
+                        intermediate_tensors=metadata.intermediate_tensors,
+                        **model_kwargs,
+                    )
+                self.stream_main.synchronize()
+        finally:
+            context.cudagraph_runtime_mode = previous_mode
+            context.capturing = previous_capturing
+
+        if not self._has_aclgraph_for_context(context):
+            raise RuntimeError(
+                "Inplace serial offset graph capture did not create an ACL "
+                f"graph entry for {batch_descriptor!r}")
+
+        if split_debug.is_enabled():
+            split_debug.log_event(
+                "inplace_lazy_capture_complete",
+                {
+                    "batch_descriptor":
+                    split_debug.batch_descriptor_info(batch_descriptor),
+                    "num_tokens": int(split_slice.num_tokens),
+                    "graph_num_tokens": int(split_slice.graph_num_tokens),
+                },
+                step_id=step_id,
+            )
+        return capture_result
```

```diff
@@
         try:
             for slice_idx, split_slice in enumerate(split_batch_slices):
                 metadata = ubatch_metadata[slice_idx]
-                with torch.npu.stream(self.stream_main):
-                    with override_forward_context(metadata.context):
-                        split_result = self.model(...)
-                        if metadata.context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
-                            self._update_attn_params_for_split_ubatch(...)
+                if self._needs_inplace_serial_offset_capture(metadata):
+                    split_result = self._run_inplace_serial_offset_capture(
+                        metadata,
+                        split_slice,
+                        model_kwargs,
+                    )
+                else:
+                    if (int(
+                            getattr(metadata.context.batch_descriptor,
+                                    "start_num_tokens", 0) or 0) > 0
+                            and metadata.context.cudagraph_runtime_mode
+                            == CUDAGraphMode.FULL
+                            and not self._has_aclgraph_for_context(
+                                metadata.context)):
+                        raise RuntimeError(
+                            "Missing inplace serial offset ACL graph before "
+                            "normal replay path: "
+                            f"{metadata.context.batch_descriptor!r}")
+                    with torch.npu.stream(self.stream_main):
+                        with override_forward_context(metadata.context):
+                            split_result = self.model(
+                                input_ids=metadata.input_ids,
+                                positions=metadata.positions,
+                                inputs_embeds=metadata.inputs_embeds,
+                                intermediate_tensors=metadata.intermediate_tensors,
+                                **model_kwargs,
+                            )
+                            if (metadata.context.cudagraph_runtime_mode
+                                    == CUDAGraphMode.FULL):
+                                self._update_attn_params_for_split_ubatch(
+                                    metadata.context,
+                                    split_slice.graph_num_tokens,
+                                    parallel_streams=False)
+                self.stream_main.synchronize()
                 results.append(
                     self._trim_split_output(split_result,
                                             split_slice.num_tokens))
```

### FIA TND seq len 模板化修正

复跑时如果仍在 capture 内看到：

```text
When layout is TND and PA not enabled, keyT(256) and valueT(256) must be equal
to the last element of actualSeqenceLengthKV(9)
```

说明 offset capture 已经进了 FIA TND op，但 `actual_seq_lengths_kv[-1]` 仍是
真实 decode seqlen，而不是 graph/capture T。之前模板化代码有两个问题：

- runner 里提前模板化使用的是 `self.block_size`。
- attention capture/update 路径里传给 `maybe_template_fia_seq_lens()` 的也是
  `block_size` / `key.shape[1]`。

这两个值都不是报错里的 `keyT`。FIA TND 约束要求最后一个 KV seqlen 等于
key/value 的 T 维，因此本次补充修正为按 `key.shape[0]` 或 split graph size
模板化。

核心 diff：

```diff
@@
-def maybe_template_fia_seq_lens(forward_context: Any, seq_lens: Any,
-                                block_size: int) -> Any:
+def _get_fia_key_t(key_tensor: Any, fallback: int) -> int:
+    if isinstance(key_tensor, torch.Tensor) and key_tensor.ndim > 0:
+        return int(key_tensor.shape[0])
+    return int(fallback)
+
+
+def maybe_template_fia_seq_lens(forward_context: Any, seq_lens: Any,
+                                target_t: int) -> Any:
@@
     templated_seq_lens = list(seq_lens)
-    templated_seq_lens[-1] = int(block_size)
+    # FIA TND requires actualSeqenceLengthKV[-1] to match key/value T.
+    templated_seq_lens[-1] = int(target_t)
     return templated_seq_lens
```

```diff
@@
-        actual_seq_lengths_kv = maybe_template_fia_seq_lens(
-            forward_context, actual_seq_lengths_kv, block_size)
+        actual_seq_lengths_kv = maybe_template_fia_seq_lens(
+            forward_context, actual_seq_lengths_kv, int(key.shape[0]))
@@
-            actual_seq_lengths_kv = maybe_template_fia_seq_lens(
-                forward_context, actual_seq_lengths_kv, int(key.shape[1]))
+            actual_seq_lengths_kv = maybe_template_fia_seq_lens(
+                forward_context, actual_seq_lengths_kv, int(key.shape[0]))
```

```diff
@@
-def _template_fia_seq_lens_list(attn_metadata: Any, block_size: int) -> int:
+def _template_fia_seq_lens_list(attn_metadata: Any, target_t: int) -> int:
@@
-        templated_seq_lens[-1] = int(block_size)
+        templated_seq_lens[-1] = int(target_t)
@@
-                templated_fia_seq_lens = _template_fia_seq_lens_list(
-                    ubatch_attn_metadata, self.block_size)
+                templated_fia_seq_lens = _template_fia_seq_lens_list(
+                    ubatch_attn_metadata, split_slice.graph_num_tokens)
```

### query_start_loc metadata 地址稳定性修正

FIA TND seq len 修正后，后续 replay 继续触发 ptr 校验：

```text
AssertionError: Attention metadata addresses for inplace aclgraphs are different
during replay.
mismatches: attn_metadata.model.layers.0.self_attn.attn.query_start_loc
```

这说明 on-demand capture 已经创建 offset graph，后续 replay 也命中了同一个
descriptor；失败点变成 attention metadata 中 `query_start_loc` 的设备地址在
capture 和 replay 之间变化。

根因在 `AscendAttentionMetadataBuilder.build()`：即使 `inplace_serial`
split-1 的 `AscendCommonAttentionMetadata.query_start_loc` 已经由
`_stabilize_inplace_common_attn_metadata()` 放进稳定 secondary buffer，builder
仍每次执行一次 CPU pinned memory 到 NPU 的临时拷贝：

```python
query_start_loc = query_start_loc_cpu.pin_memory().to(
    self.device, non_blocking=True)
```

这会为每次 build 生成新的 NPU tensor 地址，绕过稳定 buffer。

已改为优先复用 common metadata 中的稳定 NPU view，只在 device 不匹配时保留
fallback：

```diff
@@
-        # TODO: Yet another unnecessary H2D while we already have a query_start_loc on device
-        query_start_loc = query_start_loc_cpu.pin_memory().to(
-            self.device, non_blocking=True)
+        query_start_loc = common_attn_metadata.query_start_loc[:num_reqs + 1]
+        if query_start_loc.device != self.device:
+            query_start_loc = query_start_loc_cpu.pin_memory().to(
+                self.device, non_blocking=True)
```

这样 split-1 capture/replay 都会引用同一个 secondary `query_start_loc` buffer，
ptr 校验不应再在该字段失败。如果下一轮仍报 metadata ptr mismatch，应继续按
`mismatch_detail["path"]` 定位下一个漂移字段。

## 行为变化

- split-0：仍必须命中启动阶段已有普通 graph，不因本次改动自动 lazy capture。
- split-1：只有 offset descriptor、FULL graph、显式
  `allow_inplace_lazy_capture=True` 且图不存在时，才走 on-demand capture。
- on-demand capture：先 warmup，再 capture；capture 当次不 replay。
- 后续 decode step：同一 offset descriptor 已有图，走普通 replay + graph task
  update。
- 如果 offset FULL graph 缺失但 context 未授权 lazy capture，会在进入普通
  `self.model()` 前抛错，避免隐式捕获。

## 风险与假设

- warmup 使用真实 split-1 输入和 metadata，会多次写同一批 KV cache slot。这里的
  假设是：同一个 decode step、同一 slot、同一 token 数据重复写入是幂等的。如果后续
  发现某些 attention/cache op 有非幂等副作用，需要为 on-demand capture 引入更独立
  的 dummy metadata 或禁用 warmup。
- 首次遇到某个 offset descriptor 会增加一次性延迟，延迟约等于
  `cudagraph_num_of_warmups + 1` 次 split-1 forward。
- FIA 的 `capture_metadata_mode="template"` 仍依赖现有 metadata templating 和
  GraphParams key 逻辑，本次只改变捕获时机，不重写 FIA tiling 约束。
- 未在本轮执行真实 NPU correctness，因为当前环境没有重新跑用户给出的完整脚本。

## 验证

已执行静态编译检查：

```bash
python -m py_compile \
  vllm-ascend/vllm_ascend/compilation/acl_graph.py \
  vllm-ascend/vllm_ascend/attention/attention_v1.py \
  vllm-ascend/vllm_ascend/worker/model_runner_v3.py
```

结果：通过。

建议下一步在 NPU 上复跑原先固定 batch correctness：

```bash
python examples/test_split_batch_correctness_npu.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --split-mode inplace_serial \
  --fixed-batch-size 416 \
  --capture-sizes 256,384,512 \
  --max-tokens 8 \
  --validate-ptrs \
  --force-fixed-prompts \
  --output-dir /tmp/vllm_ascend_inplace_on_demand_lazy_capture
```

期望 split debug 中能看到：

- 首次 offset descriptor：`inplace_lazy_capture_prepare`、
  `inplace_lazy_capture_warmup`、`inplace_lazy_capture_complete`。
- 后续同 descriptor decode step：不再出现 prepare/warmup/complete，直接 replay。
