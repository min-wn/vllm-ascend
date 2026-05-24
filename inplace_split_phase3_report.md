# inplace split 阶段 3 完成报告

阶段 3 的目标是扩展 `CudagraphDispatcher`，支持显式的 inplace offset
lazy key。

本阶段只完成 dispatcher key 能力和测试覆盖；没有启用 inplace split，没有修改
Ascend `GraphParams` key，没有修改 `ACLGraphWrapper` capture/replay 语义，也
没有接入 split planner、input slicing 或 metadata stable buffer。

## 完成内容

### 1. 扩展 CudagraphDispatcher.dispatch()

修改文件：

```text
/vllm-workspace/vllm/vllm/v1/cudagraph_dispatcher.py
```

`dispatch()` 新增 keyword-only 参数：

```python
start_num_tokens: int = 0
allow_inplace_lazy_key: bool = False
```

默认调用保持旧行为。只有调用方显式传入：

```python
start_num_tokens > 0
allow_inplace_lazy_key=True
```

才会进入 offset key 分支。

### 2. 新增 offset descriptor 构造逻辑

新增 helper：

```python
_create_inplace_offset_batch_descriptor()
_dispatch_inplace_offset_key()
```

offset key 使用真实 `num_tokens`，不调用
`_create_padded_batch_descriptor()`，因此不会 padding 到 capture size。

示例返回 key：

```python
BatchDescriptor(
    num_tokens=3,
    num_reqs=3,
    uniform=True,
    has_lora=False,
    start_num_tokens=8,
)
```

### 3. 安全约束

offset lazy key 只允许：

- `start_num_tokens > 0`
- `allow_inplace_lazy_key=True`
- `uniform_decode=True`
- `disable_full=False`
- `cudagraph_mode.decode_mode() == CUDAGraphMode.FULL`
- `num_tokens <= max_cudagraph_capture_size`
- `num_tokens % uniform_decode_query_len == 0`

不满足显式 allow、`disable_full=True`、非 FULL decode mode 或超过最大 capture
size 时，返回 `CUDAGraphMode.NONE`，并保留
`BatchDescriptor(num_tokens=..., start_num_tokens=...)` 方便日志定位。

非 uniform decode 或 speculative query length 不能整除时，当前实现抛
`ValueError`，避免错误调用被静默降级。

### 4. 新增 dispatcher 测试

修改文件：

```text
/vllm-workspace/vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py
```

新增测试覆盖：

- 默认 dispatch 返回 key 的 `start_num_tokens == 0`。
- offset lazy key 可注册并返回 FULL。
- offset key 不 padding。
- 未显式 `allow_inplace_lazy_key=True` 时不注册 offset key。
- 同一 offset key 重复 dispatch 不增加重复 key。
- 相同 `num_tokens`、不同 `start_num_tokens` 是不同 key。
- 非 uniform decode 拒绝 offset key。
- `disable_full=True` 不注册 offset key。
- PIECEWISE mode 不注册 offset FULL key。
- 超过最大 capture size 不注册 offset key。
- speculative decode query length 不能整除时拒绝 offset key。

## 验证结果

已执行：

```bash
PYTHONPATH=/vllm-workspace/vllm:$PYTHONPATH python -m py_compile \
  vllm/v1/cudagraph_dispatcher.py \
  tests/v1/cudagraph/test_cudagraph_dispatch.py
```

结果：

```text
py_compile passed
```

尝试执行：

```bash
PYTHONPATH=/vllm-workspace/vllm:$PYTHONPATH python -m pytest \
  tests/v1/cudagraph/test_cudagraph_dispatch.py \
  -k "TestCudagraphDispatcher or batch_descriptor"
```

结果：未执行成功。collection 阶段失败，原因是当前环境缺少：

```text
ModuleNotFoundError: No module named 'tblib'
```

为覆盖同等关键断言，已执行不依赖 pytest/conftest 的 Python 校验脚本，验证：

- 默认 key offset 为 `0`。
- offset lazy key 返回 `CUDAGraphMode.FULL`。
- offset key 被加入 `cudagraph_keys[CUDAGraphMode.FULL]`。
- offset key 不 padding。
- 重复 offset key 不重复注册。
- 不同 offset 生成不同 key。
- 未允许 lazy 时不注册 key。
- `disable_full=True` 不注册 key。
- PIECEWISE mode 不注册 offset FULL key。
- 超过最大 capture size 不注册 key。
- 非 uniform decode 和 query length 不能整除时拒绝。

结果：

```text
phase3 dispatcher checks passed
```

## 未改变内容

阶段 3 明确未改变以下内容：

- 未修改 Ascend `GraphParams` key 类型。
- 未修改 `ACLGraphWrapper.__call__()`。
- 未新增真实 ACL graph lazy capture guard。
- 未修改 split planner。
- 未启用 `inplace_serial` 或 `inplace_parallel`。
- 未实现 inplace input offset view。
- 未实现 metadata stable buffer。

## 风险和注意事项

- 阶段 3 后 dispatcher 已能生成 `start_num_tokens > 0` 的 FULL key，但
  Ascend attention `GraphParams` 仍按 `int num_tokens` 索引，真实 inplace
  replay 仍不能启用。
- offset key 当前只在调用方显式传入 `allow_inplace_lazy_key=True` 时生成，
  因此默认路径和 parallel-buffer split 不受影响。
- 当前环境缺少 `tblib`，vLLM 侧 pytest 不能 collection；如需运行完整 pytest，
  可安装：

```bash
python -m pip install tblib
```

## 阶段结论

阶段 3 已完成。

当前 dispatcher 已具备表达和注册 inplace offset lazy key 的能力。下一步应进入
阶段 4：扩展 Ascend `GraphParams`，支持 descriptor-aware key，避免不同
`start_num_tokens` 的 graph 复用同一个 attention params/handle/workspace 桶。
