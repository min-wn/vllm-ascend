# inplace split 阶段 2 完成报告

阶段 2 的目标是扩展 `BatchDescriptor`，让 graph key 能表达
`start_num_tokens` offset。

本阶段只完成 descriptor/key 语义、诊断输出和单元测试覆盖；没有启用
inplace split，没有修改 `CudagraphDispatcher.dispatch()` 签名，没有新增 lazy
capture，也没有修改 `GraphParams` key 或 split 执行路径。

## 完成内容

### 1. 扩展 BatchDescriptor

修改文件：

```text
/vllm-workspace/vllm/vllm/forward_context.py
```

在 `BatchDescriptor` 尾部新增字段：

```python
start_num_tokens: int = 0
```

该字段表示当前 graph key 对应的原始 batch token 起始 offset。默认值为 `0`，
因此当前 no split 和 parallel-buffer split 路径不传 offset 时仍保持旧语义。

### 2. 保留 relax 后的 offset

`BatchDescriptor.relax_for_mixed_batch_cudagraphs()` 已保留
`start_num_tokens`：

```python
BatchDescriptor(
    self.num_tokens,
    num_reqs=None,
    uniform=False,
    has_lora=self.has_lora,
    start_num_tokens=self.start_num_tokens,
)
```

这样后续阶段即使 offset descriptor 被 relax，也不会把
`32 tokens at start=256` 和 `32 tokens at start=384` 合并成同一个 key。

### 3. 扩展 split inplace JSONL descriptor 输出

修改文件：

```text
vllm_ascend/inplace_split_debug.py
```

`batch_descriptor_info()` 现在会在 descriptor 存在该属性时输出：

```text
start_num_tokens
```

保持对旧 descriptor 或 mock descriptor 的兼容：对象没有该属性时不输出。

### 4. 增加测试覆盖

修改文件：

```text
/vllm-workspace/vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py
tests/ut/test_inplace_split_debug.py
```

新增测试覆盖：

- `BatchDescriptor._fields` 包含 `start_num_tokens`。
- 默认构造和旧 positional 调用的 `start_num_tokens == 0`。
- 仅 `start_num_tokens` 不同的 descriptor 不相等，并可作为不同 set key。
- `relax_for_mixed_batch_cudagraphs()` 保留 `start_num_tokens`。
- `batch_descriptor_info()` 输出 `start_num_tokens`。

## 预检查结果

已执行静态搜索：

```bash
rg -n "BatchDescriptor\\(" /vllm-workspace/vllm/vllm /vllm-workspace/vllm/tests vllm_ascend tests
rg -n "len\\(.*batch_descriptor|tuple\\(.*batch_descriptor|_fields|num_tokens,.*num_reqs,.*uniform,.*has_lora" /vllm-workspace/vllm/vllm vllm_ascend tests /vllm-workspace/vllm/tests
```

结论：

- 未发现 `num_tokens, num_reqs, uniform, has_lora = batch_descriptor` 这类
  四字段 unpack。
- 现有关键路径主要通过构造函数或属性访问使用 `BatchDescriptor`。

## 验证结果

已执行：

```bash
PYTHONPATH=/vllm-workspace/vllm:$PYTHONPATH python -m py_compile \
  /vllm-workspace/vllm/vllm/forward_context.py \
  vllm_ascend/inplace_split_debug.py \
  tests/ut/test_inplace_split_debug.py
PYTHONPATH=/vllm-workspace/vllm:$PYTHONPATH python -m py_compile \
  /vllm-workspace/vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py

PYTHONPATH=/vllm-workspace/vllm:$PYTHONPATH python -m pytest \
  tests/ut/test_inplace_split_debug.py
```

结果：

```text
py_compile passed
/vllm-workspace/vllm/tests/v1/cudagraph/test_cudagraph_dispatch.py py_compile passed
tests/ut/test_inplace_split_debug.py: 5 passed
```

由于当前环境缺少 `tblib`，vLLM 侧 pytest 在 collection 阶段失败：

```text
ImportError while loading conftest '/vllm-workspace/vllm/tests/conftest.py'
ModuleNotFoundError: No module named 'tblib'
```

为覆盖同等关键断言，已执行不依赖 pytest/conftest 的 Python 校验脚本，验证：

- `BatchDescriptor._fields` 包含 `start_num_tokens`。
- 默认和旧 positional 调用保持 `start_num_tokens=0`。
- offset descriptor hash/equality 可区分。
- relax 后 offset 不丢失。
- `CudagraphDispatcher.dispatch()` 未传 offset 时返回 key 的
  `start_num_tokens=0`。

结果：

```text
phase2 descriptor checks passed
```

## 未改变内容

阶段 2 明确未改变以下内容：

- 未修改 `CudagraphDispatcher.dispatch()` 签名。
- 未接入 `start_num_tokens` 到 dispatcher 产出的 runtime key。
- 未新增 `allow_inplace_lazy_key`。
- 未修改 `GraphParams` key 类型。
- 未修改 split planner 或 split 执行路径。
- 未启用 `inplace_serial` 或 `inplace_parallel`。

## 阶段结论

阶段 2 已完成。

当前 descriptor 已具备表达 offset 的能力，但 dispatcher 仍不会自动产生非零
offset key，`GraphParams` 也仍未 descriptor-aware。因此真实 inplace replay 仍
不能启用。下一步应进入阶段 3：扩展 `CudagraphDispatcher`，支持 inplace offset
lazy key。
