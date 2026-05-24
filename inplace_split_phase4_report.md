# inplace split 阶段 4 完成报告

阶段 4 的目标是扩展 Ascend `GraphParams`，支持 descriptor-aware key，避免
真实 token 数相同但 `start_num_tokens` 不同的 offset graph 复用同一个
attention params、handles、events 或 workspace 桶。

本阶段只完成 GraphParams key 能力和测试覆盖；没有启用 inplace split，没有接入
split planner、input offset slicing 或 metadata stable buffer。

## 完成内容

### 1. 新增 GraphParams key helper

修改文件：

```text
vllm_ascend/compilation/acl_graph.py
```

新增：

```python
GraphParamKey = int | BatchDescriptor
get_graph_param_key()
graph_param_key_info()
ensure_graph_param_key()
require_graph_param_key()
```

key 规则：

```text
start_num_tokens == 0  -> int runtime_shape
start_num_tokens > 0   -> BatchDescriptor
```

`get_graph_param_key()` 只接受真实 `BatchDescriptor`。如果
`forward_context.batch_descriptor` 缺失或是 mock/其他对象，则回退到
`runtime_shape`，避免测试和旧路径被误判为 offset graph。

MTP 默认仍使用 `int runtime_shape` key，不启用 offset key。

### 2. 扩展 GraphParams 类型和 workspace 写入

`GraphParams` 的四个字典 key 从 `int` 扩展为 `GraphParamKey`：

```python
events: dict[GraphParamKey, list[torch.npu.ExternalEvent]]
workspaces: dict[GraphParamKey, torch.Tensor | None]
handles: dict[GraphParamKey, list[torch_npu._C._NPUTaskGroupHandle]]
attn_params: dict[GraphParamKey, list[tuple]]
```

`_make_graph_params()` 仍按 capture size 初始化 `int` key，普通 no-offset graph
行为不变。

`update_graph_params_workspaces()` 现在可接受 `int` 或 offset
`BatchDescriptor` key，并在写入前确保对应桶存在。

### 3. 改造 capture 写入点

修改文件：

```text
vllm_ascend/attention/attention_v1.py
vllm_ascend/attention/mla_v1.py
vllm_ascend/attention/attention_cp.py
vllm_ascend/attention/mla_cp.py
```

已改造路径：

- `attention_v1.py`: `full_graph_fia()` / `full_graph_pa()`
- `mla_v1.py`: MLA decode capture，MTP 保持 int key
- `attention_cp.py`: DCP/PCP normal attention capture
- `mla_cp.py`: DCP/PCP MLA capture

这些路径现在先计算 `param_key`，再写入：

- `graph_params.events[param_key]`
- `graph_params.attn_params[param_key]`
- `graph_params.handles[param_key]`
- `graph_params.workspaces[param_key]`

### 4. 改造 update 读取点

修改文件：

```text
vllm_ascend/compilation/acl_graph.py
```

已改造路径：

- `_update_attn_pa_params()`
- `_update_attn_fia_params()`
- `update_mla_attn_params()`
- `update_attn_dcp_pcp_params()`
- `update_mla_attn_dcp_pcp_params()`

update 读取路径使用 `require_graph_param_key()`，缺 key 时显式抛
`KeyError`，避免 `zip(..., [], [], [])` 静默跳过 attention graph task update。

`runtime_shape` 仍保留用于：

- `using_paged_attention(runtime_shape, vllm_config)`
- seq_lens padding 计算
- speculative decode padding 计算

字典读取改为使用 `param_key`。

### 5. 诊断日志改造

`acl_graph_replay` 诊断 payload 现在包含：

```json
{
  "runtime_shape": 32,
  "graph_param_key": {
    "kind": "batch_descriptor",
    "num_tokens": 32,
    "start_num_tokens": 384
  }
}
```

block table replay 诊断也按 descriptor-aware key 查找 graph params。

## 测试覆盖

修改文件：

```text
tests/ut/compilation/test_acl_graph.py
```

新增 `TestGraphParamKey`，覆盖：

- 默认 descriptor 使用 int key。
- offset descriptor 使用 `BatchDescriptor` key。
- MTP 默认保持 int key。
- 相同 `num_tokens`、不同 `start_num_tokens` 的 key 互相隔离。
- 缺失 key 的 update 读取保护会抛 `KeyError`。
- `update_graph_params_workspaces()` 支持 offset key。
- `graph_param_key_info()` 输出 descriptor key 信息。

## 验证结果

已执行：

```bash
python -m py_compile \
  vllm_ascend/compilation/acl_graph.py \
  vllm_ascend/attention/attention_v1.py \
  vllm_ascend/attention/mla_v1.py \
  vllm_ascend/attention/attention_cp.py \
  vllm_ascend/attention/mla_cp.py \
  tests/ut/compilation/test_acl_graph.py
```

结果：通过。

已执行定向测试：

```bash
python -m pytest tests/ut/compilation/test_acl_graph.py -k 'GraphParamKey' -q
```

结果：

```text
7 passed, 22 deselected, 2 warnings
```

已执行完整文件测试：

```bash
python -m pytest tests/ut/compilation/test_acl_graph.py -q
```

结果：

```text
29 passed, 2 warnings
```

已执行静态搜索：

```bash
rg -n "graph_params\.(attn_params|handles|events)\[(runtime_shape|num_tokens)\]|graph_params\.workspaces\.get\((runtime_shape|num_tokens)\)|update_graph_params_workspaces\((runtime_shape|num_tokens)" \
  vllm_ascend/compilation/acl_graph.py \
  vllm_ascend/attention/attention_v1.py \
  vllm_ascend/attention/mla_v1.py \
  vllm_ascend/attention/attention_cp.py \
  vllm_ascend/attention/mla_cp.py
```

结果：无残留匹配。

## 未改变内容

阶段 4 明确未改变以下内容：

- 未启用 `inplace_serial` 或 `inplace_parallel`。
- 未实现 inplace split planner。
- 未实现 inplace input offset view。
- 未实现 metadata stable buffer。
- 未修改 `CudagraphDispatcher` 阶段 3 的 offset key allow 条件。
- 未承诺 MTP/spec offset graph 支持。
- 未承诺 CP/PCP/DCP 场景进入 inplace planner。

## 风险和注意事项

- 当前 key 能力已经支持 offset graph params 隔离，但真实 inplace split 仍不能启用，
  因为 split planner、input slicing 和 metadata stable buffer 尚未接入。
- update 路径现在对缺失 GraphParams key 更严格，会显式抛 `KeyError`。
  这是有意行为，用于避免 attention task update 静默跳过。
- MTP 仍按 int key 处理。后续若要支持 MTP offset graph，需要单独验证
  speculative padding、workspace 和 metadata 语义。
- CP/PCP/DCP 路径已完成 key 机械改造，但阶段 5 planner 仍应对这些场景 fallback。

## 阶段结论

阶段 4 已完成。

Ascend `GraphParams` 现在可以区分普通 int key 和 offset `BatchDescriptor` key。
下一步应进入阶段 5：实现 inplace split planner，只生成 2-way split，并继续保持
第一版支持范围限制。
