# Direct selected-state Cube products

This benchmark consumes the same fixed-capacity Q4 state and metadata contract
as `../multi_graph_gdn_q4_block_kernel/`. Its output replaces the gather, grouped
Q/K pack and FP32 BMM preparation for that recurrence. It does not change the
block solve or dense candidate semantics.

```python
extension.products(state, query, key, sequence, state_ids, accepted,
                   output, cores=20)
```

The tensors are on one NPU. State is BF16
`[Nslots>=1+4B,24,128,128]`, raw normalized Q/K are BF16 `[4B,8,128]`, sequence is
int32 `[B+1]` containing an initial offset followed by per-request 4/0 counts,
state IDs are int32 `[4B]`, and accepted counts are int32 `[B]`. Active tokens are
packed. Output is FP32 `[B,24,128,8]`: columns 0..3 are H0 times raw K1..4 and
columns 4..7 are H0 times raw Q1..4. There is no query scale or BF16 rounding of
products. State may have a padded slot stride; head/value/key dimensions stay
contiguous. Other tensors must be contiguous. The slot stride is passed as a
static launch argument, with no state copy or runtime host value read.
B is between 1 and 128. Inactive output rows retain their contents;
neither state nor slot 0 is written.

Each Cube task owns one request/key head. Two strided GM-to-L1 ND2NZ copies
assemble the four key rows and four query rows directly in a padded 16-by-128
NZ buffer. The three contiguous value heads are folded into M=384 and share
one Matmul invocation. Matmul reads their selected state matrices directly
from the cache slot and writes the actual 384-by-8 FP32 ND result. No gathered state or packed Q/K
intermediate is created in GM. Explicit MTE2-to-MTE1 and reverse barriers
protect the manually supplied B buffer.

The build runs the installed CANN host tiler once for the fixed 384-by-8-by-128
shape and embeds its fields in the kernel. The host tiler uses explicit A2
memory capacities and invokes no ACL/NPU API. Runtime request metadata remains
device data; it does not regenerate tiling or perform host value reads.
Sources and the generated header are staged inside the fresh build directory
so the legacy CANN precompiler sees adjacent headers without modifying the
repository source directory.

```bash
python benchmarks/multi_graph_gdn_q4_products_kernel/load.py \
  --build-dir /workspace/multi-graph-results/q4-products-build-v1
```

Same-schedule eager/replay equality, selected-state changes, random stable IDs,
mixed accepted counts, inactive rows and finite outputs must be qualified on
device. Comparison against the framework FP32-output BMM uses explicit
cross-kernel error metrics. End-to-end timing must include these products and
the complete solve/materialization endpoint; isolated products latency is not
a model replay result.
