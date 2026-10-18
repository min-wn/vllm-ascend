# Q4 block solve and dense candidate materialization

This isolated Ascend C benchmark implements the mathematical identity in
`../multi_graph_gdn_q4_block_reference.py`. It is original Apache-2.0 code and does
not copy or include the CANN-licensed recurrence implementation. It does not
replace an installed operator or initialize the NPU when built or imported.

```python
extension.emit(products, query, key, value, beta, gate, state,
               sequence, state_ids, accepted, output, cores=40)
```

All buffers are on one NPU. State supports a padded slot stride while the
head/value/key dimensions remain contiguous; other buffers are contiguous.
The supported contract is Q=4,
1 <= B <= 128, Hk=8, Hv=24, and both state dimensions 128.

| Buffer | Dtype | Shape and meaning |
| --- | --- | --- |
| products | FP32 | `[B,24,128,8]`, selected H0 times raw K1..4,Q1..4 |
| query, key | BF16 | `[4B,8,128]`, normalized, with no query scaling yet |
| value, output | BF16 | `[4B,24,128]` |
| beta | BF16 | `[4B,24]` |
| gate | FP32 | `[4B,24]`, original log decay; device `Exp` computes gamma |
| state | BF16/FP32 | `[Nslots,24,128,128]`, V-major then K; Nslots >= 1+4B |
| sequence | int32 | `[B+1]`: initial token offset, followed by per-request 4/0 counts |
| state_ids | int32 | `[4B]`, candidate destination slot for each packed active token |
| accepted | int32 | `[B]`, previous accepted count 1..4 per active request |

Products are indexed by request row even when requests are inactive. Q/K/V and
state IDs are packed by active counts. H0 is loaded from
`state_ids[begin + accepted[row] - 1]`. Products must use exactly that same H0.
The four destination slots belonging to distinct active requests must not
overlap; no request may overwrite another request's selected source. Slots may
be arbitrary stable IDs and the selected source may also be a destination of
the same request. Slot 0 is read-only. Inactive requests perform no writes.
Device metadata values are caller contracts; shape checks avoid host reads.

Each request/head task computes FP32 K/K and K/Q Gram matrices, solves the four
update vectors, and obtains all four outputs from the supplied initial-state
projections. Query scaling `1/sqrt(128)` occurs in FP32 after those products.
The task then reads the selected state in V=64 tiles, performs four ordered
FP32 decay/FMA state updates, and writes every candidate at the original cache
precision. Candidate writes do not feed back into later tokens inside a step;
the next step loads the selected candidate. FP32 state is never rounded to
BF16 at that boundary. Raw Q/K/V/beta remain BF16 for both state types.

Candidate output uses a separate two-buffer VECOUT queue. Queue ownership
protects the DMA source until completion while allowing the next token's
arithmetic to proceed. Input/scratch storage remains separate from those
candidate buffers. The Gram workspace reuses the later outer-product buffer,
keeping the FP32-state implementation within the A2 UB budget.

The block solve and Cube products reassociate floating point operations.
Official cross-algorithm bit equality is not expected. Same-implementation
eager/replay equality, continuous state and inactive/null invariants remain
required. Product gathering, packing, BMM and all candidate writes must be
included in performance measurements. This kernel alone establishes no
complete replay performance claim.

Build only, using a fresh output directory:

```bash
python benchmarks/multi_graph_gdn_q4_block_kernel/load.py \
  --build-dir /workspace/multi-graph-results/q4-block-build-v1
```
