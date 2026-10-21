# A2 Request-Tile Cube/Vector Pipeline Prototype

This isolated benchmark extension partitions the existing FP32 GDN Q4 products
and deferred solve by request and key head. It does not commit accepted states;
the existing deferred commit must run after this operation in a closed test.
No production GDN path changes implicitly.

## Execution Controls

- `transport(..., streaming=False)` produces every tile assigned to each AIC
  before releasing that AIC's two AIV consumers. This is a local producer-wave
  barrier, not a device-wide barrier. The consumer copies products through UB.
- `transport(..., streaming=True)` releases each tile separately. Optional
  `vector_iterations` adds repeated multiply-by-one work solely to qualify the
  handoff. Its performance is not a GDN performance result.
- `solve(..., streaming=False)` uses the same local producer-wave barrier and
  the actual compact Q4 solve consumer.
- `solve(..., streaming=True)` permits two outstanding products tasks per AIC,
  so products of the next request tile can overlap the current tile's solve.
- `consume(...)` runs the same solve header as a separate pure-AIV operation.
  Pair it with the original pure-Cube products operation for the unsplit control.

All modes use the full `[B,24,128,8]` FP32 products buffer. There is no ring
buffer reuse or reduced products traffic. Each publication group contains two
adjacent key heads of one request: two unchanged M384/N8/K128 Cube products and
six value-head solves. The paired AIVs handle offsets `0,2,4` and `1,3,5`,
respectively. Both the mixed and separate consumers use this fixed grouping.

The first solve build (V2) published one key head, assigning two value heads to
one AIV and one to its pair. With B100 and 20 AICs, that produces 80 versus 40
head solves per AIV pair. V3 publishes two key heads to produce 60 versus 60
head solves, while halving publication handshakes. No batch capture size or
matrix tiling changes. The producer admits at most two publication groups.

## A2 Handoff Contract

The kernel declares `KERNEL_TYPE_MIX_AIC_1_2`. AIC block `c` owns AIV blocks
`2*c` and `2*c+1`. Explicit `MatmulImpl` avoids allocating an AIV Matmul service
through the mixed-mode `Matmul` alias. FP32 products explicitly disable HF32.

After producing a task, the AIC publishes ready flag 8 on `PIPE_FIX`. Both AIVs
wait for that flag before reading the products. The consumer acknowledges flag
9 on `PIPE_MTE2`: after its products DMA in the transport case, or after issuing
all assigned solve heads in the real consumer. AIC admission allows at most two
unconsumed ready flags and drains the acknowledgements before returning.

The two-AIV acknowledgement pattern follows the repository's A2 mixed GMM
implementation. Device qualification, including repeated replay, must establish
its behavior on the installed runtime; compilation alone does not prove it.

Inactive, null, invalid accepted-prefix and inconsistent-owner tasks still
participate in the handshakes but do not read state or write products, outputs
or staged records. This avoids data-dependent mismatches between ready and wait.
Owner zero is reserved for null state. All four IDs of a valid row must identify
the same compact-state owner. No host metadata readback occurs in the binding.

## Build and API

Build with a fresh directory after every source change. The loader fingerprints
C++, headers, CMake and itself, and stages the generated fixed-shape tiling next
to the source. Host tiling generation and compilation initialize no NPU.

```bash
python benchmarks/multi_graph_gdn_q4_tile_pipeline_kernel/load.py \
  --build-dir /workspace/multi-graph-results/q4-tile-build-FRESH
```

The Python module is `gdn_q4_tile_pipeline`. `solve` takes:

```python
module.solve(state, q_fp32, k_fp32, q_bf16, k_bf16, value, beta, gate,
             sequence, state_ids, accepted, products, output,
             old_logs, old_valid, new_logs, new_valid,
             cores=20, streaming=True, request_start=0, request_stop=-1)
```

`consume` takes products first, then raw Q/K/V, beta, gate, state and metadata,
output and both log/valid banks. Its `cores` argument counts AIVs (even 2..40).
The mixed operations count AICs (1..20). Products, outputs and staged banks must
not alias any input or each other. Optional page padding in the state slot
stride is preserved.

## Scope of Evidence

Required comparison: pure products plus `consume`, mixed local-wave barrier,
and mixed streaming, followed by the identical accepted-state commit. Q/K casts,
input preparation and later full-layer/full-model work remain part of the final
performance scope. A transport test proves the handoff only. A solve microtest
cannot establish a full-model advantage over equally optimized padding.
