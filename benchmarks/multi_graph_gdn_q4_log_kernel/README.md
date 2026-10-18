# FP32 Q4 compact-log prototype

This benchmark keeps one dense base per stable request and four FP32
key/update/decay records. It is a separate experimental representation,
not a drop-in dense SSM pointer or a production cache manager.

```python
extension.step(products, query, key, value, beta, gate, base,
               sequence, base_ids, accepted, output, logs, valid, cores=40)
extension.export_candidates(base, logs, valid, candidates, cores=40)
```

`step` and the matching products entry also accept optional
`request_start=0, request_stop=-1` after `cores`. The range partitions tasks
inside the full bucket and keeps all global row/metadata addresses. It is
not a new graph capture capacity. The caller must retain unique request
ownership and join every state writer before completing the endpoint.

Base is FP32 `[Nrequest,24,128,128]` with optional page padding. Logs are
FP32 `[Nrequest,24,1056]`: four 128-element keys, four 128-element update
vectors and four eight-float padded decay scalars. Valid flags are int32
`[Nrequest,24,8]`, one full 32-byte DMA/cache-line record per request/head task.
Neighboring cores must not issue scalar writes to packed four-byte flags.
Slot zero
is null and is never mutated by `step`. Every active request owns one
distinct nonzero slot; all four packed `base_ids` entries repeat that slot.
Q/K/V/beta and the initial product layout match the dense Q4 prototype.

The pure Cube products kernel reads the base before mutation. The AIV
step loads the old log, corrects initial products through small key products,
solves current outputs and update vectors, commits the previous accepted
prefix to the base and writes the current log. One task consumes its complete
old log before overwriting it. A single bank is therefore sufficient for this
sequential two-kernel prototype; it does not prove safety for a future
multi-stream or overlapping-generation backend.

The native exporter and accepted commit share `ApplyOldLog`, including the
same FP32 multiply and FMA instructions. The device probe compares accepted
commit bitwise to the previously exported prefix. Comparison with installed
dense recurrence uses separate cross-algorithm error metrics.

The prototype assumes externally serialized, valid request generations.
Stale acceptance, retries, reassignment, prefill transitions, mixed requests,
connector exports and production generation checks are not integrated.
Convolution state and the real layer remain outside this operator experiment.
Every required dense consumer must pay export cost. The probe times both
compact representation and an all-candidate-export endpoint; it cannot claim
a model or request-splitting performance benefit.

```bash
python benchmarks/multi_graph_gdn_q4_log_kernel/load.py \
  --build-dir /workspace/multi-graph-results/q4-log-build-v1
```

Fresh directories are required after source changes. The matching products
binding permits a read-only base pool smaller than four dense candidates per
request. The Cube computation and FP32 precision remain unchanged.
