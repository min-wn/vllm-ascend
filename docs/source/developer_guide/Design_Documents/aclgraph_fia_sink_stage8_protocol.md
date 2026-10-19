# FIA sink stage 8: minimum full-model integration protocol

Status: protocol only, 2026-10-19. No Stage 8 runtime, unit test or NPU
measurement has started. This protocol depends on the completed Stage 7
result; it does not assume a Stage 7 performance outcome.

## Goal and claim boundary

Stage 8 asks whether the qualified native attention policy survives the real
Qwen3.5-27B TP2 model path with true QKV production, KV writes, GDN state,
residuals, normalization and tensor-parallel communication. It is a
default-padding model experiment: every graph still executes its complete
legal padded bucket.

The experiment cannot establish an advantage unique to graph splitting. The
production `MultiGraphV1Adapter` is not this integration path: its current
admission contract requires query length one, TP=1 and dense non-hybrid FIA,
while this model is TP2 Q4 with hybrid GDN state. Broadening that adapter is
a separate systems change.

The three controls make attribution explicit:

1. `legacy_pad`: current full-model FIA with real per-layer host updates.
2. `upstream_sink_pad`: shared sink consumer with the original native producer.
3. `whole_sink_pad`: the same sink consumer with the Stage 5 policy library.

The first versus third measures adoption plus policy. Only the second versus
third isolates the native policy difference. Complete-row assignment and FD
removal remain a combined change; no result may attribute it solely to FD
merge removal.

## Minimum implementation boundary

Reuse the complete-model TP2 benchmark and change only full attention.

| Source | Required scoped change |
| --- | --- |
| `benchmarks/multi_graph_gdn_q4_full_loader.py` | Reuse the external-launcher TP2 weight-only path and resolve the installed default catalog without supplying sizes. Select one isolated vendor per fresh process/rank. |
| `benchmarks/multi_graph_gdn_q4_full_metadata.py` | Add exact-sized int64 cumulative-Q and KV-length buffers for the sink ABI. Upload fixed Q4 endpoints once per bucket; refresh KV lengths and page tables in place. |
| `benchmarks/multi_graph_gdn_q4_full_probe.py` | Add an instance-scoped coordinator. Capture one metadata producer and sixteen sink consumers per rank while keeping GDN and TP unchanged. |
| `vllm_ascend/attention/attention_v1.py` | Adapt `AscendAttentionBackendImpl.forward_fused_infer_attention` after `reshape_and_cache` has written real K/V. Keep prefixes and unsupported geometry on the original route. |

The adapter uses the real projected query and each layer's private cache. It
must not replace the cache writer with synthetic K/V or bypass model forward.
The sink binding returns an owning output; copying it into the model output is
inside capture and timing. A new `.out` binding is not required initially.

Sharing one immutable plan does not share mutable attention workspace. Audit
all sixteen occurrence lifetimes and graph-pool aliasing before scratch reuse.
Reject geometry outside Q4/BF16/TND, Q12/KV2/D256/page128/sparse3, non-MLA,
non-quantized, no-sink and no-window. Verify the post-RoPE query/cache route.
Read actual 20 Cube/40 Vector limits on both TP ranks; do not assume 24/48.

The general metadata object exposes int32 GPU fields, while Stage 5/7
qualified exact int64 lengths. Create explicit int64 sink buffers; do not
reinterpret an oversized slab or alter other backend dtypes. Both sink
controls pay identical preparation and copy costs.

## Staged execution gates

The protocol avoids a large matrix before integration is proven.

### Gate A: three-step integration screen

Run each control in a fresh process on the first three chronological steps
with the real model and ordinary default buckets. This is not a performance
claim. Prove isolated native symbols, all 48 GDN and 16 FIA visits, control
specific producer/update counts, real K/V writes before attention, exact
current metadata, whole-row coverage where applicable, finite live hidden/residual
outputs, exact zero attention output for KV0 rows, protected inactive state,
valid TP synchronization and continued accepted-prefix state. The complete
model's padded hidden rows are not required to be zero.

Failure stops the experiment. Do not time an integration that has not passed
real-cache and model-forward checks.

After correctness passes, use a small matched checkpoint diagnostic to measure
the full-forward endpoint, actual attention critical-path share, preparation,
producer and output-copy costs. This diagnostic does not establish a continued
trajectory gain. It decides whether a larger performance study has sufficient
headroom: do not multiply an attention-chain percentage by full-model time.
The archived ready-path model mean is about 143.474 ms, whereas the Stage 7
chain is only several milliseconds; these separate experiments are scale
context, not an Amdahl bound for the new integration. Re-establish the budget
on the actual integrated path before committing to Gate C.

### Gate B: one continued 128-step qualification

After Gate A, run one complete chronological trajectory per control from the
same coherent prefix checkpoint, restoring that checkpoint before each
control. Never compare one policy after another policy has mutated state.

Every step retains the current 1024-word plan, exact length/page semantics,
fixed addresses and native geometry. Compare full hidden/residual endpoints
with explicit tolerances and inspect all 16 FIA and 48 GDN layers. Continue
accepted prefixes, paused owners, pages, recurrent bases/logs/validity and
null storage.

For the screen and qualification pass, capture three live rows (first,
middle, last) from each real FIA occurrence's projected query and current
post-write KV cache, then compare with the independent FP32 attention
reference. Gather exact ordered pages, preserve repeated page IDs and retain
the reference inputs/provenance. At each continued step, same-mode eager and
graph branches must start from identical preceding state; snapshot/restore
between those two branches and advance the accepted history exactly once.
Full-model cross-policy numerical tolerances must be declared before
measurement; the attention operator's tolerance alone is not a bound on
64-layer state drift.

Real model caches intentionally change, so Stage 7 input immutability does
not apply to active cache slots. Inactive owners, unused pages, null rows,
GDN state and layer ownership remain protected. Gate B establishes model
correctness, not performance.

### Gate C: formal three-control measurement

Only after all controls pass Gate B and the measured integration budget
justifies a performance study, run the unchanged 128-step trace with all
three controls in fresh processes. Start the critical-rank full-forward wall
endpoint before `registry.prepare_step`, frame/token preparation and copies;
finish after replay, TP communication, GDN commits, FIA completion and device
joins. Include the native producer and sink output copy. Exclude sampler,
request scheduling, draft execution, serving queues and setup.

The minimum formal order is three rounds: legacy/upstream/whole,
upstream/whole/legacy, whole/legacy/upstream. Each item is a fresh TP2
process group and one continued 128-step trajectory from the coherent
checkpoint: nine process groups and 1152 step observations in total. Record
both rank-local wall times and use their maximum for each step. Report
within-round matched differences, preserving process-order and time drift
as limitations. Do not treat individual steps as independent process trials.

Report unfiltered wall/queue samples, P50/P95/P99, per-capacity and
transition groups, memory/pool peaks, update/producer counts and all slower
counterexamples. Do not add overlapping host and queue intervals. Each trial
continues the coherent state; it is not reset per step. Use one trace and one
GDN mode before expanding the matrix.

## Correctness evidence

For every control and step, verify layer visits, update/producer counts,
current plan and exact ABI dimensions; finite live output, exact KV0 attention
zeros, legal page writes and preserved inactive/null owners; full live hidden/residual
endpoints; and continued GDN/FIA state. Use `StreamingCheckpoint` and bounded
state snapshots. Compare same-mode eager versus graph before comparing
different arithmetic policies. FD and no-FD need not be bitwise identical,
but tolerance and ownership failures are disqualifying.

Each process selects exactly one vendor. Identically named policies must not
coexist. Audit all capture pools and peak memory: seventeen independent
full-model pools may exceed memory despite safe operator-level pools.

## Static inventory

Already available:

- TP2 external-launcher Qwen3.5-27B model observer and 48-GDN/16-FIA cache
  inventory;
- unchanged 128-step trace and installed default catalog resolver;
- coherent prefixes, compact recurrent ownership, state snapshot/restore and
  critical-rank timing scaffolding;
- isolated Stage 5 libraries, 1024-word ABI validation and Stage 7 policy
  provenance/analyzer contracts;
- CPU contracts for control identity and report preservation.

Still required:

- real model-call sink adapter and one-producer/sixteen-consumer coordinator;
- exact int64 sink buffers and per-layer current-cache plumbing;
- control-specific capture/update instrumentation and output-copy accounting;
- Gate A screen, Gate B continued correctness and Gate C formal analysis.

## Result boundary

Stage 8 may claim a producer-inclusive full-model default-padding result only
after Gate C and independent analysis. It may not claim graph splitting beats
padding, serving throughput improvement or an isolated FD-merge effect. A
later split study must use the same best sink policy in both paths, exact
private descriptors per slice, required independent producers and all joins in
the critical-rank endpoint.
