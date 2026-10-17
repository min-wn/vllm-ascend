# Tensor-parallel attention boundary prototype

## Objective and scope

`benchmarks/multi_graph_tp_attention.py` extends the
[communication boundary experiment](multi_graph_tp_boundary.md) through the
first full-attention layer of Qwen3.5-27B, at layer index 3. The objective is to
hide the second request group's tensor-parallel input reduction behind useful
work for the first group. That work now includes residual normalization, QKV,
Q/K normalization, RoPE, KV writes, FIA, and the output gate. The original split
schedules then join for one shared O projection and its tensor-parallel
reduction. An optional output-overlap schedule instead projects and reduces
each group as its attention completes, allowing the first output reduction to
overlap work for the second group.

This tests whether communication overlap survives the real attention
dependencies and required host-parameter updates. It is an isolated selected
layer experiment. Rank-local preceding Down outputs and residuals are synthetic;
the preceding Down GEMM, post-attention normalization, FFN, other decoder layers,
and GDN state are absent. The script does not enable production multi-graph
execution for hybrid models or distributed serving.

The weight loader initializes the official model and TP process group but stops
before ordinary engine KV allocation, profiling, and a complete model forward.
The benchmark allocates its own selected-layer KV state. Existing production
replay model-family and KV-group restrictions remain in force.

## Default capacity contract

The workload is the previously selected batch 98, drawn once with seed 20261016
from the existing range 64 to 256. `EngineArgs` and `CudagraphDispatcher` supply
the default capture capacities. With four query tokens per request, 392 live
tokens use the default 400-token bucket.

| Schedule capacity | Captured tokens | Request capacities | Live requests at B98 |
| --- | ---: | ---: | ---: |
| Padding | 400 | 100 | 98 |
| Balanced split | 200 + 200 | 50 + 50 | 50 + 48 |
| Asymmetric split | 272 + 128 | 68 + 32 | 68 + 30 |

The prototype defaults to 272 + 128 for Q4; `--pair 200 200` selects the balanced
control. Both lane capacities must be existing default capture sizes, divisible
by the query count, and sum to the same dispatched padding capacity. The
experiment does not add a custom 49-request size, specialize captures to batch
98, remove dummy capacity from split work, or split attention heads.

The larger first group changes when useful work becomes available relative to
the second collective. It is a scheduling hypothesis, not an assumption that
an asymmetric split must be faster.

Q4 capacity selection uses
`speculative_config={"method": "ngram", "num_speculative_tokens": 3}`. This only
derives the default uniform-query capture sizes. The loader does not execute
ngram proposals, a draft model, acceptance, sampling, or the speculative
scheduler. The experiment measures one replay with Q4 and does not measure
speculative decoding throughput or depend on an acceptance-rate assumption.

## Dependency graphs

Let `P_X` be the selected layer's computation for request group X:

```text
P_X = ResidualNorm_X -> QKV_X -> QKNorm/RoPE_X
      -> KVwrite_X -> FIA_X -> gate_X -> output_slice_X
```

All modes use the same BF16 weights, NZ QKV and O layouts, input values,
aggregate reduction payload, default capacity, and TP process group. `Restore`
copies immutable synthetic Down and residual inputs to working buffers. The
collective modifies its input in place, so each mode pays these same copies.

The padding baseline completes one full-capacity reduction before attention:

```text
joined
root: Restore400 -> AR400 -> P400 -> O400 -> AR_O400 -> complete
```

The serial split is a control for the extra collective startup, fragmented QKV
and attention work, and second FIA update:

```text
split_serial
root: Restore400 -> AR_A -> AR_B -> P_A -> P_B
      -> shared O400 -> AR_O400 -> complete
```

The input-overlap candidate uses a separate communication stream to release A
while B is still reducing:

```text
split_overlap
root: Restore400 -> fork -> wait ready_A -> P_A -> wait ready_B -> P_B
                   |                           -> shared O400 -> AR_O400
comm:              +-> AR_A -> ready_A -> AR_B -> ready_B
```

In this graph, `shared O400` follows completion of both `P_A` and `P_B`.
The diagram's readiness edges are enforced by captured events. Every rank
issues collective A, then collective B, then the shared O collective in the
same order. The root stream executes both attention branches sequentially;
the candidate does not run two competing FIA kernels on separate streams.
The intended overlap is `AR_B` with `P_A`.

Each branch writes a disjoint fixed slice of a full-capacity attention output.
These three schedules use one full 400-token O projection, followed by one
full-capacity O all-reduce. The optional `--split-output-overlap` adds a fourth
schedule:

```text
split_output_overlap
root: Restore400 -> fork -> wait ready_A -> P_A -> O_A -> ready_OA
      -> wait ready_B -> P_B -> O_B -> ready_OB
      -> wait done_OA -> wait done_OB -> copy actual reduced A/B -> complete
comm: wait fork -> AR_A -> ready_A -> AR_B -> ready_B
      -> wait ready_OA -> AR_OA -> done_OA
      -> wait ready_OB -> AR_OB -> done_OB
```

All ranks issue `input_A, input_B, output_A, output_B` in that fixed order.
Captured events gate each output collective on its projection and gate the
final root-stream join on both completed reductions. `P_A`, `O_A`, `P_B`, and
`O_B` still run sequentially on the root stream; there are no concurrent FIA
kernels. The additional intended overlap is `AR_OA` with the second group's
attention-branch computation. Because `AR_B` precedes `AR_OA` on the
communication stream, input and output collectives cannot reorder across
ranks.

This is a request-group communication pipeline based on dependency readiness.
The balanced case still processes 200 + 200 tokens, equal to padding's 400.
It changes when communication can start, without removing dummy work or
requiring future inputs. Its possible benefit must pay for a second output
collective startup, fragmented O projections, and the final result join.

The communicator may return storage distinct from the projected input. The
prototype retains both projection and actual reduction-result buffers for
each group, waits for their completion, and copies the actual returned results
into a contiguous full-capacity output. Those copies are inside the captured
and timed graph. Captured identities include these output buffer addresses.

Every schedule times all O projection and reduction work. The script checks
that the loaded O module is an ordinary bias-free BF16
`AscendRowParallelLinear` with parallel input and result reduction, and checks
the explicit O-plus-reduction implementation against that module. The supported
loaded module has `custom_op=None`. This qualifies the ordinary O path in this
environment; it does not establish compatibility or performance of fused
`npu_mm_all_reduce_base` or MC2 on the old driver. The result is not a comparison
against every possible fused padding implementation.

Every collective calls the loaded model's `get_tp_group().all_reduce`. Any
future communicator improvement must also apply to padding. A win over
`split_serial` alone establishes overlap of some split overhead; the relevant
performance comparison remains each overlap candidate against `joined`.

## Inputs, KV state, and dynamic reuse

The selected layer uses its actual loaded normalization, QKV, Q/K norms, RoPE
or fused MRoPE, FIA backend, output gate, and O weights. The benchmark follows
the installed Ascend Qwen3.5 attention implementation and checks its decomposed
output against the official attention module.

The selected-layer paged KV cache is contiguous ND, explicitly cast to NPU
format 2, with block size 128. Logical contiguity alone did not establish the
required physical format on this stack.
Each request owns distinct physical pages. Splitting rebases the lane-local
query boundaries while retaining the original request's positions, slots,
and page addresses. Live KV lengths vary deterministically below the requested
maximum, 512 by default. Dummy rows have zero Down/residual inputs, zero KV
lengths, and slot mapping -1; their captured query extent remains present.

The initial prefix comes from synthetic BF16 hidden states projected through
the actual selected layer's normalization, QKV, Q/K processing, and RoPE. The
same projected prefix is copied into each request's separate pages. Current
query tokens write their actual slots through the backend. A cache snapshot
is restored before independent correctness cases and timing modes.

This prefix is coherent for the synthetic single-layer input. It is not a
complete model prefill: the prefix has not passed through earlier decoder or
GDN layers. Consequently, a successful result cannot qualify full-model KV
state, hybrid state, logits, or end-to-end decoding accuracy.

Changed-input qualification follows B98 -> B97 -> B99 -> B98, with a different
variant at every step. Live counts, hidden values, KV lengths, positions, and
slot mappings change. Graph identities and captured output, metadata, workspace,
and KV addresses remain stable. FIA workspaces are isolated per branch even
though both branches use the same layer provider.

This tests reuse within the ordinary 400-token bucket. It does not require
knowing the next step's inputs in advance. It also does not test routing across
different default buckets or the engine's actual scheduler and incremental
decode lifecycle.

## Real FIA updates on every replay

The captured graph is an `UpdatableGraph` with one real FIA task for `joined`
and two for each split mode. The script asserts that the registered operation
is `torch_npu.npu_fused_infer_attention_score.out`. A lane-aware provider maps
each occurrence to its own current host metadata.

Each replay resolves the current tasks, records preparation readiness, submits
the graph on the root stream, and invokes `graph.update` on an update stream.
The graph's per-FIA external-event mechanism gates the corresponding captured
task. Root completion also waits for the update stream's completion event.
The implementation neither freezes host sequence lengths nor omits updates
when timing repeated inputs. Split's additional update cost remains measured.

Query, key, and value layout materialization happens before the FIA update
task group. Together with benchmark-owned contiguous ND paged KV, this retains
the existing software workaround for the old driver's layout/update failure.

## Qualification and timing

Correctness uses the existing `rtol=0.02, atol=0.02` thresholds. A CPU FP32
reduction independently checks the TP input sum. Full-capacity eager ND QKV
and O weights provide the reference, including FIA, the gate, O, and its
all-reduce. Initial checks also compare the decomposed implementation with
the official complete attention module and explicit shared O with official O.
Every mode checks live-row outputs initially, after each dynamic variant, and
after each timing round. A failed check invalidates the performance result.

Each timing sample contains exactly one replay, including task resolution,
host submission, real FIA updates, all O projections and output reductions,
any output-join copies, and completion synchronization. This differs from the
first boundary experiment's amortized replay trains.

CPU rank barriers precede the timed region. Model loading, prefix construction,
capture, reference generation, input preparation, and cache restoration are
outside it. Each mode has three warmups per round. Defaults are three rounds
and eight samples per mode; the mode order reverses on alternate rounds.
Reports take the slowest rank for each sample, the median across samples for
each round, and then the median across rounds.

Wall time begins before replay task resolution. Device events cover the root
stream's dependency waits and completion, so their spans may include submission
gaps and are not pure kernel sums. Optional traces are collected separately
after formal timing and contain two replays per mode and rank.

The restore-copy cost is a harness cost paid equally by all modes. A real
preceding Down projection would produce fresh inputs. This producer and its
dependencies still need qualification before extrapolating the result to an
actual layer boundary. The selected layer's weights also remain hot throughout
this experiment.

## Driver compatibility limit

The machine retains its existing driver. Earlier FIA update attempts on the
installed software stack failed with error 207019 involving Slice/Contiguous
and `aclnnFusedInferAttentionScoreV3`. Contiguous tensors outside the update
task group address the observed layout problem but do not establish general
driver compatibility.

The Python operation name `npu_fused_infer_attention_score.out` may still lower
to `aclnnFusedInferAttentionScoreV3`. Its name is not evidence that this prototype
avoids the V3 runtime path. Capture, update, and correctness must all pass on
both ranks before timing is usable. CPU tests validate metadata construction
and lane resolution; they cannot certify the NPU runtime or this workaround.

## Running the prototype

Use a new output path for each run. The following short pilot qualifies the
current 272 + 128 candidate; its sample count is not a formal performance run:

```bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_BATCH_INVARIANT=0 \
HCCL_DETERMINISTIC=true \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/multi_graph_tp_attention.py \
  --model /models/Qwen3.5-27B-original --tensor-parallel-size 2 \
  --query 4 --pair 272 128 --kv-length 512 \
  --rounds 1 --repeats 2 \
  --output /workspace/multi-graph-results/tp-boundary/attention-pilot.json
```

After qualification, use `--rounds 3 --repeats 8` for the focused comparison
and `--trace` to collect a separate trace. Substitute `--pair 200 200` for the
balanced control. Keep all other settings and the padding implementation equal.
Add `--shared-rope --split-output-overlap` to include the fourth schedule with
the common lookup optimization enabled for every mode.

The JSON records graph scope, weight/cache shapes, real FIA update counts,
O and reduction timing inclusion, each mode's output layout and collective
order, per-rank checks, raw samples, summaries, and source hashes. A failure
writes a rank-specific failure JSON, and rank 0 also writes the main result
path. Retain these artifacts to distinguish a harness or runtime failure from
a valid negative performance result.

## Shared text-position lookup

`--shared-rope` gathers the full bucket's cos/sin rows once before the input
reduction. Text positions are identical across the three MRoPE planes, so an
expanded view supplies all three without copying the rows. Each branch reads
its token slice from that view. Padding uses the same lookup and expanded view;
the reference retains the original advanced indexing and contiguous planes.
The lookup runs inside every captured replay and uses the current position
buffer. It does not cache values across steps or assume future positions.

The fused MRoPE kernel now receives the actual plane stride as a runtime scalar.
The initial shared-lookup measurements used a `constexpr` plane-stride argument;
the current version avoids specializing a kernel for each parent-plane extent.
Its normalization and rotation arithmetic is unchanged. This supports ordinary
contiguous input,
token slices with a larger parent-plane stride, and the zero plane stride of
text-position expansion. Token and feature strides must still describe dense
rows; unsupported layouts fail before launching the kernel. This kernel
capability is available to existing callers, while shared lookup scheduling is
currently confined to this benchmark.

The NPU preflight compares both lanes' view-based Q/K/V/gate outputs with their
contiguous equivalents at `rtol=0, atol=0`, for expanded text planes and for
three genuinely different position planes. The ordinary dynamic-input checks
then compare every complete graph with the unchanged ND attention reference.
CPU wrapper contracts check view preservation and launch arguments, not kernel
numerics. Add `--shared-rope` to the invocation above to enable this experiment.

## Results and advancement gate

The formal measurements use Qwen3.5-27B BF16, TP2 on two 64 GB 910B3 devices,
batch 98, Q4, the default 400-token bucket, and a maximum KV length of 512.
The machine retains driver 24.1.rc3 with CANN 9.1, PyTorch 2.10, torch_npu,
and the vLLM 0.30 environment. Runs set `HCCL_DETERMINISTIC=true` and
`VLLM_BATCH_INVARIANT=0`. Each formal run has three alternating-order rounds
and eight single-replay samples per mode and rank. Every split mode retains
two real FIA updates per replay; joined retains one. All modes include O and
its reduction; the output-overlap mode splits this work and times its result
join, while the other modes use shared full-bucket O and reduction.

The first pilot failed before explicitly casting the benchmark-owned KV to
format 2. The subsequent `tp2-27b-attention-pilot-nd.json` passed with one round
and two samples, establishing the focused runtime qualification but not a
formal performance result. The table includes only the ensuing formal runs.

The first five formal runs passed on both ranks. The original implementation has
27 named checks per rank. Shared lookup adds exact view-versus-contiguous
MRoPE preflight comparisons, bringing each rank to 43 checks, including the
runtime-stride repeat. Complete graph checks retain `rtol=0.02, atol=0.02`;
the view preflight uses `rtol=0, atol=0`. Dynamic qualification changes the live
batch, input values, sequence lengths, positions, and slots on the captured
addresses. These checks cover this synthetic selected-layer workload only.
All four subsequent output-overlap runs also passed, with 50 checks per rank. The
seven additional checks cover the fourth mode's initial output, three dynamic
variants, and three timing rounds. Numerical thresholds are unchanged.

### Formal single-replay measurements

All times are microseconds. Each relative column is
`100 * (mode_time / joined_time - 1)` using the joined baseline in the same run.
A positive percentage means slower. Joined varies between runs, so raw
cross-run differences are not substituted for these within-run comparisons.

| Implementation | Token pair | Mode | Wall us | Wall vs joined | Device us | Device vs joined |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Original lookup | 200 + 200 | joined | 1271.66 | 0.00% | 1201.78 | 0.00% |
| Original lookup | 200 + 200 | split_serial | 1485.73 | +16.83% | 1412.21 | +17.51% |
| Original lookup | 200 + 200 | split_overlap | 1348.01 | +6.00% | 1278.65 | +6.40% |
| Original lookup | 272 + 128 | joined | 1240.24 | 0.00% | 1163.35 | 0.00% |
| Original lookup | 272 + 128 | split_serial | 1463.53 | +18.00% | 1392.40 | +19.69% |
| Original lookup | 272 + 128 | split_overlap | 1369.84 | +10.45% | 1293.24 | +11.17% |
| Shared lookup, constexpr stride | 200 + 200 | joined | 1185.75 | 0.00% | 1120.40 | 0.00% |
| Shared lookup, constexpr stride | 200 + 200 | split_serial | 1361.49 | +14.82% | 1301.46 | +16.16% |
| Shared lookup, constexpr stride | 200 + 200 | split_overlap | 1239.74 | +4.55% | 1177.32 | +5.08% |
| Shared lookup, constexpr stride | 272 + 128 | joined | 1201.30 | 0.00% | 1138.30 | 0.00% |
| Shared lookup, constexpr stride | 272 + 128 | split_serial | 1367.45 | +13.83% | 1296.47 | +13.90% |
| Shared lookup, constexpr stride | 272 + 128 | split_overlap | 1265.03 | +5.31% | 1198.33 | +5.27% |
| Shared lookup, runtime stride repeat | 200 + 200 | joined | 1189.67 | 0.00% | 1135.18 | 0.00% |
| Shared lookup, runtime stride repeat | 200 + 200 | split_serial | 1373.31 | +15.44% | 1315.46 | +15.88% |
| Shared lookup, runtime stride repeat | 200 + 200 | split_overlap | 1230.58 | +3.44% | 1170.91 | +3.15% |

Shared lookup narrows the split-overlap wall-time deficit from 6.00% to 4.55%
for 200 + 200, and from 10.45% to 5.31% for 272 + 128 in the first formal
comparison. The runtime-stride 200 + 200 repeat remains 3.44% slower than its
own joined baseline, a 40.91 us wall-time gap. This repeat qualifies the current
runtime-stride kernel; differences from the constexpr run do not isolate a
performance effect of passing stride at runtime. The later output-overlap
comparison also uses the runtime-stride kernel.

### Trace evidence and remaining gap

The original 200 + 200 rank-0 trace shows about 118.5 us of indexing kernels
for a representative split-overlap replay, versus about 65.7 us for joined.
These are the summed `aclnnIndex` broadcast, index-check, and index kernel
durations. In the shared-lookup trace, the corresponding work is one
`aclnnIndexSelect` gather, about 15-16 us for both joined and split-overlap.
The optimization therefore removes roughly 52 us of relative indexing overhead
in that trace, as well as reducing the lookup cost paid by both schedules.

These kernel durations come from separate profiler runs. They are not formal
wall measurements or additive predictions of the graph critical path. Removing
the relative indexing overhead leaves the first shared-lookup formal candidates
4.55% and 5.31% behind joined, and the runtime-stride repeat 3.44% behind.
The remaining schedule still duplicates input collective startup and FIA task
updates, fragments the attention branch, and waits for both branches before
the shared O projection and final output reduction. The existing measurements
do not assign the entire remaining gap to any single one of these costs.

### Output-reduction pipeline

The optional output-overlap schedule has completed two formal runs for each
of the existing default-capacity pairs, 200 + 200 and 272 + 128, with shared
lookup and the runtime-stride kernel. All four runs use identical benchmark
and kernel source hashes. All four modes are measured in each run; the joined
baseline receives the same shared lookup and other generic optimizations.
Relative percentages use each run's joined time.

Balanced 200 + 200:

| Run | Mode | Wall us | Wall vs joined | Device us | Device vs joined |
| --- | --- | ---: | ---: | ---: | ---: |
| Initial | joined | 1221.11 | 0.00% | 1133.28 | 0.00% |
| Initial | split_serial | 1392.39 | +14.03% | 1312.26 | +15.79% |
| Initial | split_overlap | 1311.18 | +7.38% | 1236.16 | +9.08% |
| Initial | split_output_overlap | 1223.82 | +0.22% | 1138.08 | +0.42% |
| Independent repeat | joined | 1204.79 | 0.00% | 1137.80 | 0.00% |
| Independent repeat | split_serial | 1376.89 | +14.28% | 1304.66 | +14.67% |
| Independent repeat | split_overlap | 1243.59 | +3.22% | 1175.56 | +3.32% |
| Independent repeat | split_output_overlap | 1190.17 | -1.21% | 1122.75 | -1.32% |

Output overlap improves wall time against input-only `split_overlap` by 6.66%
in the initial run and 4.30% in the repeat. Relative to padding, however, the
initial run is 2.71 us slower and the repeat is 14.62 us faster. The direction
of the padding comparison changes between runs, so these results establish a
useful reduction in split overhead, not a repeatable padding win.

The separate traces identify the intended overlap in both runs: the first
output reduction lasts about 140-153 us and is fully or almost fully hidden
by B's normalization, QKV, and RoPE work, rather than primarily by `FIA_B`.
The second output reduction remains an exposed tail of about 138-143 us, and
the final result copies cost about 9-10 us. Splitting O also adds work. In a
representative rank-1 replay of the repeat, the two O GEMMs total 77.40 us
versus 59.22 us for input-only split, and B's QKV increases from 82.36 to
101.50 us while overlapping communication. The output reductions total
284.04 us versus 248.60 us, even though their overlap shortens the graph.

These are profiler observations from separate replays, not additive wall-time
predictions or isolated causal measurements. They show why moving a reduction
off the exposed path helps while also identifying fragmentation, communication
contention, result copies, and the remaining output tail as costs. They do not
qualify the complete decoder or production scheduler.

Asymmetric 272 + 128:

| Run | Mode | Wall us | Wall vs joined | Device us | Device vs joined |
| --- | --- | ---: | ---: | ---: | ---: |
| Initial | joined | 1223.21 | 0.00% | 1157.62 | 0.00% |
| Initial | split_serial | 1377.61 | +12.62% | 1315.24 | +13.62% |
| Initial | split_overlap | 1321.67 | +8.05% | 1259.18 | +8.77% |
| Initial | split_output_overlap | 1179.72 | -3.56% | 1119.39 | -3.30% |
| Independent repeat | joined | 1203.28 | 0.00% | 1132.91 | 0.00% |
| Independent repeat | split_serial | 1353.32 | +12.47% | 1284.19 | +13.35% |
| Independent repeat | split_overlap | 1262.09 | +4.89% | 1190.29 | +5.06% |
| Independent repeat | split_output_overlap | 1148.54 | -4.55% | 1078.36 | -4.82% |

Output overlap improves wall time against the same-run joined baseline by
43.50 us initially and 54.74 us in the repeat. Compared with input-only split,
the reduction is 10.74% and 9.00%, respectively. All three output-overlap round
medians beat their corresponding joined round medians in both runs. This is a
repeatable improvement for this selected attention interval and workload;
200 + 200 remains close to parity with an unstable direction against padding.

The asymmetric traces show overlap extending through B's FIA. In a
representative rank-1 replay of the repeat, `AR_OA` takes 179.06 us; B's
53.76 us QKV and 53.72 us FIA execute within that window. The first output
reduction finishes before B's O projection completes. The exposed final
`AR_OB` takes 97.84 us and the result copies take 9.16 us. O fragmentation
still costs work: the two O GEMMs total 76.98 us, versus 57.00 us for padding.
Both ranks' two output-overlap trace replays show this dependency pattern.
Durations differ by rank: another repeat sample on rank 0 has a 215.66 us first
output reduction and a 133.48 us final reduction. The remaining tail is not
uniformly 100 us across ranks and samples.
The repeat's input-only split trace contains large communication and wait
outliers, so it is not used as a precise duration-difference model.

Reducing B's output payload alone does not explain the gain. Under the simple
linear communication model `AR(n) = alpha + beta * n`, the exposed first
input reduction plus final output reduction is
`AR(n_A) + AR(n_B) = 2 * alpha + beta * 400`. Making B smaller also makes A
larger. Shape-dependent GEMM efficiency, communication contention, and the
useful overlap window must therefore be considered together. For example,
the initial asymmetric rank-0 trace shows B's QKV rising from 52.64 us in
input-only split to 56.04 us with output overlap, much less than the roughly
19 us increase in the balanced repeat. These separate traces support a
mechanism, without isolating the effect of each shape or contention source.

### Artifacts and exact sources

Artifacts are retained in the workspace directory
`outputs/multi_graph_tp_attention/`, outside the repository checkout. Formal
JSON files contain per-rank checks and samples, aggregation results, source
hashes, shapes, and update counts. Matching `.rank0.trace.json` and
`.rank1.trace.json` files contain the separate trace replays.

| Run | Formal JSON |
| --- | --- |
| Original 200 + 200 | `tp2-27b-attention-200-200.json` |
| Original 272 + 128 | `tp2-27b-attention-272-128.json` |
| Shared constexpr 200 + 200 | `tp2-27b-attention-200-200-shared-rope.json` |
| Shared constexpr 272 + 128 | `tp2-27b-attention-272-128-shared-rope.json` |
| Shared runtime stride 200 + 200 | `tp2-27b-attention-200-200-shared-rope-dynamic-stride.json` |
| Output overlap 200 + 200 | `tp2-27b-attention-200-200-output-overlap.json` |
| Output overlap 200 + 200 repeat | `tp2-27b-attention-200-200-output-overlap-repeat.json` |
| Output overlap 272 + 128 | `tp2-27b-attention-272-128-output-overlap.json` |
| Output overlap 272 + 128 repeat | `tp2-27b-attention-272-128-output-overlap-repeat.json` |

The exact recorded SHA-256 values are:

| Source | SHA-256 |
| --- | --- |
| Original benchmark | `33213fbbb6e38ee4b5347b5b69509d64b2e228e32bba9dae9a5b83ae86ab8101` |
| Shared-lookup benchmark, all three runs | `6982e4712f3a30a03fb384050952da6675fce53d45f47d32a0a753d16feafb57` |
| Output-overlap benchmark, all four runs | `0764a871a86642e75c7c7ae3c0f59e680bfe53b54c65dc3122f04dbfc6831d66` |
| Shared constexpr-stride kernel | `1ab79bf25f51947ecd883a1e80c265d44a5086d1e7de54064f29365c2bcc70ff` |
| Shared runtime-stride kernel | `43111c26ca71e85f76c1f854a464af030214681bdb29feedbb1cfef15ad058cf` |
| Official-weight benchmark loader, all runs | `9b51d00dca1ea4b6c12c7d3a841063ea205cb58235bd8dd24397fb423bd8c9c6` |

Use the JSON's source hashes when associating a result with source; later edits
to a benchmark or kernel do not retroactively qualify those edits. The original
reports do not contain `rope_source_sha256`. The pre-shared kernel snapshot is
retained as `original_rope_kernel.py`, SHA-256
`ba3c7e740c3a2a9c8850733d3be77fefac79bc3ed02a45b4be076b58d6f7632d`.
Benchmark snapshots are `before_shared_rope.py`, `shared_rope_benchmark.py`,
and `output_overlap_benchmark.py`; the shared kernel snapshots are
`shared_rope_constexpr_kernel.py` and
`shared_rope_dynamic_kernel.py`. Each matches the corresponding hash above.

The 272 + 128 output-overlap schedule meets the advancement gate for this
selected attention interval: it beats equally optimized same-run padding,
survives an independent repeat, and exposes the intended communication overlap
in separate traces. This does not qualify a full decoder, other batches or
default buckets, or serving throughput. Earlier all-reduce-plus-QKV boundary
wins remain separate experiments with a narrower scope.

### Next boundary to qualify

The next structural experiment is to preserve the request partitions beyond
O, allowing A's post-attention normalization and real MLP to start when its
output reduction completes, while B's final output reduction is still running.
This targets the observed roughly 100-135 us exposed communication tail without
requiring advance knowledge of the next replay's inputs. The current prototype
does not implement that continuation.

The comparison must include the same real post-attention work for padding,
retain a shared full-bucket dense baseline, and measure all partitioned GEMMs,
joins, and dependencies. Additional dense fragmentation or communication
contention may outweigh the hidden tail; a win is not implied by the current
attention-only result. A real preceding shared Down producer is also needed
to remove the synthetic-input assumption at the earlier boundary.

Full hybrid/GDN state, routing between default capture buckets, and engine
integration remain separate qualification requirements. Default capacities,
truthful host lengths, shared generic optimizations, and the same communicator
and numerical thresholds remain part of every comparison.

### Local validation

The focused CPU checks pass: 17 attention benchmark tests and 16 fused MRoPE
wrapper tests. From the repository root, the commands are:

```bash
python tools/run_multi_graph_cpu_tests.py -k tp_attention
python -m pytest -q --confcutdir=tests/ut/ops \
  tests/ut/ops/test_split_qkv_rmsnorm_mrope.py
```

These validate metadata, provider bindings, and wrapper behavior;
the numerical and captured-runtime evidence comes from the NPU runs above.
Ruff and the long-function, symbolic-metadata, and logger checks pass. The
full `format.sh ci` check was attempted but is not fully green in the Windows
workspace: existing codespell findings and the direct Triton import in
`device_length_decode`, plus an unavailable hook `/bin/bash` path, missing
`shellcheck`, and an unavailable `python3` launcher, leave broader checks
unresolved. Git Bash was used to invoke `format.sh`; its availability does not
resolve those hook paths. These local tooling limits are separate from the
completed NPU correctness and timing qualification.
