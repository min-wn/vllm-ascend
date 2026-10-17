# Tensor-parallel communication boundary experiment

## Mechanism and scope

The candidate splits a tensor-parallel all-reduce at request boundaries and
allows the next local QKV projection to consume the first completed portion.
The second collective can then overlap the first QKV projection. The intended
full-model boundary is a row-parallel Down projection, its all-reduce, the next
layer's normalization, and the next layer's QKV projection.

Keep the preceding Down GEMM shared across the complete default bucket. Its
output already has contiguous request slices, so the candidate does not need to
split that GEMM just to expose two communication payloads. The next QKV GEMM is
split, and its fragmentation cost must be paid by actual overlap.

This differs from running two complete decoder branches or two competing FIA
kernels. The opportunity is to release an independent request group before the
whole collective has completed. It does not remove a mathematical dependency:
each QKV slice must wait for the reduction of its own input slice.

`benchmarks/multi_graph_tp_boundary.py` isolates all-reduce followed by QKV.
It uses synthetic rank-local inputs and an actual loaded QKV weight shard:
the first layer for Qwen3, or the first full-attention layer for Qwen3.5.
Down, residual addition, normalization, attention, FIA updates,
input preparation, and the rest of the decoder are outside this experiment.
An operator-boundary improvement therefore cannot establish decoder latency,
serving throughput, or speculative decoding speedup.

## Default capacity contract

The batch is selected once with seed 20261016 over the existing range 64 to 256.
The selected batch is 98. `EngineArgs` and `CudagraphDispatcher` determine the
default token capacities; the experiment does not add capture sizes or resample
the batch to find a favorable case.

| Query tokens per request | Live requests | Padding tokens | Split tokens | Split request capacities |
| --- | ---: | ---: | ---: | ---: |
| 1 | 98 | 104 | 56 + 48 | 56 + 48 |
| 4 | 98 | 400 | 200 + 200 | 50 + 50 |
| 4 | 98 | 400 | 272 + 128 | 68 + 32 |

For Q4, padding capacity is 100 requests. Token capacities must be divisible by
the query count, and both split capacities must belong to the default token
capture list and sum to the selected padding capacity. In particular, Q4 does
not reuse the Q1 request capacities by multiplying 56 and 48 by four.
The default is the most balanced legal pair. An explicit `--pair` must still
use two existing default token capacities, each divisible by the query count,
whose sum equals the dispatched padding capacity. The 272 + 128 candidate
changes the dependency schedule within the same bucket; it does not introduce
a custom graph capacity or split attention heads.

Q4 selection creates an `EngineArgs` configuration with
`speculative_config={"method": "ngram", "num_speculative_tokens": 3}` solely to
derive the default uniform-query token capacities. The dispatcher initializes
its keys with query length 4. The model loader does not enable a draft model or
run ngram proposals, acceptance, sampling, or the speculative scheduler. This
configuration-only selection is not qualification of an executable complete
speculative graph or of the loader's speculative metadata path.

Captured graph identities, input addresses, and output addresses remain fixed
when input contents and live request counts change. Dummy input rows are zero.
Changed-input qualification covers legal live counts inside each selected
bucket. Timed replay trains reuse prepared inputs; these measurements are not
per-step dynamic metadata preparation benchmarks.

## Three dependency schedules

All schedules use the same model shard, BF16 arithmetic, NZ QKV weights, request
ordering, aggregate communication payload, and tensor-parallel process group.

| Schedule | Dependency order | Purpose |
| --- | --- | --- |
| `joined` | All-reduce the full bucket, then one full-bucket QKV | Padding boundary baseline |
| `split_serial` | All-reduce A, all-reduce B, then QKV A and QKV B | Same split work without communication/compute overlap |
| `split_overlap` | All-reduce A, release QKV A while all-reduce B runs, then QKV B | Candidate schedule |

The overlap graph uses an explicit communication stream and readiness events.
Every rank issues collective A before collective B. The compute stream waits
for each corresponding readiness event before reading reduced input. Completing
the graph must also cover both communication results and both QKV outputs.

The experiment additionally measures full and sliced all-reduces and full and
split QKV projections. These controls distinguish collective startup costs,
GEMM fragmentation, and overlap. Differences between separately measured stages
are diagnostics, not additive predictions for graph wall time.

## Communication cost and fairness

A simple collective model is `T(n) = alpha + beta * n`, where `alpha` represents
startup and `beta * n` represents payload-dependent work. Splitting the same
payload into two calls pays two startups. Under this model, completing A early
does not save `alpha`; it only exposes part of the payload transfer before B
finishes. Additional events, QKV fragmentation, and HCCL/compute contention can
consume that opportunity. The model is a screening explanation, not a fitted
latency formula for HCCL on this machine.

The relevant objective is the measured completion time of `split_overlap`
against `joined`. Beating `split_serial` alone only shows that overlap hides
some of the additional split work. It does not show a benefit over padding.
An overlap percentage in a trace is also insufficient when concurrent kernels
or collectives become slower.

All modes call the loaded model's `get_tp_group().all_reduce`. The report records
the actual communicator type. A direct PyHccl path is not substituted only for
the split candidate: the repository's PyHccl helper is not evidence that the
model's default TP all-reduce already uses that helper. Any future communication
backend optimization must apply to the padding baseline as well.

QKV outputs are checked against original ND weights at `rtol=0.02, atol=0.02`.
The corresponding official model QKV module is also checked against that
reference. Collective output is independently checked against a CPU FP32
reduction of the original per-rank inputs. The script verifies the actual NZ
weight format, requested TP world size, and all-rank numerical results.
Changed inputs and live counts test reuse of fixed captured storage.
These checks qualify this boundary only; they do not qualify full-model hidden
states, residuals, logits, or cross-shape attention behavior.

The installed TP collective modifies its input in place. Every diagnostic
graph therefore restores one full-capacity working buffer from immutable
rank-local input before doing any measured work. The identical restore copy
is included in every schedule and component control. A real preceding Down
projection would produce fresh values instead; its cost and dependencies are
not measured here. The initial pilot omitted this restore and failed numerical
qualification through repeated summation. Its log is retained as a harness
failure, and no timing from that pilot is used.

## Measurement and next gate

Formal timing excludes profiler collection, model loading, capture, reference
generation, and input preparation. Replay trains amortize host submissions and
end with device completion. CPU rank barriers precede the timed region. Results
aggregate the slowest rank per sample before summarizing rounds; single-rank
latency is not substituted for collective completion time. Device event spans
can still contain submission gaps and are not pure kernel execution times.

Selected-layer QKV weights are reused and hot. Synthetic inputs stand in for a
previous layer's rank-local output. These conditions limit extrapolation to
multiple layers with different weights and actual normalization dependencies.

A boundary win against equally optimized `joined`, confirmed by an independent
repeat and the intended overlap in a separate trace, is a fast path to a
two-layer prototype. It is not a necessary condition for the complete
QKV-plus-FIA dependency graph: this diagnostic deliberately stops the first
branch after QKV, even if attention could use its completed output while the
second collective continues. A negative result rejects the measured boundary
schedule, not every schedule containing additional useful downstream work.
Beating `split_serial` alone still does not establish a padding advantage.

After a boundary loss, further work requires a concrete unfilled interval and
a valid dependency path for work that can occupy it. The Q4 trace below exposes
about 51 us between QKV A completing and collective B becoming ready. Whether
FIA A can fill that interval depends on its real normalization, Q/K processing,
KV writes, and completed host-length updates, and on interference with QKV B.
It is a hypothesis requiring measurement, not a 51 us predicted decoder saving.

The two-layer prototype must retain shared preceding Dense work, restore actual
residual, Norm, QKV, and FIA dependencies, then join for shared O projection.
It must use real model inputs and KV state and perform every required FIA
update with true host sequence lengths. Padding must receive the same generic
improvements, including applicable communication fusion. Only an ensuing
full-replay measurement can establish a model-level advantage; boundary results
must not be multiplied by layer count and presented as measured decoder savings.

## TP2 Qwen3-4B results

The qualified runs use BF16 Qwen3-4B on two 910B3 devices, three alternating
rounds, eight samples per mode per round, and eight graph replays per sample.
Both Q1 and Q4 passed all 63 boundary checks on each rank, including changed
inputs and live request counts. These are individual qualified runs, not
independently repeated performance claims. Times are microseconds per replay;
each column uses the slowest rank per sample and the median across rounds.
Slowdown is relative to the corresponding `joined` metric.

| Query | Schedule | Device event us | Wall us | Event slowdown | Wall slowdown |
| --- | --- | ---: | ---: | ---: | ---: |
| Q1 | `joined` | 106.833 | 113.854 | 0.00% | 0.00% |
| Q1 | `split_serial` | 141.841 | 149.039 | 32.77% | 30.90% |
| Q1 | `split_overlap` | 127.694 | 134.850 | 19.53% | 18.44% |
| Q4 | `joined` | 204.724 | 212.074 | 0.00% | 0.00% |
| Q4 | `split_serial` | 256.331 | 262.947 | 25.21% | 23.99% |
| Q4 | `split_overlap` | 225.225 | 232.477 | 10.01% | 9.62% |

The independent component diagnostics use device event measurements:

| Query | Full AR minus AR A | Split AR minus full AR | Split QKV minus full QKV | Overlap saving against same serial split |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 4.556 us | 28.586 us | 0.484 us | 14.147 us |
| Q4 | 55.292 us | 29.800 us | 5.511 us | 31.106 us |

The payload is 532,480 bytes for Q1 and 2,048,000 bytes for Q4. The report records
`NPUCommunicator` and the same `get_tp_group().all_reduce` path on both sides.
The separate traces contain BF16 HCCL `hcom_allReduce` events using
`MESH-RING-NHR`; their element counts match the full and sliced payloads.

## What the traces establish

Rank 0 has two separately profiled samples per schedule. For Q1, QKV A takes
15.940/15.800 us and overlaps collective B by 15.920/15.780 us. Collective B
takes 43.881/44.801 us and still has 27.961/29.021 us remaining after QKV A.
The corresponding serial split has no such overlap. The schedule therefore
creates real communication/compute concurrency, but Q1's small first-portion
release advantage does not pay for the additional collective.

Q4 has a materially larger release opportunity and should not be explained
only as a small-message startup problem. Its trace shows:

| Q4 device observation | Sample 1 | Sample 2 |
| --- | ---: | ---: |
| Joined QKV400 | 42.241 us | 43.001 us |
| Serial split QKV200 pair, duration sum | 63.761 us | 62.621 us |
| Overlap split QKV200 pair, duration sum | 62.761 us | 62.701 us |
| Serial split collective B | 84.102 us | 84.102 us |
| Overlap split collective B | 84.161 us | 84.101 us |
| QKV A / collective B intersection | 33.381 us | 33.661 us |
| Compute-stream wait after QKV A | 51.941 us | 51.301 us |

Neither collective B nor the QKV pair visibly slows under overlap relative to
the serial split in these samples. Readiness handoff is about 1.2 us, while
individual event resets are sub-microsecond; their observed intervals do not
explain the approximately 20 us formal boundary loss. Long event waits are
primarily dependency waits, not evidence of an intrinsically expensive event.

The late branch remains decisive: QKV A finishes well before collective B,
then QKV B still takes about 29 us. Compared with a full QKV of about 43 us,
only part of the final computation tail disappears. The isolated early-release
estimate of 55.292 us cannot be subtracted from total boundary latency. In an
idealized schedule, `T_split = AR_A + max(AR_B, QKV_A) + QKV_B`; when collective
B exceeds QKV A, making A finish still earlier does not move final completion.
The QKV-pair fragmentation visible after communication is also larger than the
independent QKV-only difference, illustrating why component medians cannot be
added to explain a combined graph exactly.

These traces do not prove absence of resource contention in other cases.
HCCL intervals include synchronization and waiting, not only link transfer.
The first joined trace sample is longer than the second, and only rank 0 is
analyzed here; trace intervals must not be mixed with the unprofiled all-rank
medians as an exact timing decomposition. They establish concurrency and the
remaining dependency window, not a complete-model performance result.

Qualified reports and rank-0 traces are archived as `tp2-q1-restored` and
`tp2-q4` under local `outputs/multi_graph_tp_boundary/` and remote
`/workspace/multi-graph-results/tp-boundary/`. The earlier Q1 harness failure
is excluded. No complete two-layer or full-replay speedup is established by
these measurements.

## Larger models and numerical qualification

The available model paths include Qwen3-8B and Qwen3.5-27B-original. The latter
has 64 layers, comprising 48 linear-attention/GDN layers and 16 full-attention
layers. Its first full-attention layer is index 3. QKV also contains the query
output gate: the packed width is 14,336, not the plain-QKV width of 8,192.
The loaded local weight shapes are `[3584, 5120]` at TP4 and `[7168, 5120]`
at TP2. The loader verifies these shapes against the actual attention module.

`multi_graph_tp_boundary_loader.py` uses the official external-launcher
executor to initialize workers, TP communication and checkpoint weights.
It stops before engine profiling, KV allocation, prefill or model execution.
The benchmark captures only the communication/projection boundary. This allows
testing a real Qwen3.5 projection without changing the existing full-decoder
loader's Qwen2/Qwen3 and single-KV-group guards. Loading the full checkpoint is
not qualification of the complete hybrid model's graph mode.

Both Qwen3-8B TP4 and Qwen3.5-27B TP4 reached graph capture and passed the joined
boundary check, but failed the split-serial output check at the unchanged
`rtol=0.02, atol=0.02` threshold. No performance result is accepted for either
four-rank configuration. The Qwen3-8B diagnostic reproduces the failure in a
separate process launch.

The untimed numerical controls isolate the source:

- Holding the full collective's output fixed, both ND and NZ split projections
  pass against the ND full projection on every rank.
- Full and sliced collective outputs each pass the elementwise input check,
  but differ by up to 0.03125 in these BF16 inputs.
- Applying the same ND full projection to the two collective outputs produces
  failing elements: 64 to 86 per rank for 8B, and 2 to 18 per rank for 27B.
  Thus changing GEMM shape or NZ layout is not required to reproduce the issue.

The evidence points to BF16 collective rounding changing with payload shape,
then being amplified by projection. It does not show a broken collective:
different reduction orders can legitimately round differently. It also does
not qualify the split path under the current final-output criterion. These
are synthetic boundary inputs, so the result must not be generalized to an
entire model's logits. `HCCL_DETERMINISTIC=true` is enabled throughout; repeat
determinism does not guarantee identical rounding across collective shapes.
All ranks start with the same random sample scaled by `0.1 * (rank + 1)`.
Consequently, the nominal reduced amplitude is 0.3 at TP2 and 1.0 at TP4, and
this diagnostic omits the next layer's real Norm. The TP2/TP4 accuracy outcomes
are not a controlled comparison with equal post-reduction input distributions.
Within each configuration the competing schedules use identical inputs.
Failures are retained per rank so an error on a nonzero rank is not replaced
only by a downstream connection error from rank zero.

Qwen3.5-27B TP2 passes all 63 checks on both ranks in two independent launches.
With the default balanced 200 + 200 split, the results are:

| Run | Joined event us | Overlap event us | Event reduction | Joined wall us | Overlap wall us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 402.794 | 395.825 | 1.73% | 411.215 | 403.590 |
| Independent repeat | 402.058 | 391.390 | 2.65% | 408.845 | 398.386 |

Rank-0 trace confirms QKV A overlaps collective B for about 86 us, leaving
about 53 us of communication after QKV A completes and about 83 us of QKV B
after communication. This larger projection hides enough work for a modest
positive boundary result. Model size alone is not an explanatory variable:
projection shape, local shard size, collective payload and TP degree all
change the balance. In particular, results from TP2 and TP4 are not a controlled
measurement of model-size scaling.

## Balancing communication and computation

The next candidate chooses the split by the critical path rather than by equal
request counts. For `AR_A + max(AR_B, QKV_A) + QKV_B`, enlarging A and shrinking B
can fill the first-branch wait and reduce the final QKV tail simultaneously.
The target is approximately `QKV_A = AR_B`, subject to existing default sizes.

A local interpolation of the measured 27B TP2 trace puts the intersection near
270 tokens. The existing 272 + 128 pair is therefore a causal single candidate,
not a search across special capture shapes. At Q4 it represents capacity for
68 + 32 requests; B98 uses 68 + 30 live requests. Kernel tiling and HCCL algorithm
thresholds can be nonlinear, so this interpolation selects a measurement and
does not predict a guaranteed speedup.

The candidate passes all 63 checks per rank in both launches, including
changed inputs and live counts B98, B97, B99, and B98 with unchanged graph and
output identities. Its independently repeated boundary measurements are:

| Run | Joined event us | Overlap event us | Event reduction | Joined wall us | Overlap wall us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 401.526 | 379.160 | 5.57% | 409.361 | 387.261 |
| Independent repeat | 401.946 | 379.918 | 5.48% | 409.302 | 387.369 |

The corresponding serial split is 452.694/453.389 us in device timing. The
gain therefore depends on overlap; merely dividing the work still loses.

The separate rank-0 trace shows collective B taking 97.442/98.642 us, fully
inside QKV A's 120.003/120.622 us interval. The earlier 53 us communication wait
disappears, and QKV B takes only 51.501/51.781 us with a roughly 0.060 us launch
gap. This supports the intended dependency change. However, QKV A at the same
272-token size takes only 99.682/100.822 us in the serial schedule. Its roughly
20 us extension under overlap consumes part of the gain; the trace alone does
not identify whether bandwidth or another shared resource causes it.

This is a measured improvement for a real 27B projection at TP2, with default
capacities and fixed-graph reuse. It remains a synthetic communication/QKV
boundary, without real residual/Norm, FIA updates, attention or a complete
hybrid decoder. It does not establish 5.5% full-model acceleration or optimality
of 272 + 128. A subsequent prototype must test the actual omitted dependency
chain and the equally optimized padding path before a model-level claim.

## Reproduction and validation

The measured machine has eight 64 GB 910B3 devices. These experiments use two
or four devices with the existing driver 24.1.rc3, CANN 9.1, PyTorch 2.10,
torch_npu, and vLLM 0.30 environment. The driver is not upgraded. From the
configured container repository, the qualified asymmetric experiment is:

```bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_BATCH_INVARIANT=0 \
HCCL_DETERMINISTIC=true python -m torch.distributed.run \
  --standalone --nproc-per-node=2 benchmarks/multi_graph_tp_boundary.py \
  --model /models/Qwen3.5-27B-original --loader qwen35-weights \
  --tensor-parallel-size 2 --query 4 --pair 272 128 \
  --rounds 3 --repeats 8 --train 8 --trace \
  --output /workspace/multi-graph-results/tp-boundary/new-run.json
```

Use a new output path for each run. Omit `--pair` for 200 + 200 and `--trace`
for a repeat without post-timing profiling. Qwen3 uses the default `replay`
loader; Qwen3.5 uses the explicit weight-only loader. Formal timing always
precedes and excludes profiling. Every report records its source SHA256;
Qwen3.5 reports also record the loader hash and actual shard metadata.

Artifacts are under remote `/workspace/multi-graph-results/tp-boundary/` and
local `outputs/multi_graph_tp_boundary/`, including logs, rank traces, JSON
reports and matching source snapshots. Qualified positive runs are
`tp2-27b-q4`, `tp2-27b-q4-repeat`, `tp2-27b-q4-272-128`, and
`tp2-27b-q4-272-128-repeat`. The failed TP4 cases are retained separately and
must not be counted as performance-qualified runs.

Targeted Ruff, Python syntax, import-policy and boolean-context checks pass.
The isolated CPU bucket and per-rank failure contracts pass 44 tests. The new loader's
CPU interface checks cover invalid projection shapes/options and cleanup on
callback failure, and its real official weight loading is verified at TP2/TP4.
The full `bash format.sh ci` check was run: formatting and Markdown checks pass,
but the repository check remains blocked by pre-existing spelling/direct
Triton imports and Windows `/bin/bash`, shellcheck and python3-launcher issues.
No full-model or serving test is claimed for this operator-only addition.
