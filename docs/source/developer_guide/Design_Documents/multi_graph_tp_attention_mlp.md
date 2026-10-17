# Tensor-parallel attention and MLP continuation

## Question and current status

Can a request group's real post-attention normalization and MLP hide the other
group's output-reduction tail, after paying for fragmented MLP projections and
the final Down reductions?

`benchmarks/multi_graph_tp_attention.py --include-mlp` extends the
[attention boundary experiment](multi_graph_tp_attention.md) to the selected
Qwen3.5-27B decoder layer's returned hidden states and residual. The attention
experiment demonstrated a repeatable gain for 272 + 128 on its narrower
endpoint. That result does not establish a gain after adding MLP.

The original split-QKV MLP pilot failed the unchanged numerical gate. Untimed
diagnostics isolate a shape-dependent QKV projection difference on the failing
inputs. A new explicit `--shared-qkv` schedule is implemented to keep input
reduction, input norm, and QKV at full-bucket shape while retaining request
splits downstream. Its captured pilot and four formal runs pass the unchanged
numerical checks. The 272 + 128 pair has no repeatable padding win; the legal
200 + 200 pair reduces full selected-layer replay wall latency by 2.81% and
2.55% in two independent runs. These are selected-layer results with synthetic
entry states, rather than full-model or serving qualification.

## Fixed workload and scope

The initial comparison keeps batch 98, Q4, TP2, maximum KV length 512, and the
default 400-token capture bucket. The existing default split is 272 + 128
tokens, or 68 + 32 request capacity, with 68 + 30 live requests. Both split
capacities already exist in the default capture list. Padding and split pay
for the same total capacity; no bespoke capacity or attention-head split is
introduced.

The first full-attention layer is layer 3. The experiment now uses its actual
input normalization, QKV, Q/K normalization, RoPE, FIA, output gate, O,
post-attention normalization, merged GateUp, activation, Down, and ordinary TP
reductions. Timed projections use NZ weights; eager decomposition uses ND
weights as a reference. The loaded MLP's activation is used directly.

The entry rank-local preceding Down outputs and residuals remain synthetic.
The preceding Down GEMM, earlier decoder layers, GDN state, full-model prefill,
sampling, draft execution, acceptance, serving, and engine scheduling remain
outside this experiment. The synthetic projected KV prefix and benchmark-owned
contiguous ND cache retain the attention experiment's scope limitations.

Q4 uses the ngram configuration only to derive default capture sizes. This
does not execute a drafter or measure acceptance-dependent throughput.

## Decoder semantics and returned residual

Let `D0` be the synthetic local preceding Down output and `R0` the input
residual. The selected layer follows this chain:

```text
H0 = AR_input(D0)
(X1, R1) = input_norm(H0, R0)           # R1 = H0 + R0
A = attention_and_gate(X1)
H1 = AR_O(O(A))
(X2, R2) = post_attention_norm(H1, R1)  # R2 = H1 + R1
L = Down(activation(GateUp(X2)))
H2 = AR_Down(L)
return H2, R2
```

The decoder returns the reduced Down output and `R2` separately. It does not
return `H2 + R2`. Both input and post-attention normalization return their
updated residual explicitly. Captured execution retains these returned values
without assuming that normalization mutates the input residual in place.
All schedules restore the original source inputs and write the input norm's
returned residual to stable storage before subsequent work.

Only unscaled dense full-attention layers are accepted. GateUp and Down must
be ordinary bias-free BF16 TP shards with compatible shapes and no custom
projection operator. GateUp does not gather its output; Down receives parallel
input and reduces its result. Explicit local Down is followed by exactly one
real reduction: the official MLP return is not reduced a second time.

The ordinary O and Down constraints leave fused MC2 or
`npu_mm_all_reduce_base` paths outside this qualification. All candidates and
padding use the same loaded communicator and generic optimizations.

## Three schedules with the same endpoint

`--include-mlp` selects exactly these three modes and enables the required
output-overlap path. `--shared-rope` remains an explicit common optimization.

| Mode | Attention/O schedule | MLP schedule | Purpose |
| --- | --- | --- | --- |
| `joined` | Full bucket | Full bucket | Padding baseline |
| `split_output_overlap` | Request-split O pipeline; input/QKV placement depends on the explicit flag below | Join, then full bucket | Isolate the attention scheduling change |
| `split_mlp_overlap` | Same attention/O pipeline | Partitioned post-norm, GateUp, activation, Down and reduction | Test the incremental MLP continuation |

All modes end only after complete reduced hidden states and the returned
residual are available. Comparing only the third mode with padding could
mistake the existing attention improvement for a new MLP scheduling benefit.
The second mode is therefore a required control.

Let `P_X` contain input norm through gated attention for group X, and let
`F_X` contain post-norm, GateUp, activation, and local Down. The MLP itself
does not add FIA tasks. The following original schedules remain selected when
`--shared-qkv` is absent; their MLP pilot failed as recorded below.

```text
joined
root: restore -> AR_input400 -> P400 -> O400 -> AR_O400
      -> F400 -> AR_Down400 -> complete(hidden, residual)

split_output_overlap
root: restore -> wait input_A -> P_A -> O_A
      -> wait input_B -> P_B -> O_B
      -> wait output_A and output_B -> join O results
      -> F400 -> AR_Down400 -> complete(hidden, residual)
comm: AR_inputA -> AR_inputB -> wait O_A -> AR_OA
      -> wait O_B -> AR_OB

split_mlp_overlap
root: restore -> wait input_A -> P_A -> O_A
      -> wait input_B -> P_B -> O_B
      -> wait output_A -> F_A -> wait output_B -> F_B
      -> wait Down_A and Down_B -> join hidden and residual -> complete
comm: AR_inputA -> AR_inputB -> wait O_A -> AR_OA
      -> wait O_B -> AR_OB -> wait local Down_A -> AR_DownA
      -> wait local Down_B -> AR_DownB
```

Captured events supply the omitted projection-ready and reduction-done edges.
Both O reductions are submitted before either Down reduction. Every rank
uses the same collective order:

| Mode | Collective order |
| --- | --- |
| `joined` | input, output, down |
| `split_output_overlap` | input_A, input_B, output_A, output_B, down |
| `split_mlp_overlap` | input_A, input_B, output_A, output_B, down_A, down_B |

The root stream runs all computation sequentially. This does not launch two
FIA kernels concurrently or run an MLP against an unfinished attention result.
The intended additional windows are `F_A` against `AR_OB`, followed by `F_B`
against `AR_DownA`. Starting `F_A` still depends on the root finishing `O_B`
and the first output reduction completing.

Collectives may return different storage from their input. The implementation
retains local projection buffers, actual reduced buffers, and post-norm
residuals. The shared-MLP control copies both actual O reduction outputs into
the full bucket before running MLP. The split-MLP mode directly consumes each
reduced O output and eventually copies the actual reduced Down results and
returned residuals into full-capacity outputs. All these joins are timed.

## Shared input reduction, normalization, and QKV

`--shared-qkv` requires `--include-mlp` and changes the front of all three
same-endpoint controls. Each replay computes the current full-bucket input
reduction, input norm, and QKV once on the root stream. Padding consumes the
whole QKV result. Split schedules consume its existing 272- and 128-token
views for Q/K norm, RoPE, KV writes, FIA, gate, O, and the selected MLP schedule.
The flag is opt-in; it does not silently replace earlier split-QKV results.

Let `T_X` be Q/K norm through gated attention, after raw QKV projection. With
the selected common `--shared-rope` option, the new shared prefix and
split-MLP candidate are:

```text
common root prefix:
  restore400 -> shared lookup400 -> AR_input400 -> input_norm400 -> QKV400

joined continuation:
root: T400 -> O400 -> AR_O400 -> F400 -> AR_Down400 -> complete

split_mlp_overlap continuation:
root: T_A -> O_A -> T_B -> O_B
      -> wait output_A -> F_A -> wait output_B -> F_B
      -> wait Down_A and Down_B -> join hidden and residual -> complete
comm: wait O_A -> AR_OA -> wait O_B -> AR_OB
      -> wait local Down_A -> AR_DownA
      -> wait local Down_B -> AR_DownB
```

The shared-MLP control has the same split attention/O continuation but waits
for both O reductions, joins their actual results, and runs `F400` plus one
`AR_Down400`. With the flag enabled, collective orders become:

| Mode | Collective order |
| --- | --- |
| `joined` | input, output, down |
| `split_output_overlap` | input, output_A, output_B, down |
| `split_mlp_overlap` | input, output_A, output_B, down_A, down_B |

All prefix operations remain inside the timed replay and read current inputs.
This shares dense computation, not values cached from a previous step, and
does not require knowing future inputs. The full bucket still comes from
ordinary default capture sizes, and the downstream split remains by requests.

The tradeoff is substantial: input reduction B can no longer overlap work for
A, and QKV for B is no longer available as work under A's output reduction.
The earlier 3.56-4.55% attention-only improvement used a different DAG and
cannot be carried forward as a prediction. Any benefit from preserving QKV
shape and dense efficiency must be measured against equally shared padding
through the complete MLP endpoint.

## Correctness and timing contract

Joined registers one real FIA update task per replay; both split schedules
register two, with separate workspaces and lane metadata. Every replay resolves
current host sequence lengths and executes real graph updates. Root completion
also waits for update completion. None of these costs is omitted.

Initial eager checks compare the explicit attention/O and MLP decomposition
with the loaded official modules. The complete selected decoder layer serves
as an additional eager reference for each dynamic input case, checking both
returned hidden states and residual. Every captured mode checks both outputs
initially, after B98 -> B97 -> B99 -> B98 changes, and after each timing round.
Input values, positions, lengths, and slots change at stable captured addresses.
Graph, output, workspace, metadata, and cache identities are checked for reuse.

The tolerance remains `rtol=0.02, atol=0.02`; the shared-RoPE view preflight
retains its exact `rtol=0, atol=0` comparisons. A failed numerical or runtime
check invalidates the performance result.

Each sample times one replay through every O and Down projection, all actual
reductions, result copies, and completion. Wall timing includes task resolution,
host submission, and FIA updates. Device events include dependency waits and
may include submission gaps; they are not sums of pure kernel durations.

CPU rank barriers, reference generation, input preparation, and cache restore
are outside timing. All schedules pay the same captured source-restore copies.
The formal comparison uses three rounds with alternating mode order and eight
samples per mode and rank. Reports take the slowest rank per sample, median
across samples, then median across rounds. Separate profiler replays identify
the actual overlap without being substituted for formal latency results.

## Run and advancement gate

Use a fresh output path. This short run qualifies the runtime before a formal
comparison:

```bash
source /etc/profile.d/ascend-multigraph.sh
ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_BATCH_INVARIANT=0 \
HCCL_DETERMINISTIC=true \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  benchmarks/multi_graph_tp_attention.py \
  --model /models/Qwen3.5-27B-original --tensor-parallel-size 2 \
  --query 4 --pair 272 128 --kv-length 512 \
  --shared-rope --include-mlp --shared-qkv --rounds 1 --repeats 2 \
  --output /workspace/multi-graph-results/tp-boundary/attention-mlp-pilot.json
```

After qualification, use `--rounds 3 --repeats 8 --trace`. A positive result
requires an independent repeat using identical source hashes and settings.
Reports include `include_mlp`, the expanded scope, actual local MLP shapes,
collective order, output-join layout, Down reduction counts, and per-rank checks.

Record two separate same-run differences:

```text
total_gain = joined_time - split_mlp_overlap_time
incremental_gain = split_output_overlap_time - split_mlp_overlap_time
```

Both differences must be positive and repeatable to attribute a net padding
win to the MLP continuation. Trace evidence must show work overlapping the
real reduction tail without moving the final wait outside the timer. If the
second difference is not positive, retain the shared-MLP control as the better
schedule for this workload, even if both split modes beat padding.

The completed attention-only results remain in the linked attention document
and are not reused as measurements for this longer endpoint.

## Original pilot failure and causal diagnostics

The original split-QKV pilot stopped at
`split_output_overlap/b99-v2/hidden` on both ranks, before a usable performance
result. One of 2,027,520 compared elements failed: index `[123, 3994]` was
0.15625 versus 0.125, an absolute error of 0.03125. The permitted combined error
there is `0.02 + 0.02 * 0.125 = 0.0225`. The test remains failed even though
most elements pass.

The numeric diagnostic records errors without stopping so their source can
be examined. It reproduces the same failing element in both split modes and
also finds `[99, 3994]` on B98 variant 3: -1.59375 versus -1.65625, an error of
0.0625 against a permitted 0.053125. Joined matches the reference on these
cases. Using the split attention/O result as the input to a shared MLP
reproduces each failure with either ND or NZ MLP weights. Switching only the
MLP weight layout therefore does not resolve the discrepancy.

The completed causal v2 diagnostic compares captured outputs with eager
execution of the same shapes. Hidden and residual outputs agree exactly in
the tested failing variants, including the split schedules. This supports a
numerical difference between shapes rather than a captured scheduling error
for those cases. Supplying full-bucket QKV to split FIA removes the observed
attention difference; splitting O afterward still passes the final tolerance.

Causal v3 then changes the QKV projection shape while retaining the original
inputs and downstream checks. It covers both failing variants on both ranks:

| Controlled change | Observation |
| --- | --- |
| Split input norm and residual only | Exact match with full-bucket results |
| Full-bucket QKV, ND or NZ weight | Exact raw QKV match; final hidden checks pass |
| QKV at 272 + 128, ND or NZ weight | Raw QKV differs; the same final hidden failures reappear |
| Full-bucket raw QKV, split Q/K norm and RoPE | Exact transformed Q/K/V/gate and attention match; final hidden checks pass |

The split raw-QKV maximum difference is 0.015625 on both ranks for B99 variant
2. For B98 variant 3 it is 0.03125 on rank 0 and 0.015625 on rank 1. These
intermediate comparisons pass their elementwise tolerance, but the downstream
layer can amplify the difference until a final hidden value fails. ND QKV
weights retain the same failure as NZ, so an ND-only conversion is not a fix.

This establishes a QKV GEMM shape-dependent numerical path as the source of
the reproduced discrepancy under these controls. It does not prove a specific
internal CANN tiling, accumulation scheme, or kernel-selection mechanism, or
that all future inputs will be qualified by shared QKV. The shared-QKV captured
graph is separately checked against the unchanged gate. Original inputs,
tolerance `rtol=0.02, atol=0.02`, and exact RoPE preflight remain unchanged.

### Diagnostic artifacts and exact sources

Artifacts are in `outputs/multi_graph_tp_attention_mlp/`, outside the checkout.
The diagnostic JSON files are explicitly marked
`diagnostic_only_not_performance_qualified`; their recorded checks include
failures and must not be counted as a passed numerical qualification.

| Artifact | Source snapshot | SHA-256 |
| --- | --- | --- |
| `tp2-27b-mlp-272-128-pilot.json` | `pilot_benchmark.py` | `d6747bbfef978a16754dcd71b66c9539184a1ce027fdba34d244f6d568eae761` |
| `tp2-27b-mlp-numeric-diagnostic.json` | `numeric_diagnostic.py` | `e916d998ef12b55078e279e409dd4fec193f0eeb75b4e311d2bb7a2363ff5ae1` |
| `tp2-27b-mlp-causal-diagnostic-v2.json` | `causal_diagnostic_v2.py` | `07aafa529990a79fa3112386896346acf379aac30cef5ff5da9009d18feb7a5d` |
| `tp2-27b-mlp-causal-diagnostic-v3.json` | `causal_diagnostic_v3.py` | `f4f766776abbe1831184a71cd05625f8d823458f347ec816889fc36f652e0c70` |

The numeric, causal v2, and causal v3 reports contain 92, 157, and 118 recorded
checks per rank respectively. The earlier `tp2-27b-mlp-causal-diagnostic.json`
attempt failed its graph/output identity check and is not used as successful
causal evidence. Its snapshot is `causal_diagnostic.py`, SHA-256
`424a1e459f0d172ba20e732f9c1b9be4d857bd74f7a3af169bb8bad4f8a59093`.

## Shared-QKV qualification and repeated formal results

The shared-QKV pilot passes 62 checks per rank with one round of two samples.
The first formal 272 + 128 run passes 74 checks per rank, including both hidden
and residual, real FIA updates, dynamic variants, and three timing-round
checks for each mode. Both runs use unchanged inputs and numerical thresholds.
The additional 12 checks in the formal run come from its two extra rounds,
three modes, and two returned tensors. Pilot latency is not used as a formal
comparison.

The actual local GateUp weight shape is `[17408, 5120]`, and local Down is
`[5120, 8704]`. The formal run uses three alternating-order rounds and eight
single-replay samples per mode and rank. Times are microseconds; percentages
are `100 * (mode / joined - 1)` within the same run, so positive means slower.

| Run | Mode | Wall us | Wall vs joined | Device us | Device vs joined |
| --- | --- | ---: | ---: | ---: | ---: |
| Initial 272 + 128 | joined | 2068.94 | 0.00% | 1989.92 | 0.00% |
| Initial 272 + 128 | split_output_overlap | 2078.00 | +0.44% | 2001.20 | +0.57% |
| Initial 272 + 128 | split_mlp_overlap | 2062.43 | -0.31% | 1987.35 | -0.13% |
| Repeat 272 + 128 | joined | 2052.32 | 0.00% | 1989.69 | 0.00% |
| Repeat 272 + 128 | split_output_overlap | 2059.33 | +0.34% | 1996.33 | +0.33% |
| Repeat 272 + 128 | split_mlp_overlap | 2054.12 | +0.09% | 1992.81 | +0.16% |
| Initial 200 + 200 | joined | 2059.03 | 0.00% | 1994.20 | 0.00% |
| Initial 200 + 200 | split_output_overlap | 2064.50 | +0.27% | 2001.88 | +0.39% |
| Initial 200 + 200 | split_mlp_overlap | 2001.18 | -2.81% | 1941.24 | -2.66% |
| Repeat 200 + 200 | joined | 2055.98 | 0.00% | 1993.08 | 0.00% |
| Repeat 200 + 200 | split_output_overlap | 2064.84 | +0.43% | 2003.50 | +0.52% |
| Repeat 200 + 200 | split_mlp_overlap | 2003.50 | -2.55% | 1937.73 | -2.78% |

The 272 + 128 result does not establish a repeatable padding win: its first
run saves 6.51 us, but the independent repeat is 1.80 us slower. Its incremental
gain against the shared-MLP control also shrinks from 15.58 to 5.21 us.

The legal 200 + 200 split passes the same 74 checks per rank in both independent
runs. It saves 57.86 and 52.48 us against same-run padding, or 2.81% and 2.55%.
Against the same-run shared-MLP control it saves 63.33 and 61.34 us, or 3.07%
and 2.97%. Thus this selected-layer workload has a repeatable gain attributable
to extending the pipeline through MLP. The magnitude remains modest; these
measurements do not establish whole-model or serving throughput gains.

The separate first-run trace confirms the intended dependency overlap. In a
representative rank-1 replay, the 103.68 us B output reduction overlaps A's
post-norm/GateUp, and the 181.24 us A Down reduction overlaps B's
post-norm/GateUp. The final B Down reduction remains exposed for 99.00 us.
Four final hidden/residual copies cost 22.34 us and remain inside the graph.

The dominant countervailing cost in this sample is GateUp. The shared-MLP
control's full 400-row projection takes 325.12 us; the split projections take
349.92 + 161.88 = 511.80 us, an increase of 186.68 us. Down, normalization,
and activation also add about 9.76, 14.38, and 10.64 us respectively. These
separate profiler observations show fragmentation and concurrent resource use
offsetting most of the hidden communication; they do not isolate a specific
kernel implementation or form an additive prediction of wall time.

With 200 + 200, the corresponding rank-1 second profiler replay shows
GateUp taking 228.46 + 216.58 = 445.04 us, compared with 314.84 us in its
shared-MLP control. The 130.20 us excess is smaller than the 186.68 us excess
observed for 272 + 128. Down takes 109.22 + 108.60 = 217.82 us versus 205.22 us.
The second O reduction overlaps the first partition's post-norm/GateUp; the
first Down reduction overlaps the second partition's post-norm/GateUp. The
final Down reduction and all four output/residual copies remain exposed and
fully timed. These profiler samples explain the direction of the result,
without substituting kernel sums for the formal replay measurements.

The next optimization target is the local GateUp projection under concurrent
communication. A useful follow-up must reduce its fragmentation penalty while
preserving request partitioning, the real final Down drain, and unchanged
accuracy. Partition selection must consider the complete downstream GEMMs and
communication windows, because the pair preferred by attention alone changes
after sharing QKV. Dynamic batch coverage beyond the tested B98/97/99/98 and
adjacent GDN integration remain open; no additional capacity has been introduced.

### Formal artifacts and source identity

The pilot and formal report are
`tp2-27b-mlp-sharedqkv-pilot.json` and
`tp2-27b-mlp-sharedqkv-formal.json` in
`outputs/multi_graph_tp_attention_mlp/`. Matching formal
`.rank0.trace.json` and `.rank1.trace.json` files contain separate profiler
replays. The independent repeats are `tp2-27b-mlp-sharedqkv-repeat.json`,
`tp2-27b-mlp-sharedqkv-200-200.json`, and
`tp2-27b-mlp-sharedqkv-200-200-repeat.json`, each with matching logs and two
rank traces in that directory. `qualification_audit.json` verifies status,
74 checks per rank, source identity, and same-run gain calculations.
`analyze_mlp_traces.py` and `mlp_trace_analysis.json` retain the separate
profiler GEMM extraction. All runs record the following exact sources:

| Source | SHA-256 |
| --- | --- |
| Benchmark, snapshot `shared_qkv_benchmark.py` | `661c6ee88b870aa3f67cb539aca4070522a442d7eda26dd3134ecc076179285e` |
| Runtime-stride RoPE kernel | `43111c26ca71e85f76c1f854a464af030214681bdb29feedbb1cfef15ad058cf` |

The focused CPU suite has 21 passing tests. Full `format.sh ci` was attempted;
Ruff and Markdown checks pass, while existing codespell findings in the resource probe, the
existing forbidden direct Triton import, and Windows shell/Python launcher
limits leave the full check unresolved. These local limitations do not change
the NPU numerical results or replace further hardware qualification.

## GateUp follow-up: measured candidates and full-model readiness

Two explicit benchmark controls retain the qualified default path:
`--gate-up-format NZ` and `--gate-up-partitions 1`. Changing either applies
identically to padding and both split schedules. All packing and format
conversion occurs before graph capture, while each projection, activation,
intermediate concatenation, Down reduction, and final drain remains timed.
These controls are experimental choices, not automatic runtime tuning.

`--gate-up-format ND` keeps the projection equations unchanged. Its three-round,
eight-sample run passes the same 74 checks per rank. Wall times are 2088.52 us
for joined, 2131.70 us for the shared-MLP control, and 2137.92 us for split MLP.
Split MLP is 2.37% slower than same-run padding. In the separate rank-1 second
profiler replay, GateUp takes 304.84 + 290.04 = 594.88 us in split MLP, versus
371.12 us in joined. This candidate does not resolve the observed penalty and
is not adopted as the default. The original NZ results remain the qualified
positive result; the ND experiment is not a new speedup.

`--gate-up-partitions 2` prepacks two matching pairs of gate/up feature-channel
blocks. Each block applies the loaded activation, then their intermediate
outputs are concatenated in original channel order before the unchanged Down.
It keeps the request split at the legal default 200 + 200 and does not split
attention heads or introduce a capture capacity. CPU tests check its complete
mathematics, residual, and request-sliced composition against the original MLP.

The two-sample, one-round paired-channel pilot passes all 62 checks per rank.
Split-MLP device time is 1989.89 us; the two previous NZ single-block formal
runs gave 1941.24 and 1937.73 us. This short screen does not show an absolute
latency improvement and is not promoted to a formal performance claim. Its
same-pilot wall advantage over its own slower padding baseline must not be
presented as an improvement over the existing best implementation. A larger
matrix is not justified by this screen alone.

The installed `torch_npu.npu_ffn` documentation was also inspected. Its
SwiGLU/GEGLU/ReGLU restriction is FP16 in high-performance mode; the documented
BF16 support for other activations does not qualify this BF16 SwiGLU layer.
No dtype or activation change was made to obtain fusion.

Artifacts are `tp2-27b-mlp-sharedqkv-200-200-gu-nd` and
`tp2-27b-mlp-sharedqkv-200-200-gu-paired-pilot`, with JSON, logs, and two rank
traces in the same local and remote artifact directories. The exact source
snapshots are `gate_up_nd_benchmark.py` (SHA-256
`e6a7591ae6fe70e91da33777a372b02feb417a4a9cf6bf36791536c1c636253a`) and
`gate_up_paired_benchmark.py` (SHA-256
`7d39e37259f875323e32a6df1ed65735d8856e2709abd9ea47827d7b556b85ed`).
`gate_up_candidate_audit.json` checks their identities and distinguishes the
formal ND run from the short channel-partition pilot. Focused CPU validation
now passes 27 tests, including case-insensitive CLI controls. Both candidates remain opt-in, with NZ and one block as
the defaults.

### Why selected-layer gains may shrink in the complete model

The measured endpoint contains one full-attention layer and its real MLP,
plus a synthetic preceding producer's input reduction. Qwen3.5-27B has 16
full-attention and 48 GDN layers. Layer counts alone do not determine the
latency fraction attributable to each path. Unchanged GDN work can dilute the
benefit; extra graph joins, state gathering, or launch gaps can erase it.
Conversely, a real adjacent layer may supply another useful overlap window.
Neither direction is established by the current selected-layer result.

A whole-model comparison must not repeat the benchmark's synthetic input
reduction at every layer: the preceding layer's real Down is already reduced.
It must also account for the actual final output endpoint and all transitions
between attention types. A per-layer final join is a concrete potential cost;
carrying request partitions through compatible boundaries would avoid repeated
repacking, but remains unimplemented and requires state-aware validation.

The startup warning about missing Triton `causal_conv1d_update_npu` belongs
to the KDA compatibility rebind in `patch_triton.py`. It is not evidence that
Qwen3.5 GDN uses the PyTorch fallback. The inspected `gdn.py` imports its
convolution and recurrent update from `fla_npu.ops.ascendc`. The installed
`causal_conv1d_update` signature accepts speculative lengths, accepted-token
counts, null state IDs, output buffers, and optional host metadata. This
source inspection establishes the actual entry point, not successful GDN
capture or correct state transitions. Those still require the qualification
below.

The next integration gate is actual layer 3 -> layer 4 with the real GDN
implementation. First qualify its official eager execution, captured replay,
and state updates on identical inputs. Then compare a joined baseline and
request pipeline through both complete layers, including both MLPs and the
last reduction. Only after this gate passes should the experiment extend to
full-model replay. Runtime bucket routing and performance selection require
measuring this complete boundary and dispatch overhead; the existing CLI
prototype does not supply an engine-level planner.

## Alternative if full MLP fragmentation loses

A smaller follow-up can partition only post-norm, GateUp, and activation,
then assemble the full intermediate activation for one shared Down and one
full-bucket Down reduction. With shared QKV, its order would be
`input, output_A, output_B, down400`, with the same final hidden and residual
endpoint as the three schedules above.

This removes a fragmented Down and an extra reduction startup, but requires
joining intermediate activations, which can be wider than hidden states.
Their copies and the final shared reduction must remain timed. This alternative
is not implemented or assumed to win. Its value depends on actual MLP shape
efficiency, the real release window, and communication contention.

The [earlier FFN readiness experiment](multi_graph_ffn_readiness.md) rejected a
large kernel rewrite for its single-card Qwen3-4B, B98/default104 scenario.
Its estimated 8-10 us shared-Down benefit was a release model, not a measured
decoder gain. Neither that estimate nor the old FIA completion gap can be
transferred to this TP27B workload. A larger release gap also cannot yield
unbounded savings once the early partition's work is fully hidden.

## Future adjacent GDN boundary

The complete cross-layer pipeline is implemented as an experimental benchmark,
but has not passed its original elementwise two-layer numerical gate. The qualification probe
described below now exercises actual layer 4. Layer 4 is a real GDN layer, so repeating
the selected full-attention layer would not qualify the adjacent model
boundary. A later experiment must satisfy these additional requirements:

- Construct the actual loader with Q4 speculative metadata support and at
  least 100 request slots. The default weight-only loader has no speculative
  configuration; capacity selection alone does not set the loaded GDN's
  `num_spec`. Assert layer 4 is `linear_attention` with `num_spec=3`, while
  retaining explicit control over capture and state allocation. `--gdn-probe`
  now enables this loader configuration; other callers retain their defaults.
- Obtain state shapes, dtypes, and cache spec through official GDN methods.
  For the inspected 27B TP2 Q4 SD layout, expected per-block Conv shape is
  `[6, 5120]` and SSM shape is `[24, 128, 128]`; DS changes the Conv orientation.
  Use the reported dtype rather than assuming FP32 recurrent state. Verify the
  loader-created packed convolution weight.
- Reserve null block 0 and assign four unique physical state blocks per
  request, retaining global request identities in both lanes. One hundred
  request slots require 401 blocks including null. GDN state tables are
  separate from the FIA token-page table.
- Use an independent official Ascend GDN metadata builder for each graph or
  lane, the real layer prefix, and graph-stable buffers. Live Q4 intervals have
  length four; inactive intervals have length zero, null state IDs, and false
  masks. FIA's full-width dummy query intervals cannot be copied unchanged.
  Reject missing metadata, because the GDN profile path can otherwise skip
  recurrent work.
- Restore identical Conv and SSM state outside every independent timed replay.
  Check hidden, residual, updated live state, inactive state, and null-block
  invariants. Separately qualify a short sequence that intentionally continues
  state and accepted-token mappings across steps instead of resetting them.

Any adjacent-boundary comparison must include real layer 3 MLP, real layer 4
GDN and MLP, every reduction, residual output, and the final drain for padding
and split. Moving an exposed tail to a later unmeasured endpoint does not
demonstrate an improvement.

## GateUp and HCCL causal probe

`--contention-probe --include-mlp --shared-qkv --shared-rope --pair 200 200`
first qualifies the selected decoder, then isolates its real post-Norm inputs,
NZ GateUp weight, and a real BF16 `[200, 5120]` Down all-reduce payload. Seven
cases compare full/left/right GEMM alone and with the same collective, plus
the collective alone. Three rounds of 12 samples alternate case order; each
sample uses the slower rank before taking medians. Restoration and final
communication completion remain inside the external replay timer. `--trace`
records a separate two-replay-per-case device trace, outside latency samples.

The initial `tp2-27b-gateup-contention.json` is **invalid for internal timing**:
events recorded inside capture returned constant timestamps across replays,
including collective durations greater than complete replay duration. Those
internal timing fields must not be used for attribution. The corrected
`tp2-27b-gateup-contention-v2.json` removes captured timing events and uses
external complete-replay events, with profiler kernel durations as a separate
diagnostic. Its complete replay medians are:

| Case | Complete replay, us |
| --- | ---: |
| Full GateUp alone | 330.01 |
| Left GateUp alone | 229.56 |
| Right GateUp alone | 227.72 |
| Full GateUp with HCCL | 355.81 |
| Left GateUp with HCCL | 256.14 |
| Right GateUp with HCCL | 256.10 |
| HCCL alone | 179.12 |

Separate device-trace medians (two observations per entry, not formal kernel
latency estimates) distinguish shape overhead from overlap effects:

| GateUp case | Rank 0, us | Rank 1, us |
| --- | ---: | ---: |
| Full alone | 296.81 | 294.24 |
| Left alone | 199.08 | 190.85 |
| Right alone | 197.14 | 188.04 |
| Full with HCCL | 305.97 | 320.71 |
| Left with HCCL | 204.32 | 220.26 |
| Right with HCCL | 201.30 | 217.07 |

The two standalone half GEMMs already cost about 379-396 us, versus 294-297 us
for the full GEMM. Communication overlap adds a rank-dependent penalty; it
does not explain all fragmentation overhead. Do not add individual probe
replay times to predict decoder latency: they include separate graph launch,
restoration, and synchronization costs. These probes do not reproduce the
decoder's exact producer release times or all concurrent operators.

The actionable next target remains cross-layer tail coverage, with a strict
budget for added GEMM fragmentation. Simply adding streams cannot recover
the standalone shape cost. No new decoder speedup is claimed by this probe.

## Actual adjacent GDN qualification

`--gdn-probe --include-mlp --shared-qkv --shared-rope --pair 200 200` loads
ngram Q4 configuration and the actual adjacent `linear_attention` layer 4.
`multi_graph_tp_gdn.py` uses its official full decoder forward, including
GDN output projection, both Norms, MLP and their TP collectives. Inputs come
from the selected layer 3 ND reference; preceding model layers remain
synthetic. Official cache binding creates 401 state blocks per rank with
Conv `[401, 6, 5120]` and SSM `[401, 24, 128, 128]` shapes.

`tp2-27b-adjacent-gdn-qualify.json` passed whole-bucket eager versus capture
checks for B98 -> B97 -> B99 -> B98 on both ranks. Hidden, residual and both
state tensors matched exactly, and inactive/null states remained unchanged.
`tp2-27b-adjacent-gdn-split.json` also passed whole and 200+200 captured paths
against the full eager reference with changing input values: 32 field checks
per rank, using the unchanged `rtol=0.02, atol=0.02`. Split results are not
bitwise equal; the largest absolute field differences were 0.03125 and
0.0625 on ranks 0 and 1, respectively, within the combined relative/absolute
tolerance. This is a qualification result, not a performance measurement.

The probe additionally checks fixed metadata buffer pointers and independent
three-step eager/full/split state trajectories with accepted counts 1, 2, 1.
`tp2-27b-adjacent-gdn-continuity.json` passed these checks on both ranks,
recording 41 check groups per rank, including the dynamic-batch comparisons.
Its tested GDN source SHA256 is
`f0a666309a8772e643c23877c6c308a9ef7a47e4d86d535caa3f0aebf7974969`.
Both state tensors demonstrably changed from their initial values, excluding
a metadata-free no-op. All state
restores used for comparison are outside any performance scope; no latency
is reported by this probe. The layer-3-to-layer-4 overlapping schedule is now
implemented separately with `--gdn-pipeline`; qualification and outcomes are
recorded below. Full-model integration remains separate work.

Raw JSON, logs and traces are retained under
`outputs/multi_graph_tp_attention_mlp/` locally and
`/workspace/multi-graph-results/tp-boundary/` remotely. Reported source hashes
identify the exact tested probe revisions; later qualification additions do
not retroactively change earlier results.

Focused CPU validation passed 36 tests (391 deselected). Scoped Ruff checks
passed. `bash format.sh ci` was attempted: repository Ruff and Markdown
checks passed, while unrelated existing codespell/forbidden-import findings
and Windows shellcheck, `/bin/bash`, and python3 launcher gaps prevent a
clean full CI result. See `contention-gdn-format-ci.log` with the raw artifacts.

## Complete two-layer scheduling experiment

`--gdn-pipeline --include-mlp --shared-qkv --shared-rope --pair 200 200`
extends the endpoint through the actual layer 4 GDN and its complete MLP.
Both hidden and residual outputs, Conv state and SSM state are checked against
the unchanged layer 3 ND oracle followed by official layer 4 execution.
Every timed sample restores the same states outside timing. FIA update remains
after graph replay submission, using the existing external-event protocol.

The joined control uses both whole layers. `split_gdn_barrier` and
`split_mlp_overlap` use the same request partitions and GDN decomposition:
the latter begins GDN A's local Norm/projection/recurrence/output-projection
after layer 3 Down_A, while Down_B can still be running. The barrier control
waits for Down_B first. Both wait for Down_B before submitting GDN collectives.
GDN output reductions A/B and MLP Down reductions A/B use one ordered
communication stream; all final waits and hidden/residual joins are timed.
The official GDN output projection's reduction is disabled only while building
the local prefix and restored in `finally`; the benchmark explicitly submits
exactly one reduction. No communication is counted twice or omitted.

The first qualification outcomes reject several candidates before timing:

| Artifact stem | Candidate | Outcome |
| --- | --- | --- |
| `tp2-27b-two-layer-pilot` | NZ GateUp, split GDN barrier control | Initial residual: 1/2,007,040 elements fails, max failing difference 0.0625 |
| `tp2-27b-two-layer-shared-gdn` | Split layer 3, complete layer 4 | Same initial residual failure at the same location |
| `tp2-27b-two-layer-nd-gateup` | ND GateUp, split GDN barrier control | Same initial residual failure; exact candidate layer 3 input passes local GDN comparison |
| `tp2-27b-two-layer-shared-mlp` | Shared layer 3 MLP and complete GDN | Initial batch passes; B97 residual has 1/1,986,560 failing elements, difference 0.0234375 |

The initial failures are present without cross-layer overlap, so they do not
establish an overlap synchronization bug. Keeping GDN whole and changing
GateUp format do not remove them. Keeping the first MLP whole removes the
initial failure but not the dynamic-input failure. Individual-layer tolerance
passes cannot be extrapolated to their composition. These candidates receive
no speedup claim, and the original tolerances remain unchanged.

Additional controls are explicit: `--gdn-shared` keeps GDN whole;
`--gdn-shared-mlp` also keeps the first MLP whole; `--gdn-shared-dense` retains
all dense GEMM row shapes and runs only the two FIA request partitions on
separate streams before the whole O projection. All use the same default
400-token capacity and legal 200+200 pair, without custom capture sizes.

The first shared-dense experiment exposed a benchmark setup error on B99:
runtime GDN metadata had captured a Python slice extent of 392 from initial
B98, truncating new requests. Even the joined baseline failed. The runtime
now fixes captured Python extents at full lane capacity and uses zero device
query intervals/null state IDs for inactive requests, matching the standalone
GDN qualification. The first `tp2-27b-two-layer-shared-dense` run is invalid
for dynamic replay and performance; subsequent results must use the fixed
runtime. This fix does not change the precision threshold.

The corrected `tp2-27b-two-layer-shared-dense-v2.json` passed 88 recorded
checks per rank, including B98 -> B97 -> B99 -> B98 outputs and full GDN
states. Three rounds of eight samples alternate mode order and use the
slower rank per sample. All FIA updates and both layers' reductions/MLPs
remain inside the timing endpoint:

| Two-layer candidate | Wall replay, us | Device replay, us |
| --- | ---: | ---: |
| Joined padding | 4807.17 | 4743.53 |
| Parallel FIA with shared dense shapes | 4853.91 | 4791.72 |

The qualified candidate is **0.97% slower in wall time** and **1.02% slower
in device time**. All three round medians have the same ordering. This is a
negative two-layer result, not an end-to-end gain. Its tested attention
benchmark SHA256 is
`d4de3404466692ea5fd77360f8f2f7f1953bb4e507309d3c46f06af9401ea87b`;
the exact attention and GDN sources are saved with the raw artifacts as
`two_layer_attention_tested.py` and `two_layer_gdn_tested.py`.

Under the original strict comparison policy, selected single-layer split-MLP gains did not extend to a
qualified two-layer speedup. The more aggressive cross-layer schedule is
implemented but blocked by its non-overlapping control's numerical failure;
the shape-preserving control qualifies but loses performance. No full-model
or serving speedup has been demonstrated. Future work must jointly preserve
numerical behavior across layer boundaries and reduce the extra split work,
rather than treating independently qualified layers as automatically
composable. None of these experimental switches enables engine routing.

Delivery validation: 45 focused CPU tests passed (391 deselected), including
collective ordering, final drain, CLI constraints and the growing-batch capture
extent regression. Scoped Ruff, codespell and Markdown checks passed. Full
`bash format.sh ci` was attempted; existing unrelated findings and Windows
tool availability still prevent a clean repository-wide result. The log is
`two-layer-format-ci.log` with the local experiment artifacts.

## Separate numerical assessment from execution correctness

`--numerical-assessment` is an explicit opt-in for the complete split
`--gdn-pipeline`. It retains the original 0.02 relative/absolute comparison
as a diagnostic, recording every failing element count, failing fraction,
maximum absolute error, RMSE, reference RMS and normalized RMSE. It does not
silently raise the tolerance or claim model-quality equivalence. Nonfinite
values and mismatched shapes remain errors. The ordinary benchmark retains
its original strict behavior.

The assessment requires additional execution checks before timing: identical
schedule eager and captured replay must match bitwise in live hidden/residual
outputs and complete Conv/SSM state, from identical inputs and initial states.
Null and inactive request states must remain exactly unchanged. The existing
graph/workspace/metadata identity checks remain in force, as do real FIA
updates and all communication dependencies.

Eight additional steps maintain independent joined, barrier-split,
overlapped-split and eager-split GDN state trajectories. Batch repeats
98, 97, 99, 98 with new input seeds and accepted counts 1, 2, 1, 3. The split
barrier, overlapped graph and identical eager schedule must remain bitwise
equal, while errors against the joined trajectory are recorded. Previously
active requests becoming inactive must retain their prior state rather than
be compared with initial state. FIA prefix caches are restored between these
synthetic steps: this is GDN state-continuity qualification, not an actual
full-model generation rollout or a logit-quality test.

Reports use `completed_with_numerical_diagnostics`, explicitly distinct from
the strict benchmark's `passed`. This permits studying performance while
preserving the evidence needed to decide whether numerical differences are
acceptable. It does not certify absence of arbitrary memory errors outside
the checked state tensors or replace model-level quality evaluation.

The first assessment (`tp2-27b-two-layer-assessment.json`, before adding the
eight-step trajectory test) completed all four input variants. Each rank
passed 12 identical-schedule eager/replay checks (three schedules times four
variants), including exact whole-state comparisons and inactive/null checks.
The split candidates had only 1-3 original-tolerance failures per roughly
two-million-element residual. Hidden, Conv and SSM had no original-tolerance
failures. Maximum normalized RMSE across both ranks and reset-state variants:

| Field | Normalized RMSE |
| --- | ---: |
| Hidden | 0.4797% |
| Residual | 0.2000% |
| Conv | 0.08284% |
| SSM | 0.06630% |

The complete two-layer wall replay medians were 4827.38 us for padding,
5248.51 us for the barrier split, and 5119.64 us for the overlapped split.
Overlap saved 2.46% against the identical split control but remained 6.05%
slower than padding. Thus permitting the sparse numerical differences does
not by itself produce a performance win. This preliminary result is followed
by the continued-state assessment, rather than being treated as proof of
arbitrary-length model quality.

`tp2-27b-two-layer-assessment-continuity.json` then completed all eight
continued-state steps and recorded 175 checks per rank. Barrier split,
overlapped split and eager split remained bitwise equal in live outputs and
complete state at every step. All inactive/null state invariants passed.
Normalized RMSE ranges against the independent joined trajectory, across
both ranks and all eight steps, were:

| Field | Minimum | Maximum | Maximum original-tolerance failing elements per comparison |
| --- | ---: | ---: | ---: |
| Hidden | 0.4466% | 0.4830% | 0 |
| Residual | 0.1773% | 0.1873% | 2 |
| Conv | 0.06955% | 0.08495% | 6 |
| SSM | 0.04591% | 0.08788% | 0 |

Hidden/residual errors did not grow steadily over this short sequence. SSM
error increased modestly while remaining below 0.09%; eight synthetic steps
do not establish an arbitrary-length bound. The evidence supports continuing
performance work with these differences treated as numerical diagnostics,
not as demonstrated execution bugs. Full-model logits and generation quality
have not been evaluated by this benchmark.

The continued-state run's three-by-eight timing reproduced the earlier
ordering: padding 4816.83 us, barrier split 5218.44 us, overlapped split
5109.06 us. Overlap saved 109.39 us (2.10%) against barrier split, but remained
292.23 us (6.07%) slower than padding. These are complete selected-two-layer
replay measurements, not serving throughput. The attention benchmark SHA256
was `7526199965dd5f0be3c1fb88415ac3c58674c8553532ed40662c5a791ec2cde5`;
the GDN helper SHA256 was
`15c334c4c12cd4d8d8441e92192acc3c163a07a8a5383601cae60d47cd2a5b6f`.

The assessment also supports `--gdn-shared`. That candidate retains a complete
GDN layer after the split first layer; its continued-state control compares
captured split with the identical eager schedule, without inventing a
cross-layer barrier/overlap comparison that it does not execute.

`tp2-27b-two-layer-assessment-shared-gdn.json` completed 139 checks per rank,
including four reset-state variants and eight independently continued-state
steps with exact eager/replay and inactive/null-state checks. Its numerical
error envelope was similar: continued-step normalized RMSE maxima were
hidden 0.4829%, residual 0.1872%, Conv 0.08494%, and SSM 0.08788%.

| Same-run shared-GDN comparison | Wall replay, us | Device replay, us |
| --- | ---: | ---: |
| Padding | 4802.04 | 4734.61 |
| Split layer 3, complete GDN layer 4 | 4779.04 | 4713.43 |

The measured wall improvement is 23.00 us (0.479%); device improvement is
0.447%. All three round medians favor this candidate, but this small result
is from one qualified run and is not a demonstrated full-model speedup or a
large robust gain. It favors preserving whole GDN computation over splitting
both layers in the current workload.

The resulting policy is to continue performance experiments with these small
cross-shape differences recorded, while retaining strict same-schedule,
state-integrity and finite-value gates. No execution correctness issue was
observed in the assessed configurations. Longer model-level quality checks
remain distinct from this decision; a handful of original-tolerance failures
alone no longer blocks the explicitly requested numerical-assessment mode.

## Subsequent branch-specific scheduling experiments

[Causal parallel-direction experiments](multi_graph_parallel_directions.md)
record the subsequent whole-GateUp/branch-Down, asymmetric branch order,
early-Norm, collective-priority and real cross-layer resource-pairing tests.
The Down pipeline improves its isolated boundary by 8.20% device, but its
complete two-layer implementation only saves 0.53% wall against padding and
does not improve on the prior split candidate. New GDN compute streams and
projection-gated Cube/vector pairing lose against the original split schedule.
Profiler evidence identifies activation and projection interference. These
findings qualify optimization hypotheses; they do not establish a large
split-over-padding or full-model gain.

Validation for this change: 47 focused CPU tests passed (391 deselected),
including metrics, nonfinite rejection and unchanged capture extents. Scoped
Ruff/codespell/Markdown checks passed. Full CI was attempted with the same
unrelated repository findings and Windows tool gaps; see
`numerical-assessment-format-ci.log`. Raw reports and exact tested sources are
saved with the preceding experiment artifacts.
