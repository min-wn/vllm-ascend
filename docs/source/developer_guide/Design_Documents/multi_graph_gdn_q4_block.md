# Q4 GDN block products and causal solve

## Decision and current scope

This route changes how the four-token recurrence is computed. It groups the
initial state's eight key/query projections into one skinny batched GEMM and
uses a four-by-four causal system to recover the four updates. The proposed
benefit is moving repeated state-vector reductions from AIV to Cube, while
retaining all four dense BF16 speculative candidates and the next-step BF16
load boundary. This is an operator change available to padding as well; it is
not evidence of a request-split advantage by itself.

The V3 product probe qualifies BF16-to-FP32 `bmm.dtype` and `bmm.dtype_out`
execution and replay on the current A2 machine. The eight-column product
endpoint, including selected-state gather and Q/K pack, takes approximately
192-199 us. The V3 complete recurrence then passes dynamic correctness, but
only reduces the median of round medians from 1000.06 to 938.82 us. Its
61.24 us saving misses the preregistered 135 us gate for layer integration.
The next candidate removes the intermediate gathered state and globally
packed RHS through direct selected-state Cube products. No full replay
improvement has been demonstrated. The preceding compact-state representation is stopped
as a latency route because its fused accepted-prefix recovery was slower than
the installed recurrence. See `multi_graph_gdn_compact_state.md`.

## Mechanism

For token `i`, the installed recurrence decays the FP32 matrix, computes a
key projection, applies a rank-one update, writes a BF16 candidate and computes
a query projection. State remains FP32 between the four tokens. Let `H0` be
the BF16 matrix selected from the previous step's accepted prefix, converted
to FP32 for accumulation. Define:

```text
alpha_i = product(gamma_t, t = 0..i)
d_i,j   = product(gamma_t, t = j+1..i), with d_i,i = 1
P       = H0 @ [k_0 k_1 k_2 k_3 q_0 q_1 q_2 q_3]
L_i,i   = 1
L_i,j   = beta_i * d_i,j * dot(k_j, k_i), j < i
L_i,j   = 0, j > i
target_i = beta_i * (v_i - alpha_i * P_key_i)
L @ U = target
H_i = alpha_i * H0 + sum(d_i,j * outer(u_j, k_j), j = 0..i)
o_i = scale * (alpha_i * P_query_i
              + sum(d_i,j * u_j * dot(k_j, q_i), j = 0..i))
```

Forward substitution requires only six prior-update terms. No expression
divides by a cumulative decay, so zero and underflowed gamma are defined.
BF16 query and key operands remain raw during the initial GEMM; applying
query scaling before casting to BF16 would introduce an additional rounding
that the installed kernel does not have. Scale is applied to FP32 products or
their FP32 consumers.

This is an identity in real arithmetic. Cube reductions, cumulative decays,
the causal solve and low-rank output evaluation change FP32 operation order.
Strict comparison is required between eager and replay for the same new
schedule. Drift from the installed recurrence must be assessed separately,
including continued accepted-state feedback and all four candidate matrices.
The CPU implementation is `benchmarks/multi_graph_gdn_q4_block_reference.py`.

## Installed API evidence

The target installation reports PyTorch `2.10.0+cpu`, torch_npu
`2.10.0.post4`, with torch_npu git revision
`5dd8ef3f9b375b5ae4a83538d5785754148c3302`. Read-only import and dispatcher
inspection, without initializing an NPU, report these actual schemas:

```text
aten::bmm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor
aten::bmm.dtype_out(Tensor self, Tensor mat2, ScalarType out_dtype,
                    *, Tensor(a!) out) -> Tensor(a!)
npu::npu_bmmV2(Tensor input, Tensor mat2, int[] output_sizes) -> Tensor
```

Both `bmm.dtype` and `bmm.dtype_out` have a registered `PrivateUse1` kernel.
This establishes an entry point, not successful BF16-to-FP32 execution on A2.
The probe must execute it and verify that the result is FP32. It never
substitutes a BF16 result followed by a cast.

For source review, official op-plugin revision
`5ac12689b6a61ec9db3fc260eab17bbc5447567e` has
[`BmmKernelNpuOpApi.cpp`](https://github.com/Ascend/op-plugin/blob/5ac12689b6a61ec9db3fc260eab17bbc5447567e/op_plugin/ops/opapi/BmmKernelNpuOpApi.cpp).
Its overload with an output dtype forwards that output tensor to
`aclnnBatchMatMul`. This source revision is a review pin, not an assertion
that it is the exact submodule used to build the installed wheel; the live
dispatcher and device qualification are authoritative for this machine.

`npu_bmmV2` has no output-dtype argument. Installed grouped-matmul documentation
limits its single-input/single-weight grouping mode to 1024 groups and lists
matching input/output types for the non-quantized BF16 case. Splitting 2352
independent head groups over multiple calls does not resolve the FP32-output
contract. The first qualification therefore uses native `bmm.dtype_out` and
`bmm.dtype`. An explicit FP32-input fallback is available only when requested;
both input conversions are included in its measured cost.

## Product probe contract

`benchmarks/multi_graph_gdn_q4_products_probe.py` uses the actual local layout:

| Item | Layout |
| --- | --- |
| Requests / value heads / key heads | 98 / 24 / 8 |
| K and V width | 128 / 128 |
| Independent matrix groups | 2352 |
| Dense BF16 candidate cache | `[1+4B, 24, 128, 128]` |
| Query/key input | raw BF16 `[B, 4, 8, 128]` |
| Product output | contiguous FP32 `[B, 24, 128, 8]` |
| Alignment diagnostic | same output with 16 columns; last eight are zero |

The selected cache slot is `1 + 4 * request_id + accepted - 1`.
Accepted counts vary per request from one through four. An inactive row uses
null slot zero. Random request reordering and live counts 98/96/97/98 change
the values at fixed graph input pointers. The cache, including all unselected,
inactive and null slots, must remain unchanged.

The primary product endpoint includes device index arithmetic, gathering the
selected dense BF16 matrix, packing and replicating Q/K for grouped heads,
optional column padding, and the FP32 batched GEMM. A separate
`product_only_diagnostic` starts from already contiguous state and RHS buffers;
it is a diagnostic lower bound. Its results must never substitute for the
gather/pack/product endpoint. The 98-group experiment is operator qualification,
not a proposal to add a bespoke ACLGraph capture size.

Each supported candidate checks CPU FP32 product error, exact same-schedule
eager/replay, changing inputs, null/inactive cache preservation and finite
outputs. The FP32 CPU product tolerances are `rtol=1e-4`, `atol=2e-6`.
Three alternating timing rounds report raw event/wall samples and event
mean/median/P95/P99. Graph and preparation buffers are retained during the
measurement; reported peak allocation is the process-wide probe allocation,
not deployment overhead for one implementation.

The root benchmark extends this to a full recurrence, with solve/state
materialization and all state updates in the timed region. Neither this
product probe nor a recurrence-only result includes the model's host FIA
updates, attention, TP communication or hidden/residual endpoint.

```bash
source /etc/profile.d/ascend-multigraph.sh
cd /vllm-workspace/vllm-ascend
timeout 180s python benchmarks/multi_graph_gdn_q4_products_probe.py \
  --api dtype_out dtype --columns 8 16 --repeats 15 --rounds 3 \
  --output /workspace/multi-graph-results/tp-boundary/gdn-q4-products-NEW.json
```

Use a new output name for each run. Schema availability or a successful
capture does not count as execution qualification. Unsupported API failures
are retained explicitly; numerical failures must be investigated before
using any performance measurement.

## V3 product qualification on the target machine

The completed device report is retained locally under
`outputs/multi_graph_tp_attention_mlp/gdn-q4-products-v3.json` and remotely
under `/workspace/multi-graph-results/tp-boundary/gdn-q4-products-v3.json`.
Its SHA256 is
`de4f306d42fdf5410ded6222e73292fe0e12ef1cc9f1a692c277e93e458be38e`.
The executed probe source SHA256 recorded in that report is
`5e87a51cb41d5e068495dd642e96486a2f35d0405548e6981a6eaefdee861161`.
The effective configuration is `ALLOW_INTERNAL_FORMAT=disable` and matmul
HF32 disabled. All reported products are genuinely FP32.

There are 32 completed dynamic correctness checks: four input/membership
transitions for each of four API/width configurations and two timing scopes.
All eager/replay comparisons are exact. The largest observed CPU-FP32 product
error is `3.7253e-8` absolute and `7.7726e-8` relative L2. The complete dense
cache, including null/inactive and unselected candidates, is unchanged.
This is a read-only product qualification, not continued recurrence-state
validation. There are 216 timing samples: nine samples in each of three
alternating rounds for each configuration/scope. Timed samples replay the
final fixed B98 input variant after the separate dynamic correctness checks;
they are not the goal's 128-step random dynamic-trajectory benchmark.

Device event times in microseconds:

| API / columns | Scope | Mean | Median | P95 | P99 |
| --- | --- | ---: | ---: | ---: | ---: |
| dtype_out / 8 | gather + pack + product | 200.28 | 199.00 | 208.86 | 211.12 |
| dtype / 8 | gather + pack + product | 192.39 | 192.24 | 195.56 | 196.02 |
| dtype_out / 16 | gather + pack + product | 231.32 | 230.14 | 237.26 | 241.78 |
| dtype / 16 | gather + pack + product | 261.03 | 261.02 | 270.92 | 273.82 |
| dtype_out / 8 | product only, diagnostic | 88.98 | 88.60 | 91.20 | 99.60 |
| dtype / 8 | product only, diagnostic | 86.89 | 85.76 | 96.82 | 98.82 |
| dtype_out / 16 | product only, diagnostic | 69.87 | 69.28 | 77.48 | 79.22 |
| dtype / 16 | product only, diagnostic | 66.73 | 64.96 | 77.40 | 78.46 |

Independent round medians, in the same units:

| API / columns | Scope | Round 0 | Round 1 | Round 2 |
| --- | --- | ---: | ---: | ---: |
| dtype_out / 8 | gather + pack + product | 199.36 | 198.92 | 199.00 |
| dtype / 8 | gather + pack + product | 193.02 | 192.10 | 192.24 |
| dtype_out / 16 | gather + pack + product | 229.96 | 230.60 | 232.84 |
| dtype / 16 | gather + pack + product | 262.52 | 260.90 | 256.76 |
| dtype_out / 8 | product only, diagnostic | 87.70 | 88.54 | 89.68 |
| dtype / 8 | product only, diagnostic | 86.02 | 84.16 | 85.80 |
| dtype_out / 16 | product only, diagnostic | 69.64 | 69.12 | 69.12 |
| dtype / 16 | product only, diagnostic | 64.96 | 65.48 | 64.78 |

The aligned 16-column GEMM is faster in isolation, but its complete product
endpoint is slower. In the eight-column `dtype` case, the full endpoint is
about 106 us longer than the contiguous-buffer diagnostic. This difference
includes preparation and possible effects of the different execution paths;
it is not an isolated measurement of gather or packing alone. These results
support carrying the eight-column endpoint into the full recurrence test,
with all preparation retained in the timer. They do not support selecting the
16-column variant based on isolated GEMM time.

The probe's peak allocation is 750,528,512 bytes and peak reservation is
1,711,276,032 bytes. Both include the retained graphs/buffers for all eight
configurations/scopes and the dense cache. They cannot be reported as the
incremental model memory of a single chosen implementation.

V1 and V2 did not execute a product. They stopped during configuration/report
preparation: V1 used the nonexistent `torch.backends.npu` namespace; V2 read
`torch.npu.config.allow_internal_format`, whose installed class has only a
setter. V3 uses the verified `torch.npu.matmul.allow_hf32` getter and reads
the internal-format option through
`torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")`. A CPU-only report
construction confirmed these fields serialize successfully while NPU
initialization remains false. These earlier failures are preflight errors,
not unsupported BMM or correctness results.

## V3 complete recurrence qualification

The full recurrence harness is
`benchmarks/multi_graph_gdn_q4_block_probe.py`, with the original Apache-2.0
device implementation under `benchmarks/multi_graph_gdn_q4_block_kernel/`.
It includes selected-state gather, raw Q/K packing, FP32 products, Gram
products, the causal solve, all four BF16 candidate writes and BF16 attention
output. It uses capacity 100, live counts 98/97/99/98 repeated twice, random
stable global request IDs and per-request accepted counts one through four.
This capacity is the operator probe's layout; model capture sizes remain
outside this experiment.

The state path is explicit. Products select the previous accepted BF16 cache
slot. The new kernel rereads that same selected state in V=64 tiles, keeps
the working tile FP32, applies four ordered gamma/FMA updates using the
solved vectors, and writes every candidate with BF16 rounding. Those writes
do not feed back into subsequent tokens inside Q4. The next replay selects
and loads a BF16 candidate, preserving the required step boundary. Distinct
requests own disjoint destination slots, while a request's selected source
may alias one of its own destinations; the complete selected tile is resident
before any of its candidate writes. Null and inactive slots remain unchanged.

Raw results are retained locally in
`outputs/multi_graph_tp_attention_mlp/gdn-q4-block-v3.json` and `.log`, with
the same names remotely under `/workspace/multi-graph-results/tp-boundary/`.
Pins:

| Artifact | SHA256 |
| --- | --- |
| Raw V3 JSON | `acd2c139908d63d7711e9a946e266942d3d38e7ecc20c127717fb7097d16059b` |
| V3 log | `399b97b160af3a189d79225f6824e017994dde57eebd24836e1f2285e15c9efa` |
| Executed harness source | `eb6215d525d8beec043b72b54608995d18b24861c6296880a6f813c901da0b9e` |
| Loaded extension | `65f599057b8e52413de276b183792e0ccbd662845ec5aa713b182a2c456ccad0` |
| V3 kernel source | `9deadff7c9b9ebc90627a923966a512a56fee134a3236cd6ab17de382b1f69c2` |

The eight dynamic steps produce 16 strict implementation/step checks. The
run uses `--recheck-eager`: each schedule also repeats eager from the same
saved state, checking products and complete state/output tensors before
testing replay. All eager repeats and same-schedule eager/replay comparisons
are exact. The eight steps continue their own official/block cache histories,
including inactive request preservation and resume. All finite checks pass.

Cross-algorithm differences are recorded separately. Worst values across
the eight dynamic steps:

| Metric | Largest observed value |
| --- | ---: |
| Attention output absolute error | 0.000015259 |
| Attention output global NRMSE | 0.000054784 |
| Attention output worst-request NRMSE | 0.000238644 |
| All candidate states absolute error | 0.000244141 |
| All candidate states global NRMSE | 0.000030968 |
| Prefix 1 worst-request NRMSE | 0.000075747 |
| Prefix 2 worst-request NRMSE | 0.000062311 |
| Prefix 3 worst-request NRMSE | 0.000075638 |
| Prefix 4 worst-request NRMSE | 0.000063187 |

Every normalized metric is below the preregistered 0.005 diagnostic threshold.
The worst-request state checks assess active stable IDs separately for each
prefix, avoiding dilution by inactive slots or the other three prefixes.
This supports this recurrence experiment; it is not a model-quality or
long-horizon inference certification.

There are 36 samples per implementation, 12 in each of three alternating
rounds. Timed samples use a frozen B98 variant, identical initial cache
contents and equal warmup/replay counts for the two implementations. They
are not the fixed random dynamic full-model trajectory. Times are in us:

| Implementation | Scope | Mean | P50 | P95 | P99 |
| --- | --- | ---: | ---: | ---: | ---: |
| Installed recurrence | device event | 1004.04 | 1000.06 | 1011.48 | 1034.40 |
| Q4 block recurrence | device event | 942.85 | 939.53 | 969.30 | 970.32 |
| Installed recurrence | host dispatch + wait | 1127.67 | 1120.27 | 1163.59 | 1254.82 |
| Q4 block recurrence | host dispatch + wait | 1114.88 | 1054.52 | 1209.87 | 2938.18 |

| Implementation | Device round 0 median | Round 1 | Round 2 | Median of round medians |
| --- | ---: | ---: | ---: | ---: |
| Installed recurrence | 1007.60 | 1000.06 | 999.95 | 1000.06 |
| Q4 block recurrence | 945.44 | 938.74 | 938.82 | 938.82 |

The mean device saving is 61.19 us, and the median-of-rounds saving is
61.24 us, about 6.12% of this recurrence. Host wall tails include a block
outlier and are not improved consistently; the block wall P99 is 2938.18 us.
The device saving would contribute only about 2.26% of the earlier 2709 us
GDN endpoint if every other cost stayed unchanged. That is a budget estimate,
not a measured layer speedup, and falls short of the 135 us integration gate.

The process peak allocation is 2,750,226,432 bytes and peak reservation is
5,492,441,088 bytes, including both graphs and correctness snapshots. The
candidate's explicit product buffer is 9,830,400 bytes, its gathered base
is 78,643,200 bytes and its packed RHS is 4,915,200 bytes at capacity 100.
These three buffers total 93,388,800 bytes; that is a named intermediate
buffer budget, not the full-model incremental peak allocation.

### V2 race and the V3 causal fix

V2's failure was an actual synchronization bug, not BF16 reassociation. In
`gdn-q4-block-v2-debug-out2.json`, repeated eager executions from identical
inputs/state differ in 363,469 of 157,679,616 state elements, with maximum
absolute error 0.98046875. Mismatches are confined to value rows 64 through
76, within the second V=64 tile. The report SHA256 is
`2fd8fea3155fe8bf556a6106ebde0da2649b01045d8db1d41c73b6baa5490a06`.
No V2 timing or numerical comparison is accepted as a performance result.

The candidate output DMA reads `bf16Buffer_` through MTE3. A following task
reuses the same UB buffer for Q/K/V/beta loads through MTE2. A `MTE3_V`
dependency only orders vector consumption; it does not prevent that MTE2
overwrite while the previous copy-out is outstanding. The next input load
covers the beginning of the shared buffer, consistent with the observed
first 13 rows of the second tile being corrupted.

The V3 kernel changes only that dependency, adding
`Sync<HardEvent::MTE3_MTE2>()` at the start of `LoadInputs`. The saved V2
kernel and V3 source differ by this event and its explanatory comment.
With the same arithmetic retained, V3 passes repeated eager and replay
checks across all eight continued-state steps. This is direct evidence for
the buffer-lifetime bug and its fix. It is categorically separate from the
small, deterministic cross-algorithm differences recorded above.

## Next structural target: direct selected-state Cube products

An additional registered-BMM control folds the three consecutive value-head
matrices that share one key/query head. Their selected H0 matrices form a
contiguous `[384,128]` matrix and use one `[128,8]` RHS, rather than three
separate `[128,128]` products with replicated RHS. The FP32 result reshapes
directly to the existing `[3,128,8]` value-head layout. This preserves the TP
partition and all request/candidate state identities.

`gdn-q4-block-v3-gqa-fold.json` passes the same 16 strict graph checks and
eight continued steps, including mixed accepted counts, random stable IDs,
inactive/null slots and per-prefix drift checks. It uses the unchanged V3
solve/materialization binary. Its three official device round medians are
1011.18/1011.42/1011.35 us; block medians are 924.35/926.18/925.12 us.
The paired median-of-rounds saving is 86.23 us, still below the 135 us gate.
Clock differences between this run and the preceding run prevent assigning
their raw block-time difference entirely to folding. Direct Cube products
will use the same folded shape to avoid three separate Matmul API calls.

The initial un-folded recurrence needs another 73.76 us removed to pass the
135 us gate; the folded control still needs 48.77 us. The product diagnostic shows
roughly 106-110 us between prepacked products and the real gather/pack/product
endpoint. Eliminating globally materialized selected-state and GQA RHS
buffers therefore identifies a concrete candidate budget. It does not prove
that all of this gap can be removed.

The next product kernel should read the selected BF16 cache slot directly,
load raw Q/K for its grouped head and form eight RHS columns locally for Cube.
It must retain FP32 products, random stable request IDs, mixed accepted
prefixes, inactive/null behavior and exact same-schedule eager/replay.
The solve/materialization endpoint must include the new product kernel and
all required synchronization. Removing the gathered base also avoids its
global write/read round trip; four dense candidate writes remain required.

If direct products approach the measured prepacked 86-89 us while keeping
the remaining recurrence cost unchanged, the budget could pass 135 us. This
is a testable upper-bound argument, not an expected or achieved speedup.
It must be rejected or redesigned if Cube tiling/packing/synchronization
cost consumes that budget. It remains a general operator optimization and
must also be applied to the padding control before any split advantage is
claimed. The framework integration is deferred pending this causal result.

## Cost budget and stop gates

### Direct products and page-strided qualification

The first direct Cube implementation reads the accepted BF16 state slot in
place. It forms raw K/Q in L1 NZ using two strided copies and folds the three
value heads per key head into one M384/N8/K128 Matmul with FP32 output. The
fixed CANN host tiling is generated at build time. The runtime kernel reads
current state IDs, counts and accepted values from device metadata; it does
not call host shape update or depend on conditional graph support.

`gdn-q4-direct-v5.json` qualifies this implementation against registered
FP32-output BMM and verifies read-only cache access. It then passes all
continued-state, repeated-eager and graph checks using the existing V3
solver/materializer. Median-of-round medians is 1010.79 us for official and
777.64 us for block: 233.15 us saved. Block round medians are
781.25/777.64/775.60 us. This exceeds the 135 us gate for a real-layer
experiment, but is still a single-device recurrence result.

The solve/materializer's V4 control adds an independent two-deep VECOUT queue
for candidate BF16 writes. It retains FP32 operation order and all four state
snapshots, and no longer waits for each individual DMA before computing the
next token. `gdn-q4-direct-queue-v4.json` passes the same checks. Paired
official round medians are 1012.45/1000.72/1000.35 us and block medians are
706.89/691.38/690.65 us, giving median-of-rounds 1000.72 versus 691.38 us.
These are endpoint measurements, not sums of isolated savings.

Real GDN `bind_kv_cache` views share one page with conv state and can have a
gap between SSM slots. Both native kernels now accept a static slot stride,
while requiring contiguous inner head/value/key dimensions. There is no
layout conversion or state copy. `gdn-q4-strided-v5.json` uses stride
`[425984,16384,128,1]` and 32768 BF16 canary elements per slot. All 17 checks
pass, including gap canaries, repeated eager, graph replay, random request
identity, mixed accepted values and continued states. Paired round medians
are official 1012.82/1013.11/1013.19 us and block 701.63/702.85/704.81 us.

The direct algorithm retains a fixed FP32 products buffer of 9,830,400 bytes
at capacity 100. It removes the named gathered-state and packed-RHS global
buffers. Reported allocator peaks still include both controls, graphs and
qualification snapshots; they are not a measured full-model memory delta.

An explicit benchmark adapter replaces only the real GDN recurrence call.
Convolution, gating, Q/K normalization, output normalization and cache
ownership remain with the official layer. It validates supported Q4 metadata
and refuses incompatible speculative calls rather than silently falling back.
Each independent metadata lane receives its own persistent products/output
buffers. Outputs are cleared inside the timed graph so inactive values remain
defined. Padding and split use the same backend; this generic operator gain
does not by itself establish an advantage of splitting. TP2 layer and full
replay results are still required.

The first real-layer attempt was deliberately rejected by the adapter: the
loaded model uses FP32 SSM state, shape `[401,24,128,128]`, stride
`[458752,16384,128,1]`. The preceding operator qualifications use BF16 state.
Their savings therefore cannot be transferred to the actual default model
without qualification at its FP32 state precision. The implementation must
preserve the model's FP32 state, rather than silently convert it to BF16 to
reuse a successful operator result. `tp2-q4-block-layout-v2.log` retains this
diagnostic. This attempt produced no layer performance measurement.

### Actual FP32 state qualification

Both native kernels now retain BF16 or FP32 state without a full-cache cast.
The direct Cube kernel uses a separate offline FP32 tiling with
`baseM=128, baseN=16, baseK=64`, double-buffered L0A/L0B, and explicit
`Matmul::SetHF32(false, 0)`. FP32 NZ uses eight elements per 32-byte C0 block.
The adapter converts only raw BF16 Q/K to FP32 inside the executed callable;
the solve/materializer continues to consume the original BF16 Q/K/V/beta.
State updates and all four candidate snapshots retain FP32 precision.

`gdn-q4-fp32-v1.json` tests the actual state stride
`[458752,16384,128,1]`, including 65536 FP32 canary elements between slots.
All 17 checks pass: direct products against registered FP32 BMM, cache
read-only access, eight continued steps, repeated eager and graph equality,
stable random request IDs, mixed accepted prefixes, inactive/null isolation,
all four candidates, finite results and page-gap preservation. The largest
reported normalized error is 0.0001718865 (0.01719%), below the preregistered
0.5% screening gate. This does not certify model quality.

Official round medians are 1000.12/999.46/999.37 us; block medians are
939.40/942.28/940.14 us. The median-of-rounds improvement is 59.32 us (5.94%)
for the recurrence alone. This does not reach the 135 us investment gate.
The earlier BF16-state savings therefore do not justify claiming the same
real-model improvement. The FP32 adapter is not advanced to a costly layer
run on the strength of that BF16 result.

`gdn-q4-fp32-profile-v2.json` repeats the complete endpoint and adds isolated
captured component diagnostics, restoring matching state outside each
component timing. Official/block medians are 1001.30/943.26 us. Component
medians are pure products 252.31 us, products including Q/K casts 252.64 us,
Q/K casts alone 41.38 us, and solve/materialization 662.26 us. These isolated
numbers are not additive: cache state, launch conditions and overlap differ.
They are neither a complete-layer result nor a split advantage.

The dominant remaining component is the solver/materializer. At live B98,
its one FP32 state read plus four dense candidate writes account for about
770.70 MB. Direct products add another 154.14 MB state read. Two equal
request chunks could ideally hide only half the products endpoint (about
126 us), before fragmentation, events and memory contention. This upper
bound does not support treating two-way stream scheduling as the next large
optimization. Reducing actual FP32 candidate traffic is a more substantial
next hypothesis; its design and qualification gates are recorded in
[Q4 log pipeline](multi_graph_gdn_q4_log_pipeline.md).

At capacity 100 the explicit products scratch is 9,830,400 bytes and FP32
Q/K conversion outputs total 3,276,800 bytes. Graph-pool retention of these
conversion outputs must be included in full-model peak memory. The reported
6.82 GB allocated / 9.50 GB reserved process peaks include both controls and
correctness snapshots, so they are not incremental model memory costs.

Exact tested native sources, generated products tiling, probe and build logs
are archived remotely under
`/workspace/multi-graph-results/q4-fp32-tested-source-v1/`. Tested block and
products library SHA256 values are respectively
`1b0ac4492614b27a657f2b17763daf2d9b831136faaf56b43a2211b495aa4883`
and `f5bf44dc2d781e8f7048ca3dabb990bd107f8eb5f3481e0667ccffb5950399f4`.

The preceding matching recurrence control is approximately 989-999 us, while
the scoped GDN layer endpoint is about 2709 us. A five-percent GDN improvement
requires at least 135 us removed from the complete recurrence path. A product
endpoint below roughly 200-300 us is the initial gate worth advancing to a
fused solve/materialization implementation; a fast GEMM alone is insufficient.
The remaining budget must cover Gram products, triangular substitution,
FP32 arithmetic, four complete BF16 candidate writes and output emission.

This algebra reduces repeated matrix-vector reductions; it does not remove
the required candidate writes. At B98, four BF16 matrix snapshots alone total
308.28 MB. A three-kernel implementation adds a 77.07 MB selected-state gather
read and write and another GEMM read, whereas the installed recurrence already
keeps its FP32 working matrix in UB. Those extra bytes can erase Cube savings.
External state traffic is therefore not claimed to be improved.

Stop or redesign if the mixed-precision GEMM is unsupported, product/gather
preparation exhausts the budget, complete recurrence does not save 135 us,
continued-state drift is unacceptable, or a fair padding control gains the
same amount without producing a complete replay advantage for splitting.
Only actual continuous-layer and full replay measurements on the fixed
dynamic trajectory can establish the thread's performance target.

The installed FLA recurrence source is pinned by the wheel
`flash_linear_attention_npu_a2-26.9.1+deva4a7958`, A2 tier, original
`recurrent_gated_delta_rule.h` SHA256
`7661897a77d6cf8f4e2c3f9c44d24c1fdb3547ee3d3e55435b0857b1192cb17f`.
Its original path, other source hashes and retained license are recorded in
`benchmarks/multi_graph_gdn_compact_state/README.md`. This route must first
compare against that actual installed recurrence rather than a weaker tiling
control or an independent Python recurrence.
