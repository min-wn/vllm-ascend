# Device-active Matmul inside a fixed ACLGraph

## Decision and scope

A fixed AscendC kernel can read a device-resident active-row count and reduce
its own Matmul work without a conditional ACLGraph or a host task update for
each linear operation. The installed CANN 9.1.0 headers and official A2 API
documentation support the required BF16, NZ-weight and runtime-shape building
blocks. This is an implementation feasibility finding, not a measured kernel
speedup or qualification of a custom operator on this machine.

The current predefined Q4 workload does not offer a promising large arithmetic
saving from this change. Its default graph capacities already match the live
row count rounded to 16. Reducing the logical row count therefore removes no
16-row Cube output blocks. Tail masks, memory traffic and scheduling can still
change; their time has not been measured and is not asserted to be zero.
No new device probe or custom dense operator is introduced for this direction.

This mechanism would also be available to padding. It is a possible general
dynamic-shape graph primitive, not a benefit exclusive to request splitting.
It does not change FIA's true host sequence-length/update requirement.

## Reproducible sources

The official source repository is <https://gitcode.com/cann/asc-devkit>.
The inspected GitHub mirror is pinned at
`a5e1e7815d50e62f6f1814e0928a443234ce4cf6`, dated 2026-10-17:

- [SetSingleShape](https://github.com/hicann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetSingleShape.md) explicitly describes runtime shape modification and reuse for tail blocks; it lists Atlas A2 support.
- [SetTail](https://github.com/hicann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetTail.md) reconfigures single-core M/N/K without changing tiling.
- [SetOrgShape](https://github.com/hicann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/SetOrgShape.md) describes original matrix dimensions and the required call order.
- [Matmul usage](https://github.com/hicann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/Matmul_usage.md) lists A2 BF16 input/output, NZ B matrices, alignment requirements and pure-Cube mode.
- [Matmul configuration](https://github.com/hicann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/adv_api/cube_compute/Matmul_Kernel/MatmulConfig.md) distinguishes tail-capable Norm/MDL from BasicBlock modes without tails.
- [Constant-tiling example](https://github.com/hicann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/examples/01_simd_cpp_api/04_advanced_api/00_matmul/matmul_constant_tiling/README_en.md) allows an actual single-core shape no larger than the configured maximum.
- [Mmad](https://github.com/hicann/asc-devkit/blob/a5e1e7815d50e62f6f1814e0928a443234ce4cf6/docs/zh/api/SIMD-API/basic_api/cube_compute_ISASI/mmad_compute/Mmad.md) explains 16-by-16 matrix blocks and invalid elements when effective dimensions are not multiples of 16. Block-count equality alone is not a timing result.

Read-only inspection of `vllm-ascend-030` used this installed header root:

```text
/usr/local/Ascend/cann-9.1.0/aarch64-linux/asc/
```

Paths below are relative to that root:

| Header | Relevant evidence |
| --- | --- |
| `include/adv_api/matmul/matmul.h` | Public SetOrgShape, SetSingleShape and SetTail declarations/comments, at lines 89, 105 and 112. |
| `impl/adv_api/detail/matmul/matmul_impl_base.h` | SetSingleShape at line 470 calls SetTail; SetTail at line 517 updates shape information and M/N/K loop modules. |
| `impl/adv_api/detail/matmul/param/matmul_shape_info.h` | CheckTailShape at line 71; BasicBlock divisibility checks at line 93. |
| `impl/adv_api/detail/matmul/scheduler/iterator/m_loop/m_loop_norm_base.h` | SetSingleShape recalculates loop state; zero iterations trigger an assertion. |
| `include/adv_api/matmul_intf.h` | ASCENDC_CUBE_ONLY selects the direct Cube implementation rather than a mixed Vector/Cube client path. |
| `impl/basic_api/kernel_tensor_impl.h` | GlobalTensor::GetValue implementation at line 1172 reads a GM scalar on device. |

SHA256 of installed `matmul_impl_base.h`:
`3756e3309f4d1f1818820ccd6d2c7b58b2913941c14fa2bee840438a7031724d`.
SHA256 of installed `matmul_shape_info.h`:
`e3866ef1f324d2a5b5ad92664f20bfec2fe62d26cd7acdf7ed7b9f70d26aef44`.
These installed headers, rather than the latest source alone, establish the
available API surface. Compiler success, graph replay and performance remain
unqualified for this proposed BF16 operator.

## What a feasible implementation would require

Keep the captured kernel launch, pointers, output allocation, workspace and
physical matrix strides fixed. Read one `n_active` scalar inside the kernel.
A fixed set of Cube cores maps only active M/N tiles, skipping tiles whose
starting M coordinate is beyond the active range. Use SetSingleShape for each
remaining single-core tile, including a partial last tile. The launch still
contains the same number of cores; reducing work inside them does not change
ACLGraph topology or require the unsupported conditional-runtime feature.

Keep SetOrgShape consistent with physical storage and output pitch. For an ND
input `[capacity, K]`, ND output `[capacity, N]`, and fixed N/K, changing the
active row count need not change either row stride. SetOrgShape is not a
substitute for changing the loop bounds with SetSingleShape. If called, it
must precede SetTensorA, SetTensorB, SetBias and SetSingleShape.

NZ weights can remain resident. BF16 NZ axes require multiples of 16, which
the current Qwen3.5 GateUp K=5120 and local N=17408 satisfy. A linear weight
stored as `[N,K]` requires consistent transpose settings in the MatmulType,
host tiling and SetTensorB call, plus correct NZ tile offsets. A pointer to a
PyTorch format-29 allocation is not sufficient proof of correct indexing.
Weight conversion must happen once, outside replay, for both comparisons.

The important API and implementation limits are:

- SetSingleShape and SetTail are the same operation in the installed implementation; they change device iteration state, not just an output mask.
- Keep `enableSetTail=true`; disabling it lets the compiler remove dynamic shape handling. SetOrgShape handling likewise requires its configuration flag when used.
- BasicBlock and SpecialBasicBlock require single-core M/N to be divisible by their basic M/N. Arbitrary tails cannot be added to those configurations without a separate tail path. Norm/MDL is the conservative API route; static tiling does not require selecting BasicBlock.
- Never pass zero active rows into the Matmul loop. Take a uniform device no-work path first. A pure-Cube implementation avoids leaving a paired Vector/Cube service waiting on only one side of a conditional path.
- A same-stream write or explicit dependency must make the current scalar visible before every replay. Reusing its address must not reuse a stale cached value. A one-time captured host scalar is not a device-active implementation.
- If unchanged downstream kernels consume inactive output rows, define those rows before consumption. Zeroing, copies, scalar preparation and all joins must be included in any comparison that needs them.

The scalar read and SetSingleShape bookkeeping have a cost, as do general tail
handling, fixed-grid idle cores and less shape-specific tiling. Their latency
is unknown. Avoid attributing a custom tiling improvement to active-row
clipping by comparing only against a library kernel with different tiling.

## Existing repository work and lower-cost alternatives

At repository base `fe412935bd728c9cf46cf590fea39a4649a4a601`, the
`multi_graph` benchmarks and design documents have no BF16 dense kernel that
reads `n_active` and changes its M work inside one fixed captured launch.
The existing dense probes vary host-known shapes or split fixed graph buckets.
NZ layout and shared-dense measurements therefore do not already test this
mechanism.

There are related runtime-shape patterns in existing GMM operators, including
`csrc/gmm/grouped_matmul_swiglu_quant_weight_nz_tensor_list/op_kernel/` and
`csrc/gmm/gmm_dequant_situ_quant/op_kernel/`. They use runtime SetOrgShape and
SetSingleShape, but are quantized/grouped paths and are not a qualified BF16
Qwen dense replacement.

Installed GroupedMatmul also reads a device group-list tensor and computes
runtime M tiles. Its `grouped_matmul.h` Process method skips empty groups;
`grouped_matmul_utils.h` GetSplitValueFromGroupList reads the counts with
GetValue. This provides an existing operator to investigate before building
a custom GEMM if the arithmetic budget later warrants it. A one-group BF16,
NZ-weight, fixed-output-capacity use must still satisfy the installed public
API's shape contract and graph support. It has not been executed or qualified.
Latest official API documentation is pinned separately at
`fd659b0e14917ece0c995e7096b48b7999254b94` in
[ops-transformer](https://github.com/hicann/ops-transformer/blob/fd659b0e14917ece0c995e7096b48b7999254b94/gmm/grouped_matmul/docs/aclnnGroupedMatmulV3.md);
it is not evidence of the installed binary's behavior.

## Why the current workload does not justify a large implementation

The captured default-capacity evidence is the workspace report
`outputs/multi_graph_tp_attention_mlp/tp2-27b-gdn-owner-boundary-v1.json`,
under `selection`. These are actual EngineArgs/dispatcher-derived sizes, not
a proposed custom capacity list:

```text
... 240, 248, 256, 272, 288, 304, 320, 336, 352, 368,
384, 400, 416, 432, 448, 464, 480, 496, 512
```

`benchmarks/multi_graph_dynamic_trace.py` predefines Q=4 and B in [64,128].
For all 65 possible batch sizes in that range, live M is in [256,512], and:

```text
default_capacity(M) = 16 * ceil(M / 16)
ceil(default_capacity(M) / 16) = ceil(M / 16)
```

The same absence of a removed M tile holds for ordinary base-M values 64 and
128, both multiples of 16. In particular:

| Live batch | Live M | Default capacity | 16-row output blocks, live / padded |
| --- | ---: | ---: | ---: |
| 97 | 388 | 400 | 25 / 25 |
| 98 | 392 | 400 | 25 / 25 |
| 99 | 396 | 400 | 25 / 25 |

For the TP2 GateUp, K=5120 and N=17408. At M=400 its logical work is
71,303,168,000 FLOPs, its BF16 weight is 178,257,920 bytes, and its BF16 output
is 13,926,400 bytes. Moving to M=392 removes 2% of logical rows but removes no
minimum 16-row output block or weight matrix. It changes only the validity of
the final block's rows, not the number of full M blocks in this accounting.
This does not rule out a tail-traffic improvement; it rules out using the
logical 2% row reduction as evidence of a large Cube-work reduction.

The conclusion is specific to these default capacities and this predefined
trace. Different default capacity ranges or workloads may cross real M-tile
boundaries. Capturing a single oversized capacity for all small steps would
create a larger saving against that artificial baseline, but would not answer
whether this mechanism beats the existing default-bucket padding path.

## Minimal causal experiment if the workload budget changes

Use one actual TP2 GateUp weight and one fixed captured allocation first.
Keep a library NZ-padding reference, a custom kernel forced to full capacity,
and the identical custom kernel reading `n_active`. The custom full/active
pair isolates clipping from differences in tiling and compiler code generation.
Include the best library ND/NZ baseline rather than assuming NZ always wins.

Change scalar and input contents over a predefined sequence without recapture;
include full capacity, partial tails, zero rows, shrinking and growing counts.
Check live rows against exact-shape eager linear outside timed execution,
output/address stability, guards and the documented inactive-row policy.
For graph timing, include scalar preparation, required tail initialization,
all output dependencies and endpoint synchronization. A library exact-shape
oracle is a correctness check, not an extra supported capture size.

Only proceed beyond that gate if active clipping beats both the same custom
full-capacity kernel and the best default-bucket library padding reference on
the predefined dynamic workload. A repeated single-weight test is only an
operator feasibility gate: it does not establish a full-model gain under a
real weight working set, TP communication, GDN state updates or FIA updates.
