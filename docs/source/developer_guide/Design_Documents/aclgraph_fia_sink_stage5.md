# FIA sink stage 5: native whole-row metadata producer

Status: the native producer passed the bounded fixed-capacity replay gate
and B1/B4/shared-plan regressions.
This stage establishes an executable metadata policy and its safety evidence.
Producer-inclusive performance, the unchanged 128-step trace, full-model
qualification and end-to-end gains have not been measured in this stage.

## Purpose and attribution

Stage four found a lower attention-chain execution envelope for whole-row
assignment, but its timed Python plan construction and copy consumed the
observed saving. Stage five moves that policy into the native AICPU metadata
producer, which reads the current device-resident lengths on each replay.
The graph retains fixed tensor addresses and physical capacity while the live
batch and KV lengths change.

The device metadata producer, shared-plan consumer and 1024-word metadata ABI
come from the pinned official source at
`bec19b8dea1e538e64838f3064f7e91a3fa9ee72`, together with the previously
qualified narrow A2 D256 port. Those mechanisms are not new contributions of
this stage. The independent extension is the guarded construction of a fresh,
complete-row, no-FlashDecoding (no-FD) plan using the native per-head costs,
including transactional publication and the replay qualification below.
This is an attention work-assignment policy inside one graph; it does not
establish a graph-splitting or multi-stream speedup.

## Native policy

The opt-in `THESIS_NO_FD_WHOLE_ROW` macro changes only
`csrc/attention/fused_infer_attention_score_v2_sink_metadata/op_kernel_aicpu/fused_infer_attention_score_v2_sink_metadata_aicpu.cpp`.
After standard length parsing and inner-tiling selection, the producer calls
the existing `CalcSplitInfo` and `CalcCostInfo` helpers. The new builder reads
`costInfo.bN2CostOfEachBatch` and constructs a plan for the current lengths.
It does not reconstruct a plan from the stage-four CPU preflight snapshot.

One indivisible work item is a complete `(request, KV head)` row. With Q4,
six query heads per KV head and `mBase=512`, each row has 24 query/group
positions and fits within one M tile. Keeping the entire KV traversal for a
row on one core allows the existing consumer to write its final result
directly, without an FD partial-result merge.

The builder orders rows in the existing batch/head traversal order. It uses
the minimum of the available core count and positive-cost row count, with
one core for the all-zero-KV case. A binary search with greedy feasibility
finds the minimum possible maximum segment cost for contiguous whole-row
segments under the supplied integer costs. Reconstruction gives each active
segment positive work when any exists; the final endpoint covers all
physical rows, including zero-KV padding.

The optimizer reuses the native cost model; its coefficients remain a
heuristic, not measured per-core execution times or an online calibration.
The gate's separately reported `core_costs` are an abstract diagnostic proxy
from the validation helper and must not be interpreted as the native
optimizer weights or NPU timing.

The output is a newly constructed `SplitResult`, rather than an existing FD
plan with its FD counts cleared. Its GS1/S2 endpoints, FD fields and reduction
vectors start at zero. The builder validates strictly advancing whole-head
endpoints, useful active segments and the complete final sentinel before
assigning the candidate to the published result. The existing consumer and
metadata ABI are unchanged: `numFD=0` and `usedFdVec=0`, with every per-core
FD field also zero.

The macro and runtime guards restrict this policy to the qualified geometry:

- Non-MLA TND query and BnBsH paged KV, 12 query heads, two KV heads, D256
  Q/K/V, page size 128, sparse mode 3, no RoPE and no sink.
- Physical batch 1 through 128, exactly Q4 per physical request, exact Q/KV
  vector dimensions and cumulative Q endpoints `(b+1)*4`.
- Nonnegative KV lengths at most 4097, with native per-head cost zero exactly
  when the corresponding KV length is zero.
- One through 20 AIC cores, twice as many AIV cores, `mBase=512`,
  `mFdBase=8` and a positive inner KV tile at most 512.
- Batch-invariant mode disabled; BF16 is the consumer dtype qualified by
  these experiments.

If a guard or candidate validation fails, the producer calls the upstream
`SplitCore` implementation. Compiling without the macro also preserves the
upstream path. These fallback conditions do not qualify untested geometries
for the new policy.

## Build and loaded-artifact verification

An initial build returned zero after compiling the reviewed metadata object,
but left the old AICPU shared library in place. The custom link output lacked
a dependency on that object, so object recompilation did not force relinking.
The first gate rejected the resulting plan. The second gate retained the
full 1024-word payload and rejected it before attention warmup or graph
capture: its base header showed zero FD counts, while per-core FD fields and
endpoints still violated the complete-row ABI. A successful compiler exit
and zero FD counts were therefore insufficient deployment evidence.

The corrected build preserved a backup of the previous library, removed the
stale output from the isolated build target and recorded a new verbose link
command containing the reviewed metadata object and required CANN AICPU
archives. The new library has a different size and hash and is staged under
the unique name `libthesis_stage5_whole_aicpu.so`. Its AICPU dispatch JSON
selects that name in a copied, isolated vendor package.

| Artifact | SHA256 |
| --- | --- |
| Include-corrected metadata source before this patch | `a3e4bbeeca240d55d09f8624efe0a83fc5e77c1f692d72f73ee55a0872e9b51d` |
| Patched metadata source | `7d48525fddf767508ad984d9d4a4afa1b8b6008647a24e20d77e9f87205b8f51` |
| Reviewed metadata object | `9fdc92e3d73d812c9de8941fd57baa0c3475d76a2192189ed462efc12d46afff` |
| Newly linked and staged AICPU library | `1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90` |
| Unchanged baseline OpAPI library | `2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40` |
| Unchanged extension | `25a41e47473a3bc9c34a50c053766181c7a764ea496b93d4d3b70180b0ecf76b` |

The host tiling library and attention AICore objects also retain their
stage-two hashes. The build report records the link command, source/object
provenance, copied-vendor inventory and materialized symlink hashes. No
system installation, baseline OPP replacement or driver update was performed.
The standalone C++ harness additionally compiled and executed the extracted
whole-row algorithm with C++17 and `-Wall -Wextra -Werror`; that is a CPU
algorithm check, not NPU qualification.

## Fixed-capacity replay gate

The corrected native v3 gate runs on Ascend910B3 with 20 AIC/40 AIV cores,
CANN 9.1.0, PyTorch `2.10.0+cpu` and `torch_npu 2.10.0.post4`. It uses one
attention occurrence at default capture capacity 400 tokens: 100 physical
Q4 requests. It does not introduce a capture size for a particular live
batch. The canonical Q4 trace digest is
`0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`.

Before any attention warmup, the runner invokes the native metadata
producer, persists all 1024 int32 words and their hash, checks loaded-symbol
provenance and validates the complete-row ABI. It repeats plan validation
after each replay. Earlier failed runs and payloads remain in the diagnostic
report rather than being replaced by the successful run.

| Replay | Live requests / tokens | Maximum live KV | Active cores | Result |
| --- | ---: | ---: | ---: | --- |
| Initial canonical step 0 | 98 / 392 | 1539 | 20 | Passed |
| Changed canonical step 6 | 99 / 396 | 1549 | 20 | Passed |
| Derived all-zero-KV fixture | 0 / 0 | 0 | 1 | Passed; all outputs exactly zero |
| Recover canonical step 0 | 98 / 392 | 1539 | 20 | Passed |

The all-zero fixture keeps the same 100 physical Q4 slots, with zero KV
lengths and zero page routes. Its base metadata contains a planning fallback
S2 value of 1024; this is not a real KV length. Its final whole-head sentinel
still covers all 200 physical `(request, KV head)` rows.

All four replays pass the full-payload no-FD checks, finite-output checks,
exact zero-padding checks and fixed-address/descriptor checks. The runner
also checks resident query, Q/KV length and page-table immutability. It does
not reread the entire resident K/V storage after every replay, so these
checks are not a full K/V memory-sanitizer proof.

For each nonempty replay, occurrence 0 checks three sampled request rows
against an FP32 reference: rows 0/49/97 initially and after recovery, and
0/49/98 after the change. The largest absolute error is
`0.00034021586179733276` in the changed replay; the initial/recovery maximum
is `0.00017317384481430054`. The inherited tolerance is `rtol=0.03` and
`atol=0.003`. The empty fixture checks exact zero over the full output.
These are sampled-reference and full-output structural checks for one
occurrence, not FP32 checks of every row or a full model.

Whole-row attention changes floating-point reduction order relative to FD
partial-softmax reduction. The successful tolerance checks do not establish
bitwise identity or batch-invariant semantics.

## Additional regression gate

B1, B4 and shared-plan regressions of the same isolated vendor package all
pass. Each graph contains one metadata producer; the shared-plan case has
two attention consumers using that plan.

| Gate | Replays | Attention invocations | Coverage |
| --- | ---: | ---: | --- |
| B1 | 8 | 8 | KV 4/7/127/129/513/1025, shortening and recovery |
| B4 | 7 | 7 | All-live, Q holes, resume, shortening, zero KV, all-empty, recovery |
| Shared plan | 7 | 14 | The same B4 sequence with two distinct attention inputs |
| Total | 22 | 29 | All passed |

Every attention invocation compares its full output against the independent
FP32 reference at `rtol=0.03`, `atol=0.003`. The fixtures use a value-64
forbidden-tail witness to distinguish correct causal masking from full
attention, with the recorded sensitivity exceeding the required normalized
difference. All cases also pass exact input-storage, zero-tail and stable
address checks; outputs for nonempty Q rows with zero KV are exactly zero.
Output allocation guards remain unavailable because the native wrapper owns
the output allocation.

The B4 and shared-plan Q-hole cases have non-Q4 physical rows and therefore
exercise the upstream fallback rather than qualifying the new policy for
mixed Q lengths. Their metadata records positive FD counts during those
cases; the final all-live Q4 replay returns to zero FD. The all-empty
fallback likewise produces zero output. This verifies the tested transitions
between the two policies, without attributing fallback execution to the
whole-row builder.

The maximum B1 absolute error is `0.18599700927734375` on the final KV4
replay. This belongs to the value-64 mask-witness fixture and passes the
elementwise relative/absolute tolerance; it is not the `0.0003402` maximum
from the separate sampled-row main gate. These regressions do not establish
bitwise identity, full-trace coverage or a performance gain.

## Remaining qualification

This stage reports `performance_measured=false`,
`full_trace_measured=false`, `full_model_qualified=false` and
`end_to_end_gain_established=false`. Successful dynamic metadata replay
establishes that this implementation can be tested; it does not establish
that native plan construction preserves the stage-four oracle saving.

The next comparison keeps the four existing default capacities 256, 400,
496 and 512, sixteen private attention occurrences and matched inputs. It
must include current-length preparation, native producer work and attention
inside the measured endpoint. Both paths need the reusable preparation
control from stage four, and the baseline must remain the existing native
shared producer rather than per-occurrence redundant planning.

Only after that comparison is positive should the same implementation run
the unchanged 128-step dynamic trajectory, including capacity transitions,
current KV lengths, padding and recovery. Full-model integration and
end-to-end measurement are later gates. A negative producer-inclusive result
would leave this as a qualified policy experiment without a performance
claim.

## Evidence

- [`fia-sink-stage5-native-gate.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage5-native-gate.json)
- [`fia-sink-stage5-build.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage5-build.json)
- [`fia-sink-stage5-diagnostics.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage5-diagnostics.json)
- [`fia-sink-stage5-regression.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage5-regression.json)
- [`fia-sink-stage5-cpp-check.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage5-cpp-check.json)

The parent workspace retains the source patch manifest, reviewed builds,
genuine relink log, v1/v2 failure reports, v3 gate report and deployed-tool
manifests under `outputs/thesis_incremental_plan/stage5-evidence/`. This
parent-workspace raw evidence is not a repository-relative documentation
link.
