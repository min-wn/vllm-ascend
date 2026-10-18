# FIA sink stage 2: D256 kernel adaptation and correctness gates

Status: bounded operator qualification completed, 2026-10-18. The isolated
native build, 25 eager cases, D128/D256 constant regression and all three
graph gates pass on the actual 910B3. This does not qualify the complete
model or establish an incremental-planning performance gain.

## Scope

The [first stage](aclgraph_fia_sink_stage1.md) showed an explicit host
dimension rejection and an insufficient K L1 slot for the target. This stage
ports the pinned sink arithmetic to BF16, TND query, Q12/KV2, D256, paged
ND128, without RoPE/sink tensors. It then checks eager semantics and fixed
address dynamic ACLGraph execution on device 0. No model or performance
matrix is required at this stage. The existing installed extension, system
OPP, driver and stage-one artifacts are retained.

The independent source and build root is
`/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e`, based on commit
`bec19b8dea1e538e64838f3064f7e91a3fa9ee72`. Dependency caches and the already
qualified host binding are copied from stage one. The binding's dispatcher
namespace remains `thesis_fia_sink_stage1`; its native symbol resolution is
checked against the separate stage-two vendor leaf before each fresh process.
Namespace reuse does not mean loading the original native operator binary.

## Kernel change

`benchmarks/aclgraph_fia_sink_stage2_patch.py` checks exact upstream file
hashes before applying three changes in an isolated source tree:

1. The previously audited metadata AICPU include-path correction.
2. A narrow host predicate for the new D256 geometry. Existing accepted
   dimensions retain their original predicate. Matching three-dimensional
   storage shapes alone does not qualify ND format; native registration,
   downstream checks and actual execution remain necessary.
3. D256-only internal MM1/MM2 subblocks, with compile-time capacity assertions.

| Operation | D256 limit | Existing capacity used |
| --- | --- | --- |
| MM1 Q in L1 | M128 x D256; M64 for the S2=1024 configuration | 49152 BF16 elements; 24576 in that alternate configuration |
| MM1 K in L1 | N64 x D256 | 24576 BF16 elements in each KP slot |
| MM1 L0 | M128 x N64 x K128 or smaller tails | A/B slots 16384 BF16 elements; C slot 16384 FP32 elements |
| MM2 P in L1/L0A | M64 x K64 | Existing KP and A slots |
| MM2 V in L1/L0B | K64 x D256 | 16384 BF16 elements in each slot |
| MM2 result in L0C | M64 x D256 | 16384 FP32 elements, exact fit |

The normal L1 layout remains 464 KiB. KP slot counts, buffer/event IDs,
ring ownership and cross-core synchronization placement do not change.
Expanding all three KP slots to 64 KiB would consume the full 512 KiB and
would still leave Q/L0C limits unresolved. Smaller internal tiles address
all three bounds without assuming spare L1.

Original `ResetLoad3DConfig` and `LoadAToL0` template constants remain.
The existing actual-height branch sets `l1H` from each smaller tile and
restores the original configuration afterward. Thus earlier dimensions
retain their original runtime blocks. Static assertions cover both normal
and S2=1024 configurations; they are not device memory-sanitizer evidence.

Proper TND prefixes keep each Q4/G6 request task at M<=24. Nevertheless,
host workspace planning uses physical packed T and wider query intervals
can create larger M. The port loops over bounded M subblocks instead of
relying on the B1 smoke's small M. A wide-query numerical boundary gate
checks this mechanism separately from the target Q4 cases.

## Correctness gates

The eager validation tool uses an independent FP32 CPU softmax reference,
with query lengths accumulated only for the native TND ABI. KV lengths are
per request. Random physical page permutations, unused-page poison and
last-page tail poison test accesses against exact logical lengths.

Positive cases include page and tile boundaries, multiple ragged requests,
long KV/FD work and a wider query interval. Causal cases use the pinned
sparse3 compressed mask contract. Forbidden-tail witnesses make ignoring
the causal mask observably wrong even for long contexts. The same-input
causal/noncausal reference separation is recorded; short random cases
alone do not qualify the long-context causal branch.

Zero KV and zero Q are separate groups. Zero KV output rows must be exactly
zero. Query guards and all input storage, including unused KV regions and
metadata, must remain unchanged. The binding allocates output internally:
this ABI cannot provide caller-controlled output allocation guard regions.
Output shape, finiteness and numerical checks are therefore distinct from
the input storage checks.

The graph gates must capture both metadata and attention. Capture at short
KV and replay longer lengths uses unchanged physical allocations, with
current lengths and page tables copied in place before replay. A fixed
capacity with paused/resumed requests checks changing cumulative Q prefixes,
packed active rows and exact zero padding. A shared-plan gate uses two
independent sets of Q/K/V with one metadata producer. This is an operator
qualification, not sixteen-layer model integration or serving validation.

Only actual stream limits at or below 20 Cube/40 Vector cores are used.
The reviewed FD LSE buffer has a separate partial-count capacity; the
serialized metadata's 36-slot size is not permission to increase core or
split counts. The port does not change metadata endpoints or FD ownership.

## Executed results

The complete bounded pipeline returned 0. All steps ran on physical device 0
with 20 Cube/40 Vector cores, CANN 9.1.0, torch 2.10.0+cpu and
torch_npu 2.10.0.post4. The driver remains 24.1.rc3. The graph child sets
`TASK_QUEUE_ENABLE=0` before importing Torch, as in the earlier model work.

| Gate | Actual result |
| --- | --- |
| Selected native package build | Pass; all 13 final kernel objects packaged. |
| Independent D128/D256 constant regression | Both pass; maximum absolute error from 1 is 0. |
| Positive eager semantics | 21/21 pass, including Q97 internal-loop and KV4097 boundaries. |
| Zero KV / zero Q eager semantics | 4/4 pass; zero KV output rows are exactly zero. |
| B1 dynamic graph | 8/8 replay steps pass, captured at KV4 then replayed through KV1025 and back. |
| B4 dynamic graph | 7/7 replay steps pass, including holes, zero KV, all inactive and recovery. |
| Shared metadata graph | 7/7 steps pass with two attention occurrences and one metadata producer. |

Across the eager gates, ordinary noncausal random cases have maximum absolute
error at most 0.000615 against the FP32 reference. The Q97 internal-loop
boundary has maximum absolute error 0.000340. Causal witnesses deliberately
put value 64 in the final tokens: their largest absolute error is 0.193299,
while satisfying the predefined elementwise `rtol=0.03, atol=0.003` check.
That large-input stress error must not be mixed with the ordinary-random
absolute-error range or presented as bitwise equality. Short KV4 and KV7
witnesses are included. Every nonempty causal fixture rejects a noncausal
reference by more than ten times the same elementwise tolerance.

Actual metadata confirms FlashDecode work: the KV4097 cases contain two FD
groups with 9 and 8 KV partials, rather than merely having a test name that
claims to cover FD. Metadata core and partial bounds pass. The wider-query
case checks inner M loops; it does not qualify arbitrary mixed-query model
execution.

All graph steps compare independent CPU references after in-place changes
to Q/K/V, page mappings and device lengths. Input descriptors and pointers,
metadata pointers and output pointers stay fixed; metadata contents change.
Every input backing region remains exact, including query guards and unused
KV pages/tails. The physical output tail and zero KV rows are exactly zero.
The shared gate checks distinct expected outputs for its two data sets in
the executed, hashed runner; it does not persist the full input tensors.
Graph metadata/attention counts describe the executed capture code and
observed replay behavior, not a profiler-derived task inventory.

## Evidence and stopping boundary

The source transformation and raw records are preserved in the parent
workspace at `outputs/thesis_incremental_plan/stage2-evidence/`, and on the
remote isolated root. The reusable summary is
[`fia-sink-stage2-qualification.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage2-qualification.json).
It links the executed scripts, library inventories, distinct case coverage,
FD/core bounds and each graph step. No original failure or build attempt is
overwritten to make the final status positive.

The generated package is 19,477,848 bytes with SHA256
`579ba0d5546eee27f5275cf6417b4427ffe79bc15b42de5941c5ea5633921581`.
The patch utility can reconstruct the source changes from the pinned
archive; `--check` verifies all input hashes without mutation.

Stage two ends at this bounded operator qualification. These checks do not
constitute a device memory-sanitizer proof, exhaustive shape qualification,
concurrent cross-stream lifetime qualification, or complete-model/serving
accuracy. B1/B4 fixed-capacity graphs are tested, not the unchanged B64..128
model trajectory or a sixteen-layer shared plan. No incremental native
planner, calibrated critical-path budget or new performance result is
established by this kernel port alone. The next milestone is to measure
one shared plan's exposed cost and the attention kernel tail before deciding
which part of the proposed scheduling algorithm merits native implementation.

That [third-stage measurement](aclgraph_fia_sink_stage3.md) is now complete.
It qualifies sampled B64..128 attention chains across all default buckets,
but remains outside complete-model accuracy and performance. The measured
single shared planner is too small to justify planner-only large speedup;
the next experimental gate targets preparation and actual attention execution.

Scoped Ruff, Python compilation and project import/function checks pass for
the saved execution tools. Required `bash format.sh ci` was attempted: Ruff
and Markdown pass. Existing unrelated spelling/import findings and missing
Windows shell/tool hooks still fail the full command. Its log is preserved
at `outputs/thesis_incremental_plan/format-ci-stage2.log` in the parent
workspace; those failures are not reported as a passing full repository CI.
