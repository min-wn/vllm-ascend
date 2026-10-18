# Thesis proposal: incremental attention plans with selective KV splitting

Status: research proposal, 2026-10-18. There is no new NPU performance result
for this algorithm. The user permits an ACLGraph optimization without request
splitting and wants an independent graduation-project contribution beyond
adopting an official PR. Replicating attention or MLP weights is excluded.

## Implementation checkpoint

The first CPU reference and independent semantic oracle are implemented in
`benchmarks/aclgraph_incremental_attention_plan.py`, its immutable types helper
and `tests/ut/worker/multi_graph/test_aclgraph_incremental_attention_plan.py`.
The initial 19 tests pass. They check exact work coverage, causal and
noncausal FP32 split-softmax merging, changing request lifetimes, inactive
rows, rollback after invalid input, preflight capacity checks, high merge-cost
rejection in full planning and split scratch bounds. This is not an
ACLGraph device producer, native metadata exporter or measured attention gain.

The prototype keeps request-local tile templates and prefix costs, patches
exact tails, constructs current row routing, and repairs contiguous partition
boundaries within a bounded search radius. It falls back to full boundary
planning when the cost/imbalance checks reject repair. Both full and incremental
controls use the same abstract cost model. Its zero default merge cost is an
optimistic structural ceiling, not a calibrated hardware assumption; nonzero
fixed-cost sensitivity is reported separately. The Python CPU timing is not
representative of a future AIV/AICPU implementation.

The first CPU screen exposed a merge-cost defect in local repair: at merge
charge 16, it retained 1717 split groups across the unchanged 128-step trace
and scored 298174 abstract cost units, versus 11 groups and 282352 units for
full planning. Repair used a weak coalesced-unsplit reference and its final
imbalance check excluded merge cost. A fresh valid no-split quality floor
and complete-objective acceptance check now reject these expensive repairs.
The merge-aware v4 screening snapshot reports:

| Policy / merge charge | Full plan cost proxy | Incremental plan cost proxy |
| --- | ---: | ---: |
| No split | 281870 | 281474 |
| Selective split / 0 | 268500 | 269914 |
| Selective split / 4 | 274840 | 276188 |
| Selective split / 16 | 282352 | 282374 |

These are sums of abstract critical-core-plus-merge cost units, not device
times. At charge 16, incremental repair retains 12 split groups but falls
back to full partitioning on 126 of the 127 post-initial steps. The quality
guard addresses that counterexample but does not establish an efficient
incremental scheduler or a universal approximation bound. Recomputing a
baseline every step consumes the intended maintenance saving; a cheap quality
bound remains an open algorithmic task.

Single-round Python totals for 128 steps are 602/828 ms for full/incremental
no-split and 869/931 ms for full/incremental selective split. These are
implementation diagnostics, not NPU speedup estimates. The report, source
hashes and fixed merge-charge sensitivity are preserved in the parent
workspace at `outputs/thesis_incremental_plan/cpu-screen-v4-merge-aware.json`.
The initial counterexample is preserved in `cpu-screen-v2.json` there.
A subsequent tested correction caps no-split partition count at the number
of attention groups and excludes the final endpoint from internal cuts.
The v4 report pins the preceding source hash; it is not a timing report for
this final small-group boundary correction.

The current container was inspected through registered Torch operator schemas,
not just `dir(torch.ops)`, which is lazy. Both sink operators, their Python
wrapper and the three native source directories are absent. The installed
PyTorch and torch_npu versions are 2.10.0+cpu and 2.10.0.post4. Devices 0/1
were idle during the inventory. This is an installation prerequisite, not a
new driver-compatibility failure.

The pinned native consumer has prefix-connected core ranges: each core starts
at the previous core's endpoint. It uses the current used-core count to
select consumers. Arbitrary persistent-request task lists and enable-flag holes
cannot be encoded directly. The CPU reference therefore retains contiguous
current-row work order; exporting valid split-reduction metadata remains a
separate implementation gate.

The pinned source also has an unresolved AICPU relative include and additional
potentially missing transitive includes. No native build has run. The target
Q4/D256/GQA6 shape follows the general GQA route rather than the specialized
D64/D128 route. A source-derived upper workspace formula at 24 cores is
145.5 MiB plus library/executor storage; this is not a measured allocation,
and the 4 KiB metadata size does not describe the total workspace.

The local inventory is preserved in the parent workspace at
`outputs/thesis_incremental_plan/sink-remote-inventory.log`. The native source
audit pins the upstream commit separately from installed capabilities.

## Existing project results and thesis scope

The existing branch supplies default-bucket dispatch, true FIA host updates,
request ownership checks, TP2 graph experiments and a fixed dynamic trace.
Its compact FP32 GDN state logs and ready-tile pipeline are public operator
improvements that can also accelerate padding. The observed ready-versus-delayed
complete-model forward/replay reduction is 4.586%, not a proven incremental
request-split gain and not an online serving result. This experiment covers
the 64 decoders and final normalization; it does not establish sampler,
request-queue or complete serving throughput. Full physical-state correctness
was checked on three continued steps per mode, not all 128 timed steps.

Those artifacts provide an implementation and evaluation foundation. Their
independent literature novelty and complete qualification would need their
own review before being presented as thesis contributions. The new thesis
candidate adds a concrete scheduling algorithm and must compare it against
full shared-device planning using equivalent attention arithmetic.

## Research question and proposed contribution

Can a fixed-address ACLGraph maintain an exact, load-balanced attention
schedule more cheaply by reusing request-local work structure across steps,
repairing membership changes and splitting only work that causes imbalance?

The proposed contribution is **incremental maintenance of an attention
execution plan under dynamic batch membership, with bounded imbalance and
selective KV-range splitting**. Dense projections and MLP retain their
ordinary full-bucket GEMMs and original TP weight shards. The split is inside
attention work assignment; it does not require splitting the whole model.

This is an algorithm to develop and falsify, not a proven performance win or
a claim of being the first incremental scheduler. The related-work review
establishes close predecessors. A thesis contribution must be the specific
maintenance algorithm, correctness contract, implementation and measured
tradeoffs, rather than a new name for plan/run or persistent kernels.

## What is inherited and what must be new

| Mechanism | Existing evidence | Role in this project |
| --- | --- | --- |
| Device exact lengths and metadata consumption | Official [FIA sink PR 15336][sink] | Strong baseline and possible native-kernel interface, not our novelty. |
| Separate planning and execution, fixed grid, compatible-layer reuse | [FlashInfer, MLSys 2025][flashinfer] | Baseline design principle, not our novelty. |
| Device-driven persistent attention and multi-token scheduling | [FlashInfer balanced GQA/Cake][cake] | Compare against nonincremental device planning; merely using a queue is insufficient. Its published target differs from A2. |
| Reusing a precomputed causal block structure during decoding | [FlexAttention for inference][flex] | Boundary reuse alone is not novel. Our question includes changing membership, load assignment and split reductions. |
| Length/occupancy-based multi-block attention | [TensorRT-LLM attention][trt] | Choosing split counts from sequence length alone is already established. |
| Current ordered-KV buckets in this repository | `worker/multi_graph/buckets.py` | Cost-estimate interpolation for split routing; it does not maintain operator execution metadata. |
| Stable request identities, local topology summaries, repair budget, imbalance trigger and bounded split ownership | Proposed here | Actual algorithm and experiments to implement. |

Do not report upstream sink adoption or sharing a single plan across layers
as the graduation project's independent contribution. It is also invalid to
compare only with legacy host-updated FIA after a stronger shared-plan sink
becomes eligible.

## Observation from the unchanged trace

The existing 128-step Q4 trace has 128 stable request IDs, randomly selected
batch sizes 64..128, shuffled row order and paused/resumed requests. Its
canonical SHA256 remains
`0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`.

Read-only CPU analysis over 127 adjacent transitions finds:

| Observation | Count |
| --- | ---: |
| Total active request observations over all 128 steps | 12493 |
| Request observations surviving into the next step | 9489 |
| Survivors crossing a 512-token KV tile boundary | 54 / 9489, 0.569% |
| Survivors crossing a 128-token page boundary | 199 / 9489, 2.097% |
| Survivors changing a 64-token cost quantum | 390 / 9489, 4.110% |
| Admissions or reappearances between adjacent steps | 2906 |
| Removals between adjacent steps | 2927 |
| Identical adjacent ordered request lists | 0 |
| Identical adjacent active sets | 1 |
| Identical adjacent complete ordered tile vectors, at 128/256/512 | 0 |

The observation supports request-local temporal structure, not whole-plan
cache hits. Churn is substantial: the mean adjacent set symmetric difference
is about 45.929 requests. Reappearance is not necessarily a new lifetime;
the request generation changes only when the identity/storage is recycled.
Both churn and tail-cost changes count toward maintenance. A tile crossing
percentage is not the fraction of total planner work or total latency saved.

The CPU calculation and per-transition evidence are preserved in the parent
workspace at `outputs/thesis_incremental_plan/audit_trace.py` and
`trace_audit.json`. Adjacent-survivor percentages use denominator 9489.
Comparisons with each request's last appearance include resumed requests and
have a different denominator; do not mix those rates.

The current trace has modest context lengths and already has many independent
requests. Its full shared planner may be cheap and its attention may already
balance well. This trace is a required control, not evidence of a large gain.

## Execution state and correctness contract

Maintain three distinct forms of state:

1. **Exact semantics, refreshed every step.** Device lengths, query offsets,
   active IDs, row mapping and page tables determine real attention and cache
   accesses. They are never replaced with rounded lengths or an old plan.
2. **Request-local structural summaries.** Stable `(request ID, lifetime
   generation)` keys own KV-range topology, query/head geometry and cost
   summaries. Batch row shuffles update routing rather than request identity.
3. **Assignment and merge plan.** Core boundaries, optional KV split ranges,
   reduction slots and plan generation determine who computes each piece.

Every active `(request, KV head group, query tile)` must cover its valid KV
interval exactly once. Split groups cover disjoint intervals, and their
FP32 online-softmax statistics and weighted outputs are merged correctly.
An output is consumed only after all its required pieces finish. Inactive
requests cannot mutate canonical state; recycled IDs/pages cannot be reached
through an old generation. Sliding windows and variable query widths change
the contract and need explicit invalidation, rather than only a length check.

Only descriptor topology is reused. Attention scores, softmax outputs and
previous-token computation are not reused. Contents of a changed tail, masks
and page mappings remain authoritative on every execution.

## Maintenance and selective splitting algorithm

Start with a deterministic algorithm that is simple enough to audit:

1. Refresh exact arrays and current row-to-request mapping. Scan O(B) rows;
   this work is unavoidable and must remain in the endpoint.
2. Compare each live request's structural signature with its stored summary.
   Update changed tails, query geometry and KV-range counts. Handle admission,
   pause, resumption and lifetime recycling explicitly.
3. Reconstruct the current row-ordered workload prefix from the summaries.
   Reuse feasible prior boundaries where possible. A permutation or removal
   can move every dense-row endpoint; reuse is conditional, not automatic.
4. Estimate each core's load including launch, page access and split-merge
   costs. Publish only plans with exact work coverage and valid merge owners.
5. Repair overloaded boundaries within a fixed work budget. Split a long KV
   interval only when the modeled reduction in the critical core's tail
   exceeds scheduling, additional KV traffic and reduction costs by a margin.
   Keep unsplit work where it is already balanced.
6. Rebuild the assignment when the repair budget, topology capacity or
   imbalance threshold is exceeded. This is a normal bounded path inside the
   same device maintenance kernel; unsupported conditional ACLGraph nodes are
   unnecessary. Correct full coverage is mandatory even when the cost model
   is wrong.

For estimated per-core loads `w_i`, define `W = sum(w_i)`, core count `C` and
largest indivisible remaining task `w_max`. A useful diagnostic is

```text
L_lower = max(W / C, w_max)
rho = max_i(w_i) / L_lower
```

`rho` bounds imbalance only under this cost model; it is not a runtime speed
guarantee. Rebuild or locally split when a chosen threshold is exceeded.
Calibrate the cost model on separate data and hold it fixed for final tests.
Do not choose thresholds after inspecting the final trajectory's speedup.

The split decision compares the estimated complete critical path before and
after replacing one task by a bounded number of KV subranges. It cannot use
`task_time / split_count` without accounting for Cube occupancy, page reads,
workspace and softmax merge. Do not create one output partial per KV page.
Use capped split groups and a preallocated maximum scratch bound.

## Two implementation levels

**Level 1: preserve the native sink arithmetic kernel.** Its metadata contains
current dense-row core endpoints (`BN2`, `GS1`, `S2`), split-reduction fields
and base/core counts. Audit that ABI and emit it from our own planner. Maintain
per-request summaries, rebuild current routing/prefixes and derive legal core
boundaries. All endpoints and merge relationships must be regenerated when
required. Matching length buckets does not justify replaying an old 4 KiB
metadata buffer.

This level can compare full device replanning against incremental planning
without changing attention arithmetic. The A2 sink must first qualify on
the actual Q4/D256/GQA6/page128 target, inactive rows, capture-to-replay
workspace growth and the installed driver. See the [research plan][research].

**Level 2: persistent request-task descriptors.** If dense-row rebuilding or
native boundary restrictions dominate, add request/row indirection to the
attention consumer and let stable task descriptors survive permutations.
Use a small repair/overflow path for new or overloaded ranges. This requires
a real kernel change; the existing sink cannot consume an arbitrary task
queue. Account for descriptor reads, indirection, synchronization and reduction
traffic. Proceed only when Level 1 identifies a bottleneck worth that cost.

Neither level duplicates QKV/O/MLP weights or changes TP ownership. Neither
requires splitting dense GEMMs into inefficient small batches.

## How request splitting could be combined later

The primary split is a KV work interval within attention, not a full-model
request branch. This preserves full-M GateUp/Down and avoids the previously
measured small-GEMM penalty. It also differs from the old persistent full-head
request owner route, which the user stopped.

If independent request groups can later overlap attention with communication,
use the same plan to expose readiness groups. Add that only after measuring
an additional complete-boundary saving against the same shared planner and
attention algorithm. It is a separate experiment, not a requirement for the
thesis's core algorithm or a way to relabel a public kernel improvement as a
request-split win.

## Minimal experiments and stop rules

1. **CPU algorithm oracle.** On unchanged traces, compare full and incremental
   plans for exact interval coverage, valid reductions, generations and
   inactive semantics. Sweep membership churn separately from length drift.
   Include adversarial permutations, all-inactive rows and full-rebuild paths.
2. **Budget on the actual A2 sink.** Measure one shared plan's exposed build
   cost and the attention core tail. Sharing already amortizes across 16
   layers; planner savings cannot be multiplied by 16. Native ABI eligibility
   and workspace safety precede performance claims.
3. **One real attention occurrence.** Compare full AICPU planning; full device
   planning; incremental device planning with splitting disabled; incremental
   planning with local repair/splitting; and the same kernel forced to full
   rebuild. If a persistent task kernel is implemented, add a nonincremental
   persistent scheduler using the identical arithmetic. Include every planner,
   copy, descriptor read, reduction and final wait.
4. **Complete model only after a positive occurrence result.** Audit continued
   hidden/residual and all physical states, then time the unchanged dynamic
   trajectory with default buckets. Report P50/P95/P99, peak memory, repaired
   tasks, rebuild frequency, tail time and maintenance time.

Separate planning savings from kernel execution savings. Against shared
planning, a complete-step budget is the exposed reduction of one planner
plus the sum of actual attention execution reductions, minus all added work.
The observed 143.474 ms model mean needs about 7.174 ms net to improve 5%.
If planning is only tens of microseconds, this proposal needs real kernel-tail
improvement for a large gain. Do not run a large model matrix to hide a missing
budget. If both planner cost and residual imbalance are small, reject the
performance hypothesis or report a narrower result; a useful implementation
alone does not prove the desired speedup.

Use distinct reporting endpoints: current single-step replay, planning plus
replay, and optional asynchronous serving. Removing a host acceptance wait
from serving cannot be credited to a replay trace that never executes it.

## Suggested thesis framing

Working title: **Incremental Attention Execution Planning and Selective KV
Splitting for Dynamic-Shape ACLGraph Inference**.

The thesis can contribute an exact-semantics/assignment separation, a
membership-aware repair algorithm with bounded scratch and fallback, an A2
implementation, and causal comparisons with strong full-planning controls.
Whether it is sufficiently original for the degree still requires related-work
review and advisor judgment. Current evidence supports investigating this
specific hypothesis; it does not establish novelty priority or a speedup.

[sink]: https://github.com/vllm-project/vllm-ascend/pull/15336
[flashinfer]: https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html
[cake]: https://github.com/flashinfer-ai/flashinfer/blob/759163ee4787a920c3c83998ecf6d1b61f327c3a/flashinfer/experimental/balanced_gqa_decode/README.md
[flex]: https://pytorch.org/blog/flexattention-for-inference/
[trt]: https://nvidia.github.io/TensorRT-LLM/advanced/gpt-attention.html
[research]: aclgraph_dynamic_shape_research.md
