# Dynamic-shape ACLGraph: literature and implementation plan

Research date: 2026-10-18. Sources were read from original papers, conference
proceedings, author repositories and official implementation discussions.
This is a research and qualification plan, not a newly measured speedup.

## Scope and decision

The user stopped the request-owner architecture because its replicated
attention weights add unacceptable memory and traffic. Do not continue that
route. Keep the original TP2 weight shards; do not add replicas of attention
or MLP weights. The user also broadened the investigation: an improvement to
dynamic-shape ACLGraph is useful even if it does not require request splitting.

The leading candidate is **fixed graph addresses with a per-step attention
execution plan, shared across compatible layers**. The immediate engineering
route is to qualify the existing A2/A3 FIA v2 sink/device-metadata proposal,
rather than rewrite attention from scratch. Plan sharing is already part of
that upstream proposal; adopting it for our geometry is compatibility and
integration work, not a new scheduling algorithm. The second candidate is
tile-granular TP compute/communication overlap with unchanged TP weights.
Its larger engineering cost is justified only by exposed communication time.

Neither candidate is guaranteed to achieve 5%. A reliable plan needs verified
operator contracts, a measured critical-path budget, matched causal controls,
and a stopping rule before expensive integration.

For the user's graduation-project requirement, upstream adoption is a
baseline rather than the independent contribution. The separate
[thesis proposal](aclgraph_incremental_attention_thesis.md) develops
membership-aware incremental planning and selective KV-range splitting,
with explicit related-work boundaries and experiments to falsify it.

The [stage-one native qualification](aclgraph_fia_sink_stage1.md) now has a
specific result: isolated native build and D128 constant eager execution
pass, but the actual Q4/D256/GQA6 target is rejected by the pinned host route.
Its K L1 allocation would also be insufficient if that check were bypassed.
Target sink integration therefore requires a kernel/tiling adaptation before
dynamic correctness or performance tests; a generic parser branch is not
evidence that the target is supported. This finding does not establish a
driver incompatibility or a scheduling speedup.

The subsequent [stage-two adaptation](aclgraph_fia_sink_stage2.md) resolves
that narrow geometry in an isolated prototype using capacity-bounded inner
tiles. The actual native build, 25 eager correctness gates and three small
dynamic/shared-plan graph gates pass. The unmodified upstream source remains
unsupported, and the prototype has no new performance or full-model result.
The next decision requires a measured exposed planner/kernel-tail budget.

## Existing evidence that constrains the search

- The current public ready-tile full-model control averages 143.474 ms over
  three trials of the fixed 128-step trajectory. Its 4.586% gain over delayed
  consumption belongs to padding as well. Three continued steps per mode,
  rather than all 128, have full correctness audits.
- Five percent of that observed mean is 7173.71 us. Eliminating host work
  cannot save that amount unless it is actually exposed on the critical path.
  Adding durations from overlapping threads/streams is not a valid budget.
- Ordinary split GateUp has a measured 29-33% aggregate GEMM penalty. A real
  GateUp/core concurrent tail remains 7.9-8.6% slower than whole-bucket work.
  More streams or a paper's resource labels do not invalidate those findings.
- Fixed-trajectory capacities 256..512 round live Q4 rows to multiples of 16.
  Active-row clipping removes no minimum 16-row Cube output tiles. It does
  not currently support a large arithmetic-saving hypothesis.
- Conditional graph creation returned unsupported on the installed driver.
  Header declarations and newer CUDA conditional nodes do not establish local
  support. No driver upgrade or reboot is part of this plan.
- The manual full-model probe is not a complete audit of every official
  compiler/static/super-kernel configuration. Qualify applicable public
  optimizations before naming a new candidate the fastest padding path.

## What the literature establishes

| Work | Verified mechanism | Transferable lesson and limit |
| --- | --- | --- |
| [FlashInfer, MLSys 2025][flashinfer] | Dynamic lengths/work assignment are planned separately from specialized kernels. Fixed-grid persistent attention and contraction consume a plan in fixed workspace sections, reusable across layers. | Direct architectural match. Its scheduler runs on CPU; it does not establish a host-free planner or unrestricted dynamic batch in one graph. See section 3 and Appendix D. |
| [FlashInfer current attention API][flashinfer-api] | `plan()` prepares reusable auxiliary structures; `run()` executes. Graph mode has fixed descriptors/buffers; plan is outside capture. | Share legal plans across layers. Cached workspace does not imply a reusable pointer-bound executor. |
| [POD-Attention, ASPLOS 2025][pod] | SM-aware scheduling in one kernel enables complementary prefill/decode work; naive streams and fusion are insufficient. | Supports explicit scheduling, but its mixed workload and NVIDIA SM/CTA contracts do not apply directly to all-Q4 A2 execution. |
| [NanoFlow, OSDI 2025][nanoflow] | Nano-batch pipelines use measured interference/resource allocation and asynchronous CPU batch formation. | Learn its cost model, not just streams. It acknowledges added weight traffic. The paper is OSDI 2025; 2024 is its arXiv draft date. |
| [Sarathi-Serve, OSDI 2024][sarathi] | Chunked prefill and stall-free scheduling stabilize iteration work and decode latency. | Relevant to a separate mixed serving experiment. vLLM already supports chunked prefill; it is not a new current-Q4 optimization. |
| [Flux, 2024][flux] | AG/GEMM and GEMM/RS overlap use GEMM prologue/epilogue and coordinated tile production. | Preserve efficient GEMM rather than many small launches. CUDA/CUTLASS/remote I/O and reported small-M regressions prevent a direct 910B claim. |
| [TileLink, MLSys 2025][tilelink] | Compute/communication tiles are decoupled, with release/acquire notify/wait and compiler-visible dependencies. | Good protocol design. Inter-layer overlap is outside its scope; raw flags alone are insufficient. |
| [Triton-distributed, 2025][triton-dist] | Symmetric memory, signal exchange and dependent loads enable programmable overlap. | Ascend work exists on `triton-v3.4`; current main's 3.7 plugin is not an Ascend deployment. SHMEM/compiler/graph compatibility need qualification. |
| [CoCoNet, ASPLOS 2022][coconet] | Collective stages and local work are reordered, fused and overlapped at tile boundaries. | Reconsider synchronization boundaries. Modified CUDA/NCCL protocols are not installed HCCL APIs. |
| [PyTorch CUDAGraph Trees][trees] | Pools/liveness are managed across captured paths; new unique shapes can require new recordings. | Shared pools already address part of this. Trees do not make arbitrary dynamic replay automatic. |
| [vLLM CUDA Graph design][vllm-graphs] | FULL/PIECEWISE dispatch has explicit batch descriptors, separate from compilation. | Preserve dispatcher contracts. Changing graph mode alone does not prove a new algorithm or speedup. |

SGLang's [original report][sglang] primarily establishes RadixAttention and
prefix-aware execution. The fixed random trace does not supply additional
shared prefixes, so prefix reuse is not included in the present gain budget.

A recent [FlashInfer experimental balanced GQA implementation][cake] provides
a device-planning example: device lengths drive long-request splitting and
short-request packing with multi-token causal verification. The inspected
path requires SM100/103, BF16, D128, GQA8, HND and page16. It is evidence of a
design direction, not an implementation for A2/D256/GQA6/ND/page128.

## Current Ascend work changes the feasibility assessment

Official [PR 15336][fia-sink], inspected at
`bec19b8dea1e538e64838f3064f7e91a3fa9ee72`, is open and unmerged at this
snapshot. Its routes explicitly distinguish A2/A3 FIA v2 sink from A5
FlashAttention/FlashMLA. This is more specific than earlier discussion of A5
or MLA-only device-length support.

Source facts at that pinned head:

- `vllm_ascend/ops/fia_v2_sink.py` passes device query/KV lengths to an AICPU
  metadata builder, then passes the metadata into the attention kernel.
- The metadata output is 1024 int32 entries, or 4 KiB. Shared plan storage
  need not grow with model weights. Attention scratch is separate and still
  requires a peak-memory audit.
- Native registration includes `ascend910b` as well as A3/A5 SoCs. This is
  source eligibility, not runtime qualification on our driver.
- Metadata reads current lengths and derives tiling. Planning moves to a
  device-side CPU; it does not disappear. Its launches, clones/copies,
  synchronization and scratch must be timed.
- The builder owns stable metadata storage. Eligible forward calls consume
  device lengths instead of Python host-length lists. One builder owns one
  metadata buffer for its compatible layer group, so upstream already
  amortizes planning across those layers.
- The submitted tests cover paged BF16/FP16 GQA and a 16-occurrence graph
  with changing lengths, but the inspected GQA case uses D128 and eight
  query tokens per request. Those 16 calls use the same Q/K/V and check the
  final output; they do not qualify 16 distinct layers with private caches.
  Reported A3 tests/A5 validation do not establish our exact A2/Q4/D256/GQA6
  geometry or old-driver compatibility.

[RFC 17479][host-contract] distinguishes device-exact, host-upper-bound and
host-exact lengths. Its [MRV2 proposal 17689][mrv2-contract] still classifies
legacy FIA as EXACT. One exact consumer in a hybrid model keeps the host
dependency. **Never skip update on the current FIA.** Replace it only after
a new backend actually consumes authoritative device data.

[PR 18376][physical-pages] separately documents full-cache temporaries from
padded FIA page strides on CANN 9.1 and proposes storage-sharing physical-page
views. Our benchmark already uses independent dense ND K/V as a software
workaround. Check for equivalent production copies, but do not count another
saving where that copy is already absent.

## Candidate A: one dynamic attention plan per step

Keep dense projections, TP partitioning and default graph buckets. For each
compatible attention geometry, use stable lengths, page-table and plan-buffer
addresses. Update contents for the current generation:

```text
current exact lengths / request map / pages
    -> one metadata plan per compatible geometry
    -> explicit plan-ready dependency
    -> 16 layer-local attention kernels consume the plan
       with their own Q/K/V/output and unchanged TP heads
```

Compatibility includes dtype, layout, Q/K/V dimensions, GQA ratio, causal or
window mode, page size, core allocation and kernel strategy. Equal batch size
alone is insufficient. Plans must not embed another layer's Q/K/V addresses.
Different local cache contents remain layer-owned.

Retain the default bucket dispatcher for dense work. The graph tensor shapes
and addresses remain those of the selected legal bucket; live lengths and
work assignments change through buffer contents. This does not make one
captured GEMM accept arbitrary tensor shapes or remove its padded Cube tiles.

Plan publication needs an explicit generation/lifetime contract. No producer
may overwrite a plan or its length/page inputs until all consumers of that
generation have finished. Ordered streams can enforce this initially;
multiple in-flight generations need separately owned slots and compatible
captured addresses. A cached plan is not reusable after lengths, mode or
geometry change without proving its validity. The 4 KiB metadata size does
not include length clones, plan-build temporaries or attention workspace.

The intended saving is repeated host tiling/task patching and its exposed
device waits. Device-driven work assignment may also reduce tail imbalance.
These are separate claims and need separate controls.

### Ordered qualification

1. Profile existing complete replay separately from formal timing. Inventory
   CPU preparation, update issue, external-event device waits, FIA kernels and
   communication. Do not move required work outside the endpoint or infer
   idle time from a synchronization that an earlier event already drained.
2. Isolate the pinned A2 sink/metadata dependencies in a fresh build location.
   Do not merge unrelated A5, DSpark or model changes. Check compiler/link
   symbols, AICPU artifacts, actual A2 kernel-template eligibility and execution
   on current CANN 9.1/driver. Parser acceptance of D256/GQA6 is insufficient.
   Establish a workspace bound for the longest permitted replay context;
   `GetMaxWorkspaceSize` forwarding to `GetWorkspaceSize` is not itself proof
   of that bound. Preserve an explicit old-FIA fallback.
3. Qualify actual TP2 geometry: BF16, Q4, 12 Q heads, two KV heads, D256,
   GQA6, dense ND page128, true causal verification and original cache
   semantics. Vary lengths/page IDs and shuffled owners; poison unused pages;
   test inactive/null rows and continued replay at fixed addresses.
   Our Q4 trace uses zero KV length for inactive rows; the upstream builder
   pads those lengths to one. Qualify zero lengths first, or explicitly guard
   dummy reads and outputs while preserving the existing physical-cache and
   inactive-state invariants. Keep cumulative query lengths distinct from
   per-request KV lengths.
4. Compare updated legacy FIA; sink with per-layer plan builds; and sink
   with one shared plan across 16 compatible occurrences. Count all planning,
   copies, workspace and final waits. Separate kernel replacement from plan
   amortization. Same plan and per-layer private Q/K/V must remain safe.
5. Integrate the real complete model only if kernel efficiency is competitive
   and the net exposed budget justifies it. Retain all 16 attention calls;
   fewer update tasks must never mean missing layers. Keep identical GDN
   state and commit algorithms on both sides.
6. Audit all 128 continued steps and repeat paired timing, tails and peak
   memory. Removing a real sampler's D2H dependency needs a separate serving
   test: this predeclared CPU trace cannot prove elimination of a rejection
   wait that its benchmark never executes.

Let `E` be the union of exposed avoidable update waits, `K` the attention
kernel improvement and `P` added plan/synchronization cost. The optimistic
budget is `E + K - P`, counting overlap once. If it cannot reach 7173.71 us
on this baseline, do not promise 5% from this candidate alone. Retain any
smaller useful result at its actual scope, or investigate Candidate B.

### Extension A2: persistent work scheduling inside attention

If the sink path is compatible but skewed KV lengths still leave a measured
core tail, extend the planner/kernel boundary rather than split the entire
model. A plan can describe `(request, KV-head group, query tile, KV interval)`
work; a fixed-size worker grid consumes that work. Long contexts create more
intervals while short contexts pack into the available grid. Partial softmax
states need a mathematically correct merge and bounded scratch.

FlashInfer's fixed-grid plan/run design and its newer balanced GQA example
motivate this direction. The upstream sink already performs length-dependent
core splitting, so an extra queue is justified only by residual imbalance
that the existing split policy cannot resolve. It can also lose through
queue contention, KV reloads or extra partial-state reduction. Compare the
same attention arithmetic with existing tiling, a preassigned balanced plan
and dynamic assignment before attributing a win to persistent scheduling.
This is a larger A2 kernel project, not a configuration switch or a promise
of a speedup on the current Q4 trace.

### Serving extension: keep speculative length correction on device

After every consumer has a valid device-exact contract, accepted-token counts
can update device lengths, slots and positions before the next plan. That
can remove a host wait needed solely to correct attention lengths. The host
still has scheduling, admission, output and KV-allocation responsibilities;
device planning alone does not remove all of those dependencies. Any GDN or
other host-exact consumer must be audited as well. This extension needs a
real asynchronous serving endpoint and cannot be credited to the current
predeclared replay trace.

## Candidate B: tile-ready TP without weight replication

Use the Flux/TileLink/CoCoNet principle: keep original TP shards and efficient
full-M GEMM, but let completed output tiles begin communication before the
last GEMM tile finishes. Start with Down/AR as a closed boundary. O/AR and
downstream early consumption follow only after a causal improvement.

Ordinary HCCL does not publish arbitrary internal tile readiness. First test
supported fused matmul/collective implementations on this installed stack.
If needed, qualify a bounded AscendC HCCL Kernel or SHMEM protocol before
building a model scheduler. Current A2 Prepare/Commit/Query/Wait and newer
A5 APIs differ. Neither an exported symbol nor an author example proves
current-driver graph support.

Controls are: best library GEMM plus HCCL; same custom GEMM plus whole-tensor
communication; identical tile communication delayed until every output is
ready; early publication. Keep final replicated outputs, BF16 arithmetic,
memory ordering and complete cleanup identical. Measure first-tile readiness,
total GEMM slowdown, communication tail, launch cost and peak buffers. One
launch alone does not prove preserved weight reuse.

Stop if memory traffic or fragmentation consumes the communication budget.
Count the union of communication-only intervals from the current trace, not
128 times an old isolated AR duration. Give generic fusion to padding.
Request splitting is optional and needs an additional causal win before any
cross-layer request scheduler is built.

## Candidate C: generation-owned graph control for asynchronous serving

The current production `ACLGraphWrapper` synchronizes the host's current
stream before ordinary full replay to protect graph update/event reuse.
Independent capture-set rotation already exists in the experimental async
executor, but is not integrated into the production full-model path.

A bounded ring of graph instances can own separate task handles, readiness
events, length/plan storage and input generations, allowing host submission
without reusing a still-live graph generation. Weights remain shared, and
canonical KV/GDN mutations must remain ordered. Shared activation pools are
safe only with proven non-overlapping liveness; additional pools/workspaces
need an actual memory bound. Replacing a host synchronize with a device event
without establishing that ownership is unsafe.

This targets exposed host-fence/submission bubbles in asynchronous serving.
It does not shorten the device arithmetic of one replay or remove a host
dependency on speculative acceptance by itself. Measure the actual bubble
before full-model integration. Pair identical graph instances with a host
fence versus generation rotation to isolate that saving; retain identical
state transitions, legal buckets and memory accounting.

## Other directions and exclusions

- Length-aware persistent attention is a later extension of A. Compare the
  same arithmetic with fixed versus dynamic assignment. Do not revive the
  slow Q1 Vector reference as a high-performance Q4/D256 backend.
- Mixed prefill/decode graph and token-budget scheduling follows Sarathi/POD,
  but needs a separately declared serving workload. It cannot replace the
  current fixed Q4 replay acceptance with an easier measurement.
- Qualify static/super-kernel compilation as a public baseline where supported.
  Configuration eligibility does not prove old-driver support for every op
  or compatibility with the manual capture path.
- Exclude attention-weight replication, large candidate-state materialization,
  unsupported conditional graphs and capture sizes picked to favor one batch.

## Deliverables and next action

First produce a current-baseline critical-path inventory and an isolated A2
sink compatibility result. Those facts decide whether A merits integration.
In parallel, inventory native TP fusion/SHMEM contracts for B without replacing
the runtime or launching another large model matrix. Keep raw results, pinned
hashes, rejection reasons and implementation on the feature branch.

Source snapshots/acquisition records are in the parent workspace's
`outputs/literature/`; the research manifest records their hashes. Third-party
speedups are not predictions for this machine. This document establishes no
new performance acceptance or completion of the original structural goal.

[flashinfer]: https://proceedings.mlsys.org/paper_files/paper/2025/hash/dbf02b21d77409a2db30e56866a8ab3a-Abstract-Conference.html
[flashinfer-api]: https://docs.flashinfer.ai/api/attention.html
[pod]: https://arxiv.org/html/2410.18038v2
[nanoflow]: https://www.usenix.org/conference/osdi25/presentation/zhu-kan
[sarathi]: https://www.usenix.org/conference/osdi24/presentation/agrawal
[flux]: https://arxiv.org/html/2406.06858v1
[tilelink]: https://arxiv.org/html/2503.20313v1
[triton-dist]: https://arxiv.org/html/2504.19442
[coconet]: https://www.microsoft.com/en-us/research/publication/breaking-the-computation-and-communication-abstraction-barrier-in-distributed-machine-learning-workloads/
[trees]: https://docs.pytorch.org/docs/2.9/torch.compiler_cudagraph_trees.html
[vllm-graphs]: https://docs.vllm.ai/en/latest/design/cuda_graphs/
[sglang]: https://arxiv.org/html/2312.07104v2
[cake]: https://github.com/flashinfer-ai/flashinfer/blob/759163ee4787a920c3c83998ecf6d1b61f327c3a/flashinfer/experimental/balanced_gqa_decode/README.md
[fia-sink]: https://github.com/vllm-project/vllm-ascend/pull/15336
[host-contract]: https://github.com/vllm-project/vllm-ascend/issues/17479
[mrv2-contract]: https://github.com/vllm-project/vllm-ascend/pull/17689
[physical-pages]: https://github.com/vllm-project/vllm-ascend/pull/18376
