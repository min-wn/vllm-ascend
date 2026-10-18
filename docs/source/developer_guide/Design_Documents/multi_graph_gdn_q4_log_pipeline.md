# FP32 Q4 GDN logs and accepted-state commit pipeline

## Decision and scope

The next structural experiment should reduce dense state work and writes,
rather than add streams around the current write-heavy Q4 operator. This
document proposes retaining an FP32 base and four ordered update records,
computing the next attention products through that representation, and
committing only the previous accepted prefix to the base. It is a research
design with a CPU mathematical oracle and qualified isolated native
recurrence, not an integrated model runtime backend.

The first gate is a closed recurrence endpoint with the real TP2-local
layout, all required state work, and continued dynamic request histories.
Only a successful result can advance to connected real layers and complete
replay. The main workload remains Qwen3.5-27B, TP2, Q4, default capture sizes
and the fixed random dynamic trajectory recorded in
`multi_graph_gdn_compact_state.md`. Drivers and host restart requirements do
not change.

This representation is available to padding and split execution equally.
Removing four dense candidate writes is a common operator improvement. Any
additional benefit attributed to request splitting must be measured against
padding using the same representation, accepted-state semantics and public
operator improvements.

## Evidence for changing the approach

The real model uses FP32 SSM state `[401,24,128,128]`, with element strides
`[458752,16384,128,1]`. Earlier BF16-state results cannot establish its
performance. The FP32 direct-products qualification passes 17 checks,
including repeated eager/replay, continued states and inactive/null handling.
It preserves FP32 state and explicitly disables HF32 in the native Cube
operator.

The completed `gdn-q4-fp32-profile-v2.json` reports the following device
times in microseconds:

| Scope | Statistic | Time |
| --- | --- | ---: |
| Installed complete recurrence | Median of three round medians | 1001.30 |
| Direct-products complete recurrence | Median of three round medians | 943.26 |
| Products with preconverted Q/K | Median of component samples | 252.31 |
| Products including Q/K casts | Median of component samples | 252.64 |
| Q/K casts alone | Median of component samples | 41.38 |
| AIV solve and four candidate materializations | Median of component samples | 662.26 |

Component profiles restore matching inputs/state outside each diagnostic
capture/replay. They are isolated measurements and are not additive. In
particular, the cast-only measurement cannot be added to products with
preconverted Q/K to predict the complete product endpoint. The complete
recurrence saves only 58.04 us in this run, below the previous 135 us
investment gate for a roughly 2709 us GDN layer.

The report SHA256 is
`b6f4e428bc53b91b52602d51c0a1563d9622f5b085877de5a74ffbc428a7b83b`.
Raw JSON/log files are retained under
`outputs/multi_graph_tp_attention_mlp/` and the remote
`/workspace/multi-graph-results/tp-boundary/` directory. The experiment is a
single-device recurrence using local TP2 dimensions, not TP2 model latency.

At 98 active requests, one FP32 matrix set contains 154,140,672 bytes.
The AIV endpoint logically reads one set and writes four, or 770,703,360
bytes. Dividing by its isolated 662.26 us gives approximately 1.16 TB/s of
logical state bytes. This is not a measured HBM bandwidth counter and does
not prove exclusive bandwidth saturation. It does identify a large required
data volume which a parallel schedule cannot remove.

For two equal request chunks, the ideal recurrence pipeline is
`P_A -> (S_A || P_B) -> S_B`. Relative to the same split serial arithmetic,
its saving cannot exceed `min(S_A, P_B)`, approximately 126 us if product
cost scales linearly with half the requests. Four equal chunks could hide at
most about three quarters of full product time, approximately 189 us, before
extra launches, events, fragmentation and memory contention. Neither estimate
is a measured endpoint or sufficient evidence for the goal's complete replay
gate. The plan therefore prioritizes deleting work and writes.

The earlier GateUp/core experiment also limits the scheduling argument.
It found real local overlap, but the closed tail lost existing AR/core
overlap, delayed the B core and paid approximately 113-118 us of GateUp
fragmentation. This design keeps expensive dense projections and GateUp
whole where possible and does not assume another local overlap will survive
the complete endpoint. See `multi_graph_gdn_compute_overlap.md`.

## Representation and canonical logical states

For one stable request and value head, store:

- An FP32 base matrix `B` of shape `[V,K]`.
- Four FP32 decay scalars `gamma[i]`, the actual operands used by the update.
- Four keys `k[i]` and four update vectors `u[i]`, stored with sufficient
  precision to reproduce the chosen update operands.
- A valid flag, generation identity and state ownership metadata.

The proposed initial implementation retains FP32 keys/updates and padded
eight-float gamma records, matching the conservative earlier log budget.
Any compression is a separate numerical and cost experiment.

Define logical candidate reconstruction through a prescribed ordered FP32
procedure, rather than an expanded polynomial:

```text
H[-1] = B
for i = 0 .. prefix-1:
    D    = round_fp32(gamma[i] * H[i-1])
    H[i] = fma_fp32(u[i], transpose(k[i]), D)
```

The matrix update uses the selected device multiplication/FMA behavior in a
fixed order. Exporting prefix `i` must reconstruct exactly that logical
candidate for this algorithm, using the stored operands and the same ordered
procedure. Export and accepted-prefix commit must agree bitwise with that
definition within a qualified device implementation. No claim is made that
the CPU FMA model certifies device instruction behavior.

The four candidates continue to exist logically. The proposal replaces
their physical representation; it does not drop speculative candidates or
permit consumers to observe an incorrect dense pointer. A cache API must
identify compact state explicitly and provide selected-prefix and all-prefix
materialization when required.

## Projecting a pending accepted prefix

Let the current persistent representation describe the previous step's four
candidates relative to `B`. The current step provides an accepted count
`a` for that previous generation. In real arithmetic, define:

```text
alpha[a] = product(gamma[t], t = 0 .. a-1)
d[a,j]   = product(gamma[t], t = j+1 .. a-1)
R        = [k_new[0..3], q_new[0..3]]

H_selected = alpha[a] * B
             + sum(d[a,j] * outer(u_old[j], k_old[j]), j = 0 .. a-1)

H_selected @ R = alpha[a] * (B @ R)
                 + sum(d[a,j] * u_old[j] * (transpose(k_old[j]) @ R),
                       j = 0 .. a-1)
```

Compute `B @ R` on Cube. The correction contains small key/R Gram products
and vector operations, with no `[V,K]` intermediate. Do not divide by
`alpha` or by any decay; zero and underflowed decays remain defined. Query
scale follows the current raw-Q product convention and is applied at the
FP32 consumer, without an additional BF16 rounding of scaled queries.

Use these eight approximate selected-state products in the existing Q4
causal solve to obtain current `u_new[0..3]` and attention outputs. This path
does not need the selected dense matrix before solving. Separately, commit
the previous accepted prefix to `B` with the ordered reconstruction above.
After commit, retain `B = H_selected` and the four new update records.

### FP32 is not exact arithmetic

The displayed projection identity is exact over real numbers. It is not
bitwise equal in FP32 to first constructing the ordered dense candidate and
then multiplying it by `R`. FP32 multiplication, FMA, cumulative products,
reductions and the rearranged additions all round. Even without a BF16
storage boundary, their reassociation can change products and, through the
causal solve, current updates and later state trajectories.

FP32 state removes an additional obstacle present in the earlier BF16
design: its next-step elementwise BF16 rounding is nonlinear and cannot be
moved through a matrix product by this formula. Removing that quantizer does
not remove FP32 rounding. The proposed projection must be treated as a new
floating-point algorithm, with its own numerical qualification.

There are three distinct comparisons:

1. Same schedule, inputs and state: eager versus replay must be bitwise equal.
2. Stored logs versus ordered materialization within the new algorithm:
   candidate export and accepted commit must agree exactly.
3. Reassociated products/outputs/state versus dense-projection or installed
   recurrence controls: assess numerical drift separately, including long
   accepted-state feedback. Exactness is not assumed or obtained by relaxing
   the first two checks.

## Accepted-state lifecycle and dependency graph

Acceptance belongs to a request generation, not to its current batch
position. A representation at generation `n` contains base `B_n` and logs
for candidates of that generation. The next active use consumes the
accepted count for generation `n` once, obtains the selected state and
publishes generation `n+1` only after its state transaction finishes.

The required ordering is:

```text
old base + old logs + accepted count + new Q/K
                       |
                 products/correction
                       |
                   new Q4 solve ---------> attention/output ready
                       |
       old accepted commit and new-log publication
                       |
                 state transaction done
```

This sketch shows logical dependencies, not a promise that two AIV-heavy
operations run concurrently. The accepted commit can start after Cube has
finished reading its old base, provided old logs remain available and
required product corrections have consumed them. Output solve consumes
products and small operands; it must not read a base being overwritten.

The implementation must enforce the following lifecycle rules:

- On first use, initialize/import the supplied real FP32 starting state and
  mark old logs invalid. Do not apply an accepted count to nonexistent logs.
- For a paused request, retain its base, logs, generation and pending accepted
  decision unchanged. On resume, consume the pending decision exactly once.
- Moving a request to another batch position changes its lookup, not state
  ownership. Reject duplicate ownership in a simultaneous replay.
- Retain old logs until both product correction and accepted commit have
  consumed them. New-log writes must not race either reader.
- Publish the new base/log pair atomically in the logical state machine.
  A failure or fallback must not expose a committed base with old logs as a
  valid new generation, or apply the accepted prefix a second time.
- Inactive rows, null state, padding and page gaps remain unchanged. Device
  counts and stable IDs determine live work at fixed captured addresses.
- The complete replay endpoint waits for both hidden/residual output and
  every required state transaction. Starting a next consumer also requires
  its request-specific state-ready dependency.

A two-bank log design is the simplest initial ownership model: read old logs
and write new logs in different banks, then publish the generation/bank.
A single-bank design is permitted only after proving that all old operands
are retained before overwrite, including parallel kernel readers. Its lower
storage cost must not be reported if the tested implementation uses two banks.

Convolution state retains the official speculative update behavior. This
proposal changes SSM representation, not convolution cache semantics. The
current model binds convolution and SSM views into shared pages; changing the
SSM allocation requires explicit cache-manager integration and must preserve
the convolution portion and all its speculative slots.

## Required consumers and materialization costs

A compact recurrence cannot silently masquerade as the existing dense
`ssm_state` tensor. Integration must enumerate consumers and define a legal
transition for each one:

| Consumer or transition | Required operation and timing |
| --- | --- |
| Next Q4 replay | Project pending accepted prefix, commit it once, then publish new logs; all are timed |
| Single accepted-state consumer | Reconstruct the requested prefix or use an explicitly supported compact interface; count required work |
| Export, checkpoint, connector transfer or eviction | Produce the required selected/all-prefix state and wait for completion in that consumer's endpoint |
| Non-spec decode, prefill, mixed path or unsupported operator fallback | Materialize the exact required state, run the real consumer and reseed/invalidate logs under an explicit protocol |
| Switching between padding and split | Share the same compact state backend; do not add hidden per-step migration or allow divergent state histories |
| Request reuse or reassignment | Finish old consumers, clear/reinitialize ownership and generation, then import the new initial state |

If a hot consumer requires all four dense candidates after each replay,
materialization must remain in that replay timer. It restores the four matrix
updates and writes that this design seeks to avoid and can invalidate the
route. Separate export during correctness testing does not prove an
export-inclusive performance benefit. Export frequency and bytes must be
reported for the actual tested endpoint.

If dense fallback is needed, it cannot simply call the installed recurrence
against a base/log pointer. The adapter must provide a correct dense state,
account for materialization, and record how the result returns to the compact
representation. Connector waits and layer-load dependencies must preserve
the official order before any cache reader or writer starts.

## Work, byte and latency budget

For B98, 24 local value heads and 128-by-128 FP32 matrices, let
`D = 154,140,672` bytes. The following are logical state-data counts, not
measured memory transactions:

| Endpoint | Matrix-set traffic, excluding logs and common inputs |
| --- | ---: |
| Installed recurrence: one selected read, four candidate writes | `5D = 770,703,360` bytes |
| Current direct-products block: product read, materialization read and four writes | `6D = 924,844,032` bytes |
| Proposed projection plus accepted-base commit: product read, commit read and one write | `3D = 462,422,016` bytes |

With the conservative padded records, one active B98 log bank contains
9,934,848 bytes. Reading an old bank and writing a new bank adds about
19,869,696 bytes, before additional metadata/scratch traffic. The resulting
`3D + 2 logs = 482,291,712` bytes is a model budget, approximately 47.85%
less than the current direct block and 37.42% less than the installed dense
recurrence. Actual cache behavior, duplicated operands, padding, allocation
and export can change the physical traffic.

More importantly, the dense current Q4 materializer executes four full-matrix
decay/FMA updates. The earlier compact V2 reconstructed `a` previous updates
and still executed four current full-matrix updates, for `a+4` such updates.
The proposed block/log algorithm executes only the `a` previous updates
needed to commit the selected base; current outputs and new update vectors
come from skinny products and the small solve. With accepted counts 1/2/4,
this removes 3/2/0 of those matrix updates respectively. The fixed workload's
actual per-request mix, rather than a chosen favorable acceptance, determines
the final cost.

At accepted4, matrix arithmetic is not reduced, but four candidate writes
become one base write. At accepted1/2, matrix arithmetic is reduced as well.
Product corrections, old-log reads, new-log emission, ownership publication
and their dependencies are new costs. They must be measured with commit in
the same closed endpoint, not subtracted from unrelated standalone profiles.

The current closed block result is 943.26 us. To improve upon the installed
1001.30 us recurrence by the earlier 135 us investment budget, the new closed
endpoint would need to be below roughly 866.30 us in comparable conditions.
That requires at least another 76.96 us net improvement over current block.
A two-layer replay near 4.8 ms needs about 240 us total saving for 5%; these
are screening budgets and do not establish a measured layer/model result.
Once the representation is shared by padding, split's final gate is still
at least 5% relative to the newly optimized padding endpoint.

For a capacity-100 illustration, one compact FP32 base including null is
158,859,264 bytes, and each conservative log bank is 10,238,976 bytes.
One-bank total is 169,098,240 bytes; two-bank total is 179,337,216 bytes,
versus 630,718,464 logical bytes for 401 dense FP32 candidates. These are
SSM tensor budgets only. Leaving the original shared-page allocation alive
does not produce this allocator saving. Runtime peak memory, convolution
allocation, product/output scratch, temporary export and retained graphs
must be reported separately.

## Additional split opportunity after write reduction

Request splitting can expose an independent state-commit lane. For disjoint
requests, once `P_A` finishes reading A's old base, A's commit/output work can
overlap B's products. After A's attention output is ready, A's output GEMM
can potentially overlap B's remaining state work. Neither launch may depend
on a partial completion event from a still-running opaque kernel: emit an
output-ready event only at a legal kernel boundary, and retain a separate
state-done event if commit is deferred.

The initial recurrence test should retain full bucket capacity and metadata,
using static request task ranges inside kernels. A range is an operator
work partition, not a new capture size. Products retain their M384/N8/K128
grouping; changing task ranges must not introduce selected-state gathers,
transposes, hidden Q/K copies or weight replication. Range boundaries follow
the bucket rule and cannot be tailored to the fixed test trajectory.

The useful comparison has three controls: full-bucket compact padding,
serial request-partitioned compact execution, and the same partition with
explicit independent dependencies. All include accepted commits and the final
join. This separates common algorithm savings, partition costs and actual
parallel savings. It must be followed by a closed real-layer control which
keeps whole input projections/GateUp where possible and counts every
collective, residual path and required state drain.

Reducing state traffic first gives this overlap more room than the current
dense writer, but does not guarantee a gain. Cube products and AIV commit
still access the same memory system. A trace must establish real overlap,
its execution inflation and the final critical path; stream labels alone
are insufficient. Padding can also overlap its own output projection with
any legally independent commit, so that common schedule must be included
in its equally optimized control.

## Qualification and stop gates

The CPU oracle first checks the mathematical and lifecycle contracts:
FP64 algebraic projection identity; FP32 projection drift; ordered candidate
export and commit; zero/underflow decays; mixed accepted counts; first use;
request permutation and pause/resume. Its current valid flag assumes normal
sequential calls; it does not detect stale acceptance, retries or request
reuse. Explicit generation/transaction checks remain an integration gate.
CPU reconstruction uses separate multiply/add, while the native exporter
and commit must share the actual device FMA path and agree bitwise. The CPU
oracle must not substitute for device FMA/reduction or performance evidence.
Finite logs and outputs do not prove finite logical candidates; exported
candidates are checked independently, including an overflow regression.

Device qualification then requires repeated identical eager execution and
same-schedule eager/replay equality, continued dynamic inputs/state, all four
logical candidate exports, inactive/null/page-gap preservation, and finite
outputs/logs/base. Compare reassociated products, outputs and continued state
to dense controls separately. Record worst-request and per-prefix errors,
not only a batch-wide mean. Longer accepted-state feedback and real decoder
outputs must pass before model performance claims.

Time complete recurrence with product correction, solve, log emission,
accepted commit and joins. Use accepted1/2/4 as causal controls and the fixed
mixed trajectory as the primary comparison. Report export-inclusive results
where the consumer needs export, and all allocator deltas. Stop or redesign
if accepted4 cost, projection drift, required dense consumers or state
management exhaust the budget; do not advance on an emit-only result.

Only then proceed to connected real layers and full dynamic replay with
true FIA host updates, input preparation, TP communication, complete hidden
and residual outputs, state drains and equally optimized padding. Report
mean and tail latency, repeatability and extra memory. The goal is achieved
only by that complete comparison, not by the research design or recurrence
gate.

## Full-model residency constraint

`multi_graph_hybrid_memory_budget.py` reads the actual checkpoint config,
index and safetensors headers without loading tensor payloads or allocating
device memory. `structural-hybrid-memory-budget-v1.json` applies those inputs
to the unchanged fixed trajectory and verifies its original digest. JSON
round-trip normalization of integer request keys preserves that digest; no
workload content is changed.

The manifest retains 128 stable request histories, 513 dense candidate slots
including null, and up to 1292 FIA pages at block size 128. With 48 GDN and
16 full-attention layers, the FP32 SSM alone is 36.07 GiB per TP2 rank.
Estimated model-style aligned GDN pages occupy 42.08 GiB. An ideal text-weight
shard adds 25.05 GiB, and retained FIA KV adds up to 5.05 GiB. Their estimated
sum is 72.18 GiB before activations, workspaces, graph pools, replicated
weights or runtime reservations, exceeding a 64 GiB device's nominal capacity.
This is a byte estimate, not an observed OOM or an allocation measurement.

The corresponding estimated sums are 36.09 GiB at TP4 and 20.81 GiB at TP8;
both still omit the named unknowns. TP8 replicates the four FIA KV heads, so
its FIA residency does not halve relative to TP4. Actual model loading and
execution support for these configurations remain unverified. The current
native Q4 kernels retain TP2-local head dimensions and cannot simply be
launched unchanged at higher TP.

Compact SSM storage could therefore address a concrete full-model capacity
constraint as well as traffic, provided the runtime genuinely removes the
four-candidate allocation. Keeping the original shared pages alive while
adding logs cannot claim that saving. A complete cache-manager integration
must preserve speculative convolution state and every required dense export.

## Native recurrence qualification

`multi_graph_gdn_q4_log_kernel` implements one full request/head task after
the direct FP32 Cube products kernel. The task loads all old log operands,
corrects projections, solves outputs/current updates, commits the previous
accepted prefix and writes the new log. Old-log reading finishes before
new-log overwriting in the same task. The initial prototype uses one log
bank and externally serialized request generations; this ownership proof
does not cover overlapping generations or a future parallel reader.

The native exporter and accepted commit call the same `ApplyOldLog` code,
with FP32 gamma multiplication followed by `MulAddDst`. This is the actual
device canonical reconstruction, unlike the CPU oracle's separate
multiply/add. Export is a read-only state consumer with explicit output.

### Rejected packed-flag version

The first build fails repeated eager state comparison in validity flags at
step zero, with maximum integer difference 1. It is rejected before timing;
the retained log is `gdn-q4-log-v1-rejected.log`. Packed four-byte flags
allowed neighboring cores to publish into the same cache line. V2 replaces
them with one 32-byte record per request/head and publishes the whole record
using DMA, removing both shared-line ownership and scalar publication. No
recurrence arithmetic changes. This is a state-publication correctness bug,
not acceptable cross-algorithm floating-point drift.

The exact rejected V1 source set is archived under
`/workspace/multi-graph-results/q4-log-v1-rejected-source/`; reconstruction
was checked against its original loader fingerprint
`49b11b8d81e77db1be99cc18bfc84abb3a1cc784976da735da6b01a018ad25de`.

### Qualified V2 and its boundary

`gdn-q4-log-v2.json` passes 25 device checks at capacity 100 and live batches
97-99. It covers eight continued dynamic-input steps, stable shuffled IDs,
mixed accepted 1..4, first use, pauses/resumption, repeated eager and
same-schedule graph equality, null/inactive state/log/flag isolation, page
canaries, finite outputs and every logical candidate. The next accepted
base is bitwise equal to the previous native exported prefix. A separate
export-inclusive graph matches its eager output, state and all candidates.
Worst-request cross-algorithm normalized error across the recorded output
and prefix checks is 0.0001723924 (0.01724%), below the preregistered 0.5%
screen. This does not certify model-level numerical behavior.

| Complete recurrence endpoint | Three device round medians (us) | Median of round medians (us) |
| --- | --- | ---: |
| Installed dense recurrence | 1005.04 / 1001.55 / 1000.06 | 1001.55 |
| Compact Q4 including accepted commit | 688.24 / 688.50 / 686.17 | 688.24 |
| Same compact Q4 plus all dense candidate export | 1459.08 / 1460.50 / 1459.27 | 1459.27 |

The closed compact endpoint saves 313.31 us (31.28%) in this recurrence
experiment, passing the 135 us real-layer investment gate. It includes Q/K
casts, Cube base products, previous-log correction, solve, the full previous
accepted-base commit, new logs, output zeroing and device completion. It is
not the goal's full-model timing boundary and does not compare split against
equally optimized padding.

All-candidate export removes the latency benefit and is slower than official.
Thus a downstream Q4 consumer must use the explicit compact representation;
the existing dense-cache interface cannot be retained by silently exporting
every step or by omitting required consumers from timing. A real-layer
adapter must integrate compact state in reset, snapshots, continued
trajectories, inactive checks and state ownership. Production
generation/retry/reuse validation, prefill and non-spec transitions remain
unimplemented. Both padding and split must receive these same changes.

Explicit compact base pages, logs and padded flags occupy respectively
185,335,808 / 10,238,976 / 77,568 bytes in this probe, versus 735,838,208
bytes of paired official pages. These are actual tensor extents, including
artificial page canaries. They are not full-model allocator savings.
The 635,437,056-byte dense export scratch is present for qualification and
the export-inclusive consumer. Peak process allocation 7,671,243,776 bytes
includes both controls, all graphs and snapshots; it is not incremental
runtime peak memory.

Tested V2 sources and probe are archived under
`/workspace/multi-graph-results/q4-log-v2-tested-source/`. The log kernel
library SHA256 is
`68a3b0098edf6cc89b95c307e01fcf2a76cd57468ff1ef71bb7abdac32c81436`;
the products library SHA256 is
`e167b52c6987a2578269ca452703ecdb9d9f68d9db63627cade8879e3fb8e823`.

### Fixed-range request pipeline

The next causal experiment adds only static request task ranges to the
existing full-capacity kernels. Buffers, metadata, row addressing and native
M384/N8/K128 products remain unchanged. The fixed rule is `boundary=B//2`;
the capacity-100 control uses ranges `[0,50)` and `[50,100)`. No graph capture
size is added or tailored to the live batch. The operations are:

```text
compact:          cast/clear -> P_full -> S_full
compact_serial:   cast/clear -> P_A -> S_A -> P_B -> S_B
compact_pipeline: cast/clear -> P_A -> (S_A || P_B) -> S_B -> root join
```

`gdn-q4-log-pipeline-v3.json` passes 41 checks. Both partitioned controls
start each continued step from the identical compact state and produce
bitwise-equal full output, base/pages, logs and flags to full compact.
The eight-step official numerical and canonical candidate/commit checks
also pass unchanged. Timing includes casts, output clear, range operations,
events, waits, accepted commit and the final root-stream join.

| Endpoint | Three device round medians (us) | Median of round medians (us) |
| --- | --- | ---: |
| Installed dense recurrence | 1012.84 / 999.68 / 1000.32 | 1000.32 |
| Full compact | 706.85 / 698.05 / 697.47 | 698.05 |
| Partitioned serial compact | 726.31 / 729.72 / 715.76 | 726.31 |
| Partitioned pipeline compact | 667.22 / 666.66 / 653.64 | 666.66 |
| Full compact plus all dense export | 1460.35 / 1453.35 / 1452.41 | 1453.35 |

The pipeline saves 59.65 us (8.21%) against the same partitioned serial
arithmetic, but partitioning costs 28.26 us relative to the full compact
control. Its net saving against equally optimized full compact is therefore
31.39 us (4.50%). This demonstrates a qualified recurrence-range effect;
it does not meet the complete-model average 5% gate. Full-layer common work
can dilute or erase this difference. A real layer is the next check, with
compact representation and all public operator improvements on both sides.

Exact tested sources and products tiling are archived under
`/workspace/multi-graph-results/q4-log-v3-tested-source/`. Log and products
library SHA256 values are respectively
`787efd722754a96e2fa5ef4194bec5b86bc070ae9ea3bc86f3826d233b2b1068`
and `49f2157b07d363f62a6a24969005973981e8f0b9a848d61e5b312cd22ee6cb6c`.
