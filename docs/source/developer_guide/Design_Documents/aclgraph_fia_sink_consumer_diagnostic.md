# Current FIA consumer bottleneck diagnostic

Status: six-process bounded measurement and two independent analyses completed,
2026-10-19. This is a small prerequisite diagnostic, not Stage 8 full-model
integration or a replacement result for the frozen Stage 7 trace.
The isolated running-O source patch and both macro builds are now complete;
the four-cell `v3` runtime comparison and independent audit are complete.
The bounded result below does not establish a stable gain. A Chinese overview is available in the
[current scheme explanation](aclgraph_dynamic_shape_current_scheme_zh.md).

## Question

Stage 7 whole-row assignment saves 89.354 us over sixteen attention occurrences
on the complete descriptor trace. At capacity512 the saving is only 15.090 us.
The earlier Stage 3 profile found a slower sink consumer than legacy FIA, but
that single-point measurement preceded the current whole-row implementation.
It cannot establish the current residual consumer gap or its cause.

Re-measure the present consumers at two original records before changing
kernel dataflow or starting a larger real-model integration.

## Fixed small experiment

| Item | Declared contract |
| --- | --- |
| Records | First original capacity400 record, step0/B98; first capacity512 record, step29/B128. |
| Shapes | BF16 Q4/TND, Q12/KV2/D256, page128, sparse3; original default catalog. |
| Inputs | Sixteen private resident synthetic Q/K/V occurrences, seed1703, original exact lengths and owner pages. |
| Controls | `upstream` sink, `whole` sink, `legacy` FIA with all real host graph-task updates. |
| Process rounds | Upstream/whole/legacy, then legacy/whole/upstream; six fresh processes, sequential on NPU0. |
| Per process | Five warmups and ten unprofiled observations per record. Three separate profiler replays per record. |
| Counts | 120 unprofiled timings; 36 profiler labels and 576 attention executions. |
| Scope | Two synthetic attention-chain records; no complete trace, loaded model, serving or split-versus-padding claim. |

Policy order is reversed in the second round, but two rounds do not establish
stable process-level statistics. Preserve every timing, including regressions.
Do not add capture sizes, remove physical padding or change request ownership.

Each bucket owns a separate graph pool. All controls use persistent pinned
preparation and retain real current metadata. Sink uploads current device KV
lengths and page routing; legacy uploads page routing and uses its true host
length update interface. Do not add useless legacy length H2D to manufacture
identical work. Report required copies explicitly and include their costs.

The full wall endpoint includes preparation, necessary copies, native planning
or every legacy update, graph replay and final completion. Queue events include
dependency waits and host submission gaps; queue span is not pure kernel time.

## Correctness and identity gates

Qualification before and after timing checks every private output for finite
values and exact KV0 padding zeros, stable addresses, exact read-only length
and page descriptors, and unchanged resident inputs. Compare three live rows
at occurrences zero, seven and fifteen with the independent FP32 reference.
The page gather uses the real ordered prefix, retaining duplicate IDs and
the exact final-page tail. There are 216 sampled row comparisons across the
six processes; timed observations are not each independently FP32 checked.

Sink qualification retains all 1024 plan words and current exact base fields.
Whole-row additionally requires complete physical coverage and zero FD fields.
Legacy instead records registered tasks and actual update execution counts;
registration alone is not evidence that replay ran the updates.
Actual task auditing wraps only pre/post qualification. Unprofiled timings
and profiler replays call the original update method without per-task Python
proxies, list comparisons or counters. Profiler task counts independently
check those uninstrumented replays.

Complete Q/K/V, mask and actual bucket query hashes must match across control
processes. Both sink controls use the same qualified consumer, extension and
non-policy native artifacts. Only the selected AICPU policy differs. Legacy
keeps its original consumer and update interface. No installed OPP is changed.

## Profiling and decision

Profiler replays are outside unprofiled timing. Retain the original Chrome
trace, labels, hashes, task identities and hardware completion evidence.
Complete preparation before each labeled replay so table H2D cannot be
mistaken for legacy update DMA. The label encloses replay and final join;
unprofiled wall still includes all required preparation and copies.
Interpret only groups with the expected sixteen attention executions and
control-specific producer/update tasks. A trace warning remains part of the
evidence, even when expected tasks are present.

For each control and capacity report:

- Unprofiled complete wall, preparation, submission and queue span.
- Hardware attention duration sum and first-start to last-end interval.
- Gaps between attention tasks and overlapping `MEM_WAIT_VALUE` waits.
- Native producer/clear tasks or real legacy CPU updates and update DMA tasks.

Hardware sums, CPU intervals and waits can overlap and must not be added to
reconstruct wall latency. Profiler overhead prevents using profiled label time
as an unprofiled performance result. The two-record result does not replace
the Stage 7 canonical frequency-weighted comparison.

If the current whole-row consumer still has a material execution deficit,
collect the device metrics needed to distinguish core imbalance, Cube/Vector
pipeline and memory effects before selecting a D256 kernel rewrite. A duration
trace alone cannot identify MM1/MM2 or FD merge as the bottleneck. If the deficit
does not remain, or the complete-endpoint budget is too small, keep whole-row
as a baseline and stop expanding this kernel route. A real-model critical-path
budget remains necessary before claiming end-to-end headroom.

## Observed result

The selected `bounded-v2` run completed all six processes, 120 unprofiled
timings and 36 profiler replays. There are twenty wall observations per
mode/capacity and six profiled replays per mode/capacity. These repeated
samples belong to only two process rounds; they are not twenty independent
process trials. All results below include both rounds.

| Capacity | Upstream wall us | Whole wall us | Legacy wall us | Whole vs upstream | Whole vs legacy |
| --- | ---: | ---: | ---: | ---: | ---: |
| 400 | 4808.460 | 4355.576 | 5354.357 | 9.418% lower | 18.654% lower |
| 512 | 6085.394 | 5471.080 | 6088.965 | 10.095% lower | 10.148% lower |

Whole improves all twenty matched observations against either control at
both capacities. Its upstream-relative reductions by process round are
7.215%/11.525% at capacity400 and 7.141%/12.895% at capacity512. Upstream
and legacy at capacity512 are effectively tied in this bounded sample,
and the sign of their comparison changes between rounds.

The separate hardware profiles localize the remaining attention-chain cost:

| Capacity | Upstream attention sum us | Whole attention sum us | Legacy attention sum us | Whole minus legacy |
| --- | ---: | ---: | ---: | ---: |
| 400 | 4216.108 | 3783.756 | 3514.050 | 269.705 us, 7.675% |
| 512 | 5489.833 | 4869.984 | 4499.837 | 370.147 us, 8.226% |

Each sum contains sixteen hardware attention executions. Whole-row producer
time is 74.128/112.112 us, and the inter-attention gap sum is only
37.711/37.444 us at capacity400/512. Native assignment changes reduce the
upstream attention sum by 432.352/619.849 us, while producer duration changes
by -5.020/+6.550 us. This points toward consumer execution rather than a
large remaining producer or inter-kernel scheduling budget. The residual
legacy gap is a measured comparison target, not a promise that all of it
can be recovered: the consumer implementations and tiling differ.

Legacy profiles contain all 192 real CPU updates and 192 update DMA tasks.
Their mean per-replay CPU update sums are 1780.600/2126.285 us, while the
attention gap sums are 1020.530/1040.548 us. Those intervals overlap device
work and waits; they are neither additive wall components nor a direct
estimate of the adoption speedup. Sink profiles contain all 24 producers
and 24 clear tasks. Both controls retain the original consumer/native
artifact identity contracts.

The formal analyzer and a separate stdlib auditor pass all task counts,
raw-file hashes and cross-process input identities. The 216 sampled FP32
rows pass with maximum absolute error 0.00029981881380081177. All
qualification-time 128 legacy task applications are independently retained;
measured legacy updates remain uninstrumented. These correctness passes
do not reference-check each timed replay individually.

## Small hardware-counter follow-up

Two additional fresh processes profile whole and legacy once per bucket,
with `ProfilerLevel.Level1`, `AiCMetrics.PipeUtilization` and text export.
The thin wrapper changes profiler configuration, preserving the frozen
probe and its preparation/update/qualification implementations. All 64
attention records have valid counters; 72 additional FP32 rows and all
input, descriptor and address checks pass. The device is idle afterward.
These four replays are separate from the 120 formal wall observations.

| Capacity | Mode | Mean kernel us | MAC time us | Vector scalar time us | AIC MTE2 time us |
| --- | --- | ---: | ---: | ---: | ---: |
| 400 | Whole | 246.983 | 33.715 | 84.608 | 180.525 |
| 400 | Legacy | 221.452 | 31.470 | 54.702 | 154.894 |
| 512 | Whole | 320.904 | 47.473 | 102.219 | 240.193 |
| 512 | Legacy | 284.126 | 44.121 | 75.851 | 212.205 |

Values are equally weighted means over sixteen occurrences in one replay,
not system utilization or individual-core measurements. MAC time increases
by only 2.245/3.352 us per operator, while Vector scalar and AIC MTE2
increase by 29.907/26.368 us and 25.631/27.988 us. AIC fixpipe time also
increases by 16.923/21.354 us. These concurrent pipeline durations must not
be added. The evidence prioritizes descriptor/control, paged-copy and
pipeline organization as source-inspection targets; it does not isolate
a source loop, prove HBM saturation or establish recoverable savings.

The [official counter definitions][counters] describe `cube_utilization`
using summed cycles, frequency, core count and task duration; this is not
MAC/FLOPS utilization. Cycle-to-time conversion also depends on frequency.
The first three whole/capacity512 records have atypical derived core times;
all remain in the result. One replay cannot separate startup, frequency and
counter transients or establish a stable causal effect. Preserve per-record
metrics and raw cycles rather than relying only on averaged ratios.

The first metrics wrapper failed before capture because replacing `Popen`
with a function broke a library subclass. The retained successful wrapper
uses a compatible subclass. That failed attempt contains no performance
sample and remains separately identifiable in the evidence archive.

[counters]: https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0067.html

## First consumer ablation target

Read-only inspection of pinned `bec19b8dea1e` identifies a concrete candidate
in `fia_block_vec_nonquant_sink.h::DealBmm2ResBaseBlock`:

- Lines 752-755 read the new MM2 partial from `mm2ResGm`.
- Lines 760-792 read the previous running output from `vec2ResGm`, rescale it
  with the current softmax factor and add the new partial.
- Lines 820-831 copy each non-final result through another UB queue and write
  it back to `vec2ResGm` for the next S2 iteration.

The isolated consumer variant now implements running FP32 output residency
in UB under the complete request/head lifetime contract. It retains the MM2
producer workspace handoff, softmax arithmetic, exact lengths, page routing
and final output format. Whole-row/no-FD ownership provides a useful lifetime
contract, but source and compile checks alone do not establish runtime safety.

`SetMSplitInfo` splits the Q4/GQA6 M=24 rows at a sixteen-row boundary:
the two AIV sides own sixteen/eight rows, requiring 16/8 KiB of FP32 running
output at D256. The patch budgets the heavier side, not an assumed 12/12 KiB
split, reclaiming 16 KiB from the existing output queue rather than increasing
the 184 KiB static UB total. The existing three-task pipeline can keep
different heads in flight. The Vec2 state therefore checks actual owner,
geometry and consecutive loop/S2 identity before reuse, with spill and event
protection on the guarded path. A blindly reused accumulator is incorrect.
The original consumer path remains for unsupported geometry, FD or an
unqualified ownership/lifetime transition. The bounded runtime checks below
qualify the two selected records, not every supported source branch.

A smaller prerequisite ablation can test removing the UB-to-UB staging copy
while retaining GM accumulation, provided the MTE3 completion protects the
source buffer lifetime. It must not be presented as equivalent to retaining
the full running output on chip. Compare the same two original records,
all pre/post correctness gates and complete wall, with profiles separated.
Keep a change only if its unprofiled endpoint improves and its affected
pipeline/task evidence agrees; counter ratios alone are not an acceptance
criterion. The separate remove-staging-only variant is not implemented.
The running-O variant and its macro-disabled source control have both passed
AscendC compilation; this is not a correctness or speedup result.

The current source-ablation binaries compile only the two selected tiling
keys `104000000010222223` and `104000000010322223`. The build tool requires
the actual `kernelList` key set to match the selection exactly, allowing
duplicate AIC/AIV entries for one key. These are restricted experimental
binaries, not a replacement package qualified for arbitrary shapes or all
fallback templates. The four `v3` cells cross upstream/whole producer with
macro-disabled/enabled consumer using the same two original records. Dynamic
chronological replay, full-model integration and split-over-padding performance
remain unvalidated for this patch. The GM running-output round trips are a
source-level target, not a measured proof that GM bandwidth is the bottleneck.

Eligibility must follow the actual native ownership, geometry and lifetime
contract, not the producer's policy name. If an upstream plan satisfies the
same consumer conditions, it receives the same fast path. Compare both
policies with the best legal consumer; otherwise a consumer-only improvement
would be falsely attributed to the whole-row assignment. The combined result
must separate consumer benefit, assignment benefit and their interaction.

On-chip online-softmax accumulation is established attention practice, not
a new algorithm claim. Our potential contribution would be its guarded
realization under the current A2 whole-head native ownership and task/UB
constraints, with evidence against the same upstream consumer. This is not
yet a benefit unique to graph splitting or a demonstrated thesis-level
novelty result.

### Bounded running-O source-ablation result

The selected `v3` run completes four fresh processes, eighty unprofiled
timings and 144 sampled FP32 row comparisons. Each policy/capacity/variant
has ten timings; there is only one fresh process per cell. The macro-disabled
control and resident variant use the same patched source, restricted tiling
keys and frozen preparation/replay protocol. Profiling is separate, with one
replay per capacity in each process.

| Policy | Capacity | Control wall us | Resident wall us | Mean reduction | Profile attention sum saved us |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upstream | 400 | 4517.510 | 4357.540 | 3.541% | 101.742 |
| Upstream | 512 | 5673.378 | 5543.449 | 2.290% | 127.862 |
| Whole | 400 | 4250.731 | 4186.756 | 1.505% | 49.021 |
| Whole | 512 | 5609.524 | 5311.093 | 5.320% | 73.001 |

The whole/capacity512 control includes a 7896.423 us wall spike. All ten
observations remain in its mean. Its median is 5362.9825 us versus
5309.507 us resident, a reduction of about 0.997%. The 5.320% mean is
therefore an unstable summary of this cell; the separate single profile
shows only 73.001 us saved across sixteen attention kernels. Profile sums,
host intervals and wall timings cannot be substituted for one another.
Fixed process order, addresses, frequency and temperature remain confounded;
the observations do not establish stable process-level speedup.

The independent raw audit passes 5393 assertions over 92 manifested files,
including all timings, 24 complete plan checks and private input identities.
Within each policy and capacity, all sixteen complete output hashes match
bitwise across macros and pre/post qualification, and all 1024 plan words
remain unchanged. The separate 144 FP32 sampled rows pass with maximum
absolute error 0.00029981881380081177. The full-output hash equality is an
observed identity, not an inference from those samples; qualification is
separate from the timed replays. NPU0 is idle afterward.

This screen observes a modest consumer benefit, including on upstream's
eligible heads. It does not show a large effect unique to whole-row ownership,
prove GM bandwidth saturation or isolate all eliminated running-output bytes
as measured HBM traffic. The affected source/plan and profile evidence support
further bounded validation; they do not establish a dynamic-trace, full-model,
serving or graph-splitting gain. On-chip accumulation remains established
practice, and the same legal fast consumer remains available to both policies.

## Why this does not supersede Stage 7

The chronological Stage 7 result remains a 1.788% observed mean reduction,
including its regressing round and P99. Repeating two fixed records here
does not establish dynamic-shape robustness or full-model benefit.

The exact resident Q/K/V, bucket query bytes and complete native plans for
step0 and step29 match Stage 7. Nevertheless, Stage 7 captured seventeen
bucket graphs and executed the chronological trace; this diagnostic captures
two and repeatedly executes one record. Last-capture allocated/reserved
memory is approximately 6.79/11.63 GB versus 5.67/6.40 GB. Process conditions,
warmup, ordering, allocation and working-set behavior also differ. Even the
same record has a different observed reduction, so trace-frequency weighting
alone cannot explain the discrepancy. Graph footprint is a candidate
confounder, not an established cause. Preserve this unresolved discrepancy
when interpreting the favorable point result.

The first launcher run, `bounded-v1`, was interrupted after a source-format
validation issue in its description string. The value was retained while
splitting the source line. Its completed and partial reports remain in a
separate directory; none contributes to the selected six-process result.

The next decision needs an allocation/order control and a narrowly scoped
consumer source ablation, rather than a larger model matrix. The minimum
context control captures and retains all seventeen Stage 7 bucket graphs
but uses exactly this two-record repeated protocol, alongside a same-session
two-bucket reference. Upstream/whole crossed with two/seventeen buckets
requires four fresh processes and eighty timings. A reduced gain would
implicate catalog construction/retention context, not prove a memory-byte
cause. An unchanged gain would leave chronological replay history as the
next control. The [completed bridge](aclgraph_fia_sink_catalog_bridge.md)
observes 6.086%/5.401% with two buckets and 7.580%/8.316% with seventeen.
The gain does not collapse with the larger resident catalog in this screen;
neither memory-byte causality nor historical magnitude stability is proved.

Only a qualified consumer change with exposed real-model headroom would
justify expanding the conditional Stage 8 integration.

## Existing-work boundary

Plan/run, device lengths and shared plans are existing mechanisms. This
diagnostic evaluates the current guarded native assignment and adapted
consumer; it adds no native incremental planner, new attention arithmetic,
multi-stream algorithm or literature novelty claim. Keep legacy-to-sink
adoption effects separate from upstream-to-whole policy effects.

See [Stage 7](aclgraph_fia_sink_stage7.md), the
[contribution audit](aclgraph_fia_sink_contribution_scope.md) and the
[conditional model-integration protocol](aclgraph_fia_sink_stage8_protocol.md).

## Saved evidence

- `benchmarks/results/multi_graph_structural/fia-sink-consumer-v1/` contains
  the selected suite, raw reports/traces, exact deployed sources and the
  separately retained interrupted run.
- `fia-sink-consumer-analysis.json` contains the qualified formal analysis,
  including both process rounds and all unprofiled/profile groups.
- `fia-sink-consumer-independent-audit.json` records the separate raw-byte,
  update, FP32, input-identity and hardware-task audit.
- `fia-sink-consumer-stage7-identity.json` records the same-record input/plan
  comparison and the differing graph allocation footprints.
- `fia-sink-consumer-stage7-review.json` independently checks the identity
  comparison across all saved process rounds and specifies the bridge control.
- `fia-sink-consumer-metrics-analysis.json` retains all 64 per-occurrence
  counter records. `fia-sink-consumer-v1/metrics/` preserves the lossless raw
  archive/manifest, the four CSVs, wrapper/configuration and post-run check.
- `fia-sink-running-o-v1/` retains the selected `v3` four-cell reports,
  restricted builds, raw profiles, manifests and source snapshots under
  `benchmarks/results/multi_graph_structural/`.
- `fia-sink-running-o-independent-audit.json` records the separate raw-file,
  key inventory, plan, output identity, FP32 and timing audit.
- `fia-sink-running-o-primary-recompute.json` independently recomputes the
  wall summaries, retained extrema, contrasts and complete-output identities.
