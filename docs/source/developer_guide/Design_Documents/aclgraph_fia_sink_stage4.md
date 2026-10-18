# FIA sink stage 4: preparation reuse and whole-row no-FD planning

Status: bounded NPU experiments completed, 2026-10-19. This stage measures
isolated attention chains and a four-capacity plan diagnostic. It is not a
complete-model, serving or end-to-end qualification.

## Preparation reuse

The first experiment keeps the stage-three graph controls and compares four
matched cases on the unchanged 128-step trace, 17 default capacities, sixteen
private attention occurrences and three trials (1536 samples):

| Case | Preparation |
| --- | --- |
| `sink_fresh` | Existing per-step CPU tensor construction and sink length/table copies |
| `sink_reuse` | One pinned Q/KV/table source per bucket; invariant cumulative Q uploaded once; KV/table refreshed each step |
| `legacy_fresh` | Existing table preparation and real sixteen FIA host updates |
| `legacy_reuse` | One pinned table source per bucket and the same real sixteen updates |

Device allocations remain separate and exact-sized. A pinned source cannot be
overwritten until the preceding graph completion. The Q4 endpoints are reused
only because every physical request in this fixed catalog remains Q4; exact KV
lengths and page tables are still filled and copied every step. The source and
device pointers, copy counts, and completion state are recorded in each report.

Both complete preparation runs are retained. In trial 1, `sink_fresh`,
capacity 480, step 117, wall time reaches 615.552 ms in the second run and
943.900 ms in the first. In the second run, 609.141 ms occurs in the host
preparation/enqueue interval, while the queue span is 6.204 ms. This locates
the delay outside the attention execution span, but does not identify its
cause. Allocation, garbage collection, instrumentation and other host/runtime
effects were not isolated. Recurrence at the same recorded point does not
establish a deterministic allocation defect.

The second run's unfiltered results, including this observation, are:

| Case | Mean wall | P50 wall | P95 wall | P99 wall |
| --- | ---: | ---: | ---: | ---: |
| `sink_fresh` | 7.082 ms | 5.546 ms | 6.809 ms | 7.052 ms |
| `sink_reuse` | 5.094 ms | 5.154 ms | 6.286 ms | 6.583 ms |
| `legacy_fresh` | 5.743 ms | 5.813 ms | 6.483 ms | 6.641 ms |
| `legacy_reuse` | 5.472 ms | 5.530 ms | 6.147 ms | 6.458 ms |

| Paired difference | Unfiltered mean wall reduction | Matched wins |
| --- | ---: | ---: |
| `sink_fresh - sink_reuse` | 1987.6 us | 384/384 |
| `legacy_fresh - legacy_reuse` | 271.1 us | 379/384 |
| `legacy_reuse - sink_reuse` | 378.2 us | 292/384 |

The `sink_fresh` mean and its 28.07% aggregate reduction are dominated by that
host observation. A separate, post hoc sensitivity calculation omitting the
matched trial 1 / step 117 pair gives 402.2 us over 383 pairs, all improving. The
first run's equivalent sensitivity result is 355.1 us. This omission was not
predeclared, is not the primary result, and does not replace the retained raw
sample or ordinary percentiles. The analyzer itself does not filter samples
or emit an independently defined outlier count.

With reusable preparation on both paths, the mean wall difference is
378.2 us, or 6.91% of the isolated legacy attention chain. This is not uniform
across capacities: at 496 it is only 11.7 us (0.20%, 16/30 pairs improve);
at 512 the sink loses 133.3 us (2.15%, only 4/27 pairs improve).

The sink reuse case removes repeated preparation work, but this is a descriptor
lifetime optimization rather than an independent attention algorithm. It
provides the preparation control for subsequent no-FD experiments, without
establishing a complete-model benefit.

## Whole-row no-FD diagnostic

The second experiment uses actual default capacities 256, 400, 496 and 512
(trace steps 24, 0, 5 and 29), sixteen private occurrences, three trials and
five repeats (240 timing samples). These are four fixed representative records,
not all 128 temporal steps. It keeps the same D256 sink consumer, internal
matrix subblocks, Q/K/V contents, exact lengths, page routes and ACLGraph
addresses. The only attention-plan change is a legal contiguous assignment of
complete `(request, KV head)` rows, with `numFD=0`, `usedFdVec=0`, all FD fields
zero and a final endpoint covering physical padding rows.

The CPU producer performs strict geometry/page/template preflight once per
representative, then uses a guarded fast builder during timing. That builder
reconstructs the plan from the KV lengths and base geometry cached by preflight;
it does not reread changed lengths on each repeat. Its identity guard and
payload comparison apply to these fixed records. A pinned int32 metadata source,
plan construction and H2D copy remain in the timed `whole_cpu` endpoint.
Changing a record requires fresh preflight or a producer that consumes its
current lengths, with that work included. This experiment does not qualify
stale-plan handling or temporal updates across the 128-step trajectory.

Four cases are compared:

| Case | Plan cost in timing |
| --- | --- |
| `native_shared` | Existing one device metadata producer, then attention |
| `native_oracle` | Existing exact metadata prepared outside timing; diagnostic |
| `whole_cpu` | Rebuild from the preflight record snapshot, pinned fill/copy and attention |
| `whole_oracle` | Whole-row plan and copy outside timing; diagnostic |

Four pre-timing qualification passes, one per representative, and one
post-timing pass compare all sixteen private outputs across controls, exact zero
padding, resident-input/descriptor guards and sampled FP32 reference rows.
These checks bracket the 240 timings; there is no independent FP32 check after
every timing sample. The whole plan is validated for complete head coverage,
contiguous core prefix, physical padding sentinel and no FD fields. Its cost
model is explicitly an `abstract_6_10_upper_bound` heuristic; it is not a
profiler-derived load balance proof.

| Case | Mean wall | Mean preparation | Mean queue span | P50 wall | P95 wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_shared` | 4.964 ms | 0.220 ms | 4.679 ms | 5.357 ms | 5.934 ms |
| `native_oracle` | 4.907 ms | 0.216 ms | 4.620 ms | 5.083 ms | 5.916 ms |
| `whole_cpu` | 5.335 ms | 0.915 ms | 4.352 ms | 5.593 ms | 6.448 ms |
| `whole_oracle` | 4.614 ms | 0.204 ms | 4.346 ms | 4.822 ms | 5.589 ms |

The whole-row attention path is faster than the native FD path in the oracle
comparison by **292.9 us in mean wall time; all 60 matched pairs improve**.
The corresponding mean queue-span reduction is 274.3 us. Queue reductions
are 324.4 us at capacity 256, 187.4 us at 400, 292.6 us at 496 and 292.7 us at 512.
The respective wall reductions are 334.6, 211.3, 301.2 and 324.3 us. These
quantities differ because the wall endpoint includes preparation and host
submission. Queue spans also include dependency waits and launch gaps; they
are not separately measured attention-task durations. The result supports
investigating this work assignment, but the oracle excludes plan generation
and copy and therefore is not an executable dynamic speedup.

With the CPU plan included, `whole_cpu` is **371.3 us slower** than
`native_shared` on average, winning only one of 60 pairs. The mean preparation
interval increases 695.2 us, while its queue span decreases 326.5 us. The current
Python snapshot rebuild/fill/copy path consumes the observed execution-path
saving. Strict preflight remains outside timing. A native whole-row producer
that reads the current lengths is a concrete follow-up hypothesis: it must
preserve enough of the measured assignment benefit after its own cost is
included. No measurement here establishes its eventual build cost or net gain.

Changing from FD partial softmax to complete-row work changes floating-point
reduction order. The successful tolerance checks do not establish bitwise
identity. The experiment also does not prove that the abstract cost model is
the optimal assignment for every trace. It measures a lower replay envelope
for the legal no-FD consumer path on four real default buckets, without
isolating the FD merge duration or individual-core tail. It does not
establish a policy for every shape or a complete dynamic
trajectory.

## Safety boundary and next gate

The helper rejects malformed page tables, unequal widths, bool/negative/out of
range page IDs when page-count provenance is supplied, stale templates,
non-Q4 physical rows and missing preflight for the fast path. It does not alter
the installed OPP or attention kernel. The NPU process uses the isolated stage
two library and retains the exact native provenance checks.

The next gate is a native metadata producer that reads current lengths while
keeping the existing 1024-word ABI and consumer. It must preserve exact
sparse/causal coverage and independently validate its whole-row endpoints.
The current cost weights remain an abstract heuristic; compare assignments and
real execution before attributing gains to load balance. The implementation
must pass B1/B4/zero-KV/padding/recovery graph checks, repeat the four-capacity
comparison including producer cost, then qualify the unchanged 128-step
trajectory. Full-model integration follows only if that endpoint is positive.
No complete-model or end-to-end gain is claimed by this stage; the bounded
292.9 us oracle difference does not establish the much larger net reduction
needed for a substantial whole-model improvement.

## Evidence

- [`fia-sink-stage4-preparation.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-preparation.json)
- [`fia-sink-stage4-preparation-analysis.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-preparation-analysis.json)
- [`fia-sink-stage4-preparation-first-run.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-preparation-first-run.json)
- [`fia-sink-stage4-preparation-first-run-analysis.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-preparation-first-run-analysis.json)
- [`fia-sink-stage4-whole-plan.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-whole-plan.json)
- [`fia-sink-stage4-whole-plan-analysis.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-whole-plan-analysis.json)
- [`fia-sink-stage4-whole-screen.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-whole-screen.json)

The parent workspace retains both complete preparation reruns, the whole-plan
formal and screen runs, launch manifests and raw logs under
`outputs/thesis_incremental_plan/stage4-evidence/`. Its source audit is
`outputs/thesis_incremental_plan/stage4-fd-source-audit.md`; that parent-workspace
artifact is not a repository-relative documentation link.
