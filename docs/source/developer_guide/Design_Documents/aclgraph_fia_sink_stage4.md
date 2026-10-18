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

The second run is the primary preparation report. One deterministic host
preparation spike recurs at `sink_fresh`, capacity 480, step 117 (615.552 ms
in the second run; 943.900 ms in the first run). Its device queue span is
6.204 ms, so it is a host allocation/copy event rather than an attention
kernel duration. It is retained in the raw report and excluded only from the
trimmed diagnostic below; the analyzer reports the ordinary percentiles and
the outlier count separately.

After removing that one explicitly identified host spike from paired
attribution, the results are:

| Paired difference | Mean wall reduction | Matched wins |
| --- | ---: | ---: |
| `sink_fresh - sink_reuse` | 402.2 us | 383/383 |
| `legacy_fresh - legacy_reuse` | 271.1 us | 379/384 |
| `legacy_reuse - sink_reuse` | 378.2 us | 292/384 |

The sink reuse case removes a repeated host allocation/copy path, but it is a
descriptor lifetime optimization rather than an independent attention
algorithm. It is a useful required baseline for every later no-FD experiment.

## Whole-row no-FD diagnostic

The second experiment uses actual default capacities 256, 400, 496 and 512
(trace steps 24, 0, 5 and 29), sixteen private occurrences, three trials and
five repeats (240 samples). It keeps the same D256 sink consumer, internal
matrix subblocks, Q/K/V contents, exact lengths, page routes and ACLGraph
addresses. The only attention-plan change is a legal contiguous assignment of
complete `(request, KV head)` rows, with `numFD=0`, `usedFdVec=0`, all FD fields
zero and a final endpoint covering physical padding rows.

The producer is initially a CPU plan. It performs strict geometry/page/template
preflight once per bucket, then uses a guarded fast builder during timing. The
fast builder cannot run without the matching preflight identity and produces
the same 1024-word payload as the strict builder. A pinned int32 metadata source
and its H2D copy remain in the timed endpoint.

Four cases are compared:

| Case | Plan cost in timing |
| --- | --- |
| `native_shared` | Existing one device metadata producer, then attention |
| `native_oracle` | Existing exact metadata prepared outside timing; diagnostic |
| `whole_cpu` | Whole-row plan construction, pinned fill/copy and attention |
| `whole_oracle` | Whole-row plan and copy outside timing; diagnostic |

All 240 samples pass output agreement, exact zero padding, resident-input and
descriptor guards, and three FP32 rows per selected record. The whole plan is
validated independently for complete head coverage, contiguous core prefix,
physical padding sentinel and no FD fields. Its cost model is explicitly an
`abstract_6_10_upper_bound` heuristic; it is not a profiler-derived load
balance proof.

| Case | Mean wall | Mean preparation | Mean queue span | P50 | P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `native_shared` | 4.964 ms | 0.220 ms | 4.679 ms | 5.357 ms | 5.932 ms |
| `native_oracle` | 4.907 ms | 0.216 ms | 4.620 ms | 5.083 ms | 5.910 ms |
| `whole_cpu` | 5.335 ms | 0.915 ms | 4.352 ms | 5.593 ms | 6.448 ms |
| `whole_oracle` | 4.614 ms | 0.204 ms | 4.346 ms | 4.822 ms | 5.588 ms |

The whole-row attention path is faster than the native FD path in the oracle
comparison by **292.9 us on average; all 60 matched pairs improve**. The queue
reduction is present at every selected capacity: 334.6 us at 256, 211.3 us at
400, 301.2 us at 496 and 324.3 us at 512. This is a strong kernel-work
allocation signal, not yet an executable dynamic speedup because the oracle
excludes plan generation and copy.

With the CPU plan included, `whole_cpu` is **371.3 us slower** than
`native_shared` on average, winning only one of 60 pairs. Its extra preparation
is the remaining budget: the same no-FD kernel advantage is visible in the
queue span, but the Python planner and metadata copy consume it. The strict
planner validation is outside timing; the fast builder is used only after a
matching preflight. This is why the next implementation target is a native
AICPU/device-side whole-row planner or an equivalent constant-time descriptor
update, not another planner-only Python optimization.

Changing from FD partial softmax to complete-row work changes floating-point
reduction order. The successful tolerance checks do not establish bitwise
identity. The experiment also does not prove that the abstract cost model is
the optimal assignment for every trace; it proves that this legal no-FD
consumer path can remove a measurable execution tail on four real default
buckets.

## Safety boundary and next gate

The helper rejects malformed page tables, unequal widths, bool/negative/out of
range page IDs when page-count provenance is supplied, stale templates,
non-Q4 physical rows and missing preflight for the fast path. It does not alter
the installed OPP or attention kernel. The NPU process uses the isolated stage
two library and retains the exact native provenance checks.

The next gate is to move the whole-row assignment into the native metadata
producer while keeping the existing 1024-word ABI and consumer. The native
implementation must reproduce exact sparse/causal ranges, not substitute the
abstract 6/10 estimate; compare its endpoints and actual tile cost against
the CPU plan before timing. It must pass B1/B4/zero-KV/padding/recovery graph
checks, then repeat the four-capacity attention comparison. Only after that
positive endpoint should it be connected to the full dynamic model. No
complete-model or end-to-end gain is claimed by this stage.

## Evidence

- [`fia-sink-stage4-preparation.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-preparation.json)
- [`fia-sink-stage4-preparation-analysis.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-preparation-analysis.json)
- [`fia-sink-stage4-whole-plan.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-whole-plan.json)
- [`fia-sink-stage4-whole-plan-analysis.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-whole-plan-analysis.json)
- [`fia-sink-stage4-whole-screen.json`](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage4-whole-screen.json)
- [`stage4-fd-source-audit.md`](../../../../outputs/thesis_incremental_plan/stage4-fd-source-audit.md)

The parent workspace retains both complete preparation reruns, the whole-plan
formal and screen runs, launch manifests and raw logs under
`outputs/thesis_incremental_plan/stage4-evidence/`.
