# FIA sink graph-catalog context bridge

Status: four-process bounded diagnostic completed, 2026-10-19. All eighty
timings and 144 sampled FP32 rows pass the independent evidence audit.
This is a context screen, not a new scheduling algorithm or complete model
performance experiment.

## Motivation

The [current-consumer diagnostic](aclgraph_fia_sink_consumer_diagnostic.md)
repeats the original step0/B98/capacity400 and step29/B128/capacity512.
Whole-row reduces wall by 9.418%/10.095% against upstream there. The complete
Stage 7 chronological descriptor trace improves only 1.788%; even the same
two records improve only 1.184%/0.332% in that trace context. Exact Q/K/V,
frame queries and all native plan words match across those experiments.
Record-frequency weighting therefore cannot explain the discrepancy alone.

The earlier experiments also differ in graph catalogs, capture allocations,
warmup and replay history, process/time conditions and working set. A larger
reserved allocation is not evidence that memory bytes caused the difference.
Before changing kernel arithmetic, separate catalog construction/retention
context from the repeated two-record execution protocol.

## Fixed four-process screen

Run upstream with two buckets, whole with two, upstream with seventeen and
whole with seventeen, in that order, on NPU0 only. Each is a fresh process
with sixteen private resident attention occurrences, seed1703 and the same
isolated vendor/extension provenance as the previous diagnostic. This single
four-cell block is a mechanism screen and cannot estimate robust process
drift. Repeated samples are not independent fresh-process replications.

The two-bucket catalog is `[400, 512]`. The seventeen-bucket catalog retains
the original chronological first-use order:

```text
400,304,288,352,272,496,464,480,320,448,384,368,432,256,512,336,416
```

Every bucket uses its original first-use descriptor and an independent graph
pool, with all owners alive until the experiment ends. Keep the installed
default catalog and canonical trace unchanged. Do not introduce new sizes.
Capture/preflight of unused buckets belongs to the declared setup context;
unused graphs are not additionally replayed during the common measured
protocol.

Graph capture records operations without executing them. Unused graph
metadata/output contents are therefore not semantically initialized. Catalog
validation checks their storage addresses, exact device descriptors and the
independently executed eager preflight plan, without reading unexecuted graph
output bytes. The two measured records acquire valid graph plans through
their ordinary qualification/replay path; their post-check reads those actual
plans. Setup still includes the retained consumer's eager warm operations.

All four processes use exactly the original consumer diagnostic's trial-zero
record order, five warmups and ten observations per record. Only step0 and
step29 participate in pre/post correctness qualification and measured
replay. There are eighty total timings and 144 sampled FP32 row comparisons.
No profiler runs are needed. Preparation, required H2D, the native producer,
replay and final completion remain inside the same complete wall endpoint.
Catalog setup and validation are outside timing in both controls.

## Attribution and correctness gates

- Preserve every source, trace, vendor/extension hash, graph pool/address and
  capture-order record; keep all outputs and private caches independently owned.
- Full resident Q/K/V, mask and every catalog bucket's query bytes must remain
  unchanged. For the two common records, query bytes match across all four
  processes; same-policy full native plans must match across catalogs.
- Both common records retain finite outputs across all sixteen occurrences,
  exact KV0 zeros, real ordered page/tail reference gathers, read-only exact
  descriptors and stable pointers. Preserve full 1024-word plans before and
  after measurement. Whole-row retains complete physical coverage and no FD.
- Report pinned descriptor preparation counts separately: measured buckets
  have seventeen step copies each (pre/post, five warmups, ten observations);
  unused buckets have setup only and no hidden step replay.
- Preserve all eighty wall observations and the fixed record schedule. Pair
  by policy/catalog, repeat and original step; do not silently rotate records
  according to cell index.

## Interpretation fixed before measurement

Compare whole/upstream wall differences independently for each catalog,
then the difference of those policy differences. Keep per-capacity and all
raw observations. Do not use profiling durations or earlier runs as extra
samples for this screen.

If the seventeen-bucket repeated-record result retains the large reduction,
catalog count/construction alone is insufficient; the next control is
chronological replay history. If it collapses, catalog construction/retention
context matters, but this does not distinguish allocation placement, capture
history, device state or memory capacity. An unstable or marginal effect
requires a reversed block before any stronger conclusion. No outcome
establishes full-model gain, split superiority or algorithmic novelty.

The kernel candidate remains separate: retain running output in UB under
the whole-head ownership/lifetime contract, with explicit slot and UB-budget
checks. This bridge does not implement or validate that change.

## Observed result

All four selected processes complete with identical frozen source. The same
private base Q/K/V and common-bucket queries match across all cells. Same-policy
complete native plans match across catalogs. No unused captured graph is
replayed. Pre/post qualification passes all sixteen outputs for finite values
and KV0 zeros; 144 sampled FP32 rows have maximum absolute error
0.00029981881380081177. NPU0 is idle after completion.

| Catalog | Original record | Upstream wall us | Whole wall us | Saved us | Reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| 2 | step0/B98/capacity400 | 4930.206 | 4630.140 | 300.067 | 6.086% |
| 2 | step29/B128/capacity512 | 6283.276 | 5943.937 | 339.340 | 5.401% |
| 17 | step0/B98/capacity400 | 4836.580 | 4469.990 | 366.590 | 7.580% |
| 17 | step29/B128/capacity512 | 6082.302 | 5576.469 | 505.833 | 8.316% |

Each row contains ten observations per policy. Capacity denotes physical
Q4 tokens: capacity400 has 100 physical requests, and capacity512 has 128.
Both catalogs retain the original capture sizes and physical padding.

The larger catalog allocates/reserves 6,789,984,256/11,632,902,144 bytes,
versus 5,669,344,768/6,400,507,904 for two buckets, matching the earlier
experiment footprints. The policy difference increases by 66.523 us at
capacity400 and 166.494 us at capacity512 in this block. This does not
support catalog count or retained allocation size alone as the explanation
for Stage 7's much smaller observed gain. It does not prove a benefit from
more graphs: cell order, process conditions and allocation context remain
confounded, and each cell has only one fresh process.

The concurrent two-bucket gains are also smaller than the previous
9.418%/10.095%, so the historical magnitude is not stable across runs.
Seventeen buckets do not collapse to Stage 7's same-record 1.184%/0.332%
in this repeated-record protocol. Chronological descriptor/replay history
and process/device drift remain unresolved. A future narrow history control
must keep the seventeen-bucket catalog and its capture order fixed, alter
only the intervening replay sequence and preserve every current descriptor.
This experiment does not establish either history or frequency as the cause.

The practical decision is to stop treating graph-memory size as a confirmed
bottleneck. The measured consumer execution gap and explicit workspace
accumulation round trips remain a concrete, separately testable kernel
target. Required preparation and native planning stay inside every endpoint.
Stage 7's 1.788% complete dynamic-descriptor-trace result, including its
negative round and P99, remains the applicable dynamic-trace evidence.

## Audit, failed attempt and reproduction

The selected `catalog-bridge-v2` bundle contains 52 raw files. An independent
stdlib audit performs 27,731 assertions over source/native identity, exact
trace reconstruction, all eighty samples, 144 FP32 row identities, 130 full
plan payloads and descriptor/address/byte contracts. Only 24 of those plan
checks refer to actual captured-producer replay output; the others identify
executed eager preflight explicitly. The source tests contain 23 probe and
19 suite cases; scoped Ruff, spelling and Markdown checks pass. Full-repository
CI still has retained vendor/raw-log spelling, pre-existing Triton import and
Windows hook/tooling failures; the original log is saved.

The initial `catalog-bridge-v1` attempt failed before any timed sample because
its audit tried to interpret unexecuted graph metadata. Its raw reports and
exact source are retained under `failed-premeasurement-v1/`. The validation
fix adds no extra graph replay and changes no frozen preparation, consumer,
producer or timing helper. No failed-attempt sample enters the table.

Saved repository artifacts:

- `benchmarks/results/multi_graph_structural/fia-sink-catalog-bridge-v1/`:
  raw suite/reports, deployed source, manifests, failed attempt and audit snapshot.
- `fia-sink-catalog-bridge-analysis.json`: complete qualified audit and summaries.
- `fia-sink-catalog-bridge-protocol-review.json`: the predeclared independent
  design review and limits on causal interpretation.
- `fia-sink-catalog-bridge-checks.json`: code hashes, tests and quality results.

The audit snapshot accepts the collected `catalog-bridge-suite.json` and an
explicit `--repo` checkout path; choose a new `--output` outside manifested
raw evidence. No NPU is needed to reproduce its checks and numerical summary.
