# FIA sink graph-catalog context bridge

Status: predeclared bounded diagnostic, 2026-10-19. Measurements are pending.
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
