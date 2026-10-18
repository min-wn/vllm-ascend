# Structural ACLGraph experiments

These raw reports retain unsuccessful experiments as well as qualified
correctness checks. They do not establish a TP2 decoder performance win.

- `a2-conditional-v1.json`: actual conditional-handle creation returns 207000
  on the current driver/CANN combination. No conditional replay ran.
- `gdn-compact-v1.json`: copied dense, compact reconstruction plus emission,
  and installed official recurrence, including all raw timing samples.
- `gdn-compact-v1-breakdown.json`: the same endpoints plus isolated emission
  and reconstruction diagnostics. The isolated modes omit required work and
  cannot be compared as replacement decoder endpoints.
- `gdn-compact-v2-fused.json`: official-matching V tiling and fused accepted
  reconstruction, including mixed per-request correctness. It remains slower
  than the installed official recurrence; it is not a decoder speedup.
- `gdn-q4-products-v3.json`: qualified BF16-to-FP32 eight-column products,
  including selected-state gather and Q/K packing. This omits solve and states.
- `gdn-q4-block-v2-debug-out2.json`: a rejected native kernel with repeat-eager
  state corruption in V rows 64-76. Product and output repetition passed; this
  failed implementation supplies no performance evidence.
- `gdn-q4-block-v3.json`: corrected DMA dependency, strict dynamic same-schedule
  checks and cross-algorithm error diagnostics. The complete recurrence saves
  about 61 us; it has not met the 135 us gate to justify a full GDN experiment.
- `gdn-q4-block-v3-gqa-fold.json`: one stacked matrix per shared Q/K head;
  complete recurrence saves about 86 us against its paired official control.
  This also remains below the 135 us gate.
- `gdn-q4-direct-v5.json`: direct selected-cache Cube products plus the V3
  solver/materializer. Complete recurrence passes its 135 us investment gate.
- `gdn-q4-direct-queue-v4.json`: same direct products with double-buffered
  candidate output. Full recurrence accounting and strict continued checks.
- `gdn-q4-strided-v5.json`: page-strided state qualification with canaries in
  each page gap; avoids a state-layout copy before real-layer integration.
- `gdn-q4-fp32-v1.json`: actual model state precision and page stride, with
  timed Q/K conversion and all four FP32 candidate writes. Strict checks pass,
  but the complete recurrence saves only about 59 us, below the 135 us gate.
- `gdn-q4-fp32-profile-v2.json`: repeat full endpoint plus isolated products,
  Q/K casts and solver/materializer diagnostics. These component times are
  not additive and do not establish a model or split speedup.
- `structural-hybrid-memory-budget-v1.json`: actual checkpoint-header and
  config inventory plus unchanged trajectory byte estimates for TP2/4/8.
  This performs no model loading or NPU allocation and is not OOM evidence.
- `structural-hybrid-memory-budget-v2-compact.json`: preserves the original
  dense estimates and adds direct compact allocation, two banks and products
  scratch budgets. TP2's named ideal subtotal is 41.76 GiB before products
  and unknown runtime overhead; actual full-model fit remains unverified.
- `gdn-q4-log-v1-rejected.log`: first compact Q4 native version fails exact
  repeated eager validity flags. It supplies no accepted timing result.
- `gdn-q4-log-v2.json`: padded per-task validity records and DMA publication;
  strict continued checks, canonical native candidate export/commit, complete
  compact endpoint and an all-candidate-export-inclusive control. This is
  still a one-device recurrence, not a model or split win.
- `gdn-q4-log-pipeline-v3.json`: fixed global request ranges; full compact,
  partitioned serial and partitioned pipeline controls. The pipeline's net
  recurrence gain is 4.50% against full compact; this is not full replay.
- `gdn-q4-log-fixed-trace-v4.json`: 128 records and 641 correctness checks
  for the immutable trace's request/acceptance/input-seed histories. This
  does not execute model sequence lengths or default bucket dispatch. Its
  timing rows are a separate B98 recurrence fixture, not trace averages.
- `gdn-q4-log-real-layer-v1-rejected.json`: official inactive hidden output
  is unspecified and contains NaNs; this failed repetition has no timing win.
- `gdn-q4-log-real-layer-v2.json`: complete actual GDN layer with TP2/MLP,
  strict same-schedule state/output checks and common inactive output restore.
  Splitting adds 2.18% against equally optimized compact padding in this test.
- `gdn-q4-log-deferred-v1.json`: exact two-phase recurrence/projection causal
  control, including commit/publication and final join. Synthetic projection
  shapes and local state qualify dependencies, not model performance.
- `gdn-q4-log-real-layer-deferred-v3.json`: complete real layer, deferred
  commit launched after gated Norm and joined at the endpoint; exact state
  controls and both-rank raw samples. This public dependency optimization
  improves the layer by 8.38% and also belongs in fair padding.
- `gdn-q4-log-real-layer-combined-v4.json`: both controls use compact state
  and deferred commit. Fixed request-range products/solve overlap adds
  2.64% in the complete real GDN layer; this remains below the final goal.
- `structural-dynamic-trace-v1.json`: predeclared 128-step random request
  trajectory. V8 now executes all steps in complete-model replay timing;
  only the first three continued steps per mode have full state audits.
  This is not a serving-engine benchmark. Its canonical `trace_digest` is
  `0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`.
- `q4-tile-full-replay-v8-queue0.rank0.json` and `.rank1.json`: all 64 real
  decoders, 17 default buckets, all 16 true FIA updates, three 128-step trials
  and initial three-step continued-state audits. Ready-tile public padding is
  143.474 ms mean, 4.586% below delayed. Raw legacy candidate labels do not
  establish splitting; complete correctness remains unqualified.
- `gdn-q4-full-replay-v8-analysis.json`: independent rank-max and statistics
  audit, raw hashes, corrected public-padding attribution and explicit missing
  full-trajectory correctness/split evidence. The companion
  `q4-tile-full-replay-v8-source-manifest.json` identifies the tested sources
  and native libraries, which predate later reporting/checkpoint edits.
- `hybrid-owner-capacity-audit-v1.json`: stable request ownership over the fixed
  trace, default equal-owner buckets and 2.904% additional aggregate capacity.
- `hybrid-owner-dense-v1-smoke.json` and `hybrid-owner-dense-v2.json`: frozen
  recurrence cut with actual owner GDN projections, original full-row TP MLP
  and all AG/RS/exit costs. Two profiles pass 60 hard checks per rank. The
  three-GDN screening budget is negative; these are not connected-layer or
  full-model measurements and do not justify further short-block integration.

The compact recurrence experiments use one NPU, capacity 100, live batches
97-99, four query tokens, and uniform accepted counts. They do not implement
the complete random trajectory or default graph dispatch. Reported speed must
include reconstruction. Required cache export must also be included whenever
a consumer needs materialized candidates.

Related design records:

- [Conditional graph qualification](../../../docs/source/developer_guide/Design_Documents/multi_graph_conditional_capture.md)
- [Compact state qualification](../../../docs/source/developer_guide/Design_Documents/multi_graph_gdn_compact_state.md)
- [Q4 block recurrence](../../../docs/source/developer_guide/Design_Documents/multi_graph_gdn_q4_block.md)

The original reports also remain under
`/workspace/multi-graph-results/tp-boundary/` on `vllm-ascend-030`. Exact tested
prototype sources and shared libraries are archived separately there; build
products are not versioned in this repository.
