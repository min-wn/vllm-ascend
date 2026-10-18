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
- `gdn-q4-log-v1-rejected.log`: first compact Q4 native version fails exact
  repeated eager validity flags. It supplies no accepted timing result.
- `gdn-q4-log-v2.json`: padded per-task validity records and DMA publication;
  strict continued checks, canonical native candidate export/commit, complete
  compact endpoint and an all-candidate-export-inclusive control. This is
  still a one-device recurrence, not a model or split win.
- `structural-dynamic-trace-v1.json`: predeclared 128-step random request
  trajectory. This is an input manifest; it has not yet been executed by an
  integrated engine benchmark. Its canonical `trace_digest` is
  `0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`.

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
