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

The original reports also remain under
`/workspace/multi-graph-results/tp-boundary/` on `vllm-ascend-030`. Exact tested
prototype sources and shared libraries are archived separately there; build
products are not versioned in this repository.
