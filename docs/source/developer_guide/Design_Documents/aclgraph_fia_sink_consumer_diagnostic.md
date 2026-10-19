# Current FIA consumer bottleneck diagnostic

Status: protocol and harness in development, 2026-10-19. Native measurements
are pending. This is a small prerequisite diagnostic, not Stage 8 full-model
integration or a replacement result for the frozen Stage 7 trace.

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

## Existing-work boundary

Plan/run, device lengths and shared plans are existing mechanisms. This
diagnostic evaluates the current guarded native assignment and adapted
consumer; it adds no native incremental planner, new attention arithmetic,
multi-stream algorithm or literature novelty claim. Keep legacy-to-sink
adoption effects separate from upstream-to-whole policy effects.

See [Stage 7](aclgraph_fia_sink_stage7.md), the
[contribution audit](aclgraph_fia_sink_contribution_scope.md) and the
[conditional model-integration protocol](aclgraph_fia_sink_stage8_protocol.md).
