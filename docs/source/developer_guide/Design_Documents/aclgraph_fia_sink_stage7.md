# FIA sink stage 7: chronological dynamic-trace native policy comparison

Status: the complete chronological descriptor-trace experiment and both
protocol audits are complete. All eight fresh-process runs pass the declared
checks. Whole-row assignment reduces the producer-inclusive wall mean by
1.788% over 1536 matched pairs; three process rounds improve and one regresses.
Overall wall P99 regresses. This is a sequential synthetic attention-chain
result, not a full-model, split-versus-padding or serving-level speedup.

## Question and comparison boundary

Stage six established a bounded producer-inclusive mean reduction for a
native whole-row assignment policy on four selected original records. It
also retained a regressing process round and a capacity-specific tail
regression. Stage seven asks whether the same executable native policy
retains a benefit when every original batch, length and page-table transition
in the chronological 128-step trace is replayed.

The controls preserve the qualified stage-six native libraries, consumer,
metadata ABI and synthetic private Q/K/V. They differ in the native producer
policy alone:

| Policy | Native work assignment |
| --- | --- |
| `upstream` | Original native `SplitCore`, with its FD split/reduction schedule. |
| `whole` | Guarded contiguous complete-row assignment, with zero FD reduction. |

Each graph contains one metadata producer followed by 16 sequential private
attention occurrences. These occurrences share current descriptors, but each
owns private Q/K/V. They are not concurrent streams, a split full model, or
request-graph splitting. The result supports this native policy on the
declared complete dynamic descriptor trace. It does not establish superiority
over padding or an end-to-end split-parallel inference improvement.

The policy changes complete-row assignment and FD removal together. The
comparison measures their combined effect and producer cost. It does not
attribute a measured difference solely to removal of the FD merge.

## Frozen inputs and native identity

The canonical digest remains
`0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524`.
The probe preserves every original trace record in chronological step order
zero through 127. It retains all 17 required original default capture sizes
and checks the archived default catalog against the installed model
configuration. No extra size is introduced. Model configuration is read;
weights are not loaded.

The qualified geometry remains Q4, 12 query heads, two KV heads, D256,
page128 and sparse mode three. True per-step KV lengths, request ownership
and page routing follow the canonical trace. Q/K/V contents are synthetic,
fixed and resident; this experiment does not produce new Q/K/V from a model
or append actual generated tokens to the cache. Chronological descriptor
changes therefore exercise native dynamic replay without simulating a
complete evolving language-model state.

| Native artifact | SHA256 |
| --- | --- |
| Upstream `libfia_v2_sink_aicpu_kernels.so` | `796983827bc6326a250009a2aacc810c44f65545ec2ae56dab517a5ec4e7111b` |
| Whole-row `libthesis_stage5_whole_aicpu.so` | `1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90` |
| Common consumer OpAPI | `2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40` |

The configured AICPU dispatch name and selected native library hash are
verified before Torch initialization. Every process report retains source
hashes, the extension hash, vendor inventory, loaded-symbol provenance and
the final native-library mappings. The stage-six capture and native-policy
helpers remain unchanged. The deployed source snapshot is commit
`7f73f2c08`, with probe SHA256
`ae985c186290e5b0ff478488b49c66a984317ed2f4d09a65f4df195b6e0b2604`.

## Predeclared chronological protocol

Four rounds alternate policy order, with a fresh interpreter for each policy:

| Round | Fresh-process order |
| --- | --- |
| 0 | `upstream`, then `whole` |
| 1 | `whole`, then `upstream` |
| 2 | `upstream`, then `whole` |
| 3 | `whole`, then `upstream` |

Each process owns 16 private Q/K/V occurrences and 17 bucket graphs. A bucket
is initialized with its first original trace record. Every bucket owns an
independent graph pool, exact-size descriptor storage, query clones and
fixed-address outputs and metadata. Bucket graphs share an occurrence's
resident KV storage; different occurrences must have disjoint Q/K/V storage.
The independent pools avoid relying on a shared-pool capture-order contract
when bucket sizes change during chronological replay.

The formal process first completes a chronological 128-step qualification
pass. It then performs three timed passes, each immediately preceded by a
complete chronological trace warmup outside timing. No pass rotates,
shuffles or groups records by capacity. A separate chronological qualification
pass follows all timing. At each step the pinned descriptor source remains
owned until the final device join, then becomes available for the next fill.

| Predeclared and observed formal quantity | Count |
| --- | ---: |
| Fresh policy processes | 8 |
| Timed passes per process | 3 |
| Timed observations per process | 384 |
| Total timed observations | 3072 |
| Matched `(round, repeat, step)` policy pairs | 1536 |
| Separate qualification passes per process | 2 |
| Total qualification step checks | 2048 |
| Sampled FP32 live-row comparisons | 6144 |

The 1536 pairs are repeated observations nested in eight processes and four
process rounds. They are not 1536 independent process trials. The 3072
warmup replays and 2048 qualification replays are outside formal timing.

The separate `--screen` gate uses only the first eight chronological records,
one occurrence, one repeat and one process round. It captures the buckets
actually used by that prefix and retains independent pools. A prefix screen
cannot qualify the complete trace or supply formal performance statistics.

Device zero, `TASK_QUEUE_ENABLE=0` and disabled batch-invariant mode apply
to both policies. A process ends before the next policy process starts.
Launchers forward cancellation to their owned process groups and join child
cleanup before returning. Every run uses a new output directory.

## Timed endpoint and transition grouping

The timed wall endpoint starts before pinned descriptor fill and ends after
graph replay and the final device/event joins. It includes current real
KV-length/page-table H2D, native metadata production, all 16 attention
occurrences, host submission and final synchronization. Invariant cumulative
Q4 endpoints are uploaded during setup for both policies and remain fixed
for a bucket. The native producer executes inside every graph replay.

| Field | Interpretation |
| --- | --- |
| `wall_us` | Complete declared attention-chain endpoint. |
| `preparation_enqueue_us` | Host fill and copy-enqueue interval, not completed H2D duration. |
| `submit_join_cpu_us` | Host submission interval, which can overlap device execution. |
| `queue_span_us` | Replay-stream events including dependency waits and host submission gaps. |

Host intervals and queue span overlap and cannot be added to reconstruct
wall time. Queue span is not pure kernel duration. Oracle gathers, D2H,
FP32 calculations, output/metadata readback, byte hashes and report writes
are outside timed intervals. The endpoint excludes QKV projection, actual
KV writes, other model layers, sampling, scheduling and serving queues.

Every sample retains the preceding record's capacity and the true capacity
change flag. Step zero is labeled `pass_start` and uses the last record as
its preceding capacity, matching the completed warmup. Later steps are
grouped as `capacity_transition` or `same_capacity`. Pass boundaries remain
a separate group even when their capacity-change flag is true. Independent
analysis can also reconstruct membership, KV-length and page-route changes
from the canonical records, outside the measured preparation path.

## Correctness and matching evidence

Before attention warmup/capture for each bucket, a standalone native metadata
call preserves all 1024 words and validates resource limits. Whole-row
metadata must additionally pass complete physical coverage, advancing
canonical endpoints, zero per-core FD fields and zero reduction counts.
Fallback to upstream FD behavior fails whole-row qualification.

Each of the separate pre/post qualification passes replays all 128 original
steps in order. Each step checks the following:

- The complete current 1024-word plan is retained. Both policies must report
  the current exact maximum Q/KV lengths and physical descriptor dimensions.
  The whole policy also repeats the structural coverage validation.
- All 16 private outputs must be finite, and all physical KV0 padding rows
  must be written as exact zero. Output buffers are neither externally cleared
  nor trimmed to manufacture this property.
- Input, output and metadata addresses, shapes and strides match capture-time
  descriptors. The exact cumulative Q, real KV lengths and current page table
  remain unchanged by replay.
- Three live request rows, zero, middle and last, are compared against the
  independent FP32 reference at occurrence `step % 16`. Every occurrence is
  sampled at eight trace steps per qualification pass. The tolerance remains
  `rtol=0.03`, `atol=0.003`.

The FP32 reference gathers only the current row's real page prefix on the
cache device before D2H. It preserves page order and repeated physical page
IDs. A local sequential page table routes the gathered BF16 K/V for the CPU
reference, which truncates to the true KV length and applies right-down
causality. Reports preserve original page IDs, local page IDs and exact KV
length so that gathering cannot silently substitute a different sequence.

Complete byte identities cover every base Q/K/V and the causal mask before
capture and after the experiment. Every captured bucket's query clones must
match the corresponding base-query prefixes immediately after capture;
complete per-bucket query hashes are then compared before and after the
qualification/timing sequence. Independent analysis matches these complete
identities across policy processes. A common random seed alone is
insufficient input-matching evidence.

Pre/post output and plan hashes are retained as repeatability diagnostics.
Equal plan hashes at different steps can be valid when exact metadata and
cuts remain equivalent. Qualification does not require each transition to
produce a new hash. Output equality across policies is also not required;
sampled numerical acceptance uses the independent FP32 tolerance.

The formal 6144 FP32 comparisons belong to separate qualification passes.
The 3072 timed observations are not each independently FP32 checked. These
checks sample selected live rows and do not prove full-model correctness,
all-element reference agreement or every possible dynamic length pattern.

## Reproduction and evidence locations

Use the frozen probe, suite and adjacent unchanged helpers in the isolated
experiment tree, after sourcing the established Ascend environment. Select
an existing frozen tools directory and a new suite output directory:

```bash
experiment_root=/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e
tools_dir="$experiment_root/stage7-tools-FROZEN_RUN"

python "$tools_dir/aclgraph_fia_sink_stage7_suite.py" \
  --root "$experiment_root" \
  --upstream-vendor "$experiment_root/opp/packages/vendors/thesis_fia_stage2_transformer" \
  --whole-vendor "$experiment_root/stage5-no-fd/relinked-v3/opp/packages/vendors/materialized-baseline-v3" \
  --output-dir "$experiment_root/stage7-suite-new-run" \
  --trace "$tools_dir/structural-dynamic-trace-v1.json" \
  --selection "$tools_dir/q4-tile-full-replay-v8-queue0.rank0.json" \
  --rounds 4 --repeats 3 --seed 1703 --timeout 1800
```

`FROZEN_RUN` denotes the deployed directory whose source hashes are preserved
in the reports. Add `--screen` for the separate bounded prefix gate.
Each direct probe accepts the same paths with `--opp-root`, the selected
`--policy upstream|whole` and `--trial ROUND`. Its raw report name is
`stage7-dynamic-probe.json`; the suite report is `stage7-suite.json`.

Copy the complete suite directory with its relative raw report paths and
logs before independent analysis. Use the matching local source snapshot:

```bash
python benchmarks/aclgraph_fia_sink_stage7_analyze.py \
  --input benchmarks/results/multi_graph_structural/fia-sink-stage7-v1/stage7-suite.json \
  --output benchmarks/results/multi_graph_structural/fia-sink-stage7-analysis.json
```

The analyzer qualifies source/native provenance, original record order,
input matching, captures, all pre/post checks, per-pass warmup and complete
timing pairs before accepting a performance comparison. The separate audit
reconstructs canonical ownership, exact lengths, pages and capacity dispatch
from the original trace and archived selection without importing the runtime
or the main analyzer. Both formal audits pass without reported issues.
Outliers and negative results remain in the aggregate. Reductions compare
matched aggregate times, not the mean of per-sample percentage ratios.
AB/BA order reduces imbalance without eliminating process drift.

Repository evidence is preserved at these locations:

- [Formal suite and all eight relative raw process reports](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage7-v1/stage7-suite.json),
  including launch/finish identity, raw byte hashes and process logs.
- [Protocol audit and paired statistics](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage7-analysis.json),
  including every matched sample and slower counterexamples.
- [Independent formal audit](../../../../benchmarks/results/multi_graph_structural/fia-sink-stage7-independent-audit.json),
  with independently reconstructed descriptor semantics and aggregate results.

The suite report SHA256 is
`b60231d2f6ef54d20373ea5dcccdb6d3039a0186e9550509d640b7f47a95b94c`.
The deployed directory is `stage7-tools-formal-v1` inside the isolated root.
The analyzer and independent audit agree on the complete observation counts,
paired wall means, every process round and every capacity.

## Complete descriptor-trace results

All 3072 timed observations are retained, yielding 1536 matched
`(round, repeat, step)` pairs. Whole-row wall time is lower in 1283 pairs
and higher in 253, with no equal pairs. Positive reduction means `whole`
is lower; all quantities below include the declared preparation and native
producer endpoint rather than subtracting an estimated planner cost.

| Endpoint mean | Upstream, ms | Whole-row, ms | Upstream minus whole, ms | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Complete wall | 4.996977 | 4.907623 | 0.089354 | 1.788% |
| Replay queue span | 4.719516 | 4.626271 | 0.093245 | 1.976% |
| Preparation enqueue | 0.210809 | 0.215181 | -0.004372 | -2.074% |
| Host submission interval | 0.210657 | 0.204284 | 0.006373 | 3.025% |

The host and queue quantities overlap and must not be added together. The
complete wall mean improves by 89.354484 us. Its P50 falls from 5.013780 ms
to 4.933561 ms, but P95 changes from 6.216767 ms to 6.217488 ms and P99
regresses from 6.550212 ms to 6.631752 ms. A mean improvement is therefore
not evidence of a tail-latency improvement.

Every round has 384 matched pairs:

| Round | Order | Upstream wall mean, ms | Whole-row wall mean, ms | Reduction | Whole faster / slower |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Upstream, whole | 5.115425 | 5.015291 | 1.957% | 322 / 62 |
| 1 | Whole, upstream | 4.905102 | 4.802917 | 2.083% | 354 / 30 |
| 2 | Upstream, whole | 5.043272 | 5.049174 | -0.117% | 232 / 152 |
| 3 | Whole, upstream | 4.924110 | 4.763110 | 3.270% | 375 / 9 |

Three rounds improve and round two regresses by 5.901464 us. Only four
process rounds are available, with repeated dependent observations inside
each round. These results do not establish a universal or stable gain across
process conditions. The complete paired records remain available for
examining process/time drift instead of removing unfavorable samples.

All 17 original capacities have positive aggregate wall differences, with
unequal weights set by their frequency in the unchanged trace:

| Physical capacity | Pairs | Upstream mean, ms | Whole-row mean, ms | Reduction | Whole faster / slower |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 36 | 3.722634 | 3.610470 | 3.013% | 35 / 1 |
| 272 | 84 | 3.813563 | 3.646923 | 4.370% | 83 / 1 |
| 288 | 132 | 4.016462 | 3.896497 | 2.987% | 129 / 3 |
| 304 | 84 | 4.143832 | 3.995385 | 3.582% | 81 / 3 |
| 320 | 72 | 4.424943 | 4.236408 | 4.261% | 63 / 9 |
| 336 | 60 | 4.448185 | 4.352777 | 2.145% | 54 / 6 |
| 352 | 96 | 4.423718 | 4.309686 | 2.578% | 90 / 6 |
| 368 | 72 | 4.851507 | 4.756146 | 1.966% | 61 / 11 |
| 384 | 72 | 4.745281 | 4.624336 | 2.549% | 67 / 5 |
| 400 | 72 | 4.854262 | 4.759440 | 1.953% | 65 / 7 |
| 416 | 108 | 5.227126 | 5.153066 | 1.417% | 91 / 17 |
| 432 | 72 | 5.354135 | 5.310764 | 0.810% | 50 / 22 |
| 448 | 84 | 5.560336 | 5.484301 | 1.367% | 64 / 20 |
| 464 | 156 | 5.638336 | 5.584449 | 0.956% | 116 / 40 |
| 480 | 108 | 5.742718 | 5.693353 | 0.860% | 81 / 27 |
| 496 | 120 | 5.867791 | 5.822915 | 0.765% | 90 / 30 |
| 512 | 108 | 6.213744 | 6.198654 | 0.243% | 63 / 45 |

The largest capacities expose a smaller aggregate benefit, reaching only
15.089602 us at capacity512. These are descriptive results of current
lengths, row composition and process conditions, not evidence that capacity
alone causes the difference. Capacity256 P99 also worsens from 4.025353 ms
to 5.977989 ms with only 36 observations per policy. Small capacity groups
cannot establish stable P99 behavior.

The transition grouping preserves pass starts separately:

| Transition class | Pairs | Upstream mean, ms | Whole-row mean, ms | Reduction | Whole faster / slower |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pass_start` | 12 | 4.665015 | 4.609778 | 1.184% | 11 / 1 |
| `capacity_transition` | 1392 | 4.990342 | 4.899921 | 1.812% | 1162 / 230 |
| `same_capacity` | 132 | 5.097128 | 5.015914 | 1.593% | 110 / 22 |

The original trace has 116 within-pass capacity changes and 11 within-pass
same-capacity transitions, plus its separate pass-start boundary. Their
timed multiplicities are 12 per canonical step across four rounds and three
passes. Membership, lengths and pages can change without a capacity change;
the `same_capacity` group is not a static-input control.

The largest retained wall loss is round two, repeat zero, step94 at
capacity256: whole-row is 3375.742 us slower even though its queue span is
52.260 us lower. That disagreement illustrates why queue duration alone
cannot establish complete-endpoint improvement or identify the cause of a
wall outlier. No outlier is excluded from these means or percentiles.

## Correctness outcomes and interpretation

All 2048 chronological qualification step checks and all 6144 sampled FP32
row comparisons pass. Maximum sampled absolute error is
`0.0003158003091812134`, within `rtol=0.03`, `atol=0.003`. All private outputs
are finite; physical KV0 outputs are exact zero; current exact descriptors
remain read-only and input/output/metadata addresses remain stable.

The retained metadata evidence contains 2184 complete 1024-word payloads:
136 standalone pre-capture payloads and 2048 pre/post step payloads. Every
whole-row payload passes the full structural no-FD contract. Current exact
base fields are checked for every qualification step under both policies.
The audits report no same-step pre/post differences among 1024 output-hash
pairs or among 1024 metadata-hash pairs. This is repeatability within each
process; different steps need not have different plan hashes, and policy
outputs are accepted numerically rather than required to match bitwise.

Complete Q/K/V and mask identities match across policy processes and remain
unchanged after execution. All 17 buckets' query clones match their base
prefixes after capture and retain their bytes through qualification and
timing. These checks qualify the declared fixed synthetic data and current
descriptor routing; they do not turn it into a real KV-update or generated
token trajectory. The sampled FP32 checks are separate from timing, not an
individual correctness check of all 3072 timed observations.

The complete-trace reduction is 1.788%, compared with the stage-six bounded
four-record reduction of 5.665%. The accepted observed benefit is narrower
than that earlier bounded result. The two experiments have different record
and capacity weights, temporal order, warmup/repeat protocols and process
conditions. Their aggregate difference is not a controlled causal estimate
of a degradation caused solely by dynamic transitions or broader sizes.
The useful performance claim is the internal matched comparison in this
stage, including its negative round, outliers and tail behavior.

The scoped Stage7 CPU regression reports 62 passing tests. Markdown and
spelling checks pass for this document. Whole-repository CI was attempted;
its existing unrelated failures remain recorded in the task checks, so this
stage does not claim a clean complete-repository CI run.

## Next gate

Stage eight should begin with a narrow real-model integration correctness
gate and a complete latency budget. It must map the qualified metadata
producer and unchanged consumer onto real Q/K/V generation and cache writes,
audit continued hidden/residual and physical states, and preserve exact
lengths, default buckets and original TP ownership. A bounded initial
integration must establish that the required operator interface and graph
workspace contract remain valid before timing a large model matrix.

The budget must include real descriptor preparation, native planning,
attention, any synchronization or adaptation cost and the final model join.
An attention-chain saving of 89.354484 us cannot simply be multiplied across
model layers or declared as end-to-end gain. Only after correctness passes
and the measured budget shows a useful exposed saving should evaluation
broaden to complete dynamic model replay and a matched padding control.
Serving throughput remains a later endpoint with separate correctness and
attribution requirements.
