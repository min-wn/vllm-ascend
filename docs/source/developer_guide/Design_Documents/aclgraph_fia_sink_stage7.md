# FIA sink stage 7: chronological dynamic-trace native policy comparison

Status: the protocol, probe and CPU contract tests are implemented. Native
screening and the formal fresh-process experiment are in progress; formal
measurements and independent protocol qualification are pending. Counts in
the protocol section describe the planned experiment, not completed evidence.

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
request-graph splitting. A positive result would support this native policy
on a complete dynamic descriptor trace; it would not establish superiority
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
helpers remain unchanged.

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

| Planned formal quantity | Count |
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

The formal destination names in this example are reserved for completed
evidence. They do not imply that those results currently exist or pass.
The analyzer must qualify source/native provenance, original record order,
input matching, captures, all pre/post checks, per-pass warmup and complete
timing pairs before a performance result is accepted. Outliers and negative
results remain in the aggregate. Reductions compare matched aggregate times,
not the mean of per-sample percentage ratios. AB/BA order reduces imbalance
without eliminating process drift.

## Results pending

The formal result is not available yet. This section must be updated from
the completed suite and independent analysis before making a stage-seven
performance claim. Required result fields are:

- Formal source snapshot, suite path, raw report hashes, analyzer path and
  protocol qualification status, including any reported issues.
- Observed timing/pair counts and correctness counts against the declared
  3072 observations, 1536 pairs, 2048 checks and 6144 FP32 samples.
- Complete wall and queue means for both policies, matched absolute and
  percentage differences, wins/losses and every process-round result.
- Per-capacity and transition-group means, pass-start behavior, retained
  outliers and tail regressions, with sample-count limits stated.
- Sampled FP32 maximum error, complete input/query/mask identity matches,
  descriptor stability and any repeatability diagnostics that differ.

The next gate depends on that result. A passing complete descriptor-trace
comparison would justify evaluating an integration with real model state;
it would not establish that integration's gain in advance. A failed or
regressing comparison requires preserving the evidence and investigating
the affected states before broadening scope. Full-model replay, a matched
padding control and serving-level measurements retain separate correctness
and attribution requirements.
