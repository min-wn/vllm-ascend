# Further split-specific opportunities: phase scheduling and KV work balance

## Scope

This follows the [first causal parallel-direction tests](multi_graph_parallel_directions.md).
The objective is to exploit independent request branches while avoiding the
measured activation and GEMM interference. Adding a stream alone is insufficient.
All runs keep the default graph capacities and real FIA updates after replay
submission. No attention-head partitioning or speculative acceptance tuning
is introduced.

## Two independent GDN scheduling windows

The first window is between the preceding layer's MLP B and the next GDN A.
The earlier projection-gated implementation still ran SwiGLU B concurrently
with the long state core, making the activation approximately 19-21 times
slower. Its GDN output GEMM then competed with Down B.

The new adapter separates the official GDN implementation into input projection,
stateful core plus gated Norm, and output projection. It uses the same registered
stateful operator and metadata, without new state copies or a replacement kernel.
The schedule first completes B's GateUp and activation, projects GDN A, then
overlaps only GDN A's vector phase with Down B and its reduction. GDN A's output
projection runs on root after Down B has been submitted, avoiding the measured
Cube/Cube collision.

The second window is inside the GDN layer itself. In four old profiler samples,
GDN A's output reduction completed 864.290-876.738 us before its MLP GateUp
started. Root was executing the independent GDN B prefix. These intervals are
not an estimate of recoverable latency: most dense computation still needs the
same hardware resources.

The combined schedule projects GDN B, completes MLP A's GateUp and activation,
then overlaps GDN B's vector core with MLP A's Down and reduction. GDN B's output
projection follows on root. GDN collectives change from output A, output B,
Down A, Down B to output A, Down A, output B, Down B, identically on both ranks.
The B output is not ready earlier and held in the communication queue; this
differs from the previously rejected depth-priority experiment.

`--gdn-pipeline --gdn-phases --numerical-assessment` compares:

| Mode | Meaning |
| --- | --- |
| `joined` | Padding with the complete two-layer endpoint |
| `split_mlp_overlap` | Previously qualified split schedule |
| `split_gdn_phased_barrier` | New phase decomposition, without concurrent A core/B Down |
| `split_gdn_zipper` | First independent window only |
| `split_gdn_zipper2` | Both windows, with the GDN collective order above |

Returned hidden states, residuals, all reductions, final joins, dynamic metadata
and state checks remain in scope. The CPU adapter tests directly extract the
repository's official GDN forward and compare all three projection layouts,
FP32/BF16 outputs, state mutations, inactive rows and failure restoration.

## KV workload balance across concurrent FIA branches

The old mixed-length experiment sorted requests to create length clusters. This
experiment instead uses capacity-constrained longest-job-first assignment to
balance total KV work between two simultaneous FIA branches. The pair remains
200 + 200 or 272 + 128; each branch consumes global request IDs and their real
physical cache pages.

`--kv-balance-probe` compares whole FIA, contiguous serial/parallel branches,
and balanced serial/parallel branches. Seeded random lengths form the ordinary
case; a sorted copy of the same histogram is a clustered diagnostic. Modes are
compared within each profile. GPU Q/K/V/gate gathers, output inverse scatter,
stream joins and every real FIA update are included in replay timing. CPU
assignment and metadata upload are reported separately and must not be silently
excluded when judging a dynamic serving implementation.

This endpoint begins with frozen projected Q/K/V and a synthetic prefix. It
does not measure RoPE/projection producers, full decoder replay or generation.
Dynamic B98, B97, B99, B98 checks refresh device permutation indices and host
lengths, compare identical eager/replay output, compare complete KV contents,
and require inactive request pages to remain untouched. Output comparisons
use active rows after restoring canonical request order.

## Quantitative screening of other proposals

The actual Q4 default capture list was enumerated for B64 through B128. The
padding capacities total 25,344 tokens. Choosing the smallest legal two-graph
sum not exceeding each padding bucket reduces that to 25,024: only 1.263%
less total capacity, before extra launches, updates or fragmented GEMMs.
B98 admits 200 + 192 instead of 400 (2% less capacity), while B127 has no legal
two-graph exact sum of 508. This is a work-count bound, not a latency prediction;
default-bucket packing alone has insufficient budget for the requested large
improvement in this range.

Cross-branch coalescing of A Down and B O reductions could reduce collective
startups, a dependency unavailable in the intact whole-batch chain. Current
readiness skew is approximately an MLP duration, much larger than the measured
startup budget, so delaying B O to pack it with A Down would repeat a known
negative result. It is screened out for the current schedule.

The CPU enumeration and exact old profiler intervals are saved as
`legal_capacity_and_gdn_window_screen.json`, with a reproducible script and
source hashes, under `outputs/multi_graph_tp_attention_mlp/`.

## First phase-scheduling result

`tp2-27b-gdn-phases-v1.json` completed all five modes with 243 checks per
rank, four metadata variants and eight continued-state steps. Same-shape
eager/replay, output/state trajectories and inactive/null state checks passed.
Three rounds of six samples, slower rank per sample:

| Mode | Wall replay, us | Device replay, us |
| --- | ---: | ---: |
| Padding | 4832.17 | 4764.46 |
| Previous split overlap | 5092.93 | 5029.79 |
| Phased barrier | 5092.92 | 5025.35 |
| First-window overlap | 5047.46 | 4976.41 |
| Both windows | 5149.75 | 5085.76 |

The first-window overlap saves 45.46 us wall and 48.94 us device against
its phased barrier (0.89% and 0.97%). All three rounds favor it. It remains
4.46% slower than padding. Adding the second window loses 102.29 us wall
against the first-window schedule; a larger readiness window is not evidence
of a larger usable overlap opportunity. Separate traces are retained for
interference and critical-path analysis.

Main source SHA256:
`f5dd0b3d6390f34175ba9b6fb8e33a80edc1ebe5477f957e04d38f196468e82c`.
Phase adapter SHA256:
`eed4e5a7d50a2f74c5626ff5c1bffcc7239c2b7973cc137847eda6f7d4cfa5d3`.

## KV work-balance result

`tp2-27b-kv-balance-2048-v1.json` completed 40 checks per rank across two
profiles, five schedules and four dynamic variants. Lengths range from 256
to 2048, with batch 98 and default capacity 400. Total KV tokens are 109,758.

| Frozen FIA endpoint | Random device, us | Clustered diagnostic device, us |
| --- | ---: | ---: |
| Whole bucket | 783.83 | 765.92 |
| Contiguous serial | 851.20 | 820.55 |
| Contiguous parallel | 852.37 | 812.39 |
| Balanced serial, including GPU reordering | 939.00 | 921.65 |
| Balanced parallel, including GPU reordering | 929.51 | 904.18 |

Random contiguous lane loads were 56,599 and 53,159 KV tokens; balanced loads
were 54,889 and 54,869. Clustered contiguous loads were 75,526 and 34,232.
Even the clustered diagnostic does not turn improved work balance into a win:
balanced parallel remains 18.05% slower than whole. Random balanced parallel
is 18.59% slower than whole. CPU planning adds further cost: roughly 0.38-0.40
ms for whole/contiguous metadata construction and 0.68-0.71 ms for balancing
plus metadata, measured separately. These Python construction costs are not
engine scheduler overhead estimates and are not included in the replay table.

The evidence rejects this implementation as a performance route on the tested
workload. Logical work balance alone does not overcome FIA interference and
GPU permutation cost. No request sorting or assignment policy is enabled.

## Preserve shared dense GDN computation

The phase traces identify a larger avoidable cost: the GDN layer's split
GateUp GEMMs take roughly 729 us in total versus 461 us whole, with split Down
also adding roughly 65 us. These are separate profiler observations, not sums
to subtract from formal replay latency.

`--gdn-core-pipeline` therefore changes the cut boundaries. It keeps the first
layer's qualified split, but starts GDN with one complete input Norm/projection.
Projected tensors are partitioned as views; each state core consumes the same
global request IDs and official metadata. Core B can overlap output A's real
TP reduction. Outputs join before one complete GDN post-Norm/GateUp/activation,
then Down A's reduction overlaps Down B. Thus dense input projection and
GateUp retain whole-bucket shapes while request-independent state and reduction
work remains split.

The same-run whole-GDN split control and padding use the original actual GDN
layer. New serial/pipeline variants share arithmetic, storage joins and phase
decomposition; serial explicitly drains each reduction before continuing.
Their same-shape eager/replay and state trajectories must be bitwise equal.
Whole-GDN versus partitioned-core differences remain numerical diagnostics.

`tp2-27b-gdn-core-pipeline-v1.json` completed 203 checks per rank and the full
dynamic/continued-state assessment. Same-run complete-two-layer results:

| Mode | Wall replay, us | Device replay, us |
| --- | ---: | ---: |
| Padding | 4826.77 | 4764.48 |
| First layer split, whole GDN | 4781.05 | 4712.57 |
| Shared dense GDN, serial core/reductions | 5098.80 | 5027.81 |
| Shared dense GDN, pipelined core/reductions | 4850.28 | 4785.36 |

Pipelining saves 248.52 us wall (4.87%) against identical serial decomposition.
It remains 23.51 us (0.49%) slower than padding and 69.23 us slower than whole
GDN in the split control. All three rounds show the same ordering. The new
boundaries remove the large split GateUp cost, but have not produced a net win.

## Parallel state cores with explicit Vector budgets

An additional comparison under `--gdn-core-concurrency` forks both independent
GDN state cores onto private streams after the full input projection. Root
consumes each completed core output, preserving output/reduction order and
the whole GateUp. The two variants give each core stream either the default
Vector resource limit or 20 Vector cores; the default 20 Cube limit is retained.
The limits persist for the exclusively owned graph streams. API results are
read back and recorded. A runtime limit alone does not prove that a custom
kernel changed its launch dimensions or that resources are physically isolated.

This tests resource distribution made possible by request independence, with
full dense shapes held constant. Eager/replay, both state trajectories, and
serial-core control equality remain strict. A preliminary run failed before
timing because a local import shadowed the enclosing `torch_npu` binding; it
was corrected before the completed comparison, with no performance inference
from that failed run.

`tp2-27b-gdn-core-concurrent-v2.json` completed 235 checks per rank and eight
continued-state steps. Same-run endpoint medians:

| Schedule | Wall replay, us | Device replay, us |
| --- | ---: | ---: |
| Padding | 4803.59 | 4740.31 |
| First layer split, whole GDN | 4775.38 | 4716.44 |
| Sequential state cores with output communication pipeline | 4845.52 | 4779.93 |
| Concurrent state cores, default Vector limit | 5174.30 | 5107.54 |
| Concurrent state cores, 20 Vector cores per stream | 5487.51 | 5423.43 |

The runtime confirmed both core streams had 20 Cube/40 Vector by default and
20 Cube/20 Vector in the budgeted case. Both concurrent variants are slower
than the sequential-core pipeline in every round. The default-limit concurrent
round medians range from 4933.83 to 5228.37 us and are substantially less stable
than the sequential pipeline's 4835.16-4850.73 us. Vector partitioning is not
a performance improvement in this workload. These findings do not prove that
every custom kernel honored a specific physical core partition.

## What the new boundaries actually remove

The core-pipeline v1 separate profiler records show that full GateUp is retained:
one rank's second replay measured 462.41 us in the whole-GDN control and
460.51 us in the pipeline. Full recurrence was 983.86 us; the two sequential
core branches total 997.78 us. Thus the previous large GateUp fragmentation
cost has been removed, and recurrence work does not materially shrink merely
by partitioning requests.

The remaining costs are concrete. In that sample, GDN O projection changes
from 88.70 us to 87.12 + 81.36 us, while GDN Down changes from 220.84 us to
146.26 + 99.28 us. The pipeline hides output A's reduction behind core B,
but output B's reduction still has an approximately 143 us exposed tail.
Down A retains an approximately 54 us uncovered remainder, followed by Down B's
approximately 129 us tail. These are annotated profiler intervals; they cannot
be summed as a predicted saving against formal latency.

This identifies why neither opening readiness windows nor preserving GateUp
alone establishes a large split win. A future operator implementation would
need to remove the fragmented output-projection cost while publishing reduced
branch outputs early enough to shorten those exposed tails. A separate-stream
change alone does not remove this work.

The two-rank trace inventory additionally identifies approximately 74 us of
extra copy/slice/concat kernel duration and 33-37 us of extra GDN norm/gating
duration in the split-core path. The core recurrence itself adds only 14-17 us.
The pipeline reduces the union of GDN communication intervals not overlapped
by actual AI kernels by approximately 186-224 us. The useful communication
overlap is real; peripheral preparation and output projections consume it.

## Validation and retained evidence

The final focused CPU suite passed 172 tests. Scoped Ruff, Markdown and
codespell checks passed. Full `bash format.sh ci` was attempted and retained
the previously recorded unrelated spelling/import findings and Windows tool
gaps; see `parallel-phases-format-ci.log`. No negative-result schedule is
enabled in production.

Raw reports, logs, both-rank traces, interval analysis scripts and exact tested
source snapshots are saved under `outputs/multi_graph_tp_attention_mlp/` and
the remote `/workspace/multi-graph-results/tp-boundary/` directory. The final
concurrent-core report's main source SHA256 is
`2b53fb16b112e63533cc1b4e90403a19b96f41834c628522af184226eed404cc`.
The committed source additionally records the stream-budget helper hash in
future reports; this reporting-only addition was made locally after deployment
and does not alter the measured schedule. The tested snapshot is preserved.

The subsequent [larger-opportunity investigation](multi_graph_parallel_research_next.md)
separates compute-phase pipelines, short request-owned GDN blocks and bounded
device-side communication prototypes. It records measured opportunity budgets,
API restrictions and stopping criteria; it does not add a new NPU result.
