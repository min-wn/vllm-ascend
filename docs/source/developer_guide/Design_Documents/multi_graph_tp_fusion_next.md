# Request-split execution with fused TP boundaries

## Decision and measured budget

The next substantial optimization target is to preserve efficient dense GEMMs
while pipelining communication at request boundaries. More stream-order tuning
of independently split GateUp GEMMs is unlikely to meet the target.

The existing complete two-layer continuity experiment measured padding at
4816.833 us and overlapped split at 5109.0585 us. Matching padding requires
292.2255 us of net savings. Beating that baseline by 5% requires 533.06715 us
of savings from the split candidate. Five percent is an investment target,
not a predicted improvement. A newly optimized padding baseline changes this
budget and must be measured again.

The isolated GateUp experiment found a 29-33% aggregate GEMM penalty when
splitting 400 rows into 200 + 200, even without communication. Removing this
one penalty cannot be assumed to recover the complete two-layer deficit.
The shared-GDN candidate's separate paired run saved only 23 us against its
4802.042 us padding baseline. These different runs must not be subtracted to
attribute an exact causal saving to a new implementation.

## Installed interfaces checked on the current machine

Read-only inspection on `vllm-ascend-030` found torch 2.10.0+cpu with
torch_npu 2.10.0.post4 and these exported operators:

| Interface | Installed contract | Remaining qualification |
| --- | --- | --- |
| `npu_mm_all_reduce_base` | Internal compute/communication pipeline; A2 BF16 and ND/NZ weights documented | Current driver, Q4 decode workload, ACL graph replay |
| `npu_mm_reduce_scatter_base` | Internal pipeline; `ai_cpu` and `aiv` modes; NZ documented only for `aiv` | Current driver, mode, graph replay and output ownership |
| `npu_all_gather_base_mm` | Communication/computation pipeline; ND input and ND/NZ weight constraints | Current driver, mode, graph replay and residual layout |

Availability is not execution qualification. Installed documentation requires
matching driver/firmware/CANN packages from 8.0.RC2 or later; comparing version
numbers across these components is insufficient. No driver update is planned.
The AllReduce documentation also says not to enable the operator for incremental
execution. The ReduceScatter `ai_cpu` documentation describes training usage.
These contracts need resolution through an appropriately scoped runtime probe
and implementation inspection, not an assumption of supported decode usage.
The bundled version lists appear older than the installed package itself.

The installed contracts restrict `comm_turn` to zero. It is not an exposed
request-block-size tuning knob. None exposes an external per-block completion
event. Internal communication overlap does not allow the next graph to consume
a partially written output. A graph-level event after an ordinary or fused
kernel only releases its completed output.

The repository already contains a GEMM/ReduceScatter fusion pass and device
wrapper. Their existence does not qualify this workload on the current device.

## Staged implementation target

1. Establish the supported fused boundary: compare ordinary Down plus AllReduce
   with fused Down/AllReduce, then fused Down/ReduceScatter plus the required
   reconstruction. Use actual model dimensions, BF16, legal capacities and
   current weight layouts. Include any required ND conversion in the accounting.
   Verify eager execution before ACL capture and repeated replay. A generic
   torch.compile graph example is not proof of ACL replay support.
2. Keep full-capacity post-norm, GateUp and activation. Split only Down and its
   reductions into the existing request capacities. Compare barrier and overlap
   variants to measure the cost of fragmented Down separately from hidden
   communication. This is a diagnostic prototype, not the expected large win.
3. If the measured critical-path budget is sufficient, implement persistent
   request ownership across a pair of TP boundaries:

   ```text
   full-M Down + ReduceScatter
       -> rank-owned request rows: residual + RMSNorm
       -> AllGather + next column-parallel GEMM
       -> existing request-split attention schedule
   ```

   ReduceScatter produces complete hidden features for a subset of token rows
   on each rank, so per-token RMSNorm is local. Each rank must select the same
   rows from its replicated residual. The next QKV/GateUp weight shard still
   requires all request rows; fused AllGather/GEMM reconstructs that input while
   overlapping communication with projection. Returned residuals need an explicit
   ownership contract or a timed reconstruction before any replicated consumer.
   Request rows cannot be reassigned between ranks without also handling the
   corresponding state and metadata requirements.
4. Only pursue custom Down block publication if the supported fusion path leaves
   a measured substantial tail. This requires a producer/consumer protocol with
   correct device memory ordering and collective order on both ranks. Reading
   unfinished ordinary GEMM output is invalid. Tile scheduling must also retain
   weight reuse; one launch alone does not guarantee full-GEMM efficiency.

Stage 3 changes communication placement and intermediate tensor ownership.
It retains full-M projections, while request splitting remains in the attention
execution. It does not introduce attention-head splitting or bespoke capture
sizes. A request group can change live length on each step within the existing
capacity and metadata protocol; it need not have fixed input values.

## Attribution and stop criteria

ReduceScatter plus AllGather is not inherently less communication than an
AllReduce. The potential gain is overlap with dense computation and moving
local work between the collectives. The gain must exceed additional launches,
layout conversions, residual reconstruction and interference.

Compare four variants with identical endpoints: ordinary padding, ordinary
split, fusion-enabled padding, and fusion-enabled split. Improvements common
to both are generic fusion benefits. Only the last paired comparison establishes
that splitting wins against an equally optimized padding implementation.
If padding obtains all the benefit, retain the generic finding but do not claim
a new split advantage.

Before building a custom kernel, measure the exposed communication intervals
on the actual two-layer critical path. Summing HCCL durations across overlapping
streams is not a latency upper bound. Reject a proposed large-gain route if its
optimistic recoverable critical path cannot meet the required net saving.

Keep true FIA lengths and replay/update ordering, slowest-rank timing, all final
waits and both returned hidden/residual tensors. Preserve finite-value, inactive
state and same-schedule execution checks; record cross-shape numerical drift
under the existing numerical-assessment policy. Start with one legal 200 + 200
case; broaden dynamic batches only after a causal improvement appears.

## Status

This update records measured budgets and installed API contracts. No fused
operator execution, new latency result, or partial-output handoff has been
qualified by this investigation. The ordinary path remains unchanged. See
[the measured two-layer results](multi_graph_tp_attention_mlp.md) for evidence
supporting the change of direction.
