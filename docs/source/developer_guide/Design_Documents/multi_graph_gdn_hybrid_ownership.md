# Hybrid request-owned GDN and tensor-parallel MLP

## User termination

On 2026-10-18 the user rejected additional attention-weight memory and traffic
and explicitly stopped this architecture, including permanent ownership
through FIA. Do not continue its runtime integration or weight replication.
Completed probes below remain negative research evidence. The
[new research plan](aclgraph_dynamic_shape_research.md) preserves original TP
shards and permits general dynamic-shape ACLGraph optimization without splitting.

## Measured investment decision

The bounded hardware gate now runs, but it does **not** justify implementing
the proposed three-GDN owner block. The naturally balanced step has a useful
interior dense/communication saving; entry/exit reconstruction consumes that
three-layer screening budget. At the immutable trajectory's worst owner
imbalance, the interior is already slower before paying any boundary.
Do not generalize the compact kernel or build three connected owner layers
on the strength of these results. The experiment omits the actual recurrence
through an explicit frozen cut and is not a layer-speed comparison.

Both ranks pass 60 hard checks across the two profiles: exact same-schedule
replay, exact RS/AR-slice, finite/inactive outputs, complete returned tensors,
component-aware QKVZ/BA reconstruction and canonical row-order oracles.
Cross-architecture comparisons pass the existing per-element 0.02/0.02 gate;
the maximum normalized RMSE is approximately 0.567%. This does not qualify
full-model numerical drift or continued recurrence state.

Three alternating-order rounds of six replays per mode, critical rank first,
produce these device medians in microseconds:

| Frozen-core dense/communication endpoint | Step 0, B98 | Step 53, B64 |
| --- | ---: | ---: |
| TP complete, original two AR | 1367.17 | 981.83 |
| TP followed by owner selection | 1405.57 | 1033.08 |
| Owner interior, AG plus RS | 1181.84 | 1076.70 |
| Owner interior, AG plus AR/slice | 1301.54 | 1150.95 |
| Owner complete, exit and both output tensors | 1717.66 | 1464.46 |
| Independent entry/exit roundtrip | 589.60 | 441.01 |
| Gross interior screening budget | +185.33 | -94.87 |
| Three-layer screening arithmetic, not measured layers | -33.61 | -725.62 |

The screening arithmetic is `3 * gross interior - roundtrip`; it must not be
reported as connected execution. Wall medians confirm the same decisions.
The preceding one-round smoke test also finds a negative short-block screen.
Raw reports are `hybrid-owner-dense-v1-smoke.json` and
`hybrid-owner-dense-v2.json` under `benchmarks/results/multi_graph_structural/`.
They record all per-rank samples, checks, shape costs and source hashes.

The optimized public full-model padding is now approximately 143.474 ms.
Five percent requires 7173.71 us of net complete-model saving. Even using the
balanced step's 589.60 us boundary as a uniform illustration, sixteen block
boundaries require gross saving above 345.98 us per GDN layer, before other
costs or core changes. The measured 185.33 us balanced-step screen is well
below this illustration. These are investment calculations, not full-trace
latency predictions. Persistent ownership through FIA is a separate possible
change that must first pass its own dense, state and host-update gates.

## Structural question and existing evidence

Keep each request's complete GDN state and all its GDN heads on one TP2 rank,
but retain the original TP2 MLP weights and full-row GateUp/Down shapes.
The request-owned GDN output is a complete hidden vector, so it needs no O
AllReduce. Exchange only the normalized MLP input, then return reduced Down
rows directly to their persistent request owners:

```text
rank-owned hidden/residual
  -> owner inputNorm/QKVZ/BA/state-core/O, all GDN heads
  -> owner postNorm
  -> AllGather(normalized input)
  -> original TP2 GateUp / activation / Down, full gathered row count
  -> ReduceScatter(Down)
  -> owner hidden/residual for the next GDN layer
```

Residuals stay local between GDN layers. At a block exit, gather both hidden and
residual and restore the original request order before a replicated consumer.
No MLP intermediate activation or full MLP weights move between ranks.

The previous full-weight owner experiment is a concrete reason to change the
architecture rather than repeat that experiment. Its owner GateUp costs
475.07 us versus 316.83 us for TP, and its retimed dense bundle costs 1081.48
versus 820.94 us. Owner core execution was approximately flat, 1145.81 versus
1154.13 us, but that used the official dense-state core. See
[ownership qualification](multi_graph_gdn_ownership_qualification.md).

The proposed protocol avoids the measured full-weight, small-M MLP penalty.
It does not assume that owner GDN projections or the newer compact core remain
equally fast. Existing isolated owner QKVZ/BA/O changes are +31.56/+1.13/+12.46
us. Those separate measurements only prioritize a new connected dense budget
probe; their sum is not a measured layer penalty.

## Dependency change and fair attribution

Ordinary TP GDN needs both ranks to contribute every request's O partial sum
before postNorm. Request ownership removes that dependency because one rank
already computes all GDN heads for its requests. The MLP still requires both
TP weight shards, but AllGather of its normalized input and ReduceScatter of
its Down output preserve ownership without recreating the GDN O reduction.

This changes request/head and communication ownership. It is not an advantage
of multiple graph objects. An owner-local padding graph can use the protocol
too; any benefit belongs to the required request partition and persistent
ownership. A baseline that already has this architecture cannot be beaten
merely by renaming its local graphs split graphs. Generic MC2/fused collectives,
layout changes and compact ready-tile kernels must be offered to both sides.

AllGather plus ReduceScatter is approximately one AllReduce's payload, not
zero communication. The candidate replaces two per-layer AllReduces with
that pair. The earlier probe measured six AllReduces at 1384.64 us; dividing by
six gives only an approximate scale of 230.77 us, not this protocol's measured
recoverable latency. Additional startup, GDN owner projection costs, dynamic
imbalance, owner-core changes and the block boundary consume that budget.

For a three-GDN block, a 100-150 us gross interior budget per layer only
justifies a bounded core experiment. It does not automatically justify
full-model integration. With sixteen three-GDN blocks, the required mean
gross saving per GDN layer is at least
`(0.05 * T_full_padding + 16 * T_owner_boundary + T_other_costs) / 48`,
using a measured complete-model baseline and measured connected boundaries.
For example, 300 us per boundary already consumes 100 us per GDN layer.
No threshold is automatically marked passed by the probe.

## Dynamic ownership without a variable-count collective assumption

Owners are fixed by `global_request_id % 2`; batch position never identifies
state. Both ranks use the larger of the two individually rounded default
owner capacities for a step. Standard equal-count AllGather/ReduceScatter
therefore suffice. The gathered MLP row count is twice that owner capacity,
and must itself exist in the official default capture catalog.

This deliberately charges extra padded work caused by unequal live owner
counts. It does not require unqualified AllGatherV, ReduceScatterV or graph
task updates of variable-count collectives.
The installed upstream `distributed/device_communicators/base_device_communicator.py`
uses `dist.all_gather_into_tensor` at line 247 and
`dist.reduce_scatter_tensor` at line 280. Its `all_gatherv` at line 270 and
`reduce_scatterv` at line 313 raise `NotImplementedError`; the installed NPU
communicator does not override those methods. These source contracts support
the equal-bucket design but do not establish runtime/capture qualification.

The CPU audit reads the immutable trace and the catalog recorded by the
complete-model allocation probe, verifies the trace digest and accounts for
all 128 steps:

| Capacity quantity | Observed value |
| --- | ---: |
| Per-rank live requests | 27..64 |
| Original padding total token capacity | 50,688 |
| Dynamic equal-owner total token capacity | 52,160 |
| Additional aggregate token capacity | 2.904% |
| Required equal owner buckets | 17 |
| Fixed 256 tokens per owner overhead | 29.293% |
| Worst step | 53: B64, owners 37/27 |
| Worst step capacities | padding 256, owner 152+152 = 304 |
| Worst step additional work capacity | 18.75% |
| Steps admitting unequal default sizes with original total | 107/128 |

Step zero is naturally balanced: B98, 49/49 live owners and capacities 200/200
versus padding 400. It is only one probe shape. Step 53 must also pass the
budget gate, because it exposes the immutable trajectory's worst capacity
overhead. Other steps remain required for final acceptance.

Fixed maximum owner capacity is rejected as the default design. Dynamic
equal buckets retain state at its original owner and avoid its 29% extra
compute, at the cost of 17 legal graph shapes and normal input compaction.
The CPU audit does not prove any device kernel honors sparse owner masking.

## Minimal dense and communication gate

`benchmarks/multi_graph_gdn_hybrid_owner_probe.py` first supports
`--capacity-audit-only` without importing the accelerator stack. Its NPU phase
uses the official weights-only loader and actual layer-4 GDN/MLP weights.

It measures inputNorm/QKVZ/BA, a frozen GDN-core output, O, postNorm, GateUp,
activation, Down and all declared communication. GDN projections use complete
owner weights at the actual owner row capacity; MLP uses original TP2 weights
at the gathered row capacity. There is no owner M200/N34816 GateUp.

The core output is an independent input. Thus QKV does not actually produce
the timed O input: this is a dense/communication screen, not a GDN forward,
and not a substitute for a three-layer or full-model result. Inputs already
have the endpoint's physical ownership; one-time block-entry selection and
state conversion are not included and must be charged before integration.

The probe compares:

| Mode | Endpoint and purpose |
| --- | --- |
| `tp_complete` | Both full-capacity hidden/residual, original two AllReduces |
| `tp_owner_endpoint` | Same TP work followed by timed owner row selection |
| `hybrid_owner_endpoint` | AG input, full-row TP MLP, RS Down; owned hidden/residual |
| `hybrid_owner_ar_slice` | Same owner path, full AR Down plus slice instead of RS |
| `hybrid_complete` | Owner protocol plus timed packed exit AG, canonical reorder and contiguous hidden/residual |
| `ownership_roundtrip` | Replicated input selection, owner packing, exit AG and canonical reconstruction, with both outputs |

`gross_interior_budget_us` compares `tp_complete` with
`hybrid_owner_endpoint`; its different output ownership is explicit and cannot
be reported as a same-endpoint speedup. `interior_bundle_saving_us` uses the
same owned endpoint and includes TP's extra selection.
`complete_bundle_saving_us` uses both complete returned tensors and includes
the owner's exit, but still excludes one-time entry and the actual state core.
The AR/slice control isolates the effect of returning only owned reduced rows.
The additional `three_layer_screen_only_us` is three times the gross interior
budget minus the separately measured connected ownership roundtrip. This is
an explicitly optimistic screening calculation. It omits actual core latency
changes, per-layer weight differences and overlap/interference; it is not a
three-layer measurement. A concrete connected block must replace it.

Changed inputs run through eager and replay with exact same-schedule checks,
finite and inactive-row checks. RS versus AR/slice uses an exact check;
cross-architecture hidden/residual and component-aware QKVZ/BA reconstruction
retain the existing `rtol=0.02, atol=0.02` gate alongside error diagnostics.
The boundary roundtrip compares to the original canonical inputs as an
independent request-order oracle. A failed gate stops before timing and saves
its partial numerical evidence. Input
preparation, graph capture and validation stay outside the device bundle timing.
Raw slowest-rank samples and alternating round order remain in the report.

Run the bounded gate with the existing environment, NPU 0/1 only, and a fresh
output path:

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_BATCH_INVARIANT=0 TASK_QUEUE_ENABLE=0 \
HCCL_DETERMINISTIC=true torchrun --standalone --nproc-per-node=2 \
  benchmarks/multi_graph_gdn_hybrid_owner_probe.py \
  --steps 0 53 --rounds 3 --repeats 6 --output /new/hybrid-owner-dense.json
```

Do not proceed if an original-tolerance failure is unexplained. A gross
interior budget below approximately
100-150 us per GDN layer rejects the current short-block investment; a larger
budget still has to pass the complete-model budget equation above. Do not
add more layers to hide a failing dense gate.
`TASK_QUEUE_ENABLE=0` retains the existing complete-model software workaround
and applies to padding and every owner control in the same process. It is
not a candidate-only optimization or a new repository environment variable.

## Next gate if the budget survives

The current public request-tile compact kernel hardcodes TP2's 24 value heads
and eight key heads. Full-head ownership needs 48/16 and actual sparse owner
IDs. Generalize the same algorithm and qualify both layouts, rather than
compare the new padding tile kernel with the old official owner core.
Equal theoretical request/head work does not prove equal runtime.

Then run three consecutive real GDN layers with different real weights,
continuous owner states, all candidate prefixes and explicit entry/exit
reconstruction. Compare equally optimized TP padding, the same owner work
with serialized owner progress, and parallel owners using a matched causal
synchronization protocol. The dense screen alone does not implement or
validate that serial/parallel experiment.

Only advance after the complete block demonstrates useful net gain with both
returned tensors and state work. Final full-model validation still requires
the entire immutable trajectory, true FIA host updates, all communication,
all CPU/H2D work, tails and memory, and at least 5% repeatable mean improvement.

If block boundaries consume the useful interior budget, one alternative is
persistent request ownership through FIA layers as well, keeping only MLP
tensor-parallel. This would remove the sixteen GDN/FIA ownership boundaries,
but needs complete-head FIA qualification, every real host-length update,
new state/page ownership and additional replicated attention weights. It is
not implemented or budget-qualified here. Its memory peak and full-model
behavior must be measured before choosing that larger design.

## Read-only next-direction screen: permanent owners through FIA

The bounded three-GDN owner design should stop on the measured dense gate.
Its independently repeated B98 gross interior budget of about 185 us/layer
cannot pay a roughly 590 us entry/exit boundary every three layers, and the
fixed trajectory's worst-capacity step loses about 95 us before any state-core
cost. Eliminating those sixteen block boundaries requires owner execution
through full-attention layers too; it cannot be achieved by omitting their
returned hidden/residual reconstruction.

A possible larger design preserves request ownership through all 64 attention
blocks while using TP MLPs at every layer. Initial hidden/residual selection
and the final canonical reconstruction occur once per complete forward. GDN
state and FIA K/V remain with stable request owners. This is a new architecture
candidate with no NPU qualification; the preceding negative result remains
valid and is not relabeled a success.

### Verified FIA dimensions and dense cost gap

The actual layer-3 safetensors headers were inspected read-only on the current
machine. They report `q_proj.weight=[12288,5120]`,
`k_proj.weight=v_proj.weight=[1024,5120]`, and
`o_proj.weight=[5120,6144]`. Q includes the output gate: it packs 24 heads of
256 Q and 256 gate channels, so there are 14,336 packed Q/gate/K/V channels.
The matching model config and its digest are retained in
`outputs/multi_graph_tp_attention_mlp/hybrid-checkpoint-inventory-v1.json` in
the parent workspace. Local TP shapes were independently recorded by the
qualified attention benchmark.

| Actual projection `(M,K,N)` | Ordinary TP2, capacity 400 | Balanced request owner |
| --- | --- | --- |
| Q/gate/K/V | `(400,5120,7168)` | `(200,5120,14336)` |
| O | `(400,3072,5120)` | `(200,6144,5120)` |
| GateUp | `(400,5120,17408)` | `(400,5120,17408)` |
| Down | `(400,8704,5120)` | `(400,8704,5120)` |

At worst step 53, the ordinary dense row count is 256, each owner has capacity
152, and the gathered MLP row count becomes 304. This extra 18.75% compute is
charged to the candidate. Fixed owner identity also creates actual FIA work
imbalance: step 53 has 38,399 and 30,694 KV tokens on the two owners, a 1.1115
ratio of the larger load to their average. Step zero has 47,062 and 47,340,
almost balanced. Neither row counts nor KV token counts prove equal kernel
latency or linear runtime scaling.

There is no retained measurement for complete-head owner FIA QKV/O GEMMs or
for owner FIA itself. Existing TP split-M QKV measurements retain half weights
and therefore do not measure these shapes. The old GDN owner QKV/O costs are
not transferred to FIA. Runtime availability of full-head shapes alone cannot
support a numerical speedup estimate.

### Memory accounting before any allocation claim

The complete FIA QKV and O BF16 matrices total 209,715,200 bytes per layer.
Replacing TP2 halves with complete owner weights adds 104,857,600 bytes/layer
per rank, or **1.5625 GiB across all sixteen FIA layers**. The corresponding
48 GDN input/O weight replication is approximately 5.1782 GiB/rank. Permanent
attention ownership therefore adds approximately **6.7407 GiB/rank** of these
weights, while retaining the existing TP MLP shards.

This estimate assumes replacement of the old attention shards. Retaining TP
shards, extra ND and NZ copies, temporary full checkpoint weights, or graph
pools changes the peak. Loading all replacements before releasing old shards
is not accounted for. Adding 6.74 GiB to the previous eager allocation of
roughly 42.9 GiB suggests roughly 49.6 GiB before additional state null slots,
layouts, owner metadata, workspaces and graph allocations. This is arithmetic,
not allocation or proof of fit on the 64 GiB device.

FIA K/V need not double: owners retain half the requests with all four KV
heads instead of all requests with two heads. The immutable maximum page
reservation contains 631 even-owner and 661 odd-owner pages. With independent
null pages and all sixteen layers, owner K/V estimates are 5,301,600,256 and
5,553,258,496 bytes per rank, versus 5,423,235,072 bytes for TP2. Local page
compaction is necessary; allocating all 1293 pages with four heads would
silently double K/V. GDN compact/conv/log allocations likewise use owner-local
state IDs and must count full-head null slots and nonuniform page totals.

### FIA implementation and real length updates

Relevant source contracts are:

- `vllm_ascend/patch/worker/patch_qwen3_5.py:48` implements QKV, Q/K norm/RoPE, FIA, gate and O using explicit local head counts.
- `benchmarks/multi_graph_tp_attention.py:364` already decomposes the real Q/gate/K/V transformation; `:402` calls the actual attention implementation with independent ND K/V.
- `vllm_ascend/attention/attention_v1.py:582` stores explicit query/KV head counts; `:710` passes both counts to FIA.
- `vllm_ascend/attention/attention_v1.py:514` resolves each FIA occurrence's actual host query/context lengths and block table; `:720` registers the updatable task.
- `benchmarks/multi_graph_gdn_q4_full_metadata.py:65` reserves stable request pages; `:81` maps current true lengths, positions, slots and null rows by request identity.

A new owner FIA adapter would set 24 Q heads and four KV heads, preserve
Q/gate interleaving when reconstructing weights, and use owner-local metadata
and ND K/V with remapped stable physical pages. It must not mutate a shared
TP attention module's head counts while TP graphs remain live. Full-head
FIA supports the same mathematical GQA ratio six, but actual current-driver
kernel/capture qualification is still missing.

All sixteen host-length graph updates remain mandatory. Each rank updates its
own selected requests' true lengths before their task readiness events release
FIA. Owner assignment does not make lengths static, and four query tokens do
not eliminate the current host update requirement. Resumed owners retain their
pending accepted position and local pages; shuffled batch order must not move
their state. Later migration would require a separately costed protocol.

### Next minimal experiment and budget

Before implementing owner FIA or generic 48-head compact kernels, measure an
FIA dense/communication gate with the same immutable step 0 and step 53. Reuse
the current probe's capacity/row plans, official weight-only loader,
component-aware sharding and eager/replay checks. Load actual full FIA
Q/gate/K/V and O weights from checkpoint components. The current GDN checkpoint
helper reads only GDN projections and cannot be reused for FIA unchanged.

Time actual inputNorm/QKV, a clearly frozen FIA/gated-output cut, O, postNorm,
the original TP full-row GateUp/activation/Down, AG and RS. Include the owner
roundtrip and both returned hidden/residual. Compare ordinary TP, owner with
AG/RS, and owner with AG/AR/slice. Validate packed Q/gate/K/V component order,
complete owner/TP dense output, real inactive rows and fixed addresses before
using timing. This gate does not perform attention and must not claim it
qualifies FIA, K/V correctness or full-model latency.

Let `G_GDN` and `G_FIA` be measured gross dense/communication savings over the
original TP endpoints. The optimistic screen for permanent attention owners
is `48*G_GDN + 16*G_FIA - T_initial_final_boundary - T_other_costs`, where
`T_other_costs` includes actual owner-core differences, owner FIA differences,
all CPU/H2D metadata, imbalance, graph updates and additional final waits.
The complete baseline currently reports approximately 143,474 us, so 5% needs
about 7,174 us net. Using the favorable B98 185 us GDN observation for every
step is invalid: the worst-capacity observation is negative. Run selected
capacity profiles first, then a weighted dense screen over the unchanged
128-step trajectory if any positive budget survives. Only the final complete
model comparison can prove the target.

If that screen cannot clear the complete-model budget with margin for actual
owner FIA/core effects, stop before building the larger ownership runtime.
No full-model owner speedup, memory fit or capture support is established by
this read-only proposal.
