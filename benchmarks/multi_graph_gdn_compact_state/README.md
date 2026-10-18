# Compact Q4 GDN State Prototype

This directory contains an isolated Ascend C prototype for Atlas A2/910B.
It does not replace an installed FLA operator or a serving cache contract.
Compilation alone does not establish runtime correctness or performance.

## Build

Run inside the existing CANN 9.1 / torch 2.10 / torch_npu environment:

```bash
python benchmarks/multi_graph_gdn_compact_state/load.py \
  --build-dir /workspace/multi-graph-results/compact-state-build-v2
```

This builds a separate `gdn_compact_state.so` and an Ascend C kernel library
under the specified output directory. It writes no installed package files.
Use a fresh output directory for changed sources: the CANN 9.1 legacy
preprocessing step does not reliably support incremental rebuilding. A completed
build with a matching source fingerprint can be reused.
The CMake target is `Ascend910B3`; the prototype uses at most 40 AIV blocks,
Q4, BF16 Q/K/V/state, FP32 scalar gates, Dk=Dv=128 and a V tile of 64.
That tile and the two output queue depths match the corresponding upstream
host tiling on the machine's 192 KiB UB. See `tiling_provenance.md` for the
source revision and calculation. The plain tiling struct is constructed explicitly. Opaque
installed host tiling and OPP registration are not reused or modified.

## Calls And State Layout

`emit` runs the installed recurrence arithmetic with either four dense state
snapshots or compact log output. Its inputs and outputs are fixed contiguous
ND tensors. Query/key normalization is a caller responsibility. Query scaling
is performed internally using `1/sqrt(128)`, matching the scoped reference.

| Tensor | Contract |
| --- | --- |
| Q/K | BF16 `[4B, Nk, 128]` |
| V/output | BF16 `[4B, Nv, 128]` |
| beta / gate | BF16 / FP32 `[4B, Nv]` |
| sequence | INT32 `[B+1]`: start offset zero, then per-request lengths 4 or 0; these are not cumulative ends |
| accepted | INT32 `[B]`, values 1 through 4 |
| dense state | BF16 `[1+4B, Nv, 128, 128]` |
| compact base | BF16 `[1+B, Nv, 128, 128]` |
| dense state IDs | INT32 `[4B]`, request r uses rows `1+4r` through `4+4r` |
| compact state IDs | INT32 `[4B]`, all four tokens of request r select row `1+r` |
| decay log | FP32 `[1+B, 4, Nv, 8]`, decay repeated across an aligned 32-byte record |
| key/update logs | FP32 `[1+B, 4, Nv, 128]` |

Inactive requests have zero sequence length and null state IDs. Null row zero
and inactive rows remain unchanged. A compact log stores the actual FP32
exponentiated decay, normalized key and computed rank-one update vector.
Key vectors are replicated per value head to keep this first prototype simple.
The log does not store an approximation of the state.

`reconstruct(base, gamma, key, update, lengths, accepted, destination)` applies
the selected prefix in the original sequence of FP32 scale and `MulAddDst`
operations, then rounds the result to BF16. `destination` may alias `base`;
each work item accesses disjoint request/head/V rows. Reconstruction completes
before the next `emit` overwrites logs. The compact step endpoint must include
both kernels, including this base write and the recurrence's subsequent read.
Inactive destination rows are preserved; use an initialized destination when
exporting logical states for qualification.

V2 adds `fused_emit(q, k, v, beta, gate, base, sequence, state_ids,
accepted, output, gamma_log, key_log, update_log, cores=40)`.
Start with ordinary compact `emit` to produce the first log records.
Each subsequent `fused_emit` reconstructs the accepted prefix of the previous
records directly into the recurrence's state UB,
rounds it to BF16, writes the new base once, and continues with the current Q4
inputs from the same rounded UB state. The timed closed step is this single
kernel, including reconstruction arithmetic, input logs, base write and new
log output. No extra state read or separate reconstruction launch is exempted.

One log bank is updated in place. Each head belongs to exactly one core, which
loads all four previous gamma/key/update records into a separate 4224-byte UB
buffer before it starts that head's recurrence. Both V tiles consume those
preserved records even after new logs overwrite GM. Distinct heads use
disjoint log regions. Consequently inactive rows retain their latest records
without copies or a global parity switch, including inactive-to-active resume.
One fixed-pointer graph can be replayed across steps. Null rows stay unchanged.
Initialize unused log rows to the identity transform (`gamma=1`, key/update
zero) when setting up the cache. Then a request's first activation, before it
has emitted any log, preserves its initial base and all four initial logical
candidates. A newly assigned serving-cache slot needs the same explicit reset;
zero decay logs are not an identity transform.

The current four log records are collected in a small double-buffered output
queue in UB. One strided batch transfer per V tile replaces V1's per-token
DMA and round-trip barriers. Previous logs are loaded once per head and used
by both V tiles. The recurrence formula and candidate rounding remain the
installed ones. These structural changes require fresh device qualification;
successful compilation is not a correctness or performance result.

All four logical candidates can be exported by reconstructing prefixes 1-4
into separate initialized temporary outputs. Such export is for correctness,
not timed execution. A serving consumer needing dense snapshots must pay its
materialization cost; this prototype does not silently exempt that consumer.

## Source And License

The three `recurrent_gated_delta_rule*` / `vendor_recurrent.h` sources derive
from the installed `flash_linear_attention_npu_a2-26.9.1+deva4a7958` wheel,
whose build metadata identifies tier `a2`. Their file-level copyright and
CANN Open Software License Agreement Version 2.0 notices are retained. The
complete applicable license is copied from the same wheel into
`LICENSE-CANN.txt`.

Original installation directory:

```text
/usr/local/python3.12.13/lib/python3.12/site-packages/fla_npu/opp/vendors/fla_npu_transformer/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/recurrent_gated_delta_rule/
```

| Original source | SHA256 |
| --- | --- |
| `recurrent_gated_delta_rule.h` | `7661897a77d6cf8f4e2c3f9c44d24c1fdb3547ee3d3e55435b0857b1192cb17f` |
| `recurrent_gated_delta_rule_struct.h` | `d5c65bc3240fb14d52e6e3ebed5b5db00fde16c78c21d1a7a15e4b105445c2de` |
| `recurrent_gated_delta_rule_tiling_data.h` | `673cc53e716e24586534a9d56cc2c2597386e699a1f11d6124952659b0c1dd45` |

The first header is renamed to `vendor_recurrent.h` and extended with optional
queued log output, suppression of dense snapshots in compact mode, and fused
accepted-prefix recovery. Repository C++ formatting is applied. The dense control retains
the original recurrence math and four-snapshot write path. The new wrapper and
reconstruction kernel are benchmark code. The native buffer allocation and
copy synchronization remain explicit and are included in device measurements.

The custom BF16 dense control must match the installed FLA kernel on active
outputs and the entire state cache before any performance result is accepted.
Every logical compact candidate and dynamic continued-state step must then
match that dense control. No model-level performance claim follows from a
recurrence-only result.
