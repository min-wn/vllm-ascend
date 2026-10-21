# Isolated Compact Q1/Q4 Prototype

This directory builds `gdn_q14.so`. It copies the structure of the frozen Q4
tile pipeline and deferred-state kernels into a separate module. It does not
change the existing Q4 sources or integrate a model runner.

## Contract

- One current query per live request has width 1 or 4. `sequence` is
  `[initial_token_offset, width_0, width_1, ...]`; a zero width is an inactive row.
  Current inputs and `ids` are packed by these widths, with maximum backing
  capacity of `4 * batch` tokens. Inactive backing rows are never current inputs.
- The stable owner is `ids[begin]`, and every actual token in the request must
  have that owner. Owner zero is reserved for padding. Live request owners must
  be unique within a launch; cross-request uniqueness is a caller invariant.
- `accepted` describes the previous published log and is independently 1..4.
  It must not exceed the previous published width. A zero previous width means
  no pending debt, so initial metadata does not replay an old update. In
  particular, a previous Q4 prefix of four can precede a current Q1 query.
- Each `[slot, head, 8]` validity record stores width 0, 1, or 4 in all eight
  int32 lanes. Each log is 1056 FP32 elements: four 128-element keys, four
  128-element updates, and four eight-lane decay values. Q1 clears all unused
  key, update, and decay fields before staging its new record.
- Shapes, dtypes, device agreement, and mutable-buffer aliasing are checked in
  the binding. The kernel skips invalid current metadata, owners, widths, and
  previous acceptance without publishing a result; it does not return a
  device-side error status. Callers must validate semantic metadata beforehand.

## Entry Points

`solve` takes the same arguments as the old tile pipeline: FP32 base/Q/K for
Cube products; BF16 Q/K/V/beta and FP32 gate for solve; packed sequence/owner
metadata; products/output; and canonical/staged log banks. It stages current
logs without mutating the canonical base or logs. `streaming=True` uses the
existing ready/consumed handoff; `False` delays consumption until products
complete. The frozen comparison uses 20 AIC and their paired 40 AIV cores.

`consume` has the original products-first consumer signature and consumes an
already complete products bank with 40 AIV cores by default. `transport` keeps
the original FP32 producer/transport signature for separate handoff checks.

`commit(base, sequence, ids, accepted, old_logs, old_valid, new_logs, new_valid,
cores=40, request_start=0, request_stop=-1)` applies exactly the accepted old
prefix to the base, then publishes the staged bank. All product and solve
readers must complete before commit. A nonzero staged width must match the
current request width. A zero staged width closes terminal debt and clears the
canonical width; this supports terminal metadata with current width one and
previous acceptance four.

`export_candidates(base, logs, valid, output, cores=40)` writes diagnostic FP32
states to `[slots,4,24,128,128]`. It applies only the published 1 or 4 candidates.
Remaining Q1 slots repeat candidate zero, and a zero-width record repeats the
base in all four slots. Dense exports are diagnostics, outside the compact hot
path.

## Arithmetic And Work

Q1 copies one K vector and one Q vector into fixed Cube columns 0 and 4. The
Cube tiling and `SetTail` remain M384 N8 K128; six Q1 product columns are invalid
and ignored. This prototype does not claim less Cube arithmetic. Consumer
input copies, casts, exponentials, valid Gram products, triangular solve, output
stores, and log publication use the actual current width. Gram reductions keep
the fixed Q4 lane layout and reduction order; unused lanes reduce zero.
Accepted-prefix materialization retains the Q4 FP32 operation order.

Build with `python load.py --build-dir <fresh-directory>` in the existing Linux
CANN/PyTorch-NPU environment. The build targets Ascend910B3 and initializes no
NPU. Source review and Python syntax checks alone do not establish device
compilation, numerical correctness, transport correctness, or performance.
