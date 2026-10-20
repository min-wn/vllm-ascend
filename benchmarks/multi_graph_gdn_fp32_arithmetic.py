# SPDX-License-Identifier: Apache-2.0
"""CPU arithmetic audit of saved step-zero inputs; no proposed precision mode.

FP64 is an offline diagnostic reference only. Native state and activations are
not changed. Chunking bounds memory without changing the full live batch.
"""

import argparse
import hashlib
import json
from pathlib import Path


def dense_recurrence(torch, state, query, key, value, gate, beta, *, scale):
    outputs = []
    query = query * scale
    gamma = gate.exp()
    for token in range(4):
        state = state * gamma[:, token, :, None, None]
        update = (value[:, token] - (state * key[:, token, :, None, :]).sum(-1)) * beta[:, token, :, None]
        state = state + update[..., None] * key[:, token, :, None, :]
        outputs.append((state * query[:, token, :, None, :]).sum(-1))
    return torch.stack(outputs, dim=1)


def compact_recurrence(torch, state, query, key, value, gate, beta, *, scale):
    from multi_graph_gdn_q4_log_reference import _solve_from_products

    right = torch.cat((key, query), dim=1).permute(0, 2, 3, 1)
    products = state @ right
    _, output, _ = _solve_from_products(products, query, key, value, gate.exp(), beta)
    return output * scale


def ulp_distance(torch, left, right):
    if left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16:
        raise ValueError("only the native BF16 output boundary is audited")

    def ordered(value):
        bits = value.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -32768 - bits, bits)

    return (ordered(left) - ordered(right)).abs()


def metrics(torch, actual, reference):
    a, b = actual.double(), reference.double()
    delta = a - b
    dimensions = tuple(range(1, a.ndim))
    ratio = (delta.square().sum(dimensions) / b.square().sum(dimensions).clamp_min(1e-30)).sqrt()
    return dict(
        worst_request_nrmse=ratio.max().item(),
        pooled_nrmse=(delta.square().sum() / b.square().sum().clamp_min(1e-30)).sqrt().item(),
        max_abs=delta.abs().max().item(),
        differing_elements=int((actual != reference).sum().item()),
    )


def inspect(torch, path, *, chunk=8):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    inputs = saved["recurrence_inputs"]
    base, logs, valid = saved["state"]
    if logs.count_nonzero().item() or valid.count_nonzero().item():
        raise ValueError("offline audit is limited to the empty-log first step")
    layer = int(next(name for name in saved["reference"] if "/recurrence/output" in name).split("/")[0][5:])
    lengths = inputs["actual_seq_lengths"]
    batch = int((lengths[1:] == 4).sum().item())
    capacity = inputs["query"].shape[0] // 4
    if batch != 98 or capacity != 100 or chunk < 1:
        raise ValueError("retain original full B98 and capacity 100 Q4 requests")
    slots = inputs["ssm_state_indices"].reshape(capacity, 4)
    owners = (slots[:batch, 0].long() - 1) // 4
    values = {}
    for name in ("query", "key", "value", "g", "beta"):
        value = inputs[name]
        values[name] = value.reshape(capacity, 4, *value.shape[1:])[:batch]
        if name in ("query", "key"):
            values[name] = values[name].repeat_interleave(3, dim=2)
    result = {(name, str(dtype)): [] for name in ("dense", "compact") for dtype in (torch.float32, torch.float64)}
    for begin in range(0, batch, chunk):
        end = min(begin + chunk, batch)
        for dtype in (torch.float32, torch.float64):
            operands = [values[name][begin:end].to(dtype) for name in ("query", "key", "value", "g", "beta")]
            state = base[owners[begin:end] + 1].to(dtype)
            for name, function in (("dense", dense_recurrence), ("compact", compact_recurrence)):
                output = function(torch, state, *operands, scale=inputs["scale"])
                result[name, str(dtype)].append(output)
    result = {f"{name}/{dtype}": torch.cat(parts) for (name, dtype), parts in result.items()}
    key = f"layer{layer}/recurrence/output"
    reference = saved["reference"][key].reshape(capacity, 4, 24, 128)[:batch]
    actual = saved["observed_recurrence"][key].reshape(capacity, 4, 24, 128)[:batch]
    oracle = result["dense/torch.float64"]
    rounded = oracle.to(torch.bfloat16)
    native_distance = ulp_distance(torch, actual, reference)
    differences = actual != reference
    rows = {name: metrics(torch, output, oracle) for name, output in result.items()}
    for name, output in (("native_official", reference), ("native_compact", actual)):
        rows[name + "/vs_fp64"] = metrics(torch, output, oracle)
        rows[name + "/vs_rounded_fp64"] = metrics(torch, output, rounded)
        rows[name + "/max_bf16_ulp_to_rounded_fp64"] = int(ulp_distance(torch, output, rounded).max().item())
    # Local FP32 contraction-order effects before and after BF16 rounding are
    # evidence of numerical sensitivity, not a claim of emulating CANN exactly.
    rows["cpu_fp32_dense_vs_compact"] = metrics(torch, result["compact/torch.float32"], result["dense/torch.float32"])
    rows["cpu_fp32_dense_vs_compact_after_bf16"] = metrics(
        torch, result["compact/torch.float32"].to(torch.bfloat16), result["dense/torch.float32"].to(torch.bfloat16)
    )
    return dict(
        snapshot=path.name,
        snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        layer=layer,
        batch=batch,
        capacity=capacity,
        owners=owners.tolist(),
        reference_kind="offline FP64 dense recurrence only, not a precision implementation or acceptance reference",
        native_bf16_pair=dict(
            differing_elements=int(differences.sum().item()),
            max_ulp=int(native_distance.max().item()),
            more_than_one_ulp=int((native_distance > 1).sum().item()),
            sign_disagreements=int((differences & (actual.signbit() != reference.signbit())).sum().item()),
        ),
        metrics=rows,
        caveat="CPU exp/reductions do not emulate native Ascend exp, Cube or vector reduction order bitwise",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("require a new output path")
    import torch

    torch.set_num_threads(8)
    rows = []
    for path in args.snapshots:
        rows.append(inspect(torch, path))
        print(path.name, json.dumps(rows[-1]["native_bf16_pair"]), flush=True)
    args.output.write_text(
        json.dumps(dict(status="complete_saved_step0_arithmetic_audit", observations=rows), indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
