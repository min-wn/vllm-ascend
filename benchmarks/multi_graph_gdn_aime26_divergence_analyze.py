# SPDX-License-Identifier: Apache-2.0
"""Offline FP64 diagnosis of the bounded Q28 native replay; never a quality gate."""

import argparse
import hashlib
import json
from pathlib import Path


def error(torch, actual, reference):
    a, r = actual.double(), reference.double()
    delta = a - r
    return dict(
        max_abs=float(delta.abs().max()),
        rmse=float(delta.square().mean().sqrt()),
        nrmse=float((delta.square().sum() / r.square().sum().clamp_min(1e-30)).sqrt()),
        differing_elements=int((a != r).sum()),
        elements=a.numel(),
        reference_sum_sq=float(r.square().sum()),
    )


def pool_errors(rows):
    squared_error = sum(row["rmse"] ** 2 * row["elements"] for row in rows)
    squared_reference = sum(row["reference_sum_sq"] for row in rows)
    count = sum(row["elements"] for row in rows)
    return dict(
        max_abs=max(row["max_abs"] for row in rows),
        rmse=(squared_error / count) ** 0.5,
        nrmse=(squared_error / max(squared_reference, 1e-30)) ** 0.5,
        elements=count,
    )


def dense_prefixes(torch, initial, query, key, value, gate, beta, scale):
    # Same recurrence as the existing FP64 audit, retaining each prefix state.
    state, outputs, states = initial, [], []
    query = query * scale
    gamma = gate.exp()
    for token in range(4):
        state = state * gamma[:, token, :, None, None]
        update = (value[:, token] - (state * key[:, token, :, None, :]).sum(-1)) * beta[:, token, :, None]
        state = state + update[..., None] * key[:, token, :, None, :]
        outputs.append((state * query[:, token, :, None, :]).sum(-1))
        states.append(state)
    return torch.stack(outputs, 1), torch.stack(states, 1)


def represented_prefixes(torch, represented, dtype):
    # Native q4_tile_solve.h WriteLog: K[0:512], U[512:1024],
    # gamma[1024+t*8:1032+t*8]. valid[...,0] guards the whole record.
    base, logs, valid = represented
    state, logs = base.to(dtype), logs.to(dtype)
    states = []
    for token in range(4):
        gamma = logs[..., 1024 + token * 8]
        key = logs[..., token * 128 : (token + 1) * 128]
        update = logs[..., 512 + token * 128 : 512 + (token + 1) * 128]
        candidate = state * gamma[..., None, None] + update[..., None] * key[..., None, :]
        state = torch.where((valid[..., 0] != 0)[..., None, None], candidate, state)
        states.append(state)
    return torch.stack(states, 1)


def local_audit(torch, saved):
    from multi_graph_gdn_fp32_arithmetic import compact_recurrence, dense_recurrence, ulp_distance

    inputs, row = saved["recurrence_inputs"], saved["target_row"]
    batch = len(saved["canonical_base"])
    operands = {}
    for name in ("query", "key", "value", "g", "beta"):
        value = inputs[name].reshape(batch, 4, *inputs[name].shape[1:])[row : row + 1].double()
        if name in ("query", "key"):
            value = value.repeat_interleave(3, dim=2)
        operands[name] = value
    base = saved["canonical_base"][row : row + 1].double()
    values = [operands[name] for name in ("query", "key", "value", "g", "beta")]
    oracle, state_oracle = dense_prefixes(torch, base, *values, inputs["scale"])
    dense_output = dense_recurrence(torch, base, *values, scale=inputs["scale"])
    formula = compact_recurrence(torch, base, *values, scale=inputs["scale"])
    if not torch.equal(oracle, dense_output):
        raise AssertionError("new state capture must preserve the existing FP64 output oracle")
    rounded = oracle.to(torch.bfloat16)
    rows = {}
    for name in ("official", "deferred_tile_delayed", "deferred_tile_ready"):
        if name == "official":
            output = saved["official_output"].reshape(batch, 4, 24, 128)[row : row + 1]
            state = saved["official_candidates"]
            extra = {}
        else:
            native = saved[name]
            output = native["output"].reshape(batch, 4, 24, 128)[row : row + 1]
            state = native["candidates"]
            represented64 = represented_prefixes(torch, native["represented"], torch.float64)
            represented32 = represented_prefixes(torch, native["represented"], torch.float32)
            extra = dict(
                import_exact=native["import_exact"],
                materialization_vs_same_native_log_fp64=error(torch, state, represented64),
                native_export_vs_cpu_fp32_log_application=error(torch, state, represented32),
                native_log_representation_vs_recurrence_fp64=error(torch, represented64, state_oracle),
            )
        rows[name] = dict(
            output_vs_fp64=error(torch, output, oracle),
            first_query_output_vs_fp64=error(torch, output[:, :1], oracle[:, :1]),
            output_vs_rounded_fp64=error(torch, output, rounded),
            max_bf16_ulp_to_rounded_reference=int(ulp_distance(torch, output, rounded).max()),
            prefix_states_vs_fp64=[error(torch, state[:, t], state_oracle[:, t]) for t in range(4)],
            all_prefix_states_vs_fp64=error(torch, state, state_oracle),
            **extra,
        )
    return dict(
        layer=0,
        batch=batch,
        target_row=row,
        common_state_origin=saved["canonical_origin"],
        common_fp32_state_import="bit-exact FP32 base copy; fresh zero logs; local test only",
        oracle="CPU FP64 sequential decay/delta/state/query; same BF16 inputs and FP32 initial state promoted exactly",
        formula_identity_output=error(torch, formula, oracle),
        native=rows,
        native_delayed_ready_output_exact=torch.equal(
            saved["deferred_tile_delayed"]["output"], saved["deferred_tile_ready"]["output"]
        ),
        native_delayed_ready_states_exact=torch.equal(
            saved["deferred_tile_delayed"]["candidates"], saved["deferred_tile_ready"]["candidates"]
        ),
        limitations=[
            "FP64 is local arithmetic reference, not a full-prefix FP64 model or task-quality reference",
            "Native output is BF16; no device pre-BF16 value was instrumented",
            "CPU FP64 exp is not a bitwise simulation of Ascend exp/reduction/Cube instructions",
            "Log application splits stored coefficient error from FP32 materialization rounding",
        ],
    )


def continuous_audit(torch, root, rank, trace):
    modes = ("official", "compact_delayed", "compact_ready")
    reports = {mode: json.loads((root / f"{mode}.rank{rank}.json").read_bytes()) for mode in modes}
    snapshots = {
        mode: torch.load(root / f"{mode}.rank{rank}.pt", map_location="cpu", weights_only=True) for mode in modes
    }
    official, delayed, ready = [reports[mode] for mode in modes]
    source = snapshots["official"]
    target = snapshots["compact_ready"]
    comparison = []
    for step, frame in enumerate(trace["steps"]):
        target_row = frame["request_ids"].index(trace["site"]["owner"])
        original = frame["expected_predictions"][target_row * 4 : (target_row + 1) * 4]
        if step + 1 < len(trace["steps"]):
            following = trace["steps"][step + 1]
            following_row = following["request_ids"].index(trace["site"]["owner"])
            emitted = following["accepted_from_previous"][following_row]
        else:
            emitted = trace["site"]["accepted_this_step"]
        comparison.append(
            dict(
                step=step,
                original_q4=original,
                emitted_count=emitted,
                **{
                    mode: reports[mode]["steps"][step]["predictions"][target_row * 4 : (target_row + 1) * 4]
                    for mode in modes
                },
            )
        )
    pairs = {}
    for name in ("recurrences", "decoders"):
        pairs[name] = {}
        for layer, native in source[name].items():
            values = native if isinstance(native, list) else [native]
            candidate = target[name][layer]
            candidate = candidate if isinstance(candidate, list) else [candidate]
            pairs[name][str(layer)] = [
                error(torch, b, a) for a, b in zip(values, candidate, strict=True) if isinstance(a, torch.Tensor)
            ]
    local_inputs_equal = {
        name: torch.equal(source["recurrence_inputs"][name], target["recurrence_inputs"][name])
        for name in (
            "query",
            "key",
            "value",
            "g",
            "beta",
            "actual_seq_lengths",
            "ssm_state_indices",
            "num_accepted_tokens",
        )
    }
    represented = target["continuous_compact_initial"]
    selected = represented_prefixes(torch, represented, torch.float64)
    row = trace["site"]["row"]
    accepted = trace["steps"][-1]["accepted_from_previous"][row]
    canonical = source["canonical_base"][row : row + 1]
    selected = selected[row + 1 : row + 2, accepted - 1]
    first_query = {}
    for layer, official_tensor in source["recurrences"].items():
        first_query[str(layer)] = error(torch, target["recurrences"][layer][:1], official_tensor[:1])
    return dict(
        rank=rank,
        initial_all_equal=official["initial"] == delayed["initial"] == ready["initial"],
        official_all_original_prediction_steps=sum(s["all_predictions_equal_original"] for s in official["steps"]),
        target_q4_predictions=comparison,
        target_accepted_emission_mismatch_steps={
            mode: [
                frame["step"]
                for frame in comparison
                if frame[mode][: frame["emitted_count"]] != frame["original_q4"][: frame["emitted_count"]]
            ]
            for mode in modes
        },
        delayed_ready=dict(
            every_step_hidden_exact=[s["hidden_sha256"] for s in delayed["steps"]]
            == [s["hidden_sha256"] for s in ready["steps"]],
            every_step_predictions_exact=[s["predictions"] for s in delayed["steps"]]
            == [s["predictions"] for s in ready["steps"]],
            final_all_layer_ssm_log_valid_conv_exact=delayed["final_states"] == ready["final_states"],
            final_logits_exact=torch.equal(snapshots["compact_delayed"]["logits"], target["logits"]),
        ),
        final_top_logits={mode: reports[mode]["top_logits"] for mode in modes},
        final_layerwise_compact_vs_official=pairs,
        first_query_recurrence_compact_vs_official=first_query,
        layer0_recurrence_inputs_equal=local_inputs_equal,
        continuous_layer0_selected_compact_vs_official_state=error(torch, selected, canonical),
        final_logits_compact_vs_official=error(torch, target["logits"], source["logits"]),
        final_first_query_hidden_compact_vs_official=error(
            torch, target["hidden"][row * 4 : row * 4 + 1], source["hidden"][row * 4 : row * 4 + 1]
        ),
    )


def head_rounding_audit(torch, root, trace, model):
    from safetensors import safe_open

    index = json.loads((model / "model.safetensors.index.json").read_bytes())["weight_map"]
    token_ids = [866, 660, 1396]
    with safe_open(model / index["lm_head.weight"], framework="pt", device="cpu") as weights:
        tensor = weights.get_slice("lm_head.weight")
        finalists = torch.cat([tensor[i : i + 1] for i in token_ids]).double()
    row = trace["site"]["row"]
    records = {}
    for mode in ("official", "compact_delayed", "compact_ready"):
        saved = torch.load(root / f"{mode}.rank0.pt", map_location="cpu", weights_only=True)
        hidden = saved["hidden"][row * 4].double()
        values = finalists @ hidden
        rounded = values.to(torch.bfloat16).double()
        actual = saved["logits"][0, token_ids].double()
        records[mode] = dict(
            unrounded_fp64_dot={str(i): float(v) for i, v in zip(token_ids, values, strict=True)},
            rounded_fp64_dot={str(i): float(v) for i, v in zip(token_ids, rounded, strict=True)},
            native_bf16_logits={str(i): float(v) for i, v in zip(token_ids, actual, strict=True)},
            unrounded_any_minus_all=float(values[0] - values[1]),
            rounded_dot_matches_native_logits=torch.equal(rounded, actual),
        )
    return dict(
        source="original BF16 lm_head rows and each path's own actual final BF16 hidden; CPU FP64 dot only",
        no_hidden_or_state_injection=True,
        not_full_fp64_model=True,
        not_a_changed_precision_candidate=True,
        token_ids=token_ids,
        controls=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("retain the first analysis; use a new path")
    import torch

    torch.set_num_threads(8)
    trace = json.loads(args.trace.read_bytes())
    rows = []
    for rank in range(2):
        local = torch.load(args.run / f"local-layer0.rank{rank}.pt", map_location="cpu", weights_only=True)
        rows.append(dict(continuous=continuous_audit(torch, args.run, rank, trace), local=local_audit(torch, local)))
    result = dict(
        status="bounded_q28_first_divergence_diagnosis",
        not_task_evaluation=True,
        not_performance_experiment=True,
        ranks=rows,
        snapshot_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.run.glob("*.pt"))},
    )
    result["pooled_local_native"] = {
        mode: {
            field: pool_errors([row["local"]["native"][mode][field] for row in rows])
            for field in ("output_vs_fp64", "first_query_output_vs_fp64", "all_prefix_states_vs_fp64")
        }
        for mode in ("official", "deferred_tile_delayed", "deferred_tile_ready")
    }
    for mode in result["pooled_local_native"]:
        result["pooled_local_native"][mode]["first_prefix_state_vs_fp64"] = pool_errors(
            [row["local"]["native"][mode]["prefix_states_vs_fp64"][0] for row in rows]
        )
    if args.model is not None:
        result["head_rounding"] = head_rounding_audit(torch, args.run, trace, args.model)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(output=str(args.output), ranks=len(rows))))


if __name__ == "__main__":
    main()
