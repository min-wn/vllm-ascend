# SPDX-License-Identifier: Apache-2.0
"""Attribute only the first frozen FP16 failure; no performance measurement."""

import argparse
import json
from pathlib import Path

from multi_graph_replayssm_protocol import Cursor, frozen_trace


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module=None):
    import torch
    import torch.distributed as dist
    from multi_graph_gdn_q4_backend import load_extension
    from multi_graph_replayssm_backend import RingBackend, RingState, installed_ring
    from multi_graph_tp_gdn import qualify_gdn
    from vllm.distributed.parallel_state import get_world_group
    from vllm.forward_context import get_forward_context

    import vllm_ascend.ops.gdn as gdn_module

    torch.npu.matmul.allow_hf32 = False
    rank, group = dist.get_rank(), get_world_group().cpu_group
    width = runner.vllm_config.model_config.hf_text_config.hidden_size
    hidden, residual = (torch.zeros(400, width, dtype=torch.bfloat16, device="npu") for _ in range(2))
    fixture = qualify_gdn(args, workload, runner, model, (hidden, residual), group, setup_only=True)
    dense = fixture.states[1]
    candidates = [[1 + owner * 4 + token for token in range(4)] for owner in range(100)]
    for slots in candidates:
        for slot in slots[1:]:
            dense[slot].copy_(dense[slots[0]])
    state = RingState(torch, dense, candidates)
    state.ring = state.ring.half()
    state.mutable = (state.base, state.ring, state.gates, state.cursors)
    backend = RingBackend(state, load_extension("gdn_replayssm", args.ring_extension))
    original = gdn_module.recurrent_gated_delta_rule
    owner, selected, token_slice = 34, 35, slice(136, 140)

    def snapshot():
        return dict(dense=tuple(t.clone() for t in fixture.states), ring=state.snapshot())

    def restore(saved):
        fixture.reset(saved["dense"])
        state.restore(saved["ring"])

    history = {"official": snapshot(), "ring": snapshot()}
    captured = {}
    stage_values = {}
    hooks = []
    layer = model.language_model.model.layers[fixture.layer_index]
    for name, part in (
        ("gated_norm", layer.linear_attn.norm),
        ("output_projection", layer.linear_attn.out_proj),
        ("post_attention_norm", layer.post_attention_layernorm),
        ("mlp_gate_up", layer.mlp.gate_up_proj),
        ("mlp_activation", layer.mlp.act_fn),
        ("mlp_down", layer.mlp.down_proj),
    ):

        def observe(module, inputs, output, name=name):
            values = output if isinstance(output, tuple) else (output,)
            stage_values[name] = tuple(t.clone() for t in values if isinstance(t, torch.Tensor))

        hooks.append(part.register_forward_hook(observe))

    def run(mode, *, substitute=None):
        stage_values.clear()

        def recurrence(q, k, v, initial, **kwargs):
            selected_initial = (
                initial[candidates[owner][int(kwargs["num_accepted_tokens"][owner].item()) - 1]].clone()
                if mode == "official"
                else None
            )
            result = (original if mode == "official" else backend)(q, k, v, initial, **kwargs)
            captured[mode] = dict(
                q=q.clone(),
                k=k.clone(),
                v=v.clone(),
                beta=kwargs["beta"].clone(),
                g=kwargs["g"].clone(),
                result=result.clone(),
                initial=selected_initial,
            )
            if substitute is not None:
                result = result.clone()
                result[token_slice] = substitute
            return result

        with installed_ring(recurrence):
            output = fixture.run(hidden, residual)
        torch.npu.synchronize()
        return dict(output=tuple(t.clone() for t in output), stages=dict(stage_values), recurrence=captured[mode])

    def nrmse(a, b):
        a, b = a.float(), b.float()
        return ((a - b).square().sum() / b.square().sum().clamp_min(1e-30)).sqrt().item()

    def stages(a, b):
        result = {}
        for name in a["stages"]:
            for index, (left, right) in enumerate(zip(a["stages"][name], b["stages"][name], strict=True)):
                if left.shape[0] == 400 * 24:
                    left, right = left.view(400, -1), right.view(400, -1)
                result[f"{name}/{index}"] = nrmse(left[token_slice], right[token_slice])
        result["hidden"] = nrmse(a["output"][0][token_slice], b["output"][0][token_slice])
        result["residual"] = nrmse(a["output"][1][token_slice], b["output"][1][token_slice])
        return result

    try:
        trace, _ = frozen_trace()
        for step, entry in enumerate(trace[:2]):
            fixture.refresh(entry["live"])
            with fixture.phases.context_for_name("full"):
                metadata = get_forward_context().attn_metadata[fixture.phases.gdn.prefix]
            metadata.num_accepted_tokens.copy_(torch.tensor(entry["accepted"], dtype=torch.int32, device="npu"))
            torch.manual_seed(entry["seed"])
            for tensor in (hidden, residual):
                tensor.copy_(torch.randn_like(tensor) * 0.1)
                tensor[entry["live"] * 4 :].zero_()
            before = history["ring"]
            results = {}
            for mode in ("official", "ring"):
                restore(history[mode])
                results[mode] = run(mode)
                history[mode] = snapshot()
            if step == 0:
                continue
            official, native = results["official"], results["ring"]
            recurrence = native["recurrence"]
            for name in ("q", "k", "v", "beta", "g"):
                torch.testing.assert_close(
                    recurrence[name][token_slice], official["recurrence"][name][token_slice], rtol=0, atol=0
                )
            cursor_fields = before["ring"][3][selected, :, 0].cpu().tolist()
            cursor = Cursor(cursor_fields[0], cursor_fields[1], bool(cursor_fields[2]), bool(cursor_fields[3])).advance(
                entry["accepted"][owner]
            )
            assert not cursor.is_flush
            checkpoint = before["ring"][0][selected].clone()
            for token in range(cursor.write_pos):
                physical = (cursor.cache_base + token) % 32
                log = before["ring"][1][selected, :, physical].float()
                gamma = before["ring"][2][selected, :, physical, 0].exp()
                checkpoint = checkpoint * gamma[:, None, None] + log[:, 128:, None] * log[:, None, :128]

            def materialized(start, recurrence=recurrence):
                current, outputs = start.clone(), []
                for token in range(136, 140):
                    k = recurrence["k"][token].float().repeat_interleave(3, 0)
                    q = recurrence["q"][token].float().repeat_interleave(3, 0)
                    current = current * recurrence["g"][token].exp()[:, None, None]
                    delta = recurrence["beta"][token].float()[:, None] * (
                        recurrence["v"][token].float() - (current @ k[:, :, None]).squeeze(-1)
                    )
                    current = current + delta[:, :, None] * k[:, None, :]
                    outputs.append(((current @ q[:, :, None]).squeeze(-1) * 128**-0.5).to(torch.bfloat16))
                return torch.stack(outputs)

            quantized_oracle = materialized(checkpoint)
            dense_oracle = materialized(official["recurrence"]["initial"])
            counterfactuals = {}
            for name, replacement in (
                ("quantized_materialized_oracle", quantized_oracle),
                ("official_dense_materialized_oracle", dense_oracle),
                ("official_recurrence_output", official["recurrence"]["result"][token_slice]),
            ):
                restore(before)
                observed = run("ring", substitute=replacement)
                counterfactuals[name] = stages(observed, official)
            row = dict(
                rank=rank,
                step=step,
                owner=owner,
                original_threshold=0.005,
                first_failure_reproduced=stages(native, official)["hidden"] > 0.005,
                input_qkv_beta_gate_exact=True,
                native_stages_vs_official=stages(native, official),
                accepted_state_nrmse=nrmse(checkpoint, official["recurrence"]["initial"]),
                native_core_vs_quantized_oracle=nrmse(native["recurrence"]["result"][token_slice], quantized_oracle),
                quantized_oracle_vs_official_core=nrmse(
                    quantized_oracle, official["recurrence"]["result"][token_slice]
                ),
                dense_oracle_vs_official_core=nrmse(dense_oracle, official["recurrence"]["result"][token_slice]),
                counterfactuals=counterfactuals,
                no_performance_measurement=True,
            )
            gathered = [None] * 2
            dist.all_gather_object(gathered, row, group=group)
            if rank == 0:
                report = dict(status="first_FP16_failure_staged_attribution", ranks=gathered)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps(report), flush=True)
    finally:
        for hook in hooks:
            hook.remove()


def main():
    from multi_graph_tp_boundary import select_workload
    from multi_graph_tp_boundary_loader import run_qwen35

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--ring-extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new diagnostic output required")
    args.tensor_parallel_size, args.pair, args.query, args.kv_length = 2, (200, 200), 4, 512
    args.dtype, args.batch_invariant, args.compiled = "bfloat16", False, False
    args.gdn_probe, args.gdn_pipeline = True, False
    args.selection = select_workload(args)
    run_qwen35(args, ReplayWorkload(98, 100, 4, 512, (50, 48)), experiment=experiment)


if __name__ == "__main__":
    main()
