"""Device qualification of compact GDN states; not a decoder benchmark.

Compare a copied official recurrence kernel with its log-emitting variant.
The compact endpoint includes accepted-prefix reconstruction on every step.
All four logical states are materialized for correctness outside timed replay.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument(
        "--mixed", action="store_true", help="include per-request accepted counts 1-4 in correctness steps"
    )
    parser.add_argument(
        "--fused", action="store_true", help="restore accepted prefix inside recurrence using one log bank"
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="also isolate non-endpoint emit/reconstruction costs"
    )
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 3:
        parser.error("new output path and at least three samples required")
    import torch
    import torch.nn.functional as functional
    import torch_npu  # noqa: F401
    from fla_npu.ops.ascendc import recurrent_gated_delta_rule

    spec = importlib.util.spec_from_file_location("gdn_compact_state", args.extension)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    batch, heads, key_heads, width = 100, 24, 8, 128
    torch.manual_seed(742)
    q, k = tuple(
        functional.normalize(torch.randn(batch * 4, key_heads, width, device=device), dim=-1).to(torch.bfloat16)
        for _ in range(2)
    )
    value = (torch.randn(batch * 4, heads, width, device=device) * 0.05).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(batch * 4, heads, device=device)).to(torch.bfloat16)
    gate = -torch.rand(batch * 4, heads, device=device) * 0.2
    initial = (torch.randn(batch + 1, heads, width, width, device=device) * 0.01).to(torch.bfloat16)
    dense_initial = torch.cat((initial[:1], initial[1:].repeat_interleave(4, dim=0)))
    dense, compact, official = dense_initial.clone(), initial.clone(), dense_initial.clone()
    sequence = torch.empty(batch + 1, dtype=torch.int32, device=device)
    lengths = torch.empty(batch, dtype=torch.int32, device=device)
    indices = torch.empty(batch * 4, dtype=torch.int32, device=device)
    compact_indices = torch.empty_like(indices)
    accepted = torch.empty(batch, dtype=torch.int32, device=device)
    outputs = {name: torch.zeros_like(value) for name in ("dense", "compact")}
    gamma_log = torch.ones(batch + 1, 4, heads, 8, device=device)
    key_log = torch.zeros(batch + 1, 4, heads, width, device=device)
    update_log = torch.zeros_like(key_log)
    dummy_logs = tuple(torch.empty_like(value) for value in (gamma_log, key_log, update_log))
    state_tensors = (dense, compact, official, gamma_log, key_log, update_log, *outputs.values())

    def snapshot():
        return tuple(tensor.clone() for tensor in state_tensors)

    def restore(saved):
        for tensor, prior in zip(state_tensors, saved, strict=True):
            tensor.copy_(prior)
        torch.npu.synchronize()

    def finite():
        for tensor in state_tensors:
            if not torch.isfinite(tensor).all().item():
                raise AssertionError("nonfinite recurrence output, state or compact log")

    def change_inputs():
        for tensor in (q, k):
            tensor.copy_(functional.normalize(torch.randn_like(tensor, dtype=torch.float32), dim=-1))
        value.copy_(torch.randn_like(value, dtype=torch.float32) * 0.05)
        beta.copy_(torch.sigmoid(torch.randn_like(beta, dtype=torch.float32)))
        gate.copy_(-torch.rand_like(gate) * 0.2)
        torch.npu.synchronize()

    def metadata(live, count):
        lengths.copy_(torch.tensor([4 if request < live else 0 for request in range(batch)], device=device))
        sequence[0] = 0
        # FLA uses a leading zero followed by per-request lengths, not offsets.
        sequence[1:].copy_(lengths)
        indices.copy_(torch.tensor([1 + token if token < live * 4 else 0 for token in range(batch * 4)], device=device))
        compact_indices.copy_(
            torch.tensor([1 + token // 4 if token < live * 4 else 0 for token in range(batch * 4)], device=device)
        )
        if count == "mixed":
            accepted.copy_(torch.tensor([1 + request % 4 for request in range(batch)], device=device))
        else:
            accepted.fill_(count)
        torch.npu.synchronize()

    def dense_op():
        module.emit(q, k, value, beta, gate, dense, sequence, indices, accepted, outputs["dense"], *dummy_logs, False)
        return outputs["dense"]

    def official_op():
        return recurrent_gated_delta_rule(
            q,
            k,
            value,
            official,
            g=gate,
            beta=beta,
            scale=width**-0.5,
            actual_seq_lengths=sequence,
            ssm_state_indices=indices,
            num_accepted_tokens=accepted,
        )

    def compact_op():
        if args.fused:
            module.fused_emit(
                q,
                k,
                value,
                beta,
                gate,
                compact,
                sequence,
                compact_indices,
                accepted,
                outputs["compact"],
                gamma_log,
                key_log,
                update_log,
            )
            return outputs["compact"]
        module.reconstruct(compact, gamma_log, key_log, update_log, lengths, accepted, compact)
        module.emit(
            q,
            k,
            value,
            beta,
            gate,
            compact,
            sequence,
            compact_indices,
            accepted,
            outputs["compact"],
            gamma_log,
            key_log,
            update_log,
            True,
        )
        return outputs["compact"]

    def export(prefix):
        count = torch.full_like(accepted, prefix)
        restored = compact.clone()
        # Inactive requests retain their previous four logical candidates.
        module.reconstruct(compact, gamma_log, key_log, update_log, torch.full_like(lengths, 4), count, restored)
        return restored

    def check_candidates():
        for prefix in (1, 2, 3, 4):
            restored = export(prefix)
            torch.npu.synchronize()
            wanted = dense[1:].view(batch, 4, heads, width, width)[:, prefix - 1]
            torch.testing.assert_close(restored[1:], wanted, rtol=0, atol=0)

    def check_preserved(before, live):
        for tensor, prior in zip((compact, gamma_log, key_log, update_log), before, strict=True):
            torch.testing.assert_close(tensor[0], prior[0], rtol=0, atol=0)
            torch.testing.assert_close(tensor[1 + live :], prior[1 + live :], rtol=0, atol=0)

    checks = []
    metadata(98, 1)
    eager_counts = (1, 2, 4, "mixed" if args.mixed else 1, 4, 2, 1, 4)
    for step, (live, count) in enumerate(zip((98, 97, 99, 98, 98, 97, 99, 98), eager_counts, strict=True)):
        change_inputs()
        metadata(live, count)
        before_dense = dense.clone()
        before_compact = tuple(tensor.clone() for tensor in (compact, gamma_log, key_log, update_log))
        dense_op()
        reference = official_op()
        compact_op()
        torch.npu.synchronize()
        torch.testing.assert_close(outputs["dense"][: live * 4], reference[: live * 4], rtol=0, atol=0)
        torch.testing.assert_close(dense, official, rtol=0, atol=0)
        torch.testing.assert_close(outputs["compact"][: live * 4], outputs["dense"][: live * 4], rtol=0, atol=0)
        check_candidates()
        torch.testing.assert_close(dense[0], before_dense[0], rtol=0, atol=0)
        torch.testing.assert_close(dense[1 + live * 4 :], before_dense[1 + live * 4 :], rtol=0, atol=0)
        check_preserved(before_compact, live)
        finite()
        checks.append(dict(step=step, live=live, accepted=count, official_and_all_candidates_exact=True))

    graphs = {}
    graph_outputs = {}
    operations = (("dense", dense_op), ("compact", compact_op), ("official", official_op))
    metadata(98, 1)
    for name, operation in operations:
        saved = snapshot()
        stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
            graph_outputs[name] = operation()
        torch.npu.synchronize()
        restore(saved)
        graphs[name] = (graph, stream)
    # Both graphs reuse stable pointers through changing live/accepted metadata.
    # Each replay starts from the exact state/log/output contents used by eager.
    graph_counts = (1, 2, "mixed" if args.mixed else 4, 1)
    for step, (live, count) in enumerate(zip((98, 97, 99, 98), graph_counts, strict=True)):
        change_inputs()
        metadata(live, count)
        before_compact = tuple(tensor.clone() for tensor in (compact, gamma_log, key_log, update_log))
        for name, operation in operations:
            saved = snapshot()
            eager_output = operation()
            torch.npu.synchronize()
            expected_output = eager_output.clone()
            expected = snapshot()
            restore(saved)
            graph, stream = graphs[name]
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                graph.replay()
            torch.npu.synchronize()
            for actual, wanted in zip(state_tensors, expected, strict=True):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
            torch.testing.assert_close(graph_outputs[name][: live * 4], expected_output[: live * 4], rtol=0, atol=0)
            checks.append(dict(mode=name, step=step, live=live, accepted=count, eager_replay_exact=True))
        torch.testing.assert_close(outputs["dense"], outputs["compact"], rtol=0, atol=0)
        torch.testing.assert_close(dense, official, rtol=0, atol=0)
        check_preserved(before_compact, live)
        check_candidates()
        finite()
    samples = []
    timing_initial = snapshot()
    if args.diagnostics:

        def emit_only():
            module.emit(
                q,
                k,
                value,
                beta,
                gate,
                compact,
                sequence,
                compact_indices,
                accepted,
                outputs["compact"],
                gamma_log,
                key_log,
                update_log,
                True,
            )
            return outputs["compact"]

        def reconstruct_only():
            module.reconstruct(compact, gamma_log, key_log, update_log, lengths, accepted, compact)
            return compact

        for name, operation in (("diagnostic_emit_only", emit_only), ("diagnostic_reconstruct_only", reconstruct_only)):
            restore(timing_initial)
            stream, graph = torch.npu.Stream(), torch.npu.NPUGraph()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.graph(graph, pool=torch.npu.graph_pool_handle(), stream=stream):
                graph_outputs[name] = operation()
            torch.npu.synchronize()
            graphs[name] = (graph, stream)
        restore(timing_initial)
    for count in (1, 2, 4):
        metadata(98, count)
        for round_id in range(3):
            for name in tuple(graphs) if round_id % 2 == 0 else tuple(reversed(graphs)):
                restore(timing_initial)
                graph, stream = graphs[name]
                stream.wait_stream(torch.npu.current_stream())
                for _ in range(3):
                    with torch.npu.stream(stream):
                        graph.replay()
                    torch.npu.synchronize()
                for sample in range(args.repeats):
                    begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
                    started = time.perf_counter()
                    with torch.npu.stream(stream):
                        begin.record(stream)
                        graph.replay()
                        end.record(stream)
                    end.synchronize()
                    samples.append(
                        dict(
                            mode=name,
                            accepted=count,
                            round=round_id,
                            sample=sample,
                            wall_us=(time.perf_counter() - started) * 1e6,
                            device_us=begin.elapsed_time(end) * 1000,
                        )
                    )
                finite()
                if not torch.isfinite(graph_outputs[name][: 98 * 4]).all().item():
                    raise AssertionError("nonfinite active output after timed replay")
    report = dict(
        status="qualified_recurrence_and_reconstruction_not_full_model",
        checks=checks,
        samples=samples,
        compact_endpoint=(
            "fused ordered accepted-prefix reconstruction, BF16 base round/write, and next recurrence"
            if args.fused
            else "in-place accepted-prefix reconstruction plus next log-emitting recurrence"
        ),
        fused=args.fused,
        mixed_per_request_correctness=args.mixed,
        excluded="convolution/gating preparation, model layers, state export and serving cache integration",
        dense_storage=list(dense.shape),
        compact_storage=list(compact.shape),
        logs=[list(t.shape) for t in (gamma_log, key_log, update_log)],
        diagnostic_modes_are_not_complete_endpoints=args.diagnostics,
        extension_sha256=hashlib.sha256(args.extension.read_bytes()).hexdigest(),
        device="single NPU 0; not TP2",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=report["status"], checks=len(checks)), indent=2))


if __name__ == "__main__":
    main()
