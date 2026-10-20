# SPDX-License-Identifier: Apache-2.0
"""Continuous graph qualification against an independently materialized oracle."""

import argparse
import json
from pathlib import Path

from multi_graph_replayssm_protocol import Cursor


def oracle_step(torch, cache, cursors, query, key, value, beta, gate, live, accepted):
    base, ring, gates = cache
    output = torch.zeros_like(value)
    for owner in range(1, live + 1):
        cursor = cursors[owner - 1].advance(accepted[owner - 1])
        cursors[owner - 1] = cursor
        state = base[owner].clone()
        for history in range(cursor.write_pos):
            physical = (cursor.cache_base + history) % 32
            logs = ring[owner, :, physical].float()
            gamma = gates[owner, :, physical, 0].exp()
            state = state * gamma[:, None, None] + logs[:, 128:, None] * logs[:, None, :128]
        if cursor.is_flush:
            base[owner].copy_(state)
        begin = (owner - 1) * 4
        for token in range(4):
            k = key[begin + token].float().repeat_interleave(3, 0)
            q = query[begin + token].float().repeat_interleave(3, 0)
            g = gate[begin + token]
            state = state * g.exp()[:, None, None]
            delta = beta[begin + token].float()[:, None] * (
                value[begin + token].float() - (state @ k[:, :, None]).squeeze(-1)
            )
            state = state + delta[:, :, None] * k[:, None, :]
            output[begin + token] = (state @ q[:, :, None]).squeeze(-1) * 128**-0.5
            physical = (cursor.cache_base + cursor.write_pos + token) % 32
            ring[owner, :, physical, :128] = k.to(ring.dtype)
            ring[owner, :, physical, 128:] = delta.to(ring.dtype)
            gates[owner, :, physical] = g[:, None]
    return output


def run(args):
    import torch
    import torch_npu  # noqa: F401
    from multi_graph_gdn_q4_backend import load_extension

    torch.npu.set_device(0)
    torch.npu.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.manual_seed(731)
    module = load_extension("gdn_replayssm", args.extension)
    batch, slots = 5, 6
    base = torch.randn(slots, 24, 128, 128, device="npu") * 0.01
    ring = torch.zeros(slots, 24, 32, 256, dtype=torch.float32, device="npu")
    gates = torch.zeros(slots, 24, 32, 8, device="npu")
    cursors = torch.zeros(slots, 4, 8, dtype=torch.int32, device="npu")
    query, key = (torch.zeros(batch * 4, 8, 128, dtype=torch.bfloat16, device="npu") for _ in range(2))
    value = torch.zeros(batch * 4, 24, 128, dtype=torch.bfloat16, device="npu")
    beta = torch.zeros(batch * 4, 24, dtype=torch.bfloat16, device="npu")
    gate = torch.zeros(batch * 4, 24, device="npu")
    qf, kf = query.float(), key.float()
    sequence = torch.full((batch + 1,), 4, dtype=torch.int32, device="npu")
    sequence[0] = 0
    ids = torch.arange(1, slots, dtype=torch.int32, device="npu").repeat_interleave(4)
    accepted = torch.ones(batch, dtype=torch.int32, device="npu")
    products = torch.empty(batch, 24, 128, 8, device="npu")
    output = torch.zeros_like(value)
    mutable = (base, ring, gates, cursors)
    initial = tuple(tensor.clone() for tensor in mutable)

    def operation(streaming):
        output.zero_()
        qf.copy_(query)
        kf.copy_(key)
        module.state(*mutable, sequence, ids, accepted, 0)
        module.state(*mutable, sequence, ids, accepted, 1)
        module.solve(
            base,
            qf,
            kf,
            query,
            key,
            value,
            beta,
            gate,
            sequence,
            ids,
            accepted,
            products,
            output,
            ring,
            gates,
            cursors,
            streaming,
        )

    graphs = {}
    for mode in (False, True):
        operation(mode)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            operation(mode)
        torch.npu.synchronize()
        graphs[mode] = graph
    for tensor, seed in zip(mutable, initial, strict=True):
        tensor.copy_(seed)
    reference = [tensor.clone() for tensor in initial[:3]]
    host_cursors = [Cursor() for _ in range(batch)]
    canonical = initial
    checks = []
    for step in range(40):
        live = (5, 4, 5, 3, 5)[step % 5]
        counts = [1 + (owner + step) % 4 for owner in range(batch)]
        torch.manual_seed(20719 + step)
        for tensor in (query, key, value):
            tensor.copy_(torch.randn_like(tensor) * 0.1)
        key.copy_(torch.nn.functional.normalize(key.float(), dim=-1))
        query.copy_(torch.nn.functional.normalize(query.float(), dim=-1))
        beta.fill_(0.5)
        gate.fill_(-0.1)
        sequence[1:].fill_(0)
        sequence[1 : live + 1].fill_(4)
        ids.zero_()
        ids[: live * 4].copy_(torch.arange(1, live + 1, dtype=torch.int32, device="npu").repeat_interleave(4))
        accepted.copy_(torch.tensor(counts, dtype=torch.int32, device="npu"))
        expected = oracle_step(torch, reference, host_cursors, query, key, value, beta, gate, live, counts)
        results = []
        for mode in (False, True):
            for target, saved in zip(mutable, canonical, strict=True):
                target.copy_(saved)
            graphs[mode].replay()
            torch.npu.synchronize()
            results.append((output.clone(), tuple(tensor.clone() for tensor in mutable)))
        for current, wanted in zip((results[1][0], *results[1][1]), (results[0][0], *results[0][1]), strict=True):
            torch.testing.assert_close(current, wanted, rtol=0, atol=0)
        nrmse = ((output.float() - expected.float()).square().sum() / expected.float().square().sum()).sqrt().item()
        if not nrmse < 0.005:
            raise AssertionError(f"step {step} oracle NRMSE {nrmse}")
        cursor_cpu = cursors.cpu()
        for owner, cursor in enumerate(host_cursors, 1):
            assert cursor_cpu[owner, :, 0].tolist() == [
                cursor.write_pos,
                cursor.cache_base,
                int(cursor.is_flush),
                int(cursor.pending),
            ]
        for target, saved in zip(mutable, canonical, strict=True):
            torch.testing.assert_close(target[0], saved[0], rtol=0, atol=0)
            torch.testing.assert_close(target[live + 1 :], saved[live + 1 :], rtol=0, atol=0)
        canonical = results[1][1]
        checks.append(dict(step=step, live=live, same_algorithm_exact=True, oracle_nrmse=nrmse))
    final_counts = [1, 2, 3, 4, 1]
    terminal = reference[0].clone()
    for owner, count in enumerate(final_counts, 1):
        cursor = host_cursors[owner - 1].advance(count)
        for token in range(cursor.write_pos):
            physical = (cursor.cache_base + token) % 32
            logs = reference[1][owner, :, physical].float()
            gamma = reference[2][owner, :, physical, 0].exp()
            terminal[owner] = terminal[owner] * gamma[:, None, None] + logs[:, 128:, None] * logs[:, None, :128]
    accepted.copy_(torch.tensor(final_counts, dtype=torch.int32, device="npu"))
    for phase in (0, 2, 3):
        module.state(*mutable, sequence, ids, accepted, phase)
    torch.npu.synchronize()
    terminal_error = (
        ((base - terminal).square().sum((1, 2, 3)) / terminal.square().sum((1, 2, 3)).clamp_min(1e-30))
        .sqrt()
        .max()
        .item()
    )
    assert terminal_error < 0.005
    assert not cursors.count_nonzero().item()
    torch.testing.assert_close(base[0], initial[0][0], rtol=0, atol=0)
    report = dict(
        status="qualified_40_step_ring_graph",
        checks=checks,
        max_oracle_nrmse=max(row["oracle_nrmse"] for row in checks),
        terminal_state_worst_nrmse=terminal_error,
        terminal_pending_empty=True,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
