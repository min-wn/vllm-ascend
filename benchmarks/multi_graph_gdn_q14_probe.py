# SPDX-License-Identifier: Apache-2.0
"""Single saved GDN recurrence site, independent Q1/Q4 compact prototype.

Continuous ledger metadata is combined with reused real layer operands. This is
not an exact replay of the ledger's original model activations or a service run.
"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from multi_graph_gdn_q4_backend import load_extension
from multi_graph_gdn_q14_protocol import load_manifest, metadata, stability_screen

OPERANDS = ("query", "key", "value", "beta", "gate")
MODES = ("forced_q4", "q14_ready", "q14_delayed", "native")
SCALE = 128**-0.5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(torch, operation):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    result = operation()
    torch.npu.synchronize()
    return result, (time.perf_counter_ns() - start) / 1e6


def require_equal(torch, first, second, label):
    if not torch.equal(first, second):
        error = (first.float() - second.float()).abs().max().item()
        raise AssertionError(f"{label}: not bitwise equal, max_abs={error}")


def error_metrics(torch, first, second):
    delta = first.float() - second.float()
    return dict(
        max_abs=delta.abs().max().item(),
        nrmse=(delta.square().mean().sqrt() / second.float().square().mean().sqrt().clamp_min(1e-30)).item(),
        finite=bool(torch.isfinite(first).all().item()),
        quality_qualification=False,
    )


def synthetic_steps():
    previous_accept = {owner: 1 for owner in range(8)}
    patterns = (
        (list(range(8)), [4] * 8),
        (list(reversed(range(8))), [1, 4, 1, 4, 4, 1, 4, 1]),
        ([7, 3, 0, 6, 4, 1], [4, 1, 1, 4, 1, 4]),
        ([2, 5, 0, 7, 1, 6, 3, 4], [1, 4, 4, 1, 1, 4, 4, 1]),
        (list(range(8)), [1] * 8),
    )
    rows = []
    for step, (owners, lengths) in enumerate(patterns):
        decisions = [{"accepted": 1 if q == 1 else 1 + (owner + step) % 4} for owner, q in zip(owners, lengths)]
        rows.append(
            dict(
                step=step,
                request_ids=owners,
                query_lengths=lengths,
                accepted_from_previous=[previous_accept[o] for o in owners],
                decisions=decisions,
            )
        )
        previous_accept.update({o: d["accepted"] for o, d in zip(owners, decisions)})
    return rows


class Fixture:
    def __init__(self, torch, host, owners, deferred):
        self.torch, self.owners = torch, owners
        available = sorted(set(host["ids"].tolist()) - {0})
        if len(available) < owners:
            raise ValueError("real site has too few distinct source owners")
        self.source_owners = available[:owners]
        source_rows = {}
        begin = 0
        for row, q in enumerate(host["sequence"].tolist()[1:]):
            if q:
                owner = int(host["ids"][begin])
                source_rows[owner] = list(range(begin, begin + q))
            begin += q
        indices = [r for owner in self.source_owners for r in source_rows[owner]]
        if len(indices) != 4 * owners:
            raise ValueError("reuse source must provide all four real rows per owner")
        self.operands = {name: host[name][indices].to("npu:0") for name in OPERANDS}
        # Materialize the source site's legal accepted state exactly once. Each
        # trial imports this common state, with no pending compact log at entry.
        selected_accept = {}
        begin = 0
        for q, a in zip(host["sequence"].tolist()[1:], host["accepted"].tolist()):
            if q:
                selected_accept[int(host["ids"][begin])] = a
            begin += q
        all_owners = host["base"].shape[0] - 1
        base, logs, valid = (host[name].to("npu:0") for name in ("base", "logs", "valid"))
        sequence = torch.tensor([0, *([4] * all_owners)], dtype=torch.int32, device="npu:0")
        ids = torch.arange(1, all_owners + 1, dtype=torch.int32, device="npu:0").repeat_interleave(4)
        accepted = torch.tensor(
            [selected_accept.get(o, 1) for o in range(1, all_owners + 1)], dtype=torch.int32, device="npu:0"
        )
        deferred.commit(base, sequence, ids, accepted, logs, valid, torch.zeros_like(logs), torch.zeros_like(valid))
        select = torch.tensor([0, *self.source_owners], dtype=torch.long, device="npu:0")
        self.initial = base.index_select(0, select).contiguous()
        torch.npu.synchronize()


class Trial:
    def __init__(self, torch, fixture, mode, capacity, new, old, deferred, gdn, fla):
        self.torch, self.fixture, self.mode, self.capacity = torch, fixture, mode, capacity
        self.new, self.old, self.deferred, self.gdn, self.fla = new, old, deferred, gdn, fla
        self.owners = fixture.owners
        self.device = "npu:0"
        self.buffers = {}
        for name, source in fixture.operands.items():
            self.buffers[name] = torch.zeros((4 * capacity, *source.shape[1:]), dtype=source.dtype, device=self.device)
        self.qf = torch.zeros_like(self.buffers["query"], dtype=torch.float32)
        self.kf = torch.zeros_like(self.buffers["key"], dtype=torch.float32)
        self.output = torch.zeros_like(self.buffers["value"])
        self.products = torch.zeros((capacity, 24, 128, 8), dtype=torch.float32, device=self.device)
        # A single metadata upload includes packed lengths, owners, acceptance,
        # and replay-source row indices. The last is fixture preparation.
        self.meta = torch.zeros(
            (capacity + 1 + 4 * capacity + capacity + 4 * capacity,), dtype=torch.int32, device=self.device
        )
        offset = 0
        self.sequence = self.meta[offset : offset + capacity + 1]
        offset += capacity + 1
        self.ids = self.meta[offset : offset + 4 * capacity]
        offset += 4 * capacity
        self.accepted = self.meta[offset : offset + capacity]
        offset += capacity
        self.source_indices = self.meta[offset:]
        self.new_logs = torch.zeros((self.owners + 1, 24, 1056), dtype=torch.float32, device=self.device)
        self.new_valid = torch.zeros((self.owners + 1, 24, 8), dtype=torch.int32, device=self.device)
        self.empty_logs, self.empty_valid = torch.zeros_like(self.new_logs), torch.zeros_like(self.new_valid)
        self.native_buffers, self.graphs = {}, {}
        self.terminal_output = torch.empty_like(fixture.initial)
        self.terminal_indices = torch.zeros(self.owners + 1, dtype=torch.long, device=self.device)
        _, self.initialization_ms = timed(torch, self.reset)

    def reset(self):
        torch = self.torch
        if self.mode == "native":
            self.base = torch.empty((4 * self.owners + 1, 24, 128, 128), dtype=torch.float32, device=self.device)
            self.base[:1].copy_(self.fixture.initial[:1])
            self.base[1::4].copy_(self.fixture.initial[1:])
        else:
            self.base = self.fixture.initial.clone()
        self.logs = torch.zeros_like(self.new_logs)
        self.valid = torch.zeros_like(self.new_valid)
        self.slot_maps = {o: list(range(4 * o + 1, 4 * o + 5)) for o in range(self.owners)}
        self.last_accept = {o: 1 for o in range(self.owners)}
        self.last_width = {}
        self.new_logs.zero_()
        self.new_valid.zero_()

    def restore(self):
        # Retain graph addresses; benchmark reset is outside the measured chain.
        if self.mode == "native":
            self.base[:1].copy_(self.fixture.initial[:1])
            self.base[1::4].copy_(self.fixture.initial[1:])
        else:
            self.base.copy_(self.fixture.initial)
        self.logs.zero_()
        self.valid.zero_()
        self.new_logs.zero_()
        self.new_valid.zero_()
        self.slot_maps = {o: list(range(4 * o + 1, 4 * o + 5)) for o in range(self.owners)}
        self.last_accept = {o: 1 for o in range(self.owners)}
        self.last_width = {}

    def prepare(self, row):
        torch = self.torch
        owners, actual = row["request_ids"], row["query_lengths"]
        previous = row["accepted_from_previous"]
        for owner, accepted in zip(owners, previous):
            if owner in self.last_width and accepted > self.last_width[owner]:
                raise ValueError("acceptance exceeds the actual previous query width")
        lengths = [4] * len(owners) if self.mode == "forced_q4" else actual
        meta = metadata(owners, lengths, previous, self.capacity)
        source_indices = [4 * o + t for o, q in zip(owners, lengths) for t in range(q)]
        source_indices += [0] * (4 * self.capacity - len(source_indices))
        payload = [*meta["sequence"], *meta["ids"], *meta["accepted"], *source_indices]
        self.meta.copy_(torch.tensor(payload, dtype=torch.int32))
        count = sum(lengths)
        # Reused real activations are already device-resident. Only actual rows
        # are gathered; this fixture-only preparation is included and reported.
        select = self.source_indices[:count].long()
        for name, source in self.fixture.operands.items():
            self.buffers[name][:count].copy_(source.index_select(0, select))
        self.count = count
        self.key = (sum(q == 1 for q in actual), sum(q == 4 for q in actual)) if self.mode == "native" else "compact"
        if self.mode == "native":
            self.prepare_native(owners, actual, previous)
        self.last_width.update(dict(zip(owners, actual)))
        self.last_accept.update({o: d["accepted"] for o, d in zip(owners, row["decisions"])})
        return meta

    def prepare_native(self, owners, lengths, previous):
        torch = self.torch
        n1, n4 = self.key
        if self.key not in self.native_buffers:
            self.native_buffers[self.key] = {
                "q1_rows": torch.empty(n1, dtype=torch.long, device=self.device),
                "q4_rows": torch.empty(4 * n4, dtype=torch.long, device=self.device),
                "q1_ids": torch.empty(n1, dtype=torch.int32, device=self.device),
                "q4_ids": torch.empty(4 * n4, dtype=torch.int32, device=self.device),
                "q1_sequence": torch.tensor([0, *([1] * n1)], dtype=torch.int32, device=self.device),
                "q4_sequence": torch.tensor([0, *([4] * n4)], dtype=torch.int32, device=self.device),
                "q4_accepted": torch.ones(n4, dtype=torch.int32, device=self.device),
                "has_state": torch.ones(n1, dtype=torch.bool, device=self.device),
            }
        lists = {name: [] for name in ("q1_rows", "q4_rows", "q1_ids", "q4_ids")}
        begin = 0
        for owner, q, a in zip(owners, lengths, previous):
            selected = self.slot_maps[owner][a - 1 if owner in self.last_width else 0]
            slots = (
                [selected] if q == 1 else [selected, *(s for s in range(4 * owner + 1, 4 * owner + 5) if s != selected)]
            )
            prefix = "q1" if q == 1 else "q4"
            lists[prefix + "_rows"].extend(range(begin, begin + q))
            lists[prefix + "_ids"].extend(slots)
            self.slot_maps[owner] = slots
            begin += q
        for name, values in lists.items():
            target = self.native_buffers[self.key][name]
            target.copy_(torch.tensor(values, dtype=target.dtype))

    def operation(self):
        self.output.zero_()
        if self.mode == "native":
            self.native_operation()
            return
        b = self.buffers
        # Same fixed-capacity casts for both compact implementations. Record
        # these explicitly; reduced logical rows do not imply reduced cast work.
        self.qf.copy_(b["query"])
        self.kf.copy_(b["key"])
        tile = self.old if self.mode == "forced_q4" else self.new
        tile.solve(
            self.base,
            self.qf,
            self.kf,
            b["query"],
            b["key"],
            b["value"],
            b["beta"],
            b["gate"],
            self.sequence,
            self.ids,
            self.accepted,
            self.products,
            self.output,
            self.logs,
            self.valid,
            self.new_logs,
            self.new_valid,
            20,
            self.mode != "q14_delayed",
        )
        commit = self.deferred if self.mode == "forced_q4" else self.new
        commit.commit(
            self.base, self.sequence, self.ids, self.accepted, self.logs, self.valid, self.new_logs, self.new_valid
        )

    def native_operation(self):
        n1, n4 = self.key
        b = self.native_buffers[self.key]

        def operands(prefix):
            return [self.buffers[name].index_select(0, b[prefix + "_rows"]) for name in OPERANDS]

        if n4:
            q, k, v, beta, gate = operands("q4")
            output = self.gdn.recurrent_gated_delta_rule(
                q,
                k,
                v,
                self.base,
                beta=beta,
                g=gate,
                scale=SCALE,
                actual_seq_lengths=b["q4_sequence"],
                ssm_state_indices=b["q4_ids"],
                num_accepted_tokens=b["q4_accepted"],
            )
            self.output.index_copy_(0, b["q4_rows"], output)
        if not n1:
            return
        q, k, v, beta, gate = operands("q1")
        if not n4:
            output = self.gdn.recurrent_gated_delta_rule(
                q,
                k,
                v,
                self.base,
                beta=beta,
                g=gate,
                scale=SCALE,
                actual_seq_lengths=b["q1_sequence"],
                ssm_state_indices=b["q1_ids"],
            )
        else:
            state = self.gdn.gather_ssm_states(self.base, b["q1_ids"], b["has_state"])
            output, final, *_ = self.fla(
                q.unsqueeze(0).transpose(1, 2).contiguous(),
                k.unsqueeze(0).transpose(1, 2).contiguous(),
                v.unsqueeze(0).transpose(1, 2).contiguous(),
                gate.unsqueeze(0).float().contiguous(),
                beta.unsqueeze(0).contiguous(),
                initial_state=state.transpose(-1, -2).contiguous(),
                output_final_state=True,
                chunk_size=64,
                cu_seqlens=tuple(range(n1 + 1)),
                chunk_indices=tuple(x for i in range(n1) for x in (i, 0)),
                scale=SCALE,
                layout="BNSD",
                use_exp2=False,
                use_qk_l2norm_in_kernel=False,
                allow_neg_eigval=False,
                disable_recompute=True,
                state_v_first=False,
            )
            self.gdn.scatter_ssm_states_(
                self.base, b["q1_ids"], final.transpose(-1, -2).contiguous().to(self.base.dtype)
            )
            output = output.reshape(n1, 24, 128)
        self.output.index_copy_(0, b["q1_rows"], output)

    def export(self):
        if self.mode == "native":
            raise ValueError("compact-only diagnostic export")
        result = self.torch.empty((self.owners + 1, 4, 24, 128, 128), dtype=self.torch.float32, device=self.device)
        extension = self.deferred if self.mode == "forced_q4" else self.new
        extension.export_candidates(self.base, self.logs, self.valid, result)
        return result

    def terminal(self):
        torch = self.torch
        if self.mode == "native":
            selected = [0, *(self.slot_maps[o][self.last_accept[o] - 1] for o in range(self.owners))]
            self.terminal_indices.copy_(torch.tensor(selected, dtype=torch.long))
            torch.index_select(self.base, 0, self.terminal_indices, out=self.terminal_output)
            return self.terminal_output
        q = 4 if self.mode == "forced_q4" else 1
        meta = metadata(
            list(range(self.owners)),
            [q] * self.owners,
            [self.last_accept[o] for o in range(self.owners)],
            self.capacity,
        )
        payload = [*meta["sequence"], *meta["ids"], *meta["accepted"], *([0] * (4 * self.capacity))]
        self.meta.copy_(torch.tensor(payload, dtype=torch.int32))
        extension = self.deferred if self.mode == "forced_q4" else self.new
        extension.commit(
            self.base, self.sequence, self.ids, self.accepted, self.logs, self.valid, self.empty_logs, self.empty_valid
        )
        return self.base

    def capture(self, rows):
        torch = self.torch
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        pool = torch.npu.graph_pool_handle()
        for row in rows:
            self.prepare(row)
            self.operation()
        torch.npu.synchronize()
        self.restore()
        for row in rows:
            self.prepare(row)
            if self.key not in self.graphs:
                torch.npu.synchronize()
                stream.wait_stream(torch.npu.current_stream())
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=pool, stream=stream):
                    self.operation()
                torch.npu.synchronize()
                self.graphs[self.key] = graph
        self.restore()
        torch.npu.synchronize()


def audit(torch, fixture, rows, capacity, new, old, deferred, gdn, fla, graph=False, native=False):
    modes = MODES if native else MODES[:3]
    trials = {m: Trial(torch, fixture, m, capacity, new, old, deferred, gdn, fla) for m in modes}
    if graph:
        for trial in trials.values():
            trial.capture(rows)
    records = []
    for row in rows:
        inactive = sorted(set(range(fixture.owners)) - set(row["request_ids"]))
        paused = {
            m: [(t.base[o + 1].clone(), t.logs[o + 1].clone(), t.valid[o + 1].clone()) for o in inactive]
            for m, t in trials.items()
            if m != "native"
        }
        for trial in trials.values():
            trial.prepare(row)
            trial.graphs[trial.key].replay() if graph else trial.operation()
        torch.npu.synchronize()
        ready, delayed, forced = (trials[m] for m in ("q14_ready", "q14_delayed", "forced_q4"))
        for name in ("output", "base", "logs", "valid", "new_logs", "new_valid"):
            require_equal(torch, getattr(ready, name), getattr(delayed, name), f"step{row['step']}/{name}/schedule")
        require_equal(torch, ready.base, forced.base, "forced-Q4 accepted base")
        e_ready, e_delayed, e_forced = (t.export() for t in (ready, delayed, forced))
        require_equal(torch, e_ready, e_delayed, "schedule candidate export")
        begin = 0
        for j, (owner, q) in enumerate(zip(row["request_ids"], row["query_lengths"])):
            require_equal(
                torch, ready.output[begin : begin + q], forced.output[4 * j : 4 * j + q], "valid output prefix"
            )
            require_equal(torch, ready.valid[owner + 1], torch.full_like(ready.valid[owner + 1], q), "published width")
            for offset, count in ((0, q * 128), (512, q * 128), (1024, q * 8)):
                require_equal(
                    torch,
                    ready.logs[owner + 1, :, offset : offset + count],
                    forced.logs[owner + 1, :, offset : offset + count],
                    "valid log prefix",
                )
            require_equal(torch, e_ready[owner + 1, :q], e_forced[owner + 1, :q], "valid candidate states")
            for start, stop in ((q * 128, 512), (512 + q * 128, 1024), (1024 + q * 8, 1056)):
                if torch.count_nonzero(ready.logs[owner + 1, :, start:stop]).item():
                    raise AssertionError("Q1 inactive log tail is not clear")
            begin += q
        for mode, before_owners in paused.items():
            trial = trials[mode]
            for o, before in zip(inactive, before_owners):
                for saved, current in zip(before, (trial.base[o + 1], trial.logs[o + 1], trial.valid[o + 1])):
                    require_equal(torch, saved, current, "paused compact state")
        native_metric = error_metrics(torch, trials["native"].output[:begin], ready.output[:begin]) if native else None
        if native and not native_metric["finite"]:
            raise AssertionError("native control has nonfinite output")
        records.append(
            dict(
                step=row["step"],
                graph=graph,
                exact_schedule=True,
                exact_forced_valid_prefixes=True,
                actual_query_tokens=begin,
                native_vs_compact_output=native_metric,
            )
        )
    terminal = {m: t.terminal() for m, t in trials.items()}
    for mode in ("q14_delayed", "forced_q4"):
        require_equal(torch, terminal["q14_ready"], terminal[mode], "terminal materialization")
    for mode, trial in trials.items():
        if mode == "native":
            continue
        if torch.count_nonzero(trial.valid).item() or torch.count_nonzero(trial.logs).item():
            raise AssertionError("terminal debt remains")
    native_terminal = error_metrics(torch, terminal["native"], terminal["q14_ready"]) if native else None
    if native and not native_terminal["finite"]:
        raise AssertionError("native control has nonfinite terminal state")
    return dict(
        steps=records,
        terminal_bitwise=True,
        quality_failures_preserved=True,
        native_vs_compact_terminal=native_terminal,
    )


def measure(torch, trials, rows, repeats):
    samples = []
    modes = ("forced_q4", "q14_ready", "native")
    for round_id in range(3):
        order = modes if round_id % 2 == 0 else tuple(reversed(modes))
        for mode in order:
            trial = trials[mode]
            for sample in range(repeats):
                _, initialization_ms = timed(torch, trial.restore)
                events = [tuple(torch.npu.Event(enable_timing=True) for _ in range(3)) for _ in rows]
                start = time.perf_counter_ns()
                for row, (begin, compute_begin, end) in zip(rows, events):
                    begin.record()
                    trial.prepare(row)
                    compute_begin.record()
                    trial.graphs[trial.key].replay()
                    end.record()
                torch.npu.synchronize()
                execute_ms = (time.perf_counter_ns() - start) / 1e6
                _, terminal_ms = timed(torch, trial.terminal)
                by_mode = {"q1_only": 0.0, "q4_only": 0.0, "mixed": 0.0}
                compute_by_mode = dict.fromkeys(by_mode, 0.0)
                prepare_by_mode = dict.fromkeys(by_mode, 0.0)
                for row, (begin, compute_begin, end) in zip(rows, events):
                    widths = set(row["query_lengths"])
                    key = "q1_only" if widths == {1} else "q4_only" if widths == {4} else "mixed"
                    by_mode[key] += begin.elapsed_time(end)
                    compute_by_mode[key] += compute_begin.elapsed_time(end)
                    prepare_by_mode[key] += begin.elapsed_time(compute_begin)
                samples.append(
                    dict(
                        round=round_id,
                        sample=sample,
                        mode=mode,
                        execution_ms=execute_ms,
                        terminal_ms=terminal_ms,
                        initialization_ms=initialization_ms,
                        lifecycle_ms=initialization_ms + execute_ms + terminal_ms,
                        device_step_endpoint_ms=by_mode,
                        device_compute_ms=compute_by_mode,
                        device_preparation_ms=prepare_by_mode,
                    )
                )
    medians = {
        mode: [
            statistics.median(s["lifecycle_ms"] for s in samples if s["mode"] == mode and s["round"] == r)
            for r in range(3)
        ]
        for mode in modes
    }
    return dict(
        samples=samples,
        round_medians_lifecycle_ms=medians,
        versus_forced_q4=stability_screen(medians["q14_ready"], medians["forced_q4"]),
        versus_native=stability_screen(medians["q14_ready"], medians["native"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-extension", type=Path, required=True)
    parser.add_argument("--old-extension", type=Path, required=True)
    parser.add_argument("--deferred-extension", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("correctness", "performance"), default="correctness")
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence; use a new output")
    import torch
    import torch_npu

    import vllm_ascend.ops.gdn as gdn
    from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    new = load_extension("gdn_q14", args.new_extension)
    old = load_extension("gdn_q4_tile_pipeline", args.old_extension)
    deferred = load_extension("gdn_q4_log_deferred", args.deferred_extension)
    profile = get_current_hardware_profile()
    fla = gdn._get_fla_gdn_prefill_op()
    if not profile.supports(HardwareCapability.FLA_GDN_PREFILL) or fla is None:
        raise RuntimeError("inspected A2 FLA dispatch unavailable; do not silently choose another native path")
    manifest = load_manifest(args.manifest)["payload"]
    report = dict(
        status="running",
        phase=args.phase,
        input_mode="real_site_reuse",
        initial_state_matches_original_ledger_step=False,
        boundary="single real GDN recurrence site after Q/K normalization, no conv/projection/norm/TP/head",
        source_owners=None,
        installed_native_mixed="Q4 recurrence plus Q1 FLA chunk below normalization",
        precision="BF16 operands/output, FP32 compact state/log/products; HF32 off; highest",
        q1_candidate_filler=0,
        cube_columns=8,
        cube_q1_useful_columns=2,
        compact_qk_cast_rows="fixed maximum 4*capacity in both implementations",
        metadata_and_fixture_gather_in_timing=True,
        service_claim=False,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        sources={
            str(p): sha(p)
            for p in (
                Path(__file__),
                args.site,
                args.manifest,
                args.new_extension,
                args.old_extension,
                args.deferred_extension,
            )
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        with torch.inference_mode():
            host = torch.load(args.site, map_location="cpu", weights_only=True)
            if args.phase == "correctness":
                fixture = Fixture(torch, host, 8, deferred)
                rows = synthetic_steps()
                report["source_owners"] = fixture.source_owners
                report["eager"] = audit(torch, fixture, rows, 8, new, old, deferred, gdn, fla)
                save()
                report["graph"] = audit(torch, fixture, rows, 8, new, old, deferred, gdn, fla, graph=True)
            else:
                rows = manifest["steps"]
                fixture = Fixture(torch, host, len(manifest["stable_owners"]), deferred)
                if manifest["stable_owners"] != list(range(fixture.owners)):
                    raise ValueError("fixture requires stable contiguous owners; do not silently remap")
                capacity = max(s["capacity_tokens"] for s in rows) // 4
                report["segment_audit"] = audit(
                    torch, fixture, rows, capacity, new, old, deferred, gdn, fla, graph=True, native=True
                )
                save()
                trials = {
                    m: Trial(torch, fixture, m, capacity, new, old, deferred, gdn, fla)
                    for m in ("forced_q4", "q14_ready", "native")
                }
                mixed = next(row for row in rows if set(row["query_lengths"]) == {1, 4})
                selected = [i for i, q in enumerate(mixed["query_lengths"]) if q == 4]
                q4_control = dict(
                    step=mixed["step"],
                    request_ids=[mixed["request_ids"][i] for i in selected],
                    query_lengths=[4] * len(selected),
                    accepted_from_previous=[mixed["accepted_from_previous"][i] for i in selected],
                    decisions=[mixed["decisions"][i] for i in selected],
                )
                for trial in trials.values():
                    trial.capture([*rows, q4_control])
                report.update(
                    source_owners=fixture.source_owners,
                    capacity_requests=capacity,
                    manifest_summary=manifest["summary"],
                    performance=measure(torch, trials, rows, args.repeats),
                    isolated_q4_control=dict(
                        selection="Q4 owners from first frozen mixed step",
                        row=q4_control,
                        measurement=measure(torch, trials, [q4_control], args.repeats),
                    ),
                )
        report["status"] = "complete_single_layer_only"
        save()
        print(json.dumps(dict(status=report["status"], output=str(args.output))), flush=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
