# SPDX-License-Identifier: Apache-2.0
"""Bounded complete decoder-layer Q1/Q4 comparison; no runner integration."""

import argparse
import copy
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

from multi_graph_gdn_q4_backend import load_extension
from multi_graph_gdn_q14_layer_protocol import CONFIGS, REPEATS, WARMUP, payload, process_order, settings
from multi_graph_gdn_q14_protocol import load_manifest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(args):
    import torch
    import torch_npu

    import vllm_ascend.ops.gdn as gdn

    paths = [
        *Path(__file__).parent.glob("multi_graph*.py"),
        args.manifest,
        Path(gdn.__file__),
        Path(inspect.getfile(gdn.causal_conv1d_update)),
    ]
    for name in ("old", "new", "deferred"):
        library = getattr(args, name + "_extension")
        paths.extend(library.parent.rglob("*.so"))
    return dict(
        files={str(p): sha(p) for p in sorted(set(paths))},
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        hf32=torch.npu.matmul.allow_hf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        internal_format=getattr(torch.npu.config, "allow_internal_format", "installed_default_unmodified"),
    )


def timed(torch, operation):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    result = operation()
    torch.npu.synchronize()
    return result, (time.perf_counter_ns() - start) / 1e6


def exact(torch, actual, expected, label):
    if not torch.equal(actual, expected):
        raise AssertionError(f"{label}: max_abs={(actual.float() - expected.float()).abs().max().item()}")


def short_schedule():
    """Exercise pure Q1 after acceptance four, then pause/reorder/resume."""
    patterns = (
        (list(range(8)), [4] * 8),
        (list(reversed(range(8))), [1] * 8),
        ([7, 3, 0, 6, 4, 1], [4, 1, 1, 4, 1, 4]),
        ([2, 5, 0, 7, 1, 6, 3, 4], [1, 4, 4, 1, 1, 4, 4, 1]),
        (list(range(8)), [4] * 8),
    )
    accepted, result = dict.fromkeys(range(8), 1), []
    for step, (owners, widths) in enumerate(patterns):
        decisions = [
            {"accepted": 1 if q == 1 else 1 + (owner + step) % 4} for owner, q in zip(owners, widths, strict=True)
        ]
        result.append(
            dict(
                step=step,
                request_ids=owners,
                query_lengths=widths,
                accepted_from_previous=[accepted[o] for o in owners],
                decisions=decisions,
            )
        )
        accepted.update({o: d["accepted"] for o, d in zip(owners, decisions, strict=True)})
    return result


class Trial:
    def __init__(self, torch, args, fixture, model, config, meta, seed, mode, libraries):
        from multi_graph_gdn_q4_log_backend import Q4LogState
        from multi_graph_gdn_q14_layer_core import PackedLayerDescriptor

        self.torch, self.args, self.mode = torch, args, mode
        self.fixture, self.model, self.config = fixture, model, config
        self.layer = model.language_model.model.layers[fixture.layer_index]
        self.gdn = self.layer.linear_attn
        self.capacity, self.rows = args.capacity, 4 * args.capacity
        self.seed = seed
        self.meta = copy.deepcopy(meta)
        self.meta.num_actual_tokens = self.rows
        self.conv = torch.empty_like(fixture.states[0])
        self.state = Q4LogState.from_base(
            torch, seed["initial_ssm"].clone(), [list(range(1 + 4 * o, 5 + 4 * o)) for o in range(self.capacity)]
        )
        self.libraries = libraries
        self.descriptor = PackedLayerDescriptor(
            sequence=torch.empty(self.capacity + 1, dtype=torch.int32, device="npu"),
            ids=torch.empty(self.rows, dtype=torch.int32, device="npu"),
            accepted=torch.empty(self.capacity, dtype=torch.int32, device="npu"),
            pack_indices=torch.empty(self.rows, dtype=torch.int64, device="npu"),
            inverse_indices=torch.empty(self.rows, dtype=torch.int64, device="npu"),
            active=torch.empty(self.rows, dtype=torch.bool, device="npu"),
        )
        self.hidden, self.residual = (torch.empty_like(seed["records"][0][name]) for name in ("hidden", "residual"))
        self.positions = torch.empty(self.rows, dtype=torch.int64, device="npu")
        self.empty_logs, self.empty_valid = torch.zeros_like(self.state.logs), torch.zeros_like(self.state.valid)
        self.terminal_conv = torch.empty((self.capacity, 3, self.conv.shape[-1]), dtype=self.conv.dtype, device="npu")
        self.terminal_slots = torch.arange(1, 1 + self.rows, 4, dtype=torch.int64, device="npu")
        self.terminal_offsets = torch.empty((self.capacity, 3, 1), dtype=torch.int64, device="npu")
        self.graph, self.stream, self.adapter = None, torch.npu.Stream(), None
        self.restore()

    def restore(self):
        self.conv[:1].copy_(self.seed["initial_conv"][:1])
        self.conv[1:].view(self.capacity, 4, *self.conv.shape[1:]).copy_(self.seed["initial_conv"][1:].unsqueeze(1))
        self.state.base.copy_(self.seed["initial_ssm"])
        self.state.logs.zero_()
        self.state.valid.zero_()
        if self.adapter is not None:
            staging = self.adapter.backend._exclusive_buffers
            staging["new_logs"].zero_()
            staging["new_valid"].zero_()

    def prepare(self, row, record, filler=None):
        torch = self.torch
        data = payload(row, self.capacity, self.mode)
        causal = self.meta.spec_decode_metadata.spec_causal_conv1d
        fields = (
            (self.descriptor.sequence, data["sequence"]),
            (self.descriptor.ids, data["ids"]),
            (self.descriptor.accepted, data["accepted"]),
            (self.meta.spec_state_indices_tensor, data["convslots"]),
            (causal.query_start_loc, data["query_start_loc"]),
            (causal.num_accepted_tokens, data["accepted"]),
            (self.meta.spec_decode_metadata.actual_seq_lengths, data["sequence"]),
        )
        for target, value in fields:
            target.copy_(torch.tensor(value, dtype=target.dtype))
        # Q4 has no packing adapter. Its one-time descriptor reservation is
        # outside performance; do not charge unused per-step uploads to it.
        if self.mode != "forced_q4" or self.adapter is None:
            for target, value in (
                (self.descriptor.pack_indices, data["pack_indices"]),
                (self.descriptor.inverse_indices, data["inverse_indices"]),
                (self.descriptor.active, [i < data["executed_query_tokens"] for i in range(self.rows)]),
            ):
                target.copy_(torch.tensor(value, dtype=target.dtype))
        if causal.cache_indices.data_ptr() != self.meta.spec_state_indices_tensor.data_ptr():
            causal.cache_indices.copy_(self.meta.spec_state_indices_tensor)
        self.hidden.copy_(record["hidden"])
        self.residual.copy_(record["residual"])
        self.positions.copy_(record["positions"])
        if filler is not None:
            mask = torch.tensor(data["valid_dense_mask"], dtype=torch.bool, device="npu")
            self.hidden.masked_fill_(~mask[:, None], filler)
            self.residual.masked_fill_(~mask[:, None], filler * 0.5)
        return data

    def installed(self):
        from multi_graph_gdn_q14_layer_core import installed

        self.gdn.kv_cache = (self.conv, self.state.handle)
        if self.adapter is not None:
            return installed(adapter=self.adapter)
        return installed(
            self.torch,
            self.gdn,
            self.meta,
            self.state,
            self.descriptor,
            self.libraries["old"],
            self.libraries["new"],
            self.libraries["deferred"],
            mode=self.mode,
            endpoint=self.layer,
        )

    def operation(self):
        from vllm.config import CUDAGraphMode

        from vllm_ascend.ascend_forward_context import set_ascend_forward_context

        with set_ascend_forward_context(
            {self.gdn.prefix: self.meta},
            self.config,
            num_tokens=self.rows,
            num_actual_tokens=self.rows,
            aclgraph_runtime_mode=CUDAGraphMode.NONE,
            model_instance=self.model,
            skip_compiled=True,
        ):
            return self.layer(hidden_states=self.hidden, residual=self.residual, positions=self.positions)

    def capture(self, row, record):
        torch = self.torch
        self.prepare(row, record)
        with self.installed() as adapter:
            self.adapter = adapter
            adapter.backend.reserve_exclusive_capacity(self.capacity)
            for _ in range(2):
                self.restore()
                self.operation()
                torch.npu.synchronize()
            self.restore()
            self.stream.wait_stream(torch.npu.current_stream())
            self.graph = torch.npu.NPUGraph()
            with torch.npu.graph(self.graph, stream=self.stream, pool=torch.npu.graph_pool_handle()):
                self.output = self.operation()
            torch.npu.synchronize()
        self.restore()

    def replay(self):
        self.stream.wait_stream(self.torch.npu.current_stream())
        with self.torch.npu.stream(self.stream):
            self.graph.replay()
        self.torch.npu.current_stream().wait_stream(self.stream)

    def snapshot(self):
        staging = self.adapter.backend._exclusive_buffers
        return tuple(
            t.clone() for t in (self.conv, *self.state.mutable_tensors, staging["new_logs"], staging["new_valid"])
        )

    def terminal(self, rows):
        torch = self.torch
        accepted = dict.fromkeys(range(self.capacity), 1)
        for row in rows:
            accepted.update({o: d["accepted"] for o, d in zip(row["request_ids"], row["decisions"], strict=True)})
        terminal_row = dict(
            request_ids=list(range(self.capacity)),
            query_lengths=[1] * self.capacity,
            accepted_from_previous=[accepted[o] for o in range(self.capacity)],
        )
        data = payload(terminal_row, self.capacity, self.mode)
        for name in ("sequence", "ids", "accepted"):
            target = getattr(self.descriptor, name)
            target.copy_(torch.tensor(data[name], dtype=target.dtype))
        extension = self.libraries["deferred"] if self.mode == "forced_q4" else self.libraries["new"]
        extension.commit(
            self.state.base,
            self.descriptor.sequence,
            self.descriptor.ids,
            self.descriptor.accepted,
            self.state.logs,
            self.state.valid,
            self.empty_logs,
            self.empty_valid,
        )
        offsets = [[accepted[o] - 1 + t for t in range(3)] for o in range(self.capacity)]
        self.terminal_offsets.copy_(
            torch.tensor(offsets, dtype=torch.int64)[:, :, None].expand_as(self.terminal_offsets)
        )
        torch.gather(
            self.conv.index_select(0, self.terminal_slots),
            1,
            self.terminal_offsets.expand_as(self.terminal_conv),
            out=self.terminal_conv,
        )


def audit(torch, trials, rows, records, *, filler=None, graph=True):
    histories, checks = {}, []
    for mode, trial in trials.items():
        trial.restore()
        observations = []
        with trial.installed() as adapter:
            if not graph:
                trial.adapter = adapter
            for row, record in zip(rows, records, strict=True):
                before = trial.snapshot()
                data = trial.prepare(row, record, filler=filler)
                result = trial.output if graph else None
                if graph:
                    trial.replay()
                else:
                    result = trial.operation()
                torch.npu.synchronize()
                valid = torch.tensor(data["valid_dense_mask"], dtype=torch.bool, device="npu")
                outputs = tuple(value[valid].clone() for value in result)
                exported = torch.empty((trial.capacity + 1, 4, 24, 128, 128), dtype=torch.float32, device="npu")
                extension = trial.libraries["deferred"] if mode == "forced_q4" else trial.libraries["new"]
                extension.export_candidates(trial.state.base, trial.state.logs, trial.state.valid, exported)
                selected, conv = [], []
                for o, decision in zip(row["request_ids"], row["decisions"], strict=True):
                    a = decision["accepted"]
                    selected.append(exported[o + 1, a - 1].clone())
                    conv.append(trial.conv[1 + 4 * o, a - 1 : a + 2].clone())
                accepted_state, history = torch.stack(selected), torch.stack(conv)
                paused = [o for o in range(trial.capacity) if o not in row["request_ids"]]
                for o in [0, *[p + 1 for p in paused]]:
                    for current, prior in zip(trial.state.mutable_tensors, before[1:4], strict=True):
                        exact(torch, current[o], prior[o], f"{mode}/paused_compact/{o}")
                slots = [0, *[1 + 4 * p + t for p in paused for t in range(4)]]
                exact(torch, trial.conv[slots], before[0][slots], f"{mode}/paused_conv")
                # Untimed diagnostic downloads keep the full 16-step audit
                # from retaining multi-gigabyte state histories in HBM.
                observations.append(
                    dict(
                        outputs=tuple(v.cpu() for v in outputs),
                        accepted=accepted_state.cpu(),
                        conv=history.cpu(),
                        snapshot=tuple(v.cpu() for v in trial.snapshot()),
                    )
                )
                checks.append(dict(mode=mode, step=row["step"], finite=True, paused_exact=True))
                for value in (*outputs, accepted_state, history):
                    if not torch.isfinite(value).all().item():
                        raise AssertionError("nonfinite valid layer data")
        trial.terminal(rows)
        torch.npu.synchronize()
        if torch.count_nonzero(trial.state.logs).item() or torch.count_nonzero(trial.state.valid).item():
            raise AssertionError(f"{mode}/terminal did not clear pending logs")
        histories[mode] = dict(steps=observations, terminal=(trial.state.base.cpu(), trial.terminal_conv.cpu()))
    # Only legal prefixes and logically selected conv histories agree across
    # widths; forced Q4 may legitimately retain other candidate cache tails.
    reference = histories["forced_q4"]
    for mode, history in histories.items():
        for row, actual, expected in zip(rows, history["steps"], reference["steps"], strict=True):
            for value in (*actual["outputs"], actual["accepted"], actual["conv"]):
                if not torch.isfinite(value).all().item():
                    raise AssertionError(f"{mode}/nonfinite/step{row['step']}")
            for a, b in zip(actual["outputs"], expected["outputs"], strict=True):
                exact(torch, a, b, f"{mode}/forced_valid_output/step{row['step']}")
            for field in ("accepted", "conv"):
                exact(torch, actual[field], expected[field], f"{mode}/forced_{field}/step{row['step']}")
            for owner, width in zip(row["request_ids"], row["query_lengths"], strict=True):
                logs, valid = actual["snapshot"][2:4]
                # The frozen Q4 ABI uses a boolean validity flag. The separate
                # Q1/Q4 ABI stores its published width in the same record.
                expected_width = 1 if mode == "forced_q4" else width
                exact(
                    torch,
                    valid[owner + 1, :, 0],
                    torch.full_like(valid[owner + 1, :, 0], expected_width),
                    "published_log_width",
                )
                for start, stop in ((0, width * 128), (512, 512 + width * 128), (1024, 1024 + width * 8)):
                    exact(
                        torch,
                        logs[owner + 1, :, start:stop],
                        expected["snapshot"][2][owner + 1, :, start:stop],
                        "effective_log_prefix",
                    )
        for a, b in zip(history["terminal"], reference["terminal"], strict=True):
            exact(torch, a, b, f"{mode}/forced_terminal")
    ready = histories["q14_ready"]
    delayed = histories["q14_delayed"]
    for row, actual, expected in zip(rows, ready["steps"], delayed["steps"], strict=True):
        for a, b in zip(actual["outputs"], expected["outputs"], strict=True):
            exact(torch, a, b, f"ready_delayed/output/step{row['step']}")
        for field in ("accepted", "conv"):
            exact(torch, actual[field], expected[field], f"ready_delayed/{field}/step{row['step']}")
        for a, b in zip(actual["snapshot"], expected["snapshot"], strict=True):
            exact(torch, a, b, f"ready_delayed/state/step{row['step']}")
    for a, b in zip(ready["terminal"], delayed["terminal"], strict=True):
        exact(torch, a, b, "ready_delayed/terminal")
    return checks, histories


def performance(torch, trials, rows, records, process, barrier):
    samples, segments = [], []
    for mode in process_order(process):
        trial = trials[mode]
        for sample in range(-WARMUP, REPEATS):
            barrier()
            _, initialization = timed(torch, trial.restore)

            def chain(trial=trial):
                for row, record in zip(rows, records, strict=True):
                    trial.prepare(row, record)
                    trial.replay()

            _, execution = timed(torch, chain)
            _, terminal = timed(torch, lambda trial=trial: trial.terminal(rows))
            if sample >= 0:
                samples.append(
                    dict(
                        process=process,
                        config=mode,
                        sample=sample,
                        initialization_ms=initialization,
                        execution_ms=execution,
                        terminal_ms=terminal,
                        lifecycle_ms=initialization + execution + terminal,
                    )
                )
        # Separate diagnostic run: event instrumentation is absent above.
        trial.restore()
        events = [tuple(torch.npu.Event(enable_timing=True) for _ in range(3)) for _ in rows]
        for (row, record), event in zip(zip(rows, records, strict=True), events, strict=True):
            event[0].record()
            data = trial.prepare(row, record)
            event[1].record()
            trial.replay()
            event[2].record()
            segments.append(dict(process=process, config=mode, step=row["step"], segment=data["segment"], events=event))
        torch.npu.synchronize()
    for segment in segments:
        first, middle, last = segment.pop("events")
        segment.update(
            prepare_ms=first.elapsed_time(middle), layer_ms=middle.elapsed_time(last), total_ms=first.elapsed_time(last)
        )
    return samples, segments


def experiment(args, workload, runner, model, inputs, context, reset_kv, *, module=None):
    import torch
    import torch.distributed as dist
    from multi_graph_gdn_q14_layer_inputs import build_layer_inputs
    from multi_graph_tp_gdn import qualify_gdn
    from vllm.distributed.parallel_state import get_world_group
    from vllm.forward_context import get_forward_context

    rank, group = dist.get_rank(), get_world_group().cpu_group
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    manifest = load_manifest(args.manifest)["payload"]
    rows = manifest["steps"]
    args.capacity = manifest["capacity_requests"]
    rank_seed = args.artifacts / f"inputs.rank{rank}.pt"
    args.artifacts.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(args)
    if args.phase == "performance":
        qualification = json.loads(args.qualification.with_suffix(f".rank{rank}.json").read_text())
        if qualification["status"] != "passed_complete_layer_correctness":
            raise ValueError("device qualification required before performance")
        if qualification["input_file_sha256"] != sha(rank_seed):
            raise ValueError("qualified inputs changed")
        if qualification["fingerprint"] != identity:
            raise ValueError("qualified dependency or precision configuration changed")
        current_sources = {p.name: sha(p) for p in Path(__file__).parent.glob("multi_graph_gdn_q14_layer*.py")}
        current_libraries = {name: sha(getattr(args, name + "_extension")) for name in ("old", "new", "deferred")}
        if qualification["source_sha256"] != current_sources or qualification["libraries"] != current_libraries:
            raise ValueError("qualified implementation changed")
    if args.phase == "correctness" and not rank_seed.exists():
        seed, generation_ms = timed(torch, lambda: build_layer_inputs(torch, model, runner.vllm_config, manifest))
        short_manifest = dict(manifest, steps=short_schedule())
        short_seed = build_layer_inputs(torch, model, runner.vllm_config, short_manifest)
        exact(torch, short_seed["initial_conv"], seed["initial_conv"], "short/main_initial_conv")
        exact(torch, short_seed["initial_ssm"], seed["initial_ssm"], "short/main_initial_ssm")
        seed["short_records"] = short_seed["records"]
        seed["short_provenance"] = short_seed["provenance"]
        host = dict(
            initial_conv=seed["initial_conv"].cpu(),
            initial_ssm=seed["initial_ssm"].cpu(),
            records=[
                {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in record.items()}
                for record in seed["records"]
            ],
            provenance=seed["provenance"],
            short_records=[
                {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in record.items()}
                for record in seed["short_records"]
            ],
            short_provenance=seed["short_provenance"],
        )
        torch.save(host, rank_seed)
    else:
        seed = torch.load(rank_seed, map_location="cpu", weights_only=False)
        for name in ("initial_conv", "initial_ssm"):
            seed[name] = seed[name].to("npu")
        for record in seed["records"]:
            for name in ("hidden", "residual", "positions"):
                record[name] = record[name].to("npu")
        if args.phase == "correctness":
            for record in seed["short_records"]:
                for name in ("hidden", "residual", "positions"):
                    record[name] = record[name].to("npu")
        generation_ms = None
    fixture = qualify_gdn(
        args,
        workload,
        runner,
        model,
        (seed["records"][0]["hidden"], seed["records"][0]["residual"]),
        group,
        setup_only=True,
    )
    with fixture.phases.context_for_name("full"):
        meta = get_forward_context().attn_metadata[fixture.phases.gdn.prefix]
    libraries = dict(
        old=load_extension("gdn_q4_tile_pipeline", args.old_extension),
        new=load_extension("gdn_q14", args.new_extension),
        deferred=load_extension("gdn_q4_log_deferred", args.deferred_extension),
    )
    modes = (*CONFIGS, "q14_delayed") if args.phase == "correctness" else CONFIGS
    trials = {
        mode: Trial(torch, args, fixture, model, runner.vllm_config, meta, seed, mode, libraries) for mode in modes
    }
    capture_start = time.perf_counter_ns()
    for trial in trials.values():
        trial.capture(rows[0], seed["records"][0])
    capture_ms = (time.perf_counter_ns() - capture_start) / 1e6
    if args.phase == "correctness":
        short = short_schedule()
        short_records = seed["short_records"]
        checks, first = audit(torch, trials, short, short_records, filler=0.125)
        more, second = audit(torch, trials, short, short_records, filler=-0.375)
        for mode in modes:
            for a, b in zip(first[mode]["steps"], second[mode]["steps"], strict=True):
                for x, y in zip(a["outputs"], b["outputs"], strict=True):
                    exact(torch, x, y, "filler/output")
                for field in ("accepted", "conv"):
                    exact(torch, a[field], b[field], f"filler/{field}")
        eager_checks, eager = audit(torch, trials, short, short_records, filler=0.125, graph=False)
        for mode, trial in trials.items():
            for actual, expected in zip(eager[mode]["steps"], first[mode]["steps"], strict=True):
                for a, b in zip(actual["outputs"], expected["outputs"], strict=True):
                    exact(torch, a, b, f"{mode}/eager_graph/output")
                for a, b in zip(actual["snapshot"], expected["snapshot"], strict=True):
                    exact(torch, a, b, f"{mode}/eager_graph/state")
        del first, second, eager
        more2, _ = audit(torch, trials, rows, seed["records"])
        report = dict(
            status="passed_complete_layer_correctness",
            checks=checks + more + eager_checks + more2,
            filler_invariant=True,
            continuous_graph_steps=16,
            synthetic_acceptance_pause=True,
        )
    else:
        samples, segments = performance(
            torch, trials, rows, seed["records"], args.process, lambda: dist.barrier(group=group)
        )
        report = dict(status="completed_performance_process", samples=samples, segments=segments)
    report.update(
        rank=rank,
        fingerprint=identity,
        settings=settings(),
        provenance=seed["provenance"],
        short_provenance=seed.get("short_provenance"),
        input_file_sha256=sha(rank_seed),
        input_generation_ms=generation_ms,
        capture_ms=capture_ms,
        loader=args.loader_metadata,
        source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("multi_graph_gdn_q14_layer*.py")},
        libraries={name: sha(getattr(args, name + "_extension")) for name in ("old", "new", "deferred")},
    )
    args.output.with_suffix(f".rank{rank}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    dist.barrier(group=group)
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(dict(status=report["status"], output=str(args.output))), flush=True)


def main():
    from multi_graph_tp_boundary_loader import run_qwen35

    from vllm_ascend.worker.multi_graph.replay_benchmark import ReplayWorkload

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("correctness", "performance"), required=True)
    parser.add_argument("--process", type=int, choices=(0, 1), default=0)
    parser.add_argument("--model", default="/models/Qwen3.5-27B-original")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    for name in ("old", "new", "deferred"):
        parser.add_argument(f"--{name}-extension", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous evidence")
    if args.phase == "performance" and args.qualification is None:
        parser.error("performance requires current device --qualification")
    args.tensor_parallel_size, args.query, args.dtype = 2, 4, "bfloat16"
    args.batch_invariant = args.compiled = args.gdn_pipeline = False
    args.gdn_probe = True
    args.pair = (112, 112)
    args.capacity = 56
    args.selection = dict(
        batch=56,
        capacity_requests=56,
        capacity_tokens=224,
        policy="inherited frozen ledger default graph capacity; no additional buckets",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_qwen35(args, ReplayWorkload(56, 56, 4, 512, (28, 28)), experiment=experiment)
    except Exception as error:
        target = args.output.with_suffix(f".rank{os.environ.get('RANK', '0')}.failure.json")
        target.write_text(
            json.dumps(dict(status="failed", error_type=type(error).__name__, error=str(error)), indent=2)
        )
        raise


if __name__ == "__main__":
    main()
