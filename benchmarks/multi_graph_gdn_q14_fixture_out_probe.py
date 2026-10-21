# SPDX-License-Identifier: Apache-2.0
"""Bounded device safety and symmetric four-configuration fixture experiment."""

import argparse
import json
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

from multi_graph_gdn_q4_backend import load_extension
from multi_graph_gdn_q14_fixture_out import CONFIGS, REPEATS, ROUNDS, OutTrial, round_order, summarize
from multi_graph_gdn_q14_probe import OPERANDS, Fixture, Trial, require_equal, sha, synthetic_steps, timed
from multi_graph_gdn_q14_protocol import load_manifest

AUDIT_MODES = ("forced_q4", "q14_ready", "q14_delayed")
STATE_FIELDS = ("output", "base", "logs", "valid", "new_logs", "new_valid")


def descriptor(tensor):
    return dict(
        pointer=tensor.data_ptr(),
        storage_pointer=tensor.untyped_storage().data_ptr(),
        storage_bytes=tensor.untyped_storage().nbytes(),
        shape=list(tensor.shape),
        stride=list(tensor.stride()),
        offset=tensor.storage_offset(),
        dtype=str(tensor.dtype),
    )


@contextmanager
def inspect_out(torch, records):
    original = torch.index_select

    def checked(source, dim, indices, *, out):
        expected = list(source.shape)
        expected[dim] = indices.numel()
        assert list(out.shape) == expected, "out shape must match before dispatch"
        assert out.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
        before = descriptor(out)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = original(source, dim, indices, out=out)
        assert not any("resiz" in str(w.message).lower() for w in caught), "unexpected out resize warning"
        assert descriptor(out) == before, "out view resized or moved"
        assert descriptor(result) == before, "returned out differs from fixed prefix"
        records.append(dict(before=before, after=descriptor(out), warnings=[str(w.message) for w in caught]))
        return result

    torch.index_select = checked
    try:
        yield
    finally:
        torch.index_select = original


def require_bits(first, second, label):
    import torch

    def raw(tensor):
        # Downloads are confined to the untimed device safety check.
        host = tensor.detach().cpu().contiguous()
        return host.view(torch.uint8).numpy().tobytes()

    if first.dtype != second.dtype or first.shape != second.shape or raw(first) != raw(second):
        raise AssertionError(f"{label}: input bytes changed")


def preparation_audit(torch, fixture, rows, capacity, extensions):
    records, calls = [], []
    for mode in AUDIT_MODES:
        legacy = Trial(torch, fixture, mode, capacity, *extensions, None, None)
        candidate = OutTrial(torch, fixture, mode, capacity, *extensions, None, None)
        identities = {
            p: {n: descriptor(b) for n, b in t.buffers.items()} for p, t in (("legacy", legacy), ("out", candidate))
        }
        for row in rows:
            count = 4 * len(row["request_ids"]) if mode == "forced_q4" else sum(row["query_lengths"])
            tails = {}
            for preparation, trial in (("legacy", legacy), ("out", candidate)):
                tails[preparation] = {}
                for index, (name, buffer) in enumerate(trial.buffers.items()):
                    buffer.fill_(index + 1 + row["step"])
                    tails[preparation][name] = buffer[count:].clone()
            legacy_meta = legacy.prepare(row)
            with inspect_out(torch, calls):
                candidate_meta = candidate.prepare(row)
            torch.npu.synchronize()
            assert legacy_meta == candidate_meta
            require_equal(torch, legacy.meta, candidate.meta, "metadata")
            for name in OPERANDS:
                require_bits(legacy.buffers[name][:count], candidate.buffers[name][:count], name)
                for preparation, trial in (("legacy", legacy), ("out", candidate)):
                    assert descriptor(trial.buffers[name]) == identities[preparation][name]
                    require_bits(trial.buffers[name][count:], tails[preparation][name], f"{preparation}/{name}/tail")
            records.append(dict(mode=mode, step=row["step"], count=count, exact_inputs=True, exact_tails=True))
    return dict(steps=records, out_calls=calls, resize=False, backing_unchanged=True)


def state_audit(torch, fixture, rows, capacity, extensions, graph):
    trials = {
        (preparation, mode): cls(torch, fixture, mode, capacity, *extensions, None, None)
        for preparation, cls in (("legacy", Trial), ("out", OutTrial))
        for mode in AUDIT_MODES
    }
    if graph:
        for trial in trials.values():
            trial.capture(rows)
    records = []
    for row in rows:
        inactive = sorted(set(range(fixture.owners)) - set(row["request_ids"]))
        paused = {
            key: [(t.base[o + 1].clone(), t.logs[o + 1].clone(), t.valid[o + 1].clone()) for o in inactive]
            for key, t in trials.items()
        }
        for trial in trials.values():
            trial.prepare(row)
            trial.graphs[trial.key].replay() if graph else trial.operation()
        torch.npu.synchronize()
        for mode in AUDIT_MODES:
            for name in STATE_FIELDS:
                require_equal(
                    torch, getattr(trials["legacy", mode], name), getattr(trials["out", mode], name), f"{mode}/{name}"
                )
        exports = {key: trial.export() for key, trial in trials.items()}
        for mode in AUDIT_MODES:
            require_equal(torch, exports["legacy", mode], exports["out", mode], "legacy/out candidates")
        for preparation in ("legacy", "out"):
            ready, delayed, forced = (trials[preparation, m] for m in ("q14_ready", "q14_delayed", "forced_q4"))
            for name in STATE_FIELDS:
                require_equal(torch, getattr(ready, name), getattr(delayed, name), f"{preparation}/{name}/schedule")
            require_equal(torch, ready.base, forced.base, "accepted base")
            e_ready, e_delayed, e_forced = (exports[preparation, m] for m in ("q14_ready", "q14_delayed", "forced_q4"))
            require_equal(torch, e_ready, e_delayed, "schedule candidates")
            begin = 0
            for j, (owner, q) in enumerate(zip(row["request_ids"], row["query_lengths"], strict=True)):
                require_equal(torch, ready.output[begin : begin + q], forced.output[4 * j : 4 * j + q], "valid output")
                require_equal(
                    torch, ready.valid[owner + 1], torch.full_like(ready.valid[owner + 1], q), "published width"
                )
                for offset, count in ((0, q * 128), (512, q * 128), (1024, q * 8)):
                    require_equal(
                        torch,
                        ready.logs[owner + 1, :, offset : offset + count],
                        forced.logs[owner + 1, :, offset : offset + count],
                        "valid log",
                    )
                for start, stop in ((q * 128, 512), (512 + q * 128, 1024), (1024 + q * 8, 1056)):
                    assert not torch.count_nonzero(ready.logs[owner + 1, :, start:stop]).item(), "invalid log tail"
                require_equal(torch, e_ready[owner + 1, :q], e_forced[owner + 1, :q], "valid candidate state")
                begin += q
        for key, snapshots in paused.items():
            trial = trials[key]
            for owner, snapshot in zip(inactive, snapshots, strict=True):
                for saved, current in zip(
                    snapshot, (trial.base[owner + 1], trial.logs[owner + 1], trial.valid[owner + 1]), strict=True
                ):
                    require_equal(torch, saved, current, "paused owner")
        records.append(
            dict(
                step=row["step"],
                graph=graph,
                exact_preparation=True,
                exact_schedule=True,
                exact_valid_prefix=True,
                paused_owners=inactive,
            )
        )
        del exports
    terminals = {key: t.terminal() for key, t in trials.items()}
    reference = terminals["legacy", "forced_q4"]
    for key, terminal in terminals.items():
        require_equal(torch, reference, terminal, str(key) + "/terminal")
        assert not torch.count_nonzero(trials[key].logs).item()
        assert not torch.count_nonzero(trials[key].valid).item()
    return dict(steps=records, terminal_bitwise=True, terminal_debt_clear=True)


def measure(torch, trials, rows):
    samples = []
    for round_id in range(ROUNDS):
        for config in round_order(round_id):
            trial = trials[config]
            for sample in range(REPEATS):
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
                by_mode = dict.fromkeys(("q1_only", "q4_only", "mixed"), 0.0)
                compute_by_mode, prepare_by_mode = by_mode.copy(), by_mode.copy()
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
                        config=config,
                        execution_ms=execute_ms,
                        initialization_ms=initialization_ms,
                        terminal_ms=terminal_ms,
                        lifecycle_ms=initialization_ms + execute_ms + terminal_ms,
                        device_step_endpoint_ms=by_mode,
                        device_compute_ms=compute_by_mode,
                        device_preparation_ms=prepare_by_mode,
                    )
                )
    return dict(samples=samples, **summarize(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("new-extension", "old-extension", "deferred-extension", "site", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=("correctness", "performance"), required=True)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence; never overwrite")
    import torch
    import torch_npu

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    source = Path(__file__).parent
    paths = [
        Path(__file__),
        *(
            source / name
            for name in (
                "multi_graph_gdn_q14_fixture_out.py",
                "multi_graph_gdn_q14_probe.py",
                "multi_graph_gdn_q14_protocol.py",
                "multi_graph_gdn_q4_backend.py",
            )
        ),
        args.new_extension,
        args.old_extension,
        args.deferred_extension,
        args.site,
        args.manifest,
    ]
    paths += sorted((args.new_extension.parent / "lib").glob("*.so"))
    sources = {str(p): sha(p) for p in paths}
    report = dict(
        status="running",
        phase=args.phase,
        sources=sources,
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        configs=CONFIGS,
        repeats=REPEATS,
        rounds=ROUNDS,
        orders=[round_order(r) for r in range(ROUNDS)],
        input_mode="real_site_reuse",
        initial_state_matches_original_ledger_step=False,
        boundary="saved normalized recurrence inputs through terminal state; not full GDN layer or service",
        precision="BF16 operands/output, FP32 compact state/log/products; HF32 off; highest",
        profiler=False,
        diagnostic_instrumentation_in_performance=False,
        quality_failures_preserved=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        if args.phase == "performance":
            if args.qualification is None:
                raise ValueError("passed device qualification required")
            qualification = json.loads(args.qualification.read_text())
            if qualification["status"] != "passed_device_correctness" or qualification["sources"] != sources:
                raise ValueError("qualification or frozen source identity mismatch")
            report["qualification_sha256"] = sha(args.qualification)
        new = load_extension("gdn_q14", args.new_extension)
        old = load_extension("gdn_q4_tile_pipeline", args.old_extension)
        deferred = load_extension("gdn_q4_log_deferred", args.deferred_extension)
        extensions = (new, old, deferred)
        manifest = load_manifest(args.manifest)["payload"]
        rows = manifest["steps"]
        capacity = max(s["capacity_tokens"] for s in rows) // 4
        with torch.inference_mode():
            host = torch.load(args.site, map_location="cpu", weights_only=True)
            if args.phase == "correctness":
                small = Fixture(torch, host, 8, deferred)
                small_rows = synthetic_steps()
                report["preparation_audit"] = preparation_audit(torch, small, small_rows, 8, extensions)
                save()
                for graph in (False, True):
                    report["small_graph" if graph else "small_eager"] = state_audit(
                        torch, small, small_rows, 8, extensions, graph
                    )
                    save()
                del small
            fixture = Fixture(torch, host, len(manifest["stable_owners"]), deferred)
            if manifest["stable_owners"] != list(range(fixture.owners)):
                raise ValueError("do not silently remap frozen owners")
            report.update(
                capacity_requests=capacity, source_owners=fixture.source_owners, manifest_summary=manifest["summary"]
            )
            if args.phase == "correctness":
                report["segment_preparation_audit"] = preparation_audit(torch, fixture, rows, capacity, extensions)
                save()
                report["segment_graph"] = state_audit(torch, fixture, rows, capacity, extensions, True)
                report["status"] = "passed_device_correctness"
            else:
                trials = {}
                for config in CONFIGS:
                    preparation, mode = config.split("_", 1)
                    cls = OutTrial if preparation == "out" else Trial
                    trial = cls(torch, fixture, mode, capacity, *extensions, None, None)
                    trial.capture(rows)
                    trials[config] = trial
                report["performance"] = measure(torch, trials, rows)
                report["status"] = "complete_single_site_performance"
        save()
        print(json.dumps(dict(status=report["status"], output=str(args.output))), flush=True)
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
