# SPDX-License-Identifier: Apache-2.0
"""Audit complete Q4 evidence; keep scheduling and algorithm precision separate."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

from multi_graph_gdn_q4_full_metadata import FIXED_TRACE_SHA256, load_fixed_trace
from multi_graph_gdn_q4_full_probe import latency_statistics
from multi_graph_gdn_ready_full_lifecycle import MODES, paired_lifecycles

STEPS = 128
REPLAY_STATUS = "qualified_full_dynamic_replay_and_closed_ready_ablation"
NUMERIC_STATUS = "complete_independent_official_full_numeric_audit"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def critical_decomposition(ranks, trial, mode):
    """Columns must come from the rank determining the closed endpoint."""
    selected = [
        (rank["rank"], sample)
        for rank in ranks
        for sample in rank["lifecycle_samples"]
        if sample["trial"] == trial and sample["mode"] == mode
    ]
    require(len(selected) == 2 and {r for r, _ in selected} == {0, 1}, "TP2 endpoint missing")
    rank, sample = max(selected, key=lambda pair: pair[1]["closed_wall_us"])
    result = dict(
        critical_rank=rank,
        execution_us=sample["wall_us"] - sample["terminal_wall_us"],
        import_us=sample["transition_us"],
        terminal_us=sample["terminal_wall_us"],
        total_us=sample["closed_wall_us"],
    )
    require(
        all(math.isfinite(result[key]) and result[key] > 0 for key in result if key.endswith("_us"))
        and math.isclose(result["total_us"], sum(result[key] for key in ("execution_us", "import_us", "terminal_us"))),
        "nonadditive lifecycle decomposition",
    )
    return result


def common_contract(documents, status):
    trace = load_fixed_trace(Path(__file__).parent / "results/multi_graph_structural/structural-dynamic-trace-v1.json")
    require(len(documents) == 2 and [d["rank"] for d in documents] == [0, 1], "TP2 reports required")
    for document in documents:
        require(document["status"] == status, "incomplete result: " + document["status"])
        require(not document.get("error"), "result contains an execution failure")
        require(document["scope"] == "all_64_decoders_and_final_norm", "model scope changed")
        require(document["trace_sha256"] == FIXED_TRACE_SHA256, "original Q4 trace changed")
        require(document["prefix"]["completed_requests"] == STEPS, "incomplete coherent initialization")
        require(document["loader"]["tensor_parallel_size"] == 2, "TP changed")
        require(document["loader"]["ssm_dtype"] == "float32", "state precision changed")
        selection = document["selection"]
        capacities = selection["capture_sizes"]
        require(capacities == [1, 2, 4, *range(8, 257, 8), *range(272, 513, 16)], "frozen default catalog changed")
        require(
            selection["config_source"] == "installed EngineArgs defaults; no capture sizes supplied", "custom catalog"
        )
        require(len(selection["dispatched_tokens"]) == STEPS, "incomplete dispatch")
        require(
            all(
                actual == min(size for size in capacities if size >= required)
                for actual, required in zip(
                    selection["dispatched_tokens"], (step["batch"] * 4 for step in trace.steps), strict=True
                )
            ),
            "dispatch no longer uses the default ceiling bucket",
        )
        require(sorted(set(selection["dispatched_tokens"])) == list(range(256, 513, 16)), "17 buckets changed")
        require(selection["required_tokens"] == list(range(256, 513, 16)), "required default catalog changed")
        require(len(document["cache_layout"]["gdn"]) == 48, "GDN layers missing")
        require(len(document["cache_layout"]["fia"]) == 16, "FIA layers missing")
    for key in ("selection", "loader", "cache_layout"):
        require(documents[0][key] == documents[1][key], "rank contract differs: " + key)
    require(documents[0]["prefix"]["token_sha256"] == documents[1]["prefix"]["token_sha256"], "prefix tokens differ")
    return trace


def replay_audit(documents, *, inherited=None):
    trace = common_contract(
        documents, "closed_ready_timing_with_prior_full_audit" if inherited is not None else REPLAY_STATUS
    )
    expected = {(mode, step) for mode in MODES for step in range(STEPS)}
    trials = len(documents[0]["paired_closed"])
    require(trials >= 3, "at least three closed pairs required")
    ranks = documents[0]["lifecycle_ranks"]
    require(ranks == documents[1]["lifecycle_ranks"], "gathered rank evidence differs")
    paired = paired_lifecycles(ranks, trials)
    require(all(d["paired_closed"] == paired for d in documents), "paired accounting differs")
    for document, rank in zip(documents, sorted(ranks, key=lambda r: r["rank"]), strict=True):
        checks = [c for c in document["checks"] if c["label"] == "complete_same_schedule_eager_replay_exact"]
        if inherited is None:
            require(
                len(checks) == len(expected) and {(c["mode"], c["step"]) for c in checks} == expected,
                "audit missing step",
            )
        else:
            proof = document["reused_qualification"]
            require(
                not checks and proof["prior_per_step_checks"] == 256 and proof["no_new_per_step_state_scan_claim"],
                "reused audit misrepresented as new full scanning",
            )
            require(document["terminal_decision_upload_timed"], "terminal host decision upload missing")
            require(
                document["reference_trajectory"] == dict(mode=MODES[0], steps=128, all_owners_closed=True),
                "no fresh continuous eager endpoint",
            )
        for check in checks:
            a, b = check["eager_audit"], check["replay_audit"]
            require(
                a["copied_to_cpu_bytes"] == a["restored_to_device_bytes"] == b["copied_to_cpu_bytes"] > 0
                and b["completed_steps"] == check["step"] + 1,
                "incomplete continued state scan",
            )
        if inherited is None:
            require(
                len(document["closed_checks"]) == len(MODES)
                and {c["mode"] for c in document["closed_checks"]} == set(MODES)
                and all(c["steps"] == STEPS and c["all_state_exact"] for c in document["closed_checks"]),
                "missing final acceptance or cross-mode exact check",
            )
        require(
            document["capture"]["graphs"] == 34 and document["capture"]["fia_tasks_per_graph"] == 16, "capture changed"
        )
        require(document["samples"] == rank["samples"], "local and gathered samples differ")
        rows = rank["samples"]
        wanted = {(trial, mode, step) for trial in range(trials) for mode in MODES for step in range(STEPS)}
        require(
            len(rows) == len(wanted) and {(s["trial"], s["mode"], s["step"]) for s in rows} == wanted,
            "timing missing step",
        )
        for row in rows:
            require(row["capacity"] == document["selection"]["dispatched_tokens"][row["step"]], "timed bucket changed")
            require(row["batch"] == trace.steps[row["step"]]["batch"], "timed active batch changed")
            require(all(math.isfinite(row[k]) and row[k] > 0 for k in ("elapsed_us", "device_us")), "invalid step time")
    endpoints = [
        dict(
            trial=trial,
            costs={mode: critical_decomposition(ranks, trial, mode) for mode in MODES},
            ready_percent=paired[trial]["ready_percent"],
        )
        for trial in range(trials)
    ]
    means = {
        mode: {
            key: statistics.mean(row["costs"][mode][key] for row in endpoints)
            for key in ("execution_us", "import_us", "terminal_us", "total_us")
        }
        for mode in MODES
    }
    step_latency = {}
    for mode in MODES:
        by_rank = [
            {(s["trial"], s["step"]): s["elapsed_us"] for s in rank["samples"] if s["mode"] == mode} for rank in ranks
        ]
        step_latency[mode] = latency_statistics([max(left, by_rank[1][key]) for key, left in by_rank[0].items()])
    return dict(
        same_algorithm_schedule_exact=True,
        per_step_audit_reused=inherited is not None,
        audited_steps_per_rank_per_mode=STEPS,
        qualification_records=sum(len(expected) for _ in documents),
        timed_closed_checks=len(MODES) * trials * len(ranks),
        trial_count=trials,
        paired=endpoints,
        mean_costs_us=means,
        mean_paired_percent=statistics.mean(row["ready_percent"] for row in endpoints),
        ratio_of_mean_totals_percent=100 * (1 - means[MODES[1]]["total_us"] / means[MODES[0]]["total_us"]),
        positive_pairs=sum(row["ready_percent"] > 0 for row in endpoints),
        step_latency_us=step_latency,
        setup_observations_us={d["rank"]: d["compact_transition_us"] for d in documents},
        accounting="additive columns from each mode's closed critical rank; step maxima not summed into lifecycle",
        step_statistics="TP max elapsed_us recomputed from both ranks; raw critical_rank_us is local in this phase",
    )


def numeric_audit(documents):
    common_contract(documents, NUMERIC_STATUS)
    token_hashes = {d["numeric_token_sha256"] for d in documents}
    require(len(token_hashes) == 1, "numeric token streams differ")
    rows = []
    for document in documents:
        require(document["numeric_completed_steps"] == STEPS, "official continued trajectory incomplete")
        require(document["numeric_contract"]["no_performance_comparison"], "offload must not be timed as baseline")
        gdn = [item["index"] for item in document["cache_layout"]["gdn"]]
        fia = [item["index"] for item in document["cache_layout"]["fia"]]
        expected = {
            (step, label, prefix)
            for step in range(STEPS)
            for label, prefix in [
                ("hidden", None),
                ("residual", None),
                *((f"gdn{layer}/candidate", prefix) for layer in gdn for prefix in range(1, 5)),
                *((f"gdn{layer}/conv", None) for layer in gdn),
                *((f"fia{layer}/{cache}/page", None) for layer in fia for cache in ("key", "value")),
            ]
        } | {(STEPS, f"gdn{layer}/{kind}", None) for layer in gdn for kind in ("terminal", "conv")}
        errors = document["numeric_errors"]
        require(
            len(errors) == len(expected) and {(r["step"], r["label"], r["prefix"]) for r in errors} == expected,
            "independent numeric observations missing",
        )
        threshold = document["numeric_contract"]["screen_nrmse"]
        require(threshold == 0.005, "original numerical screen changed")
        for row in errors:
            require(
                all(
                    math.isfinite(row[key]) and row[key] >= 0
                    for key in ("worst_request_nrmse", "pooled_nrmse", "max_abs")
                ),
                "invalid numeric metric",
            )
            require(row["screen_passed"] == (row["worst_request_nrmse"] <= threshold), "failure hidden by flag")
            rows.append(dict(row, rank=document["rank"]))
        require(document["numeric_screen_passed"] == all(r["screen_passed"] for r in errors), "numeric summary differs")
        require(
            document["numeric_output_screen_passed"]
            == all(r["screen_passed"] for r in errors if r["label"] in ("hidden", "residual")),
            "output failure hidden",
        )
    categories = {
        "hidden": lambda r: r["label"] == "hidden",
        "residual": lambda r: r["label"] == "residual",
        "logical_candidate": lambda r: r["label"].endswith("/candidate"),
        "accepted_terminal": lambda r: r["label"].endswith("/terminal"),
        "conv": lambda r: r["label"].endswith("/conv"),
        "fia_kv_pages": lambda r: r["label"].endswith("/page"),
    }
    summary = {}
    for name, belongs in categories.items():
        group = [row for row in rows if belongs(row)]
        worst = max(group, key=lambda r: r["worst_request_nrmse"])
        failed = [r for r in group if not r["screen_passed"]]
        summary[name] = dict(
            observations=len(group),
            worst=worst,
            median_worst_request_nrmse=statistics.median(r["worst_request_nrmse"] for r in group),
            failed_observations=len(failed),
            first_failed_step=min((r["step"] for r in failed), default=None),
        )
    return dict(
        complete_independent_official_trajectory=True,
        screen_nrmse=0.005,
        numerical_screen_passed=all(d["numeric_screen_passed"] for d in documents),
        output_screen_passed=all(d["numeric_output_screen_passed"] for d in documents),
        observations=len(rows),
        categories=summary,
        output_by_step=[
            dict(
                step=step,
                **{
                    label: max(r["worst_request_nrmse"] for r in rows if r["step"] == step and r["label"] == label)
                    for label in ("hidden", "residual")
                },
            )
            for step in range(STEPS)
        ],
        token_sha256=next(iter(token_hashes)),
        scope=(
            "FP32 compact vs installed official dense recurrence; "
            "independent conv/FIA/SSM histories, not generation quality"
        ),
        official_timing="CPU state banks are a correctness harness; no official performance baseline",
    )


def analyze(directory):
    for name, spec in read(directory / "evidence-manifest.json").items():
        data = (directory / name).read_bytes()
        require(len(data) == spec["bytes"] and hashlib.sha256(data).hexdigest() == spec["sha256"], "checksum: " + name)
    source_manifest = read(directory / "measured-source-manifest.json")
    with tarfile.open(directory / "measured-sources.tar.gz") as archive:
        for name, digest in source_manifest.items():
            require(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, "source archive: " + name)
    replay = [read(directory / f"replay/full.rank{rank}.json") for rank in (0, 1)]
    delayed_source = [read(directory / f"replay-delayed-source/full.rank{rank}.json") for rank in (0, 1)]
    timing = [read(directory / f"timing/full.rank{rank}.json") for rank in (0, 1)]
    numeric = [read(directory / f"numeric/numeric.rank{rank}.json") for rank in (0, 1)]
    for left, right in zip(replay, numeric, strict=True):
        for key in ("selection", "loader", "cache_layout", "terminal_contract"):
            require(left[key] == right[key], "official/replay workload differs: " + key)
        require(left["prefix"]["token_sha256"] == right["prefix"]["token_sha256"], "official/replay prefix differs")
    for folder, documents in (
        ("replay", replay),
        ("replay-delayed-source", delayed_source),
        ("timing", timing),
        ("numeric", numeric),
    ):
        deployed = read(directory / folder / "deployed-sources.json")
        for name, digest in deployed.items():
            require(source_manifest[f"{folder}/tools/{name}"] == digest, "deployed source differs: " + name)
        for document in documents:
            require(
                document["loader"]["source_sha256"] == deployed["multi_graph_gdn_q4_full_loader.py"],
                "measured loader differs",
            )
            for name, digest in document["sources"].items():
                require(deployed[name] == digest, "measured source differs: " + name)
    for rank, (document, original) in enumerate(zip(replay, delayed_source, strict=True)):
        proof = document["inherited_delayed_qualification"]
        source = directory / f"replay-delayed-source/full.rank{rank}.json"
        require(
            proof["prior_report_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest(), "delayed source differs"
        )
        require(
            proof["prior_per_step_checks"] == 128 and proof["inherited_modes"] == [MODES[0]],
            "incomplete delayed inheritance",
        )
        for key in ("selection", "loader", "cache_layout", "terminal_contract"):
            require(document[key] == original[key], "inherited delayed contract differs: " + key)
        prior = [c for c in original["checks"] if c.get("mode") == MODES[0]]
        require(
            len(prior) == 128
            and {c["step"] for c in prior} == set(range(128))
            and prior == [c for c in document["checks"] if c.get("mode") == MODES[0]],
            "inherited delayed per-step checks changed",
        )
        require(
            original["closed_checks"] == [dict(mode=MODES[0], steps=128, all_state_exact=True)],
            "inherited delayed trajectory not closed",
        )
        for name in proof["unchanged_sources"]:
            require(
                source_manifest[f"replay-delayed-source/tools/{name}"] == source_manifest[f"replay/tools/{name}"],
                "delayed arithmetic source changed: " + name,
            )
    for name in (
        "multi_graph_gdn_q4_full_probe.py",
        "multi_graph_gdn_q4_full_loader.py",
        "multi_graph_gdn_q4_full_metadata.py",
        "multi_graph_gdn_q4_log_backend.py",
        "multi_graph_gdn_ready_full_lifecycle.py",
    ):
        require(
            source_manifest[f"replay/tools/{name}"] == source_manifest[f"numeric/tools/{name}"],
            "official/compact core source changed: " + name,
        )
    preliminary = replay_audit(replay)
    scheduling, numerical = replay_audit(timing, inherited=replay), numeric_audit(numeric)
    for rank, document in enumerate(timing):
        proof = document["reused_qualification"]
        prior = directory / f"replay/full.rank{rank}.json"
        require(proof["prior_report_sha256"] == hashlib.sha256(prior.read_bytes()).hexdigest(), "reused report differs")
        require(
            proof["prior_probe_sha256"] == replay[rank]["sources"]["multi_graph_gdn_q4_full_probe.py"],
            "prior probe differs",
        )
        require(
            proof["current_probe_sha256"] == document["sources"]["multi_graph_gdn_q4_full_probe.py"],
            "new probe differs",
        )
        for key in ("selection", "loader", "cache_layout", "terminal_contract"):
            require(document[key] == replay[rank][key], "timing contract changed: " + key)
    isolation = read(directory / "timing/timing-isolation.json")
    require(isolation["closed_timing_fully_isolated_from_numeric"], "paired timing was not isolated")
    return dict(
        trace_sha256=FIXED_TRACE_SHA256,
        scheduling=scheduling,
        preliminary_scheduling=preliminary,
        official_numerical=numerical,
        timing_isolation=isolation,
        qualification_inheritance=dict(
            delayed_steps_per_rank_from_prior_complete_mode=128,
            ready_steps_per_rank_restarted_from_common_initial_state=128,
            incomplete_ready_steps_stitched=False,
            reason="CPython timed frame-dump watchdog removed after two matching native fault PCs",
        ),
        model_substitution_qualified=numerical["numerical_screen_passed"],
        split_exclusive_benefit_proven=False,
        fp16="closed; no additional precision scheme tested",
        limitations=[
            "Three pairs in one process, not independent process replications or a service benchmark.",
            "Closed TP endpoints include one compact transition, all 128 steps and final SSM acceptance.",
            "One isolated conversion observation per rank in the primary timing process, not an import distribution.",
            "Qualification-process import overlapped numerical initialization; primary timing is isolated.",
            "Model endpoint is 64 real decoders plus final norm; "
            "no LM head, sampling, draft model or acceptance-rate measurement.",
            "Original vocabulary-random token trace and frozen accepted lengths; no text quality claim.",
            "Ready scheduling uses the same compact arithmetic; padding can reuse the same operator.",
            "Official offload reference supplies numerical observations only, never a performance denominator.",
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    result["analysis_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8", newline="\n")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
