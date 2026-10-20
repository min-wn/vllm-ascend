# SPDX-License-Identifier: Apache-2.0
"""Verify frozen real-activation evidence and matched lifecycle ablations."""

import argparse
import hashlib
import json
import math
import statistics
import tarfile
from pathlib import Path

from multi_graph_gdn_ready_range_protocol import protocol


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(document):
    frozen = json.loads(json.dumps(protocol()))
    modes, rounds = frozen["modes"], frozen["rounds"]
    require(document["status"] == "qualified_real_activation_single_layer_ranges", "unqualified input")
    require(document["protocol"] == frozen, "frozen protocol differs")
    require(len(document["ranks"]) == 2 and {r["rank"] for r in document["ranks"]} == {0, 1}, "TP2 required")
    require(
        document["precision"] == dict(activation="BF16", compact_log="FP32", ring_dk="FP32", state="FP32", hf32=False),
        "precision changed",
    )
    for rank in document["ranks"]:
        require([p["name"] for p in rank["profiles"]] == [p["name"] for p in frozen["profiles"]], "profile mismatch")
        for profile, spec in zip(rank["profiles"], frozen["profiles"], strict=True):
            steps = len(spec["trace"])
            wanted = {(step, mode) for step in range(steps) for mode in modes}
            require(
                len(profile["checks"]) == len(wanted) and {(c["step"], c["mode"]) for c in profile["checks"]} == wanted,
                "incomplete continuous qualification",
            )
            require(
                all(
                    c["paused_exact"] and (not c["mode"].endswith("_ready") or c["same_algorithm_exact"])
                    for c in profile["checks"]
                ),
                "paused state or matched scheduling check failed",
            )
            expected_errors = {
                (step, mode + "/" + field)
                for step in range(steps)
                for mode in modes[1:]
                for field in ("hidden", "residual")
            } | {(steps, mode + "/terminal") for mode in modes[1:]}
            require(
                len(profile["errors"]) == len(expected_errors)
                and {(e["step"], e["label"]) for e in profile["errors"]} == expected_errors,
                "output or closed-state precision evidence missing",
            )
            require(
                all(
                    math.isfinite(e["worst_request_nrmse"])
                    and 0 <= e["worst_request_nrmse"] <= frozen["threshold_nrmse"]
                    for e in profile["errors"]
                ),
                "precision threshold failed",
            )
            wanted_samples = {(trial, mode) for trial in range(rounds) for mode in modes}
            require(
                len(profile["samples"]) == len(wanted_samples)
                and {(s["round"], s["mode"]) for s in profile["samples"]} == wanted_samples,
                "missing paired trials",
            )
            require(
                [a["step"] for a in profile["activation_manifest"]] == list(range(steps)), "activation evidence missing"
            )
            for sample in profile["samples"]:
                require(sample["terminal_exact"] and len(sample["step_device_us"]) == steps, "unclosed timing sample")
                require(
                    all(
                        math.isfinite(sample[key]) and sample[key] > 0
                        for key in ("wall_us", "closed_wall_us", "device_us", "terminal_us")
                    ),
                    "invalid timing",
                )
                require(
                    math.isclose(sample["closed_wall_us"], sample["wall_us"] + sample["import_us"])
                    and math.isclose(sample["device_us"], sum(sample["step_device_us"]) + sample["terminal_us"]),
                    "import/drain cost omitted",
                )
    return frozen


def analyze(directory):
    documents = [read(directory / f"ranges-v{index}.json") for index in (1, 2)]
    for name, spec in read(directory / "evidence-manifest.json").items():
        data = (directory / name).read_bytes()
        require(
            len(data) == spec["bytes"] and hashlib.sha256(data).hexdigest() == spec["sha256"],
            "evidence checksum: " + name,
        )
    sources = read(directory / "measured-source-manifest.json")
    with tarfile.open(directory / "measured-sources.tar.gz") as archive:
        for name, digest in sources.items():
            require(hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest, "source archive: " + name)
    frozen = audit(documents[0])
    require(read(directory / "frozen-protocol.json") == frozen, "predeclared protocol changed")
    require(documents[0]["sources"] == documents[1]["sources"], "sources differ across processes")
    require(documents[0]["token_sha256"] == documents[1]["token_sha256"], "tokens differ across processes")
    for document in documents:
        audit(document)
        for name, digest in document["sources"].items():
            require(sources["tools/" + name] == digest, "measured source differs: " + name)
    summary = {}
    for index, spec in enumerate(frozen["profiles"]):
        for rank in (0, 1):
            require(
                documents[0]["ranks"][rank]["profiles"][index]["activation_manifest"]
                == documents[1]["ranks"][rank]["profiles"][index]["activation_manifest"],
                "activations differ across processes",
            )
        rows = []
        for process, document in enumerate(documents, 1):
            for trial in range(frozen["rounds"]):
                times = {}
                for mode in frozen["modes"]:
                    samples = [
                        next(s for s in rank["profiles"][index]["samples"] if s["round"] == trial and s["mode"] == mode)
                        for rank in document["ranks"]
                    ]
                    times[mode] = {
                        key: max(s[key] for s in samples)
                        for key in ("wall_us", "closed_wall_us", "device_us", "terminal_us", "import_us")
                    }
                row = dict(process=process, round=trial, times=times)
                for family in ("compact", "ring"):
                    row[family + "_ready_percent"] = 100 * (
                        1 - times[family + "_ready"]["closed_wall_us"] / times[family + "_delayed"]["closed_wall_us"]
                    )
                row["compact_vs_official_percent"] = 100 * (
                    1 - times["compact_ready"]["closed_wall_us"] / times["official"]["closed_wall_us"]
                )
                rows.append(row)
        errors = [e for document in documents for rank in document["ranks"] for e in rank["profiles"][index]["errors"]]
        summary[spec["name"]] = dict(
            live_range=[min(t["live"] for t in spec["trace"]), max(t["live"] for t in spec["trace"])],
            paired=rows,
            mean_closed_us={
                mode: statistics.mean(r["times"][mode]["closed_wall_us"] for r in rows) for mode in frozen["modes"]
            },
            mean_percent={
                key: statistics.mean(r[key] for r in rows)
                for key in ("compact_ready_percent", "ring_ready_percent", "compact_vs_official_percent")
            },
            ready_positive_rounds={
                family: sum(r[family + "_ready_percent"] > 0 for r in rows) for family in ("compact", "ring")
            },
            max_output_nrmse=max(e["worst_request_nrmse"] for e in errors if not e["label"].endswith("/terminal")),
            max_terminal_nrmse=max(e["worst_request_nrmse"] for e in errors if e["label"].endswith("/terminal")),
        )
    attribution = read(directory / "fp16-first-failure/attribution.json")
    require(attribution["status"] == "first_FP16_failure_staged_attribution", "missing FP16 attribution")
    for rank in attribution["ranks"]:
        require(
            rank["step"] == 1
            and rank["owner"] == 34
            and rank["input_qkv_beta_gate_exact"]
            and rank["first_failure_reproduced"]
            and rank["no_performance_measurement"],
            "FP16 scope changed",
        )
        cases = rank["counterfactuals"]
        require(
            cases["quantized_materialized_oracle"]["hidden"] > rank["original_threshold"]
            and cases["official_dense_materialized_oracle"]["hidden"] <= rank["original_threshold"]
            and all(value == 0 for value in cases["official_recurrence_output"].values()),
            "FP16 attribution inconclusive",
        )
    return dict(
        decision="single_layer_precision_passed_repeatable_medium_high_ready_benefit_full_replay_still_required",
        protocol_sha256=frozen["sha256"],
        profiles=summary,
        imports_us=[
            {key: rank[key] for key in ("compact_import_us", "ring_import_us")}
            for document in documents
            for rank in document["ranks"]
        ],
        fp16_decision="retain_failed_configuration_state_approximation_amplified_no_implementation_fault_found_in_first_sample",
        qualification_checks=sum(len(p["checks"]) for d in documents for r in d["ranks"] for p in r["profiles"]),
        closed_timing_checks=sum(len(p["samples"]) for d in documents for r in d["ranks"] for p in r["profiles"]),
        limitations=[
            "All load ranges use the same 400-token graph; not default small-bucket performance at low load.",
            "Official first-four-layer real-text activations and coherent prefill; target layer 4 only timed.",
            "Scheduled accepted prefixes, no autoregressive generation or quality claim.",
            "One import observation per process/rank, not a stable import distribution.",
            "Independent difference is A2 request-group ready scheduling; "
            "compact/ring state algorithms are prior work.",
            "Low-load ready benefit is unstable; no router introduced and no all-range acceleration claim.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {key: value for key, value in profile.items() if key != "paired"}
                for name, profile in result["profiles"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
