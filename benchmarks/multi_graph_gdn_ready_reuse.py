# SPDX-License-Identifier: Apache-2.0
"""Reuse a complete per-step audit only with unchanged arithmetic and dispatch."""

import ast
import hashlib
import json

from multi_graph_gdn_q4_full_metadata import FIXED_TRACE_SHA256
from multi_graph_gdn_ready_full_lifecycle import MODES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def function_tree(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(definitions) == 1, "ambiguous arithmetic function: " + name)
    return ast.dump(definitions[0], include_attributes=False)


def validate(directory, current, report, *, extensions, delayed_only=False):
    """No shortened audit may be presented as a new full-step qualification."""
    old = json.loads((directory / f"full.rank{report['rank']}.json").read_text())
    expected_status = (
        "partial_continued_replay_qualification_not_a_performance_result"
        if delayed_only
        else "qualified_full_dynamic_replay_and_closed_ready_ablation"
    )
    require(old["status"] == expected_status, "prior audit incomplete")
    require(old["trace_sha256"] == report["trace_sha256"] == FIXED_TRACE_SHA256, "trace changed")
    require(old["scope"] == report["scope"] == "all_64_decoders_and_final_norm", "model endpoint changed")
    for key in ("selection", "loader", "cache_layout"):
        require(old[key] == report[key], "prior audit contract differs: " + key)
    require(old["prefix"]["token_sha256"] == report["prefix"]["token_sha256"], "prefix tokens differ")
    modes = MODES[:1] if delayed_only else MODES
    checks = [
        c for c in old["checks"] if c["label"] == "complete_same_schedule_eager_replay_exact" and c["mode"] in modes
    ]
    wanted = {(mode, step) for mode in modes for step in range(128)}
    require(
        len(checks) == len(wanted) and {(c["mode"], c["step"]) for c in checks} == wanted, "prior audit missing steps"
    )
    require(
        len(old["closed_checks"]) == len(modes)
        and {c["mode"] for c in old["closed_checks"]} == set(modes)
        and all(c["all_state_exact"] and c["steps"] == 128 for c in old["closed_checks"]),
        "prior audit not closed",
    )
    manifest = json.loads((directory / "deployed-sources.json").read_text())
    binaries = json.loads((directory / "runtime-binaries.json").read_text())
    require(len(extensions) == 4, "four unchanged extensions required")
    for extension in extensions:
        require(hashlib.sha256(extension.read_bytes()).hexdigest() == binaries[str(extension)], "kernel binary changed")
    unchanged = (
        "multi_graph_gdn_q4_full_metadata.py",
        "multi_graph_gdn_q4_full_loader.py",
        "multi_graph_gdn_q4_log_backend.py",
        "multi_graph_gdn_q4_backend.py",
        "multi_graph_gdn_q4_full_checks.py",
        "multi_graph_gdn_q4_full_stream_audit.py",
        "multi_graph_gdn_cpu_audit.py",
    )
    for name in unchanged:
        require(hashlib.sha256((current / name).read_bytes()).hexdigest() == manifest[name], "source changed: " + name)
    old_probe = directory / "tools/multi_graph_gdn_q4_full_probe.py"
    require(hashlib.sha256(old_probe.read_bytes()).hexdigest() == old["sources"][old_probe.name], "prior probe changed")
    functions = {
        old_probe.name: (
            "deterministic_token_ids",
            "cache_binding_identity",
            "complete_snapshot",
            "restore_complete",
            "verify_fia_tasks",
            "model_observer",
            "forward",
            "prepare",
            "replay",
            "select_mode",
        ),
        "multi_graph_gdn_ready_full_lifecycle.py": ("__init__", "run", "require_closed", "terminal_counts"),
    }
    for name, names in functions.items():
        prior = directory / "tools" / name
        require(hashlib.sha256(prior.read_bytes()).hexdigest() == manifest[name], "prior source changed: " + name)
        for function in names:
            require(
                function_tree(prior, function) == function_tree(current / name, function),
                "arithmetic changed: " + function,
            )

    # Capture construction itself must not silently acquire a different path.
    def capture_tree(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "replay_trajectories")
        capture = []
        for node in function.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "terminal_graph" for t in node.targets
            ):
                break
            capture.append(node)
        return ast.dump(ast.Module(body=capture, type_ignores=[]), include_attributes=False)

    require(capture_tree(old_probe) == capture_tree(current / old_probe.name), "graph capture changed")
    return dict(
        prior_report_sha256=hashlib.sha256((directory / f"full.rank{report['rank']}.json").read_bytes()).hexdigest(),
        prior_probe_sha256=manifest[old_probe.name],
        current_probe_sha256=hashlib.sha256((current / old_probe.name).read_bytes()).hexdigest(),
        prior_per_step_checks=len(checks),
        unchanged_sources=list(unchanged),
        unchanged_arithmetic_functions=functions,
        new_checks="one complete delayed eager trajectory and exact closed state after every timed sample",
        no_new_per_step_state_scan_claim=True,
        inherited_modes=list(modes),
        prior_directory=str(directory),
    )
