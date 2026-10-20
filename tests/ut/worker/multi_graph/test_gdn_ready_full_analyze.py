# SPDX-License-Identifier: Apache-2.0
"""A closed TP endpoint must use one rank's additive lifecycle costs."""

import copy
import importlib.util
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    spec = importlib.util.spec_from_file_location("full_analyze", BENCHMARKS / "multi_graph_gdn_ready_full_analyze.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def endpoints():
    return [
        dict(
            rank=rank,
            lifecycle_samples=[
                dict(
                    trial=0,
                    mode="ready",
                    wall_us=wall,
                    transition_us=setup,
                    terminal_wall_us=drain,
                    closed_wall_us=wall + setup,
                )
            ],
        )
        for rank, wall, setup, drain in ((0, 100.0, 10.0, 30.0), (1, 115.0, 5.0, 2.0))
    ]


def test_cost_columns_follow_the_closed_critical_rank(module):
    # Independent maxima would yield 113 + 10 + 30 = 153, a nonexistent endpoint.
    result = module.critical_decomposition(endpoints(), 0, "ready")
    assert result == dict(critical_rank=1, execution_us=113.0, import_us=5.0, terminal_us=2.0, total_us=120.0)


@pytest.mark.parametrize("bad", ("rank", "cost", "drain", "nan"))
def test_incomplete_or_nonadditive_costs_are_rejected(module, bad):
    ranks = copy.deepcopy(endpoints())
    row = ranks[1]["lifecycle_samples"][0]
    if bad == "rank":
        ranks[1]["rank"] = 0
    elif bad == "cost":
        row["closed_wall_us"] += 1
    elif bad == "drain":
        row["terminal_wall_us"] = row["wall_us"] + 1
    else:
        row["wall_us"] = float("nan")
    with pytest.raises(ValueError):
        module.critical_decomposition(ranks, 0, "ready")


@pytest.mark.parametrize("missing_terminal_conv", (False, True))
def test_numeric_audit_retains_terminal_conv_and_reports_threshold_failure(module, monkeypatch, missing_terminal_conv):
    monkeypatch.setattr(module, "common_contract", lambda documents, status: None)
    gdn, fia = [dict(index=0)], [dict(index=3)]
    rows = []
    for step in range(128):
        labels = [
            ("hidden", None),
            ("residual", None),
            *(("gdn0/candidate", prefix) for prefix in range(1, 5)),
            ("gdn0/conv", None),
            ("fia3/key/page", None),
            ("fia3/value/page", None),
        ]
        rows.extend(dict(step=step, label=label, prefix=prefix) for label, prefix in labels)
    rows.append(dict(step=128, label="gdn0/terminal", prefix=None))
    if not missing_terminal_conv:
        rows.append(dict(step=128, label="gdn0/conv", prefix=None))
    for row in rows:
        error = 0.01 if row["label"] == "hidden" else 0.0
        row.update(worst_request_nrmse=error, pooled_nrmse=error, max_abs=error, screen_passed=error <= 0.005)
    documents = [
        dict(
            rank=rank,
            numeric_completed_steps=128,
            numeric_token_sha256="same-token-stream",
            numeric_contract=dict(no_performance_comparison=True, screen_nrmse=0.005),
            cache_layout=dict(gdn=gdn, fia=fia),
            numeric_errors=rows,
            numeric_screen_passed=False,
            numeric_output_screen_passed=False,
        )
        for rank in (0, 1)
    ]
    if missing_terminal_conv:
        with pytest.raises(ValueError, match="observations missing"):
            module.numeric_audit(documents)
    else:
        result = module.numeric_audit(documents)
        assert result["complete_independent_official_trajectory"]
        assert not result["numerical_screen_passed"] and not result["output_screen_passed"]
        assert result["categories"]["conv"]["observations"] == 2 * 129
        assert result["categories"]["hidden"]["failed_observations"] == 2 * 128
