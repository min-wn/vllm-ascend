# SPDX-License-Identifier: Apache-2.0
import copy
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_gdn_ready_consumer_analyze.py"
SPEC = importlib.util.spec_from_file_location("ready_consumer_analysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report(time):
    hashes = dict(output="output", staged_logs="logs", accepted_state=["base"], terminal_state="terminal")
    site = dict(
        site="frozen.pt",
        site_sha256="sha",
        input_hashes={"q": "q"},
        aic=20,
        aiv=40,
        q4=4,
        precision="fixed",
        capacity_requests=116,
        live_requests=115,
        checks=[dict(mode=mode, hashes=hashes) for mode in ("closed_delayed", "closed_ready")],
        round_medians_us={mode: [time, time + 0.01, time - 0.01] for mode in ("closed_delayed", "closed_ready")},
    )
    return dict(status="complete_same_algorithm_diagnostic_not_model_timing", phase="candidate", sites=[site])


def test_accept_consistent_gain_above_baseline_spread():
    result = MODULE.compare([(report(100), report(95)), (report(100.1), report(95)), (report(99.9), report(95))])
    assert result["enter_complete_layer"]
    assert result["sites"][0]["ready_gain_percent"] == pytest.approx(5)


def test_reject_gain_within_baseline_spread():
    result = MODULE.compare([(report(100), report(98)), (report(101), report(99)), (report(99), report(97))])
    assert not result["enter_complete_layer"]


def test_reject_single_pair_regression_even_if_mean_improves():
    result = MODULE.compare([(report(100), report(90)), (report(100), report(90)), (report(100), report(101))])
    assert not result["enter_complete_layer"]


@pytest.mark.parametrize("field", ("site_sha256", "input_hashes", "aiv", "checks"))
def test_reject_unfair_or_numerically_changed_control(field):
    old, new = report(100), report(95)
    new["sites"][0][field] = "changed"
    with pytest.raises((ValueError, TypeError)):
        MODULE.compare([(old, new), (copy.deepcopy(old), report(95)), (copy.deepcopy(old), report(95))])


def test_require_exact_three_pairs():
    with pytest.raises(ValueError):
        MODULE.compare([(report(100), report(95))])
