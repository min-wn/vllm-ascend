# SPDX-License-Identifier: Apache-2.0
"""Reject evidence that drops lifecycle work or misses a precision failure."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    spec = importlib.util.spec_from_file_location(
        "range_analyze", BENCHMARKS / "multi_graph_gdn_ready_range_analyze.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = BENCHMARKS / "results/multi_graph_structural/gdn-ready-range-v1/ranges-v1.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    return module, document


def test_frozen_real_evidence_passes(evidence):
    module, document = evidence
    assert module.audit(document)["sha256"] == document["protocol"]["sha256"]


@pytest.mark.parametrize("bad", ["missing_step", "precision", "same_algorithm", "unclosed", "import", "drain"])
def test_incomplete_or_inconsistent_evidence_is_rejected(evidence, bad):
    module, source = evidence
    document = copy.deepcopy(source)
    profile = document["ranks"][0]["profiles"][0]
    if bad == "missing_step":
        profile["checks"].pop()
    elif bad == "precision":
        profile["errors"][0]["worst_request_nrmse"] = 0.0051
    elif bad == "same_algorithm":
        next(c for c in profile["checks"] if c["mode"] == "compact_ready")["same_algorithm_exact"] = False
    elif bad == "unclosed":
        profile["samples"][0]["terminal_exact"] = False
    elif bad == "import":
        sample = next(s for s in profile["samples"] if s["mode"] == "compact_ready")
        sample["closed_wall_us"] = sample["wall_us"]
    else:
        profile["samples"][0]["device_us"] -= profile["samples"][0]["terminal_us"]
    with pytest.raises(ValueError):
        module.audit(document)
