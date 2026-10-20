# SPDX-License-Identifier: Apache-2.0
"""Reusing a costly audit must fail when arithmetic, dispatch or kernels differ."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    spec = importlib.util.spec_from_file_location("reuse", BENCHMARKS / "multi_graph_gdn_ready_reuse.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    current, prior = tmp_path / "current", tmp_path / "prior"
    current.mkdir()
    (prior / "tools").mkdir(parents=True)
    manifest = {}
    for name in (
        "multi_graph_gdn_q4_full_probe.py",
        "multi_graph_gdn_q4_full_metadata.py",
        "multi_graph_gdn_q4_full_loader.py",
        "multi_graph_gdn_q4_log_backend.py",
        "multi_graph_gdn_q4_backend.py",
        "multi_graph_gdn_q4_full_checks.py",
        "multi_graph_gdn_q4_full_stream_audit.py",
        "multi_graph_gdn_cpu_audit.py",
        "multi_graph_gdn_ready_full_lifecycle.py",
    ):
        for target in (current / name, prior / "tools" / name):
            shutil.copyfile(BENCHMARKS / name, target)
        manifest[name] = hashlib.sha256((current / name).read_bytes()).hexdigest()
    report = dict(
        status="qualified_full_dynamic_replay_and_closed_ready_ablation",
        rank=0,
        trace_sha256=module.FIXED_TRACE_SHA256,
        scope="all_64_decoders_and_final_norm",
        selection={},
        loader={},
        cache_layout={},
        prefix=dict(token_sha256="unit-test-only"),
        checks=[
            dict(label="complete_same_schedule_eager_replay_exact", mode=mode, step=step)
            for mode in module.MODES
            for step in range(128)
        ],
        closed_checks=[dict(mode=mode, steps=128, all_state_exact=True) for mode in module.MODES],
        sources={"multi_graph_gdn_q4_full_probe.py": manifest["multi_graph_gdn_q4_full_probe.py"]},
    )
    (prior / "full.rank0.json").write_text(json.dumps(report))
    (prior / "deployed-sources.json").write_text(json.dumps(manifest))
    extensions = tuple(tmp_path / f"unit-test-only-{index}.so" for index in range(4))
    for path in extensions:
        path.write_bytes(b"not-an-accelerator-binary")
    (prior / "runtime-binaries.json").write_text(
        json.dumps({str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in extensions})
    )
    return module, prior, current, report, extensions


def test_unchanged_core_can_reuse_the_full_audit(fixture):
    module, prior, current, report, extensions = fixture
    result = module.validate(prior, current, report, extensions=extensions)
    assert result["prior_per_step_checks"] == 256 and result["no_new_per_step_state_scan_claim"]
    assert result["prior_probe_sha256"] == result["current_probe_sha256"]


@pytest.mark.parametrize("bad", ("step", "prefix", "backend", "binary", "arithmetic", "capture", "terminal"))
def test_reuse_rejects_incomplete_evidence_and_changed_work(fixture, bad):
    module, prior, current, report, extensions = fixture
    if bad == "step":
        old = json.loads((prior / "full.rank0.json").read_text())
        old["checks"].pop()
        (prior / "full.rank0.json").write_text(json.dumps(old))
    elif bad == "prefix":
        report["prefix"]["token_sha256"] = "different-tokens"
    elif bad == "backend":
        with (current / "multi_graph_gdn_q4_log_backend.py").open("a") as output:
            output.write("\n# different measured bytes\n")
    elif bad == "binary":
        extensions[0].write_bytes(b"changed-binary")
    else:
        name = "multi_graph_gdn_ready_full_lifecycle.py" if bad == "terminal" else "multi_graph_gdn_q4_full_probe.py"
        path = current / name
        source, target = {
            "arithmetic": ("tokens[capacity].copy_(token_cpu)", "tokens[capacity].copy_(token_cpu + 1)"),
            "capture": ("pool = torch.npu.graph_pool_handle()", "pool = None"),
            "terminal": ("1 + owner % 4", "1 + owner % 3"),
        }[bad]
        text = path.read_text(encoding="utf-8")
        assert source in text
        path.write_text(text.replace(source, target), encoding="utf-8")
    with pytest.raises(ValueError):
        module.validate(prior, current, report, extensions=extensions)


@pytest.mark.parametrize("missing_step", (False, True))
def test_partial_run_can_reuse_only_a_complete_closed_delayed_mode(fixture, missing_step):
    module, prior, current, report, extensions = fixture
    old = json.loads((prior / "full.rank0.json").read_text())
    old["status"] = "partial_continued_replay_qualification_not_a_performance_result"
    old["checks"] = old["checks"][: 127 if missing_step else 128] + old["checks"][128:149]
    old["closed_checks"] = old["closed_checks"][:1]
    (prior / "full.rank0.json").write_text(json.dumps(old))
    with pytest.raises(ValueError):
        module.validate(prior, current, report, extensions=extensions)
    if missing_step:
        with pytest.raises(ValueError, match="missing steps"):
            module.validate(prior, current, report, extensions=extensions, delayed_only=True)
    else:
        proof = module.validate(prior, current, report, extensions=extensions, delayed_only=True)
        assert proof["prior_per_step_checks"] == 128
        assert proof["inherited_modes"] == [module.MODES[0]]
