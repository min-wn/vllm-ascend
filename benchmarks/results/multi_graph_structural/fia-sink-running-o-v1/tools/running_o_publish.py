"""Save an exact, audited source-ablation checkpoint without changing raw bytes."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "work/vllm-ascend"
RESULTS = REPO / "benchmarks/results/multi_graph_structural"
TARGET = RESULTS / "fia-sink-running-o-v1"
assert not TARGET.exists()
audit = json.loads((HERE / "running-o-independent-audit-final.json").read_bytes())
assert audit["qualified"] and audit["timings"] == 80 and audit["sampled_fp32_rows"] == 144
source = HERE / "ro-v3-final"
manifest = json.loads((source / "manifest.json").read_bytes())
for row in manifest:
    data = (source / row["path"]).read_bytes()
    assert len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"]
shutil.copytree(source, TARGET)
tools = TARGET / "tools"
tools.mkdir()
for name in ("running_o_audit.py", "running_o_collect.py", "running_o_deploy.py", "running_o_driver.py",
             "running_o_recompute.py", "running_o_publish.py"):
    shutil.copyfile(HERE / name, tools / name)
for name in ("running-o-ub-source-review.json", "running-o-host-cpp-check.json", "running-o-host-cpp-check-v1.json",
             "running-o-target-keys.json", "running-o-scoped-checks.log", "format-ci-running-o.log"):
    shutil.copyfile(HERE / name, TARGET / name)
attempts = []
for version in (1, 2):
    failed = HERE / f"running-o-evidence-v{version}-complete"
    archive = TARGET / f"compile-attempt-v{version}.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(failed, arcname=f"compile-attempt-v{version}")
    attempts.append(dict(version=version, archive=archive.name,
                         sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                         npu_timings=0, partial=True))
selection = dict(selected_remote_version=3, manifest_sha256=hashlib.sha256((TARGET / "manifest.json").read_bytes()).hexdigest(),
                 attempts=attempts,
                 reasons={"v1": "Compiler missing retained tiling registration; failed before NPU. Deployed patch predates final freeze; no measurements used.",
                          "v2": "Controller deliberately stopped its own broad CPU compilation. Status still running is preserved with interruption sidecar; no NPU runs.",
                          "v3": "Both macro builds completed with exactly two target tiling keys; all four probes completed and audited."},
                 raw_files_unchanged_after_collection=True,
                 complete_timing_scope="Synthetic sixteen-attention chain, two original records, one fresh process/cell; no stable/full-trace/model/split-over-padding claim",
                 audit_schema_correction="Whole vendor retains its qualified extra libthesis_stage5_whole_aicpu.so; this exact immutable library and dispatcher are the only cross-policy non-consumer inventory differences.",
                 local_collection_path_adjustment="A first local extraction hit Windows MAX_PATH. Final collection shortens metadata destination names; raw bytes and original remote paths remain in manifest.")
(TARGET / "run-selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="ascii")
for src, dst in (("running-o-independent-audit-final.json", "fia-sink-running-o-independent-audit.json"),
                 ("running-o-primary-recompute-final.json", "fia-sink-running-o-primary-recompute.json")):
    assert not (RESULTS / dst).exists()
    shutil.copyfile(HERE / src, RESULTS / dst)
print(json.dumps(dict(target=str(TARGET), raw_files=len(manifest),
                      files=sum(path.is_file() for path in TARGET.rglob("*"))), indent=2))
