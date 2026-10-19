"""Preserve measured bytes and explicit binary omissions for a reviewable checkpoint."""

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "work/vllm-ascend"
SOURCE = HERE / "mm2-transfer-evidence-v1/evidence"
TARGET = REPO / "benchmarks/results/multi_graph_structural/fia-sink-mm2-transfer-v1"
assert not TARGET.exists()
audit = json.loads((HERE / "mm2-transfer-audit-v1.json").read_bytes())
assert audit["state"] == "passed" and audit["timings"] == 80
manifest = json.loads((SOURCE / "manifest.json").read_bytes())
omitted = {}
for relative, identity in manifest["files"].items():
    path = SOURCE / relative
    assert path.stat().st_size == identity["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
    if path.suffix == ".o" or relative == "screen/cpu-transfer-contract":
        omitted[relative] = identity
assert len(omitted) == 3
TARGET.mkdir()
raw = TARGET / "raw"
for path in SOURCE.rglob("*"):
    if not path.is_file() or path.relative_to(SOURCE).as_posix() in omitted:
        continue
    dest = raw / path.relative_to(SOURCE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
(TARGET / "omitted-binaries.json").write_text(json.dumps(omitted, indent=2) + "\n", encoding="ascii")
for src, dst in (
    (HERE / "mm2-transfer-audit-v1.json", TARGET / "collected-audit.json"),
    (HERE / "mm2-transfer-source-audit-v1.json", TARGET / "source-work-audit.json"),
    (HERE / "mm2-transfer-evidence-v1/collection.json", TARGET / "collection.json"),
):
    shutil.copyfile(src, dst)
tools = TARGET / "tools"
tools.mkdir()
for name in (
    "mm2_transfer_driver.py", "mm2_transfer_deploy.py", "mm2_transfer_status.py",
    "mm2_transfer_collect.py", "mm2_transfer_audit.py", "mm2_transfer_publish.py",
    "mm2_transfer_source_audit.py",
):
    shutil.copyfile(HERE / name, tools / name)
selection = dict(
    raw_files_in_collected_manifest=len(manifest["files"]),
    raw_files_retained=len(manifest["files"]) - len(omitted),
    omitted_binary_files=list(omitted),
    original_collection_all_92_files_verified=True,
    omitted_binary_hashes_verified_before_publication=True,
    saved_checkout_cannot_rehash_omitted_binaries=True,
    measured_sources_unchanged=True,
    stop_decision="Reject wide-V candidate: both process orders regress; no larger matrix",
    scope="Two original records, synthetic sixteen-attention chain; no model/serving/split-exclusive result",
)
(TARGET / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="ascii")
print(json.dumps(dict(target=str(TARGET), retained=selection["raw_files_retained"], omitted=len(omitted))))
