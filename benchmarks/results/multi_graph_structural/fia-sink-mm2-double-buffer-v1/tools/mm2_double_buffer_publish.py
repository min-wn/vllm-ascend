"""Preserve exact collected bytes; omit only native executables with identities."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "work/vllm-ascend"
SOURCE = HERE / "mm2-double-buffer-evidence-v1/evidence"
TARGET = REPO / "benchmarks/results/multi_graph_structural/fia-sink-mm2-double-buffer-v1"
assert not TARGET.exists()
manifest = json.loads((SOURCE / "manifest.json").read_bytes())
screen = json.loads((SOURCE / "screen/mm2-double-buffer-screen.json").read_bytes())
assert screen["state"] == "completed" and len(screen["runs"]) == 6
omitted = {}
for relative, identity in manifest["files"].items():
    path = SOURCE / relative
    assert path.resolve().is_relative_to(SOURCE.resolve()) and not path.is_symlink()
    assert path.stat().st_size == identity["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == identity["sha256"]
    if path.suffix == ".o" or relative == "screen/cpu-double-buffer-contract":
        omitted[relative] = identity
assert len(omitted) == 4
TARGET.mkdir()
for relative in [*manifest["files"], "manifest.json"]:
    if relative in omitted:
        continue
    path = SOURCE / relative
    dest = TARGET / "raw" / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
(TARGET / "omitted-binaries.json").write_text(json.dumps(omitted, indent=2) + "\n", encoding="ascii")
shutil.copyfile(HERE / "mm2-double-buffer-evidence-v1/collection.json", TARGET / "collection.json")
tools = TARGET / "tools"
tools.mkdir()
for name in ("mm2_double_buffer_driver.py", "mm2_double_buffer_deploy.py", "mm2_double_buffer_status.py",
             "mm2_double_buffer_collect.py", "mm2_double_buffer_publish.py"):
    shutil.copyfile(HERE / name, tools / name)
shutil.copyfile(HERE / "mm2_double_buffer_saved_audit.py", tools / "saved_audit.py")
shutil.copyfile(HERE / "mm2_double_buffer_recompute.py", tools / "recompute.py")
selection = dict(
    collected_raw_files=len(manifest["files"]), retained_raw_files=len(manifest["files"]) - len(omitted),
    omitted_binary_files=list(omitted), collected_files_all_verified=True,
    omitted_binary_hashes_verified_before_publication=True, saved_checkout_cannot_rehash_omitted_binaries=True,
    measured_sources_unchanged=True, full_model_or_split_gain=False,
    scope="Two original records, six fresh processes; synthetic sixteen-attention chain",
)
(TARGET / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="ascii")
print(json.dumps(dict(target=str(TARGET), retained=selection["retained_raw_files"], omitted=len(omitted))))
