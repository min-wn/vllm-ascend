# SPDX-License-Identifier: Apache-2.0
"""Remote: build/load isolated host bindings only; do not execute NPU work.

Run through ``python work/run_remote.py --script <this file>``. The separate
native OPP build is a prerequisite for later smoke execution, not for compiling
these dynamically resolved host bindings. Installed repositories are untouched.
"""

import difflib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import regex as re

ROOT = Path("/workspace/aclgraph-sink-stage1-bec19b8dea1e")
NAMESPACE = "thesis_fia_sink_stage1"
PINNED_HASHES = {
    "npu_fused_infer_attention_score_v2_sink.cpp": "9939be6f9230ed20ebf3794776c821486a8e677461d5240d4edb521f22f88bca",
    "npu_fused_infer_attention_score_v2_sink_metadata.cpp": (
        "e6ae5b85c6ef32d3bf9d13c050ab280eb0f5b130a7f8e9f4e56459fd34a3ac9e"
    ),
    "ops_common.cpp": "192904e9e043f39043ba347ce136f47a20da1c520bd10790361937a79d4a321e",
    "ops_common.h": "92abe2484f16fd9b7c8fe090e1eae5d9f1f73ad57a0395cf0cfe3b2b668aa103",
}

RUNNER = r'''"""Generated isolated host compilation runner; no NPU tensors/operations."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

root = Path('/workspace/aclgraph-sink-stage1-bec19b8dea1e')
status_path = root / 'extension-status.json'
manifest_path = root / 'extension-source-manifest.json'
namespace = 'thesis_fia_sink_stage1'


def persist(record):
    temporary = status_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(status_path)


status = json.loads(status_path.read_text())
status.update(state='building', runner_pid=os.getpid(), started_unix=time.time())
persist(status)
try:
    os.environ['MAX_JOBS'] = '2'
    # Imports and dispatcher registration are host-side only. In particular,
    # do not query device counts, set a device, allocate NPU tensors or run ops.
    import torch
    import torch_npu
    from torch.utils.cpp_extension import load

    torch_npu_root = Path(importlib.util.find_spec('torch_npu').origin).resolve().parent
    cann_text = os.environ.get('ASCEND_HOME_PATH') or os.environ.get('ASCEND_TOOLKIT_HOME')
    if not cann_text:
        raise RuntimeError('CANN profile must provide ASCEND_HOME_PATH or ASCEND_TOOLKIT_HOME')
    cann = Path(cann_text).resolve()
    torch_npu_include = torch_npu_root / 'include'
    torch_npu_lib = torch_npu_root / 'lib'
    cann_include, cann_lib = cann / 'include', cann / 'lib64'
    for directory in (torch_npu_include, torch_npu_lib, cann_include, cann_lib):
        if not directory.is_dir():
            raise RuntimeError(f'missing installed dependency directory: {directory}')

    manifest = json.loads(manifest_path.read_text())
    sources = [str(root / 'extension-src' / name) for name in (
        'ops_common.cpp', 'npu_fused_infer_attention_score_v2_sink.cpp',
        'npu_fused_infer_attention_score_v2_sink_metadata.cpp', 'schemas.cpp')]
    flags = [f'-L{torch_npu_lib}', f'-L{cann_lib}', '-ltorch_npu', '-lascendcl', '-ldl',
             f'-Wl,-rpath,{torch_npu_lib}', f'-Wl,-rpath,{cann_lib}']
    status.update(
        python=sys.executable, torch_version=torch.__version__, torch_npu_version=torch_npu.__version__,
        torch_cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        torch_npu_root=str(torch_npu_root), cann_root=str(cann),
        environment={key: os.environ.get(key) for key in (
            'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ASCEND_CUSTOM_OPP_PATH', 'LD_LIBRARY_PATH', 'MAX_JOBS')},
        extra_ldflags=flags, sources=sources,
        note='Compile and dispatcher-load only; native symbols/kernel/AICPU execution NOT qualified.',
    )
    persist(status)
    library = load(
        name='thesis_fia_sink_bec19b8dea1e', sources=sources,
        extra_cflags=['-O0', '-g0', '-std=c++17', '-fvisibility=hidden'],
        extra_include_paths=[str(root / 'extension-src'), str(torch_npu_include), str(cann_include)],
        extra_ldflags=flags, build_directory=str(root / 'extension-build'),
        with_cuda=False, is_python_module=False, verbose=True,
    )
    # Introspect schemas/registered kernels without invoking the operators.
    schemas = {}
    for name in ('npu_fused_infer_attention_score_v2_sink', '_npu_fused_infer_attention_score_v2_sink_metadata'):
        qualified = f'{namespace}::{name}'
        schema = torch._C._dispatch_find_schema_or_throw(qualified, '').schema()
        schemas[name] = {
            'schema': str(schema),
            'PrivateUse1': torch._C._dispatch_has_kernel_for_dispatch_key(qualified, 'PrivateUse1'),
            'Meta': torch._C._dispatch_has_kernel_for_dispatch_key(qualified, 'Meta'),
        }
        if not schemas[name]['PrivateUse1'] or not schemas[name]['Meta']:
            raise RuntimeError(f'incomplete isolated registration: {qualified}')
    library_path = Path(library).resolve()
    status.update(state='completed', returncode=0, finished_unix=time.time(), library=str(library_path),
                  extension_library=str(library_path), namespace=namespace,
                  library_sha256=hashlib.sha256(library_path.read_bytes()).hexdigest(), schemas=schemas,
                  source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  npu_work_executed=False, performance_measured=False)
    persist(status)
    print(json.dumps(status, indent=2), flush=True)
except BaseException:
    status.update(state='failed', returncode=1, finished_unix=time.time(), error=traceback.format_exc(),
                  npu_work_executed=False, performance_measured=False)
    persist(status)
    traceback.print_exc()
    raise
'''


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, contents):
    with path.open("xb") as output:
        output.write(contents)


def main():
    # Keep the experiment inside its dedicated tree and verify the pinned
    # binding inputs before publishing independently registered sources.
    if os.name != "posix" or ROOT.resolve().parent != Path("/workspace").resolve():
        raise RuntimeError("This script is intended only for the fixed remote isolated directory")
    status_path = ROOT / "extension-status.json"
    if status_path.exists():
        raise RuntimeError(f"extension already launched; inspect {status_path}, do not overwrite")
    source = ROOT / "source/csrc"
    binding = source / "attention/fused_infer_attention_score_v2_sink/torch_binding"
    if not binding.is_dir():
        raise RuntimeError("Pinned sink source has not been extracted")
    destination = ROOT / "extension-src"
    build = ROOT / "extension-build"
    destination.mkdir(exist_ok=False)
    build.mkdir(exist_ok=False)
    records, diffs = [], []
    for name, expected in PINNED_HASHES.items():
        before = (binding / name).read_bytes()
        if sha256(before) != expected:
            raise RuntimeError(f"unexpected pinned input hash: {name}")
        text = before.decode("utf-8")
        transformed = text.replace("vllm_ascend::fia_v2_sink_opapi", "thesis_fia_sink_stage1_opapi")
        transformed = transformed.replace("namespace custom {", "namespace thesis_fia_sink_stage1_binding {")
        transformed = transformed.replace("custom::", "thesis_fia_sink_stage1_binding::")
        transformed = transformed.replace("namespace custom", "namespace thesis_fia_sink_stage1_binding")
        transformed = transformed.replace("TORCH_LIBRARY_IMPL(_C_ascend,", f"TORCH_LIBRARY_IMPL({NAMESPACE},")
        changes = ["Isolate dispatcher and C++ symbol namespaces; installed extension is not modified."]
        if name == "ops_common.h":
            start = transformed.index("#define EXEC_NPU_CMD_V1(")
            end = transformed.index("#define EXEC_NPU_CMD_v0(", start)
            macro = transformed[start:end]
            if macro.count("auto workspace_tensor =") != 1:
                raise RuntimeError("workspace ownership patch no longer matches the pinned macro")
            if macro.count("fiaKeyThisCall, fiaKeepConverted]") != 1:
                raise RuntimeError("workspace callback capture patch no longer matches")
            macro = macro.replace("auto workspace_tensor =", "workspace_tensor =")
            macro = macro.replace(
                "fiaKeyThisCall, fiaKeepConverted]", "fiaKeyThisCall, fiaKeepConverted, workspace_tensor]"
            )
            transformed = transformed[:start] + macro + transformed[end:]
            changes.append(
                "Repair EXEC_NPU_CMD_V1 workspace ownership: assign outer tensor and retain it in the queued callback."
            )
        after = transformed.encode("utf-8")
        write_new(destination / name, after)
        records.append(
            {
                "file": name,
                "original_file": str((binding / name).relative_to(ROOT)),
                "before_sha256": sha256(before),
                "after_sha256": sha256(after),
                "changes": changes,
            }
        )
        diffs.extend(
            difflib.unified_diff(
                text.splitlines(keepends=True),
                transformed.splitlines(keepends=True),
                fromfile=f"pinned/{name}",
                tofile=f"extension-src/{name}",
            )
        )

    registration = (source / "torch_binding.cpp").read_text(encoding="utf-8")
    section = registration.split("#ifdef VLLM_ASCEND_ENABLE_FIA_V2_SINK\n", 1)[1].split("#endif", 1)[0]
    match = re.fullmatch(r"\s*(TORCH_LIBRARY_FRAGMENT\(_C_ascend, m\)\s*\{.*\})\s*", section, flags=re.DOTALL)
    if not match or section.count('m.def("') != 2:
        raise RuntimeError("Pinned two-operator schema fragment does not match")
    schemas = ("#include <torch/library.h>\n\n" + match.group(1).replace("_C_ascend", NAMESPACE) + "\n").encode()
    write_new(destination / "schemas.cpp", schemas)
    write_new(ROOT / "extension-source.patch", "".join(diffs).encode())
    manifest = {
        "commit": "bec19b8dea1e538e64838f3064f7e91a3fa9ee72",
        "namespace": NAMESPACE,
        "sources": records,
        "schemas_sha256": sha256(schemas),
        "schema_original_file_sha256": sha256((source / "torch_binding.cpp").read_bytes()),
        "patch_sha256": sha256((ROOT / "extension-source.patch").read_bytes()),
        "scope": "Only independent host binding compilation/loading; no native operator or NPU execution.",
        "workspace_lifetime_fix": (
            "Use the outer owning workspace tensor and capture it by value in the OpCommand callback."
        ),
        "installed_extension_modified": False,
    }
    write_new(ROOT / "extension-source-manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
    runner_path = ROOT / "run-extension-build.py"
    write_new(runner_path, RUNNER.encode())
    write_new(
        status_path,
        (
            json.dumps(
                {
                    "state": "queued",
                    "created_unix": time.time(),
                    "max_jobs": 2,
                    "runner_sha256": sha256(RUNNER.encode()),
                    "npu_work_executed": False,
                    "performance_measured": False,
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    )
    with (ROOT / "extension.log").open("xb") as output:
        process = subprocess.Popen(
            [sys.executable, str(runner_path)],
            stdout=output,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=ROOT,
            start_new_session=True,
            env=dict(os.environ, MAX_JOBS="2"),
        )
    print(
        json.dumps(
            {
                "root": str(ROOT),
                "runner_pid": process.pid,
                "status": str(status_path),
                "log": str(ROOT / "extension.log"),
                "namespace": NAMESPACE,
                "manifest": manifest,
                "npu_work_launched": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
