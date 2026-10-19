# SPDX-License-Identifier: Apache-2.0
"""Run two bounded sink smoke cases in a fresh, isolated device-0 process.

Generate/copy this script locally; execution is intended only on the remote
Linux machine after the separate native OPP and extension builds finish.
This does not capture graphs, qualify dynamic/causal semantics, benchmark
performance or load a model. Constant Q/K=0 and V=1 imply attention output 1.
"""

import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/workspace/aclgraph-sink-stage1-bec19b8dea1e")
NAMESPACE = "thesis_fia_sink_stage1"
REQUIRED_SYMBOLS = (
    "aclnnFusedInferAttentionScoreV2SinkV3GetWorkspaceSize",
    "aclnnFusedInferAttentionScoreV2SinkV3",
    "aclnnFusedInferAttentionScoreV2SinkMetadataGetWorkspaceSize",
    "aclnnFusedInferAttentionScoreV2SinkMetadata",
)
META_NAME = "_npu_fused_infer_attention_score_v2_sink_metadata"
ATTENTION_NAME = "npu_fused_infer_attention_score_v2_sink"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def persist(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def owned(path, root, *, file=False):
    target = Path(path).resolve(strict=True)
    if not target.is_relative_to(root) or (file and not target.is_file()):
        raise ValueError(f"artifact is outside the isolated root or not a file: {target}")
    return target


def symbol_provenance(library):
    """Resolve through the exact handle and prove the defining ELF with dladdr."""

    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("dli_fname", ctypes.c_char_p),
            ("dli_fbase", ctypes.c_void_p),
            ("dli_sname", ctypes.c_char_p),
            ("dli_saddr", ctypes.c_void_p),
        ]

    dl = ctypes.CDLL(ctypes.util.find_library("dl") or None)
    dl.dladdr.argtypes = (ctypes.c_void_p, ctypes.POINTER(DlInfo))
    dl.dladdr.restype = ctypes.c_int
    handle = ctypes.CDLL(str(library), mode=os.RTLD_NOW | os.RTLD_LOCAL)
    records = []
    for name in REQUIRED_SYMBOLS:
        symbol = getattr(handle, name)
        info = DlInfo()
        address = ctypes.cast(symbol, ctypes.c_void_p)
        if not address.value or not dl.dladdr(address, ctypes.byref(info)) or not info.dli_fname:
            raise RuntimeError(f"dladdr failed: {name}")
        source = Path(os.fsdecode(info.dli_fname)).resolve(strict=True)
        if source != library:
            raise RuntimeError(f"{name} resolved from {source}, expected {library}")
        records.append(dict(symbol=name, source=str(source), address=hex(address.value)))
    return handle, records


def opapi_mappings(expected):
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("Linux /proc/self/maps is required for library-isolation audit")
    mapped = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name == "libcust_opapi.so":
            mapped.add(Path(fields[5]).resolve(strict=True))
    if mapped != {expected}:
        raise RuntimeError(f"libcust_opapi.so mappings are not isolated: {sorted(map(str, mapped))}")
    return sorted(map(str, mapped))


def checked_schemas(torch, extension_status):
    expected_keywords = {
        META_NAME: (
            "num_heads_q",
            "num_heads_kv",
            "head_dim_qk",
            "head_dim_v",
            "actual_seq_lengths",
            "actual_seq_lengths_kv",
            "batch_size",
            "sparse_mode",
            "input_layout",
            "input_layout_kv",
            "block_size",
            "aic_core_num",
            "aiv_core_num",
            "batch_invariant",
        ),
        ATTENTION_NAME: (
            "query",
            "key",
            "value",
            "actual_seq_qlen",
            "actual_seq_kvlen",
            "block_table",
            "meta_data",
            "num_query_heads",
            "num_key_value_heads",
            "softmax_scale",
            "input_layout",
            "sparse_mode",
            "block_size",
            "return_softmax_lse",
        ),
    }
    records = {}
    for name, arguments in expected_keywords.items():
        qualified = f"{NAMESPACE}::{name}"
        schema = torch._C._dispatch_find_schema_or_throw(qualified, "").schema()
        names = tuple(item.name for item in schema.arguments)
        if not set(arguments).issubset(names):
            raise RuntimeError(f"operator schema contract changed: {schema}")
        if str(schema) != extension_status["schemas"][name]["schema"]:
            raise RuntimeError(f"loaded schema differs from the extension build record: {qualified}")
        if not torch._C._dispatch_has_kernel_for_dispatch_key(qualified, "PrivateUse1"):
            raise RuntimeError(f"missing NPU dispatcher implementation: {qualified}")
        records[name] = dict(schema=str(schema), argument_names=names)
    return records


def constant_case(torch, case, core_limit, progress):
    """Smoke only: constant values cannot prove length/page/causal correctness."""
    qlen, qheads, kvheads, dimension = (
        case["query"],
        case["query_heads"],
        case["kv_heads"],
        case["head_dim"],
    )
    kvlen, page = 516, 128
    blocks = (kvlen + page - 1) // page
    device = torch.device("npu:0")
    query = torch.zeros((qlen, qheads, dimension), dtype=torch.bfloat16, device=device)
    key = torch.zeros((blocks, page, kvheads * dimension), dtype=torch.bfloat16, device=device)
    value = torch.ones_like(key)
    qlengths = torch.tensor([qlen], dtype=torch.int64, device=device)
    kvlengths = torch.tensor([kvlen], dtype=torch.int64, device=device)
    pages = torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks)
    metadata_op = getattr(getattr(torch.ops, NAMESPACE), META_NAME)
    attention_op = getattr(getattr(torch.ops, NAMESPACE), ATTENTION_NAME)
    progress("metadata_started")
    metadata = metadata_op(
        qheads,
        kvheads,
        dimension,
        dimension,
        actual_seq_lengths=qlengths,
        actual_seq_lengths_kv=kvlengths,
        batch_size=1,
        sparse_mode=0,
        input_layout="TND",
        input_layout_kv="BnBsH",
        block_size=page,
        aic_core_num=core_limit["cube_core_num"],
        aiv_core_num=core_limit["vector_core_num"],
        batch_invariant=False,
    )
    torch.npu.synchronize()
    if metadata.dtype != torch.int32 or tuple(metadata.shape) != (1024,) or metadata.device != device:
        raise AssertionError(
            f"metadata contract failed: shape={metadata.shape}, dtype={metadata.dtype}, device={metadata.device}"
        )
    metadata_host = metadata.cpu()
    if not bool(torch.isfinite(metadata_host).all()) or not bool((metadata_host != 0).any()):
        raise AssertionError("metadata output is nonfinite or entirely zero")
    metadata_record = dict(
        metadata_shape=list(metadata.shape),
        metadata_dtype=str(metadata.dtype),
        metadata_nonzero=int((metadata_host != 0).sum()),
        metadata_sha256=hashlib.sha256(metadata_host.numpy().tobytes()).hexdigest(),
    )
    progress("metadata_completed", metadata=metadata_record)
    progress("attention_started")
    output, softmax_lse = attention_op(
        query,
        key,
        value,
        actual_seq_qlen=qlengths,
        actual_seq_kvlen=kvlengths,
        block_table=pages,
        meta_data=metadata,
        num_query_heads=qheads,
        num_key_value_heads=kvheads,
        softmax_scale=dimension**-0.5,
        input_layout="TND",
        sparse_mode=0,
        block_size=page,
        return_softmax_lse=False,
    )
    torch.npu.synchronize()
    expected_shape = (qlen, qheads, dimension)
    if tuple(output.shape) != expected_shape or output.dtype != torch.bfloat16 or output.device != device:
        raise AssertionError(f"attention output contract failed: {output.shape}, {output.dtype}, {output.device}")
    output_host = output.float().cpu()
    if not bool(torch.isfinite(output_host).all()):
        raise AssertionError("attention output contains nonfinite values")
    torch.testing.assert_close(output_host, torch.ones(expected_shape), rtol=3e-3, atol=3e-3)
    if softmax_lse.numel() != 0:
        raise AssertionError("return_softmax_lse=False did not produce the expected empty LSE")
    max_abs_error = float((output_host - 1).abs().max())
    progress(
        "attention_completed",
        attention=dict(output_shape=list(output.shape), output_finite=True, max_abs_error_from_one=max_abs_error),
    )
    return dict(
        **case,
        status="passed",
        batch=1,
        kv_length=kvlen,
        page_size=page,
        dtype=str(query.dtype),
        input_layout="TND",
        input_layout_kv="BnBsH",
        sparse_mode=0,
        query_shape=list(query.shape),
        kv_shape=list(key.shape),
        output_shape=list(output.shape),
        **metadata_record,
        output_finite=True,
        max_abs_error_from_one=max_abs_error,
        expected_output=1.0,
        rtol=3e-3,
        atol=3e-3,
        limitation=(
            "Constant noncausal eager smoke only; no graph/dynamic/page-ID/causal/quality/performance qualification"
        ),
    )


def child(args):
    status_path = args.output_dir / "smoke-status.json"
    status = json.loads(status_path.read_text())
    try:
        status.update(state="running", child_pid=os.getpid(), stage="symbol_provenance")
        persist(status_path, status)
        status["python_startup_opp"] = os.environ.get("ASCEND_CUSTOM_OPP_PATH")
        if status["python_startup_opp"] != str(args.opp_root):
            # This image's .pth startup prepends fla_npu paths. Restore the
            # selected vendor before Torch or our native adapter is loaded.
            if "torch" in sys.modules or "torch_npu" in sys.modules:
                raise RuntimeError("OPP changed after Torch was already imported")
            os.environ["ASCEND_CUSTOM_OPP_PATH"] = str(args.opp_root)
        status["effective_opp_before_native_load"] = os.environ["ASCEND_CUSTOM_OPP_PATH"]
        persist(status_path, status)
        if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") != "0":
            raise RuntimeError("smoke may expose physical device 0 only")
        library_handle, symbols = symbol_provenance(args.opapi_lib)
        status.update(symbols=symbols, opapi_library_sha256=sha256(args.opapi_lib))
        persist(status_path, status)
        # Imports are deliberately after child-process environment initialization.
        import torch
        import torch_npu

        extension_status = json.loads((args.root / "extension-status.json").read_text())
        if extension_status.get("state") != "completed" or extension_status.get("returncode") != 0:
            raise RuntimeError("isolated extension build has not completed successfully")
        if extension_status.get("namespace") != NAMESPACE:
            raise RuntimeError("unexpected isolated dispatcher namespace")
        if sha256(args.extension) != extension_status["library_sha256"]:
            raise RuntimeError("extension hash differs from its successful build record")
        status.update(
            stage="extension_load",
            torch_version=torch.__version__,
            torch_npu_version=torch_npu.__version__,
        )
        persist(status_path, status)
        torch.ops.load_library(str(args.extension))
        status["schemas"] = checked_schemas(torch, extension_status)
        status["opapi_mappings"] = opapi_mappings(args.opapi_lib)
        torch.npu.set_device(0)
        stream = torch.npu.current_stream()
        core_limit = dict(torch.npu.get_stream_limit(stream))
        if int(core_limit["cube_core_num"]) <= 0 or int(core_limit["vector_core_num"]) <= 0:
            raise RuntimeError(f"invalid installed stream core limit: {core_limit}")
        status.update(
            stage="constant_eager_smoke",
            physical_device=0,
            logical_device=0,
            device_name=torch.npu.get_device_name(0),
            core_limit=core_limit,
        )
        persist(status_path, status)
        cases = (
            dict(
                name="official_layout_d128_gqa4",
                query=8,
                query_heads=4,
                kv_heads=1,
                head_dim=128,
            ),
            dict(
                name="target_q4_d256_gqa6",
                query=4,
                query_heads=12,
                kv_heads=2,
                head_dim=256,
            ),
        )
        with torch.inference_mode():
            for case in cases:
                print(f"SMOKE_CASE_START {case['name']}", flush=True)
                status["npu_work_executed"] = True
                status["current_case"] = dict(**case, status="running", checkpoint="initializing")
                persist(status_path, status)

                def progress(checkpoint, **evidence):
                    status["current_case"].update(checkpoint=checkpoint, **evidence)
                    persist(status_path, status)

                status["cases"].append(constant_case(torch, case, core_limit, progress))
                status["current_case"]["status"] = "passed"
                status["opapi_mappings"] = opapi_mappings(args.opapi_lib)
                persist(status_path, status)
                print(f"SMOKE_CASE_PASSED {case['name']}", flush=True)
        if library_handle is None:
            raise AssertionError("absolute native library handle lifetime was lost")
        status.update(
            state="completed",
            returncode=0,
            stage="completed",
            finished_unix=time.time(),
            smoke_passed=True,
            performance_measured=False,
            full_model_qualified=False,
        )
        persist(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except BaseException:
        if status.get("current_case", {}).get("status") == "running":
            status["current_case"]["status"] = "failed"
        status.update(
            state="failed",
            returncode=1,
            finished_unix=time.time(),
            error=traceback.format_exc(),
            smoke_passed=False,
            performance_measured=False,
            full_model_qualified=False,
        )
        persist(status_path, status)
        traceback.print_exc()
        return 1


def launcher(args):
    if os.name != "posix" or args.root.parent != Path("/workspace").resolve():
        raise RuntimeError("This script is only for the fixed remote /workspace isolation root")
    args.opp_root = owned(args.opp_root, args.root)
    args.opapi_lib = owned(
        args.opapi_lib or args.opp_root / "op_api/lib/libcust_opapi.so",
        args.root,
        file=True,
    )
    if args.opapi_lib != (args.opp_root / "op_api/lib/libcust_opapi.so").resolve(strict=True):
        raise ValueError("--opp-root must be the vendor leaf containing this exact op_api/lib/libcust_opapi.so")
    build_status = json.loads((args.root / "extension-status.json").read_text())
    args.extension = owned(args.extension or build_status["extension_library"], args.root, file=True)
    output_dir = args.output_dir.resolve()
    if not output_dir.is_relative_to(args.root):
        raise ValueError(f"output directory is outside the isolated root: {output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)
    args.output_dir = owned(output_dir, args.root)
    status_path, log_path = (
        args.output_dir / "smoke-status.json",
        args.output_dir / "smoke.log",
    )
    if status_path.exists() or log_path.exists():
        raise RuntimeError("smoke artifacts already exist; inspect them or choose a fresh --output-dir inside the root")
    environment = dict(os.environ)
    environment["ASCEND_CUSTOM_OPP_PATH"] = str(args.opp_root)
    environment["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    environment["LD_LIBRARY_PATH"] = str(args.opapi_lib.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
    # The smoke starts a fresh Python, so the dynamic linker and binding's
    # static custom-library vectors read the selected paths before imports.
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--root",
        str(args.root),
        "--opp-root",
        str(args.opp_root),
        "--opapi-lib",
        str(args.opapi_lib),
        "--extension",
        str(args.extension),
        "--output-dir",
        str(args.output_dir),
    ]
    status = dict(
        state="queued",
        created_unix=time.time(),
        launcher_pid=os.getpid(),
        command=command,
        root=str(args.root),
        opp_vendor_leaf=str(args.opp_root),
        opapi_library=str(args.opapi_lib),
        extension_library=str(args.extension),
        namespace=NAMESPACE,
        cases=[],
        environment={
            name: environment.get(name)
            for name in (
                "ASCEND_HOME_PATH",
                "ASCEND_OPP_PATH",
                "ASCEND_CUSTOM_OPP_PATH",
                "LD_LIBRARY_PATH",
                "ASCEND_RT_VISIBLE_DEVICES",
                "TASK_QUEUE_ENABLE",
            )
        },
        script_sha256=sha256(__file__),
        extension_status_sha256=sha256(args.root / "extension-status.json"),
        scope="Two B1 constant eager operator smoke cases on device0 only; no performance or graph/model qualification",
        smoke_passed=False,
        performance_measured=False,
        full_model_qualified=False,
        npu_work_executed=False,
    )
    persist(status_path, status)
    with log_path.open("xb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=args.root,
            env=environment,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = 124
    final = json.loads(status_path.read_text())
    if final.get("state") not in ("completed", "failed") or returncode != final.get("returncode"):
        final.update(
            state="failed",
            returncode=returncode,
            finished_unix=time.time(),
            smoke_passed=False,
            error="Child timed out or terminated before a complete status was persisted",
        )
        persist(status_path, final)
    print(
        json.dumps(
            dict(
                status=str(status_path),
                log=str(log_path),
                state=final["state"],
                returncode=returncode,
            )
        )
    )
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--opp-root",
        type=Path,
        required=True,
        help="Isolated vendor leaf containing op_api/lib/libcust_opapi.so",
    )
    parser.add_argument("--opapi-lib", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.root = args.root.resolve(strict=True)
    args.output_dir = (args.output_dir or args.root).resolve()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return child(args) if args.child else launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
