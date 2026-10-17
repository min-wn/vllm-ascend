"""Qualify raw A2 ACLGraph IF execution with an isolated, bounded worker."""

import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SENTINELS = (101, 202)
CANARIES = (701, 702)


class CondTaskParams(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_void_p),
        ("type", ctypes.c_int32),
        ("size", ctypes.c_uint32),
        ("models", ctypes.POINTER(ctypes.c_void_p)),
    ]


class AclCallError(RuntimeError):
    def __init__(self, api, code, message):
        super().__init__(f"{api} returned {code}: {message}")
        self.api, self.code, self.message = api, code, message


def expected_branch_values(condition):
    if condition not in (0, 1):
        raise ValueError("IF condition must be zero or one")
    selected = 0 if condition else 1
    values = list(CANARIES)
    values[selected] = SENTINELS[selected]
    return selected, tuple(values), SENTINELS[selected]


def bind_acl(path):
    library = ctypes.CDLL(path)
    ptr, u32, i32, size = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32, ctypes.c_size_t
    pp = ctypes.POINTER(ptr)
    signatures = {
        "aclInit": (ctypes.c_char_p,),
        "aclFinalize": (),
        "aclrtSetDevice": (i32,),
        "aclrtCreateStream": (pp,),
        "aclrtDestroyStream": (ptr,),
        "aclrtSynchronizeStreamWithTimeout": (ptr, i32),
        "aclrtMalloc": (pp, size, i32),
        "aclrtFree": (ptr,),
        "aclrtMemcpy": (ptr, size, ptr, size, i32),
        "aclrtMemcpyAsync": (ptr, size, ptr, size, i32, ptr),
        "aclmdlRICaptureBegin": (ptr, i32),
        "aclmdlRICaptureGetInfo": (ptr, ctypes.POINTER(i32), pp),
        "aclmdlRICaptureEnd": (ptr, pp),
        "aclmdlRICondHandleCreate": (ptr, u32, i32, pp),
        "aclmdlRICondHandleGetCondPtr": (ptr, pp),
        "aclmdlRIAddCondTask": (CondTaskParams, ptr, u32),
        "aclmdlRICaptureToModelRIBegin": (ptr, ptr, i32),
        "aclmdlRIExecuteAsync": (ptr, ptr),
        "aclmdlRIDestroy": (ptr,),
    }
    for name, signature in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = signature, i32
    library.aclGetRecentErrMsg.argtypes = ()
    library.aclGetRecentErrMsg.restype = ctypes.c_char_p
    return library


def worker(args):
    report = dict(
        status="started",
        calls=[],
        cleanup_errors=[],
        device=args.device,
        scope="Raw ACLGraph IF with D2D branches; no FIA/HCCL/model qualification",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        no_driver_or_device_reset=True,
        condition_contract="Parent captured D2D copies conditionInput to condPtr after ASSIGN_DEFAULT initialization",
        reference="https://github.com/hicann/runtime/blob/b5c8dd9a4022aba8964feb2daa2dc8973f7cb214/example/2_advanced_features/model_ri/3_cond_model/main.cpp",
    )
    library, initialized, capturing = None, False, False
    streams, buffers = [], []
    parent = ctypes.c_void_p()

    def call(name, *values):
        result = int(getattr(library, name)(*values))
        report["calls"].append(dict(api=name, result=result))
        if result:
            raw = library.aclGetRecentErrMsg()
            raise AclCallError(name, result, raw.decode("utf-8", errors="replace") if raw else "")

    def write(pointer, value):
        source = ctypes.c_uint64(value)
        call("aclrtMemcpy", pointer, 8, ctypes.byref(source), 8, 1)

    def read(pointer):
        result = ctypes.c_uint64()
        call("aclrtMemcpy", ctypes.byref(result), 8, pointer, 8, 2)
        return result.value

    def allocate(value):
        pointer = ctypes.c_void_p()
        call("aclrtMalloc", ctypes.byref(pointer), 8, 0)
        buffers.append(pointer)
        write(pointer, value)
        return pointer

    def copy(destination, source, stream):
        call("aclrtMemcpyAsync", destination, 8, source, 8, 3, stream)

    try:
        library = bind_acl(args.library)
        call("aclInit", None)
        initialized = True
        call("aclrtSetDevice", args.device)
        for _ in range(4):
            stream = ctypes.c_void_p()
            call("aclrtCreateStream", ctypes.byref(stream))
            streams.append(stream)
        capture, child_a, child_b, execute = streams
        condition_input = allocate(0)
        sources = tuple(allocate(value) for value in SENTINELS)
        destinations = tuple(allocate(value) for value in CANARIES)
        output = allocate(703)
        call("aclmdlRICaptureBegin", capture, 1)
        capturing = True
        status = ctypes.c_int32()
        call("aclmdlRICaptureGetInfo", capture, ctypes.byref(status), ctypes.byref(parent))
        if status.value != 1 or not parent.value:
            raise ValueError("parent capture must be active with nonnull model")
        handle, cond_ptr = ctypes.c_void_p(), ctypes.c_void_p()
        call("aclmdlRICondHandleCreate", parent, 0, 1, ctypes.byref(handle))
        call("aclmdlRICondHandleGetCondPtr", handle, ctypes.byref(cond_ptr))
        if not cond_ptr.value:
            raise ValueError("condition device pointer is null")
        # Runtime restores the default condition before execution, so update
        # the condition from device memory inside the parent graph itself.
        copy(cond_ptr, condition_input, capture)
        children = (ctypes.c_void_p * 2)()
        call("aclmdlRIAddCondTask", CondTaskParams(handle, 0, 2, children), capture, 0)
        for index, stream in enumerate((child_a, child_b)):
            child = ctypes.c_void_p(children[index])
            identity = child.value
            call("aclmdlRICaptureToModelRIBegin", stream, child, 1)
            copy(destinations[index], sources[index], stream)
            copy(output, sources[index], stream)
            call("aclmdlRICaptureEnd", stream, ctypes.byref(child))
            if child.value != identity:
                raise ValueError("child model changed identity")
        call("aclmdlRICaptureEnd", capture, ctypes.byref(parent))
        capturing = False
        identities = (parent.value, cond_ptr.value)
        checks = []
        for step, condition in enumerate((0, 1, 1, 0, 1, 0) * args.repeats):
            write(condition_input, condition)
            for pointer, value in zip(destinations, CANARIES, strict=True):
                write(pointer, value)
            write(output, 703)
            call("aclmdlRIExecuteAsync", parent, execute)
            call("aclrtSynchronizeStreamWithTimeout", execute, args.sync_timeout_ms)
            selected, expected, expected_output = expected_branch_values(condition)
            actual = tuple(read(pointer) for pointer in destinations)
            actual_output, actual_condition = read(output), read(cond_ptr)
            if actual != expected or actual_output != expected_output or actual_condition != condition:
                raise ValueError(
                    f"condition={condition}, branches={actual}/{expected}, output={actual_output}/{expected_output}"
                )
            if identities != (parent.value, cond_ptr.value):
                raise ValueError("model or condition pointer changed identity")
            checks.append(
                dict(
                    step=step,
                    condition=condition,
                    selected=selected,
                    destinations=actual,
                    output=actual_output,
                    unselected_canary_unchanged=True,
                )
            )
        report.update(
            status="qualified_raw_conditional_IF",
            checks=checks,
            captures=1,
            replays=len(checks),
            model_identity=identities[0],
            condition_pointer=identities[1],
        )
    except AclCallError as error:
        report.update(
            status="unsupported" if error.code == 207000 else "failed",
            api=error.api,
            error_code=error.code,
            error=str(error),
            recent_error=error.message,
            support_gate="softwareSQ + driver dynamicSQbinding + TS_FEATURE_ACLGRAPH_COND_OP",
        )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        # Persist the result before cleanup, which is also bounded by the
        # separate parent process watchdog. No reset API is used.
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if library:
            actions = []
            if capturing and streams:
                actions.append(("aclmdlRICaptureEnd", streams[0], ctypes.byref(parent)))
            if parent.value:
                actions.append(("aclmdlRIDestroy", parent))
            actions.extend(("aclrtDestroyStream", stream) for stream in reversed(streams))
            actions.extend(("aclrtFree", pointer) for pointer in reversed(buffers))
            if initialized:
                actions.append(("aclFinalize",))
            for name, *values in actions:
                try:
                    code = int(getattr(library, name)(*values))
                    if code:
                        report["cleanup_errors"].append(dict(api=name, result=code))
                except Exception as error:
                    report["cleanup_errors"].append(dict(api=name, error=str(error)))
        report["cleanup_complete"] = not report["cleanup_errors"]
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["status"] in ("qualified_raw_conditional_IF", "unsupported") else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", default="libascendcl.so")
    parser.add_argument("--sync-timeout-ms", type=int, default=5000)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output.exists() or min(args.sync_timeout_ms, args.timeout_seconds, args.repeats) <= 0 or args.device < 0:
        parser.error("new output path, positive timeout/repeats and nonnegative device required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    try:
        completed = subprocess.run(command, timeout=args.timeout_seconds, check=False)
        if not args.output.exists():
            args.output.write_text(
                json.dumps(dict(status="worker_failed_without_report", returncode=completed.returncode))
            )
        return completed.returncode
    except subprocess.TimeoutExpired:
        previous = json.loads(args.output.read_text()) if args.output.exists() else None
        args.output.write_text(
            json.dumps(
                dict(status="timed_out", timeout_seconds=args.timeout_seconds, preserved_worker_result=previous),
                indent=2,
            )
            + "\n"
        )
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
