"""Run a bounded ABBA transfer screen without modifying qualified artifacts."""

import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/workspace/aclgraph-sink-stage2-d256-bec19b8dea1e")
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from aclgraph_fia_sink_stage7_suite import run_one
import aclgraph_fia_sink_mm2_transfer_patch as patch

OUT = ROOT / "mm2-transfer-screen-v1"
assert not OUT.exists()
OUT.mkdir()
REPORT = OUT / "mm2-transfer-screen.json"
CELLS = ("control", "wide", "wide", "control")
status = dict(
    state="running", stage="cpu_contract", started_unix=time.time(), runs=[], builds=[],
    design=dict(cells=list(CELLS), policy="whole", trial=0, seed=1703, warmups=5,
                repeats=10, profile_repeats=1, capacities=[400, 512], occurrences=16),
    source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(TOOLS.glob("*.py"))},
    scope="Two fixed synthetic attention-chain records; transfer-only source ablation; not a split-exclusive or full-model result",
    performance_measured=False, full_trace_measured=False, full_model_qualified=False,
    end_to_end_gain_established=False,
)


def save():
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(REPORT)


def idle(label):
    info = subprocess.run(["npu-smi", "info"], stdout=subprocess.PIPE, text=True, check=True).stdout
    (OUT / ("device-" + label + ".txt")).write_text(info)
    assert "No running processes found in NPU 0" in info


def cpu_contract():
    native = ROOT / "source/csrc/build/binary/ascend910b/src/fused_infer_attention_score_v2_sink/op_kernel"
    cube = (native / patch.SOURCE.name).read_bytes()
    assert hashlib.sha256(cube).hexdigest() == patch.EXPECTED_SHA256
    patched = patch.transformed(cube)
    (OUT / "cube-before.h.txt").write_bytes(cube)
    (OUT / "cube-candidate.h.txt").write_bytes(patched)
    axis = (native / "axis.h").read_text().replace('#include "kernel_operator.h"', "")
    prefix = r'''
#include <cstdint>
#include <cassert>
#include <vector>
#include <cstdio>
#define __aicore__
constexpr uint32_t BLOCK_CUBE = 16;
template <typename T> T Align(T v, T a) { return (v + a - 1) / a * a; }
'''
    function = r'''
std::vector<uint32_t> chunks(uint32_t length, bool wide, bool sink) {
    struct { uint32_t headDim = 256, headDimRope = 0; } constInfo;
    struct { bool isSinkTensor; } info{sink};
    bool THESIS_MM2_WIDE_V = wide;
    const uint32_t K_SPLIT_SIZE = 64, K_BASE = 64;
    Axis k(length);
''' + patch.SPLIT + r'''
    std::vector<uint32_t> result;
    for (auto &kL1 : KL1Slices) {
        const auto rows = kL1.AlignedSize();
        assert(rows * 256 <= (wide ? 32768U : 16384U));
        assert(rows * 64 <= 24576U);
        for (auto &kL0 : kL1.Split(K_BASE)) {
            assert(kL0.AlignedSize() * 256 <= 16384U);
            assert(kL0.start + kL0.sizeAct <= kL1.sizeAct);
            const bool last = ((k.sizeAct - kL1.start - kL0.start) <= K_BASE);
            const bool initialize = (kL1.start == 0) && (kL0.start == 0);
            result.push_back(kL1.start + kL0.start);
            result.push_back(kL0.sizeAct);
            result.push_back(last);
            result.push_back(initialize);
        }
    }
    return result;
}
int main() {
    for (uint32_t length = 0; length <= 512; ++length) {
        for (bool sink : {false, true}) {
            auto old = chunks(length, false, sink);
            auto newer = chunks(length, true, sink);
            assert(old == newer);
            assert(old.size() == 4 * ((length + 63) / 64));
            uint32_t covered = 0;
            for (uint32_t i = 0; i < old.size(); i += 4) {
                assert(old[i] == covered);
                covered += old[i + 1];
                assert(old[i + 2] == (covered == length));
                assert(old[i + 3] == (i == 0));
            }
            assert(covered == length);
        }
    }
    static_assert(2 * 16384 == 1 * 32768);
    static_assert((2 * 49152 + 32768 + 3 * 24576 + 2 * 16384) * 2 == 475136);
    std::puts("1026 old/new exact compute-order and tail contracts passed");
}
'''
    source = OUT / "cpu-transfer-contract.cpp"
    source.write_text(prefix + axis + function)
    binary = OUT / "cpu-transfer-contract"
    with (OUT / "cpu-transfer-contract.log").open("xb") as log:
        subprocess.run(["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=60)
        subprocess.run([str(binary)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=20)
    status["cpu_contract"] = dict(passed=True, cases=1026, exact_native_axis=True,
                                  scope="address, tail, ordered compute and capacity contracts; not hardware synchronization")
    save()


save()
try:
    idle("before")
    cpu_contract()
    for name, enabled in (("control", 0), ("wide", 1)):
        build = ROOT / ("mm2-transfer-build-" + name + "-v1")
        command = [sys.executable, str(TOOLS / "aclgraph_fia_sink_mm2_transfer_build.py"),
                   "--root", str(ROOT),
                   "--upstream-vendor", str(ROOT / "opp/packages/vendors/thesis_fia_stage2_transformer"),
                   "--whole-vendor", str(ROOT / "stage5-no-fd/relinked-v3/opp/packages/vendors/materialized-baseline-v3"),
                   "--output-dir", str(build), "--wide-v", str(enabled), "--timeout", "1200"]
        entry = dict(name=name, command=command, started_unix=time.time())
        status["builds"].append(entry)
        status["stage"] = "build-" + name
        save()
        with (OUT / ("build-" + name + ".log")).open("xb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1300)
        entry.update(returncode=result.returncode, finished_unix=time.time())
        save()
        assert result.returncode == 0
        metadata = json.loads((build / "mm2-transfer-build.json").read_bytes())
        assert metadata["state"] == "completed" and metadata["wide_v"] == bool(enabled)
    assert hashlib.sha256((TOOLS / "aclgraph_fia_sink_consumer_probe.py").read_bytes()).hexdigest() == \
        "b7472c5ccbdb37d4f5ea1693f2abc5145a71abb773d1e09ea788bf5daca964c9"
    for index, variant in enumerate(CELLS):
        idle("cell" + str(index))
        build = ROOT / ("mm2-transfer-build-" + variant + "-v1")
        output = OUT / (f"cell-{index:02d}-{variant}-whole")
        options = dict(root=ROOT, opp_root=build / "vendor-whole", output_dir=output,
                       trace=TOOLS / "structural-dynamic-trace-v1.json",
                       selection=TOOLS / "q4-tile-full-replay-v8-queue0.rank0.json", mode="whole", trial=0,
                       repeats=10, warmups=5, profile_repeats=1, seed=1703, timeout=900)
        command = [sys.executable, str(TOOLS / "aclgraph_fia_sink_consumer_probe.py")]
        for key, value in options.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        entry = dict(variant=variant, policy="whole", process_order_index=index, command=command,
                     launched_unix=time.time(), report_path=(output / "consumer-probe.json").relative_to(OUT).as_posix())
        status["runs"].append(entry)
        status["stage"] = "cell-" + str(index)
        save()
        rc = run_one(command, OUT / (f"cell-{index:02d}.log"), 900, entry, save)
        entry.update(returncode=rc, finished_unix=time.time())
        save()
        assert rc == 0
        report = json.loads((output / "consumer-probe.json").read_bytes())
        assert report["state"] == "completed" and len(report["samples"]) == 20
        entry["report_sha256"] = hashlib.sha256((output / "consumer-probe.json").read_bytes()).hexdigest()
        save()
    idle("after")
    status.update(state="completed", stage="completed", returncode=0,
                  performance_measured=True, finished_unix=time.time())
except BaseException:
    status.update(state="failed", returncode=1, error=traceback.format_exc(), finished_unix=time.time())
    traceback.print_exc()
save()
raise SystemExit(status["returncode"])
