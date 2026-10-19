# SPDX-License-Identifier: Apache-2.0
"""Execute address and compute-order contracts using native Axis and patch text.

This CPU harness does not qualify device synchronization or arbitrary shapes.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import aclgraph_fia_sink_mm2_double_buffer_patch as patch

HELPERS = {
    "fia_kernel_nonquant_sink.h": "785c0d14eff5874a019b570d54c5989c793e91381532e2ad6346cd98348ad041",
    "array.h": "cbce6327d5ecfb55fc8eac6df172c5dc71938202f55174dd4d308c54cc6d1c79",
    "memcopy/copy_gm_to_l1.h": "6293778b165cbeaf1a844bf7489d387abf560217ea7045abe6f1127e7e6117fc",
    "axis.h": "ea3d5fa888fc9162eb38a10dfc1be116d927f829a6ac07e4261ad8340e6645fa",
}

PREFIX = r"""
#include <cstdint>
#include <cassert>
#include <vector>
#include <cstdio>
#include <type_traits>
#define __aicore__
#define __CCE_AICORE__ 220
#define THESIS_FIA_MM2_DOUBLE_V_LAYOUT 1
#define THESIS_FIA_MM2_DOUBLE_V_WIDE 1
constexpr uint32_t BLOCK_CUBE = 16;
template <typename T> T Align(T v, T a) { return (v + a - 1) / a * a; }
struct bfloat16_t { uint16_t value; };
template <typename A, typename B> using IsSameType = std::is_same<A, B>;
enum class FIA_LAYOUT { TND, BSH, NZ };
template <typename T> struct LocalTensor {
    uint32_t offset = 0;
    LocalTensor operator[](uint32_t i) const { return {offset + i * sizeof(T)}; }
};
struct Buffer {
    uint32_t offset = 0, bytes = 0;
    template <typename T> LocalTensor<T> Get() const { return {offset}; }
};
struct Pipe {
    uint32_t used = 0;
    void InitBuffer(Buffer &b, uint32_t bytes) { b = {used, bytes}; used += bytes; }
};
struct ConstInfo {
    uint32_t headDim = 256, headDimRope = 0, keySinkNumber = 0, gSize = 6;
    uint32_t kvHeadNum = 2, qHeadNum = 12, qSeqSize = 4, headDimAlign = 256;
    uint32_t preLoadNum = 2, kvCacheBlockSize = 128, subBlockNum = 2, s2BaseSize = 512;
    int sinkNumber = 0;
    bool ropeSplitMode = false;
};
template <bool S2 = false, uint32_t Preload = 2> struct Config {
    static constexpr bool S2_BASICSIZE_IS_1024 = S2;
    static constexpr uint32_t PRELOAD_NUM = Preload;
};
template <typename Q = bfloat16_t, typename KV = bfloat16_t, typename C = Config<>,
          bool Paged = true, FIA_LAYOUT Layout = FIA_LAYOUT::TND, FIA_LAYOUT PA = FIA_LAYOUT::BSH>
struct LayoutContract {
    using Q_T = Q; using KV_T = KV; using CFG = C;
    static constexpr bool PAGE_ATTENTION = Paged;
    static constexpr FIA_LAYOUT LAYOUT_T = Layout, PA_LAYOUT = PA;
    static constexpr uint32_t L1_Q_SIZE = CFG::S2_BASICSIZE_IS_1024 ? 128 * 192 : 256 * 192;
    static constexpr uint32_t L1_SINK_V_SIZE = 128 * 128, L1_Q_BUFCNT = 2;
    static constexpr uint32_t L1_V_BUFCNT = CFG::S2_BASICSIZE_IS_1024 ? 4 : 2;
    Buffer L1_Q, L1_V;
    LocalTensor<Q_T> qL1Tensor[L1_Q_BUFCNT];
    LocalTensor<KV_T> vL1Tensor[L1_V_BUFCNT];
"""

BODY = r"""
    }
};
std::vector<uint32_t> chunks(uint32_t length, bool enabled, bool sink, bool wide) {
    struct { bool isSinkTensor; } info{sink};
    const bool THESIS_DOUBLE_V_WIDE = wide, thesisDoubleVEnabled = enabled;
    const uint32_t K_SPLIT_SIZE = 64, K_BASE = 64;
    Axis k(length);
@SPLIT@
    std::vector<uint32_t> result;
    for (auto &kL1 : KL1Slices) {
        assert(kL1.AlignedSize() * 256 <= (enabled ? 32768U : 16384U));
        assert(kL1.AlignedSize() * 64 <= 24576U);
        for (auto &kL0 : kL1.Split(K_BASE)) {
            assert(kL0.AlignedSize() * 256 <= 16384U);
            assert(kL0.start + kL0.sizeAct <= kL1.sizeAct);
            result.push_back(kL1.start + kL0.start);
            result.push_back(kL0.sizeAct);
            result.push_back((k.sizeAct - kL1.start - kL0.start) <= K_BASE);
            result.push_back(kL1.start == 0 && kL0.start == 0);
        }
    }
    return result;
}
template <typename L> void checkFallback() {
    L l; Pipe p; l.allocate(ConstInfo{}, &p);
    assert(!l.thesisDoubleVEnabled);
    assert(l.qL1Tensor[1].offset == L::L1_Q_SIZE * sizeof(typename L::Q_T));
    assert(l.vL1Tensor[1].offset == l.L1_V.offset + L::L1_SINK_V_SIZE * sizeof(typename L::KV_T));
}
int main() {
    using L = LayoutContract<>;
    for (uint32_t mutation = 0; mutation < 16; ++mutation) {
        ConstInfo c;
        switch (mutation) {
        case 1: c.headDim = 128; break;
        case 2: c.headDimRope = 64; break;
        case 3: c.ropeSplitMode = true; break;
        case 4: c.keySinkNumber = 1; break;
        case 5: c.gSize = 4; break;
        case 6: c.kvHeadNum = 1; break;
        case 7: c.qHeadNum = 16; break;
        case 8: c.qSeqSize = 5; break;
        case 9: c.headDimAlign = 272; break;
        case 10: c.preLoadNum = 1; break;
        case 11: c.kvCacheBlockSize = 64; break;
        case 12: c.subBlockNum = 1; break;
        case 13: c.sinkNumber = 1; break;
        case 14: c.s2BaseSize = 0; break;
        case 15: c.s2BaseSize = 513; break;
        }
        L l; Pipe p; l.allocate(c, &p);
        const bool enabled = mutation == 0;
        assert(l.thesisDoubleVEnabled == enabled);
        const uint32_t qs = enabled ? 65536U : 98304U, vs = enabled ? 65536U : 32768U;
        assert(l.L1_Q.bytes == qs * 2 && l.L1_V.bytes == vs * 2);
        assert(l.L1_V.offset == l.L1_Q.offset + qs * 2);
        for (uint32_t i = 0; i < 2; ++i) {
            assert(l.qL1Tensor[i].offset == i * qs);
            assert(l.vL1Tensor[i].offset == l.L1_V.offset + i * vs);
        }
        // The unmodified three KP slots and two K-sink slots follow Q/V.
        assert(p.used + (3 * 24576 + 2 * 16384) * 2 == 464 * 1024);
    }
    // Execute the actual compile-time gate with each predicate independently false.
    checkFallback<LayoutContract<float>>();
    checkFallback<LayoutContract<bfloat16_t, float>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<true>>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<false, 1>>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<>, false>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<>, true, FIA_LAYOUT::BSH>>();
    checkFallback<LayoutContract<bfloat16_t, bfloat16_t, Config<>, true, FIA_LAYOUT::TND, FIA_LAYOUT::NZ>>();
    // Actual Q NZ storage, including padding, both 128-column Load3D products and both slots.
    for (uint32_t actualM = 1; actualM <= 128; ++actualM) {
        const uint32_t alignedM = Align(actualM, 16U);
        L l; Pipe p; l.allocate(ConstInfo{}, &p);
        for (uint32_t slot = 0; slot < 2; ++slot) {
            const uint32_t begin = l.qL1Tensor[slot].offset;
            for (uint32_t k = 0; k < 256; ++k) {
                for (uint32_t m = 0; m < alignedM; ++m) {
                    const uint32_t nz = (k / 16) * alignedM * 16 + m * 16 + k % 16;
                    assert(nz < 32768);
                    assert(begin + (nz + 1) * 2 <= begin + 65536);
                }
            }
        }
    }
    for (uint32_t length = 0; length <= 512; ++length) {
        for (bool sink : {false, true}) {
            const auto original = chunks(length, false, sink, false);
            assert(original == chunks(length, true, sink, false));
            assert(original == chunks(length, true, sink, true));
            assert(original == chunks(length, false, sink, true));
            assert(original.size() == 4 * ((length + 63) / 64));
            uint32_t covered = 0;
            for (uint32_t i = 0; i < original.size(); i += 4) {
                assert(original[i] == covered);
                covered += original[i + 1];
                assert(original[i + 2] == (covered == length));
                assert(original[i + 3] == (i == 0));
            }
            assert(covered == length);
        }
    }
    std::puts("16 runtime layouts, 7 static fallbacks, 128 Q NZ tails x 2 slots, 1026 ordered K contracts passed");
}
"""


def run(native, output):
    native, output = Path(native).resolve(strict=True), Path(output).resolve(strict=True)
    for name, expected in HELPERS.items():
        data = (native / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("native helper drift: " + name)
        (output / (name.replace("/", "_") + ".txt")).write_bytes(data)
    before = (native / patch.SOURCE.name).read_bytes()
    candidate = patch.transformed(before)
    (output / "cube-before.h.txt").write_bytes(before)
    (output / "cube-candidate.h.txt").write_bytes(candidate)
    axis = (native / "axis.h").read_text().replace('#include "kernel_operator.h"', "")
    program = (
        PREFIX
        + patch.GUARD
        + "void allocate(const ConstInfo &constInfo, Pipe *pipe) {\n"
        + patch.Q_ALLOCATION
        + patch.V_ALLOCATION
        + BODY.replace("@SPLIT@", patch.SPLIT)
    )
    source, binary = output / "cpu-double-buffer-contract.cpp", output / "cpu-double-buffer-contract"
    source.write_text(program.replace("struct bfloat16_t", axis + "\nstruct bfloat16_t", 1), encoding="utf-8")
    with (output / "cpu-double-buffer-contract.log").open("xb") as log:
        subprocess.run(
            ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", str(source), "-o", str(binary)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=60,
        )
        subprocess.run([str(binary)], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=20)
    result = dict(
        passed=True,
        native_helpers=HELPERS,
        native_axis=True,
        actual_patch_fragments=True,
        runtime_layout_cases=16,
        static_fallback_cases=7,
        q_nz_tail_cases=128,
        q_slots=2,
        ordered_k_cases=1026,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="CPU address, layout and ordered products only; device synchronization remains unqualified",
    )
    (output / "cpu-double-buffer-contract.json").write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.native, args.output), indent=2))


if __name__ == "__main__":
    main()
