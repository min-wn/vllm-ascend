# SPDX-License-Identifier: Apache-2.0
"""Reallocate qualified D256 Q/V slots without dropping either V buffer.

Runtime geometry selects the complete layout before allocating any L1 region.
Other geometries retain the original strides, allocations and MM2 K split.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

import aclgraph_fia_sink_mm2_transfer_patch as original

SOURCE = original.SOURCE
EXPECTED_SHA256 = original.EXPECTED_SHA256
MACRO = "THESIS_FIA_MM2_DOUBLE_V_WIDE"

GUARD = """#if defined(THESIS_FIA_MM2_DOUBLE_V_LAYOUT) && THESIS_FIA_MM2_DOUBLE_V_LAYOUT && (__CCE_AICORE__ == 220)
    static constexpr bool THESIS_DOUBLE_V_STATIC =
        IsSameType<Q_T, bfloat16_t>::value && IsSameType<KV_T, bfloat16_t>::value &&
        PAGE_ATTENTION && LAYOUT_T == FIA_LAYOUT::TND && PA_LAYOUT == FIA_LAYOUT::BSH &&
        !CFG::S2_BASICSIZE_IS_1024 && CFG::PRELOAD_NUM == 2;
#else
    static constexpr bool THESIS_DOUBLE_V_STATIC = false;
#endif
#if defined(THESIS_FIA_MM2_DOUBLE_V_WIDE) && THESIS_FIA_MM2_DOUBLE_V_WIDE
    static constexpr bool THESIS_DOUBLE_V_WIDE = true;
#else
    static constexpr bool THESIS_DOUBLE_V_WIDE = false;
#endif
    bool thesisDoubleVEnabled = false;
    static constexpr uint32_t THESIS_DOUBLE_SLOT_ELEMENTS = 128 * 256;
    static_assert(2 * THESIS_DOUBLE_SLOT_ELEMENTS + 2 * THESIS_DOUBLE_SLOT_ELEMENTS ==
                  2 * (256 * 192) + 2 * (128 * 128));
"""

Q_ALLOCATION = """    // Select the whole layout before allocation; unsupported geometry keeps its old bounds.
    thesisDoubleVEnabled = THESIS_DOUBLE_V_STATIC && constInfo.headDim == 256U &&
        constInfo.headDimRope == 0U && !constInfo.ropeSplitMode && constInfo.keySinkNumber == 0U &&
        constInfo.gSize == 6U && constInfo.kvHeadNum == 2U && constInfo.qHeadNum == 12U &&
        constInfo.qSeqSize == 4U && constInfo.headDimAlign == 256U && constInfo.preLoadNum == 2U &&
        constInfo.kvCacheBlockSize == 128U && constInfo.subBlockNum == 2U && constInfo.sinkNumber == 0 &&
        constInfo.s2BaseSize > 0U && constInfo.s2BaseSize <= 512U;
    const uint32_t qL1Stride = thesisDoubleVEnabled ? THESIS_DOUBLE_SLOT_ELEMENTS : L1_Q_SIZE;
    pipe->InitBuffer(L1_Q, qL1Stride * L1_Q_BUFCNT * sizeof(Q_T));
    for (uint32_t i = 0; i < L1_Q_BUFCNT; ++i) {
        qL1Tensor[i] = L1_Q.Get<Q_T>()[i * qL1Stride];
    }"""

V_ALLOCATION = """    const uint32_t vL1Stride = thesisDoubleVEnabled ? THESIS_DOUBLE_SLOT_ELEMENTS : L1_SINK_V_SIZE;
    pipe->InitBuffer(L1_V, vL1Stride * L1_V_BUFCNT * sizeof(KV_T));
    for (uint32_t i = 0; i < L1_V_BUFCNT; ++i) {
        vL1Tensor[i] = L1_V.Get<KV_T>()[i * vL1Stride];
    }"""

SPLIT = """    // L0B still consumes the same ordered 64-row D256 products.
    const uint32_t kL1Split = THESIS_DOUBLE_V_WIDE && thesisDoubleVEnabled && !info.isSinkTensor
        ? 128U : K_SPLIT_SIZE;
    auto KL1Slices = k.Split(kL1Split);"""


def transformed(before):
    if hashlib.sha256(before).hexdigest() != EXPECTED_SHA256:
        raise ValueError("expected the pinned, qualified D256 cube source")
    text = before.decode("utf-8")
    replace = original.replace_once
    marker = "    // =================================L1 Buffer================================="
    text = replace(text, marker, GUARD + marker)
    text = replace(
        text, "Array<LocalTensor<Q_T>, L1_Q_BUFCNT, L1_Q_SIZE> qL1Tensor;", "LocalTensor<Q_T> qL1Tensor[L1_Q_BUFCNT];"
    )
    text = replace(
        text,
        "Array<LocalTensor<KV_T>, L1_V_BUFCNT, L1_SINK_V_SIZE> vL1Tensor;",
        "LocalTensor<KV_T> vL1Tensor[L1_V_BUFCNT];",
    )
    text = replace(
        text,
        "    pipe->InitBuffer(L1_Q, L1_Q_SIZE * L1_Q_BUFCNT * sizeof(Q_T));\n    qL1Tensor.Init(L1_Q.Get<Q_T>());",
        Q_ALLOCATION,
    )
    text = replace(
        text,
        "    pipe->InitBuffer(L1_V, L1_SINK_V_SIZE * L1_V_BUFCNT * sizeof(KV_T)); "
        "// L1_SINK_V_SIZE:128*128\n    vL1Tensor.Init(L1_V.Get<KV_T>());",
        V_ALLOCATION,
    )
    return replace(text, "    auto KL1Slices = k.Split(K_SPLIT_SIZE);", SPLIT).encode("utf-8")


def patch_tree(source, check=False):
    source = Path(source).resolve(strict=True)
    path = (source / SOURCE).resolve(strict=True)
    if not path.is_relative_to(source) or not path.is_file():
        raise ValueError("cube source escapes the isolated source tree")
    before = path.read_bytes()
    after = transformed(before)
    manifest = dict(
        source_file=SOURCE.as_posix(),
        before_sha256=hashlib.sha256(before).hexdigest(),
        after_sha256=hashlib.sha256(after).hexdigest(),
        macro=MACRO,
        macro_enabled_by_patch_tool=False,
        original_q_slot_bytes=98304,
        original_v_slot_bytes=32768,
        qualified_q_slot_bytes=65536,
        qualified_v_slot_bytes=65536,
        q_slots=2,
        v_slots=2,
        kp_slots=3,
        l1_total_bytes=475136,
        l0_compute_k=64,
        original_or_control_l1_transfer_k=64,
        candidate_l1_transfer_k=128,
        geometry_gate_before_allocation=True,
        fallback_uses_original_slot_strides=True,
        event_ids_and_slot_counts_unchanged=True,
        unchanged="MM1, Q copy/Load3D, compute loop, snapshots, event protocol, plan, softmax, graph capacities",
        qualification="source only; compile and device checks required; unsupported shapes are not runtime qualified",
        graph_split_exclusive=False,
    )
    if not check:
        evidence = source / "mm2-double-buffer-patch.json"
        diff = source / "mm2-double-buffer-source.patch"
        if evidence.exists() or diff.exists():
            raise ValueError("refusing to overwrite existing patch evidence")
        evidence.write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
        diff.write_text(
            "".join(
                difflib.unified_diff(
                    before.decode().splitlines(True),
                    after.decode().splitlines(True),
                    fromfile="a/" + SOURCE.as_posix(),
                    tofile="b/" + SOURCE.as_posix(),
                )
            ),
            encoding="utf-8",
        )
        path.write_bytes(after)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(json.dumps(patch_tree(args.source, args.check), indent=2))


if __name__ == "__main__":
    main()
