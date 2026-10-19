# SPDX-License-Identifier: Apache-2.0
"""Separate MM2 L1 transfers from safe L0 compute tiles in an isolated copy.

The candidate combines two 32 KiB V slots into one 64 KiB slot. D256 can then
transfer 128 rows while computing the same ordered 64-row partial products.
No additional L1 storage, KV cache, graph capacity, or plan field is added.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

SOURCE = Path("csrc/attention/fused_infer_attention_score_v2_sink/op_kernel/fia_block_cube_nonquant_gqa_sink.h")
EXPECTED_SHA256 = "82779741be8ae3d1954af2758d657b5adeba89267ba9100138e8ff24f771362e"
MACRO = "THESIS_FIA_MM2_WIDE_V"

GUARD = """#if defined(THESIS_FIA_MM2_WIDE_V) && THESIS_FIA_MM2_WIDE_V && (__CCE_AICORE__ == 220)
    static constexpr bool THESIS_MM2_WIDE_V =
        IsSameType<Q_T, bfloat16_t>::value && IsSameType<KV_T, bfloat16_t>::value &&
        PAGE_ATTENTION && LAYOUT_T == FIA_LAYOUT::TND && PA_LAYOUT == FIA_LAYOUT::BSH &&
        !CFG::S2_BASICSIZE_IS_1024;
#else
    static constexpr bool THESIS_MM2_WIDE_V = false;
#endif
"""

SPLIT = """    // Preserve the safe L0B bound and reduction order; only batch L1 transfers.
    const uint32_t kL1Split = THESIS_MM2_WIDE_V && constInfo.headDim == 256U &&
                             constInfo.headDimRope == 0U && !info.isSinkTensor
        ? 128U : K_SPLIT_SIZE;
    auto KL1Slices = k.Split(kL1Split);"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("qualified source anchor changed")
    return text.replace(old, new, 1)


def transformed(before):
    if hashlib.sha256(before).hexdigest() != EXPECTED_SHA256:
        raise ValueError("expected the pinned, qualified D256 cube source")
    text = before.decode("utf-8")
    marker = "    // =================================L1 Buffer================================="
    text = replace_once(text, marker, GUARD + marker)
    text = replace_once(
        text,
        "    static constexpr uint32_t L1_SINK_V_SIZE = 128 * 128;",
        "    static constexpr uint32_t L1_SINK_V_SIZE = THESIS_MM2_WIDE_V ? 128 * 256 : 128 * 128;",
    )
    text = replace_once(
        text,
        "    static constexpr uint32_t L1_V_BUFCNT = CFG::S2_BASICSIZE_IS_1024 ? 4 : 2;",
        "    static constexpr uint32_t L1_V_BUFCNT = CFG::S2_BASICSIZE_IS_1024 ? 4 :\n"
        "        (THESIS_MM2_WIDE_V ? 1 : 2);\n"
        "    static_assert(L1_SINK_V_SIZE * L1_V_BUFCNT ==\n"
        "                  128 * 128 * (CFG::S2_BASICSIZE_IS_1024 ? 4 : 2));",
    )
    return replace_once(text, "    auto KL1Slices = k.Split(K_SPLIT_SIZE);", SPLIT).encode("utf-8")


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
        target="A2 BF16 TND paged ND128 Q4/GQA6/D256, S2<=512, no sink/RoPE",
        l1_v_bytes=dict(
            control_slots=2,
            control_bytes_per_slot=32768,
            candidate_slots=1,
            candidate_bytes_per_slot=65536,
            total=65536,
        ),
        l1_total_bytes=475136,
        l0_compute_k=64,
        l1_transfer_k=128,
        tradeoff="fewer transfer/lookup/event groups, less V buffering; no guaranteed latency reduction",
        unchanged="MM1, metadata plan, L0 compute order, softmax, workspace recurrence, graph capacities",
        fallback="other compile-time geometries retain two V slots; non-D256 tasks retain their safe K split",
        qualification="source only; restricted binary and device qualification required",
        graph_split_exclusive=False,
    )
    if not check:
        evidence = source / "mm2-transfer-patch.json"
        diff = source / "mm2-transfer-source.patch"
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
