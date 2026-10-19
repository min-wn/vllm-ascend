# SPDX-License-Identifier: Apache-2.0
"""Apply a pinned, capacity-checked D256 sink prototype to an isolated tree.

This is a research port for BF16, TND, Q12/KV2 and paged ND128, without
RoPE/sink tensors. It does not qualify dynamic graphs or establish a speedup.
The native files retain their upstream CANN license. No installed OPP is edited.
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path

COMMIT = "bec19b8dea1e538e64838f3064f7e91a3fa9ee72"
ATTENTION = "csrc/attention/fused_infer_attention_score_v2_sink/"
HOST = ATTENTION + "op_host/fused_infer_attention_score_v2_sink_tiling_v3.cpp"
CUBE = ATTENTION + "op_kernel/fia_block_cube_nonquant_gqa_sink.h"
METADATA = (
    "csrc/attention/fused_infer_attention_score_v2_sink_metadata/"
    "op_kernel_aicpu/fused_infer_attention_score_v2_sink_metadata_aicpu.cpp"
)
HASHES = {
    HOST: "a8c628b253010b14e5a7f855dc964b40fdc5a25c85ef3d64581ae70f4001fa06",
    CUBE: "600b15942b202996c480a5b978eb56d7045591aeb40012f6e0463d475964ac06",
    METADATA: "bffd08092e0c431674f1c361681be58315a714d517d80136a18edb1faf1bbeec",
}

HOST_HELPER = """// Research prototype: only the audited BF16/TND/paged-ND128 geometry.
// Dynamic exact query lengths remain device inputs; inner M loops are bounded
// independently of total packed T and do not assume a single request.
bool CheckD256PrototypeSupport(gert::TilingContext *context)
{
    const auto *attrs = context->GetAttrs();
    if (std::string(attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX)) != "TND" ||
        *attrs->GetAttrPointer<uint32_t>(ATTR_N_INDEX) != 12 ||
        *attrs->GetAttrPointer<uint32_t>(ATTR_NUM_KV_HEADS_INDEX) != 2 ||
        *attrs->GetAttrPointer<uint32_t>(ATTR_BLOCK_SIZE_INDEX) != 128 ||
        *attrs->GetAttrPointer<int64_t>(ATTR_SINK_NUMBER_INDEX) != 0 ||
        context->GetOptionalInputShape(BLOCK_TABLE_INDEX) == nullptr) {
        return false;
    }
    for (const auto index : {QUERY_INDEX, KEY_INDEX, VALUE_INDEX}) {
        const auto *desc = context->GetInputDesc(index);
        const auto *shape = context->GetInputShape(index);
        if (desc == nullptr || desc->GetDataType() != ge::DT_BF16 ||
            shape == nullptr || shape->GetStorageShape().GetDimNum() != 3) {
            return false;
        }
    }
    for (const auto index : {QUERY_ROPE_INDEX, KEY_ROPE_INDEX, LEARNABLE_SINK_INDEX,
                             KEY_SINK_INDEX, KEY_ROPE_SINK_INDEX, VALUE_SINK_INDEX}) {
        if (context->GetOptionalInputShape(index) != nullptr) {
            return false;
        }
    }
    const auto &q = context->GetInputShape(QUERY_INDEX)->GetStorageShape();
    const auto &k = context->GetInputShape(KEY_INDEX)->GetStorageShape();
    const auto &v = context->GetInputShape(VALUE_INDEX)->GetStorageShape();
    return q.GetDim(1) == 12 && q.GetDim(2) == 256 &&
           k.GetDim(1) == 128 && k.GetDim(2) == 512 &&
           v.GetDim(1) == 128 && v.GetDim(2) == 512;
}

"""

CAPACITY_CONSTANTS = """    // D256 keeps the existing L1/flag layout. Retile rather than widening
    // each KP slot to 64 KiB (which would consume all 512 KiB of default L1).
    static constexpr uint32_t D256_HEAD_DIM = 256;
    static constexpr uint32_t D256_MM1_QUERY_ROWS = CFG::S2_BASICSIZE_IS_1024 ? 64 : 128;
    static constexpr uint32_t D256_MM1_KEY_ROWS = 64;
    static constexpr uint32_t D256_MM2_QUERY_ROWS = 64;
    static constexpr uint32_t D256_MM2_KEY_ROWS = 64;
    static_assert(D256_MM1_QUERY_ROWS * D256_HEAD_DIM <= L1_Q_SIZE);
    static_assert(D256_MM1_KEY_ROWS * D256_HEAD_DIM <= L1_SINK_KP_SIZE);
    static_assert(D256_MM1_KEY_ROWS * D256_HEAD_DIM <= L1_K_SINK_SIZE);
    static_assert(128 * 128 <= L0A_PP_SIZE);
    static_assert(D256_MM1_KEY_ROWS * 128 <= L0B_PP_SIZE);
    static_assert(128 * D256_MM1_KEY_ROWS <= L0C_PP_SIZE);
    static_assert(D256_MM2_QUERY_ROWS * D256_MM2_KEY_ROWS <= L1_SINK_KP_SIZE);
    static_assert(D256_MM2_QUERY_ROWS * D256_MM2_KEY_ROWS <= L0A_PP_SIZE);
    static_assert(D256_MM2_KEY_ROWS * D256_HEAD_DIM <= L1_SINK_V_SIZE);
    static_assert(D256_MM2_KEY_ROWS * D256_HEAD_DIM <= L0B_PP_SIZE);
    static_assert(D256_MM2_QUERY_ROWS * D256_HEAD_DIM <= L0C_PP_SIZE);

"""


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"pinned context changed: {old[:100]!r}")
    return text.replace(old, new, 1)


def transformed(relative, text):
    if relative == METADATA:
        return replace_once(
            text, '#include "../../common/aicpu/cpu_context_util.h"', '#include "../aicpu/cpu_context_util.h"'
        )
    if relative == HOST:
        text = replace_once(
            text,
            "bool CheckGqaDSupport(gert::TilingContext *context)",
            HOST_HELPER + "bool CheckGqaDSupport(gert::TilingContext *context)",
        )
        marker = "    // D的组合(128+0,128)(64+0,64)(128+64,128)(192+0,128)(192+0,192)"
        return replace_once(
            text,
            marker,
            "    if (queryD == 256 && queryRopeD == 0 && valueD == 256) {\n"
            "        return CheckD256PrototypeSupport(context);\n    }\n\n" + marker,
        )
    text = replace_once(
        text,
        "    Array<LocalTensor<Q_T>, L0AB_BUFCNT, L0A_PP_SIZE> aL0Tensor;",
        CAPACITY_CONSTANTS + "    Array<LocalTensor<Q_T>, L0AB_BUFCNT, L0A_PP_SIZE> aL0Tensor;",
    )
    text = replace_once(
        text,
        "    const uint32_t N_SPLIT_SIZE = 128;",
        "    const bool isD256 = constInfo.headDim == D256_HEAD_DIM;\n"
        "    const uint32_t mSplit = isD256 ? D256_MM1_QUERY_ROWS : M_SPLIT_SIZE;\n"
        "    const uint32_t N_SPLIT_SIZE = isD256 ? D256_MM1_KEY_ROWS : 128;",
    )
    text = replace_once(text, "    auto mSlices = m.Split(M_SPLIT_SIZE);", "    auto mSlices = m.Split(mSplit);")
    text = replace_once(
        text,
        "    constexpr uint32_t M_SPLIT_SIZE = 128;\n    // L0B",
        "    constexpr uint32_t M_SPLIT_SIZE = 128;\n"
        "    const uint32_t mSplit = constInfo.headDim == D256_HEAD_DIM\n"
        "        ? D256_MM2_QUERY_ROWS : M_SPLIT_SIZE;\n    // L0B",
    )
    return replace_once(
        text, "    for (auto &mL1 : m.Split(M_SPLIT_SIZE)) {", "    for (auto &mL1 : m.Split(mSplit)) {"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Fresh pinned source tree containing csrc/")
    parser.add_argument("--check", action="store_true", help="Verify all hashes/contexts without writing")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    changes, patches = [], []
    for relative, expected in HASHES.items():
        path = (source / relative).resolve(strict=True)
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f"source path escapes isolated tree: {path}")
        before = path.read_bytes()
        if hashlib.sha256(before).hexdigest() != expected:
            raise ValueError(f"source hash differs from pinned commit: {relative}")
        after = transformed(relative, before.decode("utf-8")).encode("utf-8")
        changes.append((path, before, after))
        patches.append(
            "".join(
                difflib.unified_diff(
                    before.decode().splitlines(True),
                    after.decode().splitlines(True),
                    fromfile="a/" + relative,
                    tofile="b/" + relative,
                )
            )
        )
    manifest = {
        "source_commit": COMMIT,
        "scope": "isolated D256 prototype; not runtime qualification",
        "files": [
            {
                "file": str(path.relative_to(source)),
                "before_sha256": hashlib.sha256(before).hexdigest(),
                "after_sha256": hashlib.sha256(after).hexdigest(),
            }
            for path, before, after in changes
        ],
    }
    if not args.check:
        artifact = source.parent / "d256-patch-manifest.json"
        diff = source.parent / "d256-source.patch"
        if artifact.exists() or diff.exists():
            raise RuntimeError("patch evidence already exists; do not overwrite a previous attempt")
        artifact.write_text(json.dumps(manifest, indent=2) + "\n")
        diff.write_text("".join(patches))
        for path, _, after in changes:
            path.write_bytes(after)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
