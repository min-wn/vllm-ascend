# SPDX-License-Identifier: Apache-2.0
"""Patch an isolated stage-two source copy with an opt-in whole-row producer."""

import argparse
import hashlib
import json
from pathlib import Path

METADATA = Path(
    "csrc/attention/fused_infer_attention_score_v2_sink_metadata/op_kernel_aicpu/fused_infer_attention_score_v2_sink_metadata_aicpu.cpp"
)

EXPECTED_STAGE2_SHA256 = "a3e4bbeeca240d55d09f8624efe0a83fc5e77c1f692d72f73ee55a0872e9b51d"

HELPER = r"""
#ifdef THESIS_NO_FD_WHOLE_ROW
// This consumer needs canonical whole-head endpoints, not an FD plan with
// reduction counts erased. Publish a fully fresh result only after validation.
static bool BuildWholeRowPlan(const BaseInfo &baseInfo, uint32_t coreNum,
                              const CostInfo &costInfo, SplitResult &result)
{
    if (baseInfo.batchInvariant || !baseInfo.isS1G || baseInfo.sparseMode != 3 ||
        baseInfo.n2Size != 2U || baseInfo.gSize != 6U || baseInfo.s1Size != 4U ||
        baseInfo.bSize == 0U || baseInfo.bSize > 128U || coreNum == 0U || coreNum > 20U ||
        result.vecCubeRatio != 2U || !baseInfo.isAccumSeqS1 || baseInfo.isAccumSeqS2 ||
        baseInfo.actualLenQDims != baseInfo.bSize || baseInfo.actualLenKvDims != baseInfo.bSize ||
        baseInfo.actualSeqS1Size.size() != baseInfo.bSize ||
        baseInfo.actualSeqS2Size.size() != baseInfo.bSize ||
        baseInfo.sinkNumber != 0 || baseInfo.keySinkNumber != 0 ||
        costInfo.bN2CostOfEachBatch.size() != baseInfo.bSize) {
        return false;
    }
    const uint32_t totalHeads = baseInfo.bSize * baseInfo.n2Size;
    std::vector<int64_t> costs(totalHeads, 0);
    std::vector<uint32_t> positive;
    positive.reserve(totalHeads);
    for (uint32_t batch = 0; batch < baseInfo.bSize; ++batch) {
        if (baseInfo.actualSeqS1Size[batch] != static_cast<int64_t>((batch + 1U) * 4U) ||
            baseInfo.actualSeqS2Size[batch] < 0 || baseInfo.actualSeqS2Size[batch] > 4097) {
            return false;
        }
        const int64_t cost = costInfo.bN2CostOfEachBatch[batch];
        if (cost < 0 || ((baseInfo.actualSeqS2Size[batch] == 0) != (cost == 0))) {
            return false;
        }
        for (uint32_t head = 0; head < baseInfo.n2Size; ++head) {
            const uint32_t index = batch * baseInfo.n2Size + head;
            costs[index] = cost;
            if (cost > 0) {
                positive.push_back(index);
            }
        }
    }
    const uint32_t used = positive.empty() ? 1U : std::min(coreNum, static_cast<uint32_t>(positive.size()));
    std::vector<uint32_t> ends;
    ends.reserve(used);
    if (used == 1U) {
        ends.push_back(totalHeads);
    } else {
        int64_t lower = 0;
        int64_t upper = 0;
        for (const auto cost : costs) {
            lower = std::max(lower, cost);
            upper += cost;
        }
        lower = std::max(lower, (upper + used - 1) / used);
        auto groupsNeeded = [&](int64_t limit) {
            uint32_t groups = 1U;
            int64_t current = 0;
            for (const auto index : positive) {
                if (current + costs[index] > limit) {
                    ++groups;
                    current = 0;
                }
                current += costs[index];
            }
            return groups;
        };
        while (lower < upper) {
            const int64_t middle = lower + (upper - lower) / 2;
            if (groupsNeeded(middle) <= used) {
                upper = middle;
            } else {
                lower = middle + 1;
            }
        }
        int64_t current = 0;
        for (uint32_t position = 0; position < positive.size(); ++position) {
            const uint32_t index = positive[position];
            const uint32_t groupsLeft = used - static_cast<uint32_t>(ends.size());
            const uint32_t headsLeft = static_cast<uint32_t>(positive.size()) - position;
            if (current > 0 && (current + costs[index] > lower || headsLeft == groupsLeft - 1U)) {
                ends.push_back(index);
                current = 0;
            }
            current += costs[index];
        }
        ends.push_back(totalHeads);
    }
    if (ends.size() != used) {
        return false;
    }
    SplitResult candidate{coreNum, result.vecCubeRatio};
    candidate.usedCoreNum = used;
    uint32_t start = 0U;
    for (uint32_t core = 0; core < used; ++core) {
        if (ends[core] <= start || ends[core] > totalHeads) {
            return false;
        }
        candidate.bN2End[core] = ends[core];
        int64_t load = 0;
        for (uint32_t index = start; index < ends[core]; ++index) {
            load += costs[index];
        }
        if (!positive.empty() && load == 0) {
            return false;
        }
        candidate.maxCost = std::max(candidate.maxCost, load);
        start = ends[core];
    }
    if (start != totalHeads) {
        return false;
    }
    // Fresh vectors retain zero GS1/S2 endpoints and every FD/merge field.
    result = candidate;
    return true;
}
#endif
"""

COMPUTE_POLICY = r"""#ifdef THESIS_NO_FD_WHOLE_ROW
    bool wholeRowBuilt = false;
    if (!isMlaSink && inputLayout_ == "TND" && inputLayoutKv_ == "BnBsH" &&
        numHeadsQ_ == 12U && numHeadsKv_ == 2U && headDimQK_ == 256U && headDimV_ == 256U &&
        ropeHeadDim_ == 0U && blockSize_ == 128 && sparseMode_ == 3U && !batchInvariant_ &&
        sinkNum_ == 0 && kSinkNum_ == 0 && bSize > 0U && bSize <= 128U &&
        s1Size == 4U && s2Size > 0 && s2Size <= 4097 &&
        actualLenQDims == bSize && actualLenKvDims == bSize &&
        aicCoreNum_ > 0U && aicCoreNum_ <= 20U && aivCoreNum_ == aicCoreNum_ * 2U &&
        mBaseSize_ == 512U && mFdBaseSize_ == 8U && sInnerSize_ > 0U && sInnerSize_ <= 512U) {
        SplitContext wholeContext(baseInfo, splitParam);
        CalcSplitInfo(wholeContext);
        CalcCostInfo(wholeContext);
        wholeRowBuilt = BuildWholeRowPlan(baseInfo, aicCoreNum_, wholeContext.costInfo, res);
    }
    if (!wholeRowBuilt) {
        SplitCore(aicCoreNum_, baseInfo, splitParam, res);
    }
#else
    SplitCore(aicCoreNum_, baseInfo, splitParam, res);
#endif"""


def transformed(original):
    if hashlib.sha256(original).hexdigest() != EXPECTED_STAGE2_SHA256:
        raise ValueError("metadata source differs from the pinned stage-two include-corrected source")
    text = original.decode("utf-8")
    marker = "namespace aicpu {\nnamespace {"
    if text.count(marker) != 1 or "THESIS_NO_FD_WHOLE_ROW" in text:
        raise ValueError("unexpected or already patched metadata source")
    text = text.replace(marker, marker + "\n" + HELPER, 1)
    old = "    SplitCore(aicCoreNum_, baseInfo, splitParam, res);"
    if text.count(old) != 1:
        raise ValueError("unexpected SplitCore call count")
    return text.replace(old, COMPUTE_POLICY, 1).encode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    path = (source / METADATA).resolve(strict=True)
    if not path.is_relative_to(source):
        raise ValueError("metadata source escapes the isolated source tree")
    original = path.read_bytes()
    patched = transformed(original)
    manifest = {
        "source_file": str(path.relative_to(source)),
        "before_sha256": hashlib.sha256(original).hexdigest(),
        "after_sha256": hashlib.sha256(patched).hexdigest(),
        "macro": "THESIS_NO_FD_WHOLE_ROW",
        "scope": (
            "Q4/TND/BnBsH/qualified-BF16-consumer/Q12-KV2/D256/page128/sparse3/no-sink-no-RoPE; "
            "native whole-head costs and fresh transactional plan; unsupported cases use upstream"
        ),
    }
    if not args.check:
        path.write_bytes(patched)
        (source / "stage5-no-fd-patch.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="ascii")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
