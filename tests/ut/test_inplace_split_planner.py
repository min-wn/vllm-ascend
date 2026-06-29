import numpy as np
import pytest

from vllm_ascend.worker.ubatch_utils import (
    inplace_split_preserves_attention_backend,
    INPLACE_SPLIT_DRY_RUN,
    MIXED_REQUEST_SPLIT_DRY_RUN,
    NO_SPLIT_ABOVE_MAX_CAPTURE_SIZE,
    NO_SPLIT_EXACT_GRAPH_HIT,
    NO_SPLIT_GRAPH_BUCKET_MISSING,
    NO_SPLIT_INPLACE_PADDING_SAVING_TOO_SMALL,
    NO_SPLIT_INVALID_FIRST_TOKENS_POLICY,
    NO_SPLIT_INVALID_MIXED_REQUEST_SPLIT_POLICY,
    NO_SPLIT_MACRO_GRAPH_MISS,
    NO_SPLIT_MIN_TOKENS_PER_SPLIT,
    NO_SPLIT_MIXED_DECODE_ONLY,
    NO_SPLIT_MIXED_BELOW_MIN_TOTAL_TOKENS,
    NO_SPLIT_NO_LOWER_CAPTURE_SIZE,
    NO_SPLIT_NO_OFFSET_CAPTURE_SIZE,
    NO_SPLIT_OFFSET_BUCKET_TOO_SMALL,
    NO_SPLIT_OFFSET_GRAPH_BELOW_MIN_SIZE,
    NO_SPLIT_OFFSET_GRAPH_EXCEEDS_START_CAP,
    NO_SPLIT_OFFSET_GRAPH_EXCEEDS_PADDED_BATCH,
    NO_SPLIT_OFFSET_PADDING_TOO_LARGE,
    NO_SPLIT_PADDING_TOO_LARGE,
    NO_SPLIT_REMAINDER_TOO_LARGE,
    NO_SPLIT_SINGLE_REQUEST_DOMINATES,
    create_inplace_split_batch_slices,
    create_macro_inplace_split_batch_slices,
    create_mixed_request_split_batch_slices,
)
from vllm_ascend.ascend_config import SplitBatchConfig


def _tokens(num_reqs: int, query_len: int = 1) -> np.ndarray:
    return np.full(num_reqs, query_len, dtype=np.int32)


def _macro_graph_config(**overrides):
    config = {
        "enabled": True,
        "mode": "inplace_parallel",
        "num_splits": 2,
        "enable_parallel_streams": True,
        "enable_inplace_lazy_capture": False,
        "inplace_split_planner_policy": "macro_cube_balanced",
        "macro_graph_config": {
            "enabled": True,
            "plan_source": "explicit",
            "capture_plans": [{
                "total_tokens": 427,
                "split_actual_tokens": [224, 203],
                "split_graph_tokens": [224, 224],
            }],
        },
    }
    macro_graph_config = config["macro_graph_config"]
    macro_graph_config.update(overrides)
    return SplitBatchConfig(config).macro_graph_config


def test_macro_inplace_split_uses_explicit_capture_plan():
    plan, reason = create_macro_inplace_split_batch_slices(
        _tokens(427),
        total_num_tokens=427,
        uniform_decode_query_len=1,
        macro_graph_config=_macro_graph_config(),
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens_policy == "macro_cube_balanced"
    assert plan.first_tokens == 224
    assert plan.second_actual_tokens == 203
    assert plan.second_graph_tokens == 224
    assert plan.second_padding_tokens == 21
    assert plan.split_slices[0].token_slice == slice(0, 224)
    assert plan.split_slices[0].graph_num_tokens == 224
    assert plan.split_slices[1].token_slice == slice(224, 427)
    assert plan.split_slices[1].graph_num_tokens == 224
    assert plan.split_slices[1].start_num_tokens == 224


def test_macro_compact_split_uses_explicit_request_counts():
    plan, reason = create_macro_inplace_split_batch_slices(
        _tokens(68),
        total_num_tokens=68,
        uniform_decode_query_len=1,
        macro_graph_config=_macro_graph_config(capture_plans=[{
            "total_tokens": 68,
            "split_actual_tokens": [36, 32],
            "split_graph_tokens": [36, 32],
            "split_start_tokens": [0, 0],
            "split_num_reqs": [5, 1],
        }]),
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.offset_match_policy == "compact"
    assert plan.first_reqs == 5
    assert plan.second_reqs == 1
    assert plan.split_slices[0].request_slice == slice(0, 5)
    assert plan.split_slices[0].token_slice == slice(0, 36)
    assert plan.split_slices[0].start_num_tokens == 0
    assert plan.split_slices[1].request_slice == slice(5, 6)
    assert plan.split_slices[1].token_slice == slice(36, 68)
    assert plan.split_slices[1].start_num_tokens == 0


def test_macro_inplace_split_misses_unlisted_explicit_plan():
    plan, reason = create_macro_inplace_split_batch_slices(
        _tokens(428),
        total_num_tokens=428,
        uniform_decode_query_len=1,
        macro_graph_config=_macro_graph_config(),
    )

    assert plan is None
    assert reason == NO_SPLIT_MACRO_GRAPH_MISS


def test_macro_inplace_split_planner_generates_cube_balanced_plan():
    plan, reason = create_macro_inplace_split_batch_slices(
        _tokens(427),
        total_num_tokens=427,
        uniform_decode_query_len=1,
        macro_graph_config=_macro_graph_config(
            plan_source="planner",
            capture_plans=[],
            capture_total_tokens=[427],
            graph_token_alignment=64,
            min_split_graph_tokens=192,
        ),
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 256
    assert plan.second_actual_tokens == 171
    assert plan.split_slices[0].graph_num_tokens == 256
    assert plan.split_slices[1].graph_num_tokens == 192
    assert plan.split_slices[1].start_num_tokens == 256


def test_inplace_split_uses_largest_lower_capture_size():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(416),
        total_num_tokens=416,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 384
    assert plan.second_tokens == 32
    assert plan.first_reqs == 384
    assert plan.second_reqs == 32
    assert plan.padded_num_tokens_without_split == 512
    assert len(plan.split_slices) == 2

    first, second = plan.split_slices
    assert first.request_slice == slice(0, 384)
    assert first.token_slice == slice(0, 384)
    assert first.padded_num_tokens == 384
    assert first.graph_num_tokens == 384
    assert first.start_num_tokens == 0
    assert second.request_slice == slice(384, 416)
    assert second.token_slice == slice(384, 416)
    assert second.padded_num_tokens == 32
    assert second.graph_num_tokens == 32
    assert second.start_num_tokens == 384
    assert plan.offset_match_policy == "exact"
    assert plan.first_tokens_policy == "largest_lower"
    assert plan.second_actual_tokens == 32
    assert plan.second_graph_tokens == 32
    assert plan.second_padding_tokens == 0


def test_inplace_split_balanced_policy_chooses_even_graph_workload():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(416),
        total_num_tokens=416,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256, 384, 512},
        offset_match_policy="bucket",
        offset_capture_sizes={32, 256},
        first_tokens_policy="balanced",
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens_policy == "balanced"
    assert plan.first_tokens == 256
    assert plan.second_actual_tokens == 160
    assert plan.second_graph_tokens == 256
    assert plan.second_padding_tokens == 96
    assert plan.split_slices[1].start_num_tokens == 256


def test_inplace_split_balanced_policy_respects_allowed_offset_graphs():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(385),
        total_num_tokens=385,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256, 384, 512},
        offset_match_policy="bucket",
        offset_capture_sizes={32, 64, 128, 256},
        offset_allowed_graph_tokens_by_start={
            256: [256],
            384: [32],
        },
        first_tokens_policy="balanced",
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 256
    assert plan.second_actual_tokens == 129
    assert plan.second_graph_tokens == 256
    assert plan.split_slices[1].start_num_tokens == 256


def test_inplace_split_rejects_invalid_first_tokens_policy():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(416),
        total_num_tokens=416,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
        first_tokens_policy="unknown",
    )

    assert plan is None
    assert reason == NO_SPLIT_INVALID_FIRST_TOKENS_POLICY


def test_inplace_split_handles_query_len_greater_than_one():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(208, query_len=2),
        total_num_tokens=416,
        uniform_decode_query_len=2,
        cudagraph_capture_sizes={256, 384, 512},
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 384
    assert plan.second_tokens == 32
    assert plan.first_reqs == 192
    assert plan.second_reqs == 16
    assert plan.split_slices[1].start_num_tokens == 384


def test_inplace_split_bucket_pads_second_split_graph_tokens():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(249),
        total_num_tokens=249,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={64, 128},
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 128
    assert plan.second_tokens == 121
    assert plan.second_actual_tokens == 121
    assert plan.second_graph_tokens == 128
    assert plan.second_padding_tokens == 7
    assert plan.offset_match_policy == "bucket"
    assert plan.offset_capture_sizes_considered == [64, 128]
    assert plan.padded_num_tokens_without_split == 256

    second = plan.split_slices[1]
    assert second.request_slice == slice(128, 249)
    assert second.token_slice == slice(128, 249)
    assert second.num_tokens == 121
    assert second.graph_num_tokens == 128
    assert second.start_num_tokens == 128


def test_inplace_split_respects_min_padding_saved_tokens():
    capture_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256}

    plan, reason = create_inplace_split_batch_slices(
        _tokens(130),
        total_num_tokens=130,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=capture_sizes,
        offset_match_policy="bucket",
        offset_capture_sizes={32, 64, 128},
        inplace_min_padding_saved_tokens=96,
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 128
    assert plan.second_actual_tokens == 2
    assert plan.second_graph_tokens == 32
    assert plan.debug_payload()["padding_saved_tokens"] == 96
    assert plan.debug_payload()["inplace_min_padding_saved_tokens"] == 96
    assert plan.debug_payload()["inplace_split_overhead_tokens"] == 0
    assert plan.debug_payload()["inplace_effective_padding_saved_tokens"] == 96
    assert plan.debug_payload()["inplace_required_padding_saved_tokens"] == 96

    plan, reason = create_inplace_split_batch_slices(
        _tokens(130),
        total_num_tokens=130,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=capture_sizes,
        offset_match_policy="bucket",
        offset_capture_sizes={32, 64, 128},
        inplace_min_padding_saved_tokens=97,
    )

    assert plan is None
    assert reason == NO_SPLIT_INPLACE_PADDING_SAVING_TOO_SMALL

    plan, reason = create_inplace_split_batch_slices(
        _tokens(130),
        total_num_tokens=130,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=capture_sizes,
        offset_match_policy="bucket",
        offset_capture_sizes={32, 64, 128},
        inplace_min_padding_saved_tokens=64,
        inplace_split_overhead_tokens=32,
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.debug_payload()["padding_saved_tokens"] == 96
    assert plan.debug_payload()["inplace_min_padding_saved_tokens"] == 64
    assert plan.debug_payload()["inplace_split_overhead_tokens"] == 32
    assert plan.debug_payload()["inplace_effective_padding_saved_tokens"] == 64
    assert plan.debug_payload()["inplace_required_padding_saved_tokens"] == 96

    plan, reason = create_inplace_split_batch_slices(
        _tokens(130),
        total_num_tokens=130,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=capture_sizes,
        offset_match_policy="bucket",
        offset_capture_sizes={32, 64, 128},
        inplace_min_padding_saved_tokens=64,
        inplace_split_overhead_tokens=33,
    )

    assert plan is None
    assert reason == NO_SPLIT_INPLACE_PADDING_SAVING_TOO_SMALL


def test_inplace_split_bucket_maps_adjacent_remainders_to_same_graph():
    plan_a, reason_a = create_inplace_split_batch_slices(
        _tokens(231),
        total_num_tokens=231,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={64, 128},
    )
    plan_b, reason_b = create_inplace_split_batch_slices(
        _tokens(249),
        total_num_tokens=249,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={64, 128},
    )

    assert reason_a == INPLACE_SPLIT_DRY_RUN
    assert reason_b == INPLACE_SPLIT_DRY_RUN
    assert plan_a is not None
    assert plan_b is not None
    assert plan_a.split_slices[1].start_num_tokens == 128
    assert plan_b.split_slices[1].start_num_tokens == 128
    assert plan_a.split_slices[1].graph_num_tokens == 128
    assert plan_b.split_slices[1].graph_num_tokens == 128
    assert plan_a.second_actual_tokens == 103
    assert plan_b.second_actual_tokens == 121


def test_inplace_split_bucket_uses_lower_candidate_when_start_cap_rejects():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(300),
        total_num_tokens=300,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={64, 128, 256, 384},
        offset_match_policy="bucket",
        offset_capture_sizes={64, 128, 256},
        offset_max_graph_tokens_by_start={256: 32},
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 128
    assert plan.second_actual_tokens == 172
    assert plan.second_graph_tokens == 256
    assert plan.split_slices[1].start_num_tokens == 128
    assert plan.offset_max_graph_tokens_by_start == {256: 32}


def test_inplace_split_bucket_rejects_when_all_candidates_exceed_start_cap():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(249),
        total_num_tokens=249,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="exact",
        offset_max_graph_tokens_by_start={128: 64},
    )

    assert plan is None
    assert reason == NO_SPLIT_OFFSET_GRAPH_EXCEEDS_START_CAP


def test_inplace_split_bucket_respects_min_offset_graph_tokens():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(65),
        total_num_tokens=65,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={64, 128},
        offset_match_policy="bucket",
        offset_capture_sizes={1, 2, 4, 8, 16, 32, 64},
        offset_min_graph_tokens=32,
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.second_actual_tokens == 1
    assert plan.second_graph_tokens == 32
    assert plan.offset_capture_sizes_considered == [32, 64]
    assert plan.offset_min_graph_tokens == 32


def test_inplace_split_bucket_respects_allowed_sizes_by_start():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(65),
        total_num_tokens=65,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={32, 64, 128},
        offset_match_policy="bucket",
        offset_capture_sizes={16, 32, 64},
        offset_allowed_graph_tokens_by_start={
            32: [16, 32],
            64: [32, 64],
        },
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 64
    assert plan.second_graph_tokens == 32
    assert plan.offset_allowed_graph_tokens_by_start == {
        32: [16, 32],
        64: [32, 64],
    }


def test_inplace_split_exact_rejects_offset_graph_below_min_size():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(65),
        total_num_tokens=65,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={64, 128},
        offset_match_policy="exact",
        offset_min_graph_tokens=32,
    )

    assert plan is None
    assert reason == NO_SPLIT_OFFSET_GRAPH_BELOW_MIN_SIZE


def test_inplace_split_bucket_filters_sizes_by_query_len():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(124, query_len=2),
        total_num_tokens=248,
        uniform_decode_query_len=2,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={64, 127, 128},
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.second_actual_tokens == 120
    assert plan.second_graph_tokens == 128
    assert plan.offset_capture_sizes_considered == [64, 128]


def test_inplace_split_bucket_rejects_when_no_aligned_offset_size():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(124, query_len=2),
        total_num_tokens=248,
        uniform_decode_query_len=2,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={3, 5},
    )

    assert plan is None
    assert reason == NO_SPLIT_NO_OFFSET_CAPTURE_SIZE


def test_inplace_split_bucket_rejects_when_bucket_too_small():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(249),
        total_num_tokens=249,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={64},
    )

    assert plan is None
    assert reason == NO_SPLIT_OFFSET_BUCKET_TOO_SMALL


def test_inplace_split_bucket_rejects_padding_token_limit():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(249),
        total_num_tokens=249,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={128},
        offset_max_padding_tokens=6,
    )

    assert plan is None
    assert reason == NO_SPLIT_OFFSET_PADDING_TOO_LARGE


def test_inplace_split_bucket_ignores_padding_ratio_limit():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(129),
        total_num_tokens=129,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={128, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={128},
        offset_max_padding_ratio=8.0,
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.second_actual_tokens == 1
    assert plan.second_graph_tokens == 128


def test_inplace_split_bucket_rejects_graph_slice_beyond_padded_batch():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(193),
        total_num_tokens=193,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={192, 256},
        offset_match_policy="bucket",
        offset_capture_sizes={128},
    )

    assert plan is None
    assert reason == NO_SPLIT_OFFSET_GRAPH_EXCEEDS_PADDED_BATCH


def test_inplace_split_handles_query_len_four_spec_decode_shape():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(104, query_len=4),
        total_num_tokens=416,
        uniform_decode_query_len=4,
        cudagraph_capture_sizes={256, 384, 512},
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 384
    assert plan.second_tokens == 32
    assert plan.first_reqs == 96
    assert plan.second_reqs == 8
    assert plan.split_slices[0].request_slice == slice(0, 96)
    assert plan.split_slices[0].token_slice == slice(0, 384)
    assert plan.split_slices[1].request_slice == slice(96, 104)
    assert plan.split_slices[1].token_slice == slice(384, 416)


def test_inplace_split_rejects_spec_decode_without_aligned_lower_capture():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(103, query_len=4),
        total_num_tokens=412,
        uniform_decode_query_len=4,
        cudagraph_capture_sizes={258, 386, 512},
    )

    assert plan is None
    assert reason == NO_SPLIT_NO_LOWER_CAPTURE_SIZE


def test_inplace_split_rejects_exact_graph_hit():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(384),
        total_num_tokens=384,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
    )

    assert plan is None
    assert reason == NO_SPLIT_EXACT_GRAPH_HIT


def test_inplace_split_rejects_batch_above_max_capture_size():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(640),
        total_num_tokens=640,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
    )

    assert plan is None
    assert reason == NO_SPLIT_ABOVE_MAX_CAPTURE_SIZE


def test_inplace_split_rejects_when_no_lower_capture_size_exists():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(128),
        total_num_tokens=128,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
    )

    assert plan is None
    assert reason == NO_SPLIT_NO_LOWER_CAPTURE_SIZE


def test_inplace_split_skips_unaligned_lower_capture_sizes():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(84, query_len=5),
        total_num_tokens=420,
        uniform_decode_query_len=5,
        cudagraph_capture_sizes={256, 384, 512},
    )

    assert plan is None
    assert reason == NO_SPLIT_NO_LOWER_CAPTURE_SIZE


def test_inplace_split_uses_aligned_lower_capture_size_when_available():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(84, query_len=5),
        total_num_tokens=420,
        uniform_decode_query_len=5,
        cudagraph_capture_sizes={250, 384, 512},
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 250
    assert plan.first_reqs == 50
    assert plan.second_tokens == 170
    assert plan.second_reqs == 34
    assert plan.split_slices[1].start_num_tokens == 250


def test_inplace_split_rejects_remainder_limit():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(416),
        total_num_tokens=416,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
        inplace_max_remainder_tokens=16,
    )

    assert plan is None
    assert reason == NO_SPLIT_REMAINDER_TOO_LARGE


def test_inplace_split_backend_guard_rejects_mixed_pa_shapes():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(416),
        total_num_tokens=416,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert not inplace_split_preserves_attention_backend(
        plan, lambda shape: shape in {512})


def test_inplace_split_backend_guard_accepts_same_backend_shapes():
    plan, reason = create_inplace_split_batch_slices(
        _tokens(416),
        total_num_tokens=416,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes={256, 384, 512},
    )

    assert reason == INPLACE_SPLIT_DRY_RUN
    assert plan is not None
    assert inplace_split_preserves_attention_backend(
        plan, lambda shape: shape in {32, 384, 512})


def _mixed_tokens(values: list[int]) -> np.ndarray:
    return np.array(values, dtype=np.int32)


def test_mixed_request_split_chooses_request_boundary_plan():
    plan, reason = create_mixed_request_split_batch_slices(
        _mixed_tokens([1, 1, 64, 64]),
        total_num_tokens=130,
        cudagraph_capture_sizes={64, 66, 128},
        decode_threshold=1,
    )

    assert reason == MIXED_REQUEST_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.reason == MIXED_REQUEST_SPLIT_DRY_RUN
    assert plan.first_tokens == 66
    assert plan.second_tokens == 64
    assert plan.first_reqs == 3
    assert plan.second_reqs == 1
    assert plan.offset_match_policy == "compact"

    first, second = plan.split_slices
    assert first.request_slice == slice(0, 3)
    assert first.token_slice == slice(0, 66)
    assert first.graph_num_tokens == 66
    assert first.start_num_tokens == 0
    assert second.request_slice == slice(3, 4)
    assert second.token_slice == slice(66, 130)
    assert second.graph_num_tokens == 64
    assert second.start_num_tokens == 0

    payload = plan.debug_payload()
    assert payload["mixed_request_split"] is True
    assert payload["split_req_index"] == 3
    assert payload["num_decode_reqs"] == 2
    assert payload["num_prefill_reqs"] == 2
    assert payload["num_decode_tokens"] == 2
    assert payload["num_prefill_tokens"] == 128
    assert payload["first_attention_cost"] == 258
    assert payload["second_attention_cost"] == 256


def test_mixed_request_split_supports_prefill_only_multi_request_batch():
    plan, reason = create_mixed_request_split_batch_slices(
        _mixed_tokens([64, 64, 64, 64]),
        total_num_tokens=256,
        cudagraph_capture_sizes={128, 256},
        decode_threshold=1,
    )

    assert reason == MIXED_REQUEST_SPLIT_DRY_RUN
    assert plan is not None
    assert plan.first_tokens == 128
    assert plan.second_tokens == 128
    assert plan.split_slices[0].request_slice == slice(0, 2)
    assert plan.split_slices[1].request_slice == slice(2, 4)
    assert all(s.start_num_tokens == 0 for s in plan.split_slices)


def test_mixed_request_split_rejects_decode_only_batch():
    plan, reason = create_mixed_request_split_batch_slices(
        _mixed_tokens([1, 1, 1, 1]),
        total_num_tokens=4,
        cudagraph_capture_sizes={4},
        decode_threshold=1,
        min_total_tokens=1,
    )

    assert plan is None
    assert reason == NO_SPLIT_MIXED_DECODE_ONLY


def test_mixed_request_split_rejects_single_dominant_request():
    plan, reason = create_mixed_request_split_batch_slices(
        _mixed_tokens([1, 1, 128]),
        total_num_tokens=130,
        cudagraph_capture_sizes={2, 128},
        decode_threshold=1,
        min_total_tokens=1,
    )

    assert plan is None
    assert reason == NO_SPLIT_SINGLE_REQUEST_DOMINATES


def test_mixed_request_split_rejects_below_min_total_tokens():
    plan, reason = create_mixed_request_split_batch_slices(
        _mixed_tokens([32, 32]),
        total_num_tokens=64,
        cudagraph_capture_sizes={32},
        decode_threshold=1,
        min_total_tokens=128,
    )

    assert plan is None
    assert reason == NO_SPLIT_MIXED_BELOW_MIN_TOTAL_TOKENS


def test_mixed_request_split_rejects_min_tokens_per_split():
    plan, reason = create_mixed_request_split_batch_slices(
        _mixed_tokens([1, 1, 64, 64]),
        total_num_tokens=130,
        cudagraph_capture_sizes={64, 66, 128},
        decode_threshold=1,
        min_tokens_per_split=128,
        min_total_tokens=1,
    )

    assert plan is None
    assert reason == NO_SPLIT_MIN_TOKENS_PER_SPLIT


def test_mixed_request_split_rejects_missing_graph_bucket():
    plan, reason = create_mixed_request_split_batch_slices(
        _mixed_tokens([64, 64, 64]),
        total_num_tokens=192,
        cudagraph_capture_sizes={64},
        decode_threshold=1,
        max_padding_ratio_per_split=None,
    )

    assert plan is None
    assert reason == NO_SPLIT_GRAPH_BUCKET_MISSING


def test_mixed_request_split_rejects_padding_ratio():
    plan, reason = create_mixed_request_split_batch_slices(
        _mixed_tokens([1, 1, 64, 64]),
        total_num_tokens=130,
        cudagraph_capture_sizes={64, 128},
        decode_threshold=1,
    )

    assert plan is None
    assert reason == NO_SPLIT_PADDING_TOO_LARGE


def test_mixed_request_split_rejects_invalid_policy():
    plan, reason = create_mixed_request_split_batch_slices(
        _mixed_tokens([64, 64]),
        total_num_tokens=128,
        cudagraph_capture_sizes={64},
        decode_threshold=1,
        split_policy="unknown",
    )

    assert plan is None
    assert reason == NO_SPLIT_INVALID_MIXED_REQUEST_SPLIT_POLICY


def test_split_batch_config_accepts_mixed_request_split_config():
    cfg = SplitBatchConfig({
        "enabled": True,
        "mode": "inplace_parallel",
        "num_splits": 2,
        "enable_parallel_streams": True,
        "inplace_parallel_replay_policy": "piecewise_attention_parallel",
        "enable_mixed_request_split": True,
        "mixed_request_min_total_tokens": 256,
        "mixed_request_min_tokens_per_split": 128,
        "mixed_request_max_single_request_ratio": 0.8,
        "mixed_request_max_padding_ratio_per_split": None,
        "mixed_request_decode_weight": 2,
        "mixed_request_prefill_weight": 5,
    })

    assert cfg.enable_mixed_request_split is True
    assert cfg.mixed_request_split_policy == "balanced_attention"
    assert cfg.mixed_request_split_execution_mode == "dry_run"
    assert cfg.mixed_request_min_total_tokens == 256
    assert cfg.mixed_request_min_tokens_per_split == 128
    assert cfg.mixed_request_max_single_request_ratio == 0.8
    assert cfg.mixed_request_max_padding_ratio_per_split is None
    assert cfg.mixed_request_decode_weight == 2
    assert cfg.mixed_request_prefill_weight == 5


def test_split_batch_config_accepts_mixed_request_split_serial_execution():
    cfg = SplitBatchConfig({
        "enabled": True,
        "mode": "inplace_serial",
        "num_splits": 2,
        "enable_mixed_request_split": True,
        "mixed_request_split_execution_mode": "serial",
    })

    assert cfg.enable_mixed_request_split is True
    assert cfg.mixed_request_split_execution_mode == "serial"


def test_split_batch_config_accepts_mixed_request_split_piecewise_parallel_execution():
    cfg = SplitBatchConfig({
        "enabled": True,
        "mode": "inplace_parallel",
        "num_splits": 2,
        "enable_parallel_streams": True,
        "inplace_parallel_replay_policy": "piecewise_attention_parallel",
        "enable_mixed_request_split": True,
        "mixed_request_split_execution_mode":
        "piecewise_attention_parallel",
    })

    assert cfg.enable_mixed_request_split is True
    assert (cfg.mixed_request_split_execution_mode ==
            "piecewise_attention_parallel")


def test_split_batch_config_rejects_mixed_request_split_parallel_buffer():
    with pytest.raises(ValueError,
                       match="enable_mixed_request_split requires mode"):
        SplitBatchConfig({
            "enabled": True,
            "mode": "parallel_buffer",
            "enable_mixed_request_split": True,
        })


def test_split_batch_config_rejects_mixed_request_split_invalid_execution_mode():
    with pytest.raises(ValueError,
                       match="mixed_request_split_execution_mode"):
        SplitBatchConfig({
            "enabled": True,
            "mode": "inplace_serial",
            "enable_mixed_request_split": True,
            "mixed_request_split_execution_mode": "parallel",
        })


def test_split_batch_config_rejects_mixed_request_split_serial_execution_in_parallel_mode():
    with pytest.raises(ValueError, match="requires mode='inplace_serial'"):
        SplitBatchConfig({
            "enabled": True,
            "mode": "inplace_parallel",
            "num_splits": 2,
            "enable_parallel_streams": True,
            "inplace_parallel_replay_policy": "piecewise_attention_parallel",
            "enable_mixed_request_split": True,
            "mixed_request_split_execution_mode": "serial",
        })


def test_split_batch_config_rejects_mixed_request_split_piecewise_parallel_execution_in_serial_mode():
    with pytest.raises(ValueError, match="requires mode='inplace_parallel'"):
        SplitBatchConfig({
            "enabled": True,
            "mode": "inplace_serial",
            "num_splits": 2,
            "enable_mixed_request_split": True,
            "mixed_request_split_execution_mode":
            "piecewise_attention_parallel",
        })


def test_split_batch_config_rejects_mixed_request_split_full_graph_policy():
    with pytest.raises(ValueError,
                       match="piecewise_attention_parallel"):
        SplitBatchConfig({
            "enabled": True,
            "mode": "inplace_parallel",
            "num_splits": 2,
            "enable_parallel_streams": True,
            "inplace_parallel_replay_policy": "full_graph_parallel",
            "enable_mixed_request_split": True,
        })
