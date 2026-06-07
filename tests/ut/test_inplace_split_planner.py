import numpy as np

from vllm_ascend.worker.ubatch_utils import (
    inplace_split_preserves_attention_backend,
    INPLACE_SPLIT_DRY_RUN,
    NO_SPLIT_ABOVE_MAX_CAPTURE_SIZE,
    NO_SPLIT_EXACT_GRAPH_HIT,
    NO_SPLIT_INVALID_FIRST_TOKENS_POLICY,
    NO_SPLIT_NO_LOWER_CAPTURE_SIZE,
    NO_SPLIT_NO_OFFSET_CAPTURE_SIZE,
    NO_SPLIT_OFFSET_BUCKET_TOO_SMALL,
    NO_SPLIT_OFFSET_GRAPH_BELOW_MIN_SIZE,
    NO_SPLIT_OFFSET_GRAPH_EXCEEDS_START_CAP,
    NO_SPLIT_OFFSET_GRAPH_EXCEEDS_PADDED_BATCH,
    NO_SPLIT_OFFSET_PADDING_TOO_LARGE,
    NO_SPLIT_REMAINDER_TOO_LARGE,
    create_inplace_split_batch_slices,
)


def _tokens(num_reqs: int, query_len: int = 1) -> np.ndarray:
    return np.full(num_reqs, query_len, dtype=np.int32)


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
