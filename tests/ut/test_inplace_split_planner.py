import numpy as np

from vllm_ascend.worker.ubatch_utils import (
    inplace_split_preserves_attention_backend,
    INPLACE_SPLIT_DRY_RUN,
    NO_SPLIT_ABOVE_MAX_CAPTURE_SIZE,
    NO_SPLIT_EXACT_GRAPH_HIT,
    NO_SPLIT_NO_LOWER_CAPTURE_SIZE,
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
