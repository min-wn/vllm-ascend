import pytest
import torch

from vllm_ascend.worker.multi_graph.kv_layout import contiguous_dense_kv_views


def test_contiguous_kv_views_share_allocation_without_overlapping():
    raw = torch.zeros(2 * 3 * 4 * 2 * 8)
    key, value = contiguous_dense_kv_views(raw, (2, 3, 4, 2, 8))
    assert key.is_contiguous() and value.is_contiguous()
    assert key.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert value.storage_offset() == key.numel()
    key[1, 2, 0].fill_(3)
    value[1, 2, 0].fill_(5)
    assert torch.equal(raw.view(2, 3, 4, 2, 8)[0, 1, 2, 0], torch.full((8,), 3.0))
    assert torch.equal(raw.view(2, 3, 4, 2, 8)[1, 1, 2, 0], torch.full((8,), 5.0))
    assert key[0].count_nonzero() == 0


@pytest.mark.parametrize("shape", [(3, 3, 4, 2, 8), (2, 0, 4, 2, 8), (2, 3, 4, 16)])
def test_reject_unsupported_shape(shape):
    with pytest.raises(ValueError):
        contiguous_dense_kv_views(torch.empty(384), shape)


def test_reject_padding_and_noncontiguous_allocation():
    with pytest.raises(ValueError):
        contiguous_dense_kv_views(torch.empty(400), (2, 3, 4, 2, 8))
    with pytest.raises(ValueError):
        contiguous_dense_kv_views(torch.empty(768)[::2], (2, 3, 4, 2, 8))
