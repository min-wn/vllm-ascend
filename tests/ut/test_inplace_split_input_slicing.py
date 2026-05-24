import torch

from vllm_ascend.attention.utils import (slice_model_inputs_by_token,
                                         slice_positions_by_token)


def test_slice_model_inputs_uses_offset_views_for_1d_positions():
    input_ids = torch.arange(416, dtype=torch.int32)
    positions = torch.arange(416, dtype=torch.int64)
    token_slice = slice(384, 416)

    sliced_input_ids, sliced_positions, sliced_inputs_embeds = (
        slice_model_inputs_by_token(input_ids, positions, None, token_slice))

    assert torch.equal(sliced_input_ids, input_ids[token_slice])
    assert torch.equal(sliced_positions, positions[token_slice])
    assert sliced_inputs_embeds is None
    assert sliced_input_ids.shape == (32, )
    assert sliced_positions.shape == (32, )
    assert sliced_input_ids.storage_offset() == 384
    assert sliced_positions.storage_offset() == 384
    assert sliced_input_ids.data_ptr() == input_ids[token_slice].data_ptr()
    assert sliced_positions.data_ptr() == positions[token_slice].data_ptr()


def test_slice_positions_by_token_handles_2d_positions():
    positions = torch.arange(3 * 416, dtype=torch.int64).reshape(3, 416)
    token_slice = slice(384, 416)

    sliced_positions = slice_positions_by_token(positions, token_slice)

    assert torch.equal(sliced_positions, positions[:, token_slice])
    assert sliced_positions.shape == (3, 32)
    assert sliced_positions.storage_offset() == 384
    assert sliced_positions.data_ptr() == positions[:, token_slice].data_ptr()


def test_slice_model_inputs_handles_inputs_embeds():
    hidden_size = 4
    input_ids = torch.arange(416, dtype=torch.int32)
    positions = torch.arange(416, dtype=torch.int64)
    inputs_embeds = torch.arange(416 * hidden_size,
                                 dtype=torch.float32).reshape(
                                     416, hidden_size)
    token_slice = slice(384, 416)

    sliced_input_ids, sliced_positions, sliced_inputs_embeds = (
        slice_model_inputs_by_token(input_ids, positions, inputs_embeds,
                                    token_slice))

    assert torch.equal(sliced_input_ids, input_ids[token_slice])
    assert torch.equal(sliced_positions, positions[token_slice])
    assert torch.equal(sliced_inputs_embeds, inputs_embeds[token_slice])
    assert sliced_inputs_embeds.shape == (32, hidden_size)
    assert sliced_inputs_embeds.storage_offset() == 384 * hidden_size
    assert sliced_inputs_embeds.data_ptr() == inputs_embeds[
        token_slice].data_ptr()


def test_slice_model_inputs_allows_input_ids_none():
    hidden_size = 4
    positions = torch.arange(416, dtype=torch.int64)
    inputs_embeds = torch.arange(416 * hidden_size,
                                 dtype=torch.float32).reshape(
                                     416, hidden_size)
    token_slice = slice(384, 416)

    sliced_input_ids, sliced_positions, sliced_inputs_embeds = (
        slice_model_inputs_by_token(None, positions, inputs_embeds,
                                    token_slice))

    assert sliced_input_ids is None
    assert torch.equal(sliced_positions, positions[token_slice])
    assert torch.equal(sliced_inputs_embeds, inputs_embeds[token_slice])
