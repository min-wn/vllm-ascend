from dataclasses import dataclass, field

import pytest
import torch

from vllm_ascend.worker.multi_graph.buffers import LaneBuffers, assert_disjoint
from vllm_ascend.worker.multi_graph.metadata import clone_metadata
from vllm_ascend.worker.multi_graph.types import GraphKey, SlicePlan


@dataclass
class DecodeMetadata:
    num_prefills: int = 0
    num_decodes: int = 4
    num_actual_tokens: int = 4
    num_decode_tokens: int = 4
    max_query_len: int = 1
    attn_state: str = "DecodeOnly"
    seq_lens: torch.Tensor = None
    seq_lens_cpu: torch.Tensor = None
    seq_lens_gpu: torch.Tensor = None
    seq_lens_list: list[int] = None
    actual_seq_lengths_q: list[int] = None
    query_start_loc: torch.Tensor = None
    query_start_loc_gpu: torch.Tensor = None
    block_tables: torch.Tensor = None
    slot_mapping: torch.Tensor = None
    attn_mask: torch.Tensor = None
    reshape_cache_event: object = None
    pcp_local_num_input_tokens: int = None
    qfa_metadata_cache: dict = field(default_factory=dict)


@pytest.fixture
def metadata():
    lengths = torch.tensor([11, 23, 35, 47], dtype=torch.int32)
    return DecodeMetadata(
        seq_lens=lengths,
        seq_lens_cpu=lengths,
        seq_lens_gpu=lengths,
        seq_lens_list=lengths.tolist(),
        actual_seq_lengths_q=[1, 2, 3, 4],
        query_start_loc=torch.arange(5, dtype=torch.int32),
        query_start_loc_gpu=torch.arange(5, dtype=torch.int32),
        block_tables=torch.tensor([[3, 9], [4, 10], [5, 11], [6, 12]], dtype=torch.int32),
        slot_mapping=torch.tensor([395, 524, 673, 812]),
        attn_mask=torch.zeros((2, 2)),
    )


def test_metadata_rebases_query_but_preserves_kv_addresses(signature, metadata):
    part = SlicePlan(2, 4, 2, 4, GraphKey(1, 2, 2, signature))
    lane = clone_metadata(metadata, part)
    assert lane.query_start_loc.tolist() == [0, 1, 2]
    assert lane.actual_seq_lengths_q == [1, 2]
    assert lane.seq_lens_list == [35, 47]
    assert lane.block_tables.tolist() == [[5, 11], [6, 12]]
    assert lane.slot_mapping.tolist() == [673, 812]
    assert lane.num_actual_tokens == lane.num_decodes == lane.num_decode_tokens == 2
    assert_disjoint((metadata.seq_lens, metadata.block_tables), (lane.seq_lens, lane.block_tables))
    lane.seq_lens_list[0] = 999
    lane.block_tables.fill_(0)
    assert metadata.seq_lens_list[2] == 35
    assert metadata.block_tables[2, 0] == 5


def test_refresh_preserves_every_tensor_address(signature, metadata):
    part = SlicePlan(0, 2, 0, 2, GraphKey(0, 2, 2, signature))
    source = {"layer0": metadata, "layer1": metadata}
    ids = torch.tensor([1, 2, 3, 4])
    positions = torch.tensor([10, 22, 34, 46])
    buffers = LaneBuffers.create(ids, positions, source, part)
    assert buffers.metadata["layer0"] is buffers.metadata["layer1"]
    addresses = tuple(tensor.data_ptr() for tensor in buffers.writable_tensors())
    metadata.seq_lens.add_(10)
    metadata.seq_lens_list[:] = metadata.seq_lens.tolist()
    metadata.slot_mapping.add_(100)
    values = buffers.preflight(ids + 1, positions + 1, source, part)
    buffers.refresh(ids + 1, positions + 1, part, values)
    assert addresses == tuple(tensor.data_ptr() for tensor in buffers.writable_tensors())
    assert buffers.input_ids.tolist() == [2, 3]
    assert buffers.metadata["layer0"].seq_lens_list == [21, 33]
    assert buffers.metadata["layer0"].slot_mapping.tolist() == [495, 624]


def test_layout_failure_is_detected_before_any_mutation(signature, metadata):
    part = SlicePlan(0, 2, 0, 2, GraphKey(0, 2, 2, signature))
    ids = torch.arange(4)
    buffers = LaneBuffers.create(ids, ids, {"layer": metadata}, part)
    metadata.block_tables = torch.zeros((4, 3), dtype=torch.int32)
    with pytest.raises(ValueError, match="block_tables"):
        buffers.preflight(ids + 100, ids, {"layer": metadata}, part)
    assert buffers.input_ids.tolist() == [0, 1]


def test_storage_aliasing_rejects_nonoverlapping_views_of_same_pool():
    tensor = torch.empty(8)
    with pytest.raises(ValueError, match="share writable storage"):
        assert_disjoint((tensor[:4],), (tensor[4:],))
