# SPDX-License-Identifier: Apache-2.0
"""CPU checks for owner history and the real-activation preparation boundary."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "benchmarks/multi_graph_gdn_q14_layer_inputs.py"


def module():
    assert SOURCE.exists(), "teacher-forced layer input helper is not implemented"
    spec = importlib.util.spec_from_file_location("q14_layer_inputs_tested", SOURCE)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def payload():
    return dict(
        stable_owners=[0, 1],
        capacity_requests=2,
        steps=[
            dict(step=54, request_ids=[1, 0], query_lengths=[4, 1], accepted_from_previous=[4, 3]),
            dict(step=55, request_ids=[1], query_lengths=[1], accepted_from_previous=[3]),
            dict(step=56, request_ids=[0, 1], query_lengths=[4, 4], accepted_from_previous=[1, 1]),
        ],
    )


def test_owner_cursor_follows_acceptance_across_reorder_pause_and_resume():
    frozen = payload()
    before = copy.deepcopy(frozen)
    tokens = [list(range(100)), list(range(1000, 1100))]
    rows = module().activation_plan(frozen, tokens)
    assert [row["offsets"] for row in rows] == [[16, 16], [19], [17, 20]]
    assert rows[0]["source_tokens"] == [[1016, 1017, 1018, 1019], [16, 17, 18, 19]]
    assert rows[2]["positions"] == [17, 18, 19, 20, 20, 21, 22, 23]
    assert rows[1]["positions"] == [19, 20, 21, 22, 0, 0, 0, 0]
    assert frozen == before


def test_invalid_acceptance_after_q1_and_exhausted_text_are_rejected():
    frozen = payload()
    frozen["steps"][2]["accepted_from_previous"][0] = 2
    with pytest.raises(ValueError, match="previous query"):
        module().activation_plan(frozen, [list(range(100))] * 2)
    with pytest.raises(ValueError, match="four source tokens"):
        module().activation_plan(payload(), [list(range(20))] * 2)


class Tensor(np.ndarray):
    def clone(self):
        return self.copy()

    def copy_(self, value):
        self[...] = value
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return np.asarray(self)


def tensor(value, dtype=None, **kwargs):
    return np.asarray(value, dtype=dtype).view(Tensor)


class Source:
    def __init__(self):
        self.tokens = [list(range(100)), list(range(1000, 1100))]
        self.token_sha256 = "frozen-token-table"
        self.bundle = SimpleNamespace(device="cpu")
        self.target = SimpleNamespace(conv=tensor([[0], [0], [0]]), base=tensor([[0], [0], [0]]))

    def batch(self, owners, lengths, *, target=False):
        result = []
        for owner, length in zip(owners, lengths, strict=True):
            if target:
                self.target.conv[owner + 1] = length + owner
                self.target.base[owner + 1] = 100 + length + owner
            values = tensor([[owner * 1000 + pos] for pos in range(length - 4, length)], dtype=np.float32)
            result.append((values, -values))
        return result


def test_preparation_retains_quartets_positions_and_independent_prefill_snapshot():
    torch = SimpleNamespace(
        bfloat16=np.float32,
        int64=np.int64,
        uint8=np.uint8,
        zeros=lambda *shape, dtype, **kwargs: np.zeros(shape, dtype=dtype).view(Tensor),
        tensor=tensor,
        isfinite=np.isfinite,
        npu=SimpleNamespace(synchronize=lambda: None),
    )
    source = Source()
    config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(hidden_size=1)))
    prepared = module().build_layer_inputs(torch, None, config, payload(), source=source, batch_size=1)
    assert prepared["initial_conv"].tolist() == [[0], [16], [17]]
    assert prepared["initial_ssm"].tolist() == [[0], [116], [117]]
    assert prepared["records"][0]["hidden"].reshape(-1).tolist() == [1016, 1017, 1018, 1019, 16, 17, 18, 19]
    assert prepared["records"][1]["hidden"].reshape(-1).tolist() == [1019, 1020, 1021, 1022, 0, 0, 0, 0]
    assert prepared["records"][2]["positions"].tolist() == [17, 18, 19, 20, 20, 21, 22, 23]
    assert not prepared["provenance"]["initial_state_matches_original_ledger_step"]
    assert len(prepared["records"][0]["input_manifest"]["tensor_sha256"]["hidden"]) == 64
    source.target.conv[...] = -1
    assert prepared["initial_conv"].tolist() == [[0], [16], [17]]


def test_frozen_sixteen_step_metadata_produces_224_rows_without_original_sequence_offsets():
    manifest = ROOT / "benchmarks/results/multi_graph_structural/gdn-compact-q14-v1/manifest.json"
    frozen = json.loads(manifest.read_bytes())["payload"]
    rows = module().activation_plan(frozen, [list(range(256))] * 56)
    assert [row["step"] for row in rows] == list(range(54, 70))
    assert all(len(row["positions"]) == 224 for row in rows)
    assert rows[0]["offsets"] == [16] * 56
    assert rows[-1]["offsets"] == [31] * 56
    assert rows[1]["query_lengths"] == frozen["steps"][1]["query_lengths"]
