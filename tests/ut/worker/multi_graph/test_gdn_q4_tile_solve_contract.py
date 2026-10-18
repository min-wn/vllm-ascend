# SPDX-License-Identifier: Apache-2.0
"""Keep the transport experiment's consumer identical to the public solve.

These source contracts establish that a later overlap comparison does not also
change recurrence math or state publication. Device qualification remains
necessary to test transport visibility and eager/replay determinism.
"""

import re
from pathlib import Path

import pytest

BENCHMARKS = Path(__file__).resolve().parents[4] / "benchmarks"
BASELINE = BENCHMARKS / "multi_graph_gdn_q4_log_deferred_kernel" / "q4_log_deferred_kernel.cpp"
CONSUMER = BENCHMARKS / "multi_graph_gdn_q4_tile_pipeline_kernel" / "q4_tile_solve.h"


def method_body(source, name):
    signature = re.search(r"__aicore__\s+inline\s+void\s+" + name + r"\s*\(", source)
    assert signature is not None, name
    start = source.index("{", signature.end())
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def code_tokens(source):
    source = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?[fFuUlL]*|\S", source)


@pytest.mark.parametrize(
    "name", ["LoadInputs", "LoadOldLog", "ComputeGram", "Projection", "Solve", "WriteOutput", "WriteLog"]
)
def test_tile_consumer_preserves_public_deferred_solve_operations(name):
    baseline = BASELINE.read_text(encoding="utf-8")
    consumer = CONSUMER.read_text(encoding="utf-8")
    assert code_tokens(method_body(consumer, name)) == code_tokens(method_body(baseline, name))


def test_tile_consumer_preserves_geometry_and_fp32_scale():
    pattern = r"constexpr\s+(?:uint32_t|float)\s+(\w+)\s*=\s*([^;]+);"
    baseline = dict(re.findall(pattern, BASELINE.read_text(encoding="utf-8")))
    consumer = dict(re.findall(pattern, CONSUMER.read_text(encoding="utf-8")))
    assert consumer == baseline


def test_tile_consumer_cannot_commit_base_or_publish_canonical_log():
    consumer = CONSUMER.read_text(encoding="utf-8")
    assert "CommitBase" not in consumer
    assert "PublishLog" not in consumer
    assert not re.search(r"GlobalTensor<float>\s+state_", consumer)
    assert "stateOutputQueue_" not in consumer
    assert "DataCopy(newLogs_[" in method_body(consumer, "WriteLog")
    assert "DataCopy(newValid_[" in method_body(consumer, "WriteLog")
