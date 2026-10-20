# SPDX-License-Identifier: Apache-2.0
import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[4] / "benchmarks" / "multi_graph_replayssm_protocol.py"
SPEC = importlib.util.spec_from_file_location("replayssm_protocol", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_multiple_flushes_wraps_and_pause_resume():
    trace, final = MODULE.frozen_trace()
    observed = MODULE.coverage(trace)
    assert min(observed["flushes"]) >= 2
    assert min(observed["flushes"][:99]) >= 3
    assert min(observed["wraps"][:99]) >= 1
    assert observed["pauses"][98] and observed["resumes"][98]
    assert set(final) == {1, 2, 3, 4}


def test_flush_keeps_current_window_at_old_physical_write_position():
    cursor = MODULE.Cursor(13, 27, True, True)
    current_start = (cursor.cache_base + cursor.write_pos) % MODULE.RING_LENGTH
    next_cursor = cursor.advance(3)
    assert next_cursor.cache_base == current_start == 8
    assert next_cursor.write_pos == 3
    assert not next_cursor.is_flush


def test_first_step_does_not_commit_nonexistent_window():
    assert MODULE.Cursor().advance(4) == MODULE.Cursor(pending=True)


@pytest.mark.parametrize("accepted", [0, 5, True, 1.5])
def test_reject_bad_acceptance(accepted):
    with pytest.raises(ValueError):
        MODULE.Cursor().advance(accepted)
