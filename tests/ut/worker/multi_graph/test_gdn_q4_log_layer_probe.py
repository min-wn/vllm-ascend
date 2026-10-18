# SPDX-License-Identifier: Apache-2.0
"""CPU scope checks for the real-layer compact/padding causal comparison."""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture
def probe():
    path = Path(__file__).resolve().parents[4] / "benchmarks/multi_graph_gdn_q4_log_layer_probe.py"
    spec = importlib.util.spec_from_file_location("q4_log_layer_probe_cpu_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qualification_acceptance_tracks_request_generation_through_pause_and_resume(probe):
    schedule = probe.qualification_schedule(5, 7)
    assert len(schedule) == 8
    assert [entry["live"] for entry in schedule] == [5, 4, 6, 5] * 2
    generations = [0] * 7
    for step, entry in enumerate(schedule):
        assert entry["step"] == step
        assert entry["accepted"] == [1 + (request + generations[request]) % 4 for request in range(7)]
        assert all(1 <= accepted <= 4 for accepted in entry["accepted"])
        for request in range(entry["live"]):
            generations[request] += 1
    # Request 4 pauses during step 1, then consumes the same pending decision.
    assert schedule[1]["accepted"][4] == schedule[2]["accepted"][4]
    # Request 5 first becomes active at step 2; no inactive step advances it.
    assert schedule[0]["accepted"][5] == schedule[2]["accepted"][5]
    assert schedule[3]["accepted"][5] == schedule[6]["accepted"][5]
    # This request never participates and retains its first-use decision.
    assert len({entry["accepted"][6] for entry in schedule}) == 1


def test_qualification_schedule_is_fixed_and_does_not_mutate_another_call(probe):
    schedule = probe.qualification_schedule(98, 100)
    assert [entry["live"] for entry in schedule] == [98, 97, 99, 98] * 2
    assert schedule == probe.qualification_schedule(98, 100)
    assert len({entry["accepted"][0] for entry in schedule}) == 4
    schedule[0]["accepted"][0] = 99
    assert probe.qualification_schedule(98, 100)[0]["accepted"][0] == 1


@pytest.mark.parametrize("batch,capacity", [(2, 100), (101, 100), (0, 0)])
def test_invalid_schedule_bounds_are_rejected(probe, batch, capacity):
    with pytest.raises(ValueError):
        probe.qualification_schedule(batch, capacity)


@pytest.mark.parametrize(
    "mode,backend",
    [
        ("official_padding", None),
        ("compact_padding", "padding"),
        ("compact_serial", "serial"),
        ("compact_pipeline", "pipeline"),
        ("compact_deferred_serial", "deferred_serial"),
        ("compact_deferred_overlap", "deferred_overlap"),
        ("compact_deferred_pipeline", "deferred_pipeline"),
        ("compact_deferred_tile_delayed", "deferred_tile_delayed"),
        ("compact_deferred_tile_ready", "deferred_tile_ready"),
        ("compact_deferred_tile_separate", "deferred_tile_separate"),
    ],
)
def test_layer_controls_select_only_the_recurrence_backend(probe, mode, backend):
    assert probe.mode_backend_name(mode) == backend


def test_layer_control_refuses_unknown_backend(probe):
    with pytest.raises(ValueError):
        probe.mode_backend_name("split_dense_projections")


def test_deferred_controls_require_the_explicit_extension_option(probe):
    baseline = ("official_padding", "compact_padding", "compact_serial", "compact_pipeline")
    assert probe.layer_modes(None) == baseline
    assert probe.layer_modes(Path("deferred.so")) == baseline + (
        "compact_deferred_serial",
        "compact_deferred_overlap",
        "compact_deferred_pipeline",
    )
    assert probe.layer_modes(Path("deferred.so"), None) == probe.layer_modes(Path("deferred.so"))


def test_request_tile_controls_require_both_explicit_extensions_and_keep_all_prior_controls(probe):
    with pytest.raises(ValueError, match="require the deferred extension"):
        probe.layer_modes(None, Path("tile.so"))
    original = probe.layer_modes(Path("deferred.so"))
    assert probe.layer_modes(Path("deferred.so"), Path("tile.so")) == original + (
        "compact_deferred_tile_delayed",
        "compact_deferred_tile_ready",
        "compact_deferred_tile_separate",
    )


def test_deferred_pipeline_comparison_uses_equally_optimized_padding_control(probe):
    comparison = probe.deferred_pipeline_comparison(
        dict(
            official_padding=[3000, 3000, 3000],
            compact_padding=[2600, 2600, 2600],
            compact_deferred_serial=[2500, 2500, 2500],
            compact_deferred_overlap=[2400, 2300, 2500],
            compact_deferred_pipeline=[2280, 2185, 2375],
        )
    )
    assert comparison["baseline"] == "compact_deferred_overlap"
    assert comparison["candidate"] == "compact_deferred_pipeline"
    assert comparison["paired_round_latency_reduction_percent"] == pytest.approx([5, 5, 5])
    assert comparison["median_of_round_medians_reduction_percent"] == pytest.approx(5)
    assert probe.deferred_pipeline_comparison(dict(compact_padding=[2600])) is None


def test_deferred_pipeline_comparison_keeps_losses_and_pairing_mismatch_visible(probe):
    comparison = probe.deferred_pipeline_comparison(
        dict(compact_deferred_overlap=[2400, 2400], compact_deferred_pipeline=[2520, 2520])
    )
    assert comparison["paired_round_latency_reduction_percent"] == pytest.approx([-5, -5])
    with pytest.raises(ValueError):
        probe.deferred_pipeline_comparison(
            dict(compact_deferred_overlap=[2400, 2400], compact_deferred_pipeline=[2280])
        )


@pytest.mark.parametrize(
    "separate_times,delayed_times,expected_baseline,expected_improvement",
    [
        ([2400] * 3, [2250] * 3, "compact_deferred_overlap", 10),
        ([1800] * 3, [2250] * 3, "compact_deferred_tile_separate", 0),
        ([2400] * 3, [1875] * 3, "compact_deferred_tile_delayed", 4),
    ],
)
def test_tile_comparison_credits_only_increment_beyond_best_public_padding(
    probe, separate_times, delayed_times, expected_baseline, expected_improvement
):
    comparison = probe.tile_pipeline_comparison(
        dict(
            official_padding=[3000, 3000, 3000],
            compact_deferred_overlap=[2000, 2000, 2000],
            compact_deferred_tile_separate=separate_times,
            compact_deferred_tile_delayed=delayed_times,
            compact_deferred_tile_ready=[1800, 1800, 1800],
            compact_deferred_pipeline=[1900, 1900, 1900],
        )
    )
    best = comparison["versus_best_public_padding"]
    assert best["baseline"] == expected_baseline
    assert best["median_of_round_medians_reduction_percent"] == pytest.approx(expected_improvement)
    assert set(comparison["versus_each_public_padding"]) == {
        "compact_deferred_overlap",
        "compact_deferred_tile_separate",
        "compact_deferred_tile_delayed",
    }
    assert comparison["same_kernel_readiness_increment"]["paired_round_latency_reduction_percent"] == pytest.approx(
        [100 * (1 - 1800 / previous) for previous in delayed_times]
    )
    assert comparison["versus_previous_request_pipeline"]["baseline"] == "compact_deferred_pipeline"
    assert "not establish full-model" in comparison["scope"]
    assert probe.tile_pipeline_comparison(dict(compact_deferred_overlap=[2000])) is None


def test_tile_comparison_preserves_losses_round_pairing_and_fast_public_consumer(probe):
    times = dict(
        compact_deferred_overlap=[2400, 2000, 2200],
        compact_deferred_tile_separate=[2200, 1900, 2100],
        compact_deferred_tile_delayed=[2100, 1800, 2000],
        compact_deferred_tile_ready=[2310, 1995, 2205],
        compact_deferred_pipeline=[2300, 1950, 2150],
    )
    comparison = probe.tile_pipeline_comparison(times)
    best = comparison["versus_best_public_padding"]
    assert best["baseline"] == "compact_deferred_tile_delayed"
    assert best["paired_round_latency_reduction_percent"] == pytest.approx([-10.0, -10.833333333, -10.25])
    assert best["median_of_round_medians_reduction_percent"] == pytest.approx(-10.25)
    assert comparison["versus_each_public_padding"]["compact_deferred_tile_separate"][
        "paired_round_latency_reduction_percent"
    ] == pytest.approx([-5, -5, -5])
    assert comparison["same_kernel_readiness_increment"]["paired_round_latency_reduction_percent"] == pytest.approx(
        [-10.0, -10.833333333, -10.25]
    )
    times["compact_deferred_tile_ready"] = [2310, 1995]
    with pytest.raises(ValueError):
        probe.tile_pipeline_comparison(times)


def test_tile_comparison_does_not_assign_fused_delayed_kernel_gain_to_readiness(probe):
    comparison = probe.tile_pipeline_comparison(
        dict(
            compact_deferred_overlap=[2340.41, 2349.94, 2347.46],
            compact_deferred_tile_delayed=[2343.40, 2333.54, 2337.04],
            compact_deferred_tile_ready=[2196.82, 2206.53, 2185.73],
            compact_deferred_tile_separate=[2607, 2608, 2609],
            compact_deferred_pipeline=[2300, 2301, 2302],
        )
    )
    best = comparison["versus_best_public_padding"]
    assert best["baseline"] == "compact_deferred_tile_delayed"
    assert best["median_of_round_medians_reduction_percent"] == pytest.approx(100 * (1 - 2196.82 / 2337.04))
    assert best == comparison["same_kernel_readiness_increment"]
    assert "waits for each producer wave" in comparison["attribution"]


@pytest.mark.parametrize("raise_inside", [False, True])
def test_commit_pre_hook_runs_before_real_projection_and_is_removed_on_exit(probe, raise_inside):
    calls = []
    projection = torch.nn.Linear(3, 2, bias=False)

    class Backend:
        def launch_pending_commit(self):
            calls.append("launch")
            return False  # A no-pending backend must not replace linear inputs.

    observer = projection.register_forward_hook(lambda *args: calls.append("projection_complete"))
    value = torch.ones(1, 3)
    expected = projection(value).clone()
    calls.clear()
    try:
        with probe.installed_commit_pre_hook(projection, Backend()):
            actual = projection(value)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert calls == ["launch", "projection_complete"]
            if raise_inside:
                raise RuntimeError("fixture failure")
    except RuntimeError:
        assert raise_inside
    calls.clear()
    projection(value)
    assert calls == ["projection_complete"]
    observer.remove()


def test_dynamic_inactive_output_restoration_preserves_live_rows_and_clears_nan(probe):
    hidden = torch.tensor([[1.0, 2.0], [float("nan"), 3.0], [4.0, float("inf")]])
    residual = hidden + 1
    active = torch.tensor([[True], [False], [False]])
    result = probe.restore_active_outputs(torch, (hidden, residual), active)
    for actual, source in zip(result, (hidden, residual), strict=True):
        torch.testing.assert_close(actual[0], source[0], rtol=0, atol=0)
        assert torch.isfinite(actual).all() and not actual[1:].count_nonzero()
    active[1] = True
    changed = probe.restore_active_outputs(torch, (torch.nan_to_num(hidden),), active)[0]
    assert changed[1, 1] == 3 and not changed[2].count_nonzero()
