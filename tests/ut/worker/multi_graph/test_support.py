from dataclasses import replace

import pytest

from vllm_ascend.worker.multi_graph.support import UnsupportedFeature, rejection_reasons


def test_verified_supported_profile(profile, signature):
    assert rejection_reasons(profile, signature) == ()
    assert rejection_reasons(replace(profile, device_family="910C"), signature) == ()


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"runner_version": "v2"}, "V1"),
        ({"device_family": "310P"}, "910B/910C"),
        ({"model_verified": False}, "ownership"),
        ({"parallel_sizes": (1, 1, 1, 1)}, "TP=PP=DP=PCP=DCP=1"),
        ({"parallel_sizes": (1, 1, 1, 1, 2)}, "TP=PP=DP=PCP=DCP=1"),
        ({"expert_parallel": True}, "expert"),
        ({"configured_graph_mode": "FULL"}, "FULL_DECODE_ONLY"),
    ],
)
def test_unsupported_runtime_has_reason(profile, signature, changes, reason):
    assert any(reason in failure for failure in rejection_reasons(replace(profile, **changes), signature))


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype": "float32"},
        {"attention_backend": "AscendMLABackend"},
        {"attention_variant": "PA"},
        {"attention_variant": "FIAV2"},
        {"graph_mode": "PIECEWISE"},
        {"attention_state": "ChunkedPrefill"},
        {"query_len": 2},
    ],
)
def test_unsupported_capture_variant(profile, signature, changes):
    assert rejection_reasons(profile, replace(signature, **changes))


@pytest.mark.parametrize("feature", list(UnsupportedFeature))
def test_each_stateful_feature_is_rejected(profile, signature, feature):
    changed = replace(profile, unsupported_features=frozenset({feature}))
    assert feature.value in rejection_reasons(changed, signature)


def test_reports_all_reasons_deterministically(profile, signature):
    changed = replace(
        profile,
        runner_version="v2",
        unsupported_features=frozenset({UnsupportedFeature.LORA, UnsupportedFeature.KV_TRANSFER}),
    )
    reasons = rejection_reasons(changed, signature)
    assert len(reasons) == 3
    assert reasons == rejection_reasons(changed, signature)
