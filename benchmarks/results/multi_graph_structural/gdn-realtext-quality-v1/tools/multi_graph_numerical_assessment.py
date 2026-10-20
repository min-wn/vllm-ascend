"""Diagnostic error measurements; these do not certify model output quality."""


def error_metrics(actual, expected):
    import torch

    if actual.shape != expected.shape:
        raise ValueError("numerical assessment requires matching shapes")
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("nonfinite values are not accepted as rounding differences")
    delta = actual - expected
    squared_error = delta.double().square().mean().item()
    reference_energy = expected.double().square().mean().item()
    failures = (delta.abs() > 0.02 + 0.02 * expected.abs()).sum().item()
    return dict(
        elements=actual.numel(),
        original_tolerance_failures=failures,
        original_tolerance_failure_fraction=failures / actual.numel(),
        max_abs=delta.abs().max().item(),
        rmse=squared_error**0.5,
        normalized_rmse=(squared_error / max(reference_energy, 1e-30)) ** 0.5,
        reference_rms=reference_energy**0.5,
    )
