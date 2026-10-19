# SPDX-License-Identifier: Apache-2.0
"""Independently recompute all ABC/CBA wall differences from decimal raw samples."""

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite a recomputation")
    root = Path(__file__).resolve().parents[1]
    audit = json.loads((root / "saved-audit.json").read_bytes())
    paths = sorted((root / "raw/screen").glob("cell-*/consumer-probe.json"))
    assert len(paths) == 6
    summaries, values_by_cell, hashes = [], {}, {}
    sample_count = 0
    for index, path in enumerate(paths):
        report = json.loads(path.read_bytes())
        for capacity in (400, 512):
            values = sorted(Decimal(str(s["wall_us"])) for s in report["samples"] if s["capacity"] == capacity)
            assert len(values) == 10
            sample_count += len(values)
            mean, median = sum(values) / len(values), (values[4] + values[5]) / 2
            (old,) = [s for s in audit["summaries"] if s["cell"] == index and s["capacity"] == capacity]
            assert abs(mean - Decimal(str(old["mean_us"]))) < Decimal("1e-8")
            assert abs(median - Decimal(str(old["median_us"]))) < Decimal("1e-8")
            values_by_cell[index, capacity] = mean, median
            summaries.append(
                dict(
                    cell=index,
                    capacity=capacity,
                    mean_us=float(mean),
                    median_us=float(median),
                )
            )
            for phase in ("pre_checks", "post_checks"):
                (check,) = [r for r in report[phase] if r["capacity"] == capacity]
                if capacity in hashes:
                    assert hashes[capacity] == check["output_hashes"]
                hashes[capacity] = check["output_hashes"]
    pairs = []
    for contrast, directional in (
        ("layout_only", ((0, 1), (5, 4))),
        ("transfer_same_layout", ((1, 2), (4, 3))),
        ("candidate_vs_original", ((0, 2), (5, 3))),
    ):
        for direction, (control, wide) in enumerate(directional):
            for capacity in (400, 512):
                old, new = (
                    values_by_cell[control, capacity],
                    values_by_cell[wide, capacity],
                )
                reductions = [100 * (a - b) / a for a, b in zip(old, new, strict=True)]
                (expected,) = [
                    r
                    for r in audit["directional_pairs"]
                    if r["contrast"] == contrast and r["pair"] == direction and r["capacity"] == capacity
                ]
                for metric, number in zip(
                    ("mean_reduction_percent", "median_reduction_percent"),
                    reductions,
                    strict=True,
                ):
                    assert abs(number - Decimal(str(expected[metric]))) < Decimal("1e-8")
                pairs.append(
                    dict(
                        contrast=contrast,
                        direction=direction,
                        capacity=capacity,
                        mean_reduction_percent=float(reductions[0]),
                        median_reduction_percent=float(reductions[1]),
                    )
                )
    assert sample_count == 120 and len(pairs) == 12
    result = dict(
        state="passed",
        samples=sample_count,
        summaries=summaries,
        pairs=pairs,
        complete_output_hashes_equal=True,
        full_model_or_split_gain=False,
        recomputer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps({k: v for k, v in result.items() if k != "summaries"}, indent=2))


if __name__ == "__main__":
    main()
