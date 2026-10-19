# SPDX-License-Identifier: Apache-2.0
"""Recompute wall statistics directly from raw samples, without the main analyzer."""

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
    assert len(paths) == 4
    summaries = []
    sample_count = 0
    hashes = {}
    for index, path in enumerate(paths):
        report = json.loads(path.read_bytes())
        for capacity in (400, 512):
            values = sorted(Decimal(str(s["wall_us"])) for s in report["samples"] if s["capacity"] == capacity)
            assert len(values) == 10
            sample_count += len(values)
            mean = sum(values) / len(values)
            median = (values[4] + values[5]) / 2
            (old,) = [s for s in audit["summaries"] if s["cell"] == index and s["capacity"] == capacity]
            assert abs(mean - Decimal(str(old["mean_us"]))) < Decimal("1e-8")
            assert abs(median - Decimal(str(old["median_us"]))) < Decimal("1e-8")
            summaries.append(dict(cell=index, capacity=capacity, mean_us=float(mean), median_us=float(median)))
            for phase in ("pre_checks", "post_checks"):
                (check,) = [r for r in report[phase] if r["capacity"] == capacity]
                if capacity in hashes:
                    assert hashes[capacity] == check["output_hashes"]
                hashes[capacity] = check["output_hashes"]
    pairs = []
    for control, wide in ((0, 1), (3, 2)):
        for capacity in (400, 512):
            (old,) = [r for r in summaries if r["cell"] == control and r["capacity"] == capacity]
            (new,) = [r for r in summaries if r["cell"] == wide and r["capacity"] == capacity]
            pairs.append(
                dict(
                    control_cell=control,
                    wide_cell=wide,
                    capacity=capacity,
                    mean_increase_percent=100 * (new["mean_us"] / old["mean_us"] - 1),
                    median_increase_percent=100 * (new["median_us"] / old["median_us"] - 1),
                )
            )
    assert sample_count == 80 and all(r["median_increase_percent"] > 0 for r in pairs)
    result = dict(
        state="passed",
        samples=sample_count,
        summaries=summaries,
        pairs=pairs,
        complete_output_hashes_equal=True,
        conclusion="Reject wide-V candidate on both process orders",
        full_model_or_split_gain=False,
        recomputer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps({k: v for k, v in result.items() if k != "summaries"}, indent=2))


if __name__ == "__main__":
    main()
