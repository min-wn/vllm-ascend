# SPDX-License-Identifier: Apache-2.0
"""Reject unpaired inputs, changed state results and gains within baseline noise."""

import argparse
import json
import statistics
from pathlib import Path

MIN_PAIRS = 3
MIN_GAIN_PERCENT = 1.0


def compare(pairs):
    if len(pairs) != MIN_PAIRS:
        raise ValueError("exactly three independent old/new pairs required")
    site_names = None
    records = {}
    for pair_index, (old, new) in enumerate(pairs):
        for report in (old, new):
            if report["status"] != "complete_same_algorithm_diagnostic_not_model_timing":
                raise ValueError("incomplete process")
            if report["phase"] != "candidate":
                raise ValueError("use the same closed endpoint in both candidate controls")
        old_sites = {site["site"]: site for site in old["sites"]}
        new_sites = {site["site"]: site for site in new["sites"]}
        if old_sites.keys() != new_sites.keys() or len(old_sites) != len(old["sites"]):
            raise ValueError("missing or duplicated site")
        if site_names is None:
            site_names = set(old_sites)
        if site_names != set(old_sites):
            raise ValueError("site coverage changed between process pairs")
        for name, before in old_sites.items():
            after = new_sites[name]
            if before["site_sha256"] != after["site_sha256"] or before["input_hashes"] != after["input_hashes"]:
                raise ValueError("not the same frozen inputs and continuous initial state")
            for key in ("aic", "aiv", "q4", "precision", "capacity_requests", "live_requests"):
                if before[key] != after[key]:
                    raise ValueError(f"comparison contract changed: {key}")
            old_checks = {c["mode"]: c["hashes"] for c in before["checks"]}
            new_checks = {c["mode"]: c["hashes"] for c in after["checks"]}
            if old_checks != new_checks or set(old_checks) != {"closed_delayed", "closed_ready"}:
                raise ValueError("output/publication/accepted/terminal state mismatch")
            row = records.setdefault(
                name, dict(site=name, capacity=before["capacity_requests"], live=before["live_requests"], pairs=[])
            )
            times = {
                tag: {
                    mode: statistics.median(site["round_medians_us"][mode])
                    for mode in ("closed_delayed", "closed_ready")
                }
                for tag, site in (("old", before), ("new", after))
            }
            row["pairs"].append(
                dict(
                    pair=pair_index,
                    device_us=times,
                    gain_percent={
                        mode: 100 * (1 - times["new"][mode] / times["old"][mode])
                        for mode in ("closed_delayed", "closed_ready")
                    },
                )
            )
    for row in records.values():
        before = [pair["device_us"]["old"]["closed_ready"] for pair in row["pairs"]]
        after = [pair["device_us"]["new"]["closed_ready"] for pair in row["pairs"]]
        noise = 100 * (max(before) - min(before)) / statistics.median(before)
        gain = 100 * (1 - statistics.mean(after) / statistics.mean(before))
        row.update(
            old_ready_us=statistics.mean(before),
            new_ready_us=statistics.mean(after),
            ready_gain_percent=gain,
            old_repeat_spread_percent=noise,
            gate_percent=max(MIN_GAIN_PERCENT, 2 * noise),
            pass_gate=(
                gain > max(MIN_GAIN_PERCENT, 2 * noise)
                and all(pair["gain_percent"]["closed_ready"] > 0 for pair in row["pairs"])
            ),
        )
    return dict(
        status="complete",
        sites=list(records.values()),
        enter_complete_layer=bool(records) and all(row["pass_gate"] for row in records.values()),
        scope="closed single-layer recurrence; not complete layer or model improvement",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve previous analysis")
    pairs = [
        tuple(json.loads((args.root / f"pair{pair}.{tag}.json").read_bytes()) for tag in ("old", "new"))
        for pair in range(MIN_PAIRS)
    ]
    args.output.write_text(json.dumps(compare(pairs), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
