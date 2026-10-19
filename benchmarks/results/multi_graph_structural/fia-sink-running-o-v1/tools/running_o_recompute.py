"""Primary independent arithmetic from frozen four-cell raw reports."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--bundle", type=Path, default=HERE / "ro-v3")
parser.add_argument("--output", type=Path, default=HERE / "running-o-primary-recompute-v3.json")
args = parser.parse_args()
ROOT = args.bundle
OUT = args.output
assert not OUT.exists()
reports = {}
means = {}
rows = []
for path in sorted(ROOT.glob("cell-*/consumer-probe.json")):
    report = json.loads(path.read_bytes())
    assert report["state"] == "completed" and report["returncode"] == 0
    variant = path.parent.name.split("-")[2]
    policy = report["policy"]
    assert (variant, policy) not in reports
    reports[variant, policy] = report
    for cap in (400, 512):
        values = [item["wall_us"] for item in report["samples"] if item["capacity"] == cap]
        assert len(values) == 10
        means[variant, policy, cap] = statistics.mean(values)
        rows.append(dict(variant=variant, policy=policy, capacity=cap, samples=10,
                         mean_us=statistics.mean(values), median_us=statistics.median(values),
                         min_us=min(values), max_us=max(values),
                         raw_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(reports) == 4
contrasts = []
equality = []
for cap in (400, 512):
    for policy in ("upstream", "whole"):
        old, new = (means[variant, policy, cap] for variant in ("control", "resident"))
        contrasts.append(dict(kind="consumer", policy=policy, capacity=cap,
                              saved_us=old-new, reduction_percent=100*(old-new)/old))
        hashes = []
        plans = []
        for variant in ("control", "resident"):
            report = reports[variant, policy]
            for phase in ("pre_checks", "post_checks"):
                check, = [item for item in report[phase] if item["capacity"] == cap]
                assert len(check["output_hashes"]) == 16
                hashes.append(check["output_hashes"])
                plans.append(check["plan"]["words"])
        equality.append(dict(policy=policy, capacity=cap,
                             full_16_output_hashes_equal_across_macro_and_pre_post=all(x == hashes[0] for x in hashes),
                             full_1024_word_plans_equal=all(x == plans[0] for x in plans)))
    for variant in ("control", "resident"):
        old, new = (means[variant, policy, cap] for policy in ("upstream", "whole"))
        contrasts.append(dict(kind="policy", variant=variant, capacity=cap,
                              saved_us=old-new, reduction_percent=100*(old-new)/old))
    old, new = means["control", "upstream", cap], means["resident", "whole", cap]
    contrasts.append(dict(kind="combined", capacity=cap,
                          saved_us=old-new, reduction_percent=100*(old-new)/old))
result = dict(timing=rows, contrasts=contrasts, equality=equality,
              process_replicates_per_cell=1, timings=80,
              scope="Two fixed records, synthetic sixteen-attention chain; arithmetic is not a stable gain/full-model or split-over-padding claim",
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
print(json.dumps(result, indent=2))
