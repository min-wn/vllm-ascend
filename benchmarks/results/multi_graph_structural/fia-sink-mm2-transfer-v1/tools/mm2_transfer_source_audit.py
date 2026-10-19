"""Recompute transfer work from frozen records; no latency or HBM prediction."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / "work/vllm-ascend"
BUNDLE = REPO / "benchmarks/results/multi_graph_structural/fia-sink-running-o-v1"
REFERENCE = BUNDLE / "cell-03-control-whole/consumer-probe.json"
report = json.loads(REFERENCE.read_bytes())
assert report["state"] == "completed" and report["policy"] == "whole"
rows = []
for record in report["selected_records"]:
    cap = record["physical_tokens"]
    check, = [r for r in report["pre_checks"] if r["capacity"] == cap]
    base = check["plan"]["words"][576:586]
    assert base[:3] == [512, 512, 8] and base[3] == base[5] == 0
    old_groups = new_groups = old_rows = new_rows = compute_tiles = outer_tiles = 0
    for length in record["kv_lengths"]:
        for start in range(0, length, 512):
            size = min(512, length - start)
            for _head in range(2):
                outer_tiles += 1
                old_groups += (size + 63) // 64
                new_groups += (size + 127) // 128
                compute_tiles += (size + 63) // 64
                old_rows += sum(min(64, size - k) for k in range(0, size, 64))
                new_rows += sum(min(128, size - k) for k in range(0, size, 128))
    assert old_rows == new_rows == 2 * sum(record["kv_lengths"])
    rows.append(dict(
        step=record["step"], batch=record["batch"], capacity=cap,
        outer_s2_tasks=outer_tiles,
        old_v_transfers=old_groups, candidate_v_transfers=new_groups,
        old_p_transfers=old_groups, candidate_p_transfers=new_groups,
        eliminated_transfer_groups=old_groups-new_groups,
        transfer_group_reduction_percent=100*(old_groups-new_groups)/old_groups,
        old_mm2_compute_tiles=compute_tiles, candidate_mm2_compute_tiles=compute_tiles,
        old_v_page_lookups=old_groups, candidate_v_page_lookups=new_groups,
        logical_v_bytes_per_attention=old_rows*256*2,
        logical_p_bytes_per_attention=old_rows*24*2,
        logical_kv_bytes_per_attention=old_rows*256*2*2,
        valid_row_sequence_unchanged=True,
    ))
result = dict(
    scope="Whole-head Q4/GQA6/D256/page128 source work counts for two frozen records; not runtime counters",
    reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
    rows=rows,
    assumptions=["one actual M=24 tile per request/KV head", "exact ordered pages, all real lengths",
                 "only MM2 L1 grouping changes; inner compute K remains 64", "no FD or sink tensor"],
    no_measured_hbm_or_latency_claim=True,
    graph_split_required_for_candidate=False,
)
out = HERE / "mm2-transfer-source-audit-v1.json"
assert not out.exists()
out.write_text(json.dumps(result, indent=2)+"\n", encoding="ascii")
print(json.dumps(result, indent=2))
