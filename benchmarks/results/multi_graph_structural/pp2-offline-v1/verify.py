# SPDX-License-Identifier: Apache-2.0
"""Independent CPU checks of the offline ledger, not an NPU qualification."""

import importlib
import itertools
import json
import sys

import audit


def exhaustive_cost(batch, groups, capacities):
    """Enumerate compositions via cut positions, independently of the DP."""
    best = None
    for cuts in itertools.combinations(range(1, batch), groups - 1):
        boundaries = (0, *cuts, batch)
        live = [right - left for left, right in itertools.pairwise(boundaries)]
        caps = [capacities[n] for n in live]
        objective = (sum(caps) + max(caps), sum(n * n for n in caps), sum(n * n for n in live))
        if best is None or objective < best:
            best = objective
    return best


def main():
    report = audit.read(audit.HERE / "audit.json")
    rebuilt = json.loads(json.dumps(audit.build()))
    assert report == rebuilt
    capacities = audit.capacity_table(report["capture_sizes"], 128)
    partitions = audit.partition_table(capacities, 8)
    cases = [(batch, 2) for batch in range(2, 129)]
    cases += [(batch, 4) for batch in range(4, 23)]
    cases += [(batch, 8) for batch in range(8, 17)]
    for batch, groups in cases:
        live = partitions[groups][batch]
        caps = [capacities[n] for n in live]
        actual = (sum(caps) + max(caps), sum(n * n for n in caps), sum(n * n for n in live))
        assert actual == exhaustive_cost(batch, groups, capacities), (batch, groups, actual)

    # Use the actual reservation implementation to independently check the
    # inventory estimate's null-page convention and immutable request coverage.
    sys.path.insert(0, str(audit.ROOT / "benchmarks"))
    metadata = importlib.import_module("multi_graph_gdn_q4_full_metadata")
    trace = metadata.load_fixed_trace(audit.RESULTS / "structural-dynamic-trace-v1.json")
    pages = metadata.reserve_owner_pages(trace)
    allocated = [page for owner in pages.pages for page in owner]
    assert sorted(allocated) == list(range(1, pages.page_count))
    assert pages.page_count == report["pp2_residency"]["actual_fia_page_count_including_null"] == 1293
    assert sum(pages.maximum_lengths) == 156445
    for candidate in report["partitions"]:
        for row, step in zip(candidate["rows"], trace.steps, strict=True):
            position = 0
            covered = []
            for count, capacity in zip(row["live_requests_per_group"], row["group_capacity_tokens"], strict=True):
                owners = step["request_ids"][position : position + count]
                assert len(owners) == count and count * 4 <= capacity and capacity in report["capture_sizes"]
                covered.extend(owners)
                position += count
            assert tuple(covered) == step["request_ids"]

    for sizes in ((2,), (3, 3), (7, 2, 9, 1), (104, 96, 96, 96)):
        assert audit.linear_pipeline(sizes, sizes) == sum(sizes) + max(sizes)
    assert audit.linear_pipeline([2, 2], [2, 2], [1, 1]) == 7
    try:
        audit.linear_pipeline([1], [-1])
    except ValueError:
        pass
    else:
        raise AssertionError("negative resource time was accepted")

    headers = audit.read(audit.HERE / "checkpoint-headers.json")
    shapes = report["batch98_projection_shapes"]
    full_shapes = {name: item["full_weight_n_k"] for name, item in shapes.items()}
    for layer in (headers["layers"][0], headers["layers"][3]):
        schema = headers["layer_schemas"][layer["schema_sha256"]]
        assert full_shapes["gate_up"] == [2 * schema["mlp.gate_proj.weight"]["shape"][0], 5120]
        assert full_shapes["down"] == schema["mlp.down_proj.weight"]["shape"]
    schema = headers["layer_schemas"][headers["layers"][0]["schema_sha256"]]
    assert full_shapes["gdn_qkvz"] == [
        schema["linear_attn.in_proj_qkv.weight"]["shape"][0] + schema["linear_attn.in_proj_z.weight"]["shape"][0],
        5120,
    ]
    schema = headers["layer_schemas"][headers["layers"][3]["schema_sha256"]]
    assert full_shapes["fia_qgate_kv"] == [
        sum(schema[f"self_attn.{component}_proj.weight"]["shape"][0] for component in ("q", "k", "v")),
        5120,
    ]
    result = dict(
        scope="CPU ledger verification only; not graph capture, NPU correctness or measured performance",
        regeneration_matches=True,
        independent_exhaustive_partition_cases=len(cases),
        all_step_group_owner_coverage_checks=128 * 3,
        actual_reservation_null_page_checked=True,
        checkpoint_header_and_gemm_shapes_checked=True,
        pipeline_recurrence_and_negative_time_checked=True,
    )
    (audit.HERE / "checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
