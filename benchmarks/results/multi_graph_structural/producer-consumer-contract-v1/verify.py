# SPDX-License-Identifier: Apache-2.0
"""Independently reconstruct inventories and exhaust small pipeline models."""

import itertools
import json
import math
from decimal import Decimal

import audit


def two_servers(chunks, total, communication, projection):
    communication_end, projection_end = Decimal(0), Decimal(0)
    for chunk in chunks:
        communication_end += communication * chunk / total
        projection_end = max(communication_end, projection_end) + projection * chunk / total
    return projection_end


def exhaust(total, count, allowed, communication, projection):
    best, checked = None, 0
    for prefix in itertools.product(allowed, repeat=count - 1):
        tail = total - sum(prefix)
        if tail not in allowed:
            continue
        end = two_servers((*prefix, tail), total, communication, projection)
        best = end if best is None else min(best, end)
        checked += 1
    assert best is not None
    return best, checked


def main():
    document = audit.read(audit.HERE / "audit.json")
    assert json.loads(json.dumps(audit.build())) == document
    traces = document["trace_records"]
    for row in document["conditional_budget"]["rows"]:
        values = {}
        for mode in ("padding", "fused_padding"):
            descriptor = next(t for t in traces if t["rank"] == row["rank"] and t["mode"] == mode)
            path = audit.RESULTS / "two-layer-fused-v1/results" / descriptor["file"]
            assert audit.digest(path) == descriptor["sha256"]
            raw = json.loads(path.read_bytes(), parse_float=Decimal)
            events = raw if isinstance(raw, list) else raw["traceEvents"]
            events = sorted((e for e in events if e.get("ph") == "X"), key=lambda e: Decimal(str(e["ts"])))
            mm = [
                e
                for e in events
                if "MatMul" in e["name"] and e.get("args", {}).get("Task Type", "").startswith("KERNEL_")
            ]
            comm = [e for e in events if e["name"].startswith("hcom_")]
            if mode == "padding":
                ar = [e for e in comm if e["name"].startswith("hcom_allReduce")]
                assert len(mm) == 9 and len(ar) == 4
                values["down"] = Decimal(str(mm[3]["dur"]))
                values["reduce"] = Decimal(str(ar[1]["dur"]))
                values["projection"] = sum(Decimal(str(e["dur"])) for e in mm[4:6])
            else:
                fused = [e for e in events if e["name"] == "MatmulReduceScatterV2"]
                ag = [e for e in comm if e["name"].startswith("hcom_allGather")]
                assert len(mm) == 8 and len(fused) == len(ag) == 1
                values["baseline"] = sum(Decimal(str(e["dur"])) for e in (*fused, *ag, *mm[3:5]))
        for field, key in (
            ("ordinary_down_us", "down"),
            ("ordinary_reduce_us", "reduce"),
            ("ordinary_input_projections_us", "projection"),
            ("fused_inventory_us", "baseline"),
        ):
            assert math.isclose(row[field], float(values[key]), abs_tol=1e-8)
        assert math.isclose(
            row["fluid_ideal_difference_us"],
            float(values["baseline"] - values["down"] - max(values["reduce"], values["projection"])),
            abs_tol=1e-8,
        )

    selection = audit.read(audit.RESULTS / "two-layer-fused-v1/analysis.json")["selection"]
    total = selection["capacity_tokens"]
    allowed = [size for size in selection["capture_sizes"] if size % 8 == 0 and size <= total]
    model_plans = 0
    for row in document["conditional_budget"]["rows"]:
        comm = Decimal(str(row["ordinary_reduce_us"]))
        projection = Decimal(str(row["ordinary_input_projections_us"]))
        for plan in row["linear_default_capacity_pipeline"]:
            chunks = plan["chunks"]
            assert len(chunks) == plan["groups"] and sum(chunks) == total
            assert all(chunk in allowed for chunk in chunks)
            observed = two_servers(chunks, total, comm, projection)
            assert math.isclose(float(observed), plan["reduction_projection_span_us"], abs_tol=1e-8)
            if plan["groups"] in (2, 4):
                best, count = exhaust(total, plan["groups"], allowed, comm, projection)
                assert observed == best
                model_plans += count

    small_cases = 0
    for comm, projection in ((10, 4), (4, 10), (6, 6)):
        for groups in (1, 2, 3, 4):
            best, count = exhaust(8, groups, (1, 2, 4, 8), Decimal(comm), Decimal(projection))
            dp = audit.optimal_linear_pipeline(8, groups, (1, 2, 4, 8), Decimal(comm), Decimal(projection))
            assert math.isclose(float(best), dp["reduction_projection_span_us"], abs_tol=1e-8)
            model_plans += count
            small_cases += 1

    for plan in document["conditional_budget"]["common_plans"]:
        assert sum(plan["chunks"]) == total and all(chunk in allowed for chunk in plan["chunks"])
        endpoints = []
        for row in document["conditional_budget"]["rows"]:
            end = Decimal(str(row["ordinary_down_us"])) + two_servers(
                plan["chunks"],
                total,
                Decimal(str(row["ordinary_reduce_us"])),
                Decimal(str(row["ordinary_input_projections_us"])),
            )
            endpoints.append(float(end))
        assert all(
            math.isclose(a, b, abs_tol=1e-8) for a, b in zip(endpoints, plan["per_rank_inventory_us"], strict=True)
        )
        assert max(endpoints) == plan["critical_inventory_us"]
    result = dict(
        scope="CPU archive/source and conditional-model verification only; no NPU execution",
        deterministic_regeneration=True,
        public_source_files=11,
        archived_trace_hashes=12,
        independently_reconstructed_trace_inventories=4,
        archived_source_files=len(document["checked_archived_sources"]),
        exhaustive_model_plans=model_plans,
        small_model_cases=small_cases,
        actual_cost_model_two_and_four_group_dp_matches_exhaustive=True,
        common_rank_plan_evaluation=True,
    )
    (audit.HERE / "checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
