# SPDX-License-Identifier: Apache-2.0
"""Source contracts and conditional budgets, without new NPU measurements."""

import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE.parent
WORK = ROOT.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from aclgraph_two_layer_fused_analysis import analyze  # noqa: E402

REVISIONS = {
    "ops": (WORK / "reference-ops-transformer-20261020", "b58ac682305ab2ddc753dccde4111acab8bd6af5"),
    "catccos": (WORK / "reference-catccos-20261020", "4359f7fb80decb4a83236f41af5dea980232acc5"),
    "shmem": (
        WORK / "reference-catccos-20261020/3rdparty/shmem",
        "aa1e4c24f706a4f9d63975c77e527ef1eb37ca9f",
    ),
    "catlass": (
        WORK / "reference-catccos-20261020/3rdparty/catlass",
        "db93081ce9b0ad6c04e99b9ed5f2fba081ccdb00",
    ),
}
SOURCE_CHECKS = (
    (
        "ops",
        "mc2/matmul_reduce_scatter_v2/op_host/op_tiling/arch22/matmul_reduce_scatter_v2_aiv_mode_tiling.cpp",
        (
            "GetAlgorithmPolicy",
            "M <= 2048",
            "N > 512",
            "SetScheduleMode(batch_mode)",
            "CalcTschBlockDim(aivNum, aicNum, aivNum)",
            "SetBlockDim(numBlocks)",
        ),
    ),
    (
        "ops",
        "mc2/matmul_reduce_scatter_v2/op_kernel/arch22/matmul_reduce_scatter_v2.cpp",
        ("KERNEL_TYPE_MIX_AIC_1_2", "MatmulReduceScatterAivModeSmallM"),
    ),
    (
        "ops",
        "mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_smallM.h",
        (
            "NotifyRemoteDataCopy(",
            "CompleteRemoteDataCopy(",
            "taskSplit.GetCulumnStartPos(cal_idx)",
            "taskSplit.GetCulumnEndPos(cal_idx)",
            "SyncAll<true>()",
            "n_st * n0",
            "CommColumnSplitter taskSplit",
        ),
    ),
    (
        "ops",
        "mc2/matmul_reduce_scatter_v2/op_kernel/matmul_reduce_scatter_aiv_mode_util.h",
        ("class CommColumnSplitter", "GetCulumnStartPos", "GetCulumnEndPos"),
    ),
    (
        "ops",
        "mc2/all_gather_matmul_v2/op_host/config/ascend910b/all_gather_matmul_v2_binary.json",
        ("AllGatherMatmulV2_AIV_MODE_ND_NZ_BF16_BF16",),
    ),
    (
        "catccos",
        "include/catccos/dgemm/kernel/matmul_reduce_scatter.hpp",
        (
            "blockPerComm = aicoreNum * params.commInterval",
            "CrossCoreSetFlag<0x2, PIPE_FIX>",
            "CrossCoreWaitFlag(flagAicFinishStore",
            "aclshmemx_barrier_all_vec()",
        ),
    ),
    (
        "catccos",
        "include/catccos/dgemm/device/device_dgemm.hpp",
        ("KernelAdapter<DGemmKernel>", "aclrtSynchronizeStream(stream)"),
    ),
    (
        "catccos",
        "examples/matmul_reduce_scatter/main.cpp",
        ("using ElementA = half", "using LayoutB = Catlass::layout::RowMajor"),
    ),
    (
        "shmem",
        "docs/quickstart.md",
        ("Ascend HDK 25.0.RC1.1", "8.5.0", "8.3.RC1"),
    ),
    (
        "shmem",
        "include/device/gm2gm/shmem_device_cc.h",
        ("only in MIX kernels", "conflict with SyncAll"),
    ),
    (
        "catlass",
        "include/catlass/arch/cross_core_sync.hpp",
        ("MAX_REVERSE_DEPTH = 15", "CrossCoreSetFlagWithReverse", "CrossCoreWaitFlagWithReverse"),
    ),
)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_records():
    records = []
    for name, path, anchors in SOURCE_CHECKS:
        checkout, revision = REVISIONS[name]
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError(f"reference HEAD changed: {name}")
        data = subprocess.check_output(["git", "-C", str(checkout), "show", f"{revision}:{path}"])
        lines = data.decode("utf-8").splitlines()
        locations = {}
        for anchor in anchors:
            hits = [i for i, line in enumerate(lines, 1) if anchor in line]
            if not hits:
                raise ValueError(f"source anchor missing: {name}:{path}:{anchor}")
            locations[anchor] = hits
        records.append(
            dict(
                repository=name,
                revision=revision,
                path=path,
                git_blob_sha256=hashlib.sha256(data).hexdigest(),
                anchors=locations,
            )
        )
    for name in ("shmem", "catlass"):
        checkout, revision = REVISIONS["catccos"]
        tree = subprocess.check_output(
            ["git", "-C", str(checkout), "ls-tree", revision, f"3rdparty/{name}"], text=True
        ).split()
        if tree[:3] != ["160000", "commit", REVISIONS[name][1]]:
            raise ValueError("submodule revision differs from CATCCOS gitlink")
    return records


def optimal_linear_pipeline(capacity, stages, allowed, communication, projection):
    """Exact DP for this idealized divisible two-server model, not hardware."""
    previous = {0: (Decimal(0), ())}
    for _ in range(stages):
        current = {}
        for prefix, (finish, chunks) in previous.items():
            for chunk in allowed:
                total = prefix + chunk
                if total > capacity:
                    continue
                ready = communication * total / capacity
                end = max(finish, ready) + projection * chunk / capacity
                choice = (end, (*chunks, chunk))
                if total not in current or choice < current[total]:
                    current[total] = choice
        previous = current
    finish, chunks = previous[capacity]
    return dict(chunks=list(chunks), reduction_projection_span_us=float(finish))


def conditional_budgets(report):
    rows = []
    selection = report["selection"]
    capacity = selection["capacity_tokens"]
    allowed = [size for size in selection["capture_sizes"] if size % 8 == 0 and size <= capacity]
    for rank in (0, 1):
        plain = next(t for t in report["traces"] if t["rank"] == rank and t["mode"] == "padding")
        fused = next(t for t in report["traces"] if t["rank"] == rank and t["mode"] == "fused_padding")
        down, comm, projection = map(
            Decimal,
            map(str, (plain["previous_down_matmul_us"], plain["first_collective_us"], plain["input_projection_us"])),
        )
        baseline = sum(
            map(
                Decimal,
                map(str, (fused["down_fused_work_us"], fused["all_gather_work_us"], fused["input_projection_us"])),
            )
        )
        ideal = down + max(comm, projection)
        pipelines = []
        for groups in (1, 2, 4, 8):
            row = optimal_linear_pipeline(capacity, groups, allowed, comm, projection)
            predicted = down + Decimal(str(row["reduction_projection_span_us"]))
            pipelines.append(
                dict(
                    groups=groups,
                    **row,
                    candidate_inventory_us=float(predicted),
                    difference_from_fused_inventory_us=float(baseline - predicted),
                )
            )
        rows.append(
            dict(
                rank=rank,
                ordinary_down_us=float(down),
                ordinary_reduce_us=float(comm),
                ordinary_input_projections_us=float(projection),
                fused_inventory_us=float(baseline),
                fluid_ideal_inventory_us=float(ideal),
                fluid_ideal_difference_us=float(baseline - ideal),
                linear_default_capacity_pipeline=pipelines,
            )
        )
    common_plans = []
    for groups in (1, 2, 4, 8):
        candidates = {
            tuple(p["chunks"]) for r in rows for p in r["linear_default_capacity_pipeline"] if p["groups"] == groups
        }
        evaluated = []
        for chunks in sorted(candidates):
            endpoints = []
            for row in rows:
                comm, projection = map(
                    Decimal, map(str, (row["ordinary_reduce_us"], row["ordinary_input_projections_us"]))
                )
                prefix, end = 0, Decimal(0)
                for chunk in chunks:
                    prefix += chunk
                    end = max(end, comm * prefix / capacity) + projection * chunk / capacity
                endpoints.append(float(Decimal(str(row["ordinary_down_us"])) + end))
            evaluated.append(
                dict(chunks=list(chunks), per_rank_inventory_us=endpoints, critical_inventory_us=max(endpoints))
            )
        best = min(evaluated, key=lambda r: (r["critical_inventory_us"], r["chunks"]))
        common_plans.append(
            dict(
                groups=groups,
                **best,
                difference_from_critical_fused_inventory_us=(
                    max(r["fused_inventory_us"] for r in rows) - best["critical_inventory_us"]
                ),
            )
        )
    return dict(
        scope="conditional cross-profile arithmetic; NOT measured gain, a universal bound or a latency prediction",
        assumptions=[
            "Hold ordinary Down, ordinary reduction and projection totals at their saved profiled values.",
            "After whole Down completes, reduction supplies contiguous row groups; projection consumes them in order.",
            "Reduction and projection are independent servers with costs exactly proportional to group rows.",
            "No norm, interference, extra launches, flags, copies, joins or final state costs are included.",
            "Chunk capacities are from the unchanged default table, divisible by eight and sum to full capacity.",
            "One total projection service time represents both required projections; this is optimistic aggregation.",
            "Different communication or a preserved fused producer invalidates these fixed-cost numbers.",
            "Per-rank optima are optimistic screens, not permission for rank-dependent collective ordering.",
        ],
        capacity_tokens=capacity,
        rows=rows,
        common_plans=common_plans,
        common_plan_selection=(
            "Evaluate both per-rank optimum proposals on both ranks; best of this set, "
            "not a joint-optimality claim; no inter-rank arrival skew model."
        ),
    )


def build():
    directory = RESULTS / "two-layer-fused-v1"
    report = json.loads(json.dumps(analyze(directory)))
    if read(directory / "analysis.json") != report:
        raise ValueError("recomputed two-layer analysis differs from archive")
    exports, sdk = read(HERE / "installed-exports.json"), read(HERE / "installed-sdk.json")
    if not all(exports["found_in_these_libraries"].values()):
        raise ValueError("queried ELF symbol absent from inspected library set")
    sources = read(directory / "launch.json")["sources"]
    return dict(
        scope="pinned public-source review, read-only installed-file inventory and archived trace recomputation",
        limitations=[
            "Public source revisions are not proven matches for installed tiling or kernel binaries.",
            "Full-core mixed launch requests and internal barriers are not hardware residency measurements.",
            "ELF exports do not prove ABI, old-driver compatibility, memory visibility, progress or capture safety.",
            "This audit does not establish new model correctness, NPU performance or multigraph-exclusive value.",
        ],
        source_records=source_records(),
        installed_snapshots={name: digest(HERE / name) for name in ("installed-sdk.json", "installed-exports.json")},
        installed_driver=sdk["driver"]["fields"]["Version"],
        elf_symbols_found=len(exports["found_in_these_libraries"]),
        trace_records=[{k: t[k] for k in ("rank", "mode", "file", "sha256")} for t in report["traces"]],
        checked_archived_sources=sources,
        archived_analysis_sha256=digest(directory / "analysis.json"),
        analyzer_sha256={
            name: digest(ROOT / "benchmarks" / name)
            for name in (
                "aclgraph_two_layer_fused_analysis.py",
                "aclgraph_bounded_lead_analysis.py",
                "aclgraph_bounded_lead_probe.py",
                "aclgraph_full_path_budget.py",
            )
        },
        conditional_budget=conditional_budgets(report),
    )


if __name__ == "__main__":
    result = build()
    (HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "source_files": len(result["source_records"]),
                "trace_files": len(result["trace_records"]),
                "budget": result["conditional_budget"]["rows"],
            },
            indent=2,
        )
    )
