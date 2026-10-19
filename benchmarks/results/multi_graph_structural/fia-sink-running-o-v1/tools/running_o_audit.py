"""Independent stdlib raw-read audit of a four-cell running-O source ablation.

No benchmark or previous audit modules are imported. Raw qualification is
separate from timings; one process per cell cannot establish a stable gain.
"""

import argparse
import hashlib
import json
import math
import statistics
import struct
from decimal import Decimal
from pathlib import Path, PurePosixPath, PureWindowsPath

CELLS = [("control", "upstream"), ("resident", "upstream"), ("resident", "whole"), ("control", "whole")]
CAPS = [400, 512]
CANONICAL = "0519bc80410bd15d0893e0283189425ec285d53003198c163183b79e6249e524"
PROBE = "b7472c5ccbdb37d4f5ea1693f2abc5145a71abb773d1e09ea788bf5daca964c9"
PATCH = "0de5b73e22626c825cec46448488c45f5511c223f719ac7d5029162732b9dba4"
OPAPI = "2c1660f393e0e8cc9c8d935929420e3e3b61297163994ade48eec5ac14c9ad40"
EXTENSION = "25a41e47473a3bc9c34a50c053766181c7a764ea496b93d4d3b70180b0ecf76b"
PARAM = "9384517184de3478d50eec0461cd78a5b1e1aeee2ccc86a6ae927b70cd474880"
TILING_KEYS = ["104000000010222223", "104000000010322223"]
BINARY = "FusedInferAttentionScoreV2Sink_cacc23924ba5ed25533f5fc5005e0673"
HEADER = "fused_infer_attention_score_v2_sink/op_kernel/fia_block_vec_nonquant_sink.h"
WRAPPER = "fused_infer_attention_score_v2_sink/FusedInferAttentionScoreV2Sink.py"
POLICIES = {
    "upstream": ("libfia_v2_sink_aicpu_kernels.so", "796983827bc6326a250009a2aacc810c44f65545ec2ae56dab517a5ec4e7111b"),
    "whole": ("libthesis_stage5_whole_aicpu.so", "1c4ab9fcf2b7d2935be3e60f2a215fb65dc85859bc6fe99fe5271b4c4784ea90"),
}
METRICS = ("wall_us", "queue_span_us", "preparation_enqueue_us", "submit_join_cpu_us")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0

    def require(self, condition, label):
        self.checks += 1
        if not condition:
            raise ValueError(label)

    def same(self, left, right, label):
        self.require(left == right, label)


def owned(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive or ".." in PurePosixPath(relative).parts:
        raise ValueError("invalid relative raw evidence path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("raw evidence path escapes bundle")
    return path


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def tensor(item, shape, audit, dtype="torch.bfloat16"):
    audit.same(item["shape"], shape, "complete tensor shape")
    audit.same(item["dtype"], dtype, "complete tensor dtype")
    digest = item["sha256"]
    audit.require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "complete tensor byte SHA256")


def plan_check(item, record, policy, audit):
    words = item["words"]
    audit.require(isinstance(words, list) and len(words) == 1024 and all(type(v) is int and 0 <= v < 2**31 for v in words), "full int32[1024] plan")
    audit.same(hashlib.sha256(struct.pack("<1024i", *words)).hexdigest(), item["sha256"], "full plan byte hash")
    maximum, count = max(record["kv_lengths"]) or 1024, record["physical_batch"]
    base = words[576:586]
    audit.same(base[:3] + base[6:], [512, min(512, maximum), 8, 4, maximum, count, count], "native plan true current geometry")
    audit.require(item["current_exact_base_verified"] is True and 1 <= base[4] <= 20, "plan current-base qualification")
    if policy == "whole":
        audit.same([base[3], base[5]], [0, 0], "whole plan has no FD")
        expected, begin = [0] * 1024, 0
        expected[576:586] = base
        for core in range(base[4]):
            slot = words[core * 10:core * 10 + 10]
            audit.require(slot[0] == 1 and slot[2:] == [0] * 8 and begin < slot[1] <= count * 2, "whole full-head advancing ownership")
            expected[core * 10:core * 10 + 10] = slot
            begin = slot[1]
        audit.same(begin, count * 2, "whole complete physical head coverage")
        for vector in range(base[4] * 2):
            expected[360 + 3 * vector] = 1
        audit.same(words, expected, "whole complete unused/FD payload")
    return words


def eligibility(words, record, audit):
    base = words[576:586]
    fd = {}
    for core in range(base[4]):
        slot = words[10 * core:10 * core + 10]
        if slot[6]:
            audit.require(slot[5] == 0 and 0 <= slot[4] < record["physical_batch"] * 2, "FD group true head coordinate")
            audit.require(slot[4] not in fd, "FD group declares one unique head")
            fd[slot[4]] = slot[6]
    audit.same(len(fd), base[3], "FD group count matches actual serialized plan")
    heads = [head for head in range(record["physical_batch"] * 2) if record["kv_lengths"][head // 2] > 0]
    eligible = [head for head in heads if head not in fd]
    tiles = lambda head: (record["kv_lengths"][head // 2] + base[1] - 1) // base[1]
    logical = sum(max(0, tiles(head) - 1) * 24 * 256 * 4 * 2 for head in eligible)
    return dict(positive_kv_heads=len(heads), fd_heads=len(fd), source_plan_eligible_complete_heads=len(eligible),
                eligible_head_fraction=len(eligible) / len(heads) if heads else 0.0,
                positive_kv_s2_tiles=sum(map(tiles, heads)), eligible_s2_tiles=sum(map(tiles, eligible)),
                logical_running_o_gm_read_write_bytes_avoidable_per_occurrence=logical,
                qualification_method="Pinned source guards plus actual 1024-word plan and Q4 geometry; no runtime eligibility counter",
                not_measured_hbm_traffic=True, fd_head_ids=sorted(fd))


def input_storage(report, audit):
    audit.same(report["input_identity"], report["post_input_identity"], "all private QKV immutable")
    audit.same(report["mask_identity"], report["post_mask_identity"], "full mask immutable")
    audit.same(report["frame_query_identity"], report["post_frame_query_identity"], "captured query frames immutable")
    value = report["input_identity"]
    audit.same(value["sha256"], identity(value["occurrences"]), "complete QKV aggregate hash")
    audit.same(len(value["occurrences"]), 16, "sixteen private occurrences")
    shapes = dict(query=[512, 12, 256], key=[1293, 128, 2, 256], value=[1293, 128, 2, 256])
    for occurrence in value["occurrences"]:
        audit.same(set(occurrence), set(shapes), "QKV exact key set")
        for name, shape in shapes.items():
            tensor(occurrence[name], shape, audit)
    tensor(report["mask_identity"], [2048, 2048], audit, "torch.bool")
    ranges = report["private_storage_ranges"]
    audit.same({(row["occurrence"], row["name"]) for row in ranges}, {(i, name) for i in range(16) for name in shapes}, "private QKV ranges cover each tensor once")
    audit.same(len(ranges), 48, "private QKV range count")
    for row in ranges:
        audit.require(type(row["start"]) is int and row["start"] > 0, "positive private address")
        audit.same(row["end"] - row["start"], math.prod(shapes[row["name"]]) * 2, "private exact tensor byte extent")
    intervals = sorted((row["start"], row["end"]) for row in ranges)
    audit.require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), "all private QKV nonoverlap")
    for cap in CAPS:
        frames = report["frame_query_identity"][str(cap)]
        audit.same(len(frames), 16, "sixteen captured frame Q tensors")
        for frame in frames:
            tensor(frame, [cap, 12, 256], audit)
        descriptors = report["captured_addresses"][str(cap)]
        shapes = dict(q_lengths=[cap // 4], kv_lengths=[cap // 4], block_table=[cap // 4, 15], atten_mask=[2048, 2048], metadata=[1024])
        shapes.update({name + str(i): [cap, 12, 256] for name in ("query", "output") for i in range(16)})
        audit.same(set(descriptors), set(shapes), "all captured descriptors")
        for name, shape in shapes.items():
            saved = descriptors[name]
            audit.same(saved["shape"], shape, "captured exact shape")
            audit.same(saved["stride"], [math.prod(shape[i + 1:]) for i in range(len(shape))], "captured contiguous stride")
            audit.require(type(saved["pointer"]) is int and saved["pointer"] > 0, "captured positive address")
        prep = report["prepared_descriptors"][str(cap)]
        expected = dict(mode="sink", physical_tokens=cap, physical_requests=cap // 4,
                        host_pinned=dict(q_lengths=True, kv_lengths=True, block_table=True),
                        stable_host_allocations=True, stable_device_allocations=True,
                        host_allocation_count=3, device_allocation_count=0,
                        setup_copies=dict(q_lengths=1, kv_lengths=0, block_table=0),
                        step_copies=dict(q_lengths=0, kv_lengths=18, block_table=18),
                        prepared_steps=18, completed_steps=18, inflight=False,
                        q4_invariant=dict(query_length=4, cumulative_q=list(range(4, cap + 1, 4))))
        for key, value in expected.items():
            audit.same(prep[key], value, "complete prepared-descriptor accounting " + key)
        for name in ("q_lengths", "kv_lengths", "block_table"):
            for field in ("pointer", "shape", "stride"):
                audit.same(prep["device"][name][field], descriptors[name][field], "prepared source equals captured descriptor")


def interval(event):
    start, duration = Decimal(str(event["ts"])), Decimal(str(event["dur"]))
    if not start.is_finite() or not duration.is_finite() or duration < 0:
        raise ValueError("nonfinite profiler interval")
    return start, start + duration


def duration(events):
    return float(sum((interval(event)[1] - interval(event)[0] for event in events), Decimal(0)))


def profile_check(document, profile, policy, audit):
    events = document["traceEvents"] if isinstance(document, dict) else document
    labels = [row for row in events if row.get("ph") == "X" and str(row.get("name", "")).startswith("consumer/")]
    audit.same([row["name"] for row in labels], [f"consumer/{policy}/cap{profile['capacity']}/repeat0"], "one actual profiled replay label")
    start, end = interval(labels[0])
    contained = [row for row in events if row.get("ph") == "X" and interval(row)[0] >= start and interval(row)[1] <= end + 1]
    hardware = [row for row in contained if row.get("args", {}).get("Task Type")]
    launches = [row for row in hardware if row["args"]["Task Type"] == "MODEL_EXECUTE"]
    completions = [row for row in hardware if row["args"]["Task Type"] == "NOTIFY_RECORD"]
    audit.same([len(launches), len(completions)], [1, 1], "one model launch/final notify")
    model, stream = completions[0]["args"]["Model Id"], completions[0]["args"]["Physic Stream Id"]
    audit.require(model != 4294967295, "valid model ID")
    begin, finish = interval(launches[0])[0], interval(completions[0])[1]
    tasks = [row for row in events if row.get("ph") == "X" and row.get("args", {}).get("Model Id") == model and interval(row)[0] >= begin and interval(row)[1] <= finish]
    audit.require(all(row["args"]["Physic Stream Id"] == stream for row in tasks), "one model physical stream")
    kernels = [row for row in tasks if str(row["args"]["Task Type"]).startswith("KERNEL_")]
    attention = sorted([row for row in kernels if row["args"]["Task Type"] == "KERNEL_MIX_AIC"], key=lambda row: interval(row)[0])
    metadata = [row for row in kernels if row["args"]["Task Type"] == "KERNEL_AICPU"]
    zero = [row for row in kernels if row["args"]["Task Type"] == "KERNEL_AIVEC"]
    audit.same([len(kernels), len(attention), len(metadata), len(zero)], [18, 16, 1, 1], "actual attention/producer/clear kernel counts")
    audit.require(all(interval(a)[1] <= interval(b)[0] for a, b in zip(attention, attention[1:])), "sixteen nonoverlapping attention kernels")
    audit.require(not any(row["args"]["Task Type"] in ("PCIE_DMA_SQE", "SDMA_SQE") for row in hardware), "no update DMA inside prepared label")
    audit.require(not any(row.get("cat") == "cpu_op" and row.get("name") == "npu::npu_fused_infer_attention_score" for row in contained), "no hidden legacy update")
    total = duration(attention)
    return dict(attention_kernels=16, attention_sum_us=total,
                between_attention_gap_us=float(interval(attention[-1])[1] - interval(attention[0])[0]) - total,
                producer_sum_us=duration(metadata), clear_sum_us=duration(zero),
                profile_scope="one replay after formal timings; does not estimate stable kernel latency")


def build_checks(base, seen, reports, deployed, audit):
    candidates = [(path, load(path)) for path in seen if path.name == "running-o-build.json"]
    builds = {}
    for variant, enabled in (("control", False), ("resident", True)):
        matching = [(path, value) for path, value in candidates if value.get("state") == "completed" and value.get("resident") is enabled]
        audit.same(len(matching), 1, "one completed selected build per macro")
        path, value = matching[0]
        audit.same(value["returncode"], 0, "successful native build")
        audit.same(value["compile_returncode"], 0, "successful compiler")
        audit.require(value["npu_work_executed"] is False and value["system_install_performed"] is False, "isolated compile only")
        audit.same(value["baseline_parameter"]["sha256"], PARAM, "frozen BF16 compiler parameters")
        audit.same(set(value["changed_source_files"]), {HEADER, WRAPPER}, "only header and macro wrapper changed")
        audit.same(value["source_before"][HEADER], "4f1a8185300f1f6bbf7728b778d47ce49dc929658b629d8e330582e49866f1e0", "pinned native vector header")
        audit.same(set(value["source_before"]), set(value["source_after"]), "no copied source files added/deleted")
        differences = {name for name in value["source_before"] if value["source_before"][name] != value["source_after"][name]}
        audit.same(differences, {HEADER, WRAPPER}, "complete source inventory mutation scope")
        audit.same(set(value["artifacts"]), {BINARY + ".o", BINARY + ".json"}, "selected BF16 binary pair")
        binary_scope = value["binary_qualification"]
        audit.same(binary_scope["scope"], "restricted_tiling_keys", "binary restricted-key qualification scope")
        audit.same(binary_scope["requested_tiling_keys"], TILING_KEYS, "exact requested two tiling keys")
        audit.same(binary_scope["actual_tiling_keys"], TILING_KEYS, "exact compiled two tiling keys")
        audit.require(binary_scope["exact_selected_keys_verified"] is True and binary_scope["arbitrary_shape_fallback_qualified"] is False,
                      "restricted consumer does not qualify arbitrary fallback geometry")
        audit.require(binary_scope["npu_runtime_qualified"] is False, "build-stage qualification correctly excludes NPU execution")
        metadata = value["binary_metadata"]
        kernels = metadata["kernelList"]
        actual_keys = sorted({str(row["tilingKey"]) for row in kernels}, key=int)
        audit.same(actual_keys, TILING_KEYS, "actual raw compiled kernelList keys")
        audit.same(binary_scope["kernel_entries"], len(kernels), "actual kernel entry count")
        audit.same([arg for arg in value["command"] if arg.startswith("--tiling_key")],
                   ["--tiling_key=" + ",".join(TILING_KEYS)], "one exact restricted compiler tiling-key option")
        for policy in POLICIES:
            audit.same(value["vendors"][policy]["available_tiling_keys"], TILING_KEYS, "both staged vendors declare same restricted keys")
            audit.require(value["vendors"][policy]["arbitrary_shape_fallback_qualified"] is False, "staged vendor excludes arbitrary shape fallback")
        builds[variant] = value
    control, resident = builds["control"], builds["resident"]
    audit.same(control["source_before"], resident["source_before"], "identical copied baseline sources")
    audit.same({k: v for k, v in control["source_after"].items() if k != WRAPPER}, {k: v for k, v in resident["source_after"].items() if k != WRAPPER}, "same patched sources except macro wrapper")
    by_hash = {}
    for path in seen:
        by_hash.setdefault(sha(path), []).append(path)
    wrappers = {}
    for variant, enabled in (("control", 0), ("resident", 1)):
        build = builds[variant]
        for name in (HEADER, WRAPPER):
            audit.require(build["source_after"][name] in by_hash, "raw exact post-build header/wrapper snapshot manifested")
        binary_json_hash = build["artifacts"][BINARY + ".json"]["sha256"]
        audit.require(binary_json_hash in by_hash, "actual compiled binary metadata raw bytes manifested")
        for binary_json_path in by_hash[binary_json_hash]:
            audit.same(load(binary_json_path), build["binary_metadata"], "actual compiled binary JSON matches embedded report metadata")
        raw = by_hash[build["source_after"][WRAPPER]][0].read_text(encoding="utf-8")
        marker = f'    options.append("-DTHESIS_FIA_RUNNING_O_UB={enabled}")\n'
        audit.same(raw.count(marker), 1, "correct isolated compile macro")
        wrappers[variant] = raw.replace(marker, "")
        for policy in POLICIES:
            artifact = reports[variant, policy]["opp_artifacts"]
            for suffix in (".o", ".json"):
                relative = "op_impl/ai_core/tbe/kernel/ascend910b/fused_infer_attention_score_v2_sink/" + BINARY + suffix
                audit.same(artifact[relative], build["artifacts"][BINARY + suffix]["sha256"], "same selected compiled consumer for both producer policies")
    changed_consumer = {"op_impl/ai_core/tbe/kernel/ascend910b/fused_infer_attention_score_v2_sink/" + BINARY + suffix for suffix in (".o", ".json")}
    for policy in POLICIES:
        audit.same({key: value for key, value in reports["control", policy]["opp_artifacts"].items() if key not in changed_consumer},
                   {key: value for key, value in reports["resident", policy]["opp_artifacts"].items() if key not in changed_consumer},
                   "all staged artifacts except selected consumer pair unchanged by macro ablation")
    for variant in ("control", "resident"):
        dispatcher = "op_impl/cpu/config/cust_aicpu_kernel.json"
        whole_library = "op_impl/cpu/aicpu_kernel/impl/" + POLICIES["whole"][0]
        upstream_artifacts = reports[variant, "upstream"]["opp_artifacts"]
        whole_artifacts = reports[variant, "whole"]["opp_artifacts"]
        audit.require(whole_library not in upstream_artifacts, "upstream vendor does not contain extra whole producer")
        audit.same(whole_artifacts[whole_library], POLICIES["whole"][1], "whole-only extra immutable producer exact bytes")
        audit.same(set(whole_artifacts) - set(upstream_artifacts), {whole_library}, "only one permitted extra whole producer library")
        audit.same(set(upstream_artifacts) - set(whole_artifacts), set(), "whole vendor does not omit upstream artifacts")
        audit.same({key: value for key, value in upstream_artifacts.items() if key != dispatcher},
                   {key: value for key, value in whole_artifacts.items() if key not in (dispatcher, whole_library)},
                   "producer vendors differ only by dispatcher and exact immutable whole producer")
    audit.same(wrappers["control"], wrappers["resident"], "wrappers differ only by macro value")
    audit.require(control["artifacts"][BINARY + ".o"]["sha256"] != resident["artifacts"][BINARY + ".o"]["sha256"], "ablation emits different consumer objects")
    audit.same(deployed["aclgraph_fia_sink_running_o_patch.py"], PATCH, "frozen measured patch source")
    return {variant: dict(object_sha256=build["artifacts"][BINARY + ".o"]["sha256"], header_sha256=build["source_after"][HEADER], wrapper_sha256=build["source_after"][WRAPPER],
                          restricted_tiling_keys=TILING_KEYS, raw_binary_metadata_verified=True,
                          arbitrary_shape_fallback_qualified=False,
                          build_stage_npu_runtime_qualified=False) for variant, build in builds.items()}


def audit_bundle(base, audit):
    base = base.resolve(strict=True)
    collection = load(base / "collection.json")
    audit.require(collection["partial"] is False and collection["controller_interrupted"] is False,
                  "only a complete uninterrupted selected collection can qualify")
    manifest = load(base / "manifest.json")
    seen = set()
    for item in manifest:
        path = owned(base, item["path"])
        audit.require(path not in seen, "unique manifested raw file")
        audit.same(sha(path), item["sha256"], "manifest full byte hash " + item["path"])
        seen.add(path)
    suites = [path for path in seen if path.name == "running-o-screen.json"]
    audit.same(len(suites), 1, "one selected screen suite")
    suite_path = suites[0]
    suite = load(suite_path)
    tool_manifests = [path for path in seen if path.name == "deployed-tools.json"]
    audit.same(len(tool_manifests), 1, "one frozen selected tools manifest")
    deployed = load(tool_manifests[0])
    audit.same(deployed["aclgraph_fia_sink_consumer_probe.py"], PROBE, "unchanged consumer probe")
    audit.same(suite["state"], "completed", "complete selected screen")
    audit.same(suite["returncode"], 0, "successful screen")
    audit.require(suite["performance_measured"] is True, "formal timings exist")
    for flag in ("full_model_qualified", "full_trace_measured", "end_to_end_gain_established"):
        audit.require(suite[flag] is False, "scope excludes " + flag)
    audit.same(suite["design"], dict(cells=[list(pair) for pair in CELLS], trial=0, seed=1703, warmups=5, repeats=10, profile_repeats=1, steps=[0, 29], capacities=CAPS, occurrences=16), "predeclared fixed four-cell screen")
    for name, digest in suite["source_sha256"].items():
        audit.same(deployed[name], digest, "driver frozen dependency " + name)
    audit.same(len(suite["runs"]), 4, "four fresh child processes")
    reports, means, profiles, outputs, plans, qualifications = {}, {}, {}, {}, {}, {}
    parent_pids, child_pids = set(), set()
    first, rows_count, timings_count, max_error, plan_checks = None, 0, 0, 0.0, 0
    previous = suite["started_unix"]
    for index, (entry, (variant, policy)) in enumerate(zip(suite["runs"], CELLS, strict=True)):
        audit.same([entry["variant"], entry["policy"], entry["process_order_index"]], [variant, policy, index], "fixed cell process order")
        audit.require(previous <= entry["launched_unix"] < entry["finished_unix"] <= suite["finished_unix"], "serial fresh-process intervals")
        previous = entry["finished_unix"]
        path = (suite_path.parent / entry["report_path"]).resolve(strict=True)
        audit.require(path.is_relative_to(base) and path in seen, "raw process report manifested")
        audit.same(sha(path), entry["report_sha256"], "suite binds exact process report bytes")
        report = load(path)
        reports[variant, policy] = report
        audit.same(entry["returncode"], 0, "successful child process")
        command = entry["command"]
        for option, wanted in (("--mode", policy), ("--trial", "0"), ("--repeats", "10"), ("--warmups", "5"), ("--profile-repeats", "1"), ("--seed", "1703")):
            audit.same(command.count(option), 1, "one original CLI option " + option)
            audit.same(command[command.index(option) + 1], wanted, "exact frozen CLI option " + option)
        for field, bag, value in (("launcher", parent_pids, entry["pid"]), ("native", child_pids, report["child_pid"])):
            audit.require(type(value) is int and value > 0 and value not in bag, "unique " + field + " PID")
            bag.add(value)
        expected = dict(state="completed", stage="completed", returncode=0, mode=policy, policy=policy,
                        trial=0, repeats=10, occurrences=16, profile_repeats=1, warmup_replays=5, warmup_count=5,
                        core_limit=dict(cube_core_num=20, vector_core_num=40), npu_work_executed=True,
                        performance_measured=True, canonical_digest=CANONICAL, independent_graph_pool_per_bucket=True,
                        separate_qualification=True, timed_samples_individually_qualified=False,
                        frame_queries_match_base_prefixes=True, full_model_qualified=False,
                        full_trace_measured=False, end_to_end_gain_established=False)
        for key, value in expected.items():
            audit.same(report[key], value, "bounded probe report " + key)
        for name, digest in report["scripts"].items():
            audit.same(deployed[name], digest, "runtime frozen dependency " + name)
        audit.same(report["scripts"]["aclgraph_fia_sink_consumer_probe.py"], PROBE, "actual unchanged probe")
        audit.same(report["extension_sha256"], EXTENSION, "common extension")
        provenance = report["policy_provenance"]
        name, digest = POLICIES[policy]
        audit.same(provenance["policy"], policy, "actual policy dispatch")
        audit.same(PurePosixPath(provenance["library"]).name, name, "actual metadata library")
        audit.same(provenance["library_sha256"], digest, "unchanged metadata producer bytes")
        audit.same(provenance["opapi_sha256"], OPAPI, "unchanged OpAPI")
        artifact = report["opp_artifacts"]
        audit.same(artifact["op_api/lib/libcust_opapi.so"], OPAPI, "staged OpAPI bytes")
        audit.same(artifact["op_impl/cpu/aicpu_kernel/impl/" + name], digest, "staged producer bytes")
        native = report["final_native_provenance"]
        audit.require(native["alternate_sink_symbols_absent"] is True and native["selected_library_handle_retained"] is True, "no alternative native symbol owner")
        selected_opapi = str(PurePosixPath(provenance["library"]).parents[4] / "op_api/lib/libcust_opapi.so")
        audit.same({row["symbol"] for row in native["sink_symbols"]}, {"aclnnFusedInferAttentionScoreV2SinkV3GetWorkspaceSize", "aclnnFusedInferAttentionScoreV2SinkV3", "aclnnFusedInferAttentionScoreV2SinkMetadataGetWorkspaceSize", "aclnnFusedInferAttentionScoreV2SinkMetadata"}, "all four measured native sink symbols")
        audit.require(len(native["sink_symbols"]) == 4 and all(row["source"] == selected_opapi for row in native["sink_symbols"]), "all sink symbols come from selected OpAPI")
        input_storage(report, audit)
        keys = ("input_identity", "mask_identity", "frame_query_identity", "selected_records", "scripts", "extension_sha256", "torch_version", "torch_npu_version", "device_name", "trace_contract")
        if first is None:
            first = report
        for key in keys:
            audit.same(report[key], first[key], "cross-cell common exact identity " + key)
        for field, filename in (("trace_file_sha256", "structural-dynamic-trace-v1.json"), ("selection_sha256", "q4-tile-full-replay-v8-queue0.rank0.json")):
            audit.same(report["trace_contract"][field], deployed[filename], "unchanged canonical source " + field)
        records = {item["physical_tokens"]: item for item in report["selected_records"]}
        audit.same(sorted(records), CAPS, "two original record capacities")
        audit.same([records[cap]["step"] for cap in CAPS], [0, 29], "two original trace steps")
        audit.same([records[cap]["batch"] for cap in CAPS], [98, 128], "original real batch sizes")
        for item in report["metadata_preflight"]:
            cap = item["capacity"]
            audit.same(item["record"], records[cap], "preflight exact descriptor record")
            words = plan_check(item, records[cap], policy, audit)
            plan_checks += 1
            key = policy, cap
            if key in plans:
                audit.same(words, plans[key], "control/resident full 1024-word plans unchanged")
            plans[key] = words
            qualifications[key] = eligibility(words, records[cap], audit)
        for phase in ("pre_checks", "post_checks"):
            audit.same(len(report[phase]), 2, "two separate qualification records")
            for item in report[phase]:
                cap, record = item["capacity"], records[item["capacity"]]
                audit.same([item["step"], item["private_occurrences"]], [record["step"], 16], "qualification identity")
                for flag in ("finite", "all_outputs_finite", "exact_zero_padding", "stable_addresses", "exact_descriptors_read_only"):
                    audit.require(item[flag] is True, "native qualification " + flag)
                audit.same([item["legacy_update_calls_delta"], item["legacy_updated_tasks_delta"]], [0, 0], "sink has no substituted legacy update")
                audit.same(plan_check(item["plan"], record, policy, audit), plans[policy, cap], "actual replay plan equals eager/current cross-variant plan")
                plan_checks += 1
                rows = item["fp32_sampled_rows"]
                wanted = {(occ, row) for occ in (0, 7, 15) for row in (0, record["batch"] // 2, record["batch"] - 1)}
                audit.same({(row["occurrence"], row["row"]) for row in rows}, wanted, "FP32 exact sampled identities")
                audit.same(len(rows), 9, "nine independent FP32 rows")
                for row in rows:
                    length = record["kv_lengths"][row["row"]]
                    pages = record["block_table"][row["row"]][:(length + 127) // 128]
                    audit.same([row["kv_length"], row["original_page_ids"], row["local_page_ids"]], [length, pages, list(range(len(pages)))], "FP32 oracle uses current ordered true pages")
                    audit.require(finite(row["max_abs_error"]), "finite FP32 error")
                    max_error = max(max_error, row["max_abs_error"])
                    rows_count += 1
                audit.same(len(item["output_hashes"]), 16, "all private output hashes")
                for output in item["output_hashes"]:
                    tensor(output, [cap, 12, 256], audit)
                outputs[variant, policy, cap, phase] = item["output_hashes"]
        audit.same(report["warmups"], [dict(capacity=cap, replays=5, outside_timing=True) for cap in CAPS], "fixed warmups outside formal timings")
        samples = report["samples"]
        audit.same(len(samples), 20, "twenty unfiltered timings")
        audit.same([(row["repeat"], row["capacity"]) for row in samples], [(repeat, cap) for repeat in range(10) for cap in (CAPS if repeat % 2 == 0 else CAPS[::-1])], "fixed alternating sample order")
        for row in samples:
            record = records[row["capacity"]]
            audit.same([row["trial"], row["mode"], row["policy"], row["step"], row["batch"]], [0, policy, policy, record["step"], record["batch"]], "formal timing original identity")
            audit.require(all(finite(row[name]) for name in METRICS) and row["wall_us"] > 0, "all formal metrics finite")
            audit.require(row["preparation_enqueue_us"] + row["submit_join_cpu_us"] <= row["wall_us"] + 1e-6, "full wall endpoint same-clock subcosts")
            audit.same([row["legacy_update_calls_delta"], row["legacy_updated_tasks_delta"]], [0, 0], "formal sink producer retained")
            timings_count += 1
        for cap in CAPS:
            means[variant, policy, cap] = {metric: statistics.mean(row[metric] for row in samples if row["capacity"] == cap) for metric in METRICS}
            wall = [row["wall_us"] for row in samples if row["capacity"] == cap]
            means[variant, policy, cap].update(wall_median_us=statistics.median(wall),
                                              wall_min_us=min(wall), wall_max_us=max(wall),
                                              wall_range_us=max(wall)-min(wall))
        audit.same(len(report["profiles"]), 2, "two raw actual device profiles")
        for item in report["profiles"]:
            profile_path = (path.parent / item["path"]).resolve(strict=True)
            audit.require(profile_path.is_relative_to(base) and profile_path in seen, "profile raw bytes manifested")
            audit.same(sha(profile_path), item["sha256"], "profile raw byte hash")
            audit.same(profile_path.stat().st_size, item["bytes"], "profile exact byte length")
            audit.same([item["repeats"], item["outside_timing"], item["preparation_in_trace_outside_label"]], [1, True, True], "one diagnostic profile outside formal endpoint")
            profiles[variant, policy, item["capacity"]] = profile_check(load(profile_path), item, policy, audit)
    audit.require(parent_pids.isdisjoint(child_pids), "disjoint launcher/native PIDs")
    audit.same([timings_count, rows_count], [80, 144], "complete timing and FP32 counts")
    build_summary = build_checks(base, seen, reports, deployed, audit)
    full_output_identity = []
    for policy in POLICIES:
        for cap in CAPS:
            control = outputs["control", policy, cap, "pre_checks"]
            resident = outputs["resident", policy, cap, "pre_checks"]
            pre_post = {variant: outputs[variant, policy, cap, "pre_checks"] == outputs[variant, policy, cap, "post_checks"] for variant in ("control", "resident")}
            full_output_identity.append(dict(policy=policy, capacity=cap,
                control_resident_all_16_outputs_bitwise_equal=control == resident,
                matching_occurrences=sum(a == b for a, b in zip(control, resident)),
                pre_post_bitwise_equal=pre_post,
                numerical_scope="Full-output hashes establish equality only when identical; otherwise rely on separate frozen-probe sampled FP32 assert_close, not full-output error proof"))
    differences, interactions = [], []
    for cap in CAPS:
        for policy in POLICIES:
            old, new = means["control", policy, cap]["wall_us"], means["resident", policy, cap]["wall_us"]
            differences.append(dict(policy=policy, capacity=cap, control_wall_us=old, resident_wall_us=new,
                                    saved_wall_us=old-new, reduction_percent=100*(old-new)/old,
                                    control_wall_median_us=means["control", policy, cap]["wall_median_us"],
                                    resident_wall_median_us=means["resident", policy, cap]["wall_median_us"],
                                    median_reduction_percent=100*(means["control", policy, cap]["wall_median_us"]-means["resident", policy, cap]["wall_median_us"])/means["control", policy, cap]["wall_median_us"],
                                    profile_attention_saved_us=profiles["control", policy, cap]["attention_sum_us"]-profiles["resident", policy, cap]["attention_sum_us"]))
        old_gap = means["control", "upstream", cap]["wall_us"]-means["control", "whole", cap]["wall_us"]
        new_gap = means["resident", "upstream", cap]["wall_us"]-means["resident", "whole", cap]["wall_us"]
        interactions.append(dict(capacity=cap, control_upstream_minus_whole_us=old_gap,
                                 resident_upstream_minus_whole_us=new_gap,
                                 interaction_us_resident_minus_control=new_gap-old_gap))
    after_files = [path for path in seen if path.name == "device-after.txt"]
    audit.same(len(after_files), 1, "final device-idle raw evidence")
    audit.require("No running processes found in NPU 0" in after_files[0].read_text(), "NPU0 idle after selected screen")
    observed_control = reports["control", "whole"]["samples"]
    observed_maximum = max((row for row in observed_control if row["capacity"] == 512), key=lambda row: row["wall_us"])
    baseline_median = means["control", "whole", 512]["wall_median_us"]
    outlier_observation = dict(policy="whole", capacity=512, control_maximum_sample=observed_maximum,
                              control_median_wall_us=baseline_median,
                              control_maximum_minus_median_us=observed_maximum["wall_us"]-baseline_median,
                              control_maximum_over_median=observed_maximum["wall_us"]/baseline_median,
                              control_mean_wall_us=means["control", "whole", 512]["wall_us"],
                              resident_mean_wall_us=means["resident", "whole", 512]["wall_us"],
                              resident_median_wall_us=means["resident", "whole", 512]["wall_median_us"],
                              profile_attention_saved_us=profiles["control", "whole", 512]["attention_sum_us"]-profiles["resident", "whole", 512]["attention_sum_us"],
                              mean_wall_profile_mismatch_flag=True,
                              samples_removed=0,
                              interpretation="The control whole/cap512 maximum has a large CPU submit/join component. Mean wall saving exceeds median and single-profile attention saving; do not attribute the full mean difference to the resident kernel or call it a stable 5.32% gain.")
    return dict(qualified=True, assertions=audit.checks, manifest_files=len(seen), process_count=4,
                timings=timings_count, sampled_fp32_rows=rows_count, maximum_fp32_absolute_error=max_error,
                complete_plan_checks=plan_checks, same_policy_1024_word_plans_unchanged=True,
                all_private_qkv_query_mask_bytes_match=True, output_identity=full_output_identity,
                builds=build_summary,
                timing=[dict(variant=v, policy=p, capacity=c, samples=10, **row) for (v,p,c), row in means.items()],
                profile=[dict(variant=v, policy=p, capacity=c, **row) for (v,p,c), row in profiles.items()],
                consumer_changes=differences, policy_interactions=interactions,
                wall_outlier_observation=outlier_observation,
                eligibility=[dict(policy=p, capacity=c, **row) for (p,c), row in qualifications.items()],
                source=dict(bundle=str(base), suite_sha256=sha(suite_path), auditor_sha256=sha(__file__)),
                npu0_idle_after=True, process_replicates_per_cell=1,
                stable_performance_gain_established=False, full_trace_measured=False,
                full_model_qualified=False, end_to_end_gain_established=False,
                limitations=["Four fresh processes in one fixed order; order, addresses, frequency and temperature remain confounded.",
                             "Synthetic sixteen-attention chain with two original records; no dynamic/full-model/serving or split-over-padding claim.",
                             "All formal samples retained; pre/post qualification and profiler replays are separate from timed replay.",
                             "Full output bitwise equality is reported as observed, never inferred from a few FP32 samples.",
                             "Eligibility and avoidable workspace bytes are source/plan deductions, not runtime counters or measured HBM bytes.",
                             "One profiler replay per capacity is attribution evidence only."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_bundle(args.bundle, Audit())
    if args.output:
        if args.output.exists():
            raise ValueError("refusing to overwrite audit evidence")
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="ascii")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
