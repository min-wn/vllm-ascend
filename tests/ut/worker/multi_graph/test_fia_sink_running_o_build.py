# SPDX-License-Identifier: Apache-2.0
"""CPU-only isolation and dispatch contracts for the running-output builder."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def builder(monkeypatch):
    parent = Path(__file__).resolve().parents[4] / "benchmarks"
    monkeypatch.syspath_prepend(str(parent))
    patch_name = "aclgraph_fia_sink_running_o_patch"
    if not (parent / (patch_name + ".py")).exists():
        patch = ModuleType(patch_name)
        patch.__file__ = str(parent / (patch_name + ".py"))
        monkeypatch.setitem(sys.modules, patch_name, patch)
    spec = importlib.util.spec_from_file_location(
        "running_o_builder_cpu", parent / "aclgraph_fia_sink_running_o_build.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def configured(builder, tmp_path_factory, monkeypatch):
    # Shorten fake names to keep the generated Windows path bounded.
    monkeypatch.setattr(builder, "BINARY", "qualified_bf16")
    monkeypatch.setattr(builder, "OP", "fia")
    root = tmp_path_factory.mktemp("ro") / "experiment"
    build = root / "source/csrc/build"
    source = build / "binary/ascend910b/src"
    write(source / builder.OP / builder.WRAPPER, "generated wrapper")
    write(source / builder.OP / "op_kernel/fia_block_vec_nonquant_sink.h", "qualified kernel")
    param_data = {"op_list": [{"bin_filename": builder.BINARY, "inputs": [{"dtype": "bfloat16"}]}]}
    param = write(build / "binary/ascend910b/gen" / (builder.BINARY + "_param.json"), json.dumps(param_data))
    vendors = {}
    hashes = {param: builder.PARAM_HASH}
    for policy, (name, digest) in builder.POLICY_LIBRARIES.items():
        vendor = root / "vendors" / policy
        write(
            vendor / "op_impl/cpu/config/cust_aicpu_kernel.json",
            json.dumps({"FusedInferAttentionScoreV2SinkMetadata": {"opInfo": {"kernelSo": name}}}),
        )
        hashes[vendor / "op_impl/cpu/aicpu_kernel/impl" / name] = digest
        hashes[vendor / "op_api/lib/libcust_opapi.so"] = builder.OPAPI_HASH
        vendors[policy] = vendor
    monkeypatch.setattr(builder, "sha256", lambda path: hashes[Path(path)])
    return SimpleNamespace(
        root=root,
        build=build,
        source=source,
        param=param,
        param_data=param_data,
        vendors=vendors,
        hashes=hashes,
        output=root / "new-consumer",
    )


def preflight(builder, configured, output=None):
    return builder.preflight(
        configured.root,
        configured.vendors["upstream"],
        configured.vendors["whole"],
        configured.output if output is None else output,
    )


def test_preflight_accepts_fresh_isolated_output_without_mutating_inputs(builder, configured):
    original = {path: path.read_bytes() for path in configured.root.rglob("*") if path.is_file()}
    result = preflight(builder, configured)
    assert result == (
        configured.root.resolve(),
        configured.build.resolve(),
        configured.source.resolve(),
        configured.param.resolve(),
        configured.vendors,
        configured.output.resolve(),
    )
    assert not configured.output.exists()
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("location", ("outside", "root", "existing", "upstream", "whole", "build", "source"))
def test_output_must_be_fresh_and_not_overlap_qualified_inputs(builder, configured, location):
    paths = {
        "outside": configured.root.parent / "outside",
        "root": configured.root,
        "existing": configured.root / "existing",
        "upstream": configured.vendors["upstream"] / "new",
        "whole": configured.vendors["whole"] / "new",
        "build": configured.build / "new",
        "source": configured.source / "new",
    }
    (configured.root / "existing").mkdir()
    with pytest.raises(ValueError, match="escapes|fresh|overlaps"):
        preflight(builder, configured, paths[location])
    assert not configured.output.exists()


@pytest.mark.parametrize("policy", ("upstream", "whole"))
def test_dispatcher_cannot_silently_select_another_producer(builder, configured, policy):
    config = configured.vendors[policy] / "op_impl/cpu/config/cust_aicpu_kernel.json"
    write(config, json.dumps({"FusedInferAttentionScoreV2SinkMetadata": {"opInfo": {"kernelSo": "another.so"}}}))
    with pytest.raises(ValueError, match="dispatcher"):
        preflight(builder, configured)
    assert not configured.output.exists()


@pytest.mark.parametrize("policy", ("upstream", "whole"))
@pytest.mark.parametrize("library", ("producer", "opapi"))
def test_qualified_producer_and_common_opapi_hashes_are_required(builder, configured, policy, library):
    vendor = configured.vendors[policy]
    name = builder.POLICY_LIBRARIES[policy][0]
    relative = "op_impl/cpu/aicpu_kernel/impl/" + name if library == "producer" else "op_api/lib/libcust_opapi.so"
    path = vendor / relative
    configured.hashes[path] = "unexpected hash"
    with pytest.raises(ValueError, match="producer|OpAPI"):
        preflight(builder, configured)
    assert not configured.output.exists()


@pytest.mark.parametrize("change", ("float16", "float32", "another_binary"))
def test_other_dtype_or_artifact_parameters_are_rejected(builder, configured, change):
    data = configured.param_data
    if change == "another_binary":
        data["op_list"][0]["bin_filename"] = "unqualified_binary"
    else:
        data["op_list"][0]["inputs"][0]["dtype"] = change
    configured.param.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="qualified BF16"):
        preflight(builder, configured)
    assert not configured.output.exists()


def test_bf16_parameter_byte_drift_is_rejected_even_with_matching_dtype_and_name(builder, configured):
    configured.hashes[configured.param] = "unqualified compiler geometry or options"
    with pytest.raises(ValueError, match="compiler parameters"):
        preflight(builder, configured)
    assert not configured.output.exists()


@pytest.mark.parametrize("resident", (True, -1, 2, "1"))
def test_invalid_variant_is_rejected_before_creating_output(builder, configured, resident):
    args = SimpleNamespace(
        root=configured.root,
        upstream_vendor=configured.vendors["upstream"],
        whole_vendor=configured.vendors["whole"],
        output_dir=configured.output,
        resident=resident,
        timeout=60,
    )

    def no_compile(*args, **kwargs):
        pytest.fail("invalid variant must not invoke patching or compilation")

    with pytest.raises(ValueError, match="resident"):
        builder.build(args, runner=no_compile)
    assert not configured.output.exists()


@pytest.mark.parametrize("resident", (0, 1))
def test_wrapper_injects_one_compile_definition_and_preserves_generated_call(builder, resident):
    call = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    text = "def build():\n" + call + "\n"
    changed = builder.wrapper_options(text, resident)
    assert changed.count("-DTHESIS_FIA_RUNNING_O_UB=" + str(resident)) == 1
    assert changed.index("options.append") < changed.index(call)
    assert changed.count(call) == 1
    with pytest.raises(ValueError, match="already patched"):
        builder.wrapper_options(changed, resident)


@pytest.mark.parametrize("case", ("missing", "duplicated", "macro_already_present"))
def test_wrapper_drift_or_repeated_injection_is_rejected(builder, case):
    call = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    text = {
        "missing": "different generated wrapper",
        "duplicated": call + "\n" + call,
        "macro_already_present": "# THESIS_FIA_RUNNING_O_UB\n" + call,
    }[case]
    with pytest.raises(ValueError, match="compile anchor|already patched"):
        builder.wrapper_options(text, 1)


def test_compile_command_preserves_bf16_param_and_isolates_output_paths(builder, tmp_path):
    wrapper = tmp_path / "source with spaces" / "wrapper.py"
    param = tmp_path / "qualified bf16.json"
    output = tmp_path / "binary with spaces"
    command = builder.compile_command(wrapper, param, output)
    assert command == [
        "opc",
        str(wrapper),
        "--main_func=fused_infer_attention_score_v2_sink",
        "--input_param=" + str(param),
        "--soc_version=Ascend910B1",
        "--output=" + str(output),
        "--impl_mode=high_performance,optional",
        "--simplified_key_mode=0",
        "--op_mode=dynamic",
    ]


def test_tiling_key_csv_is_normalized_without_changing_numeric_identity(builder):
    assert builder.parse_tiling_keys("104000000010322223,2,0") == ("0", "2", "104000000010322223")
    assert builder.parse_tiling_keys(str((1 << 64) - 1)) == (str((1 << 64) - 1),)


@pytest.mark.parametrize(
    "value",
    ("", "1,", ",1", "1,,2", "01", "+1", "-1", "1.0", "0x1", "1,1", "1, 2", "１", "1\n", "18446744073709551616"),
)
def test_malformed_duplicate_or_out_of_range_tiling_key_csv_is_rejected(builder, value):
    with pytest.raises(ValueError, match="tiling key"):
        builder.parse_tiling_keys(value)


@pytest.mark.parametrize("keys", (("2", "1"), ("01",), ("1", "1"), (), ["1"], (1,), "1"))
def test_invalid_tiling_key_contract_is_rejected_before_output_or_compile(builder, configured, keys):
    args = SimpleNamespace(
        root=configured.root,
        upstream_vendor=configured.vendors["upstream"],
        whole_vendor=configured.vendors["whole"],
        output_dir=configured.output,
        resident=1,
        timeout=60,
        tiling_keys=keys,
    )

    def no_compile(*args, **kwargs):
        pytest.fail("invalid key selection must not invoke patching or compilation")

    with pytest.raises(ValueError, match="tiling key"):
        builder.build(args, runner=no_compile)
    assert not configured.output.exists()


def test_selected_tiling_keys_are_passed_to_opc_without_leaking_into_wrapper_options(builder, tmp_path):
    keys = ("104000000010322223", "104000000010322224")
    wrapper, param, output = (tmp_path / name for name in ("wrapper.py", "param.json", "binary"))
    base = builder.compile_command(wrapper, param, output)
    assert builder.compile_command(wrapper, param, output, keys) == [*base, "--tiling_key=" + ",".join(keys)]
    call = "    compile_op(src, origin_func_name, op_info, options, code_channel, '{}', {'valueDepend': {}})"
    assert "tiling_key" not in builder.wrapper_options(call, 1)


def test_binary_qualification_accepts_duplicate_aic_aiv_entries_but_marks_restricted_scope(builder):
    metadata = {
        "kernelList": [
            {"tilingKey": 2, "kernelType": "MIX_AIC"},
            {"tilingKey": "2", "kernelType": "MIX_AIV"},
            {"tilingKey": 12, "kernelType": "MIX_AIC"},
        ]
    }
    result = builder.binary_qualification(metadata, ("2", "12"))
    assert result["scope"] == "restricted_tiling_keys"
    assert result["requested_tiling_keys"] == result["actual_tiling_keys"] == ["2", "12"]
    assert result["kernel_entries"] == 3
    assert result["exact_selected_keys_verified"] is True
    assert result["arbitrary_shape_fallback_qualified"] is False
    assert result["npu_runtime_qualified"] is False


@pytest.mark.parametrize("actual", ((2,), (2, 12, 13)))
def test_missing_or_unrequested_compiled_keys_fail_qualification(builder, actual):
    metadata = {"kernelList": [{"tilingKey": key} for key in actual]}
    with pytest.raises(ValueError, match="differ from the requested"):
        builder.binary_qualification(metadata, ("2", "12"))


@pytest.mark.parametrize(
    "metadata",
    (
        None,
        {},
        {"kernelList": []},
        {"kernelList": [{}]},
        {"kernelList": [1]},
        {"kernelList": [{"tilingKey": True}]},
        {"kernelList": [{"tilingKey": "02"}]},
        {"kernelList": [{"tilingKey": 1 << 64}]},
    ),
)
def test_unproven_or_invalid_compiled_key_inventory_fails_qualification(builder, metadata):
    with pytest.raises(ValueError, match="kernel|tiling key"):
        builder.binary_qualification(metadata, ("2",))


def test_default_full_parameter_build_still_does_not_claim_arbitrary_shape_runtime_qualification(builder):
    result = builder.binary_qualification({"kernelList": [{"tilingKey": "2"}]}, None)
    assert result["scope"] == "all_parameter_tiling_keys"
    assert result["requested_tiling_keys"] is None
    assert result["actual_tiling_keys"] == ["2"]
    assert result["exact_selected_keys_verified"] is False
    assert result["arbitrary_shape_fallback_qualified"] is False
    assert result["npu_runtime_qualified"] is False
