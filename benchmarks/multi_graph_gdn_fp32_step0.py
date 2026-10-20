# SPDX-License-Identifier: Apache-2.0
"""Bounded FP32 attribution on the unchanged full-batch Q4 step zero.

Instrumentation is eager and untimed. It neither qualifies the continued
trajectory nor changes native kernels, precision, thresholds or scheduling.
"""

import hashlib
from contextlib import contextmanager

from multi_graph_gdn_cpu_audit import identical, zero
from multi_graph_gdn_full_numeric import SCREEN_NRMSE, OfficialBanks, error_record
from multi_graph_gdn_q4_full_metadata import decode_values
from multi_graph_gdn_q4_full_probe import complete_snapshot, deterministic_token_ids, restore_complete
from multi_graph_gdn_q4_log_backend import CompactSSMRegistry, installed_log_registry


def tensor_digest(torch, value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def token_error(torch, actual, reference, *, label, capacity, step):
    """Normalize flattened head/norm tensors without losing token/owner order."""
    if actual.shape != reference.shape or actual.numel() % capacity:
        raise ValueError(f"observation lost its token axis: {label}")
    live = step["batch"] * 4
    a, b = (v.reshape(capacity, -1)[:live].reshape(step["batch"], 4, -1) for v in (actual, reference))
    row = error_record(torch, a, b, label=label, step=0, owners=list(step["request_ids"]))
    row.update(
        differing_elements=int((a != b).sum().item()),
        bitwise_identical=identical(torch, a, b),
        shape=list(actual.shape),
        dtype=str(actual.dtype),
    )
    return row


def tensors(value, path=""):
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensors(item, f"{path}/{index}")
    elif hasattr(value, "detach"):
        yield path, value


class StepObserver:
    """Copy hook observations immediately, before fused residual updates."""

    def __init__(self, torch, step, capacity):
        self.torch, self.step, self.capacity = torch, step, capacity
        self.reference, self.errors, self.recurrence, self.observed_recurrence = {}, [], {}, {}
        self.phase = "reference"

    def observe(self, label, value):
        for suffix, tensor in tensors(value):
            name = label + suffix
            saved = tensor.detach().to(device="cpu", copy=True)
            if self.phase == "reference":
                if name in self.reference:
                    raise RuntimeError(f"repeated reference observation: {name}")
                self.reference[name] = saved
            else:
                if name not in self.reference:
                    raise RuntimeError(f"unpaired observation: {name}")
                if "/recurrence/" in name:
                    self.observed_recurrence[name] = saved
                self.errors.append(
                    token_error(
                        self.torch, saved, self.reference[name], label=name, capacity=self.capacity, step=self.step
                    )
                )

    @contextmanager
    def installed(self, model, indices=None):
        handles = []
        layers = model.language_model.model.layers
        try:
            for index in range(len(layers)) if indices is None else indices:
                layer = layers[index]
                prefix = f"layer{index}"

                def entry(module, inputs, kwargs, prefix=prefix):
                    self.observe(prefix + "/input/hidden", kwargs.get("hidden_states", inputs[0] if inputs else None))
                    self.observe(prefix + "/input/residual", kwargs.get("residual"))

                handles.append(layer.register_forward_pre_hook(entry, with_kwargs=True))
                targets = [
                    ("input_norm", layer.input_layernorm),
                    ("post_attention_norm", layer.post_attention_layernorm),
                    ("mlp", layer.mlp),
                    ("decoder", layer),
                ]
                for name in ("gate_up_proj", "act_fn", "down_proj"):
                    module = getattr(layer.mlp, name, None)
                    if module is not None:
                        targets.append(("mlp/" + name, module))
                if layer.layer_type == "linear_attention":
                    gdn = layer.linear_attn
                    targets.append(("attention", gdn))
                    for name in ("in_proj_qkv", "in_proj_z", "in_proj_qkvz", "in_proj_ba", "norm", "out_proj"):
                        module = getattr(gdn, name, None)
                        if module is not None:
                            targets.append(("gdn/" + name, module))

                    def norm_input(module, inputs, prefix=prefix):
                        self.observe(prefix + "/gdn/norm_input", inputs)

                    handles.append(gdn.norm.register_forward_pre_hook(norm_input))
                else:
                    targets.append(("attention", layer.self_attn))
                for name, module in targets:

                    def output(module, inputs, result, label=prefix + "/" + name):
                        self.observe(label, result)

                    handles.append(module.register_forward_hook(output))
            if indices is None:

                def endpoint(module, inputs, result):
                    self.observe("final_norm", result)

                handles.append(model.language_model.model.norm.register_forward_hook(endpoint))
            yield
        finally:
            for handle in reversed(handles):
                handle.remove()

    @contextmanager
    def tap_recurrence(self, entries):
        import vllm_ascend.ops.gdn as gdn

        original, visits = gdn.recurrent_gated_delta_rule, []

        def call(query, key, value, initial_state, **kwargs):
            if len(visits) >= len(entries):
                raise RuntimeError("unexpected extra recurrence")
            item = entries[len(visits)]
            visits.append(item.index)
            inputs = dict(query=query, key=key, value=value, **kwargs)
            if self.phase == "reference":
                self.recurrence[item.index] = {
                    name: val.detach().cpu().clone() if hasattr(val, "detach") else val for name, val in inputs.items()
                }
            else:
                expected = self.recurrence[item.index]
                for name, tensor in inputs.items():
                    if name in ("query", "key", "value", "g", "beta"):
                        self.observe(f"layer{item.index}/recurrence/input/{name}", tensor)
                    elif hasattr(tensor, "detach"):
                        if not identical(self.torch, tensor.cpu(), expected[name]):
                            raise AssertionError(f"recurrence metadata differs: layer{item.index}/{name}")
                    elif tensor != expected[name]:
                        raise AssertionError(f"recurrence option differs: layer{item.index}/{name}")
            if self.phase == "reference":
                for name in ("query", "key", "value", "g", "beta"):
                    self.observe(f"layer{item.index}/recurrence/input/{name}", inputs[name])
            result = original(query, key, value, initial_state, **kwargs)
            self.observe(f"layer{item.index}/recurrence/output", result)
            return result

        gdn.recurrent_gated_delta_rule = call
        try:
            yield
            if visits != [item.index for item in entries]:
                raise AssertionError("recurrence traversal differs")
        finally:
            gdn.recurrent_gated_delta_rule = original


def audit_start(torch, bundle, banks, initial, frame, step):
    values = decode_values(step, frame.capacity_tokens, bundle.pages)
    for request, length in zip(step["request_ids"], step["seq_lengths"], strict=True):
        if length - 4 != bundle.trace.initial_prefix_by_request[request]:
            raise AssertionError("step-zero position does not immediately follow its coherent prefix")
    for name, expected in (("positions", values["positions"]), ("slot_mapping", values["slots"])):
        if getattr(frame, name).cpu().tolist() != expected:
            raise AssertionError(f"decode input differs: {name}")
    rows = []
    for item, snapshot, conv in zip(bundle.gdn_entries, initial["compact"].states, initial["conv"], strict=True):
        base, logs, valid = snapshot.tensors
        dense = banks.dense[item.index]
        if not identical(torch, dense[1:].view(128, 4, 24, 128, 128), base[1:, None].expand(-1, 4, -1, -1, -1)):
            raise AssertionError(f"initial accepted candidate mismatch: layer{item.index}")
        if not identical(torch, conv, banks.conv[item.index].cpu()):
            raise AssertionError(f"initial convolution mismatch: layer{item.index}")
        if any(not zero(torch, value) for value in (base[0], dense[0], logs, valid)):
            raise AssertionError("initial null/log state is not empty")
        if snapshot.last_step != -1 or snapshot.pending is not None or any(snapshot.generations):
            raise AssertionError("initial compact host lifecycle is not new")
        lookup = item.state.device_owner_lookup.cpu().tolist()
        if lookup != [0, *(1 + owner for owner in range(128) for _ in range(4))]:
            raise AssertionError("candidate/owner mapping differs")
        rows.append(
            dict(layer=item.index, base_sha256=tensor_digest(torch, base), conv_sha256=tensor_digest(torch, conv))
        )
    fia_rows = []
    for item, pair in zip(bundle.fia_entries, initial["fia"], strict=True):
        for label, saved, reference in zip(("key", "value"), pair, banks.fia[item.index], strict=True):
            if not identical(torch, saved, reference.cpu()):
                raise AssertionError(f"initial FIA history mismatch: layer{item.index}/{label}")
            fia_rows.append(dict(layer=item.index, field=label, sha256=tensor_digest(torch, saved)))
    meta = frame.mapping[bundle.gdn_entries[0].prefix]
    spec = meta.spec_decode_metadata
    accepted = spec.spec_causal_conv1d.num_accepted_tokens.cpu().tolist()
    ids = meta.spec_state_indices_tensor.cpu().reshape(-1).tolist()
    if accepted != values["accepted"] or ids != [slot for row in values["gdn_blocks"] for slot in row]:
        raise AssertionError("official accepted counts or candidate IDs differ from owner contract")
    return dict(
        passed=True,
        gdn=rows,
        fia=fia_rows,
        input_values=values,
        actual_seq_lengths=spec.actual_seq_lengths.cpu().tolist(),
        candidate_ids=ids,
        accepted=accepted,
        all_initial_candidates_equal=True,
        common_weights_and_input_embedding=True,
        output_order="first B*4 tokens reshape(B,4,-1), owner axis in unchanged request_ids order",
    )


def select_sites(rows, gdn_indices):
    """At most two local sites: first differing recurrence and first failed block."""
    recurrent = [r for r in rows if r["label"].endswith("/recurrence/output")]
    first = next((r for r in recurrent if r["differing_elements"]), None)
    failed = next((r for r in rows if "/decoder/" in r["label"] and not r["screen_passed"]), None)
    indices = []
    for row in (first, failed):
        if row is not None:
            index = int(row["label"].split("/")[0][5:])
            index = max(i for i in gdn_indices if i <= index)
            if index not in indices:
                indices.append(index)
    return dict(first_differing_recurrence=first, first_failed_decoder=failed, local_layer_indices=indices)


@contextmanager
def layer_context(model, bundle, frame, config):
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    from vllm_ascend.ascend_forward_context import set_ascend_forward_context

    with set_ascend_forward_context(
        frame.mapping,
        config,
        num_tokens=frame.capacity_tokens,
        num_actual_tokens=frame.capacity_tokens,
        aclgraph_runtime_mode=CUDAGraphMode.NONE,
        model_instance=model,
        skip_compiled=True,
    ):
        ctx = get_forward_context()
        ctx.slot_mapping = {item.backend.layer_name: frame.slot_mapping for item in bundle.fia_entries}
        ctx.capturing = False
        yield


def local_pair(torch, model, bundle, registry, banks, initial, observer, frame, config, index):
    item = next(item for item in bundle.gdn_entries if item.index == index)
    offset = bundle.gdn_entries.index(item)
    state = initial["compact"].states[offset]
    backend = registry.backends[item.state.handle]
    backend.state.restore(state)
    item.conv.copy_(initial["conv"][offset])
    local = StepObserver(torch, observer.step, frame.capacity_tokens)
    hidden = observer.reference[f"layer{index}/input/hidden"].to(bundle.device)
    residual = observer.reference.get(f"layer{index}/input/residual")
    residual = residual.to(bundle.device) if residual is not None else None
    original_cache = item.gdn.kv_cache
    banks.scratch[0].copy_(item.base[0])
    banks.scratch[1:].view(128, 4, 24, 128, 128).copy_(item.base[1:, None])
    conv = item.conv.clone()
    try:
        item.gdn.kv_cache = (conv, banks.scratch)
        with (
            layer_context(model, bundle, frame, config),
            local.installed(model, (index,)),
            local.tap_recurrence([item]),
        ):
            model.language_model.model.layers[index](
                hidden_states=hidden.clone(),
                residual=residual.clone() if residual is not None else None,
                positions=frame.positions,
            )
        torch.npu.synchronize()
    finally:
        item.gdn.kv_cache = original_cache
    # Nothing in the official isolated call touched compact base/log/conv.
    one = CompactSSMRegistry()
    one.register(item.state.handle, backend, item.gdn.out_proj)
    one.mode = "deferred_tile_ready"
    local.phase = "compact"
    one.prepare_step(0, observer.step["request_ids"], observer.step["accepted_from_previous"])
    with installed_log_registry(one), layer_context(model, bundle, frame, config):
        with local.installed(model, (index,)), local.tap_recurrence([item]):
            model.language_model.model.layers[index](
                hidden_states=hidden.clone(),
                residual=residual.clone() if residual is not None else None,
                positions=frame.positions,
            )
            one.finish_pending_commits()
        torch.npu.synchronize()
    one.finish_step(0)
    return local


def diagnose(args, torch, dist, model, bundle, factory, registry, forward, vocabulary, report, *, checkpoint):
    from vllm.config import get_current_vllm_config

    torch.npu.matmul.allow_hf32 = False
    torch.set_num_threads(8)
    step = bundle.trace.steps[0]
    if step["step"] != 0 or step["batch"] != 98 or args.selection["dispatched_tokens"][0] != 400:
        raise ValueError("this bounded attribution is frozen to original B98, default 400-token step zero")
    registry.mode = "deferred_tile_ready"
    frame = factory.decode(step, 400)
    ids = torch.zeros(400, dtype=torch.int64)
    ids[: step["batch"] * 4] = deterministic_token_ids(torch, step["batch"] * 4, vocabulary, step["input_seed"])
    report["step0_contract"] = dict(
        step=dict(step),
        batch=98,
        capacity_tokens=400,
        token_sha256=tensor_digest(torch, ids),
        screen_nrmse=SCREEN_NRMSE,
        performance_rounds=0,
        continued_qualification=False,
        weight_activation_dtype="bfloat16",
        ssm_log_dtype="float32",
        hf32=False,
    )
    checkpoint()
    initial = complete_snapshot(torch, bundle, registry)
    banks = OfficialBanks(torch, bundle)
    report["step0_start_audit"] = audit_start(torch, bundle, banks, initial, frame, step)
    checkpoint()
    print(f"rank={dist.get_rank()} FP32 step0 initial-state audit passed", flush=True)
    observer = StepObserver(torch, step, 400)
    ids = ids.to(bundle.device)
    with (
        banks.installed(model, step["request_ids"]),
        observer.installed(model),
        observer.tap_recurrence(bundle.gdn_entries),
    ):
        reference = tuple(v.cpu().clone() for v in forward(frame, ids))
    # Explicitly verify the official control did not alter compact histories.
    for item, saved in zip(bundle.gdn_entries, initial["compact"].states, strict=True):
        if not identical(torch, item.base.cpu(), saved.tensors[0]):
            raise AssertionError("official control modified compact initial state")
    for item, conv in zip(bundle.gdn_entries, initial["conv"], strict=True):
        if not identical(torch, item.conv.cpu(), conv):
            raise AssertionError("official control modified compact convolution history")
    for item, pair in zip(bundle.fia_entries, initial["fia"], strict=True):
        if any(
            not identical(torch, current.cpu(), saved)
            for current, saved in zip((item.key, item.value), pair, strict=True)
        ):
            raise AssertionError("official control modified compact FIA history")
    observer.phase = "compact"
    registry.prepare_step(0, step["request_ids"], step["accepted_from_previous"])
    with installed_log_registry(registry), observer.installed(model), observer.tap_recurrence(bundle.gdn_entries):
        actual = tuple(v.cpu().clone() for v in forward(frame, ids, registry=registry))
    registry.finish_step(0)
    report["step0_errors"] = observer.errors
    report["step0_endpoint"] = [
        token_error(torch, a, b, label=label, capacity=400, step=step)
        for label, a, b in zip(("hidden", "residual"), actual, reference, strict=True)
    ]
    sites = select_sites(observer.errors, [item.index for item in bundle.gdn_entries])
    report["step0_sites"] = sites
    report.update(status="partial_fp32_step0_attribution", step0_local=[])
    checkpoint()
    print(f"rank={dist.get_rank()} FP32 step0 full pair complete, sites={sites['local_layer_indices']}", flush=True)
    for index in sites["local_layer_indices"]:
        local = local_pair(
            torch, model, bundle, registry, banks, initial, observer, frame, get_current_vllm_config(), index
        )
        record = dict(layer=index, identical_decoder_inputs=True, identical_initial_state=True, errors=local.errors)
        artifact = args.output.parent / f"step0-layer{index}.rank{dist.get_rank()}.pt"
        torch.save(
            dict(
                decoder_hidden=observer.reference[f"layer{index}/input/hidden"],
                decoder_residual=observer.reference.get(f"layer{index}/input/residual"),
                state=initial["compact"]
                .states[bundle.gdn_entries.index(next(i for i in bundle.gdn_entries if i.index == index))]
                .tensors,
                recurrence_inputs=local.recurrence[index],
                reference=local.reference,
                observed_recurrence=local.observed_recurrence,
            ),
            artifact,
        )
        record.update(snapshot_path=str(artifact), snapshot_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        report["step0_local"].append(record)
        checkpoint()
        print(f"rank={dist.get_rank()} FP32 step0 same-input layer={index} complete", flush=True)
    # One intervention only: restore the whole original prefix, then replace
    # the earliest differing recurrence output with its official observation.
    # It diagnoses a single site and does not alter any native state algorithm.
    first = sites["first_differing_recurrence"]
    if first is not None:
        index = int(first["label"].split("/")[0][5:])
        restore_complete(torch, bundle, registry, initial)
        replacement = observer.reference[first["label"]].to(bundle.device)
        import vllm_ascend.ops.gdn as gdn

        registry.prepare_step(0, step["request_ids"], step["accepted_from_previous"])
        with installed_log_registry(registry):
            original, calls = gdn.recurrent_gated_delta_rule, []

            def substitute(query, key, value, initial_state, **kwargs):
                position = len(calls)
                calls.append(bundle.gdn_entries[position].index)
                output = original(query, key, value, initial_state, **kwargs)
                return replacement if calls[-1] == index else output

            gdn.recurrent_gated_delta_rule = substitute
            try:
                replaced = tuple(v.cpu().clone() for v in forward(frame, ids, registry=registry))
            finally:
                gdn.recurrent_gated_delta_rule = original
        registry.finish_step(0)
        report["step0_single_substitution"] = dict(
            layer=index,
            boundary="recurrence output only; compact state commit still executes",
            replacements=sum(i == index for i in calls),
            endpoint=[
                token_error(torch, a, b, label=label, capacity=400, step=step)
                for label, a, b in zip(("hidden", "residual"), replaced, reference, strict=True)
            ],
        )
    report.update(status="complete_bounded_fp32_step0_attribution", full_trajectory_numerical_qualification=False)
    checkpoint()
    print(f"rank={dist.get_rank()} bounded FP32 step0 attribution complete", flush=True)
