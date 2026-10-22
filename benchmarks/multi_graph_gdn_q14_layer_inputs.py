# SPDX-License-Identifier: Apache-2.0
"""Prepare a new teacher-forced layer comparison from the frozen Q1/Q4 metadata.

Only the owner/query/acceptance schedule comes from the original ledger. Real
text, a common 16-token prefill, and official layers 0..3 construct new layer-4
inputs outside timing. These are not the original ledger's model activations.
"""

import hashlib
import json

PREFIX_TOKENS = 16
QUERY_CAPACITY = 4


def _json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def activation_plan(manifest, tokens, *, prefix_tokens=PREFIX_TOKENS):
    """Keep each owner's accepted text offset independent of its row position."""
    owners = manifest["stable_owners"]
    capacity = manifest["capacity_requests"]
    if owners != list(range(len(owners))) or not owners or capacity < 1 or capacity > len(owners):
        raise ValueError("expected contiguous stable owners and a valid request capacity")
    if len(tokens) < len(owners) or type(prefix_tokens) is not int or prefix_tokens < 1:
        raise ValueError("missing source text or invalid prefix length")
    offsets = dict.fromkeys(owners, prefix_tokens)
    previous_query = {}
    result = []
    for entry in manifest["steps"]:
        live = entry["request_ids"]
        widths, accepted = entry["query_lengths"], entry["accepted_from_previous"]
        if len(live) > capacity or len(set(live)) != len(live):
            raise ValueError("duplicate owners or request capacity exceeded")
        if len(live) != len(widths) or len(live) != len(accepted):
            raise ValueError("owner/query/acceptance lengths differ")
        if entry.get("capacity_tokens", capacity * QUERY_CAPACITY) != capacity * QUERY_CAPACITY:
            raise ValueError("retain the common four-row dense projection capacity")
        row_offsets, source_tokens, positions = [], [], []
        for owner, width, count in zip(live, widths, accepted, strict=True):
            if type(owner) is not int or owner not in offsets or width not in (1, 4):
                raise ValueError("unknown owner or query other than Q1/Q4")
            if type(count) is not int or not 1 <= count <= QUERY_CAPACITY:
                raise ValueError("previous acceptance must be in 1..4")
            if owner in previous_query:
                if count > previous_query[owner]:
                    raise ValueError("acceptance exceeds the previous query")
                offsets[owner] += count
            # Segment entry has no pending log: its old-ledger acceptance does
            # not advance the new common prefill a second time.
            offset = offsets[owner]
            window = list(tokens[owner][offset : offset + QUERY_CAPACITY])
            if len(window) != QUERY_CAPACITY:
                raise ValueError("source text has fewer than four source tokens remaining")
            row_offsets.append(offset)
            source_tokens.append(window)
            positions.extend(range(offset, offset + QUERY_CAPACITY))
            previous_query[owner] = width
        positions.extend([0] * (QUERY_CAPACITY * (capacity - len(live))))
        result.append(
            dict(
                step=entry["step"],
                request_ids=list(live),
                query_lengths=list(widths),
                accepted_from_previous=list(accepted),
                offsets=row_offsets,
                source_tokens=source_tokens,
                source_tokens_sha256=_json_sha256(source_tokens),
                positions=positions,
            )
        )
    return result


def _tensor_sha256(torch, tensor):
    return hashlib.sha256(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def build_layer_inputs(torch, model, config, manifest, *, source=None, batch_size=8):
    """Return device-resident shared inputs and initial conv/SSM plus CPU provenance.

    Call before graph capture and measurement. Save provenance as JSON and, if
    required, save the returned tensors per TP rank after moving them to CPU.
    Each record retains four dense rows per request, including Q1 filler rows;
    its frozen query lengths alone determine which rows recurrence consumes.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("activation generation batch size must be positive")
    if source is None:
        from multi_graph_gdn_ready_activations import ActivationSource

        source = ActivationSource(torch, model, config, owners=len(manifest["stable_owners"]))
    plan = activation_plan(manifest, source.tokens)
    owners = manifest["stable_owners"]
    # ActivationSource.initial_target() assumes 100 owners. Prepare only this
    # frozen owner's pool without changing the established source helper.
    for start in range(0, len(owners), batch_size):
        batch = owners[start : start + batch_size]
        source.batch(batch, [PREFIX_TOKENS] * len(batch), target=True)
    torch.npu.synchronize()
    initial_conv, initial_ssm = (value.clone() for value in (source.target.conv, source.target.base))
    hidden_size = config.model_config.hf_text_config.hidden_size
    rows = manifest["capacity_requests"] * QUERY_CAPACITY
    records = []
    for entry in plan:
        hidden, residual = (
            torch.zeros(rows, hidden_size, dtype=torch.bfloat16, device=source.bundle.device) for _ in range(2)
        )
        for start in range(0, len(entry["request_ids"]), batch_size):
            batch = entry["request_ids"][start : start + batch_size]
            lengths = [offset + QUERY_CAPACITY for offset in entry["offsets"][start : start + batch_size]]
            values = source.batch(batch, lengths)
            if len(values) != len(batch):
                raise ValueError("activation source returned a different request count")
            for row, pair in enumerate(values, start):
                if any(tuple(value.shape) != (QUERY_CAPACITY, hidden_size) for value in pair):
                    raise ValueError("activation source did not return four full hidden/residual rows")
                hidden[row * QUERY_CAPACITY : (row + 1) * QUERY_CAPACITY].copy_(pair[0])
                residual[row * QUERY_CAPACITY : (row + 1) * QUERY_CAPACITY].copy_(pair[1])
        positions = torch.tensor(entry["positions"], dtype=torch.int64, device=source.bundle.device)
        if not all(torch.isfinite(value).all().item() for value in (hidden, residual)):
            raise ValueError("nonfinite teacher-forced activation")
        hashes = {name: _tensor_sha256(torch, value) for name, value in dict(hidden=hidden, residual=residual).items()}
        input_manifest = dict(entry, tensor_sha256=hashes, positions_sha256=_tensor_sha256(torch, positions))
        records.append(
            dict(
                step=entry["step"], hidden=hidden, residual=residual, positions=positions, input_manifest=input_manifest
            )
        )
    torch.npu.synchronize()
    return dict(
        initial_conv=initial_conv,
        initial_ssm=initial_ssm,
        records=records,
        provenance=dict(
            scope="new teacher-forced single-layer comparison with frozen owner/query/acceptance metadata",
            activation_scope="same real text through embedding and official layers 0..3; layer4 inputs saved per step",
            initial_state_matches_original_ledger_step=False,
            initial_state="same real 16-token layer4 prefill; no pending accepted-state debt",
            prefix_tokens=PREFIX_TOKENS,
            source_token_sha256=source.token_sha256,
            common_dense_rows=rows,
            initial_conv_sha256=_tensor_sha256(torch, initial_conv),
            initial_ssm_sha256=_tensor_sha256(torch, initial_ssm),
            initial_tokens=[dict(owner=owner, tokens=list(source.tokens[owner][:PREFIX_TOKENS])) for owner in owners],
            steps=[record["input_manifest"] for record in records],
            generation_excluded_from_timing=True,
        ),
    )
