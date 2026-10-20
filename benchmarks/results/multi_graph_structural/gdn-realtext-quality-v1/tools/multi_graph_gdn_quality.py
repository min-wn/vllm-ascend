# SPDX-License-Identifier: Apache-2.0
"""Predeclared token alignment and full-vocabulary quality measurements.

This module never changes model arithmetic or the old hidden NRMSE screen.
Quality limits are engineering screening budgets, not universal tolerances.
"""

import hashlib
import json
import math
import random

SCHEMA = "gdn-real-text-quality-v1"
BOOTSTRAP_SEED = 20261020
BOOTSTRAP_DRAWS = 10000
LIMITS = dict(
    ppl_ratio_upper95=1.01,
    mean_kl=0.001,
    p99_kl=0.01,
    mean_tv=0.01,
    p99_tv=0.05,
    top1_disagreement=0.02,
    max_owner_delta_nll=0.05,
    minimum_source_articles=16,
    task_accuracy_drop=0.02,
    task_new_failure_fraction=0.03,
)


def digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def score_layout(trace):
    """Each accepted next-token position is counted once, including the tail.

    Input at p predicts p+1. The first decode input token was predicted by the
    identical official prefill and is excluded. Last accept decisions match
    the existing terminal materialization, even for paused owners.
    """
    last = {owner: step["step"] for step in trace.steps for owner in step["request_ids"]}
    cursor = {owner: length + 1 for owner, length in enumerate(trace.initial_prefix_by_request)}
    result = []
    for step in trace.steps:
        rows = []
        for row, (owner, length, accepted) in enumerate(
            zip(step["request_ids"], step["seq_lengths"], step["accepted_after_step"], strict=True)
        ):
            count = 1 + owner % 4 if last[owner] == step["step"] else accepted
            start = length - 4 + 1
            if start != cursor[owner]:
                raise ValueError("duplicate or missing next-token position in continued request")
            rows.extend(dict(flat=row * 4 + j, owner=owner, position=start + j) for j in range(count))
            cursor[owner] += count
        result.append(rows)
    return result


def distribution_metrics(torch, reference, actual, targets):
    if reference.ndim != 2 or actual.shape != reference.shape or targets.shape != reference.shape[:1]:
        raise ValueError("paired full-vocabulary logits and one next-token target per row required")
    if not torch.isfinite(reference).all().item() or not torch.isfinite(actual).all().item():
        raise ValueError("nonfinite logits")
    log_p = reference.float().log_softmax(-1)
    log_q = actual.float().log_softmax(-1)
    p, q = log_p.exp(), log_q.exp()
    index = targets.long().unsqueeze(-1)
    official = -log_p.gather(-1, index).squeeze(-1)
    compact = -log_q.gather(-1, index).squeeze(-1)
    mixture = torch.logaddexp(log_p, log_q) - math.log(2)
    top = reference.float().topk(2, dim=-1)
    prediction, candidate = log_p.argmax(-1), log_q.argmax(-1)
    values = dict(
        official_nll=official,
        compact_nll=compact,
        delta_nll=compact - official,
        kl=(p * (log_p - log_q)).sum(-1).clamp_min(0),
        js=(0.5 * (p * (log_p - mixture) + q * (log_q - mixture))).sum(-1).clamp_min(0),
        tv=(0.5 * (p - q).abs()).sum(-1),
        official_top1=prediction,
        compact_top1=candidate,
        top1_disagreement=prediction != candidate,
        official_margin=top.values[:, 0] - top.values[:, 1],
        official_correct=prediction == targets,
        compact_correct=candidate == targets,
    )
    # Only O(tokens) observations leave the accelerator. Logits are not kept.
    return {key: value.double().cpu().tolist() for key, value in values.items()}


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]) * (index - lower)


def summarize(rows, *, expected_tokens):
    if not rows or len(rows) != expected_tokens:
        raise ValueError("incomplete predeclared quality trajectory")
    identities = {(row["owner"], row["position"]) for row in rows}
    if len(identities) != len(rows):
        raise ValueError("duplicate scored position")
    if any(not math.isfinite(value) for row in rows for value in row.values() if isinstance(value, float)):
        raise ValueError("nonfinite observation")
    articles, owners = {}, {}
    for row in rows:
        for mapping, key in ((articles, row["article"]), (owners, row["owner"])):
            item = mapping.setdefault(key, [0.0, 0])
            item[0] += row["delta_nll"]
            item[1] += 1
    blocks = list(articles.values())
    rng = random.Random(BOOTSTRAP_SEED)
    ratios = []
    for _ in range(BOOTSTRAP_DRAWS):
        selected = [rng.choice(blocks) for _ in blocks]
        ratios.append(math.exp(sum(x[0] for x in selected) / sum(x[1] for x in selected)))
    count = len(rows)
    means = {
        key: sum(row[key] for row in rows) / count
        for key in (
            "official_nll",
            "compact_nll",
            "delta_nll",
            "kl",
            "js",
            "tv",
            "top1_disagreement",
            "official_correct",
            "compact_correct",
        )
    }
    report = dict(
        tokens=count,
        owners=len(owners),
        source_articles=len(articles),
        means=means,
        official_ppl=math.exp(means["official_nll"]),
        compact_ppl=math.exp(means["compact_nll"]),
        ppl_ratio=math.exp(means["delta_nll"]),
        ppl_ratio_upper95=percentile(ratios, 0.95),
        p99_kl=percentile([r["kl"] for r in rows], 0.99),
        p99_tv=percentile([r["tv"] for r in rows], 0.99),
        max_owner_delta_nll=max(total / n for total, n in owners.values()),
        bootstrap=dict(unit="source article", seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS, one_sided=0.95),
    )
    gates = dict(
        complete_owners=len(owners) == 128,
        source_articles=len(articles) >= LIMITS["minimum_source_articles"],
        ppl_noninferiority=report["ppl_ratio_upper95"] <= LIMITS["ppl_ratio_upper95"],
        mean_kl=means["kl"] <= LIMITS["mean_kl"],
        p99_kl=report["p99_kl"] <= LIMITS["p99_kl"],
        mean_tv=means["tv"] <= LIMITS["mean_tv"],
        p99_tv=report["p99_tv"] <= LIMITS["p99_tv"],
        top1=means["top1_disagreement"] <= LIMITS["top1_disagreement"],
        owner_tail=report["max_owner_delta_nll"] <= LIMITS["max_owner_delta_nll"],
    )
    report.update(gates=gates, screen_passed=all(gates.values()), limits=LIMITS)
    return report
