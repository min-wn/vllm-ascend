from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from vllm_ascend.worker.multi_graph.costs import DecodeCostDocument, DecodeCostRecord
from vllm_ascend.worker.multi_graph.planner import CostKey, MeasuredCost
from vllm_ascend.worker.multi_graph.types import CaptureSignature


@dataclass(frozen=True)
class CalibrationPolicy:
    samples_per_mode: int = 5
    maximum_keys: int = 128
    sample_interval: int = 16

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("calibration limits must be positive integers")


class OnlineCalibrator:
    """Alternate real steps; never replay a request to compare two KV writers.

    The caller supplies synchronized forward operations, a synchronization
    callback and current capture identity. Exact KV vectors must recur to form
    pairs. Inference correctness must be validated separately on the device.
    """

    def __init__(self, signature: CaptureSignature, policy: CalibrationPolicy):
        self.signature = signature
        self.policy = policy
        self.samples: dict[CostKey, tuple[list[float], list[float]]] = {}
        self.steps = 0
        self.failed = False
        self.active = False
        self.revision = 0

    def dispatch(
        self,
        key: CostKey,
        baseline: Callable[[], Any],
        split: Callable[[], Any],
        synchronize: Callable[[], None],
        clock: Callable[[], int] = perf_counter_ns,
    ) -> Any:
        if self.failed or self.active:
            raise RuntimeError("calibrator is failed or reentered")
        if any(graph.signature != self.signature for graph in key.graph_keys):
            raise ValueError("calibration signature changed")
        if (
            key.execution not in ("serial", "parallel")
            or len(key.kv_lengths) != sum(graph.graph_reqs for graph in key.graph_keys)
            or any(type(length) is not int or length <= 0 for length in key.kv_lengths)
            or tuple(graph.lane_id for graph in key.graph_keys) != (0, 1)
        ):
            raise ValueError("invalid calibration key")
        self.steps += 1
        if self.steps % self.policy.sample_interval:
            return baseline()
        if key not in self.samples:
            if len(self.samples) >= self.policy.maximum_keys:
                return baseline()
            self.samples[key] = ([], [])
        single_samples, split_samples = self.samples[key]
        if min(len(single_samples), len(split_samples)) >= self.policy.samples_per_mode:
            return baseline()
        use_split = len(single_samples) > len(split_samples)
        operation = split if use_split else baseline
        self.active = True
        try:
            synchronize()
            start = clock()
            result = operation()
            synchronize()
            elapsed = (clock() - start) / 1000
            MeasuredCost(elapsed, elapsed)
            (split_samples if use_split else single_samples).append(elapsed)
            if min(len(single_samples), len(split_samples)) == self.policy.samples_per_mode:
                self.revision += 1
            return result
        except Exception:
            self.failed = True
            raise
        finally:
            self.active = False

    def document(self) -> DecodeCostDocument:
        if self.failed or self.active:
            raise RuntimeError("cannot export failed or active calibration")
        records = []
        for key, (single, split) in self.samples.items():
            count = min(len(single), len(split))
            if count < self.policy.samples_per_mode:
                continue
            records.append(
                DecodeCostRecord(
                    lane_sizes=tuple(graph.graph_tokens for graph in key.graph_keys),
                    execution=key.execution,
                    kv_lengths=key.kv_lengths,
                    single_graph_us=single[:count],
                    split_us=split[:count],
                )
            )
        return DecodeCostDocument(
            metric="synchronized_decode_dispatch_us", signature=self.signature, records=tuple(records)
        )

    def save(self, path: Path) -> None:
        payload = self.document().model_dump_json(indent=2)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as target:
                temporary = Path(target.name)
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
