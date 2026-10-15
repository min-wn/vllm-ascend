from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from vllm_ascend.config_utils import config

PositiveInt = Annotated[int, Field(strict=True, gt=0)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]


@config(config=ConfigDict(frozen=True))
class MultiGraphConfig:
    """Explicit resource limits for the two-lane experimental decode path.

    Sizes describe distinct entries in each lane, including equal-size entries.
    Enabling requires an explicit memory reservation; there is no guessed budget.
    The runner validates supported combinations before allocating resources.
    """

    enabled: bool = False
    execution: Literal["serial", "parallel"] = "parallel"
    num_lanes: Literal[2] = 2
    workload: Literal["uniform_decode"] = "uniform_decode"
    input_layout: Literal["lane_buffer"] = "lane_buffer"
    kv_cache_layout: Literal["default", "contiguous"] = "default"
    capture_sizes_per_lane: tuple[tuple[PositiveInt, ...], tuple[PositiveInt, ...]] = ((), ())
    padding_policy: Literal["exact"] = "exact"
    capture_policy: Literal["startup"] = "startup"
    miss_policy: Literal["single_graph"] = "single_graph"
    selection_policy: Literal["measured", "force"] = "measured"
    max_graph_entries: PositiveInt = 10
    graph_memory_budget_bytes: NonNegativeInt = 0
    cost_table_path: str | None = None
    minimum_cost_samples: PositiveInt = 5
    minimum_gain_fraction: Annotated[float, Field(ge=0, lt=1, allow_inf_nan=False)] = 0.05
    kv_bucket_upper_bounds: tuple[PositiveInt, ...] = ()
    minimum_bucket_vectors: Annotated[int, Field(strict=True, ge=2)] = 2
    online_calibration: bool = False
    calibration_sample_interval: PositiveInt = 16
    calibration_maximum_keys: PositiveInt = 128
    calibration_output_path: str | None = None

    @model_validator(mode="after")
    def validate_resources(self) -> MultiGraphConfig:
        if self.online_calibration and (not self.calibration_output_path or self.selection_policy != "measured"):
            raise ValueError("online calibration requires measured selection and a calibration_output_path")
        if tuple(sorted(set(self.kv_bucket_upper_bounds))) != self.kv_bucket_upper_bounds:
            raise ValueError("kv_bucket_upper_bounds must be strictly increasing")
        for sizes in self.capture_sizes_per_lane:
            if tuple(sorted(set(sizes))) != sizes:
                raise ValueError("capture_sizes_per_lane must contain strictly increasing sizes")
        if sum(map(len, self.capture_sizes_per_lane)) > self.max_graph_entries:
            raise ValueError("capture_sizes_per_lane exceeds max_graph_entries across both lanes")
        if self.enabled:
            if any(not sizes for sizes in self.capture_sizes_per_lane):
                raise ValueError("enabled multi-graph execution requires capture sizes in both lanes")
            if self.graph_memory_budget_bytes == 0:
                raise ValueError("enabled multi-graph execution requires a positive graph_memory_budget_bytes")
        return self

    def compute_hash(self) -> str:
        """Return a stable signature for capture entries and measured costs."""
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def capture_hash(self) -> str:
        values = dataclasses.asdict(self)
        for name in (
            "selection_policy",
            "execution",
            "cost_table_path",
            "minimum_cost_samples",
            "minimum_gain_fraction",
            "kv_bucket_upper_bounds",
            "minimum_bucket_vectors",
            "online_calibration",
            "calibration_sample_interval",
            "calibration_maximum_keys",
            "calibration_output_path",
        ):
            values.pop(name)
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()
