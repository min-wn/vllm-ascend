#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

from vllm.logger import logger
from vllm.triton_utils import HAS_TRITON


def check_kv_extra_config(vllm_config):

    def _check(name: str, config: dict):
        tp_key = "tp_size"
        dp_key = "dp_size"
        if tp_key in config:
            config_tp = config[tp_key]
            vllm_tp = vllm_config.parallel_config.tensor_parallel_size
            if config_tp != vllm_tp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting tensor parallel size. "
                    f"Expected {vllm_tp}, but got {config_tp}.")
        if dp_key in config:
            config_dp = config[dp_key]
            vllm_dp = vllm_config.parallel_config.data_parallel_size
            if config_dp != vllm_dp:
                raise ValueError(
                    f"KV transfer '{name}' config has a conflicting data parallel size. "
                    f"Expected {vllm_dp}, but got {config_dp}.")

    if vllm_config.kv_transfer_config.is_kv_producer:
        _check(
            "prefill",
            vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {}))
    if vllm_config.kv_transfer_config.is_kv_consumer:
        _check(
            "decode",
            vllm_config.kv_transfer_config.get_from_extra_config("decode", {}))


class AscendConfig:
    """
    Configuration Object for additional_config from vllm.configs.
    """

    def __init__(self, vllm_config):
        additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}

        xlite_graph_config = additional_config.get("xlite_graph_config", {})
        self.xlite_graph_config = XliteGraphConfig(xlite_graph_config,
                                                   vllm_config)

        ascend_compilation_config = additional_config.get(
            "ascend_compilation_config", {})
        self.ascend_compilation_config = AscendCompilationConfig(
            **ascend_compilation_config)

        finegrained_tp_config = additional_config.get("finegrained_tp_config",
                                                      {})
        self.finegrained_tp_config = FinegrainedTPConfig(
            finegrained_tp_config, vllm_config)

        # Dump / PrecisionDebugger configuration
        dump_config_path = additional_config.get("dump_config", None)
        self.dump_config = DumpConfig(dump_config_path)

        weight_prefetch_config = additional_config.get(
            "weight_prefetch_config", {})
        self.weight_prefetch_config = WeightPrefetchConfig(
            weight_prefetch_config)

        # Split-batch (decode micro-splitting across requests)
        split_batch_config = additional_config.get("split_batch_config", {})
        self.split_batch_config = SplitBatchConfig(split_batch_config)

        # Todo: Once https://github.com/vllm-project/vllm/issues/22246 is merged in vllm. Remove this config
        self.expert_map_path = additional_config.get("expert_map_path", None)
        self.eplb_policy_type = additional_config.get("eplb_policy_type", 1)
        self.expert_map_record_path = additional_config.get(
            "expert_map_record_path",
            None)  # Provide path to export expert map
        self.init_redundancy_expert = additional_config.get(
            "init_redundancy_expert", 0)
        self.dynamic_eplb = additional_config.get("dynamic_eplb", False)
        self.num_iterations_eplb_update = additional_config.get(
            "num_iterations_eplb_update", 400)
        self.gate_eplb = additional_config.get("gate_eplb", False)
        self.num_wait_worker_iterations = additional_config.get(
            "num_wait_worker_iterations", 30)
        self.chunked_prefill_for_mla = additional_config.get(
            "chunked_prefill_for_mla", False)
        self.enable_shared_expert_dp = additional_config.get(
            "enable_shared_expert_dp",
            False) and vllm_config.parallel_config.enable_expert_parallel
        if self.enable_shared_expert_dp:
            from vllm_ascend.utils import enable_sp
            assert enable_sp(vllm_config=vllm_config,
                             enable_shared_expert_dp=True)
        self.multistream_overlap_shared_expert = additional_config.get(
            "multistream_overlap_shared_expert", False)
        self.multistream_overlap_gate = additional_config.get(
            "multistream_overlap_gate", False)
        self.recompute_scheduler_enable = additional_config.get(
            "recompute_scheduler_enable", False)
        self.enable_cpu_binding = additional_config.get(
            "enable_cpu_binding", False)

        if vllm_config.kv_transfer_config is not None:
            check_kv_extra_config(vllm_config)

        self.pd_tp_ratio = 1
        self.pd_head_ratio = 1
        self.num_head_replica = 1
        if vllm_config.kv_transfer_config is not None and not vllm_config.model_config.is_deepseek_mla:
            prefill_tp_size = vllm_config.kv_transfer_config.get_from_extra_config(
                "prefill", {"tp_size": 1})["tp_size"]
            decode_tp_size = vllm_config.kv_transfer_config.get_from_extra_config(
                "decode", {"tp_size": 1})["tp_size"]
            assert prefill_tp_size % decode_tp_size == 0, "Prefill TP size must be divisible by Decode TP size."
            self.pd_tp_ratio = prefill_tp_size // decode_tp_size
            if self.pd_tp_ratio > 1:
                try:
                    # only support Qwen model now
                    # TODO: use a more robust method to get kv_head_num
                    num_kv_head = vllm_config.model_config.hf_config.num_key_value_heads
                    self.num_head_replica = prefill_tp_size // num_kv_head if prefill_tp_size >= num_kv_head else 1
                    prefill_tp_size = min(prefill_tp_size, num_kv_head)
                    decode_tp_size = min(decode_tp_size, num_kv_head)
                    self.pd_head_ratio = prefill_tp_size // decode_tp_size
                except Exception:
                    raise AssertionError(
                        "Can not get num_key_value_heads from model_config")

            if self.pd_tp_ratio == 0:
                raise AssertionError(
                    "Only support P node tp size lagger then D node tp size")
        self.SLO_limits_for_dynamic_batch = additional_config.get(
            "SLO_limits_for_dynamic_batch", -1)
        from vllm_ascend.utils import get_flashcomm2_config_and_validate
        self.flashcomm2_oproj_tensor_parallel_size, self.flashcomm2_oproj_shared = get_flashcomm2_config_and_validate(
            self, vllm_config)
        self.enable_npugraph_ex = additional_config.get(
            "enable_npugraph_ex", False)
        # We find that _npu_paged_attention still performs better than
        # npu_fused_infer_attention_score in some cases. We allow to execute
        # _npu_paged_attention in this cases. This should be removed once
        # npu_fused_infer_attention_score performs better on all scenarios.
        self.pa_shape_list = additional_config.get("pa_shape_list", [])

        kv_cfg = vllm_config.kv_transfer_config
        if kv_cfg is not None and not getattr(kv_cfg, "_engine_id_patched",
                                              False):
            kv_cfg.engine_id = f"{kv_cfg.engine_id}-{uuid4().hex}"
            kv_cfg._engine_id_patched = True
        self.enable_async_exponential = additional_config.get(
            "enable_async_exponential", 0)
        if self.enable_async_exponential not in (0, 1):
            raise AssertionError(
                "Enable async exponential can only be set to 0 or 1.")


class FinegrainedTPConfig:
    """
    Configuration Object for finegrained_tp_config from additional_config
    """

    def __init__(self, finegrained_tp_config: dict, vllm_config):
        self.oproj_tensor_parallel_size = finegrained_tp_config.get(
            "oproj_tensor_parallel_size", 0)
        self.lmhead_tensor_parallel_size = finegrained_tp_config.get(
            "lmhead_tensor_parallel_size", 0)
        self.embedding_tensor_parallel_size = finegrained_tp_config.get(
            "embedding_tensor_parallel_size", 0)
        self.mlp_tensor_parallel_size = finegrained_tp_config.get(
            "mlp_tensor_parallel_size", 0)

        enabled_configs = []
        if self.oproj_tensor_parallel_size > 0:
            enabled_configs.append(
                f"oproj_tensor_parallel_size={self.oproj_tensor_parallel_size}"
            )
            # dummy_run does not run the entire attention module in eager mode,, so the o_proj tp split can only be used in graph mode.
            if vllm_config.model_config.enforce_eager is True:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in graph mode"
                )
            if vllm_config.kv_transfer_config is None or not vllm_config.kv_transfer_config.is_kv_consumer:
                raise AssertionError(
                    "oproj_tensor_parallel_size is only supported in pd scenario and can only be used in D node."
                )
        if self.lmhead_tensor_parallel_size > 0:
            enabled_configs.append(
                f"lmhead_tensor_parallel_size={self.lmhead_tensor_parallel_size}"
            )
        if self.embedding_tensor_parallel_size > 0:
            enabled_configs.append(
                f"embedding_tensor_parallel_size={self.embedding_tensor_parallel_size}"
            )
        if self.mlp_tensor_parallel_size > 0:
            enabled_configs.append(
                f"mlp_tensor_parallel_size={self.mlp_tensor_parallel_size}")
        module_tp_sizes = [
            self.oproj_tensor_parallel_size,
            self.lmhead_tensor_parallel_size,
            self.embedding_tensor_parallel_size,
            self.mlp_tensor_parallel_size,
        ]
        for module_tp_size in module_tp_sizes:
            if module_tp_size > 0 and vllm_config.parallel_config.data_parallel_size % module_tp_size != 0:
                raise AssertionError(
                    "module tp sizes must divide data_parallel_size")
        if any(size > 0 for size in module_tp_sizes) and enabled_configs:
            logger.info(
                f"finegrained_tp_config enabled: {', '.join(enabled_configs)}")


class AscendCompilationConfig:
    """
    Configuration for controlling the behavior of Ascend graph optimization.

    This class provides a way to configure graph fusion optimizations.
    These configurations directly impact the performance and behavior of models
    deployed on Ascend platforms.
    """

    def __init__(self,
                 fuse_norm_quant: bool = True,
                 fuse_qknorm_rope: bool = False,
                 **kwargs):
        """
        Initialize the configuration.
        
        Args:
            fuse_norm_quant (bool): Whether to enable norm and quant fusion optimization.
                When set to True, the system will optimize norm and quant operations.
                Default: True
            fuse_qknorm_rope (bool): Whether to enable qknorm and rope fusion optimization.
                Default: False
            **kwargs: Additional optional parameters for forward compatibility and configuration extension.
        """
        self.fuse_norm_quant = fuse_norm_quant
        self.fuse_qknorm_rope = HAS_TRITON or fuse_qknorm_rope


class XliteGraphConfig:
    """
    Configuration Object for xlite_graph_config from additional_config
    """

    def __init__(self, xlite_graph_config, vllm_config):
        self.enabled = xlite_graph_config.get("enabled", False)
        self.full_mode = xlite_graph_config.get("full_mode", False)
        if self.enabled:
            if bool(vllm_config.speculative_config):
                raise RuntimeError(
                    "Xlite graph mode is not compatible with speculative decoding. Please disable speculative decoding."
                )
            if vllm_config.parallel_config.pipeline_parallel_size > 1:
                raise RuntimeError(
                    "Xlite graph mode is not compatible with pipeline parallelism. Please set pipeline_parallel_size to 1."
                )
            if vllm_config.cache_config.block_size != 128:
                raise RuntimeError(
                    "Xlite graph mode is only compatible with block_size of 128. Please set block_size to 128."
                )


class DumpConfig:
    """
    Configuration object for dump/PrecisionDebugger settings.
    """

    def __init__(self, dump_config_path: Optional[str] = None):
        # enable_dump is True when dump_cfg exists and config_path is not empty
        self.enable_dump: bool = bool(dump_config_path)
        # Path to msprobe config json; may be None.
        self.config_path: Optional[str] = dump_config_path


class WeightPrefetchConfig:
    """
    Configuration Object for weight_prefetch_config from additional_config
    """

    prefetch_ratio: dict = {
        "attn": {
            "qkv": 1.0,
            "o": 1.0,
        },
        "moe": {
            "gate_up": 0.8
        }
    }

    def __init__(self, weight_prefetch_config: dict):
        self.enabled = weight_prefetch_config.get("enabled", False)
        self.prefetch_ratio = weight_prefetch_config.get(
            "prefetch_ratio", self.prefetch_ratio)


@dataclass(frozen=True)
class MacroGraphCapturePlan:
    total_tokens: int
    split_actual_tokens: tuple[int, int]
    split_graph_tokens: tuple[int, int]
    split_start_tokens: tuple[int, int]
    split_num_reqs: Optional[tuple[int, int]] = None
    split_req_caps: Optional[tuple[int, int]] = None

    @classmethod
    def from_config(cls, raw_plan: dict[str, Any]) -> "MacroGraphCapturePlan":
        total_tokens = int(raw_plan.get("total_tokens", 0))
        split_actual_tokens = _parse_two_ints(raw_plan,
                                              "split_actual_tokens")
        split_graph_tokens = _parse_two_ints(raw_plan, "split_graph_tokens")
        raw_num_reqs = raw_plan.get("split_num_reqs", None)
        split_num_reqs = None
        if raw_num_reqs is not None:
            split_num_reqs = tuple(int(v) for v in raw_num_reqs)
            if len(split_num_reqs) != 2:
                raise ValueError(
                    "macro_graph_config.capture_plans[].split_num_reqs "
                    "must contain exactly 2 integers")
        raw_req_caps = raw_plan.get("split_req_caps", None)
        split_req_caps = None
        if raw_req_caps is not None:
            split_req_caps = tuple(int(v) for v in raw_req_caps)
            if len(split_req_caps) != 2:
                raise ValueError(
                    "macro_graph_config.capture_plans[].split_req_caps "
                    "must contain exactly 2 integers")
        raw_start_tokens = raw_plan.get("split_start_tokens", None)
        if raw_start_tokens is None:
            split_start_tokens = (0, split_actual_tokens[0])
        else:
            split_start_tokens = tuple(int(v) for v in raw_start_tokens)
            if len(split_start_tokens) != 2:
                raise ValueError(
                    "macro_graph_config.capture_plans[]."
                    "split_start_tokens must contain exactly 2 integers")

        if total_tokens < 1:
            raise ValueError(
                "macro_graph_config.capture_plans[].total_tokens must be >= 1")
        if sum(split_actual_tokens) != total_tokens:
            raise ValueError(
                "macro_graph_config.capture_plans[].split_actual_tokens "
                "must sum to total_tokens")
        if any(v < 1 for v in split_actual_tokens):
            raise ValueError(
                "macro_graph_config.capture_plans[].split_actual_tokens "
                "must contain positive integers")
        if any(v < 1 for v in split_graph_tokens):
            raise ValueError(
                "macro_graph_config.capture_plans[].split_graph_tokens "
                "must contain positive integers")
        if any(graph < actual for graph, actual in zip(
                split_graph_tokens, split_actual_tokens)):
            raise ValueError(
                "macro_graph_config.capture_plans[].split_graph_tokens "
                "must be >= split_actual_tokens")
        if split_start_tokens[0] != 0 or any(v < 0 for v in split_start_tokens):
            raise ValueError(
                "macro_graph_config.capture_plans[].split_start_tokens "
                "must start at 0 and be non-negative")
        if split_num_reqs is not None and any(v < 1 for v in split_num_reqs):
            raise ValueError(
                "macro_graph_config.capture_plans[].split_num_reqs "
                "must contain positive integers")
        if split_req_caps is not None and any(v < 1 for v in split_req_caps):
            raise ValueError(
                "macro_graph_config.capture_plans[].split_req_caps "
                "must contain positive integers")
        if (split_num_reqs is not None and split_req_caps is not None
                and any(cap < req for cap, req in zip(split_req_caps,
                                                      split_num_reqs))):
            raise ValueError(
                "macro_graph_config.capture_plans[].split_req_caps must be "
                ">= split_num_reqs")
        return cls(
            total_tokens=total_tokens,
            split_actual_tokens=split_actual_tokens,
            split_graph_tokens=split_graph_tokens,
            split_start_tokens=split_start_tokens,
            split_num_reqs=split_num_reqs,
            split_req_caps=split_req_caps,
        )


def _parse_two_ints(raw: dict[str, Any], key: str) -> tuple[int, int]:
    value = raw.get(key, None)
    if value is None:
        raise ValueError(
            f"macro_graph_config.capture_plans[].{key} is required")
    parsed = tuple(int(v) for v in value)
    if len(parsed) != 2:
        raise ValueError(
            f"macro_graph_config.capture_plans[].{key} must contain "
            "exactly 2 integers")
    return parsed


def _parse_dual_stream_two_ints(raw: dict[str, Any],
                                key: str) -> tuple[int, int]:
    value = raw.get(key, None)
    if value is None:
        raise ValueError(
            f"dual_stream_attention_config.capture_plans[].{key} "
            "is required")
    parsed = tuple(int(v) for v in value)
    if len(parsed) != 2:
        raise ValueError(
            f"dual_stream_attention_config.capture_plans[].{key} "
            "must contain exactly 2 integers")
    return parsed


@dataclass(frozen=True)
class DualStreamAttentionCapturePlan:
    total_tokens: int
    split_actual_tokens: tuple[int, int]
    split_graph_tokens: tuple[int, int]
    split_start_tokens: tuple[int, int]

    @classmethod
    def from_config(
            cls,
            raw_plan: dict[str, Any]) -> "DualStreamAttentionCapturePlan":
        total_tokens = int(raw_plan.get("total_tokens", 0))
        split_actual_tokens = _parse_dual_stream_two_ints(
            raw_plan, "split_actual_tokens")
        split_graph_tokens = _parse_dual_stream_two_ints(
            raw_plan, "split_graph_tokens")
        raw_start_tokens = raw_plan.get("split_start_tokens", None)
        if raw_start_tokens is None:
            split_start_tokens = (0, split_actual_tokens[0])
        else:
            split_start_tokens = tuple(int(v) for v in raw_start_tokens)
            if len(split_start_tokens) != 2:
                raise ValueError(
                    "dual_stream_attention_config.capture_plans[]."
                    "split_start_tokens must contain exactly 2 integers")

        if total_tokens < 1:
            raise ValueError(
                "dual_stream_attention_config.capture_plans[]."
                "total_tokens must be >= 1")
        if sum(split_actual_tokens) != total_tokens:
            raise ValueError(
                "dual_stream_attention_config.capture_plans[]."
                "split_actual_tokens must sum to total_tokens")
        if any(v < 1 for v in split_actual_tokens):
            raise ValueError(
                "dual_stream_attention_config.capture_plans[]."
                "split_actual_tokens must contain positive integers")
        if any(v < 1 for v in split_graph_tokens):
            raise ValueError(
                "dual_stream_attention_config.capture_plans[]."
                "split_graph_tokens must contain positive integers")
        if any(graph < actual for graph, actual in zip(
                split_graph_tokens, split_actual_tokens)):
            raise ValueError(
                "dual_stream_attention_config.capture_plans[]."
                "split_graph_tokens must be >= split_actual_tokens")
        if split_start_tokens[0] != 0 or any(v < 0 for v in split_start_tokens):
            raise ValueError(
                "dual_stream_attention_config.capture_plans[]."
                "split_start_tokens must start at 0 and be non-negative")
        if split_start_tokens[1] != split_actual_tokens[0]:
            raise ValueError(
                "dual_stream_attention_config.capture_plans[]."
                "split_start_tokens[1] must equal split_actual_tokens[0]")
        return cls(
            total_tokens=total_tokens,
            split_actual_tokens=split_actual_tokens,
            split_graph_tokens=split_graph_tokens,
            split_start_tokens=split_start_tokens,
        )

    @property
    def graph_tokens(self) -> int:
        return sum(self.split_graph_tokens)


class DualStreamAttentionConfig:
    """Configuration for full graph + dual-stream FIA attention prototype."""

    def __init__(self, dual_stream_attention_config: dict):
        self.enabled: bool = bool(
            dual_stream_attention_config.get("enabled", False))
        self.backend: str = str(
            dual_stream_attention_config.get("backend", "fia"))
        self.plan_source: str = str(
            dual_stream_attention_config.get("plan_source", "explicit"))
        self.secondary_stream_mode: str = str(
            dual_stream_attention_config.get("secondary_stream_mode",
                                             "dedicated_pair"))
        self.actual_q_policy: str = str(
            dual_stream_attention_config.get("actual_q_policy", "graph"))
        self.miss_policy: str = str(
            dual_stream_attention_config.get("miss_policy", "error"))
        self.validate_decode_only: bool = bool(
            dual_stream_attention_config.get("validate_decode_only", True))
        self.max_capture_graphs: int = int(
            dual_stream_attention_config.get("max_capture_graphs", 16))
        raw_capture_plans = dual_stream_attention_config.get(
            "capture_plans", [])
        self.capture_plans: list[DualStreamAttentionCapturePlan] = [
            DualStreamAttentionCapturePlan.from_config(raw_plan)
            for raw_plan in raw_capture_plans
        ]

        valid_backends = ("fia", )
        if self.backend not in valid_backends:
            raise ValueError(
                "dual_stream_attention_config.backend must be one of "
                f"{valid_backends}, got {self.backend!r}")
        valid_plan_sources = ("explicit", )
        if self.plan_source not in valid_plan_sources:
            raise ValueError(
                "dual_stream_attention_config.plan_source must be one of "
                f"{valid_plan_sources}, got {self.plan_source!r}")
        valid_secondary_stream_modes = ("fork_join", "dedicated_pair")
        if self.secondary_stream_mode not in valid_secondary_stream_modes:
            raise ValueError(
                "dual_stream_attention_config.secondary_stream_mode must be "
                f"one of {valid_secondary_stream_modes}, got "
                f"{self.secondary_stream_mode!r}")
        valid_actual_q_policies = ("graph", "actual")
        if self.actual_q_policy not in valid_actual_q_policies:
            raise ValueError(
                "dual_stream_attention_config.actual_q_policy must be one of "
                f"{valid_actual_q_policies}, got {self.actual_q_policy!r}")
        valid_miss_policies = ("error", "disable")
        if self.miss_policy not in valid_miss_policies:
            raise ValueError(
                "dual_stream_attention_config.miss_policy must be one of "
                f"{valid_miss_policies}, got {self.miss_policy!r}")
        if self.max_capture_graphs < 1:
            raise ValueError(
                "dual_stream_attention_config.max_capture_graphs must be >= 1"
            )
        if len(self.capture_plans) > self.max_capture_graphs:
            raise ValueError(
                "dual_stream_attention_config.capture_plans exceeds "
                "max_capture_graphs")
        if self.enabled and not self.capture_plans:
            raise ValueError(
                "dual_stream_attention_config.capture_plans must be "
                "non-empty when enabled=True")

    def find_plan(
            self,
            num_tokens: int) -> Optional[DualStreamAttentionCapturePlan]:
        num_tokens = int(num_tokens)
        for plan in self.capture_plans:
            if plan.total_tokens == num_tokens or plan.graph_tokens == num_tokens:
                return plan
        return None


class MacroGraphConfig:
    """Configuration object for split-batch macro graph precapture."""

    def __init__(self, macro_graph_config: dict):
        self.enabled: bool = bool(macro_graph_config.get("enabled", False))
        self.capture_timing: str = str(
            macro_graph_config.get("capture_timing", "load_time"))
        self.schedule: str = str(
            macro_graph_config.get("schedule",
                                   "matmul_serial_attention_parallel"))
        self.backend: str = str(
            macro_graph_config.get("backend", "npugraph_ex"))
        self.backend_options: dict[str, Any] = dict(
            macro_graph_config.get("backend_options", {}))
        self.miss_policy: str = str(
            macro_graph_config.get("miss_policy", "error"))
        self.plan_source: str = str(
            macro_graph_config.get("plan_source", "explicit"))
        self.max_capture_graphs: int = int(
            macro_graph_config.get("max_capture_graphs", 16))
        self.validate_no_inner_aclgraph: bool = bool(
            macro_graph_config.get("validate_no_inner_aclgraph", True))
        self.allow_bucket_match: bool = bool(
            macro_graph_config.get("allow_bucket_match", False))
        self.allow_padded_replay: bool = bool(
            macro_graph_config.get("allow_padded_replay", False))
        self.relax_mixed_request_gates: bool = bool(
            macro_graph_config.get("relax_mixed_request_gates", False))
        self.bucket_min_total_tokens: int = int(
            macro_graph_config.get("bucket_min_total_tokens", 2))
        self.bucket_min_tokens_per_split: int = int(
            macro_graph_config.get("bucket_min_tokens_per_split", 1))
        self.bucket_min_actual_tokens_per_split: int = int(
            macro_graph_config.get("bucket_min_actual_tokens_per_split", 8))
        self.bucket_min_prefill_reqs_for_prefill_split: int = int(
            macro_graph_config.get(
                "bucket_min_prefill_reqs_for_prefill_split", 1))
        self.bucket_max_single_request_ratio: float = float(
            macro_graph_config.get("bucket_max_single_request_ratio", 1.0))
        self.bucket_padding_ratio_grace_tokens: int = int(
            macro_graph_config.get("bucket_padding_ratio_grace_tokens", 0))
        raw_max_padding_ratio = macro_graph_config.get(
            "max_padding_ratio_per_split", 0.0)
        if raw_max_padding_ratio is None:
            self.max_padding_ratio_per_split: Optional[float] = None
        else:
            self.max_padding_ratio_per_split = float(raw_max_padding_ratio)
        raw_capture_plans = macro_graph_config.get("capture_plans", [])
        self.capture_plans: list[MacroGraphCapturePlan] = [
            MacroGraphCapturePlan.from_config(raw_plan)
            for raw_plan in raw_capture_plans
        ]
        raw_capture_total_tokens = macro_graph_config.get(
            "capture_total_tokens", [])
        self.capture_total_tokens: list[int] = sorted({
            int(num_tokens)
            for num_tokens in raw_capture_total_tokens
        })
        self.planner_policy: str = str(
            macro_graph_config.get("planner_policy",
                                   macro_graph_config.get(
                                       "inplace_split_planner_policy",
                                       "macro_cube_balanced")))
        self.graph_token_alignment: int = int(
            macro_graph_config.get("graph_token_alignment", 64))
        self.min_split_graph_tokens: int = int(
            macro_graph_config.get("min_split_graph_tokens", 192))
        self.min_padding_saved_tokens: int = int(
            macro_graph_config.get("min_padding_saved_tokens", 64))

        valid_capture_timings = ("load_time", )
        if self.capture_timing not in valid_capture_timings:
            raise ValueError(
                "macro_graph_config.capture_timing must be one of "
                f"{valid_capture_timings}, got {self.capture_timing!r}")
        valid_schedules = ("matmul_serial_attention_parallel", )
        if self.schedule not in valid_schedules:
            raise ValueError(
                "macro_graph_config.schedule must be one of "
                f"{valid_schedules}, got {self.schedule!r}")
        valid_backends = ("npugraph_ex", "torchair_tagged_event")
        if self.backend not in valid_backends:
            raise ValueError(
                "macro_graph_config.backend must be one of "
                f"{valid_backends}, got {self.backend!r}")
        valid_miss_policies = ("error", "padding")
        if self.miss_policy not in valid_miss_policies:
            raise ValueError(
                "macro_graph_config.miss_policy must be one of "
                f"{valid_miss_policies}, got {self.miss_policy!r}")
        valid_plan_sources = ("explicit", "planner")
        if self.plan_source not in valid_plan_sources:
            raise ValueError(
                "macro_graph_config.plan_source must be one of "
                f"{valid_plan_sources}, got {self.plan_source!r}")
        valid_planner_policies = ("macro_cube_balanced", )
        if self.planner_policy not in valid_planner_policies:
            raise ValueError(
                "macro_graph_config.planner_policy must be one of "
                f"{valid_planner_policies}, got {self.planner_policy!r}")
        if self.max_capture_graphs < 1:
            raise ValueError(
                "macro_graph_config.max_capture_graphs must be >= 1")
        if len(self.capture_plans) > self.max_capture_graphs:
            raise ValueError(
                "macro_graph_config.capture_plans exceeds "
                "max_capture_graphs")
        if self.graph_token_alignment < 1:
            raise ValueError(
                "macro_graph_config.graph_token_alignment must be >= 1")
        if self.min_split_graph_tokens < 1:
            raise ValueError(
                "macro_graph_config.min_split_graph_tokens must be >= 1")
        if self.min_padding_saved_tokens < 0:
            raise ValueError(
                "macro_graph_config.min_padding_saved_tokens must be >= 0")
        if (self.max_padding_ratio_per_split is not None
                and self.max_padding_ratio_per_split < 0):
            raise ValueError(
                "macro_graph_config.max_padding_ratio_per_split must be >= 0")
        if self.bucket_min_total_tokens < 1:
            raise ValueError(
                "macro_graph_config.bucket_min_total_tokens must be >= 1")
        if self.bucket_min_tokens_per_split < 1:
            raise ValueError(
                "macro_graph_config.bucket_min_tokens_per_split must be >= 1")
        if self.bucket_min_actual_tokens_per_split < 1:
            raise ValueError(
                "macro_graph_config.bucket_min_actual_tokens_per_split must be >= 1")
        if self.bucket_min_prefill_reqs_for_prefill_split < 0:
            raise ValueError(
                "macro_graph_config."
                "bucket_min_prefill_reqs_for_prefill_split must be >= 0")
        if self.bucket_max_single_request_ratio <= 0:
            raise ValueError(
                "macro_graph_config.bucket_max_single_request_ratio must be > 0")
        if self.bucket_padding_ratio_grace_tokens < 0:
            raise ValueError(
                "macro_graph_config.bucket_padding_ratio_grace_tokens must be >= 0")
        if any(num_tokens < 1 for num_tokens in self.capture_total_tokens):
            raise ValueError(
                "macro_graph_config.capture_total_tokens must contain "
                "positive integers")
        if (self.plan_source == "explicit" and self.enabled
                and not self.capture_plans):
            raise ValueError(
                "macro_graph_config.capture_plans must be non-empty when "
                "enabled=True and plan_source='explicit'")
        if (self.plan_source == "planner" and self.enabled
                and not self.capture_total_tokens):
            raise ValueError(
                "macro_graph_config.capture_total_tokens must be non-empty "
                "when enabled=True and plan_source='planner'")


class SplitBatchConfig:
    """Configuration object for split_batch_config from additional_config.

    This is used by NPUModelRunner/AscendSplitBatchWrapper.
    """

    def __init__(self, split_batch_config: dict):
        self.enabled: bool = bool(split_batch_config.get("enabled", False))
        self.enable_parallel_streams: bool = bool(
            split_batch_config.get("enable_parallel_streams", False))
        self.mode: str = str(split_batch_config.get("mode",
                                                    "parallel_buffer"))
        self.num_splits: int = int(split_batch_config.get("num_splits", 2))
        self.min_batch_size_for_split: int = int(
            split_batch_config.get("min_batch_size_for_split", 4))

        # Optional separate capture sizes for the parallel-stream graph pool.
        # When set, the Second capture in _capture_model will use these sizes
        # instead of the main cudagraph_capture_sizes.  None means "reuse main
        # capture sizes" (legacy behaviour).
        raw_parallel_sizes = split_batch_config.get(
            "parallel_capture_sizes", None)
        if raw_parallel_sizes is not None:
            self.parallel_capture_sizes: list[int] = sorted(
                int(s) for s in raw_parallel_sizes)
        else:
            self.parallel_capture_sizes = None

        # When True, always split as (largest_main_graph_hit + remainder) and
        # skip the padding-saved threshold check.  Useful for benchmarking the
        # split path regardless of how much padding would be saved.
        self.force_split: bool = bool(
            split_batch_config.get("force_split", False))

        self.enable_inplace_lazy_capture: bool = bool(
            split_batch_config.get("enable_inplace_lazy_capture", True))
        self.inplace_serial_first: bool = bool(
            split_batch_config.get("inplace_serial_first", True))
        self.inplace_parallel_replay_policy: str = str(
            split_batch_config.get("inplace_parallel_replay_policy",
                                   "full_graph_parallel"))
        self.piecewise_scheduler_sync_policy: str = str(
            split_batch_config.get("piecewise_scheduler_sync_policy",
                                   "event_chain"))
        self.piecewise_attention_enqueue_policy: str = str(
            split_batch_config.get("piecewise_attention_enqueue_policy",
                                   "persistent_thread"))
        self.enable_mixed_request_split: bool = bool(
            split_batch_config.get("enable_mixed_request_split", False))
        self.mixed_request_split_policy: str = str(
            split_batch_config.get("mixed_request_split_policy",
                                   "balanced_attention"))
        self.mixed_request_split_execution_mode: str = str(
            split_batch_config.get("mixed_request_split_execution_mode",
                                   "dry_run"))
        self.mixed_request_min_total_tokens: int = int(
            split_batch_config.get("mixed_request_min_total_tokens", 128))
        self.mixed_request_min_tokens_per_split: int = int(
            split_batch_config.get("mixed_request_min_tokens_per_split", 64))
        self.mixed_request_max_single_request_ratio: float = float(
            split_batch_config.get("mixed_request_max_single_request_ratio",
                                   0.70))
        raw_mixed_max_padding_tokens = split_batch_config.get(
            "mixed_request_max_padding_tokens_per_split", None)
        if raw_mixed_max_padding_tokens is None:
            self.mixed_request_max_padding_tokens_per_split: Optional[
                int] = None
        else:
            self.mixed_request_max_padding_tokens_per_split = int(
                raw_mixed_max_padding_tokens)
        raw_mixed_max_padding_ratio = split_batch_config.get(
            "mixed_request_max_padding_ratio_per_split", 0.0)
        if raw_mixed_max_padding_ratio is None:
            self.mixed_request_max_padding_ratio_per_split: Optional[
                float] = None
        else:
            self.mixed_request_max_padding_ratio_per_split = float(
                raw_mixed_max_padding_ratio)
        self.mixed_request_min_prefill_reqs_for_prefill_split: int = int(
            split_batch_config.get(
                "mixed_request_min_prefill_reqs_for_prefill_split", 2))
        self.mixed_request_decode_weight: int = int(
            split_batch_config.get("mixed_request_decode_weight", 1))
        self.mixed_request_prefill_weight: int = int(
            split_batch_config.get("mixed_request_prefill_weight", 4))
        raw_inplace_max_remainder_tokens = split_batch_config.get(
            "inplace_max_remainder_tokens", None)
        if raw_inplace_max_remainder_tokens is None:
            self.inplace_max_remainder_tokens: Optional[int] = None
        else:
            self.inplace_max_remainder_tokens = int(
                raw_inplace_max_remainder_tokens)
        self.inplace_validate_metadata_ptrs: bool = bool(
            split_batch_config.get("inplace_validate_metadata_ptrs", False))
        self.inplace_force_pa_for_offset: bool = bool(
            split_batch_config.get("inplace_force_pa_for_offset", False))
        self.enable_inplace_spec_decode: bool = bool(
            split_batch_config.get("enable_inplace_spec_decode", False))
        self.enable_inplace_mrope: bool = bool(
            split_batch_config.get("enable_inplace_mrope", False))
        raw_inplace_split_planner_policy = split_batch_config.get(
            "inplace_split_planner_policy",
            split_batch_config.get("inplace_split_first_tokens_policy",
                                   "largest_lower"))
        self.inplace_split_planner_policy: str = str(
            raw_inplace_split_planner_policy)
        self.inplace_split_first_tokens_policy = (
            self.inplace_split_planner_policy)
        self.inplace_offset_match_policy: str = str(
            split_batch_config.get("inplace_offset_match_policy", "exact"))
        raw_inplace_offset_capture_sizes = split_batch_config.get(
            "inplace_offset_capture_sizes", None)
        if raw_inplace_offset_capture_sizes is not None:
            self.inplace_offset_capture_sizes: Optional[list[int]] = sorted(
                {int(s) for s in raw_inplace_offset_capture_sizes})
        else:
            self.inplace_offset_capture_sizes = None
        self.inplace_offset_min_graph_tokens: int = int(
            split_batch_config.get("inplace_offset_min_graph_tokens", 1))
        raw_inplace_offset_max_padding_tokens = split_batch_config.get(
            "inplace_offset_max_padding_tokens", None)
        if raw_inplace_offset_max_padding_tokens is None:
            self.inplace_offset_max_padding_tokens: Optional[int] = None
        else:
            self.inplace_offset_max_padding_tokens = int(
                raw_inplace_offset_max_padding_tokens)
        raw_inplace_offset_max_padding_ratio = split_batch_config.get(
            "inplace_offset_max_padding_ratio", None)
        if raw_inplace_offset_max_padding_ratio is None:
            self.inplace_offset_max_padding_ratio: Optional[float] = None
        else:
            self.inplace_offset_max_padding_ratio = float(
                raw_inplace_offset_max_padding_ratio)
        raw_inplace_offset_max_graph_tokens_by_start = (
            split_batch_config.get(
                "inplace_offset_max_graph_tokens_by_start", None))
        if raw_inplace_offset_max_graph_tokens_by_start is None:
            self.inplace_offset_max_graph_tokens_by_start: Optional[
                dict[int, int]] = None
        else:
            self.inplace_offset_max_graph_tokens_by_start = {
                int(start): int(max_graph_tokens)
                for start, max_graph_tokens in
                raw_inplace_offset_max_graph_tokens_by_start.items()
            }
        raw_inplace_offset_allowed_graph_tokens_by_start = (
            split_batch_config.get(
                "inplace_offset_allowed_graph_tokens_by_start", None))
        if raw_inplace_offset_allowed_graph_tokens_by_start is None:
            self.inplace_offset_allowed_graph_tokens_by_start: Optional[
                dict[int, list[int]]] = None
        else:
            self.inplace_offset_allowed_graph_tokens_by_start = {
                int(start): sorted({int(size) for size in sizes})
                for start, sizes in
                raw_inplace_offset_allowed_graph_tokens_by_start.items()
            }
        self.inplace_offset_prefer_cached_graph: bool = bool(
            split_batch_config.get("inplace_offset_prefer_cached_graph",
                                   False))
        self.inplace_offset_fallback_on_miss: bool = bool(
            split_batch_config.get("inplace_offset_fallback_on_miss", False))
        self.macro_graph_config = MacroGraphConfig(
            split_batch_config.get("macro_graph_config", {}))
        self.dual_stream_attention_config = DualStreamAttentionConfig(
            split_batch_config.get("dual_stream_attention_config", {}))

        valid_modes = ("parallel_buffer", "inplace_serial",
                       "inplace_parallel")
        if self.mode not in valid_modes:
            raise ValueError(
                "split_batch_config.mode must be one of "
                f"{valid_modes}, got {self.mode!r}")
        if self.num_splits < 2:
            raise ValueError("split_batch_config.num_splits must be >= 2")
        if self.min_batch_size_for_split < 1:
            raise ValueError(
                "split_batch_config.min_batch_size_for_split must be >= 1")
        if self.mode.startswith("inplace") and self.num_splits != 2:
            raise ValueError(
                "inplace split currently supports "
                "split_batch_config.num_splits=2 only")
        valid_replay_policies = ("full_graph_parallel",
                                 "piecewise_attention_parallel")
        if self.inplace_parallel_replay_policy not in valid_replay_policies:
            raise ValueError(
                "split_batch_config.inplace_parallel_replay_policy must be "
                f"one of {valid_replay_policies}, got "
                f"{self.inplace_parallel_replay_policy!r}")
        valid_piecewise_sync_policies = ("host_sync", "event_chain")
        if self.piecewise_scheduler_sync_policy not in (
                valid_piecewise_sync_policies):
            raise ValueError(
                "split_batch_config.piecewise_scheduler_sync_policy must be "
                f"one of {valid_piecewise_sync_policies}, got "
                f"{self.piecewise_scheduler_sync_policy!r}")
        valid_piecewise_attention_policies = ("per_piece_thread",
                                              "persistent_thread")
        if self.piecewise_attention_enqueue_policy not in (
                valid_piecewise_attention_policies):
            raise ValueError(
                "split_batch_config.piecewise_attention_enqueue_policy must "
                f"be one of {valid_piecewise_attention_policies}, got "
                f"{self.piecewise_attention_enqueue_policy!r}")
        valid_mixed_request_policies = ("balanced_attention", )
        if self.mixed_request_split_policy not in valid_mixed_request_policies:
            raise ValueError(
                "split_batch_config.mixed_request_split_policy must be one "
                f"of {valid_mixed_request_policies}, got "
                f"{self.mixed_request_split_policy!r}")
        valid_mixed_request_execution_modes = (
            "dry_run", "serial", "piecewise_attention_parallel")
        if self.mixed_request_split_execution_mode not in (
                valid_mixed_request_execution_modes):
            raise ValueError(
                "split_batch_config.mixed_request_split_execution_mode must "
                f"be one of {valid_mixed_request_execution_modes}, got "
                f"{self.mixed_request_split_execution_mode!r}")
        if self.enable_mixed_request_split:
            if self.mode not in ("inplace_serial", "inplace_parallel"):
                raise ValueError(
                    "split_batch_config.enable_mixed_request_split requires "
                    "mode to be 'inplace_serial' or 'inplace_parallel'")
            if (self.mixed_request_split_execution_mode == "serial"
                    and self.mode != "inplace_serial"):
                raise ValueError(
                    "split_batch_config.mixed_request_split_execution_mode="
                    "'serial' requires mode='inplace_serial'")
            if (self.mixed_request_split_execution_mode
                    == "piecewise_attention_parallel"
                    and self.mode != "inplace_parallel"):
                raise ValueError(
                    "split_batch_config.mixed_request_split_execution_mode="
                    "'piecewise_attention_parallel' requires "
                    "mode='inplace_parallel'")
            if self.mode == "inplace_parallel" and not self.enable_parallel_streams:
                raise ValueError(
                    "split_batch_config.enable_mixed_request_split with "
                    "mode='inplace_parallel' requires "
                    "enable_parallel_streams=True")
            if (self.mode == "inplace_parallel"
                    and self.inplace_parallel_replay_policy !=
                    "piecewise_attention_parallel"):
                raise ValueError(
                    "split_batch_config.enable_mixed_request_split with "
                    "mode='inplace_parallel' requires "
                    "inplace_parallel_replay_policy="
                    "'piecewise_attention_parallel'")
        if self.mixed_request_min_total_tokens < 1:
            raise ValueError(
                "split_batch_config.mixed_request_min_total_tokens must be "
                ">= 1")
        if self.mixed_request_min_tokens_per_split < 1:
            raise ValueError(
                "split_batch_config.mixed_request_min_tokens_per_split must "
                "be >= 1")
        if not (0 < self.mixed_request_max_single_request_ratio <= 1):
            raise ValueError(
                "split_batch_config.mixed_request_max_single_request_ratio "
                "must be in (0, 1]")
        if (self.mixed_request_max_padding_tokens_per_split is not None
                and self.mixed_request_max_padding_tokens_per_split < 0):
            raise ValueError(
                "split_batch_config."
                "mixed_request_max_padding_tokens_per_split must be >= 0")
        if (self.mixed_request_max_padding_ratio_per_split is not None
                and self.mixed_request_max_padding_ratio_per_split < 0):
            raise ValueError(
                "split_batch_config."
                "mixed_request_max_padding_ratio_per_split must be >= 0")
        if self.mixed_request_min_prefill_reqs_for_prefill_split < 1:
            raise ValueError(
                "split_batch_config."
                "mixed_request_min_prefill_reqs_for_prefill_split must be "
                ">= 1")
        if self.mixed_request_decode_weight < 1:
            raise ValueError(
                "split_batch_config.mixed_request_decode_weight must be >= 1")
        if self.mixed_request_prefill_weight < 1:
            raise ValueError(
                "split_batch_config.mixed_request_prefill_weight must be >= 1"
            )
        if (self.inplace_max_remainder_tokens is not None
                and self.inplace_max_remainder_tokens < 1):
            raise ValueError(
                "split_batch_config.inplace_max_remainder_tokens must be >= 1"
            )
        valid_offset_match_policies = ("exact", "bucket")
        if self.inplace_offset_match_policy not in valid_offset_match_policies:
            raise ValueError(
                "split_batch_config.inplace_offset_match_policy must be one "
                f"of {valid_offset_match_policies}, got "
                f"{self.inplace_offset_match_policy!r}")
        valid_first_tokens_policies = ("largest_lower", "balanced",
                                       "macro_cube_balanced")
        if self.inplace_split_planner_policy not in valid_first_tokens_policies:
            raise ValueError(
                "split_batch_config.inplace_split_planner_policy must be "
                f"one of {valid_first_tokens_policies}, got "
                f"{self.inplace_split_planner_policy!r}")
        if (self.inplace_split_planner_policy == "macro_cube_balanced"
                and not self.macro_graph_config.enabled):
            raise ValueError(
                "split_batch_config.inplace_split_planner_policy="
                "'macro_cube_balanced' requires "
                "split_batch_config.macro_graph_config.enabled=True")
        if self.macro_graph_config.enabled:
            if self.mode != "inplace_parallel":
                raise ValueError(
                    "split_batch_config.macro_graph_config.enabled requires "
                    "split_batch_config.mode='inplace_parallel'")
            if not self.enable_parallel_streams:
                raise ValueError(
                    "split_batch_config.macro_graph_config.enabled requires "
                    "split_batch_config.enable_parallel_streams=True")
            if self.enable_inplace_lazy_capture:
                raise ValueError(
                    "split_batch_config.macro_graph_config.enabled requires "
                    "split_batch_config.enable_inplace_lazy_capture=False")
        if (self.inplace_offset_capture_sizes is not None
                and any(size < 1
                        for size in self.inplace_offset_capture_sizes)):
            raise ValueError(
                "split_batch_config.inplace_offset_capture_sizes must contain "
                "positive integers")
        if self.inplace_offset_min_graph_tokens < 1:
            raise ValueError(
                "split_batch_config.inplace_offset_min_graph_tokens must be "
                ">= 1")
        if (self.inplace_offset_max_padding_tokens is not None
                and self.inplace_offset_max_padding_tokens < 0):
            raise ValueError(
                "split_batch_config.inplace_offset_max_padding_tokens must be "
                ">= 0")
        if (self.inplace_offset_max_padding_ratio is not None
                and self.inplace_offset_max_padding_ratio < 1.0):
            raise ValueError(
                "split_batch_config.inplace_offset_max_padding_ratio must be "
                ">= 1.0")
        if self.inplace_offset_max_graph_tokens_by_start is not None:
            if any(start < 0 or max_graph_tokens < 1 for start,
                   max_graph_tokens in
                   self.inplace_offset_max_graph_tokens_by_start.items()):
                raise ValueError(
                    "split_batch_config."
                    "inplace_offset_max_graph_tokens_by_start must map "
                    "non-negative starts to positive graph token limits")
        if self.inplace_offset_allowed_graph_tokens_by_start is not None:
            if any(start < 0 or not sizes or any(size < 1 for size in sizes)
                   for start, sizes in
                   self.inplace_offset_allowed_graph_tokens_by_start.items()):
                raise ValueError(
                    "split_batch_config."
                    "inplace_offset_allowed_graph_tokens_by_start must map "
                    "non-negative starts to non-empty positive graph token "
                    "lists")


_ASCEND_CONFIG: Optional[AscendConfig] = None


def init_ascend_config(vllm_config):
    additional_config = vllm_config.additional_config if vllm_config.additional_config is not None else {}
    refresh = additional_config.get("refresh",
                                    False) if additional_config else False
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is not None and not refresh:
        return _ASCEND_CONFIG
    _ASCEND_CONFIG = AscendConfig(vllm_config)
    return _ASCEND_CONFIG


def clear_ascend_config():
    global _ASCEND_CONFIG
    _ASCEND_CONFIG = None


def get_ascend_config():
    global _ASCEND_CONFIG
    if _ASCEND_CONFIG is None:
        raise RuntimeError(
            "Ascend config is not initialized. Please call init_ascend_config first."
        )
    return _ASCEND_CONFIG
