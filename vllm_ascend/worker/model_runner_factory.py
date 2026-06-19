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

import os
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config


def create_npu_model_runner(vllm_config: VllmConfig,
                            device: torch.device) -> Any:
    """Create the configured Ascend model runner.

    `VLLM_ASCEND_MODEL_RUNNER` is a convenient profiling override.  The
    structured config remains the preferred way to make benchmark runs
    reproducible.
    """
    ascend_config = get_ascend_config()
    configured_type = getattr(ascend_config.model_runner_config, "type", "v3")
    runner_type = os.getenv("VLLM_ASCEND_MODEL_RUNNER", configured_type)

    if runner_type == "v3":
        from vllm_ascend.worker.model_runner_v3 import NPUModelRunner

        return NPUModelRunner(vllm_config, device)

    if runner_type == "attention_only_macro":
        from vllm_ascend.worker.model_runner_attention_only_macro import (
            AttentionOnlyMacroNPUModelRunner,
        )

        logger.warning(
            "Using experimental Ascend model runner: attention_only_macro")
        return AttentionOnlyMacroNPUModelRunner(vllm_config, device)

    raise ValueError(
        "Unsupported Ascend model runner type "
        f"{runner_type!r}. Expected one of ('v3', 'attention_only_macro').")
