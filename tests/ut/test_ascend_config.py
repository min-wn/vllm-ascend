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
# This file is a part of the vllm-ascend project.
#

from vllm.config import VllmConfig

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import (SplitBatchConfig, clear_ascend_config,
                                       get_ascend_config, init_ascend_config)


class TestAscendConfig(TestBase):

    @staticmethod
    def _clean_up_ascend_config(func):

        def wrapper(*args, **kwargs):
            clear_ascend_config()
            func(*args, **kwargs)
            clear_ascend_config()

        return wrapper

    @_clean_up_ascend_config
    def test_init_ascend_config_without_additional_config(self):
        test_vllm_config = VllmConfig()
        # No additional config given, check the default value here.
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertIsNone(ascend_config.expert_map_path)
        self.assertFalse(ascend_config.multistream_overlap_shared_expert)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertTrue(ascend_compilation_config.fuse_norm_quant)

    @_clean_up_ascend_config
    def test_init_ascend_config_with_additional_config(self):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "ascend_compilation_config": {
                "fuse_norm_quant": False,
            },
            "multistream_overlap_shared_expert": True,
            "expert_map_path": "test_expert_map_path",
            "refresh": True,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(ascend_config.expert_map_path, "test_expert_map_path")
        self.assertTrue(ascend_config.multistream_overlap_shared_expert)
        self.assertFalse(ascend_config.enable_npugraph_ex)

        ascend_compilation_config = ascend_config.ascend_compilation_config
        self.assertFalse(ascend_compilation_config.fuse_norm_quant)

    @_clean_up_ascend_config
    def test_init_ascend_config_enable_npugraph_ex(self):
        test_vllm_config = VllmConfig()
        test_vllm_config.additional_config = {
            "enable_npugraph_ex": True,
            "refresh": True,
        }
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertTrue(ascend_config.enable_npugraph_ex)

    @_clean_up_ascend_config
    def test_get_ascend_config(self):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)

    @_clean_up_ascend_config
    def test_get_ascend_config_without_init(self):
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    @_clean_up_ascend_config
    def test_clear_ascend_config(self):
        test_vllm_config = VllmConfig()
        ascend_config = init_ascend_config(test_vllm_config)
        self.assertEqual(get_ascend_config(), ascend_config)
        clear_ascend_config()
        with self.assertRaises(RuntimeError):
            get_ascend_config()

    def test_split_batch_config_defaults_keep_parallel_buffer(self):
        split_config = SplitBatchConfig({})

        self.assertFalse(split_config.enabled)
        self.assertFalse(split_config.enable_parallel_streams)
        self.assertEqual(split_config.mode, "parallel_buffer")
        self.assertEqual(split_config.num_splits, 2)
        self.assertEqual(split_config.min_batch_size_for_split, 4)
        self.assertIsNone(split_config.parallel_capture_sizes)
        self.assertFalse(split_config.force_split)
        self.assertTrue(split_config.enable_inplace_lazy_capture)
        self.assertTrue(split_config.inplace_serial_first)
        self.assertIsNone(split_config.inplace_max_remainder_tokens)
        self.assertFalse(split_config.inplace_validate_metadata_ptrs)
        self.assertFalse(split_config.inplace_force_pa_for_offset)
        self.assertFalse(split_config.enable_inplace_spec_decode)
        self.assertFalse(split_config.enable_inplace_mrope)
        self.assertEqual(split_config.inplace_split_planner_policy,
                         "largest_lower")
        self.assertEqual(split_config.inplace_split_first_tokens_policy,
                         "largest_lower")
        self.assertEqual(split_config.inplace_offset_match_policy, "exact")
        self.assertIsNone(split_config.inplace_offset_capture_sizes)
        self.assertEqual(split_config.inplace_offset_min_graph_tokens, 1)
        self.assertIsNone(split_config.inplace_offset_max_padding_tokens)
        self.assertIsNone(split_config.inplace_offset_max_padding_ratio)
        self.assertIsNone(
            split_config.inplace_offset_max_graph_tokens_by_start)
        self.assertIsNone(
            split_config.inplace_offset_allowed_graph_tokens_by_start)
        self.assertFalse(split_config.inplace_offset_prefer_cached_graph)
        self.assertFalse(split_config.inplace_offset_fallback_on_miss)

    def test_split_batch_config_legacy_parallel_streams_compat(self):
        split_config = SplitBatchConfig({
            "enabled": True,
            "enable_parallel_streams": True,
            "num_splits": 3,
            "min_batch_size_for_split": 8,
            "parallel_capture_sizes": [128, 64],
            "force_split": True,
        })

        self.assertTrue(split_config.enabled)
        self.assertTrue(split_config.enable_parallel_streams)
        self.assertEqual(split_config.mode, "parallel_buffer")
        self.assertEqual(split_config.num_splits, 3)
        self.assertEqual(split_config.min_batch_size_for_split, 8)
        self.assertEqual(split_config.parallel_capture_sizes, [64, 128])
        self.assertTrue(split_config.force_split)

    def test_split_batch_config_accepts_inplace_serial(self):
        split_config = SplitBatchConfig({
            "enabled": True,
            "mode": "inplace_serial",
            "num_splits": 2,
            "enable_inplace_lazy_capture": False,
            "inplace_serial_first": False,
            "inplace_max_remainder_tokens": 64,
            "inplace_validate_metadata_ptrs": True,
            "inplace_force_pa_for_offset": False,
            "enable_inplace_spec_decode": True,
            "enable_inplace_mrope": True,
            "inplace_split_planner_policy": "balanced",
            "inplace_offset_match_policy": "bucket",
            "inplace_offset_capture_sizes": [128, 32, 64],
            "inplace_offset_min_graph_tokens": 32,
            "inplace_offset_max_padding_tokens": 127,
            "inplace_offset_max_padding_ratio": 8.0,
            "inplace_offset_max_graph_tokens_by_start": {
                "128": 64,
                256: 128,
            },
            "inplace_offset_allowed_graph_tokens_by_start": {
                "32": [32, 16],
                64: [64, 16, 32],
            },
            "inplace_offset_prefer_cached_graph": True,
            "inplace_offset_fallback_on_miss": True,
        })

        self.assertEqual(split_config.mode, "inplace_serial")
        self.assertFalse(split_config.enable_inplace_lazy_capture)
        self.assertFalse(split_config.inplace_serial_first)
        self.assertEqual(split_config.inplace_max_remainder_tokens, 64)
        self.assertTrue(split_config.inplace_validate_metadata_ptrs)
        self.assertFalse(split_config.inplace_force_pa_for_offset)
        self.assertTrue(split_config.enable_inplace_spec_decode)
        self.assertTrue(split_config.enable_inplace_mrope)
        self.assertEqual(split_config.inplace_split_planner_policy,
                         "balanced")
        self.assertEqual(split_config.inplace_split_first_tokens_policy,
                         "balanced")
        self.assertEqual(split_config.inplace_offset_match_policy, "bucket")
        self.assertEqual(split_config.inplace_offset_capture_sizes,
                         [32, 64, 128])
        self.assertEqual(split_config.inplace_offset_min_graph_tokens, 32)
        self.assertEqual(split_config.inplace_offset_max_padding_tokens, 127)
        self.assertEqual(split_config.inplace_offset_max_padding_ratio, 8.0)
        self.assertEqual(
            split_config.inplace_offset_max_graph_tokens_by_start, {
                128: 64,
                256: 128,
            })
        self.assertEqual(
            split_config.inplace_offset_allowed_graph_tokens_by_start, {
                32: [16, 32],
                64: [16, 32, 64],
            })
        self.assertTrue(split_config.inplace_offset_prefer_cached_graph)
        self.assertTrue(split_config.inplace_offset_fallback_on_miss)

    def test_split_batch_config_accepts_inplace_force_pa_for_offset(self):
        split_config = SplitBatchConfig({
            "enabled": True,
            "mode": "inplace_serial",
            "num_splits": 2,
            "inplace_force_pa_for_offset": True,
        })

        self.assertTrue(split_config.inplace_force_pa_for_offset)

    def test_split_batch_config_accepts_inplace_parallel(self):
        split_config = SplitBatchConfig({
            "enabled": True,
            "mode": "inplace_parallel",
            "num_splits": 2,
        })

        self.assertEqual(split_config.mode, "inplace_parallel")
        self.assertEqual(split_config.num_splits, 2)

    def test_split_batch_config_rejects_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "split_batch_config.mode"):
            SplitBatchConfig({"mode": "auto"})

    def test_split_batch_config_rejects_inplace_num_splits_not_two(self):
        with self.assertRaisesRegex(ValueError, "num_splits=2"):
            SplitBatchConfig({
                "mode": "inplace_serial",
                "num_splits": 3,
            })
        with self.assertRaisesRegex(ValueError, "num_splits=2"):
            SplitBatchConfig({
                "mode": "inplace_parallel",
                "num_splits": 3,
            })

    def test_split_batch_config_rejects_invalid_inplace_max_remainder_tokens(
            self):
        with self.assertRaisesRegex(ValueError,
                                    "inplace_max_remainder_tokens"):
            SplitBatchConfig({"inplace_max_remainder_tokens": 0})

    def test_split_batch_config_rejects_invalid_offset_policy(self):
        with self.assertRaisesRegex(ValueError,
                                    "inplace_offset_match_policy"):
            SplitBatchConfig({"inplace_offset_match_policy": "relaxed"})
        with self.assertRaisesRegex(
                ValueError, "inplace_split_planner_policy"):
            SplitBatchConfig({"inplace_split_planner_policy": "round_robin"})

    def test_split_batch_config_rejects_invalid_offset_padding_limits(self):
        with self.assertRaisesRegex(ValueError,
                                    "inplace_offset_capture_sizes"):
            SplitBatchConfig({"inplace_offset_capture_sizes": [32, 0]})
        with self.assertRaisesRegex(ValueError,
                                    "inplace_offset_min_graph_tokens"):
            SplitBatchConfig({"inplace_offset_min_graph_tokens": 0})
        with self.assertRaisesRegex(ValueError,
                                    "inplace_offset_max_padding_tokens"):
            SplitBatchConfig({"inplace_offset_max_padding_tokens": -1})
        with self.assertRaisesRegex(ValueError,
                                    "inplace_offset_max_padding_ratio"):
            SplitBatchConfig({"inplace_offset_max_padding_ratio": 0.5})
        with self.assertRaisesRegex(
                ValueError, "inplace_offset_max_graph_tokens_by_start"):
            SplitBatchConfig({
                "inplace_offset_max_graph_tokens_by_start": {
                    -1: 64,
                }
            })
        with self.assertRaisesRegex(
                ValueError, "inplace_offset_max_graph_tokens_by_start"):
            SplitBatchConfig({
                "inplace_offset_max_graph_tokens_by_start": {
                    128: 0,
                }
            })
        with self.assertRaisesRegex(
                ValueError, "inplace_offset_allowed_graph_tokens_by_start"):
            SplitBatchConfig({
                "inplace_offset_allowed_graph_tokens_by_start": {
                    32: []
                }
            })
