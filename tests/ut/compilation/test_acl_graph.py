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

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, ForwardContext

from tests.ut.base import TestBase
from vllm_ascend.attention.attention_v1 import (AscendMetadata,
                                                AscendMetadataForDecode)
from vllm_ascend.attention.mla_v1 import (AscendMLADecodeMetadata,
                                          AscendMLAMetadata)
from vllm_ascend.compilation.acl_graph import (
    ACLGraphEntry, ACLGraphWrapper, GraphParams, ensure_graph_param_key,
    _collect_attn_metadata_tensor_infos,
    get_graph_param_key, get_graph_params, get_mtp_graph_params,
    graph_param_key_info, maybe_template_fia_seq_lens,
    require_graph_param_key, set_graph_params, set_mtp_graph_params,
    update_attn_dcp_pcp_params, _update_attn_fia_params,
    update_graph_params_workspaces,
    update_mla_attn_dcp_pcp_params, update_mtp_graph_params_workspaces)


class TestACLGraphEntry(TestBase):

    def test_aclgraph_entry_initialization(self):
        """Test ACLGraphEntry initialization with default values"""
        batch_descriptor = BatchDescriptor(
            num_tokens=30,
            uniform=False,
        )

        entry = ACLGraphEntry(batch_descriptor=batch_descriptor)

        self.assertEqual(entry.batch_descriptor, batch_descriptor)
        self.assertIsNone(entry.aclgraph)
        self.assertIsNone(entry.output)
        self.assertIsNone(entry.input_addresses)


    def test_aclgraph_entry_with_values(self):
        """Test ACLGraphEntry initialization with specified values"""
        batch_descriptor = BatchDescriptor(
            num_tokens=30,
            uniform=False,
        )

        mock_graph = MagicMock()
        mock_output = MagicMock()
        input_addresses = [12345, 67890]

        entry = ACLGraphEntry(batch_descriptor=batch_descriptor,
                              aclgraph=mock_graph,
                              output=mock_output,
                              input_addresses=input_addresses)

        self.assertEqual(entry.batch_descriptor, batch_descriptor)
        self.assertEqual(entry.aclgraph, mock_graph)
        self.assertEqual(entry.output, mock_output)
        self.assertEqual(entry.input_addresses, input_addresses)


class TestGraphParamKey(TestBase):

    def test_get_graph_param_key_uses_int_for_default_descriptor(self):
        context = MagicMock()
        context.is_mtp_model = False
        context.batch_descriptor = BatchDescriptor(num_tokens=32,
                                                   num_reqs=32,
                                                   uniform=True,
                                                   has_lora=False)

        self.assertEqual(get_graph_param_key(context, 32), 32)

    def test_get_graph_param_key_uses_descriptor_for_offset(self):
        descriptor = BatchDescriptor(num_tokens=32,
                                     num_reqs=32,
                                     uniform=True,
                                     has_lora=False,
                                     start_num_tokens=384,
                                     graph_variant="inplace_serial",
                                     attention_backend="fia")
        context = MagicMock()
        context.is_mtp_model = False
        context.batch_descriptor = descriptor

        self.assertEqual(get_graph_param_key(context, 32), descriptor)

    def test_get_graph_param_key_keeps_mtp_int_by_default(self):
        descriptor = BatchDescriptor(num_tokens=32,
                                     num_reqs=32,
                                     uniform=True,
                                     has_lora=False,
                                     start_num_tokens=384,
                                     graph_variant="inplace_serial",
                                     attention_backend="fia")
        context = MagicMock()
        context.is_mtp_model = True
        context.batch_descriptor = descriptor

        self.assertEqual(get_graph_param_key(context, 32), 32)

    def test_offset_keys_are_isolated(self):
        desc_1 = BatchDescriptor(num_tokens=32,
                                 num_reqs=32,
                                 uniform=True,
                                 has_lora=False,
                                 start_num_tokens=256)
        desc_2 = BatchDescriptor(num_tokens=32,
                                 num_reqs=32,
                                 uniform=True,
                                 has_lora=False,
                                 start_num_tokens=384)
        graph_params = GraphParams({}, {}, {}, {})

        ensure_graph_param_key(graph_params, desc_1)
        ensure_graph_param_key(graph_params, desc_2)
        graph_params.attn_params[desc_1].append("a")
        graph_params.attn_params[desc_2].append("b")

        self.assertEqual(graph_params.attn_params[desc_1], ["a"])
        self.assertEqual(graph_params.attn_params[desc_2], ["b"])
        self.assertNotEqual(graph_params.attn_params[desc_1],
                            graph_params.attn_params[desc_2])

    def test_require_graph_param_key_rejects_missing_key(self):
        graph_params = GraphParams({}, {}, {}, {})
        descriptor = BatchDescriptor(num_tokens=32, start_num_tokens=384)

        with self.assertRaises(KeyError):
            require_graph_param_key(graph_params,
                                    descriptor,
                                    op="unit_test")

    def test_update_graph_params_workspaces_supports_offset_key(self):
        descriptor = BatchDescriptor(num_tokens=32, start_num_tokens=384)
        workspace = object()
        graph_params = GraphParams({}, {}, {}, {})

        with patch('vllm_ascend.compilation.acl_graph._graph_params',
                   new=graph_params):
            with patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors',
                       side_effect=lambda value: value):
                update_graph_params_workspaces(descriptor, workspace)

        self.assertIs(graph_params.workspaces[descriptor], workspace)
        self.assertNotIn(32, graph_params.workspaces)

    def test_graph_param_key_info_for_descriptor(self):
        descriptor = BatchDescriptor(num_tokens=32,
                                     num_reqs=32,
                                     uniform=True,
                                     has_lora=False,
                                     start_num_tokens=384,
                                     graph_variant="inplace_serial",
                                     attention_backend="fia")

        self.assertEqual(
            graph_param_key_info(descriptor),
            {
                "kind": "batch_descriptor",
                "num_tokens": 32,
                "num_reqs": 32,
                "uniform": True,
                "has_lora": False,
                "start_num_tokens": 384,
                "graph_variant": "inplace_serial",
                "attention_backend": "fia",
                "capture_metadata_mode": "",
            },
        )

    def test_maybe_template_fia_seq_lens_copies_tail_to_target_t(self):
        descriptor = BatchDescriptor(num_tokens=32,
                                     num_reqs=32,
                                     uniform=True,
                                     has_lora=False,
                                     start_num_tokens=384,
                                     graph_variant="inplace_serial",
                                     attention_backend="fia",
                                     capture_metadata_mode="template")
        context = SimpleNamespace(batch_descriptor=descriptor)
        seq_lens = [9, 9]

        templated_seq_lens = maybe_template_fia_seq_lens(
            context, seq_lens, target_t=32)

        self.assertEqual(templated_seq_lens, [9, 32])
        self.assertEqual(seq_lens, [9, 9])

    def test_maybe_template_fia_seq_lens_ignores_regular_fia_descriptor(self):
        descriptor = BatchDescriptor(num_tokens=384,
                                     num_reqs=384,
                                     uniform=True,
                                     has_lora=False,
                                     attention_backend="fia")
        context = SimpleNamespace(batch_descriptor=descriptor)
        seq_lens = [9, 9]

        templated_seq_lens = maybe_template_fia_seq_lens(
            context, seq_lens, target_t=32)

        self.assertIs(templated_seq_lens, seq_lens)

    def test_update_attn_fia_params_templates_offset_seq_lens(self):
        descriptor = BatchDescriptor(num_tokens=32,
                                     num_reqs=32,
                                     uniform=True,
                                     has_lora=False,
                                     start_num_tokens=384,
                                     graph_variant="inplace_serial",
                                     attention_backend="fia",
                                     capture_metadata_mode="template")
        forward_context = SimpleNamespace(
            is_mtp_model=False,
            batch_descriptor=descriptor,
            attn_metadata={
                "layer.0":
                SimpleNamespace(seq_lens_list=[9, 9],
                                actual_seq_lengths_q=[32])
            },
        )
        event = SimpleNamespace(record=Mock())
        param = (
            "query",
            torch.empty(32),
            "value",
            "block_tables",
            "attn_mask",
            256,
            [9, 9],
            [32],
            2,
            14,
            1.0,
            "attn_output",
            "softmax_lse",
        )
        graph_params = GraphParams(
            events={descriptor: [event]},
            workspaces={descriptor: "workspace"},
            handles={descriptor: ["handle"]},
            attn_params={descriptor: [param]},
        )

        with patch(
                "vllm_ascend.compilation.acl_graph.get_graph_params",
                return_value=graph_params), patch(
                    "vllm_ascend.compilation.acl_graph.torch.npu.stream",
                    return_value=nullcontext()), patch(
                        "vllm_ascend.compilation.acl_graph.torch.npu.graph_task_update_begin"
                    ), patch(
                        "vllm_ascend.compilation.acl_graph.torch.npu.graph_task_update_end"
                    ), patch(
                        "vllm_ascend.compilation.acl_graph.torch_npu.npu_fused_infer_attention_score.out"
                    ) as mock_fia:
            _update_attn_fia_params("stream", forward_context, 32)

        self.assertEqual(
            mock_fia.call_args.kwargs["actual_seq_lengths_kv"], [9, 32])
        self.assertEqual(
            forward_context.attn_metadata["layer.0"].seq_lens_list, [9, 9])
        event.record.assert_called_once_with("stream")


class TestACLGraphWrapper(TestBase):

    def setUp(self):
        """Set up test fixtures"""
        super().setUp()

        # Mock VllmConfig
        self.mock_vllm_config = MagicMock(spec=VllmConfig)
        self.mock_vllm_config.compilation_config = MagicMock()

        # Mock runnable function
        self.mock_runnable = MagicMock(return_value="test_output")

        # Mock graph pool
        self.mock_graph_pool = MagicMock()

        # Mock CUDAGraphOptions
        self.mock_cudagraph_options = MagicMock(spec=CUDAGraphOptions)
        self.mock_cudagraph_options.debug_log_enable = False
        self.mock_cudagraph_options.gc_disable = False
        self.mock_cudagraph_options.weak_ref_output = False

        # Mock BatchDescriptor
        self.mock_batch_descriptor = BatchDescriptor(
            num_tokens=30,
            uniform=False,
        )

        # Mock ForwardContext
        self.mock_forward_context = MagicMock(spec=ForwardContext)
        self.mock_forward_context.batch_descriptor = self.mock_batch_descriptor
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_initialization_with_default_options(self, mock_envs,
                                                 mock_current_platform):
        """Test ACLGraphWrapper initialization with default CUDAGraphOptions"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool

        wrapper = ACLGraphWrapper(runnable=self.mock_runnable,
                                  vllm_config=self.mock_vllm_config,
                                  runtime_mode=CUDAGraphMode.FULL)

        self.assertEqual(wrapper.runnable, self.mock_runnable)
        self.assertEqual(wrapper.vllm_config, self.mock_vllm_config)
        self.assertEqual(wrapper.graph_pool, self.mock_graph_pool)
        self.assertEqual(wrapper.runtime_mode, CUDAGraphMode.FULL)
        self.assertFalse(wrapper.is_debugging_mode)
        self.assertIsInstance(wrapper.aclgraph_options, CUDAGraphOptions)
        self.assertEqual(wrapper.concrete_aclgraph_entries, {})

    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_initialization_with_custom_options(self, mock_envs,
                                                mock_current_platform):
        """Test ACLGraphWrapper initialization with custom CUDAGraphOptions"""
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        self.assertEqual(wrapper.runnable, self.mock_runnable)
        self.assertEqual(wrapper.vllm_config, self.mock_vllm_config)
        self.assertEqual(wrapper.graph_pool, self.mock_graph_pool)
        self.assertEqual(wrapper.runtime_mode, CUDAGraphMode.FULL)
        self.assertTrue(wrapper.is_debugging_mode)
        self.assertEqual(wrapper.aclgraph_options, self.mock_cudagraph_options)
        self.assertEqual(wrapper.concrete_aclgraph_entries, {})

    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_initialization_assertion_error(self, mock_envs,
                                            mock_current_platform):
        """Test ACLGraphWrapper initialization raises AssertionError for NONE mode"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool

        with self.assertRaises(AssertionError):
            ACLGraphWrapper(runnable=self.mock_runnable,
                            vllm_config=self.mock_vllm_config,
                            runtime_mode=CUDAGraphMode.NONE)

    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_call_with_none_runtime_mode(self, mock_envs,
                                         mock_current_platform,
                                         mock_get_forward_context):
        """Test __call__ method when runtime mode is NONE"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.NONE

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        result = wrapper("arg1", "arg2")

        # Should call the runnable directly without graph capture
        self.mock_runnable.assert_called_once_with("arg1", "arg2")
        self.assertEqual(result, "test_output")

    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_call_with_mismatched_runtime_mode(self, mock_envs,
                                               mock_current_platform,
                                               mock_get_forward_context):
        """Test __call__ method when runtime mode doesn't match wrapper mode"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.PIECEWISE  # Different from FULL

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        result = wrapper("arg1", "arg2")

        # Should call the runnable directly without graph capture
        self.mock_runnable.assert_called_once_with("arg1", "arg2")
        self.assertEqual(result, "test_output")

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_capture_graph_first_time(
            self, mock_weak_ref_tensors, mock_compilation_counter, mock_envs,
            mock_current_platform, mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method captures graph for the first time"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors to return the same output
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Set up the compilation counter mock
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # Call the wrapper
        result = wrapper(test_tensor, "arg2")

        # Verify graph capture happened
        mock_validate_cudagraph_capturing_enabled.assert_called_once()
        mock_torch.npu.NPUGraph.assert_called_once()
        mock_torch.npu.graph.assert_called_once_with(mock_npu_graph,
                                                     pool=self.mock_graph_pool)
        self.mock_runnable.assert_called_once_with(test_tensor, "arg2")

        # Verify the entry was created and updated
        self.assertIn(self.mock_batch_descriptor,
                      wrapper.concrete_aclgraph_entries)
        entry = wrapper.concrete_aclgraph_entries[self.mock_batch_descriptor]
        self.assertEqual(entry.aclgraph, mock_npu_graph)
        self.assertEqual(entry.output, "weak_ref_output")

        # Verify compilation counter was incremented
        self.assertEqual(mock_compilation_counter.num_cudagraph_captured, 1)

        # Should return the original output (not weak ref)
        self.assertEqual(result, "test_output")

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_replay_graph(self, mock_weak_ref_tensors,
                               mock_compilation_counter, mock_envs,
                               mock_current_platform, mock_get_forward_context,
                               mock_validate_cudagraph_capturing_enabled,
                               mock_torch):
        """Test __call__ method replays graph when already captured"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors to return the same output
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Set up the compilation counter mock
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # First call to capture the graph
        first_result = wrapper(test_tensor, "arg2")

        # Verify graph capture happened during first call
        mock_validate_cudagraph_capturing_enabled.assert_called_once()
        mock_torch.npu.NPUGraph.assert_called_once()
        mock_torch.npu.graph.assert_called_once()

        # Reset mock to track second call
        self.mock_runnable.reset_mock()
        mock_npu_graph.reset_mock()

        # Second call should replay the graph
        second_result = wrapper(test_tensor, "arg2")

        # Verify runnable was called only during capture (not during replay)
        self.mock_runnable.assert_not_called()

        # Verify graph replay happened
        mock_npu_graph.replay.assert_called_once()

        # Both calls should return the weak ref output
        self.assertEqual(first_result, "test_output")  # Original output
        self.assertEqual(second_result, "weak_ref_output")  # Weak ref output

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_with_debug_mode_input_address_check(
            self, mock_weak_ref_tensors, mock_envs, mock_current_platform,
            mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method with debug mode input address checking"""
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"  # Enable debug mode
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Create a mock tensor as the output of runnable
        mock_output_tensor = torch.tensor([4, 5, 6])
        self.mock_runnable.return_value = mock_output_tensor

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # First call to capture the graph
        tensor = torch.tensor([1, 2, 3])  # Create tensor once
        _ = wrapper(tensor, "arg2")

        # Second call with same tensor addresses should work
        _ = wrapper(tensor, "arg2")  # Use the same tensor object

        # Should not raise AssertionError
        self.assertTrue(True)

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_with_debug_mode_input_address_mismatch(
            self, mock_weak_ref_tensors, mock_envs, mock_current_platform,
            mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method with debug mode input address mismatch raises AssertionError"""
        mock_envs.VLLM_LOGGING_LEVEL = "DEBUG"  # Enable debug mode
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors
        mock_weak_ref_tensors.return_value = "weak_ref_output"

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Create a mock tensor as the output of runnable
        mock_output_tensor = torch.tensor([4, 5, 6])
        self.mock_runnable.return_value = mock_output_tensor

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # First call to capture the graph
        tensor1 = torch.tensor([1, 2, 3])
        _ = wrapper(tensor1, "arg2")

        # Second call with different tensor addresses should raise AssertionError
        tensor2 = torch.tensor([4, 5,
                                6])  # Different values, different address

        with self.assertRaises(AssertionError) as context:
            wrapper(tensor2, "arg2")

        self.assertIn("Input addresses for aclgraphs are different",
                      str(context.exception))

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    @patch('vllm_ascend.compilation.acl_graph.patch')
    def test_call_capture_graph_with_gc_disable(
            self, mock_patch, mock_weak_ref_tensors, mock_compilation_counter,
            mock_envs, mock_current_platform, mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method captures graph with gc_disable option enabled"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Enable gc_disable option
        self.mock_cudagraph_options.gc_disable = True
        # weak_ref_output is not enabled by default

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock patch context manager
        mock_exit_stack = MagicMock()
        mock_patch.return_value = mock_exit_stack
        mock_exit_stack.enter_context = Mock()

        # Mock weak_ref_tensors to simulate the actual behavior:
        # 1. First call (inside the graph context) should return "inner_output"
        # 2. Second call (for entry.output) should return "weak_ref_output"
        mock_weak_ref_tensors.side_effect = ["inner_output", "weak_ref_output"]

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Set up the compilation counter mock
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # Call the wrapper
        result = wrapper(test_tensor, "arg2")

        # Verify patch was called to disable gc
        self.assertTrue(mock_patch.called)

        # Verify graph capture happened
        mock_validate_cudagraph_capturing_enabled.assert_called_once()
        mock_torch.npu.NPUGraph.assert_called_once()
        mock_torch.npu.graph.assert_called_once_with(mock_npu_graph,
                                                     pool=self.mock_graph_pool)

        # Should return the original output (not weak ref) since weak_ref_output is not enabled
        self.assertEqual(result, "test_output")

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch(
        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
    )
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_call_capture_graph_with_weak_ref_output(
            self, mock_weak_ref_tensors, mock_compilation_counter, mock_envs,
            mock_current_platform, mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        """Test __call__ method captures graph with weak_ref_output option enabled"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Enable weak_ref_output option
        self.mock_cudagraph_options.weak_ref_output = True

        # Mock torch.npu.NPUGraph
        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph

        # Mock torch.npu.graph context manager
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)

        # Mock weak_ref_tensors to simulate the actual behavior:
        # 1. First call (inside the graph context with weak_ref_output=True) should return "weak_ref_output"
        # 2. Second call (for entry.output) should return "weak_ref_output"
        mock_weak_ref_tensors.side_effect = [
            "weak_ref_output", "weak_ref_output"
        ]

        # Ensure torch.Tensor can be correctly identified by isinstance
        mock_torch.Tensor = torch.Tensor

        # Set up the compilation counter mock
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Create a real torch tensor for the test, not a mock
        test_tensor = torch.tensor([1, 2, 3])

        # Call the wrapper
        result = wrapper(test_tensor, "arg2")

        # Verify weak_ref_tensors was called twice (once for inner output, once for final output)
        self.assertEqual(mock_weak_ref_tensors.call_count, 2)

        # Verify graph capture happened
        mock_validate_cudagraph_capturing_enabled.assert_called_once()
        mock_torch.npu.NPUGraph.assert_called_once()
        mock_torch.npu.graph.assert_called_once_with(mock_npu_graph,
                                                     pool=self.mock_graph_pool)

        # Should return the weak ref output when weak_ref_output option is enabled
        self.assertEqual(result, "weak_ref_output")

    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.logger')
    def test_call_capture_graph_with_debug_log(self, mock_logger, mock_envs,
                                               mock_current_platform,
                                               mock_get_forward_context):
        """Test __call__ method captures graph with debug logging enabled"""
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL

        # Enable debug logging
        self.mock_cudagraph_options.debug_log_enable = True
        # weak_ref_output is not enabled by default

        # Mock torch
        with patch('vllm_ascend.compilation.acl_graph.torch') as mock_torch:
            # Mock torch.npu.NPUGraph
            mock_npu_graph = MagicMock()
            mock_torch.npu.NPUGraph.return_value = mock_npu_graph

            # Mock torch.npu.graph context manager
            mock_graph_context = MagicMock()
            mock_torch.npu.graph.return_value = mock_graph_context
            mock_graph_context.__enter__ = Mock(return_value=None)
            mock_graph_context.__exit__ = Mock(return_value=None)

            # Ensure torch.Tensor can be correctly identified by isinstance
            mock_torch.Tensor = torch.Tensor

            # Mock weak_ref_tensors
            with patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors'
                       ) as mock_weak_ref_tensors:
                # Mock weak_ref_tensors to simulate the actual behavior:
                # 1. First call (inside the graph context) should return "inner_output"
                # 2. Second call (for entry.output) should return "weak_ref_output"
                mock_weak_ref_tensors.side_effect = [
                    "inner_output", "weak_ref_output"
                ]

                # Mock validate_cudagraph_capturing_enabled
                with patch(
                        'vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled'
                ):
                    wrapper = ACLGraphWrapper(
                        runnable=self.mock_runnable,
                        vllm_config=self.mock_vllm_config,
                        runtime_mode=CUDAGraphMode.FULL,
                        cudagraph_options=self.mock_cudagraph_options)

                    # Create a real torch tensor for the test, not a mock
                    test_tensor = torch.tensor([1, 2, 3])

                    # Call the wrapper
                    _ = wrapper(test_tensor, "arg2")

                    # Verify debug log was called
                    mock_logger.debug.assert_called_once()

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_inplace_offset_lazy_capture_temporarily_enables_capture(
            self, mock_weak_ref_tensors, mock_compilation_counter, mock_envs,
            mock_current_platform, mock_get_forward_context, mock_torch):
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        descriptor = BatchDescriptor(num_tokens=32,
                                     num_reqs=32,
                                     uniform=True,
                                     has_lora=False,
                                     start_num_tokens=384,
                                     graph_variant="inplace_serial",
                                     attention_backend="fia")
        self.mock_forward_context.batch_descriptor = descriptor
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        self.mock_forward_context.allow_inplace_lazy_capture = True
        self.mock_forward_context.split_inplace_mode = "inplace_serial"
        mock_get_forward_context.return_value = self.mock_forward_context

        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)
        mock_torch.Tensor = torch.Tensor
        mock_weak_ref_tensors.return_value = "weak_ref_output"
        mock_compilation_counter.num_cudagraph_captured = 0

        from vllm.compilation import monitor as compilation_monitor
        previous_enabled = compilation_monitor.cudagraph_capturing_enabled
        compilation_monitor.set_cudagraph_capturing_enabled(False)
        try:
            wrapper = ACLGraphWrapper(
                runnable=self.mock_runnable,
                vllm_config=self.mock_vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                cudagraph_options=self.mock_cudagraph_options)

            result = wrapper(torch.tensor([1, 2, 3]))
        finally:
            restored_enabled = compilation_monitor.cudagraph_capturing_enabled
            compilation_monitor.set_cudagraph_capturing_enabled(
                previous_enabled)

        self.assertEqual(result, "test_output")
        self.assertFalse(restored_enabled)
        self.assertIn(descriptor, wrapper.concrete_aclgraph_entries)
        mock_torch.npu.NPUGraph.assert_called_once()

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_inplace_offset_capture_requires_context_flag(
            self, mock_envs, mock_current_platform, mock_get_forward_context,
            mock_torch):
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        self.mock_forward_context.batch_descriptor = BatchDescriptor(
            num_tokens=32,
            num_reqs=32,
            uniform=True,
            has_lora=False,
            start_num_tokens=384,
            graph_variant="inplace_serial",
            attention_backend="fia")
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        self.mock_forward_context.allow_inplace_lazy_capture = False
        self.mock_forward_context.split_inplace_mode = "inplace_serial"
        mock_get_forward_context.return_value = self.mock_forward_context
        mock_torch.Tensor = torch.Tensor

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        with self.assertRaisesRegex(RuntimeError,
                                    "Refusing to capture an inplace offset"):
            wrapper(torch.tensor([1, 2, 3]))

        mock_torch.npu.NPUGraph.assert_not_called()

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch('vllm_ascend.compilation.acl_graph.validate_cudagraph_capturing_enabled')
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    @patch('vllm_ascend.compilation.acl_graph.compilation_counter')
    @patch('vllm_ascend.compilation.acl_graph.weak_ref_tensors')
    def test_inplace_input_ptr_validation_without_debug_mode(
            self, mock_weak_ref_tensors, mock_compilation_counter, mock_envs,
            mock_current_platform, mock_get_forward_context,
            mock_validate_cudagraph_capturing_enabled, mock_torch):
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_get_forward_context.return_value = self.mock_forward_context
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        self.mock_forward_context.validate_inplace_input_ptrs = True

        mock_npu_graph = MagicMock()
        mock_torch.npu.NPUGraph.return_value = mock_npu_graph
        mock_graph_context = MagicMock()
        mock_torch.npu.graph.return_value = mock_graph_context
        mock_graph_context.__enter__ = Mock(return_value=None)
        mock_graph_context.__exit__ = Mock(return_value=None)
        mock_torch.Tensor = torch.Tensor
        mock_weak_ref_tensors.return_value = "weak_ref_output"
        mock_compilation_counter.num_cudagraph_captured = 0

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        tensor1 = torch.arange(1024)
        tensor2 = torch.arange(2048)
        wrapper(tensor1)
        with self.assertRaisesRegex(AssertionError, "Input addresses"):
            wrapper(tensor2)

    def test_collect_attn_metadata_tensor_infos_recurses_common_shapes(self):
        metadata = {
            "layer": SimpleNamespace(
                common_attn_metadata=SimpleNamespace(
                    query_start_loc=torch.arange(3, dtype=torch.int32),
                    seq_lens=torch.arange(2, dtype=torch.int32),
                ),
                decode=SimpleNamespace(
                    block_table=torch.arange(4, dtype=torch.int32).reshape(
                        2, 2),
                ),
            )
        }

        addresses, infos = _collect_attn_metadata_tensor_infos(metadata)

        self.assertEqual(len(addresses), 3)
        paths = {info["path"] for info in infos}
        self.assertIn(
            "attn_metadata.layer.common_attn_metadata.query_start_loc",
            paths)
        self.assertIn("attn_metadata.layer.common_attn_metadata.seq_lens",
                      paths)
        self.assertIn("attn_metadata.layer.decode.block_table", paths)

    @patch('vllm_ascend.compilation.acl_graph.torch')
    @patch('vllm_ascend.compilation.acl_graph.get_forward_context')
    @patch('vllm_ascend.compilation.acl_graph.current_platform')
    @patch('vllm_ascend.compilation.acl_graph.envs')
    def test_inplace_metadata_ptr_validation_mismatch(
            self, mock_envs, mock_current_platform, mock_get_forward_context,
            mock_torch):
        mock_envs.VLLM_LOGGING_LEVEL = "INFO"
        mock_current_platform.get_global_graph_pool.return_value = self.mock_graph_pool
        mock_torch.Tensor = torch.Tensor
        descriptor = self.mock_batch_descriptor
        self.mock_forward_context.batch_descriptor = descriptor
        self.mock_forward_context.cudagraph_runtime_mode = CUDAGraphMode.FULL
        self.mock_forward_context.validate_inplace_metadata_ptrs = True
        self.mock_forward_context.attn_metadata = {
            "layer": Mock(seq_lens=torch.arange(2, dtype=torch.int32))
        }
        mock_get_forward_context.return_value = self.mock_forward_context

        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)
        entry = ACLGraphEntry(batch_descriptor=descriptor,
                              aclgraph=MagicMock(),
                              output="cached",
                              attn_metadata_addresses=[123],
                              attn_metadata_tensor_infos=[{
                                  "path": "old"
                              }])
        wrapper.concrete_aclgraph_entries[descriptor] = entry

        with self.assertRaisesRegex(AssertionError,
                                    "Attention metadata addresses"):
            wrapper()

    def test_getattr_access_runnable_attributes(self):
        """Test __getattr__ method accesses runnable attributes"""
        mock_runnable = MagicMock()
        mock_runnable.test_attr = "test_value"

        wrapper = ACLGraphWrapper(
            runnable=mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Should be able to access attributes of the runnable
        self.assertEqual(wrapper.test_attr, "test_value")

    def test_getattr_attribute_not_exists(self):
        """Test __getattr__ method raises AttributeError for non-existent attributes"""

        # Create a simple object without any attributes
        class EmptyRunnable:
            pass

        mock_runnable = EmptyRunnable()

        wrapper = ACLGraphWrapper(
            runnable=mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        # Should raise AttributeError for non-existent attributes
        with self.assertRaises(AttributeError) as context:
            _ = wrapper.non_existent_attr

        self.assertIn("Attribute non_existent_attr not exists",
                      str(context.exception))

    def test_unwrap_method(self):
        """Test unwrap method returns the original runnable"""
        wrapper = ACLGraphWrapper(
            runnable=self.mock_runnable,
            vllm_config=self.mock_vllm_config,
            runtime_mode=CUDAGraphMode.FULL,
            cudagraph_options=self.mock_cudagraph_options)

        unwrapped = wrapper.unwrap()
        self.assertEqual(unwrapped, self.mock_runnable)


class TestMTPGraphParams(TestBase):

    def test_set_mtp_graph_params(self):
        with patch('vllm_ascend.compilation.acl_graph._mtp_graph_params',
                   new=None):
            set_mtp_graph_params([4])
            from vllm_ascend.compilation.acl_graph import _mtp_graph_params
            self.assertIsNotNone(_mtp_graph_params)

    @patch('vllm_ascend.compilation.acl_graph._mtp_graph_params')
    def test_update_mtp_graph_params_workspaces(self, mtp_graph_params_mock):
        mtp_graph_params_mock.workspaces = {4: 5}
        update_mtp_graph_params_workspaces(4, 6)
        self.assertEqual(mtp_graph_params_mock.workspaces[4], 6)

    @patch('vllm_ascend.compilation.acl_graph._mtp_graph_params')
    def test_get_mtp_graph_params(self, mtp_graph_params_mock):
        graph_params = get_mtp_graph_params()
        self.assertIs(mtp_graph_params_mock, graph_params)


class TestPCPDCPGraphParams(TestBase):

    def setUp(self):
        self.update_stream = MagicMock(name="FakeStream")
        graph_params = get_graph_params()
        if graph_params is None:
            set_graph_params(set([4]))
            self.graph_params = get_graph_params()
        else:
            self.graph_params = graph_params
        mock_event = torch.npu.ExternalEvent()
        mock_event.record = MagicMock()
        self.graph_params.events[4] = []
        self.graph_params.handles[4] = []
        self.graph_params.events[4].append(mock_event)
        self.graph_params.handles[4].append(MagicMock())

    @patch('torch.npu.graph_task_update_end', )
    @patch('torch.npu.graph_task_update_begin', MagicMock())
    @patch('torch_npu.atb.npu_multi_head_latent_attention', MagicMock())
    def test_update_mla_dcp_pcp_params(self, _mock_graph_task_end):
        input_positions = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        block_table = torch.zeros(2, 5, dtype=torch.long)
        seq_lens = torch.tensor([4, 4])
        cp_seq_len = torch.tensor([2, 2])
        max_seq_lens = 4
        seq_lens_list = [4, 4]
        slot_mapping = torch.zeros(8, dtype=torch.long)
        query_start_loc = torch.tensor([0, 4])
        block_tables = torch.zeros(2, 5, dtype=torch.long)

        decode = AscendMLADecodeMetadata(input_positions,
                                         block_table,
                                         seq_lens,
                                         max_seq_lens,
                                         seq_lens_list,
                                         cp_seq_len=cp_seq_len)
        metadata = AscendMLAMetadata(8,
                                     8,
                                     slot_mapping,
                                     query_start_loc,
                                     seq_lens,
                                     block_tables,
                                     4,
                                     4,
                                     0,
                                     decode=decode)
        forward_context = MagicMock()
        forward_context.attn_metadata = {"attn_layer_0": metadata}
        forward_context.is_mtp_model = False

        num_heads = 256
        scale = 0.1
        num_kv_heads = 8
        qk_head_dim = 96
        qk_rope_head_dim = 32
        qk_nope_head_dim = 64
        query = torch.randn(4, num_heads, qk_head_dim)
        q_pe = query[..., qk_nope_head_dim:]
        q_nope = query[..., :qk_nope_head_dim]
        k_nope = torch.randn(4, num_heads, qk_nope_head_dim)
        k_pe = torch.randn(4, num_heads, qk_rope_head_dim)
        out = torch.randn(2, 16, 128)
        lse = torch.randn(2, 16, 8)
        self.graph_params.attn_params[4] = []
        self.graph_params.attn_params[4].append(
            (q_nope, q_pe, k_nope, k_pe, block_table, seq_lens, num_heads,
             scale, num_kv_heads, out, lse))

        with patch("torch_npu._C._npu_setStream", return_value=None):
            update_mla_attn_dcp_pcp_params(self.update_stream, forward_context,
                                           4)

        _mock_graph_task_end.assert_called_once()

    @patch('torch.npu.graph_task_update_end', )
    @patch('torch.npu.graph_task_update_begin', MagicMock())
    @patch('torch_npu.npu_fused_infer_attention_score.out', MagicMock())
    def test_update_attn_dcp_pcp_params(self, _mock_graph_task_end):
        block_table = torch.zeros(2, 5, dtype=torch.long)
        num_heads = 256
        scale = 0.1
        num_kv_heads = 8
        qk_head_dim = 96
        qk_nope_head_dim = 64
        query = torch.randn(4, num_heads, qk_head_dim)
        q_nope = query[..., :qk_nope_head_dim]
        k_nope = torch.randn(4, num_heads, qk_nope_head_dim)
        actual_seq_lengths_kv = [1, 1]
        actual_seq_lengths_q = np.array([1, 1])
        out = torch.randn(2, 16, 128)
        lse = torch.randn(2, 16, 8)

        num_computed_tokens_of_pcp_dcp = np.array([[[1, 1], [1, 1]],
                                                   [[1, 1], [1, 1]]])
        decode = AscendMetadataForDecode(num_computed_tokens_of_pcp_dcp)
        metadata = AscendMetadata(num_actual_tokens_pcp_padded=[1, 1],
                                  actual_seq_lengths_q=actual_seq_lengths_q,
                                  num_decode_tokens=1,
                                  decode_meta=decode)
        forward_context = MagicMock()
        forward_context.attn_metadata = {"attn_layer_0": metadata}
        forward_context.is_mtp_model = False

        self.graph_params.attn_params[4] = []
        self.graph_params.attn_params[4].append(
            (q_nope, k_nope, k_nope, num_heads, num_kv_heads, scale,
             block_table, 128, actual_seq_lengths_kv, actual_seq_lengths_q,
             out, lse, 2, 0, 0))

        with patch("torch_npu._C._npu_setStream", return_value=None):
            update_attn_dcp_pcp_params(self.update_stream, forward_context, 4)

        _mock_graph_task_end.assert_called_once()
