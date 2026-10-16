# Experimental exact multi-graph decode

This branch implements startup capture and serial/parallel replay for two independent whole-model graphs. CPU tensor and mocked-executor tests pass. Actual Ascend capture, model accuracy and device overlap have not been verified locally; hardware tests are required before treating the path as validated.

## Supported configuration

The adapter accepts V1, the standard Ascend device adaptor, LlamaForCausalLM/Qwen2ForCausalLM, FP16/BF16, one dense KV group and standard FIA attention. TP/PP/DP/PCP/DCP must all be one. Each request contributes one decode token. Capture sizes are exact; online misses use the existing forward path.

Set FULL_DECODE_ONLY, compilation mode NONE (0), and disable npugraph_ex. LoRA, speculative decoding, quantization, hybrid/MoE, sinks/sliding windows, active device metadata, ENPU, breakable/Xlite/super kernels, offload, KV transfer/sharing, sleep and hot weight reload are rejected. Explicit kv_cache_memory_bytes is rejected because graph resources must be reserved before automatic KV sizing. PA sizes must not overlap multi-graph capture sizes.

Example for a small correctness run:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    dtype="bfloat16",
    max_num_seqs=8,
    max_model_len=512,
    max_num_batched_tokens=512,
    compilation_config={
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [2, 4, 6, 8],
    },
    additional_config={
        "ascend_compilation_config": {"enable_npugraph_ex": False},
        "pa_shape_list": [],
        "multi_graph_config": {
            "enabled": True,
            "execution": "parallel",
            "selection_policy": "force",
            "capture_sizes_per_lane": [[2, 3, 4], [2, 3, 4]],
            "max_graph_entries": 6,
            "graph_memory_budget_bytes": 1073741824,
        },
    },
)
```

Set VLLM_USE_V2_MODEL_RUNNER=0. The 1 GiB budget is an explicit example reservation, not a measured requirement. Startup raises when capture resource consumption exceeds it. The configuration defaults to disabled.

Use execution=serial for the split correctness control. selection_policy=force is for correctness/benchmark runs. The default measured policy keeps the existing forward path without a matching cost record.

## Contiguous dense KV storage

Set `multi_graph_config.kv_cache_layout="contiguous"` to partition each unused dense allocation into contiguous K and V regions before the first cache write. This avoids FIA's implicit full-cache contiguous conversion for block-interleaved K/V. It allocates no extra KV storage and performs no per-step full-cache copy. The default `"default"` preserves the existing layout. Changing this option requires a new worker and fresh cache; existing cache contents cannot be reinterpreted.

The option is limited to the same single-rank Llama/Qwen2 dense FP16/BF16 configurations as the experimental adapter. Padded pages, hybrid models, KV transfer and offload are excluded. It may also be set with `enabled=false` for eager/single-graph controls; the runtime restrictions still apply. Capture and cost identities include the layout. Use `--kv-cache-layout contiguous` with the benchmark to apply the same storage contract to every mode. This workaround does not provide general compatibility for other operations unsupported by an old driver.

On 2026-10-15, Qwen2.5-0.5B-Instruct passed eager/single/serial/parallel comparisons for FP16 and BF16 with batches 6, 7 and 8 on Ascend 910B3, driver 24.1.rc3, CANN 9.1.0 and torch-npu 2.10.0.post4 using contiguous KV. All six comparisons produced identical tokens and selected-token log probabilities; split modes asserted actual split execution. Each mode ran in a fresh process with one warmup and two measured 16-token generations; correctness compared the last generation. This is bounded correctness evidence, not long-running stability or speedup evidence.

## Measured routing

Run the benchmark with `--decode-costs --repeats 5` to produce `serial.costs.json` and `parallel.costs.json`. A separate fresh process per mode synchronizes before and after each decode model forward, including the adapter's buffer refresh, graph updates and output merge. Scheduler preprocessing and logits/sampling outside the forward are excluded. The primary generation timing has no per-step synchronization. Profiling runs in another fresh process. Generation latency cannot be imported as decode cost.

Set `selection_policy="measured"`, `cost_table_path="/absolute/path/parallel.costs.json"`, `minimum_cost_samples=5` and `minimum_gain_fraction=0.05` using the same model, capture sizes, graph budget and software/device environment used for calibration. Capture identity excludes routing policy, execution mode and cost file path. Records still distinguish serial and parallel execution. Model configuration, capture configuration, device name/index, dtype, backend, KV layout, graph generation and torch/torch-npu/vLLM/plugin versions must match. An invalid, oversized or incompatible file fails startup before lane stream allocation.

Measurements use medians of paired samples with equal ordered KV lengths and confirmed split execution. Exact KV vectors deliberately avoid interpolating performance to unmeasured length buckets. Missing records or fewer than the required samples use the existing forward path; gains below the threshold also use that path. The adapter exposes `execution_count` and `fallback_counts` for planner decisions. No calibration data is fabricated or shipped with the feature.

The benchmark export requires equal decode step counts, matching KV vectors and identical output tokens between single and split runs. This checks reproducibility for cost import; use the separate model accuracy test for logprob acceptance. Calibration changes host timing through synchronization, so hardware profiling and repeatability remain required before interpreting gains.

## Ownership and failures

Each entry has its own input/metadata/output buffers, graph and pool; lanes have separate compute/update streams. Weights and physical KV allocations are shared. Duplicate write slots are checked using CPU block metadata, and KV allocation addresses are verified. Query boundaries are rebased; physical block IDs, slots and KV lengths retain their meanings.

All entry layouts and all FIA tasks are checked/resolved before replay. Replay precedes update, following the UpdatableGraph external-event protocol. The parent stream joins completion and merges hidden states in request order. Existing logits and sampling run once. A host fence protects buffers and output consumption across steps.

Any submission failure enters FAILED; no eager re-execution or waiting on potentially unsignalled external events is attempted. The worker must terminate/restart. Configuration changes, weight reload and weight transfer are rejected; KV allocation changes invalidate entries and require a new worker.

## Validation

Run in the supported Linux Ascend environment:

```bash
pytest -q tests/ut/worker/multi_graph
pytest -q tests/ut/test_ascend_config.py -k multi_graph
pytest -q tests/ut/compilation/test_acl_graph.py
pytest -q tests/e2e/pull_request/one_card/test_multi_graph_fia.py
pytest -q tests/e2e/pull_request/one_card/test_multi_graph_split.py
python benchmarks/multi_graph_split.py --batch-size 7 --dtype bfloat16 --trace --output-dir multi_graph_results
```

The FIA test captures two independent pools, changes query tensors and KV lengths over repeated replays, and compares with the same eager FIA operator. A subprocess timeout detects hangs. The model test covers even/odd batches and both dtypes across eager, single, serial split and parallel split; it asserts split execution occurred and compares tokens and selected-token log probabilities.

The benchmark uses fresh processes, records repeated warm generation latency plus versions/device identity, and optionally exports NPU traces in a separate profiling run. Generation latency includes prefill and sampling; it is not a decode-only kernel measurement. Inspect traces to establish actual stream overlap. No speedup is claimed from CPU tests.

## Padding performance comparison

Use the same card for sequential mode runs. For a 7-request batch, a single graph of capacity 8 pads one row; split sizes 4 and 3 contain no padding:

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python benchmarks/multi_graph_split.py \
  --model /workspace/models/Qwen2.5-0.5B-Instruct \
  --kv-cache-layout contiguous \
  --batch-size 7 --single-capacity 8 --lane-sizes 4 3 \
  --prompt-tokens 32 --prompt-pattern staggered --prompt-spread 32 \
  --output-tokens 64 --max-model-len 512 \
  --kv-blocks 256 --prefix-cache off \
  --warmups 3 --repeats 20 --decode-costs --trace \
  --output-dir padding_comparison
```

`--batch-size`, `--single-capacity` and `--lane-sizes` are independent. For example, 416 requests can compare a capacity-512 single graph with exact split sizes 384 and 32, subject to hardware resource limits and correctness validation. Lane sizes must sum to the actual batch; padded split entries remain unsupported. The single capture list contains only the requested capacity so the reference cannot silently select a smaller exact graph. All graph modes still capture that single graph for fallback; split modes additionally capture the two lane entries.

Synthetic prompts use valid tokenizer IDs with exact requested token lengths. Patterns are `uniform`, `staggered` and `bimodal`; `--prompt-spread` controls the difference in input lengths. Maximum prompt plus output length is checked. All modes use block size 128 and the same explicit `num_gpu_blocks_override`, either `--kv-blocks` or a derived capacity. An insufficient workload KV budget fails before execution. Split modes retain the explicit startup graph reservation; because the block override fixes actual KV capacity, that reservation does not give the baseline more KV blocks. JSON reports actual blocks, memory after warmup, peak allocated/reserved memory, startup time and graph reservation. Changing KV capacity can change workspace/algorithm selection, so retain these fields when comparing runs.

The default scheduler token budget covers the complete prefill batch and the single-graph capacity, preventing the benchmark from unintentionally becoming a mixed/partially admitted decode test. `--max-batched-tokens` may override that budget but must still cover both. With prefix caching off, the audit requires exactly `repeats * (output_tokens - 1)` decode steps; missing steps invalidate the measurement.

The default prefix cache policy is off; `--prefix-cache on` instead measures repeated warm-cache requests and must be reported as such. Default warmup count is 3 and measured generation count is 20. Each generation boundary is synchronized. JSON includes raw samples, p50/p90/p95/p99, standard deviation, coefficient of variation and output tokens/second. A p99 from a small sample is descriptive rather than a stable tail estimate. These statistics do not replace repeated runs under controlled device load and thermal conditions.

Use `--mode-order parallel serial single eager` to repeat the experiment in reversed order and check order/thermal effects. The order is saved in the summary. Per-mode metadata records the source commit, dirty-worktree flag, benchmark file hash, driver/CANN information and package versions.

Primary files are `eager.json`, `single.json`, `serial.json` and `parallel.json`. With `--decode-costs`, additional `*.decode.json` files contain synchronized forward samples and per-step audits. The audit observes actual single-graph replay capacity, actual lane sizes, KV lengths, padding fraction and fallback reasons; an unexpected route aborts the comparison. With `--trace`, `*.trace.json` records profiler-run metadata and `*.trace.trace.json` contains Chrome traces; those runs have one measured generation and are excluded from throughput ratios. Inspect NPU kernel intervals on the lane streams to establish overlap; multiple streams alone are not evidence of parallel execution.

The profiler uses one explicit active step and completes that step before stopping. Trace export validates that positive-duration Ascend Hardware execution events exist, rejecting a CPU-only trace; per-mode JSON records device event counts and stream IDs. Both list-root and `traceEvents`-wrapped Chrome formats are accepted. Device events include more than compute kernels, so their stream count is not a lane-overlap measurement.

`summary.json` checks tokens and selected-token log probabilities across all measured repetitions against the eager/single controls, checks common KV block capacity, and reports ratios only for primary generation measurements. Decode cost records pair identical ordered KV vectors and executed split shapes. The model-forward metric includes adapter overhead and synchronization effects; it is neither isolated kernel latency nor production inter-token latency. Use the serial split to distinguish split overhead from benefits of parallel execution.

## CPU verification

With Python 3.12, CPU PyTorch 2.10, pytest, pydantic and numpy installed:

```bash
python tools/run_multi_graph_cpu_tests.py --junitxml=multi_graph_cpu_tests.xml
```

The launcher isolates parent fixtures that require vLLM/NPU and stubs the package logger. Tensor operations and core modules run on real CPU PyTorch. Tests exercise capture orchestration with mocked device APIs, event ownership/order, cost persistence and rejection, worker guards and failure states. The CPU workflow uploads JUnit results. These tests do not establish NPU API compatibility, correctness or overlap.

## Trace attribution

The separate trace pass adds `mg.decode.*` ranges around eligible decode forwards and nested ranges for metadata preflight/refresh, resource draining, task resolution and graph updates. These annotations are installed only inside the benchmark child; normal generation/decode measurements do not install phase wrappers. Direct child runs also select the inline V1 engine so the observers execute in the model process.

Analyze an exported trace without vLLM or NPU dependencies:

```bash
python tools/analyze_multi_graph_trace.py parallel.trace.json --output parallel.analysis.json
```

After identifying the physical graph kernel streams from the trace, add `--lane-streams 37 38` to calculate their kernel interval intersection. Stream IDs vary between runs. The analyzer rejects missing kernel streams and excludes control/wait tasks from device busy intervals. Copy intervals are included in overall device activity, but excluded from lane kernel overlap. Per-step device windows extend between consecutive host decode starts, include sampling/transfers, and omit the final unbounded window. Nested host phase durations overlap; do not add parent and child totals. Profiler latency is diagnostic and must not replace normal generation results.

Captured metadata shared by several layer names is validated and refreshed once per lane. A change from one shared source to distinct per-layer sources is rejected before mutation because the captured aliases can no longer represent it safely.

## Extension components

V1 supports opt-in KV bucket routing with `kv_bucket_upper_bounds` and `minimum_bucket_vectors` (at least two). Exact observations take precedence. Ordered bucket vectors preserve lane imbalance; bucket estimates use the fastest measured baseline and slowest split. Unknown buckets fall back. Validate representativeness with hardware data before enabling interpolation.

Online calibration is opt-in with `online_calibration=true`, `selection_policy="measured"`, a `calibration_output_path` in an existing directory, `calibration_sample_interval` and `calibration_maximum_keys`. It alternates baseline/split on separate real steps and never repeats a step to compare KV writers. Matching exact KV vectors must recur to reach `minimum_cost_samples`. Complete records update routing and persist atomically. The dispatch metric is `synchronized_decode_dispatch_us`; synchronization and persistence increase calibration overhead. Device correctness and latency impact remain unverified.

Independent extension modules provide capacity planning/padded inputs/output trimming (`request_plan.py`), mixed request metadata (`mixed_metadata.py`), completion-query capture-set rotation (`async_executor.py`), V2 input normalization/exact manager bridge (`runner_v2.py`), and serial-lane rank agreement/Gloo transport (`distributed.py`). They are covered by CPU tests but are not integrated as full runtime support. V1 startup still requires exact decode, host fences and a single rank. Padding requires allocator-reserved, initialized scratch KV; V2 requires native capture/metadata integration; distributed model execution requires HCCL collective validation.

Run `python tools/multi_graph_gloo_probe.py --output-dir multi_graph_gloo` in Linux CPU CI to exercise two-rank agreement, common fallback and divergent-request rejection. The local Windows PyTorch build rejected Gloo device creation, so local transport validation is incomplete.
