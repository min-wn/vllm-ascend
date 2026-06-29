import os

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_ASCEND_SPLIT_INPLACE_DEBUG"] = "1"
os.environ["VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE"] = "/tmp/vllm_ascend_inplace_split.jsonl"

from vllm import LLM, SamplingParams


def main():
    topics = [
        "artificial intelligence",
        "quantum computing",
        "climate change",
        "space exploration",
        "ancient history",
        "modern art",
        "human psychology",
        "economics",
        "philosophy"
    ]
    raw_prompts = [f"Explain {topic}" for topic in topics]
    sampling_params = SamplingParams(max_tokens=50, temperature=0.0)
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        additional_config={
            "split_batch_config": {
                "enabled": True,
                "mode": "inplace_serial",
                "cudagraph_mode": "FULL",
                "num_splits": 2,
                "enable_parallel_streams": False,
                "enable_inplace_lazy_capture": True,
                "inplace_split_planner_policy": "largest_lower",
                "inplace_offset_match_policy": "exact",
            }
        },
        gpu_memory_utilization=0.6,
    )

    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True
        )
        for text in raw_prompts
    ]
    
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print(f">>> Diag log: {os.environ['VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE']} <<<")

if __name__ == "__main__":
    main()