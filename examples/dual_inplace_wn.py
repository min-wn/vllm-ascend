# /// script
# dependencies = []
# ///
"""Compare split disabled vs enabled for wn-vllm original code."""
import os, sys, json

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODE = sys.argv[1]
OUT = sys.argv[2]

sys.path.insert(0, "/vllm-workspace/vllm")
sys.path.insert(0, "/vllm-workspace")

from vllm import LLM, SamplingParams

def main():
    topics = [
        "artificial intelligence", "quantum computing", "climate change",
        "space exploration", "ancient history", "modern art",
        "human psychology", "economics", "philosophy",
    ]
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        additional_config={
            "split_batch_config": {
                "enabled": MODE == "enabled",
                "mode": "inplace_serial",
                "num_splits": 2,
                "enable_parallel_streams": False,
                "min_batch_size_for_split": 4,
                "enable_inplace_lazy_capture": True,
            }
            | ({"force_split": True} if MODE == "enabled" else {}),
        },
        gpu_memory_utilization=0.6,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Explain {t}"}],
            tokenize=False, add_generation_prompt=True,
        )
        for t in topics
    ]
    outs = llm.generate(prompts, SamplingParams(max_tokens=50, temperature=0.0))
    results = [{"idx": i, "token_ids": list(o.outputs[0].token_ids), "text": o.outputs[0].text}
               for i, o in enumerate(outs)]
    with open(OUT, "w") as f:
        json.dump(results, f)
    print(f"DONE {MODE}")

if __name__ == "__main__":
    main()
