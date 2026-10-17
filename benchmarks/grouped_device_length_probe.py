"""Isolated BF16 Qwen3-4B operator feasibility experiment."""

import hashlib
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from grouped_device_length_kernel import launch

# Same captured graph must consume changed device lengths and pages.
torch.npu.set_device(0)
torch.manual_seed(42)
q = torch.randn((2, 32, 128), dtype=torch.bfloat16, device="npu")
k = torch.randn((8, 128, 8, 128), dtype=q.dtype, device="npu")
v = torch.randn_like(k)
lengths = torch.tensor([512, 512], dtype=torch.int32, device="npu")
t = torch.arange(8, dtype=torch.int32, device="npu").view(2, 4)
o = torch.empty_like(q)
launch(q, k, v, lengths, t, o)
torch.npu.synchronize()
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    launch(q, k, v, lengths, t, o)
torch.npu.synchronize()
qc, kc, vc = q.cpu().float(), k.cpu().float(), v.cpu().float()
checks = []
for actual, pages in [
    ([512, 512], [[0, 1, 2, 3], [4, 5, 6, 7]]),
    ([129, 17], [[4, 5, -1, -1], [0, -1, -1, -1]]),
    ([0, 0], [[-1] * 4, [-1] * 4]),
    ([511, 512], [[3, 2, 1, 0], [7, 6, 5, 4]]),
]:
    lengths.copy_(torch.tensor(actual, dtype=torch.int32))
    t.copy_(torch.tensor(pages, dtype=torch.int32))
    o.fill_(float("nan"))
    torch.npu.synchronize()
    g.replay()
    torch.npu.synchronize()
    reference = torch.zeros_like(qc)
    for row, length in enumerate(actual):
        if not length:
            continue
        keys = kc[[p for p in pages[row] if p >= 0]].reshape(-1, 8, 128)[:length].repeat_interleave(4, dim=1)
        values = vc[[p for p in pages[row] if p >= 0]].reshape(-1, 8, 128)[:length].repeat_interleave(4, dim=1)
        scores = torch.einsum("hd,thd->ht", qc[row], keys) * 128**-0.5
        reference[row] = torch.einsum("ht,thd->hd", scores.softmax(-1), values)
    observed = o.cpu().float()
    torch.testing.assert_close(observed, reference, rtol=0.02, atol=0.01)
    checks.append(dict(lengths=actual, max_abs_error=(observed - reference).abs().max().item()))

# Reuse the established FIA update/event benchmark unchanged except descriptors,
# smaller measurement budget, and one additional grouped operation.
source = Path("benchmarks/device_length_decode.py").read_text()
source = source[: source.index("def main():")]
source = source.replace("14, 2, 64, 128", "32, 8, 128, 128").replace(
    'dtype=torch.float16, device="npu"', 'dtype=torch.bfloat16, device="npu"'
)
source = source.replace(
    "    prototype_graph, _, _ = capture(prototype)",
    "    launch(query, key, value, lengths, table, output)\n"
    "    torch.npu.synchronize()\n"
    "    torch.testing.assert_close(output, fia_output, rtol=0.02, atol=0.01)\n"
    "    grouped_graph, _, _ = capture(lambda: launch(query, key, value, lengths, table, output))\n"
    "    prototype_graph, _, _ = capture(prototype)",
)
source = source.replace(
    '"prototype_replay_us": prototype_graph.replay,',
    '"grouped_replay_us": grouped_graph.replay,\n        "prototype_replay_us": prototype_graph.replay,',
)
source = source.replace("range(5):", "range(3):").replace("rounds=5", "rounds=3")
namespace = {"launch": launch}
exec(compile(source, "benchmarks/device_length_decode.py", "exec"), namespace)
result = dict(
    source_sha256={
        name: hashlib.sha256(Path("benchmarks", name).read_bytes()).hexdigest()
        for name in ("grouped_device_length_probe.py", "grouped_device_length_kernel.py", "device_length_decode.py")
    },
    checks=checks,
    measurement=namespace["benchmark"](104, 512, 20),
    limitations="Operator feasibility only, all 104 rows active. Not a split/padding comparison or end-to-end gain.",
)
path = Path("/workspace/multi-graph-results/grouped-device-length/pilot.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
