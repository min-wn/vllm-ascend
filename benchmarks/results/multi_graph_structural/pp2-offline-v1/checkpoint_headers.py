# SPDX-License-Identifier: Apache-2.0
"""Print a layer-resolved checkpoint inventory; read headers, never tensor data.

Run through SSH stdin with the checkpoint directory as the only argument.
The schema hashes avoid repeating the same layer tensor descriptions 64 times.
"""

import hashlib
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def collect(model_dir):
    root = Path(model_dir).resolve()
    config_raw = (root / "config.json").read_bytes()
    index_raw = (root / "model.safetensors.index.json").read_bytes()
    index = json.loads(index_raw)
    layer_pattern = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
    layer_tensors = defaultdict(dict)
    others, files, names = {}, [], set()
    header_read = 0
    for filename in sorted(set(index["weight_map"].values())):
        path = (root / filename).resolve()
        if path.parent != root:
            raise ValueError("checkpoint shard must remain in the requested directory")
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ValueError("truncated header length")
            size = struct.unpack("<Q", prefix)[0]
            if size > path.stat().st_size - 8:
                raise ValueError("invalid header size")
            raw = source.read(size)
        header_read += 8 + size
        for name, item in json.loads(raw).items():
            if name == "__metadata__":
                continue
            if name in names or index["weight_map"].get(name) != filename:
                raise ValueError("duplicate name or index mismatch")
            names.add(name)
            start, stop = item["data_offsets"]
            if not 0 <= start <= stop <= path.stat().st_size - 8 - size:
                raise ValueError("invalid payload offsets")
            spec = dict(dtype=item["dtype"], shape=item["shape"], bytes=stop - start)
            match = layer_pattern.fullmatch(name)
            if match:
                layer_tensors[int(match[1])][match[2]] = spec
            else:
                others[name] = spec
        files.append(dict(name=filename, header_sha256=sha256(raw), header_bytes=8 + size))
    if names != set(index["weight_map"]):
        raise ValueError("tensor name set differs from index")
    schemas, layers = {}, []
    for number, tensors in sorted(layer_tensors.items()):
        canonical = json.dumps(tensors, sort_keys=True, separators=(",", ":")).encode("ascii")
        key = sha256(canonical)
        schemas[key] = tensors
        layers.append(dict(index=number, schema_sha256=key, bytes=sum(t["bytes"] for t in tensors.values())))
    groups = defaultdict(int)
    text_nonlayer = {}
    for name, spec in others.items():
        if name == "lm_head.weight":
            group = "lm_head"
        elif name == "model.language_model.embed_tokens.weight":
            group = "text_embedding"
        elif name.startswith("model.language_model."):
            group = "text_nonlayer"
        elif name.startswith("mtp."):
            group = "mtp"
        elif "visual." in name:
            group = "vision"
        else:
            group = "other"
        groups[group] += spec["bytes"]
        if group in ("lm_head", "text_embedding", "text_nonlayer"):
            text_nonlayer[name] = spec
    total = sum(layer["bytes"] for layer in layers) + sum(groups.values())
    if total != index["metadata"]["total_size"]:
        raise ValueError("payload total differs from index")
    return dict(
        status="header_inventory_not_loaded_device_memory",
        model_dir=str(root),
        config_sha256=sha256(config_raw),
        index_sha256=sha256(index_raw),
        tensor_count=len(names),
        total_payload_bytes=total,
        safetensors_bytes_read=header_read,
        tensor_payloads_read=False,
        files=files,
        layer_schemas=schemas,
        layers=layers,
        other_group_bytes=dict(groups),
        text_nonlayer_tensors=text_nonlayer,
    )


if __name__ == "__main__":
    print(json.dumps(collect(sys.argv[1]), indent=2, sort_keys=True))
