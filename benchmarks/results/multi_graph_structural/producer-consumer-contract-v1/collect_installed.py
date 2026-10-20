# SPDX-License-Identifier: Apache-2.0
"""Read public ELF exports over SSH; never load a library or initialize NPU."""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

REMOTE = r"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

symbols = (
    "HcclCommGetHandleWithName", "HcclGetHcclBuffer", "HcclCreateOpResCtx",
    "HcclCommSymWinRegister", "HcclSymWinGetPeerPointer",
    "HcclThreadAcquireWithStream", "HcclEngineCtxCreate",
)
base = Path("/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64")
rows = []
for name in ("libhccl.so", "libhcomm.so", "libhccl_compat.so"):
    path = (base / name).resolve(strict=True)
    exports = subprocess.check_output(["nm", "-D", "--defined-only", str(path)], text=True)
    definitions = {}
    for line in exports.splitlines():
        fields = line.split()
        if len(fields) >= 3:
            symbol = fields[-1].split("@")[0]
            if symbol in symbols:
                definitions[symbol] = line.strip()
    dynamic = subprocess.check_output(["readelf", "-d", str(path)], text=True)
    rows.append(dict(
        path=str(path), bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        needed=re.findall(r"Shared library: \[(.*?)\]", dynamic),
        matching_definitions=definitions,
    ))
print(json.dumps(dict(
    scope="read-only public ELF files; no dlopen, symbol invocation or NPU initialization",
    queried_symbols=list(symbols), libraries=rows,
    found_in_these_libraries={s: [r["path"] for r in rows if s in r["matching_definitions"]]
                             for s in symbols},
    limitations=("Exports establish neither ABI/call compatibility nor driver, capture or device-memory "
                 "qualification; not an exhaustive library search."),
), indent=2))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="vllm-ascend-030")
    args = parser.parse_args()
    command = "python3 -c " + shlex.quote(REMOTE)
    raw = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", args.host, command], timeout=60
    )
    document = json.loads(raw)
    document["ssh_host"] = args.host
    target = Path(__file__).with_name("installed-exports.json")
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document["found_in_these_libraries"], indent=2))


if __name__ == "__main__":
    main()
