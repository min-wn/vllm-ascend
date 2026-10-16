"""Optional local C++ FIA submission experiment, never imported by serving."""

import hashlib
from pathlib import Path


def load_native_update():
    import torch_npu
    from torch.utils.cpp_extension import load

    root = Path(torch_npu.__file__).parent
    source = Path(__file__).with_name("multi_graph_native_update.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    extension = load(
        name=f"multi_graph_native_update_{digest[:12]}",
        sources=[str(source)],
        extra_include_paths=[str(root / "include"), str(root / "include/third_party/acl/inc")],
        extra_cflags=["-O2"],
        extra_ldflags=[f"-L{root / 'lib'}", "-ltorch_npu", f"-Wl,-rpath,{root / 'lib'}"],
        verbose=True,
    )
    return extension, digest
