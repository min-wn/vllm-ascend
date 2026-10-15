import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def load_method(path, name, namespace):
    source = ast.parse(path.read_text(encoding="utf-8"))
    method = next(
        method
        for node in source.body
        if isinstance(node, ast.ClassDef)
        for method in node.body
        if isinstance(method, ast.FunctionDef) and method.name == name
    )
    ast.fix_missing_locations(method)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("name", ["update_config", "reload_weights"])
@pytest.mark.parametrize("enabled", [False, True])
def test_worker_guards_preserve_disabled_behavior(name, enabled):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/worker.py"
    namespace = {
        "Any": object,
        "get_ascend_config": lambda: SimpleNamespace(multi_graph_config=SimpleNamespace(enabled=enabled)),
    }
    method = load_method(path, name, namespace)
    runner = SimpleNamespace(update_config=Mock(), reload_weights=Mock())
    worker = SimpleNamespace(model_runner=runner)
    if enabled:
        with pytest.raises(ValueError, match="multi-graph"):
            method(worker, {})
        assert not runner.update_config.called and not runner.reload_weights.called
    else:
        method(worker, {})
        getattr(runner, name).assert_called_once_with({})
