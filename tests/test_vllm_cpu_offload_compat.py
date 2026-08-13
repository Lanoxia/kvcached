# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import copy
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).parents[1]
PATCHES = ROOT / "kvcached/integration/vllm/patches.py"
AUTOPATCH = ROOT / "kvcached/integration/vllm/autopatch.py"


def load_uniform_cache_wrapper(
    original_method: Callable[..., bool],
    *,
    enabled: bool,
) -> Callable[..., bool]:
    tree = ast.parse(PATCHES.read_text(encoding="utf-8"))
    wrapper = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patched_use_uniform_kv_cache"
    )
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(wrapper)], type_ignores=[]))
    namespace: dict[str, Any] = {
        "Any": Any,
        "enable_kvcached": lambda: enabled,
        "original_method": original_method,
    }
    exec(compile(module, str(PATCHES), "exec"), namespace)
    return namespace[wrapper.name]


def test_kvcached_forces_per_layer_vmm_allocation_for_both_upstream_signatures():
    one_arg = load_uniform_cache_wrapper(lambda groups: True, enabled=True)
    two_args = load_uniform_cache_wrapper(
        lambda groups, cache_dtype: True,
        enabled=True,
    )

    assert one_arg(["attention-group"]) is False
    assert two_args(["attention-group"], "auto") is False


def test_disabled_patch_preserves_old_and_new_upstream_calls():
    one_arg = load_uniform_cache_wrapper(
        lambda groups: groups == ["attention-group"],
        enabled=False,
    )
    two_args = load_uniform_cache_wrapper(
        lambda groups, cache_dtype: (groups, cache_dtype) == (["attention-group"], "auto"),
        enabled=False,
    )

    assert one_arg(["attention-group"]) is True
    assert two_args(["attention-group"], cache_dtype="auto") is True


def test_cpu_offload_patch_is_registered_for_vllm_012_and_newer():
    source = AUTOPATCH.read_text(encoding="utf-8")

    assert "KVConnectorMixinPatch" in source
    assert "(KVConnectorMixinPatch(), VLLM_V12_RANGE)" in source


def test_gpu_smoke_requires_real_offload_and_baseline_equality():
    script = (ROOT / "tools" / "run_vllm_cpu_offload_smoke.sh").read_text()

    assert "vllm:kv_offload_store_bytes" in script
    assert "vllm:kv_offload_load_bytes" in script
    assert "CPU-offload output differs from the no-kvcached baseline" in script
    assert "Successfully patched vllm:.*kv_connector_mixin" in script
    assert 'extra_config["num_cpu_blocks"]' in script
    assert 'extra_config["cpu_bytes_to_use"]' in script
    assert "inspect.getsource(CPUOffloadingSpec.__init__)" in script
    assert 'if "cpu_bytes_to_use" in spec_source' in script
    assert "metrics-before-replay.prom" in script
    assert "metrics-after-replay.prom" in script
    assert "MANIFEST.sha256" in script
