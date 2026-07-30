# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "run_vllm_nixl_pd_smoke.sh"


def run_preflight(**overrides: str) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(
        {
            "CHECK_ONLY": "1",
            "CLIENT_ENDPOINT": "chat",
            "KVCACHED_NIXL_CONTIGUOUS_LAYOUT": "false",
            "MAX_MODEL_LEN": "512",
            "BLOCK_SIZE": "128",
            "NUM_REQUESTS": "1",
            "MAX_TOKENS": "8",
            "MIN_REMOTE_BLOCKS": "1",
            "REQUEST_TIMEOUT": "30",
            "WATCHDOG_INTERVAL": "2",
            **overrides,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def test_check_only_passes_without_installing_or_starting_servers():
    completed = run_preflight()

    assert completed.returncode == 0
    assert "CHECK_ONLY=1; skipping dependency installation" in completed.stdout
    assert "[nixl-smoke] PASS" in completed.stdout


def test_preflight_rejects_unknown_client_endpoint():
    completed = run_preflight(CLIENT_ENDPOINT="unknown")

    assert completed.returncode != 0
    assert "CLIENT_ENDPOINT must be 'chat' or 'completions'" in completed.stdout


def test_preflight_rejects_contiguous_kvcached_layout():
    completed = run_preflight(KVCACHED_NIXL_CONTIGUOUS_LAYOUT="true")

    assert completed.returncode != 0
    assert "requires KVCACHED_NIXL_CONTIGUOUS_LAYOUT=false" in completed.stdout


def test_preflight_rejects_nonpositive_counts():
    for name in ("MAX_MODEL_LEN", "BLOCK_SIZE", "NUM_REQUESTS", "MAX_TOKENS"):
        completed = run_preflight(**{name: "0"})
        assert completed.returncode != 0
        assert f"{name} must be a positive integer" in completed.stdout
