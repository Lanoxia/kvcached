#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# Run the pytest subset that is intentionally isolated from GPU, vLLM,
# SGLang, and the kvcached C++ extension. Keep this list in sync with the
# CPU Tests GitHub Actions workflow.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-python}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" -m pytest \
  tests/test_make_cache_key.py \
  tests/test_prefix_cache.py \
  tests/test_vllm_nixl_compat.py \
  "$@"
