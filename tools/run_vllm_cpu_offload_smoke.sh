#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright contributors to the kvcached project
# SPDX-License-Identifier: Apache-2.0

# End-to-end smoke test for kvcached with vLLM's OffloadingConnector.
# The test fills the GPU prefix cache, replays the first prompt, and requires
# both GPU-to-CPU stores and a CPU-to-GPU load in vLLM's metrics.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8100}"
CPU_BYTES_TO_USE="${CPU_BYTES_TO_USE:-4294967296}"
OFFLOAD_BLOCK_SIZE="${OFFLOAD_BLOCK_SIZE:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024}"
EVICTION_REQUESTS="${EVICTION_REQUESTS:-32}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
INSTALL_EDITABLE="${INSTALL_EDITABLE:-1}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
BASELINE_URL="${BASELINE_URL:-}"
LOG_DIR="${LOG_DIR:-$(mktemp -d /tmp/kvcached-vllm-offload.XXXXXX)}"
SERVER_LOG="${LOG_DIR}/server.log"
RESULT_JSON="${LOG_DIR}/result.json"
SERVER_PID=""
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
archive="${LOG_DIR%/}.tar.gz"
mkdir -p "${LOG_DIR}"

log() {
  printf '[cpu-offload-smoke] %s\n' "$*" >&2
}

cleanup() {
  set +e
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill -TERM -- "-${SERVER_PID}" >/dev/null 2>&1 \
      || kill -TERM "${SERVER_PID}" >/dev/null 2>&1 \
      || true
    sleep 3
    kill -KILL -- "-${SERVER_PID}" >/dev/null 2>&1 \
      || kill -KILL "${SERVER_PID}" >/dev/null 2>&1 \
      || true
  fi
}

finalize() {
  local exit_code=$?
  trap - EXIT
  cleanup
  set +e
  nvidia-smi >"${LOG_DIR}/nvidia-smi-final.txt" 2>&1
  {
    printf 'status=%s\n' "$([[ "${exit_code}" -eq 0 ]] && printf passed || printf failed)"
    printf 'exit_code=%s\n' "${exit_code}"
    printf 'started_at=%s\n' "${started_at}"
    printf 'finished_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'model=%s\n' "${MODEL}"
    printf 'baseline_url=%s\n' "${BASELINE_URL}"
  } >"${LOG_DIR}/run-status.txt"
  (
    cd "${LOG_DIR}" || exit
    find . -type f ! -name MANIFEST.sha256 -print0 \
      | sort -z \
      | xargs -0 sha256sum >MANIFEST.sha256
  )
  tar -czf "${archive}" -C "$(dirname "${LOG_DIR}")" "$(basename "${LOG_DIR}")"
  sha256sum "${archive}" >"${archive}.sha256"
  log "Artifacts: ${LOG_DIR}"
  log "Archive: ${archive}"
  exit "${exit_code}"
}
trap finalize EXIT

die() {
  printf '[cpu-offload-smoke][FAIL] %s\n' "$*" >&2
  if [[ -f "${SERVER_LOG}" ]]; then
    printf '\n--- server log tail ---\n' >&2
    tail -240 "${SERVER_LOG}" >&2 || true
  fi
  exit 1
}

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v "${VLLM_BIN}" >/dev/null 2>&1 || die "${VLLM_BIN} is not installed"
command -v nvidia-smi >/dev/null 2>&1 || die "an NVIDIA GPU is required"
nvidia-smi -L >/dev/null 2>&1 || die "nvidia-smi cannot access a GPU"

nvidia-smi >"${LOG_DIR}/nvidia-smi-initial.txt" 2>&1
{
  printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
  printf 'git_branch=%s\n' "$(git branch --show-current)"
  printf 'python=%s\n' "$(python --version 2>&1)"
  printf 'vllm=%s\n' "$("${VLLM_BIN}" --version 2>&1 || true)"
  uname -a
  df -h "${LOG_DIR}"
} >"${LOG_DIR}/environment.txt"
if [[ "${INSTALL_EDITABLE}" == "1" ]]; then
  log "Installing kvcached from ${ROOT_DIR}"
  python -m pip install -e .
fi

KV_TRANSFER_CONFIG="$(python - "${CPU_BYTES_TO_USE}" "${OFFLOAD_BLOCK_SIZE}" <<'PY'
import json
import sys

print(json.dumps({
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "cpu_bytes_to_use": int(sys.argv[1]),
        "block_size": int(sys.argv[2]),
    },
}))
PY
)"

server_cmd=(
  "${VLLM_BIN}" serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --enable-prefix-caching
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --kv-transfer-config "${KV_TRANSFER_CONFIG}"
)
if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  # Intentional word splitting lets callers pass additional CLI flags.
  # shellcheck disable=SC2206
  extra_args=(${VLLM_EXTRA_ARGS})
  server_cmd+=("${extra_args[@]}")
fi

log "Starting vLLM; logs and results will be kept in ${LOG_DIR}"
if command -v setsid >/dev/null 2>&1; then
  ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 \
    setsid "${server_cmd[@]}" >"${SERVER_LOG}" 2>&1 &
else
  ENABLE_KVCACHED=true KVCACHED_AUTOPATCH=1 \
    "${server_cmd[@]}" >"${SERVER_LOG}" 2>&1 &
fi
SERVER_PID=$!

deadline=$((SECONDS + REQUEST_TIMEOUT))
until curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; do
  kill -0 "${SERVER_PID}" >/dev/null 2>&1 \
    || die "vLLM exited before becoming ready"
  [[ "${SECONDS}" -le "${deadline}" ]] \
    || die "timed out waiting for vLLM"
  sleep 2
done

grep -q "Successfully patched vllm:.*kv_connector_mixin" "${SERVER_LOG}" \
  || die "vLLM became ready without the kvcached CPU-offload compatibility patch"

python - "${HOST}" "${PORT}" "${MODEL}" "${EVICTION_REQUESTS}" \
  "${RESULT_JSON}" "${LOG_DIR}" "${BASELINE_URL}" <<'PY'
import json
import sys
import urllib.request

host, port, model, eviction_requests, result_path, log_dir, baseline_url = sys.argv[1:]
base_url = f"http://{host}:{port}"


def completion(url: str, prompt: str) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": 8,
        "temperature": 0,
        "seed": 17,
    }).encode()
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.load(response)
    return body["choices"][0]["text"]


def metrics() -> str:
    with urllib.request.urlopen(f"{base_url}/metrics", timeout=30) as response:
        return response.read().decode()


def counter(metrics_text: str, metric_name: str) -> float:
    total = 0.0
    for line in metrics_text.splitlines():
        if line.startswith("#") or not line:
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if name in (metric_name, f"{metric_name}_total"):
            total += float(line.rsplit(None, 1)[1])
    return total


shared_context = " ".join(
    ["Paris is the capital city of France and a major European city."] * 40
)
prompt = (
    f"{shared_context}\nQuestion: What is the capital of France? "
    "Answer with only the city name.\nAnswer:"
)
baseline_output = completion(baseline_url, prompt) if baseline_url else None
first_output = completion(base_url, prompt)

for index in range(int(eviction_requests)):
    filler = " ".join(
        [f"Unique cache pressure request {index} contains token group {index}."] * 45
    )
    completion(base_url, f"{filler}\nReturn the number {index}.\nAnswer:")

before_replay = metrics()
with open(f"{log_dir}/metrics-before-replay.prom", "w", encoding="utf-8") as output:
    output.write(before_replay)
replay_output = completion(base_url, prompt)
after_replay = metrics()
with open(f"{log_dir}/metrics-after-replay.prom", "w", encoding="utf-8") as output:
    output.write(after_replay)

store_bytes = counter(after_replay, "vllm:kv_offload_store_bytes")
load_before = counter(before_replay, "vllm:kv_offload_load_bytes")
load_after = counter(after_replay, "vllm:kv_offload_load_bytes")
result = {
    "first_output": first_output,
    "baseline_output": baseline_output,
    "baseline_matches": baseline_output is None or baseline_output == first_output,
    "replay_output": replay_output,
    "outputs_equal": first_output == replay_output,
    "store_bytes": store_bytes,
    "load_bytes_before_replay": load_before,
    "load_bytes_after_replay": load_after,
    "replay_loaded_bytes": load_after - load_before,
}
with open(result_path, "w", encoding="utf-8") as output:
    json.dump(result, output, indent=2, ensure_ascii=True)
print(json.dumps(result, indent=2))

if first_output != replay_output:
    raise SystemExit("replayed output differs after CPU restore")
if baseline_output is not None and baseline_output != first_output:
    raise SystemExit("CPU-offload output differs from the no-kvcached baseline")
if store_bytes <= 0:
    raise SystemExit("no GPU-to-CPU store was observed")
if load_after <= load_before:
    raise SystemExit(
        "no CPU-to-GPU replay load was observed; increase EVICTION_REQUESTS "
        "or lower GPU_MEMORY_UTILIZATION"
    )
PY

log "PASS: CPU store, CPU restore, and output equality were verified"
log "Result: ${RESULT_JSON}"
