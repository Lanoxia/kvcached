# kvcached H20 Validation Report

Date: 2026-07-23

Author: Zixuan Li

Fork: https://github.com/Lanoxia/kvcached

Latest commit validated: `9959758 Respect kvcached GPU utilization in KV allocation`

Rebased follow-up branches:

- `zixuan/fix-gpu-utilization-sizing`
- `zixuan/nixl-pd-smoke-preflight`
- `zixuan/gpu-ci-runner`

## Summary

I set up a reproducible GPU validation environment for kvcached on a Tencent AnyDev machine with two NVIDIA H20 GPUs, built kvcached from source, installed the vLLM/NIXL stack, and ran end-to-end prefill/decode disaggregation smoke tests.

The validation covers both the plain vLLM+NIXL baseline and the kvcached-enabled path. I also added a follow-up fix after the first GPU run revealed that the Python integration layer did not respect `KVCACHED_GPU_UTILIZATION` when sizing the virtual KV tensors.

## Contributions Validated

- Added a CPU pytest workflow and a local CPU test runner.
- Added a vLLM + NIXL prefill/decode smoke harness.
- Added preflight/check-only support for the smoke harness.
- Fixed Python 3.9 test compatibility in the prefix-cache tests.
- Fixed `KVCACHED_GPU_UTILIZATION` handling in both vLLM and SGLang integration layers.
- Added regression coverage for the GPU-utilization sizing behavior.

## H20 Environment

- Host: Tencent AnyDev GPU environment
- GPU: 2 x NVIDIA H20
- GPU memory: 97,871 MiB per GPU
- Driver: 535.247.01
- Python: 3.9.20
- PyTorch: 2.8.0+cu128
- CUDA reported by PyTorch: 12.8
- vLLM: 0.10.2
- NIXL: 0.8.0 package installed
- kvcached checkout: `/data/workspace/kvcached`
- Virtual environment: `/data/workspace/kvcached-venv`
- Compiler environment: `/data/workspace/kvcached-compiler`

## Test Results

| Validation | Result |
| --- | --- |
| `bash tools/run_cpu_tests.sh` | 56 passed |
| `pytest tests/test_alloc_kv_cache_alignment.py tests/test_vllm_nixl_compat.py` | 48 passed |
| Editable kvcached build on H20 | passed |
| vLLM + NIXL baseline P/D smoke | passed |
| vLLM + NIXL + kvcached P/D smoke | passed |

## GPU Smoke Matrix

All passing runs used `Qwen/Qwen2.5-1.5B-Instruct` with prefill on GPU 0 and decode on GPU 1.

| Run | Baseline | kvcached | Requests | Block Size | Max Model Len | Prompt Repetitions | Remote Blocks | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Full smoke | yes | yes | 3 | 128 | 512 | 1 | 2 | PASS |
| Stability smoke | yes | yes | 10 | 128 | 512 | 1 | 2 | PASS |
| Longer prompt smoke | yes | yes | 5 | 128 | 1024 | 3 | 4 | PASS |
| Deep transfer smoke | no | yes | 3 | 128 | 2048 | 6 | 8 | PASS |
| kvcached 20-request smoke | no | yes | 20 | 128 | 512 | 1 | 2 | PASS |
| kvcached block-size 64 smoke | no | yes | 5 | 64 | 1024 | 3 | 8 | PASS |
| kvcached block-size 32 smoke | no | yes | 3 | 32 | 1024 | 3 | 16 | PASS |

One boundary run with `BLOCK_SIZE=256` was rejected by vLLM 0.10.2 before startup because its CLI only accepts block sizes `1, 8, 16, 32, 64, 128`. This is not a kvcached runtime failure.

## Representative Logs

Remote log directory:

`/data/workspace/kvcached-professor-logs`

Key logs:

- `/data/workspace/kvcached-professor-logs/h20-full-smoke-smallpool-20260723-045714.log`
- `/data/workspace/kvcached-professor-logs/h20-stability-10req-20260723-050301.log`
- `/data/workspace/kvcached-professor-logs/h20-longprompt-5req-20260723-050443.log`
- `/data/workspace/kvcached-professor-logs/h20-kvcached-deeptransfer-3req-20260723-050641.log`
- `/data/workspace/kvcached-professor-logs/h20-kvcached-20req-20260723-051333.log`
- `/data/workspace/kvcached-professor-logs/h20-kvcached-block64-5req-20260723-051436.log`
- `/data/workspace/kvcached-professor-logs/h20-kvcached-block32-3req-20260723-051615.log`

Representative successful client outputs include:

- `{"mode": "with_kvcached", "requests": 3, "status": "PD_CLIENT_OK", "total_elapsed_sec": 5.613}`
- `{"mode": "with_kvcached", "requests": 10, "status": "PD_CLIENT_OK", "total_elapsed_sec": 5.962}`
- `{"mode": "with_kvcached", "requests": 20, "status": "PD_CLIENT_OK", "total_elapsed_sec": 6.751}`
- Deep-transfer runs reached `remote_blocks=8`.
- The `BLOCK_SIZE=32` run reached `remote_blocks=16`.

## Notes From Debugging

The initial H20 run showed that the default kvcached virtual KV tensor sizing exposed a very large region to NIXL. The vLLM/SGLang Python integration path was using total GPU memory directly, while the lower-level allocator exposed `KVCACHED_GPU_UTILIZATION`.

I fixed the integration layer to apply `GPU_UTILIZATION` before deriving the per-layer K/V tensor size, and added regression tests for both vLLM and SGLang. After this fix, small-pool H20 smoke runs completed successfully and also exercised block reuse under repeated requests.

## Reproduction Command Pattern

```bash
ssh -p 36000 root@minkali-any4.devcloud.woa.com
cd /data/workspace/kvcached
. /data/workspace/kvcached-venv/bin/activate
export PATH=/data/workspace/kvcached-compiler/bin:$PATH
export CC=/data/workspace/kvcached-compiler/bin/x86_64-conda-linux-gnu-gcc
export CXX=/data/workspace/kvcached-compiler/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=/data/workspace/kvcached-compiler/bin/x86_64-conda-linux-gnu-g++

LOG_DIR=/data/workspace/kvcached-professor-logs \
INSTALL_VLLM=0 \
INSTALL_EDITABLE=0 \
RUN_UNIT_TESTS=0 \
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
GPU_MEMORY_UTILIZATION=0.092 \
KVCACHED_GPU_UTILIZATION=0.0012 \
NUM_REQUESTS=10 \
MAX_TOKENS=8 \
bash tools/run_vllm_nixl_pd_smoke.sh
```
