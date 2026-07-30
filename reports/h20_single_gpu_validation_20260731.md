# kvcached portable single-H20 validation report

Date: 2026-07-31

## Scope

This run used one NVIDIA H20 to validate the self-hosted GPU CI path, restore a
previously skipped resize regression, verify integration sizing tests, and
measure pinned-memory transfer costs for the CPU-offload design.

The validation is independent of the machine provider. It starts from the
public kvcached repository and uses standard Linux, CUDA, PyTorch, pytest, and
GitHub Actions interfaces. No organization-specific service, model path,
credential, scheduler, or storage API is required by the implementation.

## Environment

- GPU: 1 x NVIDIA H20
- GPU memory: 97,871 MiB
- Driver: 535.247.01
- System CUDA toolkit: 12.1
- Base Python: 3.9.20
- Base PyTorch: 2.4.1+cu121
- Initial system compiler: GCC 8.5.0
- Working compiler: Conda GCC 11.4.0

## GPU CI results

The first build found two real self-hosted-runner requirements:

1. PyTorch 2.4 rejects GCC 8 and requires GCC 9 or newer.
2. The container mounted the runtime driver as `libcuda.so.1`, while the
   extension linker searched for `libcuda.so`.

After installing GCC 11 and linking through the CUDA toolkit stub, the core GPU
profile passed:

- initial core profile: 35 passed, 1 skipped;
- repeated core profile: five successful iterations;
- skipped resize test, after inspection and re-enabling: passed;
- full core profile with resize enabled: 36 passed in each of five iterations.

This is 180 consecutive passing test executions in the final stability run.
No compute process remained on the GPU after the profile completed.

Two follow-up commits were validated on the H20 and pushed:

- `7c6987f` re-enables the stale resize regression;
- `acc486e` automatically links CUDA extensions against the toolkit stub.

## Completed persistent-runner delivery

Commit `e4045f0` completes the provider-neutral deployment layer around the
validated GPU entry point:

- separate core, vLLM, and SGLang Python environments and extension builds;
- one-command environment provisioning with `pip check`;
- daily, manual, push, and maintainer-approved pull-request triggers;
- idle-GPU protection and a host-wide concurrency lock;
- independent vLLM and SGLang correctness requests;
- complete artifact capture on success and failure;
- configurable one-GPU engine profiles and a two-GPU NIXL profile.

The deployment contracts, shell syntax, workflow YAML, and engine-selection
logic passed 27 CPU-only tests. The repository contains no provider-specific
host, path, scheduler, storage, or credential dependency.

## Additional validation

- GPU-utilization allocation sizing: 31 passed.
- NIXL compatibility and smoke-contract tests: 21 passed.
- NIXL check-only preflight: passed and correctly detected one visible GPU.
- SGLang legacy page-accounting compatibility: 8 passed.
- CPU-offload control-plane and benchmark-statistics tests: 16 passed.

The engine smoke entry points remain provider-neutral:

- `GPU_CI_PROFILE=vllm` starts one public vLLM correctness smoke test;
- `GPU_CI_PROFILE=sglang` starts one public SGLang correctness smoke test;
- `GPU_CI_PROFILE=engines` runs both sequentially;
- `GPU_CI_PROFILE=nixl` runs the two-GPU vLLM/NIXL P/D smoke.

They accept a configurable model identifier or local model directory and do
not contain environment-specific paths.

## CPU-offload transfer baseline

Each kvcached logical page contains one 2 MiB slice per layer and per K/V
buffer. Results below are averages over 100 iterations using pinned host
memory.

| Layers | Logical page | D2H | D2H bandwidth | H2D | H2D bandwidth |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 32 MiB | 0.666 ms | 50.41 GB/s | 0.624 ms | 53.80 GB/s |
| 16 | 64 MiB | 1.345 ms | 49.88 GB/s | 1.246 ms | 53.86 GB/s |
| 32 | 128 MiB | 2.691 ms | 49.87 GB/s | 2.493 ms | 53.84 GB/s |
| 40 | 160 MiB | 3.360 ms | 49.93 GB/s | 3.114 ms | 53.87 GB/s |

The upgraded benchmark also records mean, P50, P95, minimum, and maximum
latency. Its code and CPU tests are in commit `05ee08e`.

## Conclusions

The GPU CI implementation is executable on a real H20 and stable across
repeated runs. The run also converted two environment-specific failures into
portable repository fixes and demonstrated that the resize path no longer
needs to remain skipped.

For CPU offloading, the measured lower bound for restoring one 32-layer logical
page is about 2.5 ms. This is promising, but an engine should still compare
that cost with prefix recomputation and include queueing, metadata, page
mapping, and CUDA synchronization overhead before deciding to offload.

## Reproduction

On any Linux x86-64 CUDA runner with a CUDA-enabled PyTorch environment and
GCC 9 or newer:

```bash
git clone --branch zixuan/gpu-ci-runner \
  https://github.com/Lanoxia/kvcached.git
cd kvcached
GPU_CI_PROFILE=core GPU_CI_REPEAT=5 bash tools/run_gpu_ci.sh
```

This branch is the current public validation branch. After the changes are
merged upstream, the same command can use the OVG repository's default branch.

For the CPU-offload transfer baseline:

```bash
python tools/benchmark_cpu_offload.py \
  --page-size-mb 2 \
  --layers 32 \
  --kv-buffers 2 \
  --iterations 100 \
  --report cpu-offload-transfer.json
```
