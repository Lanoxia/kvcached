# vLLM NIXL P/D Disaggregation

This example runs an end-to-end smoke test for kvcached with vLLM
prefill/decode disaggregation through `NixlConnector`.

The smoke test starts one prefill server and one decode server, sends direct
P/D requests, verifies that the prefill response publishes remote KV blocks,
and checks logs for the kvcached NIXL compatibility patch.

## What It Covers

- Baseline vLLM + NIXL P/D path without kvcached.
- kvcached + NIXL P/D path with `KVCACHED_CONTIGUOUS_LAYOUT=false`.
- Response correctness with a deterministic prompt.
- Failure signatures such as `set_stride`, inconsistent block counts, and NIXL
  transfer errors.

## Requirements

- A CUDA GPU environment with PyTorch and CUDA available.
- `curl` and `setsid`.
- Enough memory to serve the selected model. The default model is
  `Qwen/Qwen2.5-1.5B-Instruct`.

The helper script can install its pinned vLLM/NIXL stack and kvcached in
editable mode. To reuse an existing environment, set `INSTALL_VLLM=0` and/or
`INSTALL_EDITABLE=0`.

## Run

From the repository root:

```bash
bash tools/run_vllm_nixl_pd_smoke.sh
```

Useful overrides:

```bash
INSTALL_VLLM=0 \
MODEL=Qwen/Qwen2.5-1.5B-Instruct \
PREFILL_GPU=0 \
DECODE_GPU=1 \
GPU_MEMORY_UTILIZATION=0.35 \
BLOCK_SIZE=128 \
NUM_REQUESTS=3 \
EXPECTED_SUBSTRING=Paris \
MIN_REMOTE_BLOCKS=2 \
bash tools/run_vllm_nixl_pd_smoke.sh
```

On a single-GPU machine, omit `DECODE_GPU`; the script will use GPU 0 for both
servers when only one GPU is visible.

## Expected Pass Signals

The script prints `PD_CLIENT_OK` after the client validates the P/D response.
At the end of a successful run it prints:

```text
[nixl-smoke] without_kvcached PASS
[nixl-smoke] with_kvcached PASS
[nixl-smoke] PASS
```

The logs are written under the reported `LOG_DIR`. For the kvcached case, the
script checks for evidence that the NIXL compatibility patch was applied and
that the layout/block-count reconciliation path ran.

## Notes

- kvcached + NIXL currently requires non-contiguous KV layout because vLLM's
  `NixlConnector` assumes each layer's K/V regions are block-contiguous.
  The smoke script defaults `KVCACHED_NIXL_CONTIGUOUS_LAYOUT=false`.
- Set `RUN_BASELINE=0` to skip the baseline pass while iterating on kvcached.
- Set `STRICT_EXPECTED_SUBSTRING=0` or `CLIENT_ENDPOINT=completions` for
  lower-level transport debugging where generated text quality is not the
  pass/fail condition.
