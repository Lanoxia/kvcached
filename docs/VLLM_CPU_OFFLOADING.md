# vLLM CPU KV offloading

kvcached can use vLLM's `OffloadingConnector` while keeping GPU KV tensors in
the elastic VMM pool. The integration forces vLLM's per-layer allocation path;
the connector then copies inactive KV blocks between those tensors and CPU
memory without bypassing kvcached.

Enable kvcached and request a bounded CPU cache:

```bash
export ENABLE_KVCACHED=true
export KVCACHED_AUTOPATCH=1

vllm serve Qwen/Qwen2-1.5B \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cpu_bytes_to_use": 4294967296,
      "block_size": 64
    }
  }'
```

The patch applies to vLLM 0.12 and newer. It accepts both historical
`use_uniform_kv_cache(attn_groups, cache_dtype)` and current
`use_uniform_kv_cache(attn_groups)` signatures.

Older vLLM releases used `--kv-offloading-size 4` for a 4 GiB CPU cache.
Some of those releases, including 0.16.0, reject `OffloadingConnector` when
the hybrid KV cache manager is enabled. For those releases, also add:

```bash
--disable-hybrid-kv-cache-manager
```

Current vLLM implements `SupportsHMA` in `OffloadingConnector`, so that
workaround is not required there. GPU validation should cover an offload, a CPU
cache hit restored to GPU, and output equality against the same run without
offloading.

The repository provides a repeatable smoke test for that runtime gate:

```bash
bash tools/run_vllm_cpu_offload_smoke.sh
```

It records vLLM's offload counters before and after replaying an evicted prompt,
requires both transfer directions to occur, checks deterministic output
equality, and saves the server log and a JSON result under `LOG_DIR`.
