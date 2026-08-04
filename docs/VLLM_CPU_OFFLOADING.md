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
  --kv-offloading-size 4
```

The patch applies to vLLM 0.12 and newer. It accepts both historical
`use_uniform_kv_cache(attn_groups, cache_dtype)` and current
`use_uniform_kv_cache(attn_groups)` signatures.

Some older vLLM releases, including 0.16.0, reject `OffloadingConnector` when
the hybrid KV cache manager is enabled. For those releases, add:

```bash
--disable-hybrid-kv-cache-manager
```

Current vLLM implements `SupportsHMA` in `OffloadingConnector`, so that
workaround is not required there. GPU validation should cover an offload, a CPU
cache hit restored to GPU, and output equality against the same run without
offloading.
