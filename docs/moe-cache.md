# MoE cache

When the experts of MoE layers stay in host memory (`-cmoe`, `-ncmoe`, `-ot`), the MoE cache keeps the most used ones in device memory for small batches (decode, speculative decoding). Large batches still upload the experts they use.

| Argument | |
|---|---|
| `--moe-cache-mib N` | cache size in MiB, split among the GPUs like the layers (default: 0, disabled) |
| `--moe-cache-cpu-share F` | share of the cache misses computed on the CPU from the host experts instead of uploaded, once the cache is full (default: 0) |
| `--moe-cache-direct-io` | read the cache misses from the model file instead of copying them from the host experts (default: off) |

The cache hits run on the device. With `--moe-cache-cpu-share F` the CPU computes a share `F` of the misses while the device computes the hits and uploads the other misses into the cache. This is faster when the CPU reads the host experts faster than the device can upload them, for example with fast RAM, a PCIe 4 link or experts that are not in pinned memory. The host experts are pinned with `-lm none`, `-lm mlock` and `-lm dio`, unless `--no-host`, but not when they are mmap'd (the default). The device uploads pinned experts at the full speed of the link, so the CPU share often does not help then. Until the cache is full every miss is uploaded, so the cache fills as fast as without the CPU share. The best share depends on the machine, the model and the cache size.

## Picking the arguments

Put the script in front of your server command:

```bash
python3 scripts/moe-cache-tune.py -- build/bin/llama-server -m model.gguf -cmoe -t 8 -c 32768
```

It starts the server 6 times, at two cache sizes × shares 0, 0.5 and 0.95, and times the generation of a few thousand tokens. That takes a few minutes, mostly model loading.

From these runs it fits the upload and CPU speeds of the machine, then replays the routing of all the runs through a copy of the cache logic for other sizes and shares. It prints the measured speeds, the predicted t/s and the recommended arguments: the lowest share within 5% of the fastest.

- **Server command:** keep your own arguments (`-t`, `-c`, expert placement, load mode); the speeds and the free device memory depend on them.
- **`--max-mib N`:** tunes a smaller cache than what fits on the device. The runs in `--out` are reused while the command stays the same, so trying another value is instant.
- **`--markdown`:** prints Markdown tables to paste in a PR.

## Models larger than the RAM: direct I/O

With `--moe-cache-direct-io` a cache miss is read from the model file straight into its cache slot, and the host copy of the expert is not touched. With a memory-mapped model (`-lm mmap`) the experts kept in the CPU are then never read into RAM, so the experts can be several times larger than the RAM: only the cache and the non-expert weights are resident.

```bash
build/bin/llama-completion -m model.gguf -ngl 99 -cmoe -lm mmap --no-repack --moe-cache-mib 40000 --moe-cache-direct-io -ub 32
```

- **Where it works:** the cache must be device memory that the host can write in place, which today means Metal with shared buffers (the default on Apple silicon). Elsewhere the option logs a warning and the misses are copied from the host experts as before. It is not combined with `--moe-cache-cpu-share`.
- **Reads:** one `pread` per expert tensor of a missing expert (three per expert for separate gate, up and down tensors), consecutive experts going to consecutive slots in one read, with up to `LLAMA_MOE_CACHE_IO_THREADS` reads in flight (default 4). On macOS the reads go around the file cache, so the experts are not held twice.
- **Alignment:** macOS reads around the file cache only when the file offset, the size and the address are multiples of 4096 bytes. A GGUF file aligns tensor data to 32 bytes by default, so its experts are read into an aligned buffer and copied into the cache, which costs one copy; a warning says so. A file written with `general.alignment = 4096` is read in place.
- **Prefetch:** with this option the expert tensors are left out of the prefetch of a memory-mapped model (`llama_model_params::moe_no_prefetch`), which would otherwise pull the whole file through the page cache at load.
- **Expert types:** the experts must stay in plain host memory. Types that the CPU backend repacks (Q4_0, Q4_K, Q2_K, Q8_0 and others) need `--no-repack`.
- **Batches:** only batches of at most 32 tokens whose experts fit the cache read it. A larger batch reads the host experts, which pages them in from the disk; keep the micro-batch (`-ub`) at 32 or less. A warning is logged once when a larger batch is built.
- **Counters:** with `-v` the cache logs its hits and misses at exit, and with direct I/O the bytes read, the number of reads and the time spent waiting for them.
