# Ornith-1.5 patches for Strata v0.2.9 (r1)

One consolidated diff that lets the released Strata v0.2.9 source (engine 0.1.55, commit `c62d5e3a`) serve
Ornith-1.5 35B A3B, plus the prebuilt engine it produced.

## What is in it

| Change | Why |
|---|---|
| `ssm_alpha` / `ssm_beta` as K-quants | the official Ornith quants store them as Q4_K, stock expects F32/Q8_0 and refuses the file (upstream PR 3925) |
| `--kv fp16\|int8\|k8v4`, `--gqa-attn`, `--attn-mma` | cache modes and the grouped-query attention path (upstream PR 3903) |
| Q4_K expert down projections, 1024-wide CPU expert pool | the 35B's expert width, which the stock kernels do not cover |
| `q36_attn.cuh` | the attention pieces the above two need |
| `--vram-headroom`, `--shared-card` | place experts on a second card that holds less than a full layer |

## Apply

```
apply.cmd "C:\path\to\Strata"
```

The tree must be v0.2.9 (engine 0.1.55): `CMakeLists.txt` says `project(strata VERSION 0.1.55)`. The older five-diff
set in `patches/legacy-0.1.54` targets the pre-release tree and does not apply to v0.2.9; it is kept for history.

## Build

The released engine used:

```
cmake -S <tree> -B <tree>\build-qwen35 -G Ninja -DCMAKE_BUILD_TYPE=Release
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
  -DSTRATA_Q6K_EXPERTS=ON
  -DCMAKE_CUDA_COMPILER=...\nvcc.exe -DSTRATA_GGML_DIR=<pinned llama.cpp>
ninja -C <tree>\build-qwen35 strata strata_kq
```

## Measured

RTX 3090, Ornith-1.5-35B-Q4_K_M, `--mtp 2`, prompt 3,733 tokens, 300 generated:

```
prompt 3733 tokens read in 1844 ms (2007 tok/s), 300 generated in 2302 ms (130.0 tok/s), drafts accepted 198 of 204
```

Three interleaved runs against the previous engine (built on the pre-release 0.1.54 source) read 129.8 / 130.1 /
130.3 against 130.3 / 130.3 / 130.0 tok/s: the rebuild is performance-neutral. An older note quoted 188.9 tok/s at
"4K"; that came from a different measurement setup and is not comparable.

## Files without an MTP block

Some repacks are published without the MTP block, and `strata-q36` then refuses them with
`missing qwen35moe.nextn_predict_layers` even without `--mtp`. Check the model's own repository first: the official
`AtomicChat/Ornith-1.5-35B-A3B-GGUF` ships `mtp-Ornith-1.5-35B-A3B-Q8_0.gguf` next to the quants.

If you do not have that file, there are two ways to get a working head:

1. **Extract it from a quant that carries it.** The block is named `blk.<n_layers>.*`: 20 tensors for the 35B, a full
   layer plus the `nextn.*` projections.

   ```
   python tools/extract_mtp.py "Ornith-1.5-35B-Q4_K_M.gguf" "mtp-blk40-ornith35b.gguf"
   strata.exe --native <file>.gguf --max-context 262144 --mtp 2 --mtp-gguf mtp-blk40-ornith35b.gguf
   ```

   The loader requires every tensor of the `--mtp-gguf` file to start with `blk.<n_layers>.`, which is what the tool
   writes.

2. **Download one:** [SizorCloud/Ornith-1.5-35B-A3B-MTP](https://huggingface.co/SizorCloud/Ornith-1.5-35B-A3B-MTP)
   holds a head derived from Qwen3.6-35B-A3B.

Measured on `Ornith-1.5-35B-A3B-AD-Q6_K.gguf`, RTX 3090, 2,739-token prompt, 120 generated:

| head | decode | drafts |
|---|---|---|
| none (`--mtp 0`) | 49.7 tok/s | 0 |
| derived from Qwen3.6-35B-A3B | 58.3 tok/s | 65 of 110 (59%) |
| extracted from Ornith-1.5-35B-Q4_K_M | **67.9 tok/s** | **79 of 84 (94%)** |

The head from the same model family stays closest.

One trap: the server applies a default presence penalty of 1.5 when a config has no `sampling` block, and that
disables MTP drafts entirely (`drafts accepted 0 of 0`). Put `"presence_penalty": 0.0` in the config's `sampling`.

## Note

The DeepSeek-V4.1 CPU lane patches are a separate set:
`github.com/AndyAnttle/strata-deepseek-patches`. Both patch sets touch `CMakeLists.txt`, in different hunks, so they
can be applied in either order.
