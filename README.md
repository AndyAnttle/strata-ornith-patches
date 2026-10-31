# Ornith-1.5 on Strata: patched engine, patches, installer

**Compatibility: everything here is cut for Strata v0.2.9** (the `v0.2.9` tag, engine reports `strata 0.1.55`).
The patches are diffs against that tag's source tree and were verified to apply to it exactly; the prebuilt engine
was built from that tree. On a newer Strata they may still apply, but check the first hunk that fails and fix it
against the new source rather than the patch, and rebuild — a Strata release expects its own engine version, and a
patched 0.1.55 engine next to a newer server is not a combination anyone tested. On an older tree, do not try:
the KV modes and the `ssm_alpha` reading are not there to extend.

Strata v0.2.9 cannot serve the official Ornith quants out of the box: its `qwen35` engine path refuses a
GGUF whose `ssm_alpha` / `ssm_beta` are K-quants (`blk.0.ssm_alpha.weight is Q4_K, expected F32, BF16, F16 or
Q8_0`), and it has no K/V cache modes for that path, so every expert spills out of VRAM. These patches fix
that; the release also ships the engine already built, so most people never touch a compiler.

The 122B work that came before this (MTP heads, expert profiles, why the model was dropped) is written up in
[`122B-experiments.md`](122B-experiments.md).

The patches modify Strata (<https://github.com/Niko1221/Strata>), which is MIT licensed; this patch set is
distributed under the same terms (see `LICENSE`).

## Sources

Where each patch's approach comes from, so it does not live in a vacuum:

| patch | source |
|---|---|
| 01 | upstream pull request Niko1221/Strata#3925 — reading `ssm_alpha` / `ssm_beta` as K-quants (`Niko1221/Strata`); the idea is that PR's, re-cut against v0.2.9 |
| 02 | upstream pull request Niko1221/Strata#3903 — the K/V cache modes and the grouped-query decode kernel; re-cut against v0.2.9. The kernel itself cites Niko1221/Strata#3641 in its header |
| 03 | ours, from the IQ3_S 122B work: the expert kernels had to take Q4_K downs and the CPU pool had to hold a 1024-wide expert |
| 04 | ours, from measurements on a 12 GB card: the placement must tell a dedicated card from a shared one, and keep room on the latter |
| 05 | ours, same work: NVML cannot answer "who drives the display" on WDDM (it returns `ERROR_NOT_SUPPORTED` for the graphics-process list), so the engine measures the pre-load footprint and takes the installer's verdict as `STRATA_Q36_SHARED` |

The engine's own comments keep the issue numbers they were written with (№3641 for the grouped-query kernel, №3911
for the draft-vocabulary row bound, №3976 for the layer-major prompt walk); they are upstream's and are left as they
are.

## Where things go

Strata normally unpacks to a folder that holds `START-HERE.bat`, `engine\`, `serve\`; this README calls it
`<Strata>`. Unpack the three downloads like this:

| download | unpack to | what appears |
|---|---|---|
| `strata-engine-0.2.9-qwen35.zip` | **`<Strata>\`** (beside `engine\`) | `<Strata>\engine-0.2.9-qwen35\{strata.exe, strata_kq.dll, strata-vision.exe, README.txt}` |
| `ornith-install.zip` | **`<Strata>\`** | `<Strata>\ornith-install\{INSTALL-ORNITH.bat, install.py, README.md}` |
| `strata-ornith-patches-0.2.9.zip` | anywhere (only for building from source) | `patches\`, `apply.cmd`, `apply.sh`, `tools\apply_patch.py` |

The installer looks for the engine in `<Strata>\engine-0.2.9-qwen35\`, then one level up (for a layout where
`engine-*` sits beside the Strata folder), then in the older `engine-qkv-mtp\`, then the stock `engine\`.

## Quick start (no build)

1. Unpack the two zips as in the table.
2. Run `<Strata>\ornith-install\INSTALL-ORNITH.bat`.
   Menu: quantization (Q4_K_M 21.7 GB or Q6_K 29.2 GB) → GPU → K/V cache → context length.
   Recommended: `k8v4` on a 20 GB+ card, context 128K.
3. Start the model with the generated `run-ornith-1.5-35b-…bat`, or through `START-HERE.bat`.

The vision encoder (`mmproj-…-BF16.gguf`) is downloaded with the model and wired into the config.

## Building the engine yourself

```bat
apply.cmd "C:\path\to\<Strata>"
```

`apply.cmd` applies the three diffs in order. It uses Git's `patch.exe` when that is on `PATH` and falls
back to the bundled `tools\apply_patch.py` (plain Python, no Git needed). `apply.sh` does the same under bash.
Both stop at the first hunk that does not match instead of leaving a silently half-patched tree.

Then:

```bat
cmake -S "<Strata>" -B "<Strata>\build-qwen35" -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF ^
  -DSTRATA_Q6K_EXPERTS=ON -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe" ^
  -DSTRATA_GGML_DIR="C:\path\to\pinned\llama.cpp"
ninja -C "<Strata>\build-qwen35" strata strata_kq
```

Copy the two files into `<Strata>\engine-0.2.9-qwen35\` (with `strata-vision.exe` from the release) and you
have what the zip above contains. `-DSTRATA_Q6K_EXPERTS=ON` is what lets a Q6_K file run; Q4_K_M decodes
identically with or without it (verified: same token ids).

## The three patches

| file | what it does | origin |
|---|---|---|
| `01-pr3925-kquant-ssm.diff` | reads `ssm_alpha` / `ssm_beta` as K-quants or IQ4_XS; also the `<fstream>` / `<iterator>` includes the second patch needs | upstream PR #3925 (open) |
| `02-pr3903-kv-modes.diff` | `--kv fp16\|int8\|k8v4`, `--gqa-attn`, `--attn-mma`, the grouped-query decode kernel (`q36_attn.cuh`), `--tokens-file` | upstream PR #3903 (open), re-cut against v0.2.9 |
| `03-qwen35moe-kernels.diff` | Q4_K expert down projections in `STRATA_D_FMTS`, a CPU expert pool sized for a 1024-wide expert | ours |
| `04-vram-headroom.diff` | the expert placement keeps a fifth of the card free (768 MiB to 2.5 GiB) instead of a flat 768 MiB, and warns after the warmup when a hybrid placement leaves under 512 MiB | ours |
| `05-shared-card.diff` | reads how much VRAM was in use before the model started, and whether a display drives the card, then keeps room on a shared card instead of filling it; `STRATA_Q36_SHARED=1` forces that from the outside (the installer sets it) | ours |

Patches 02 and 03 are offered as diffs **against the real v0.2.9 tree** rather than as the raw pull-request
patches: upstream moved 470 lines in `CMakeLists.txt` and inserted an include the PR predates, so the raw
diffs only apply with fuzz. Re-cut diffs apply exactly, which is also what lets the Python applier work.
A patch that cannot apply exactly is a signal that the tree is newer than v0.2.9, and then the hunk wants
fixing against the new source.

Patch 04 comes from a measured failure on a 12 GB card that also drives its display: with the old 768 MiB reserve
the placement filled 11.98 of 12 GB, Windows moved part of the model into shared memory as soon as the desktop
needed VRAM, and an 8K-token prompt decoded at 4.4 tok/s. With the reserve a fifth of the card the same run
decoded at 32.4 tok/s. The all-in-VRAM path keeps its own 256 MiB reserve and is untouched.

Verification: applying 01+02+03+04 to a fresh v0.2.9 tarball reproduces the built tree byte for byte, through
`apply.cmd` (patch.exe) and through the Python applier alike.

## Measured

Ornith-1.5-35B-A3B Q4_K_M, one RTX 3090 24 GB, greedy:

| context | K/V cache | decode | notes |
|---|---|---|---|
| 131072 (32K prompt) | `k8v4` | **168 tok/s** | every main-layer expert in VRAM (18.17 GiB) |
| 131072 | `int8` | 99-105 tok/s | 222 of 256 experts resident |
| 131072 | `fp16` (stock behaviour) | 24 tok/s | experts spill to the CPU pool |
| 4096 | `k8v4` | 189-191 tok/s | smoke test, MTP 7 of 12 drafts |

The speed comes from the freed VRAM, not from the attention math: the cache modes are what let the experts
stay on the card.

## Limits

- `--kv k8v4` and `--kv fp16` need at most 8 query heads per KV head. The 35B has 16 heads over 2 KV heads
  and fits; models with 32 heads over 2 cannot use them (`--kv int8 --gqa-attn` is the ceiling there).
- `k8v4` shrinks the engine's VRAM reserve to 256 MiB to seat every expert: keep that card free of other
  work, or the placement falls back to the hybrid mode and the speedup with it.
- No `--mtp-gguf`: a `qwen35moe` file that carries no MTP block of its own is refused. That is why the
  installer no longer offers the 122B.
