# The Ornith-1.5-122B experiment: what we tried, what it measured, why it is not in this release

This is the record of the work that came before the 35B patch set. None of it is needed to run Ornith-1.5-35B-A3B;
it is here so that the same ground is not covered twice, and so the numbers that ended the experiment stay
checkable.

Everything below was measured on one RTX 3090 24 GB (or two, where it says so), Windows, MSVC 2022, CUDA 13.0.

## The model and the hope

`Ornith-1.5-122B` (GSQ-RCO IQ3_S, 53.4 GB over two shards): 48 layers, 256 experts with top-8 routing, 2 KV heads,
head_dim 256, 262,144 positions. Of the 48 layers, 36 are linear-attention (a fixed-size recurrent state) and 12
are full attention, so the K/V cache only exists in a quarter of the layers.

The hope was a 122B-class model on a single 24 GB card: 47.5 GiB of routed experts against 24 GB of VRAM, with the
rest streamed from host memory. The 35B runs at 168-187 tok/s in this same box, so the question was how much of
that a model 3.5x larger would keep.

## MTP: two heads, one of them useless

Speculative decoding for this family needs the MTP block, and the checkpoint's own block turned out to be at
initialization:

- its norms sit at ~1.000 (`enorm`, `hnorm`, `shared_head_norm`), where a trained block (Qwen3.6-35B, the same
  architecture) reads 0.40 / 0.44 / 3.11;
- its expert tensors have std 0.015 with 0.74% exact zeros, i.e. untouched weights;
- measured: **0 of 22 drafts accepted**, against 60% for Qwen3.6-35B with its trained block.

We converted the block from the official checkpoint with llama.cpp's converter, grafted its 20 tensors into the
first GGUF shard (block_count 48, +2.7 GB, 517 tensors, `split.tensors.count` 879 -> 899, the original shard kept
as `.orig`), and verified the copy tensor by tensor: max |diff| 0.0097 at std 0.027, ~0.6% (the Q8_0 noise level).
The graft worked and the engine still accepted 0 drafts, which is exactly what an untrained block looks like.

A Q8_0 head built from the MLX split of the same checkpoint (`ToPo-ToPo/Ornith-1.5-122B-MTP-bf16`) behaved
differently: **38% of drafts accepted on a code prompt** (10 of 26), 11-13% on structured extraction prompts. That
is the head the engine ended up using, through a side-GGUF flag (`--mtp-gguf`) that this patch set does not carry.

Both head repositories are published under `SizorCloud/Ornith-1.5-122B-MTP` and `SizorCloud/Ornith-1.5-35B-A3B-MTP`.
The 35B one is still useful for quantizations that ship without an MTP block; the 122B one only matters if you run
that model, which this release does not.

## Speed: what one card can do with 47.5 GiB of experts

K/V cache `int8` plus the grouped-query decode kernel, which is what makes 262,144 positions fit at all:

| context | experts in VRAM | expert work on the CPU | decode | prefill |
|---|---|---|---|---|
| 262,144 | 25 of 256 (4.64 GiB) | 89% | 3.7-3.9 tok/s | 182 tok/s |
| 32,768 | 46 of 256 (8.53 GiB) | 81% | 4.2-4.3 tok/s | 180 tok/s |

Real conversations on the same card ran 5.8-8.3 tok/s. The CPU expert pool carries 60-90% of the expert
matmuls, so its thread count matters: `--cpu-threads 16` (of 20 logical) moved a held-out run from 7.0 to
7.7 tok/s. That is the only tuning that helped, and it is in the installer for the 35B too.

For scale, the same box runs:

| model | cards | decode | expert work on the CPU |
|---|---|---|---|
| Qwen3.8-Flash-Next, UD-Q4_K_XL | 1 | 44.6 tok/s | cache hit 67.6% |
| Qwen3.8-Flash-Next, UD-Q4_K_XL | 2 | 50.8 tok/s | cache hit 82.8% |
| DeepSeek-V4.1-Flash, 2.00 bpw | 2 | 11.5-15.3 tok/s | **0.9%** |
| Ornith-1.5-122B, IQ3_S | 1 | 3.7-8.3 tok/s | 60-90% |

A bigger model being faster is not a paradox: DeepSeek keeps 99% of its expert calls on the GPUs, and the 122B
keeps 10-40%. See "why" below.

## The expert-profile experiment (the one that should have worked)

If expert routing is skewed, a static ranking of (layer, expert) pairs should beat a cache that only reacts to the
current conversation. We implemented the whole path in the engine:

- `--dump-routing PATH` writes the routed ids of every layer's forward in the format of `tools/make_profile.py`;
- `--expert-profile PATH` reads the main engine's `STRP` profile and seats the VRAM slots in rank order instead of
  "the first S expert ids", which is what the engine did before;
- `STRATA_Q36_SWAP_BUDGET` / `STRATA_Q36_PROMPT_SWAPS` bound the slot-replacement bursts (each swap copies ~4 MB).

The collection is re-usable: `bench/profile_corpus.json` (45 prompts, code/maths/reasoning/tool use/7 languages)
through `tools/collect_profile_traces.py`, then `tools/make_profile_mix.py`. We ran 15 of them on the 122B
(decode windows only, 1.15M routed ids):

| slots | share of routed mass the ranking covers |
|---|---|
| 8 | 11.2% |
| 24 | 25.3% |
| 46 | **39.7%** |
| 96 | 63.7% |
| 128 | 75.1% |

The top expert of a layer holds only 0.9-3.8% of that layer's routing (uniform would be 0.39%), so the routing is
skewed, but not nearly as much as a 384-expert model's: DeepSeek's shipped profile holds 60-85% of the experts its
prompts route at 5,805 slots, and its own documentation records 69.2% measured against 94.9% claimed.

Held-out prompts, the profile never saw them, context 32768, 46 slots:

| prompt | adaptive (what the engine does now) | profile, slots frozen |
|---|---|---|
| science | **41.6% hit, 8.2 tok/s** | 35.4%, 6.5 |
| tool use | **36.1%, 5.5** | 30.1%, 4.5 |

Context 262,144, 25 slots: adaptive 28.2% and 6.4-7.0 tok/s, profile 24.5% and 6.2, no profile and no swaps 12.9%
and 5.1. The profile and the adaptive policy are equal there within run-to-run noise, and both beat frozen slots.

Verdict: at 25-46 slots the per-conversation policy already extracts what a top-K selection can give, and a static
ranking does not beat it. The code was left in the engine, off by default; the 35B does not use it.

One correction worth recording: an early baseline of "18.9% hit" that motivated the work came from a synthetic
8k-token filler prompt, long and out of domain. On realistic short prompts the same engine reports 36-46%. The
profile was chasing a number that did not exist in real use.

## Why the model never got fast

1. **It does not fit.** 47.5 GiB of routed experts against 24 GB of VRAM leaves 10-18% of the pairs resident, and
   the routing is too flat for a small cache to cover more.
2. **The CPU pool is then the engine.** 60-90% of expert matmuls run on 10-16 threads, ~160 us per row.
3. **Two cards would help, and would also end the point.** Giving the model a second card means porting the main
   engine's layer split into the qwen35 path: 1-2 weeks, and after it the model needs both cards while still being
   a 122B at IQ3_S. The measured worth of a second card on the *well-optimised* path (Flash-Next) is +14% (44.6 ->
   50.8 tok/s); for the 122B it would be larger in relative terms, but it stops being "the heavy model that leaves
   a card free", which was its only advantage.
4. **Its niche is taken twice.** For quality, DeepSeek-V4.1-Flash on two cards is stronger and faster (11.5-15.3
   tok/s, 0.9% CPU). For a single card, Ornith-1.5-35B-A3B runs 168-187 tok/s and scores SWE-bench Verified 79
   against the 122B's 84.5: better quality per second by a wide margin.

So the 122B was dropped from the installer and from the engine patches. The local files stay on the author's disk;
nothing in this repository depends on them.

## Numbers kept for reference

- Engine at the time: `qwen36` path, 0.1.54, with PR #3925 and PR #3903 applied plus our own fixes.
- 122B: 48 layers (12 full attention), 256 experts top-8, 2 KV heads, 256 head_dim, 262,144 positions, 53.4 GB.
- K/V cache at 262,144 in `int8`: 3.2 GB; in `fp16`: 6.4 GB (did not fit beside the experts).
- Expert slice: 4.15 MB; a swap copies one slice into a slot (~4 MB), the post-prompt seating burst was capped at
  2,048 swaps and that alone was measurable at long context.
- The 35B's own MTP block in the official quants is trained (norms 0.402 / 3.109, same as Qwen3.6-35B); the 35B
  needs no side head at all.
