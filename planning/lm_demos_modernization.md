# The language demos: TinyGPT, TinyStories, bigram — modernization backlog

**Opened 2026-09-09.** Audit of `demos/MainTinyGptShakespeare.lean`, `demos/MainTinyStories.lean`,
`demos/MainBigramShakespeare.lean`. Companion to `planning/archive/tinygpt_demo_v2.md` (current and
complete for what it planned) and `planning/archive/demo_xla_port.md` (the runtime record). The
TinyGPT book entry was rewritten the same day (`3c8c5dce`) on a fresh run; this is what is left.

## §0 Verified today

* **Runtime: XLA/PJRT since `bc946fe6` (2026-08-26).** Zero IREE tokens in the three sources; all
  link `lowererLink`; `LowererSession` throughout; `graphArtifact` picks `.mlir`/`.vmfb`. Nothing
  is needed to "be on XLA".
* **Re-run 2026-09-09** (`runs/2026-09-09-tinygpt-nano-xla/`): nano, 10K steps, **2.279
  bits/char held-out**, train 1.99, **173 s wall clock compile included** (17 ms/step), one 4060 Ti.
  Reproduces the IREE-era 2.27/2.00. Bigram floor re-measured: 3.56 bits/char. The book,
  `demos/README.md`, and the front-door README quote these now.
* **Still stale, deliberately left:** `historical/RESULTS.md:58,87` ("CUDA / IREE", "IREE
  pipeline") — historical by policy; `planning/tour_realignment.md:48,166` (1.45 nats/char);
  `scripts/run_tinystories_8k.sh` is an IREE-era launcher (`IREE_BACKEND=cuda`, `iree-compile`)
  for a cloud A100; `LeanMlir/Train.lean` prints "Compiling vmfbs..." on XLA (a code string).

## §1 Bugs (fix these first; each is small)

1. ⛔ **Checkpoint resume is documented but missing.** `MainTinyStories.lean` ~:134 and
   `MainTinyGptShakespeare.lean` ~:295 call `heInitParams` unconditionally; `tinygpt_demo_v2.md:253`
   says "resume-by-rerun". A rerun restarts from random init and **overwrites** the checkpoint
   (today's run overwrote the 2026-08-26 one). TinyStories was stopped at ~3200/12000 steps, so
   there is no trained final model. ~6 lines: load-if-exists, keyed on the spec name.
2. **TinyStories never stops on the EOT token.** `preprocess_tinystories.py:100` writes `eot_id`
   to `meta.txt` "so the sampler can stop on it"; `MainTinyStories.lean:215-256` never reads it.
3. **No `PJRT_FFI_RESIDENT=1` / `SHIM_WORKERS` on any LM run line** (docstrings :30-33 / :25-27,
   `run_tinystories_8k.sh`). Off by default (`ffi/pjrt_ffi.c:284`); TinyStories at 2.7 s/step is
   host-bound, the exact profile residency fixes. Add to the run lines and re-measure.
4. **`gradClipNorm` 0 and no dropout** (`MainTinyGptShakespeare.lean:133-141`,
   `MainTinyStories.lean:74-83`) while `tiny` overfits from step ~3500 (`RESULTS.md:65`). Dropout
   is the indicated fix, not more steps. `useEMA := false` — if turned on, the EMA warm-up bug
   (`planning/archive/ema.md:203`) applies.

## §2 Conventions the LM demos lack

| convention | status | effort |
|---|---|---|
| bf16 | ⛔ structurally unavailable: `MlirCodegen.lean` has zero bf16; `cfg.bf16 := true` on a `NetSpec` demo is silently ignored | high (a bf16 arm in the generic walk) |
| verified-render tier | attention/LN hand-rolled; `Proofs/Nets/ViT/*` (`mhsa_has_vjp_mat`, `layerNormVec_has_vjp`, `transformerBlockV_has_vjp_mat`) unused; causal mask has no proven analogue; `content.tex`'s "same VJP machinery" is loose | very high (a project) |
| JAX twin | none (`jax/` has no GPT; `Jax/Codegen.lean` hardcodes NCHW). The gather path's only validation is a loss-sequence tie against the one-hot path — good but self-referential | medium |
| `runs/` dir | ✅ nano since 2026-09-09; none for `tiny` or TinyStories | low + GPU |
| CI | not in any workflow; `scripts/rope_test.sh` is the closest gate | medium |
| data pipeline docs | the `.venv-tokenizers` requirement is recorded only in `demo_xla_port.md:288`; `download_tinystories.sh` guards on file existence so a truncated 1.9 GB download passes forever | low |

Post-v2 additions (flash attention, RoPE, no-pos, the 8K config) exist only as configs
(`MainTinyGptShakespeare.lean:88-107`, `MainTinyStories.lean:55`) with no planning record.

## §3 The figure the entry still lacks

The causal-mask figure landed (`3c8c5dce`). The train-vs-val bits/char curve — Gate A of
`tinygpt_demo_v2.md:92`, and the demo's actual lesson — now HAS its nano data: today's
`train.log` carries loss every 100 steps and val every 500. `tiny` still needs a 10K run (~4 min)
to draw its val line turning up. `scripts/log_to_pgfplots.py` needs ~10 lines of new regex
(`step {n}/{N}: loss=` and `── val @ step {n}: … bits/char`). Companion: the sample-vs-checkpoint
verbatim panel — data already in `blueprint/src/figures/tinygpt/` (five orphaned `.txt` files).

## §4 Order

§1.1 → §1.2 → §1.3 (one commit, re-run nano + tiny with the flags, ~10 min GPU) → §3 curve →
§1.4 as an experiment → §2 rows as separate sessions. Ask before TinyStories' 12K-step run.
