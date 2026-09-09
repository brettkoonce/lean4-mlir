# 2026-09-09 — TinyGPT nano, 10K steps on XLA

The book's TinyGPT entry quoted an IREE-era wall clock (11 min on gfx1100) and a
TRAIN loss (1.45 nats/char). This is the re-run on the current stack, one RTX
4060 Ti through XLA/PJRT, and the numbers the entry quotes now.

## The result

**2.279 bits/char held-out** (1.580 nats) after 10K Adam steps; train loss 1.377
nats = 1.99 bits. **173 s wall clock** for the whole `train` command,
compile included (first step at +6 s; 10K steps in 167 s ≈ 17 ms/step with a val
pass every 500). Reproduces the IREE-era 2.27 val / 2.00 train in
`historical/RESULTS.md` to the second decimal.

```
LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=0 \
  .lake/build/bin/tinygpt-shakespeare train nano 10000 32 30
.lake/build/bin/tinygpt-shakespeare sample nano 600 80 0 100 1 "ROMEO:"   # sample_romeo.txt
.lake/build/bin/tinygpt-shakespeare suite nano   # rewrites blueprint/src/figures/tinygpt/prompt_suite_nano.txt
```

Bigram floor re-measured the same session: `bigram-shakespeare train` ends at
2.47 nats = **3.56 bits/char**. Uniform over 65 characters is log2(65) = 6.02.

## Val curve

| step | val nats/char | val bits/char |
|---|---|---|
| 1000 | 1.959 | 2.827 |
| 2000 | 1.787 | 2.578 |
| 3000 | 1.719 | 2.480 |
| 4000 | 1.675 | 2.416 |
| 5000 | 1.645 | 2.374 |
| 6000 | 1.619 | 2.336 |
| 7000 | 1.608 | 2.320 |
| 8000 | 1.593 | 2.299 |
| 9000 | 1.581 | 2.281 |
| 10000 | 1.580 | 2.279 |

Monotone to step 9000, flat after — the cosine floor (10% of peak) does the last
500 steps. No overfit at this size; the 1.2M `tiny` rung is where train and val
part ways (RESULTS.md).

## Provenance

- Binary `.lake/build/bin/tinygpt-shakespeare` built 2026-09-02. Commits since then
  touching the demo, `Train.lean`, `MlirCodegen.lean`, `Types.lean` and `ffi/` are
  `c5b489b4` (IREE gate-helper dedup, not on the XLA train path), `5deb3cd6`
  (planning-path comment moves) and `732c7750` (docstring links) — verified by
  diffing non-comment lines. The PJRT shim was rebuilt from `ffi/pjrt_ffi.c` with the
  README one-liner before the run.
- Engine: XLA/PJRT, cuDNN 9.23.2 from the pinned `.venv` (`jax/requirements-cuda-lock.txt`).
  `LEAN_MLIR_LOWERER` unset = XLA default; nothing IREE was involved.
- Files: `train.log` (timestamped), `timing.txt`, `sample_romeo.txt`, `suite.log`,
  `shim_build.log`. Checkpoint: `.lake/build/tinygpt_shakespeare_params.bin`
  (overwrites the 2026-08-26 one; the demo has no resume path).
