# TODO: the IREE/GPU runs owed after the MNv4 Conv-S → Conv-M conversion (and R50's)

## ✅ BOTH MNv4 TIES RAN AND PASSED, 2026-09-07 — only the R50 item (§3) is left

| tie | result |
|---|---|
| forward (`scripts/mnv4_forward_tie.py`, B = 2, seed 42) | ✅ **max \|Δ\| = 3.770e-06** over the logits, mean 1.269e-06, tol 1e-4. Conv-S's was 1.423e-06 — same order, as this file predicted. |
| gradient, raw (`scripts/grad_tie.py --net mnv4`) | ✅ 232 of 233 slots carry a non-trivial reference gradient and **0 of 232 are worse than 10×** the reference's own relu-discontinuity floor. Worst render error **1.281e+00 vs a 1.468e+00 floor** — inside the reference's own fp32 noise. |
| gradient, `--nokink` | ✅ 0 of 212 live parameters over the threshold. |

▶ **What that buys.** Block ORDER is now pinned by measurement, not by types: a pre/post-DW swap
is invisible to `toSpecs`, to every `#guard`, and (at stride 1) to the types, and these two runs
are the only things in the repo that can see it. The Lean tiers written on top of this render
therefore certify the net the reference computes, not merely a net. The two-conv head and the
strided depthwise placement are covered by the same runs.

⚠ Still NOT established by any of this: an accuracy. Conv-M has no Imagenette run and no verified
ImageNet run; `RESULTS.md`'s 84.58% belongs to the superseded Conv-S table.

### Two setup facts that cost time, recorded so they do not again

⛔⛔ **`iree-run-module --device=local-task` SEGFAULTS on `@mnv4_fwd`** — exit 245 / −11 with
**empty stderr**, no output file and no diagnostic, on both the source-built runtime and the pip
one. `--device=local-sync` runs the IDENTICAL vmfb in **one second**. `scripts/grad_tie.py` has
carried that fallback since `planning/mnv4_verified.md` §3f hit it on `efficientnet_fwd`;
`mnv4_forward_tie.py` did not, so the first Conv-M run looked like a broken artifact. Both scripts
now try `local-task` then `local-sync` and say which one ran.

⭐ **The IREE pairing in §2b is right, but both scripts hard-coded `.venv/bin/iree-compile`.**
`IREE_C` is now `os.environ.get("IREE_COMPILE", …)` in both, so the run is:

```
IREE_COMPILE=/home/skoonce/lean4-mlir/.venv/bin/iree-compile \
IREE_RUN_MODULE=/home/skoonce/lean/klawd_max_power/iree-build/tools/iree-run-module \
JAX_PLATFORMS=cpu .venv/bin/python scripts/mnv4_forward_tie.py
```

⚠ The reference side needs the repo's **pinned** `.venv` (jax 0.11.0); the system python has a
`jax` with no `jaxlib`. ⚠ `jax/.lake/build/generated_mobilenet_v4.py` is CURRENT and Conv-M — this
file's "it does not exist at all" warning was true on 2026-08-14 and is not now; it reports
`Parameters: 8447322`, the same number `VLayer.toSpecs` derives.

---

**Status 2026-08-14, kept as the record of what was owed.** The Conv-M conversion landed
source-side and every gate that does not need a GPU is green. What is owed is the compute: the two
MNv4 ties, plus an R50 run that was already outstanding before this work.

---

## What already passes, so you know where the line is

| gate | result |
|---|---|
| `lake build LeanMlir Proofs` | green |
| `mnv4-fwd-smoke` | green — 233 params, 8,447,322, 77 convs, 30 depthwise, signature ties `VLayer.toSpecs` shape-for-shape |
| `mnv4-train-smoke` | green — 858 inputs / 856 outputs, forward body verbatim, 54 relus paired with 54 `selectPos` masks |
| `scripts/regen_verified_mlir.sh proofs` | green — all 6 mnv4 artifacts re-rendered, prefix audit holds, no other artifact moved |
| `scripts/check_render_coverage.py` | green — 139/157 diffed, 18 known-unguarded |
| the four `#guard` families in `VerifiedNets.lean` | green, including the `bnChannels` stat-alignment gate |

⭐ **The strongest cross-check available without a GPU already ran**: `jax/MainMobilenetV4.lean`'s
`totalParams` reads **8,447,322**, the same number `mobilenetv4Verified`'s `#guard` derives from
`VLayer.toSpecs`. Two independent implementations, same count. That pins the block table and the
two-conv head. It does NOT pin block ORDER or the backward's dispatch — which is exactly what the
ties below are for.

---

## 1. MNv4 forward tie — re-run against the Conv-M reference

```
cd jax && lake exe mobilenet-v4        # regenerates jax/.lake/build/generated_mobilenet_v4.py
cd .. && lake build mnv4-fwd-smoke && .lake/build/bin/mnv4-fwd-smoke
scripts/mnv4_forward_tie.py
```

⚠⚠ **The reference `.py` is currently STALE — it does not exist in `jax/.lake/build/` at all, and
if it is rebuilt from an older checkout it will be the Conv-S table.** `mobilenet-v4` must be
re-run after the 2026-08-14 conversion or the tie compares two different networks and the failure
will look like a block-order bug.

▶ **Why this tie and not the `#guard`s.** A pre/post-DW swap is invisible to everything in
`VerifiedNets.lean`: same `k`, same channels ⇒ same `toSpecs`, and at stride 1 both positions are
shape-preserving so the types pass too. The forward tie is the only thing that can see it. Last
Conv-S value was **1.423e-06**; expect the same order of magnitude.

⚠ The stem's padding is the known first suspect on a miss: XLA `SAME` on a 3×3/s2 at 224 pads
(0,1), not (1,1). `mnv4_forward_tie.py --stem-symmetric` isolates it. That is already handled by
`.convStridedXla` in the render, so a miss there means something regressed.

## 2. MNv4 gradient tie — phase 2's actual gate

```
lake build mnv4-train-smoke && .lake/build/bin/mnv4-train-smoke
scripts/grad_tie.py --net mnv4 --nokink
```

✅ **DONE 2026-09-07.** `scripts/grad_tie.py`'s `NETS["mnv4"]` read `nparams=158, nstats=104` —
Conv-S's numbers, at which this gate could not have lined up its arguments at all. Now **233** and
**154**, counted off `verified_mlir/mnv4_adam_train_step.mlir`'s signature, and the tie passes.

▶ This is the check that the backward differentiates each family as the family it is. The
two-conv head is new code (`MobileNetV4RenderB.lean`: a second `convBackBatched` /
`convWeightGradB` / BN-back triple), and `mnv4-train-smoke` only counts it — it cannot see a wrong
contraction. **Nothing yet has checked the new head's gradient numerically.**

## 2b. ⭐ The IREE binaries are NOT where the scripts look — this pairing works

Both scripts hardcode/default to paths that do not exist. What is actually on the box:

```
compiler: /home/skoonce/lean4-mlir/.venv/bin/iree-compile            (3.12.0rc20260428)
runtime:  /home/skoonce/lean/klawd_max_power/iree-build/tools/iree-run-module
```

Verified together on a trivial stablehlo.add module: compiles and returns `4xf32=11 22 33 44`.

⚠ **Do NOT use `/home/skoonce/src/iree-build/tools/iree-run-module`** with that compiler. It
fails with `import function hal.command_buffer.dispatch signature mismatch between m and source
hal` — a runtime/compiler version skew, which reads like a broken module rather than a bad
pairing. The repo `.venv` has no `iree` package at all (`.venv/bin/iree-compile` does not exist),
so `mnv4_forward_tie.py`'s `IREE_C` and `grad_tie.py` both need pointing at the paths above.

## 3. R50 — outstanding before this work

The R50 IREE run was already owed and is unrelated to the Conv-M conversion. Carried here so the
two are scheduled together rather than rediscovered separately.

---

## Environment notes, so the run does not fail on setup

- ⚠ The repo `.venv` has NO iree at all, and `scripts/mnv4_forward_tie.py`'s default
  `IREE_RUN_MODULE` (`/home/skoonce/lean/claude_max/lean4-jax/.venv/bin/iree-run-module`) does not
  exist either — checked 2026-08-14. Use the pairing in §2b above.
- Use `--iree-cuda-target=sm_86` on RTX 40-series; `sm_89` is broken in IREE 3.11 (issue #21122).
- ⚠ Ask before starting anything long. This box has crashed on long runs before.

## When the ties pass — ✅ all three done 2026-09-07

1. ✅ `scripts/grad_tie.py`'s `nparams`/`nstats` are 233 / 154.
2. ✅ The "has NOT been re-run" sentence is gone from `mnv4ImagenetVerified`'s docstring, replaced
   by the numbers and by what the Imagenette-render tie does and does not carry to the 1000-class
   spec.
3. This file is kept rather than deleted: §3's R50 item is still open, and the two setup findings
   at the top (the `local-task` segfault, the env-overridable compiler path) are worth more here
   than in a commit message.
