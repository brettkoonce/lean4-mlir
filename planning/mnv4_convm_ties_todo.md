# TODO: the IREE/GPU runs owed after the MNv4 Conv-S → Conv-M conversion (and R50's)

**Status 2026-08-14.** The Conv-M conversion landed source-side and every gate that does not
need a GPU is green. What is owed is the compute: the two MNv4 ties, plus an R50 run that was
already outstanding before this work.

⚠ **Nothing in the repo currently claims these ties pass.** `VerifiedNets.lean`'s
`mnv4ImagenetVerified` docstring says the forward tie has not been re-run, and this file is what
it points at. Do not quote a tie number until one of the commands below has actually produced it.

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

⚠⚠ **`scripts/grad_tie.py`'s `NETS["mnv4"]` still says `nparams=158, nstats=104`.** Those are the
Conv-S numbers. They must become **233** and **154** or the tie will not even line up its
arguments. This edit is NOT yet made — it is deliberately left here rather than made blind,
because the right time to change it is when someone can run the tie and see it pass.

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

## When the ties pass

1. Update `scripts/grad_tie.py`'s `nparams`/`nstats` (above) as part of the same change.
2. Drop the "has NOT been re-run" sentence from `mnv4ImagenetVerified`'s docstring.
3. Delete this file, or cut it down to the R50 item if that is still open.
