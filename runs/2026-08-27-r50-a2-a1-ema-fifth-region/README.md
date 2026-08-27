# 2026-08-27 — RSB-A2/A1 get the model-EMA shadow: the fifth blob region

**What this closes.** `planning/verified_side_quest_counterparts.md` §4a found that the eight A2/A1
renders it had just landed were **not faithful A2/A1**, because two of A2's regularisers had no
expression on the verified path. This run closes the first of the two:

> ⛔ **Model EMA (`emaDecay := 0.9999`) is UNREPRESENTABLE, not merely unrendered.** The EMA shadow
> and the gradient accumulator are the **same fourth region** of `[θ|m|v|·]`, and
> `VerifiedTrain.lean:1156` already throws on the pairing. […] Lifting it means a **fifth region**
> in the driver's pack/unpack and in every optimizer's return list. That is its own project, not a
> flag.

It is now a fifth region. `nRegions` is 3, 4 or 5; `nScalars` is 3, 5 or 7; the layout is
`[θ|m|v|G|E]` and the scalar tail is `lr,bc₁,bc₂,aup,akeep,emad,oemad`.

⚠ **The second delta is NOT closed.** Stochastic depth `dropPath := 0.05` still has no importer on
the residual family — neither `ResNet34RenderB` nor `ResNet50RenderB` imports `DropPath.lean`, and
`r34AdamVariant` has no `drop` marker. So the four artifacts below are A2's graph minus **one**
regulariser rather than minus two. Ghost-BN group 64-vs-128 is unchanged. Quote them the way A3's
deltas are quoted, never as "RSB-A2 reproduced".

---

## The artifacts

| artifact | replicas | baked `%wd` | regions |
|---|---|---|---|
| `resnet50in_emalambaccdp8x64wxclipbce_train_step.mlir` | 4 | 0.02 (A2) | 5 |
| `resnet50in_emalambacc8x64wxclipbce_train_step.mlir` | 1 | 0.02 (A2) | 5 |
| `resnet50in_emalambaccdp8x64wxclipbcewd001_train_step.mlir` | 4 | 0.01 (A1) | 5 |
| `resnet50in_emalambacc8x64wxclipbcewd001_train_step.mlir` | 1 | 0.01 (A1) | 5 |

920 inputs, 918 outputs, 1.6 MB each. The 1-replica peers are not optional: `r50-accum-tie` and
`r50-accum-shard-tie` both compare a DP render against a single-device one.

⚠ **No bf16 twins, and that is a decision.** §4c's rule is that a tier shipping without its
precision peer "reads as a decision and is really an accident of ordering" — but the shadow's
arithmetic is `e' = d·e + (1−d)·θ′` on MASTER weights, which stay f32 in every bf16 render in this
tree. A bf16 twin would differ from the fp32 twin in no line of the EMA block. Render them the day
a bf16 A2 run is scheduled.

---

## `opt_step_tie.log` — the composition, measured

⭐ The new `emalambacc8wxclip` row runs `generated_resnet50_imagenet_a2accum.py`'s **own**
`ema_update` — extracted and `exec`'d, not re-implemented — against the `E` slot the render emits.

```
  ✓ emalambacc8wxclip K=8  ref=r50_a2accum.py  worst rel 1.20e-07
  worst 3: emalambacc8wxclip/E'[0] (shadow) 1.20e-07, …
✓ 7 variants agree with the shipped reference within rtol 2e-06
```

⭐⭐ **The decay scalars come out of the reference too, not out of a formula written here.**
`ema_update` at `(ema := 1, params := 0)` returns `d·1 + (1−d)·0 = d` exactly, so `%emad`/`%oemad`
are tied to the reference as well as the op that consumes them. This repo has shipped the
warmup-corrected decay wrong once already (`planning/ema.md`: a shadow holding 12.8% of its init at
3.1 tau, scoring 0.00% top-1 beside live weights at 70.48%).

## `ema_control.log` — and it goes RED on both defects it claims to catch

```
  rtol gate                                 2e-06
  TEST     E' vs ema_update(e0, theta')     1.20e-07
  CONTROL a  E' vs ema_update(e0, theta)    1.14e-02   <- one-step lag
  CONTROL b  E' read at offset 9 (G's slot) 1.91e+01   <- swapped region order
  CONTROL c  G' read at offset 12 (E's slot)1.01e+00   <- swapped region order
```

⚠⚠ **Control (a) is the one worth reading twice.** A shadow wired to the INCOMING θ instead of the
updated θ′ is off by only **1.1%** — because at lr 0.005 one step barely moves θ. It is four orders
above this gate and would be invisible to a 1e-2 tolerance, and it trains, descends and prints a
normal curve. The reference's ordering (`ema_update` follows the `train_step` call) is the only
thing that says which is right.

⚠ The incoming shadow `e0` is deliberately NOT `theta`, even though the driver seeds it there: with
`e0 = θ` the update reads `d·θ + (1−d)·θ′ ≈ θ′` and the row would pass with its two operands
swapped.

## `arity_check.log` — the driver's blob against the artifact's signature

The shim refuses a count mismatch, but only at run time on a GPU. This reproduces the count
arithmetic against the committed text, in both directions — the driver does `pbuf := out`, so the
previous output IS the next input and the two layouts have to line up either way.

```
✓ 23/23 artifacts agree with the driver's packed layout
```

⭐ All three region counts are exercised: 12 three-region, 7 four-region, 4 five-region.

## `variant_predicates.log`

`✓ 75 variant spellings, 5 axes, no collision`, and its region guard is now a **three-way** sum
with a populated-bucket check beside it. It used to read `nRegions v == (if e || accOn v then 4
else 3)` — true of every spelling that existed, and silently true of a five-region one.

---

## What was NOT run here, and why

* **No GPU step.** Every gate above is CPU: XLA-on-CPU for the optimizer tie, text for the arity
  check, `#guard` for the predicates. A five-region graph has not yet been *loaded* by the trainer,
  and the shim's buffer-count refusal is the check that would say so. `arity_check.py` reproduces
  that arithmetic statically, which is the strongest cheap substitute and not a replacement.
* **No A2 or A1 training run.** These are 300- and 600-epoch schedules.
* **No re-measure of the phase-2 costs.** `runs/2026-08-27-jax-sb-tier-step-probe/` has them and §0
  of the planning doc says not to start there.

## Order of operations, recorded because the reverse is tempting

The refusal in `trainAdamSched` came off **LAST**. While it stood, a wrong render failed loudly at
load; the moment it was gone, a five-region graph packed wrongly trains and reports a number. The
renderer, the driver's pack/unpack, the three-way partition and the tie row all landed first.
