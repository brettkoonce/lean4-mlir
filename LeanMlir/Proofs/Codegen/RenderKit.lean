import LeanMlir.Proofs.Codegen.StableHLO

/-! # The renderers' shared optimizer tail

Every batched train-step renderer ends the same way: fold a per-parameter optimizer step over the
parameter list, so the θ/m/v outputs come out in signature order. This file holds that step once.

| step | used by | ops per parameter |
|---|---|---|
| `adamOne` | MobileNetV2, MobileNetV4, EfficientNet | all-reduce (DP only) + the AdamW triple |
| `rmsOne` | MobileNetV2, EfficientNet | all-reduce + coupled L2 + mean-square + buffer + SGD |
| `adamOneEma` | ViT, ConvNeXt | `adamOne` + the model-EMA shadow, with the clip's `preAvg` |

and the train step's **packed interface** — the one positional contract the driver's blob walks:
`packedTrainSig` (arguments) and `packedTrainRetTys` (results), used by every batched renderer.

`ResNet34RenderB.optOne` is the multi-optimizer step (AdamW / LAMB / heavy-ball / accumulation /
EMA) that ResNet-34 and ResNet-50 fold; it reads the same `PGrad`.

At `replicas ≤ 1` the all-reduce emits nothing and threads the raw gradient, so a single-device
render is unchanged by it. The AdamW triple consumes the averaged gradient as an `.operand`,
exactly as it consumed the raw one, so the `den` side does not shift.
-/

namespace Proofs.StableHLO

/-- A trainable parameter: emitted name (no `%`), gradient SSA name, and shape. The optimizer tail
    is a fold over this list, so the θ/m/v output order cannot drift from the signature order. -/
structure PGrad where
  nm   : String          -- parameter name without `%` (`%{nm}`, `%{nm}m`, `%{nm}v` are the args)
  grad : String          -- SSA name of its un-fused gradient
  ds   : List Nat        -- parameter shape, for the emitted optimizer ops
deriving Inhabited

/-- `(θ', m', v')` for one parameter under **AdamW**: the replica mean of its gradient
    (`prettyAllReduceMean`, `pretty` of the `allReduceMeanF` node) and then the proven
    `adamMNextF`/`adamVNextF`/`adamWParamF` triple (`prettyAdamW`). -/
def adamOne (B : Nat) (replicas : Nat) (g : PGrad) :
    StateM Proofs.StableHLO.EmitS (String × String × String × String) := do
  let (arS, gAvg) ← Proofs.StableHLO.prettyAllReduceMean g.grad g.ds g.nm replicas
  let (c, nT, nM, nV) ← prettyAdamW B g.nm g.ds gAvg
  pure (arS ++ c, nT, nM, nV)

/-- `(θ', b', s')` for one parameter under **RMSProp with momentum** — the `adamOne` peer. Only ONE
    of the four ops is new. Reading the reference
    ([`jax/Jax/Codegen.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/jax/Jax/Codegen.lean), the `.rmsprop` branch) top to bottom:

    | reference line | emitted here |
    |---|---|
    | `grads = g + WD * p` | `momVNextF` at `(μ := wd, v := θ)` — `Proofs.momVNext_as_coupled_l2` |
    | `sq = RHO*s + (1-RHO)*g*g` | **`adamVNextF` at `β₂ := ρ`** — `Proofs.rmsSqNext_eq_adamVNext` |
    | `buf = MOMENTUM*b + g/sqrt(sq+EPS)` | `rmsBufNextF` — the new op, ε INSIDE the root |
    | `params = p - lr*buf` | `sgdParamF` on the buffer's SSA |

    ⚠ **The weight decay is COUPLED and goes FIRST**, so the accumulator sees the decayed gradient.
    Reversing that order — decaying after the accumulator, AdamW-style — is a different optimizer
    and would not show up as an arity or type error anywhere.

    ⚠ **EfficientNet's ε is 1e-3, where MobileNetV2's is 1.0** — the placement's sensitive end: at a
    collapsed mean-square the textbook spelling takes a step **31.6×** larger
    (`Proofs.rmsBufNext_eps_placement_at_zero`). A green MobileNetV2 tie does not license the
    EfficientNet render; `rms-tie efficientnet` is its own gate.

    Slot mapping: the packed `[θ|m|v]` signature is reused verbatim with **`m` carrying the
    momentum buffer and `v` the running mean-square**, the same slot reinterpretation the Nesterov
    render does for its velocity. That is why the driver and the interface do not move. -/
def rmsOne (B : Nat) (replicas : Nat) (g : PGrad) :
    StateM Proofs.StableHLO.EmitS (String × String × String × String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let (arS, gAvg) ← Proofs.StableHLO.prettyAllReduceMean g.grad g.ds g.nm replicas
  let (cW, nW) ← pretty B (.momVNextF s!"%{g.nm}" "%wd" g.ds 0 z (.operand gAvg z))
  let gr : SHlo n := .operand nW z
  let (cS, nS) ← pretty B (.adamVNextF s!"%{g.nm}v" "%rho" "%orho" g.ds 0 z gr)
  let (cB, nB) ← pretty B (.rmsBufNextF s!"%{g.nm}v" s!"%{g.nm}m" "%rho" "%orho" "%mu" "%eps"
                    g.ds 0 0 0 z z gr)
  -- θ' threads b' by SSA NAME, not by re-nesting `rmsBufNextF` inside `sgdParamF`: `pretty` has no
  -- CSE, so re-nesting would emit the whole 13-op buffer block a second time.
  let (cT, nT) ← pretty B (.sgdParamF s!"%{g.nm}" "%lr" g.ds 0 z (.operand nB z))
  pure (arS ++ cW ++ cS ++ cB ++ cT, nT, nB, nS)

/-- `(θ', m', v', e')` for one parameter under **AdamW with the model-EMA shadow** — ViT's and
    ConvNeXt's step. `e'` is `""` at `ema := false`.

    * `wdName` — `"%wd"`, or the zero constant for timm's `no_weight_decay` parameters. The AdamW
      ops take the decay as an OPERAND NAME, so excluding a parameter binds that name to a zero.
    * `preAvg` — the caller has ALREADY averaged (and clipped) this gradient, so the collective is
      not emitted a second time. Under data parallelism the clip must come AFTER the `all_reduce`
      (the reference clips the combined gradient; clipping per replica clips PARTIAL gradients — a
      different function that still trains and descends), and the clip needs every gradient at
      once while this step is per parameter, so at `clip := true` the caller hoists both
      (`planning/archive/grad_clip.md` §4).
    * `ema` — the shadow `e' = d·e + (1−d)·θ'` is `adamMNextF` at `(β₁ := d, m := e, g := θ')`:
      `Proofs.adamMNext` IS the reference's `ema_update` (`jax/Jax/Codegen.lean:2459`), so it needs
      no new op and `adamMNextF_faithful` closes the `den` side by `rfl`. ⚠ It reads `nT`, the
      UPDATED parameter — the shadow averages weights after the optimizer moves them. ⚠
      `%emad`/`%oemad` are function ARGS, not constants: the reference's decay is warmup-corrected,
      `d = min(decay, (1+t)/(10+t))` (`planning/archive/ema.md` §2 — without it a shadow held 12.8%
      of the random init and scored 0.00%). At `ema := false` no `pretty` call happens, so the
      fresh-name counter does not move and the non-EMA renders are unchanged by the flag.

    ⚠ The shadow's SSA name is `%{nm}e` here; `ResNet34RenderB.optOne` uses `%{nm}ema` because at
    suffix `e` the stem BN gamma `%sg` collides with `select_and_scatter`'s block-local `%sge`.
    ViT and ConvNeXt have no max-pool, so `e` is safe for them. -/
def adamOneEma (B : Nat) (replicas : Nat) (g : PGrad)
    (ema : Bool := false) (wdName : String := "%wd") (preAvg : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × String × String × String × String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let replicas := if preAvg then 1 else replicas
  let (arS, gAvg) ← Proofs.StableHLO.prettyAllReduceMean g.grad g.ds g.nm replicas
  let (cA, nT, nM, nV) ← prettyAdamW B g.nm g.ds gAvg wdName
  let (cE, nE) ← if ema then
      pretty B (.adamMNextF s!"%{g.nm}e" "%emad" "%oemad" g.ds 0 z (.operand nT z))
    else pure ("", "")
  pure (arS ++ cA ++ cE, nT, nM, nV, nE)

-- ════════════════════════════════════════════════════════════════
-- § The packed train-step interface `[θ|m|v|G|E] + scalars`
-- ════════════════════════════════════════════════════════════════

/-- The packed train step's **argument list after `%x`**: the parameter regions in the order the
    driver packs them — `θ`, `m`, `v`, then the accumulator `G` (`<p>a`, only under gradient
    accumulation) and the model-EMA shadow `E` (`<p>{emaSuf}`, only under `ema`) — followed by the
    runtime scalars `%lr, %bc1, %bc2`, `%aup, %akeep` (accumulation) and `%emad, %oemad` (EMA).
    `ps` is `(%name, type)` per parameter, in signature order.

    ⚠ `G` precedes `E` and never follows it: `[θ|m|v|G|E]` is the order that leaves both single-axis
    layouts at the index they already occupy. ⚠ `emaSuf` is `"e"` except on the ResNet family, whose
    stem BN gamma `%sg` + `e` would be `%sge` — `select_and_scatter`'s block-local name in the
    max-pool backward — so there it is `"ema"` (`ResNet34RenderB.optOne`). -/
def packedTrainSig (ps : List (String × String)) (acc : Bool := false) (ema : Bool := false)
    (emaSuf : String := "e") : String :=
  let region (suf : String) := String.intercalate ", " (ps.map fun (n, t) => s!"{n}{suf}: {t}")
  let sufs := ["", "m", "v"] ++ (if acc then ["a"] else []) ++ (if ema then [emaSuf] else [])
  String.intercalate ", " (sufs.map region) ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>" ++
    (if acc then ", %aup: tensor<f32>, %akeep: tensor<f32>" else "") ++
    (if ema then ", %emad: tensor<f32>, %oemad: tensor<f32>" else "")

/-- The packed train step's **leading result types**, in `packedTrainSig`'s order: the updated
    regions `θ', m', v'[, G'][, E']`, then `%loss, %bc1, %bc2`, then the handed-back
    accumulation / EMA scalars. `pTy` is the parameter types in signature order. -/
def packedTrainRetTys (pTy : List String) (acc : Bool := false) (ema : Bool := false) :
    List String :=
  pTy ++ pTy ++ pTy ++ (if acc then pTy else []) ++ (if ema then pTy else []) ++
    ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++
    (if acc then ["tensor<f32>", "tensor<f32>"] else []) ++
    (if ema then ["tensor<f32>", "tensor<f32>"] else [])

/-- The stochastic-depth mask arguments `, %dp<i>: tensor<Bxf32>` for the sites `idxs`, in the
    order given — the order the driver's `dropScales` writes them into the blob. Empty when `sd` is
    off, which keeps every non-SD render byte-identical. Each net passes its own site list
    (`vitDropSig`, `cnxDropSig`, `enetDropSig`, `r50DropSig`). -/
def dropMaskSig (B : Nat) (sd : Bool) (idxs : List Nat) : String :=
  if sd then String.join (idxs.map (fun i => s!", {dpName i}: {ty [B]}")) else ""

end Proofs.StableHLO
