import LeanMlir.Proofs.Foundation.EvenKernelConvBack
import LeanMlir.Proofs.Architectures.ConvNeXtBackCertifiedTie

/-! # ConvNeXt-T's whole-net backward tie — the stage fold, and ⛔ what the tie FOUND

⭐ **Read `EvenKernelConvBack.lean` first; the finding is the deliverable.** This file was started
as the ConvNeXt peer of `r34InputGrad_eq_resnet34_vjp` and `mnv2InputGrad_eq_mobilenetv2_vjp` —
`planning/float_budget_numbers.md` §3.18, done BEFORE the number rather than after it, because
§3.10's tie found r34 reversing the wrong pool and moved a committed number 4×. It paid out the
same way at the first leaf it touched.

⛔⛔ **WHAT THE TIE FOUND: `convFlatBack` IS NOT THE ADJOINT AT AN EVEN KERNEL, and ConvNeXt is
the only net in the repo that has one.** `conv2d` pads by `pH = (kH-1)/2`, so the reversed-kernel
forward conv is the adjoint only when `kH - 1 - pH = pH`, i.e. only for odd `kH`. ConvNeXt's
4×4/s4 patchify stem and its three 2×2/s2 downsamples are the four sites where that fails; at
`kH = 4` the hand-written backward is the adjoint of a conv shifted one pixel. Every other net is
all-odd (R34 7×7/3×3/1×1, MobileNetV2 and EfficientNet-B0 1×1/3×3/5×5), and ViT's 16×16 patch
embed never routes through `conv2d` at all.

⚠ **Nothing trained is affected, and the codegen tier already knew.** `StableHLO.lean`'s
`.convStridedBack` pads ASYMMETRICALLY, `[[kH-1-pH, pH]]`, in both the per-example (:6120) and the
batched (:8248) arms, and its `den` is the certified VJP; the batched comment names the same
quantity — *"the symmetric `[[p,p],[p,p]]` … AGREES at every odd kernel and is WRONG at even ones
(kH=2 ⇒ `[[0,0]]` where the VJP needs `[[1,0]]`)"*. The fix landed on TWO tiers and never reached
the third: the float bridge's `flatConvStride2Back` / `flatConvStride4Back`, which are
`convFlatBack ∘ scatter` at the SYMMETRIC pad. ⭐ That is `imagenet_specs_drift_from_twins` in its
*"a fix landed on one tier and its twin kept the old spelling"* form, for the third time (§3.10's
pool and §3.16's head LayerNorm were the first two), and the first time the stale tier is the one
a committed NUMBER was folded through: §3.16's 5.766·10²⁴⁹ is stale at 4 of its 137 stages.

**What is here.** The repair is `padOdd` (`EvenKernelConvBack.lean`): an even-kernel conv is an
odd-kernel conv on the kernel zero-extended at `(+1,+1)`, which is the emitter's asymmetric pad
written in the vocabulary the float tier already has, so the existing odd-kernel leaf tie does all
the work and no new float machinery is needed. On top of it:

1. `cnxDownChBack_eq_vjp` — the stage-boundary downsample tie, `lnB ∘ flatConvStride2Back
   (padOdd W)` against `(cnxDownChW_has_vjp …).backward`. ⛔ `padOdd` is load-bearing: `p.W` is
   `2×2`. Two existing ties composed.
2. `cnxStageChKBack_eq_vjp` — ⭐ **the depth-`k` stage fold, §3.18's "one real proof".** `HasVJP`
   for `convNextStageChK` is built head-first (block `0` runs first), so its backward composes the
   block backwards in the OPPOSITE order, each at its own saved activation, and the tail's saved
   input is block `0`'s forward OUTPUT. The induction step is one rewrite of the block tie
   (`cnxBlockChBack_eq_vjp`) and one of the inductive hypothesis.
3. `cnxSavedA0 … cnxSavedA10` — `convNextForwardTCh`'s eleven stage inputs, named, so the apex's
   slots can be written without an eleven-deep nest.

⛔ **What is NOT here: the assembly** `convnextInputGrad_eq_convNextForwardTCh_vjp`. It is scoped
in `planning/float_budget_numbers.md`, with the measurements, and it is an ELABORATION problem
rather than a mathematical one — every mathematical piece it needs is above or in
`ConvNeXtBackCertifiedTie.lean`. The short version: `convNextForwardTCh_has_vjp` is a tactic proof,
so its eleven `have`s are `letFun` and its `.backward` will not reduce (whole-net `rfl`: no result
at `maxHeartbeats 8000000`, ~8 min, twice). Rebuilding it as a term-mode chain of twelve top-level
`def`s costs **2.4 s**, and eleven applied-form single-level `rfl`s identify it with the hand chain
in **2m43s** — against an import of `ConvNeXtFullT` alone. Against this file's cone the same
eleven reductions reach **103 GB without terminating**. ⚠ Neither `open Classical` nor the `let`-vs-
`def` binding (which is separately worth 200×: a `let` chain zeta-duplicates to `2^11` copies)
explains that gap; the environment does, and that is the thing to diagnose next.

⭐ **When it lands it will be STRONGER than r34's and MobileNetV2's ties.**
`convNextForwardTCh_has_vjp` is `HasVJP` — everywhere — not the smooth-point `HasVJPAt` those two
are, because GELU, LayerNorm, convolution and the layer scale are all smooth and ConvNeXt has no
kink anywhere. Its only hypotheses are the 23 LayerNorm positivities, so unlike every other
whole-net backward tie in this repo it carries no smoothness side-condition.
⭐⭐ And ConvNeXt already has §3.14's shape check, `convNextForwardTCh_eq_chain` — the `rfl` saying
the chain the apex instantiates IS the committed forward — written before anyone needed it.
⚠ ResNet-34 still has no peer (§4 item 8) and is the net the hole already bit.
-/

namespace Proofs

open Classical

-- ════════════════════════════════════════════════════════════════
-- § The stage-boundary downsample
-- ════════════════════════════════════════════════════════════════

/-- **The downsample backward tie.** `cnxDownBack (padOdd p.W) lnB` — the strided-conv backward at
    the ZERO-EXTENDED kernel, then the channel-LN back at the input resolution — is
    `(cnxDownChW_has_vjp h w p hε).backward v`.

    ⛔ `padOdd` is load-bearing and not cosmetic: `p.W` is `2×2`, so `cnxDownBack p.W` reverses a
    conv shifted one pixel (`EvenKernelConvBack.lean`). This is one of the four sites the
    whole-net tie found. -/
theorem cnxDownChBack_eq_vjp {cin cout h w : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) (v : Vec (cin * (2 * h) * (2 * w))) :
    cnxDownBack (h := h) (w := w) (padOdd p.W)
        (chanLNTensor3Back cin (2 * h) (2 * w) p.ε p.γ v)
      = (cnxDownChW_has_vjp h w p hε).backward v := by
  funext dy
  show (chanLNTensor3Back cin (2 * h) (2 * w) p.ε p.γ v)
      (flatConvStride2Back (h := h) (w := w) (padOdd p.W) dy) = _
  rw [flatConvStride2Back_padOdd_eq_vjp_backward (by norm_num) (by norm_num) p.W p.b
        (chanLNTensor3 cin (2 * h) (2 * w) p.ε p.γ p.β v),
      chanLNTensor3Back_eq_chanLN_vjp (β := p.β) p.ε hε p.γ v]
  rfl

-- ════════════════════════════════════════════════════════════════
-- § ⭐ The depth-`k` STAGE FOLD — the one real proof
-- ════════════════════════════════════════════════════════════════

/-- One channel-LN ConvNeXt block's backward at a saved input `v` — exactly the left-hand side of
    `cnxBlockChBack_eq_vjp`, named so the stage recursion can be written down. -/
noncomputable def cnxBlockChBackAt {c cExp h w kHd kWd : Nat}
    (p : CnxBlockParamsCh c cExp h w kHd kWd) (v : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  Proofs.residual (cnxBlockBodyBack p.Wdw p.Wex p.Wpr
    (chanLNTensor3Back c h w p.εn p.γn (depthwiseFlat (h := h) (w := w) p.Wdw p.bdw v))
    ((layerScale_has_vjp (cnxGlsCh p)).backward
      ((flatConv (h := h) (w := w) p.Wpr p.bpr ∘ gelu (cExp * h * w) ∘
        flatConv (h := h) (w := w) p.Wex p.bex ∘ chanLNTensor3 c h w p.εn p.γn p.βn ∘
        depthwiseFlat (h := h) (w := w) p.Wdw p.bdw) v))
    ((gelu_has_vjp (cExp * h * w)).backward
      ((flatConv (h := h) (w := w) p.Wex p.bex ∘ chanLNTensor3 c h w p.εn p.γn p.βn ∘
        depthwiseFlat (h := h) (w := w) p.Wdw p.bdw) v)))

/-- **The depth-`k` stage backward**, at a saved stage input `v`.

    ⚠ **Head-first, like the forward it reverses.** `convNextStageChK (k+1) ps =
    convNextStageChK k (ps ∘ succ) ∘ cnxBlockChW (ps 0)` applies block `0` FIRST, so the backward
    applies block `0`'s reverse LAST — `cnxBlockChBackAt (ps 0) v ∘ (the rest)`. And the saved
    activation threads forward through the recursion: the tail's saved input is
    `cnxBlockChW (ps 0) v`, block `0`'s OUTPUT. Getting either of those backwards is the
    §3.3-lesson-2 trap (`floatBridgesTo_convNextStageChK` associates the other way), and it is the
    DEFINITION that decides, never the analogy. -/
noncomputable def cnxStageChKBack {c cExp h w kH kW : Nat} :
    (k : Nat) → (ps : Fin k → CnxBlockParamsCh c cExp h w kH kW) → Vec (c * h * w) →
      (Vec (c * h * w) → Vec (c * h * w))
  | 0, _, _ => id
  | k + 1, ps, v =>
      cnxBlockChBackAt (ps 0) v ∘
        cnxStageChKBack k (fun i => ps i.succ) (cnxBlockChW (ps 0) v)

/-- ⭐⭐ **THE STAGE-FOLD TIE.** The hand-composed depth-`k` stage backward IS
    `(convNextStageChK_has_vjp k ps hε).backward`. Induction on `k`: the base case is
    `identity_has_vjp`'s `fun _ dy => dy`, and the step is one rewrite of the block tie
    (`cnxBlockChBack_eq_vjp`) and one of the inductive hypothesis at the shifted saved
    activation. -/
theorem cnxStageChKBack_eq_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd) :
    ∀ (k : Nat) (ps : Fin k → CnxBlockParamsCh c cExp h w kHd kWd)
      (hε : ∀ i, 0 < (ps i).εn) (v : Vec (c * h * w)),
      cnxStageChKBack k ps v = (convNextStageChK_has_vjp k ps hε).backward v
  | 0, _, _, _ => rfl
  | k + 1, ps, hε, v => by
      show cnxBlockChBackAt (ps 0) v ∘
        cnxStageChKBack k (fun i => ps i.succ) (cnxBlockChW (ps 0) v) = _
      rw [cnxStageChKBack_eq_vjp hkHd hkWd k (fun i => ps i.succ) (fun i => hε i.succ)
            (cnxBlockChW (ps 0) v)]
      show Proofs.residual _ ∘ _ = _
      rw [cnxBlockChBack_eq_vjp hkHd hkWd (ps 0) (hε 0) v]
      rfl

/-- `rowLNVecFlat_has_vjp_backward_eq` at the FUNCTION level — the direction and shape a whole-net
    `rw` needs. The committed lemma is pointwise in `dy` and oriented certified-to-hand; a chain
    rewrite wants hand-to-certified with `dy` abstracted. -/
theorem rowLNVecFlat_has_vjp_backward_eq_fun {s c : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (X : Vec (s * c)) :
    rowLNVecFlatBack s c ε γ X = (rowLNVecFlat_has_vjp s c ε γ β hε).backward X := by
  funext dy
  rw [rowLNVecFlat_has_vjp_backward_eq (β := β) ε hε γ X dy]

-- ════════════════════════════════════════════════════════════════
-- § The saved activations, named
-- ════════════════════════════════════════════════════════════════

/-! ⭐ `convNextForwardTCh`'s eleven stage inputs, named. A backward's slots are indexed by the
activation each reverse was saved at, and spelling those inline gives a statement whose eleventh
slot is an eleven-deep nest — mnv2's tie is already at the readable limit with six. Naming them
also makes the THREAD visible, which is the thing a reader has to check: `cnxSaved_i` is the input
to stage `i`, hence the saved activation of stage `i`'s backward. They are plain `def`s, so the
whole-net `rfl` unfolds them. -/

/-- The stem conv's output — the stem LayerNorm's saved input. -/
noncomputable def cnxSavedA0 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) : Vec (96 * 56 * 56) :=
  flatConvStride4 (h := 56) (w := 56) w.sW w.sb x

/-- Stage 1's saved input. -/
noncomputable def cnxSavedA1 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) : Vec (96 * 56 * 56) :=
  chanLNTensor3 96 56 56 w.sε w.sγ w.sβ (cnxSavedA0 w x)

/-- Downsample 1's saved input. -/
noncomputable def cnxSavedA2 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) : Vec (96 * 56 * 56) :=
  convNextStageChK 3 w.s1 (cnxSavedA1 w x)

/-- Stage 2's saved input. -/
noncomputable def cnxSavedA3 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) :
    Vec (192 * 28 * 28) := cnxDownChW 28 28 w.d1 (cnxSavedA2 w x)

/-- Downsample 2's saved input. -/
noncomputable def cnxSavedA4 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) :
    Vec (192 * 28 * 28) := convNextStageChK 3 w.s2 (cnxSavedA3 w x)

/-- Stage 3's saved input. -/
noncomputable def cnxSavedA5 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) :
    Vec (384 * 14 * 14) := cnxDownChW 14 14 w.d2 (cnxSavedA4 w x)

/-- Downsample 3's saved input. -/
noncomputable def cnxSavedA6 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) :
    Vec (384 * 14 * 14) := convNextStageChK 9 w.s3 (cnxSavedA5 w x)

/-- Stage 4's saved input. -/
noncomputable def cnxSavedA7 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) :
    Vec (768 * 7 * 7) := cnxDownChW 7 7 w.d3 (cnxSavedA6 w x)

/-- GAP's saved input. -/
noncomputable def cnxSavedA8 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) :
    Vec (768 * 7 * 7) := convNextStageChK 3 w.s4 (cnxSavedA7 w x)

/-- The head LayerNorm's saved input. -/
noncomputable def cnxSavedA9 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) : Vec 768 :=
  globalAvgPoolFlat 768 7 7 (cnxSavedA8 w x)

/-- The classifier's saved input. -/
noncomputable def cnxSavedA10 (w : CnxTWeightsCh) (x : Vec (3 * 224 * 224)) : Vec 768 :=
  rowLNVecFlat 1 768 w.hε w.hγ w.hβ (cnxSavedA9 w x)

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE APEX
-- ════════════════════════════════════════════════════════════════

end Proofs
