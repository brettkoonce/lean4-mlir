import LeanMlir.Proofs.Foundation.EvenKernelConvBack
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackCertifiedTie

/-! # ConvNeXt-T's whole-net backward tie — the stage fold, and ⛔ what the tie FOUND

⭐ **Read `EvenKernelConvBack.lean` first; the finding is the deliverable.** This file was started
as the ConvNeXt peer of `r34InputGrad_eq_resnet34_vjp` and `mnv2InputGrad_eq_mobilenetv2_vjp` —
`planning/archive/float_budget_numbers_log.md` §3.18, done BEFORE the number rather than after it, because
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
the third: `BackwardMaps.lean`'s `flatConvStride2Back` / `flatConvStride4Back`, which are
`convFlatBack ∘ scatter` at the SYMMETRIC pad. ⭐ That is `imagenet_specs_drift_from_twins` in its
*"a fix landed on one tier and its twin kept the old spelling"* form, for the third time (§3.10's
pool and §3.16's head LayerNorm were the first two).

**What is here.** The repair is `padOdd` (`EvenKernelConvBack.lean`): an even-kernel conv is an
odd-kernel conv on the kernel zero-extended at `(+1,+1)`, which is the emitter's asymmetric pad
written in the vocabulary `BackwardMaps.lean` already has, so the existing odd-kernel leaf tie does
all the work and no new conv machinery is needed. On top of it:

1. `cnxDownChBack_eq_vjp` — the stage-boundary downsample tie, `lnB ∘ flatConvStride2Back
   (padOdd W)` against `(cnxDownChW_has_vjp …).backward`. ⛔ `padOdd` is load-bearing: `p.W` is
   `2×2`. Two existing ties composed.
2. `cnxStageChKBack_eq_vjp` — ⭐ **the depth-`k` stage fold, §3.18's "one real proof".** `HasVJP`
   for `convNextStageChK` is built head-first (block `0` runs first), so its backward composes the
   block backwards in the OPPOSITE order, each at its own saved activation, and the tail's saved
   input is block `0`'s forward OUTPUT. The induction step is one rewrite of the block tie
   (`cnxBlockChBack_eq_vjp`) and one of the inductive hypothesis.
3. `cnxSavedA0 … cnxSavedA10` — `convNextForwardTCh`'s eleven stage inputs, named as FUNCTIONS, so
   that the same twelve constants are both the activations the backward's slots are saved at and
   the `f` argument of each `vjp_comp`.
4. ⭐⭐ **`convnextInputGrad_eq_convNextForwardTCh_vjp` — THE APEX.** `convnextInputGrad`, with
   every slot pinned to the certified per-op backward at its own saved activation, IS
   `(convNextForwardTCh_has_vjp …).backward x`. The ConvNeXt peer of
   `r34InputGrad_eq_resnet34_vjp` and `mnv2InputGrad_eq_mobilenetv2_vjp`, and **stronger than
   both**: `convNextForwardTCh_has_vjp` is `HasVJP` — everywhere — not the smooth-point `HasVJPAt`
   those two are, because GELU, LayerNorm, convolution and the layer scale are all smooth and
   ConvNeXt has no kink anywhere. Its only hypotheses are the 23 LayerNorm positivities, so unlike
   every other whole-net backward tie in this repo it carries no smoothness side-condition.
   ⭐⭐ And ConvNeXt has its shape check too
   (`planning/archive/float_budget_numbers_log.md` §3.14): `convNextForwardTCh_eq_chain`, the `rfl` saying the
   chain the apex instantiates IS the committed forward, written before anyone needed it.

⭐⭐ **WHAT THE ASSEMBLY COST, AND IT IS ONE RULE:** *never hand the unifier two spellings of the
same thing in an APPLIED position.* Every expensive step in this file was an instance, and each is
free once the spelling is normalised at a definition:

* `cnxDownChW h w p` is declared over `Vec (cin * (2 * h) * (2 * w))` where the chain spells
  `Vec (96 * 56 * 56)`. Both are closed terms and equal, and the unifier still descends into the
  semantics of both sides rather than reducing `2 * 28` — the diagnostics reach
  `conv2d_input_grad_formula`, `Finset.sum`, `Mat.unflatten`, `cnxBlockChW`. Measured one link at a
  time, the three downsamples cost 3 s, 15 s and then do not finish, while every stage, LayerNorm,
  GAP and dense link is free. `cnxDn1`/`cnxDn2`/`cnxDn3` below are the whole fix: a one-line `def`
  with the type ascribed in the chain's spelling, plus `Differentiable`/`HasVJP` peers ascribed the
  same way. Same for `cnxLNh` at `Vec 768` against `rowLNVecFlat 1 768`'s `Vec (1 * 768)`.
* A leaf tie goes the OTHER way — state it in the LEMMA's spelling, not the chain's
  (`cnxLNhBack_eq_vjp` takes `v : Vec (1 * 768)`); at `Vec 768` the same statement does not finish.
* The saved activations are functions, so each `cnxTk` is a one-step iota with syntactically
  identical sides. Stated the other way — the chain's own `f x` against an applied
  `cnxSavedA k w x` — identifying the two costs 2 s at depth one and does not finish at depth two.
* The closing step is `simp only [Function.comp_apply, cnxV0]`, not `rfl`: after the eleven peels
  the two sides differ only by `Function.comp`, and `rfl` will not take that route.

⚠ `planning/archive/float_budget_numbers_log.md` §3.7(d) records this trap in its other guise, where the
computed dimension meets a metavariable (`2 * ?h = 112`) and the unification is higher-order; there
the fix is to pin the implicit. Here `h` is given explicitly and it still costs — two CLOSED
spellings of one numeral are enough. ⛔ And it is invisible in an unapplied position:
`convNextForwardTCh_vjp_chain`'s ascription compares the whole twelve-factor composition against
the committed one and is free, because no `x` is in sight to evaluate.

⛔ **The other half of the shape is the term-mode chain, and it is not a preference.**
`convNextForwardTCh_has_vjp` is a tactic proof, so its eleven `have`s are `letFun` and its
`.backward` does not reduce; the whole-net `rfl` against it returned no result at
`maxHeartbeats 8000000`, twice, ~8 min each. `HasVJP.backward_unique` transfers through `.correct`
instead, which costs nothing, and the term-mode peer must be top-level `def`s rather than a `let`
chain — a `let` used twice per level zeta-expands to `2^11` copies of the prefix.

⚠ ResNet-34's shape check is `resnet34Forward_full_pc_eq_chain` (`ResNet34BackCertifiedTie.lean`),
and it is the net the hole first bit.
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
    §3.3-lesson-2 trap (the stage fold once associated the other way), and it is the
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
-- § ⭐⭐ THE VJP CHAIN, at normalised dimension spellings
-- ════════════════════════════════════════════════════════════════

/-! ⭐⭐ **Every stage whose declared type carries a COMPUTED dimension gets a wrapper here, and
that is the whole reason this section closes in seconds.** `cnxDownChW h w p` is declared over
`Vec (cin * (2 * h) * (2 * w))`; the chain spells the same type `Vec (96 * 56 * 56)`. Both are
closed terms and they are equal, but in an APPLIED position the unifier does not reduce `2 * 28`
to `56` — it descends into the semantics of both sides instead, and the diagnostics name what it
reaches: `conv2d_input_grad_formula`, `Finset.sum`, `Mat.unflatten`, `cnxBlockChW`. Measured, one
tie at a time: the three downsample links cost 3 s, 15 s and then do not finish, while every
stage, LayerNorm, GAP and dense link is free. With the wrappers below — a one-line `def` per
offending stage, its type ascribed in the chain's spelling, and its `Differentiable`/`HasVJP`
peers ascribed the same way — the twelve chain defs and all eleven links together cost
**2.9 s**, of which the links are ~0.3 s.

⚠ `planning/archive/float_budget_numbers_log.md` §3.7(d) records this trap in its other guise, where the
computed dimension meets a metavariable (`2 * ?h = 112`) and the unification is higher-order.
There the fix is to pin the implicit. Here `h` is already given explicitly and it still costs:
two CLOSED spellings of one numeral are enough. ⛔ And it is invisible in an unapplied position —
`convNextForwardTCh_vjp_chain`'s ascription below compares the whole twelve-factor composition
against the committed one and is free, because no `x` is in sight to evaluate. -/

/-- Downsample 1 at the chain's dimension spelling. -/
private noncomputable def cnxDn1 (w : CnxTWeightsCh) : Vec (96 * 56 * 56) → Vec (192 * 28 * 28) :=
  cnxDownChW 28 28 w.d1
private theorem cnxDn1Diff (w : CnxTWeightsCh) (hd1 : 0 < w.d1.ε) :
    Differentiable ℝ (cnxDn1 w) := cnxDownChW_diff 28 28 w.d1 hd1
private noncomputable def cnxDn1Vjp (w : CnxTWeightsCh) (hd1 : 0 < w.d1.ε) :
    HasVJP (cnxDn1 w) := cnxDownChW_has_vjp 28 28 w.d1 hd1

/-- Downsample 2 at the chain's dimension spelling. -/
private noncomputable def cnxDn2 (w : CnxTWeightsCh) : Vec (192 * 28 * 28) → Vec (384 * 14 * 14) :=
  cnxDownChW 14 14 w.d2
private theorem cnxDn2Diff (w : CnxTWeightsCh) (hd2 : 0 < w.d2.ε) :
    Differentiable ℝ (cnxDn2 w) := cnxDownChW_diff 14 14 w.d2 hd2
private noncomputable def cnxDn2Vjp (w : CnxTWeightsCh) (hd2 : 0 < w.d2.ε) :
    HasVJP (cnxDn2 w) := cnxDownChW_has_vjp 14 14 w.d2 hd2

/-- Downsample 3 at the chain's dimension spelling. -/
private noncomputable def cnxDn3 (w : CnxTWeightsCh) : Vec (384 * 14 * 14) → Vec (768 * 7 * 7) :=
  cnxDownChW 7 7 w.d3
private theorem cnxDn3Diff (w : CnxTWeightsCh) (hd3 : 0 < w.d3.ε) :
    Differentiable ℝ (cnxDn3 w) := cnxDownChW_diff 7 7 w.d3 hd3
private noncomputable def cnxDn3Vjp (w : CnxTWeightsCh) (hd3 : 0 < w.d3.ε) :
    HasVJP (cnxDn3 w) := cnxDownChW_has_vjp 7 7 w.d3 hd3

/-- The head LayerNorm at `Vec 768`, not `Vec (1 * 768)` — the same normalisation, at the one
    site where the computed dimension is a `1 *` rather than a `2 *`. -/
private noncomputable def cnxLNh (w : CnxTWeightsCh) : Vec 768 → Vec 768 :=
  rowLNVecFlat 1 768 w.hε w.hγ w.hβ
private theorem cnxLNhDiff (w : CnxTWeightsCh) (hhε : 0 < w.hε) :
    Differentiable ℝ (cnxLNh w) := rowLNVecFlat_diff 1 768 w.hε w.hγ w.hβ hhε
private noncomputable def cnxLNhVjp (w : CnxTWeightsCh) (hhε : 0 < w.hε) :
    HasVJP (cnxLNh w) := rowLNVecFlat_has_vjp 1 768 w.hε w.hγ w.hβ hhε


-- ── the forward prefixes: `cnxSavedA k w x` is stage `k`'s saved input ──

/-! ⭐ `convNextForwardTCh`'s eleven stage inputs, named — and named as FUNCTIONS, so that the
same twelve constants are both the saved activations the backward's slots are indexed by and the
`f` argument of each `vjp_comp`. That is what makes every link below a one-step iota with
syntactically identical sides: the alternative — an applied `cnxSavedA k w x` on one side and the
chain's own `f x` on the other — is defeq, and identifying the two costs 2 s at depth one and
does not finish at depth two. -/

/-- The stem conv's output — the stem LayerNorm's saved input. -/
noncomputable def cnxSavedA0 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (96 * 56 * 56) :=
    flatConvStride4 (h := 56) (w := 56) w.sW w.sb

private theorem cnxD0 (w : CnxTWeightsCh) : Differentiable ℝ (cnxSavedA0 w) :=
    flatConvStride4_differentiable (h := 56) (w := 56) w.sW w.sb
private noncomputable def cnxV0 (w : CnxTWeightsCh) : HasVJP (cnxSavedA0 w) :=
    flatConvStride4_has_vjp (h := 56) (w := 56) w.sW w.sb

/-- Stage 1's saved input. -/
noncomputable def cnxSavedA1 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (96 * 56 * 56) :=
  chanLNTensor3 96 56 56 w.sε w.sγ w.sβ ∘ cnxSavedA0 w

private theorem cnxD1 (w : CnxTWeightsCh) (hsε : 0 < w.sε) : Differentiable ℝ (cnxSavedA1 w) :=
  (chanLNTensor3_diff 96 56 56 w.sε w.sγ w.sβ hsε).comp (cnxD0 w)
private noncomputable def cnxV1 (w : CnxTWeightsCh) (hsε : 0 < w.sε) : HasVJP (cnxSavedA1 w) :=
  vjp_comp (cnxSavedA0 w) _ (cnxD0 w) (chanLNTensor3_diff 96 56 56 w.sε w.sγ w.sβ hsε) (cnxV0 w)
    (chanLNTensor3_has_vjp 96 56 56 w.sε w.sγ w.sβ hsε)

/-- Downsample 1's saved input. -/
noncomputable def cnxSavedA2 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (96 * 56 * 56) :=
  convNextStageChK 3 w.s1 ∘ cnxSavedA1 w

private theorem cnxD2 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn) :
    Differentiable ℝ (cnxSavedA2 w) :=
  (convNextStageChK_diff 3 w.s1 h1).comp (cnxD1 w hsε)
private noncomputable def cnxV2 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn) :
    HasVJP (cnxSavedA2 w) :=
  vjp_comp (cnxSavedA1 w) _ (cnxD1 w hsε) (convNextStageChK_diff 3 w.s1 h1) (cnxV1 w hsε)
    (convNextStageChK_has_vjp 3 w.s1 h1)

/-- Stage 2's saved input. -/
noncomputable def cnxSavedA3 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (192 * 28 * 28) :=
  cnxDn1 w ∘ cnxSavedA2 w

private theorem cnxD3 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) : Differentiable ℝ (cnxSavedA3 w) :=
  (cnxDn1Diff w hd1).comp (cnxD2 w hsε h1)
private noncomputable def cnxV3 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) : HasVJP (cnxSavedA3 w) :=
  vjp_comp (cnxSavedA2 w) _ (cnxD2 w hsε h1) (cnxDn1Diff w hd1) (cnxV2 w hsε h1) (cnxDn1Vjp w hd1)

/-- Downsample 2's saved input. -/
noncomputable def cnxSavedA4 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (192 * 28 * 28) :=
  convNextStageChK 3 w.s2 ∘ cnxSavedA3 w

private theorem cnxD4 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) : Differentiable ℝ (cnxSavedA4 w) :=
  (convNextStageChK_diff 3 w.s2 h2).comp (cnxD3 w hsε h1 hd1)
private noncomputable def cnxV4 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) : HasVJP (cnxSavedA4 w) :=
  vjp_comp (cnxSavedA3 w) _ (cnxD3 w hsε h1 hd1) (convNextStageChK_diff 3 w.s2 h2)
    (cnxV3 w hsε h1 hd1) (convNextStageChK_has_vjp 3 w.s2 h2)

/-- Stage 3's saved input. -/
noncomputable def cnxSavedA5 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (384 * 14 * 14) :=
  cnxDn2 w ∘ cnxSavedA4 w

private theorem cnxD5 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) : Differentiable ℝ
    (cnxSavedA5 w) :=
  (cnxDn2Diff w hd2).comp (cnxD4 w hsε h1 hd1 h2)
private noncomputable def cnxV5 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) : HasVJP (cnxSavedA5 w) :=
  vjp_comp (cnxSavedA4 w) _ (cnxD4 w hsε h1 hd1 h2) (cnxDn2Diff w hd2) (cnxV4 w hsε h1 hd1 h2)
    (cnxDn2Vjp w hd2)

/-- Downsample 3's saved input. -/
noncomputable def cnxSavedA6 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (384 * 14 * 14) :=
  convNextStageChK 9 w.s3 ∘ cnxSavedA5 w

private theorem cnxD6 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn) :
    Differentiable ℝ (cnxSavedA6 w) :=
  (convNextStageChK_diff 9 w.s3 h3).comp (cnxD5 w hsε h1 hd1 h2 hd2)
private noncomputable def cnxV6 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn) :
    HasVJP (cnxSavedA6 w) :=
  vjp_comp (cnxSavedA5 w) _ (cnxD5 w hsε h1 hd1 h2 hd2) (convNextStageChK_diff 9 w.s3 h3)
    (cnxV5 w hsε h1 hd1 h2 hd2) (convNextStageChK_has_vjp 9 w.s3 h3)

/-- Stage 4's saved input. -/
noncomputable def cnxSavedA7 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (768 * 7 * 7) :=
  cnxDn3 w ∘ cnxSavedA6 w

private theorem cnxD7 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) : Differentiable ℝ (cnxSavedA7 w) :=
  (cnxDn3Diff w hd3).comp (cnxD6 w hsε h1 hd1 h2 hd2 h3)
private noncomputable def cnxV7 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) : HasVJP (cnxSavedA7 w) :=
  vjp_comp (cnxSavedA6 w) _ (cnxD6 w hsε h1 hd1 h2 hd2 h3) (cnxDn3Diff w hd3)
    (cnxV6 w hsε h1 hd1 h2 hd2 h3) (cnxDn3Vjp w hd3)

/-- GAP's saved input. -/
noncomputable def cnxSavedA8 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (768 * 7 * 7) :=
  convNextStageChK 3 w.s4 ∘ cnxSavedA7 w

private theorem cnxD8 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) : Differentiable ℝ (cnxSavedA8 w) :=
  (convNextStageChK_diff 3 w.s4 h4).comp (cnxD7 w hsε h1 hd1 h2 hd2 h3 hd3)
private noncomputable def cnxV8 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) : HasVJP (cnxSavedA8 w) :=
  vjp_comp (cnxSavedA7 w) _ (cnxD7 w hsε h1 hd1 h2 hd2 h3 hd3) (convNextStageChK_diff 3 w.s4 h4)
    (cnxV7 w hsε h1 hd1 h2 hd2 h3 hd3) (convNextStageChK_has_vjp 3 w.s4 h4)

/-- The head LayerNorm's saved input. -/
noncomputable def cnxSavedA9 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (768) :=
  globalAvgPoolFlat 768 7 7 ∘ cnxSavedA8 w

private theorem cnxD9 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) : Differentiable ℝ (cnxSavedA9 w) :=
  (globalAvgPoolFlat_differentiable 768 7 7).comp (cnxD8 w hsε h1 hd1 h2 hd2 h3 hd3 h4)
private noncomputable def cnxV9 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) : HasVJP (cnxSavedA9 w) :=
  vjp_comp (cnxSavedA8 w) _ (cnxD8 w hsε h1 hd1 h2 hd2 h3 hd3 h4)
    (globalAvgPoolFlat_differentiable 768 7 7) (cnxV8 w hsε h1 hd1 h2 hd2 h3 hd3 h4)
    (globalAvgPoolFlat_has_vjp 768 7 7)

/-- The classifier's saved input. -/
noncomputable def cnxSavedA10 (w : CnxTWeightsCh) : Vec (3 * 224 * 224) → Vec (768) :=
  cnxLNh w ∘ cnxSavedA9 w

private theorem cnxD10 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) : Differentiable ℝ
    (cnxSavedA10 w) :=
  (cnxLNhDiff w hhε).comp (cnxD9 w hsε h1 hd1 h2 hd2 h3 hd3 h4)
private noncomputable def cnxV10 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) : HasVJP (cnxSavedA10 w) :=
  vjp_comp (cnxSavedA9 w) _ (cnxD9 w hsε h1 hd1 h2 hd2 h3 hd3 h4) (cnxLNhDiff w hhε)
    (cnxV9 w hsε h1 hd1 h2 hd2 h3 hd3 h4) (cnxLNhVjp w hhε)

private theorem cnxD11 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) :
    Differentiable ℝ (dense w.Wd w.bd ∘ cnxSavedA10 w) :=
  (dense_differentiable w.Wd w.bd).comp (cnxD10 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
private noncomputable def cnxV11 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) :
    HasVJP (dense w.Wd w.bd ∘ cnxSavedA10 w) :=
  vjp_comp (cnxSavedA10 w) _ (cnxD10 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
    (dense_differentiable w.Wd w.bd) (cnxV10 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
    (dense_has_vjp w.Wd w.bd)

-- ── the eleven single-level reductions ──

private theorem cnxT1 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (x : Vec (3 * 224 * 224))
    (dy : Vec (96 * 56 * 56)) :
    (cnxV1 w hsε).backward x dy
      = (cnxV0 w).backward x
        ((chanLNTensor3_has_vjp 96 56 56 w.sε w.sγ w.sβ hsε).backward (cnxSavedA0 w x) dy) := rfl

private theorem cnxT2 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (x : Vec (3 * 224 * 224))
    (dy : Vec (96 * 56 * 56)) :
    (cnxV2 w hsε h1).backward x dy
      = (cnxV1 w hsε).backward x
        ((convNextStageChK_has_vjp 3 w.s1 h1).backward (cnxSavedA1 w x) dy) := rfl

private theorem cnxT3 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (x : Vec (3 * 224 * 224))
    (dy : Vec (192 * 28 * 28)) :
    (cnxV3 w hsε h1 hd1).backward x dy
      = (cnxV2 w hsε h1).backward x ((cnxDn1Vjp w hd1).backward (cnxSavedA2 w x) dy) := rfl

private theorem cnxT4 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (x : Vec (3 * 224 * 224))
    (dy : Vec (192 * 28 * 28)) :
    (cnxV4 w hsε h1 hd1 h2).backward x dy
      = (cnxV3 w hsε h1 hd1).backward x
        ((convNextStageChK_has_vjp 3 w.s2 h2).backward (cnxSavedA3 w x) dy) := rfl

private theorem cnxT5 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (x : Vec (3 * 224 * 224))
    (dy : Vec (384 * 14 * 14)) :
    (cnxV5 w hsε h1 hd1 h2 hd2).backward x dy
      = (cnxV4 w hsε h1 hd1 h2).backward x ((cnxDn2Vjp w hd2).backward (cnxSavedA4 w x) dy) := rfl

private theorem cnxT6 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (x : Vec (3 * 224 * 224))
    (dy : Vec (384 * 14 * 14)) :
    (cnxV6 w hsε h1 hd1 h2 hd2 h3).backward x dy
      = (cnxV5 w hsε h1 hd1 h2 hd2).backward x
        ((convNextStageChK_has_vjp 9 w.s3 h3).backward (cnxSavedA5 w x) dy) := rfl

private theorem cnxT7 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (x : Vec (3 * 224 * 224))
    (dy : Vec (768 * 7 * 7)) :
    (cnxV7 w hsε h1 hd1 h2 hd2 h3 hd3).backward x dy
      = (cnxV6 w hsε h1 hd1 h2 hd2 h3).backward x ((cnxDn3Vjp w hd3).backward (cnxSavedA6 w x) dy)
        := rfl

private theorem cnxT8 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (x : Vec (3 * 224 * 224))
    (dy : Vec (768 * 7 * 7)) :
    (cnxV8 w hsε h1 hd1 h2 hd2 h3 hd3 h4).backward x dy
      = (cnxV7 w hsε h1 hd1 h2 hd2 h3 hd3).backward x
        ((convNextStageChK_has_vjp 3 w.s4 h4).backward (cnxSavedA7 w x) dy) := rfl

private theorem cnxT9 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (x : Vec (3 * 224 * 224))
    (dy : Vec (768)) :
    (cnxV9 w hsε h1 hd1 h2 hd2 h3 hd3 h4).backward x dy
      = (cnxV8 w hsε h1 hd1 h2 hd2 h3 hd3 h4).backward x
        ((globalAvgPoolFlat_has_vjp 768 7 7).backward (cnxSavedA8 w x) dy) := rfl

private theorem cnxT10 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) (x : Vec (3 * 224 * 224))
    (dy : Vec (768)) :
    (cnxV10 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x dy
      = (cnxV9 w hsε h1 hd1 h2 hd2 h3 hd3 h4).backward x
        ((cnxLNhVjp w hhε).backward (cnxSavedA9 w x) dy) := rfl

private theorem cnxT11 (w : CnxTWeightsCh) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) (x : Vec (3 * 224 * 224))
    (dy : Vec (10)) :
    (cnxV11 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x dy
      = (cnxV10 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x
        ((dense_has_vjp w.Wd w.bd).backward (cnxSavedA10 w x) dy) := rfl
set_option maxRecDepth 100000

-- ── the three normalised leaf ties the wrappers need ──

private theorem cnxDn1Back_eq_vjp (w : CnxTWeightsCh) (hd1 : 0 < w.d1.ε) (v : Vec (96 * 56 * 56)) :
    cnxDownBack (h := 28) (w := 28) (padOdd w.d1.W)
        (chanLNTensor3Back 96 56 56 w.d1.ε w.d1.γ v)
      = (cnxDn1Vjp w hd1).backward v :=
  cnxDownChBack_eq_vjp (h := 28) (w := 28) w.d1 hd1 v

private theorem cnxDn2Back_eq_vjp (w : CnxTWeightsCh) (hd2 : 0 < w.d2.ε) (v : Vec (192 * 28 * 28)) :
    cnxDownBack (h := 14) (w := 14) (padOdd w.d2.W)
        (chanLNTensor3Back 192 28 28 w.d2.ε w.d2.γ v)
      = (cnxDn2Vjp w hd2).backward v :=
  cnxDownChBack_eq_vjp (h := 14) (w := 14) w.d2 hd2 v

private theorem cnxDn3Back_eq_vjp (w : CnxTWeightsCh) (hd3 : 0 < w.d3.ε) (v : Vec (384 * 14 * 14)) :
    cnxDownBack (h := 7) (w := 7) (padOdd w.d3.W)
        (chanLNTensor3Back 384 14 14 w.d3.ε w.d3.γ v)
      = (cnxDn3Vjp w hd3).backward v :=
  cnxDownChBack_eq_vjp (h := 7) (w := 7) w.d3 hd3 v

private theorem cnxLNhBack_eq_vjp (w : CnxTWeightsCh) (hhε : 0 < w.hε) (v : Vec (1 * 768)) :
    rowLNVecFlatBack 1 768 w.hε w.hγ v = (cnxLNhVjp w hhε).backward v :=
  rowLNVecFlat_has_vjp_backward_eq_fun (β := w.hβ) w.hε hhε w.hγ v

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE APEX
-- ════════════════════════════════════════════════════════════════

/-- **`convNextForwardTCh_has_vjp` as a TERM-mode `vjp_comp` chain.** -/
noncomputable def convNextForwardTCh_vjp_chain (w : CnxTWeightsCh)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) :
    HasVJP
      (dense w.Wd w.bd ∘
        rowLNVecFlat 1 768 w.hε w.hγ w.hβ ∘
        globalAvgPoolFlat 768 7 7 ∘
        convNextStageChK 3 w.s4 ∘
        cnxDownChW 7 7 w.d3 ∘
        convNextStageChK 9 w.s3 ∘
        cnxDownChW 14 14 w.d2 ∘
        convNextStageChK 3 w.s2 ∘
        cnxDownChW 28 28 w.d1 ∘
        convNextStageChK 3 w.s1 ∘
        chanLNTensor3 96 56 56 w.sε w.sγ w.sβ ∘
        flatConvStride4 (h := 56) (w := 56) w.sW w.sb) :=
  cnxV11 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε

set_option maxHeartbeats 1000000 in
/-- ⭐⭐ **`convnextInputGrad` IS the certified whole-net ConvNeXt-T gradient.** -/
theorem convnextInputGrad_eq_convNextForwardTCh_vjp (w : CnxTWeightsCh)
    (hsε : 0 < w.sε)
    (h1 : ∀ i, 0 < (w.s1 i).εn) (hd1 : 0 < w.d1.ε)
    (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε)
    (h3 : ∀ i, 0 < (w.s3 i).εn) (hd3 : 0 < w.d3.ε)
    (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε)
    (x : Vec (3 * 224 * 224)) :
    convnextInputGrad w.Wd (padOdd w.sW)
        (chanLNTensor3Back 96 56 56 w.sε w.sγ (cnxSavedA0 w x))
        (rowLNVecFlatBack 1 768 w.hε w.hγ (cnxSavedA9 w x))
        (cnxStageChKBack 3 w.s1 (cnxSavedA1 w x))
        (cnxDownBack (h := 28) (w := 28) (padOdd w.d1.W)
          (chanLNTensor3Back 96 56 56 w.d1.ε w.d1.γ (cnxSavedA2 w x)))
        (cnxStageChKBack 3 w.s2 (cnxSavedA3 w x))
        (cnxDownBack (h := 14) (w := 14) (padOdd w.d2.W)
          (chanLNTensor3Back 192 28 28 w.d2.ε w.d2.γ (cnxSavedA4 w x)))
        (cnxStageChKBack 9 w.s3 (cnxSavedA5 w x))
        (cnxDownBack (h := 7) (w := 7) (padOdd w.d3.W)
          (chanLNTensor3Back 384 14 14 w.d3.ε w.d3.γ (cnxSavedA6 w x)))
        (cnxStageChKBack 3 w.s4 (cnxSavedA7 w x))
      = (convNextForwardTCh_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x := by
  rw [cnxStageChKBack_eq_vjp (by norm_num) (by norm_num) 3 w.s1 h1 (cnxSavedA1 w x),
      cnxStageChKBack_eq_vjp (by norm_num) (by norm_num) 3 w.s2 h2 (cnxSavedA3 w x),
      cnxStageChKBack_eq_vjp (by norm_num) (by norm_num) 9 w.s3 h3 (cnxSavedA5 w x),
      cnxStageChKBack_eq_vjp (by norm_num) (by norm_num) 3 w.s4 h4 (cnxSavedA7 w x),
      cnxDn1Back_eq_vjp w hd1, cnxDn2Back_eq_vjp w hd2, cnxDn3Back_eq_vjp w hd3,
      cnxLNhBack_eq_vjp w hhε,
      chanLNTensor3Back_eq_chanLN_vjp (β := w.sβ) w.sε hsε w.sγ (cnxSavedA0 w x)]
  unfold convnextInputGrad
  rw [flatConvStride4Back_padOdd_eq_vjp_backward (h := 56) (w := 56) (by norm_num) (by norm_num)
        w.sW w.sb x,
      gapBack_eq_vjp_backward 768 7 7 (cnxSavedA8 w x),
      dense_transpose_eq_vjp_backward w.Wd w.bd (cnxSavedA10 w x)]
  funext dy
  rw [HasVJP.backward_unique (convNextForwardTCh_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
        (convNextForwardTCh_vjp_chain w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε) x dy,
      convNextForwardTCh_vjp_chain,
      cnxT11 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x,
      cnxT10 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x,
      cnxT9 w hsε h1 hd1 h2 hd2 h3 hd3 h4 x,
      cnxT8 w hsε h1 hd1 h2 hd2 h3 hd3 h4 x,
      cnxT7 w hsε h1 hd1 h2 hd2 h3 hd3 x,
      cnxT6 w hsε h1 hd1 h2 hd2 h3 x,
      cnxT5 w hsε h1 hd1 h2 hd2 x,
      cnxT4 w hsε h1 hd1 h2 x,
      cnxT3 w hsε h1 hd1 x,
      cnxT2 w hsε h1 x,
      cnxT1 w hsε x]
  simp only [Function.comp_apply, cnxV0]

end Proofs
