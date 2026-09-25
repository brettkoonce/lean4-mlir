import LeanMlir.Proofs.Architectures.EvenKernelConvBack
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtBackCertifiedTie

/-! # ConvNeXt-T's whole-net backward tie — the stage fold, and ⛔ what the tie FOUND

⭐ **Read `EvenKernelConvBack.lean` first; the finding is the deliverable.** This file was started
as the ConvNeXt peer of ResNet-34's whole-net tie (today `r34InputGradB_eq_r34B_full_vjp`) —
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
`.convStridedBack` pads ASYMMETRICALLY, `[[kH-1-pH, pH]]`, in both the per-example (`.convStridedBack`) and the
batched (`.convStridedBackBatched`) arms, and its `den` is the certified VJP; the batched comment names the same
quantity — *"the symmetric `[[p,p],[p,p]]` … AGREES at every odd kernel and is WRONG at even ones
(kH=2 ⇒ `[[0,0]]` where the VJP needs `[[1,0]]`)"*. The fix landed on TWO tiers and never reached
the third: `BackwardMaps.lean`'s `flatConvStride2Back` / `flatConvStride4Back`, which are
`convFlatBack ∘ scatter` at the SYMMETRIC pad. ⭐ That is the recurring twin-drift pattern in its
*"a fix landed on one tier and its twin kept the old spelling"* form, for the third time (§3.10's
pool and §3.16's head LayerNorm were the first two).

**What is here.** The repair is `padOdd` (`EvenKernelConvBack.lean`): an even-kernel conv is an
odd-kernel conv on the kernel zero-extended at `(+1,+1)`, which is the emitter's asymmetric pad
written in the vocabulary `BackwardMaps.lean` already has, so the existing odd-kernel leaf tie does
all the work and no new conv machinery is needed. On top of it:

1. `cnxDownChBack_eq_vjp` — the stage-boundary downsample tie, `lnB ∘ flatConvStride2Back
   (padOdd W)` against `(cnxDownChWHasVJP …).backward`. ⛔ `padOdd` is load-bearing: `p.W` is
   `2×2`. Two existing ties composed.
2. `cnxStageChKBack_eq_vjp` — ⭐ **the depth-`k` stage fold, §3.18's "one real proof".** `HasVJP`
   for `convNextStageChK` is built head-first (block `0` runs first), so its backward composes the
   block backwards in the OPPOSITE order, each at its own saved activation, and the tail's saved
   input is block `0`'s forward OUTPUT. The induction step is one rewrite of the block tie
   (`cnxBlockChBack_eq_vjp`) and one of the inductive hypothesis.
3. `cnxSavedA0 … cnxSavedA10` — `convNextForwardTCh`'s eleven stage inputs, named as FUNCTIONS, so
   that the same twelve constants are both the activations the backward's slots are saved at and
   the `f` argument of each `vjpComp`.
4. ⭐⭐ **`convnextInputGrad_eq_convNextForwardTCh_vjp` — THE APEX.** `convnextInputGrad`, with
   every slot pinned to the certified per-op backward at its own saved activation, IS
   `(convNextForwardTChHasVJP …).backward x`. The ConvNeXt peer of
   `r34InputGradB_eq_r34B_full_vjp`, and **stronger**: `convNextForwardTChHasVJP` is `HasVJP` —
   everywhere — not the smooth-point `HasVJPAt` that one is, because GELU, LayerNorm, convolution and the layer scale are all smooth and
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
  `conv2dInputGradFormula`, `Finset.sum`, `Mat.unflatten`, `cnxBlockChW`. Measured one link at a
  time, the three downsamples cost 3 s, 15 s and then do not finish, while every stage, LayerNorm,
  GAP and dense link is free. `cnxDn1`/`cnxDn2`/`cnxDn3` below are the whole fix: a one-line `def`
  with the type ascribed in the chain's spelling, plus `Differentiable`/`HasVJP` peers ascribed the
  same way. Same for `cnxLNh` at `Vec 768` against `rowLNVecFlat 1 768`'s `Vec (1 * 768)`.
* A leaf tie goes the OTHER way — state it in the LEMMA's spelling, not the chain's
  (`cnxLNhBack_eq_vjp` takes `v : Vec (1 * 768)`); at `Vec 768` the same statement does not finish.
* The saved activations are functions, so each `cnxTk` is a one-step iota with syntactically
  identical sides. Stated the other way — the chain's own `f x` against an applied
  `cnxSavedA k w x` — identifying the two costs 2 s at depth one and does not finish at depth two.
* The closing step is `rw [cnxV0]` and `rw [Function.comp_apply]`, not `rfl`: after the eleven peels
  the two sides differ only by `Function.comp` and `cnxV0`, and `rfl` will not take that route.
  ⛔ Nor `simp only [Function.comp_apply, cnxV0]`, which elaborates just as fast: both lemmas are
  definitional, so simp records no step and the KERNEL re-derives the whole chain by unfolding —
  17 s and 6 GB for this module on Lean 4.32.2, 6 min and 48 GB on 4.34.0. The `rw`s hand it
  syntactic rewrites instead: 3 s and 3 GB on 4.34.0.

⚠ `planning/archive/float_budget_numbers_log.md` §3.7(d) records this trap in its other guise, where the
computed dimension meets a metavariable (`2 * ?h = 112`) and the unification is higher-order; there
the fix is to pin the implicit. Here `h` is given explicitly and it still costs — two CLOSED
spellings of one numeral are enough. ⛔ And it is invisible in an unapplied position:
`convNextForwardTChVjpChain`'s ascription compares the whole twelve-factor composition against
the committed one and is free, because no `x` is in sight to evaluate.

⛔ **The other half of the shape is the term-mode chain, and it is not a preference.**
`convNextForwardTChHasVJP` is a tactic proof, so its eleven `have`s are `letFun` and its
`.backward` does not reduce; the whole-net `rfl` against it returned no result at
`maxHeartbeats 8000000`, twice, ~8 min each. `HasVJP.backward_unique` transfers through `.correct`
instead, which costs nothing, and the term-mode peer must be top-level `def`s rather than a `let`
chain — a `let` used twice per level zeta-expands to `2^11` copies of the prefix.

⚠ ResNet-34's shape check is `resnet34ForwardBFull_eq_slots` (`ResNet34BackCertifiedTieB.lean`),
and it is the net the hole first bit.
-/

namespace Proofs


-- ════════════════════════════════════════════════════════════════
-- § The stage-boundary downsample
-- ════════════════════════════════════════════════════════════════

/-- **The downsample backward tie.** `cnxDownBack (padOdd p.W) lnB` — the strided-conv backward at
    the ZERO-EXTENDED kernel, then the channel-LN back at the input resolution — is
    `(cnxDownChWHasVJP h w p hε).backward v`.

    ⛔ `padOdd` is load-bearing and not cosmetic: `p.W` is `2×2`, so `cnxDownBack p.W` reverses a
    conv shifted one pixel (`EvenKernelConvBack.lean`). This is one of the four sites the
    whole-net tie found. -/
theorem cnxDownChBack_eq_vjp {cin cout h w : Nat} (p : CnxDownParamsCh cin cout)
    (hε : 0 < p.ε) (v : Vec (cin * (2 * h) * (2 * w))) :
    cnxDownBack (h := h) (w := w) (padOdd p.W)
        (chanLNTensor3Back cin (2 * h) (2 * w) p.ε p.γ v)
      = (cnxDownChWHasVJP h w p hε).backward v := by
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
    ((layerScaleHasVJP (cnxGlsCh p)).backward
      ((flatConv (h := h) (w := w) p.Wpr p.bpr ∘ gelu (cExp * h * w) ∘
        flatConv (h := h) (w := w) p.Wex p.bex ∘ chanLNTensor3 c h w p.εn p.γn p.βn ∘
        depthwiseFlat (h := h) (w := w) p.Wdw p.bdw) v))
    ((geluHasVJP (cExp * h * w)).backward
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
    `(convNextStageChKHasVJP k ps hε).backward`. Induction on `k`: the base case is
    `identityHasVJP`'s `fun _ dy => dy`, and the step is one rewrite of the block tie
    (`cnxBlockChBack_eq_vjp`) and one of the inductive hypothesis at the shifted saved
    activation. -/
theorem cnxStageChKBack_eq_vjp {c cExp h w kHd kWd : Nat}
    (hkHd : 2 * ((kHd - 1) / 2) + 1 = kHd) (hkWd : 2 * ((kWd - 1) / 2) + 1 = kWd) :
    ∀ (k : Nat) (ps : Fin k → CnxBlockParamsCh c cExp h w kHd kWd)
      (hε : ∀ i, 0 < (ps i).εn) (v : Vec (c * h * w)),
      cnxStageChKBack k ps v = (convNextStageChKHasVJP k ps hε).backward v
  | 0, _, _, _ => rfl
  | k + 1, ps, hε, v => by
      show cnxBlockChBackAt (ps 0) v ∘
        cnxStageChKBack k (fun i => ps i.succ) (cnxBlockChW (ps 0) v) = _
      rw [cnxStageChKBack_eq_vjp hkHd hkWd k (fun i => ps i.succ) (fun i => hε i.succ)
            (cnxBlockChW (ps 0) v)]
      show Proofs.residual _ ∘ _ = _
      rw [cnxBlockChBack_eq_vjp hkHd hkWd (ps 0) (hε 0) v]
      rfl

/-- `rowLNVecFlatHasVJP_backward_eq` at the FUNCTION level — the direction and shape a whole-net
    `rw` needs. The committed lemma is pointwise in `dy` and oriented certified-to-hand; a chain
    rewrite wants hand-to-certified with `dy` abstracted. -/
theorem rowLNVecFlatHasVJP_backward_eq_fun {s c : Nat} (ε : ℝ) (hε : 0 < ε) (γ β : Vec c)
    (X : Vec (s * c)) :
    rowLNVecFlatBack s c ε γ X = (rowLNVecFlatHasVJP s c ε γ β hε).backward X := by
  funext dy
  rw [rowLNVecFlatHasVJP_backward_eq (β := β) ε hε γ X dy]


-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE VJP CHAIN, at normalised dimension spellings
-- ════════════════════════════════════════════════════════════════

/-! ⭐⭐ **Every stage whose declared type carries a COMPUTED dimension gets a wrapper here, and
that is the whole reason this section closes in seconds.** `cnxDownChW h w p` is declared over
`Vec (cin * (2 * h) * (2 * w))`; the chain spells the same type `Vec (96 * 56 * 56)`. Both are
closed terms and they are equal, but in an APPLIED position the unifier does not reduce `2 * 28`
to `56` — it descends into the semantics of both sides instead, and the diagnostics name what it
reaches: `conv2dInputGradFormula`, `Finset.sum`, `Mat.unflatten`, `cnxBlockChW`. Measured, one
tie at a time: the three downsample links cost 3 s, 15 s and then do not finish, while every
stage, LayerNorm, GAP and dense link is free. With the wrappers below — a one-line `def` per
offending stage, its type ascribed in the chain's spelling, and its `Differentiable`/`HasVJP`
peers ascribed the same way — the twelve chain defs and all eleven links together cost
**2.9 s**, of which the links are ~0.3 s.

⚠ `planning/archive/float_budget_numbers_log.md` §3.7(d) records this trap in its other guise, where the
computed dimension meets a metavariable (`2 * ?h = 112`) and the unification is higher-order.
There the fix is to pin the implicit. Here `h` is already given explicitly and it still costs:
two CLOSED spellings of one numeral are enough. ⛔ And it is invisible in an unapplied position —
`convNextForwardTChVjpChain`'s ascription below compares the whole twelve-factor composition
against the committed one and is free, because no `x` is in sight to evaluate. -/

/-! ⭐ The wrappers, their `Differentiable`/`HasVJP` peers, the stem's `cnxSavedA0_differentiable`/`cnxV0` and the four
normalised leaf ties below are PUBLIC: `ConvNeXtWholeBackCertifiedTieB.lean` lifts the same
twelve stages over a batch and needs them at exactly these spellings. -/

/-- Downsample 1 at the chain's dimension spelling. -/
noncomputable def cnxDn1 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (96 * 56 * 56) → Vec (192 * 28 * 28) :=
  cnxDownChW 28 28 w.d1
theorem cnxDn1_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hd1 : 0 < w.d1.ε) :
    Differentiable ℝ (cnxDn1 w) := cnxDownChW_differentiable 28 28 w.d1 hd1
noncomputable def cnxDn1Vjp {nC : Nat} (w : CnxTWeightsCh nC) (hd1 : 0 < w.d1.ε) :
    HasVJP (cnxDn1 w) := cnxDownChWHasVJP 28 28 w.d1 hd1

/-- Downsample 2 at the chain's dimension spelling. -/
noncomputable def cnxDn2 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (192 * 28 * 28) → Vec (384 * 14 * 14) :=
  cnxDownChW 14 14 w.d2
theorem cnxDn2_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hd2 : 0 < w.d2.ε) :
    Differentiable ℝ (cnxDn2 w) := cnxDownChW_differentiable 14 14 w.d2 hd2
noncomputable def cnxDn2Vjp {nC : Nat} (w : CnxTWeightsCh nC) (hd2 : 0 < w.d2.ε) :
    HasVJP (cnxDn2 w) := cnxDownChWHasVJP 14 14 w.d2 hd2

/-- Downsample 3 at the chain's dimension spelling. -/
noncomputable def cnxDn3 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (384 * 14 * 14) → Vec (768 * 7 * 7) :=
  cnxDownChW 7 7 w.d3
theorem cnxDn3_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hd3 : 0 < w.d3.ε) :
    Differentiable ℝ (cnxDn3 w) := cnxDownChW_differentiable 7 7 w.d3 hd3
noncomputable def cnxDn3Vjp {nC : Nat} (w : CnxTWeightsCh nC) (hd3 : 0 < w.d3.ε) :
    HasVJP (cnxDn3 w) := cnxDownChWHasVJP 7 7 w.d3 hd3

/-- The head LayerNorm at `Vec 768`, not `Vec (1 * 768)` — the same normalisation, at the one
    site where the computed dimension is a `1 *` rather than a `2 *`. -/
noncomputable def cnxLNh {nC : Nat} (w : CnxTWeightsCh nC) : Vec 768 → Vec 768 :=
  rowLNVecFlat 1 768 w.hε w.hγ w.hβ
theorem cnxLNh_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hhε : 0 < w.hε) :
    Differentiable ℝ (cnxLNh w) := rowLNVecFlat_differentiable 1 768 w.hε w.hγ w.hβ hhε
noncomputable def cnxLNhVjp {nC : Nat} (w : CnxTWeightsCh nC) (hhε : 0 < w.hε) :
    HasVJP (cnxLNh w) := rowLNVecFlatHasVJP 1 768 w.hε w.hγ w.hβ hhε


-- ── the forward prefixes: `cnxSavedA k w x` is stage `k`'s saved input ──

/-! ⭐ `convNextForwardTCh`'s eleven stage inputs, named — and named as FUNCTIONS, so that the
same twelve constants are both the saved activations the backward's slots are indexed by and the
`f` argument of each `vjpComp`. That is what makes every link below a one-step iota with
syntactically identical sides: the alternative — an applied `cnxSavedA k w x` on one side and the
chain's own `f x` on the other — is defeq, and identifying the two costs 2 s at depth one and
does not finish at depth two. -/

/-- The stem conv's output — the stem LayerNorm's saved input. -/
noncomputable def cnxSavedA0 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (96 * 56 * 56) :=
    flatConvStride4 (h := 56) (w := 56) w.sW w.sb

theorem cnxSavedA0_differentiable {nC : Nat} (w : CnxTWeightsCh nC) : Differentiable ℝ (cnxSavedA0 w) :=
    flatConvStride4_differentiable (h := 56) (w := 56) w.sW w.sb
noncomputable def cnxV0 {nC : Nat} (w : CnxTWeightsCh nC) : HasVJP (cnxSavedA0 w) :=
    flatConvStride4HasVJP (h := 56) (w := 56) w.sW w.sb

/-- Stage 1's saved input. -/
noncomputable def cnxSavedA1 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (96 * 56 * 56) :=
  chanLNTensor3 96 56 56 w.sε w.sγ w.sβ ∘ cnxSavedA0 w

private theorem cnxSavedA1_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) : Differentiable ℝ (cnxSavedA1 w) :=
  (chanLNTensor3_differentiable 96 56 56 w.sε w.sγ w.sβ hsε).comp (cnxSavedA0_differentiable w)
private noncomputable def cnxV1 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) : HasVJP (cnxSavedA1 w) :=
  vjpComp (cnxSavedA0 w) _ (cnxSavedA0_differentiable w) (chanLNTensor3_differentiable 96 56 56 w.sε w.sγ w.sβ hsε) (cnxV0 w)
    (chanLNTensor3HasVJP 96 56 56 w.sε w.sγ w.sβ hsε)

/-- Downsample 1's saved input. -/
noncomputable def cnxSavedA2 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (96 * 56 * 56) :=
  convNextStageChK 3 w.s1 ∘ cnxSavedA1 w

private theorem cnxSavedA2_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn) :
    Differentiable ℝ (cnxSavedA2 w) :=
  (convNextStageChK_differentiable 3 w.s1 h1).comp (cnxSavedA1_differentiable w hsε)
private noncomputable def cnxV2 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn) :
    HasVJP (cnxSavedA2 w) :=
  vjpComp (cnxSavedA1 w) _ (cnxSavedA1_differentiable w hsε) (convNextStageChK_differentiable 3 w.s1 h1) (cnxV1 w hsε)
    (convNextStageChKHasVJP 3 w.s1 h1)

/-- Stage 2's saved input. -/
noncomputable def cnxSavedA3 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (192 * 28 * 28) :=
  cnxDn1 w ∘ cnxSavedA2 w

private theorem cnxSavedA3_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) : Differentiable ℝ (cnxSavedA3 w) :=
  (cnxDn1_differentiable w hd1).comp (cnxSavedA2_differentiable w hsε h1)
private noncomputable def cnxV3 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) : HasVJP (cnxSavedA3 w) :=
  vjpComp (cnxSavedA2 w) _ (cnxSavedA2_differentiable w hsε h1) (cnxDn1_differentiable w hd1) (cnxV2 w hsε h1) (cnxDn1Vjp w hd1)

/-- Downsample 2's saved input. -/
noncomputable def cnxSavedA4 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (192 * 28 * 28) :=
  convNextStageChK 3 w.s2 ∘ cnxSavedA3 w

private theorem cnxSavedA4_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) : Differentiable ℝ (cnxSavedA4 w) :=
  (convNextStageChK_differentiable 3 w.s2 h2).comp (cnxSavedA3_differentiable w hsε h1 hd1)
private noncomputable def cnxV4 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) : HasVJP (cnxSavedA4 w) :=
  vjpComp (cnxSavedA3 w) _ (cnxSavedA3_differentiable w hsε h1 hd1) (convNextStageChK_differentiable 3 w.s2 h2)
    (cnxV3 w hsε h1 hd1) (convNextStageChKHasVJP 3 w.s2 h2)

/-- Stage 3's saved input. -/
noncomputable def cnxSavedA5 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (384 * 14 * 14) :=
  cnxDn2 w ∘ cnxSavedA4 w

private theorem cnxSavedA5_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) : Differentiable ℝ
    (cnxSavedA5 w) :=
  (cnxDn2_differentiable w hd2).comp (cnxSavedA4_differentiable w hsε h1 hd1 h2)
private noncomputable def cnxV5 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) : HasVJP (cnxSavedA5 w) :=
  vjpComp (cnxSavedA4 w) _ (cnxSavedA4_differentiable w hsε h1 hd1 h2) (cnxDn2_differentiable w hd2) (cnxV4 w hsε h1 hd1 h2)
    (cnxDn2Vjp w hd2)

/-- Downsample 3's saved input. -/
noncomputable def cnxSavedA6 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (384 * 14 * 14) :=
  convNextStageChK 9 w.s3 ∘ cnxSavedA5 w

private theorem cnxSavedA6_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn) :
    Differentiable ℝ (cnxSavedA6 w) :=
  (convNextStageChK_differentiable 9 w.s3 h3).comp (cnxSavedA5_differentiable w hsε h1 hd1 h2 hd2)
private noncomputable def cnxV6 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn) :
    HasVJP (cnxSavedA6 w) :=
  vjpComp (cnxSavedA5 w) _ (cnxSavedA5_differentiable w hsε h1 hd1 h2 hd2) (convNextStageChK_differentiable 9 w.s3 h3)
    (cnxV5 w hsε h1 hd1 h2 hd2) (convNextStageChKHasVJP 9 w.s3 h3)

/-- Stage 4's saved input. -/
noncomputable def cnxSavedA7 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (768 * 7 * 7) :=
  cnxDn3 w ∘ cnxSavedA6 w

private theorem cnxSavedA7_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) : Differentiable ℝ (cnxSavedA7 w) :=
  (cnxDn3_differentiable w hd3).comp (cnxSavedA6_differentiable w hsε h1 hd1 h2 hd2 h3)
private noncomputable def cnxV7 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) : HasVJP (cnxSavedA7 w) :=
  vjpComp (cnxSavedA6 w) _ (cnxSavedA6_differentiable w hsε h1 hd1 h2 hd2 h3) (cnxDn3_differentiable w hd3)
    (cnxV6 w hsε h1 hd1 h2 hd2 h3) (cnxDn3Vjp w hd3)

/-- GAP's saved input. -/
noncomputable def cnxSavedA8 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (768 * 7 * 7) :=
  convNextStageChK 3 w.s4 ∘ cnxSavedA7 w

private theorem cnxSavedA8_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) : Differentiable ℝ (cnxSavedA8 w) :=
  (convNextStageChK_differentiable 3 w.s4 h4).comp (cnxSavedA7_differentiable w hsε h1 hd1 h2 hd2 h3 hd3)
private noncomputable def cnxV8 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) : HasVJP (cnxSavedA8 w) :=
  vjpComp (cnxSavedA7 w) _ (cnxSavedA7_differentiable w hsε h1 hd1 h2 hd2 h3 hd3) (convNextStageChK_differentiable 3 w.s4 h4)
    (cnxV7 w hsε h1 hd1 h2 hd2 h3 hd3) (convNextStageChKHasVJP 3 w.s4 h4)

/-- The head LayerNorm's saved input. -/
noncomputable def cnxSavedA9 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (768) :=
  globalAvgPoolFlat 768 7 7 ∘ cnxSavedA8 w

private theorem cnxSavedA9_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) : Differentiable ℝ (cnxSavedA9 w) :=
  (globalAvgPoolFlat_differentiable 768 7 7).comp (cnxSavedA8_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4)
private noncomputable def cnxV9 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) : HasVJP (cnxSavedA9 w) :=
  vjpComp (cnxSavedA8 w) _ (cnxSavedA8_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4)
    (globalAvgPoolFlat_differentiable 768 7 7) (cnxV8 w hsε h1 hd1 h2 hd2 h3 hd3 h4)
    (globalAvgPoolFlatHasVJP 768 7 7)

/-- The classifier's saved input. -/
noncomputable def cnxSavedA10 {nC : Nat} (w : CnxTWeightsCh nC) : Vec (3 * 224 * 224) → Vec (768) :=
  cnxLNh w ∘ cnxSavedA9 w

private theorem cnxSavedA10_differentiable {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) : Differentiable ℝ
    (cnxSavedA10 w) :=
  (cnxLNh_differentiable w hhε).comp (cnxSavedA9_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4)
private noncomputable def cnxV10 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) : HasVJP (cnxSavedA10 w) :=
  vjpComp (cnxSavedA9 w) _ (cnxSavedA9_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4) (cnxLNh_differentiable w hhε)
    (cnxV9 w hsε h1 hd1 h2 hd2 h3 hd3 h4) (cnxLNhVjp w hhε)

private noncomputable def cnxV11 {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) :
    HasVJP (dense w.Wd w.bd ∘ cnxSavedA10 w) :=
  vjpComp (cnxSavedA10 w) _ (cnxSavedA10_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
    (dense_differentiable w.Wd w.bd) (cnxV10 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
    (denseHasVJP w.Wd w.bd)

-- ── the eleven single-level reductions ──

private theorem cnxV1_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (x : Vec (3 * 224 * 224))
    (dy : Vec (96 * 56 * 56)) :
    (cnxV1 w hsε).backward x dy
      = (cnxV0 w).backward x
        ((chanLNTensor3HasVJP 96 56 56 w.sε w.sγ w.sβ hsε).backward (cnxSavedA0 w x) dy) := rfl

private theorem cnxV2_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (x : Vec (3 * 224 * 224))
    (dy : Vec (96 * 56 * 56)) :
    (cnxV2 w hsε h1).backward x dy
      = (cnxV1 w hsε).backward x
        ((convNextStageChKHasVJP 3 w.s1 h1).backward (cnxSavedA1 w x) dy) := rfl

private theorem cnxV3_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (x : Vec (3 * 224 * 224))
    (dy : Vec (192 * 28 * 28)) :
    (cnxV3 w hsε h1 hd1).backward x dy
      = (cnxV2 w hsε h1).backward x ((cnxDn1Vjp w hd1).backward (cnxSavedA2 w x) dy) := rfl

private theorem cnxV4_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (x : Vec (3 * 224 * 224))
    (dy : Vec (192 * 28 * 28)) :
    (cnxV4 w hsε h1 hd1 h2).backward x dy
      = (cnxV3 w hsε h1 hd1).backward x
        ((convNextStageChKHasVJP 3 w.s2 h2).backward (cnxSavedA3 w x) dy) := rfl

private theorem cnxV5_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (x : Vec (3 * 224 * 224))
    (dy : Vec (384 * 14 * 14)) :
    (cnxV5 w hsε h1 hd1 h2 hd2).backward x dy
      = (cnxV4 w hsε h1 hd1 h2).backward x ((cnxDn2Vjp w hd2).backward (cnxSavedA4 w x) dy) := rfl

private theorem cnxV6_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (x : Vec (3 * 224 * 224))
    (dy : Vec (384 * 14 * 14)) :
    (cnxV6 w hsε h1 hd1 h2 hd2 h3).backward x dy
      = (cnxV5 w hsε h1 hd1 h2 hd2).backward x
        ((convNextStageChKHasVJP 9 w.s3 h3).backward (cnxSavedA5 w x) dy) := rfl

private theorem cnxV7_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (x : Vec (3 * 224 * 224))
    (dy : Vec (768 * 7 * 7)) :
    (cnxV7 w hsε h1 hd1 h2 hd2 h3 hd3).backward x dy
      = (cnxV6 w hsε h1 hd1 h2 hd2 h3).backward x ((cnxDn3Vjp w hd3).backward (cnxSavedA6 w x) dy)
        := rfl

private theorem cnxV8_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (x : Vec (3 * 224 * 224))
    (dy : Vec (768 * 7 * 7)) :
    (cnxV8 w hsε h1 hd1 h2 hd2 h3 hd3 h4).backward x dy
      = (cnxV7 w hsε h1 hd1 h2 hd2 h3 hd3).backward x
        ((convNextStageChKHasVJP 3 w.s4 h4).backward (cnxSavedA7 w x) dy) := rfl

private theorem cnxV9_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (x : Vec (3 * 224 * 224))
    (dy : Vec (768)) :
    (cnxV9 w hsε h1 hd1 h2 hd2 h3 hd3 h4).backward x dy
      = (cnxV8 w hsε h1 hd1 h2 hd2 h3 hd3 h4).backward x
        ((globalAvgPoolFlatHasVJP 768 7 7).backward (cnxSavedA8 w x) dy) := rfl

private theorem cnxV10_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) (x : Vec (3 * 224 * 224))
    (dy : Vec (768)) :
    (cnxV10 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x dy
      = (cnxV9 w hsε h1 hd1 h2 hd2 h3 hd3 h4).backward x
        ((cnxLNhVjp w hhε).backward (cnxSavedA9 w x) dy) := rfl

private theorem cnxV11_backward {nC : Nat} (w : CnxTWeightsCh nC) (hsε : 0 < w.sε) (h1 : ∀ i, 0 < (w.s1 i).εn)
    (hd1 : 0 < w.d1.ε) (h2 : ∀ i, 0 < (w.s2 i).εn) (hd2 : 0 < w.d2.ε) (h3 : ∀ i, 0 < (w.s3 i).εn)
    (hd3 : 0 < w.d3.ε) (h4 : ∀ i, 0 < (w.s4 i).εn) (hhε : 0 < w.hε) (x : Vec (3 * 224 * 224))
    (dy : Vec nC) :
    (cnxV11 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x dy
      = (cnxV10 w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x
        ((denseHasVJP w.Wd w.bd).backward (cnxSavedA10 w x) dy) := rfl

-- ── the three normalised leaf ties the wrappers need ──

theorem cnxDn1Back_eq_vjp {nC : Nat} (w : CnxTWeightsCh nC) (hd1 : 0 < w.d1.ε) (v : Vec (96 * 56 * 56)) :
    cnxDownBack (h := 28) (w := 28) (padOdd w.d1.W)
        (chanLNTensor3Back 96 56 56 w.d1.ε w.d1.γ v)
      = (cnxDn1Vjp w hd1).backward v :=
  cnxDownChBack_eq_vjp (h := 28) (w := 28) w.d1 hd1 v

theorem cnxDn2Back_eq_vjp {nC : Nat} (w : CnxTWeightsCh nC) (hd2 : 0 < w.d2.ε) (v : Vec (192 * 28 * 28)) :
    cnxDownBack (h := 14) (w := 14) (padOdd w.d2.W)
        (chanLNTensor3Back 192 28 28 w.d2.ε w.d2.γ v)
      = (cnxDn2Vjp w hd2).backward v :=
  cnxDownChBack_eq_vjp (h := 14) (w := 14) w.d2 hd2 v

theorem cnxDn3Back_eq_vjp {nC : Nat} (w : CnxTWeightsCh nC) (hd3 : 0 < w.d3.ε) (v : Vec (384 * 14 * 14)) :
    cnxDownBack (h := 7) (w := 7) (padOdd w.d3.W)
        (chanLNTensor3Back 384 14 14 w.d3.ε w.d3.γ v)
      = (cnxDn3Vjp w hd3).backward v :=
  cnxDownChBack_eq_vjp (h := 7) (w := 7) w.d3 hd3 v

theorem cnxLNhBack_eq_vjp {nC : Nat} (w : CnxTWeightsCh nC) (hhε : 0 < w.hε) (v : Vec (1 * 768)) :
    rowLNVecFlatBack 1 768 w.hε w.hγ v = (cnxLNhVjp w hhε).backward v :=
  rowLNVecFlatHasVJP_backward_eq_fun (β := w.hβ) w.hε hhε w.hγ v

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ THE APEX
-- ════════════════════════════════════════════════════════════════

/-- **`convNextForwardTChHasVJP` as a TERM-mode `vjpComp` chain.** -/
noncomputable def convNextForwardTChVjpChain {nC : Nat} (w : CnxTWeightsCh nC)
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

/-- ⭐⭐ **`convnextInputGrad` IS the certified whole-net ConvNeXt-T gradient.** -/
theorem convnextInputGrad_eq_convNextForwardTCh_vjp {nC : Nat} (w : CnxTWeightsCh nC)
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
      = (convNextForwardTChHasVJP w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε).backward x := by
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
  rw [HasVJP.backward_unique (convNextForwardTChHasVJP w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
        (convNextForwardTChVjpChain w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε) x dy,
      convNextForwardTChVjpChain,
      cnxV11_backward w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x,
      cnxV10_backward w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε x,
      cnxV9_backward w hsε h1 hd1 h2 hd2 h3 hd3 h4 x,
      cnxV8_backward w hsε h1 hd1 h2 hd2 h3 hd3 h4 x,
      cnxV7_backward w hsε h1 hd1 h2 hd2 h3 hd3 x,
      cnxV6_backward w hsε h1 hd1 h2 hd2 h3 x,
      cnxV5_backward w hsε h1 hd1 h2 hd2 x,
      cnxV4_backward w hsε h1 hd1 h2 x,
      cnxV3_backward w hsε h1 hd1 x,
      cnxV2_backward w hsε h1 x,
      cnxV1_backward w hsε x]
  rw [cnxV0]
  repeat rw [Function.comp_apply]

end Proofs
