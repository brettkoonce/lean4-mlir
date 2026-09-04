import LeanMlir.Proofs.Float.FloatBudgetEnvBackMBConv
import LeanMlir.Proofs.Float.FloatBudgetEnvLN
import LeanMlir.Proofs.Float.SEBackFloatBridge
import LeanMlir.Proofs.Float.EfficientNetWholeBackFloatBridge

/-! # The `Maps` kit the SQUEEZE-EXCITE backward needs, on top of `FloatBudgetEnvBackMBConv`

The fourth backward net's kit, beside `FloatBudgetEnvBack.lean`'s (ResNet-34),
`FloatBudgetEnvBackMBConv.lean`'s (MobileNetV2) and `FloatBudgetEnvBackLN.lean`'s (ConvNeXt-T).
With it the EfficientNet-B0 backward cone is closed at the `Maps` tier.

⭐ **Exactly ONE leaf here is new in kind: `Maps.broadcastBack`,** the squeeze-excite gate's
spatial reduce. Everything else the SE backward needs already existed at both tiers —
`Maps.diagBack` (the two saved-vector scales, and both `diagBack`s of the product rule),
`Maps.linBack`, `Maps.gapBack`, `Maps.biPathSum`, `Maps.convBack`, `Maps.depthwiseBack`,
`Maps.depthwiseStride2Back` and `Maps.batchMap` — which is `planning/float_budget_numbers.md`
§3.9's costing holding to the leaf, as §3.13's did for MobileNetV2.

⛔ **What was NOT already there is the TIER.** `SEBackFloatBridge.lean` carried three
`floatBridges_` and **zero** `floatBridgesTo_`, and `EfficientNetWholeBackFloatBridge.lean` two
more — and a budget file cannot use an `∃`-tier bridge, because `FloatBridges` discards the float
map and a `Maps` envelope has to name one. That is §3.5.1's surprise for the fourth time (§3.17
was the third), and it is what §3.17 says to check before costing a budget file: **grep the cone
for `floatBridgesTo_`, not for the defs, which exist either way.**

⭐ **The structural fact the SE backward turns on, and it is `planning`'s §3.8 answer in one
line:** `seInputGrad g xinp gateBack = biPathSum (diagBack g) (gateBack ∘ diagBack xinp)` — the
saved gate `g` and the saved input `xinp` are CONSTANTS, both branches are linear in the
cotangent, and nothing multiplies the cotangent by itself. The forward's SE is quadratic in the
window because the gate is grown out of the same input the rescale multiplies (§0.1's third
reducing site); on the backward that input is not the cotangent, so the fold is a fold.

⚠ **`broadcastBack` is the one stage that MULTIPLIES by a fan-in the gate then divides back out.**
Its window carries `c·h·w` where the honest count is `h·w` — the `(c−1)·h·w` masked entries are
identically zero and `floatClose_broadcastBack`'s `hsumabs` bounds them by `A` anyway (§3.9
finding 7). Worth 6 orders on B0's fold (10¹⁶⁹ → 10¹⁶⁴) at NO new hypothesis. ⛔ It is priced and
NOT taken here, and §3.9's "cheap to fix" is wrong: the honest count needs the cardinality of
`flatChannel c h w`'s fibre, and that lemma does not exist — it is a `Finset.card` argument
through two `finProdFinEquiv`s, not a one-liner. The leaf below is the bound as PROVED.

⚠ **It imports `FloatBudgetEnvLN.lean` for ONE leaf, `Maps.diagBack`** — the saved-vector
pointwise scale, which the squeeze-excite backward uses four times (both saved derivatives inside
the gate, and both branches of the product rule) and which happens to live in the file written for
ConvNeXt's layer scale. That pulls the LayerNorm kit into B0's backward cone for no mathematical
reason; the clean cut is the `FloatBudgetEnvCore` split §3.11 priced and declined, on exactly this
evidence. `FloatBudgetEnvLN.lean` already does the same thing in the other direction (it imports
`FloatBudgetEnvMBConv` for `Maps.depthwise`).

⚠ **Why this file and not `FloatBudgetEnvBackMBConv.lean`.** A `Maps` lemma names its bridge, and
these name the squeeze-excite cone; putting them there would make MobileNetV2's backward budget
depend on EfficientNet's. Same reasoning that created `FloatBudgetEnvMBConv.lean` for the forward
(§0) and `FloatBudgetEnvBackLN.lean` for ConvNeXt's backward.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The one new leaf: the squeeze-excite gate's spatial reduce
-- ════════════════════════════════════════════════════════════════

/-- **The broadcast adjoint float-bridges TO its float peer** — the `FloatBridgesTo` peer of
    `floatBridges_broadcastBack`, with the float map NAMED (`FloatModel.broadcastBackFlatF`, the
    rounded per-channel reduction the deployed kernel runs).

    ⚠ The fan-in charged is `c·h·w`, the full masked vector, because that is what
    `floatClose_broadcastBack` proves — see the file header for why the honest `h·w` is priced and
    not taken. -/
noncomputable def floatBridgesTo_broadcastBack {c h w : Nat} (M : FloatModel) (hc : 0 < c) :
    FloatBridgesTo (broadcastBackFlat c h w) (M.broadcastBackFlatF (c := c) (h := h) (w := w)) :=
  ⟨fun A => ((c * h * w : ℕ) : ℝ) * A
              + ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * A),
   fun A e => ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * (A + e))
                + ((c * h * w : ℕ) : ℝ) * e,
   fun A hA => ⟨(floatClose_broadcastBack M A).cod_nonneg hA hc,
     floatClose_broadcastBack M A⟩⟩

namespace FloatBridgesTo

/-- ⭐ **An envelope through the squeeze-excite gate's spatial reduce.** The `Vec (c·h·w) → Vec c`
    channel-wise sum: one rounded reduction of fan-in `c·h·w`, so the window is multiplied by that
    count and the modulus carries the Higham `γ` of a `c·h·w`-term sum.

    ⭐ This is the one stage of the SE backward that GROWS the cotangent window; `Maps.gapBack`
    four stages later divides by `h·w` and gives most of it back. Read the pair together: the gate
    path costs magnitude, not nonlinearity. -/
theorem Maps.broadcastBack {c h w : Nat} (M : FloatModel) (hc : 0 < c)
    {g Ā Ē Ā' Ē' : ℝ} (hg : (1 + M.u) ^ (c * h * w + 1) - 1 ≤ g)
    (hĀ' : ((c * h * w : ℕ) : ℝ) * Ā + g * (((c * h * w : ℕ) : ℝ) * Ā) ≤ Ā')
    (hĒ' : g * (((c * h * w : ℕ) : ℝ) * (Ā + Ē)) + ((c * h * w : ℕ) : ℝ) * Ē ≤ Ē') :
    (floatBridgesTo_broadcastBack (h := h) (w := w) M hc).Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    have hN0 : (0 : ℝ) ≤ ((c * h * w : ℕ) : ℝ) := by positivity
    have hγ0 : (0 : ℝ) ≤ (1 + M.u) ^ (c * h * w + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
    show ((c * h * w : ℕ) : ℝ) * A
        + ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * A) ≤ Ā'
    have h1 : ((c * h * w : ℕ) : ℝ) * A ≤ ((c * h * w : ℕ) : ℝ) * Ā :=
      mul_le_mul_of_nonneg_left hle hN0
    have h2 : ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * A)
        ≤ g * (((c * h * w : ℕ) : ℝ) * Ā) :=
      mul_le_mul hg h1 (mul_nonneg hN0 h0) (le_trans hγ0 hg)
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    have hN0 : (0 : ℝ) ≤ ((c * h * w : ℕ) : ℝ) := by positivity
    have hγ0 : (0 : ℝ) ≤ (1 + M.u) ^ (c * h * w + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
    show ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * (A + E))
        + ((c * h * w : ℕ) : ℝ) * E ≤ Ē'
    have h1 : ((c * h * w : ℕ) : ℝ) * (A + E) ≤ ((c * h * w : ℕ) : ℝ) * (Ā + Ē) :=
      mul_le_mul_of_nonneg_left (by linarith) hN0
    have h2 : ((1 + M.u) ^ (c * h * w + 1) - 1) * (((c * h * w : ℕ) : ℝ) * (A + E))
        ≤ g * (((c * h * w : ℕ) : ℝ) * (Ā + Ē)) :=
      mul_le_mul hg h1 (mul_nonneg hN0 (by linarith)) (le_trans hγ0 hg)
    have h3 : ((c * h * w : ℕ) : ℝ) * E ≤ ((c * h * w : ℕ) : ℝ) * Ē :=
      mul_le_mul_of_nonneg_left hEle hN0
    linarith

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The squeeze-excite GATE backward, at a named float peer
-- ════════════════════════════════════════════════════════════════

/-- The float squeeze-excite gate input-gradient — `seGateInputGrad`'s deployed peer, every stage
    the float map its bridge names. The exact reverse of the gate's six forward stages
    (`broadcastFlat ∘ sigmoid ∘ dense W₂ ∘ swish ∘ dense W₁ ∘ GAP`), rounded. -/
noncomputable def seGateInputGradF {c h w r : Nat} (M : FloatModel)
    (W₁ : Mat c r) (W₂ : Mat r c) (fssig : Vec c) (fssw : Vec r) :
    Vec (c * h * w) → Vec (c * h * w) :=
  gapBackF M c h w
  ∘ M.dense (Mat.transpose W₁) (0 : Vec c)
  ∘ M.diagBackF fssw
  ∘ M.dense (Mat.transpose W₂) (0 : Vec r)
  ∘ M.diagBackF fssig
  ∘ M.broadcastBackFlatF (c := c) (h := h) (w := w)

/-- **The SE gate's backward float-bridges TO its float peer** — the `FloatBridgesTo` peer of
    `floatBridges_seGateBack`, with the float map NAMED. Six concrete stages, no supplied bridge:
    the spatial reduce, the sigmoid `diagBack`, `linBack W₂`, the swish `diagBack`, `linBack W₁`,
    and `gapBack`.

    ⭐ The two saved-derivative vectors enter as bounds, not as windows: `Ssig` is `1 + esig`
    (a sigmoid gate cannot exceed 1) and `Ssw` is `2` at the caller (`swishScalarDeriv_abs_le`,
    `Architectures/SwishSaturation.lean`) — the GLOBAL bound whose absence put this fold at
    10⁴³¹ (§3.9 finding 3, §3.12). Neither imports the forward's certified window. -/
noncomputable def floatBridgesTo_seGateBack {c h w r : Nat} (M : FloatModel)
    (W₁ : Mat c r) (W₂ : Mat r c) (ssig fssig : Vec c) (ssw fssw : Vec r)
    {w' Ssig esig Ssw eswish : ℝ} (hw' : 0 ≤ w')
    (hc : 0 < c) (hr : 0 < r) (hh : 0 < h) (hww : 0 < w)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hW₂ : ∀ i j, |W₂ i j| ≤ w')
    (hssig : ∀ i, |ssig i| ≤ Ssig) (hfssig : ∀ i, |fssig i - ssig i| ≤ esig)
    (hssw : ∀ i, |ssw i| ≤ Ssw) (hfssw : ∀ i, |fssw i - ssw i| ≤ eswish) :
    FloatBridgesTo (seGateInputGrad (h := h) (w := w) W₁ W₂ ssig ssw)
      (seGateInputGradF (h := h) (w := w) M W₁ W₂ fssig fssw) :=
  (((((floatBridgesTo_broadcastBack (h := h) (w := w) M hc).comp
    (floatBridgesTo_diagBack M ssig fssig hc hssig hfssig)).comp
    (floatBridgesTo_linBack M W₂ hw' hc hW₂)).comp
    (floatBridgesTo_diagBack M ssw fssw hr hssw hfssw)).comp
    (floatBridgesTo_linBack M W₁ hw' hr hW₁)).comp
    (floatBridgesTo_gapBack M c h w hc hh hww)

-- ════════════════════════════════════════════════════════════════
-- § The squeeze-excite BLOCK backward: the product-rule two-branch fan-in
-- ════════════════════════════════════════════════════════════════

/-- The float squeeze-excite block input-gradient — `seInputGrad`'s deployed peer. The rounded
    two-branch sum `fl(diagBack g dy + gateBack (diagBack x dy))`, in exactly the shape
    `FloatBridgesTo.biPathSum` names. -/
noncomputable def seInputGradF {n : Nat} (M : FloatModel) (fg fx : Vec n)
    (gateBackF : Vec n → Vec n) : Vec n → Vec n :=
  fun v j => M.add (M.diagBackF fg v j) ((gateBackF ∘ M.diagBackF fx) v j)

/-- **The SE block input-gradient float-bridges TO its float peer** — the `FloatBridgesTo` peer of
    `floatBridges_seBack`, with the float map NAMED.

    ⭐⭐ **This is the theorem behind §0.1's closing sentence.** The product rule fans out and
    rejoins, which is what made squeeze-excite the obvious suspect for a backward that does not
    fold — and it is not one: the gate `g` and the input `xinp` are SAVED CONSTANTS, both branches
    are `diagBack` scales of the cotangent, and nothing multiplies the cotangent by itself. The
    forward's SE is quadratic in the window because the gate is grown out of the same input the
    rescale multiplies; on the backward that input is not the cotangent. -/
noncomputable def floatBridgesTo_seBack {n : Nat} (M : FloatModel)
    (g fg xinp fx : Vec n) {Sg eg Sx ex : ℝ} (hn : 0 < n)
    (hg : ∀ i, |g i| ≤ Sg) (hfg : ∀ i, |fg i - g i| ≤ eg)
    (hx : ∀ i, |xinp i| ≤ Sx) (hfx : ∀ i, |fx i - xinp i| ≤ ex)
    {gateBack gateBackF : Vec n → Vec n} (hgateBack : FloatBridgesTo gateBack gateBackF) :
    FloatBridgesTo (seInputGrad g xinp gateBack) (seInputGradF M fg fx gateBackF) :=
  FloatBridgesTo.biPathSum M
    (floatBridgesTo_diagBack M g fg hn hg hfg)
    ((floatBridgesTo_diagBack M xinp fx hn hx hfx).comp hgateBack)

namespace FloatBridgesTo

/-- ⭐ **An envelope through the whole squeeze-excite GATE backward.** Six numeric stages —
    `broadcastBack → diagBack σ′ → linBack W₂ → diagBack swish′ → linBack W₁ → gapBack` — twelve
    inequalities.

    ⭐ Read stages 1 and 6 as a pair: `broadcastBack` multiplies the cotangent window by the
    reduce's `c·h·w` fan-in and `gapBack` divides by `h·w`, so what the gate path costs the fold
    is a factor of roughly `c` per site and NOT any nonlinearity — the four dense/scale stages in
    between are all magnitude-non-increasing at B0's profile. -/
theorem Maps.seGateBack {c h w r : Nat} (M : FloatModel)
    (W₁ : Mat c r) (W₂ : Mat r c) (ssig fssig : Vec c) (ssw fssw : Vec r)
    {w' Ssig esig Ssw eswish : ℝ} (hw' : 0 ≤ w')
    (hc : 0 < c) (hr : 0 < r) (hh : 0 < h) (hww : 0 < w)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hW₂ : ∀ i j, |W₂ i j| ≤ w')
    (hssig : ∀ i, |ssig i| ≤ Ssig) (hfssig : ∀ i, |fssig i - ssig i| ≤ esig)
    (hssw : ∀ i, |ssw i| ≤ Ssw) (hfssw : ∀ i, |fssw i - ssw i| ≤ eswish)
    {q gbc g2 g1 Ā Ē A1 E1 A2 E2 A3 E3 A4 E4 A5 E5 Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hSsig0 : 0 ≤ Ssig) (hesig0 : 0 ≤ esig) (hSsw0 : 0 ≤ Ssw) (heswish0 : 0 ≤ eswish)
    (hgbc : (1 + M.u) ^ (c * h * w + 1) - 1 ≤ gbc)
    (hg2 : (1 + M.u) ^ (c + 2) - 1 ≤ g2) (hg1 : (1 + M.u) ^ (r + 2) - 1 ≤ g1)
    (bcA : ((c * h * w : ℕ) : ℝ) * Ā + gbc * (((c * h * w : ℕ) : ℝ) * Ā) ≤ A1)
    (bcE : gbc * (((c * h * w : ℕ) : ℝ) * (Ā + Ē)) + ((c * h * w : ℕ) : ℝ) * Ē ≤ E1)
    (sigA : Ssig * A1 + FloatModel.mulErr q Ssig A1 esig 0 ≤ A2)
    (sigE : FloatModel.mulErr q Ssig A1 esig 0 + Ssig * E1 ≤ E2)
    (d2A : (1 + g2) * ((c : ℝ) * w' * A2 + 0) ≤ A3)
    (d2E : g2 * ((c : ℝ) * w' * (A2 + E2) + 0) + (c : ℝ) * w' * E2 ≤ E3)
    (swA : Ssw * A3 + FloatModel.mulErr q Ssw A3 eswish 0 ≤ A4)
    (swE : FloatModel.mulErr q Ssw A3 eswish 0 + Ssw * E3 ≤ E4)
    (d1A : (1 + g1) * ((r : ℝ) * w' * A4 + 0) ≤ A5)
    (d1E : g1 * ((r : ℝ) * w' * (A4 + E4) + 0) + (r : ℝ) * w' * E4 ≤ E5)
    (gpA : 1 / ((h : ℝ) * (w : ℝ)) * A5 + FloatModel.mulErr q (1 / ((h : ℝ) * (w : ℝ))) A5 0 0 ≤ Ā')
    (gpE : FloatModel.mulErr q (1 / ((h : ℝ) * (w : ℝ))) A5 0 0
            + 1 / ((h : ℝ) * (w : ℝ)) * E5 ≤ Ē') :
    (floatBridgesTo_seGateBack (h := h) (w := w) M W₁ W₂ ssig fssig ssw fssw hw' hc hr hh hww
      hW₁ hW₂ hssig hfssig hssw hfssw).Maps Ā Ē Ā' Ē' :=
  (((((Maps.broadcastBack (h := h) (w := w) M hc hgbc bcA bcE).comp hc
    (Maps.diagBack M ssig fssig hc hssig hfssig hq hSsig0 hesig0 sigA sigE)).comp hc
    (Maps.linBack M W₂ hw' hc hW₂ hg2 d2A d2E)).comp hr
    (Maps.diagBack M ssw fssw hr hssw hfssw hq hSsw0 heswish0 swA swE)).comp hr
    (Maps.linBack M W₁ hw' hr hW₁ hg1 d1A d1E)).comp hc
    (Maps.gapBack M c h w hc hh hww hq gpA gpE)

/-- ⭐⭐ **An envelope through the squeeze-excite BLOCK backward — the product rule.** Two
    branches over the SAME incoming envelope (`Ā, Ē`): the main path `diagBack g` at the saved
    gate, and the gate path `gateBack ∘ diagBack xinp` at the saved input, joined by a rounded
    add.

    ⚠ `Sx` — the bound on the SE's saved INPUT — is the one place this block still reaches for a
    magnitude the forward supplies, and it is the second of §3.9's two window imports. The first,
    `|swish′|`, was removed by `swishScalarDeriv_abs_le` (§3.12) and took the fold from 10⁴³¹ to
    10¹⁶⁹; this one is what an operating-point `|x| ≤ 16` would remove, and it is worth a further
    71 orders (`b0_back_chain(sx = 16)`). It is NOT taken: 10¹⁶⁹ is already 84 orders under
    §3.7(a)'s ~10²⁵³ ceiling, and §3.13's rule is that an operating point is what you pay when the
    fold does not fit, not a thing to buy for its own sake. -/
theorem Maps.seBack {n : Nat} (M : FloatModel)
    (g fg xinp fx : Vec n) {Sg eg Sx ex : ℝ} (hn : 0 < n)
    (hg : ∀ i, |g i| ≤ Sg) (hfg : ∀ i, |fg i - g i| ≤ eg)
    (hx : ∀ i, |xinp i| ≤ Sx) (hfx : ∀ i, |fx i - xinp i| ≤ ex)
    {gateBack gateBackF : Vec n → Vec n} (hgateBack : FloatBridgesTo gateBack gateBackF)
    {q Ā Ē Pd Ep A1 E1 Bd Ed Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hSg0 : 0 ≤ Sg) (heg0 : 0 ≤ eg) (hSx0 : 0 ≤ Sx) (hex0 : 0 ≤ ex)
    (mainA : Sg * Ā + FloatModel.mulErr q Sg Ā eg 0 ≤ Pd)
    (mainE : FloatModel.mulErr q Sg Ā eg 0 + Sg * Ē ≤ Ep)
    (preA : Sx * Ā + FloatModel.mulErr q Sx Ā ex 0 ≤ A1)
    (preE : FloatModel.mulErr q Sx Ā ex 0 + Sx * Ē ≤ E1)
    (mgate : hgateBack.Maps A1 E1 Bd Ed)
    (hĀ' : Pd + Bd + q * (Pd + Bd) ≤ Ā')
    (hĒ' : q * (Pd + Ep + Bd + Ed) + (Ep + Ed) ≤ Ē') :
    (floatBridgesTo_seBack M g fg xinp fx hn hg hfg hx hfx hgateBack).Maps Ā Ē Ā' Ē' :=
  Maps.biPathSum M hn
    (Maps.diagBack M g fg hn hg hfg hq hSg0 heg0 mainA mainE)
    ((Maps.diagBack M xinp fx hn hx hfx hq hSx0 hex0 preA preE).comp hn mgate)
    hq hĀ' hĒ'

end FloatBridgesTo

-- ════════════════════════════════════════════════════════════════
-- § The three MBConv body backwards, at named float peers
-- ════════════════════════════════════════════════════════════════

/-- The float no-expand MBConv body input-gradient (`b1`) — `mbNoExpBodyBack`'s deployed peer.
    `depthwiseBack ∘ seBack ∘ projectBack`, every stage the float map its bridge names. -/
noncomputable def mbNoExpBodyBackF {cin cout h w kHd kWd kHp kWp : Nat} (M : FloatModel)
    (Wd : DepthwiseKernel cin kHd kWd) (Wp : Kernel4 cout cin kHp kWp)
    (bnBdF swBdF seBF : Vec (cin * h * w) → Vec (cin * h * w))
    (bnBpF : Vec (cout * h * w) → Vec (cout * h * w)) :
    Vec (cout * h * w) → Vec (cin * h * w) :=
  (M.depthwiseFlatF (h := h) (w := w) (dwReverse Wd) (fun _ => 0) ∘ bnBdF ∘ swBdF)
  ∘ seBF
  ∘ (M.flatConvF (h := h) (w := w) (IR.reverseSwap Wp) (fun _ => 0) ∘ bnBpF)

/-- **The no-expand MBConv body backward float-bridges TO its float peer** (`b1`) — the
    `FloatBridgesTo` peer of `floatBridges_mbNoExpBodyBack`. The BN backs, the swish back and the
    squeeze-excite back are supplied (discharge with `floatBridgesTo_bnBack`,
    `floatBridgesTo_diagBack` and `floatBridgesTo_seBack`); the depthwise and project convs are
    concrete. -/
noncomputable def floatBridgesTo_mbNoExpBodyBack {cin cout h w kHd kWd kHp kWp : Nat}
    (M : FloatModel) (Wd : DepthwiseKernel cin kHd kWd) (Wp : Kernel4 cout cin kHp kWp)
    {bnBd swBd seB bnBdF swBdF seBF : Vec (cin * h * w) → Vec (cin * h * w)}
    {bnBp bnBpF : Vec (cout * h * w) → Vec (cout * h * w)}
    {wd wp : ℝ} (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd) (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hnI : 0 < cin * h * w) (hnO : 0 < cout * h * w)
    (hbnBd : FloatBridgesTo bnBd bnBdF) (hswBd : FloatBridgesTo swBd swBdF)
    (hseB : FloatBridgesTo seB seBF) (hbnBp : FloatBridgesTo bnBp bnBpF) :
    FloatBridgesTo (mbNoExpBodyBack Wd Wp bnBd swBd seB bnBp)
      (mbNoExpBodyBackF M Wd Wp bnBdF swBdF seBF bnBpF) :=
  ((hbnBp.comp (floatBridgesTo_convBack (h := h) (w := w) M Wp hwp hnO hWp)).comp hseB).comp
    ((hswBd.comp hbnBd).comp
      (floatBridgesTo_depthwiseBack (h := h) (w := w) M Wd hwd hnI hWd))

/-- The float strided MBConv body input-gradient (`b2`) — `mbStridedBodyBack`'s deployed peer.
    The expand arm sits at the PRE-downsample `2h × 2w`, and the depthwise stage threads the
    stride-2 backward (zero-upsample scatter, then the reversed-kernel depthwise). -/
noncomputable def mbStridedBodyBackF {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel) (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    (bnBeF swBeF : Vec (cmid * (2 * h) * (2 * w)) → Vec (cmid * (2 * h) * (2 * w)))
    (bnBdF swBdF seBF : Vec (cmid * h * w) → Vec (cmid * h * w))
    (bnBpF : Vec (cout * h * w) → Vec (cout * h * w)) :
    Vec (cout * h * w) → Vec (cin * (2 * h) * (2 * w)) :=
  (M.flatConvF (h := 2 * h) (w := 2 * w) (IR.reverseSwap We) (fun _ => 0) ∘ bnBeF ∘ swBeF)
  ∘ ((M.depthwiseFlatF (h := 2 * h) (w := 2 * w) (dwReverse Wd) (fun _ => 0)
        ∘ decimateBack cmid h w) ∘ bnBdF ∘ swBdF)
  ∘ seBF
  ∘ (M.flatConvF (h := h) (w := w) (IR.reverseSwap Wp) (fun _ => 0) ∘ bnBpF)

/-- **The strided MBConv body backward float-bridges TO its float peer** (`b2`) — the
    `FloatBridgesTo` peer of `floatBridges_mbStridedBodyBack`. Same shape as `b1`'s with the
    stride-2 depthwise backward and the expand arm at the doubled resolution. -/
noncomputable def floatBridgesTo_mbStridedBodyBack
    {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat} (M : FloatModel)
    (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    {bnBe swBe bnBeF swBeF : Vec (cmid * (2 * h) * (2 * w)) → Vec (cmid * (2 * h) * (2 * w))}
    {bnBd swBd seB bnBdF swBdF seBF : Vec (cmid * h * w) → Vec (cmid * h * w)}
    {bnBp bnBpF : Vec (cout * h * w) → Vec (cout * h * w)}
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hnM2 : 0 < cmid * (2 * h) * (2 * w)) (hnO : 0 < cout * h * w)
    (hbnBe : FloatBridgesTo bnBe bnBeF) (hswBe : FloatBridgesTo swBe swBeF)
    (hbnBd : FloatBridgesTo bnBd bnBdF) (hswBd : FloatBridgesTo swBd swBdF)
    (hseB : FloatBridgesTo seB seBF) (hbnBp : FloatBridgesTo bnBp bnBpF) :
    FloatBridgesTo (mbStridedBodyBack We Wd Wp bnBe swBe bnBd swBd seB bnBp)
      (mbStridedBodyBackF M We Wd Wp bnBeF swBeF bnBdF swBdF seBF bnBpF) :=
  (((hbnBp.comp (floatBridgesTo_convBack (h := h) (w := w) M Wp hwp hnO hWp)).comp hseB).comp
    ((hswBd.comp hbnBd).comp
      (floatBridgesTo_depthwiseStride2Back (h := h) (w := w) M Wd hwd hnM2 hWd))).comp
    ((hswBe.comp hbnBe).comp
      (floatBridgesTo_convBack (h := 2 * h) (w := 2 * w) M We hwe hnM2 hWe))

/-- The float full MBConv body input-gradient (`b3`'s body) — `mbconvBodyBack`'s deployed peer.
    The `b1` shape with the expand arm restored, all at one resolution. -/
noncomputable def mbconvBodyBackF {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel) (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    (bnBeF bnBdF swBeF swBdF seBF : Vec (cmid * h * w) → Vec (cmid * h * w))
    (bnBpF : Vec (cout * h * w) → Vec (cout * h * w)) :
    Vec (cout * h * w) → Vec (cin * h * w) :=
  (M.flatConvF (h := h) (w := w) (IR.reverseSwap We) (fun _ => 0) ∘ bnBeF ∘ swBeF)
  ∘ (M.depthwiseFlatF (h := h) (w := w) (dwReverse Wd) (fun _ => 0) ∘ bnBdF ∘ swBdF)
  ∘ seBF
  ∘ (M.flatConvF (h := h) (w := w) (IR.reverseSwap Wp) (fun _ => 0) ∘ bnBpF)

/-- **The full MBConv body backward float-bridges TO its float peer** (`b3`'s body) — the
    `FloatBridgesTo` peer of `floatBridges_mbconvBodyBack`. The caller wraps it in
    `FloatBridgesTo.residual` for a residual block, exactly as MobileNetV2's `b2`/`b4` do
    (§3.13: the block record does not have to own its skip). -/
noncomputable def floatBridgesTo_mbconvBodyBack {cin cmid cout h w kHe kWe kHd kWd kHp kWp : Nat}
    (M : FloatModel) (We : Kernel4 cmid cin kHe kWe) (Wd : DepthwiseKernel cmid kHd kWd)
    (Wp : Kernel4 cout cmid kHp kWp)
    {bnBe bnBd swBe swBd seB bnBeF bnBdF swBeF swBdF seBF :
      Vec (cmid * h * w) → Vec (cmid * h * w)}
    {bnBp bnBpF : Vec (cout * h * w) → Vec (cout * h * w)}
    {we wd wp : ℝ} (hwe : 0 ≤ we) (hwd : 0 ≤ wd) (hwp : 0 ≤ wp)
    (hWe : ∀ o c kh kw, |We o c kh kw| ≤ we) (hWd : ∀ ch kh kw, |Wd ch kh kw| ≤ wd)
    (hWp : ∀ o c kh kw, |Wp o c kh kw| ≤ wp)
    (hnM : 0 < cmid * h * w) (hnO : 0 < cout * h * w)
    (hbnBe : FloatBridgesTo bnBe bnBeF) (hbnBd : FloatBridgesTo bnBd bnBdF)
    (hswBe : FloatBridgesTo swBe swBeF) (hswBd : FloatBridgesTo swBd swBdF)
    (hseB : FloatBridgesTo seB seBF) (hbnBp : FloatBridgesTo bnBp bnBpF) :
    FloatBridgesTo (mbconvBodyBack We Wd Wp bnBe bnBd swBe swBd seB bnBp)
      (mbconvBodyBackF M We Wd Wp bnBeF bnBdF swBeF swBdF seBF bnBpF) :=
  (((hbnBp.comp (floatBridgesTo_convBack (h := h) (w := w) M Wp hwp hnO hWp)).comp hseB).comp
    ((hswBd.comp hbnBd).comp
      (floatBridgesTo_depthwiseBack (h := h) (w := w) M Wd hwd hnM hWd))).comp
    ((hswBe.comp hbnBe).comp
      (floatBridgesTo_convBack (h := h) (w := w) M We hwe hnM hWe))

-- ════════════════════════════════════════════════════════════════
-- § ⭐ The kit EXERCISED — B0's `b1` squeeze-excite site at `b0_back_chain`'s numerals
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **EfficientNet-B0's `b1` squeeze-excite backward, closed end to end at the numerals the
    generator emits** (`b0_back_chain(ssw = 2)`, `scripts/float_budget_envelope.py`): cotangent in
    `(1.720·10¹³⁸, 2.709·10¹³⁷)` — what the block's project-conv backward hands the SE — out
    `(7.834·10¹⁵², 1.699·10¹⁵²)`. Ten numeric stages: the product rule's two `diagBack`s, the six
    gate stages, and the rounded join.

    ⭐⭐ **And the saved swish derivative is the REAL one** — `fun i => swishScalarDeriv (xsw i)`
    at `Ssw = 2`, discharged by `swishScalarDeriv_abs_le`. That is the whole of §3.12 in one
    argument slot: the repo's only prior bound was `swishScalar_lipschitz_abs`'s `1 + A/4` at the
    forward's window, which is `1.216·10⁵¹` at B0's head alone and put this fold at 10⁴³¹.

    ⚠ §5's rule — a `Maps` leaf nothing composes is a leaf nobody has checked composes, and this
    is simultaneously the check that the generator's arithmetic IS these lemmas'. It caught the
    fan-in convention on `Maps.linBack`: the SE's two denses are `Mat 32 8` and `Mat 8 32`, so the
    reduce charges `32` and the expand `8`, not the other way round. -/
example (M : FloatModel) (hMu : M.u ≤ u32)
    (W₁ : Mat 32 8) (W₂ : Mat 8 32)
    (hW₁ : ∀ i j, |W₁ i j| ≤ 37 / 10) (hW₂ : ∀ i j, |W₂ i j| ≤ 37 / 10)
    (gate fgate xinp fx : Vec (32 * 112 * 112))
    (hg : ∀ i, |gate i| ≤ 101 / 100) (hfg : ∀ i, |fgate i - gate i| ≤ 1 / 100)
    (hx : ∀ i, |xinp i| ≤ 7572000000) (hfx : ∀ i, |fx i - xinp i| ≤ 1 / 100)
    (ssig fssig : Vec 32) (xsw fssw : Vec 8)
    (hssig : ∀ i, |ssig i| ≤ 1 / 4) (hfssig : ∀ i, |fssig i - ssig i| ≤ 1 / 100)
    (hfssw : ∀ i, |fssw i - swishScalarDeriv (xsw i)| ≤ 1 / 100) :
    (floatBridgesTo_seBack M gate fgate xinp fx (by norm_num) hg hfg hx hfx
      (floatBridgesTo_seGateBack (h := 112) (w := 112) M W₁ W₂ ssig fssig
        (fun i => swishScalarDeriv (xsw i)) fssw (Ssw := 2) (by norm_num)
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) hW₁ hW₂
        hssig hfssig (fun _i => swishScalarDeriv_abs_le _) hfssw)).Maps
      (1720 * 10 ^ 135) (2709 * 10 ^ 134) (7834 * 10 ^ 149) (1699 * 10 ^ 149) :=
  FloatBridgesTo.Maps.seBack M gate fgate xinp fx (by norm_num) hg hfg hx hfx _
    (q := u32) hMu (by norm_num) (by norm_num) (by norm_num) (by norm_num)
    (Pd := 1755 * 10 ^ 135) (Ep := 2909 * 10 ^ 134)
    (A1 := 1303 * 10 ^ 145) (E1 := 2052 * 10 ^ 144)
    (Bd := 7833 * 10 ^ 149) (Ed := 1698 * 10 ^ 149)
    (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
    (FloatBridgesTo.Maps.seGateBack (h := 112) (w := 112) M W₁ W₂ ssig fssig
      (fun i => swishScalarDeriv (xsw i)) fssw (Ssw := 2) (by norm_num)
      (by norm_num) (by norm_num) (by norm_num) (by norm_num) hW₁ hW₂
      hssig hfssig (fun _i => swishScalarDeriv_abs_le _) hfssw
      (q := u32) hMu (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (gbc := 613 / 25000) (g2 := 2027 / 10 ^ 9) (g1 := 5961 / 10 ^ 10)
      (M.gamma_num (k := 32 * 112 * 112 + 1) (q := 613 / 25000) hMu
        (by norm_num [u32]) (by norm_num [u32]))
      (M.gamma_num (k := 32 + 2) (q := 2027 / 10 ^ 9) hMu
        (by norm_num [u32]) (by norm_num [u32]))
      (M.gamma_num (k := 8 + 2) (q := 5961 / 10 ^ 10) hMu
        (by norm_num [u32]) (by norm_num [u32]))
      (A1 := 5359 * 10 ^ 150) (E1 := 9722 * 10 ^ 149)
      (A2 := 1394 * 10 ^ 150) (E2 := 2967 * 10 ^ 149)
      (A3 := 1651 * 10 ^ 152) (E3 := 3513 * 10 ^ 151)
      (A4 := 3319 * 10 ^ 152) (E4 := 7192 * 10 ^ 151)
      (A5 := 9825 * 10 ^ 153) (E5 := 2129 * 10 ^ 153)
      (by norm_num) (by norm_num)
      (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
      (by norm_num) (by norm_num)
      (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])
      (by norm_num) (by norm_num)
      (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
    (by norm_num [u32]) (by norm_num [u32])

end Proofs
