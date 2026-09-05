import LeanMlir.Proofs.Float.FloatBudgetEnvBackLN
import LeanMlir.Proofs.Foundation.EvenKernelConvBack

/-! # A NUMBER for a LAYERNORM net's whole-net BACKWARD — ConvNeXt-T, and it is the FOLD

The third whole-net input-gradient number (`Resnet34BackFloatBudget.lean`,
`MobileNetV2BackFloatBudget.lean`), and the first for a net whose FORWARD number is a cap.

    certified window ≤ 1.023·10²⁵¹      (`cnxGradBridge_mag_le`)
    fresh budget     ≤ 1.563·10²⁵⁰      (`cnxGradBridge_fresh_le`)
    budget / window  = 0.153            — ⭐ the interval FOLD, no `capped` anywhere

at the measured 300-epoch per-kind profile (conv/dense kernels `≤ 6/10`, LayerNorm γ `≤ 48/10`,
layer scale `≤ 84/10`), `ε ≥ 10⁻⁵`, `u ≤ 2⁻²⁴`, saved-activation accuracies `10⁻²`, at the
operating point `|istd| ≤ 16`, on loss cotangents of magnitude `≤ 1`. 137 numeric stages, 322
rational inequalities, generated and re-asserted by `cnx_back_chain` / `verify_cnx_back`
(`scripts/float_budget_envelope.py`) before a line of this file was written.

⭐⭐ **This is the repo's first honest whole-net FOLD for a LayerNorm net**, and the point is the
contrast with the same net's forward: `cnx_float_logits_le` is `budget/window = 2.00`, the triangle
inequality, because LayerNorm reduces its statistics out of its own input and has no eval mode to
freeze them in (`planning/archive/float_budget_numbers_log.md` §0.1). A VJP reads those statistics off the
SAVED activations, which the cotangent does not perturb, so the quadratic never appears and the
fold exists. §0.1's list of quadratic sites is a FORWARD-only list.

⛔⛔ **READ THIS BEFORE QUOTING THE NUMBER — it does not compose with the forward, and not for the
reason r34's and MobileNetV2's do not.** On those two the caveat is quantitative: `es`/`exh` are
supplied at `10⁻²` and their own training-mode forward fold says `10⁷⁴¹⁷`, but both nets have an
INFERENCE mode in which the forward statement is itself a fold. ConvNeXt has no such mode. Its
forward's certified statement is `FloatBridgesTo.capped`, whose modulus is `2·mag` **by
construction**, so the saved-activation accuracy it supplies is `2 × window ≈ 10²²⁷` and no
tightening of the profile or the operating point moves it below that. So the honest reading is:
*an honest fold of the BACKWARD kernel's rounding, at a hypothesised operating point, given
saved-activation accuracies this net's forward cannot supply in any mode.* It is not, and cannot
be assembled into, a statement about a deployed forward-then-backward composition (§9).

⭐⭐ **What the number IS about is settled**: `convnextInputGrad_eq_convNextForwardTCh_vjp`
(`Foundation/ConvNeXtWholeBackCertifiedTie.lean`) says this exact term, at these slots, IS
`(convNextForwardTCh_has_vjp …).backward x`. ⭐ Unlike ResNet-34 and MobileNetV2, whose budget files
define their own `*GradR` skeleton, this one is stated **directly on `convnextInputGrad`** — the
committed backward the tie names — so nothing stands between the number and the certified gradient.

⭐ **Three leaf facts carry it, and all three were already in the repo** (§3.3.0(b) for the eighth
time): `bnXhat_sq_le`'s `|x̂| ≤ √n` at the CHANNEL count (without it the fold is 10⁵¹⁴⁷),
`geluScalarDeriv_abs_le`'s global `|gelu′| ≤ 3/2` (without it, 10⁶³³⁰ — the exact shape of B0's
swish blocker, one net over, and ConvNeXt's own FORWARD ablation had measured this constant as
*not* load-bearing), and `padOdd`'s repair of the even-kernel conv backward, whose fan-ins this
file charges at `cout·3·3` and `96·5·5`.

⚠ **The operating point is paid, unlike MobileNetV2's.** At the unconditional `ε`-floor
`|istd| ≤ 317` the same fold is 6.847·10²⁸⁰, past §3.7(a)'s shape-dependent `norm_num` wall;
`|istd| ≤ 16` buys 30 orders. §3.13's rule for the fourth time: an operating-point hypothesis is
not a property of backwards, it is what you pay when the `ε`-floor fold does not fit. -/

namespace Proofs

open FloatModel
open Classical

-- ════════════════════════════════════════════════════════════════
-- § The numeric profile, and the net's backward data
-- ════════════════════════════════════════════════════════════════

/-- The numeric profile the ConvNeXt-T backward fold runs at. ⚠ `S` is the OPERATING POINT — an
    assumption about the saved activations (§0.1's escape 2), not a consequence of `hε5`. -/
structure CnxBackProfile (M : FloatModel) (ε wk gl sl S es exh esav q : ℝ) : Prop where
  hwk : 0 ≤ wk
  hgl : 0 ≤ gl
  hsl : 0 ≤ sl
  hS0 : 0 ≤ S
  hes : 0 ≤ es
  hexh : 0 ≤ exh
  hesav : 0 ≤ esav
  hq : M.u ≤ q
  hε : 0 < ε

/-- A conv / patchify / downsample kernel with its magnitude bound. -/
structure CnxKerB (oc ic kH kW : Nat) (wk : ℝ) where
  W : Kernel4 oc ic kH kW
  hW : ∀ o c kh kw, |W o c kh kw| ≤ wk

/-- A depthwise kernel with its magnitude bound. -/
structure CnxDwKerB (c kH kW : Nat) (wk : ℝ) where
  W : DepthwiseKernel c kH kW
  hW : ∀ ch kh kw, |W ch kh kw| ≤ wk

/-- The classifier. -/
structure CnxHeadB (m n : Nat) (wk : ℝ) where
  W : Mat m n
  hW : ∀ i j, |W i j| ≤ wk

/-- One CHANNEL-LayerNorm backward site's data: γ, the saved forward activation it normalised,
    and the deployed float inverse-stddev and normalised activation read off it. ⚠ `hSabs` is the
    operating point; `hxh` is NOT a field — `bnXhat_abs_le_num` derives it from `bnXhat_sq_le`. -/
structure CnxLnBack (c h w : Nat) (ε gl S es exh : ℝ) where
  γ : Vec c
  fγ : Vec c
  x : Vec (c * h * w)
  fs : Fin (h * w) → ℝ
  fxh : Fin (h * w) → Vec c
  hγ : ∀ i, |γ i| ≤ gl
  hfγ : ∀ i, |fγ i - γ i| ≤ 0
  hst : ∀ r, |fs r - bnIstd c (Mat.unflatten (chanLNRows c h w x) r) ε| ≤ es
  hSabs : ∀ r, |bnIstd c (Mat.unflatten (chanLNRows c h w x) r) ε| ≤ S
  hfxh : ∀ r i, |fxh r i - bnXhat c ε (Mat.unflatten (chanLNRows c h w x) r) i| ≤ exh

/-- The HEAD LayerNorm backward site — `rowLNVecFlat 1 768`, one row, so the vector-LN spelling
    rather than the channel one. -/
structure CnxRowLnBack (s c : Nat) (ε gl S es exh : ℝ) where
  γ : Vec c
  fγ : Vec c
  X : Vec (s * c)
  fs : Fin s → ℝ
  fxh : Fin s → Vec c
  hγ : ∀ i, |γ i| ≤ gl
  hfγ : ∀ i, |fγ i - γ i| ≤ 0
  hst : ∀ r, |fs r - bnIstd c (Mat.unflatten X r) ε| ≤ es
  hSabs : ∀ r, |bnIstd c (Mat.unflatten X r) ε| ≤ S
  hfxh : ∀ r i, |fxh r i - bnXhat c ε (Mat.unflatten X r) i| ≤ exh

/-- One ConvNeXt block's backward data: the three kernels, the LayerNorm site, the layer scale
    (a stored weight, so its float peer IS itself) and the saved GELU derivative. -/
structure CnxBlockBack (c cExp h w : Nat) (ε wk gl sl S es exh esav : ℝ) where
  kdw : CnxDwKerB c 7 7 wk
  kex : CnxKerB cExp c 1 1 wk
  kpr : CnxKerB c cExp 1 1 wk
  ln : CnxLnBack c h w ε gl S es exh
  γls : Vec (c * h * w)
  hγls : ∀ i, |γls i| ≤ sl
  sge : Vec (cExp * h * w)
  fsge : Vec (cExp * h * w)
  hsge : ∀ i, |sge i| ≤ 3 / 2
  hfsge : ∀ i, |fsge i - sge i| ≤ esav

/-- One stage-boundary downsample's backward data. ⛔ The kernel is the committed `2×2`; the
    backward runs it through `padOdd` — `convFlatBack` is the adjoint only at an ODD kernel
    (`Foundation/EvenKernelConvBack.lean`), so the fan-in a numeral charges is `cout·3·3`. -/
structure CnxDownB (cin cout h w : Nat) (ε wk gl S es exh : ℝ) where
  k : CnxKerB cout cin 2 2 wk
  ln : CnxLnBack cin (2 * h) (2 * w) ε gl S es exh

/-- **The whole net's backward data** — 23 LayerNorm backward sites, each with its saved
    activation, and the 4×4/s4 patchify stem's kernel (also even, also through `padOdd`). -/
structure CnxBackWeights (ε wk gl sl S es exh esav : ℝ) where
  sW : CnxKerB 96 3 4 4 wk
  lnStem : CnxLnBack 96 56 56 ε gl S es exh
  s1 : Fin 3 → CnxBlockBack 96 384 56 56 ε wk gl sl S es exh esav
  d1 : CnxDownB 96 192 28 28 ε wk gl S es exh
  s2 : Fin 3 → CnxBlockBack 192 768 28 28 ε wk gl sl S es exh esav
  d2 : CnxDownB 192 384 14 14 ε wk gl S es exh
  s3 : Fin 9 → CnxBlockBack 384 1536 14 14 ε wk gl sl S es exh esav
  d3 : CnxDownB 384 768 7 7 ε wk gl S es exh
  s4 : Fin 3 → CnxBlockBack 768 3072 7 7 ε wk gl sl S es exh esav
  lnHead : CnxRowLnBack 1 768 ε gl S es exh
  fc : CnxHeadB 768 10 wk

-- ════════════════════════════════════════════════════════════════
-- § The real net, its float peer, and the closed bridge
-- ════════════════════════════════════════════════════════════════

section Net

variable {ε wk gl sl S es exh esav q : ℝ}

/-- The certified channel-LayerNorm backward at this site. -/
noncomputable def CnxLnBack.real {c h w : Nat} (s : CnxLnBack c h w ε gl S es exh) :
    Vec (c * h * w) → Vec (c * h * w) := chanLNTensor3Back c h w ε s.γ s.x

/-- Its deployed float peer, at the supplied float statistics. -/
noncomputable def CnxLnBack.float {c h w : Nat} (s : CnxLnBack c h w ε gl S es exh)
    (M : FloatModel) : Vec (c * h * w) → Vec (c * h * w) :=
  chanLNTensor3BackF M s.fγ s.fs s.fxh

/-- The site's bridge. ⭐ `hxh` is not a field: `bnXhat_abs_le_num` turns `bnXhat_sq_le`'s
    `x̂² ≤ c` into the rational `Xh`, at the CEILING root (`√96 → 10`, …, `√768 → 28`) since none
    of ConvNeXt's four channel counts is a square. -/
noncomputable def CnxLnBack.bridge {c h w : Nat} (s : CnxLnBack c h w ε gl S es exh)
    (M : FloatModel) (P : CnxBackProfile M ε wk gl sl S es exh esav q)
    (hhw : 0 < h * w) (hc : 0 < c)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((c : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo s.real (s.float M) :=
  floatBridgesTo_chanLNTensor3Back M s.γ s.fγ s.x s.fs s.fxh hhw hc s.hγ s.hfγ s.hst s.hSabs
    (fun _r i => bnXhat_abs_le_num (X := Xh) P.hε _ hXh0 hnX i) s.hfxh

/-- The head LayerNorm's certified backward. -/
noncomputable def CnxRowLnBack.real {s c : Nat} (t : CnxRowLnBack s c ε gl S es exh) :
    Vec (s * c) → Vec (s * c) := rowLNVecFlatBack s c ε t.γ t.X

/-- Its deployed float peer. -/
noncomputable def CnxRowLnBack.float {s c : Nat} (t : CnxRowLnBack s c ε gl S es exh)
    (M : FloatModel) : Vec (s * c) → Vec (s * c) := rowLNVecFlatBackF M t.fγ t.fs t.fxh

/-- The head site's bridge. -/
noncomputable def CnxRowLnBack.bridge {s c : Nat} (t : CnxRowLnBack s c ε gl S es exh)
    (M : FloatModel) (P : CnxBackProfile M ε wk gl sl S es exh esav q)
    (hs0 : 0 < s) (hc : 0 < c)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((c : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo t.real (t.float M) :=
  floatBridgesTo_rowLNVecFlatBack M t.γ t.fγ t.X t.fs t.fxh hs0 hc t.hγ t.hfγ t.hst t.hSabs
    (fun _r i => bnXhat_abs_le_num (X := Xh) P.hε _ hXh0 hnX i) t.hfxh

/-- One block's certified body backward, under its additive skip. -/
noncomputable def CnxBlockBack.real {c cExp h w : Nat}
    (b : CnxBlockBack c cExp h w ε wk gl sl S es exh esav) :
    Vec (c * h * w) → Vec (c * h * w) :=
  Proofs.residual (cnxBlockBodyBack b.kdw.W b.kex.W b.kpr.W b.ln.real (diagBack b.γls)
    (diagBack b.sge))

/-- Its deployed float peer. -/
noncomputable def CnxBlockBack.float {c cExp h w : Nat}
    (b : CnxBlockBack c cExp h w ε wk gl sl S es exh esav) (M : FloatModel) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun v j => M.add (cnxBlockBodyBackF M b.kdw.W b.kex.W b.kpr.W (b.ln.float M)
    (M.diagBackF b.γls) (M.diagBackF b.fsge) v j) (v j)

/-- The block's bridge, closed at real weights. -/
noncomputable def CnxBlockBack.bridge {c cExp h w : Nat}
    (b : CnxBlockBack c cExp h w ε wk gl sl S es exh esav) (M : FloatModel)
    (P : CnxBackProfile M ε wk gl sl S es exh esav q)
    (hnC : 0 < c * h * w) (hnE : 0 < cExp * h * w) (hhw : 0 < h * w) (hc : 0 < c)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((c : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo b.real (b.float M) :=
  (floatBridgesTo_cnxBlockBodyBack M b.kdw.W b.kex.W b.kpr.W P.hwk P.hwk P.hwk
    b.kdw.hW b.kex.hW b.kpr.hW hnC hnE (b.ln.bridge M P hhw hc hXh0 hnX)
    (floatBridgesTo_diagBack M b.γls b.γls (es := 0) hnC b.hγls (fun _ => by simp))
    (floatBridgesTo_diagBack M b.sge b.fsge hnE b.hsge b.hfsge)).residual M

/-- One downsample's certified backward — ⛔ at `padOdd`. -/
noncomputable def CnxDownB.real {cin cout h w : Nat} (dn : CnxDownB cin cout h w ε wk gl S es exh) :
    Vec (cout * h * w) → Vec (cin * (2 * h) * (2 * w)) :=
  cnxDownBack (h := h) (w := w) (padOdd dn.k.W) dn.ln.real

/-- Its deployed float peer. -/
noncomputable def CnxDownB.float {cin cout h w : Nat}
    (dn : CnxDownB cin cout h w ε wk gl S es exh) (M : FloatModel) :
    Vec (cout * h * w) → Vec (cin * (2 * h) * (2 * w)) :=
  cnxDownBackF (h := h) (w := w) M (padOdd dn.k.W) (dn.ln.float M)

/-- The downsample's bridge. -/
noncomputable def CnxDownB.bridge {cin cout h w : Nat}
    (dn : CnxDownB cin cout h w ε wk gl S es exh) (M : FloatModel)
    (P : CnxBackProfile M ε wk gl sl S es exh esav q)
    (hn : 0 < cout * (2 * h) * (2 * w)) (hhw : 0 < (2 * h) * (2 * w)) (hc : 0 < cin)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((cin : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo dn.real (dn.float M) :=
  floatBridgesTo_cnxDownBack (h := h) (w := w) M (padOdd dn.k.W) P.hwk
    (padOdd_abs_le dn.k.W P.hwk dn.k.hW) hn (dn.ln.bridge M P hhw hc hXh0 hnX)


/-- Stage `s4`'s certified backward: 3 block backwards, block `0`'s applied LAST because the
    forward applies it first. -/
noncomputable def cnxs4Back (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec (768 * 7 * 7) → Vec (768 * 7 * 7) :=
  (w.s4 0).real ∘ (w.s4 1).real ∘ (w.s4 2).real

/-- Its deployed float peer. -/
noncomputable def cnxs4BackF (M : FloatModel) (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec (768 * 7 * 7) → Vec (768 * 7 * 7) :=
  (w.s4 0).float M ∘ (w.s4 1).float M ∘ (w.s4 2).float M

/-- Stage `s3`'s certified backward: 9 block backwards, block `0`'s applied LAST because the
    forward applies it first. -/
noncomputable def cnxs3Back (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec (384 * 14 * 14) → Vec (384 * 14 * 14) :=
  (w.s3 0).real ∘ (w.s3 1).real ∘ (w.s3 2).real ∘ (w.s3 3).real ∘ (w.s3 4).real ∘ (w.s3 5).real ∘
    (w.s3 6).real ∘ (w.s3 7).real ∘ (w.s3 8).real

/-- Its deployed float peer. -/
noncomputable def cnxs3BackF (M : FloatModel) (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec (384 * 14 * 14) → Vec (384 * 14 * 14) :=
  (w.s3 0).float M ∘ (w.s3 1).float M ∘ (w.s3 2).float M ∘ (w.s3 3).float M ∘ (w.s3 4).float M ∘
    (w.s3 5).float M ∘ (w.s3 6).float M ∘ (w.s3 7).float M ∘ (w.s3 8).float M

/-- Stage `s2`'s certified backward: 3 block backwards, block `0`'s applied LAST because the
    forward applies it first. -/
noncomputable def cnxs2Back (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec (192 * 28 * 28) → Vec (192 * 28 * 28) :=
  (w.s2 0).real ∘ (w.s2 1).real ∘ (w.s2 2).real

/-- Its deployed float peer. -/
noncomputable def cnxs2BackF (M : FloatModel) (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec (192 * 28 * 28) → Vec (192 * 28 * 28) :=
  (w.s2 0).float M ∘ (w.s2 1).float M ∘ (w.s2 2).float M

/-- Stage `s1`'s certified backward: 3 block backwards, block `0`'s applied LAST because the
    forward applies it first. -/
noncomputable def cnxs1Back (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec (96 * 56 * 56) → Vec (96 * 56 * 56) :=
  (w.s1 0).real ∘ (w.s1 1).real ∘ (w.s1 2).real

/-- Its deployed float peer. -/
noncomputable def cnxs1BackF (M : FloatModel) (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec (96 * 56 * 56) → Vec (96 * 56 * 56) :=
  (w.s1 0).float M ∘ (w.s1 1).float M ∘ (w.s1 2).float M

/-- Stage `s4`'s bridge. -/
noncomputable def cnxs4BackBridge (M : FloatModel)
    (P : CnxBackProfile M ε wk gl sl S es exh esav q)
    (w : CnxBackWeights ε wk gl sl S es exh esav) :
    FloatBridgesTo (cnxs4Back w) (cnxs4BackF M w) :=
  (((w.s4 2).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 28) (by norm_num) (by norm_num)).comp
    ((w.s4 1).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 28) (by norm_num) (by norm_num))
      ).comp
    ((w.s4 0).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 28) (by norm_num) (by norm_num))

/-- Stage `s3`'s bridge. -/
noncomputable def cnxs3BackBridge (M : FloatModel)
    (P : CnxBackProfile M ε wk gl sl S es exh esav q)
    (w : CnxBackWeights ε wk gl sl S es exh esav) :
    FloatBridgesTo (cnxs3Back w) (cnxs3BackF M w) :=
  (((((((((w.s3 8).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num) (by norm_num)).comp
    ((w.s3 7).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num) (by norm_num))
      ).comp
    ((w.s3 6).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num) (by norm_num))
      ).comp
    ((w.s3 5).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num) (by norm_num))
      ).comp
    ((w.s3 4).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num) (by norm_num))
      ).comp
    ((w.s3 3).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num) (by norm_num))
      ).comp
    ((w.s3 2).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num) (by norm_num))
      ).comp
    ((w.s3 1).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num) (by norm_num))
      ).comp
    ((w.s3 0).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num) (by norm_num))

/-- Stage `s2`'s bridge. -/
noncomputable def cnxs2BackBridge (M : FloatModel)
    (P : CnxBackProfile M ε wk gl sl S es exh esav q)
    (w : CnxBackWeights ε wk gl sl S es exh esav) :
    FloatBridgesTo (cnxs2Back w) (cnxs2BackF M w) :=
  (((w.s2 2).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 14) (by norm_num) (by norm_num)).comp
    ((w.s2 1).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 14) (by norm_num) (by norm_num))
      ).comp
    ((w.s2 0).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 14) (by norm_num) (by norm_num))

/-- Stage `s1`'s bridge. -/
noncomputable def cnxs1BackBridge (M : FloatModel)
    (P : CnxBackProfile M ε wk gl sl S es exh esav q)
    (w : CnxBackWeights ε wk gl sl S es exh esav) :
    FloatBridgesTo (cnxs1Back w) (cnxs1BackF M w) :=
  (((w.s1 2).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 10) (by norm_num) (by norm_num)).comp
    ((w.s1 1).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 10) (by norm_num) (by norm_num))
      ).comp
    ((w.s1 0).bridge M P (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Xh := 10) (by norm_num) (by norm_num))

/-- ⭐ **The whole ConvNeXt-T input-gradient, at the COMMITTED `convnextInputGrad`** — every slot
    pinned to the certified per-op backward at its own saved activation. ⛔ The stem and the three
    downsamples run at `padOdd`; `convnextInputGrad_eq_convNextForwardTCh_vjp` is the theorem that
    says this term IS `(convNextForwardTCh_has_vjp …).backward`. -/
noncomputable def cnxGradR (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec 10 → Vec (3 * 224 * 224) :=
  convnextInputGrad w.fc.W (padOdd w.sW.W) w.lnStem.real w.lnHead.real
    (cnxs1Back w) (w.d1.real) (cnxs2Back w) (w.d2.real) (cnxs3Back w) (w.d3.real) (cnxs4Back w)

/-- The deployed float peer, at the committed `convnextInputGradF`. -/
noncomputable def cnxGradF (M : FloatModel) (w : CnxBackWeights ε wk gl sl S es exh esav) :
    Vec 10 → Vec (3 * 224 * 224) :=
  convnextInputGradF M w.fc.W (padOdd w.sW.W) (w.lnStem.float M) (w.lnHead.float M)
    (cnxs1BackF M w) (w.d1.float M) (cnxs2BackF M w) (w.d2.float M) (cnxs3BackF M w)
    (w.d3.float M) (cnxs4BackF M w)

set_option maxRecDepth 400000 in
set_option maxHeartbeats 2000000 in
/-- ⭐ **The whole ConvNeXt-T input-gradient VJP float-bridges TO its float peer, CLOSED** — all 23
    LayerNorm backwards, 18 blocks and 3 downsamples discharged at the record's real data, nothing
    left but `es`/`exh` and the operating point. ⚠ Grouped exactly as `convnextInputGrad` groups. -/
noncomputable def cnxGradBridge (M : FloatModel)
    (P : CnxBackProfile M ε wk gl sl S es exh esav q)
    (w : CnxBackWeights ε wk gl sl S es exh esav) :
    FloatBridgesTo (cnxGradR w) (cnxGradF M w) :=
  ((((((((((floatBridgesTo_linBack M w.fc.W P.hwk (by norm_num) w.fc.hW).comp
      (w.lnHead.bridge M P (by norm_num) (by norm_num) (Xh := 28) (by norm_num)
        (by norm_num))).comp
      (floatBridgesTo_gapBack M 768 7 7 (by norm_num) (by norm_num) (by norm_num))).comp
      (cnxs4BackBridge M P w)).comp
      (w.d3.bridge M P (by norm_num) (by norm_num) (by norm_num) (Xh := 20) (by norm_num)
        (by norm_num))).comp
      (cnxs3BackBridge M P w)).comp
      (w.d2.bridge M P (by norm_num) (by norm_num) (by norm_num) (Xh := 14) (by norm_num)
        (by norm_num))).comp
      (cnxs2BackBridge M P w)).comp
      (w.d1.bridge M P (by norm_num) (by norm_num) (by norm_num) (Xh := 10) (by norm_num)
        (by norm_num))).comp
      (cnxs1BackBridge M P w)).comp
      ((w.lnStem.bridge M P (by norm_num) (by norm_num) (Xh := 10) (by norm_num)
        (by norm_num)).comp
        (floatBridgesTo_flatConvStride4Back (h := 56) (w := 56) M (padOdd w.sW.W) P.hwk
          (by norm_num) (padOdd_abs_le w.sW.W P.hwk w.sW.hW)))

end Net

-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- The measured 300-epoch per-kind profile, at the operating point `|istd| ≤ 16`.
    ⭐ The per-kind split is worth **68 orders** here — by far the largest of any net in this file
    — because ConvNeXt's outlier kind is the LAYER SCALE (8.38 against the conv kernels' 0.60),
    and it multiplies inside every block, so every conv fan-in in the chain would pay for a
    uniform bound. At a uniform `84/10` the same fold is 2.406·10³¹⁷ and there is no theorem. -/
theorem cnxBackProfile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    CnxBackProfile M ε (6/10) (48/10) (84/10) 16 (1/100) (1/100) (1/100) u32 where
  hwk := by norm_num
  hgl := by norm_num
  hsl := by norm_num
  hS0 := by norm_num
  hes := by norm_num
  hexh := by norm_num
  hesav := by norm_num
  hq := hMu
  hε := by linarith


set_option maxRecDepth 400000 in
set_option maxHeartbeats 4000000 in
/-- ⭐⭐ **The 137-stage envelope for ConvNeXt-T's whole-net input gradient**, at the committed
    profile and `|istd| ≤ 16`. Every numeral is `cnx_back_chain(pad_odd = True, S = 16)`'s, and
    every one of the 322 rounded inequalities was re-asserted exactly by `verify_cnx_back` before
    this file was written. ⭐ Four `Kr`/`Kb` pairs serve all 23 LayerNorm sites — ConvNeXt has only
    four distinct reduction widths, and `Maps.rowLNVecFlatBack`'s per-unit-gain form is homogeneous
    of degree 1 in the window, so the expensive constant is evaluated once per width. -/
theorem cnxGradBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : CnxBackWeights ε (6/10) (48/10) (84/10) 16 (1/100) (1/100) (1/100)) :
    (cnxGradBridge M (cnxBackProfile_committed M hMu hε5) w).Maps 1 0
      (1023 * 10 ^ 248) (1563 * 10 ^ 247) := by
  have P := cnxBackProfile_committed M hMu hε5
  have hK96r : bnGradInputReMag 96 1 1 16 10 ≤ 1632 := by
    norm_num [bnGradInputReMag]
  have hK96b : bnGradInputBudgetG u32 (5782 / 10 ^ 9) 96 1 1 16 10
      (1/100) (1/100) ≤ 4234 / 10 ^ 3 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have hK192r : bnGradInputReMag 192 1 1 16 14 ≤ 3168 := by
    norm_num [bnGradInputReMag]
  have hK192b : bnGradInputBudgetG u32 (1151 / 10 ^ 8) 192 1 1 16 14
      (1/100) (1/100) ≤ 6502 / 10 ^ 3 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have hK384r : bnGradInputReMag 384 1 1 16 20 ≤ 6432 := by
    norm_num [bnGradInputReMag]
  have hK384b : bnGradInputBudgetG u32 (2295 / 10 ^ 8) 384 1 1 16 20
      (1/100) (1/100) ≤ 1058 / 10 ^ 2 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have hK768r : bnGradInputReMag 768 1 1 16 28 ≤ 1258 * 10 ^ 1 := by
    norm_num [bnGradInputReMag]
  have hK768b : bnGradInputBudgetG u32 (4584 / 10 ^ 8) 768 1 1 16 28
      (1/100) (1/100) ≤ 1741 / 10 ^ 2 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
      bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have m0 := (FloatBridgesTo.Maps.linBack M w.fc.W P.hwk (by norm_num) w.fc.hW
    (M.gamma_num (q := 7153 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 6001 / 10 ^ 3) (Ē' := 4292 / 10 ^ 9)
    (by norm_num [u32]) (by norm_num [u32])).comp (by norm_num)
    (FloatBridgesTo.Maps.rowLNVecFlatBack M w.lnHead.γ w.lnHead.fγ w.lnHead.X w.lnHead.fs
      w.lnHead.fxh (by norm_num) (by norm_num) w.lnHead.hγ w.lnHead.hfγ w.lnHead.hst w.lnHead.hSabs
      (fun _r i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i) w.lnHead.hfxh
      hMu (M.gamma_num (q := 4584 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
      (Kr := 1258 * 10 ^ 1) (Kb := 1741 / 10 ^ 2) hK768r hK768b (by norm_num) (by norm_num)
      (Ā := 6001 / 10 ^ 3) (Ē := 4292 / 10 ^ 9) (Ā' := 3629 * 10 ^ 2) (Ē' := 5018 / 10 ^ 1)
      (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
  have m1 := m0.comp (by norm_num)
    (FloatBridgesTo.Maps.gapBack M 768 7 7 (by norm_num) (by norm_num) (by norm_num) P.hq
      (Ā := 3629 * 10 ^ 2) (Ē := 5018 / 10 ^ 1)
      (Ā' := 7407) (Ē' := 1025 / 10 ^ 2) (by norm_num [FloatModel.mulErr, u32])
        (by norm_num [FloatModel.mulErr, u32]) )
  have m2 := m1.comp (by norm_num)
    (((FloatBridgesTo.Maps.residual M (m := 768 * 7 * 7) (by norm_num)
        (Ā := 7407) (Ē := 1025 / 10 ^ 2) (Bd := 1420 * 10 ^ 14) (Ed := 1364 * 10 ^ 12)
        (Ā' := 1421 * 10 ^ 14) (Ē' := 1365 * 10 ^ 12)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 7) (w := 7) M (w.s4 2).kdw.W (w.s4 2).kex.W (w.s4 2).kpr.W
          P.hwk P.hwk P.hwk (w.s4 2).kdw.hW (w.s4 2).kex.hW (w.s4 2).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 4590 / 10 ^ 8) (gex := 1833 / 10 ^ 7) (gdw := 3040 / 10 ^ 9)
          (A1 := 6222 * 10 ^ 1) (E1 := 8611 / 10 ^ 2) (A2 := 2868 * 10 ^ 4) (E2 := 4100 * 10 ^ 1)
          (A3 := 4331 * 10 ^ 4) (E3 := 3484 * 10 ^ 2) (A4 := 7985 * 10 ^ 7) (E4 := 6570 * 10 ^ 5)
          (A5 := 4829 * 10 ^ 12) (E5 := 4635 * 10 ^ 10)
          (Ā := 7407) (Ē := 1025 / 10 ^ 2) (Ā' := 1420 * 10 ^ 14) (Ē' := 1364 * 10 ^ 12)
          (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s4 2).γls (w.s4 2).γls (es := 0) (by norm_num) (w.s4 2).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 7407) (Ē := 1025 / 10 ^ 2) (Ā' := 6222 * 10 ^ 1) (Ē' := 8611 / 10 ^ 2)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s4 2).sge (w.s4 2).fsge (by norm_num) (w.s4 2).hsge (w.s4 2).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 2868 * 10 ^ 4) (Ē := 4100 * 10 ^ 1) (Ā' := 4331 * 10 ^ 4) (Ē' := 3484 * 10 ^ 2)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 7) (w := 7) M (w.s4 2).ln.γ (w.s4 2).ln.fγ (w.s4 2).ln.x (w.s4 2).ln.fs (w.s4 2).ln.fxh
              (by norm_num) (by norm_num) (w.s4 2).ln.hγ (w.s4 2).ln.hfγ (w.s4 2).ln.hst
                (w.s4 2).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
                (w.s4 2).ln.hfxh
              hMu (M.gamma_num (q := 4584 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 1258 * 10 ^ 1) (Kb := 1741 / 10 ^ 2) hK768r hK768b (by norm_num) (by norm_num)
              (Ā := 7985 * 10 ^ 7) (Ē := 6570 * 10 ^ 5) (Ā' := 4829 * 10 ^ 12)
                (Ē' := 4635 * 10 ^ 10)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32])).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 768 * 7 * 7) (by norm_num)
        (Ā := 1421 * 10 ^ 14) (Ē := 1365 * 10 ^ 12) (Bd := 2724 * 10 ^ 27) (Ed := 4838 * 10 ^ 25)
        (Ā' := 2725 * 10 ^ 27) (Ē' := 4839 * 10 ^ 25)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 7) (w := 7) M (w.s4 1).kdw.W (w.s4 1).kex.W (w.s4 1).kpr.W
          P.hwk P.hwk P.hwk (w.s4 1).kdw.hW (w.s4 1).kex.hW (w.s4 1).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 4590 / 10 ^ 8) (gex := 1833 / 10 ^ 7) (gdw := 3040 / 10 ^ 9)
          (A1 := 1194 * 10 ^ 15) (E1 := 1147 * 10 ^ 13) (A2 := 5503 * 10 ^ 17)
            (E2 := 5311 * 10 ^ 15)
          (A3 := 8310 * 10 ^ 17) (E3 := 1347 * 10 ^ 16) (A4 := 1532 * 10 ^ 21)
            (E4 := 2512 * 10 ^ 19)
          (A5 := 9264 * 10 ^ 25) (E5 := 1645 * 10 ^ 24)
          (Ā := 1421 * 10 ^ 14) (Ē := 1365 * 10 ^ 12) (Ā' := 2724 * 10 ^ 27) (Ē' := 4838 * 10 ^ 25)
          (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s4 1).γls (w.s4 1).γls (es := 0) (by norm_num) (w.s4 1).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 1421 * 10 ^ 14) (Ē := 1365 * 10 ^ 12) (Ā' := 1194 * 10 ^ 15)
              (Ē' := 1147 * 10 ^ 13) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s4 1).sge (w.s4 1).fsge (by norm_num) (w.s4 1).hsge (w.s4 1).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 5503 * 10 ^ 17) (Ē := 5311 * 10 ^ 15) (Ā' := 8310 * 10 ^ 17)
              (Ē' := 1347 * 10 ^ 16) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 7) (w := 7) M (w.s4 1).ln.γ (w.s4 1).ln.fγ (w.s4 1).ln.x (w.s4 1).ln.fs (w.s4 1).ln.fxh
              (by norm_num) (by norm_num) (w.s4 1).ln.hγ (w.s4 1).ln.hfγ (w.s4 1).ln.hst
                (w.s4 1).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
                (w.s4 1).ln.hfxh
              hMu (M.gamma_num (q := 4584 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 1258 * 10 ^ 1) (Kb := 1741 / 10 ^ 2) hK768r hK768b (by norm_num) (by norm_num)
              (Ā := 1532 * 10 ^ 21) (Ē := 2512 * 10 ^ 19) (Ā' := 9264 * 10 ^ 25)
                (Ē' := 1645 * 10 ^ 24)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 768 * 7 * 7) (by norm_num)
        (Ā := 2725 * 10 ^ 27) (Ē := 4839 * 10 ^ 25) (Bd := 5231 * 10 ^ 40) (Ed := 1351 * 10 ^ 39)
        (Ā' := 5232 * 10 ^ 40) (Ē' := 1352 * 10 ^ 39)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 7) (w := 7) M (w.s4 0).kdw.W (w.s4 0).kex.W (w.s4 0).kpr.W
          P.hwk P.hwk P.hwk (w.s4 0).kdw.hW (w.s4 0).kex.hW (w.s4 0).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 4590 / 10 ^ 8) (gex := 1833 / 10 ^ 7) (gdw := 3040 / 10 ^ 9)
          (A1 := 2290 * 10 ^ 28) (E1 := 4065 * 10 ^ 26) (A2 := 1056 * 10 ^ 31)
            (E2 := 1879 * 10 ^ 29)
          (A3 := 1595 * 10 ^ 31) (E3 := 3875 * 10 ^ 29) (A4 := 2941 * 10 ^ 34)
            (E4 := 7198 * 10 ^ 32)
          (A5 := 1779 * 10 ^ 39) (E5 := 4593 * 10 ^ 37)
          (Ā := 2725 * 10 ^ 27) (Ē := 4839 * 10 ^ 25) (Ā' := 5231 * 10 ^ 40) (Ē' := 1351 * 10 ^ 39)
          (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 1833 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s4 0).γls (w.s4 0).γls (es := 0) (by norm_num) (w.s4 0).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 2725 * 10 ^ 27) (Ē := 4839 * 10 ^ 25) (Ā' := 2290 * 10 ^ 28)
              (Ē' := 4065 * 10 ^ 26) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s4 0).sge (w.s4 0).fsge (by norm_num) (w.s4 0).hsge (w.s4 0).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 1056 * 10 ^ 31) (Ē := 1879 * 10 ^ 29) (Ā' := 1595 * 10 ^ 31)
              (Ē' := 3875 * 10 ^ 29) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 7) (w := 7) M (w.s4 0).ln.γ (w.s4 0).ln.fγ (w.s4 0).ln.x (w.s4 0).ln.fs (w.s4 0).ln.fxh
              (by norm_num) (by norm_num) (w.s4 0).ln.hγ (w.s4 0).ln.hfγ (w.s4 0).ln.hst
                (w.s4 0).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
                (w.s4 0).ln.hfxh
              hMu (M.gamma_num (q := 4584 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 1258 * 10 ^ 1) (Kb := 1741 / 10 ^ 2) hK768r hK768b (by norm_num) (by norm_num)
              (Ā := 2941 * 10 ^ 34) (Ē := 7198 * 10 ^ 32) (Ā' := 1779 * 10 ^ 39)
                (Ē' := 4593 * 10 ^ 37)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32])))
  have m3 := m2.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownBack (h := 7) (w := 7) M (padOdd w.d3.k.W)
      P.hwk (padOdd_abs_le w.d3.k.W P.hwk w.d3.k.hW) (by norm_num) (by norm_num)
      _ (g := 4123 / 10 ^ 7) (Ā := 5232 * 10 ^ 40) (Ē := 1352 * 10 ^ 39) (A1 := 2171 * 10 ^ 44)
        (E1 := 5699 * 10 ^ 42)
      (Ā' := 6714 * 10 ^ 48) (Ē' := 1870 * 10 ^ 47)
      (M.gamma_num (q := 4123 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num [u32]) (by norm_num [u32])
      (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M w.d3.ln.γ w.d3.ln.fγ w.d3.ln.x w.d3.ln.fs w.d3.ln.fxh
          (by norm_num) (by norm_num) w.d3.ln.hγ w.d3.ln.hfγ w.d3.ln.hst w.d3.ln.hSabs
          (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
            w.d3.ln.hfxh
          hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
          (Ā := 2171 * 10 ^ 44) (Ē := 5699 * 10 ^ 42) (Ā' := 6714 * 10 ^ 48) (Ē' := 1870 * 10 ^ 47)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])))
  have m4 := m3.comp (by norm_num)
    (((((((((FloatBridgesTo.Maps.residual M (m := 384 * 14 * 14) (by norm_num)
        (Ā := 6714 * 10 ^ 48) (Ē := 1870 * 10 ^ 47) (Bd := 1647 * 10 ^ 61) (Ed := 5922 * 10 ^ 59)
        (Ā' := 1648 * 10 ^ 61) (Ē' := 5923 * 10 ^ 59)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 14) (w := 14) M (w.s3 8).kdw.W (w.s3 8).kex.W (w.s3 8).kpr.W
          P.hwk P.hwk P.hwk (w.s3 8).kdw.hW (w.s3 8).kex.hW (w.s3 8).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 2301 / 10 ^ 8) (gex := 9169 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 5640 * 10 ^ 49) (E1 := 1571 * 10 ^ 48) (A2 := 1300 * 10 ^ 52)
            (E2 := 3623 * 10 ^ 50)
          (A3 := 1964 * 10 ^ 52) (E3 := 6735 * 10 ^ 50) (A4 := 1811 * 10 ^ 55)
            (E4 := 6225 * 10 ^ 53)
          (A5 := 5601 * 10 ^ 59) (E5 := 2014 * 10 ^ 58)
          (Ā := 6714 * 10 ^ 48) (Ē := 1870 * 10 ^ 47) (Ā' := 1647 * 10 ^ 61) (Ē' := 5922 * 10 ^ 59)
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s3 8).γls (w.s3 8).γls (es := 0) (by norm_num) (w.s3 8).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 6714 * 10 ^ 48) (Ē := 1870 * 10 ^ 47) (Ā' := 5640 * 10 ^ 49)
              (Ē' := 1571 * 10 ^ 48) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s3 8).sge (w.s3 8).fsge (by norm_num) (w.s3 8).hsge (w.s3 8).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 1300 * 10 ^ 52) (Ē := 3623 * 10 ^ 50) (Ā' := 1964 * 10 ^ 52)
              (Ē' := 6735 * 10 ^ 50) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M (w.s3 8).ln.γ (w.s3 8).ln.fγ (w.s3 8).ln.x (w.s3 8).ln.fs (w.s3 8).ln.fxh
              (by norm_num) (by norm_num) (w.s3 8).ln.hγ (w.s3 8).ln.hfγ (w.s3 8).ln.hst
                (w.s3 8).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
                (w.s3 8).ln.hfxh
              hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
              (Ā := 1811 * 10 ^ 55) (Ē := 6225 * 10 ^ 53) (Ā' := 5601 * 10 ^ 59)
                (Ē' := 2014 * 10 ^ 58)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32])).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 384 * 14 * 14) (by norm_num)
        (Ā := 1648 * 10 ^ 61) (Ē := 5923 * 10 ^ 59) (Bd := 4040 * 10 ^ 73) (Ed := 1780 * 10 ^ 72)
        (Ā' := 4041 * 10 ^ 73) (Ē' := 1781 * 10 ^ 72)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 14) (w := 14) M (w.s3 7).kdw.W (w.s3 7).kex.W (w.s3 7).kpr.W
          P.hwk P.hwk P.hwk (w.s3 7).kdw.hW (w.s3 7).kex.hW (w.s3 7).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 2301 / 10 ^ 8) (gex := 9169 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 1385 * 10 ^ 62) (E1 := 4976 * 10 ^ 60) (A2 := 3192 * 10 ^ 64)
            (E2 := 1148 * 10 ^ 63)
          (A3 := 4820 * 10 ^ 64) (E3 := 2042 * 10 ^ 63) (A4 := 4443 * 10 ^ 67)
            (E4 := 1887 * 10 ^ 66)
          (A5 := 1374 * 10 ^ 72) (E5 := 6052 * 10 ^ 70)
          (Ā := 1648 * 10 ^ 61) (Ē := 5923 * 10 ^ 59) (Ā' := 4040 * 10 ^ 73) (Ē' := 1780 * 10 ^ 72)
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s3 7).γls (w.s3 7).γls (es := 0) (by norm_num) (w.s3 7).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 1648 * 10 ^ 61) (Ē := 5923 * 10 ^ 59) (Ā' := 1385 * 10 ^ 62)
              (Ē' := 4976 * 10 ^ 60) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s3 7).sge (w.s3 7).fsge (by norm_num) (w.s3 7).hsge (w.s3 7).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 3192 * 10 ^ 64) (Ē := 1148 * 10 ^ 63) (Ā' := 4820 * 10 ^ 64)
              (Ē' := 2042 * 10 ^ 63) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M (w.s3 7).ln.γ (w.s3 7).ln.fγ (w.s3 7).ln.x (w.s3 7).ln.fs (w.s3 7).ln.fxh
              (by norm_num) (by norm_num) (w.s3 7).ln.hγ (w.s3 7).ln.hfγ (w.s3 7).ln.hst
                (w.s3 7).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
                (w.s3 7).ln.hfxh
              hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
              (Ā := 4443 * 10 ^ 67) (Ē := 1887 * 10 ^ 66) (Ā' := 1374 * 10 ^ 72)
                (Ē' := 6052 * 10 ^ 70)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 384 * 14 * 14) (by norm_num)
        (Ā := 4041 * 10 ^ 73) (Ē := 1781 * 10 ^ 72) (Bd := 9911 * 10 ^ 85) (Ed := 5161 * 10 ^ 84)
        (Ā' := 9912 * 10 ^ 85) (Ē' := 5162 * 10 ^ 84)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 14) (w := 14) M (w.s3 6).kdw.W (w.s3 6).kex.W (w.s3 6).kpr.W
          P.hwk P.hwk P.hwk (w.s3 6).kdw.hW (w.s3 6).kex.hW (w.s3 6).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 2301 / 10 ^ 8) (gex := 9169 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 3395 * 10 ^ 74) (E1 := 1497 * 10 ^ 73) (A2 := 7823 * 10 ^ 76)
            (E2 := 3451 * 10 ^ 75)
          (A3 := 1182 * 10 ^ 77) (E3 := 5959 * 10 ^ 75) (A4 := 1090 * 10 ^ 80)
            (E4 := 5503 * 10 ^ 78)
          (A5 := 3371 * 10 ^ 84) (E5 := 1755 * 10 ^ 83)
          (Ā := 4041 * 10 ^ 73) (Ē := 1781 * 10 ^ 72) (Ā' := 9911 * 10 ^ 85) (Ē' := 5161 * 10 ^ 84)
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s3 6).γls (w.s3 6).γls (es := 0) (by norm_num) (w.s3 6).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 4041 * 10 ^ 73) (Ē := 1781 * 10 ^ 72) (Ā' := 3395 * 10 ^ 74)
              (Ē' := 1497 * 10 ^ 73) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s3 6).sge (w.s3 6).fsge (by norm_num) (w.s3 6).hsge (w.s3 6).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 7823 * 10 ^ 76) (Ē := 3451 * 10 ^ 75) (Ā' := 1182 * 10 ^ 77)
              (Ē' := 5959 * 10 ^ 75) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M (w.s3 6).ln.γ (w.s3 6).ln.fγ (w.s3 6).ln.x (w.s3 6).ln.fs (w.s3 6).ln.fxh
              (by norm_num) (by norm_num) (w.s3 6).ln.hγ (w.s3 6).ln.hfγ (w.s3 6).ln.hst
                (w.s3 6).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
                (w.s3 6).ln.hfxh
              hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
              (Ā := 1090 * 10 ^ 80) (Ē := 5503 * 10 ^ 78) (Ā' := 3371 * 10 ^ 84)
                (Ē' := 1755 * 10 ^ 83)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 384 * 14 * 14) (by norm_num)
        (Ā := 9912 * 10 ^ 85) (Ē := 5162 * 10 ^ 84) (Bd := 2430 * 10 ^ 98) (Ed := 1459 * 10 ^ 97)
        (Ā' := 2431 * 10 ^ 98) (Ē' := 1460 * 10 ^ 97)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 14) (w := 14) M (w.s3 5).kdw.W (w.s3 5).kex.W (w.s3 5).kpr.W
          P.hwk P.hwk P.hwk (w.s3 5).kdw.hW (w.s3 5).kex.hW (w.s3 5).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 2301 / 10 ^ 8) (gex := 9169 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 8327 * 10 ^ 86) (E1 := 4337 * 10 ^ 85) (A2 := 1919 * 10 ^ 89)
            (E2 := 9998 * 10 ^ 87)
          (A3 := 2898 * 10 ^ 89) (E3 := 1692 * 10 ^ 88) (A4 := 2672 * 10 ^ 92)
            (E4 := 1562 * 10 ^ 91)
          (A5 := 8263 * 10 ^ 96) (E5 := 4959 * 10 ^ 95)
          (Ā := 9912 * 10 ^ 85) (Ē := 5162 * 10 ^ 84) (Ā' := 2430 * 10 ^ 98) (Ē' := 1459 * 10 ^ 97)
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s3 5).γls (w.s3 5).γls (es := 0) (by norm_num) (w.s3 5).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 9912 * 10 ^ 85) (Ē := 5162 * 10 ^ 84) (Ā' := 8327 * 10 ^ 86)
              (Ē' := 4337 * 10 ^ 85) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s3 5).sge (w.s3 5).fsge (by norm_num) (w.s3 5).hsge (w.s3 5).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 1919 * 10 ^ 89) (Ē := 9998 * 10 ^ 87) (Ā' := 2898 * 10 ^ 89)
              (Ē' := 1692 * 10 ^ 88) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M (w.s3 5).ln.γ (w.s3 5).ln.fγ (w.s3 5).ln.x (w.s3 5).ln.fs (w.s3 5).ln.fxh
              (by norm_num) (by norm_num) (w.s3 5).ln.hγ (w.s3 5).ln.hfγ (w.s3 5).ln.hst
                (w.s3 5).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
                (w.s3 5).ln.hfxh
              hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
              (Ā := 2672 * 10 ^ 92) (Ē := 1562 * 10 ^ 91) (Ā' := 8263 * 10 ^ 96)
                (Ē' := 4959 * 10 ^ 95)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 384 * 14 * 14) (by norm_num)
        (Ā := 2431 * 10 ^ 98) (Ē := 1460 * 10 ^ 97) (Bd := 5960 * 10 ^ 110) (Ed := 4049 * 10 ^ 109)
        (Ā' := 5961 * 10 ^ 110) (Ē' := 4050 * 10 ^ 109)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 14) (w := 14) M (w.s3 4).kdw.W (w.s3 4).kex.W (w.s3 4).kpr.W
          P.hwk P.hwk P.hwk (w.s3 4).kdw.hW (w.s3 4).kex.hW (w.s3 4).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 2301 / 10 ^ 8) (gex := 9169 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 2043 * 10 ^ 99) (E1 := 1227 * 10 ^ 98) (A2 := 4708 * 10 ^ 101)
            (E2 := 2829 * 10 ^ 100)
          (A3 := 7110 * 10 ^ 101) (E3 := 4715 * 10 ^ 100) (A4 := 6554 * 10 ^ 104)
            (E4 := 4352 * 10 ^ 103)
          (A5 := 2027 * 10 ^ 109) (E5 := 1377 * 10 ^ 108)
          (Ā := 2431 * 10 ^ 98) (Ē := 1460 * 10 ^ 97) (Ā' := 5960 * 10 ^ 110)
            (Ē' := 4049 * 10 ^ 109)
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s3 4).γls (w.s3 4).γls (es := 0) (by norm_num) (w.s3 4).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 2431 * 10 ^ 98) (Ē := 1460 * 10 ^ 97) (Ā' := 2043 * 10 ^ 99)
              (Ē' := 1227 * 10 ^ 98) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s3 4).sge (w.s3 4).fsge (by norm_num) (w.s3 4).hsge (w.s3 4).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 4708 * 10 ^ 101) (Ē := 2829 * 10 ^ 100) (Ā' := 7110 * 10 ^ 101)
              (Ē' := 4715 * 10 ^ 100) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M (w.s3 4).ln.γ (w.s3 4).ln.fγ (w.s3 4).ln.x (w.s3 4).ln.fs (w.s3 4).ln.fxh
              (by norm_num) (by norm_num) (w.s3 4).ln.hγ (w.s3 4).ln.hfγ (w.s3 4).ln.hst
                (w.s3 4).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
                (w.s3 4).ln.hfxh
              hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
              (Ā := 6554 * 10 ^ 104) (Ē := 4352 * 10 ^ 103) (Ā' := 2027 * 10 ^ 109)
                (Ē' := 1377 * 10 ^ 108)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 384 * 14 * 14) (by norm_num)
        (Ā := 5961 * 10 ^ 110) (Ē := 4050 * 10 ^ 109) (Bd := 1462 * 10 ^ 123)
          (Ed := 1108 * 10 ^ 122)
        (Ā' := 1463 * 10 ^ 123) (Ē' := 1109 * 10 ^ 122)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 14) (w := 14) M (w.s3 3).kdw.W (w.s3 3).kex.W (w.s3 3).kpr.W
          P.hwk P.hwk P.hwk (w.s3 3).kdw.hW (w.s3 3).kex.hW (w.s3 3).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 2301 / 10 ^ 8) (gex := 9169 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 5008 * 10 ^ 111) (E1 := 3403 * 10 ^ 110) (A2 := 1154 * 10 ^ 114)
            (E2 := 7844 * 10 ^ 112)
          (A3 := 1743 * 10 ^ 114) (E3 := 1293 * 10 ^ 113) (A4 := 1607 * 10 ^ 117)
            (E4 := 1194 * 10 ^ 116)
          (A5 := 4970 * 10 ^ 121) (E5 := 3768 * 10 ^ 120)
          (Ā := 5961 * 10 ^ 110) (Ē := 4050 * 10 ^ 109) (Ā' := 1462 * 10 ^ 123)
            (Ē' := 1108 * 10 ^ 122)
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s3 3).γls (w.s3 3).γls (es := 0) (by norm_num) (w.s3 3).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 5961 * 10 ^ 110) (Ē := 4050 * 10 ^ 109) (Ā' := 5008 * 10 ^ 111)
              (Ē' := 3403 * 10 ^ 110) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s3 3).sge (w.s3 3).fsge (by norm_num) (w.s3 3).hsge (w.s3 3).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 1154 * 10 ^ 114) (Ē := 7844 * 10 ^ 112) (Ā' := 1743 * 10 ^ 114)
              (Ē' := 1293 * 10 ^ 113) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M (w.s3 3).ln.γ (w.s3 3).ln.fγ (w.s3 3).ln.x (w.s3 3).ln.fs (w.s3 3).ln.fxh
              (by norm_num) (by norm_num) (w.s3 3).ln.hγ (w.s3 3).ln.hfγ (w.s3 3).ln.hst
                (w.s3 3).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
                (w.s3 3).ln.hfxh
              hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
              (Ā := 1607 * 10 ^ 117) (Ē := 1194 * 10 ^ 116) (Ā' := 4970 * 10 ^ 121)
                (Ē' := 3768 * 10 ^ 120)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 384 * 14 * 14) (by norm_num)
        (Ā := 1463 * 10 ^ 123) (Ē := 1109 * 10 ^ 122) (Bd := 3587 * 10 ^ 135)
          (Ed := 2999 * 10 ^ 134)
        (Ā' := 3588 * 10 ^ 135) (Ē' := 3000 * 10 ^ 134)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 14) (w := 14) M (w.s3 2).kdw.W (w.s3 2).kex.W (w.s3 2).kpr.W
          P.hwk P.hwk P.hwk (w.s3 2).kdw.hW (w.s3 2).kex.hW (w.s3 2).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 2301 / 10 ^ 8) (gex := 9169 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 1229 * 10 ^ 124) (E1 := 9316 * 10 ^ 122) (A2 := 2832 * 10 ^ 126)
            (E2 := 2148 * 10 ^ 125)
          (A3 := 4277 * 10 ^ 126) (E3 := 3506 * 10 ^ 125) (A4 := 3943 * 10 ^ 129)
            (E4 := 3236 * 10 ^ 128)
          (A5 := 1220 * 10 ^ 134) (E5 := 1020 * 10 ^ 133)
          (Ā := 1463 * 10 ^ 123) (Ē := 1109 * 10 ^ 122) (Ā' := 3587 * 10 ^ 135)
            (Ē' := 2999 * 10 ^ 134)
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s3 2).γls (w.s3 2).γls (es := 0) (by norm_num) (w.s3 2).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 1463 * 10 ^ 123) (Ē := 1109 * 10 ^ 122) (Ā' := 1229 * 10 ^ 124)
              (Ē' := 9316 * 10 ^ 122) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s3 2).sge (w.s3 2).fsge (by norm_num) (w.s3 2).hsge (w.s3 2).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 2832 * 10 ^ 126) (Ē := 2148 * 10 ^ 125) (Ā' := 4277 * 10 ^ 126)
              (Ē' := 3506 * 10 ^ 125) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M (w.s3 2).ln.γ (w.s3 2).ln.fγ (w.s3 2).ln.x (w.s3 2).ln.fs (w.s3 2).ln.fxh
              (by norm_num) (by norm_num) (w.s3 2).ln.hγ (w.s3 2).ln.hfγ (w.s3 2).ln.hst
                (w.s3 2).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
                (w.s3 2).ln.hfxh
              hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
              (Ā := 3943 * 10 ^ 129) (Ē := 3236 * 10 ^ 128) (Ā' := 1220 * 10 ^ 134)
                (Ē' := 1020 * 10 ^ 133)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 384 * 14 * 14) (by norm_num)
        (Ā := 3588 * 10 ^ 135) (Ē := 3000 * 10 ^ 134) (Bd := 8794 * 10 ^ 147)
          (Ed := 8030 * 10 ^ 146)
        (Ā' := 8795 * 10 ^ 147) (Ē' := 8031 * 10 ^ 146)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 14) (w := 14) M (w.s3 1).kdw.W (w.s3 1).kex.W (w.s3 1).kpr.W
          P.hwk P.hwk P.hwk (w.s3 1).kdw.hW (w.s3 1).kex.hW (w.s3 1).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 2301 / 10 ^ 8) (gex := 9169 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 3014 * 10 ^ 136) (E1 := 2521 * 10 ^ 135) (A2 := 6945 * 10 ^ 138)
            (E2 := 5811 * 10 ^ 137)
          (A3 := 1049 * 10 ^ 139) (E3 := 9412 * 10 ^ 137) (A4 := 9669 * 10 ^ 141)
            (E4 := 8684 * 10 ^ 140)
          (A5 := 2991 * 10 ^ 146) (E5 := 2731 * 10 ^ 145)
          (Ā := 3588 * 10 ^ 135) (Ē := 3000 * 10 ^ 134) (Ā' := 8794 * 10 ^ 147)
            (Ē' := 8030 * 10 ^ 146)
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s3 1).γls (w.s3 1).γls (es := 0) (by norm_num) (w.s3 1).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 3588 * 10 ^ 135) (Ē := 3000 * 10 ^ 134) (Ā' := 3014 * 10 ^ 136)
              (Ē' := 2521 * 10 ^ 135) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s3 1).sge (w.s3 1).fsge (by norm_num) (w.s3 1).hsge (w.s3 1).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 6945 * 10 ^ 138) (Ē := 5811 * 10 ^ 137) (Ā' := 1049 * 10 ^ 139)
              (Ē' := 9412 * 10 ^ 137) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M (w.s3 1).ln.γ (w.s3 1).ln.fγ (w.s3 1).ln.x (w.s3 1).ln.fs (w.s3 1).ln.fxh
              (by norm_num) (by norm_num) (w.s3 1).ln.hγ (w.s3 1).ln.hfγ (w.s3 1).ln.hst
                (w.s3 1).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
                (w.s3 1).ln.hfxh
              hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
              (Ā := 9669 * 10 ^ 141) (Ē := 8684 * 10 ^ 140) (Ā' := 2991 * 10 ^ 146)
                (Ē' := 2731 * 10 ^ 145)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 384 * 14 * 14) (by norm_num)
        (Ā := 8795 * 10 ^ 147) (Ē := 8031 * 10 ^ 146) (Bd := 2156 * 10 ^ 160)
          (Ed := 2133 * 10 ^ 159)
        (Ā' := 2157 * 10 ^ 160) (Ē' := 2134 * 10 ^ 159)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 14) (w := 14) M (w.s3 0).kdw.W (w.s3 0).kex.W (w.s3 0).kpr.W
          P.hwk P.hwk P.hwk (w.s3 0).kdw.hW (w.s3 0).kex.hW (w.s3 0).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 2301 / 10 ^ 8) (gex := 9169 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 7388 * 10 ^ 148) (E1 := 6747 * 10 ^ 147) (A2 := 1703 * 10 ^ 151)
            (E2 := 1555 * 10 ^ 150)
          (A3 := 2572 * 10 ^ 151) (E3 := 2503 * 10 ^ 150) (A4 := 2371 * 10 ^ 154)
            (E4 := 2310 * 10 ^ 153)
          (A5 := 7333 * 10 ^ 158) (E5 := 7253 * 10 ^ 157)
          (Ā := 8795 * 10 ^ 147) (Ē := 8031 * 10 ^ 146) (Ā' := 2156 * 10 ^ 160)
            (Ē' := 2133 * 10 ^ 159)
          (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 9169 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s3 0).γls (w.s3 0).γls (es := 0) (by norm_num) (w.s3 0).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 8795 * 10 ^ 147) (Ē := 8031 * 10 ^ 146) (Ā' := 7388 * 10 ^ 148)
              (Ē' := 6747 * 10 ^ 147) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s3 0).sge (w.s3 0).fsge (by norm_num) (w.s3 0).hsge (w.s3 0).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 1703 * 10 ^ 151) (Ē := 1555 * 10 ^ 150) (Ā' := 2572 * 10 ^ 151)
              (Ē' := 2503 * 10 ^ 150) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 14) (w := 14) M (w.s3 0).ln.γ (w.s3 0).ln.fγ (w.s3 0).ln.x (w.s3 0).ln.fs (w.s3 0).ln.fxh
              (by norm_num) (by norm_num) (w.s3 0).ln.hγ (w.s3 0).ln.hfγ (w.s3 0).ln.hst
                (w.s3 0).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 20) P.hε _ (by norm_num) (by norm_num) i)
                (w.s3 0).ln.hfxh
              hMu (M.gamma_num (q := 2295 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 6432) (Kb := 1058 / 10 ^ 2) hK384r hK384b (by norm_num) (by norm_num)
              (Ā := 2371 * 10 ^ 154) (Ē := 2310 * 10 ^ 153) (Ā' := 7333 * 10 ^ 158)
                (Ē' := 7253 * 10 ^ 157)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32])))
  have m5 := m4.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownBack (h := 14) (w := 14) M (padOdd w.d2.k.W)
      P.hwk (padOdd_abs_le w.d2.k.W P.hwk w.d2.k.hW) (by norm_num) (by norm_num)
      _ (g := 2062 / 10 ^ 7) (Ā := 2157 * 10 ^ 160) (Ē := 2134 * 10 ^ 159) (A1 := 4474 * 10 ^ 163)
        (E1 := 4436 * 10 ^ 162)
      (Ā' := 6818 * 10 ^ 167) (Ē' := 6886 * 10 ^ 166)
      (M.gamma_num (q := 2062 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num [u32]) (by norm_num [u32])
      (FloatBridgesTo.Maps.chanLNTensor3Back (h := 28) (w := 28) M w.d2.ln.γ w.d2.ln.fγ w.d2.ln.x w.d2.ln.fs w.d2.ln.fxh
          (by norm_num) (by norm_num) w.d2.ln.hγ w.d2.ln.hfγ w.d2.ln.hst w.d2.ln.hSabs
          (fun _r i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
            w.d2.ln.hfxh
          hMu (M.gamma_num (q := 1151 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (Kr := 3168) (Kb := 6502 / 10 ^ 3) hK192r hK192b (by norm_num) (by norm_num)
          (Ā := 4474 * 10 ^ 163) (Ē := 4436 * 10 ^ 162) (Ā' := 6818 * 10 ^ 167)
            (Ē' := 6886 * 10 ^ 166)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])))
  have m6 := m5.comp (by norm_num)
    (((FloatBridgesTo.Maps.residual M (m := 192 * 28 * 28) (by norm_num)
        (Ā := 6818 * 10 ^ 167) (Ē := 6886 * 10 ^ 166) (Bd := 2058 * 10 ^ 179)
          (Ed := 2240 * 10 ^ 178)
        (Ā' := 2059 * 10 ^ 179) (Ē' := 2241 * 10 ^ 178)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 28) (w := 28) M (w.s2 2).kdw.W (w.s2 2).kex.W (w.s2 2).kpr.W
          P.hwk P.hwk P.hwk (w.s2 2).kdw.hW (w.s2 2).kex.hW (w.s2 2).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 1157 / 10 ^ 8) (gex := 4590 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 5728 * 10 ^ 168) (E1 := 5785 * 10 ^ 167) (A2 := 6599 * 10 ^ 170)
            (E2 := 6666 * 10 ^ 169)
          (A3 := 9965 * 10 ^ 170) (E3 := 1066 * 10 ^ 170) (A4 := 4593 * 10 ^ 173)
            (E4 := 4915 * 10 ^ 172)
          (A5 := 6999 * 10 ^ 177) (E5 := 7618 * 10 ^ 176)
          (Ā := 6818 * 10 ^ 167) (Ē := 6886 * 10 ^ 166) (Ā' := 2058 * 10 ^ 179)
            (Ē' := 2240 * 10 ^ 178)
          (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s2 2).γls (w.s2 2).γls (es := 0) (by norm_num) (w.s2 2).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 6818 * 10 ^ 167) (Ē := 6886 * 10 ^ 166) (Ā' := 5728 * 10 ^ 168)
              (Ē' := 5785 * 10 ^ 167) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s2 2).sge (w.s2 2).fsge (by norm_num) (w.s2 2).hsge (w.s2 2).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 6599 * 10 ^ 170) (Ē := 6666 * 10 ^ 169) (Ā' := 9965 * 10 ^ 170)
              (Ē' := 1066 * 10 ^ 170) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 28) (w := 28) M (w.s2 2).ln.γ (w.s2 2).ln.fγ (w.s2 2).ln.x (w.s2 2).ln.fs (w.s2 2).ln.fxh
              (by norm_num) (by norm_num) (w.s2 2).ln.hγ (w.s2 2).ln.hfγ (w.s2 2).ln.hst
                (w.s2 2).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
                (w.s2 2).ln.hfxh
              hMu (M.gamma_num (q := 1151 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 3168) (Kb := 6502 / 10 ^ 3) hK192r hK192b (by norm_num) (by norm_num)
              (Ā := 4593 * 10 ^ 173) (Ē := 4915 * 10 ^ 172) (Ā' := 6999 * 10 ^ 177)
                (Ē' := 7618 * 10 ^ 176)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32])).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 192 * 28 * 28) (by norm_num)
        (Ā := 2059 * 10 ^ 179) (Ē := 2241 * 10 ^ 178) (Bd := 6219 * 10 ^ 190)
          (Ed := 7251 * 10 ^ 189)
        (Ā' := 6220 * 10 ^ 190) (Ē' := 7252 * 10 ^ 189)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 28) (w := 28) M (w.s2 1).kdw.W (w.s2 1).kex.W (w.s2 1).kpr.W
          P.hwk P.hwk P.hwk (w.s2 1).kdw.hW (w.s2 1).kex.hW (w.s2 1).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 1157 / 10 ^ 8) (gex := 4590 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 1730 * 10 ^ 180) (E1 := 1883 * 10 ^ 179) (A2 := 1993 * 10 ^ 182)
            (E2 := 2170 * 10 ^ 181)
          (A3 := 3010 * 10 ^ 182) (E3 := 3455 * 10 ^ 181) (A4 := 1388 * 10 ^ 185)
            (E4 := 1593 * 10 ^ 184)
          (A5 := 2115 * 10 ^ 189) (E5 := 2466 * 10 ^ 188)
          (Ā := 2059 * 10 ^ 179) (Ē := 2241 * 10 ^ 178) (Ā' := 6219 * 10 ^ 190)
            (Ē' := 7251 * 10 ^ 189)
          (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s2 1).γls (w.s2 1).γls (es := 0) (by norm_num) (w.s2 1).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 2059 * 10 ^ 179) (Ē := 2241 * 10 ^ 178) (Ā' := 1730 * 10 ^ 180)
              (Ē' := 1883 * 10 ^ 179) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s2 1).sge (w.s2 1).fsge (by norm_num) (w.s2 1).hsge (w.s2 1).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 1993 * 10 ^ 182) (Ē := 2170 * 10 ^ 181) (Ā' := 3010 * 10 ^ 182)
              (Ē' := 3455 * 10 ^ 181) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 28) (w := 28) M (w.s2 1).ln.γ (w.s2 1).ln.fγ (w.s2 1).ln.x (w.s2 1).ln.fs (w.s2 1).ln.fxh
              (by norm_num) (by norm_num) (w.s2 1).ln.hγ (w.s2 1).ln.hfγ (w.s2 1).ln.hst
                (w.s2 1).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
                (w.s2 1).ln.hfxh
              hMu (M.gamma_num (q := 1151 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 3168) (Kb := 6502 / 10 ^ 3) hK192r hK192b (by norm_num) (by norm_num)
              (Ā := 1388 * 10 ^ 185) (Ē := 1593 * 10 ^ 184) (Ā' := 2115 * 10 ^ 189)
                (Ē' := 2466 * 10 ^ 188)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 192 * 28 * 28) (by norm_num)
        (Ā := 6220 * 10 ^ 190) (Ē := 7252 * 10 ^ 189) (Bd := 1878 * 10 ^ 202)
          (Ed := 2335 * 10 ^ 201)
        (Ā' := 1879 * 10 ^ 202) (Ē' := 2336 * 10 ^ 201)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 28) (w := 28) M (w.s2 0).kdw.W (w.s2 0).kex.W (w.s2 0).kpr.W
          P.hwk P.hwk P.hwk (w.s2 0).kdw.hW (w.s2 0).kex.hW (w.s2 0).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 1157 / 10 ^ 8) (gex := 4590 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 5225 * 10 ^ 191) (E1 := 6092 * 10 ^ 190) (A2 := 6020 * 10 ^ 193)
            (E2 := 7019 * 10 ^ 192)
          (A3 := 9091 * 10 ^ 193) (E3 := 1114 * 10 ^ 193) (A4 := 4190 * 10 ^ 196)
            (E4 := 5136 * 10 ^ 195)
          (A5 := 6385 * 10 ^ 200) (E5 := 7941 * 10 ^ 199)
          (Ā := 6220 * 10 ^ 190) (Ē := 7252 * 10 ^ 189) (Ā' := 1878 * 10 ^ 202)
            (Ē' := 2335 * 10 ^ 201)
          (M.gamma_num (q := 1157 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 4590 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s2 0).γls (w.s2 0).γls (es := 0) (by norm_num) (w.s2 0).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 6220 * 10 ^ 190) (Ē := 7252 * 10 ^ 189) (Ā' := 5225 * 10 ^ 191)
              (Ē' := 6092 * 10 ^ 190) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s2 0).sge (w.s2 0).fsge (by norm_num) (w.s2 0).hsge (w.s2 0).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 6020 * 10 ^ 193) (Ē := 7019 * 10 ^ 192) (Ā' := 9091 * 10 ^ 193)
              (Ē' := 1114 * 10 ^ 193) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 28) (w := 28) M (w.s2 0).ln.γ (w.s2 0).ln.fγ (w.s2 0).ln.x (w.s2 0).ln.fs (w.s2 0).ln.fxh
              (by norm_num) (by norm_num) (w.s2 0).ln.hγ (w.s2 0).ln.hfγ (w.s2 0).ln.hst
                (w.s2 0).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
                (w.s2 0).ln.hfxh
              hMu (M.gamma_num (q := 1151 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 3168) (Kb := 6502 / 10 ^ 3) hK192r hK192b (by norm_num) (by norm_num)
              (Ā := 4190 * 10 ^ 196) (Ē := 5136 * 10 ^ 195) (Ā' := 6385 * 10 ^ 200)
                (Ē' := 7941 * 10 ^ 199)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32])))
  have m7 := m6.comp (by norm_num)
    (FloatBridgesTo.Maps.cnxDownBack (h := 28) (w := 28) M (padOdd w.d1.k.W)
      P.hwk (padOdd_abs_le w.d1.k.W P.hwk w.d1.k.hW) (by norm_num) (by norm_num)
      _ (g := 1032 / 10 ^ 7) (Ā := 1879 * 10 ^ 202) (Ē := 2336 * 10 ^ 201) (A1 := 1949 * 10 ^ 205)
        (E1 := 2425 * 10 ^ 204)
      (Ā' := 1531 * 10 ^ 209) (Ē' := 1940 * 10 ^ 208)
      (M.gamma_num (q := 1032 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (by norm_num [u32]) (by norm_num [u32])
      (FloatBridgesTo.Maps.chanLNTensor3Back (h := 56) (w := 56) M w.d1.ln.γ w.d1.ln.fγ w.d1.ln.x w.d1.ln.fs w.d1.ln.fxh
          (by norm_num) (by norm_num) w.d1.ln.hγ w.d1.ln.hfγ w.d1.ln.hst w.d1.ln.hSabs
          (fun _r i => bnXhat_abs_le_num (X := 10) P.hε _ (by norm_num) (by norm_num) i)
            w.d1.ln.hfxh
          hMu (M.gamma_num (q := 5782 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
            (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
          (Kr := 1632) (Kb := 4234 / 10 ^ 3) hK96r hK96b (by norm_num) (by norm_num)
          (Ā := 1949 * 10 ^ 205) (Ē := 2425 * 10 ^ 204) (Ā' := 1531 * 10 ^ 209)
            (Ē' := 1940 * 10 ^ 208)
          (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32])))
  have m8 := m7.comp (by norm_num)
    (((FloatBridgesTo.Maps.residual M (m := 96 * 56 * 56) (by norm_num)
        (Ā := 1531 * 10 ^ 209) (Ē := 1940 * 10 ^ 208) (Bd := 5963 * 10 ^ 219)
          (Ed := 8027 * 10 ^ 218)
        (Ā' := 5964 * 10 ^ 219) (Ē' := 8028 * 10 ^ 218)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 56) (w := 56) M (w.s1 2).kdw.W (w.s1 2).kex.W (w.s1 2).kpr.W
          P.hwk P.hwk P.hwk (w.s1 2).kdw.hW (w.s1 2).kex.hW (w.s1 2).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 5842 / 10 ^ 9) (gex := 2301 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 1287 * 10 ^ 210) (E1 := 1630 * 10 ^ 209) (A2 := 7414 * 10 ^ 211)
            (E2 := 9390 * 10 ^ 210)
          (A3 := 1120 * 10 ^ 212) (E3 := 1483 * 10 ^ 211) (A4 := 2581 * 10 ^ 214)
            (E4 := 3418 * 10 ^ 213)
          (A5 := 2028 * 10 ^ 218) (E5 := 2730 * 10 ^ 217)
          (Ā := 1531 * 10 ^ 209) (Ē := 1940 * 10 ^ 208) (Ā' := 5963 * 10 ^ 219)
            (Ē' := 8027 * 10 ^ 218)
          (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s1 2).γls (w.s1 2).γls (es := 0) (by norm_num) (w.s1 2).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 1531 * 10 ^ 209) (Ē := 1940 * 10 ^ 208) (Ā' := 1287 * 10 ^ 210)
              (Ē' := 1630 * 10 ^ 209) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s1 2).sge (w.s1 2).fsge (by norm_num) (w.s1 2).hsge (w.s1 2).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 7414 * 10 ^ 211) (Ē := 9390 * 10 ^ 210) (Ā' := 1120 * 10 ^ 212)
              (Ē' := 1483 * 10 ^ 211) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 56) (w := 56) M (w.s1 2).ln.γ (w.s1 2).ln.fγ (w.s1 2).ln.x (w.s1 2).ln.fs (w.s1 2).ln.fxh
              (by norm_num) (by norm_num) (w.s1 2).ln.hγ (w.s1 2).ln.hfγ (w.s1 2).ln.hst
                (w.s1 2).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 10) P.hε _ (by norm_num) (by norm_num) i)
                (w.s1 2).ln.hfxh
              hMu (M.gamma_num (q := 5782 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 1632) (Kb := 4234 / 10 ^ 3) hK96r hK96b (by norm_num) (by norm_num)
              (Ā := 2581 * 10 ^ 214) (Ē := 3418 * 10 ^ 213) (Ā' := 2028 * 10 ^ 218)
                (Ē' := 2730 * 10 ^ 217)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32])).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 96 * 56 * 56) (by norm_num)
        (Ā := 5964 * 10 ^ 219) (Ē := 8028 * 10 ^ 218) (Bd := 2321 * 10 ^ 230)
          (Ed := 3308 * 10 ^ 229)
        (Ā' := 2322 * 10 ^ 230) (Ē' := 3309 * 10 ^ 229)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 56) (w := 56) M (w.s1 1).kdw.W (w.s1 1).kex.W (w.s1 1).kpr.W
          P.hwk P.hwk P.hwk (w.s1 1).kdw.hW (w.s1 1).kex.hW (w.s1 1).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 5842 / 10 ^ 9) (gex := 2301 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 5010 * 10 ^ 220) (E1 := 6744 * 10 ^ 219) (A2 := 2886 * 10 ^ 222)
            (E2 := 3885 * 10 ^ 221)
          (A3 := 4358 * 10 ^ 222) (E3 := 6117 * 10 ^ 221) (A4 := 1005 * 10 ^ 225)
            (E4 := 1410 * 10 ^ 224)
          (A5 := 7894 * 10 ^ 228) (E5 := 1125 * 10 ^ 228)
          (Ā := 5964 * 10 ^ 219) (Ē := 8028 * 10 ^ 218) (Ā' := 2321 * 10 ^ 230)
            (Ē' := 3308 * 10 ^ 229)
          (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s1 1).γls (w.s1 1).γls (es := 0) (by norm_num) (w.s1 1).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 5964 * 10 ^ 219) (Ē := 8028 * 10 ^ 218) (Ā' := 5010 * 10 ^ 220)
              (Ē' := 6744 * 10 ^ 219) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s1 1).sge (w.s1 1).fsge (by norm_num) (w.s1 1).hsge (w.s1 1).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 2886 * 10 ^ 222) (Ē := 3885 * 10 ^ 221) (Ā' := 4358 * 10 ^ 222)
              (Ē' := 6117 * 10 ^ 221) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 56) (w := 56) M (w.s1 1).ln.γ (w.s1 1).ln.fγ (w.s1 1).ln.x (w.s1 1).ln.fs (w.s1 1).ln.fxh
              (by norm_num) (by norm_num) (w.s1 1).ln.hγ (w.s1 1).ln.hfγ (w.s1 1).ln.hst
                (w.s1 1).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 10) P.hε _ (by norm_num) (by norm_num) i)
                (w.s1 1).ln.hfxh
              hMu (M.gamma_num (q := 5782 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 1632) (Kb := 4234 / 10 ^ 3) hK96r hK96b (by norm_num) (by norm_num)
              (Ā := 1005 * 10 ^ 225) (Ē := 1410 * 10 ^ 224) (Ā' := 7894 * 10 ^ 228)
                (Ē' := 1125 * 10 ^ 228)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32]))).comp (by norm_num)
      (FloatBridgesTo.Maps.residual M (m := 96 * 56 * 56) (by norm_num)
        (Ā := 2322 * 10 ^ 230) (Ē := 3309 * 10 ^ 229) (Bd := 9038 * 10 ^ 240)
          (Ed := 1359 * 10 ^ 240)
        (Ā' := 9039 * 10 ^ 240) (Ē' := 1360 * 10 ^ 240)
        (FloatBridgesTo.Maps.cnxBlockBodyBack (h := 56) (w := 56) M (w.s1 0).kdw.W (w.s1 0).kex.W (w.s1 0).kpr.W
          P.hwk P.hwk P.hwk (w.s1 0).kdw.hW (w.s1 0).kex.hW (w.s1 0).kpr.hW (by norm_num)
            (by norm_num)
          _ _ _
          (gpr := 5842 / 10 ^ 9) (gex := 2301 / 10 ^ 8) (gdw := 3040 / 10 ^ 9)
          (A1 := 1951 * 10 ^ 231) (E1 := 2780 * 10 ^ 230) (A2 := 1124 * 10 ^ 233)
            (E2 := 1602 * 10 ^ 232)
          (A3 := 1698 * 10 ^ 233) (E3 := 2516 * 10 ^ 232) (A4 := 3913 * 10 ^ 235)
            (E4 := 5798 * 10 ^ 234)
          (A5 := 3074 * 10 ^ 239) (E5 := 4622 * 10 ^ 238)
          (Ā := 2322 * 10 ^ 230) (Ē := 3309 * 10 ^ 229) (Ā' := 9038 * 10 ^ 240)
            (Ē' := 1359 * 10 ^ 240)
          (M.gamma_num (q := 5842 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 2301 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
            (M.gamma_num (q := 3040 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (FloatBridgesTo.Maps.diagBack M (w.s1 0).γls (w.s1 0).γls (es := 0) (by norm_num) (w.s1 0).hγls
            (fun _ => by simp) hMu (by norm_num) le_rfl
            (Ā := 2322 * 10 ^ 230) (Ē := 3309 * 10 ^ 229) (Ā' := 1951 * 10 ^ 231)
              (Ē' := 2780 * 10 ^ 230) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.diagBack M (w.s1 0).sge (w.s1 0).fsge (by norm_num) (w.s1 0).hsge (w.s1 0).hfsge
            hMu (by norm_num) (by norm_num)
            (Ā := 1124 * 10 ^ 233) (Ē := 1602 * 10 ^ 232) (Ā' := 1698 * 10 ^ 233)
              (Ē' := 2516 * 10 ^ 232) (by norm_num [FloatModel.mulErr, u32])
              (by norm_num [FloatModel.mulErr, u32]) )
          (by norm_num [u32]) (by norm_num [u32])
          (FloatBridgesTo.Maps.chanLNTensor3Back (h := 56) (w := 56) M (w.s1 0).ln.γ (w.s1 0).ln.fγ (w.s1 0).ln.x (w.s1 0).ln.fs (w.s1 0).ln.fxh
              (by norm_num) (by norm_num) (w.s1 0).ln.hγ (w.s1 0).ln.hfγ (w.s1 0).ln.hst
                (w.s1 0).ln.hSabs
              (fun _r i => bnXhat_abs_le_num (X := 10) P.hε _ (by norm_num) (by norm_num) i)
                (w.s1 0).ln.hfxh
              hMu (M.gamma_num (q := 5782 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
                (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
              (Kr := 1632) (Kb := 4234 / 10 ^ 3) hK96r hK96b (by norm_num) (by norm_num)
              (Ā := 3913 * 10 ^ 235) (Ē := 5798 * 10 ^ 234) (Ā' := 3074 * 10 ^ 239)
                (Ē' := 4622 * 10 ^ 238)
              (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
          (by norm_num [u32]) (by norm_num [u32]))
        hMu (by norm_num [u32]) (by norm_num [u32])))
  have m9 := m8.comp (by norm_num)
    ((FloatBridgesTo.Maps.chanLNTensor3Back (h := 56) (w := 56) M w.lnStem.γ w.lnStem.fγ w.lnStem.x w.lnStem.fs w.lnStem.fxh
        (by norm_num) (by norm_num) w.lnStem.hγ w.lnStem.hfγ w.lnStem.hst w.lnStem.hSabs
        (fun _r i => bnXhat_abs_le_num (X := 10) P.hε _ (by norm_num) (by norm_num) i) w.lnStem.hfxh
        hMu (M.gamma_num (q := 5782 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          (by norm_num) le_rfl (by norm_num) (by norm_num) (by norm_num) (by norm_num)
        (Kr := 1632) (Kb := 4234 / 10 ^ 3) hK96r hK96b (by norm_num) (by norm_num)
        (Ā := 9039 * 10 ^ 240) (Ē := 1360 * 10 ^ 240) (Ā' := 7100 * 10 ^ 244)
          (Ē' := 1084 * 10 ^ 244)
        (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]) ).comp
          (by norm_num)
      (FloatBridgesTo.Maps.flatConvStride4Back (h := 56) (w := 56) M (padOdd w.sW.W) P.hwk
        (by norm_num) (by norm_num) (padOdd_abs_le w.sW.W P.hwk w.sW.hW)
        (M.gamma_num (q := 1432 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (Ā := 7100 * 10 ^ 244) (Ē := 1084 * 10 ^ 244) (Ā' := 1023 * 10 ^ 248)
          (Ē' := 1563 * 10 ^ 247)
        (by norm_num [u32]) (by norm_num [u32])))
  exact m9

/-- The certified output window of ConvNeXt-T's input-gradient: `≤ 1.023·10²⁵¹` per input pixel,
    on loss cotangents of magnitude `≤ 1`. -/
theorem cnxGradBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : CnxBackWeights ε (6/10) (48/10) (84/10) 16 (1/100) (1/100) (1/100)) :
    (cnxGradBridge M (cnxBackProfile_committed M hMu hε5) w).mag 1 ≤ 1023 * 10 ^ 248 :=
  (cnxGradBridge_maps M hMu hε5 w).mag_le 1 (by norm_num) le_rfl

/-- The fresh budget of ConvNeXt-T's input-gradient: `≤ 1.563·10²⁵⁰`. -/
theorem cnxGradBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : CnxBackWeights ε (6/10) (48/10) (84/10) 16 (1/100) (1/100) (1/100)) :
    (cnxGradBridge M (cnxBackProfile_committed M hMu hε5) w).fresh 1 ≤ 1563 * 10 ^ 247 :=
  (cnxGradBridge_maps M hMu hε5 w).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed ConvNeXt-T float input-gradient is within `1.563·10²⁵⁰` of the certified
    real one, per input pixel**, on loss cotangents of magnitude `≤ 1`, at the measured per-kind
    profile, `ε ≥ 10⁻⁵`, `u ≤ 2⁻²⁴`, saved-activation accuracies `10⁻²` and the operating point
    `|istd| ≤ 16`. Certified window `1.023·10²⁵¹`, so `budget / window = 0.153` — ⭐ **the interval
    FOLD**, at a net whose own FORWARD number is a `2.00` cap.

    ⛔ **Read the file header before quoting it.** This does NOT compose with the forward: this
    net's forward statement is `capped`, so the saved-activation accuracy it can supply is
    `2 × window ≈ 10²²⁷`, and LayerNorm has no eval mode to switch to. The fold is real and the
    forward-then-backward composition does not exist; say both halves. -/
theorem cnx_grad_float_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : CnxBackWeights ε (6/10) (48/10) (84/10) 16 (1/100) (1/100) (1/100))
    (dy : Vec 10) (hdy : ∀ k, |dy k| ≤ 1) (j : Fin (3 * 224 * 224)) :
    |cnxGradF M w dy j - cnxGradR w dy j| ≤ 1563 * 10 ^ 247 :=
  (cnxGradBridge_maps M hMu hε5 w).budget_le (by norm_num) le_rfl dy hdy j

end Proofs