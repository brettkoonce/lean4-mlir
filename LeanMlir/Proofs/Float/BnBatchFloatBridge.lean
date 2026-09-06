import LeanMlir.Proofs.Float.FloatBudgetEnv
import LeanMlir.Proofs.Float.FloatBudgetEnvBack
import LeanMlir.Proofs.Float.BnPerChannelFloatBridge

/-! # ℝ→Float32 bridge for TRUE batch-norm — the batched world's missing leaf

`BnPerChannelFloatBridge.lean` and `FloatBudgetEnvBack.lean` carry the PER-EXAMPLE BatchNorm
leaves (`floatBridgesTo_bnPerChannelTensor3`, `floatBridgesTo_bnPerChannelBack`) that every
whole-net number in the repo goes through. This file adds their peers for **`bnBatchTensor4`**,
the training-mode BatchNorm that reduces μ/var over the batch+spatial axes `[0,2,3]` — the one op
in a batched net that couples examples, and the normalisation every Adam/momentum train step in
`verified_mlir/` actually emits.

## Why it did not exist

⛔ Before this file the batched world had **no float leaf in either direction**. That is not the
same gap `planning/proofs_tier_to_paper_nets.md` §4 named: it priced a missing *backward* leaf,
and the forward was missing too. `EfficientNetWholeFloatBridge.lean` takes `bnBatchLA`'s bridge as
a HYPOTHESIS (`hbn : FloatBridges (StableHLO.bnBatchLA …)`, twenty-odd of them), and a legacy
`FloatBridges` constrains no float implementation at all (`formalization.yaml` 4d) — so those
hypotheses named nothing and nothing discharged them. B0's own numbers dodge the hole two ways:
`b0_float_logits_le` is at INFERENCE BN (frozen statistics, `batchMap` of a per-example op) and
`b0_grad_float_le` is at `N = 1`, where the batched reduction width `N·h·w` coincides with the
per-example `h·w` and `floatBridgesTo_bnPerChannelBack` is the honest leaf.

## Why it is short

⭐⭐ **`bnBatchTensor4` IS `bnPerChannelTensor3` at a different width.** Both are
`bnPerChannelFlat oc m` conjugated by a permutation (`PerChannelBN.lean`): the per-example op
relabels `(oc·h)·w ↔ oc·(h·w)` and takes `m = h·w`; the batched op relabels
`[N,C,H,W] ↔ [C, N·H·W]` and takes `m = N·(h·w)`. `floatBridgesTo_bnPerChannelFlat` and
`floatBridgesTo_bnPerChannelFlatBack` are already generic in `m`, and `floatBridgesTo_gather`
holds for ANY equiv, so the leaf is a re-conjugation: one new `Equiv` (`bnchwEquiv`, from the
two round-trip lemmas that were already proven) and the same two `.comp`s.

⭐ The `Maps` envelopes are shorter still, because their proofs never mention the layout — they
are statements about `bnLeafMag`/`bnLeafMod` (forward) and `bnGradInputReMag`/`bnGradInputBudgetG`
(backward), which take a WIDTH and no indices. `Maps.bnLeafCore` / `Maps.bnGradLeafCore` below
factor that out, so the batched envelopes are three lines each and a fourth copy (R50, MNv4)
costs nothing.

## What the batch axis actually changes

⚠ **The width is the entire content of the batch world, and it is not cosmetic.** Every hypothesis
here quantifies over `Vec (N * (h * w))`, so the supplied statistic moduli `emean`/`eistd`/`es`/
`exh` are claims about a reduction `N` times wider, and `bnGradInputReMag`'s gain `S·G·(2 + Xh²)`
carries `Xh² = n = N·h·w`. A number built on this leaf therefore MOVES WITH THE BATCH SIZE — one
theorem per `N`, not one theorem for all `N` the way an inference-BN forward gets. That is B0's
`b0_grad_float_le` qualifier stated at the leaf instead of at the net.

⚠ This file states the leaf at `bnBatchTensor4` (`Foundation/PerChannelBN.lean`), not at
`StableHLO.bnBatchLA`, so it stays out of the `Codegen` cone like every other `Float/` leaf.
`bnBatchLA` is `bnBatchTensor4` conjugated by one more `Fin.cast` reindex — magnitude-stable with
modulus `id` — so a net-level bridge composes `floatBridgesTo_gather` on each side and inherits
these `mag`/`mod` unchanged.
-/

namespace Proofs

open FloatModel
open scoped Real

-- ════════════════════════════════════════════════════════════════
-- § The transpose, as an `Equiv`
-- ════════════════════════════════════════════════════════════════

/-- The `[N,C,H,W] ↔ [C, N·H·W]` transpose as an `Equiv` — `bnchwFwdIdx` and `bnchwBackIdx` are
    mutual inverses (`bnchwFwdIdx_bnchwBackIdx`, `bnchwBackIdx_bnchwFwdIdx`, both already proven
    in `PerChannelBN.lean`), so this is a genuine relabeling. The batched peer of `reassocEquiv`,
    and the only genuinely new artifact in this file. -/
noncomputable def bnchwEquiv (N oc h w : Nat) :
    Fin (oc * (N * (h * w))) ≃ Fin (N * (oc * (h * w))) where
  toFun := bnchwFwdIdx N oc h w
  invFun := bnchwBackIdx N oc h w
  left_inv := bnchwBackIdx_bnchwFwdIdx N oc h w
  right_inv := bnchwFwdIdx_bnchwBackIdx N oc h w

-- ════════════════════════════════════════════════════════════════
-- § FORWARD: the batched BatchNorm leaf
-- ════════════════════════════════════════════════════════════════

/-- The float true-batch-norm on the `[N,C,H,W]` layout — the same `bnchw` conjugation as the
    real op, both permutations exact in float. The batched peer of `bnPerChannelTensor3FV`, and
    note the statistics `fμ`/`fistdv` now read a `Vec (N * (h * w))`: one μ/var per channel for
    the WHOLE batch. -/
noncomputable def bnBatchTensor4FV {N oc h w : Nat} (M : FloatModel) (γ β : Vec oc)
    (fμ fistdv : Fin oc → Vec (N * (h * w)) → ℝ) :
    Vec (N * (oc * (h * w))) → Vec (N * (oc * (h * w))) :=
  bnchwBack N oc h w ∘ bnPerChannelFlatFV M γ β fμ fistdv ∘ bnchwFwd N oc h w

/-- ⭐ **True batch-norm float-bridges TO its float map.** `floatBridgesTo_bnPerChannelTensor3`'s
    conjugation with `reassocEquiv` replaced by `bnchwEquiv` and the reduction width `h·w`
    replaced by `N·(h·w)`; the two gathers are magnitude-stable with modulus `id`, so the
    composite's `mag`/`mod` are the BN's own `bnLeafMag`/`bnLeafMod` — the SAME pair the
    per-example leaf carries, at the wider `S`/`emean`/`eistd`. -/
noncomputable def floatBridgesTo_bnBatchTensor4 {N oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec (N * (h * w)) → ℝ) (emean eistd : ℝ → ℝ)
    {G Bbnd S : ℝ}
    (hoc : 0 < oc) (hNhw : 0 < N * (h * w)) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec (N * (h * w)), (∀ k, |v k| ≤ A) →
        |fμ c v - bnMean (N * (h * w)) v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec (N * (h * w)), (∀ k, |v k| ≤ A) →
        |fistdv c v - bnIstd (N * (h * w)) v ε| ≤ eistd A)
    (hS : ∀ v : Vec (N * (h * w)), |bnIstd (N * (h * w)) v ε| ≤ S) :
    FloatBridgesTo (bnBatchTensor4 N oc h w ε γ β) (bnBatchTensor4FV M γ β fμ fistdv) :=
  ⟨bnLeafMag M.u S G Bbnd emean eistd, bnLeafMod M.u ε S G Bbnd emean eistd,
   ((floatBridgesTo_gather (bnchwEquiv N oc h w)).comp
      (floatBridgesTo_bnPerChannelFlat M γ β fμ fistdv emean eistd hoc hNhw hε hγ hβ
        hmean histd hS)).comp
     (floatBridgesTo_gather (bnchwEquiv N oc h w).symm) |>.close⟩

/-- **The `ε`-floor instantiation** — the batched BN leaf with `S := 1/√ε`, closed
    unconditionally (`bnIstd_abs_le`). The batched peer of
    `floatBridgesTo_bnPerChannelTensor3_eps`, and the form a whole-net number uses when it
    declines an operating point on the inverse standard deviation. -/
noncomputable def floatBridgesTo_bnBatchTensor4_eps {N oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec (N * (h * w)) → ℝ) (emean eistd : ℝ → ℝ)
    {G Bbnd : ℝ}
    (hoc : 0 < oc) (hNhw : 0 < N * (h * w)) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec (N * (h * w)), (∀ k, |v k| ≤ A) →
        |fμ c v - bnMean (N * (h * w)) v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec (N * (h * w)), (∀ k, |v k| ≤ A) →
        |fistdv c v - bnIstd (N * (h * w)) v ε| ≤ eistd A) :
    FloatBridgesTo (bnBatchTensor4 N oc h w ε γ β) (bnBatchTensor4FV M γ β fμ fistdv) :=
  floatBridgesTo_bnBatchTensor4 M γ β fμ fistdv emean eistd hoc hNhw hε hγ hβ
    hmean histd (fun v => bnIstd_abs_le v hε)

-- ════════════════════════════════════════════════════════════════
-- § BACKWARD: the batched BatchNorm backward leaf
-- ════════════════════════════════════════════════════════════════

/-- The float batch-norm BACKWARD on the `[N,C,H,W]` layout — the same `bnchw` conjugation as
    `bnBatchTensor4_grad_input`, both permutations exact in float. The batched peer of
    `bnPerChannelTensor3BackFV`; the saved normalised activation `fxh` is now one
    `Vec (N * (h * w))` per channel, read off the whole batch's forward. -/
noncomputable def bnBatchTensor4BackFV {N oc h w : Nat} (M : FloatModel) (γ : Vec oc)
    (fs : Fin oc → ℝ) (fxh : Fin oc → Vec (N * (h * w))) :
    Vec (N * (oc * (h * w))) → Vec (N * (oc * (h * w))) :=
  bnchwBack N oc h w ∘ bnPerChannelFlatBackFV M γ fs fxh ∘ bnchwFwd N oc h w

/-- ⭐ **True batch-norm BACKWARD float-bridges TO its float map** — the leaf §4 of
    `planning/proofs_tier_to_paper_nets.md` priced as "the real cost" of the batched world.
    `floatBridgesTo_bnPerChannelBack`'s conjugation at `bnchwEquiv` and width `N·(h·w)`.

    ⭐ Like its per-example twin this is an honest FOLD, not a cap: the statistics are read off
    the SAVED forward activations, which the cotangent does not perturb, so `mod` is linear in
    the inherited error. A VJP at a fixed point is a linear map and the batch axis does not
    change that — it only widens `n`. -/
noncomputable def floatBridgesTo_bnBatchBack {N oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ : Vec oc) (x : Vec (N * (oc * (h * w)))) (fs : Fin oc → ℝ)
    (fxh : Fin oc → Vec (N * (h * w)))
    {G S Xh es exh : ℝ} (hoc : 0 < oc) (hNhw : 0 < N * (h * w))
    (hγ : ∀ c, |γ c| ≤ G)
    (hs : ∀ c, |fs c
      - bnIstd (N * (h * w)) (Mat.unflatten (bnchwFwd N oc h w x) c) ε| ≤ es)
    (hSabs : ∀ c, |bnIstd (N * (h * w)) (Mat.unflatten (bnchwFwd N oc h w x) c) ε| ≤ S)
    (hxh : ∀ c i,
      |bnXhat (N * (h * w)) ε (Mat.unflatten (bnchwFwd N oc h w x) c) i| ≤ Xh)
    (hfxh : ∀ c i,
      |fxh c i - bnXhat (N * (h * w)) ε (Mat.unflatten (bnchwFwd N oc h w x) c) i| ≤ exh) :
    FloatBridgesTo (fun dy => bnBatchTensor4_grad_input N oc h w ε γ x dy)
      (bnBatchTensor4BackFV M γ fs fxh) :=
  ⟨fun A => bnGradInputReMag (N * (h * w)) G A S Xh
              + M.bnGradInputBudget (N * (h * w)) G A S Xh es exh,
   fun A e => M.bnGradInputBudget (N * (h * w)) G A S Xh es exh
              + bnGradInputReMag (N * (h * w)) G e S Xh,
   (((floatBridgesTo_gather (bnchwEquiv N oc h w)).comp
      (floatBridgesTo_bnPerChannelFlatBack M γ (bnchwFwd N oc h w x) fs fxh hoc hNhw
        hγ hs hSabs hxh hfxh)).comp
      (floatBridgesTo_gather (bnchwEquiv N oc h w).symm)).close⟩

-- ════════════════════════════════════════════════════════════════
-- § `Maps` envelopes — the numeral-facing half
-- ════════════════════════════════════════════════════════════════

namespace FloatBridgesTo

/-! ⭐ The two cores below are `Maps.bnPerChannelTensor3` / `Maps.bnPerChannelTensor3Capped` /
`Maps.bnPerChannelBack` with the LAYOUT abstracted away. Those proofs never mention `oc`, `h` or
`w` except to pick a channel index for a nonnegativity side condition: they are statements about
`bnLeafMag`/`bnLeafMod` and `bnGradInputReMag`/`bnGradInputBudgetG`, which take a WIDTH and no
indices. Factoring that out is what makes the batched envelopes three lines each — and what keeps
R50's and MobileNetV4's from being a fourth and fifth copy of the same `linarith` chain. -/

/-- **The BatchNorm forward envelope, layout-free.** Any bridge whose `mag`/`mod` are the BN
    leaf's pair maps `(Ā, Ē)` into `(Ā', Ē')` under the two closing inequalities. ⚠ The second is
    QUADRATIC in the window (`2·Ā·(8·Ā·Ē·Tq)`) — that is training-mode BatchNorm's modulus and the
    reason a deep training-BN forward needs `capped` below rather than this. -/
theorem Maps.bnLeafCore {m n : Nat} {f fF : Vec m → Vec n} (M : FloatModel) {ε : ℝ}
    (b : FloatBridgesTo f fF) {G Bbnd S : ℝ} {emean eistd : ℝ → ℝ}
    (hbmag : b.mag = bnLeafMag M.u S G Bbnd emean eistd)
    (hbmod : b.mod = bnLeafMod M.u ε S G Bbnd emean eistd)
    (hε : 0 < ε) (hemn : ∀ A, 0 ≤ A → 0 ≤ emean A) (hein : ∀ A, 0 ≤ A → 0 ≤ eistd A)
    {q em ei Sq Tq Ā Ē Ā' Ē' : ℝ}
    (hq : M.u ≤ q) (hG0 : 0 ≤ G) (hB0 : 0 ≤ Bbnd) (hS0 : 0 ≤ S)
    (hĀ0 : 0 ≤ Ā) (hĒ0 : 0 ≤ Ē)
    (hem : ∀ A, 0 ≤ A → A ≤ Ā → emean A ≤ em) (hei : ∀ A, 0 ≤ A → A ≤ Ā → eistd A ≤ ei)
    (hSq : 1 / Real.sqrt ε ≤ Sq) (hTq : 1 / (2 * ε * Real.sqrt ε) ≤ Tq)
    (hĀ' : G * (2 * Ā * S) + Bbnd + bnNormBudget q (2 * Ā) S G Bbnd em ei ≤ Ā')
    (hĒ' : bnNormBudget q (2 * Ā) S G Bbnd em ei
             + G * ((Ē + Ē) * Sq + 2 * Ā * (8 * Ā * Ē * Tq)) ≤ Ē') :
    b.Maps Ā Ē Ā' Ē' := by
  have hu := M.u_nonneg
  have hsε : 0 < Real.sqrt ε := Real.sqrt_pos.mpr hε
  have hinv0 : (0:ℝ) ≤ 1 / Real.sqrt ε := by positivity
  have hlip0 : (0:ℝ) ≤ 1 / (2 * ε * Real.sqrt ε) := by positivity
  constructor
  · intro A h0 hle
    rw [hbmag]
    unfold bnLeafMag
    have hnb := bnNormBudget_mono (u := M.u) (u' := q) (D := 2 * A) (D' := 2 * Ā)
      (S := S) (G := G) (Bb := Bbnd) hu hq (by linarith) (by linarith) hS0 hG0 hB0
      (hemn A h0) (hem A h0 hle) (hein A h0) (hei A h0 hle)
    have hmg : G * (2 * A * S) ≤ G * (2 * Ā * S) :=
      mul_le_mul_of_nonneg_left (by nlinarith) hG0
    linarith
  · intro A E h0 hE0 hle hEle
    rw [hbmod]
    unfold bnLeafMod bnReluBudget
    have hnb := bnNormBudget_mono (u := M.u) (u' := q) (D := 2 * A) (D' := 2 * Ā)
      (S := S) (G := G) (Bb := Bbnd) hu hq (by linarith) (by linarith) hS0 hG0 hB0
      (hemn A h0) (hem A h0 hle) (hein A h0) (hei A h0 hle)
    have ht1 : (E + E) * (1 / Real.sqrt ε) ≤ (Ē + Ē) * Sq :=
      mul_le_mul (by linarith) hSq hinv0 (by linarith)
    have hdiv : (8 * A * E) / (2 * ε * Real.sqrt ε) ≤ 8 * Ā * Ē * Tq := by
      rw [div_eq_mul_one_div]
      exact mul_le_mul (by nlinarith) hTq hlip0 (by nlinarith)
    have hdiv0 : (0:ℝ) ≤ (8 * A * E) / (2 * ε * Real.sqrt ε) := by positivity
    have ht2 : 2 * A * ((8 * A * E) / (2 * ε * Real.sqrt ε)) ≤ 2 * Ā * (8 * Ā * Ē * Tq) :=
      mul_le_mul (by linarith) hdiv hdiv0 (by linarith)
    have htail : G * ((E + E) * (1 / Real.sqrt ε)
          + 2 * A * ((8 * A * E) / (2 * ε * Real.sqrt ε)))
        ≤ G * ((Ē + Ē) * Sq + 2 * Ā * (8 * Ā * Ē * Tq)) :=
      mul_le_mul_of_nonneg_left (by linarith) hG0
    linarith

/-- **The CAPPED BatchNorm forward envelope, layout-free** — `Maps.capped` over the WINDOW
    clause alone. ⛔ A number built on this is the triangle inequality, not the fold: it says the
    float and the real forward both land in the certified window, and its tell is
    `budget / window = 2.00`. The quadratic modulus is never turned into a numeral. -/
theorem Maps.bnLeafCoreCapped {m n : Nat} {f fF : Vec m → Vec n} (M : FloatModel)
    (b : FloatBridgesTo f fF) {G Bbnd S : ℝ} {emean eistd : ℝ → ℝ}
    (hbmag : b.mag = bnLeafMag M.u S G Bbnd emean eistd)
    (hemn : ∀ A, 0 ≤ A → 0 ≤ emean A) (hein : ∀ A, 0 ≤ A → 0 ≤ eistd A)
    {q em ei Ā Ē Ā' Ē' : ℝ}
    (hq : M.u ≤ q) (hG0 : 0 ≤ G) (hB0 : 0 ≤ Bbnd) (hS0 : 0 ≤ S)
    (hem : ∀ A, 0 ≤ A → A ≤ Ā → emean A ≤ em) (hei : ∀ A, 0 ≤ A → A ≤ Ā → eistd A ≤ ei)
    (hĀ' : G * (2 * Ā * S) + Bbnd + bnNormBudget q (2 * Ā) S G Bbnd em ei ≤ Ā')
    (hĒ' : 2 * Ā' ≤ Ē') :
    b.capped.Maps Ā Ē Ā' Ē' := by
  refine Maps.capped (Ē := Ē) (fun A h0 hle => ?_) hĒ'
  have hu := M.u_nonneg
  rw [hbmag]
  unfold bnLeafMag
  have hnb := bnNormBudget_mono (u := M.u) (u' := q) (D := 2 * A) (D' := 2 * Ā)
    (S := S) (G := G) (Bb := Bbnd) hu hq (by linarith) (by linarith) hS0 hG0 hB0
    (hemn A h0) (hem A h0 hle) (hein A h0) (hei A h0 hle)
  have hmg : G * (2 * A * S) ≤ G * (2 * Ā * S) :=
    mul_le_mul_of_nonneg_left (by nlinarith) hG0
  linarith

/-- **The BatchNorm BACKWARD envelope, layout-free.** Both closing inequalities are stated over
    the rational budget `bnGradInputBudgetG` at the input window `Ā` and transported to every
    `A ≤ Ā` by the two monotonicity lemmas — the leaf is linear in the cotangent, so there is
    nothing cleverer to do and nothing is lost. Not capped, and it does not need to be. -/
theorem Maps.bnGradLeafCore {m n₀ : Nat} {f fF : Vec m → Vec n₀} (M : FloatModel)
    (b : FloatBridgesTo f fF) (n : Nat) {G S Xh es exh : ℝ}
    (hbmag : b.mag = fun A => bnGradInputReMag n G A S Xh + M.bnGradInputBudget n G A S Xh es exh)
    (hbmod : b.mod = fun A e => M.bnGradInputBudget n G A S Xh es exh
                                 + bnGradInputReMag n G e S Xh)
    {q gn Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hgn : (1 + M.u) ^ (n + 1) - 1 ≤ gn)
    (hG0 : 0 ≤ G) (hS0 : 0 ≤ S) (hXh0 : 0 ≤ Xh) (hes0 : 0 ≤ es) (hexh0 : 0 ≤ exh)
    (hĀ' : bnGradInputReMag n G Ā S Xh + bnGradInputBudgetG q gn n G Ā S Xh es exh ≤ Ā')
    (hĒ' : bnGradInputBudgetG q gn n G Ā S Xh es exh + bnGradInputReMag n G Ē S Xh ≤ Ē') :
    b.Maps Ā Ē Ā' Ē' where
  mag_le := fun A h0 hle => by
    rw [hbmag]
    have h1 := bnGradInputReMag_mono n (G := G) (S := S) (Xh := Xh) h0 hle hG0 hS0 hXh0
    have h2 := bnGradInputBudget_le M n (G := G) (S := S) (Xh := Xh) (es := es)
      (exh := exh) hq hgn h0 hle hG0 hS0 hXh0 hes0 hexh0
    linarith
  mod_le := fun A E h0 hE0 hle hEle => by
    rw [hbmod]
    have h1 := bnGradInputReMag_mono n (G := G) (S := S) (Xh := Xh) hE0 hEle hG0 hS0 hXh0
    have h2 := bnGradInputBudget_le M n (G := G) (S := S) (Xh := Xh) (es := es)
      (exh := exh) hq hgn h0 hle hG0 hS0 hXh0 hes0 hexh0
    linarith

-- ════════════════════════════════════════════════════════════════
-- § The batched envelopes
-- ════════════════════════════════════════════════════════════════

/-- ⭐ **An envelope through a TRUE batch-norm.** `Maps.bnPerChannelTensor3`'s statement at the
    batched reduction width `N·(h·w)`; the proof is the layout-free core, applied at `rfl`.

    ⚠ `em`/`ei` bound the SUPPLIED float mean/inv-stddev accuracy over a reduction `N` times
    wider than the per-example op's. A GPU `rsqrt` has no IEEE spec, so — as on the per-example
    leaf — the BN statistics are modelled, not derived. -/
theorem Maps.bnBatchTensor4 {N oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec (N * (h * w)) → ℝ) (emean eistd : ℝ → ℝ)
    {G Bbnd S : ℝ}
    (hoc : 0 < oc) (hNhw : 0 < N * (h * w)) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec (N * (h * w)), (∀ k, |v k| ≤ A) →
        |fμ c v - bnMean (N * (h * w)) v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec (N * (h * w)), (∀ k, |v k| ≤ A) →
        |fistdv c v - bnIstd (N * (h * w)) v ε| ≤ eistd A)
    (hSb : ∀ v : Vec (N * (h * w)), |bnIstd (N * (h * w)) v ε| ≤ S)
    {q em ei Sq Tq Ā Ē Ā' Ē' : ℝ}
    (hq : M.u ≤ q) (hG0 : 0 ≤ G) (hB0 : 0 ≤ Bbnd) (hS0 : 0 ≤ S)
    (hĀ0 : 0 ≤ Ā) (hĒ0 : 0 ≤ Ē)
    (hem : ∀ A, 0 ≤ A → A ≤ Ā → emean A ≤ em) (hei : ∀ A, 0 ≤ A → A ≤ Ā → eistd A ≤ ei)
    (hSq : 1 / Real.sqrt ε ≤ Sq) (hTq : 1 / (2 * ε * Real.sqrt ε) ≤ Tq)
    (hĀ' : G * (2 * Ā * S) + Bbnd + bnNormBudget q (2 * Ā) S G Bbnd em ei ≤ Ā')
    (hĒ' : bnNormBudget q (2 * Ā) S G Bbnd em ei
             + G * ((Ē + Ē) * Sq + 2 * Ā * (8 * Ā * Ē * Tq)) ≤ Ē') :
    (floatBridgesTo_bnBatchTensor4 M γ β fμ fistdv emean eistd hoc hNhw hε hγ hβ
      hmean histd hSb).Maps Ā Ē Ā' Ē' :=
  Maps.bnLeafCore M _ rfl rfl hε
    (fun A hA => (abs_nonneg _).trans (hmean ⟨0, hoc⟩ A hA 0 (fun _ => by simpa using hA)))
    (fun A hA => (abs_nonneg _).trans (histd ⟨0, hoc⟩ A hA 0 (fun _ => by simpa using hA)))
    hq hG0 hB0 hS0 hĀ0 hĒ0 hem hei hSq hTq hĀ' hĒ'

/-- ⭐⭐ **The CAPPED true batch-norm envelope** — the batched peer of
    `Maps.bnPerChannelTensor3Capped`, and the form a deep TRAINING-BN forward needs. Only the
    window clause survives, plus `Maps.capped`'s own `2·Ā' ≤ Ē'`; no `Sq`, no `Tq`, no bound on
    the inherited error, so the quadratic modulus is never turned into a numeral. -/
theorem Maps.bnBatchTensor4Capped {N oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ β : Vec oc) (fμ fistdv : Fin oc → Vec (N * (h * w)) → ℝ) (emean eistd : ℝ → ℝ)
    {G Bbnd S : ℝ}
    (hoc : 0 < oc) (hNhw : 0 < N * (h * w)) (hε : 0 < ε)
    (hγ : ∀ c, |γ c| ≤ G) (hβ : ∀ c, |β c| ≤ Bbnd)
    (hmean : ∀ c A, 0 ≤ A → ∀ v : Vec (N * (h * w)), (∀ k, |v k| ≤ A) →
        |fμ c v - bnMean (N * (h * w)) v| ≤ emean A)
    (histd : ∀ c A, 0 ≤ A → ∀ v : Vec (N * (h * w)), (∀ k, |v k| ≤ A) →
        |fistdv c v - bnIstd (N * (h * w)) v ε| ≤ eistd A)
    (hSb : ∀ v : Vec (N * (h * w)), |bnIstd (N * (h * w)) v ε| ≤ S)
    {q em ei Ā Ē Ā' Ē' : ℝ}
    (hq : M.u ≤ q) (hG0 : 0 ≤ G) (hB0 : 0 ≤ Bbnd) (hS0 : 0 ≤ S)
    (hem : ∀ A, 0 ≤ A → A ≤ Ā → emean A ≤ em) (hei : ∀ A, 0 ≤ A → A ≤ Ā → eistd A ≤ ei)
    (hĀ' : G * (2 * Ā * S) + Bbnd + bnNormBudget q (2 * Ā) S G Bbnd em ei ≤ Ā')
    (hĒ' : 2 * Ā' ≤ Ē') :
    (floatBridgesTo_bnBatchTensor4 M γ β fμ fistdv emean eistd hoc hNhw hε hγ hβ
      hmean histd hSb).capped.Maps Ā Ē Ā' Ē' :=
  Maps.bnLeafCoreCapped M _ rfl
    (fun A hA => (abs_nonneg _).trans (hmean ⟨0, hoc⟩ A hA 0 (fun _ => by simpa using hA)))
    (fun A hA => (abs_nonneg _).trans (histd ⟨0, hoc⟩ A hA 0 (fun _ => by simpa using hA)))
    hq hG0 hB0 hS0 hem hei hĀ' hĒ'

/-- ⭐⭐ **An envelope through a TRUE batch-norm BACKWARD** — the numeral-facing end of the leaf
    §4 priced as the batched world's real cost, and an honest FOLD at training-mode BatchNorm.

    ⚠ Every occurrence of the width is `N·(h·w)`, and `bnGradInputReMag`'s gain is
    `S·G·(2 + Xh²)` with `Xh² = n`: this envelope's numerals grow with the batch size. One
    theorem per `N`. -/
theorem Maps.bnBatchBack {N oc h w : Nat} (M : FloatModel) {ε : ℝ}
    (γ : Vec oc) (x : Vec (N * (oc * (h * w)))) (fs : Fin oc → ℝ)
    (fxh : Fin oc → Vec (N * (h * w)))
    {G S Xh es exh : ℝ} (hoc : 0 < oc) (hNhw : 0 < N * (h * w))
    (hγ : ∀ c, |γ c| ≤ G)
    (hs : ∀ c, |fs c
      - bnIstd (N * (h * w)) (Mat.unflatten (bnchwFwd N oc h w x) c) ε| ≤ es)
    (hSabs : ∀ c, |bnIstd (N * (h * w)) (Mat.unflatten (bnchwFwd N oc h w x) c) ε| ≤ S)
    (hxh : ∀ c i,
      |bnXhat (N * (h * w)) ε (Mat.unflatten (bnchwFwd N oc h w x) c) i| ≤ Xh)
    (hfxh : ∀ c i,
      |fxh c i - bnXhat (N * (h * w)) ε (Mat.unflatten (bnchwFwd N oc h w x) c) i| ≤ exh)
    {q gn Ā Ē Ā' Ē' : ℝ} (hq : M.u ≤ q)
    (hgn : (1 + M.u) ^ (N * (h * w) + 1) - 1 ≤ gn)
    (hG0 : 0 ≤ G) (hS0 : 0 ≤ S) (hXh0 : 0 ≤ Xh) (hes0 : 0 ≤ es) (hexh0 : 0 ≤ exh)
    (hĀ' : bnGradInputReMag (N * (h * w)) G Ā S Xh
            + bnGradInputBudgetG q gn (N * (h * w)) G Ā S Xh es exh ≤ Ā')
    (hĒ' : bnGradInputBudgetG q gn (N * (h * w)) G Ā S Xh es exh
            + bnGradInputReMag (N * (h * w)) G Ē S Xh ≤ Ē') :
    (floatBridgesTo_bnBatchBack M γ x fs fxh hoc hNhw hγ hs hSabs hxh hfxh).Maps Ā Ē Ā' Ē' :=
  Maps.bnGradLeafCore M _ (N * (h * w)) rfl rfl hq hgn hG0 hS0 hXh0 hes0 hexh0 hĀ' hĒ'

end FloatBridgesTo

end Proofs
