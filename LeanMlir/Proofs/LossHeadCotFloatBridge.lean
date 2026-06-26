import LeanMlir.Proofs.FloatBridge
import LeanMlir.Proofs.FloatComposeBridge

/-! # ℝ→Float32 bridge: the loss-head cotangent seed (A3 §1g — "from the loss")

A3 (planning/a3_backward_deepnet_assembly.md §1g): lift the per-entry softmax−onehot cotangent
closeness (`FloatModel.softmax_ce_cot_close`, `FloatBridge.lean`) to a `FloatClose`/`FloatBridges`
**seed** so a whole-net backward starts *from the loss*, not from an abstract cotangent `dy`.

The cross-entropy loss cotangent at the logits is `∂(CE)/∂logits = softmax(z) − onehot(label)`
(`softmaxCE_grad`); the deployed float head computes `M.softmaxCECotF fexp` (rounded softmax minus
onehot, at a float `exp` within `eexp`). `softmax_ce_cot_close` bounds the two per entry by `cotErr`
(softmax is bounded in `[0,1]`, so the seed never blows up — magnitude `1 + cotErr(0)` regardless of
the logit scale; `softmax_perturb` gives the Lipschitz part). Wrapping that as a `FloatBridges`
(`floatBridges_lossSeed`) lets it `.comp` with **any** `<net>_grad_floatBridges`
(`floatBridges_gradFromLoss`): the deployed float "loss → input-gradient" map is within an explicit
budget of the certified real one — the end-to-end backward certificate.

The softmax regime hypotheses (`hfexp`/`smRho < 1`) are the same transcendental-`exp` budget the
forward softmax uses; they are independent of the logits, so the seed is `FloatClose` on **every**
magnitude domain `A`.
-/

namespace Proofs

open scoped Real

-- ════════════════════════════════════════════════════════════════
-- § The loss-head cotangent as a FloatClose / FloatBridges seed
-- ════════════════════════════════════════════════════════════════

/-- **The softmax−onehot loss cotangent is `FloatClose`** — the seed of the from-the-loss backward.
    Real map `z ↦ softmax(z) − onehot(label)` (the CE input-gradient), float map
    `z ↦ M.softmaxCECotF fexp z label` (rounded softmax minus onehot). Output magnitude
    `1 + cotErr(0)` (the softmax is bounded in `[0,1]`, so `|softmax − onehot| ≤ 1` independent of the
    logit scale), modulus `e ↦ cotErr(e)` — `softmax_ce_cot_close` at `δ := e`. Holds on every domain
    `A` (the bound is logit-scale-free). -/
theorem floatClose_lossSeed (M : FloatModel) (fexp : ℝ → ℝ) {eexp : ℝ} {n : ℕ}
    (label : Fin n) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp n < 1) (A : ℝ) :
    FloatClose A (1 + FloatModel.cotErr M.u eexp 0 n)
      (fun z k => softmax n z k - oneHot n label k)
      (fun z => M.softmaxCECotF fexp z label)
      (fun e => FloatModel.cotErr M.u eexp e n) := by
  have hcot0 : 0 ≤ FloatModel.cotErr M.u eexp 0 n := M.cotErr_nonneg heexp0 le_rfl hρ1
  refine ⟨fun z _hz k => ?_, fun zt za e _hza _hzt hd k => ?_⟩
  · -- magnitude: |real| ≤ 1, |float| ≤ 1 + cotErr(0)
    have hReal : |softmax n z k - oneHot n label k| ≤ 1 := by
      have hs1 : softmax n z k ≤ 1 := (abs_le.mp (FloatModel.softmax_abs_le_one z k)).2
      have hs0 : 0 ≤ softmax n z k :=
        div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun j _ => (Real.exp_pos _).le)
      unfold oneHot
      by_cases hk : k = label
      · simp only [if_pos hk]; rw [abs_le]; constructor <;> linarith
      · simp only [if_neg hk, sub_zero, abs_of_nonneg hs0]; exact hs1
    have hClose0 : |M.softmaxCECotF fexp z label k - (softmax n z k - oneHot n label k)|
        ≤ FloatModel.cotErr M.u eexp 0 n :=
      M.softmax_ce_cot_close fexp z z label heexp0 heexp1 hfexp hρ1 (fun k' => by simp) k
    refine ⟨hReal.trans (by linarith), ?_⟩
    calc |M.softmaxCECotF fexp z label k|
        ≤ |M.softmaxCECotF fexp z label k - (softmax n z k - oneHot n label k)|
          + |softmax n z k - oneHot n label k| := by
            simpa using abs_sub_le (M.softmaxCECotF fexp z label k)
              (softmax n z k - oneHot n label k) 0
      _ ≤ FloatModel.cotErr M.u eexp 0 n + 1 := add_le_add hClose0 hReal
      _ = 1 + FloatModel.cotErr M.u eexp 0 n := by ring
  · -- error: softmax_ce_cot_close at δ := e (inferred from hd)
    exact M.softmax_ce_cot_close fexp zt za label heexp0 heexp1 hfexp hρ1 hd k

/-- **The loss-head cotangent seed float-bridges.** The existential closure of `floatClose_lossSeed`
    (output magnitude `1 + cotErr(0) ≥ 0`). The seed every from-the-loss whole-net backward prepends
    via `.comp`. -/
theorem floatBridges_lossSeed (M : FloatModel) (fexp : ℝ → ℝ) {eexp : ℝ} {n : ℕ}
    (label : Fin n) (hn : 0 < n) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp n < 1) :
    FloatBridges (fun z k => softmax n z k - oneHot n label k) := fun A hA =>
  ⟨_, _, _, (floatClose_lossSeed M fexp label heexp0 heexp1 hfexp hρ1 A).cod_nonneg hA hn,
    floatClose_lossSeed M fexp label heexp0 heexp1 hfexp hρ1 A⟩

-- ════════════════════════════════════════════════════════════════
-- § The from-the-loss whole-net backward (seed .comp <net>_grad)
-- ════════════════════════════════════════════════════════════════

/-- **From-the-loss backward float-bridges.** Prepending the loss-head cotangent seed to any
    `<net>_grad` (the input-gradient VJP at an abstract cotangent) yields the whole "logits → input
    gradient" backward, and it float-bridges: the deployed float loss-gradient is within an explicit
    budget of the certified real one. One `.comp` — instantiate `netGrad := mnv2InputGrad …` /
    `cifar8InputGrad …` / `r34InputGrad …` / `convnextInputGrad …` to upgrade that net's `_grad` from
    "≈ at an abstract `dy`" to "≈ from the loss". -/
theorem floatBridges_gradFromLoss {n inDim : Nat} (M : FloatModel) (fexp : ℝ → ℝ) {eexp : ℝ}
    (label : Fin n) (hn : 0 < n) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp n < 1)
    {netGrad : Vec n → Vec inDim} (hnet : FloatBridges netGrad) :
    FloatBridges (netGrad ∘ (fun z k => softmax n z k - oneHot n label k)) :=
  (floatBridges_lossSeed M fexp label hn heexp0 heexp1 hfexp hρ1).comp hnet

end Proofs
