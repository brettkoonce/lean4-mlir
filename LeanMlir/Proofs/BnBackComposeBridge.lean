import LeanMlir.Proofs.BnBackFloatBridge
import LeanMlir.Proofs.FloatComposeBridge

/-! # BN backward as a composable `FloatClose` MAP (A3 1c)

The BN-backward keystone (`BnBackFloatBridge.bnGradInput_close`) bounds the float input-gradient
*per entry, at a fixed cotangent* `dy` (with `Cdy` bounding `dy`). To slot BatchNorm-back into a
whole-net `.comp` backward chain it must become a `FloatClose A B f fF L` over the **cotangent**:
on `dy` within magnitude `A`, the float map `bnGradInputF` is within `L e` of the real
`bn_grad_input` at cotangent error `e`.

The per-entry keystone is most of the error clause. The two extra pieces, both about the *real*
map (which is linear in `dy`, so genuinely Lipschitz):
* **magnitude** `|bn_grad_input … dy i| ≤ B` (`bn_grad_input_abs_le`) — the three-term bracket
  `n·dx̂ᵢ − Σdx̂ − x̂ᵢ·Σ(x̂·dx̂)` has magnitude `≤ MTr`, scaled by `|1/n·s| ≤ S/n`;
* **modulus** `L e` (`bn_grad_input_diff_abs_le`) — the real map is linear in `dy`, so its
  response to a cotangent perturbation `e` is the SAME bound at `Cdy := e`.

So `B = ReMag(A) + budget(A)`, `L e = budget(A) + ReMag(e)`, where `ReMag` is the real-map
magnitude and `budget` is the keystone's float roundoff. The supplied float `s` (`fs`) and `x̂`
(`fxh`) carry their accuracy `es`/`exh` exactly as the forward keystone did (rsqrt has no IEEE
spec). The shared backward op every BN/LN net's gradient folds (`cifarBn`, `r34`, `convnext`/`vit`
LayerNorm).
-/

namespace Proofs

open scoped Real

/-- Magnitude of the real BN input-gradient at cotangent magnitude `Cdy`: `(1/n·S)·MTr` with
    `MTr = n·(G·Cdy) + n·(G·Cdy) + Xh·(n·(Xh·(G·Cdy)))` — the three-term bracket bound scaled by
    the `(1/n)·s` factor. **Linear in `Cdy`**, which is exactly why it doubles as the Lipschitz
    modulus. -/
noncomputable def bnGradInputReMag (n : Nat) (G Cdy S Xh : ℝ) : ℝ :=
  1 / (n : ℝ) * S * ((n : ℝ) * (G * Cdy) + (n : ℝ) * (G * Cdy) + Xh * ((n : ℝ) * (Xh * (G * Cdy))))

/-- `bn_grad_input` spelled out (zeta-reduced): `(1/n)·s·(n·(γ·dyᵢ) − Σγ·dy − x̂ᵢ·Σ x̂·(γ·dy))`. -/
theorem bn_grad_input_eq {n : Nat} (ε γ : ℝ) (x dy : Vec n) (i : Fin n) :
    bn_grad_input n ε γ x dy i
      = 1 / (n : ℝ) * bnIstd n x ε
          * ((n : ℝ) * (γ * dy i) - (∑ k, γ * dy k)
              - bnXhat n ε x i * (∑ k, bnXhat n ε x k * (γ * dy k))) := by
  simp only [bn_grad_input]

/-- The real BN input-gradient at the zero cotangent is zero (it is linear in `dy`). -/
theorem bn_grad_input_zero {n : Nat} (ε γ : ℝ) (x : Vec n) (i : Fin n) :
    bn_grad_input n ε γ x (0 : Vec n) i = 0 := by
  simp [bn_grad_input]

/-- **Real BN input-gradient is Lipschitz in the cotangent** — for `dy`-perturbations within `e`,
    the output difference is `≤ ReMag(e)`. The map is the linear `(1/n)·s·(three-term)`, so this is
    the magnitude bound at `Cdy := e`. -/
theorem bn_grad_input_diff_abs_le {n : Nat} {ε γ : ℝ} (x vt va : Vec n)
    {G S Xh e : ℝ} (hn : 0 < n)
    (hγ : |γ| ≤ G) (hd : ∀ k, |vt k - va k| ≤ e)
    (hSabs : |bnIstd n x ε| ≤ S) (hxh : ∀ i, |bnXhat n ε x i| ≤ Xh) (i : Fin n) :
    |bn_grad_input n ε γ x vt i - bn_grad_input n ε γ x va i| ≤ bnGradInputReMag n G e S Xh := by
  have hnR : (0 : ℝ) ≤ (n : ℝ) := Nat.cast_nonneg n
  have hninv : (0 : ℝ) ≤ 1 / (n : ℝ) := by positivity
  have hG0 : 0 ≤ G := (abs_nonneg _).trans hγ
  have hXh0 : 0 ≤ Xh := (abs_nonneg _).trans (hxh i)
  have hS0 : 0 ≤ S := (abs_nonneg _).trans hSabs
  -- three per-group bounds on the bracket difference
  have htA : |(n : ℝ) * (γ * vt i) - (n : ℝ) * (γ * va i)| ≤ (n : ℝ) * (G * e) := by
    have he : (n : ℝ) * (γ * vt i) - (n : ℝ) * (γ * va i) = (n : ℝ) * (γ * (vt i - va i)) := by ring
    rw [he, abs_mul, abs_of_nonneg hnR]
    refine mul_le_mul_of_nonneg_left ?_ hnR
    rw [abs_mul]; exact mul_le_mul hγ (hd i) (abs_nonneg _) hG0
  have htS : |(∑ k, γ * vt k) - (∑ k, γ * va k)| ≤ (n : ℝ) * (G * e) := by
    calc |(∑ k, γ * vt k) - (∑ k, γ * va k)|
        = |∑ k, (γ * vt k - γ * va k)| := by rw [Finset.sum_sub_distrib]
      _ ≤ ∑ k, |γ * vt k - γ * va k| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _k : Fin n, G * e := Finset.sum_le_sum (fun k _ => by
            have he : γ * vt k - γ * va k = γ * (vt k - va k) := by ring
            rw [he, abs_mul]; exact mul_le_mul hγ (hd k) (abs_nonneg _) hG0)
      _ = (n : ℝ) * (G * e) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have htC : |bnXhat n ε x i * (∑ k, bnXhat n ε x k * (γ * vt k))
              - bnXhat n ε x i * (∑ k, bnXhat n ε x k * (γ * va k))|
            ≤ Xh * ((n : ℝ) * (Xh * (G * e))) := by
    rw [← mul_sub, abs_mul]
    refine mul_le_mul (hxh i) ?_ (abs_nonneg _) hXh0
    calc |(∑ k, bnXhat n ε x k * (γ * vt k)) - (∑ k, bnXhat n ε x k * (γ * va k))|
        = |∑ k, (bnXhat n ε x k * (γ * vt k) - bnXhat n ε x k * (γ * va k))| := by
          rw [Finset.sum_sub_distrib]
      _ ≤ ∑ k, |bnXhat n ε x k * (γ * vt k) - bnXhat n ε x k * (γ * va k)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _k : Fin n, Xh * (G * e) := Finset.sum_le_sum (fun k _ => by
            have he : bnXhat n ε x k * (γ * vt k) - bnXhat n ε x k * (γ * va k)
                = bnXhat n ε x k * (γ * (vt k - va k)) := by ring
            rw [he, abs_mul]
            exact mul_le_mul (hxh k)
              (by rw [abs_mul]; exact mul_le_mul hγ (hd k) (abs_nonneg _) hG0)
              (abs_nonneg _) hXh0)
      _ = (n : ℝ) * (Xh * (G * e)) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  -- the bracket difference bound (the three groups via triangle)
  have hBd : |((n : ℝ) * (γ * vt i) - (∑ k, γ * vt k)
                - bnXhat n ε x i * (∑ k, bnXhat n ε x k * (γ * vt k)))
              - ((n : ℝ) * (γ * va i) - (∑ k, γ * va k)
                - bnXhat n ε x i * (∑ k, bnXhat n ε x k * (γ * va k)))|
            ≤ (n : ℝ) * (G * e) + (n : ℝ) * (G * e) + Xh * ((n : ℝ) * (Xh * (G * e))) := by
    have heq : ((n : ℝ) * (γ * vt i) - (∑ k, γ * vt k)
                - bnXhat n ε x i * (∑ k, bnXhat n ε x k * (γ * vt k)))
              - ((n : ℝ) * (γ * va i) - (∑ k, γ * va k)
                - bnXhat n ε x i * (∑ k, bnXhat n ε x k * (γ * va k)))
            = ((n : ℝ) * (γ * vt i) - (n : ℝ) * (γ * va i))
              - ((∑ k, γ * vt k) - (∑ k, γ * va k))
              - (bnXhat n ε x i * (∑ k, bnXhat n ε x k * (γ * vt k))
                  - bnXhat n ε x i * (∑ k, bnXhat n ε x k * (γ * va k))) := by ring
    rw [heq]
    exact (abs_sub _ _).trans (add_le_add ((abs_sub _ _).trans (add_le_add htA htS)) htC)
  -- scale bound and assemble
  have hc : |1 / (n : ℝ) * bnIstd n x ε| ≤ 1 / (n : ℝ) * S := by
    rw [abs_mul, abs_of_nonneg hninv]
    exact mul_le_mul_of_nonneg_left hSabs hninv
  rw [bn_grad_input_eq, bn_grad_input_eq, ← mul_sub, abs_mul]
  exact mul_le_mul hc hBd (abs_nonneg _) (mul_nonneg hninv hS0)

/-- **Real BN input-gradient magnitude** — `|bn_grad_input … dy i| ≤ ReMag(Cdy)` for `dy` within
    `Cdy`. The Lipschitz bound at `va = 0` (`bn_grad_input … 0 = 0`). -/
theorem bn_grad_input_abs_le {n : Nat} {ε γ : ℝ} (x dy : Vec n) {G Cdy S Xh : ℝ} (hn : 0 < n)
    (hγ : |γ| ≤ G) (hdy : ∀ k, |dy k| ≤ Cdy)
    (hSabs : |bnIstd n x ε| ≤ S) (hxh : ∀ i, |bnXhat n ε x i| ≤ Xh) (i : Fin n) :
    |bn_grad_input n ε γ x dy i| ≤ bnGradInputReMag n G Cdy S Xh := by
  have h := bn_grad_input_diff_abs_le (ε := ε) (γ := γ) x dy (0 : Vec n)
    (G := G) (S := S) (Xh := Xh) (e := Cdy) hn hγ (fun k => by simpa using hdy k) hSabs hxh i
  rwa [bn_grad_input_zero, sub_zero] at h

-- ════════════════════════════════════════════════════════════════
-- § BN backward as a `FloatClose` map over the cotangent
-- ════════════════════════════════════════════════════════════════

/-- **BN backward is `FloatClose`** (input-gradient, over the cotangent `dy`). The float
    three-term input-gradient `bnGradInputF` (at a supplied float inverse-stddev `fs` and
    normalized `fxh`, close within `es`/`exh`) is within `budget(A) + ReMag(e)` of the certified
    `bn_grad_input` at cotangent error `e`, both outputs bounded by `ReMag(A) + budget(A)`. The
    backward peer of `floatClose_bn`; folds into the whole-net backward via `.comp`. -/
theorem floatClose_bnBack {n : Nat} (M : FloatModel) {ε γ : ℝ} (x fxh : Vec n) (fs : ℝ)
    {G S Xh es exh A : ℝ} (hn : 0 < n)
    (hγ : |γ| ≤ G)
    (hs : |fs - bnIstd n x ε| ≤ es) (hSabs : |bnIstd n x ε| ≤ S)
    (hxh : ∀ i, |bnXhat n ε x i| ≤ Xh) (hfxh : ∀ i, |fxh i - bnXhat n ε x i| ≤ exh) :
    FloatClose A
      (bnGradInputReMag n G A S Xh + M.bnGradInputBudget n G A S Xh es exh)
      (fun dy => bn_grad_input n ε γ x dy)
      (fun dy i => M.bnGradInputF γ fs fxh dy i)
      (fun e => M.bnGradInputBudget n G A S Xh es exh + bnGradInputReMag n G e S Xh) := by
  refine ⟨fun v hv i => ?_, fun vt va e _ hvt hd i => ?_⟩
  · -- magnitude clause (real and float on the same cotangent `v`)
    have hAge : 0 ≤ A := (abs_nonneg _).trans (hv ⟨0, hn⟩)
    have hrealmag : |bn_grad_input n ε γ x v i| ≤ bnGradInputReMag n G A S Xh :=
      bn_grad_input_abs_le x v hn hγ hv hSabs hxh i
    have hfc : |M.bnGradInputF γ fs fxh v i - bn_grad_input n ε γ x v i|
        ≤ M.bnGradInputBudget n G A S Xh es exh :=
      bnGradInput_close M x v fxh fs hn hγ hv hs hSabs hxh hfxh i
    have hbudge : 0 ≤ M.bnGradInputBudget n G A S Xh es exh := by
      have h0 := bnGradInput_close M x (0 : Vec n) fxh fs hn hγ
        (fun k => by simpa using hAge) hs hSabs hxh hfxh i
      exact (abs_nonneg _).trans h0
    refine ⟨hrealmag.trans (le_add_of_nonneg_right hbudge), ?_⟩
    calc |M.bnGradInputF γ fs fxh v i|
        ≤ |M.bnGradInputF γ fs fxh v i - bn_grad_input n ε γ x v i|
          + |bn_grad_input n ε γ x v i| := by
          simpa using abs_sub_le (M.bnGradInputF γ fs fxh v i) (bn_grad_input n ε γ x v i) 0
      _ ≤ M.bnGradInputBudget n G A S Xh es exh + bnGradInputReMag n G A S Xh :=
          add_le_add hfc hrealmag
      _ = bnGradInputReMag n G A S Xh + M.bnGradInputBudget n G A S Xh es exh := by ring
  · -- error clause: float roundoff at `vt` + real Lipschitz response to `dy`-shift `e`
    have hfc : |M.bnGradInputF γ fs fxh vt i - bn_grad_input n ε γ x vt i|
        ≤ M.bnGradInputBudget n G A S Xh es exh :=
      bnGradInput_close M x vt fxh fs hn hγ hvt hs hSabs hxh hfxh i
    have hreal : |bn_grad_input n ε γ x vt i - bn_grad_input n ε γ x va i|
        ≤ bnGradInputReMag n G e S Xh :=
      bn_grad_input_diff_abs_le x vt va hn hγ hd hSabs hxh i
    calc |M.bnGradInputF γ fs fxh vt i - bn_grad_input n ε γ x va i|
        ≤ |M.bnGradInputF γ fs fxh vt i - bn_grad_input n ε γ x vt i|
          + |bn_grad_input n ε γ x vt i - bn_grad_input n ε γ x va i| := abs_sub_le _ _ _
      _ ≤ M.bnGradInputBudget n G A S Xh es exh + bnGradInputReMag n G e S Xh :=
          add_le_add hfc hreal

/-- **BN backward float-bridges** (over the cotangent). The shared BN/LN backward op in
    `FloatBridges` form; `.comp`-chains into a whole-net backward. Output magnitude nonnegativity
    is read off the certificate via `FloatClose.cod_nonneg`. -/
theorem floatBridges_bnBack {n : Nat} (M : FloatModel) {ε γ : ℝ} (x fxh : Vec n) (fs : ℝ)
    {G S Xh es exh : ℝ} (hn : 0 < n)
    (hγ : |γ| ≤ G)
    (hs : |fs - bnIstd n x ε| ≤ es) (hSabs : |bnIstd n x ε| ≤ S)
    (hxh : ∀ i, |bnXhat n ε x i| ≤ Xh) (hfxh : ∀ i, |fxh i - bnXhat n ε x i| ≤ exh) :
    FloatBridges (fun dy => bn_grad_input n ε γ x dy) := by
  intro A hA
  have hfc := floatClose_bnBack M x fxh fs hn hγ hs hSabs hxh hfxh (A := A)
  exact ⟨_, _, _, hfc.cod_nonneg hA hn, hfc⟩

end Proofs
