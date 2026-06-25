import LeanMlir.Proofs.SgdDescentMlp
import LeanMlir.Proofs.ConvLossFold
import LeanMlir.Proofs.MnistCNN

/-! # Lipschitz constants for the CNN softmax-CE loss — descent through the pool

`SgdDescentMlp` discharged `sgd_descends`' smoothness hypothesis for every
MLP weight layer; this file extends the program to the Chapter-4 MNIST CNN
(`conv → relu → conv → relu → maxpool → dense → relu → dense → relu →
dense`). What's genuinely new versus the MLP:

* **The dense head is free.** Below the pool the CNN *is* an MLP at the
  pooled activation: the loss-of-`W₅`/`W₄`/`W₃` maps are literal instances
  of `linear_sgd_descends` / `mlp_hidden_sgd_descends` /
  `mlp_input_sgd_descends` at `x := maxPoolFlat (…)`. No new theorems are
  needed (the MLP statements are generic in the fixed activation vector).

* **The max-pool needs a quantitative SELECTION margin.** `MaxPool2Smooth`
  (pairwise-distinct window cells) is the qualitative off-the-kink
  condition; descent needs its quantitative form `MaxPool2MarginQ δ`
  (pairwise window gaps exceed `2δ`): a perturbation of at most `δ` per
  entry then cannot reorder any window, so the argmax — hence the
  pool's routing pattern — FREEZES along the step segment
  (`MaxPool2MarginQ.isArgmax_iff`), exactly as the ReLU margins freeze the
  masks. The pool is also 1-Lipschitz per entry
  (`maxPoolFlat_entry_lipschitz`) and `ℓ1`-contractive across entries
  (`maxPoolFlat_l1_contract` — the 2×2 stride-2 windows partition the
  input), so drift passes through it unamplified.

* **Conv layers are dense layers with weight sharing.** The conv output is
  affine in the kernel; each output entry reads one kernel slab against
  bounded input values (`flatConv_kernel_drift`), and the `ℓ1` drift picks
  up the spatial multiplicity `h·w` — each kernel entry touches every
  spatial position (`flatConv_kernel_drift_sum`).

The capstone `cnn_conv2_sgd_descends` mirrors `mlp_input_sgd_descends`:
under the four margins (relu₂, pool selection, relu₃, relu₄) at the step
radius and the small-step condition, one inexact SGD step on the second
conv kernel provably decreases the cross-entropy loss by ≥ `lr·‖∇L‖₂²/2`,
with the segment-Lipschitz constant explicit.

`cnn_conv1_sgd_descends` extends the program one layer deeper: the step
now crosses conv2 AS A FUNCTION OF ITS INPUT. Conv is linear there, its
Jacobian entry a single kernel tap (`convTap`, extracted point-free from
the certified input-VJP), and its `ℓ1` operator factor is LOCALITY —
`(channels)·kH·kW·w₂`, not a spatial count. Under FIVE margins (relu₁ +
the conv2 four, at conv1 radii) every routing decision freezes and the
loss provably drops.

`cnn_conv2_bias_sgd_descends` / `cnn_conv1_bias_sgd_descends` close the
biases: the bias-map Jacobian is a Kronecker channel indicator
(`conv2d_bias_pdiv`, extracted from the certified bias VJP), the
per-entry drift is exactly `|e o|` (no input bound `a`), and the rungs
are the kernel arguments verbatim with the conv stage's `a·D` radii
replaced by the bare `D` and `a² ↦ 1` in the constants. EVERY parameter
of the Chapter-4 CNN — both conv kernels, both conv biases, and the
dense head — now has a proven descent statement. -/

namespace Proofs

open StableHLO Classical

-- ════════════════════════════════════════════════════════════════
-- § Max is Lipschitz: the 2×2 window max moves no more than its cells
-- ════════════════════════════════════════════════════════════════

/-- A four-way max moves by at most the largest cell movement (`ℓ∞`). -/
theorem max4_sub_abs_le {a b c d a' b' c' d' δ : ℝ}
    (h1 : |a - a'| ≤ δ) (h2 : |b - b'| ≤ δ)
    (h3 : |c - c'| ≤ δ) (h4 : |d - d'| ≤ δ) :
    |max (max a b) (max c d) - max (max a' b') (max c' d')| ≤ δ := by
  have hab : |max a b - max a' b'| ≤ δ :=
    le_trans (abs_max_sub_max_le_max a b a' b') (max_le h1 h2)
  have hcd : |max c d - max c' d'| ≤ δ :=
    le_trans (abs_max_sub_max_le_max c d c' d') (max_le h3 h4)
  exact le_trans (abs_max_sub_max_le_max _ _ _ _) (max_le hab hcd)

/-- A four-way max moves by at most the *sum* of the cell movements
    (`ℓ1`) — the per-window step of the pool's `ℓ1` contraction. -/
theorem max4_sub_abs_le_sum {a b c d a' b' c' d' : ℝ} :
    |max (max a b) (max c d) - max (max a' b') (max c' d')| ≤
      |a - a'| + |b - b'| + |c - c'| + |d - d'| := by
  refine max4_sub_abs_le (δ := |a - a'| + |b - b'| + |c - c'| + |d - d'|)
    ?_ ?_ ?_ ?_ <;>
    nlinarith [abs_nonneg (a - a'), abs_nonneg (b - b'),
      abs_nonneg (c - c'), abs_nonneg (d - d')]

-- ════════════════════════════════════════════════════════════════
-- § Index plumbing: window cells tile the input, flat sums = tensor sums
-- ════════════════════════════════════════════════════════════════

/-- Flat index of a `Tensor3` coordinate (the suite's row-major layout). -/
def t3Idx {c h w : Nat} (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    Fin (c * h * w) :=
  finProdFinEquiv (finProdFinEquiv (ci, hi), wi)

/-- `t3Idx` reads back through `Tensor3.unflatten`. -/
theorem unflatten_t3Idx {c h w : Nat} (v : Vec (c * h * w))
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    Tensor3.unflatten v ci hi wi = v (t3Idx ci hi wi) := rfl

/-- `Tensor3.flatten` reads off at a `t3Idx`. -/
theorem flatten_t3Idx {c h w : Nat} (T : Tensor3 c h w)
    (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    Tensor3.flatten T (t3Idx ci hi wi) = T ci hi wi := by
  unfold Tensor3.flatten t3Idx
  simp

/-- A flat sum is the triple tensor sum. -/
theorem sum_t3 {c h w : Nat} (f : Fin (c * h * w) → ℝ) :
    ∑ k, f k = ∑ ci : Fin c, ∑ hi : Fin h, ∑ wi : Fin w,
      f (t3Idx ci hi wi) := by
  calc ∑ k, f k
      = ∑ p : Fin (c * h) × Fin w, f (finProdFinEquiv p) :=
        (Equiv.sum_comp finProdFinEquiv f).symm
    _ = ∑ q : Fin (c * h), ∑ wi : Fin w, f (finProdFinEquiv (q, wi)) :=
        Fintype.sum_prod_type _
    _ = ∑ p : Fin c × Fin h, ∑ wi : Fin w,
          f (finProdFinEquiv (finProdFinEquiv p, wi)) :=
        (Equiv.sum_comp finProdFinEquiv
          (fun q => ∑ wi : Fin w, f (finProdFinEquiv (q, wi)))).symm
    _ = ∑ ci : Fin c, ∑ hi : Fin h, ∑ wi : Fin w,
          f (t3Idx ci hi wi) := Fintype.sum_prod_type _

/-- Every flat spatial index is a `t3Idx` — lets a per-cell bound be lifted to
    the whole flattened conv-output vector (`∀ k`), the form `relu_close` /
    `maxPoolFlat_close` / `dense_close` consume. -/
theorem t3Idx_surj {c h w : Nat} (k : Fin (c * h * w)) :
    ∃ (ci : Fin c) (hi : Fin h) (wi : Fin w), k = t3Idx ci hi wi := by
  refine ⟨(finProdFinEquiv.symm (finProdFinEquiv.symm k).1).1,
    (finProdFinEquiv.symm (finProdFinEquiv.symm k).1).2,
    (finProdFinEquiv.symm k).2, ?_⟩
  simp only [t3Idx, Prod.mk.eta, Equiv.apply_symm_apply]

/-- The window-cell parameterization `(out-row, sub-row) ↦ in-row` is a
    bijection — pooled windows tile the rows. -/
def winRowEquiv (h : Nat) : Fin h × Fin 2 ≃ Fin (2 * h) where
  toFun p := winRowInv p.1 p.2
  invFun hi := (winRow hi, winRowMod hi)
  left_inv p := by
    ext
    · exact congrArg Fin.val (winRow_winRowInv p.1 p.2)
    · exact congrArg Fin.val (winRowMod_winRowInv p.1 p.2)
  right_inv hi := winRowInv_winRow hi

/-- Column version of `winRowEquiv`. -/
def winColEquiv (w : Nat) : Fin w × Fin 2 ≃ Fin (2 * w) where
  toFun p := winColInv p.1 p.2
  invFun wi := (winCol wi, winColMod wi)
  left_inv p := by
    ext
    · exact congrArg Fin.val (winCol_winColInv p.1 p.2)
    · exact congrArg Fin.val (winColMod_winColInv p.1 p.2)
  right_inv wi := winColInv_winCol wi

/-- Summing a function over all window cells of all windows is summing it
    over the whole spatial grid — the 2×2 stride-2 windows partition the
    input. -/
theorem sum_window_cells {h w : Nat} (g : Fin (2 * h) → Fin (2 * w) → ℝ) :
    ∑ ho : Fin h, ∑ wo : Fin w, ∑ ab : Fin 2 × Fin 2,
        g (winRowInv ho ab.1) (winColInv wo ab.2) =
      ∑ hi : Fin (2 * h), ∑ wi : Fin (2 * w), g hi wi := by
  have hcol : ∀ g' : Fin (2 * w) → ℝ,
      ∑ wo : Fin w, ∑ b : Fin 2, g' (winColInv wo b) = ∑ wi, g' wi := by
    intro g'
    calc ∑ wo : Fin w, ∑ b : Fin 2, g' (winColInv wo b)
        = ∑ q : Fin w × Fin 2, g' (winColInv q.1 q.2) :=
          (Fintype.sum_prod_type
            (fun q : Fin w × Fin 2 => g' (winColInv q.1 q.2))).symm
      _ = ∑ wi, g' wi := Equiv.sum_comp (winColEquiv w) g'
  have hrow : ∀ g' : Fin (2 * h) → ℝ,
      ∑ ho : Fin h, ∑ a : Fin 2, g' (winRowInv ho a) = ∑ hi, g' hi := by
    intro g'
    calc ∑ ho : Fin h, ∑ a : Fin 2, g' (winRowInv ho a)
        = ∑ p : Fin h × Fin 2, g' (winRowInv p.1 p.2) :=
          (Fintype.sum_prod_type
            (fun p : Fin h × Fin 2 => g' (winRowInv p.1 p.2))).symm
      _ = ∑ hi, g' hi := Equiv.sum_comp (winRowEquiv h) g'
  calc ∑ ho : Fin h, ∑ wo : Fin w, ∑ ab : Fin 2 × Fin 2,
        g (winRowInv ho ab.1) (winColInv wo ab.2)
      = ∑ ho : Fin h, ∑ a : Fin 2, ∑ wo : Fin w, ∑ b : Fin 2,
          g (winRowInv ho a) (winColInv wo b) := by
        refine Finset.sum_congr rfl fun ho _ => ?_
        calc ∑ wo : Fin w, ∑ ab : Fin 2 × Fin 2,
              g (winRowInv ho ab.1) (winColInv wo ab.2)
            = ∑ wo : Fin w, ∑ a : Fin 2, ∑ b : Fin 2,
                g (winRowInv ho a) (winColInv wo b) :=
              Finset.sum_congr rfl fun wo _ => Fintype.sum_prod_type _
          _ = ∑ a : Fin 2, ∑ wo : Fin w, ∑ b : Fin 2,
                g (winRowInv ho a) (winColInv wo b) := Finset.sum_comm
    _ = ∑ ho : Fin h, ∑ a : Fin 2, ∑ wi : Fin (2 * w),
          g (winRowInv ho a) wi := by
        refine Finset.sum_congr rfl fun ho _ => ?_
        exact Finset.sum_congr rfl fun a _ =>
          hcol (fun wi => g (winRowInv ho a) wi)
    _ = ∑ hi : Fin (2 * h), ∑ wi : Fin (2 * w), g hi wi :=
        hrow (fun hi => ∑ wi : Fin (2 * w), g hi wi)

-- ════════════════════════════════════════════════════════════════
-- § The pool is 1-Lipschitz per entry and ℓ1-contractive across entries
-- ════════════════════════════════════════════════════════════════

/-- The pooled entry at `(ci, ho, wo)` is the four-way max of its window
    cells, in flat coordinates. -/
theorem maxPoolFlat_apply {c h w : Nat} (u : Vec (c * (2*h) * (2*w)))
    (ci : Fin c) (ho : Fin h) (wo : Fin w) :
    maxPoolFlat c h w u (t3Idx ci ho wo) =
      max (max (u (t3Idx ci (winRowInv ho 0) (winColInv wo 0)))
               (u (t3Idx ci (winRowInv ho 1) (winColInv wo 0))))
          (max (u (t3Idx ci (winRowInv ho 0) (winColInv wo 1)))
               (u (t3Idx ci (winRowInv ho 1) (winColInv wo 1)))) := by
  show Tensor3.flatten (maxPool2 (Tensor3.unflatten u)) (t3Idx ci ho wo) = _
  rw [flatten_t3Idx, winRowInv_zero, winRowInv_one, winColInv_zero,
    winColInv_one]
  rfl

/-- Per-entry pool Lipschitz bound: if every input entry moves by at most
    `δ`, every pooled entry moves by at most `δ`. -/
theorem maxPoolFlat_entry_lipschitz {c h w : Nat}
    (u v : Vec (c * (2*h) * (2*w))) {δ : ℝ}
    (hδ : ∀ k, |u k - v k| ≤ δ) (q : Fin (c * h * w)) :
    |maxPoolFlat c h w u q - maxPoolFlat c h w v q| ≤ δ := by
  obtain ⟨p, rfl⟩ := finProdFinEquiv.surjective q
  obtain ⟨pp, wo⟩ := p
  obtain ⟨r, rfl⟩ := finProdFinEquiv.surjective pp
  obtain ⟨ci, ho⟩ := r
  rw [show finProdFinEquiv (finProdFinEquiv (ci, ho), wo) =
        t3Idx ci ho wo from rfl,
    maxPoolFlat_apply, maxPoolFlat_apply]
  exact max4_sub_abs_le (hδ _) (hδ _) (hδ _) (hδ _)

/-- `ℓ1` contraction: the pooled drift, summed over all pooled entries, is
    at most the input drift summed over all input entries (windows are
    disjoint, max is 1-Lipschitz). The pool passes `ℓ1` budgets through
    unamplified. -/
theorem maxPoolFlat_l1_contract {c h w : Nat}
    (u v : Vec (c * (2*h) * (2*w))) :
    ∑ q, |maxPoolFlat c h w u q - maxPoolFlat c h w v q| ≤
      ∑ k, |u k - v k| := by
  rw [sum_t3 (fun q => |maxPoolFlat c h w u q - maxPoolFlat c h w v q|),
    sum_t3 (fun k => |u k - v k|)]
  refine Finset.sum_le_sum fun ci _ => ?_
  calc ∑ ho : Fin h, ∑ wo : Fin w,
        |maxPoolFlat c h w u (t3Idx ci ho wo) -
          maxPoolFlat c h w v (t3Idx ci ho wo)|
      ≤ ∑ ho : Fin h, ∑ wo : Fin w, ∑ ab : Fin 2 × Fin 2,
          |u (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2)) -
            v (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2))| := by
        refine Finset.sum_le_sum fun ho _ => Finset.sum_le_sum fun wo _ => ?_
        have hexp : ∑ ab : Fin 2 × Fin 2,
            |u (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2)) -
              v (t3Idx ci (winRowInv ho ab.1) (winColInv wo ab.2))| =
            ∑ a : Fin 2, ∑ b : Fin 2,
              |u (t3Idx ci (winRowInv ho a) (winColInv wo b)) -
                v (t3Idx ci (winRowInv ho a) (winColInv wo b))| :=
          Fintype.sum_prod_type _
        rw [maxPoolFlat_apply, maxPoolFlat_apply, hexp, Fin.sum_univ_two,
          Fin.sum_univ_two, Fin.sum_univ_two]
        refine le_trans max4_sub_abs_le_sum (le_of_eq ?_)
        ring
    _ = ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          |u (t3Idx ci hi wi) - v (t3Idx ci hi wi)| :=
        sum_window_cells (fun hi wi =>
          |u (t3Idx ci hi wi) - v (t3Idx ci hi wi)|)

-- ════════════════════════════════════════════════════════════════
-- § The selection margin: window gaps beyond 2δ freeze the argmax
-- ════════════════════════════════════════════════════════════════

/-- Two values farther apart than `2δ` cannot be equalized by
    `δ`-perturbations. -/
theorem ne_of_gap_of_close {xa xb ya yb δ : ℝ} (hgap : 2 * δ < |xa - xb|)
    (ha : |ya - xa| ≤ δ) (hb : |yb - xb| ≤ δ) : ya ≠ yb := by
  intro heq
  have h1 := abs_le.mp ha
  have h2 := abs_le.mp hb
  have heq' : ya - yb = 0 := by rw [heq]; ring
  have hle : |xa - xb| ≤ 2 * δ :=
    abs_le.mpr ⟨by linarith [h1.1, h1.2, h2.1, h2.2],
      by linarith [h1.1, h1.2, h2.1, h2.2]⟩
  linarith

/-- Strict order survives `δ`-perturbations across a `2δ` gap. -/
theorem lt_of_lt_gap_of_close {xa xb ya yb δ : ℝ}
    (hlt : 2 * δ < xb - xa) (ha : |ya - xa| ≤ δ) (hb : |yb - xb| ≤ δ) :
    ya < yb := by
  have h1 := abs_le.mp ha
  have h2 := abs_le.mp hb
  linarith [h1.1, h1.2, h2.1, h2.2]

/-- **Quantitative pool-selection margin**: every two cells of every 2×2
    window differ by more than `2δ`. The quantitative form of
    `MaxPool2Smooth` — a perturbation of at most `δ` per entry can neither
    create a tie nor reorder a window, so the pool's argmax routing
    freezes. The pool peer of the ReLU margin `a·D < |zⱼ|`. -/
def MaxPool2MarginQ {c h w : Nat} (δ : ℝ)
    (x : Tensor3 c (2*h) (2*w)) : Prop :=
  ∀ (ci : Fin c) (ho : Fin h) (wo : Fin w)
    (ab ab' : Fin 2 × Fin 2), ab ≠ ab' →
    2 * δ < |x ci (winRowInv ho ab.1) (winColInv wo ab.2) -
             x ci (winRowInv ho ab'.1) (winColInv wo ab'.2)|

/-- Every point within `δ` of a margined point is smooth (no window
    ties). -/
theorem MaxPool2MarginQ.smooth_of_close {c h w : Nat} {δ : ℝ}
    {x y : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x)
    (hclose : ∀ ci hi wi, |y ci hi wi - x ci hi wi| ≤ δ) :
    MaxPool2Smooth y := fun ci ho wo ab ab' hne =>
  ne_of_gap_of_close (hm ci ho wo ab ab' hne)
    (hclose ci (winRowInv ho ab.1) (winColInv wo ab.2))
    (hclose ci (winRowInv ho ab'.1) (winColInv wo ab'.2))

/-- A margined point is itself smooth. -/
theorem MaxPool2MarginQ.smooth {c h w : Nat} {δ : ℝ} (hδ0 : 0 ≤ δ)
    {x : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x) :
    MaxPool2Smooth x :=
  hm.smooth_of_close (fun ci hi wi => by simp [hδ0])

/-- **The argmax freezes**: within `δ` of a margined point, every window's
    argmax cell is the same as at the margined point. -/
theorem MaxPool2MarginQ.isArgmax_iff {c h w : Nat} {δ : ℝ}
    {x y : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x)
    (hclose : ∀ ci hi wi, |y ci hi wi - x ci hi wi| ≤ δ)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    MaxPool2IsArgmax y ci hi wi ↔ MaxPool2IsArgmax x ci hi wi := by
  have hxw : x ci (winRowInv (winRow hi) (winRowMod hi))
      (winColInv (winCol wi) (winColMod wi)) = x ci hi wi := by
    rw [winRowInv_winRow, winColInv_winCol]
  have hyw : y ci (winRowInv (winRow hi) (winRowMod hi))
      (winColInv (winCol wi) (winColMod wi)) = y ci hi wi := by
    rw [winRowInv_winRow, winColInv_winCol]
  constructor
  · -- y-argmax at (hi,wi) ⇒ x-argmax at (hi,wi), by contraposition on cells
    intro hy a b
    by_contra hnot
    have hlt : x ci hi wi <
        x ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b) :=
      not_le.mp hnot
    have hne : ((a, b) : Fin 2 × Fin 2) ≠ (winRowMod hi, winColMod wi) := by
      intro hEq
      have ha' : a = winRowMod hi := congrArg Prod.fst hEq
      have hb' : b = winColMod wi := congrArg Prod.snd hEq
      rw [ha', hb', hxw] at hlt
      exact lt_irrefl _ hlt
    have hgap := hm ci (winRow hi) (winCol wi) (a, b)
      (winRowMod hi, winColMod wi) hne
    rw [hxw] at hgap
    have hgap' : 2 * δ <
        x ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b) -
          x ci hi wi := by
      rwa [abs_of_pos (by linarith)] at hgap
    have hylt : y ci hi wi <
        y ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b) :=
      lt_of_lt_gap_of_close hgap' (hclose ci hi wi)
        (hclose ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b))
    exact absurd (hy a b) (not_le.mpr hylt)
  · -- x-argmax at (hi,wi) ⇒ y-argmax at (hi,wi)
    intro hx a b
    by_cases hEq : ((a, b) : Fin 2 × Fin 2) = (winRowMod hi, winColMod wi)
    · have ha' : a = winRowMod hi := congrArg Prod.fst hEq
      have hb' : b = winColMod wi := congrArg Prod.snd hEq
      rw [ha', hb', hyw]
    · have hle := hx a b
      have hgap := hm ci (winRow hi) (winCol wi)
        (winRowMod hi, winColMod wi) (a, b) (Ne.symm hEq)
      rw [hxw] at hgap
      have hgap' : 2 * δ < x ci hi wi -
          x ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b) := by
        rwa [abs_of_nonneg (sub_nonneg.mpr hle)] at hgap
      exact le_of_lt (lt_of_lt_gap_of_close hgap'
        (hclose ci (winRowInv (winRow hi) a) (winColInv (winCol wi) b))
        (hclose ci hi wi))

-- ════════════════════════════════════════════════════════════════
-- § Conv-kernel drift: a dense layer with weight sharing
-- ════════════════════════════════════════════════════════════════

/-- The padded input read that multiplies kernel entry `(·, c, kh, kw)` at
    output position `(hi, wi)` — names the `dite` inside `conv2d` so the
    affine-in-the-kernel structure can be stated. Depends on the input
    only, never the kernel. -/
noncomputable def convPad {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    ℝ :=
  if hpad : (kH - 1) / 2 ≤ kh.val + hi.val ∧
      kh.val + hi.val - (kH - 1) / 2 < h ∧
      (kW - 1) / 2 ≤ kw.val + wi.val ∧
      kw.val + wi.val - (kW - 1) / 2 < w then
    x c ⟨kh.val + hi.val - (kH - 1) / 2, hpad.2.1⟩
        ⟨kw.val + wi.val - (kW - 1) / 2, hpad.2.2.2⟩
  else 0

/-- `conv2d` through `convPad`: bias plus the kernel-linear form. -/
theorem conv2d_eq_convPad {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W b x o hi wi =
      b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * convPad kH kW x c kh kw hi wi := rfl

/-- Padded reads are bounded by the input bound (out-of-bounds reads are
    zero). -/
theorem abs_convPad_le {ic h w kH kW : Nat} (x : Tensor3 ic h w) {a : ℝ}
    (ha : 0 ≤ a) (hx : ∀ c i j, |x c i j| ≤ a)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    |convPad kH kW x c kh kw hi wi| ≤ a := by
  unfold convPad
  split_ifs with h
  · exact hx _ _ _
  · simpa using ha

/-- Flat index of a `Kernel4` entry (the suite's row-major layout). -/
def k4Idx {oc ic kH kW : Nat} (o : Fin oc) (c : Fin ic)
    (kh : Fin kH) (kw : Fin kW) : Fin (oc * ic * kH * kW) :=
  finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o, c), kh), kw)

/-- `k4Idx` reads back through `Kernel4.unflatten`. -/
theorem unflatten_k4Idx {oc ic kH kW : Nat} (v : Vec (oc * ic * kH * kW))
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    Kernel4.unflatten v o c kh kw = v (k4Idx o c kh kw) := rfl

/-- `Kernel4.flatten` reads off at a `k4Idx` — the forward peer of
    `unflatten_k4Idx`, lifting a per-entry kernel bound to the flattened vector. -/
theorem flatten_k4Idx {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    Kernel4.flatten W (k4Idx o c kh kw) = W o c kh kw := by
  simp only [Kernel4.flatten, k4Idx, Equiv.symm_apply_apply]

/-- Every flat kernel index is a `k4Idx` — lets the abstract `∀ idx` gradient
    accuracy be discharged per `(o,cc,kh,kw)` by `cnn_conv2_grad_close`. -/
theorem k4Idx_surj {oc ic kH kW : Nat} (idx : Fin (oc * ic * kH * kW)) :
    ∃ (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW),
      idx = k4Idx o c kh kw := by
  refine ⟨(finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1).1,
    (finProdFinEquiv.symm
      (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).1).2,
    (finProdFinEquiv.symm (finProdFinEquiv.symm idx).1).2,
    (finProdFinEquiv.symm idx).2, ?_⟩
  simp only [k4Idx, Prod.mk.eta, Equiv.apply_symm_apply]

theorem k4Idx_inj {oc ic kH kW : Nat} {o : Fin oc} {c c' : Fin ic}
    {kh kh' : Fin kH} {kw kw' : Fin kW}
    (hEq : k4Idx o c kh kw = k4Idx o c' kh' kw') :
    c = c' ∧ kh = kh' ∧ kw = kw' := by
  unfold k4Idx at hEq
  have h1 := finProdFinEquiv.injective hEq
  have hkw : kw = kw' := (Prod.ext_iff.mp h1).2
  have h2 := finProdFinEquiv.injective (Prod.ext_iff.mp h1).1
  have hkh : kh = kh' := (Prod.ext_iff.mp h2).2
  have h3 := finProdFinEquiv.injective (Prod.ext_iff.mp h2).1
  exact ⟨(Prod.ext_iff.mp h3).2, hkh, hkw⟩

/-- The `ℓ1` mass of one output-channel slab is at most the total `ℓ1`
    mass — the conv analogue of a dense column being part of the flat
    parameter vector. -/
theorem sum_abs_kernel_slab_le {oc ic kH kW : Nat}
    (e : Vec (oc * ic * kH * kW)) (o : Fin oc) :
    ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW, |e (k4Idx o c kh kw)| ≤
      ∑ idx, |e idx| := by
  have hcollapse : ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
      |e (k4Idx o c kh kw)| =
      ∑ p : (Fin ic × Fin kH) × Fin kW, |e (k4Idx o p.1.1 p.1.2 p.2)| := by
    calc ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW, |e (k4Idx o c kh kw)|
        = ∑ q : Fin ic × Fin kH, ∑ kw : Fin kW, |e (k4Idx o q.1 q.2 kw)| :=
          (Fintype.sum_prod_type (fun q : Fin ic × Fin kH =>
            ∑ kw : Fin kW, |e (k4Idx o q.1 q.2 kw)|)).symm
      _ = ∑ p : (Fin ic × Fin kH) × Fin kW,
            |e (k4Idx o p.1.1 p.1.2 p.2)| :=
          (Fintype.sum_prod_type (fun p : (Fin ic × Fin kH) × Fin kW =>
            |e (k4Idx o p.1.1 p.1.2 p.2)|)).symm
  rw [hcollapse]
  have himg : ∑ idx ∈ Finset.univ.image
      (fun p : (Fin ic × Fin kH) × Fin kW => k4Idx o p.1.1 p.1.2 p.2),
      |e idx| =
      ∑ p : (Fin ic × Fin kH) × Fin kW, |e (k4Idx o p.1.1 p.1.2 p.2)| :=
    Finset.sum_image fun p _ p' _ hpq => by
      obtain ⟨h1, h2, h3⟩ := k4Idx_inj hpq
      exact Prod.ext (Prod.ext h1 h2) h3
  rw [← himg]
  exact Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ _)
    (fun idx _ _ => abs_nonneg _)

/-- The slabs tile the kernel: summing the slab masses over the output
    channels recovers the total `ℓ1` mass. -/
theorem sum_abs_k4 {oc ic kH kW : Nat} (e : Vec (oc * ic * kH * kW)) :
    ∑ idx, |e idx| =
      ∑ o : Fin oc, ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        |e (k4Idx o c kh kw)| := by
  calc ∑ idx, |e idx|
      = ∑ p : Fin (oc * ic * kH) × Fin kW, |e (finProdFinEquiv p)| :=
        (Equiv.sum_comp finProdFinEquiv (fun idx => |e idx|)).symm
    _ = ∑ q : Fin (oc * ic * kH), ∑ kw : Fin kW,
          |e (finProdFinEquiv (q, kw))| := Fintype.sum_prod_type _
    _ = ∑ p : Fin (oc * ic) × Fin kH, ∑ kw : Fin kW,
          |e (finProdFinEquiv (finProdFinEquiv p, kw))| :=
        (Equiv.sum_comp finProdFinEquiv (fun q => ∑ kw : Fin kW,
          |e (finProdFinEquiv (q, kw))|)).symm
    _ = ∑ q : Fin (oc * ic), ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (finProdFinEquiv (finProdFinEquiv (q, kh), kw))| :=
        Fintype.sum_prod_type _
    _ = ∑ p : Fin oc × Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx p.1 p.2 kh kw)| :=
        (Equiv.sum_comp finProdFinEquiv (fun q => ∑ kh : Fin kH,
          ∑ kw : Fin kW,
            |e (finProdFinEquiv (finProdFinEquiv (q, kh), kw))|)).symm
    _ = ∑ o : Fin oc, ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx o c kh kw)| := Fintype.sum_prod_type _

-- ════════════════════════════════════════════════════════════════
-- § Conv forward rounding budget (planning §1b-A): conv = dense at the
--   conv fan-in, so the float conv close IS `dense_close` on the
--   per-output-coordinate flattened window.
-- ════════════════════════════════════════════════════════════════

/-- Flat index of a conv *window* slot `(c, kh, kw)` — `k4Idx` without the
    output channel (row-major, fan-in `ic·kH·kW`). -/
def w3Idx {ic kH kW : Nat} (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    Fin (ic * kH * kW) :=
  finProdFinEquiv (finProdFinEquiv (c, kh), kw)

/-- The triple conv-window sum collapses to one flat sum over the fan-in —
    the conv analogue of `dot` being a single-index sum (mirrors `sum_abs_k4`,
    one fewer axis). -/
theorem sum_w3 {ic kH kW : Nat} (g : Fin (ic * kH * kW) → ℝ) :
    ∑ idx, g idx =
      ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW, g (w3Idx c kh kw) := by
  calc ∑ idx, g idx
      = ∑ p : Fin (ic * kH) × Fin kW, g (finProdFinEquiv p) :=
        (Equiv.sum_comp finProdFinEquiv g).symm
    _ = ∑ q : Fin (ic * kH), ∑ kw : Fin kW, g (finProdFinEquiv (q, kw)) :=
        Fintype.sum_prod_type _
    _ = ∑ p : Fin ic × Fin kH, ∑ kw : Fin kW,
          g (finProdFinEquiv (finProdFinEquiv p, kw)) :=
        (Equiv.sum_comp finProdFinEquiv (fun q => ∑ kw : Fin kW,
          g (finProdFinEquiv (q, kw)))).symm
    _ = ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW, g (w3Idx c kh kw) :=
        Fintype.sum_prod_type _

/-- The per-output-coordinate conv *window* as a flat `Vec` over the fan-in:
    the (padded) input reads that the kernel slab dots against. -/
noncomputable def convWindow {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w)
    (hi : Fin h) (wi : Fin w) : Vec (ic * kH * kW) :=
  fun idx =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    convPad kH kW x q.1 q.2 p.2 hi wi

/-- The kernel as a `Mat (ic·kH·kW) oc` — column `o` is the flattened slab. -/
noncomputable def convKernelMat {oc ic kH kW : Nat}
    (W : Kernel4 oc ic kH kW) : Mat (ic * kH * kW) oc :=
  fun idx o =>
    let p := finProdFinEquiv.symm idx
    let q := finProdFinEquiv.symm p.1
    W o q.1 q.2 p.2

@[simp] theorem convWindow_w3 {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w)
    (hi : Fin h) (wi : Fin w) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    convWindow kH kW x hi wi (w3Idx c kh kw) = convPad kH kW x c kh kw hi wi := by
  simp [convWindow, w3Idx, Equiv.symm_apply_apply]

@[simp] theorem convKernelMat_w3 {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    convKernelMat W (w3Idx c kh kw) o = W o c kh kw := by
  simp [convKernelMat, w3Idx, Equiv.symm_apply_apply]

/-- **conv2d is a dense layer at the conv fan-in** — `conv = dense-with-sharing`
    made exact: each output coordinate is `Proofs.dense` of the kernel slab
    against the flattened window. The structural fact that lets the float conv
    budget reuse `dense_close`. -/
theorem conv2d_eq_dense {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W b x o hi wi =
      Proofs.dense (convKernelMat W) b (convWindow kH kW x hi wi) o := by
  rw [conv2d_eq_convPad]
  show b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
      W o c kh kw * convPad kH kW x c kh kw hi wi
    = (∑ idx, convWindow kH kW x hi wi idx * convKernelMat W idx o) + b o
  rw [sum_w3 (fun idx => convWindow kH kW x hi wi idx * convKernelMat W idx o),
      add_comm]
  refine congrArg (· + b o) ?_
  refine Finset.sum_congr rfl fun c _ => Finset.sum_congr rfl fun kh _ =>
    Finset.sum_congr rfl fun kw _ => ?_
  rw [convWindow_w3, convKernelMat_w3]; ring

/-- Padded reads of inputs within `e` stay within `e` (the read is either a
    coordinate, diff `≤ e`, or `0`, diff `0`). -/
theorem convPad_close {ic h w kH kW : Nat} (xt xa : Tensor3 ic h w) {e : ℝ}
    (he : 0 ≤ e) (hx : ∀ c i j, |xt c i j - xa c i j| ≤ e)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    |convPad kH kW xt c kh kw hi wi - convPad kH kW xa c kh kw hi wi| ≤ e := by
  unfold convPad
  split_ifs with h
  · exact hx _ _ _
  · simpa using he

/-- **The float conv layer** — `M.dense` of the kernel slab against the
    flattened window, per output coordinate. The float peer of `conv2d`
    (every product/accumulate/bias-add rounded), in the dense form. -/
noncomputable def FloatModel.convF {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w) :
    Tensor3 oc h w :=
  fun o hi wi => M.dense (convKernelMat W) b (convWindow kH kW x hi wi) o

/-- **Conv forward rounding budget (Item A).** The rounded conv at a float input
    within `e` of the real activation is within the conv-fan-in `denseErr` of the
    real conv — `dense_close` at the flattened window. The compounded Higham
    factor rides the fan-in `ic·kH·kW` (the dense column length here), exactly as
    the planning doc calls for. -/
theorem FloatModel.convF_close {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (xt xa : Tensor3 ic h w) {e : ℝ}
    (he : 0 ≤ e) (hx : ∀ c i j, |xt c i j - xa c i j| ≤ e)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |M.convF W b xt o hi wi - conv2d W b xa o hi wi| ≤
      M.denseErr (convKernelMat W) b (convWindow kH kW xa hi wi) e o := by
  rw [conv2d_eq_dense, FloatModel.convF]
  refine M.dense_close (convKernelMat W) b (convWindow kH kW xt hi wi)
    (convWindow kH kW xa hi wi) e he ?_ o
  intro idx
  simp only [convWindow]
  exact convPad_close xt xa he hx _ _ _ hi wi

/-- Kernel-slab entries inherit the uniform kernel magnitude bound. -/
theorem convKernelMat_abs_le {oc ic kH kW : Nat} {W : Kernel4 oc ic kH kW}
    {w' : ℝ} (hW : ∀ o c kh kw, |W o c kh kw| ≤ w')
    (i : Fin (ic * kH * kW)) (j : Fin oc) : |convKernelMat W i j| ≤ w' := by
  simp only [convKernelMat]; exact hW _ _ _ _

/-- Window reads inherit the uniform input magnitude bound (padding reads 0). -/
theorem convWindow_abs_le {ic h w kH kW : Nat} {x : Tensor3 ic h w} {a : ℝ}
    (ha : 0 ≤ a) (hx : ∀ c i j, |x c i j| ≤ a) (hi : Fin h) (wi : Fin w)
    (idx : Fin (ic * kH * kW)) : |convWindow kH kW x hi wi idx| ≤ a := by
  simp only [convWindow]; exact abs_convPad_le x ha hx _ _ _ hi wi

/-- **Conv output magnitude bound** = `dense_abs_le` at the fan-in: conv is a
    dense layer, so `|conv2dⱼ| ≤ layerAct (ic·kH·kW) w β a`. -/
theorem conv2d_abs_le {ic oc h w kH kW : Nat} {W : Kernel4 oc ic kH kW}
    {b : Vec oc} {x : Tensor3 ic h w} {w' β a : ℝ} (ha : 0 ≤ a)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    (hx : ∀ c i j, |x c i j| ≤ a) (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |conv2d W b x o hi wi| ≤ FloatModel.layerAct (ic * kH * kW) w' β a := by
  rw [conv2d_eq_dense]
  exact FloatModel.dense_abs_le ha (fun i j => convKernelMat_abs_le hW i j) hb
    (fun idx => convWindow_abs_le ha hx hi wi idx) o

-- ════════════════════════════════════════════════════════════════
-- § Vec-space float conv: the form the MNIST-CNN forward composes
-- ════════════════════════════════════════════════════════════════

/-- **Vec-space float conv** — the float peer of `flatConv`
    (`flatten ∘ conv2d ∘ unflatten`), with the rounded `convF` inside. -/
noncomputable def FloatModel.flatConvF {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    Vec (ic * h * w) → Vec (oc * h * w) :=
  fun v => Tensor3.flatten (M.convF W b (Tensor3.unflatten v))

/-- **Vec-space conv forward budget, uniform.** The rounded `flatConvF` at a
    float input within `e` of the real activation is within the conv-fan-in
    `layerBudget` of the real `flatConv` — every output coordinate, one closed
    form. The conv layer threads exactly like a dense layer at fan-in
    `ic·kH·kW`. -/
theorem FloatModel.flatConvF_close {ic oc h w kH kW : Nat} (M : FloatModel)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (vt va : Vec (ic * h * w))
    {w' β a e : ℝ} (hw' : 0 ≤ w') (ha : 0 ≤ a) (he : 0 ≤ e)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    (hva : ∀ k, |va k| ≤ a) (hvte : ∀ k, |vt k - va k| ≤ e)
    (k : Fin (oc * h * w)) :
    |M.flatConvF W b vt k - flatConv W b va k| ≤
      FloatModel.layerBudget M.u (ic * kH * kW) w' β a e := by
  have huf_e : ∀ c i j,
      |Tensor3.unflatten vt c i j - Tensor3.unflatten va c i j| ≤ e := by
    intro c i j; simp only [Tensor3.unflatten]; exact hvte _
  have huf_a : ∀ c i j, |Tensor3.unflatten va c i j| ≤ a := by
    intro c i j; simp only [Tensor3.unflatten]; exact hva _
  simp only [FloatModel.flatConvF, flatConv, Tensor3.flatten]
  refine (M.convF_close W b (Tensor3.unflatten vt) (Tensor3.unflatten va)
    he huf_e _ _ _).trans ?_
  exact M.denseErr_le_uniform hw' he (fun i j => convKernelMat_abs_le hW i j) hb
    (fun idx => convWindow_abs_le ha huf_a _ _ idx) _

/-- Vec-space conv magnitude bound (the activation-norm pass-through). -/
theorem flatConv_abs_le {ic oc h w kH kW : Nat} {W : Kernel4 oc ic kH kW}
    {b : Vec oc} {v : Vec (ic * h * w)} {w' β a : ℝ} (ha : 0 ≤ a)
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') (hb : ∀ o, |b o| ≤ β)
    (hv : ∀ k, |v k| ≤ a) (k : Fin (oc * h * w)) :
    |flatConv W b v k| ≤ FloatModel.layerAct (ic * kH * kW) w' β a := by
  have huf : ∀ c i j, |Tensor3.unflatten v c i j| ≤ a := by
    intro c i j; simp only [Tensor3.unflatten]; exact hv _
  simp only [flatConv, Tensor3.flatten]
  exact conv2d_abs_le ha hW hb huf _ _ _

-- ════════════════════════════════════════════════════════════════
-- § Whole-net MNIST-CNN forward rounding budget (Item A capstone)
-- ════════════════════════════════════════════════════════════════

/-- **The float MNIST-CNN (no BN) forward** — the float peer of
    `mnistCnnNoBnForward`: rounded conv (`flatConvF`) and rounded dense
    (`M.dense`); `relu` and `maxPoolFlat` appear bare (exact in float). -/
noncomputable def FloatModel.mnistCnnNoBnForwardF
    {ic c h w d1 nClasses kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) :
    Vec (ic * (2*h) * (2*w)) → Vec nClasses :=
  M.dense W₅ b₅
  ∘ (relu d1 ∘ M.dense W₄ b₄)
  ∘ (relu d1 ∘ M.dense W₃ b₃)
  ∘ maxPoolFlat c h w
  ∘ (relu (c * (2*h) * (2*w)) ∘ M.flatConvF (h := 2*h) (w := 2*w) W₂ b₂)
  ∘ (relu (c * (2*h) * (2*w)) ∘ M.flatConvF (h := 2*h) (w := 2*w) W₁ b₁)

/-- **Whole-net MNIST-CNN forward rounding budget (Item A capstone).** The
    rounded forward is within an explicit closed-form `layerBudget` of the real
    `conv→relu→conv→relu→maxpool→dense→relu→dense→relu→dense` forward, per
    output logit — the binary32 forward-error bound for the Chapter-4 CNN.

    Each weight layer threads identically: conv layers as `dense` at their
    fan-in (`ic·kH·kW`, then `c·kH·kW`), the dense head at `c·h·w` / `d1`; relu
    and maxpool pass error through exactly (no rounding, no amplification). The
    budget is the `mlp_float_close_uniform` nest extended to the CNN's six
    layers — `norm_num`-evaluable at a concrete net and magnitude profile. -/
theorem FloatModel.cnn_float_close
    {ic c h w d1 nClasses kH kW : Nat} (M : FloatModel)
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d1) (b₃ : Vec d1)
    (W₄ : Mat d1 d1) (b₄ : Vec d1)
    (W₅ : Mat d1 nClasses) (b₅ : Vec nClasses) (x : Vec (ic * (2*h) * (2*w)))
    {w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ a : ℝ}
    (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄)
    (hw₅ : 0 ≤ w₅) (ha : 0 ≤ a)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hx : ∀ i, |x i| ≤ a) (k : Fin nClasses) :
    |M.mnistCnnNoBnForwardF W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x k -
        mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x k| ≤
      FloatModel.layerBudget M.u d1 w₅ β₅
        (FloatModel.layerAct d1 w₄ β₄
          (FloatModel.layerAct (c * h * w) w₃ β₃
            (FloatModel.layerAct (c * kH * kW) w₂ β₂
              (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))))
        (FloatModel.layerBudget M.u d1 w₄ β₄
          (FloatModel.layerAct (c * h * w) w₃ β₃
            (FloatModel.layerAct (c * kH * kW) w₂ β₂
              (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)))
          (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
            (FloatModel.layerAct (c * kH * kW) w₂ β₂
              (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
            (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
              (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
              (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)))) := by
  simp only [FloatModel.mnistCnnNoBnForwardF, mnistCnnNoBnForward, Function.comp]
  -- real activation magnitudes, layer by layer
  set A1 := FloatModel.layerAct (ic * kH * kW) w₁ β₁ a with hA1
  set A2 := FloatModel.layerAct (c * kH * kW) w₂ β₂ A1 with hA2
  set A3 := FloatModel.layerAct (c * h * w) w₃ β₃ A2 with hA3
  set A4 := FloatModel.layerAct d1 w₄ β₄ A3 with hA4
  set E1 := FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 with hE1
  set E2 := FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ A1 E1 with hE2
  set E3 := FloatModel.layerBudget M.u (c * h * w) w₃ β₃ A2 E2 with hE3
  set E4 := FloatModel.layerBudget M.u d1 w₄ β₄ A3 E3 with hE4
  have hA1_0 : 0 ≤ A1 := FloatModel.layerAct_nonneg hw₁ hβ₁ ha
  have hE1_0 : 0 ≤ E1 := FloatModel.layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl
  have hA2_0 : 0 ≤ A2 := FloatModel.layerAct_nonneg hw₂ hβ₂ hA1_0
  have hE2_0 : 0 ≤ E2 := FloatModel.layerBudget_nonneg M.u_nonneg hw₂ hβ₂ hA1_0 hE1_0
  have hA3_0 : 0 ≤ A3 := FloatModel.layerAct_nonneg hw₃ hβ₃ hA2_0
  have hE3_0 : 0 ≤ E3 := FloatModel.layerBudget_nonneg M.u_nonneg hw₃ hβ₃ hA2_0 hE2_0
  have hA4_0 : 0 ≤ A4 := FloatModel.layerAct_nonneg hw₄ hβ₄ hA3_0
  -- real activation magnitude bounds
  have mA1 : ∀ j, |relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x) j| ≤ A1 :=
    fun j => (FloatModel.relu_abs_le _ j).trans (flatConv_abs_le ha hW₁ hb₁ hx j)
  have mA2 : ∀ j, |relu (c * (2*h) * (2*w))
      (flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x))) j| ≤ A2 :=
    fun j => (FloatModel.relu_abs_le _ j).trans
      (flatConv_abs_le hA1_0 hW₂ hb₂ mA1 j)
  have mAp : ∀ j, |maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x)))) j| ≤ A2 :=
    fun j => maxPoolFlat_abs_le mA2 j
  have mA3 : ∀ j, |relu d1 (Proofs.dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x)))))) j| ≤ A3 :=
    fun j => (FloatModel.relu_abs_le _ j).trans (FloatModel.dense_abs_le hA2_0 hW₃ hb₃ mAp j)
  have mA4 : ∀ j, |relu d1 (Proofs.dense W₄ b₄ (relu d1 (Proofs.dense W₃ b₃
      (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (flatConv W₂ b₂
        (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x)))))))) j| ≤ A4 :=
    fun j => (FloatModel.relu_abs_le _ j).trans (FloatModel.dense_abs_le hA3_0 hW₄ hb₄ mA3 j)
  -- float-vs-real error, layer by layer
  have e1 : ∀ j, |M.flatConvF W₁ b₁ x j - flatConv W₁ b₁ x j| ≤ E1 :=
    fun j => M.flatConvF_close W₁ b₁ x x hw₁ ha le_rfl hW₁ hb₁ hx
      (fun i => by simp) j
  have r1 : ∀ j, |relu (c * (2*h) * (2*w)) (M.flatConvF W₁ b₁ x) j -
      relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x) j| ≤ E1 :=
    fun j => FloatModel.relu_close _ _ E1 e1 j
  have e2 : ∀ j, |M.flatConvF W₂ b₂ (relu (c * (2*h) * (2*w)) (M.flatConvF W₁ b₁ x)) j -
      flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x)) j| ≤ E2 :=
    fun j => M.flatConvF_close W₂ b₂ _ _ hw₂ hA1_0 hE1_0 hW₂ hb₂ mA1 r1 j
  have r2 : ∀ j, |relu (c * (2*h) * (2*w))
      (M.flatConvF W₂ b₂ (relu (c * (2*h) * (2*w)) (M.flatConvF W₁ b₁ x))) j -
      relu (c * (2*h) * (2*w))
      (flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x))) j| ≤ E2 :=
    fun j => FloatModel.relu_close _ _ E2 e2 j
  have ep : ∀ j, |maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (M.flatConvF W₂ b₂ (relu (c * (2*h) * (2*w)) (M.flatConvF W₁ b₁ x)))) j -
      maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x)))) j| ≤ E2 :=
    fun j => maxPoolFlat_close _ _ r2 j
  have e3 : ∀ j, |M.dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (M.flatConvF W₂ b₂ (relu (c * (2*h) * (2*w)) (M.flatConvF W₁ b₁ x))))) j -
      Proofs.dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x))))) j| ≤ E3 :=
    fun j => (M.dense_close W₃ b₃ _ _ E2 hE2_0 ep j).trans
      (M.denseErr_le_uniform hw₃ hE2_0 hW₃ hb₃ mAp j)
  have r3 : ∀ j, |relu d1 (M.dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (M.flatConvF W₂ b₂ (relu (c * (2*h) * (2*w)) (M.flatConvF W₁ b₁ x)))))) j -
      relu d1 (Proofs.dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x)))))) j| ≤ E3 :=
    fun j => FloatModel.relu_close _ _ E3 e3 j
  have e4 : ∀ j, |M.dense W₄ b₄ (relu d1 (M.dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (M.flatConvF W₂ b₂ (relu (c * (2*h) * (2*w)) (M.flatConvF W₁ b₁ x))))))) j -
      Proofs.dense W₄ b₄ (relu d1 (Proofs.dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x))))))) j| ≤ E4 :=
    fun j => (M.dense_close W₄ b₄ _ _ E3 hE3_0 r3 j).trans
      (M.denseErr_le_uniform hw₄ hE3_0 hW₄ hb₄ mA3 j)
  have r4 : ∀ j, |relu d1 (M.dense W₄ b₄ (relu d1 (M.dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (M.flatConvF W₂ b₂ (relu (c * (2*h) * (2*w)) (M.flatConvF W₁ b₁ x)))))))) j -
      relu d1 (Proofs.dense W₄ b₄ (relu d1 (Proofs.dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (flatConv W₂ b₂ (relu (c * (2*h) * (2*w)) (flatConv W₁ b₁ x)))))))) j| ≤ E4 :=
    fun j => FloatModel.relu_close _ _ E4 e4 j
  -- final dense layer
  have hE4_0 : 0 ≤ E4 :=
    FloatModel.layerBudget_nonneg M.u_nonneg hw₄ hβ₄ hA3_0 hE3_0
  exact (M.dense_close W₅ b₅ _ _ E4 hE4_0 r4 k).trans
    (M.denseErr_le_uniform hw₅ hE4_0 hW₅ hb₅ mA4 k)
theorem conv2d_kernel_sub {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) (v e : Vec (oc * ic * kH * kW))
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
      conv2d (Kernel4.unflatten v) b x o hi wi =
      ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi := by
  have hb : conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
      conv2d (Kernel4.unflatten v) b x o hi wi =
      (∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        Kernel4.unflatten (v + e) o c kh kw *
          convPad kH kW x c kh kw hi wi) -
      ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        Kernel4.unflatten v o c kh kw *
          convPad kH kW x c kh kw hi wi := by
    rw [conv2d_eq_convPad, conv2d_eq_convPad]
    ring
  rw [hb, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun c _ => ?_
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun kh _ => ?_
  rw [← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun kw _ => ?_
  rw [unflatten_k4Idx, unflatten_k4Idx]
  show (v + e) (k4Idx o c kh kw) * _ - v (k4Idx o c kh kw) * _ = _
  rw [Pi.add_apply]
  ring

/-- **Per-entry conv drift, slab-refined**: a kernel perturbation moves the
    output entry `(o, hi, wi)` by at most `a` times the `ℓ1` mass of the
    channel-`o` slab (each output reads only its own slab). -/
theorem conv2d_kernel_drift {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW))
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
      conv2d (Kernel4.unflatten v) b x o hi wi| ≤
      a * ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        |e (k4Idx o c kh kw)| := by
  rw [conv2d_kernel_sub]
  calc |∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi|
      ≤ ∑ c : Fin ic, |∑ kh : Fin kH, ∑ kw : Fin kW,
          e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, |∑ kw : Fin kW,
          e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi| :=
        Finset.sum_le_sum fun c _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx o c kh kw) * convPad kH kW x c kh kw hi wi| :=
        Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
          Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx o c kh kw)| * a := by
        refine Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
          Finset.sum_le_sum fun kw _ => ?_
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_left
          (abs_convPad_le x ha hx c kh kw hi wi) (abs_nonneg _)
    _ = (∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx o c kh kw)|) * a := by
        rw [Finset.sum_mul]
        refine Finset.sum_congr rfl fun c _ => ?_
        rw [Finset.sum_mul]
        refine Finset.sum_congr rfl fun kh _ => ?_
        rw [Finset.sum_mul]
    _ = a * ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |e (k4Idx o c kh kw)| := mul_comm _ _

/-- Per-entry conv drift against the TOTAL `ℓ1` mass — the form the relu
    margins consume. -/
theorem conv2d_kernel_drift_total {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW))
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    |conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
      conv2d (Kernel4.unflatten v) b x o hi wi| ≤ a * ∑ idx, |e idx| :=
  le_trans (conv2d_kernel_drift b x ha hx v e o hi wi)
    (mul_le_mul_of_nonneg_left (sum_abs_kernel_slab_le e o) ha)

/-- **`ℓ1` conv drift**: summed over all output entries, the drift is at
    most `(h·w)·a·‖e‖₁` — the spatial multiplicity `h·w` is the price of
    weight sharing (each kernel entry touches every spatial position). -/
theorem conv2d_kernel_drift_sum {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW)) :
    ∑ o : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
        |conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
          conv2d (Kernel4.unflatten v) b x o hi wi| ≤
      ((h * w : ℕ) : ℝ) * (a * ∑ idx, |e idx|) := by
  calc ∑ o : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
        |conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
          conv2d (Kernel4.unflatten v) b x o hi wi|
      ≤ ∑ o : Fin oc, ∑ _hi : Fin h, ∑ _wi : Fin w,
          a * ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            |e (k4Idx o c kh kw)| := by
        refine Finset.sum_le_sum fun o _ => Finset.sum_le_sum fun hi _ =>
          Finset.sum_le_sum fun wi _ => ?_
        exact conv2d_kernel_drift b x ha hx v e o hi wi
    _ = ∑ o : Fin oc, ((h * w : ℕ) : ℝ) *
          (a * ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            |e (k4Idx o c kh kw)|) := by
        refine Finset.sum_congr rfl fun o _ => ?_
        rw [Finset.sum_const, Finset.sum_const, Finset.card_univ,
          Finset.card_univ, Fintype.card_fin, Fintype.card_fin,
          nsmul_eq_mul, nsmul_eq_mul]
        push_cast
        ring
    _ = ((h * w : ℕ) : ℝ) * (a * ∑ o : Fin oc, ∑ c : Fin ic,
          ∑ kh : Fin kH, ∑ kw : Fin kW, |e (k4Idx o c kh kw)|) := by
        simp only [Finset.mul_sum]
    _ = ((h * w : ℕ) : ℝ) * (a * ∑ idx, |e idx|) := by
        rw [← sum_abs_k4]

/-- **The pool's routing pattern is frozen**: under the margin, the
    `pdiv3` Jacobian of the pool is entry-for-entry IDENTICAL at the
    margined point and at any `δ`-close point. This is what lets the pool
    behave as a fixed linear selector along the whole step segment. -/
theorem MaxPool2MarginQ.pdiv3_eq {c h w : Nat} {δ : ℝ} (hδ0 : 0 ≤ δ)
    {x y : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x)
    (hclose : ∀ ci hi wi, |y ci hi wi - x ci hi wi| ≤ δ)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w))
    (co : Fin c) (ho : Fin h) (wo : Fin w) :
    pdiv3 maxPool2 y ci hi wi co ho wo =
      pdiv3 maxPool2 x ci hi wi co ho wo := by
  rw [pdiv3_maxPool2_smooth y (hm.smooth_of_close hclose) ci hi wi co ho wo,
    pdiv3_maxPool2_smooth x (hm.smooth hδ0) ci hi wi co ho wo]
  have hiff := hm.isArgmax_iff hclose ci hi wi
  by_cases hA : MaxPool2IsArgmax x ci hi wi
  · have hAy : MaxPool2IsArgmax y ci hi wi := hiff.mpr hA
    simp [hA, hAy]
  · have hAy : ¬ MaxPool2IsArgmax y ci hi wi := fun h => hA (hiff.mp h)
    simp [hA, hAy]

/-- **Float pool-backward closeness** (Increment 1 keystone). Under the pool
    margin the float post-relu argmax matches the real one
    (`isArgmax_iff`), so the pool's backward selector
    `𝟙[(ci,hi,wi) is its window's argmax]·(pooled cotangent)` differs from the
    certified one only through the pooled cotangent value — an indicator
    pass-through (`indicator ∈ {0,1}`), the pool peer of `reluMask_close`.
    The two cotangent values `ay` (float) / `ax` (real) enter only via their
    closeness `|ay − ax| ≤ e`. -/
theorem MaxPool2MarginQ.poolBack_close {c h w : Nat} {δ : ℝ}
    {x y : Tensor3 c (2*h) (2*w)} (hm : MaxPool2MarginQ δ x)
    (hclose : ∀ ci hi wi, |y ci hi wi - x ci hi wi| ≤ δ)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w))
    {ay ax e : ℝ} (ha : |ay - ax| ≤ e) :
    |(if MaxPool2IsArgmax y ci hi wi then ay else 0) -
      (if MaxPool2IsArgmax x ci hi wi then ax else 0)| ≤ e := by
  have hiff := hm.isArgmax_iff hclose ci hi wi
  by_cases hA : MaxPool2IsArgmax x ci hi wi
  · rw [if_pos hA, if_pos (hiff.mpr hA)]; exact ha
  · rw [if_neg hA, if_neg (fun h => hA (hiff.mp h)), sub_zero, abs_zero]
    exact le_trans (abs_nonneg _) ha

-- ════════════════════════════════════════════════════════════════
-- § The 3-dense head above the pool: input-gradient closed form
-- ════════════════════════════════════════════════════════════════

/-- Folds the raw `finProdFinEquiv` encoding back into `t3Idx`. -/
theorem t3Idx_def {c h w : Nat} (ci : Fin c) (hi : Fin h) (wi : Fin w) :
    finProdFinEquiv (finProdFinEquiv (ci, hi), wi) = t3Idx ci hi wi := rfl

/-- `t3Idx` is injective componentwise — the spatial peer of `k4Idx_inj`. -/
theorem t3Idx_inj {c h w : Nat} {ci ci' : Fin c} {hi hi' : Fin h}
    {wi wi' : Fin w} (hEq : t3Idx ci hi wi = t3Idx ci' hi' wi') :
    ci = ci' ∧ hi = hi' ∧ wi = wi' := by
  unfold t3Idx at hEq
  have h1 := finProdFinEquiv.injective hEq
  have hwi : wi = wi' := (Prod.ext_iff.mp h1).2
  have h2 := finProdFinEquiv.injective (Prod.ext_iff.mp h1).1
  exact ⟨(Prod.ext_iff.mp h2).1, (Prod.ext_iff.mp h2).2, hwi⟩

/-- The 3-dense head `CE ∘ d₅ ∘ relu ∘ d₄ ∘ relu ∘ d₃` is differentiable
    at any point whose two ReLU pre-activations are off the kinks. -/
theorem ce_head3_differentiableAt {p d₃ d₄ nC : Nat} (W₃ : Mat p d₃)
    (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC)
    (b₅ : Vec nC) (label : Fin nC) (u : Vec p)
    (hz3 : ∀ l, dense W₃ b₃ u l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)) q ≠ 0) :
    DifferentiableAt ℝ
      (fun y : Vec p => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y)))))
        label) u := by
  rw [differentiableAt_pi]
  intro _
  have h1 : DifferentiableAt ℝ
      (fun y : Vec p => relu d₃ (dense W₃ b₃ y)) u :=
    (relu_differentiableAt_of_smooth d₃ _ hz3).comp
      (f := fun y : Vec p => dense W₃ b₃ y) u ((dense_differentiable W₃ b₃) u)
  have h2 : DifferentiableAt ℝ
      (fun y : Vec p => dense W₄ b₄ (relu d₃ (dense W₃ b₃ y))) u :=
    ((dense_differentiable W₄ b₄) _).comp
      (f := fun y : Vec p => relu d₃ (dense W₃ b₃ y)) u h1
  have h3 : DifferentiableAt ℝ
      (fun y : Vec p => relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y)))) u :=
    (relu_differentiableAt_of_smooth d₄ _ hz4).comp
      (f := fun y : Vec p => dense W₄ b₄ (relu d₃ (dense W₃ b₃ y))) u h2
  have h4 : DifferentiableAt ℝ
      (fun y : Vec p =>
        dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y))))) u :=
    ((dense_differentiable W₅ b₅) _).comp
      (f := fun y : Vec p => relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y))))
      u h3
  exact (crossEntropy_differentiable nC label).differentiableAt.comp
    (f := fun y : Vec p =>
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y))))) u h4

/-- **Loss input-gradient of the 3-dense head** `CE∘d₅∘relu∘d₄∘relu∘d₃`
    at the pooled vector — one `pdiv_comp` hop (peel `dense W₃`) on top of
    `ce_head2_input_grad`, exactly as `ce_head2` was one hop on
    `ce_head_relu`. Note there is NO leading mask: the pool output feeds
    `dense W₃` directly. -/
theorem ce_head3_input_grad {p d₃ d₄ nC : Nat} (W₃ : Mat p d₃)
    (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC)
    (b₅ : Vec nC) (label : Fin nC) (u : Vec p)
    (hz3 : ∀ l, dense W₃ b₃ u l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)) q ≠ 0) (j : Fin p) :
    pdiv (fun y : Vec p => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y)))))
        label) u j 0
      = ∑ l, W₃ j l *
          ((if dense W₃ b₃ u l > 0 then (1:ℝ) else 0) *
            ∑ q, W₄ l q *
              ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)) q > 0
                  then (1:ℝ) else 0) *
                ∑ k, W₅ q k *
                  (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                      (relu d₃ (dense W₃ b₃ u))))) k -
                    oneHot nC label k))) := by
  have hH : DifferentiableAt ℝ
      (fun z : Vec d₃ => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ z)))) label)
      (dense W₃ b₃ u) := by
    rw [differentiableAt_pi]
    intro _
    have h1 : DifferentiableAt ℝ
        (fun z : Vec d₃ => relu d₄ (dense W₄ b₄ (relu d₃ z)))
        (dense W₃ b₃ u) :=
      (relu_differentiableAt_of_smooth d₄ _ hz4).comp
        (f := fun z : Vec d₃ => dense W₄ b₄ (relu d₃ z)) _
        (((dense_differentiable W₄ b₄) _).comp (f := relu d₃) _
          (relu_differentiableAt_of_smooth d₃ _ hz3))
    have h2 : DifferentiableAt ℝ
        (fun z : Vec d₃ => dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ z))))
        (dense W₃ b₃ u) :=
      ((dense_differentiable W₅ b₅) _).comp
        (f := fun z : Vec d₃ => relu d₄ (dense W₄ b₄ (relu d₃ z))) _ h1
    exact (crossEntropy_differentiable nC label).differentiableAt.comp
      (f := fun z : Vec d₃ =>
        dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ z)))) _ h2
  rw [show (fun y : Vec p => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ y)))))
          label)
        = (fun z : Vec d₃ => fun _ : Fin 1 => crossEntropy nC
            (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ z)))) label)
          ∘ (dense W₃ b₃) from rfl,
      pdiv_comp _ _ _ ((dense_differentiable W₃ b₃) u) hH]
  refine Finset.sum_congr rfl fun l _ => ?_
  rw [pdiv_dense, ce_head2_input_grad W₄ b₄ W₅ b₅ label _ hz3 hz4 l]

-- ════════════════════════════════════════════════════════════════
-- § Through the pool: the loss gradient at the conv output
-- ════════════════════════════════════════════════════════════════

/-- The whole head above the conv output — `CE∘head3∘maxPoolFlat∘relu` —
    is differentiable at any point with the relu₂ pre-activation off the
    kinks, no pool ties (POST-relu), and the two head masks off the
    kinks. -/
theorem pool_head_differentiableAt {c h w d₃ d₄ nC : Nat}
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (z₂ : Vec (c * (2*h) * (2*w))) (hz2 : ∀ k, z₂ k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₂) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) z₂)) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) z₂)))) q ≠ 0) :
    DifferentiableAt ℝ
      (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label)
      z₂ := by
  have hHd := ce_head3_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label
    (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) z₂)) hz3 hz4
  have hpt : Tensor3.flatten (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₂) : Tensor3 c (2*h) (2*w)) =
      relu (c * (2*h) * (2*w)) z₂ := Tensor3.flatten_unflatten _
  have hmp_d : DifferentiableAt ℝ (maxPoolFlat c h w)
      (relu (c * (2*h) * (2*w)) z₂) := by
    rw [← hpt]
    exact maxPoolFlat_differentiableAt _ hmp hc hh hw
  have h1 : DifferentiableAt ℝ
      (fun y : Vec (c * (2*h) * (2*w)) =>
        maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y)) z₂ :=
    hmp_d.comp (f := relu (c * (2*h) * (2*w))) z₂
      (relu_differentiableAt_of_smooth _ _ hz2)
  exact hHd.comp
    (f := fun y : Vec (c * (2*h) * (2*w)) =>
      maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y)) z₂ h1

/-- **Loss input-gradient at the conv output** — the key glue of the conv
    rung. The chain `pdiv`s through the relu (mask) and the pool (frozen
    selector): at a smooth point the sum over pooled coordinates collapses
    to the single argmax term, so

    `∂(CE∘head3∘pool∘relu)/∂z₂[ci,hi,wi] = relu'(z₂[ci,hi,wi]) ·
       𝟙[(ci,hi,wi) is its window's argmax] · head3grad(window(ci,hi,wi))`.

    NB the pool acts on the POST-relu activation, so the smoothness and
    argmax conditions are stated on `relu z₂`, not `z₂`. -/
theorem pool_relu_input_grad {c h w d₃ d₄ nC : Nat}
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (z₂ : Vec (c * (2*h) * (2*w))) (hz2 : ∀ k, z₂ k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₂) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) z₂)) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) z₂)))) q ≠ 0)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label)
        z₂ (t3Idx ci hi wi) 0
      = (if z₂ (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
          (if MaxPool2IsArgmax
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) z₂)) ci hi wi
            then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
              ((if dense W₃ b₃ (maxPoolFlat c h w
                    (relu (c * (2*h) * (2*w)) z₂)) l > 0
                  then (1:ℝ) else 0) *
                ∑ q, W₄ l q *
                  ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                        (relu (c * (2*h) * (2*w)) z₂)))) q > 0
                      then (1:ℝ) else 0) *
                    ∑ k, W₅ q k *
                      (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                          (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) z₂))))))) k -
                        oneHot nC label k)))
            else 0) := by
  have hc : 0 < c := Fin.pos ci
  have hh : 0 < h := by have := Fin.pos hi; omega
  have hw : 0 < w := by have := Fin.pos wi; omega
  have hHd := ce_head3_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label
    (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) z₂)) hz3 hz4
  have hpt : Tensor3.flatten (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₂) : Tensor3 c (2*h) (2*w)) =
      relu (c * (2*h) * (2*w)) z₂ := Tensor3.flatten_unflatten _
  have hmp_d : DifferentiableAt ℝ (maxPoolFlat c h w)
      (relu (c * (2*h) * (2*w)) z₂) := by
    rw [← hpt]
    exact maxPoolFlat_differentiableAt _ hmp hc hh hw
  have hG : DifferentiableAt ℝ
      ((fun u : Vec (c * h * w) => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))))
          label) ∘ (maxPoolFlat c h w))
      (relu (c * (2*h) * (2*w)) z₂) :=
    hHd.comp _ hmp_d
  -- hop 1: peel the relu; the chain picks up the mask
  rw [show (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
          crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
            (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label)
        = ((fun u : Vec (c * h * w) => fun _ : Fin 1 => crossEntropy nC
            (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))))
            label) ∘ (maxPoolFlat c h w)) ∘ (relu (c * (2*h) * (2*w)))
        from rfl,
      pdiv_comp _ _ _
        (relu_differentiableAt_of_smooth (c * (2*h) * (2*w)) z₂ hz2) hG]
  simp_rw [pdiv_relu (c * (2*h) * (2*w)) z₂ hz2 (t3Idx ci hi wi), ite_mul,
    zero_mul]
  rw [Finset.sum_ite_eq]
  simp only [Finset.mem_univ, if_true]
  congr 1
  -- hop 2: through the pool; the routing collapses to the argmax cell
  rw [pdiv_comp (maxPoolFlat c h w) _ _ hmp_d hHd (t3Idx ci hi wi) 0,
    sum_t3 (fun q : Fin (c * h * w) =>
      pdiv (maxPoolFlat c h w) (relu (c * (2*h) * (2*w)) z₂)
        (t3Idx ci hi wi) q *
      pdiv (fun u : Vec (c * h * w) => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))))
          label) (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) z₂)) q 0)]
  -- the pool pdiv IS pdiv3 of maxPool2, which the margin collapses
  have hglue : ∀ (co : Fin c) (ho : Fin h) (wo : Fin w),
      pdiv (maxPoolFlat c h w) (relu (c * (2*h) * (2*w)) z₂)
        (t3Idx ci hi wi) (t3Idx co ho wo) =
      (if co = ci ∧ ho = winRow hi ∧ wo = winCol wi ∧
          MaxPool2IsArgmax (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) z₂)) ci hi wi
        then (1:ℝ) else 0) := by
    intro co ho wo
    have h1 : pdiv (maxPoolFlat c h w) (relu (c * (2*h) * (2*w)) z₂)
        (t3Idx ci hi wi) (t3Idx co ho wo) =
        pdiv3 maxPool2 (Tensor3.unflatten (relu (c * (2*h) * (2*w)) z₂))
          ci hi wi co ho wo := by
      unfold pdiv3
      rw [hpt]
      rfl
    rw [h1, pdiv3_maxPool2_smooth _ hmp ci hi wi co ho wo]
  simp_rw [hglue]
  rw [Finset.sum_eq_single ci
      (fun co _ hne_co => by
        rw [Finset.sum_eq_zero]
        intro ho _
        rw [Finset.sum_eq_zero]
        intro wo _
        rw [if_neg (fun hcon => hne_co hcon.1)]
        ring)
      (fun habs => absurd (Finset.mem_univ ci) habs)]
  rw [Finset.sum_eq_single (winRow hi)
      (fun ho _ hne_ho => by
        rw [Finset.sum_eq_zero]
        intro wo _
        rw [if_neg (fun hcon => hne_ho hcon.2.1)]
        ring)
      (fun habs => absurd (Finset.mem_univ _) habs)]
  rw [Finset.sum_eq_single (winCol wi)
      (fun wo _ hne_wo => by
        rw [if_neg (fun hcon => hne_wo hcon.2.2.1)]
        ring)
      (fun habs => absurd (Finset.mem_univ _) habs)]
  by_cases hA : MaxPool2IsArgmax
      (Tensor3.unflatten (relu (c * (2*h) * (2*w)) z₂)) ci hi wi
  · rw [if_pos ⟨rfl, rfl, rfl, hA⟩, if_pos hA,
      ce_head3_input_grad W₃ b₃ W₄ b₄ W₅ b₅ label _ hz3 hz4]
    simp only [ite_mul, one_mul, zero_mul]
  · rw [if_neg (fun hcon => hA hcon.2.2.2), if_neg hA, zero_mul]

-- ════════════════════════════════════════════════════════════════
-- § The conv weight-map Jacobian: closed form, point-free, ℓ1 row mass
-- ════════════════════════════════════════════════════════════════

/-- **Closed form of the conv weight-map `pdiv`** — extracted from the
    certified VJP (`conv2d_weight_grad_has_vjp`) by contracting its
    `.correct` field against a basis vector. Kernel entry `(o,cc,kh,kw)`
    touches output `(co,hi,wi)` iff `co = o`, with coefficient the padded
    input read `convPad`. NB the right-hand side does not mention `v`:
    the weight map is affine, so its Jacobian is point-free — this is
    what lets the gradient difference along a step segment collapse to
    the head drift alone. -/
theorem conv2d_weight_pdiv {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) (v : Vec (oc * ic * kH * kW))
    (o : Fin oc) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW)
    (co : Fin oc) (hi : Fin h) (wi : Fin w) :
    pdiv (fun v' : Vec (oc * ic * kH * kW) =>
        Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v
      (k4Idx o cc kh kw) (t3Idx co hi wi)
      = if co = o then convPad kH kW x cc kh kw hi wi else 0 := by
  have hb := conv_weight_grad_bridge b x v (basisVec (t3Idx co hi wi))
    (k4Idx o cc kh kw)
  have hsum : ∑ j : Fin (oc * h * w),
      pdiv (fun v' : Vec (oc * ic * kH * kW) =>
          Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v
        (k4Idx o cc kh kw) j * basisVec (t3Idx co hi wi) j
      = pdiv (fun v' : Vec (oc * ic * kH * kW) =>
          Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v
        (k4Idx o cc kh kw) (t3Idx co hi wi) := by
    rw [Finset.sum_eq_single (t3Idx co hi wi)
      (fun j _ hne => by rw [basisVec_apply, if_neg hne, mul_zero])
      (fun habs => absurd (Finset.mem_univ _) habs)]
    rw [basisVec_apply, if_pos rfl, mul_one]
  rw [← hsum, ← hb]
  -- evaluate the transpose-trick backward at the basis vector
  simp only [conv2d_weight_grad_has_vjp, k4Idx, Equiv.symm_apply_apply,
    basisVec_apply, convPad]
  simp only [t3Idx_def]
  rcases eq_or_ne co o with hco | hco
  · subst hco
    rw [if_pos rfl,
      Finset.sum_eq_single hi
        (fun hi' _ hne_hi => by
          rw [Finset.sum_eq_zero]
          intro wi' _
          rw [if_neg (fun heq => hne_hi
            (t3Idx_inj (show t3Idx co hi' wi' = t3Idx co hi wi
              from heq)).2.1), mul_zero])
        (fun habs => absurd (Finset.mem_univ _) habs),
      Finset.sum_eq_single wi
        (fun wi' _ hne_wi => by
          rw [if_neg (fun heq => hne_wi
            (t3Idx_inj (show t3Idx co hi wi' = t3Idx co hi wi
              from heq)).2.2), mul_zero])
        (fun habs => absurd (Finset.mem_univ _) habs),
      if_pos rfl, mul_one]
  · rw [if_neg hco, Finset.sum_eq_zero]
    intro hi' _
    rw [Finset.sum_eq_zero]
    intro wi' _
    rw [if_neg (fun heq => hco
      ((t3Idx_inj (show t3Idx o hi' wi' = t3Idx co hi wi from heq)).1).symm),
      mul_zero]

/-- The `ℓ1` mass of one Jacobian row of the conv weight map: kernel
    entry `(o,cc,kh,kw)` touches the `(h·w)` outputs of its slab, each
    with a padded read bounded by `a` — the quantitative form of "weight
    sharing costs a spatial multiplicity". -/
theorem conv2d_weight_pdiv_row_l1 {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v : Vec (oc * ic * kH * kW))
    (o : Fin oc) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    ∑ k : Fin (oc * h * w),
        |pdiv (fun v' : Vec (oc * ic * kH * kW) =>
            Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v
          (k4Idx o cc kh kw) k| ≤ ((h * w : ℕ) : ℝ) * a := by
  rw [sum_t3 (fun k : Fin (oc * h * w) =>
    |pdiv (fun v' : Vec (oc * ic * kH * kW) =>
        Tensor3.flatten (conv2d (Kernel4.unflatten v') b x)) v
      (k4Idx o cc kh kw) k|)]
  simp_rw [conv2d_weight_pdiv b x v o cc kh kw]
  rw [Finset.sum_eq_single o
    (fun co _ hne_co => by
      rw [Finset.sum_eq_zero]
      intro hi _
      rw [Finset.sum_eq_zero]
      intro wi _
      rw [if_neg hne_co, abs_zero])
    (fun habs => absurd (Finset.mem_univ _) habs)]
  calc ∑ hi : Fin h, ∑ wi : Fin w,
        |if o = o then convPad kH kW x cc kh kw hi wi else 0|
      ≤ ∑ _hi : Fin h, ∑ _wi : Fin w, a := by
        refine Finset.sum_le_sum fun hi _ => Finset.sum_le_sum fun wi _ => ?_
        rw [if_pos rfl]
        exact abs_convPad_le x ha hx cc kh kw hi wi
    _ = ((h * w : ℕ) : ℝ) * a := by
        rw [Finset.sum_const, Finset.sum_const, Finset.card_univ,
          Finset.card_univ, Fintype.card_fin, Fintype.card_fin, smul_smul,
          nsmul_eq_mul]

-- ════════════════════════════════════════════════════════════════
-- § Conv gradient-step rounding (planning §1b-B): the conv weight grad is
--   a spatial correlation (a dot), the bias grad a spatial sum — so both
--   rounded SGD steps reduce to the generic dot/sum step closes.
-- ════════════════════════════════════════════════════════════════

/-- The spatial `(hi, wi)` sum collapses to one flat sum over `Fin (h·w)`. -/
theorem sum_s2 {h w : Nat} (g : Fin (h * w) → ℝ) :
    ∑ s, g s = ∑ hi : Fin h, ∑ wi : Fin w, g (finProdFinEquiv (hi, wi)) := by
  calc ∑ s, g s = ∑ p : Fin h × Fin w, g (finProdFinEquiv p) :=
        (Equiv.sum_comp finProdFinEquiv g).symm
    _ = ∑ hi : Fin h, ∑ wi : Fin w, g (finProdFinEquiv (hi, wi)) :=
        Fintype.sum_prod_type _

/-- The padded-input window for a fixed kernel slot, flattened over the
    `(hi, wi)` spatial grid — the left operand of the conv weight-grad dot. -/
noncomputable def convPadWin {ic h w : Nat} (kH kW : Nat) (x : Tensor3 ic h w)
    (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) : Vec (h * w) :=
  fun s => convPad kH kW x cc kh kw (finProdFinEquiv.symm s).1
    (finProdFinEquiv.symm s).2

/-- The cotangent slab for a fixed output channel, flattened over `(hi, wi)`. -/
noncomputable def cotWin {oc h w : Nat} (cot : Tensor3 oc h w) (o : Fin oc) :
    Vec (h * w) :=
  fun s => cot o (finProdFinEquiv.symm s).1 (finProdFinEquiv.symm s).2

@[simp] theorem convPadWin_apply {ic h w : Nat} (kH kW : Nat)
    (x : Tensor3 ic h w) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW)
    (hi : Fin h) (wi : Fin w) :
    convPadWin kH kW x cc kh kw (finProdFinEquiv (hi, wi)) =
      convPad kH kW x cc kh kw hi wi := by
  simp [convPadWin, Equiv.symm_apply_apply]

@[simp] theorem cotWin_apply {oc h w : Nat} (cot : Tensor3 oc h w)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    cotWin cot o (finProdFinEquiv (hi, wi)) = cot o hi wi := by
  simp [cotWin, Equiv.symm_apply_apply]

/-- **The conv weight gradient is the spatial dot** `Σ_{hi,wi} convPad · cot`
    (the contraction `conv2d_weight_pdiv` certifies as `∂L/∂W_{o,cc,kh,kw}`),
    re-expressed as a flat `Fin (h·w)` dot of the padded-input window against
    the cotangent slab — the form the float dot rounds. -/
theorem convWeightGrad_eq_dot {ic oc h w kH kW : Nat} (x : Tensor3 ic h w)
    (cot : Tensor3 oc h w) (o : Fin oc) (cc : Fin ic) (kh : Fin kH)
    (kw : Fin kW) :
    ∑ s, convPadWin kH kW x cc kh kw s * cotWin cot o s =
      ∑ hi : Fin h, ∑ wi : Fin w,
        convPad kH kW x cc kh kw hi wi * cot o hi wi := by
  rw [sum_s2 (fun s => convPadWin kH kW x cc kh kw s * cotWin cot o s)]
  refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
  rw [convPadWin_apply, cotWin_apply]

/-- The conv bias gradient is the spatial sum `Σ_{hi,wi} cot`. -/
theorem convBiasGrad_eq_sum {oc h w : Nat} (cot : Tensor3 oc h w) (o : Fin oc) :
    ∑ s, cotWin cot o s = ∑ hi : Fin h, ∑ wi : Fin w, cot o hi wi := by
  rw [sum_s2 (fun s => cotWin cot o s)]
  refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
  rw [cotWin_apply]

/-- **Rounded conv weight update (Item B).** The float update
    `fl(Wₒ,cc,kh,kw − fl(lr·fl(convPadWin · cotWin)))` — the conv weight
    gradient is a correlation, a dot over the `h·w` spatial positions — is
    within `sgdErr` of the real step `W − lr·(Σ_{hi,wi} convPad·cot)`, the
    dot's Higham γ (fan-in `h·w`) as the gradient-error slot. Reuses the
    generic `dotSgd_step_close`; the cotangent is supplied (the loss-head
    `exp` accuracy lives in `cotErr`). -/
theorem FloatModel.cnn_convW_step_float_close {ic oc h w kH kW : Nat}
    (M : FloatModel) (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w)
    (cot : Tensor3 oc h w) {lr G : ℝ} (o : Fin oc) (cc : Fin ic)
    (kh : Fin kH) (kw : Fin kW)
    (hG : |∑ s, convPadWin kH kW x cc kh kw s * cotWin cot o s| ≤ G)
    (hlr : 0 ≤ lr) :
    |M.sub (W o cc kh kw)
        (M.mul lr (M.dot (convPadWin kH kW x cc kh kw) (cotWin cot o))) -
      (W o cc kh kw - lr * ∑ s,
        convPadWin kH kW x cc kh kw s * cotWin cot o s)| ≤
      sgdErr M.u lr |W o cc kh kw| G
        (((1 + M.u) ^ (h * w + 1) - 1) *
          ∑ s, |convPadWin kH kW x cc kh kw s * cotWin cot o s|) :=
  M.dotSgd_step_close (W o cc kh kw) (convPadWin kH kW x cc kh kw)
    (cotWin cot o) hG hlr

/-- **Rounded conv bias update (Item B)** — the bias gradient is the spatial
    sum `Σ cot`, so the rounded update reduces to `sumSgd_step_close`. -/
theorem FloatModel.cnn_convb_step_float_close {oc h w : Nat} (M : FloatModel)
    (b : Vec oc) (cot : Tensor3 oc h w) {lr G : ℝ} (o : Fin oc)
    (hG : |∑ s, cotWin cot o s| ≤ G) (hlr : 0 ≤ lr) :
    |M.sub (b o) (M.mul lr (M.sum (cotWin cot o))) -
      (b o - lr * ∑ s, cotWin cot o s)| ≤
      sgdErr M.u lr |b o| G
        (((1 + M.u) ^ (h * w + 1) - 1) * ∑ s, |cotWin cot o s|) :=
  M.sumSgd_step_close (b o) (cotWin cot o) hG hlr

/-- **Numeric conv-weight-step capstone at the committed MNIST-CNN dims (Item
    C).** The Chapter-4 conv2 is `32→32`, `3×3`, at `28×28` (the conv output
    grid, before maxpool), so the weight gradient is a dot over `28·28 = 784`
    spatial positions. At binary32 (`u ≤ 2⁻²⁴`), `lr = 1/10`, kernel `|W| ≤ 3/5`
    (the trained-magnitude bound, matching the MLP capstone), every rounded
    conv2 weight SGD entry is within **`(a·g)/250 + 10⁻⁷`** of the certified
    real step — where `a` bounds the conv2-input activation and `g` the conv2
    cotangent magnitude.

    Both `a` and `g` are **a-posteriori / measured** quantities (the conv input
    and back-propagated cotangent are not intrinsically `≤ 1`, unlike the
    softmax−onehot loss head), supplied as hypotheses — the same worst-case→
    measured hand-off as the forward `δ`. The decimal rate `1/250 ≈ 0.4%` is
    dominated by `lr·γ₇₈₅` (the gradient's Higham error at learning-rate scale):
    the conv weight step is as accurate as the gradient itself, no worse. -/
theorem FloatModel.mnist_cnn_convW_step_float_budget (M : FloatModel)
    (hMu : M.u ≤ u32) (W : Kernel4 32 32 3 3) (act : Tensor3 32 28 28)
    (cot : Tensor3 32 28 28) {a g : ℝ} (ha : 0 ≤ a) (hg : 0 ≤ g)
    (hW : ∀ o cc kh kw, |W o cc kh kw| ≤ 3/5)
    (hact : ∀ c i j, |act c i j| ≤ a) (hcot : ∀ o i j, |cot o i j| ≤ g)
    (o cc : Fin 32) (kh kw : Fin 3) :
    |M.sub (W o cc kh kw)
        (M.mul (1/10) (M.dot (convPadWin 3 3 act cc kh kw) (cotWin cot o))) -
      (W o cc kh kw - (1/10) * ∑ s,
        convPadWin 3 3 act cc kh kw s * cotWin cot o s)| ≤
      (a * g) / 250 + 1/10000000 := by
  have hu := M.u_nonneg
  -- per-term and summed magnitude of the conv weight gradient
  have hterm : ∀ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤ a * g := by
    intro s
    rw [abs_mul]
    refine mul_le_mul ?_ ?_ (abs_nonneg _) ha
    · simp only [convPadWin]; exact abs_convPad_le act ha hact _ _ _ _ _
    · simp only [cotWin]; exact hcot _ _ _
  have hsum : ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤
      784 * (a * g) := by
    calc ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s|
        ≤ ∑ _s : Fin (28 * 28), a * g := Finset.sum_le_sum fun s _ => hterm s
      _ = 784 * (a * g) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul]; norm_num
  have hG : |∑ s, convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤ 784 * (a * g) :=
    (Finset.abs_sum_le_sum_abs _ _).trans hsum
  -- the conv weight step budget (Item B), with G := 784·a·g
  have hstep := M.cnn_convW_step_float_close W act cot o cc kh kw hG
    (by norm_num : (0:ℝ) ≤ 1/10)
  refine hstep.trans ?_
  -- eg ≤ (47/10⁶)·784·a·g  (γ₇₈₅ × the summed gradient mass)
  have hk1 : ((28 * 28 + 1 : ℕ) : ℝ) * u32 < 1 := by norm_num [u32]
  have hk2 : ((28 * 28 + 1 : ℕ) : ℝ) * u32 / (1 - ((28 * 28 + 1 : ℕ) : ℝ) * u32)
      ≤ 47/1000000 := by norm_num [u32]
  have hhigham : (1 + M.u) ^ (28 * 28 + 1) - 1 ≤ 47/1000000 :=
    M.gamma_num hMu hk1 hk2
  have hhigham0 : 0 ≤ (1 + M.u) ^ (28 * 28 + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith))
  have hsum0 : 0 ≤ ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| :=
    Finset.sum_nonneg fun s _ => abs_nonneg _
  have heg : ((1 + M.u) ^ (28 * 28 + 1) - 1) *
      ∑ s, |convPadWin 3 3 act cc kh kw s * cotWin cot o s| ≤
      (47/1000000) * (784 * (a * g)) :=
    mul_le_mul hhigham hsum hsum0 (by norm_num)
  have hag0 : (0:ℝ) ≤ a * g := mul_nonneg ha hg
  have h1 : u32 ≤ 1/16000000 := by norm_num [u32]
  -- push u → the LITERAL 1/16000000, |W| → 3/5, eg → its rational bound (G fixed),
  -- so the closing goal is linear in a·g with constant coefficients
  refine (sgdErr_mono hu (hMu.trans h1) (by norm_num) (abs_nonneg _)
    (hW o cc kh kw) (mul_nonneg (by norm_num) hag0)
    (mul_nonneg hhigham0 hsum0) heg).trans ?_
  set s := a * g with hs
  have hs0 : (0:ℝ) ≤ s := hag0
  unfold FloatModel.sgdErr
  linarith [hs0]

-- ════════════════════════════════════════════════════════════════
-- § The conv2 loss-of-kernel map: differentiability and gradient
-- ════════════════════════════════════════════════════════════════

/-- The loss-of-conv2-kernel map is differentiable wherever the relu₂
    pre-activation is off the kinks, no pool window ties (POST-relu),
    and the two head pre-activations are off the kinks. -/
theorem cnn_conv2_loss_differentiableAt {c h w d₃ d₄ nC kH kW : Nat}
    (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (v : Vec (c * c * kH * kW))
    (hz2 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q ≠ 0) :
    DifferentiableAt ℝ
      (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label) v := by
  have hG := pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label hc hh hw
    (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)) hz2 hmp hz3 hz4
  have h0 : DifferentiableAt ℝ
      (fun v' : Vec (c * c * kH * kW) =>
        Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)) v :=
    (conv2d_weight_differentiable b₂ x₁) v
  exact ((differentiableAt_pi.mp hG) 0).comp
    (f := fun v' : Vec (c * c * kH * kW) =>
      Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)) v h0

/-- **Closed form of the conv2 loss gradient** at any four-margin point —
    the EXISTING fold `conv_total_loss_grad_fold` (generic in the
    downstream `G`) contracted with the pool-collapsed head gradient
    (`pool_relu_input_grad`) and the point-free conv weight Jacobian
    (`conv2d_weight_pdiv`). The conv-layer peer of
    `mlp_input_loss_gradAt`; the spatial triple sum (vs the MLP's
    Kronecker collapse) is weight sharing. -/
theorem cnn_conv2_loss_gradAt {c h w d₃ d₄ nC kH kW : Nat}
    (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (hh : 0 < h) (hw : 0 < w)
    (v : Vec (c * c * kH * kW))
    (hz2 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q ≠ 0)
    (o cc : Fin c) (kh : Fin kH) (kw : Fin kW) :
    gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label)
        v (k4Idx o cc kh kw)
      = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then convPad kH kW x₁ cc kh kw hi wi else 0) *
            ((if Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                    (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
                  ci hi wi
                then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
                  ((if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
                        l > 0 then (1:ℝ) else 0) *
                    ∑ q, W₄ l q *
                      ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q > 0
                          then (1:ℝ) else 0) *
                        ∑ k, W₅ q k *
                          (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                              (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d (Kernel4.unflatten v) b₂ x₁))))))))) k -
                            oneHot nC label k)))
                else 0)) := by
  have hc : 0 < c := Fin.pos o
  have hdiff := cnn_conv2_loss_differentiableAt b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅
    label hc hh hw v hz2 hmp hz3 hz4
  have hG := pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label hc hh hw
    (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)) hz2 hmp hz3 hz4
  calc gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label)
        v (k4Idx o cc kh kw)
      = pdiv (fun v' : Vec (c * c * kH * kW) => fun _ : Fin 1 =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label)
          v (k4Idx o cc kh kw) 0 := gradAt_eq_pdiv _ _ hdiff _
    _ = pdiv (fun v' : Vec (c * c * kH * kW) => fun _ : Fin 1 =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label)
          (Kernel4.flatten (Kernel4.unflatten v)) (k4Idx o cc kh kw) 0 := by
        rw [Kernel4.flatten_unflatten]
    _ = ∑ k : Fin (c * (2*h) * (2*w)),
          pdiv (fun v' : Vec (c * c * kH * kW) =>
              Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁))
            (Kernel4.flatten (Kernel4.unflatten v)) (k4Idx o cc kh kw) k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w
                  (relu (c * (2*h) * (2*w)) y))))))) label)
            (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)) k 0 :=
        conv_total_loss_grad_fold b₂ x₁ (Kernel4.unflatten v)
          (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w
                (relu (c * (2*h) * (2*w)) y))))))) label)
          hG (k4Idx o cc kh kw)
    _ = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          pdiv (fun v' : Vec (c * c * kH * kW) =>
              Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁))
            (Kernel4.flatten (Kernel4.unflatten v)) (k4Idx o cc kh kw)
            (t3Idx ci hi wi) *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w
                  (relu (c * (2*h) * (2*w)) y))))))) label)
            (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))
            (t3Idx ci hi wi) 0 :=
        sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
          pdiv (fun v' : Vec (c * c * kH * kW) =>
              Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁))
            (Kernel4.flatten (Kernel4.unflatten v)) (k4Idx o cc kh kw) k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w
                  (relu (c * (2*h) * (2*w)) y))))))) label)
            (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)) k 0)
    _ = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then convPad kH kW x₁ cc kh kw hi wi else 0) *
            ((if Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                    (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
                  ci hi wi
                then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
                  ((if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
                        l > 0 then (1:ℝ) else 0) *
                    ∑ q, W₄ l q *
                      ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q > 0
                          then (1:ℝ) else 0) *
                        ∑ k, W₅ q k *
                          (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                              (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d (Kernel4.unflatten v) b₂ x₁))))))))) k -
                            oneHot nC label k)))
                else 0)) := by
        refine Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl
          fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
        rw [conv2d_weight_pdiv b₂ x₁ _ o cc kh kw ci hi wi,
          pool_relu_input_grad W₃ b₃ W₄ b₄ W₅ b₅ label _ hz2 hmp hz3 hz4
            ci hi wi]

/-- The unmasked peer of `reluMask_dense_transpose_eq`: a bare `Wᵀ`
    contraction `∑ₖ Wₗₖ·cₖ = dense (transpose W) 0 c l`. The pool feeds
    `dense W₃` with **no** leading ReLU mask, so the W₃ contraction in the
    certified conv-2 gradient collapses through this, where the masked W₄/W₅
    contractions collapse through `reluMask_dense_transpose_eq`. NB this is
    generic in `(W, c)`, so fire it only where the goal has no *other* matrix
    contraction (e.g. the spatial `∑ convPad·cot`) — see `head3_cot_reluMask`. -/
theorem dense_transpose_eq {p n : Nat} (W : Mat p n) (c : Vec n) (l : Fin p) :
    (∑ k, W l k * c k) = dense (fun j i' => W i' j) (fun _ => 0) c l := by
  show (∑ k, W l k * c k) = (∑ k, c k * W l k) + (0:ℝ)
  rw [add_zero]
  exact Finset.sum_congr rfl fun k _ => mul_comm _ _

/-- **The 3-dense head cotangent in `dense`/`reluMask` form.** The raw nested
    `∑ₗ W₃·(𝟙[z₃]·∑_q W₄·(𝟙[z₄]·∑_k W₅·(softmax−onehot)))` that
    `pool_relu_input_grad` / `cnn_conv2_loss_gradAt` leave at the pooled vector
    `u` equals `dense W₃ᵀ 0 (mask z₃ (dense W₄ᵀ 0 (mask z₄ (dense W₅ᵀ 0
    (softmax−onehot)))))` — the two masked contractions via
    `reluMask_dense_transpose_eq`, the unmasked W₃ via `dense_transpose_eq`.
    Stated head-locally (no spatial sum) so the generic `dense_transpose_eq`
    fires only on the W₃ row. The head peer the conv grad-close bounds against
    via `dense_close` (W₃) and `cot_step_close` (W₄/W₅). -/
theorem head3_cot_reluMask {p d₃ d₄ nC : Nat} (W₃ : Mat p d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC) (u : Vec p) (j : Fin p) :
    (∑ l, W₃ j l *
        ((if dense W₃ b₃ u l > 0 then (1:ℝ) else 0) *
          ∑ q, W₄ l q *
            ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)) q > 0 then (1:ℝ) else 0) *
              ∑ k, W₅ q k *
                (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                    (relu d₃ (dense W₃ b₃ u))))) k - oneHot nC label k))))
      = dense (fun j' i' => W₃ i' j') (fun _ => 0)
          (FloatModel.reluMask (dense W₃ b₃ u)
            (dense (fun j' i' => W₄ i' j') (fun _ => 0)
              (FloatModel.reluMask (dense W₄ b₄ (relu d₃ (dense W₃ b₃ u)))
                (dense (fun j' i' => W₅ i' j') (fun _ => 0)
                  (fun k => softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                      (relu d₃ (dense W₃ b₃ u))))) k - oneHot nC label k)))))
          j := by
  simp_rw [reluMask_dense_transpose_eq]
  rw [dense_transpose_eq]

/-- **The certified conv-2 loss gradient, head restated in `dense`/`reluMask`
    form** — the conv peer of `mlp_input_loss_gradAt_reluMask` (Increment 1
    keystone). The two head `Wᵀ` contractions (under the d₄/d₃ ReLU masks)
    collapse via `reluMask_dense_transpose_eq`, the unmasked W₃ contraction via
    `dense_transpose_eq`; the conv-output ReLU mask `𝟙[z₂>0]` and the pool
    argmax selector are kept explicit (their float closeness is handled by
    `reluMask_close` and `MaxPool2MarginQ.poolBack_close`). The whole conv
    gradient is then packaged as the spatial dot `∑ₛ convPadWin·cotWin`
    (`convWeightGrad_eq_dot`) — the exact quantity the rendered trainer's float
    conv-weight dot rounds, so the conv grad-close bounds against this. -/
theorem cnn_conv2_loss_gradAt_reluMask {c h w d₃ d₄ nC kH kW : Nat}
    (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (hh : 0 < h) (hw : 0 < w)
    (v : Vec (c * c * kH * kW))
    (hz2 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q ≠ 0)
    (o cc : Fin c) (kh : Fin kH) (kw : Fin kW) :
    gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label)
        v (k4Idx o cc kh kw)
      = ∑ s, convPadWin kH kW x₁ cc kh kw s *
          cotWin (fun ci hi wi =>
            (if Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                    (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
                  ci hi wi
                then dense (fun j i' => W₃ i' j) (fun _ => 0)
                  (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                        (conv2d (Kernel4.unflatten v) b₂ x₁)))))
                    (dense (fun j i' => W₄ i' j) (fun _ => 0)
                      (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                          (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d (Kernel4.unflatten v) b₂ x₁)))))))
                        (dense (fun j i' => W₅ i' j) (fun _ => 0)
                          (fun k => softmax nC (dense W₅ b₅ (relu d₄
                              (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                                  (Tensor3.flatten (conv2d (Kernel4.unflatten v)
                                    b₂ x₁))))))))) k - oneHot nC label k)))))
                  (t3Idx ci (winRow hi) (winCol wi))
                else 0)) o s := by
  rw [cnn_conv2_loss_gradAt b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw v hz2 hmp hz3
      hz4 o cc kh kw]
  -- restate the head into dense/reluMask form (head-local lemma — does not
  -- touch the spatial `∑ convPad·cot` sum), then package as the spatial dot
  simp_rw [head3_cot_reluMask]
  rw [convWeightGrad_eq_dot x₁ _ o cc kh kw]
  -- collapse the `if ci = o` conv-channel selector
  rw [Finset.sum_eq_single o
    (fun ci _ hne => Finset.sum_eq_zero fun hi _ => Finset.sum_eq_zero fun wi _ =>
      by rw [if_neg hne, zero_mul])
    (fun habs => absurd (Finset.mem_univ o) habs)]
  refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
  rw [if_pos (rfl : o = o)]

-- ════════════════════════════════════════════════════════════════
-- § Increment 2 — the conv2 float-backward grad-close (two generic cores)
-- ════════════════════════════════════════════════════════════════

/-- **Scalar ReLU-mask freeze** — the `(if z>0 then 1 else 0)·x` peer of
    `reluMask_close`. Under the sign margin `ez < |z|` the float and real masks
    agree, so the masked value is 1-Lipschitz in `x`. The conv-output ReLU mask
    `𝟙[z₂>0]` in the conv-2 grad-close sits on a scalar cell (not a `Vec`), so
    it needs this rather than the vector `reluMask_close`. -/
theorem mask_scalar_close {zt z xt x ez ex : ℝ}
    (hz : |zt - z| ≤ ez) (hm : ez < |z|) (hx : |xt - x| ≤ ex) (hex : 0 ≤ ex) :
    |(if zt > 0 then (1:ℝ) else 0) * xt -
      (if z > 0 then (1:ℝ) else 0) * x| ≤ ex := by
  have hzi := abs_le.mp hz
  rcases lt_trichotomy z 0 with hneg | hzero | hpos
  · have h1 : ¬ z > 0 := by linarith
    have h2 : ¬ zt > 0 := by
      rw [not_lt]; rw [abs_of_neg hneg] at hm; linarith [hzi.2]
    rw [if_neg h1, if_neg h2]; simpa using hex
  · exfalso; rw [hzero, abs_zero] at hm
    linarith [(abs_nonneg (zt - z)).trans hz]
  · have h2 : zt > 0 := by
      rw [abs_of_pos hpos] at hm; linarith [hzi.1]
    rw [if_pos hpos, if_pos h2, one_mul, one_mul]; exact hx

/-- **Float dot against a perturbed cotangent** — the conv-2 grad-close's final
    contraction (the conv peer of the MLP's scalar `mul_close`: a dot, because
    of weight sharing). `M.dot A B̃` (exact left operand `A`, float cotangent
    `B̃`) vs the certified `∑ Aᵢ·Bᵢ` splits into the Higham dot rounding on `A·B̃`
    (`dot_close`, fan-in `n`) plus the per-entry cotangent drift
    `|B̃ᵢ − Bᵢ| ≤ eB`. With `|Aᵢ| ≤ a` and `|B̃ᵢ| ≤ Ct`, the bound is closed-form
    and `norm_num`-evaluable. -/
theorem FloatModel.dot_perturbed_close {n : ℕ} (M : FloatModel)
    (A Bt B : Vec n) {a Ct eB : ℝ} (ha : 0 ≤ a)
    (hA : ∀ i, |A i| ≤ a) (hBt : ∀ i, |Bt i| ≤ Ct)
    (hB : ∀ i, |Bt i - B i| ≤ eB) :
    |M.dot A Bt - ∑ i, A i * B i| ≤
      ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * (a * Ct)) + (n : ℝ) * (a * eB) := by
  have hγ0 : (0:ℝ) ≤ (1 + M.u) ^ (n + 1) - 1 :=
    sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
  -- rounding term: ∑|A·B̃| ≤ n·a·Ct
  have h2 : (∑ i, |A i * Bt i|) ≤ (n : ℝ) * (a * Ct) := by
    calc (∑ i, |A i * Bt i|) ≤ ∑ _i : Fin n, a * Ct := by
          refine Finset.sum_le_sum fun i _ => ?_
          rw [abs_mul]; exact mul_le_mul (hA i) (hBt i) (abs_nonneg _) ha
      _ = (n : ℝ) * (a * Ct) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have h1' : |M.dot A Bt - ∑ i, A i * Bt i| ≤
      ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * (a * Ct)) :=
    (M.dot_close A Bt).trans (mul_le_mul_of_nonneg_left h2 hγ0)
  -- drift term: ∑|A|·|B̃ − B| ≤ n·a·eB
  have h3 : |(∑ i, A i * Bt i) - ∑ i, A i * B i| ≤ (n : ℝ) * (a * eB) := by
    rw [← Finset.sum_sub_distrib]
    refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
    calc (∑ i, |A i * Bt i - A i * B i|) ≤ ∑ _i : Fin n, a * eB := by
          refine Finset.sum_le_sum fun i _ => ?_
          rw [show A i * Bt i - A i * B i = A i * (Bt i - B i) from by ring, abs_mul]
          exact mul_le_mul (hA i) (hB i) (abs_nonneg _) ha
      _ = (n : ℝ) * (a * eB) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  calc |M.dot A Bt - ∑ i, A i * B i|
      ≤ |M.dot A Bt - ∑ i, A i * Bt i| +
          |(∑ i, A i * Bt i) - ∑ i, A i * B i| := abs_sub_le _ _ _
    _ ≤ ((1 + M.u) ^ (n + 1) - 1) * ((n : ℝ) * (a * Ct)) +
          (n : ℝ) * (a * eB) := add_le_add h1' h3

/-- **The binary32 conv-2 weight gradient the rendered trainer computes** — the
    conv peer of `mlpInputFloatGrad`. At kernel entry `(o,cc,kh,kw)` it is the
    float dot of the (exact) padded-input window `convPadWin` against the float
    conv-output cotangent slab `cotWin c̃Conv o`, where the float cotangent
    `c̃Conv` rounds every step of the backward — conv-output ReLU mask `𝟙[z̃₂>0]`,
    pool argmax selector (read on the FLOAT post-relu), and the head
    `W₃ᵀ·mask(z̃₃)·W₄ᵀ·mask(z̃₄)·W₅ᵀ·(float softmax−onehot)` at the float
    pre-activations. The `M`-free `reluMask`/`maxPoolFlat`/`relu` are exact in
    float; `M.convF`/`M.dense`/`M.softmaxCECotF` carry the rounding. -/
noncomputable def FloatModel.cnnConv2FloatGrad {c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC)
    (v : Vec (c * c * kH * kW)) : Vec (c * c * kH * kW) :=
  Kernel4.flatten fun o cc kh kw =>
    M.dot (convPadWin kH kW x₁ cc kh kw)
      (cotWin (fun ci hi wi =>
        (if Tensor3.flatten (M.convF (Kernel4.unflatten v) b₂ x₁)
              (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
          (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (M.convF (Kernel4.unflatten v) b₂ x₁)))) ci hi wi
            then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
              (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                  (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                    (M.convF (Kernel4.unflatten v) b₂ x₁)))))
                (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                  (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                      (M.dense W₃ b₃ (maxPoolFlat c h w
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (M.convF (Kernel4.unflatten v) b₂ x₁)))))))
                    (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                      (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                          (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                            (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                              (Tensor3.flatten (M.convF (Kernel4.unflatten v)
                                b₂ x₁))))))))) label)))))
              (t3Idx ci (winRow hi) (winCol wi))
            else 0)) o)

@[simp] theorem FloatModel.cnnConv2FloatGrad_apply {c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC)
    (v : Vec (c * c * kH * kW)) (o cc : Fin c) (kh : Fin kH) (kw : Fin kW) :
    M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label v (k4Idx o cc kh kw) =
      M.dot (convPadWin kH kW x₁ cc kh kw)
        (cotWin (fun ci hi wi =>
          (if Tensor3.flatten (M.convF (Kernel4.unflatten v) b₂ x₁)
                (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
            (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (M.convF (Kernel4.unflatten v) b₂ x₁)))) ci hi wi
              then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
                (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (M.convF (Kernel4.unflatten v) b₂ x₁)))))
                  (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                    (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                        (M.dense W₃ b₃ (maxPoolFlat c h w
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (M.convF (Kernel4.unflatten v) b₂ x₁)))))))
                      (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                        (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                            (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                              (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                                (Tensor3.flatten (M.convF (Kernel4.unflatten v)
                                  b₂ x₁))))))))) label)))))
                (t3Idx ci (winRow hi) (winCol wi))
              else 0)) o) := by
  simp only [FloatModel.cnnConv2FloatGrad, Kernel4.flatten, k4Idx,
    Equiv.symm_apply_apply]

/-- **The conv-2 float-backward grad-close budget** — the closed-form `η` the
    rounded `W₂` gradient stays within of the certified one. Bottom-up: the
    forward rounding nest (`Econv → E₃ → E₄ → δlogit`, conv at fan-in
    `c·kH·kW`, the dense head at `c·h·w / d₃`) feeds the head `cotErr`; the
    backward then rides two `cot_step` `layerBudget`s (W₅/W₄) and the unmasked
    W₃ `layerBudget` to `econv`; finally the spatial dot (fan-in `(2h)·(2w)`)
    contributes its Higham γ on the float-cotangent magnitude `Ctilde` plus the
    per-entry cotangent drift `econv`. The conv peer of the MLP's
    `mulErr/layerBudget/cotErr` nest, deeper by the pool + the dot. -/
noncomputable def FloatModel.cnnConv2GradBudget (M : FloatModel)
    (c h w d₃ d₄ nC kH kW : ℕ) (a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ) : ℝ :=
  let A2 := FloatModel.layerAct (c * kH * kW) w₂ β₂ a
  let A3 := FloatModel.layerAct (c * h * w) w₃ β₃ A2
  let A4 := FloatModel.layerAct d₃ w₄ β₄ A3
  let Econv := FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0
  let E3 := FloatModel.layerBudget M.u (c * h * w) w₃ β₃ A2 Econv
  let E4 := FloatModel.layerBudget M.u d₃ w₄ β₄ A3 E3
  let δlogit := FloatModel.layerBudget M.u d₄ w₅ β₅ A4 E4
  let C4 := FloatModel.layerAct nC w₅ 0 1
  let C3 := FloatModel.layerAct d₄ w₄ 0 C4
  let CPooled := FloatModel.layerAct d₃ w₃ 0 C3
  let ecHead := FloatModel.cotErr M.u eexp δlogit nC
  let ec4 := FloatModel.layerBudget M.u nC w₅ 0 1 ecHead
  let ec3 := FloatModel.layerBudget M.u d₄ w₄ 0 C4 ec4
  let econv := FloatModel.layerBudget M.u d₃ w₃ 0 C3 ec3
  let Ctilde := CPooled + econv
  ((1 + M.u) ^ ((2 * h) * (2 * w) + 1) - 1) *
      (((2 * h) * (2 * w) : ℕ) * (a * Ctilde)) +
    (((2 * h) * (2 * w) : ℕ) * (a * econv))

open FloatModel in
/-- **The binary32 conv-2 weight gradient is within an explicit budget of the
    certified one** (Increment 2 capstone) — the conv-layer peer of
    `mlp_w0_grad_close`, the project's deepest float-backward grad-close. With
    the conv-2 input `x₁` exact, the rendered trainer's `W₂` gradient
    `M.cnnConv2FloatGrad …` stays within `cnnConv2GradBudget` of the certified
    `gradAt`. The chain: float forward (`convF_close` → `dense_close`×3, relu
    and pool error-transparent) ⟶ head (`softmax_ce_cot_close`) ⟶ two masked
    `Wᵀ` `cot_step_close` (W₅ under z̃₄, W₄ under z̃₃) ⟶ unmasked W₃ `dense_close`
    ⟶ pool-backward freeze (`poolBack_close`) ⟶ conv-output ReLU mask freeze
    (`mask_scalar_close`) ⟶ the spatial dot (`dot_perturbed_close`). Four
    quantitative margins are carried (conv-output `Econv`, pool `Econv` POST-relu,
    z̃₃ `E₃`, z̃₄ `E₄`); the bridge `cnn_conv2_loss_gradAt_reluMask` turns the
    `gradAt` into the dot the float gradient rounds. -/
theorem cnn_conv2_grad_close {c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    (v : Vec (c * c * kH * kW))
    {a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (hh : 0 < h) (hw : 0 < w)
    (ha : 0 ≤ a) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂) (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃)
    (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅) (hβ₅ : 0 ≤ β₅)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx₁ : ∀ ci i j, |x₁ ci i j| ≤ a)
    (hv2 : ∀ idx, |v idx| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmarginConv : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0 <
      |Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q|)
    (o cc : Fin c) (kh : Fin kH) (kw : Fin kW) :
    |M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label v (k4Idx o cc kh kw) -
      gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label) v (k4Idx o cc kh kw)|
      ≤ M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp := by
  have hv2' : ∀ o' c' kh' kw', |Kernel4.unflatten v o' c' kh' kw'| ≤ w₂ :=
    fun o' c' kh' kw' => by rw [unflatten_k4Idx]; exact hv2 _
  -- abbreviate the forward values (real / float)
  set ZC := Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) with hZC
  set ZCF := Tensor3.flatten (M.convF (Kernel4.unflatten v) b₂ x₁) with hZCF
  set PR := maxPoolFlat c h w (relu (c * (2*h) * (2*w)) ZC) with hPR
  set PF := maxPoolFlat c h w (relu (c * (2*h) * (2*w)) ZCF) with hPF
  set Z3 := dense W₃ b₃ PR with hZ3
  set Z3F := M.dense W₃ b₃ PF with hZ3F
  set Z4 := dense W₄ b₄ (relu d₃ Z3) with hZ4
  set Z4F := M.dense W₄ b₄ (relu d₃ Z3F) with hZ4F
  set Z5 := dense W₅ b₅ (relu d₄ Z4) with hZ5
  set Z5F := M.dense W₅ b₅ (relu d₄ Z4F) with hZ5F
  -- abbreviate the budgets (real activation magnitudes / forward + backward drifts)
  set A2 := FloatModel.layerAct (c * kH * kW) w₂ β₂ a with hA2
  set A3 := FloatModel.layerAct (c * h * w) w₃ β₃ A2 with hA3
  set A4 := FloatModel.layerAct d₃ w₄ β₄ A3 with hA4
  set Econv := FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0 with hEc
  set E3 := FloatModel.layerBudget M.u (c * h * w) w₃ β₃ A2 Econv with hE3
  set E4 := FloatModel.layerBudget M.u d₃ w₄ β₄ A3 E3 with hE4
  set DL := FloatModel.layerBudget M.u d₄ w₅ β₅ A4 E4 with hDL
  set C4 := FloatModel.layerAct nC w₅ 0 1 with hC4
  set C3 := FloatModel.layerAct d₄ w₄ 0 C4 with hC3
  set CP := FloatModel.layerAct d₃ w₃ 0 C3 with hCP
  set ecH := FloatModel.cotErr M.u eexp DL nC with hecH
  set ec4 := FloatModel.layerBudget M.u nC w₅ 0 1 ecH with hec4
  set ec3 := FloatModel.layerBudget M.u d₄ w₄ 0 C4 ec4 with hec3
  set ecv := FloatModel.layerBudget M.u d₃ w₃ 0 C3 ec3 with hecv
  -- nonnegativity facts
  have A2nn : 0 ≤ A2 := layerAct_nonneg hw₂ hβ₂ ha
  have A3nn : 0 ≤ A3 := layerAct_nonneg hw₃ hβ₃ A2nn
  have A4nn : 0 ≤ A4 := layerAct_nonneg hw₄ hβ₄ A3nn
  have Ecnn : 0 ≤ Econv := layerBudget_nonneg M.u_nonneg hw₂ hβ₂ ha le_rfl
  have E3nn : 0 ≤ E3 := layerBudget_nonneg M.u_nonneg hw₃ hβ₃ A2nn Ecnn
  have E4nn : 0 ≤ E4 := layerBudget_nonneg M.u_nonneg hw₄ hβ₄ A3nn E3nn
  have DLnn : 0 ≤ DL := layerBudget_nonneg M.u_nonneg hw₅ hβ₅ A4nn E4nn
  have C4nn : 0 ≤ C4 := layerAct_nonneg hw₅ le_rfl zero_le_one
  have C3nn : 0 ≤ C3 := layerAct_nonneg hw₄ le_rfl C4nn
  have CPnn : 0 ≤ CP := layerAct_nonneg hw₃ le_rfl C3nn
  have ecHnn : 0 ≤ ecH := M.cotErr_nonneg heexp0 DLnn hρ1
  have ec4nn : 0 ≤ ec4 := layerBudget_nonneg M.u_nonneg hw₅ le_rfl zero_le_one ecHnn
  have ec3nn : 0 ≤ ec3 := layerBudget_nonneg M.u_nonneg hw₄ le_rfl C4nn ec4nn
  have ecvnn : 0 ≤ ecv := layerBudget_nonneg M.u_nonneg hw₃ le_rfl C3nn ec3nn
  -- forward magnitude bounds (real activations)
  have hMconv : ∀ k, |ZC k| ≤ A2 := by
    intro k; obtain ⟨ci, hi, wi, rfl⟩ := t3Idx_surj k
    rw [hZC, flatten_t3Idx]; exact conv2d_abs_le ha hv2' hb₂ hx₁ ci hi wi
  have hMpool : ∀ j, |PR j| ≤ A2 :=
    fun j => maxPoolFlat_abs_le (fun k => (relu_abs_le _ k).trans (hMconv k)) j
  have hM3 : ∀ l, |relu d₃ Z3 l| ≤ A3 :=
    fun l => (relu_abs_le _ l).trans (dense_abs_le A2nn hW₃ hb₃ hMpool l)
  have hM4 : ∀ q, |relu d₄ Z4 q| ≤ A4 :=
    fun q => (relu_abs_le _ q).trans (dense_abs_le A3nn hW₄ hb₄ hM3 q)
  -- forward closeness (float vs real), layer by layer
  have hEconv : ∀ k, |ZCF k - ZC k| ≤ Econv := by
    intro k; obtain ⟨ci, hi, wi, rfl⟩ := t3Idx_surj k
    rw [hZCF, hZC, flatten_t3Idx, flatten_t3Idx]
    exact (M.convF_close (Kernel4.unflatten v) b₂ x₁ x₁ le_rfl
        (fun _ _ _ => by simp) ci hi wi).trans
      (M.denseErr_le_uniform hw₂ le_rfl (fun i j => convKernelMat_abs_le hv2' i j)
        hb₂ (fun idx => convWindow_abs_le ha hx₁ hi wi idx) ci)
  have hRelu : ∀ k, |relu (c * (2*h) * (2*w)) ZCF k -
      relu (c * (2*h) * (2*w)) ZC k| ≤ Econv := fun k => relu_close _ _ _ hEconv k
  have hPool : ∀ k, |PF k - PR k| ≤ Econv := fun k => maxPoolFlat_close _ _ hRelu k
  have hE3close : ∀ l, |Z3F l - Z3 l| ≤ E3 := fun l =>
    (M.dense_close W₃ b₃ PF PR Econv Ecnn hPool l).trans
      (M.denseErr_le_uniform hw₃ Ecnn hW₃ hb₃ hMpool l)
  have hRelu3 : ∀ l, |relu d₃ Z3F l - relu d₃ Z3 l| ≤ E3 :=
    fun l => relu_close _ _ _ hE3close l
  have hE4close : ∀ q, |Z4F q - Z4 q| ≤ E4 := fun q =>
    (M.dense_close W₄ b₄ (relu d₃ Z3F) (relu d₃ Z3) E3 E3nn hRelu3 q).trans
      (M.denseErr_le_uniform hw₄ E3nn hW₄ hb₄ hM3 q)
  have hRelu4 : ∀ q, |relu d₄ Z4F q - relu d₄ Z4 q| ≤ E4 :=
    fun q => relu_close _ _ _ hE4close q
  have hDLclose : ∀ k, |Z5F k - Z5 k| ≤ DL := fun k =>
    (M.dense_close W₅ b₅ (relu d₄ Z4F) (relu d₄ Z4) E4 E4nn hRelu4 k).trans
      (M.denseErr_le_uniform hw₅ E4nn hW₅ hb₅ hM4 k)
  -- head cotangent (float softmax−onehot within cotErr); real head ∈ [−1,1]
  have hHeadCot : ∀ k, |M.softmaxCECotF fexp Z5F label k -
      (softmax nC Z5 k - oneHot nC label k)| ≤ ecH := fun k =>
    M.softmax_ce_cot_close fexp Z5F Z5 label heexp0 heexp1 hfexp hρ1 hDLclose k
  have hHeadMag : ∀ k, |softmax nC Z5 k - oneHot nC label k| ≤ 1 := by
    intro k
    have hD : 0 < ∑ t, Real.exp (Z5 t) :=
      Finset.sum_pos (fun t _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩
    have hs0 : 0 ≤ softmax nC Z5 k :=
      div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun t _ => (Real.exp_pos _).le)
    have hs1 : softmax nC Z5 k ≤ 1 :=
      (div_le_one hD).mpr
        (Finset.single_le_sum (fun t _ => (Real.exp_pos _).le) (Finset.mem_univ k))
    simp only [oneHot]
    by_cases hkl : k = label
    · rw [if_pos hkl, abs_le]; constructor <;> linarith
    · rw [if_neg hkl, abs_le]; constructor <;> linarith
  -- two masked Wᵀ cotangent steps + the unmasked W₃ step
  have hc4 : ∀ q, |FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j)
        (fun _ => 0) (M.softmaxCECotF fexp Z5F label)) q -
      FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
        (fun k => softmax nC Z5 k - oneHot nC label k)) q| ≤ ec4 := fun q =>
    M.cot_step_close W₅ Z4F Z4 (M.softmaxCECotF fexp Z5F label)
      (fun k => softmax nC Z5 k - oneHot nC label k) hw₅ zero_le_one ecHnn hW₅
      hHeadMag hHeadCot hE4close hmargin4 q
  have hc4Mag : ∀ q, |FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j)
      (fun _ => 0) (fun k => softmax nC Z5 k - oneHot nC label k)) q| ≤ C4 :=
    fun q => (reluMask_abs_le _ _ q).trans
      (dense_abs_le zero_le_one (fun i j => hW₅ j i) (fun _ => by simp) hHeadMag q)
  have hc3 : ∀ l, |FloatModel.reluMask Z3F (M.dense (fun j i' => W₄ i' j)
        (fun _ => 0) (FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j)
          (fun _ => 0) (M.softmaxCECotF fexp Z5F label)))) l -
      FloatModel.reluMask Z3 (dense (fun j i' => W₄ i' j) (fun _ => 0)
        (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
          (fun k => softmax nC Z5 k - oneHot nC label k)))) l| ≤ ec3 := fun l =>
    M.cot_step_close W₄ Z3F Z3
      (FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
        (M.softmaxCECotF fexp Z5F label)))
      (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
        (fun k => softmax nC Z5 k - oneHot nC label k)))
      hw₄ C4nn ec4nn hW₄ hc4Mag hc4 hE3close hmargin3 l
  have hc3Mag : ∀ l, |FloatModel.reluMask Z3 (dense (fun j i' => W₄ i' j)
      (fun _ => 0) (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j)
        (fun _ => 0) (fun k => softmax nC Z5 k - oneHot nC label k)))) l| ≤ C3 :=
    fun l => (reluMask_abs_le _ _ l).trans
      (dense_abs_le C4nn (fun i j => hW₄ j i) (fun _ => by simp) hc4Mag l)
  -- the unmasked W₃ contraction (pool feeds it directly): float vs real
  have hcPool : ∀ j, |M.dense (fun j' i' => W₃ i' j') (fun _ => 0)
        (FloatModel.reluMask Z3F (M.dense (fun j' i' => W₄ i' j') (fun _ => 0)
          (FloatModel.reluMask Z4F (M.dense (fun j' i' => W₅ i' j') (fun _ => 0)
            (M.softmaxCECotF fexp Z5F label))))) j -
      dense (fun j' i' => W₃ i' j') (fun _ => 0)
        (FloatModel.reluMask Z3 (dense (fun j' i' => W₄ i' j') (fun _ => 0)
          (FloatModel.reluMask Z4 (dense (fun j' i' => W₅ i' j') (fun _ => 0)
            (fun k => softmax nC Z5 k - oneHot nC label k))))) j| ≤ ecv := fun j =>
    (M.dense_close (fun j' i' => W₃ i' j') (fun _ => 0) _ _ ec3 ec3nn hc3 j).trans
      (M.denseErr_le_uniform hw₃ ec3nn (fun i j' => hW₃ j' i) (fun _ => by simp)
        hc3Mag j)
  have hcPoolMag : ∀ j, |dense (fun j' i' => W₃ i' j') (fun _ => 0)
      (FloatModel.reluMask Z3 (dense (fun j' i' => W₄ i' j') (fun _ => 0)
        (FloatModel.reluMask Z4 (dense (fun j' i' => W₅ i' j') (fun _ => 0)
          (fun k => softmax nC Z5 k - oneHot nC label k))))) j| ≤ CP :=
    fun j => dense_abs_le C3nn (fun i j' => hW₃ j' i) (fun _ => by simp) hc3Mag j
  -- the conv-output cotangent tensors (float / real), in reluMask form
  set cotF : Tensor3 c (2*h) (2*w) := fun ci hi wi =>
    (if ZCF (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
      (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w)) ZCF))
          ci hi wi
        then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
          (FloatModel.reluMask Z3F (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
            (FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
              (M.softmaxCECotF fexp Z5F label)))))
          (t3Idx ci (winRow hi) (winCol wi))
        else 0) with hcotFdef
  set cotR : Tensor3 c (2*h) (2*w) := fun ci hi wi =>
    (if ZC (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
      (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w)) ZC))
          ci hi wi
        then dense (fun j i' => W₃ i' j) (fun _ => 0)
          (FloatModel.reluMask Z3 (dense (fun j i' => W₄ i' j) (fun _ => 0)
            (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
              (fun k => softmax nC Z5 k - oneHot nC label k)))))
          (t3Idx ci (winRow hi) (winCol wi))
        else 0) with hcotRdef
  -- per-cell cotangent closeness (pool freeze + conv-mask freeze) and magnitude
  have hPostRelu : ∀ ci hi wi,
      |Tensor3.unflatten (relu (c * (2*h) * (2*w)) ZCF) ci hi wi -
        Tensor3.unflatten (relu (c * (2*h) * (2*w)) ZC) ci hi wi| ≤ Econv := by
    intro ci hi wi; rw [unflatten_t3Idx, unflatten_t3Idx]; exact hRelu (t3Idx ci hi wi)
  have hcotcell : ∀ ci hi wi, |cotF ci hi wi - cotR ci hi wi| ≤ ecv := by
    intro ci hi wi
    have hpb := hmarginPool.poolBack_close hPostRelu ci hi wi
      (hcPool (t3Idx ci (winRow hi) (winCol wi)))
    simp only [hcotFdef, hcotRdef]
    exact mask_scalar_close (hEconv (t3Idx ci hi wi)) (hmarginConv (t3Idx ci hi wi))
      hpb ecvnn
  have hcellMagR : ∀ ci hi wi, |cotR ci hi wi| ≤ CP := by
    intro ci hi wi
    simp only [hcotRdef]
    by_cases hz : ZC (t3Idx ci hi wi) > 0
    · rw [if_pos hz, one_mul]
      split_ifs with hA
      · exact hcPoolMag (t3Idx ci (winRow hi) (winCol wi))
      · simpa using CPnn
    · rw [if_neg hz, zero_mul, abs_zero]; exact CPnn
  have hcellMag : ∀ ci hi wi, |cotF ci hi wi| ≤ CP + ecv := by
    intro ci hi wi
    have htri := abs_sub_abs_le_abs_sub (cotF ci hi wi) (cotR ci hi wi)
    linarith [hcellMagR ci hi wi, hcotcell ci hi wi, htri]
  -- assemble: rewrite to the dot form (apply + bridge), then the dot composite
  rw [M.cnnConv2FloatGrad_apply b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label v o cc kh kw,
    cnn_conv2_loss_gradAt_reluMask b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw v
      (fun k => abs_pos.mp (lt_of_le_of_lt Ecnn (hmarginConv k)))
      (hmarginPool.smooth Ecnn)
      (fun l => abs_pos.mp (lt_of_le_of_lt E3nn (hmargin3 l)))
      (fun q => abs_pos.mp (lt_of_le_of_lt E4nn (hmargin4 q)))
      o cc kh kw]
  -- the three dot premises against cotF / cotR
  have hA : ∀ s, |convPadWin kH kW x₁ cc kh kw s| ≤ a := fun s => by
    simp only [convPadWin]; exact abs_convPad_le x₁ ha hx₁ cc kh kw _ _
  have hBt : ∀ s, |cotWin cotF o s| ≤ CP + ecv := fun s => by
    simp only [cotWin]; exact hcellMag o _ _
  have hB : ∀ s, |cotWin cotF o s - cotWin cotR o s| ≤ ecv := fun s => by
    simp only [cotWin]; exact hcotcell o _ _
  simp only [FloatModel.cnnConv2GradBudget]
  exact M.dot_perturbed_close (convPadWin kH kW x₁ cc kh kw) (cotWin cotF o)
    (cotWin cotR o) ha hA hBt hB

-- ════════════════════════════════════════════════════════════════
-- § Drift transport: conv → relu → pool → dense → relu → dense → logits
-- ════════════════════════════════════════════════════════════════

/-- The `ℓ1` mass of a scaled step. -/
theorem smul_l1_mass {n : Nat} (e : Vec n) {t : ℝ} (ht0 : 0 ≤ t) :
    (∑ idx, |(t • e) idx|) = t * ∑ idx, |e idx| := by
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun idx _ => by
    simp [abs_mul, abs_of_nonneg ht0]

/-- A `t`-scaled step stays inside the step radius for `t ∈ [0,1]`. -/
theorem smul_l1_mass_le {n : Nat} (e : Vec n) {t D : ℝ} (ht0 : 0 ≤ t)
    (ht1 : t ≤ 1) (he : (∑ idx, |e idx|) ≤ D) :
    (∑ idx, |(t • e) idx|) ≤ D := by
  rw [smul_l1_mass e ht0]
  calc t * ∑ idx, |e idx|
      ≤ 1 * D := mul_le_mul ht1 he
        (Finset.sum_nonneg fun _ _ => abs_nonneg _) zero_le_one
    _ = D := one_mul D

/-- A dense layer's output moves by at most `w·‖Δinput‖₁` per entry — the
    `ℓ1→ℓ∞` operator bound used at every dense crossing of the chain. -/
theorem dense_input_drift {m n : Nat} (W : Mat m n) (b : Vec n)
    {wb : ℝ} (hW : ∀ i j, |W i j| ≤ wb)
    (u u' : Vec m) (j : Fin n) :
    |dense W b u' j - dense W b u j| ≤ wb * ∑ i, |u' i - u i| := by
  have hdiff : dense W b u' j - dense W b u j =
      ∑ i, (u' i - u i) * W i j := by
    have h2 : (∑ i, u' i * W i j) - (∑ i, u i * W i j) =
        ∑ i, (u' i - u i) * W i j := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => by ring
    show ((∑ i, u' i * W i j) + b j) - ((∑ i, u i * W i j) + b j) = _
    linarith [h2]
  rw [hdiff]
  calc |∑ i, (u' i - u i) * W i j|
      ≤ ∑ i, |(u' i - u i) * W i j| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i, |u' i - u i| * wb :=
        Finset.sum_le_sum fun i _ => by
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_left (hW i j) (abs_nonneg _)
    _ = wb * ∑ i, |u' i - u i| := by
        rw [← Finset.sum_mul]
        ring

/-- Per-entry conv drift, flat-index form of `conv2d_kernel_drift_total`. -/
theorem conv2d_flat_kernel_drift_total {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW))
    (k : Fin (oc * h * w)) :
    |Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b x) k -
      Tensor3.flatten (conv2d (Kernel4.unflatten v) b x) k| ≤
      a * ∑ idx, |e idx| := by
  obtain ⟨p, rfl⟩ := finProdFinEquiv.surjective k
  obtain ⟨pp, wi⟩ := p
  obtain ⟨q, rfl⟩ := finProdFinEquiv.surjective pp
  obtain ⟨o, hi⟩ := q
  rw [show finProdFinEquiv (finProdFinEquiv (o, hi), wi) =
        t3Idx o hi wi from rfl,
    flatten_t3Idx, flatten_t3Idx]
  exact conv2d_kernel_drift_total b x ha hx v e o hi wi

/-- `ℓ1` conv drift, flat-index form of `conv2d_kernel_drift_sum`. -/
theorem conv2d_flat_kernel_drift_sum {ic oc h w kH kW : Nat} (b : Vec oc)
    (x : Tensor3 ic h w) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ c i j, |x c i j| ≤ a) (v e : Vec (oc * ic * kH * kW)) :
    ∑ k, |Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b x) k -
        Tensor3.flatten (conv2d (Kernel4.unflatten v) b x) k| ≤
      ((h * w : ℕ) : ℝ) * (a * ∑ idx, |e idx|) := by
  rw [sum_t3 (fun k : Fin (oc * h * w) =>
    |Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b x) k -
      Tensor3.flatten (conv2d (Kernel4.unflatten v) b x) k|)]
  calc ∑ o : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
        |Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b x)
            (t3Idx o hi wi) -
          Tensor3.flatten (conv2d (Kernel4.unflatten v) b x)
            (t3Idx o hi wi)|
      = ∑ o : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
          |conv2d (Kernel4.unflatten (v + e)) b x o hi wi -
            conv2d (Kernel4.unflatten v) b x o hi wi| := by
        refine Finset.sum_congr rfl fun o _ => Finset.sum_congr rfl
          fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
        rw [flatten_t3Idx, flatten_t3Idx]
    _ ≤ ((h * w : ℕ) : ℝ) * (a * ∑ idx, |e idx|) :=
        conv2d_kernel_drift_sum b x ha hx v e

/-- **Pooled `ℓ1` drift**: kernel perturbation → conv (`ℓ1`, spatial
    multiplicity) → relu (contraction) → pool (contraction). -/
theorem cnn_pool_l1_drift {c h w kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ cc i j, |x₁ cc i j| ≤ a) (v e : Vec (c * c * kH * kW)) :
    ∑ q, |maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b₂ x₁))) q -
        maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))) q| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|) :=
  le_trans (maxPoolFlat_l1_contract _ _)
    (le_trans (Finset.sum_le_sum fun k _ => relu_entry_lipschitz _ _ _ k)
      (conv2d_flat_kernel_drift_sum b₂ x₁ ha hx v e))

/-- Per-entry POST-relu tensor drift — the form the pool margin
    (`MaxPool2MarginQ`) consumes. -/
theorem cnn_postrelu_close {c h w kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ cc i j, |x₁ cc i j| ≤ a) (v e : Vec (c * c * kH * kW))
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b₂ x₁))) :
          Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))) :
          Tensor3 c (2*h) (2*w)) ci hi wi| ≤
      a * ∑ idx, |e idx| := by
  rw [unflatten_t3Idx, unflatten_t3Idx]
  exact le_trans (relu_entry_lipschitz _ _ _ _)
    (conv2d_flat_kernel_drift_total b₂ x₁ ha hx v e _)

/-- Per-entry drift of the relu₃ pre-activation. -/
theorem cnn_z3_drift {c h w d₃ kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {a w₃ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (v e : Vec (c * c * kH * kW)) (l : Fin d₃) :
    |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten (v + e)) b₂ x₁)))) l -
      dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l| ≤
      w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|)) :=
  le_trans (dense_input_drift W₃ b₃ hW₃ _ _ l)
    (mul_le_mul_of_nonneg_left (cnn_pool_l1_drift b₂ x₁ ha hx v e) hw₃)

/-- Per-entry drift of the relu₄ pre-activation. -/
theorem cnn_z4_drift {c h w d₃ d₄ kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {a w₃ w₄ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (v e : Vec (c * c * kH * kW)) (q : Fin d₄) :
    |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (v + e)) b₂ x₁)))))) q -
      dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q| ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * ∑ idx, |e idx|)))) := by
  refine le_trans (dense_input_drift W₄ b₄ hW₄ _ _ q)
    (mul_le_mul_of_nonneg_left ?_ hw₄)
  calc ∑ l, |relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (v + e)) b₂ x₁))))) l -
        relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten v) b₂ x₁))))) l|
      ≤ ∑ l, |dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten (v + e)) b₂ x₁)))) l -
          dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten v) b₂ x₁)))) l| :=
        Finset.sum_le_sum fun l _ => relu_entry_lipschitz _ _ _ l
    _ ≤ ∑ _l : Fin d₃, w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * ∑ idx, |e idx|)) :=
        Finset.sum_le_sum fun l _ =>
          cnn_z3_drift b₂ x₁ W₃ b₃ ha hx hw₃ hW₃ v e l
    _ = (d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * ∑ idx, |e idx|))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

/-- **Logit drift through the whole conv2 chain**: kernel perturbation →
    conv → relu → pool → d₃ → relu → d₄ → relu → d₅. Each dense crossing
    contributes its `ℓ1→ℓ1` operator factor `dᵢ·wᵢ`; the conv contributes
    the weight-sharing multiplicity `(2h)·(2w)`. -/
theorem cnn_conv2_logit_drift {c h w d₃ d₄ nC kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    {a w₃ w₄ w₅ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (v e : Vec (c * c * kH * kW)) (k : Fin nC) :
    |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (v + e)) b₂ x₁)))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten v) b₂ x₁)))))))) k| ≤
      w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|)))))) := by
  refine le_trans (dense_input_drift W₅ b₅ hW₅ _ _ k)
    (mul_le_mul_of_nonneg_left ?_ hw₅)
  calc ∑ q, |relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (v + e)) b₂ x₁))))))) q -
        relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten v) b₂ x₁))))))) q|
      ≤ ∑ q, |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten (v + e)) b₂ x₁)))))) q -
          dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q| :=
        Finset.sum_le_sum fun q _ => relu_entry_lipschitz _ _ _ q
    _ ≤ ∑ _q : Fin d₄, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * ∑ idx, |e idx|)))) :=
        Finset.sum_le_sum fun q _ =>
          cnn_z4_drift b₂ x₁ W₃ b₃ W₄ b₄ ha hx hw₃ hW₃ hw₄ hW₄ v e q
    _ = (d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * ∑ idx, |e idx|))))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

-- ════════════════════════════════════════════════════════════════
-- § The margins freeze every routing decision along the segment
-- ════════════════════════════════════════════════════════════════

/-- The relu₂ margin keeps the conv pre-activation off the kink, same
    sign, along the whole step segment. -/
theorem cnn_margin2_keeps_offkink {c h w kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) {a D : ℝ} (ha : 0 ≤ a)
    (hx : ∀ cc i j, |x₁ cc i j| ≤ a) (v e : Vec (c * c * kH * kW))
    (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ k, a * D <
      |Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (k : Fin (c * (2*h) * (2*w))) :
    Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • e)) b₂ x₁) k ≠ 0 ∧
      (0 < Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • e)) b₂ x₁) k
        ↔ 0 < Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k) := by
  refine sign_stable_of_close ?_ (hm k)
  have h1 := conv2d_flat_kernel_drift_total b₂ x₁ ha hx v (t • e) k
  have h2 : a * (∑ idx, |(t • e) idx|) ≤ a * D :=
    mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) ha
  linarith

/-- The POST-relu tensor stays within the pool margin radius `a·D` along
    the whole step segment — what `MaxPool2MarginQ.{smooth_of_close,
    isArgmax_iff, pdiv3_eq}` consume. -/
theorem cnn_postrelu_close_seg {c h w kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) {a D : ℝ} (ha : 0 ≤ a)
    (hx : ∀ cc i j, |x₁ cc i j| ≤ a) (v e : Vec (c * c * kH * kW))
    (he : (∑ idx, |e idx|) ≤ D)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • e)) b₂ x₁))) :
          Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))) :
          Tensor3 c (2*h) (2*w)) ci hi wi| ≤ a * D :=
  le_trans (cnn_postrelu_close b₂ x₁ ha hx v (t • e) ci hi wi)
    (mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) ha)

/-- The relu₃ margin keeps the first head pre-activation off the kink,
    same sign, along the whole step segment. -/
theorem cnn_margin3_keeps_offkink {c h w d₃ kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {a w₃ D : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (v e : Vec (c * c * kH * kW)) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * D)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (l : Fin d₃) :
    dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • e)) b₂ x₁))))
        l ≠ 0 ∧
      (0 < dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • e)) b₂ x₁))))
          l ↔
        0 < dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l) := by
  refine sign_stable_of_close ?_ (hm l)
  have h1 := cnn_z3_drift b₂ x₁ W₃ b₃ ha hx hw₃ hW₃ v (t • e) l
  have h2 : w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |(t • e) idx|)) ≤
      w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * D)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) ha)
      (Nat.cast_nonneg _)) hw₃
  linarith

/-- The relu₄ margin keeps the second head pre-activation off the kink,
    same sign, along the whole step segment. -/
theorem cnn_margin4_keeps_offkink {c h w d₃ d₄ kH kW : Nat} (b₂ : Vec c)
    (x₁ : Tensor3 c (2*h) (2*w)) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {a w₃ w₄ D : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (v e : Vec (c * c * kH * kW)) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * D)))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (q : Fin d₄) :
    dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (v + t • e)) b₂ x₁)))))) q ≠ 0 ∧
      (0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (v + t • e)) b₂ x₁)))))) q ↔
        0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q) := by
  refine sign_stable_of_close ?_ (hm q)
  have h1 := cnn_z4_drift b₂ x₁ W₃ b₃ W₄ b₄ ha hx hw₃ hW₃ hw₄ hW₄
    v (t • e) q
  have h2 : w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
      (a * ∑ idx, |(t • e) idx|)))) ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) ha)
        (Nat.cast_nonneg _)) hw₃) (Nat.cast_nonneg _)) hw₄
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The head-gradient drift under frozen masks
-- ════════════════════════════════════════════════════════════════

/-- **Frozen-mask head-gradient drift**: with the two head masks frozen
    (0/1-valued, shared between the two points) and the softmax drifting
    by at most `Δ`, the head3 gradient closed form drifts by at most
    `d₃·w₃·d₄·w₄·nC·w₅·Δ` — the oneHot cancels in the difference. -/
theorem head3_sum_drift {p d₃ d₄ nC : Nat} (W₃ : Mat p d₃)
    (W₄ : Mat d₃ d₄) (W₅ : Mat d₄ nC) {w₃ w₄ w₅ Δ : ℝ}
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (m₃ : Fin d₃ → ℝ) (hm₃ : ∀ l, |m₃ l| ≤ 1)
    (m₄ : Fin d₄ → ℝ) (hm₄ : ∀ r, |m₄ r| ≤ 1)
    (s s' oh : Vec nC) (hs : ∀ k, |s' k - s k| ≤ Δ) (q : Fin p) :
    |(∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s' k - oh k)))) -
      ∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s k - oh k)))| ≤
      (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * Δ))))) := by
  have hcoll : (∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s' k - oh k)))) -
      (∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s k - oh k)))) =
      ∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s' k - s k))) := by
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun l _ => ?_
    rw [← mul_sub, ← mul_sub, ← Finset.sum_sub_distrib]
    congr 2
    refine Finset.sum_congr rfl fun r _ => ?_
    rw [← mul_sub, ← mul_sub, ← Finset.sum_sub_distrib]
    congr 2
    exact Finset.sum_congr rfl fun k _ => by ring
  rw [hcoll]
  have hinner : ∀ r, |∑ k, W₅ r k * (s' k - s k)| ≤
      (nC : ℝ) * (w₅ * Δ) := by
    intro r
    calc |∑ k, W₅ r k * (s' k - s k)|
        ≤ ∑ k, |W₅ r k * (s' k - s k)| := Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _k : Fin nC, w₅ * Δ :=
          Finset.sum_le_sum fun k _ => by
            rw [abs_mul]
            exact mul_le_mul (hW₅ r k) (hs k) (abs_nonneg _) hw₅
      _ = (nC : ℝ) * (w₅ * Δ) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul]
  have hmid : ∀ l, |∑ r, W₄ l r *
      (m₄ r * ∑ k, W₅ r k * (s' k - s k))| ≤
      (d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * Δ))) := by
    intro l
    calc |∑ r, W₄ l r * (m₄ r * ∑ k, W₅ r k * (s' k - s k))|
        ≤ ∑ r, |W₄ l r * (m₄ r * ∑ k, W₅ r k * (s' k - s k))| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ _r : Fin d₄, w₄ * ((nC : ℝ) * (w₅ * Δ)) := by
          refine Finset.sum_le_sum fun r _ => ?_
          rw [abs_mul]
          refine mul_le_mul (hW₄ l r) ?_ (abs_nonneg _) hw₄
          rw [abs_mul]
          exact le_trans (mul_le_of_le_one_left (abs_nonneg _) (hm₄ r))
            (hinner r)
      _ = (d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * Δ))) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul]
  calc |∑ l, W₃ q l * (m₃ l * ∑ r, W₄ l r *
        (m₄ r * ∑ k, W₅ r k * (s' k - s k)))|
      ≤ ∑ l, |W₃ q l * (m₃ l * ∑ r, W₄ l r *
          (m₄ r * ∑ k, W₅ r k * (s' k - s k)))| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _l : Fin d₃, w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * Δ)))) := by
        refine Finset.sum_le_sum fun l _ => ?_
        rw [abs_mul]
        refine mul_le_mul (hW₃ q l) ?_ (abs_nonneg _) hw₃
        rw [abs_mul]
        exact le_trans (mul_le_of_le_one_left (abs_nonneg _) (hm₃ l))
          (hmid l)
    _ = (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) * (w₅ * Δ))))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

-- ════════════════════════════════════════════════════════════════
-- § Segment-Lipschitz gradient for the conv2 loss, explicit constant
-- ════════════════════════════════════════════════════════════════

/-- **Segment-Lipschitz gradient for the conv2-kernel loss, explicit
    constant.** Under the four margins at step radius `D` — relu₂
    (`a·D`), pool selection (`MaxPool2MarginQ (a·D)` of the POST-relu
    tensor), relu₃ (`w₃·4hw·a·D`), relu₄ (`w₄·d₃·w₃·4hw·a·D`) — every
    routing decision (masks AND pool argmaxes) freezes along `[v, v+d]`,
    the point-free conv Jacobian factors out, and the difference
    collapses to the softmax drift exactly as in
    `mlp_input_loss_grad_lipschitz`. The conv-layer peer of that
    theorem; the constant picks up the weight-sharing multiplicity
    `((2h)·(2w))²`. -/
theorem cnn_conv2_loss_grad_lipschitz {c h w d₃ d₄ nC kH kW : Nat}
    (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (hh : 0 < h) (hw : 0 < w)
    {a w₃ w₄ w₅ D : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (v d : Vec (c * c * kH * kW)) (hd : (∑ idx, |d idx|) ≤ D)
    (hm2 : ∀ k, a * D <
      |Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k|)
    (hmq : MaxPool2MarginQ (a * D) (Tensor3.unflatten
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * D)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * D)))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))) < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1)
    (idx : Fin (c * c * kH * kW)) :
    |gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label)
        (v + t • d) idx -
      gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label)
        v idx| ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))) * (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D :=
    le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  have haD0 : 0 ≤ a * D := mul_nonneg ha hD0
  have hδ0 : (0:ℝ) ≤ w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))) :=
    mul_nonneg hw₅ (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
        (mul_nonneg (Nat.cast_nonneg _) haD0)))))
  have hden : (0:ℝ) < 1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))) := by linarith
  obtain ⟨p1, rfl⟩ := finProdFinEquiv.surjective idx
  obtain ⟨p2, kw⟩ := p1
  obtain ⟨p3, rfl⟩ := finProdFinEquiv.surjective p2
  obtain ⟨p4, kh⟩ := p3
  obtain ⟨p5, rfl⟩ := finProdFinEquiv.surjective p4
  obtain ⟨o, cc⟩ := p5
  rw [show finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o, cc), kh),
        kw) = k4Idx o cc kh kw from rfl]
  -- base-point conditions from the margins
  have hz2_v : ∀ k,
      Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁) k ≠ 0 :=
    fun k h0 => by
      have hk := hm2 k
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr haD0)
  have hmp_v : MaxPool2Smooth (Tensor3.unflatten
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))) :
      Tensor3 c (2*h) (2*w)) := hmq.smooth haD0
  have hz3_v : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l ≠ 0 :=
    fun l h0 => by
      have hk := hm3 l
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr (mul_nonneg hw₃
        (mul_nonneg (Nat.cast_nonneg _) haD0)))
  have hz4_v : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q ≠ 0 :=
    fun q h0 => by
      have hk := hm4 q
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr (mul_nonneg hw₄
        (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
          (mul_nonneg (Nat.cast_nonneg _) haD0)))))
  -- segment-point conditions: everything frozen
  have hstab2 := fun k =>
    cnn_margin2_keeps_offkink b₂ x₁ ha hx v d hd hm2 t ht0 ht1 k
  have hz2_t : ∀ k, Tensor3.flatten
      (conv2d (Kernel4.unflatten (v + t • d)) b₂ x₁) k ≠ 0 :=
    fun k => (hstab2 k).1
  have hclose := fun ci hi wi =>
    cnn_postrelu_close_seg b₂ x₁ ha hx v d hd t ht0 ht1 ci hi wi
  have hmp_t : MaxPool2Smooth (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten (v + t • d)) b₂ x₁))) :
      Tensor3 c (2*h) (2*w)) := hmq.smooth_of_close hclose
  have hstab3 := fun l =>
    cnn_margin3_keeps_offkink b₂ x₁ W₃ b₃ ha hx hw₃ hW₃ v d hd hm3
      t ht0 ht1 l
  have hz3_t : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten (v + t • d)) b₂ x₁)))) l ≠ 0 :=
    fun l => (hstab3 l).1
  have hstab4 := fun q =>
    cnn_margin4_keeps_offkink b₂ x₁ W₃ b₃ W₄ b₄ ha hx hw₃ hW₃ hw₄ hW₄
      v d hd hm4 t ht0 ht1 q
  have hz4_t : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten (v + t • d)) b₂ x₁)))))) q ≠ 0 :=
    fun q => (hstab4 q).1
  -- both gradients in closed form
  rw [cnn_conv2_loss_gradAt b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw
      (v + t • d) hz2_t hmp_t hz3_t hz4_t o cc kh kw,
    cnn_conv2_loss_gradAt b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw
      v hz2_v hmp_v hz3_v hz4_v o cc kh kw]
  -- the frozen masks and the frozen routing
  have hmask2 : ∀ (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)),
      (if Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • d)) b₂ x₁)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) =
      (if Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) := by
    intro ci hi wi
    by_cases hp : Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)
        (t3Idx ci hi wi) > 0
    · rw [if_pos ((hstab2 _).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab2 _).2.mp hgt)), if_neg hp]
  have hmask3 : ∀ l : Fin d₃,
      (if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • d)) b₂ x₁))))
          l > 0 then (1:ℝ) else 0) =
      (if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
          l > 0 then (1:ℝ) else 0) := by
    intro l
    by_cases hp : dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁)))) l > 0
    · rw [if_pos ((hstab3 l).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab3 l).2.mp hgt)), if_neg hp]
  have hmask4 : ∀ q : Fin d₄,
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (v + t • d)) b₂ x₁)))))) q > 0
        then (1:ℝ) else 0) =
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q > 0
        then (1:ℝ) else 0) := by
    intro q
    by_cases hp : dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q > 0
    · rw [if_pos ((hstab4 q).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab4 q).2.mp hgt)), if_neg hp]
  have hargiff : ∀ (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)),
      MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • d)) b₂ x₁))))
        ci hi wi ↔
      MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))
        ci hi wi :=
    fun ci hi wi => hmq.isArgmax_iff hclose ci hi wi
  -- the softmax drift along the segment
  have hzdrift : ∀ k, |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • d))
          b₂ x₁)))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten v) b₂ x₁)))))))) k| ≤
      t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))) := by
    intro k
    have h1 := cnn_conv2_logit_drift b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ ha hx
      hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ v (t • d) k
    rw [smul_l1_mass d ht0] at h1
    have h2 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * (t * ∑ idx, |d idx|))))))) =
        t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |d idx|))))))) := by
      ring
    rw [h2] at h1
    have h3 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |d idx|)))))) ≤
        w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left hd ha) (Nat.cast_nonneg _)) hw₃)
          (Nat.cast_nonneg _)) hw₄) (Nat.cast_nonneg _)) hw₅
    have h4 := mul_le_mul_of_nonneg_left h3 ht0
    linarith
  have hδlt : 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) < 1 := by
    nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
  have hexp := FloatModel.exp_sub_one_le hδlt
  have hmono : 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) /
        (1 - 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))) ≤
      2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) := by
    refine div_le_div_of_nonneg_left
      (by nlinarith [mul_nonneg ht0 hδ0]) hden ?_
    nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
  have hS : ∀ k, |softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • d))
          b₂ x₁))))))))) k -
      softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten v) b₂ x₁))))))))) k| ≤
      2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) :=
    fun k => le_trans (FloatModel.softmax_perturb _ _ hzdrift k)
      (le_trans hexp hmono)
  have hΔ0 : (0:ℝ) ≤ 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) /
      (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) :=
    div_nonneg (mul_nonneg (by norm_num) (mul_nonneg ht0 hδ0)) hden.le
  have hM0 : (0:ℝ) ≤ (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
      (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))))))) :=
    mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
        (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₅ hΔ0)))))
  -- the conv Jacobian row mass
  have hcp : ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
      |if ci = o then convPad kH kW x₁ cc kh kw hi wi else 0| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * a := by
    rw [Finset.sum_eq_single o
      (fun ci _ hne => by
        rw [Finset.sum_eq_zero]
        intro hi _
        rw [Finset.sum_eq_zero]
        intro wi _
        rw [if_neg hne, abs_zero])
      (fun habs => absurd (Finset.mem_univ _) habs)]
    calc ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          |if o = o then convPad kH kW x₁ cc kh kw hi wi else 0|
        ≤ ∑ _hi : Fin (2*h), ∑ _wi : Fin (2*w), a := by
          refine Finset.sum_le_sum fun hi _ =>
            Finset.sum_le_sum fun wi _ => ?_
          rw [if_pos rfl]
          exact abs_convPad_le x₁ ha hx cc kh kw hi wi
      _ = ((2*h * (2*w) : ℕ) : ℝ) * a := by
          rw [Finset.sum_const, Finset.sum_const, Finset.card_univ,
            Finset.card_univ, Fintype.card_fin, Fintype.card_fin,
            smul_smul, nsmul_eq_mul]
  -- the endgame: combine, freeze, collapse to the softmax drift
  have hfinal : ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
      (|if ci = o then convPad kH kW x₁ cc kh kw hi wi else 0| *
        ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
          (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
            (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))))))))) ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))) * (t * D) := by
    calc ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
        (|if ci = o then convPad kH kW x₁ cc kh kw hi wi else 0| *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))))))))))
        = (∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
            |if ci = o then convPad kH kW x₁ cc kh kw hi wi else 0|) *
            ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
              (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) /
                (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                  (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))))))))) := by
          simp only [← Finset.sum_mul]
      _ ≤ (((2*h * (2*w) : ℕ) : ℝ) * a) *
            ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
              (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))) /
                (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                  (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))))))))) :=
          mul_le_mul_of_nonneg_right hcp hM0
      _ = (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
            (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))) * (t * D) := by
          ring
  refine le_trans (le_trans (by
    rw [← Finset.sum_sub_distrib]
    refine le_trans (le_of_eq (congrArg abs (Finset.sum_congr rfl
      fun ci _ => by rw [← Finset.sum_sub_distrib]))) ?_
    refine le_trans (le_of_eq (congrArg abs (Finset.sum_congr rfl
      fun ci _ => Finset.sum_congr rfl fun hi _ => by
        rw [← Finset.sum_sub_distrib]))) ?_
    refine le_trans (Finset.abs_sum_le_sum_abs _ _) ?_
    exact Finset.sum_le_sum fun ci _ => le_trans
      (Finset.abs_sum_le_sum_abs _ _)
      (Finset.sum_le_sum fun hi _ => Finset.abs_sum_le_sum_abs _ _))
    (Finset.sum_le_sum fun ci _ => Finset.sum_le_sum fun hi _ =>
      Finset.sum_le_sum fun wi _ => ?_)) hfinal
  -- per-term: freeze the masks and the route, collapse to the drift
  rw [hmask2 ci hi wi]
  simp only [hmask3, hmask4]
  by_cases hA : MaxPool2IsArgmax (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten v) b₂ x₁)))) ci hi wi
  · rw [if_pos ((hargiff ci hi wi).mpr hA), if_pos hA, ← mul_sub,
      abs_mul, ← mul_sub, abs_mul]
    refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
    refine le_trans (mul_le_of_le_one_left (abs_nonneg _) ?_) ?_
    · split_ifs <;> simp
    · exact head3_sum_drift W₃ W₄ W₅ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
        (fun l => if dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten v) b₂ x₁)))) l > 0
          then (1:ℝ) else 0)
        (fun l => by split_ifs <;> simp)
        (fun q => if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten v) b₂ x₁)))))) q > 0
          then (1:ℝ) else 0)
        (fun q => by split_ifs <;> simp)
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v) b₂ x₁))))))))))
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten (v + t • d))
              b₂ x₁))))))))))
        (oneHot nC label) hS (t3Idx ci (winRow hi) (winCol wi))
  · rw [if_neg (fun hA' => hA ((hargiff ci hi wi).mp hA')), if_neg hA]
    simp only [mul_zero, sub_self, abs_zero]
    exact mul_nonneg (abs_nonneg _) hM0

-- ════════════════════════════════════════════════════════════════
-- § The conv2 capstone: one inexact SGD step provably descends
-- ════════════════════════════════════════════════════════════════

/-- **One inexact SGD step on the CNN's second conv kernel provably
    decreases the cross-entropy loss.** All of `sgd_descends`'
    hypotheses discharged for the loss-of-conv2-kernel map:
    differentiability along the segment and the segment-Lipschitz
    constant both come from the FOUR margin hypotheses at the step
    radius `D = lr·(‖∇L‖₁ + |kernel|·η)` — relu₂, the pool-selection
    margin (POST-relu), relu₃, relu₄ — which freeze every mask and the
    pool's entire routing pattern along the step. Remaining hypotheses
    are checkable arithmetic: the oracle accuracy `η`, the margins, the
    small-step condition, and the two dominance conditions. Conclusion:
    the loss drops by ≥ `lr·‖∇L‖₂²/2`. The conv-layer peer of
    `mlp_input_sgd_descends`; the descent program now reaches through
    weight sharing and max-pooling. -/
theorem cnn_conv2_sgd_descends {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (gh : Vec (c * c * kH * kW))
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    {lr η a w₃ w₄ w₅ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx -
      gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label) (Kernel4.flatten W₂) idx| ≤ η)
    (hm2 : ∀ k, a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) * η)) <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmq : MaxPool2MarginQ (a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) * η)))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx,
        |gradAt (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) * η)))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * (lr * ((∑ idx, |gradAt
          (fun v' : Vec (c * c * kH * kW) =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten
                  (conv2d (Kernel4.unflatten v') b₂ x₁))))))))) label)
            (Kernel4.flatten W₂) idx|) +
          ((c * c * kH * kW : ℕ) : ℝ) * η)))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) * η))))))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) ≤
      lr * (∑ idx, gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt
            (fun v' : Vec (c * c * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten
                    (conv2d (Kernel4.unflatten v') b₂ x₁))))))))) label)
              (Kernel4.flatten W₂) idx|) +
            ((c * c * kH * kW : ℕ) : ℝ) * η))))))))))) *
        (lr * ((∑ idx, |gradAt
          (fun v' : Vec (c * c * kH * kW) =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten
                  (conv2d (Kernel4.unflatten v') b₂ x₁))))))))) label)
            (Kernel4.flatten W₂) idx|) +
          ((c * c * kH * kW : ℕ) : ℝ) * η)) ^ 2 ≤
      lr * (∑ idx, gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx ^ 2) / 4) :
    crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d
            (Kernel4.unflatten (Kernel4.flatten W₂ - lr • gh))
            b₂ x₁))))))))) label ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d
            (Kernel4.unflatten (Kernel4.flatten W₂)) b₂ x₁)))))))))
          label -
        lr * (∑ idx, gradAt
          (fun v' : Vec (c * c * kH * kW) =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten
                  (conv2d (Kernel4.unflatten v') b₂ x₁))))))))) label)
          (Kernel4.flatten W₂) idx ^ 2) / 2 := by
  set f : Vec (c * c * kH * kW) → ℝ :=
    fun v' : Vec (c * c * kH * kW) =>
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
        label with hf
  have hden : (0:ℝ) < 1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx,
        |gradAt f (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) * η))))))))) := by
    linarith
  have hC0 : (0:ℝ) ≤ 2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 *
      (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
      (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx,
          |gradAt f (Kernel4.flatten W₂) idx|) +
          ((c * c * kH * kW : ℕ) : ℝ) * η)))))))))) :=
    div_nonneg (by positivity) hden.le
  -- the margins, restated at the `unflatten ∘ flatten` parameter point
  have hm2' : ∀ k, a * (lr * ((∑ idx,
      |gradAt f (Kernel4.flatten W₂) idx|) +
      ((c * c * kH * kW : ℕ) : ℝ) * η)) <
      |Tensor3.flatten (conv2d (Kernel4.unflatten (Kernel4.flatten W₂))
        b₂ x₁) k| := fun k => by
    rw [Kernel4.unflatten_flatten]
    exact hm2 k
  have hmq' : MaxPool2MarginQ (a * (lr * ((∑ idx,
      |gradAt f (Kernel4.flatten W₂) idx|) +
      ((c * c * kH * kW : ℕ) : ℝ) * η)))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten (Kernel4.flatten W₂))
          b₂ x₁)))) := by
    rw [Kernel4.unflatten_flatten]
    exact hmq
  have hm3' : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx,
      |gradAt f (Kernel4.flatten W₂) idx|) +
      ((c * c * kH * kW : ℕ) : ℝ) * η)))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten (Kernel4.flatten W₂))
          b₂ x₁)))) l| := fun l => by
    rw [Kernel4.unflatten_flatten]
    exact hm3 l
  have hm4' : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
      (a * (lr * ((∑ idx, |gradAt f (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) * η)))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (Kernel4.flatten W₂)) b₂ x₁))))))
        q| := fun q => by
    rw [Kernel4.unflatten_flatten]
    exact hm4 q
  -- ℓ1 radius of the step
  have hD : (∑ idx, |(-(lr • gh)) idx|) ≤
      lr * ((∑ idx, |gradAt f (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) * η) := by
    calc (∑ idx, |(-(lr • gh)) idx|) = ∑ idx, lr * |gh idx| := by
          refine Finset.sum_congr rfl fun idx _ => ?_
          simp [abs_mul, abs_of_nonneg hlr]
      _ ≤ ∑ idx, lr * (|gradAt f (Kernel4.flatten W₂) idx| + η) := by
          refine Finset.sum_le_sum fun idx _ => ?_
          refine mul_le_mul_of_nonneg_left ?_ hlr
          have h3 : |gh idx| ≤
              |gh idx - gradAt f (Kernel4.flatten W₂) idx| +
              |gradAt f (Kernel4.flatten W₂) idx| := by
            simpa using abs_sub_le (gh idx)
              (gradAt f (Kernel4.flatten W₂) idx) 0
          linarith [hgh idx]
      _ = lr * ((∑ idx, |gradAt f (Kernel4.flatten W₂) idx|) +
            ((c * c * kH * kW : ℕ) : ℝ) * η) := by
          rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const,
            Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hmain := sgd_descends f (Kernel4.flatten W₂) gh hlr hη hC0 hgh
    (fun t ht => cnn_conv2_loss_differentiableAt b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅
      label hc hh hw _
      (fun k => (cnn_margin2_keeps_offkink b₂ x₁ ha hx
        (Kernel4.flatten W₂) (-(lr • gh)) hD hm2' t ht.1 ht.2 k).1)
      (hmq'.smooth_of_close (fun ci hi wi => cnn_postrelu_close_seg b₂ x₁
        ha hx (Kernel4.flatten W₂) (-(lr • gh)) hD t ht.1 ht.2 ci hi wi))
      (fun l => (cnn_margin3_keeps_offkink b₂ x₁ W₃ b₃ ha hx hw₃ hW₃
        (Kernel4.flatten W₂) (-(lr • gh)) hD hm3' t ht.1 ht.2 l).1)
      (fun q => (cnn_margin4_keeps_offkink b₂ x₁ W₃ b₃ W₄ b₄ ha hx hw₃ hW₃
        hw₄ hW₄ (Kernel4.flatten W₂) (-(lr • gh)) hD hm4' t ht.1 ht.2 q).1))
    (fun t ht idx => by
      have h := cnn_conv2_loss_grad_lipschitz b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅
        label hh hw ha hx hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ (Kernel4.flatten W₂)
        (-(lr • gh)) hD hm2' hmq' hm3' hm4' hsmall t ht idx
      simpa [hf] using h)
    h1 h2
  simpa [hf] using hmain

open FloatModel in
/-- **One binary32 SGD step on the CNN's second conv kernel provably decreases
    the cross-entropy loss — with NO abstract gradient-accuracy parameter**
    (Increment 3, the conv-2 rung capstone). The conv peer of
    `mlp_input_float_sgd_descends`: the gradient is the *actual* binary32 `W₂`
    gradient `M.cnnConv2FloatGrad …`, and its accuracy is *proven* by
    `cnn_conv2_grad_close` (η := `cnnConv2GradBudget`, discharged per kernel
    entry via `k4Idx_surj`), not assumed. The two rounding-margin families are
    carried as hypotheses (the honest first cut): the per-layer ROUND margins
    (`hmarginConv/Pool/3/4`, feeding the grad-close) and the gradient-radius
    STEP margins + `hsmall`/`h1`/`h2` (feeding `cnn_conv2_sgd_descends`'s
    drift-freeze and the descent geometry). The conv-2 input `x₁` is exact. -/
theorem cnn_conv2_float_sgd_descends {c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    {lr a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (ha : 0 ≤ a) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂) (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃)
    (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅) (hβ₅ : 0 ≤ β₅) (hlr : 0 ≤ lr)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx : ∀ cc i j, |x₁ cc i j| ≤ a)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmarginConv : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0 <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hm2 : ∀ k, a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) *
          M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
            eexp)) <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmq : MaxPool2MarginQ (a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) *
          M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
            eexp)))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx,
        |gradAt (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) *
          M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
            eexp)))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * (lr * ((∑ idx, |gradAt
          (fun v' : Vec (c * c * kH * kW) =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten
                  (conv2d (Kernel4.unflatten v') b₂ x₁))))))))) label)
            (Kernel4.flatten W₂) idx|) +
          ((c * c * kH * kW : ℕ) : ℝ) *
            M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
              eexp)))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) +
        ((c * c * kH * kW : ℕ) : ℝ) *
          M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
            eexp))))))))) < 1)
    (h1 : lr * (M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
          eexp) * (∑ idx, |gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx|) ≤
      lr * (∑ idx, gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt
            (fun v' : Vec (c * c * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten
                    (conv2d (Kernel4.unflatten v') b₂ x₁))))))))) label)
              (Kernel4.flatten W₂) idx|) +
            ((c * c * kH * kW : ℕ) : ℝ) *
              M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅
                β₅ eexp))))))))))) *
        (lr * ((∑ idx, |gradAt
          (fun v' : Vec (c * c * kH * kW) =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten
                  (conv2d (Kernel4.unflatten v') b₂ x₁))))))))) label)
            (Kernel4.flatten W₂) idx|) +
          ((c * c * kH * kW : ℕ) : ℝ) *
            M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
              eexp)) ^ 2 ≤
      lr * (∑ idx, gradAt
        (fun v' : Vec (c * c * kH * kW) =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
            label) (Kernel4.flatten W₂) idx ^ 2) / 4) :
    crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d
            (Kernel4.unflatten (Kernel4.flatten W₂ -
              lr • M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
                (Kernel4.flatten W₂)))
            b₂ x₁))))))))) label ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d
            (Kernel4.unflatten (Kernel4.flatten W₂)) b₂ x₁)))))))))
          label -
        lr * (∑ idx, gradAt
          (fun v' : Vec (c * c * kH * kW) =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten
                  (conv2d (Kernel4.unflatten v') b₂ x₁))))))))) label)
          (Kernel4.flatten W₂) idx ^ 2) / 2 := by
  have hu := M.u_nonneg
  -- nonnegativity of the proven budget
  have A2nn : 0 ≤ FloatModel.layerAct (c * kH * kW) w₂ β₂ a :=
    layerAct_nonneg hw₂ hβ₂ ha
  have A3nn : 0 ≤ FloatModel.layerAct (c * h * w) w₃ β₃
      (FloatModel.layerAct (c * kH * kW) w₂ β₂ a) := layerAct_nonneg hw₃ hβ₃ A2nn
  have A4nn : 0 ≤ FloatModel.layerAct d₃ w₄ β₄
      (FloatModel.layerAct (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)) := layerAct_nonneg hw₄ hβ₄ A3nn
  have E4nn : 0 ≤ FloatModel.layerBudget M.u d₃ w₄ β₄
      (FloatModel.layerAct (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a))
      (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)) :=
    layerBudget_nonneg hu hw₄ hβ₄ A3nn
      (layerBudget_nonneg hu hw₃ hβ₃ A2nn
        (layerBudget_nonneg hu hw₂ hβ₂ ha le_rfl))
  have DLnn : 0 ≤ FloatModel.layerBudget M.u d₄ w₅ β₅
      (FloatModel.layerAct d₃ w₄ β₄ (FloatModel.layerAct (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)))
      (FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0))) :=
    layerBudget_nonneg hu hw₅ hβ₅ A4nn E4nn
  have C4nn : 0 ≤ FloatModel.layerAct nC w₅ 0 1 :=
    layerAct_nonneg hw₅ le_rfl zero_le_one
  have C3nn : 0 ≤ FloatModel.layerAct d₄ w₄ 0 (FloatModel.layerAct nC w₅ 0 1) :=
    layerAct_nonneg hw₄ le_rfl C4nn
  have CPnn : 0 ≤ FloatModel.layerAct d₃ w₃ 0
      (FloatModel.layerAct d₄ w₄ 0 (FloatModel.layerAct nC w₅ 0 1)) :=
    layerAct_nonneg hw₃ le_rfl C3nn
  have ecHnn : 0 ≤ FloatModel.cotErr M.u eexp
      (FloatModel.layerBudget M.u d₄ w₅ β₅
        (FloatModel.layerAct d₃ w₄ β₄ (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)))
        (FloatModel.layerBudget M.u d₃ w₄ β₄
          (FloatModel.layerAct (c * h * w) w₃ β₃
            (FloatModel.layerAct (c * kH * kW) w₂ β₂ a))
          (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
            (FloatModel.layerAct (c * kH * kW) w₂ β₂ a)
            (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ a 0)))) nC :=
    M.cotErr_nonneg heexp0 DLnn hρ1
  have ec4nn : 0 ≤ FloatModel.layerBudget M.u nC w₅ 0 1 _ :=
    layerBudget_nonneg hu hw₅ le_rfl zero_le_one ecHnn
  have ec3nn : 0 ≤ FloatModel.layerBudget M.u d₄ w₄ 0
      (FloatModel.layerAct nC w₅ 0 1) _ :=
    layerBudget_nonneg hu hw₄ le_rfl C4nn ec4nn
  have ecvnn : 0 ≤ FloatModel.layerBudget M.u d₃ w₃ 0
      (FloatModel.layerAct d₄ w₄ 0 (FloatModel.layerAct nC w₅ 0 1)) _ :=
    layerBudget_nonneg hu hw₃ le_rfl C3nn ec3nn
  have hη0 : 0 ≤ M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄
      w₅ β₅ eexp := by
    simp only [FloatModel.cnnConv2GradBudget]
    have hγ : (0:ℝ) ≤ (1 + M.u) ^ ((2 * h) * (2 * w) + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith))
    have hn : (0:ℝ) ≤ (((2 * h) * (2 * w) : ℕ) : ℝ) := Nat.cast_nonneg _
    exact add_nonneg
      (mul_nonneg hγ (mul_nonneg hn (mul_nonneg ha (add_nonneg CPnn ecvnn))))
      (mul_nonneg hn (mul_nonneg ha ecvnn))
  -- the flattened kernel inherits the per-entry bound
  have hv2 : ∀ idx, |Kernel4.flatten W₂ idx| ≤ w₂ := by
    intro idx
    obtain ⟨o', c', kh', kw', rfl⟩ := k4Idx_surj idx
    rw [flatten_k4Idx]; exact hW₂ o' c' kh' kw'
  -- discharge the abstract gradient accuracy by the proven grad-close
  have hgh : ∀ idx, |M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
      (Kernel4.flatten W₂) idx -
      gradAt (fun v' : Vec (c * c * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten v') b₂ x₁)))))))))
          label) (Kernel4.flatten W₂) idx| ≤
      M.cnnConv2GradBudget c h w d₃ d₄ nC kH kW a w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp := by
    intro idx
    obtain ⟨o', c', kh', kw', rfl⟩ := k4Idx_surj idx
    exact cnn_conv2_grad_close M b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label fexp
      (Kernel4.flatten W₂) hh hw ha hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅
      heexp0 heexp1 hfexp hρ1 hx hv2 hb₂ hW₃ hb₃ hW₄ hb₄ hW₅ hb₅
      (fun k => by rw [Kernel4.unflatten_flatten]; exact hmarginConv k)
      (by rw [Kernel4.unflatten_flatten]; exact hmarginPool)
      (fun l => by rw [Kernel4.unflatten_flatten]; exact hmargin3 l)
      (fun q => by rw [Kernel4.unflatten_flatten]; exact hmargin4 q)
      o' c' kh' kw'
  exact cnn_conv2_sgd_descends W₂ b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label
    (M.cnnConv2FloatGrad b₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ fexp label (Kernel4.flatten W₂))
    hc hh hw ha hx hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ hlr hη0 hgh hm2 hmq hm3 hm4 hsmall h1 h2

-- ════════════════════════════════════════════════════════════════
-- § Conv as a function of its INPUT: the tap Jacobian and its masses
--
-- The conv1 rung crosses conv2 as a function of its input. Conv is
-- LINEAR in its input; the Jacobian entry pairing input `(ci,hi,wi)`
-- with output `(co,ho,wo)` is a single kernel tap (`convTap`, the
-- input-side peer of `convPad`), extracted from the certified input-VJP
-- (`conv2d_has_vjp3`) by contracting `.correct` against a basis
-- cotangent — point-free, exactly like `conv2d_weight_pdiv`. Each
-- input entry feeds at most `oc·kH·kW` outputs and each output reads at
-- most `ic·kH·kW` inputs (`convTap_out_l1` / `convTap_in_l1`): the
-- `ℓ1` operator factor of a conv crossing is `(channels)·kH·kW·w₂ᶜ`,
-- NOT a spatial count — locality is what keeps the conv1 constant
-- usable at trained magnitudes.
-- ════════════════════════════════════════════════════════════════

/-- A 0/1-pinned sum is at most its pinned value: if `P` holds for at
    most one index, `∑ i, (if P i then X else 0) ≤ X`. -/
theorem sum_pinned_le {n : Nat} {X : ℝ} (hX : 0 ≤ X) (P : Fin n → Prop)
    [DecidablePred P] (huniq : ∀ i j, P i → P j → i = j) :
    ∑ i : Fin n, (if P i then X else 0) ≤ X := by
  by_cases hex : ∃ i, P i
  · obtain ⟨i₀, hi₀⟩ := hex
    rw [Finset.sum_eq_single i₀
      (fun j _ hne => if_neg (fun hP => hne (huniq j i₀ hP hi₀)))
      (fun h => absurd (Finset.mem_univ _) h), if_pos hi₀]
  · rw [Finset.sum_eq_zero (fun i _ => if_neg (fun hP => hex ⟨i, hP⟩))]
    exact hX

/-- Rotate the innermost summation index of a triple sum to the front. -/
theorem sum_swap_12_3 {α β γ : Type*} [Fintype α] [Fintype β] [Fintype γ]
    (f : α → β → γ → ℝ) :
    ∑ a : α, ∑ b : β, ∑ c : γ, f a b c =
      ∑ c : γ, ∑ a : α, ∑ b : β, f a b c :=
  calc ∑ a : α, ∑ b : β, ∑ c : γ, f a b c
      = ∑ a : α, ∑ c : γ, ∑ b : β, f a b c :=
        Finset.sum_congr rfl fun _a _ => Finset.sum_comm
    _ = ∑ c : γ, ∑ a : α, ∑ b : β, f a b c := Finset.sum_comm

/-- Swap the two index pairs of a quadruple sum. -/
theorem sum_swap_pair_pair {α β γ δ : Type*}
    [Fintype α] [Fintype β] [Fintype γ] [Fintype δ]
    (f : α → β → γ → δ → ℝ) :
    ∑ a : α, ∑ b : β, ∑ c : γ, ∑ d : δ, f a b c d =
      ∑ c : γ, ∑ d : δ, ∑ a : α, ∑ b : β, f a b c d :=
  calc ∑ a : α, ∑ b : β, ∑ c : γ, ∑ d : δ, f a b c d
      = ∑ c : γ, ∑ a : α, ∑ b : β, ∑ d : δ, f a b c d :=
        sum_swap_12_3 (fun a b c => ∑ d : δ, f a b c d)
    _ = ∑ c : γ, ∑ a : α, ∑ d : δ, ∑ b : β, f a b c d :=
        Finset.sum_congr rfl fun _c _ =>
          Finset.sum_congr rfl fun _a _ => Finset.sum_comm
    _ = ∑ c : γ, ∑ d : δ, ∑ a : α, ∑ b : β, f a b c d :=
        Finset.sum_congr rfl fun _c _ => Finset.sum_comm

/-- Triangle inequality for a difference of triple sums. -/
theorem abs_triple_sum_sub_le {α β γ : Type*}
    [Fintype α] [Fintype β] [Fintype γ] (f g : α → β → γ → ℝ) :
    |(∑ a : α, ∑ b : β, ∑ c : γ, f a b c) -
        ∑ a : α, ∑ b : β, ∑ c : γ, g a b c| ≤
      ∑ a : α, ∑ b : β, ∑ c : γ, |f a b c - g a b c| := by
  calc |(∑ a : α, ∑ b : β, ∑ c : γ, f a b c) -
        ∑ a : α, ∑ b : β, ∑ c : γ, g a b c|
      = |∑ a : α, ((∑ b : β, ∑ c : γ, f a b c) -
          ∑ b : β, ∑ c : γ, g a b c)| := by
        rw [← Finset.sum_sub_distrib]
    _ ≤ ∑ a : α, |(∑ b : β, ∑ c : γ, f a b c) -
          ∑ b : β, ∑ c : γ, g a b c| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ a : α, ∑ b : β, ∑ c : γ, |f a b c - g a b c| := by
        refine Finset.sum_le_sum fun a _ => ?_
        calc |(∑ b : β, ∑ c : γ, f a b c) - ∑ b : β, ∑ c : γ, g a b c|
            = |∑ b : β, ((∑ c : γ, f a b c) - ∑ c : γ, g a b c)| := by
              rw [← Finset.sum_sub_distrib]
          _ ≤ ∑ b : β, |(∑ c : γ, f a b c) - ∑ c : γ, g a b c| :=
              Finset.abs_sum_le_sum_abs _ _
          _ ≤ ∑ b : β, ∑ c : γ, |f a b c - g a b c| := by
              refine Finset.sum_le_sum fun b _ => ?_
              calc |(∑ c : γ, f a b c) - ∑ c : γ, g a b c|
                  = |∑ c : γ, (f a b c - g a b c)| := by
                    rw [← Finset.sum_sub_distrib]
                _ ≤ ∑ c : γ, |f a b c - g a b c| :=
                    Finset.abs_sum_le_sum_abs _ _

/-- The kernel tap that multiplies input entry `(ci,hi,wi)` in output
    entry `(co,ho,wo)` — the input-side Jacobian entry of `conv2d`.
    Depends on the kernel only, never the input (conv is linear in its
    input). Deliberately let-free, like `convPad`. -/
noncomputable def convTap {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) : ℝ :=
  if hpad : ho.val ≤ hi.val + (kH - 1) / 2 ∧
      hi.val + (kH - 1) / 2 - ho.val < kH ∧
      wo.val ≤ wi.val + (kW - 1) / 2 ∧
      wi.val + (kW - 1) / 2 - wo.val < kW then
    W co ci ⟨hi.val + (kH - 1) / 2 - ho.val, hpad.2.1⟩
            ⟨wi.val + (kW - 1) / 2 - wo.val, hpad.2.2.2⟩
  else 0

/-- A single conv tap is bounded by the kernel magnitude (out-of-pad taps are
    zero) — the per-entry bound the conv-2 backward `dot_perturbed_close` uses. -/
theorem convTap_abs_le {ic oc h w kH kW : Nat} {W : Kernel4 oc ic kH kW}
    {w' : ℝ} (hw' : 0 ≤ w') (hW : ∀ o c kh kw, |W o c kh kw| ≤ w')
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) :
    |convTap W ci hi wi co ho wo| ≤ w' := by
  unfold convTap
  split_ifs with hpad
  · exact hW _ _ _ _
  · simpa using hw'

/-- **The tap as a kernel-offset indicator sum**: `|convTap|` is the sum
    over kernel offsets `(kh,kw)` of `|W co ci kh kw|` pinned to the
    unique offset aligning input `(hi,wi)` with output `(ho,wo)`. The
    workhorse for both mass bounds: summing it over OUTPUTS pins
    `(ho,wo)` per offset, summing it over INPUTS pins `(hi,wi)`. -/
theorem abs_convTap_expand {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) :
    |convTap W ci hi wi co ho wo| =
      ∑ kh : Fin kH, ∑ kw : Fin kW,
        if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
            kw.val + wo.val = wi.val + (kW - 1) / 2
          then |W co ci kh kw| else 0 := by
  unfold convTap
  split_ifs with hpad
  · rw [Finset.sum_eq_single
        (⟨hi.val + (kH - 1) / 2 - ho.val, hpad.2.1⟩ : Fin kH)
        (fun kh _ hne => by
          rw [Finset.sum_eq_zero]
          intro kw _
          exact if_neg (fun hcon => hne (Fin.ext (by
            show kh.val = hi.val + (kH - 1) / 2 - ho.val
            omega))))
        (fun habs => absurd (Finset.mem_univ _) habs),
      Finset.sum_eq_single
        (⟨wi.val + (kW - 1) / 2 - wo.val, hpad.2.2.2⟩ : Fin kW)
        (fun kw _ hne => if_neg (fun hcon => hne (Fin.ext (by
          show kw.val = wi.val + (kW - 1) / 2 - wo.val
          omega))))
        (fun habs => absurd (Finset.mem_univ _) habs),
      if_pos ⟨by show hi.val + (kH - 1) / 2 - ho.val + ho.val = _; omega,
        by show wi.val + (kW - 1) / 2 - wo.val + wo.val = _; omega⟩]
  · rw [abs_zero]
    symm
    rw [Finset.sum_eq_zero]
    intro kh _
    rw [Finset.sum_eq_zero]
    intro kw _
    refine if_neg (fun hcon => hpad ?_)
    have hk1 := kh.isLt
    have hk2 := kw.isLt
    omega

/-- Output-side tap mass: one input entry feeds at most `oc·kH·kW`
    outputs, each through a tap bounded by `wK` — the `ℓ1→ℓ1` operator
    factor of a conv crossing as a function of its input. -/
theorem convTap_out_l1 {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    {wK : ℝ} (hW : ∀ o c kh kw, |W o c kh kw| ≤ wK)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w) :
    ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        |convTap W ci hi wi co ho wo| ≤
      ((oc * kH * kW : ℕ) : ℝ) * wK := by
  calc ∑ co : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        |convTap W ci hi wi co ho wo|
      ≤ ∑ _co : Fin oc, ∑ _kh : Fin kH, ∑ _kw : Fin kW, wK := by
        refine Finset.sum_le_sum fun co _ => ?_
        calc ∑ ho : Fin h, ∑ wo : Fin w, |convTap W ci hi wi co ho wo|
            = ∑ ho : Fin h, ∑ wo : Fin w, ∑ kh : Fin kH, ∑ kw : Fin kW,
                (if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = wi.val + (kW - 1) / 2
                  then |W co ci kh kw| else 0) := by
              refine Finset.sum_congr rfl fun ho _ =>
                Finset.sum_congr rfl fun wo _ => ?_
              exact abs_convTap_expand W ci hi wi co ho wo
          _ = ∑ kh : Fin kH, ∑ kw : Fin kW, ∑ ho : Fin h, ∑ wo : Fin w,
                (if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = wi.val + (kW - 1) / 2
                  then |W co ci kh kw| else 0) := by
              exact sum_swap_pair_pair _
          _ ≤ ∑ kh : Fin kH, ∑ kw : Fin kW, |W co ci kh kw| := by
              refine Finset.sum_le_sum fun kh _ =>
                Finset.sum_le_sum fun kw _ => ?_
              calc ∑ ho : Fin h, ∑ wo : Fin w,
                    (if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                        kw.val + wo.val = wi.val + (kW - 1) / 2
                      then |W co ci kh kw| else 0)
                  ≤ ∑ ho : Fin h,
                      (if kh.val + ho.val = hi.val + (kH - 1) / 2
                        then |W co ci kh kw| else 0) := by
                    refine Finset.sum_le_sum fun ho _ => ?_
                    by_cases hrow : kh.val + ho.val = hi.val + (kH - 1) / 2
                    · rw [if_pos hrow]
                      refine sum_pinned_le (abs_nonneg _) _ ?_
                      intro i j hPi hPj
                      exact Fin.ext (by omega)
                    · rw [if_neg hrow]
                      refine le_of_eq (Finset.sum_eq_zero fun wo _ => ?_)
                      exact if_neg (fun hcon => hrow hcon.1)
                _ ≤ |W co ci kh kw| := by
                    refine sum_pinned_le (abs_nonneg _) _ ?_
                    intro i j hPi hPj
                    exact Fin.ext (by omega)
          _ ≤ ∑ _kh : Fin kH, ∑ _kw : Fin kW, wK :=
              Finset.sum_le_sum fun kh _ => Finset.sum_le_sum fun kw _ =>
                hW co ci kh kw
    _ = ((oc * kH * kW : ℕ) : ℝ) * wK := by
        rw [Finset.sum_const, Finset.sum_const, Finset.sum_const,
          Finset.card_univ, Finset.card_univ, Finset.card_univ,
          Fintype.card_fin, Fintype.card_fin, Fintype.card_fin,
          smul_smul, smul_smul, nsmul_eq_mul]

/-- Input-side tap mass: one output entry reads at most `ic·kH·kW`
    inputs, each through a tap bounded by `wK`. -/
theorem convTap_in_l1 {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    {wK : ℝ} (hW : ∀ o c kh kw, |W o c kh kw| ≤ wK)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) :
    ∑ ci : Fin ic, ∑ hi : Fin h, ∑ wi : Fin w,
        |convTap W ci hi wi co ho wo| ≤
      ((ic * kH * kW : ℕ) : ℝ) * wK := by
  calc ∑ ci : Fin ic, ∑ hi : Fin h, ∑ wi : Fin w,
        |convTap W ci hi wi co ho wo|
      ≤ ∑ _ci : Fin ic, ∑ _kh : Fin kH, ∑ _kw : Fin kW, wK := by
        refine Finset.sum_le_sum fun ci _ => ?_
        calc ∑ hi : Fin h, ∑ wi : Fin w, |convTap W ci hi wi co ho wo|
            = ∑ hi : Fin h, ∑ wi : Fin w, ∑ kh : Fin kH, ∑ kw : Fin kW,
                (if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = wi.val + (kW - 1) / 2
                  then |W co ci kh kw| else 0) := by
              refine Finset.sum_congr rfl fun hi _ =>
                Finset.sum_congr rfl fun wi _ => ?_
              exact abs_convTap_expand W ci hi wi co ho wo
          _ = ∑ kh : Fin kH, ∑ kw : Fin kW, ∑ hi : Fin h, ∑ wi : Fin w,
                (if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = wi.val + (kW - 1) / 2
                  then |W co ci kh kw| else 0) := by
              exact sum_swap_pair_pair _
          _ ≤ ∑ kh : Fin kH, ∑ kw : Fin kW, |W co ci kh kw| := by
              refine Finset.sum_le_sum fun kh _ =>
                Finset.sum_le_sum fun kw _ => ?_
              calc ∑ hi : Fin h, ∑ wi : Fin w,
                    (if kh.val + ho.val = hi.val + (kH - 1) / 2 ∧
                        kw.val + wo.val = wi.val + (kW - 1) / 2
                      then |W co ci kh kw| else 0)
                  ≤ ∑ hi : Fin h,
                      (if kh.val + ho.val = hi.val + (kH - 1) / 2
                        then |W co ci kh kw| else 0) := by
                    refine Finset.sum_le_sum fun hi _ => ?_
                    by_cases hrow : kh.val + ho.val = hi.val + (kH - 1) / 2
                    · rw [if_pos hrow]
                      refine sum_pinned_le (abs_nonneg _) _ ?_
                      intro i j hPi hPj
                      exact Fin.ext (by omega)
                    · rw [if_neg hrow]
                      refine le_of_eq (Finset.sum_eq_zero fun wi _ => ?_)
                      exact if_neg (fun hcon => hrow hcon.1)
                _ ≤ |W co ci kh kw| := by
                    refine sum_pinned_le (abs_nonneg _) _ ?_
                    intro i j hPi hPj
                    exact Fin.ext (by omega)
          _ ≤ ∑ _kh : Fin kH, ∑ _kw : Fin kW, wK :=
              Finset.sum_le_sum fun kh _ => Finset.sum_le_sum fun kw _ =>
                hW co ci kh kw
    _ = ((ic * kH * kW : ℕ) : ℝ) * wK := by
        rw [Finset.sum_const, Finset.sum_const, Finset.sum_const,
          Finset.card_univ, Finset.card_univ, Finset.card_univ,
          Fintype.card_fin, Fintype.card_fin, Fintype.card_fin,
          smul_smul, smul_smul, nsmul_eq_mul]

/-- **Closed form of the conv input-map `pdiv3`** — extracted from the
    certified input-VJP (`conv2d_has_vjp3`) by contracting its
    `.correct` field against a basis cotangent. Point-free in `x`:
    conv is linear in its input. -/
theorem conv2d_input_pdiv3 {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w)
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) :
    pdiv3 (conv2d W b) x ci hi wi co ho wo =
      convTap W ci hi wi co ho wo := by
  have hb := (conv2d_has_vjp3 W b).correct x
    (fun co' ho' wo' =>
      if co' = co ∧ ho' = ho ∧ wo' = wo then (1:ℝ) else 0) ci hi wi
  have hsum : ∑ co' : Fin oc, ∑ ho' : Fin h, ∑ wo' : Fin w,
      pdiv3 (conv2d W b) x ci hi wi co' ho' wo' *
        (if co' = co ∧ ho' = ho ∧ wo' = wo then (1:ℝ) else 0) =
      pdiv3 (conv2d W b) x ci hi wi co ho wo := by
    rw [Finset.sum_eq_single co
      (fun co' _ hne => by
        rw [Finset.sum_eq_zero]
        intro ho' _
        rw [Finset.sum_eq_zero]
        intro wo' _
        rw [if_neg (fun hcon => hne hcon.1), mul_zero])
      (fun habs => absurd (Finset.mem_univ _) habs),
      Finset.sum_eq_single ho
      (fun ho' _ hne => by
        rw [Finset.sum_eq_zero]
        intro wo' _
        rw [if_neg (fun hcon => hne hcon.2.1), mul_zero])
      (fun habs => absurd (Finset.mem_univ _) habs),
      Finset.sum_eq_single wo
      (fun wo' _ hne => by
        rw [if_neg (fun hcon => hne hcon.2.2), mul_zero])
      (fun habs => absurd (Finset.mem_univ _) habs),
      if_pos ⟨rfl, rfl, rfl⟩, mul_one]
  rw [← hsum, ← hb]
  -- evaluate the explicit input-gradient formula at the basis cotangent
  simp only [conv2d_has_vjp3, conv2d_input_grad_formula]
  rw [Finset.sum_eq_single co
    (fun co' _ hne => Finset.sum_eq_zero fun ho' _ =>
      Finset.sum_eq_zero fun wo' _ => by simp [hne])
    (fun habs => absurd (Finset.mem_univ _) habs),
    Finset.sum_eq_single ho
    (fun ho' _ hne => Finset.sum_eq_zero fun wo' _ => by simp [hne])
    (fun habs => absurd (Finset.mem_univ _) habs),
    Finset.sum_eq_single wo
    (fun wo' _ hne => by simp [hne])
    (fun habs => absurd (Finset.mem_univ _) habs)]
  simp only [and_self, if_true, mul_one]
  rfl

/-- Flat-coordinate form of `conv2d_input_pdiv3` — the shape the chain
    rule through `flatConv W₂ b₂` consumes. -/
theorem conv2d_flat_input_pdiv {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (y : Vec (ic * h * w))
    (ci : Fin ic) (hi : Fin h) (wi : Fin w)
    (co : Fin oc) (ho : Fin h) (wo : Fin w) :
    pdiv (fun u : Vec (ic * h * w) =>
        Tensor3.flatten (conv2d W b (Tensor3.unflatten u))) y
      (t3Idx ci hi wi) (t3Idx co ho wo) =
      convTap W ci hi wi co ho wo := by
  have h1 : pdiv (fun u : Vec (ic * h * w) =>
      Tensor3.flatten (conv2d W b (Tensor3.unflatten u))) y
      (t3Idx ci hi wi) (t3Idx co ho wo) =
      pdiv3 (conv2d W b) (Tensor3.unflatten y) ci hi wi co ho wo := by
    unfold pdiv3
    rw [Tensor3.flatten_unflatten]
    rfl
  rw [h1, conv2d_input_pdiv3]

-- ════════════════════════════════════════════════════════════════
-- § Conv input drift: per-entry (ℓ∞) and total (ℓ1), locality factors
-- ════════════════════════════════════════════════════════════════

/-- Padded reads move no more than the input entries. -/
theorem abs_convPad_sub_le {ic h w kH kW : Nat} (x x' : Tensor3 ic h w)
    {δ : ℝ} (hδ : 0 ≤ δ) (hclose : ∀ c i j, |x' c i j - x c i j| ≤ δ)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    |convPad kH kW x' c kh kw hi wi - convPad kH kW x c kh kw hi wi| ≤
      δ := by
  unfold convPad
  split_ifs with hcond
  · exact hclose _ _ _
  · simpa using hδ

/-- **Per-entry conv input drift**: each output reads `ic·kH·kW` padded
    inputs through taps bounded by `wK`. -/
theorem conv2d_input_entry_drift {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x x' : Tensor3 ic h w)
    {wK δ : ℝ} (hwK : 0 ≤ wK) (hW : ∀ o c kh kw, |W o c kh kw| ≤ wK)
    (hδ : 0 ≤ δ) (hclose : ∀ c i j, |x' c i j - x c i j| ≤ δ)
    (o : Fin oc) (ho : Fin h) (wo : Fin w) :
    |conv2d W b x' o ho wo - conv2d W b x o ho wo| ≤
      ((ic * kH * kW : ℕ) : ℝ) * (wK * δ) := by
  rw [conv2d_eq_convPad, conv2d_eq_convPad]
  have hdiff : (b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * convPad kH kW x' c kh kw ho wo) -
      (b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * convPad kH kW x c kh kw ho wo) =
      ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * (convPad kH kW x' c kh kw ho wo -
          convPad kH kW x c kh kw ho wo) := by
    have h1 : ∀ c : Fin ic,
        (∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * convPad kH kW x' c kh kw ho wo) -
        (∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * convPad kH kW x c kh kw ho wo) =
        ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo) := by
      intro c
      rw [← Finset.sum_sub_distrib]
      refine Finset.sum_congr rfl fun kh _ => ?_
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun kw _ => by ring
    have h2 : (∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * convPad kH kW x' c kh kw ho wo) -
        (∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * convPad kH kW x c kh kw ho wo) =
        ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo) := by
      rw [← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun c _ => h1 c
    linarith [h2]
  rw [hdiff]
  calc |∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * (convPad kH kW x' c kh kw ho wo -
          convPad kH kW x c kh kw ho wo)|
      ≤ ∑ c : Fin ic, |∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo)| :=
        Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, |∑ kw : Fin kW,
          W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo)| :=
        Finset.sum_le_sum fun c _ => Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          |W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo)| :=
        Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
          Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ _c : Fin ic, ∑ _kh : Fin kH, ∑ _kw : Fin kW, wK * δ := by
        refine Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
          Finset.sum_le_sum fun kw _ => ?_
        rw [abs_mul]
        exact mul_le_mul (hW o c kh kw)
          (abs_convPad_sub_le x x' hδ hclose c kh kw ho wo)
          (abs_nonneg _) hwK
    _ = ((ic * kH * kW : ℕ) : ℝ) * (wK * δ) := by
        rw [Finset.sum_const, Finset.sum_const, Finset.sum_const,
          Finset.card_univ, Finset.card_univ, Finset.card_univ,
          Fintype.card_fin, Fintype.card_fin, Fintype.card_fin,
          smul_smul, smul_smul, nsmul_eq_mul]

/-- The padded-read drift as a position-pinned indicator sum — the
    input-side peer of `abs_convTap_expand`, for the `ℓ1` bound. -/
theorem abs_convPad_sub_expand {ic h w kH kW : Nat} (x x' : Tensor3 ic h w)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (ho : Fin h) (wo : Fin w) :
    |convPad kH kW x' c kh kw ho wo - convPad kH kW x c kh kw ho wo| =
      ∑ i : Fin h, ∑ j : Fin w,
        if kh.val + ho.val = i.val + (kH - 1) / 2 ∧
            kw.val + wo.val = j.val + (kW - 1) / 2
          then |x' c i j - x c i j| else 0 := by
  unfold convPad
  split_ifs with hpad
  · rw [Finset.sum_eq_single
        (⟨kh.val + ho.val - (kH - 1) / 2, hpad.2.1⟩ : Fin h)
        (fun i _ hne => by
          rw [Finset.sum_eq_zero]
          intro j _
          exact if_neg (fun hcon => hne (Fin.ext (by
            show i.val = kh.val + ho.val - (kH - 1) / 2
            omega))))
        (fun habs => absurd (Finset.mem_univ _) habs),
      Finset.sum_eq_single
        (⟨kw.val + wo.val - (kW - 1) / 2, hpad.2.2.2⟩ : Fin w)
        (fun j _ hne => if_neg (fun hcon => hne (Fin.ext (by
          show j.val = kw.val + wo.val - (kW - 1) / 2
          omega))))
        (fun habs => absurd (Finset.mem_univ _) habs),
      if_pos ⟨by show _ = kh.val + ho.val - (kH - 1) / 2 + _; omega,
        by show _ = kw.val + wo.val - (kW - 1) / 2 + _; omega⟩]
  · rw [sub_zero, abs_zero]
    symm
    rw [Finset.sum_eq_zero]
    intro i _
    rw [Finset.sum_eq_zero]
    intro j _
    refine if_neg (fun hcon => hpad ?_)
    have h1 := i.isLt
    have h2 := j.isLt
    omega

/-- **`ℓ1` conv input drift**: each input entry feeds at most `oc·kH·kW`
    outputs, so the total output drift is at most `oc·kH·kW·wK` times
    the total input drift — locality, not a spatial count. -/
theorem conv2d_input_l1_drift {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x x' : Tensor3 ic h w)
    {wK : ℝ} (hwK : 0 ≤ wK) (hW : ∀ o c kh kw, |W o c kh kw| ≤ wK) :
    ∑ o : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        |conv2d W b x' o ho wo - conv2d W b x o ho wo| ≤
      ((oc * kH * kW : ℕ) : ℝ) *
        (wK * ∑ c : Fin ic, ∑ i : Fin h, ∑ j : Fin w,
          |x' c i j - x c i j|) := by
  have hentry : ∀ (o : Fin oc) (ho : Fin h) (wo : Fin w),
      |conv2d W b x' o ho wo - conv2d W b x o ho wo| ≤
      ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        wK * |convPad kH kW x' c kh kw ho wo -
          convPad kH kW x c kh kw ho wo| := by
    intro o ho wo
    rw [conv2d_eq_convPad, conv2d_eq_convPad]
    have hdiff : (b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * convPad kH kW x' c kh kw ho wo) -
        (b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * convPad kH kW x c kh kw ho wo) =
        ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo) := by
      have h1 : ∀ c : Fin ic,
          (∑ kh : Fin kH, ∑ kw : Fin kW,
            W o c kh kw * convPad kH kW x' c kh kw ho wo) -
          (∑ kh : Fin kH, ∑ kw : Fin kW,
            W o c kh kw * convPad kH kW x c kh kw ho wo) =
          ∑ kh : Fin kH, ∑ kw : Fin kW,
            W o c kh kw * (convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo) := by
        intro c
        rw [← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl fun kh _ => ?_
        rw [← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun kw _ => by ring
      have h2 : (∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            W o c kh kw * convPad kH kW x' c kh kw ho wo) -
          (∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            W o c kh kw * convPad kH kW x c kh kw ho wo) =
          ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            W o c kh kw * (convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo) := by
        rw [← Finset.sum_sub_distrib]
        exact Finset.sum_congr rfl fun c _ => h1 c
      linarith [h2]
    rw [hdiff]
    calc |∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * (convPad kH kW x' c kh kw ho wo -
            convPad kH kW x c kh kw ho wo)|
        ≤ ∑ c : Fin ic, |∑ kh : Fin kH, ∑ kw : Fin kW,
            W o c kh kw * (convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo)| :=
          Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, |∑ kw : Fin kW,
            W o c kh kw * (convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo)| :=
          Finset.sum_le_sum fun c _ => Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            |W o c kh kw * (convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo)| :=
          Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
            Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            wK * |convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo| := by
          refine Finset.sum_le_sum fun c _ => Finset.sum_le_sum
            fun kh _ => Finset.sum_le_sum fun kw _ => ?_
          rw [abs_mul]
          exact mul_le_mul_of_nonneg_right (hW o c kh kw) (abs_nonneg _)
  calc ∑ o : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
        |conv2d W b x' o ho wo - conv2d W b x o ho wo|
      ≤ ∑ o : Fin oc, ∑ ho : Fin h, ∑ wo : Fin w,
          ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
            wK * |convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo| :=
        Finset.sum_le_sum fun o _ => Finset.sum_le_sum fun ho _ =>
          Finset.sum_le_sum fun wo _ => hentry o ho wo
    _ = ∑ o : Fin oc, ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          ∑ ho : Fin h, ∑ wo : Fin w,
            wK * |convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo| := by
        refine Finset.sum_congr rfl fun o _ => ?_
        exact (sum_swap_12_3 _).trans
          (Finset.sum_congr rfl fun c _ => sum_swap_pair_pair _)
    _ ≤ ∑ _o : Fin oc, ∑ c : Fin ic, ∑ _kh : Fin kH, ∑ _kw : Fin kW,
          wK * (∑ i : Fin h, ∑ j : Fin w, |x' c i j - x c i j|) := by
        refine Finset.sum_le_sum fun o _ => Finset.sum_le_sum fun c _ =>
          Finset.sum_le_sum fun kh _ => Finset.sum_le_sum fun kw _ => ?_
        have hfact : ∑ ho : Fin h, ∑ wo : Fin w,
            wK * |convPad kH kW x' c kh kw ho wo -
              convPad kH kW x c kh kw ho wo| =
            wK * ∑ ho : Fin h, ∑ wo : Fin w,
              |convPad kH kW x' c kh kw ho wo -
                convPad kH kW x c kh kw ho wo| := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun ho _ => ?_
          rw [Finset.mul_sum]
        rw [hfact]
        refine mul_le_mul_of_nonneg_left ?_ hwK
        calc ∑ ho : Fin h, ∑ wo : Fin w,
              |convPad kH kW x' c kh kw ho wo -
                convPad kH kW x c kh kw ho wo|
            = ∑ ho : Fin h, ∑ wo : Fin w, ∑ i : Fin h, ∑ j : Fin w,
                (if kh.val + ho.val = i.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = j.val + (kW - 1) / 2
                  then |x' c i j - x c i j| else 0) := by
              refine Finset.sum_congr rfl fun ho _ =>
                Finset.sum_congr rfl fun wo _ => ?_
              exact abs_convPad_sub_expand x x' c kh kw ho wo
          _ = ∑ i : Fin h, ∑ j : Fin w, ∑ ho : Fin h, ∑ wo : Fin w,
                (if kh.val + ho.val = i.val + (kH - 1) / 2 ∧
                    kw.val + wo.val = j.val + (kW - 1) / 2
                  then |x' c i j - x c i j| else 0) := by
              exact sum_swap_pair_pair _
          _ ≤ ∑ i : Fin h, ∑ j : Fin w, |x' c i j - x c i j| := by
              refine Finset.sum_le_sum fun i _ =>
                Finset.sum_le_sum fun j _ => ?_
              calc ∑ ho : Fin h, ∑ wo : Fin w,
                    (if kh.val + ho.val = i.val + (kH - 1) / 2 ∧
                        kw.val + wo.val = j.val + (kW - 1) / 2
                      then |x' c i j - x c i j| else 0)
                  ≤ ∑ ho : Fin h,
                      (if kh.val + ho.val = i.val + (kH - 1) / 2
                        then |x' c i j - x c i j| else 0) := by
                    refine Finset.sum_le_sum fun ho _ => ?_
                    by_cases hrow : kh.val + ho.val = i.val + (kH - 1) / 2
                    · rw [if_pos hrow]
                      refine sum_pinned_le (abs_nonneg _) _ ?_
                      intro p q hPp hPq
                      exact Fin.ext (by omega)
                    · rw [if_neg hrow]
                      refine le_of_eq (Finset.sum_eq_zero fun wo _ => ?_)
                      exact if_neg (fun hcon => hrow hcon.1)
                _ ≤ |x' c i j - x c i j| := by
                    refine sum_pinned_le (abs_nonneg _) _ ?_
                    intro p q hPp hPq
                    exact Fin.ext (by omega)
    _ = ((oc * kH * kW : ℕ) : ℝ) *
          (wK * ∑ c : Fin ic, ∑ i : Fin h, ∑ j : Fin w,
            |x' c i j - x c i j|) := by
        have hinner : ∑ c : Fin ic, ∑ _kh : Fin kH, ∑ _kw : Fin kW,
            wK * (∑ i : Fin h, ∑ j : Fin w, |x' c i j - x c i j|) =
            ((kH * kW : ℕ) : ℝ) * (wK * ∑ c : Fin ic, ∑ i : Fin h,
              ∑ j : Fin w, |x' c i j - x c i j|) := by
          calc ∑ c : Fin ic, ∑ _kh : Fin kH, ∑ _kw : Fin kW,
              wK * (∑ i : Fin h, ∑ j : Fin w, |x' c i j - x c i j|)
              = ∑ c : Fin ic, ((kH * kW : ℕ) : ℝ) *
                  (wK * ∑ i : Fin h, ∑ j : Fin w, |x' c i j - x c i j|) := by
                refine Finset.sum_congr rfl fun c _ => ?_
                rw [Finset.sum_const, Finset.sum_const, Finset.card_univ,
                  Finset.card_univ, Fintype.card_fin, Fintype.card_fin,
                  smul_smul, nsmul_eq_mul]
            _ = ((kH * kW : ℕ) : ℝ) * (wK * ∑ c : Fin ic, ∑ i : Fin h,
                  ∑ j : Fin w, |x' c i j - x c i j|) := by
                rw [Finset.mul_sum, Finset.mul_sum]
        calc ∑ _o : Fin oc, ∑ c : Fin ic, ∑ _kh : Fin kH, ∑ _kw : Fin kW,
            wK * (∑ i : Fin h, ∑ j : Fin w, |x' c i j - x c i j|)
            = ∑ _o : Fin oc, ((kH * kW : ℕ) : ℝ) *
                (wK * ∑ c : Fin ic, ∑ i : Fin h, ∑ j : Fin w,
                  |x' c i j - x c i j|) :=
              Finset.sum_congr rfl fun o _ => hinner
          _ = ((oc * kH * kW : ℕ) : ℝ) *
                (wK * ∑ c : Fin ic, ∑ i : Fin h, ∑ j : Fin w,
                  |x' c i j - x c i j|) := by
              rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
                nsmul_eq_mul]
              push_cast
              ring

-- ════════════════════════════════════════════════════════════════
-- § The conv1 drift chain: through BOTH convs to the logits
-- ════════════════════════════════════════════════════════════════

/-- POST-relu₁ tensor drift under a conv1 kernel perturbation. -/
theorem cnn1_postrelu1_close {ic c h w kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) {a : ℝ} (ha : 0 ≤ a)
    (hx : ∀ cc i j, |x₀ cc i j| ≤ a) (u e : Vec (c * ic * kH * kW))
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten (u + e)) b₁ x₀))) :
          Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀))) :
          Tensor3 c (2*h) (2*w)) ci hi wi| ≤
      a * ∑ idx, |e idx| := by
  rw [unflatten_t3Idx, unflatten_t3Idx]
  exact le_trans (relu_entry_lipschitz _ _ _ _)
    (conv2d_flat_kernel_drift_total b₁ x₀ ha hx u e _)

/-- Per-entry conv2-preactivation drift under a conv1 kernel
    perturbation: the perturbation crosses conv2 as a function of its
    INPUT, picking up the locality factor `c·kH·kW·w₂`. -/
theorem cnn1_z2_entry_drift {ic c h w kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {a w₂ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (u e : Vec (c * ic * kH * kW)) (k : Fin (c * (2*h) * (2*w))) :
    |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (u + e)) b₁ x₀))))) k -
      Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))) k| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * ∑ idx, |e idx|)) := by
  obtain ⟨p, rfl⟩ := finProdFinEquiv.surjective k
  obtain ⟨pp, wo⟩ := p
  obtain ⟨q, rfl⟩ := finProdFinEquiv.surjective pp
  obtain ⟨o, ho⟩ := q
  rw [show finProdFinEquiv (finProdFinEquiv (o, ho), wo) =
        t3Idx o ho wo from rfl,
    flatten_t3Idx, flatten_t3Idx]
  exact conv2d_input_entry_drift W₂ b₂ _ _ hw₂ hW₂
    (mul_nonneg ha (Finset.sum_nonneg fun _ _ => abs_nonneg _))
    (fun cc i j => cnn1_postrelu1_close b₁ x₀ ha hx u e cc i j) o ho wo

/-- POST-relu₂ tensor drift under a conv1 kernel perturbation — what the
    pool margin consumes on the conv1 rung. -/
theorem cnn1_postrelu2_close {ic c h w kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {a w₂ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (u e : Vec (c * ic * kH * kW))
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + e)) b₁ x₀))))))) :
        Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀))))))) :
        Tensor3 c (2*h) (2*w)) ci hi wi| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * ∑ idx, |e idx|)) := by
  rw [unflatten_t3Idx, unflatten_t3Idx]
  exact le_trans (relu_entry_lipschitz _ _ _ _)
    (cnn1_z2_entry_drift b₁ x₀ W₂ b₂ ha hx hw₂ hW₂ u e _)

/-- Pooled `ℓ1` drift under a conv1 kernel perturbation: conv1 (`ℓ1`,
    spatial multiplicity) → relu → conv2-as-input (`ℓ1`, LOCALITY
    multiplicity `c·kH·kW`) → relu → pool. -/
theorem cnn1_pool_l1_drift {ic c h w kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {a w₂ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (u e : Vec (c * ic * kH * kW)) :
    ∑ q, |maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten (u + e)) b₁ x₀))))))) q -
        maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten u) b₁ x₀))))))) q| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * ∑ idx, |e idx|))) := by
  refine le_trans (maxPoolFlat_l1_contract _ _) (le_trans
    (Finset.sum_le_sum fun k _ => relu_entry_lipschitz _ _ _ k) ?_)
  calc ∑ k, |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + e)) b₁ x₀))))) k -
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀))))) k|
      = ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
          |conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten (u + e)) b₁ x₀)))) co ho wo -
            conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u) b₁ x₀)))) co ho wo| := by
        rw [sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
          |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten (u + e)) b₁ x₀))))) k -
            Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u) b₁ x₀))))) k|)]
        refine Finset.sum_congr rfl fun co _ => Finset.sum_congr rfl
          fun ho _ => Finset.sum_congr rfl fun wo _ => ?_
        rw [flatten_t3Idx, flatten_t3Idx]
    _ ≤ ((c * kH * kW : ℕ) : ℝ) * (w₂ *
          ∑ cc : Fin c, ∑ i : Fin (2*h), ∑ j : Fin (2*w),
            |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (conv2d (Kernel4.unflatten (u + e))
                  b₁ x₀))) : Tensor3 c (2*h) (2*w)) cc i j -
              (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (conv2d (Kernel4.unflatten u)
                  b₁ x₀))) : Tensor3 c (2*h) (2*w)) cc i j|) :=
        conv2d_input_l1_drift W₂ b₂ _ _ hw₂ hW₂
    _ ≤ ((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * ∑ idx, |e idx|))) := by
        refine mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left ?_ hw₂) (Nat.cast_nonneg _)
        calc ∑ cc : Fin c, ∑ i : Fin (2*h), ∑ j : Fin (2*w),
              |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d (Kernel4.unflatten (u + e))
                    b₁ x₀))) : Tensor3 c (2*h) (2*w)) cc i j -
                (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d (Kernel4.unflatten u)
                    b₁ x₀))) : Tensor3 c (2*h) (2*w)) cc i j|
            = ∑ k, |relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d (Kernel4.unflatten (u + e)) b₁ x₀)) k -
                relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d (Kernel4.unflatten u) b₁ x₀)) k| :=
              (sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
                |relu (c * (2*h) * (2*w)) (Tensor3.flatten
                    (conv2d (Kernel4.unflatten (u + e)) b₁ x₀)) k -
                  relu (c * (2*h) * (2*w)) (Tensor3.flatten
                    (conv2d (Kernel4.unflatten u) b₁ x₀)) k|)).symm
          _ ≤ ∑ k, |Tensor3.flatten
                  (conv2d (Kernel4.unflatten (u + e)) b₁ x₀) k -
                Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k| :=
              Finset.sum_le_sum fun k _ => relu_entry_lipschitz _ _ _ k
          _ ≤ ((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|) :=
              conv2d_flat_kernel_drift_sum b₁ x₀ ha hx u e

/-- Per-entry drift of the relu₃ pre-activation, conv1 rung. -/
theorem cnn1_z3_drift {ic c h w d₃ kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {a w₂ w₃ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (u e : Vec (c * ic * kH * kW)) (l : Fin d₃) :
    |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + e)) b₁ x₀)))))))) l -
      dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l| ≤
      w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * ∑ idx, |e idx|)))) :=
  le_trans (dense_input_drift W₃ b₃ hW₃ _ _ l)
    (mul_le_mul_of_nonneg_left
      (cnn1_pool_l1_drift b₁ x₀ W₂ b₂ ha hx hw₂ hW₂ u e) hw₃)

/-- Per-entry drift of the relu₄ pre-activation, conv1 rung. -/
theorem cnn1_z4_drift {ic c h w d₃ d₄ kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {a w₂ w₃ w₄ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (u e : Vec (c * ic * kH * kW)) (q : Fin d₄) :
    |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + e)) b₁ x₀)))))))))) q -
      dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q| ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|)))))) := by
  refine le_trans (dense_input_drift W₄ b₄ hW₄ _ _ q)
    (mul_le_mul_of_nonneg_left ?_ hw₄)
  calc ∑ l, |relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten (u + e)) b₁ x₀))))))))) l -
        relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten u) b₁ x₀))))))))) l|
      ≤ ∑ l, |dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten (u + e)) b₁ x₀)))))))) l -
          dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l| :=
        Finset.sum_le_sum fun l _ => relu_entry_lipschitz _ _ _ l
    _ ≤ ∑ _l : Fin d₃, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|)))) :=
        Finset.sum_le_sum fun l _ =>
          cnn1_z3_drift b₁ x₀ W₂ b₂ W₃ b₃ ha hx hw₂ hW₂ hw₃ hW₃ u e l
    _ = (d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
          (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|))))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

/-- Logit drift through the whole conv1 chain. -/
theorem cnn1_logit_drift {ic c h w d₃ d₄ nC kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    {a w₂ w₃ w₄ w₅ : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (u e : Vec (c * ic * kH * kW)) (k : Fin nC) :
    |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten (u + e))
              b₁ x₀)))))))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten u)
              b₁ x₀)))))))))))) k| ≤
      w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) *
        (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|)))))))) := by
  refine le_trans (dense_input_drift W₅ b₅ hW₅ _ _ k)
    (mul_le_mul_of_nonneg_left ?_ hw₅)
  calc ∑ q, |relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten (u + e)) b₁ x₀))))))))))) q -
        relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten u) b₁ x₀))))))))))) q|
      ≤ ∑ q, |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten (u + e)) b₁ x₀)))))))))) q -
          dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q| :=
        Finset.sum_le_sum fun q _ => relu_entry_lipschitz _ _ _ q
    _ ≤ ∑ _q : Fin d₄, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) *
          (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|)))))) :=
        Finset.sum_le_sum fun q _ =>
          cnn1_z4_drift b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ ha hx hw₂ hW₂ hw₃ hW₃
            hw₄ hW₄ u e q
    _ = (d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) *
          (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |e idx|))))))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

-- ════════════════════════════════════════════════════════════════
-- § conv1 margins freeze every routing decision along the segment
-- ════════════════════════════════════════════════════════════════

/-- The relu₁ margin keeps the conv1 pre-activation off the kink. -/
theorem cnn1_margin1_keeps_offkink {ic c h w kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) {a D : ℝ} (ha : 0 ≤ a)
    (hx : ∀ cc i j, |x₀ cc i j| ≤ a) (u e : Vec (c * ic * kH * kW))
    (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ k, a * D <
      |Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (k : Fin (c * (2*h) * (2*w))) :
    Tensor3.flatten (conv2d (Kernel4.unflatten (u + t • e)) b₁ x₀) k ≠ 0 ∧
      (0 < Tensor3.flatten (conv2d (Kernel4.unflatten (u + t • e)) b₁ x₀) k
        ↔ 0 < Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k) := by
  refine sign_stable_of_close ?_ (hm k)
  have h1 := conv2d_flat_kernel_drift_total b₁ x₀ ha hx u (t • e) k
  have h2 : a * (∑ idx, |(t • e) idx|) ≤ a * D :=
    mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) ha
  linarith

/-- The relu₂ margin (at the conv1 radius) keeps the conv2
    pre-activation off the kink. -/
theorem cnn1_margin2_keeps_offkink {ic c h w kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {a w₂ D : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (u e : Vec (c * ic * kH * kW)) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * D)) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))) k|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (k : Fin (c * (2*h) * (2*w))) :
    Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (u + t • e)) b₁ x₀))))) k ≠ 0 ∧
      (0 < Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + t • e)) b₁ x₀))))) k ↔
        0 < Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀))))) k) := by
  refine sign_stable_of_close ?_ (hm k)
  have h1 := cnn1_z2_entry_drift b₁ x₀ W₂ b₂ ha hx hw₂ hW₂ u (t • e) k
  have h2 : ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * ∑ idx, |(t • e) idx|)) ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * D)) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) ha) hw₂)
      (Nat.cast_nonneg _)
  linarith

/-- The POST-relu₂ tensor stays within the conv1-rung pool margin radius
    along the whole step segment. -/
theorem cnn1_postrelu2_close_seg {ic c h w kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {a w₂ D : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (u e : Vec (c * ic * kH * kW)) (he : (∑ idx, |e idx|) ≤ D)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + t • e)) b₁ x₀))))))) :
        Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀))))))) :
        Tensor3 c (2*h) (2*w)) ci hi wi| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * D)) :=
  le_trans (cnn1_postrelu2_close b₁ x₀ W₂ b₂ ha hx hw₂ hW₂ u (t • e)
      ci hi wi)
    (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) ha) hw₂)
      (Nat.cast_nonneg _))

/-- The relu₃ margin (at the conv1 radius) keeps the first head
    pre-activation off the kink. -/
theorem cnn1_margin3_keeps_offkink {ic c h w d₃ kH kW : Nat} (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {a w₂ w₃ D : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (u e : Vec (c * ic * kH * kW)) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (l : Fin d₃) :
    dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + t • e)) b₁ x₀)))))))) l ≠ 0 ∧
      (0 < dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten (u + t • e)) b₁ x₀)))))))) l ↔
        0 < dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l) := by
  refine sign_stable_of_close ?_ (hm l)
  have h1 := cnn1_z3_drift b₁ x₀ W₂ b₂ W₃ b₃ ha hx hw₂ hW₂ hw₃ hW₃
    u (t • e) l
  have h2 : w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |(t • e) idx|)))) ≤
      w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) ha)
        (Nat.cast_nonneg _)) hw₂) (Nat.cast_nonneg _)) hw₃
  linarith

/-- The relu₄ margin (at the conv1 radius) keeps the second head
    pre-activation off the kink. -/
theorem cnn1_margin4_keeps_offkink {ic c h w d₃ d₄ kH kW : Nat}
    (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {a w₂ w₃ w₄ D : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (u e : Vec (c * ic * kH * kW)) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (q : Fin d₄) :
    dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + t • e)) b₁ x₀))))))))))
        q ≠ 0 ∧
      (0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten (u + t • e)) b₁ x₀))))))))))
          q ↔
        0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q) := by
  refine sign_stable_of_close ?_ (hm q)
  have h1 := cnn1_z4_drift b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ ha hx hw₂ hW₂ hw₃ hW₃
    hw₄ hW₄ u (t • e) q
  have h2 : w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) * (a * ∑ idx, |(t • e) idx|)))))) ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (smul_l1_mass_le e ht0 ht1 he) ha)
          (Nat.cast_nonneg _)) hw₂) (Nat.cast_nonneg _)) hw₃)
      (Nat.cast_nonneg _)) hw₄
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The conv1 head gradient: through relu₁, conv2-as-input, and the
--   pool to the 3-dense head
-- ════════════════════════════════════════════════════════════════

/-- The whole head above the conv1 output — `CE∘head3∘pool∘relu∘
    (flatConv W₂ b₂)∘relu` — is differentiable at any five-condition
    point. -/
theorem cnn1_pool_head_differentiableAt {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (z₁ : Vec (c * (2*h) * (2*w))) (hz1 : ∀ k, z₁ k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₁))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁))))) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁)))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) z₁)))))))) q ≠ 0) :
    DifferentiableAt ℝ
      (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) y))))))))))) label) z₁ := by
  have hG2 := pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label hc hh hw
    (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₁)))) hz2 hmp hz3 hz4
  have hflat : DifferentiableAt ℝ
      (fun v : Vec (c * (2*h) * (2*w)) =>
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v)))
      (relu (c * (2*h) * (2*w)) z₁) :=
    (flatConv_differentiable (h := 2*h) (w := 2*w) W₂ b₂) _
  have hGF : DifferentiableAt ℝ
      ((fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
            (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label) ∘
        (fun v : Vec (c * (2*h) * (2*w)) =>
          Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v))))
      (relu (c * (2*h) * (2*w)) z₁) :=
    hG2.comp (relu (c * (2*h) * (2*w)) z₁) hflat
  exact hGF.comp (f := relu (c * (2*h) * (2*w))) z₁
    (relu_differentiableAt_of_smooth _ z₁ hz1)

/-- **Loss input-gradient at the conv1 output** — the conv1 peer of
    `pool_relu_input_grad`. One more relu mask and one conv-as-input
    crossing: the chain picks up `relu'(z₁)` and contracts the point-free
    tap Jacobian of conv2 with the pool-collapsed conv2-rung gradient. -/
theorem cnn1_pool_head_input_grad {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (z₁ : Vec (c * (2*h) * (2*w))) (hz1 : ∀ k, z₁ k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₁))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁))))) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁)))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) z₁)))))))) q ≠ 0)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) y))))))))))) label)
        z₁ (t3Idx ci hi wi) 0
      = (if z₁ (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
          ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
            convTap W₂ ci hi wi co ho wo *
              ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) z₁))) (t3Idx co ho wo) > 0
                  then (1:ℝ) else 0) *
                (if MaxPool2IsArgmax (Tensor3.unflatten
                      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                        (conv2d W₂ b₂ (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) z₁)))))) co ho wo
                  then ∑ l, W₃ (t3Idx co (winRow ho) (winCol wo)) l *
                    ((if dense W₃ b₃ (maxPoolFlat c h w
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) z₁)))))) l > 0
                        then (1:ℝ) else 0) *
                      ∑ q, W₄ l q *
                        ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃
                              (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                                (Tensor3.flatten (conv2d W₂ b₂
                                  (Tensor3.unflatten (relu
                                    (c * (2*h) * (2*w)) z₁)))))))) q > 0
                            then (1:ℝ) else 0) *
                          ∑ k, W₅ q k *
                            (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                                (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                                  (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (conv2d W₂ b₂
                                      (Tensor3.unflatten (relu
                                        (c * (2*h) * (2*w))
                                        z₁))))))))))) k -
                              oneHot nC label k)))
                  else 0)) := by
  have hc : 0 < c := Fin.pos ci
  have hh : 0 < h := by have := Fin.pos hi; omega
  have hw : 0 < w := by have := Fin.pos wi; omega
  have hG2 := pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label hc hh hw
    (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) z₁)))) hz2 hmp hz3 hz4
  have hflat : DifferentiableAt ℝ
      (fun v : Vec (c * (2*h) * (2*w)) =>
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v)))
      (relu (c * (2*h) * (2*w)) z₁) :=
    (flatConv_differentiable (h := 2*h) (w := 2*w) W₂ b₂) _
  have hGF : DifferentiableAt ℝ
      ((fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 => crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
            (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label) ∘
        (fun v : Vec (c * (2*h) * (2*w)) =>
          Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v))))
      (relu (c * (2*h) * (2*w)) z₁) :=
    hG2.comp (relu (c * (2*h) * (2*w)) z₁) hflat
  -- hop 1: peel relu₁; the chain picks up the mask
  rw [show (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
          crossEntropy nC
          (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
            (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) y))))))))))) label)
        = ((fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
            crossEntropy nC
            (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
              (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y)))))))
            label) ∘
          (fun v : Vec (c * (2*h) * (2*w)) =>
            Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v)))) ∘
          (relu (c * (2*h) * (2*w)))
        from rfl,
      pdiv_comp _ _ _
        (relu_differentiableAt_of_smooth (c * (2*h) * (2*w)) z₁ hz1) hGF]
  simp_rw [pdiv_relu (c * (2*h) * (2*w)) z₁ hz1 (t3Idx ci hi wi)]
  rw [Finset.sum_eq_single (t3Idx ci hi wi)
    (fun j _ hne => by rw [if_neg (fun heq => hne heq.symm), zero_mul])
    (fun habs => absurd (Finset.mem_univ _) habs),
    if_pos rfl]
  congr 1
  -- hop 2: through conv2 as a function of its input
  have hop2 : pdiv ((fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
        crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label) ∘
        (fun v : Vec (c * (2*h) * (2*w)) =>
          Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v))))
      (relu (c * (2*h) * (2*w)) z₁) (t3Idx ci hi wi) 0
      = ∑ k : Fin (c * (2*h) * (2*w)),
          pdiv (fun v : Vec (c * (2*h) * (2*w)) =>
              Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v)))
            (relu (c * (2*h) * (2*w)) z₁) (t3Idx ci hi wi) k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC
              (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y)))))))
              label)
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) z₁)))) k 0 :=
    pdiv_comp _ _ _ hflat hG2 _ _
  rw [hop2, sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
    pdiv (fun v : Vec (c * (2*h) * (2*w)) =>
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten v)))
      (relu (c * (2*h) * (2*w)) z₁) (t3Idx ci hi wi) k *
    pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
        crossEntropy nC
        (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) y))))))) label)
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) z₁)))) k 0)]
  refine Finset.sum_congr rfl fun co _ => Finset.sum_congr rfl
    fun ho _ => Finset.sum_congr rfl fun wo _ => ?_
  rw [conv2d_flat_input_pdiv W₂ b₂ _ ci hi wi co ho wo,
    pool_relu_input_grad W₃ b₃ W₄ b₄ W₅ b₅ label _ hz2 hmp hz3 hz4
      co ho wo]

-- ════════════════════════════════════════════════════════════════
-- § The conv1 loss-of-kernel map: differentiability and gradient
-- ════════════════════════════════════════════════════════════════

/-- The loss-of-conv1-kernel map is differentiable at any
    five-condition point. -/
theorem cnn_conv1_loss_differentiableAt {ic c h w d₃ d₄ nC kH kW : Nat}
    (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC) (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (u : Vec (c * ic * kH * kW))
    (hz1 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
      k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten u) b₁ x₀))))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q ≠ 0) :
    DifferentiableAt ℝ
      (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label) u := by
  have hG1 := cnn1_pool_head_differentiableAt W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
    label hc hh hw (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀))
    hz1 hz2 hmp hz3 hz4
  have h0 : DifferentiableAt ℝ
      (fun u' : Vec (c * ic * kH * kW) =>
        Tensor3.flatten (conv2d (Kernel4.unflatten u') b₁ x₀)) u :=
    (conv2d_weight_differentiable b₁ x₀) u
  exact ((differentiableAt_pi.mp hG1) 0).comp
    (f := fun u' : Vec (c * ic * kH * kW) =>
      Tensor3.flatten (conv2d (Kernel4.unflatten u') b₁ x₀)) u h0

/-- **Closed form of the conv1 loss gradient** at any five-margin point —
    the same fold, contracted with the conv1 head gradient
    (`cnn1_pool_head_input_grad`): the conv1 weight Jacobian
    (`convPad` reads of the IMAGE) times relu₁'s mask times the
    point-free conv2 tap Jacobian times the pool-collapsed head. Two
    spatial triple-sums: weight sharing at conv1, locality at conv2. -/
theorem cnn_conv1_loss_gradAt {ic c h w d₃ d₄ nC kH kW : Nat}
    (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC) (hh : 0 < h) (hw : 0 < w)
    (u : Vec (c * ic * kH * kW))
    (hz1 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
      k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten u) b₁ x₀))))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q ≠ 0)
    (o : Fin c) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label)
        u (k4Idx o cc kh kw)
      = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then convPad kH kW x₀ cc kh kw hi wi else 0) *
            ((if Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
                convTap W₂ ci hi wi co ho wo *
                  ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d (Kernel4.unflatten u) b₁ x₀)))))
                        (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                    (if MaxPool2IsArgmax (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d (Kernel4.unflatten u)
                                  b₁ x₀)))))))) co ho wo
                      then ∑ l, W₃ (t3Idx co (winRow ho) (winCol wo)) l *
                        ((if dense W₃ b₃ (maxPoolFlat c h w
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₂ b₂ (Tensor3.unflatten
                                  (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (conv2d
                                      (Kernel4.unflatten u)
                                      b₁ x₀)))))))) l > 0
                            then (1:ℝ) else 0) *
                          ∑ q, W₄ l q *
                            ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                  (maxPoolFlat c h w (relu
                                    (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (conv2d W₂ b₂ (Tensor3.unflatten
                                      (relu (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d
                                          (Kernel4.unflatten u)
                                          b₁ x₀)))))))))) q > 0
                                then (1:ℝ) else 0) *
                              ∑ k, W₅ q k *
                                (softmax nC (dense W₅ b₅ (relu d₄
                                    (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                      (maxPoolFlat c h w (relu
                                        (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d W₂ b₂
                                          (Tensor3.unflatten (relu
                                            (c * (2*h) * (2*w))
                                            (Tensor3.flatten (conv2d
                                              (Kernel4.unflatten u)
                                              b₁ x₀))))))))))))) k -
                                  oneHot nC label k)))
                      else 0))) := by
  have hc : 0 < c := Fin.pos o
  have hdiff := cnn_conv1_loss_differentiableAt b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄
    W₅ b₅ label hc hh hw u hz1 hz2 hmp hz3 hz4
  have hG1 := cnn1_pool_head_differentiableAt W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
    label hc hh hw (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀))
    hz1 hz2 hmp hz3 hz4
  calc gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label)
        u (k4Idx o cc kh kw)
      = pdiv (fun u' : Vec (c * ic * kH * kW) => fun _ : Fin 1 =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
            label)
          u (k4Idx o cc kh kw) 0 := gradAt_eq_pdiv _ _ hdiff _
    _ = pdiv (fun u' : Vec (c * ic * kH * kW) => fun _ : Fin 1 =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
            label)
          (Kernel4.flatten (Kernel4.unflatten u)) (k4Idx o cc kh kw)
          0 := by
        rw [Kernel4.flatten_unflatten]
    _ = ∑ k : Fin (c * (2*h) * (2*w)),
          pdiv (fun u' : Vec (c * ic * kH * kW) =>
              Tensor3.flatten (conv2d (Kernel4.unflatten u') b₁ x₀))
            (Kernel4.flatten (Kernel4.unflatten u)) (k4Idx o cc kh kw) k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) y)))))))))))
                label)
            (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)) k 0 :=
        conv_total_loss_grad_fold b₁ x₀ (Kernel4.unflatten u)
          (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                  (relu (c * (2*h) * (2*w)) y)))))))))))
              label)
          hG1 (k4Idx o cc kh kw)
    _ = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          pdiv (fun u' : Vec (c * ic * kH * kW) =>
              Tensor3.flatten (conv2d (Kernel4.unflatten u') b₁ x₀))
            (Kernel4.flatten (Kernel4.unflatten u)) (k4Idx o cc kh kw)
            (t3Idx ci hi wi) *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) y)))))))))))
                label)
            (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀))
            (t3Idx ci hi wi) 0 :=
        sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
          pdiv (fun u' : Vec (c * ic * kH * kW) =>
              Tensor3.flatten (conv2d (Kernel4.unflatten u') b₁ x₀))
            (Kernel4.flatten (Kernel4.unflatten u)) (k4Idx o cc kh kw) k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) y)))))))))))
                label)
            (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)) k 0)
    _ = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then convPad kH kW x₀ cc kh kw hi wi else 0) *
            ((if Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
                convTap W₂ ci hi wi co ho wo *
                  ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d (Kernel4.unflatten u) b₁ x₀)))))
                        (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                    (if MaxPool2IsArgmax (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d (Kernel4.unflatten u)
                                  b₁ x₀)))))))) co ho wo
                      then ∑ l, W₃ (t3Idx co (winRow ho) (winCol wo)) l *
                        ((if dense W₃ b₃ (maxPoolFlat c h w
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₂ b₂ (Tensor3.unflatten
                                  (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (conv2d
                                      (Kernel4.unflatten u)
                                      b₁ x₀)))))))) l > 0
                            then (1:ℝ) else 0) *
                          ∑ q, W₄ l q *
                            ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                  (maxPoolFlat c h w (relu
                                    (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (conv2d W₂ b₂ (Tensor3.unflatten
                                      (relu (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d
                                          (Kernel4.unflatten u)
                                          b₁ x₀)))))))))) q > 0
                                then (1:ℝ) else 0) *
                              ∑ k, W₅ q k *
                                (softmax nC (dense W₅ b₅ (relu d₄
                                    (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                      (maxPoolFlat c h w (relu
                                        (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d W₂ b₂
                                          (Tensor3.unflatten (relu
                                            (c * (2*h) * (2*w))
                                            (Tensor3.flatten (conv2d
                                              (Kernel4.unflatten u)
                                              b₁ x₀))))))))))))) k -
                                  oneHot nC label k)))
                      else 0))) := by
        refine Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl
          fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
        rw [conv2d_weight_pdiv b₁ x₀ _ o cc kh kw ci hi wi,
          cnn1_pool_head_input_grad W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label _
            hz1 hz2 hmp hz3 hz4 ci hi wi]

/-- **The certified conv-1 loss gradient, head restated in `dense`/`reluMask`
    form** — the conv-1 peer of `cnn_conv2_loss_gradAt_reluMask` (Increment 4
    keystone). One conv-backward deeper than conv-2: the conv-1-output
    cotangent is `𝟙[z₁>0] · ∑_{co,ho,wo} convTap·(conv-2-output cotangent)`,
    with the 3-dense head collapsed by `head3_cot_reluMask` exactly as in
    conv-2. The conv-1 ReLU mask, the conv-2 backward tap (`convTap`, the
    point-free conv-2 input Jacobian), the conv-2 ReLU mask and the pool
    selector all stay explicit (their float closeness is `mask_scalar_close` /
    `dot_perturbed_close` / `poolBack_close`). Packaged as the spatial dot
    `∑ₛ convPadWin x₀·cotWin` (`convWeightGrad_eq_dot`) the float conv-1 weight
    dot rounds. -/
theorem cnn_conv1_loss_gradAt_reluMask {ic c h w d₃ d₄ nC kH kW : Nat}
    (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC) (hh : 0 < h) (hw : 0 < w)
    (u : Vec (c * ic * kH * kW))
    (hz1 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten u) b₁ x₀))))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q ≠ 0)
    (o : Fin c) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label)
        u (k4Idx o cc kh kw)
      = ∑ s, convPadWin kH kW x₀ cc kh kw s *
          cotWin (fun ci hi wi =>
            (if Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
                convTap W₂ ci hi wi co ho wo *
                  ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d (Kernel4.unflatten u) b₁ x₀)))))
                        (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                    (if MaxPool2IsArgmax (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d (Kernel4.unflatten u) b₁ x₀))))))))
                          co ho wo
                      then dense (fun j i' => W₃ i' j) (fun _ => 0)
                        (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d W₂ b₂ (Tensor3.unflatten
                                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))
                          (dense (fun j i' => W₄ i' j) (fun _ => 0)
                            (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                                (dense W₃ b₃ (maxPoolFlat c h w (relu
                                  (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (conv2d W₂ b₂ (Tensor3.unflatten (relu
                                      (c * (2*h) * (2*w)) (Tensor3.flatten
                                        (conv2d (Kernel4.unflatten u)
                                          b₁ x₀)))))))))))
                              (dense (fun j i' => W₅ i' j) (fun _ => 0)
                                (fun k => softmax nC (dense W₅ b₅ (relu d₄
                                    (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                      (maxPoolFlat c h w (relu
                                        (c * (2*h) * (2*w)) (Tensor3.flatten
                                          (conv2d W₂ b₂ (Tensor3.unflatten
                                            (relu (c * (2*h) * (2*w))
                                              (Tensor3.flatten (conv2d
                                                (Kernel4.unflatten u)
                                                b₁ x₀))))))))))))) k -
                                  oneHot nC label k)))))
                        (t3Idx co (winRow ho) (winCol wo))
                      else 0))) o s := by
  rw [cnn_conv1_loss_gradAt b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw u
      hz1 hz2 hmp hz3 hz4 o cc kh kw]
  simp_rw [head3_cot_reluMask]
  rw [convWeightGrad_eq_dot x₀ _ o cc kh kw]
  rw [Finset.sum_eq_single o
    (fun ci _ hne => Finset.sum_eq_zero fun hi _ => Finset.sum_eq_zero fun wi _ =>
      by rw [if_neg hne, zero_mul])
    (fun habs => absurd (Finset.mem_univ o) habs)]
  refine Finset.sum_congr rfl fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
  rw [if_pos (rfl : o = o)]

/-- **The binary32 conv-1 weight gradient the rendered trainer computes** — the
    conv-1 peer of `cnnConv2FloatGrad`, one conv-backward deeper. At kernel
    entry `(o,cc,kh,kw)` it is the float dot of the (exact) padded-input window
    `convPadWin x₀` against the float conv-1-output cotangent slab; that
    cotangent is the conv-1 ReLU mask `𝟙[z̃₁>0]` times the float conv-2 backward
    `M.dot (convTap W₂ slab) (float conv-2-output cotangent slab)`, where the
    conv-2 cotangent is exactly `cnnConv2FloatGrad`'s, but at the FLOAT conv-2
    input `relu(z̃₁)` (a function of the conv-1 kernel `u`). All `M`-ops carry
    the rounding; `reluMask`/`maxPoolFlat`/`relu`/`convTap` are exact. -/
noncomputable def FloatModel.cnnConv1FloatGrad {ic c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC)
    (u : Vec (c * ic * kH * kW)) : Vec (c * ic * kH * kW) :=
  Kernel4.flatten fun o cc kh kw =>
    M.dot (convPadWin kH kW x₀ cc kh kw)
      (cotWin (fun ci hi wi =>
        (if Tensor3.flatten (M.convF (Kernel4.unflatten u) b₁ x₀)
              (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
          M.dot (Tensor3.flatten (fun co ho wo => convTap W₂ ci hi wi co ho wo))
            (Tensor3.flatten (fun co ho wo =>
              (if Tensor3.flatten (M.convF W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (M.convF (Kernel4.unflatten u) b₁ x₀)))))
                    (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                      (Tensor3.flatten (M.convF W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (M.convF (Kernel4.unflatten u) b₁ x₀)))))))) co ho wo
                  then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
                    (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (M.convF W₂ b₂ (Tensor3.unflatten (relu
                            (c * (2*h) * (2*w)) (Tensor3.flatten
                              (M.convF (Kernel4.unflatten u) b₁ x₀)))))))))
                      (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                        (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                            (M.dense W₃ b₃ (maxPoolFlat c h w (relu
                              (c * (2*h) * (2*w)) (Tensor3.flatten
                                (M.convF W₂ b₂ (Tensor3.unflatten (relu
                                  (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (M.convF (Kernel4.unflatten u)
                                      b₁ x₀)))))))))))
                          (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                            (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                                (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                                  (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (M.convF W₂ b₂
                                      (Tensor3.unflatten (relu
                                        (c * (2*h) * (2*w)) (Tensor3.flatten
                                          (M.convF (Kernel4.unflatten u)
                                            b₁ x₀))))))))))))) label)))))
                    (t3Idx co (winRow ho) (winCol wo))
                  else 0)))) o)

@[simp] theorem FloatModel.cnnConv1FloatGrad_apply {ic c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (fexp : ℝ → ℝ) (label : Fin nC)
    (u : Vec (c * ic * kH * kW)) (o : Fin c) (cc : Fin ic) (kh : Fin kH)
    (kw : Fin kW) :
    M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label u
        (k4Idx o cc kh kw) =
    M.dot (convPadWin kH kW x₀ cc kh kw)
      (cotWin (fun ci hi wi =>
        (if Tensor3.flatten (M.convF (Kernel4.unflatten u) b₁ x₀)
              (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
          M.dot (Tensor3.flatten (fun co ho wo => convTap W₂ ci hi wi co ho wo))
            (Tensor3.flatten (fun co ho wo =>
              (if Tensor3.flatten (M.convF W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (M.convF (Kernel4.unflatten u) b₁ x₀)))))
                    (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                      (Tensor3.flatten (M.convF W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (M.convF (Kernel4.unflatten u) b₁ x₀)))))))) co ho wo
                  then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
                    (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (M.convF W₂ b₂ (Tensor3.unflatten (relu
                            (c * (2*h) * (2*w)) (Tensor3.flatten
                              (M.convF (Kernel4.unflatten u) b₁ x₀)))))))))
                      (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                        (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                            (M.dense W₃ b₃ (maxPoolFlat c h w (relu
                              (c * (2*h) * (2*w)) (Tensor3.flatten
                                (M.convF W₂ b₂ (Tensor3.unflatten (relu
                                  (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (M.convF (Kernel4.unflatten u)
                                      b₁ x₀)))))))))))
                          (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                            (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                                (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                                  (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (M.convF W₂ b₂
                                      (Tensor3.unflatten (relu
                                        (c * (2*h) * (2*w)) (Tensor3.flatten
                                          (M.convF (Kernel4.unflatten u)
                                            b₁ x₀))))))))))))) label)))))
                    (t3Idx co (winRow ho) (winCol wo))
                  else 0)))) o) := by
  simp only [FloatModel.cnnConv1FloatGrad, Kernel4.flatten, k4Idx,
    Equiv.symm_apply_apply]

/-- **The conv-1 float-backward grad-close budget** — the conv-2 budget
    (`cnnConv2GradBudget`-shaped) deepened by one conv layer at the bottom (the
    forward nest now starts at conv-1 fan-in `ic·kH·kW`, the conv-2 input
    carries the conv-1 rounding `E₁`) and one conv-backward at the top: the
    conv-2 backward Higham γ over the slab `c·(2h)·(2w)` against the
    float-cotangent magnitude `C2t` plus the per-entry conv-2-cotangent drift
    `e₂` gives `eback`; the conv-1 spatial dot (fan-in `(2h)·(2w)`) then rides
    `eback` and the float conv-1-cotangent magnitude `C1t`. -/
noncomputable def FloatModel.cnnConv1GradBudget (M : FloatModel)
    (ic c h w d₃ d₄ nC kH kW : ℕ)
    (a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ) : ℝ :=
  let A1 := FloatModel.layerAct (ic * kH * kW) w₁ β₁ a
  let A2 := FloatModel.layerAct (c * kH * kW) w₂ β₂ A1
  let A3 := FloatModel.layerAct (c * h * w) w₃ β₃ A2
  let A4 := FloatModel.layerAct d₃ w₄ β₄ A3
  let E1 := FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0
  let E2 := FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ A1 E1
  let E3 := FloatModel.layerBudget M.u (c * h * w) w₃ β₃ A2 E2
  let E4 := FloatModel.layerBudget M.u d₃ w₄ β₄ A3 E3
  let δlogit := FloatModel.layerBudget M.u d₄ w₅ β₅ A4 E4
  let C4 := FloatModel.layerAct nC w₅ 0 1
  let C3 := FloatModel.layerAct d₄ w₄ 0 C4
  let CP := FloatModel.layerAct d₃ w₃ 0 C3
  let ecHead := FloatModel.cotErr M.u eexp δlogit nC
  let ec4 := FloatModel.layerBudget M.u nC w₅ 0 1 ecHead
  let ec3 := FloatModel.layerBudget M.u d₄ w₄ 0 C4 ec4
  let e2 := FloatModel.layerBudget M.u d₃ w₃ 0 C3 ec3
  let C2t := CP + e2
  let eback := ((1 + M.u) ^ ((c * (2 * h) * (2 * w)) + 1) - 1) *
        (((c * (2 * h) * (2 * w) : ℕ) : ℝ) * (w₂ * C2t)) +
      (((c * (2 * h) * (2 * w) : ℕ) : ℝ) * (w₂ * e2))
  let C1t := ((c * (2 * h) * (2 * w) : ℕ) : ℝ) * (w₂ * CP) + eback
  ((1 + M.u) ^ ((2 * h) * (2 * w) + 1) - 1) *
      (((2 * h) * (2 * w) : ℕ) * (a * C1t)) +
    (((2 * h) * (2 * w) : ℕ) * (a * eback))

/-- The conv-2-output cotangent error budget, as a function of the conv-2
    input magnitude `aX2` and rounding `eX2` — the `e₂` of `cnnConv2GradBudget`
    (where `aX2 = a`, `eX2 = 0`) and the `e₂` inside `cnnConv1GradBudget` (where
    `aX2 = A₁`, `eX2 = E₁`). Factored so the conv-1 rung reuses the conv-2
    cotangent chain at a FLOAT conv-2 input. -/
noncomputable def FloatModel.cnnConv2CotBudget (M : FloatModel)
    (c h w d₃ d₄ nC kH kW : ℕ) (aX2 eX2 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ) : ℝ :=
  let A2 := FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2
  let A3 := FloatModel.layerAct (c * h * w) w₃ β₃ A2
  let A4 := FloatModel.layerAct d₃ w₄ β₄ A3
  let E2 := FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2
  let E3 := FloatModel.layerBudget M.u (c * h * w) w₃ β₃ A2 E2
  let E4 := FloatModel.layerBudget M.u d₃ w₄ β₄ A3 E3
  let δlogit := FloatModel.layerBudget M.u d₄ w₅ β₅ A4 E4
  let C4 := FloatModel.layerAct nC w₅ 0 1
  let C3 := FloatModel.layerAct d₄ w₄ 0 C4
  let ecHead := FloatModel.cotErr M.u eexp δlogit nC
  let ec4 := FloatModel.layerBudget M.u nC w₅ 0 1 ecHead
  let ec3 := FloatModel.layerBudget M.u d₄ w₄ 0 C4 ec4
  FloatModel.layerBudget M.u d₃ w₃ 0 C3 ec3

/-- The real conv-2-output cotangent magnitude bound — `aX2`/`eX2`-independent
    (the head cotangent and the two masked `Wᵀ` steps are magnitude-frozen). -/
noncomputable def FloatModel.cnnConv2CotMag (d₃ d₄ nC : ℕ)
    (w₃ w₄ w₅ : ℝ) : ℝ :=
  FloatModel.layerAct d₃ w₃ 0 (FloatModel.layerAct d₄ w₄ 0
    (FloatModel.layerAct nC w₅ 0 1))

open FloatModel in
/-- **The conv-2-output cotangent is float-close at a float conv-2 input**
    (Increment 4 keystone) — Increment 2's conv-2 cotangent chain, factored to
    take the conv-2 input `(X2, X2F)` with `|X2F − X2| ≤ eX2`, `|X2| ≤ aX2`. The
    conv-1 rung instantiates `X2 = relu(z₁)`, `X2F = relu(z̃₁)`, `eX2 = E₁`. The
    chain: float forward from `X2` (`convF_close` → `dense_close`×3) → head
    (`softmax_ce_cot_close`) → `cot_step_close`×2 → unmasked W₃ `dense_close` →
    pool-back (`poolBack_close`) → conv-2 ReLU mask (`mask_scalar_close`). -/
theorem cnn_conv2_cot_close {c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (X2 X2F : Tensor3 c (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    {aX2 eX2 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (haX2 : 0 ≤ aX2) (heX2 : 0 ≤ eX2) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅)
    (hβ₅ : 0 ≤ β₅) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hX2 : ∀ co i j, |X2F co i j - X2 co i j| ≤ eX2)
    (hX2mag : ∀ co i j, |X2 co i j| ≤ aX2)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmarginConv : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2 <
      |Tensor3.flatten (conv2d W₂ b₂ X2) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2)
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ X2)))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2)
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ X2)))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2)
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2)) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ X2)))))) q|)
    (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) :
    |((if Tensor3.flatten (M.convF W₂ b₂ X2F) (t3Idx co ho wo) > 0
          then (1:ℝ) else 0) *
        (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (M.convF W₂ b₂ X2F)))) co ho wo
          then M.dense (fun j i' => W₃ i' j) (fun _ => 0)
            (FloatModel.reluMask (M.dense W₃ b₃ (maxPoolFlat c h w
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (M.convF W₂ b₂ X2F)))))
              (M.dense (fun j i' => W₄ i' j) (fun _ => 0)
                (FloatModel.reluMask (M.dense W₄ b₄ (relu d₃
                    (M.dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                      (Tensor3.flatten (M.convF W₂ b₂ X2F)))))))
                  (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
                    (M.softmaxCECotF fexp (M.dense W₅ b₅ (relu d₄
                        (M.dense W₄ b₄ (relu d₃ (M.dense W₃ b₃
                          (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                            (Tensor3.flatten (M.convF W₂ b₂ X2F))))))))) label)))))
            (t3Idx co (winRow ho) (winCol wo))
          else 0)) -
      ((if Tensor3.flatten (conv2d W₂ b₂ X2) (t3Idx co ho wo) > 0
          then (1:ℝ) else 0) *
        (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ X2)))) co ho wo
          then dense (fun j i' => W₃ i' j) (fun _ => 0)
            (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ X2)))))
              (dense (fun j i' => W₄ i' j) (fun _ => 0)
                (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                    (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                      (Tensor3.flatten (conv2d W₂ b₂ X2)))))))
                  (dense (fun j i' => W₅ i' j) (fun _ => 0)
                    (fun k => softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                        (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu
                          (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ X2))))))))) k - oneHot nC label k)))))
            (t3Idx co (winRow ho) (winCol wo))
          else 0))| ≤
      M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW aX2 eX2 w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅
        eexp := by
  -- abbreviate the forward values (real / float) from the conv-2 input X2/X2F
  set Z2C := Tensor3.flatten (conv2d W₂ b₂ X2) with hZ2C
  set Z2CF := Tensor3.flatten (M.convF W₂ b₂ X2F) with hZ2CF
  set PR := maxPoolFlat c h w (relu (c * (2*h) * (2*w)) Z2C) with hPR
  set PF := maxPoolFlat c h w (relu (c * (2*h) * (2*w)) Z2CF) with hPF
  set Z3 := dense W₃ b₃ PR with hZ3
  set Z3F := M.dense W₃ b₃ PF with hZ3F
  set Z4 := dense W₄ b₄ (relu d₃ Z3) with hZ4
  set Z4F := M.dense W₄ b₄ (relu d₃ Z3F) with hZ4F
  set Z5 := dense W₅ b₅ (relu d₄ Z4) with hZ5
  set Z5F := M.dense W₅ b₅ (relu d₄ Z4F) with hZ5F
  set A2 := FloatModel.layerAct (c * kH * kW) w₂ β₂ aX2 with hA2
  set A3 := FloatModel.layerAct (c * h * w) w₃ β₃ A2 with hA3
  set A4 := FloatModel.layerAct d₃ w₄ β₄ A3 with hA4
  set E2 := FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂ aX2 eX2 with hE2
  set E3 := FloatModel.layerBudget M.u (c * h * w) w₃ β₃ A2 E2 with hE3
  set E4 := FloatModel.layerBudget M.u d₃ w₄ β₄ A3 E3 with hE4
  set DL := FloatModel.layerBudget M.u d₄ w₅ β₅ A4 E4 with hDL
  set C4 := FloatModel.layerAct nC w₅ 0 1 with hC4
  set C3 := FloatModel.layerAct d₄ w₄ 0 C4 with hC3
  set ecH := FloatModel.cotErr M.u eexp DL nC with hecH
  set ec4 := FloatModel.layerBudget M.u nC w₅ 0 1 ecH with hec4
  set ec3 := FloatModel.layerBudget M.u d₄ w₄ 0 C4 ec4 with hec3
  have A2nn : 0 ≤ A2 := layerAct_nonneg hw₂ hβ₂ haX2
  have A3nn : 0 ≤ A3 := layerAct_nonneg hw₃ hβ₃ A2nn
  have A4nn : 0 ≤ A4 := layerAct_nonneg hw₄ hβ₄ A3nn
  have E2nn : 0 ≤ E2 := layerBudget_nonneg M.u_nonneg hw₂ hβ₂ haX2 heX2
  have E3nn : 0 ≤ E3 := layerBudget_nonneg M.u_nonneg hw₃ hβ₃ A2nn E2nn
  have E4nn : 0 ≤ E4 := layerBudget_nonneg M.u_nonneg hw₄ hβ₄ A3nn E3nn
  have DLnn : 0 ≤ DL := layerBudget_nonneg M.u_nonneg hw₅ hβ₅ A4nn E4nn
  have C4nn : 0 ≤ C4 := layerAct_nonneg hw₅ le_rfl zero_le_one
  have C3nn : 0 ≤ C3 := layerAct_nonneg hw₄ le_rfl C4nn
  have ecHnn : 0 ≤ ecH := M.cotErr_nonneg heexp0 DLnn hρ1
  have ec4nn : 0 ≤ ec4 := layerBudget_nonneg M.u_nonneg hw₅ le_rfl zero_le_one ecHnn
  have ec3nn : 0 ≤ ec3 := layerBudget_nonneg M.u_nonneg hw₄ le_rfl C4nn ec4nn
  have ecvnn : 0 ≤ FloatModel.layerBudget M.u d₃ w₃ 0 C3 ec3 :=
    layerBudget_nonneg M.u_nonneg hw₃ le_rfl C3nn ec3nn
  -- forward magnitudes (real)
  have hMconv : ∀ k, |Z2C k| ≤ A2 := by
    intro k; obtain ⟨ci, hi, wi, rfl⟩ := t3Idx_surj k
    rw [hZ2C, flatten_t3Idx]; exact conv2d_abs_le haX2 hW₂ hb₂ hX2mag ci hi wi
  have hMpool : ∀ j, |PR j| ≤ A2 :=
    fun j => maxPoolFlat_abs_le (fun k => (relu_abs_le _ k).trans (hMconv k)) j
  have hM3 : ∀ l, |relu d₃ Z3 l| ≤ A3 :=
    fun l => (relu_abs_le _ l).trans (dense_abs_le A2nn hW₃ hb₃ hMpool l)
  have hM4 : ∀ q, |relu d₄ Z4 q| ≤ A4 :=
    fun q => (relu_abs_le _ q).trans (dense_abs_le A3nn hW₄ hb₄ hM3 q)
  -- forward closeness (float vs real)
  have hEconv : ∀ k, |Z2CF k - Z2C k| ≤ E2 := by
    intro k; obtain ⟨ci, hi, wi, rfl⟩ := t3Idx_surj k
    rw [hZ2CF, hZ2C, flatten_t3Idx, flatten_t3Idx]
    exact (M.convF_close W₂ b₂ X2F X2 heX2 hX2 ci hi wi).trans
      (M.denseErr_le_uniform hw₂ heX2 (fun i j => convKernelMat_abs_le hW₂ i j)
        hb₂ (fun idx => convWindow_abs_le haX2 hX2mag hi wi idx) ci)
  have hRelu : ∀ k, |relu (c * (2*h) * (2*w)) Z2CF k -
      relu (c * (2*h) * (2*w)) Z2C k| ≤ E2 := fun k => relu_close _ _ _ hEconv k
  have hPool : ∀ k, |PF k - PR k| ≤ E2 := fun k => maxPoolFlat_close _ _ hRelu k
  have hE3close : ∀ l, |Z3F l - Z3 l| ≤ E3 := fun l =>
    (M.dense_close W₃ b₃ PF PR E2 E2nn hPool l).trans
      (M.denseErr_le_uniform hw₃ E2nn hW₃ hb₃ hMpool l)
  have hRelu3 : ∀ l, |relu d₃ Z3F l - relu d₃ Z3 l| ≤ E3 :=
    fun l => relu_close _ _ _ hE3close l
  have hE4close : ∀ q, |Z4F q - Z4 q| ≤ E4 := fun q =>
    (M.dense_close W₄ b₄ (relu d₃ Z3F) (relu d₃ Z3) E3 E3nn hRelu3 q).trans
      (M.denseErr_le_uniform hw₄ E3nn hW₄ hb₄ hM3 q)
  have hRelu4 : ∀ q, |relu d₄ Z4F q - relu d₄ Z4 q| ≤ E4 :=
    fun q => relu_close _ _ _ hE4close q
  have hDLclose : ∀ k, |Z5F k - Z5 k| ≤ DL := fun k =>
    (M.dense_close W₅ b₅ (relu d₄ Z4F) (relu d₄ Z4) E4 E4nn hRelu4 k).trans
      (M.denseErr_le_uniform hw₅ E4nn hW₅ hb₅ hM4 k)
  -- head cotangent + real head magnitude
  have hHeadCot : ∀ k, |M.softmaxCECotF fexp Z5F label k -
      (softmax nC Z5 k - oneHot nC label k)| ≤ ecH := fun k =>
    M.softmax_ce_cot_close fexp Z5F Z5 label heexp0 heexp1 hfexp hρ1 hDLclose k
  have hHeadMag : ∀ k, |softmax nC Z5 k - oneHot nC label k| ≤ 1 := by
    intro k
    have hD : 0 < ∑ t, Real.exp (Z5 t) :=
      Finset.sum_pos (fun t _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩
    have hs0 : 0 ≤ softmax nC Z5 k :=
      div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun t _ => (Real.exp_pos _).le)
    have hs1 : softmax nC Z5 k ≤ 1 :=
      (div_le_one hD).mpr
        (Finset.single_le_sum (fun t _ => (Real.exp_pos _).le) (Finset.mem_univ k))
    simp only [oneHot]
    by_cases hkl : k = label
    · rw [if_pos hkl, abs_le]; constructor <;> linarith
    · rw [if_neg hkl, abs_le]; constructor <;> linarith
  -- two masked Wᵀ cotangent steps + unmasked W₃ step
  have hc4 : ∀ q, |FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j)
        (fun _ => 0) (M.softmaxCECotF fexp Z5F label)) q -
      FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
        (fun k => softmax nC Z5 k - oneHot nC label k)) q| ≤ ec4 := fun q =>
    M.cot_step_close W₅ Z4F Z4 (M.softmaxCECotF fexp Z5F label)
      (fun k => softmax nC Z5 k - oneHot nC label k) hw₅ zero_le_one ecHnn hW₅
      hHeadMag hHeadCot hE4close hmargin4 q
  have hc4Mag : ∀ q, |FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j)
      (fun _ => 0) (fun k => softmax nC Z5 k - oneHot nC label k)) q| ≤ C4 :=
    fun q => (reluMask_abs_le _ _ q).trans
      (dense_abs_le zero_le_one (fun i j => hW₅ j i) (fun _ => by simp) hHeadMag q)
  have hc3 : ∀ l, |FloatModel.reluMask Z3F (M.dense (fun j i' => W₄ i' j)
        (fun _ => 0) (FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j)
          (fun _ => 0) (M.softmaxCECotF fexp Z5F label)))) l -
      FloatModel.reluMask Z3 (dense (fun j i' => W₄ i' j) (fun _ => 0)
        (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
          (fun k => softmax nC Z5 k - oneHot nC label k)))) l| ≤ ec3 := fun l =>
    M.cot_step_close W₄ Z3F Z3
      (FloatModel.reluMask Z4F (M.dense (fun j i' => W₅ i' j) (fun _ => 0)
        (M.softmaxCECotF fexp Z5F label)))
      (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j) (fun _ => 0)
        (fun k => softmax nC Z5 k - oneHot nC label k)))
      hw₄ C4nn ec4nn hW₄ hc4Mag hc4 hE3close hmargin3 l
  have hc3Mag : ∀ l, |FloatModel.reluMask Z3 (dense (fun j i' => W₄ i' j)
      (fun _ => 0) (FloatModel.reluMask Z4 (dense (fun j i' => W₅ i' j)
        (fun _ => 0) (fun k => softmax nC Z5 k - oneHot nC label k)))) l| ≤ C3 :=
    fun l => (reluMask_abs_le _ _ l).trans
      (dense_abs_le C4nn (fun i j => hW₄ j i) (fun _ => by simp) hc4Mag l)
  have hcPool : ∀ j, |M.dense (fun j' i' => W₃ i' j') (fun _ => 0)
        (FloatModel.reluMask Z3F (M.dense (fun j' i' => W₄ i' j') (fun _ => 0)
          (FloatModel.reluMask Z4F (M.dense (fun j' i' => W₅ i' j') (fun _ => 0)
            (M.softmaxCECotF fexp Z5F label))))) j -
      dense (fun j' i' => W₃ i' j') (fun _ => 0)
        (FloatModel.reluMask Z3 (dense (fun j' i' => W₄ i' j') (fun _ => 0)
          (FloatModel.reluMask Z4 (dense (fun j' i' => W₅ i' j') (fun _ => 0)
            (fun k => softmax nC Z5 k - oneHot nC label k))))) j| ≤
      FloatModel.layerBudget M.u d₃ w₃ 0 C3 ec3 := fun j =>
    (M.dense_close (fun j' i' => W₃ i' j') (fun _ => 0) _ _ ec3 ec3nn hc3 j).trans
      (M.denseErr_le_uniform hw₃ ec3nn (fun i j' => hW₃ j' i) (fun _ => by simp)
        hc3Mag j)
  -- pool freeze + conv-2 ReLU mask freeze → the per-cell cotangent close
  have hPostRelu : ∀ ci hi wi,
      |Tensor3.unflatten (relu (c * (2*h) * (2*w)) Z2CF) ci hi wi -
        Tensor3.unflatten (relu (c * (2*h) * (2*w)) Z2C) ci hi wi| ≤ E2 := by
    intro ci hi wi; rw [unflatten_t3Idx, unflatten_t3Idx]
    exact hRelu (t3Idx ci hi wi)
  rw [FloatModel.cnnConv2CotBudget]
  have hpb := hmarginPool.poolBack_close hPostRelu co ho wo
    (hcPool (t3Idx co (winRow ho) (winCol wo)))
  exact mask_scalar_close (hEconv (t3Idx co ho wo)) (hmarginConv (t3Idx co ho wo))
    hpb ecvnn

open FloatModel in
/-- **The real conv-2-output cotangent is magnitude-bounded** by `cnnConv2CotMag`
    — the `aX2`/`eX2`-independent ℓ∞ bound (the conv-2 ReLU mask and pool
    selector only shrink, the head cotangent is in `[−1,1]`, the two masked `Wᵀ`
    steps and the unmasked W₃ ride `layerAct`). Used to bound the real conv-1
    cotangent `∑ convTap·c₂` in the conv-1 rung. -/
theorem cnn_conv2_cot_real_abs_le {c h w d₃ d₄ nC kH kW : Nat}
    (X2 : Tensor3 c (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    {w₃ w₄ w₅ : ℝ} (hw₃ : 0 ≤ w₃) (hw₄ : 0 ≤ w₄) (hw₅ : 0 ≤ w₅)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) :
    |(if Tensor3.flatten (conv2d W₂ b₂ X2) (t3Idx co ho wo) > 0
          then (1:ℝ) else 0) *
        (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ X2)))) co ho wo
          then dense (fun j i' => W₃ i' j) (fun _ => 0)
            (FloatModel.reluMask (dense W₃ b₃ (maxPoolFlat c h w
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ X2)))))
              (dense (fun j i' => W₄ i' j) (fun _ => 0)
                (FloatModel.reluMask (dense W₄ b₄ (relu d₃
                    (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                      (Tensor3.flatten (conv2d W₂ b₂ X2)))))))
                  (dense (fun j i' => W₅ i' j) (fun _ => 0)
                    (fun k => softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                        (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu
                          (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ X2))))))))) k - oneHot nC label k)))))
            (t3Idx co (winRow ho) (winCol wo))
          else 0)| ≤ FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ := by
  rw [FloatModel.cnnConv2CotMag]
  set PR := maxPoolFlat c h w (relu (c * (2*h) * (2*w))
    (Tensor3.flatten (conv2d W₂ b₂ X2))) with hPR
  set Z5 := dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ PR)))) with hZ5
  have hHeadMag : ∀ k, |softmax nC Z5 k - oneHot nC label k| ≤ 1 := by
    intro k
    have hD : 0 < ∑ t, Real.exp (Z5 t) :=
      Finset.sum_pos (fun t _ => Real.exp_pos _) ⟨k, Finset.mem_univ k⟩
    have hs0 : 0 ≤ softmax nC Z5 k :=
      div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun t _ => (Real.exp_pos _).le)
    have hs1 : softmax nC Z5 k ≤ 1 :=
      (div_le_one hD).mpr
        (Finset.single_le_sum (fun t _ => (Real.exp_pos _).le) (Finset.mem_univ k))
    simp only [oneHot]
    by_cases hkl : k = label
    · rw [if_pos hkl, abs_le]; constructor <;> linarith
    · rw [if_neg hkl, abs_le]; constructor <;> linarith
  have hc4Mag : ∀ q, |FloatModel.reluMask
      (dense W₄ b₄ (relu d₃ (dense W₃ b₃ PR))) (dense (fun j i' => W₅ i' j)
      (fun _ => 0) (fun k => softmax nC Z5 k - oneHot nC label k)) q| ≤
      FloatModel.layerAct nC w₅ 0 1 := fun q => (reluMask_abs_le _ _ q).trans
    (dense_abs_le zero_le_one (fun i j => hW₅ j i) (fun _ => by simp) hHeadMag q)
  have hc3Mag : ∀ l, |FloatModel.reluMask (dense W₃ b₃ PR)
      (dense (fun j i' => W₄ i' j) (fun _ => 0)
        (FloatModel.reluMask (dense W₄ b₄ (relu d₃ (dense W₃ b₃ PR)))
          (dense (fun j i' => W₅ i' j) (fun _ => 0)
            (fun k => softmax nC Z5 k - oneHot nC label k)))) l| ≤
      FloatModel.layerAct d₄ w₄ 0 (FloatModel.layerAct nC w₅ 0 1) :=
    fun l => (reluMask_abs_le _ _ l).trans
      (dense_abs_le (layerAct_nonneg hw₅ le_rfl zero_le_one) (fun i j => hW₄ j i)
        (fun _ => by simp) hc4Mag l)
  have hcPoolMag : ∀ j, |dense (fun j' i' => W₃ i' j') (fun _ => 0)
      (FloatModel.reluMask (dense W₃ b₃ PR) (dense (fun j' i' => W₄ i' j')
        (fun _ => 0) (FloatModel.reluMask (dense W₄ b₄ (relu d₃ (dense W₃ b₃ PR)))
          (dense (fun j' i' => W₅ i' j') (fun _ => 0)
            (fun k => softmax nC Z5 k - oneHot nC label k))))) j| ≤
      FloatModel.layerAct d₃ w₃ 0 (FloatModel.layerAct d₄ w₄ 0
        (FloatModel.layerAct nC w₅ 0 1)) :=
    fun j => dense_abs_le (layerAct_nonneg hw₄ le_rfl
      (layerAct_nonneg hw₅ le_rfl zero_le_one)) (fun i j' => hW₃ j' i)
      (fun _ => by simp) hc3Mag j
  by_cases hz : Tensor3.flatten (conv2d W₂ b₂ X2) (t3Idx co ho wo) > 0
  · rw [if_pos hz, one_mul]
    split_ifs with hA
    · exact hcPoolMag (t3Idx co (winRow ho) (winCol wo))
    · simpa using FloatModel.layerAct_nonneg hw₃ le_rfl (FloatModel.layerAct_nonneg hw₄
        le_rfl (FloatModel.layerAct_nonneg hw₅ le_rfl zero_le_one))
  · rw [if_neg hz, zero_mul, abs_zero]
    exact FloatModel.layerAct_nonneg hw₃ le_rfl (FloatModel.layerAct_nonneg hw₄
      le_rfl (FloatModel.layerAct_nonneg hw₅ le_rfl zero_le_one))

/-- `|a| ≤ C + e` from `|a − b| ≤ e` and `|b| ≤ C` — lifts a closeness + a
    base magnitude to a float magnitude (the float cotangent bound from the
    real bound plus the drift). -/
theorem abs_le_of_close {a b e C : ℝ} (h1 : |a - b| ≤ e) (h2 : |b| ≤ C) :
    |a| ≤ C + e := by
  have := abs_sub_abs_le_abs_sub a b
  linarith

/-- **The float conv-2 backward (transpose conv) against a perturbed
    cotangent.** The rounded `M.dot` of the exact `convTap` slab against the
    float conv-2-output cotangent `c2F`, vs the certified `∑ convTap·c2R` — the
    `convTap`-flattening (`sum_t3`) plus `dot_perturbed_close` (fan-in
    `c·(2h)·(2w)`, per-entry tap bound `w₂`, float-cotangent magnitude `C2t`,
    drift `e₂`). Generic in `(c2F, c2R)` so the conv-1 rung passes the conv-2
    cotangent tensors abstractly. -/
theorem convTap_back_close {c h w kH kW : Nat} (M : FloatModel)
    (W₂ : Kernel4 c c kH kW) (c2F c2R : Tensor3 c (2*h) (2*w))
    {w₂ C2t e2 : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hc2F : ∀ co ho wo, |c2F co ho wo| ≤ C2t)
    (hc2close : ∀ co ho wo, |c2F co ho wo - c2R co ho wo| ≤ e2)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |M.dot (Tensor3.flatten (fun co ho wo => convTap W₂ ci hi wi co ho wo))
        (Tensor3.flatten c2F) -
      ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
        convTap W₂ ci hi wi co ho wo * c2R co ho wo| ≤
      ((1 + M.u) ^ ((c * (2*h) * (2*w)) + 1) - 1) *
          (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * C2t)) +
        (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * e2)) := by
  have htap_eq : (∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
        convTap W₂ ci hi wi co ho wo * c2R co ho wo) =
      ∑ s, (Tensor3.flatten (fun co ho wo => convTap W₂ ci hi wi co ho wo)) s *
        (Tensor3.flatten c2R) s := by
    rw [sum_t3 (fun s => (Tensor3.flatten
      (fun co ho wo => convTap W₂ ci hi wi co ho wo)) s * (Tensor3.flatten c2R) s)]
    refine Finset.sum_congr rfl fun co _ => Finset.sum_congr rfl fun ho _ =>
      Finset.sum_congr rfl fun wo _ => ?_
    rw [flatten_t3Idx, flatten_t3Idx]
  rw [htap_eq]
  exact M.dot_perturbed_close
    (Tensor3.flatten (fun co ho wo => convTap W₂ ci hi wi co ho wo))
    (Tensor3.flatten c2F) (Tensor3.flatten c2R) hw₂
    (fun s => by obtain ⟨co, ho, wo, rfl⟩ := t3Idx_surj s
                 rw [flatten_t3Idx]; exact convTap_abs_le hw₂ hW₂ ci hi wi co ho wo)
    (fun s => by obtain ⟨co, ho, wo, rfl⟩ := t3Idx_surj s
                 rw [flatten_t3Idx]; exact hc2F co ho wo)
    (fun s => by obtain ⟨co, ho, wo, rfl⟩ := t3Idx_surj s
                 rw [flatten_t3Idx, flatten_t3Idx]; exact hc2close co ho wo)

/-- The real conv-2 backward `∑ convTap·c2R` is magnitude-bounded by the tap
    ℓ∞-mass `(c·(2h)·(2w))·w₂` times the cotangent bound `CP` — the (loose,
    uniform) bound on the real conv-1 cotangent. -/
theorem convTap_back_abs_le {c h w kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (c2R : Tensor3 c (2*h) (2*w))
    {w₂ CP : ℝ} (hw₂ : 0 ≤ w₂)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hc2R : ∀ co ho wo, |c2R co ho wo| ≤ CP)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
        convTap W₂ ci hi wi co ho wo * c2R co ho wo| ≤
      ((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * CP) := by
  have hbound : (∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
      |convTap W₂ ci hi wi co ho wo * c2R co ho wo|) ≤
      ((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * CP) := by
    calc (∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
            |convTap W₂ ci hi wi co ho wo * c2R co ho wo|)
        ≤ ∑ _co : Fin c, ∑ _ho : Fin (2*h), ∑ _wo : Fin (2*w), w₂ * CP := by
          refine Finset.sum_le_sum fun co _ => Finset.sum_le_sum fun ho _ =>
            Finset.sum_le_sum fun wo _ => ?_
          rw [abs_mul]
          exact mul_le_mul (convTap_abs_le hw₂ hW₂ ci hi wi co ho wo)
            (hc2R co ho wo) (abs_nonneg _) hw₂
      _ = ((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * CP) := by
          simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
            nsmul_eq_mul]
          push_cast; ring
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  refine (Finset.sum_le_sum fun co _ => Finset.abs_sum_le_sum_abs _ _).trans ?_
  refine (Finset.sum_le_sum fun co _ => Finset.sum_le_sum fun ho _ =>
    Finset.abs_sum_le_sum_abs _ _).trans hbound

open FloatModel in
/-- **The binary32 conv-1 weight gradient is within an explicit budget of the
    certified one** (Increment 4 capstone) — the conv-1 peer of
    `cnn_conv2_grad_close`, one conv-backward deeper. With `x₀` exact, the
    rendered trainer's `W₁` gradient `M.cnnConv1FloatGrad …` stays within
    `cnnConv1GradBudget`. The conv-2 cotangent chain is reused at a FLOAT conv-2
    input `relu(z̃₁)` (`cnn_conv2_cot_close`); the conv-2 backward is a rounded
    dot of the (exact) `convTap` slab against the float conv-2 cotangent slab
    (`dot_perturbed_close` over `c·(2h)·(2w)`); the conv-1 ReLU mask freezes
    (`mask_scalar_close`); the conv-1 weight dot rounds it
    (`dot_perturbed_close` over `(2h)·(2w)`). Five quantitative margins are
    carried; the bridge `cnn_conv1_loss_gradAt_reluMask` turns the `gradAt`
    into the dot. -/
theorem cnn_conv1_grad_close {ic c h w d₃ d₄ nC kH kW : Nat} (M : FloatModel)
    (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄)
    (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    (u : Vec (c * ic * kH * kW))
    {a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (hh : 0 < h) (hw : 0 < w)
    (ha : 0 ≤ a) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅)
    (hβ₅ : 0 ≤ β₅) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx₀ : ∀ ci i j, |x₀ ci i j| ≤ a)
    (hu1 : ∀ idx, |u idx| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hmargin1 : ∀ k, FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 <
      |Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k|)
    (hmargin2 : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀))))) k|)
    (hmarginPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))
    (hmargin3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
          (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l|)
    (hmargin4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃ (FloatModel.layerAct (c * kH * kW)
          w₂ β₂ (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
            (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q|)
    (o : Fin c) (cc : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    |M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label u
        (k4Idx o cc kh kw) -
      gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label) u (k4Idx o cc kh kw)|
      ≤ M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
          w₅ β₅ eexp := by
  have hc : 0 < c := Fin.pos o
  have hu2' : ∀ o' c' kh' kw', |Kernel4.unflatten u o' c' kh' kw'| ≤ w₁ :=
    fun o' c' kh' kw' => by rw [unflatten_k4Idx]; exact hu1 _
  -- off-kink + smooth conditions from the quantitative margins
  have hz1 : ∀ k, Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k ≠ 0 :=
    fun k => abs_pos.mp (lt_of_le_of_lt
      (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl) (hmargin1 k))
  have hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten u) b₁ x₀))))) k ≠ 0 :=
    fun k => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₂ hβ₂
      (layerAct_nonneg hw₁ hβ₁ ha)
      (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl)) (hmargin2 k))
  have hmp := hmarginPool.smooth (layerBudget_nonneg M.u_nonneg hw₂ hβ₂
    (layerAct_nonneg hw₁ hβ₁ ha)
    (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl))
  have hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l ≠ 0 :=
    fun l => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₃ hβ₃
      (layerAct_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha))
      (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha)
        (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl))) (hmargin3 l))
  have hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))))))) ) q ≠ 0 :=
    fun q => abs_pos.mp (lt_of_le_of_lt (layerBudget_nonneg M.u_nonneg hw₄ hβ₄
      (layerAct_nonneg hw₃ hβ₃ (layerAct_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha)))
      (layerBudget_nonneg M.u_nonneg hw₃ hβ₃
        (layerAct_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha))
        (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ (layerAct_nonneg hw₁ hβ₁ ha)
          (layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl)))) (hmargin4 q))
  rw [M.cnnConv1FloatGrad_apply b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label u
      o cc kh kw,
    cnn_conv1_loss_gradAt_reluMask b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw u
      hz1 hz2 hmp hz3 hz4 o cc kh kw]
  -- abbreviate the conv-1 forward and the conv-2 input
  set Z1C := Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) with hZ1C
  set Z1CF := Tensor3.flatten (M.convF (Kernel4.unflatten u) b₁ x₀) with hZ1CF
  set X2 := Tensor3.unflatten (relu (c * (2*h) * (2*w)) Z1C) with hX2
  set X2F := Tensor3.unflatten (relu (c * (2*h) * (2*w)) Z1CF) with hX2F
  -- budgets
  set A1 := FloatModel.layerAct (ic * kH * kW) w₁ β₁ a with hA1
  set E1 := FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 with hE1
  set e2 := M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW A1 E1 w₂ β₂ w₃ β₃ w₄ β₄ w₅
    β₅ eexp with he2
  set CP := FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ with hCP
  set eback := ((1 + M.u) ^ ((c * (2*h) * (2*w)) + 1) - 1) *
      (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * (CP + e2))) +
      (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * e2)) with heback
  have hA1nn : 0 ≤ A1 := layerAct_nonneg hw₁ hβ₁ ha
  have hE1nn : 0 ≤ E1 := layerBudget_nonneg M.u_nonneg hw₁ hβ₁ ha le_rfl
  have hCPnn : 0 ≤ CP := by
    rw [hCP, FloatModel.cnnConv2CotMag]
    exact layerAct_nonneg hw₃ le_rfl (layerAct_nonneg hw₄ le_rfl
      (layerAct_nonneg hw₅ le_rfl zero_le_one))
  have he2nn : 0 ≤ e2 := by
    rw [he2, FloatModel.cnnConv2CotBudget]
    exact layerBudget_nonneg M.u_nonneg hw₃ le_rfl
      (layerAct_nonneg hw₄ le_rfl (layerAct_nonneg hw₅ le_rfl zero_le_one))
      (layerBudget_nonneg M.u_nonneg hw₄ le_rfl
        (layerAct_nonneg hw₅ le_rfl zero_le_one)
        (layerBudget_nonneg M.u_nonneg hw₅ le_rfl zero_le_one
          (M.cotErr_nonneg heexp0 (layerBudget_nonneg M.u_nonneg hw₅ hβ₅
            (layerAct_nonneg hw₄ hβ₄ (layerAct_nonneg hw₃ hβ₃
              (layerAct_nonneg hw₂ hβ₂ hA1nn)))
            (layerBudget_nonneg M.u_nonneg hw₄ hβ₄ (layerAct_nonneg hw₃ hβ₃
              (layerAct_nonneg hw₂ hβ₂ hA1nn))
              (layerBudget_nonneg M.u_nonneg hw₃ hβ₃
                (layerAct_nonneg hw₂ hβ₂ hA1nn)
                (layerBudget_nonneg M.u_nonneg hw₂ hβ₂ hA1nn hE1nn)))) hρ1)))
  have hebacknn : 0 ≤ eback := by
    rw [heback]
    have hγ : (0:ℝ) ≤ (1 + M.u) ^ ((c * (2*h) * (2*w)) + 1) - 1 :=
      sub_nonneg.mpr (one_le_pow₀ (by linarith [M.u_nonneg]))
    have hn : (0:ℝ) ≤ ((c * (2*h) * (2*w) : ℕ) : ℝ) := Nat.cast_nonneg _
    exact add_nonneg (mul_nonneg hγ (mul_nonneg hn (mul_nonneg hw₂
      (add_nonneg hCPnn he2nn)))) (mul_nonneg hn (mul_nonneg hw₂ he2nn))
  -- conv-1 forward closeness, conv-2 input closeness + magnitude
  have hZ1close : ∀ k, |Z1CF k - Z1C k| ≤ E1 := by
    intro k; obtain ⟨ci, hi, wi, rfl⟩ := t3Idx_surj k
    rw [hZ1CF, hZ1C, flatten_t3Idx, flatten_t3Idx]
    exact (M.convF_close (Kernel4.unflatten u) b₁ x₀ x₀ le_rfl
        (fun _ _ _ => by simp) ci hi wi).trans
      (M.denseErr_le_uniform hw₁ le_rfl (fun i j => convKernelMat_abs_le hu2' i j)
        hb₁ (fun idx => convWindow_abs_le ha hx₀ hi wi idx) ci)
  have hX2close : ∀ co i j, |X2F co i j - X2 co i j| ≤ E1 := by
    intro co i j; rw [hX2F, hX2, unflatten_t3Idx, unflatten_t3Idx]
    exact relu_close _ _ _ hZ1close (t3Idx co i j)
  have hX2mag : ∀ co i j, |X2 co i j| ≤ A1 := by
    intro co i j; rw [hX2, unflatten_t3Idx]
    refine (relu_abs_le _ _).trans ?_
    rw [hZ1C, flatten_t3Idx]; exact conv2d_abs_le ha hu2' hb₁ hx₀ co i j
  -- conv-2 cotangent closeness / magnitudes (factored, at the float conv-2 input)
  have hc2close := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    cnn_conv2_cot_close M X2 X2F W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label fexp hA1nn hE1nn
      hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅ heexp0 heexp1 hfexp hρ1 hX2close hX2mag
      hW₂ hb₂ hW₃ hb₃ hW₄ hb₄ hW₅ hb₅ hmargin2 hmarginPool hmargin3 hmargin4 co ho wo
  have hc2realmag := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    cnn_conv2_cot_real_abs_le X2 W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label hw₃ hw₄ hw₅
      hW₃ hW₄ hW₅ co ho wo
  have hc2floatmag := fun (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)) =>
    abs_le_of_close (hc2close co ho wo) (hc2realmag co ho wo)
  -- the conv-1 weight dot (conv-1 ReLU mask freeze + conv-2 backward dot)
  simp only [FloatModel.cnnConv1GradBudget]
  refine M.dot_perturbed_close (convPadWin kH kW x₀ cc kh kw) _ _ ha
    (fun s => by simp only [convPadWin]; exact abs_convPad_le x₀ ha hx₀ cc kh kw _ _)
    (fun s => by
      simp only [cotWin]; rw [abs_mul]
      refine le_trans (mul_le_mul (by split_ifs <;> simp) ?_ (abs_nonneg _)
        zero_le_one) (le_of_eq (one_mul _))
      exact abs_le_of_close
        (convTap_back_close M W₂ _ _ hw₂ hW₂ hc2floatmag hc2close o _ _)
        (convTap_back_abs_le W₂ _ hw₂ hW₂ hc2realmag o _ _))
    (fun s => by
      simp only [cotWin]
      exact mask_scalar_close (hZ1close _) (hmargin1 _)
        (convTap_back_close M W₂ _ _ hw₂ hW₂ hc2floatmag hc2close o _ _) hebacknn)

-- ════════════════════════════════════════════════════════════════
-- § Segment-Lipschitz gradient for the conv1 loss, explicit constant
-- ════════════════════════════════════════════════════════════════

/-- **Segment-Lipschitz gradient for the conv1-kernel loss, explicit
    constant.** Under the FIVE margins at step radius `D` — relu₁
    (`a·D`), relu₂ (`c·kH·kW·w₂·a·D`), pool selection (same radius,
    POST-relu₂), relu₃, relu₄ — every routing decision freezes along
    `[u, u+d]`, BOTH conv Jacobians factor out point-free, and the
    difference collapses to the softmax drift. The constant picks up the
    conv1 weight-sharing multiplicity `((2h)·(2w))²` AND the conv2
    locality multiplicity `(c·kH·kW)²·w₂²`. -/
theorem cnn_conv1_loss_grad_lipschitz {ic c h w d₃ d₄ nC kH kW : Nat}
    (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW)
    (b₂ : Vec c) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC) (hh : 0 < h) (hw : 0 < w)
    {a w₂ w₃ w₄ w₅ D : ℝ} (ha : 0 ≤ a) (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (u d : Vec (c * ic * kH * kW)) (hd : (∑ idx, |d idx|) ≤ D)
    (hm1 : ∀ k, a * D <
      |Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * D)) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))) k|)
    (hmq : MaxPool2MarginQ (((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * D)))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * D))))))))) < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1)
    (idx : Fin (c * ic * kH * kW)) :
    |gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
        (u + t • d) idx -
      gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
        u idx| ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 *
        ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 *
        w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            (a * D))))))))))) * (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D :=
    le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  have haD0 : 0 ≤ a * D := mul_nonneg ha hD0
  have hδ0 : (0:ℝ) ≤ w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * D)))))))) :=
    mul_nonneg hw₅ (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
        (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₂
          (mul_nonneg (Nat.cast_nonneg _) haD0)))))))
  have hden : (0:ℝ) < 1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * D))))))))) := by linarith
  obtain ⟨p1, rfl⟩ := finProdFinEquiv.surjective idx
  obtain ⟨p2, kw⟩ := p1
  obtain ⟨p3, rfl⟩ := finProdFinEquiv.surjective p2
  obtain ⟨p4, kh⟩ := p3
  obtain ⟨p5, rfl⟩ := finProdFinEquiv.surjective p4
  obtain ⟨o, cc⟩ := p5
  rw [show finProdFinEquiv (finProdFinEquiv (finProdFinEquiv (o, cc), kh),
        kw) = k4Idx o cc kh kw from rfl]
  have hKw0 : (0:ℝ) ≤ ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * D)) :=
    mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₂ haD0)
  -- base-point conditions from the margins
  have hz1_v : ∀ k,
      Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀) k ≠ 0 :=
    fun k h0 => by
      have hk := hm1 k
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr haD0)
  have hz2_v : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten u) b₁ x₀))))) k ≠ 0 :=
    fun k h0 => by
      have hk := hm2 k
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr hKw0)
  have hmp_v : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))))) :
      Tensor3 c (2*h) (2*w)) := hmq.smooth hKw0
  have hz3_v : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l ≠ 0 :=
    fun l h0 => by
      have hk := hm3 l
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr (mul_nonneg hw₃ (mul_nonneg
        (Nat.cast_nonneg _) (mul_nonneg hw₂ (mul_nonneg
          (Nat.cast_nonneg _) haD0)))))
  have hz4_v : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q ≠ 0 :=
    fun q h0 => by
      have hk := hm4 q
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr (mul_nonneg hw₄ (mul_nonneg
        (Nat.cast_nonneg _) (mul_nonneg hw₃ (mul_nonneg
          (Nat.cast_nonneg _) (mul_nonneg hw₂ (mul_nonneg
            (Nat.cast_nonneg _) haD0)))))))
  -- segment-point conditions: everything frozen
  have hstab1 := fun k =>
    cnn1_margin1_keeps_offkink b₁ x₀ ha hx u d hd hm1 t ht0 ht1 k
  have hz1_t : ∀ k, Tensor3.flatten
      (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀) k ≠ 0 :=
    fun k => (hstab1 k).1
  have hstab2 := fun k =>
    cnn1_margin2_keeps_offkink b₁ x₀ W₂ b₂ ha hx hw₂ hW₂ u d hd hm2
      t ht0 ht1 k
  have hz2_t : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀))))) k ≠ 0 :=
    fun k => (hstab2 k).1
  have hclose := fun ci hi wi =>
    cnn1_postrelu2_close_seg b₁ x₀ W₂ b₂ ha hx hw₂ hW₂ u d hd
      t ht0 ht1 ci hi wi
  have hmp_t : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀))))))) :
      Tensor3 c (2*h) (2*w)) := hmq.smooth_of_close hclose
  have hstab3 := fun l =>
    cnn1_margin3_keeps_offkink b₁ x₀ W₂ b₂ W₃ b₃ ha hx hw₂ hW₂ hw₃ hW₃
      u d hd hm3 t ht0 ht1 l
  have hz3_t : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀)))))))) l ≠ 0 :=
    fun l => (hstab3 l).1
  have hstab4 := fun q =>
    cnn1_margin4_keeps_offkink b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ ha hx hw₂ hW₂
      hw₃ hW₃ hw₄ hW₄ u d hd hm4 t ht0 ht1 q
  have hz4_t : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀)))))))))) q ≠ 0 :=
    fun q => (hstab4 q).1
  -- both gradients in closed form
  rw [cnn_conv1_loss_gradAt b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw
      (u + t • d) hz1_t hz2_t hmp_t hz3_t hz4_t o cc kh kw,
    cnn_conv1_loss_gradAt b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw
      u hz1_v hz2_v hmp_v hz3_v hz4_v o cc kh kw]
  -- the frozen masks and the frozen routing
  have hmask1 : ∀ (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)),
      (if Tensor3.flatten (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) =
      (if Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) := by
    intro ci hi wi
    by_cases hp : Tensor3.flatten (conv2d (Kernel4.unflatten u) b₁ x₀)
        (t3Idx ci hi wi) > 0
    · rw [if_pos ((hstab1 _).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab1 _).2.mp hgt)), if_neg hp]
  have hmask2 : ∀ k : Fin (c * (2*h) * (2*w)),
      (if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀))))) k > 0
        then (1:ℝ) else 0) =
      (if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀))))) k > 0
        then (1:ℝ) else 0) := by
    intro k
    by_cases hp : Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten u) b₁ x₀))))) k > 0
    · rw [if_pos ((hstab2 _).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab2 _).2.mp hgt)), if_neg hp]
  have hmask3 : ∀ l : Fin d₃,
      (if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀))))))))
          l > 0 then (1:ℝ) else 0) =
      (if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten u) b₁ x₀))))))))
          l > 0 then (1:ℝ) else 0) := by
    intro l
    by_cases hp : dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l > 0
    · rw [if_pos ((hstab3 l).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab3 l).2.mp hgt)), if_neg hp]
  have hmask4 : ∀ q : Fin d₄,
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀))))))))))
          q > 0 then (1:ℝ) else 0) =
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten u) b₁ x₀))))))))))
          q > 0 then (1:ℝ) else 0) := by
    intro q
    by_cases hp : dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀)))))))))) q > 0
    · rw [if_pos ((hstab4 q).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab4 q).2.mp hgt)), if_neg hp]
  have hargiff : ∀ (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)),
      MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀))))))))
        co ho wo ↔
      MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀))))))))
        co ho wo :=
    fun co ho wo => hmq.isArgmax_iff hclose co ho wo
  -- the softmax drift along the segment
  have hzdrift : ∀ k, |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + t • d)) b₁ x₀)))))))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten u)
              b₁ x₀)))))))))))) k| ≤
      t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * D))))))))) := by
    intro k
    have h1 := cnn1_logit_drift b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ ha hx
      hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ u (t • d) k
    rw [smul_l1_mass d ht0] at h1
    have h2 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * (t * ∑ idx, |d idx|))))))))) =
        t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            (a * ∑ idx, |d idx|))))))))) := by
      ring
    rw [h2] at h1
    have h3 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * ∑ idx, |d idx|)))))))) ≤
        w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            (a * D)))))))) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
              (mul_le_mul_of_nonneg_left hd ha) (Nat.cast_nonneg _)) hw₂)
            (Nat.cast_nonneg _)) hw₃) (Nat.cast_nonneg _)) hw₄)
        (Nat.cast_nonneg _)) hw₅
    have h4 := mul_le_mul_of_nonneg_left h3 ht0
    linarith
  have hδlt : 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * D)))))))))) < 1 := by
    nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
  have hexp := FloatModel.exp_sub_one_le hδlt
  have hmono : 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * D)))))))))) /
        (1 - 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            (a * D))))))))))) ≤
      2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * D)))))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            (a * D)))))))))) := by
    refine div_le_div_of_nonneg_left
      (by nlinarith [mul_nonneg ht0 hδ0]) hden ?_
    nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
  have hS : ∀ k, |softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (u + t • d))
              b₁ x₀))))))))))))) k -
      softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d (Kernel4.unflatten u)
              b₁ x₀))))))))))))) k| ≤
      2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * D)))))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            (a * D)))))))))) :=
    fun k => le_trans (FloatModel.softmax_perturb _ _ hzdrift k)
      (le_trans hexp hmono)
  have hΔ0 : (0:ℝ) ≤ 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * D)))))))))) /
      (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * D)))))))))) :=
    div_nonneg (mul_nonneg (by norm_num) (mul_nonneg ht0 hδ0)) hden.le
  have hM0 : (0:ℝ) ≤ (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
      (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (a * D)))))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            (a * D)))))))))))))))) :=
    mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
        (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₅ hΔ0)))))
  -- the conv1 Jacobian row mass
  have hcp : ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
      |if ci = o then convPad kH kW x₀ cc kh kw hi wi else 0| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * a := by
    rw [Finset.sum_eq_single o
      (fun ci _ hne => by
        rw [Finset.sum_eq_zero]
        intro hi _
        rw [Finset.sum_eq_zero]
        intro wi _
        rw [if_neg hne, abs_zero])
      (fun habs => absurd (Finset.mem_univ _) habs)]
    calc ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          |if o = o then convPad kH kW x₀ cc kh kw hi wi else 0|
        ≤ ∑ _hi : Fin (2*h), ∑ _wi : Fin (2*w), a := by
          refine Finset.sum_le_sum fun hi _ =>
            Finset.sum_le_sum fun wi _ => ?_
          rw [if_pos rfl]
          exact abs_convPad_le x₀ ha hx cc kh kw hi wi
      _ = ((2*h * (2*w) : ℕ) : ℝ) * a := by
          rw [Finset.sum_const, Finset.sum_const, Finset.card_univ,
            Finset.card_univ, Fintype.card_fin, Fintype.card_fin,
            smul_smul, nsmul_eq_mul]
  -- the endgame
  have hfinal : ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
      (|if ci = o then convPad kH kW x₀ cc kh kw hi wi else 0| *
        (((c * kH * kW : ℕ) : ℝ) * w₂ *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))))))))))))) ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 *
        ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 *
        w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            (a * D))))))))))) * (t * D) := by
    calc ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
        (|if ci = o then convPad kH kW x₀ cc kh kw hi wi else 0| *
          (((c * kH * kW : ℕ) : ℝ) * w₂ *
            ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
              (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))) /
                (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                  (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                    (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))))))))))))
        = (∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
            |if ci = o then convPad kH kW x₀ cc kh kw hi wi else 0|) *
            (((c * kH * kW : ℕ) : ℝ) * w₂ *
              ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
                (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) *
                  (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                    (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))) /
                  (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                    (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                      (((2*h * (2*w) : ℕ) : ℝ) *
                        (a * D)))))))))))))))))) := by
          simp only [← Finset.sum_mul]
      _ ≤ (((2*h * (2*w) : ℕ) : ℝ) * a) *
            (((c * kH * kW : ℕ) : ℝ) * w₂ *
              ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
                (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) *
                  (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                    (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))) /
                  (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                    (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                      (((2*h * (2*w) : ℕ) : ℝ) *
                        (a * D)))))))))))))))))) :=
          mul_le_mul_of_nonneg_right hcp
            (mul_nonneg (mul_nonneg (Nat.cast_nonneg _) hw₂) hM0)
      _ = (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 *
            ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 *
            w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))))) *
            (t * D) := by
          ring
  refine le_trans (le_trans (abs_triple_sum_sub_le _ _)
    (Finset.sum_le_sum fun ci _ => Finset.sum_le_sum fun hi _ =>
      Finset.sum_le_sum fun wi _ => ?_)) hfinal
  -- per-term: freeze relu₁'s mask, then bound the conv2 contraction
  rw [hmask1 ci hi wi]
  simp only [hmask2, hmask3, hmask4]
  rw [← mul_sub, abs_mul, ← mul_sub, abs_mul]
  refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
  refine le_trans (mul_le_of_le_one_left (abs_nonneg _) ?_) ?_
  · split_ifs <;> simp
  -- the conv2 contraction: point-free taps times the frozen-route drift
  have hlast := calc ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
      (|convTap W₂ ci hi wi co ho wo| *
        ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
          (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
            (((c * kH * kW : ℕ) : ℝ) * (w₂ *
              (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))) /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D))))))))))))))))))
      = (∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
          |convTap W₂ ci hi wi co ho wo|) *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) *
                    (a * D))))))))))))))))) := by
                        simp only [← Finset.sum_mul]
    _ ≤ (((c * kH * kW : ℕ) : ℝ) * w₂) *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) *
                    (a * D))))))))))))))))) :=
        mul_le_mul_of_nonneg_right
          (convTap_out_l1 W₂ hW₂ ci hi wi) hM0
    _ = ((c * kH * kW : ℕ) : ℝ) * w₂ *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) *
                    (a * D))))))))))))))))) := by
          ring
  refine le_trans (abs_triple_sum_sub_le _ _) ?_
  refine le_trans (Finset.sum_le_sum fun co _ => Finset.sum_le_sum
    fun ho _ => Finset.sum_le_sum fun wo _ => ?_) hlast
  show |convTap W₂ ci hi wi co ho wo * _ -
        convTap W₂ ci hi wi co ho wo * _| ≤
      |convTap W₂ ci hi wi co ho wo| *
        ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
          (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
            (((c * kH * kW : ℕ) : ℝ) * (w₂ *
              (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))) /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * (a * D)))))))))))))))))
  rw [← mul_sub, abs_mul]
  refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
  by_cases hA : MaxPool2IsArgmax (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten u) b₁ x₀))))))))
      co ho wo
  · rw [if_pos ((hargiff co ho wo).mpr hA), if_pos hA, ← mul_sub,
      abs_mul]
    refine le_trans (mul_le_of_le_one_left (abs_nonneg _) ?_) ?_
    · split_ifs <;> simp
    · exact head3_sum_drift W₃ W₄ W₅ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
        (fun l => if dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u) b₁ x₀)))))))) l > 0
          then (1:ℝ) else 0)
        (fun l => by split_ifs <;> simp)
        (fun q => if dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u)
                  b₁ x₀)))))))))) q > 0
          then (1:ℝ) else 0)
        (fun q => by split_ifs <;> simp)
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d (Kernel4.unflatten u)
                    b₁ x₀))))))))))))))
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d (Kernel4.unflatten (u + t • d))
                    b₁ x₀))))))))))))))
        (oneHot nC label) hS (t3Idx co (winRow ho) (winCol wo))
  · rw [if_neg (fun hA' => hA ((hargiff co ho wo).mp hA')),
      if_neg hA]
    simp only [mul_zero, sub_self, abs_zero]
    exact hM0
-- ════════════════════════════════════════════════════════════════
-- § The conv1 capstone: one inexact SGD step provably descends
-- ════════════════════════════════════════════════════════════════

/-- **One inexact SGD step on the CNN's FIRST conv kernel provably
    decreases the cross-entropy loss.** The deepest rung: the step
    crosses relu₁, conv2 (as a function of its input — the point-free
    tap Jacobian with locality factor `c·kH·kW·w₂`), relu₂, the pool,
    and the 3-dense head. Under the FIVE margins at the step radius
    `D = lr·(‖∇L‖₁ + |kernel|·η)`, every mask and the pool's routing
    pattern freeze along the step, and the loss drops by
    ≥ `lr·‖∇L‖₂²/2`. With this, every conv kernel of the Chapter-4 CNN
    has a proven descent statement. -/
theorem cnn_conv1_sgd_descends {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (gh : Vec (c * ic * kH * kW))
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    {lr η a w₂ w₃ w₄ w₅ : ℝ} (ha : 0 ≤ a)
    (hx : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx - (gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁)) idx| ≤ η)
    (hm1 : ∀ k, a * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)) < |(Tensor3.flatten (conv2d W₁ b₁ x₀)) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)))) < |(Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))) k|)
    (hmq : MaxPool2MarginQ (((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η))))) (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)))))) < |(dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η))))))))
      < |(dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η))))))))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) ≤
      lr * (∑ idx, (gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁)) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 / (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η))))))))))))) * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)) ^ 2 ≤
      lr * (∑ idx, (gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁)) idx ^ 2) / 4) :
    crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d (Kernel4.unflatten (Kernel4.flatten W₁ - lr • gh)) b₁ x₀))))))))))))) label ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d (Kernel4.unflatten (Kernel4.flatten W₁)) b₁ x₀))))))))))))) label -
        lr * (∑ idx, (gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁)) idx ^ 2) / 2 := by
  set f : Vec (c * ic * kH * kW) → ℝ :=
    fun u' : Vec (c * ic * kH * kW) =>
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label
    with hf
  have hden : (0:ℝ) < 1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt f (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η))))))))))) := by linarith
  have hC0 : (0:ℝ) ≤ 2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 / (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt f (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)))))))))))) :=
    div_nonneg (by positivity) hden.le
  have hm1' : ∀ k, a * (lr * ((∑ idx, |gradAt f (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)) <
      |Tensor3.flatten (conv2d (Kernel4.unflatten (Kernel4.flatten W₁))
        b₁ x₀) k| := fun k => by
    rw [Kernel4.unflatten_flatten]
    exact hm1 k
  have hm2' : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (lr * ((∑ idx, |gradAt f (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)))) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d (Kernel4.unflatten (Kernel4.flatten W₁)) b₁ x₀)))))
        k| := fun k => by
    rw [Kernel4.unflatten_flatten]
    exact hm2 k
  have hmq' : MaxPool2MarginQ (((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (lr * ((∑ idx, |gradAt f (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)))))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (Kernel4.flatten W₁))
              b₁ x₀)))))))) := by
    rw [Kernel4.unflatten_flatten]
    exact hmq
  have hm3' : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx, |gradAt f (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)))))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (Kernel4.flatten W₁))
              b₁ x₀)))))))) l| := fun l => by
    rw [Kernel4.unflatten_flatten]
    exact hm3 l
  have hm4' : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
      (a * (lr * ((∑ idx, |gradAt f (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η)))))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d (Kernel4.unflatten (Kernel4.flatten W₁))
              b₁ x₀)))))))))) q| := fun q => by
    rw [Kernel4.unflatten_flatten]
    exact hm4 q
  have hD : (∑ idx, |(-(lr • gh)) idx|) ≤ lr * ((∑ idx, |gradAt f (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η) := by
    calc (∑ idx, |(-(lr • gh)) idx|) = ∑ idx, lr * |gh idx| := by
          refine Finset.sum_congr rfl fun idx _ => ?_
          simp [abs_mul, abs_of_nonneg hlr]
      _ ≤ ∑ idx, lr * (|gradAt f (Kernel4.flatten W₁) idx| + η) := by
          refine Finset.sum_le_sum fun idx _ => ?_
          refine mul_le_mul_of_nonneg_left ?_ hlr
          have h3 : |gh idx| ≤
              |gh idx - gradAt f (Kernel4.flatten W₁) idx| +
              |gradAt f (Kernel4.flatten W₁) idx| := by
            simpa using abs_sub_le (gh idx)
              (gradAt f (Kernel4.flatten W₁) idx) 0
          linarith [hgh idx]
      _ = lr * ((∑ idx, |gradAt f (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) * η) := by
          rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const,
            Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hmain := sgd_descends f (Kernel4.flatten W₁) gh hlr hη hC0 hgh
    (fun t ht => cnn_conv1_loss_differentiableAt b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄
      W₅ b₅ label hc hh hw _
      (fun k => (cnn1_margin1_keeps_offkink b₁ x₀ ha hx
        (Kernel4.flatten W₁) (-(lr • gh)) hD hm1' t ht.1 ht.2 k).1)
      (fun k => (cnn1_margin2_keeps_offkink b₁ x₀ W₂ b₂ ha hx hw₂ hW₂
        (Kernel4.flatten W₁) (-(lr • gh)) hD hm2' t ht.1 ht.2 k).1)
      (hmq'.smooth_of_close (fun ci hi wi => cnn1_postrelu2_close_seg
        b₁ x₀ W₂ b₂ ha hx hw₂ hW₂ (Kernel4.flatten W₁) (-(lr • gh)) hD
        t ht.1 ht.2 ci hi wi))
      (fun l => (cnn1_margin3_keeps_offkink b₁ x₀ W₂ b₂ W₃ b₃ ha hx
        hw₂ hW₂ hw₃ hW₃ (Kernel4.flatten W₁) (-(lr • gh)) hD hm3'
        t ht.1 ht.2 l).1)
      (fun q => (cnn1_margin4_keeps_offkink b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄
        ha hx hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ (Kernel4.flatten W₁) (-(lr • gh))
        hD hm4' t ht.1 ht.2 q).1))
    (fun t ht idx => by
      have hlip := cnn_conv1_loss_grad_lipschitz b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄
        W₅ b₅ label hh hw ha hx hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
        (Kernel4.flatten W₁) (-(lr • gh)) hD hm1' hm2' hmq' hm3' hm4'
        hsmall t ht idx
      simpa [hf] using hlip)
    h1 h2
  simpa [hf] using hmain

open FloatModel in
/-- **One binary32 SGD step on the CNN's FIRST conv kernel provably decreases
    the cross-entropy loss — with NO abstract gradient-accuracy parameter**
    (Increment 4 capstone, the deepest descent statement). The conv-1 peer of
    `cnn_conv2_float_sgd_descends`: the gradient is the actual binary32 `W₁`
    gradient `M.cnnConv1FloatGrad …`, accuracy *proven* by `cnn_conv1_grad_close`
    (η := `cnnConv1GradBudget`, discharged per kernel entry via `k4Idx_surj`),
    wired into the abstract `cnn_conv1_sgd_descends`. Five per-layer ROUND
    margins feed the grad-close; the gradient-radius STEP margins +
    `hsmall`/`h1`/`h2` feed the drift-freeze and descent geometry. Every conv
    kernel of the Chapter-4 CNN now has a float-faithful descent statement. -/
theorem cnn_conv1_float_sgd_descends {ic c h w d₃ d₄ nC kH kW : Nat}
    (M : FloatModel) (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c)
    (x₀ : Tensor3 ic (2*h) (2*w)) (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC) (fexp : ℝ → ℝ)
    {lr a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp : ℝ}
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (ha : 0 ≤ a) (hw₁ : 0 ≤ w₁) (hβ₁ : 0 ≤ β₁) (hw₂ : 0 ≤ w₂) (hβ₂ : 0 ≤ β₂)
    (hw₃ : 0 ≤ w₃) (hβ₃ : 0 ≤ β₃) (hw₄ : 0 ≤ w₄) (hβ₄ : 0 ≤ β₄) (hw₅ : 0 ≤ w₅)
    (hβ₅ : 0 ≤ β₅) (hlr : 0 ≤ lr) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hρ1 : FloatModel.smRho M.u eexp nC < 1)
    (hx₀ : ∀ cc i j, |x₀ cc i j| ≤ a)
    (hW₁ : ∀ o cc kh kw, |W₁ o cc kh kw| ≤ w₁) (hb₁ : ∀ o, |b₁ o| ≤ β₁)
    (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂) (hb₂ : ∀ o, |b₂ o| ≤ β₂)
    (hW₃ : ∀ i j, |W₃ i j| ≤ w₃) (hb₃ : ∀ j, |b₃ j| ≤ β₃)
    (hW₄ : ∀ i j, |W₄ i j| ≤ w₄) (hb₄ : ∀ j, |b₄ j| ≤ β₄)
    (hW₅ : ∀ i j, |W₅ i j| ≤ w₅) (hb₅ : ∀ j, |b₅ j| ≤ β₅)
    (hr1 : ∀ k, FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 <
      |Tensor3.flatten (conv2d W₁ b₁ x₀) k|)
    (hr2 : ∀ k, FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₁ b₁ x₀))))) k|)
    (hrPool : MaxPool2MarginQ
      (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
        (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
        (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hr3 : ∀ l, FloatModel.layerBudget M.u (c * h * w) w₃ β₃
        (FloatModel.layerAct (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
        (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
          (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
          (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))) l|)
    (hr4 : ∀ q, FloatModel.layerBudget M.u d₃ w₄ β₄
        (FloatModel.layerAct (c * h * w) w₃ β₃ (FloatModel.layerAct (c * kH * kW)
          w₂ β₂ (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)))
        (FloatModel.layerBudget M.u (c * h * w) w₃ β₃
          (FloatModel.layerAct (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a))
          (FloatModel.layerBudget M.u (c * kH * kW) w₂ β₂
            (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
            (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))) q|)
    (hm1 : ∀ k, a * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) *
          M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
            w₅ β₅ eexp)) < |(Tensor3.flatten (conv2d W₁ b₁ x₀)) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (lr * ((∑ idx,
              |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) *
          M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
            w₅ β₅ eexp)))) < |(Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))) k|)
    (hmq : MaxPool2MarginQ (((c * kH * kW : ℕ) : ℝ) * (w₂ * (a * (lr * ((∑ idx,
              |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) *
          M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
            w₅ β₅ eexp))))) (Tensor3.unflatten (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu
                (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        (a * (lr * ((∑ idx, |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) *
          M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
            w₅ β₅ eexp)))))) < |(dense W₃ b₃ (maxPoolFlat c h w (relu
              (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁ b₁ x₀))))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx,
              |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) *
          M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
            w₅ β₅ eexp)))))))) < |(dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat
              c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
                (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁ b₁ x₀))))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) :
        ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx,
              |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) *
          M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
            w₅ β₅ eexp))))))))))) < 1)
    (h1 : lr * (M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃
          w₄ β₄ w₅ β₅ eexp) * (∑ idx,
              |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) ≤
      lr * (∑ idx, (gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁)) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2
        * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 * a ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) :
          ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (a * (lr * ((∑ idx,
              |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) *
          M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
            w₅ β₅ eexp))))))))))))) * (lr * ((∑ idx,
              |gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁) idx|) + ((c * ic * kH * kW : ℕ) : ℝ) *
          M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄
            w₅ β₅ eexp)) ^ 2 ≤
      lr * (∑ idx, (gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁)) idx ^ 2) / 4) :
    crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d
            (Kernel4.unflatten (Kernel4.flatten W₁ -
              lr • M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
                (Kernel4.flatten W₁)))
            b₁ x₀))))))))))))) label ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d
            (Kernel4.unflatten (Kernel4.flatten W₁)) b₁ x₀))))))))))))) label -
        lr * (∑ idx, (gradAt (fun u' : Vec (c * ic * kH * kW) =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d (Kernel4.unflatten u') b₁ x₀))))))))))))) label)
              (Kernel4.flatten W₁)) idx ^ 2) / 2 := by
  have hu := M.u_nonneg
  -- nonnegativity of the proven budget
  have hA1nn : 0 ≤ FloatModel.layerAct (ic * kH * kW) w₁ β₁ a :=
    layerAct_nonneg hw₁ hβ₁ ha
  have hE1nn : 0 ≤ FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0 :=
    layerBudget_nonneg hu hw₁ hβ₁ ha le_rfl
  have hCPnn : 0 ≤ FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ := by
    rw [FloatModel.cnnConv2CotMag]
    exact layerAct_nonneg hw₃ le_rfl (layerAct_nonneg hw₄ le_rfl
      (layerAct_nonneg hw₅ le_rfl zero_le_one))
  have he2nn : 0 ≤ M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW
      (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
      (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)
      w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp := by
    rw [FloatModel.cnnConv2CotBudget]
    exact layerBudget_nonneg hu hw₃ le_rfl
      (layerAct_nonneg hw₄ le_rfl (layerAct_nonneg hw₅ le_rfl zero_le_one))
      (layerBudget_nonneg hu hw₄ le_rfl (layerAct_nonneg hw₅ le_rfl zero_le_one)
        (layerBudget_nonneg hu hw₅ le_rfl zero_le_one
          (M.cotErr_nonneg heexp0 (layerBudget_nonneg hu hw₅ hβ₅
            (layerAct_nonneg hw₄ hβ₄ (layerAct_nonneg hw₃ hβ₃
              (layerAct_nonneg hw₂ hβ₂ hA1nn)))
            (layerBudget_nonneg hu hw₄ hβ₄ (layerAct_nonneg hw₃ hβ₃
              (layerAct_nonneg hw₂ hβ₂ hA1nn))
              (layerBudget_nonneg hu hw₃ hβ₃ (layerAct_nonneg hw₂ hβ₂ hA1nn)
                (layerBudget_nonneg hu hw₂ hβ₂ hA1nn hE1nn)))) hρ1)))
  have hη0 : 0 ≤ M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃
      w₄ β₄ w₅ β₅ eexp := by
    simp only [FloatModel.cnnConv1GradBudget]
    have hγ : ∀ m : ℕ, (0:ℝ) ≤ (1 + M.u) ^ (m + 1) - 1 :=
      fun m => sub_nonneg.mpr (one_le_pow₀ (by linarith))
    have hn : ∀ m : ℕ, (0:ℝ) ≤ ((m : ℕ) : ℝ) := fun m => Nat.cast_nonneg _
    have hebacknn : (0:ℝ) ≤ ((1 + M.u) ^ ((c * (2*h) * (2*w)) + 1) - 1) *
          (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ *
            (FloatModel.cnnConv2CotMag d₃ d₄ nC w₃ w₄ w₅ +
              M.cnnConv2CotBudget c h w d₃ d₄ nC kH kW
                (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
                (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)
                w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp))) +
          (((c * (2*h) * (2*w) : ℕ) : ℝ) * (w₂ * M.cnnConv2CotBudget c h w d₃ d₄
            nC kH kW (FloatModel.layerAct (ic * kH * kW) w₁ β₁ a)
            (FloatModel.layerBudget M.u (ic * kH * kW) w₁ β₁ a 0)
            w₂ β₂ w₃ β₃ w₄ β₄ w₅ β₅ eexp)) :=
      add_nonneg (mul_nonneg (hγ _) (mul_nonneg (hn _) (mul_nonneg hw₂
        (add_nonneg hCPnn he2nn)))) (mul_nonneg (hn _) (mul_nonneg hw₂ he2nn))
    exact add_nonneg (mul_nonneg (hγ _) (mul_nonneg (hn _) (mul_nonneg ha
      (add_nonneg (mul_nonneg (hn _) (mul_nonneg hw₂ hCPnn)) hebacknn))))
      (mul_nonneg (hn _) (mul_nonneg ha hebacknn))
  -- the flattened conv-1 kernel inherits the per-entry bound
  have huf : ∀ idx, |Kernel4.flatten W₁ idx| ≤ w₁ := by
    intro idx
    obtain ⟨o', c', kh', kw', rfl⟩ := k4Idx_surj idx
    rw [flatten_k4Idx]; exact hW₁ o' c' kh' kw'
  -- discharge the abstract gradient accuracy by the proven grad-close
  have hgh : ∀ idx, |M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
      (Kernel4.flatten W₁) idx -
      gradAt (fun u' : Vec (c * ic * kH * kW) =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d (Kernel4.unflatten u') b₁ x₀)))))))))))))
          label) (Kernel4.flatten W₁) idx| ≤
      M.cnnConv1GradBudget ic c h w d₃ d₄ nC kH kW a w₁ β₁ w₂ β₂ w₃ β₃ w₄ β₄ w₅
        β₅ eexp := by
    intro idx
    obtain ⟨o', c', kh', kw', rfl⟩ := k4Idx_surj idx
    exact cnn_conv1_grad_close M b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label fexp
      (Kernel4.flatten W₁) hh hw ha hw₁ hβ₁ hw₂ hβ₂ hw₃ hβ₃ hw₄ hβ₄ hw₅ hβ₅
      heexp0 heexp1 hfexp hρ1 hx₀ huf hb₁ hW₂ hb₂ hW₃ hb₃ hW₄ hb₄ hW₅ hb₅
      (fun k => by rw [Kernel4.unflatten_flatten]; exact hr1 k)
      (fun k => by rw [Kernel4.unflatten_flatten]; exact hr2 k)
      (by rw [Kernel4.unflatten_flatten]; exact hrPool)
      (fun l => by rw [Kernel4.unflatten_flatten]; exact hr3 l)
      (fun q => by rw [Kernel4.unflatten_flatten]; exact hr4 q)
      o' c' kh' kw'
  exact cnn_conv1_sgd_descends W₁ b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label
    (M.cnnConv1FloatGrad b₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ fexp label
      (Kernel4.flatten W₁))
    hc hh hw ha hx₀ hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ hlr hη0 hgh hm1 hm2 hmq hm3
    hm4 hsmall h1 h2

-- ════════════════════════════════════════════════════════════════
-- § The conv bias: affine difference, drift, and the Kronecker Jacobian
--
-- The bias rungs. `conv2d` is affine in its bias with the SIMPLEST
-- possible Jacobian: output `(co,hi,wi)` reads bias entry `co` with
-- coefficient 1 — a Kronecker channel indicator, point-free. The
-- per-entry drift is exactly `|e o|` (no input bound `a`, no kernel
-- mass); only the `ℓ1` drift picks up the spatial multiplicity `h·w`
-- (one bias entry feeds a whole channel — sharing, exactly as for the
-- kernel). Everything downstream of the conv is the kernel-rung
-- argument verbatim.
-- ════════════════════════════════════════════════════════════════

/-- The conv output difference under a bias perturbation, exactly:
    output `(o,hi,wi)` moves by `e o` — `conv2d` is affine in the bias. -/
theorem conv2d_bias_sub {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (x : Tensor3 ic h w) (b e : Vec oc)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W (b + e) x o hi wi - conv2d W b x o hi wi = e o := by
  rw [conv2d_eq_convPad, conv2d_eq_convPad, Pi.add_apply]
  ring

/-- Per-entry conv drift under a bias perturbation: the perturbation's
    own entry — no `a` factor, no kernel mass. Flat-index form. -/
theorem conv2d_flat_bias_drift_total {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b e : Vec oc)
    (k : Fin (oc * h * w)) :
    |Tensor3.flatten (conv2d W (b + e) x) k -
      Tensor3.flatten (conv2d W b x) k| ≤ ∑ idx, |e idx| := by
  obtain ⟨p, rfl⟩ := finProdFinEquiv.surjective k
  obtain ⟨pp, wi⟩ := p
  obtain ⟨q, rfl⟩ := finProdFinEquiv.surjective pp
  obtain ⟨o, hi⟩ := q
  rw [show finProdFinEquiv (finProdFinEquiv (o, hi), wi) =
        t3Idx o hi wi from rfl,
    flatten_t3Idx, flatten_t3Idx, conv2d_bias_sub]
  exact Finset.single_le_sum (f := fun idx => |e idx|)
    (fun idx _ => abs_nonneg _) (Finset.mem_univ o)

/-- **`ℓ1` conv bias drift**: summed over all output entries, at most
    `(h·w)·‖e‖₁` — one bias entry feeds every spatial position of its
    channel. -/
theorem conv2d_flat_bias_drift_sum {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (x : Tensor3 ic h w) (b e : Vec oc) :
    ∑ k, |Tensor3.flatten (conv2d W (b + e) x) k -
        Tensor3.flatten (conv2d W b x) k| ≤
      ((h * w : ℕ) : ℝ) * ∑ idx, |e idx| := by
  rw [sum_t3 (fun k : Fin (oc * h * w) =>
    |Tensor3.flatten (conv2d W (b + e) x) k -
      Tensor3.flatten (conv2d W b x) k|)]
  calc ∑ o : Fin oc, ∑ hi : Fin h, ∑ wi : Fin w,
        |Tensor3.flatten (conv2d W (b + e) x) (t3Idx o hi wi) -
          Tensor3.flatten (conv2d W b x) (t3Idx o hi wi)|
      = ∑ o : Fin oc, ∑ _hi : Fin h, ∑ _wi : Fin w, |e o| := by
        refine Finset.sum_congr rfl fun o _ => Finset.sum_congr rfl
          fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
        rw [flatten_t3Idx, flatten_t3Idx, conv2d_bias_sub]
    _ = ∑ o : Fin oc, ((h * w : ℕ) : ℝ) * |e o| := by
        refine Finset.sum_congr rfl fun o _ => ?_
        rw [Finset.sum_const, Finset.sum_const, Finset.card_univ,
          Finset.card_univ, Fintype.card_fin, Fintype.card_fin,
          smul_smul, nsmul_eq_mul]
    _ = ((h * w : ℕ) : ℝ) * ∑ idx, |e idx| := by
        rw [Finset.mul_sum]
    _ ≤ ((h * w : ℕ) : ℝ) * ∑ idx, |e idx| := le_refl _

/-- **Closed form of the conv bias-map `pdiv`** — extracted from the
    certified VJP (`conv2d_bias_grad_has_vjp`) by contracting its
    `.correct` field against a basis vector, exactly as
    `conv2d_weight_pdiv`. Bias entry `o` touches output `(co,hi,wi)`
    iff `co = o`, with coefficient 1 — the Kronecker channel indicator.
    Point-free (the bias map is affine), so along a step segment only
    the head gradient moves. -/
theorem conv2d_bias_pdiv {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (x : Tensor3 ic h w) (b : Vec oc) (o : Fin oc)
    (co : Fin oc) (hi : Fin h) (wi : Fin w) :
    pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b
      o (t3Idx co hi wi)
      = if co = o then (1:ℝ) else 0 := by
  have hb := conv_bias_grad_bridge W x b (basisVec (t3Idx co hi wi)) o
  have hsum : ∑ j : Fin (oc * h * w),
      pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o j *
        basisVec (t3Idx co hi wi) j
      = pdiv (fun b' : Vec oc => Tensor3.flatten (conv2d W b' x)) b o
          (t3Idx co hi wi) := by
    rw [Finset.sum_eq_single (t3Idx co hi wi)
      (fun j _ hne => by rw [basisVec_apply, if_neg hne, mul_zero])
      (fun habs => absurd (Finset.mem_univ _) habs)]
    rw [basisVec_apply, if_pos rfl, mul_one]
  rw [← hsum, ← hb]
  -- evaluate the spatial-sum backward at the basis vector
  simp only [conv2d_bias_grad_has_vjp, basisVec_apply]
  simp only [t3Idx_def]
  rcases eq_or_ne co o with hco | hco
  · subst hco
    rw [if_pos rfl,
      Finset.sum_eq_single hi
        (fun hi' _ hne_hi => by
          rw [Finset.sum_eq_zero]
          intro wi' _
          rw [if_neg (fun heq => hne_hi
            (t3Idx_inj (show t3Idx co hi' wi' = t3Idx co hi wi
              from heq)).2.1)])
        (fun habs => absurd (Finset.mem_univ _) habs),
      Finset.sum_eq_single wi
        (fun wi' _ hne_wi => by
          rw [if_neg (fun heq => hne_wi
            (t3Idx_inj (show t3Idx co hi wi' = t3Idx co hi wi
              from heq)).2.2)])
        (fun habs => absurd (Finset.mem_univ _) habs),
      if_pos rfl]
  · rw [if_neg hco, Finset.sum_eq_zero]
    intro hi' _
    rw [Finset.sum_eq_zero]
    intro wi' _
    rw [if_neg (fun heq => hco
      ((t3Idx_inj (show t3Idx o hi' wi' = t3Idx co hi wi
        from heq)).1).symm)]

-- ════════════════════════════════════════════════════════════════
-- § The conv2-BIAS drift chain: the kernel chain with the conv stage's
--   `a·‖e‖₁` replaced by the bare `‖e‖₁`
-- ════════════════════════════════════════════════════════════════

/-- Pooled `ℓ1` drift under a conv2 bias perturbation: conv (`ℓ1`,
    spatial multiplicity, no `a`) → relu (contraction) → pool
    (contraction). -/
theorem cnnb2_pool_l1_drift {c h w kH kW : Nat} (W₂ : Kernel4 c c kH kW)
    (x₁ : Tensor3 c (2*h) (2*w)) (b e : Vec c) :
    ∑ q, |maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ (b + e) x₁))) q -
        maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b x₁))) q| ≤
      ((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx| :=
  le_trans (maxPoolFlat_l1_contract _ _)
    (le_trans (Finset.sum_le_sum fun k _ => relu_entry_lipschitz _ _ _ k)
      (conv2d_flat_bias_drift_sum W₂ x₁ b e))

/-- Per-entry POST-relu tensor drift under a conv2 bias perturbation —
    the form the pool margin consumes. -/
theorem cnnb2_postrelu_close {c h w kH kW : Nat} (W₂ : Kernel4 c c kH kW)
    (x₁ : Tensor3 c (2*h) (2*w)) (b e : Vec c)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ (b + e) x₁))) :
          Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b x₁))) :
          Tensor3 c (2*h) (2*w)) ci hi wi| ≤
      ∑ idx, |e idx| := by
  rw [unflatten_t3Idx, unflatten_t3Idx]
  exact le_trans (relu_entry_lipschitz _ _ _ _)
    (conv2d_flat_bias_drift_total W₂ x₁ b e _)

/-- Per-entry drift of the relu₃ pre-activation, conv2-bias rung. -/
theorem cnnb2_z3_drift {c h w d₃ kH kW : Nat} (W₂ : Kernel4 c c kH kW)
    (x₁ : Tensor3 c (2*h) (2*w)) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {w₃ : ℝ} (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (b e : Vec c) (l : Fin d₃) :
    |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ (b + e) x₁)))) l -
      dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁)))) l| ≤
      w₃ * (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx|) :=
  le_trans (dense_input_drift W₃ b₃ hW₃ _ _ l)
    (mul_le_mul_of_nonneg_left (cnnb2_pool_l1_drift W₂ x₁ b e) hw₃)

/-- Per-entry drift of the relu₄ pre-activation, conv2-bias rung. -/
theorem cnnb2_z4_drift {c h w d₃ d₄ kH kW : Nat} (W₂ : Kernel4 c c kH kW)
    (x₁ : Tensor3 c (2*h) (2*w)) (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {w₃ w₄ : ℝ} (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (b e : Vec c) (q : Fin d₄) :
    |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ (b + e) x₁)))))) q -
      dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b x₁)))))) q| ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        ∑ idx, |e idx|))) := by
  refine le_trans (dense_input_drift W₄ b₄ hW₄ _ _ q)
    (mul_le_mul_of_nonneg_left ?_ hw₄)
  calc ∑ l, |relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ (b + e) x₁))))) l -
        relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ b x₁))))) l|
      ≤ ∑ l, |dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₂ (b + e) x₁)))) l -
          dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₂ b x₁)))) l| :=
        Finset.sum_le_sum fun l _ => relu_entry_lipschitz _ _ _ l
    _ ≤ ∑ _l : Fin d₃, w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          ∑ idx, |e idx|) :=
        Finset.sum_le_sum fun l _ =>
          cnnb2_z3_drift W₂ x₁ W₃ b₃ hw₃ hW₃ b e l
    _ = (d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          ∑ idx, |e idx|)) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

/-- Logit drift through the whole conv2-bias chain. -/
theorem cnnb2_logit_drift {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    {w₃ w₄ w₅ : ℝ} (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (b e : Vec c) (k : Fin nC) :
    |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ (b + e) x₁)))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b x₁)))))))) k| ≤
      w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx|))))) := by
  refine le_trans (dense_input_drift W₅ b₅ hW₅ _ _ k)
    (mul_le_mul_of_nonneg_left ?_ hw₅)
  calc ∑ q, |relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ (b + e) x₁))))))) q -
        relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ b x₁))))))) q|
      ≤ ∑ q, |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₂ (b + e) x₁)))))) q -
          dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₂ b x₁)))))) q| :=
        Finset.sum_le_sum fun q _ => relu_entry_lipschitz _ _ _ q
    _ ≤ ∑ _q : Fin d₄, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          ∑ idx, |e idx|))) :=
        Finset.sum_le_sum fun q _ =>
          cnnb2_z4_drift W₂ x₁ W₃ b₃ W₄ b₄ hw₃ hW₃ hw₄ hW₄ b e q
    _ = (d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
          ∑ idx, |e idx|)))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

-- ════════════════════════════════════════════════════════════════
-- § conv2-bias margins freeze every routing decision along the segment
-- ════════════════════════════════════════════════════════════════

/-- The relu₂ margin (at the bias radius `D`) keeps the conv
    pre-activation off the kink along the whole step segment. -/
theorem cnnb2_margin2_keeps_offkink {c h w kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    {D : ℝ} (b e : Vec c) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ k, D < |Tensor3.flatten (conv2d W₂ b x₁) k|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (k : Fin (c * (2*h) * (2*w))) :
    Tensor3.flatten (conv2d W₂ (b + t • e) x₁) k ≠ 0 ∧
      (0 < Tensor3.flatten (conv2d W₂ (b + t • e) x₁) k
        ↔ 0 < Tensor3.flatten (conv2d W₂ b x₁) k) := by
  refine sign_stable_of_close ?_ (hm k)
  have h1 := conv2d_flat_bias_drift_total W₂ x₁ b (t • e) k
  have h2 : (∑ idx, |(t • e) idx|) ≤ D := smul_l1_mass_le e ht0 ht1 he
  linarith

/-- The POST-relu tensor stays within the bias-rung pool margin radius
    `D` along the whole step segment. -/
theorem cnnb2_postrelu_close_seg {c h w kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    {D : ℝ} (b e : Vec c) (he : (∑ idx, |e idx|) ≤ D)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ (b + t • e) x₁))) :
          Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b x₁))) :
          Tensor3 c (2*h) (2*w)) ci hi wi| ≤ D :=
  le_trans (cnnb2_postrelu_close W₂ x₁ b (t • e) ci hi wi)
    (smul_l1_mass_le e ht0 ht1 he)

/-- The relu₃ margin (at the bias radius) keeps the first head
    pre-activation off the kink along the whole step segment. -/
theorem cnnb2_margin3_keeps_offkink {c h w d₃ kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {w₃ D : ℝ} (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (b e : Vec c) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * D) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁)))) l|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (l : Fin d₃) :
    dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ (b + t • e) x₁)))) l ≠ 0 ∧
      (0 < dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ (b + t • e) x₁)))) l ↔
        0 < dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b x₁)))) l) := by
  refine sign_stable_of_close ?_ (hm l)
  have h1 := cnnb2_z3_drift W₂ x₁ W₃ b₃ hw₃ hW₃ b (t • e) l
  have h2 : w₃ * (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |(t • e) idx|) ≤
      w₃ * (((2*h * (2*w) : ℕ) : ℝ) * D) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (smul_l1_mass_le e ht0 ht1 he) (Nat.cast_nonneg _)) hw₃
  linarith

/-- The relu₄ margin (at the bias radius) keeps the second head
    pre-activation off the kink along the whole step segment. -/
theorem cnnb2_margin4_keeps_offkink {c h w d₃ d₄ kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {w₃ w₄ D : ℝ} (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (b e : Vec c) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * D))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b x₁)))))) q|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (q : Fin d₄) :
    dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ (b + t • e) x₁)))))) q ≠ 0 ∧
      (0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ (b + t • e) x₁)))))) q ↔
        0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ b x₁)))))) q) := by
  refine sign_stable_of_close ?_ (hm q)
  have h1 := cnnb2_z4_drift W₂ x₁ W₃ b₃ W₄ b₄ hw₃ hW₃ hw₄ hW₄ b (t • e) q
  have h2 : w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
      ∑ idx, |(t • e) idx|))) ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * D))) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (smul_l1_mass_le e ht0 ht1 he) (Nat.cast_nonneg _)) hw₃)
      (Nat.cast_nonneg _)) hw₄
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The conv2 loss-of-bias map: differentiability and gradient
-- ════════════════════════════════════════════════════════════════

/-- The loss-of-conv2-bias map is differentiable at any four-condition
    point. -/
theorem cnn_conv2_bias_loss_differentiableAt {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (b : Vec c)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b x₁) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b x₁))) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b x₁)))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁)))))) q ≠ 0) :
    DifferentiableAt ℝ
      (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b := by
  have hG := pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label hc hh hw
    (Tensor3.flatten (conv2d W₂ b x₁)) hz2 hmp hz3 hz4
  have h0 : DifferentiableAt ℝ
      (fun b' : Vec c => Tensor3.flatten (conv2d W₂ b' x₁)) b :=
    (conv2d_bias_differentiable W₂ x₁) b
  exact ((differentiableAt_pi.mp hG) 0).comp
    (f := fun b' : Vec c => Tensor3.flatten (conv2d W₂ b' x₁)) b h0

/-- **Closed form of the conv2 bias loss gradient** at any four-margin
    point — the EXISTING fold `conv_bias_total_loss_grad_fold` contracted
    with the pool-collapsed head gradient (`pool_relu_input_grad`, reused
    verbatim) and the Kronecker bias Jacobian (`conv2d_bias_pdiv`). -/
theorem cnn_conv2_bias_loss_gradAt {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (hh : 0 < h) (hw : 0 < w)
    (b : Vec c)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b x₁) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b x₁))) : Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b x₁)))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁)))))) q ≠ 0)
    (o : Fin c) :
    gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
        b o
      = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then (1:ℝ) else 0) *
            ((if Tensor3.flatten (conv2d W₂ b x₁)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                    (Tensor3.flatten (conv2d W₂ b x₁))))
                  ci hi wi
                then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
                  ((if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                        (Tensor3.flatten (conv2d W₂ b x₁))))
                        l > 0 then (1:ℝ) else 0) *
                    ∑ q, W₄ l q *
                      ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d W₂ b x₁)))))) q > 0
                          then (1:ℝ) else 0) *
                        ∑ k, W₅ q k *
                          (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                              (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d W₂ b x₁))))))))) k -
                            oneHot nC label k)))
                else 0)) := by
  have hc : 0 < c := Fin.pos o
  have hdiff := cnn_conv2_bias_loss_differentiableAt W₂ x₁ W₃ b₃ W₄ b₄
    W₅ b₅ label hc hh hw b hz2 hmp hz3 hz4
  have hG := pool_head_differentiableAt W₃ b₃ W₄ b₄ W₅ b₅ label hc hh hw
    (Tensor3.flatten (conv2d W₂ b x₁)) hz2 hmp hz3 hz4
  calc gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
        b o
      = pdiv (fun b' : Vec c => fun _ : Fin 1 =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
          b o 0 := gradAt_eq_pdiv _ _ hdiff _
    _ = ∑ k : Fin (c * (2*h) * (2*w)),
          pdiv (fun b' : Vec c =>
              Tensor3.flatten (conv2d W₂ b' x₁)) b o k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w
                  (relu (c * (2*h) * (2*w)) y))))))) label)
            (Tensor3.flatten (conv2d W₂ b x₁)) k 0 :=
        conv_bias_total_loss_grad_fold W₂ x₁ b
          (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w
                (relu (c * (2*h) * (2*w)) y))))))) label)
          hG o
    _ = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          pdiv (fun b' : Vec c =>
              Tensor3.flatten (conv2d W₂ b' x₁)) b o (t3Idx ci hi wi) *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w
                  (relu (c * (2*h) * (2*w)) y))))))) label)
            (Tensor3.flatten (conv2d W₂ b x₁)) (t3Idx ci hi wi) 0 :=
        sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
          pdiv (fun b' : Vec c =>
              Tensor3.flatten (conv2d W₂ b' x₁)) b o k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w
                  (relu (c * (2*h) * (2*w)) y))))))) label)
            (Tensor3.flatten (conv2d W₂ b x₁)) k 0)
    _ = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then (1:ℝ) else 0) *
            ((if Tensor3.flatten (conv2d W₂ b x₁)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              (if MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                    (Tensor3.flatten (conv2d W₂ b x₁))))
                  ci hi wi
                then ∑ l, W₃ (t3Idx ci (winRow hi) (winCol wi)) l *
                  ((if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                        (Tensor3.flatten (conv2d W₂ b x₁))))
                        l > 0 then (1:ℝ) else 0) *
                    ∑ q, W₄ l q *
                      ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                              (conv2d W₂ b x₁)))))) q > 0
                          then (1:ℝ) else 0) *
                        ∑ k, W₅ q k *
                          (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄
                              (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
                                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                  (conv2d W₂ b x₁))))))))) k -
                            oneHot nC label k)))
                else 0)) := by
        refine Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl
          fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
        rw [conv2d_bias_pdiv W₂ x₁ b o ci hi wi,
          pool_relu_input_grad W₃ b₃ W₄ b₄ W₅ b₅ label _ hz2 hmp hz3 hz4
            ci hi wi]

-- ════════════════════════════════════════════════════════════════
-- § Segment-Lipschitz gradient for the conv2 bias loss, explicit constant
-- ════════════════════════════════════════════════════════════════

/-- **Segment-Lipschitz gradient for the conv2-bias loss, explicit
    constant.** The kernel-rung argument with the conv stage's `a·D`
    radius replaced by the bare `D` — the bias Jacobian is a Kronecker
    indicator with row mass `(2h)·(2w)`, no input bound. Constant:
    the kernel constant with `a² ↦ 1`. -/
theorem cnn_conv2_bias_loss_grad_lipschitz {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (hh : 0 < h) (hw : 0 < w)
    {w₃ w₄ w₅ D : ℝ}
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (b d : Vec c) (hd : (∑ idx, |d idx|) ≤ D)
    (hm2 : ∀ k, D < |Tensor3.flatten (conv2d W₂ b x₁) k|)
    (hmq : MaxPool2MarginQ D (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b x₁)))))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * D) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) * D))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * D)))))) < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1)
    (o : Fin c) :
    |gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
        (b + t • d) o -
      gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
        b o| ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * D)))))))) * (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D :=
    le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  have hδ0 : (0:ℝ) ≤ w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * D))))) :=
    mul_nonneg hw₅ (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
        (mul_nonneg (Nat.cast_nonneg _) hD0)))))
  have hden : (0:ℝ) < 1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * D)))))) := by linarith
  -- base-point conditions from the margins
  have hz2_v : ∀ k, Tensor3.flatten (conv2d W₂ b x₁) k ≠ 0 :=
    fun k h0 => by
      have hk := hm2 k
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr hD0)
  have hmp_v : MaxPool2Smooth (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b x₁))) :
      Tensor3 c (2*h) (2*w)) := hmq.smooth hD0
  have hz3_v : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁)))) l ≠ 0 :=
    fun l h0 => by
      have hk := hm3 l
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr (mul_nonneg hw₃
        (mul_nonneg (Nat.cast_nonneg _) hD0)))
  have hz4_v : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₂ b x₁)))))) q ≠ 0 :=
    fun q h0 => by
      have hk := hm4 q
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr (mul_nonneg hw₄
        (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
          (mul_nonneg (Nat.cast_nonneg _) hD0)))))
  -- segment-point conditions: everything frozen
  have hstab2 := fun k =>
    cnnb2_margin2_keeps_offkink W₂ x₁ b d hd hm2 t ht0 ht1 k
  have hz2_t : ∀ k, Tensor3.flatten (conv2d W₂ (b + t • d) x₁) k ≠ 0 :=
    fun k => (hstab2 k).1
  have hclose := fun ci hi wi =>
    cnnb2_postrelu_close_seg W₂ x₁ b d hd t ht0 ht1 ci hi wi
  have hmp_t : MaxPool2Smooth (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₂ (b + t • d) x₁))) :
      Tensor3 c (2*h) (2*w)) := hmq.smooth_of_close hclose
  have hstab3 := fun l =>
    cnnb2_margin3_keeps_offkink W₂ x₁ W₃ b₃ hw₃ hW₃ b d hd hm3
      t ht0 ht1 l
  have hz3_t : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₂ (b + t • d) x₁)))) l ≠ 0 :=
    fun l => (hstab3 l).1
  have hstab4 := fun q =>
    cnnb2_margin4_keeps_offkink W₂ x₁ W₃ b₃ W₄ b₄ hw₃ hW₃ hw₄ hW₄
      b d hd hm4 t ht0 ht1 q
  have hz4_t : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₂ (b + t • d) x₁)))))) q ≠ 0 :=
    fun q => (hstab4 q).1
  -- both gradients in closed form
  rw [cnn_conv2_bias_loss_gradAt W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw
      (b + t • d) hz2_t hmp_t hz3_t hz4_t o,
    cnn_conv2_bias_loss_gradAt W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw
      b hz2_v hmp_v hz3_v hz4_v o]
  -- the frozen masks and the frozen routing
  have hmask2 : ∀ (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)),
      (if Tensor3.flatten (conv2d W₂ (b + t • d) x₁)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) =
      (if Tensor3.flatten (conv2d W₂ b x₁)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) := by
    intro ci hi wi
    by_cases hp : Tensor3.flatten (conv2d W₂ b x₁)
        (t3Idx ci hi wi) > 0
    · rw [if_pos ((hstab2 _).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab2 _).2.mp hgt)), if_neg hp]
  have hmask3 : ∀ l : Fin d₃,
      (if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ (b + t • d) x₁))))
          l > 0 then (1:ℝ) else 0) =
      (if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b x₁))))
          l > 0 then (1:ℝ) else 0) := by
    intro l
    by_cases hp : dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁)))) l > 0
    · rw [if_pos ((hstab3 l).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab3 l).2.mp hgt)), if_neg hp]
  have hmask4 : ∀ q : Fin d₄,
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ (b + t • d) x₁)))))) q > 0
        then (1:ℝ) else 0) =
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ b x₁)))))) q > 0
        then (1:ℝ) else 0) := by
    intro q
    by_cases hp : dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b x₁)))))) q > 0
    · rw [if_pos ((hstab4 q).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab4 q).2.mp hgt)), if_neg hp]
  have hargiff : ∀ (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)),
      MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ (b + t • d) x₁))))
        ci hi wi ↔
      MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b x₁))))
        ci hi wi :=
    fun ci hi wi => hmq.isArgmax_iff hclose ci hi wi
  -- the softmax drift along the segment
  have hzdrift : ∀ k, |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ (b + t • d) x₁)))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b x₁)))))))) k| ≤
      t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * D)))))) := by
    intro k
    have h1 := cnnb2_logit_drift W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅
      hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ b (t • d) k
    rw [smul_l1_mass d ht0] at h1
    have h2 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (t * ∑ idx, |d idx|)))))) =
        t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |d idx|)))))) := by
      ring
    rw [h2] at h1
    have h3 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |d idx|))))) ≤
        w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * D))))) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left hd (Nat.cast_nonneg _)) hw₃)
          (Nat.cast_nonneg _)) hw₄) (Nat.cast_nonneg _)) hw₅
    have h4 := mul_le_mul_of_nonneg_left h3 ht0
    linarith
  have hδlt : 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * D))))))) < 1 := by
    nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
  have hexp := FloatModel.exp_sub_one_le hδlt
  have hmono : 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))))))) /
        (1 - 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * D)))))))) ≤
      2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * D))))))) := by
    refine div_le_div_of_nonneg_left
      (by nlinarith [mul_nonneg ht0 hδ0]) hden ?_
    nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
  have hS : ∀ k, |softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ (b + t • d) x₁))))))))) k -
      softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b x₁))))))))) k| ≤
      2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * D))))))) :=
    fun k => le_trans (FloatModel.softmax_perturb _ _ hzdrift k)
      (le_trans hexp hmono)
  have hΔ0 : (0:ℝ) ≤ 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * D))))))) /
      (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))))))) :=
    div_nonneg (mul_nonneg (by norm_num) (mul_nonneg ht0 hδ0)) hden.le
  have hM0 : (0:ℝ) ≤ (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
      (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * D))))))))))))) :=
    mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
        (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₅ hΔ0)))))
  -- the bias Jacobian row mass: the Kronecker indicator sums to (2h)·(2w)
  have hcp : ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
      |if ci = o then (1:ℝ) else 0| ≤ ((2*h * (2*w) : ℕ) : ℝ) := by
    rw [Finset.sum_eq_single o
      (fun ci _ hne => by
        rw [Finset.sum_eq_zero]
        intro hi _
        rw [Finset.sum_eq_zero]
        intro wi _
        rw [if_neg hne, abs_zero])
      (fun habs => absurd (Finset.mem_univ _) habs)]
    calc ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          |if o = o then (1:ℝ) else 0|
        ≤ ∑ _hi : Fin (2*h), ∑ _wi : Fin (2*w), (1:ℝ) := by
          refine Finset.sum_le_sum fun hi _ =>
            Finset.sum_le_sum fun wi _ => ?_
          rw [if_pos rfl, abs_one]
      _ = ((2*h * (2*w) : ℕ) : ℝ) := by
          rw [Finset.sum_const, Finset.sum_const, Finset.card_univ,
            Finset.card_univ, Fintype.card_fin, Fintype.card_fin,
            smul_smul, nsmul_eq_mul, mul_one]
  -- the endgame: combine, freeze, collapse to the softmax drift
  have hfinal : ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
      (|if ci = o then (1:ℝ) else 0| *
        ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
          (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
            (((2*h * (2*w) : ℕ) : ℝ) * D))))))) /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((2*h * (2*w) : ℕ) : ℝ) * D))))))))))))))) ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * D)))))))) * (t * D) := by
    calc ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
        (|if ci = o then (1:ℝ) else 0| *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((2*h * (2*w) : ℕ) : ℝ) * D))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((2*h * (2*w) : ℕ) : ℝ) * D)))))))))))))))
        = (∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
            |if ci = o then (1:ℝ) else 0|) *
            ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
              (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((2*h * (2*w) : ℕ) : ℝ) * D))))))) /
                (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                  (((2*h * (2*w) : ℕ) : ℝ) * D)))))))))))))) := by
          simp only [← Finset.sum_mul]
      _ ≤ ((2*h * (2*w) : ℕ) : ℝ) *
            ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
              (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((2*h * (2*w) : ℕ) : ℝ) * D))))))) /
                (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                  (((2*h * (2*w) : ℕ) : ℝ) * D)))))))))))))) :=
          mul_le_mul_of_nonneg_right hcp hM0
      _ = (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
            (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((2*h * (2*w) : ℕ) : ℝ) * D)))))))) * (t * D) := by
          ring
  refine le_trans (le_trans (abs_triple_sum_sub_le _ _)
    (Finset.sum_le_sum fun ci _ => Finset.sum_le_sum fun hi _ =>
      Finset.sum_le_sum fun wi _ => ?_)) hfinal
  -- per-term: freeze the masks and the route, collapse to the drift
  rw [hmask2 ci hi wi]
  simp only [hmask3, hmask4]
  by_cases hA : MaxPool2IsArgmax (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₂ b x₁)))) ci hi wi
  · rw [if_pos ((hargiff ci hi wi).mpr hA), if_pos hA, ← mul_sub,
      abs_mul, ← mul_sub, abs_mul]
    refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
    refine le_trans (mul_le_of_le_one_left (abs_nonneg _) ?_) ?_
    · split_ifs <;> simp
    · exact head3_sum_drift W₃ W₄ W₅ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
        (fun l => if dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ b x₁)))) l > 0
          then (1:ℝ) else 0)
        (fun l => by split_ifs <;> simp)
        (fun q => if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ b x₁)))))) q > 0
          then (1:ℝ) else 0)
        (fun q => by split_ifs <;> simp)
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b x₁))))))))))
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ (b + t • d) x₁))))))))))
        (oneHot nC label) hS (t3Idx ci (winRow hi) (winCol wi))
  · rw [if_neg (fun hA' => hA ((hargiff ci hi wi).mp hA')), if_neg hA]
    simp only [mul_zero, sub_self, abs_zero]
    exact mul_nonneg (abs_nonneg _) hM0

-- ════════════════════════════════════════════════════════════════
-- § The conv2-bias capstone: one inexact SGD step provably descends
-- ════════════════════════════════════════════════════════════════

/-- **One inexact SGD step on the CNN's second conv BIAS provably
    decreases the cross-entropy loss.** The conv2-kernel capstone with
    the bias-rung radii: the four margins at the step radius
    `D = lr·(‖∇L‖₁ + c·η)` carry no input bound `a` (the bias Jacobian
    is a Kronecker indicator), and the parameter needs no
    flatten/unflatten plumbing — the bias IS a vector. -/
theorem cnn_conv2_bias_sgd_descends {c h w d₃ d₄ nC kH kW : Nat}
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c) (x₁ : Tensor3 c (2*h) (2*w))
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (gh : Vec c)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    {lr η w₃ w₄ w₅ : ℝ}
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ o, |gh o -
      gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o| ≤ η)
    (hm2 : ∀ k, lr * ((∑ o, |gradAt
        (fun b' : Vec c =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o|) +
        (c : ℝ) * η) <
      |Tensor3.flatten (conv2d W₂ b₂ x₁) k|)
    (hmq : MaxPool2MarginQ (lr * ((∑ o, |gradAt
        (fun b' : Vec c =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o|) +
        (c : ℝ) * η))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))))
    (hm3 : ∀ l, w₃ * (((2*h * (2*w) : ℕ) : ℝ) * (lr * ((∑ o,
        |gradAt (fun b' : Vec c =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o|) +
        (c : ℝ) * η))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ x₁)))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((2*h * (2*w) : ℕ) : ℝ) *
        (lr * ((∑ o, |gradAt
          (fun b' : Vec c =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o|) +
          (c : ℝ) * η))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁)))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (lr * ((∑ o, |gradAt
        (fun b' : Vec c =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o|) +
        (c : ℝ) * η)))))))) < 1)
    (h1 : lr * η * (∑ o, |gradAt
        (fun b' : Vec c =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o|) ≤
      lr * (∑ o, gradAt
        (fun b' : Vec c =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 *
        (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((2*h * (2*w) : ℕ) : ℝ) * (lr * ((∑ o, |gradAt
            (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
              b₂ o|) +
            (c : ℝ) * η)))))))))) *
        (lr * ((∑ o, |gradAt
          (fun b' : Vec c =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label) b₂ o|) +
          (c : ℝ) * η)) ^ 2 ≤
      lr * (∑ o, gradAt
        (fun b' : Vec c =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
          b₂ o ^ 2) / 4) :
    crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ (b₂ - lr • gh) x₁))))))))) label ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ x₁))))))))) label -
        lr * (∑ o, gradAt
          (fun b' : Vec c =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label)
            b₂ o ^ 2) / 2 := by
  set f : Vec c → ℝ :=
    fun b' : Vec c =>
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b' x₁))))))))) label with hf
  have hden : (0:ℝ) < 1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((2*h * (2*w) : ℕ) : ℝ) * (lr * ((∑ o, |gradAt f b₂ o|) +
        (c : ℝ) * η)))))))) := by
    linarith
  have hC0 : (0:ℝ) ≤ 2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 *
      (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
      (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((2*h * (2*w) : ℕ) : ℝ) * (lr * ((∑ o, |gradAt f b₂ o|) +
          (c : ℝ) * η))))))))) :=
    div_nonneg (by positivity) hden.le
  -- ℓ1 radius of the step
  have hD : (∑ o, |(-(lr • gh)) o|) ≤
      lr * ((∑ o, |gradAt f b₂ o|) + (c : ℝ) * η) := by
    calc (∑ o, |(-(lr • gh)) o|) = ∑ o, lr * |gh o| := by
          refine Finset.sum_congr rfl fun o _ => ?_
          simp [abs_mul, abs_of_nonneg hlr]
      _ ≤ ∑ o, lr * (|gradAt f b₂ o| + η) := by
          refine Finset.sum_le_sum fun o _ => ?_
          refine mul_le_mul_of_nonneg_left ?_ hlr
          have h3 : |gh o| ≤
              |gh o - gradAt f b₂ o| + |gradAt f b₂ o| := by
            simpa using abs_sub_le (gh o) (gradAt f b₂ o) 0
          linarith [hgh o]
      _ = lr * ((∑ o, |gradAt f b₂ o|) + (c : ℝ) * η) := by
          rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const,
            Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hmain := sgd_descends f b₂ gh hlr hη hC0 hgh
    (fun t ht => cnn_conv2_bias_loss_differentiableAt W₂ x₁ W₃ b₃ W₄ b₄
      W₅ b₅ label hc hh hw _
      (fun k => (cnnb2_margin2_keeps_offkink W₂ x₁
        b₂ (-(lr • gh)) hD hm2 t ht.1 ht.2 k).1)
      (hmq.smooth_of_close (fun ci hi wi => cnnb2_postrelu_close_seg W₂ x₁
        b₂ (-(lr • gh)) hD t ht.1 ht.2 ci hi wi))
      (fun l => (cnnb2_margin3_keeps_offkink W₂ x₁ W₃ b₃ hw₃ hW₃
        b₂ (-(lr • gh)) hD hm3 t ht.1 ht.2 l).1)
      (fun q => (cnnb2_margin4_keeps_offkink W₂ x₁ W₃ b₃ W₄ b₄ hw₃ hW₃
        hw₄ hW₄ b₂ (-(lr • gh)) hD hm4 t ht.1 ht.2 q).1))
    (fun t ht o => by
      have h := cnn_conv2_bias_loss_grad_lipschitz W₂ x₁ W₃ b₃ W₄ b₄ W₅ b₅
        label hh hw hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ b₂
        (-(lr • gh)) hD hm2 hmq hm3 hm4 hsmall t ht o
      simpa [hf] using h)
    h1 h2
  simpa [hf] using hmain

-- ════════════════════════════════════════════════════════════════
-- § The conv1-BIAS drift chain: the conv1-kernel chain with the conv1
--   stage's `a·‖e‖₁` replaced by the bare `‖e‖₁`
-- ════════════════════════════════════════════════════════════════

/-- POST-relu₁ tensor drift under a conv1 bias perturbation. -/
theorem cnnb1_postrelu1_close {ic c h w kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w)) (b e : Vec c)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ (b + e) x₀))) :
          Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₁ b x₀))) :
          Tensor3 c (2*h) (2*w)) ci hi wi| ≤
      ∑ idx, |e idx| := by
  rw [unflatten_t3Idx, unflatten_t3Idx]
  exact le_trans (relu_entry_lipschitz _ _ _ _)
    (conv2d_flat_bias_drift_total W₁ x₀ b e _)

/-- Per-entry conv2-preactivation drift under a conv1 bias
    perturbation: the perturbation crosses conv2 as a function of its
    INPUT, picking up the locality factor `c·kH·kW·w₂`. -/
theorem cnnb1_z2_entry_drift {ic c h w kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {w₂ : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (b e : Vec c) (k : Fin (c * (2*h) * (2*w))) :
    |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ (b + e) x₀))))) k -
      Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀))))) k| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * ∑ idx, |e idx|) := by
  obtain ⟨p, rfl⟩ := finProdFinEquiv.surjective k
  obtain ⟨pp, wo⟩ := p
  obtain ⟨q, rfl⟩ := finProdFinEquiv.surjective pp
  obtain ⟨o, ho⟩ := q
  rw [show finProdFinEquiv (finProdFinEquiv (o, ho), wo) =
        t3Idx o ho wo from rfl,
    flatten_t3Idx, flatten_t3Idx]
  exact conv2d_input_entry_drift W₂ b₂ _ _ hw₂ hW₂
    (Finset.sum_nonneg fun _ _ => abs_nonneg _)
    (fun cc i j => cnnb1_postrelu1_close W₁ x₀ b e cc i j) o ho wo

/-- POST-relu₂ tensor drift under a conv1 bias perturbation — what the
    pool margin consumes on the conv1-bias rung. -/
theorem cnnb1_postrelu2_close {ic c h w kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {w₂ : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (b e : Vec c)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + e) x₀))))))) :
        Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀))))))) :
        Tensor3 c (2*h) (2*w)) ci hi wi| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * ∑ idx, |e idx|) := by
  rw [unflatten_t3Idx, unflatten_t3Idx]
  exact le_trans (relu_entry_lipschitz _ _ _ _)
    (cnnb1_z2_entry_drift W₁ x₀ W₂ b₂ hw₂ hW₂ b e _)

/-- Pooled `ℓ1` drift under a conv1 bias perturbation: conv1 (`ℓ1`,
    spatial multiplicity, no `a`) → relu → conv2-as-input (`ℓ1`,
    LOCALITY multiplicity `c·kH·kW`) → relu → pool. -/
theorem cnnb1_pool_l1_drift {ic c h w kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {w₂ : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (b e : Vec c) :
    ∑ q, |maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ (b + e) x₀))))))) q -
        maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ b x₀))))))) q| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        ∑ idx, |e idx|)) := by
  refine le_trans (maxPoolFlat_l1_contract _ _) (le_trans
    (Finset.sum_le_sum fun k _ => relu_entry_lipschitz _ _ _ k) ?_)
  calc ∑ k, |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + e) x₀))))) k -
        Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀))))) k|
      = ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
          |conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ (b + e) x₀)))) co ho wo -
            conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b x₀)))) co ho wo| := by
        rw [sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
          |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ (b + e) x₀))))) k -
            Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b x₀))))) k|)]
        refine Finset.sum_congr rfl fun co _ => Finset.sum_congr rfl
          fun ho _ => Finset.sum_congr rfl fun wo _ => ?_
        rw [flatten_t3Idx, flatten_t3Idx]
    _ ≤ ((c * kH * kW : ℕ) : ℝ) * (w₂ *
          ∑ cc : Fin c, ∑ i : Fin (2*h), ∑ j : Fin (2*w),
            |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (conv2d W₁ (b + e)
                  x₀))) : Tensor3 c (2*h) (2*w)) cc i j -
              (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (conv2d W₁ b
                  x₀))) : Tensor3 c (2*h) (2*w)) cc i j|) :=
        conv2d_input_l1_drift W₂ b₂ _ _ hw₂ hW₂
    _ ≤ ((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          ∑ idx, |e idx|)) := by
        refine mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left ?_ hw₂) (Nat.cast_nonneg _)
        calc ∑ cc : Fin c, ∑ i : Fin (2*h), ∑ j : Fin (2*w),
              |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₁ (b + e)
                    x₀))) : Tensor3 c (2*h) (2*w)) cc i j -
                (Tensor3.unflatten (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₁ b
                    x₀))) : Tensor3 c (2*h) (2*w)) cc i j|
            = ∑ k, |relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁ (b + e) x₀)) k -
                relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁ b x₀)) k| :=
              (sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
                |relu (c * (2*h) * (2*w)) (Tensor3.flatten
                    (conv2d W₁ (b + e) x₀)) k -
                  relu (c * (2*h) * (2*w)) (Tensor3.flatten
                    (conv2d W₁ b x₀)) k|)).symm
          _ ≤ ∑ k, |Tensor3.flatten
                  (conv2d W₁ (b + e) x₀) k -
                Tensor3.flatten (conv2d W₁ b x₀) k| :=
              Finset.sum_le_sum fun k _ => relu_entry_lipschitz _ _ _ k
          _ ≤ ((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx| :=
              conv2d_flat_bias_drift_sum W₁ x₀ b e

/-- Per-entry drift of the relu₃ pre-activation, conv1-bias rung. -/
theorem cnnb1_z3_drift {ic c h w d₃ kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {w₂ w₃ : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (b e : Vec c) (l : Fin d₃) :
    |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + e) x₀)))))))) l -
      dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀)))))))) l| ≤
      w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        ∑ idx, |e idx|))) :=
  le_trans (dense_input_drift W₃ b₃ hW₃ _ _ l)
    (mul_le_mul_of_nonneg_left
      (cnnb1_pool_l1_drift W₁ x₀ W₂ b₂ hw₂ hW₂ b e) hw₃)

/-- Per-entry drift of the relu₄ pre-activation, conv1-bias rung. -/
theorem cnnb1_z4_drift {ic c h w d₃ d₄ kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {w₂ w₃ w₄ : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (b e : Vec c) (q : Fin d₄) :
    |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + e) x₀)))))))))) q -
      dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀)))))))))) q| ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx|))))) := by
  refine le_trans (dense_input_drift W₄ b₄ hW₄ _ _ q)
    (mul_le_mul_of_nonneg_left ?_ hw₄)
  calc ∑ l, |relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ (b + e) x₀))))))))) l -
        relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ b x₀))))))))) l|
      ≤ ∑ l, |dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ (b + e) x₀)))))))) l -
          dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b x₀)))))))) l| :=
        Finset.sum_le_sum fun l _ => relu_entry_lipschitz _ _ _ l
    _ ≤ ∑ _l : Fin d₃, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
          (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx|))) :=
        Finset.sum_le_sum fun l _ =>
          cnnb1_z3_drift W₁ x₀ W₂ b₂ W₃ b₃ hw₂ hW₂ hw₃ hW₃ b e l
    _ = (d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
          (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx|)))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

/-- Logit drift through the whole conv1-bias chain. -/
theorem cnnb1_logit_drift {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    {w₂ w₃ w₄ w₅ : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (b e : Vec c) (k : Fin nC) :
    |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₁ (b + e)
              x₀)))))))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₁ b
              x₀)))))))))))) k| ≤
      w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) *
        (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx|))))))) := by
  refine le_trans (dense_input_drift W₅ b₅ hW₅ _ _ k)
    (mul_le_mul_of_nonneg_left ?_ hw₅)
  calc ∑ q, |relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ (b + e) x₀))))))))))) q -
        relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ b x₀))))))))))) q|
      ≤ ∑ q, |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ (b + e) x₀)))))))))) q -
          dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
              (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b x₀)))))))))) q| :=
        Finset.sum_le_sum fun q _ => relu_entry_lipschitz _ _ _ q
    _ ≤ ∑ _q : Fin d₄, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) *
          (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx|))))) :=
        Finset.sum_le_sum fun q _ =>
          cnnb1_z4_drift W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ hw₂ hW₂ hw₃ hW₃
            hw₄ hW₄ b e q
    _ = (d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) *
          (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |e idx|)))))) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul]

-- ════════════════════════════════════════════════════════════════
-- § conv1-bias margins freeze every routing decision along the segment
-- ════════════════════════════════════════════════════════════════

/-- The relu₁ margin (at the bias radius `D`) keeps the conv1
    pre-activation off the kink. -/
theorem cnnb1_margin1_keeps_offkink {ic c h w kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    {D : ℝ} (b e : Vec c) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ k, D < |Tensor3.flatten (conv2d W₁ b x₀) k|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (k : Fin (c * (2*h) * (2*w))) :
    Tensor3.flatten (conv2d W₁ (b + t • e) x₀) k ≠ 0 ∧
      (0 < Tensor3.flatten (conv2d W₁ (b + t • e) x₀) k
        ↔ 0 < Tensor3.flatten (conv2d W₁ b x₀) k) := by
  refine sign_stable_of_close ?_ (hm k)
  have h1 := conv2d_flat_bias_drift_total W₁ x₀ b (t • e) k
  have h2 : (∑ idx, |(t • e) idx|) ≤ D := smul_l1_mass_le e ht0 ht1 he
  linarith

/-- The relu₂ margin (at the conv1-bias radius) keeps the conv2
    pre-activation off the kink. -/
theorem cnnb1_margin2_keeps_offkink {ic c h w kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {w₂ D : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (b e : Vec c) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * D) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀))))) k|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (k : Fin (c * (2*h) * (2*w))) :
    Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ (b + t • e) x₀))))) k ≠ 0 ∧
      (0 < Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + t • e) x₀))))) k ↔
        0 < Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀))))) k) := by
  refine sign_stable_of_close ?_ (hm k)
  have h1 := cnnb1_z2_entry_drift W₁ x₀ W₂ b₂ hw₂ hW₂ b (t • e) k
  have h2 : ((c * kH * kW : ℕ) : ℝ) * (w₂ * ∑ idx, |(t • e) idx|) ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * D) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (smul_l1_mass_le e ht0 ht1 he) hw₂) (Nat.cast_nonneg _)
  linarith

/-- The POST-relu₂ tensor stays within the conv1-bias pool margin radius
    along the whole step segment. -/
theorem cnnb1_postrelu2_close_seg {ic c h w kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    {w₂ D : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (b e : Vec c) (he : (∑ idx, |e idx|) ≤ D)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1)
    (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)) :
    |(Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + t • e) x₀))))))) :
        Tensor3 c (2*h) (2*w)) ci hi wi -
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀))))))) :
        Tensor3 c (2*h) (2*w)) ci hi wi| ≤
      ((c * kH * kW : ℕ) : ℝ) * (w₂ * D) :=
  le_trans (cnnb1_postrelu2_close W₁ x₀ W₂ b₂ hw₂ hW₂ b (t • e)
      ci hi wi)
    (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (smul_l1_mass_le e ht0 ht1 he) hw₂) (Nat.cast_nonneg _))

/-- The relu₃ margin (at the conv1-bias radius) keeps the first head
    pre-activation off the kink. -/
theorem cnnb1_margin3_keeps_offkink {ic c h w d₃ kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    {w₂ w₃ D : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (b e : Vec c) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀)))))))) l|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (l : Fin d₃) :
    dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + t • e) x₀)))))))) l ≠ 0 ∧
      (0 < dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ (b + t • e) x₀)))))))) l ↔
        0 < dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ b x₀)))))))) l) := by
  refine sign_stable_of_close ?_ (hm l)
  have h1 := cnnb1_z3_drift W₁ x₀ W₂ b₂ W₃ b₃ hw₂ hW₂ hw₃ hW₃
    b (t • e) l
  have h2 : w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |(t • e) idx|))) ≤
      w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (smul_l1_mass_le e ht0 ht1 he) (Nat.cast_nonneg _)) hw₂)
      (Nat.cast_nonneg _)) hw₃
  linarith

/-- The relu₄ margin (at the conv1-bias radius) keeps the second head
    pre-activation off the kink. -/
theorem cnnb1_margin4_keeps_offkink {ic c h w d₃ d₄ kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    {w₂ w₃ w₄ D : ℝ} (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (b e : Vec c) (he : (∑ idx, |e idx|) ≤ D)
    (hm : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀)))))))))) q|)
    (t : ℝ) (ht0 : 0 ≤ t) (ht1 : t ≤ 1) (q : Fin d₄) :
    dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + t • e) x₀))))))))))
        q ≠ 0 ∧
      (0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ (b + t • e) x₀))))))))))
          q ↔
        0 < dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ b x₀)))))))))) q) := by
  refine sign_stable_of_close ?_ (hm q)
  have h1 := cnnb1_z4_drift W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ hw₂ hW₂ hw₃ hW₃
    hw₄ hW₄ b (t • e) q
  have h2 : w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
      (((2*h * (2*w) : ℕ) : ℝ) * ∑ idx, |(t • e) idx|))))) ≤
      w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))))) :=
    mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
          (smul_l1_mass_le e ht0 ht1 he) (Nat.cast_nonneg _)) hw₂)
        (Nat.cast_nonneg _)) hw₃) (Nat.cast_nonneg _)) hw₄
  linarith

-- ════════════════════════════════════════════════════════════════
-- § The conv1 loss-of-bias map: differentiability and gradient
-- ════════════════════════════════════════════════════════════════

/-- The loss-of-conv1-bias map is differentiable at any five-condition
    point. -/
theorem cnn_conv1_bias_loss_differentiableAt {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC) (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    (b : Vec c)
    (hz1 : ∀ k, Tensor3.flatten (conv2d W₁ b x₀) k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₁ b x₀))))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀))))))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀)))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀)))))))))) q ≠ 0) :
    DifferentiableAt ℝ
      (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b' x₀)))))))))))))
          label) b := by
  have hG1 := cnn1_pool_head_differentiableAt W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
    label hc hh hw (Tensor3.flatten (conv2d W₁ b x₀))
    hz1 hz2 hmp hz3 hz4
  have h0 : DifferentiableAt ℝ
      (fun b' : Vec c => Tensor3.flatten (conv2d W₁ b' x₀)) b :=
    (conv2d_bias_differentiable W₁ x₀) b
  exact ((differentiableAt_pi.mp hG1) 0).comp
    (f := fun b' : Vec c => Tensor3.flatten (conv2d W₁ b' x₀)) b h0

/-- **Closed form of the conv1 bias loss gradient** at any five-margin
    point — the bias fold at conv1, contracted with the conv1 head
    gradient (`cnn1_pool_head_input_grad`, reused verbatim): the
    Kronecker bias Jacobian times relu₁'s mask times the point-free
    conv2 tap Jacobian times the pool-collapsed head. -/
theorem cnn_conv1_bias_loss_gradAt {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC) (hh : 0 < h) (hw : 0 < w)
    (b : Vec c)
    (hz1 : ∀ k, Tensor3.flatten (conv2d W₁ b x₀) k ≠ 0)
    (hz2 : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₁ b x₀))))) k ≠ 0)
    (hmp : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀))))))) :
      Tensor3 c (2*h) (2*w)))
    (hz3 : ∀ l, dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀)))))))) l ≠ 0)
    (hz4 : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀)))))))))) q ≠ 0)
    (o : Fin c) :
    gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b' x₀)))))))))))))
          label)
        b o
      = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then (1:ℝ) else 0) *
            ((if Tensor3.flatten (conv2d W₁ b x₀)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
                convTap W₂ ci hi wi co ho wo *
                  ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d W₁ b x₀)))))
                        (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                    (if MaxPool2IsArgmax (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₁ b
                                  x₀)))))))) co ho wo
                      then ∑ l, W₃ (t3Idx co (winRow ho) (winCol wo)) l *
                        ((if dense W₃ b₃ (maxPoolFlat c h w
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₂ b₂ (Tensor3.unflatten
                                  (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (conv2d
                                      W₁ b
                                      x₀)))))))) l > 0
                            then (1:ℝ) else 0) *
                          ∑ q, W₄ l q *
                            ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                  (maxPoolFlat c h w (relu
                                    (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (conv2d W₂ b₂ (Tensor3.unflatten
                                      (relu (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d
                                          W₁ b
                                          x₀)))))))))) q > 0
                                then (1:ℝ) else 0) *
                              ∑ k, W₅ q k *
                                (softmax nC (dense W₅ b₅ (relu d₄
                                    (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                      (maxPoolFlat c h w (relu
                                        (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d W₂ b₂
                                          (Tensor3.unflatten (relu
                                            (c * (2*h) * (2*w))
                                            (Tensor3.flatten (conv2d
                                              W₁ b
                                              x₀))))))))))))) k -
                                  oneHot nC label k)))
                      else 0))) := by
  have hc : 0 < c := Fin.pos o
  have hdiff := cnn_conv1_bias_loss_differentiableAt W₁ x₀ W₂ b₂ W₃ b₃
    W₄ b₄ W₅ b₅ label hc hh hw b hz1 hz2 hmp hz3 hz4
  have hG1 := cnn1_pool_head_differentiableAt W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
    label hc hh hw (Tensor3.flatten (conv2d W₁ b x₀))
    hz1 hz2 hmp hz3 hz4
  calc gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b' x₀)))))))))))))
          label)
        b o
      = pdiv (fun b' : Vec c => fun _ : Fin 1 =>
          crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
            (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
              (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁ b' x₀)))))))))))))
            label)
          b o 0 := gradAt_eq_pdiv _ _ hdiff _
    _ = ∑ k : Fin (c * (2*h) * (2*w)),
          pdiv (fun b' : Vec c =>
              Tensor3.flatten (conv2d W₁ b' x₀)) b o k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) y)))))))))))
                label)
            (Tensor3.flatten (conv2d W₁ b x₀)) k 0 :=
        conv_bias_total_loss_grad_fold W₁ x₀ b
          (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
            crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
              (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                  (relu (c * (2*h) * (2*w)) y)))))))))))
              label)
          hG1 o
    _ = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          pdiv (fun b' : Vec c =>
              Tensor3.flatten (conv2d W₁ b' x₀)) b o (t3Idx ci hi wi) *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) y)))))))))))
                label)
            (Tensor3.flatten (conv2d W₁ b x₀))
            (t3Idx ci hi wi) 0 :=
        sum_t3 (fun k : Fin (c * (2*h) * (2*w)) =>
          pdiv (fun b' : Vec c =>
              Tensor3.flatten (conv2d W₁ b' x₀)) b o k *
          pdiv (fun y : Vec (c * (2*h) * (2*w)) => fun _ : Fin 1 =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) y)))))))))))
                label)
            (Tensor3.flatten (conv2d W₁ b x₀)) k 0)
    _ = ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          (if ci = o then (1:ℝ) else 0) *
            ((if Tensor3.flatten (conv2d W₁ b x₀)
                  (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) *
              ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
                convTap W₂ ci hi wi co ho wo *
                  ((if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                          (conv2d W₁ b x₀)))))
                        (t3Idx co ho wo) > 0 then (1:ℝ) else 0) *
                    (if MaxPool2IsArgmax (Tensor3.unflatten
                          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                            (conv2d W₂ b₂ (Tensor3.unflatten
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₁ b
                                  x₀)))))))) co ho wo
                      then ∑ l, W₃ (t3Idx co (winRow ho) (winCol wo)) l *
                        ((if dense W₃ b₃ (maxPoolFlat c h w
                              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                                (conv2d W₂ b₂ (Tensor3.unflatten
                                  (relu (c * (2*h) * (2*w))
                                    (Tensor3.flatten (conv2d
                                      W₁ b
                                      x₀)))))))) l > 0
                            then (1:ℝ) else 0) *
                          ∑ q, W₄ l q *
                            ((if dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                  (maxPoolFlat c h w (relu
                                    (c * (2*h) * (2*w)) (Tensor3.flatten
                                    (conv2d W₂ b₂ (Tensor3.unflatten
                                      (relu (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d
                                          W₁ b
                                          x₀)))))))))) q > 0
                                then (1:ℝ) else 0) *
                              ∑ k, W₅ q k *
                                (softmax nC (dense W₅ b₅ (relu d₄
                                    (dense W₄ b₄ (relu d₃ (dense W₃ b₃
                                      (maxPoolFlat c h w (relu
                                        (c * (2*h) * (2*w))
                                        (Tensor3.flatten (conv2d W₂ b₂
                                          (Tensor3.unflatten (relu
                                            (c * (2*h) * (2*w))
                                            (Tensor3.flatten (conv2d
                                              W₁ b
                                              x₀))))))))))))) k -
                                  oneHot nC label k)))
                      else 0))) := by
        refine Finset.sum_congr rfl fun ci _ => Finset.sum_congr rfl
          fun hi _ => Finset.sum_congr rfl fun wi _ => ?_
        rw [conv2d_bias_pdiv W₁ x₀ b o ci hi wi,
          cnn1_pool_head_input_grad W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label _
            hz1 hz2 hmp hz3 hz4 ci hi wi]

-- ════════════════════════════════════════════════════════════════
-- § Segment-Lipschitz gradient for the conv1 bias loss, explicit constant
-- ════════════════════════════════════════════════════════════════

/-- **Segment-Lipschitz gradient for the conv1-bias loss, explicit
    constant.** The conv1-kernel argument with the conv1 stage's `a·D`
    radii replaced by the bare `D` — the bias Jacobian is a Kronecker
    indicator with row mass `(2h)·(2w)`. Constant: the conv1-kernel
    constant with `a² ↦ 1`. -/
theorem cnn_conv1_bias_loss_grad_lipschitz {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃)
    (W₄ : Mat d₃ d₄) (b₄ : Vec d₄) (W₅ : Mat d₄ nC) (b₅ : Vec nC)
    (label : Fin nC) (hh : 0 < h) (hw : 0 < w)
    {w₂ w₃ w₄ w₅ D : ℝ}
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (b d : Vec c) (hd : (∑ idx, |d idx|) ≤ D)
    (hm1 : ∀ k, D < |Tensor3.flatten (conv2d W₁ b x₀) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * D) <
      |Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀))))) k|)
    (hmq : MaxPool2MarginQ (((c * kH * kW : ℕ) : ℝ) * (w₂ * D))
      (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀)))))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))) <
      |dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀)))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
        (((2*h * (2*w) : ℕ) : ℝ) * D))))) <
      |dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀)))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        D)))))))) < 1)
    (t : ℝ) (ht : t ∈ Set.Icc (0:ℝ) 1)
    (o : Fin c) :
    |gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b' x₀))))))))))))) label)
        (b + t • d) o -
      gradAt (fun b' : Vec c =>
        crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b' x₀))))))))))))) label)
        b o| ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 *
        ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 *
        w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            D)))))))))) * (t * D) := by
  obtain ⟨ht0, ht1⟩ := ht
  have hD0 : 0 ≤ D :=
    le_trans (Finset.sum_nonneg fun _ _ => abs_nonneg _) hd
  have hδ0 : (0:ℝ) ≤ w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        D))))))) :=
    mul_nonneg hw₅ (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
        (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₂
          (mul_nonneg (Nat.cast_nonneg _) hD0)))))))
  have hden : (0:ℝ) < 1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        D)))))))) := by linarith
  have hKw0 : (0:ℝ) ≤ ((c * kH * kW : ℕ) : ℝ) * (w₂ * D) :=
    mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₂ hD0)
  -- base-point conditions from the margins
  have hz1_v : ∀ k, Tensor3.flatten (conv2d W₁ b x₀) k ≠ 0 :=
    fun k h0 => by
      have hk := hm1 k
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr hD0)
  have hz2_v : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₁ b x₀))))) k ≠ 0 :=
    fun k h0 => by
      have hk := hm2 k
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr hKw0)
  have hmp_v : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀))))))) :
      Tensor3 c (2*h) (2*w)) := hmq.smooth hKw0
  have hz3_v : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀)))))))) l ≠ 0 :=
    fun l h0 => by
      have hk := hm3 l
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr (mul_nonneg hw₃ (mul_nonneg
        (Nat.cast_nonneg _) (mul_nonneg hw₂ (mul_nonneg
          (Nat.cast_nonneg _) hD0)))))
  have hz4_v : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀)))))))))) q ≠ 0 :=
    fun q h0 => by
      have hk := hm4 q
      rw [h0, abs_zero] at hk
      exact absurd hk (not_lt.mpr (mul_nonneg hw₄ (mul_nonneg
        (Nat.cast_nonneg _) (mul_nonneg hw₃ (mul_nonneg
          (Nat.cast_nonneg _) (mul_nonneg hw₂ (mul_nonneg
            (Nat.cast_nonneg _) hD0)))))))
  -- segment-point conditions: everything frozen
  have hstab1 := fun k =>
    cnnb1_margin1_keeps_offkink W₁ x₀ b d hd hm1 t ht0 ht1 k
  have hz1_t : ∀ k, Tensor3.flatten
      (conv2d W₁ (b + t • d) x₀) k ≠ 0 :=
    fun k => (hstab1 k).1
  have hstab2 := fun k =>
    cnnb1_margin2_keeps_offkink W₁ x₀ W₂ b₂ hw₂ hW₂ b d hd hm2
      t ht0 ht1 k
  have hz2_t : ∀ k, Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₁ (b + t • d) x₀))))) k ≠ 0 :=
    fun k => (hstab2 k).1
  have hclose := fun ci hi wi =>
    cnnb1_postrelu2_close_seg W₁ x₀ W₂ b₂ hw₂ hW₂ b d hd
      t ht0 ht1 ci hi wi
  have hmp_t : MaxPool2Smooth (Tensor3.unflatten (relu (c * (2*h) * (2*w))
      (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ (b + t • d) x₀))))))) :
      Tensor3 c (2*h) (2*w)) := hmq.smooth_of_close hclose
  have hstab3 := fun l =>
    cnnb1_margin3_keeps_offkink W₁ x₀ W₂ b₂ W₃ b₃ hw₂ hW₂ hw₃ hW₃
      b d hd hm3 t ht0 ht1 l
  have hz3_t : ∀ l, dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ (b + t • d) x₀)))))))) l ≠ 0 :=
    fun l => (hstab3 l).1
  have hstab4 := fun q =>
    cnnb1_margin4_keeps_offkink W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ hw₂ hW₂
      hw₃ hW₃ hw₄ hW₄ b d hd hm4 t ht0 ht1 q
  have hz4_t : ∀ q, dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
        (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ (b + t • d) x₀)))))))))) q ≠ 0 :=
    fun q => (hstab4 q).1
  -- both gradients in closed form
  rw [cnn_conv1_bias_loss_gradAt W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw
      (b + t • d) hz1_t hz2_t hmp_t hz3_t hz4_t o,
    cnn_conv1_bias_loss_gradAt W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ label hh hw
      b hz1_v hz2_v hmp_v hz3_v hz4_v o]
  -- the frozen masks and the frozen routing
  have hmask1 : ∀ (ci : Fin c) (hi : Fin (2*h)) (wi : Fin (2*w)),
      (if Tensor3.flatten (conv2d W₁ (b + t • d) x₀)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) =
      (if Tensor3.flatten (conv2d W₁ b x₀)
          (t3Idx ci hi wi) > 0 then (1:ℝ) else 0) := by
    intro ci hi wi
    by_cases hp : Tensor3.flatten (conv2d W₁ b x₀)
        (t3Idx ci hi wi) > 0
    · rw [if_pos ((hstab1 _).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab1 _).2.mp hgt)), if_neg hp]
  have hmask2 : ∀ k : Fin (c * (2*h) * (2*w)),
      (if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + t • d) x₀))))) k > 0
        then (1:ℝ) else 0) =
      (if Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀))))) k > 0
        then (1:ℝ) else 0) := by
    intro k
    by_cases hp : Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₁ b x₀))))) k > 0
    · rw [if_pos ((hstab2 _).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab2 _).2.mp hgt)), if_neg hp]
  have hmask3 : ∀ l : Fin d₃,
      (if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ (b + t • d) x₀))))))))
          l > 0 then (1:ℝ) else 0) =
      (if dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ b x₀))))))))
          l > 0 then (1:ℝ) else 0) := by
    intro l
    by_cases hp : dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀)))))))) l > 0
    · rw [if_pos ((hstab3 l).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab3 l).2.mp hgt)), if_neg hp]
  have hmask4 : ∀ q : Fin d₄,
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ (b + t • d) x₀))))))))))
          q > 0 then (1:ℝ) else 0) =
      (if dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
            (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ b x₀))))))))))
          q > 0 then (1:ℝ) else 0) := by
    intro q
    by_cases hp : dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w
        (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂
          (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀)))))))))) q > 0
    · rw [if_pos ((hstab4 q).2.mpr hp), if_pos hp]
    · rw [if_neg (fun hgt => hp ((hstab4 q).2.mp hgt)), if_neg hp]
  have hargiff : ∀ (co : Fin c) (ho : Fin (2*h)) (wo : Fin (2*w)),
      MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + t • d) x₀))))))))
        co ho wo ↔
      MaxPool2IsArgmax (Tensor3.unflatten (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀))))))))
        co ho wo :=
    fun co ho wo => hmq.isArgmax_iff hclose co ho wo
  -- the softmax drift along the segment
  have hzdrift : ∀ k, |dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + t • d) x₀)))))))))))) k -
      dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₁ b
              x₀)))))))))))) k| ≤
      t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          D)))))))) := by
    intro k
    have h1 := cnnb1_logit_drift W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅
      hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ b (t • d) k
    rw [smul_l1_mass d ht0] at h1
    have h2 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          (t * ∑ idx, |d idx|)))))))) =
        t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            ∑ idx, |d idx|)))))))) := by
      ring
    rw [h2] at h1
    have h3 : w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          ∑ idx, |d idx|))))))) ≤
        w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            D))))))) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
        (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
          (mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left
              (mul_le_mul_of_nonneg_left hd (Nat.cast_nonneg _)) hw₂)
            (Nat.cast_nonneg _)) hw₃) (Nat.cast_nonneg _)) hw₄)
        (Nat.cast_nonneg _)) hw₅
    have h4 := mul_le_mul_of_nonneg_left h3 ht0
    linarith
  have hδlt : 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        D))))))))) < 1 := by
    nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
  have hexp := FloatModel.exp_sub_one_le hδlt
  have hmono : 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          D))))))))) /
        (1 - 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            D)))))))))) ≤
      2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          D))))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            D))))))))) := by
    refine div_le_div_of_nonneg_left
      (by nlinarith [mul_nonneg ht0 hδ0]) hden ?_
    nlinarith [mul_le_mul_of_nonneg_right ht1 hδ0]
  have hS : ∀ k, |softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
      (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
        (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ (b + t • d)
              x₀))))))))))))) k -
      softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃
        (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten
          (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₁ b
              x₀))))))))))))) k| ≤
      2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          D))))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            D))))))))) :=
    fun k => le_trans (FloatModel.softmax_perturb _ _ hzdrift k)
      (le_trans hexp hmono)
  have hΔ0 : (0:ℝ) ≤ 2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
      (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
        D))))))))) /
      (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
          D))))))))) :=
    div_nonneg (mul_nonneg (by norm_num) (mul_nonneg ht0 hδ0)) hden.le
  have hM0 : (0:ℝ) ≤ (d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
      (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
        (((c * kH * kW : ℕ) : ℝ) * (w₂ *
          (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ *
            (((2*h * (2*w) : ℕ) : ℝ) * D))))))))))))))) :=
    mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₃
      (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₄
        (mul_nonneg (Nat.cast_nonneg _) (mul_nonneg hw₅ hΔ0)))))
  -- the conv1-bias Jacobian row mass
  have hcp : ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
      |if ci = o then (1:ℝ) else 0| ≤ ((2*h * (2*w) : ℕ) : ℝ) := by
    rw [Finset.sum_eq_single o
      (fun ci _ hne => by
        rw [Finset.sum_eq_zero]
        intro hi _
        rw [Finset.sum_eq_zero]
        intro wi _
        rw [if_neg hne, abs_zero])
      (fun habs => absurd (Finset.mem_univ _) habs)]
    calc ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
          |if o = o then (1:ℝ) else 0|
        ≤ ∑ _hi : Fin (2*h), ∑ _wi : Fin (2*w), (1:ℝ) := by
          refine Finset.sum_le_sum fun hi _ =>
            Finset.sum_le_sum fun wi _ => ?_
          rw [if_pos rfl, abs_one]
      _ = ((2*h * (2*w) : ℕ) : ℝ) := by
          rw [Finset.sum_const, Finset.sum_const, Finset.card_univ,
            Finset.card_univ, Fintype.card_fin, Fintype.card_fin,
            smul_smul, nsmul_eq_mul, mul_one]
  -- the endgame
  have hfinal : ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
      (|if ci = o then (1:ℝ) else 0| *
        (((c * kH * kW : ℕ) : ℝ) * w₂ *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) * D)))))))))))))))))) ≤
      (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 *
        ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 *
        w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
        (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
          (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) *
            D)))))))))) * (t * D) := by
    calc ∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
        (|if ci = o then (1:ℝ) else 0| *
          (((c * kH * kW : ℕ) : ℝ) * w₂ *
            ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
              (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
                (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                  (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                    (((2*h * (2*w) : ℕ) : ℝ) * D))))))))))))))))))
        = (∑ ci : Fin c, ∑ hi : Fin (2*h), ∑ wi : Fin (2*w),
            |if ci = o then (1:ℝ) else 0|) *
            (((c * kH * kW : ℕ) : ℝ) * w₂ *
              ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
                (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) *
                  (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                    (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
                  (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                    (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                      (((2*h * (2*w) : ℕ) : ℝ) *
                        D))))))))))))))))) := by
          simp only [← Finset.sum_mul]
      _ ≤ ((2*h * (2*w) : ℕ) : ℝ) *
            (((c * kH * kW : ℕ) : ℝ) * w₂ *
              ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
                (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) *
                  (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                    (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
                  (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                    (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                      (((2*h * (2*w) : ℕ) : ℝ) *
                        D))))))))))))))))) :=
          mul_le_mul_of_nonneg_right hcp
            (mul_nonneg (mul_nonneg (Nat.cast_nonneg _) hw₂) hM0)
      _ = (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 *
            ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 *
            w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * D)))))))))) *
            (t * D) := by
          ring
  refine le_trans (le_trans (abs_triple_sum_sub_le _ _)
    (Finset.sum_le_sum fun ci _ => Finset.sum_le_sum fun hi _ =>
      Finset.sum_le_sum fun wi _ => ?_)) hfinal
  -- per-term: freeze relu₁'s mask, then bound the conv2 contraction
  rw [hmask1 ci hi wi]
  simp only [hmask2, hmask3, hmask4]
  rw [← mul_sub, abs_mul, ← mul_sub, abs_mul]
  refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
  refine le_trans (mul_le_of_le_one_left (abs_nonneg _) ?_) ?_
  · split_ifs <;> simp
  -- the conv2 contraction: point-free taps times the frozen-route drift
  have hlast := calc ∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
      (|convTap W₂ ci hi wi co ho wo| *
        ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
          (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
            (((c * kH * kW : ℕ) : ℝ) * (w₂ *
              (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * D)))))))))))))))))
      = (∑ co : Fin c, ∑ ho : Fin (2*h), ∑ wo : Fin (2*w),
          |convTap W₂ ci hi wi co ho wo|) *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) *
                    D)))))))))))))))) := by
                        simp only [← Finset.sum_mul]
    _ ≤ (((c * kH * kW : ℕ) : ℝ) * w₂) *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) *
                    D)))))))))))))))) :=
        mul_le_mul_of_nonneg_right
          (convTap_out_l1 W₂ hW₂ ci hi wi) hM0
    _ = ((c * kH * kW : ℕ) : ℝ) * w₂ *
          ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
            (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
              (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
                (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                  (((2*h * (2*w) : ℕ) : ℝ) *
                    D)))))))))))))))) := by
          ring
  refine le_trans (abs_triple_sum_sub_le _ _) ?_
  refine le_trans (Finset.sum_le_sum fun co _ => Finset.sum_le_sum
    fun ho _ => Finset.sum_le_sum fun wo _ => ?_) hlast
  show |convTap W₂ ci hi wi co ho wo * _ -
        convTap W₂ ci hi wi co ho wo * _| ≤
      |convTap W₂ ci hi wi co ho wo| *
        ((d₃ : ℝ) * (w₃ * ((d₄ : ℝ) * (w₄ * ((nC : ℝ) *
          (w₅ * (2 * (t * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
            (((c * kH * kW : ℕ) : ℝ) * (w₂ *
              (((2*h * (2*w) : ℕ) : ℝ) * D))))))))) /
            (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ *
              (((c * kH * kW : ℕ) : ℝ) * (w₂ *
                (((2*h * (2*w) : ℕ) : ℝ) * D))))))))))))))))
  rw [← mul_sub, abs_mul]
  refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
  by_cases hA : MaxPool2IsArgmax (Tensor3.unflatten
      (relu (c * (2*h) * (2*w)) (Tensor3.flatten
        (conv2d W₂ b₂ (Tensor3.unflatten
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₁ b x₀))))))))
      co ho wo
  · rw [if_pos ((hargiff co ho wo).mpr hA), if_pos hA, ← mul_sub,
      abs_mul]
    refine le_trans (mul_le_of_le_one_left (abs_nonneg _) ?_) ?_
    · split_ifs <;> simp
    · exact head3_sum_drift W₃ W₄ W₅ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅
        (fun l => if dense W₃ b₃ (maxPoolFlat c h w
          (relu (c * (2*h) * (2*w)) (Tensor3.flatten
            (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁ b x₀)))))))) l > 0
          then (1:ℝ) else 0)
        (fun l => by split_ifs <;> simp)
        (fun q => if dense W₄ b₄ (relu d₃ (dense W₃ b₃
          (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
            (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
              (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                (conv2d W₁
                  b x₀)))))))))) q > 0
          then (1:ℝ) else 0)
        (fun q => by split_ifs <;> simp)
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁
                    b x₀))))))))))))))
        (softmax nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
          (dense W₃ b₃ (maxPoolFlat c h w
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₂ b₂ (Tensor3.unflatten
                (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                  (conv2d W₁ (b + t • d)
                    x₀))))))))))))))
        (oneHot nC label) hS (t3Idx co (winRow ho) (winCol wo))
  · rw [if_neg (fun hA' => hA ((hargiff co ho wo).mp hA')),
      if_neg hA]
    simp only [mul_zero, sub_self, abs_zero]
    exact hM0


-- ════════════════════════════════════════════════════════════════
-- § The conv1-bias capstone: one inexact SGD step provably descends
-- ════════════════════════════════════════════════════════════════

/-- **One inexact SGD step on the CNN's FIRST conv BIAS provably
    decreases the cross-entropy loss.** The conv1-kernel capstone with
    the bias-rung radii: the FIVE margins at the step radius
    `D = lr·(‖∇L‖₁ + c·η)` carry no input bound `a` (the bias Jacobian
    is a Kronecker indicator) and the parameter needs no
    flatten/unflatten plumbing. With this theorem every parameter of
    the Chapter-4 CNN — both conv kernels, both conv biases, and the
    three dense layers (weights and biases via the MLP rungs) — has a
    proven descent statement. -/
theorem cnn_conv1_bias_sgd_descends {ic c h w d₃ d₄ nC kH kW : Nat}
    (W₁ : Kernel4 c ic kH kW) (b₁ : Vec c) (x₀ : Tensor3 ic (2*h) (2*w))
    (W₂ : Kernel4 c c kH kW) (b₂ : Vec c)
    (W₃ : Mat (c * h * w) d₃) (b₃ : Vec d₃) (W₄ : Mat d₃ d₄) (b₄ : Vec d₄)
    (W₅ : Mat d₄ nC) (b₅ : Vec nC) (label : Fin nC)
    (gh : Vec c)
    (hc : 0 < c) (hh : 0 < h) (hw : 0 < w)
    {lr η w₂ w₃ w₄ w₅ : ℝ}
    (hw₂ : 0 ≤ w₂) (hW₂ : ∀ o cc kh kw, |W₂ o cc kh kw| ≤ w₂)
    (hw₃ : 0 ≤ w₃) (hW₃ : ∀ i j, |W₃ i j| ≤ w₃)
    (hw₄ : 0 ≤ w₄) (hW₄ : ∀ i j, |W₄ i j| ≤ w₄)
    (hw₅ : 0 ≤ w₅) (hW₅ : ∀ i j, |W₅ i j| ≤ w₅)
    (hlr : 0 ≤ lr) (hη : 0 ≤ η)
    (hgh : ∀ idx, |gh idx - (gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁) idx| ≤ η)
    (hm1 : ∀ k, lr * (((∑ idx, |gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁ idx|) + (c : ℝ) * η)) < |(Tensor3.flatten (conv2d W₁ b₁ x₀)) k|)
    (hm2 : ∀ k, ((c * kH * kW : ℕ) : ℝ) * (w₂ * (lr * (((∑ idx, |gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁ idx|) + (c : ℝ) * η)))) < |(Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))) k|)
    (hmq : MaxPool2MarginQ (((c * kH * kW : ℕ) : ℝ) * (w₂ * (lr * (((∑ idx, |gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁ idx|) + (c : ℝ) * η))))) (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀)))))))))
    (hm3 : ∀ l, w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁ idx|) + (c : ℝ) * η)))))) < |(dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))) l|)
    (hm4 : ∀ q, w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁ idx|) + (c : ℝ) * η))))))))
      < |(dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))))) q|)
    (hsmall : 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁ idx|) + (c : ℝ) * η))))))))))) < 1)
    (h1 : lr * η * (∑ idx, |gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁ idx|) ≤
      lr * (∑ idx, (gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁) idx ^ 2) / 4)
    (h2 : (2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 / (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (lr * (((∑ idx, |gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁ idx|) + (c : ℝ) * η))))))))))))) * (lr * ((∑ idx, |gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁ idx|) + (c : ℝ) * η)) ^ 2 ≤
      lr * (∑ idx, (gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁) idx ^ 2) / 4) :
    crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ (b₁ - lr • gh) x₀))))))))))))) label ≤
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃ (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten (relu (c * (2*h) * (2*w)) (Tensor3.flatten (conv2d W₁ b₁ x₀))))))))))))) label -
        lr * (∑ idx, (gradAt (fun b' : Vec c =>
              crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
                (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
                  (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
                    (relu (c * (2*h) * (2*w)) (Tensor3.flatten
                      (conv2d W₁ b' x₀))))))))))))) label)
              b₁) idx ^ 2) / 2 := by
  set f : Vec c → ℝ :=
    fun b' : Vec c =>
      crossEntropy nC (dense W₅ b₅ (relu d₄ (dense W₄ b₄ (relu d₃
        (dense W₃ b₃ (maxPoolFlat c h w (relu (c * (2*h) * (2*w))
          (Tensor3.flatten (conv2d W₂ b₂ (Tensor3.unflatten
            (relu (c * (2*h) * (2*w)) (Tensor3.flatten
              (conv2d W₁ b' x₀))))))))))))) label
    with hf
  have hden : (0:ℝ) < 1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (lr * ((∑ idx, |gradAt f b₁ idx|) + (c : ℝ) * η)))))))))) := by linarith
  have hC0 : (0:ℝ) ≤ 2 * (nC : ℝ) * ((2*h * (2*w) : ℕ) : ℝ) ^ 2 * ((c * kH * kW : ℕ) : ℝ) ^ 2 * (d₃ : ℝ) ^ 2 * (d₄ : ℝ) ^ 2 * w₂ ^ 2 * w₃ ^ 2 * w₄ ^ 2 * w₅ ^ 2 / (1 - 2 * (w₅ * ((d₄ : ℝ) * (w₄ * ((d₃ : ℝ) * (w₃ * (((c * kH * kW : ℕ) : ℝ) * (w₂ * (((2*h * (2*w) : ℕ) : ℝ) * (lr * ((∑ idx, |gradAt f b₁ idx|) + (c : ℝ) * η))))))))))) :=
    div_nonneg (by positivity) hden.le
  have hD : (∑ idx, |(-(lr • gh)) idx|) ≤ lr * ((∑ idx, |gradAt f b₁ idx|) + (c : ℝ) * η) := by
    calc (∑ idx, |(-(lr • gh)) idx|) = ∑ idx, lr * |gh idx| := by
          refine Finset.sum_congr rfl fun idx _ => ?_
          simp [abs_mul, abs_of_nonneg hlr]
      _ ≤ ∑ idx, lr * (|gradAt f b₁ idx| + η) := by
          refine Finset.sum_le_sum fun idx _ => ?_
          refine mul_le_mul_of_nonneg_left ?_ hlr
          have h3 : |gh idx| ≤
              |gh idx - gradAt f b₁ idx| + |gradAt f b₁ idx| := by
            simpa using abs_sub_le (gh idx) (gradAt f b₁ idx) 0
          linarith [hgh idx]
      _ = lr * ((∑ idx, |gradAt f b₁ idx|) + (c : ℝ) * η) := by
          rw [← Finset.mul_sum, Finset.sum_add_distrib, Finset.sum_const,
            Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hmain := sgd_descends f b₁ gh hlr hη hC0 hgh
    (fun t ht => cnn_conv1_bias_loss_differentiableAt W₁ x₀ W₂ b₂ W₃ b₃
      W₄ b₄ W₅ b₅ label hc hh hw _
      (fun k => (cnnb1_margin1_keeps_offkink W₁ x₀
        b₁ (-(lr • gh)) hD hm1 t ht.1 ht.2 k).1)
      (fun k => (cnnb1_margin2_keeps_offkink W₁ x₀ W₂ b₂ hw₂ hW₂
        b₁ (-(lr • gh)) hD hm2 t ht.1 ht.2 k).1)
      (hmq.smooth_of_close (fun ci hi wi => cnnb1_postrelu2_close_seg
        W₁ x₀ W₂ b₂ hw₂ hW₂ b₁ (-(lr • gh)) hD t ht.1 ht.2 ci hi wi))
      (fun l => (cnnb1_margin3_keeps_offkink W₁ x₀ W₂ b₂ W₃ b₃ hw₂ hW₂
        hw₃ hW₃ b₁ (-(lr • gh)) hD hm3 t ht.1 ht.2 l).1)
      (fun q => (cnnb1_margin4_keeps_offkink W₁ x₀ W₂ b₂ W₃ b₃ W₄ b₄
        hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ b₁ (-(lr • gh)) hD hm4 t ht.1 ht.2 q).1))
    (fun t ht o => by
      have hlip := cnn_conv1_bias_loss_grad_lipschitz W₁ x₀ W₂ b₂ W₃ b₃
        W₄ b₄ W₅ b₅ label hh hw hw₂ hW₂ hw₃ hW₃ hw₄ hW₄ hw₅ hW₅ b₁
        (-(lr • gh)) hD hm1 hm2 hmq hm3 hm4 hsmall t ht o
      simpa [hf] using hlip)
    h1 h2
  simpa [hf] using hmain

end Proofs
