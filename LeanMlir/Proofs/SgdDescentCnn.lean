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
with the segment-Lipschitz constant explicit. The first conv layer (one
more conv+relu crossing, drift factor `(kH·kW)·ic·w₂`-style) and the
biases are the same argument and are left open. -/

namespace Proofs

open StableHLO

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

/-- The conv output difference under a kernel perturbation, exactly:
    `conv2d` is affine in the kernel. -/
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

end Proofs
