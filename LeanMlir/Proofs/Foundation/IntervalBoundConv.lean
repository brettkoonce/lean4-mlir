import LeanMlir.Proofs.Architectures.CNN

/-! # IBP at conv depth — a compositional interval-bound engine

`IntervalBound.lean` proves the `L∞` certificate for exactly one shape:
`dense ∘ relu ∘ dense`, dense-only, hard-wired at two layers
(`ibp2_certified_at_eps`). That is the shape the 784→16→10 scorecards use, and
it is the reason the IBP tier stops at one hidden layer.

This file removes both restrictions. The soundness argument for IBP is
*compositional* — "the box propagates" is a property of one layer, and depth is
just `∘` — so the engine here is:

* `InBox` / `InBox3` — a point lies in an axis-aligned box (flat / tensor);
* `BoxSound f Flo Fhi` — `(Flo, Fhi)` is a sound interval transformer for `f`;
* `BoxSound.comp` — **depth**: sound transformers compose exactly as the layers
  do, so an `n`-layer net's transformer is `n` compositions and no new proof;
* per-layer instances — `dense` (sign-split), `relu` (endpoint `max`), **`conv2d`
  / `flatConv`** (sign-split over the padded taps), **`maxPool2` / `maxPoolFlat`**
  (`max` is monotone, so pool the endpoints), plus `id`;
* `ibp_certified_of_boxSound` — the capstone: any sound transformer whose output
  boxes separate on `x ∓ ε` certifies `x` at `L∞` radius `ε`, at any depth.

The conv transformer is stated against the repo's own `conv2d`
(`CNN.lean`, SAME padding, the definition the VJP suite and the codegen use) —
not a re-modelled convolution. Padding contributes `0` to both endpoints, which
is why the sign-split can sit *outside* the pad test (`convTap` is monotone,
and constant `0` where the window falls off the image).

`convLo_uniform` / `convHi_uniform` are the conv peers of
`denseLo_uniform`/`denseHi_uniform`: on the first layer's uniform box `x ∓ ε`
the interval image collapses to `conv2d W b x ∓ ε · conv2d |W| 0 𝟙`, i.e. one
extra convolution by the absolute kernel against the all-ones image. That keeps
the generated per-image data at "one forward pass + one `|W|` pass" instead of
two full sign-split sums — the conv analogue of the `⟨w,x⟩ ∓ ε‖w‖₁` collapse the
784-dim files already exploit.

Nonlinearity note: as in `IntervalBound.lean`, unstable (sign-crossing) ReLUs
and tied max-pool windows are handled *soundly*, not assumed away — the box
simply contains both branches. The certificate is sound and incomplete; counts
derived from it are lower bounds only.

Everything is elementary and closes under `propext / Classical.choice /
Quot.sound`. -/

namespace Proofs
namespace IBP

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Boxes
-- ════════════════════════════════════════════════════════════════

/-- `u` lies in the axis-aligned box `[lo, hi]` (flat vectors). -/
def InBox {n : Nat} (lo hi u : Vec n) : Prop := ∀ i, lo i ≤ u i ∧ u i ≤ hi i

/-- `u` lies in the axis-aligned box `[lo, hi]` (rank-3 tensors). -/
def InBox3 {c h w : Nat} (lo hi u : Tensor3 c h w) : Prop :=
  ∀ a b d, lo a b d ≤ u a b d ∧ u a b d ≤ hi a b d

/-- **A sound interval transformer.** `Flo lo hi` / `Fhi lo hi` bracket `f` on
    every point of the box `[lo, hi]`. This is the only property a layer needs
    to contribute to a certificate — it composes (`BoxSound.comp`) and it is
    what the capstone consumes. -/
def BoxSound {n m : Nat} (f : Vec n → Vec m) (Flo Fhi : Vec n → Vec n → Vec m) : Prop :=
  ∀ lo hi u, InBox lo hi u → InBox (Flo lo hi) (Fhi lo hi) (f u)

/-- The rank-3 peer, for the layers that live in tensor space. -/
def BoxSound3 {c h w c' h' w' : Nat} (f : Tensor3 c h w → Tensor3 c' h' w')
    (Flo Fhi : Tensor3 c h w → Tensor3 c h w → Tensor3 c' h' w') : Prop :=
  ∀ lo hi u, InBox3 lo hi u → InBox3 (Flo lo hi) (Fhi lo hi) (f u)

/-- **Depth, for free.** Sound transformers compose exactly as their layers do:
    feed the first layer's output box into the second layer's transformer. An
    `n`-layer net therefore needs `n` per-layer soundness facts and `n-1` uses
    of this lemma — no new argument per depth. -/
theorem BoxSound.comp {n m p : Nat} {f : Vec n → Vec m} {g : Vec m → Vec p}
    {Flo Fhi : Vec n → Vec n → Vec m} {Glo Ghi : Vec m → Vec m → Vec p}
    (hg : BoxSound g Glo Ghi) (hf : BoxSound f Flo Fhi) :
    BoxSound (g ∘ f) (fun lo hi => Glo (Flo lo hi) (Fhi lo hi))
                     (fun lo hi => Ghi (Flo lo hi) (Fhi lo hi)) :=
  fun lo hi u hu => hg _ _ _ (hf lo hi u hu)

/-- Identity is its own transformer (the base case of a layer fold). -/
theorem boxSound_id {n : Nat} : BoxSound (id : Vec n → Vec n) (fun lo _ => lo) (fun _ hi => hi) :=
  fun _ _ _ hu => hu

/-- Depth in tensor space — the `BoxSound3` peer of `BoxSound.comp`, so a conv
    body (`maxpool ∘ relu ∘ conv ∘ …`) composes without round-tripping through
    `flatten` between every layer. -/
theorem BoxSound3.comp {c h w c' h' w' c'' h'' w'' : Nat}
    {f : Tensor3 c h w → Tensor3 c' h' w'} {g : Tensor3 c' h' w' → Tensor3 c'' h'' w''}
    {Flo Fhi : Tensor3 c h w → Tensor3 c h w → Tensor3 c' h' w'}
    {Glo Ghi : Tensor3 c' h' w' → Tensor3 c' h' w' → Tensor3 c'' h'' w''}
    (hg : BoxSound3 g Glo Ghi) (hf : BoxSound3 f Flo Fhi) :
    BoxSound3 (g ∘ f) (fun lo hi => Glo (Flo lo hi) (Fhi lo hi))
                      (fun lo hi => Ghi (Flo lo hi) (Fhi lo hi)) :=
  fun lo hi u hu => hg _ _ _ (hf lo hi u hu)

-- ════════════════════════════════════════════════════════════════
-- § Dense (`Proofs.dense`, biased, `Vec`-space)
-- ════════════════════════════════════════════════════════════════

/-- Sign-split LOWER image of `dense W b`: positive weights pull from `lo`,
    negative from `hi`. Biases shift both endpoints equally. -/
noncomputable def denseLoV {m n : Nat} (W : Mat m n) (b : Vec n) (lo hi : Vec m) : Vec n :=
  fun j => (∑ i, if 0 ≤ W i j then lo i * W i j else hi i * W i j) + b j

/-- Sign-split UPPER image of `dense W b`. -/
noncomputable def denseHiV {m n : Nat} (W : Mat m n) (b : Vec n) (lo hi : Vec m) : Vec n :=
  fun j => (∑ i, if 0 ≤ W i j then hi i * W i j else lo i * W i j) + b j

theorem denseV_boxSound {m n : Nat} (W : Mat m n) (b : Vec n) :
    BoxSound (dense W b) (denseLoV W b) (denseHiV W b) := by
  intro lo hi u hu j
  have hlo : (∑ i, if 0 ≤ W i j then lo i * W i j else hi i * W i j)
      ≤ ∑ i, u i * W i j := by
    refine Finset.sum_le_sum fun i _ => ?_
    by_cases hW : 0 ≤ W i j
    · rw [if_pos hW]; exact mul_le_mul_of_nonneg_right (hu i).1 hW
    · rw [if_neg hW]; exact mul_le_mul_of_nonpos_right (hu i).2 (le_of_not_ge hW)
  have hhi : (∑ i, u i * W i j)
      ≤ ∑ i, if 0 ≤ W i j then hi i * W i j else lo i * W i j := by
    refine Finset.sum_le_sum fun i _ => ?_
    by_cases hW : 0 ≤ W i j
    · rw [if_pos hW]; exact mul_le_mul_of_nonneg_right (hu i).2 hW
    · rw [if_neg hW]; exact mul_le_mul_of_nonpos_right (hu i).1 (le_of_not_ge hW)
  simp only [denseLoV, denseHiV, dense]
  exact ⟨by linarith [hlo], by linarith [hhi]⟩

/-- **Uniform-box collapse for dense** — the `Vec`-space peer of
    `denseLo_uniform`: on `x ∓ ε` the sign split evaluates to
    `⟨x, W⟩ + b ∓ ε·‖W_{·j}‖₁`, so an instance needs one dot and one `ℓ1` fact
    per output, not a sign-split sum. -/
theorem denseLoV_uniform {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) (ε : ℝ) (j : Fin n) :
    denseLoV W b (fun i => x i - ε) (fun i => x i + ε) j
      = ((∑ i, x i * W i j) + b j) - ε * ∑ i, |W i j| := by
  have key : (∑ i, if 0 ≤ W i j then (x i - ε) * W i j else (x i + ε) * W i j)
      = (∑ i, x i * W i j) - ε * ∑ i, |W i j| := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    by_cases hW : 0 ≤ W i j
    · rw [if_pos hW, abs_of_nonneg hW]; ring
    · rw [if_neg hW, abs_of_neg (lt_of_not_ge hW)]; ring
  simp only [denseLoV]
  rw [key]; ring

/-- Sibling of `denseLoV_uniform` for the upper endpoint. -/
theorem denseHiV_uniform {m n : Nat} (W : Mat m n) (b : Vec n) (x : Vec m) (ε : ℝ) (j : Fin n) :
    denseHiV W b (fun i => x i - ε) (fun i => x i + ε) j
      = ((∑ i, x i * W i j) + b j) + ε * ∑ i, |W i j| := by
  have key : (∑ i, if 0 ≤ W i j then (x i + ε) * W i j else (x i - ε) * W i j)
      = (∑ i, x i * W i j) + ε * ∑ i, |W i j| := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    by_cases hW : 0 ≤ W i j
    · rw [if_pos hW, abs_of_nonneg hW]; ring
    · rw [if_neg hW, abs_of_neg (lt_of_not_ge hW)]; ring
  simp only [denseHiV]
  rw [key]; ring

-- ════════════════════════════════════════════════════════════════
-- § ReLU
-- ════════════════════════════════════════════════════════════════

/-- `Proofs.relu`'s `if · > 0` form is the `max` form the box endpoints use. -/
theorem relu_apply_eq_max {n : Nat} (x : Vec n) (i : Fin n) : relu n x i = max (x i) 0 := by
  simp only [relu]
  by_cases h : x i > 0
  · rw [if_pos h, max_eq_left h.le]
  · rw [if_neg h, max_eq_right (le_of_not_gt h)]

noncomputable def reluLoV {n : Nat} (lo : Vec n) : Vec n := fun i => max (lo i) 0
noncomputable def reluHiV {n : Nat} (hi : Vec n) : Vec n := fun i => max (hi i) 0

/-- ReLU is monotone, so the box endpoints are just the ReLU'd endpoints —
    and a *sign-crossing* (unstable) neuron is handled here, not excluded: its
    box becomes `[0, relu hi]`, which contains both branches. -/
theorem reluV_boxSound {n : Nat} :
    BoxSound (relu n) (fun lo _ => reluLoV lo) (fun _ hi => reluHiV hi) := by
  intro lo hi u hu i
  rw [relu_apply_eq_max]
  exact ⟨max_le_max_right 0 (hu i).1, max_le_max_right 0 (hu i).2⟩

/-- Coordinatewise ReLU on tensors — the form a conv body uses between conv and
    pool, so the body never leaves tensor space. -/
noncomputable def reluT {c h w : Nat} (x : Tensor3 c h w) : Tensor3 c h w :=
  fun o i j => max (x o i j) 0

/-- `reluT` IS the repo's `Proofs.relu`, read through `flatten` — so the conv
    body's activation is the same function the VJP suite differentiates, not a
    second ReLU. -/
theorem flatten_reluT {c h w : Nat} (T : Tensor3 c h w) :
    Tensor3.flatten (reluT T) = relu (c * h * w) (Tensor3.flatten T) := by
  funext i
  rw [relu_apply_eq_max]
  rfl

theorem reluT_boxSound3 {c h w : Nat} :
    BoxSound3 (reluT : Tensor3 c h w → Tensor3 c h w)
      (fun lo _ => reluT lo) (fun _ hi => reluT hi) := by
  intro lo hi u hu o i j
  exact ⟨max_le_max_right 0 (hu o i j).1, max_le_max_right 0 (hu o i j).2⟩

/-- The input box of a `CertifiedAtLinfV` goal is stated on the FLAT vector
    `x ∓ ε`; these two put it back in tensor space so the conv transformer's
    uniform-collapse lemmas apply. -/
theorem unflatten_sub_const {c h w : Nat} (T : Tensor3 c h w) (ε : ℝ) :
    Tensor3.unflatten (fun i => Tensor3.flatten T i - ε) = fun a b d => T a b d - ε := by
  funext a b d
  unfold Tensor3.unflatten Tensor3.flatten
  simp [Equiv.symm_apply_apply]

theorem unflatten_add_const {c h w : Nat} (T : Tensor3 c h w) (ε : ℝ) :
    Tensor3.unflatten (fun i => Tensor3.flatten T i + ε) = fun a b d => T a b d + ε := by
  funext a b d
  unfold Tensor3.unflatten Tensor3.flatten
  simp [Equiv.symm_apply_apply]

-- ════════════════════════════════════════════════════════════════
-- § Convolution (`Proofs.conv2d`, SAME padding)
-- ════════════════════════════════════════════════════════════════

/-- The padded input lookup `conv2d` performs, factored out verbatim: the tap
    at kernel offset `(kh, kw)` for output position `(hi, wi)`, or `0` where the
    window falls off the image. -/
noncomputable def convTap {ic h w kH kW : Nat} (x : Tensor3 ic h w)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) : ℝ :=
  let pH := (kH - 1) / 2
  let pW := (kW - 1) / 2
  let hh := kh.val + hi.val
  let ww := kw.val + wi.val
  if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
    x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
  else 0

/-- `conv2d` in tap form — definitional, so the engine below is talking about
    the repo's convolution and not a re-modelled one. -/
theorem conv2d_eq_tap {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w)
    (o : Fin oc) (hi : Fin h) (wi : Fin w) :
    conv2d W b x o hi wi
      = b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * convTap x c kh kw hi wi := rfl

/-- Taps are monotone in the input tensor (the pad branch is the constant `0`,
    which is monotone trivially) — the only fact the conv sign-split needs. -/
theorem convTap_mono {ic h w kH kW : Nat} {lo hi u : Tensor3 ic h w}
    (hu : InBox3 lo hi u) (c : Fin ic) (kh : Fin kH) (kw : Fin kW)
    (hI : Fin h) (wI : Fin w) :
    convTap lo c kh kw hI wI ≤ convTap u c kh kw hI wI ∧
      convTap u c kh kw hI wI ≤ convTap hi c kh kw hI wI := by
  unfold convTap
  by_cases hpad : ((kH - 1) / 2 ≤ kh.val + hI.val ∧ kh.val + hI.val - (kH - 1) / 2 < h ∧
      (kW - 1) / 2 ≤ kw.val + wI.val ∧ kw.val + wI.val - (kW - 1) / 2 < w)
  · rw [dif_pos hpad, dif_pos hpad, dif_pos hpad]
    exact hu c _ _
  · rw [dif_neg hpad, dif_neg hpad, dif_neg hpad]
    exact ⟨le_refl 0, le_refl 0⟩

/-- Sign-split LOWER image of `conv2d W b`. The split sits outside the pad test
    because `convTap` is `0` on both endpoints there. -/
noncomputable def convLo {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (lo hi : Tensor3 ic h w) : Tensor3 oc h w :=
  fun o hI wI =>
    b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
      if 0 ≤ W o c kh kw then W o c kh kw * convTap lo c kh kw hI wI
      else W o c kh kw * convTap hi c kh kw hI wI

/-- Sign-split UPPER image of `conv2d W b`. -/
noncomputable def convHi {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (lo hi : Tensor3 ic h w) : Tensor3 oc h w :=
  fun o hI wI =>
    b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
      if 0 ≤ W o c kh kw then W o c kh kw * convTap hi c kh kw hI wI
      else W o c kh kw * convTap lo c kh kw hI wI

/-- **Convolution propagates boxes soundly.** The rank-3 conv peer of
    `denseLo_le`/`le_denseHi`. -/
theorem conv2d_boxSound3 {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    BoxSound3 (conv2d W b : Tensor3 ic h w → Tensor3 oc h w) (convLo W b) (convHi W b) := by
  intro lo hi u hu o hI wI
  have hlo : (∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        (if 0 ≤ W o c kh kw then W o c kh kw * convTap lo c kh kw hI wI
         else W o c kh kw * convTap hi c kh kw hI wI))
      ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          W o c kh kw * convTap u c kh kw hI wI := by
    refine Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
      Finset.sum_le_sum fun kw _ => ?_
    by_cases hW : 0 ≤ W o c kh kw
    · rw [if_pos hW]
      exact mul_le_mul_of_nonneg_left (convTap_mono hu c kh kw hI wI).1 hW
    · rw [if_neg hW]
      exact mul_le_mul_of_nonpos_left (convTap_mono hu c kh kw hI wI).2 (le_of_not_ge hW)
  have hhi : (∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
        W o c kh kw * convTap u c kh kw hI wI)
      ≤ ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          (if 0 ≤ W o c kh kw then W o c kh kw * convTap hi c kh kw hI wI
           else W o c kh kw * convTap lo c kh kw hI wI) := by
    refine Finset.sum_le_sum fun c _ => Finset.sum_le_sum fun kh _ =>
      Finset.sum_le_sum fun kw _ => ?_
    by_cases hW : 0 ≤ W o c kh kw
    · rw [if_pos hW]
      exact mul_le_mul_of_nonneg_left (convTap_mono hu c kh kw hI wI).2 hW
    · rw [if_neg hW]
      exact mul_le_mul_of_nonpos_left (convTap_mono hu c kh kw hI wI).1 (le_of_not_ge hW)
  rw [conv2d_eq_tap]
  simp only [convLo, convHi]
  exact ⟨by linarith [hlo], by linarith [hhi]⟩

-- ════════════════════════════════════════════════════════════════
-- § Max pooling
-- ════════════════════════════════════════════════════════════════

/-- Max-pool is monotone, so its box endpoints are the pooled endpoints. Tied
    windows need no special handling: `max` is monotone regardless of which
    entry attains it, so the certificate never touches the argmax. -/
theorem maxPool2_boxSound3 {c h w : Nat} :
    BoxSound3 (maxPool2 : Tensor3 c (2*h) (2*w) → Tensor3 c h w)
      (fun lo _ => maxPool2 lo) (fun _ hi => maxPool2 hi) := by
  intro lo hi u hu ch hI wI
  simp only [maxPool2]
  constructor
  · exact max_le_max (max_le_max (hu _ _ _).1 (hu _ _ _).1)
      (max_le_max (hu _ _ _).1 (hu _ _ _).1)
  · exact max_le_max (max_le_max (hu _ _ _).2 (hu _ _ _).2)
      (max_le_max (hu _ _ _).2 (hu _ _ _).2)

-- ════════════════════════════════════════════════════════════════
-- § Flat ↔ tensor transfer (so conv/pool join the `Vec` composition chain)
-- ════════════════════════════════════════════════════════════════

theorem inBox3_unflatten {c h w : Nat} {lo hi u : Vec (c * h * w)} (hu : InBox lo hi u) :
    InBox3 (Tensor3.unflatten lo : Tensor3 c h w) (Tensor3.unflatten hi) (Tensor3.unflatten u) :=
  fun _ _ _ => hu _

theorem inBox_flatten {c h w : Nat} {lo hi u : Tensor3 c h w} (hu : InBox3 lo hi u) :
    InBox (Tensor3.flatten lo) (Tensor3.flatten hi) (Tensor3.flatten u) :=
  fun _ => hu _ _ _

/-- Lift a rank-3 sound transformer through `flatten ∘ · ∘ unflatten`. Both maps
    are index bijections, so soundness transfers verbatim. -/
theorem BoxSound3.flat {c h w c' h' w' : Nat} {f : Tensor3 c h w → Tensor3 c' h' w'}
    {Flo Fhi : Tensor3 c h w → Tensor3 c h w → Tensor3 c' h' w'} (hf : BoxSound3 f Flo Fhi) :
    BoxSound (fun v : Vec (c * h * w) => Tensor3.flatten (f (Tensor3.unflatten v)))
      (fun lo hi => Tensor3.flatten (Flo (Tensor3.unflatten lo) (Tensor3.unflatten hi)))
      (fun lo hi => Tensor3.flatten (Fhi (Tensor3.unflatten lo) (Tensor3.unflatten hi))) :=
  fun _ _ _ hu => inBox_flatten (hf _ _ _ (inBox3_unflatten hu))

/-- Flat conv box endpoints. -/
noncomputable def flatConvLo {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (lo hi : Vec (ic * h * w)) : Vec (oc * h * w) :=
  Tensor3.flatten (convLo W b (Tensor3.unflatten lo) (Tensor3.unflatten hi))

noncomputable def flatConvHi {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (lo hi : Vec (ic * h * w)) : Vec (oc * h * w) :=
  Tensor3.flatten (convHi W b (Tensor3.unflatten lo) (Tensor3.unflatten hi))

/-- **`flatConv` propagates boxes soundly** — conv joins the `Vec`-space
    composition chain, so it can sit anywhere in a `BoxSound.comp` stack. -/
theorem flatConv_boxSound {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    BoxSound (flatConv W b : Vec (ic * h * w) → Vec (oc * h * w))
      (flatConvLo W b) (flatConvHi W b) :=
  (conv2d_boxSound3 W b).flat

/-- Flat max-pool box endpoints. -/
noncomputable def maxPoolFlatLo (c h w : Nat) (lo _hi : Vec (c * (2*h) * (2*w))) :
    Vec (c * h * w) := maxPoolFlat c h w lo

noncomputable def maxPoolFlatHi (c h w : Nat) (_lo hi : Vec (c * (2*h) * (2*w))) :
    Vec (c * h * w) := maxPoolFlat c h w hi

theorem maxPoolFlat_boxSound (c h w : Nat) :
    BoxSound (maxPoolFlat c h w) (maxPoolFlatLo c h w) (maxPoolFlatHi c h w) :=
  (maxPool2_boxSound3 (c := c) (h := h) (w := w)).flat

-- ════════════════════════════════════════════════════════════════
-- § The capstone, at any depth
-- ════════════════════════════════════════════════════════════════

/-- `f` is *certified at `L∞` radius ε* on input `x` with class `y`: every
    coordinatewise-bounded perturbation keeps `y` the strict argmax. The
    `Vec`-space peer of `LipschitzCertDemo.CertifiedAtLinf`. -/
def CertifiedAtLinfV {n k : Nat} (f : Vec n → Vec k) (ε : ℝ) (x : Vec n) (y : Fin k) : Prop :=
  ∀ δ : Vec n, (∀ i, |δ i| ≤ ε) → ∀ j, j ≠ y → f (x + δ) j < f (x + δ) y

/-- **IBP `L∞` certificate at arbitrary depth and layer mix.** Given *any*
    sound interval transformer for the network — built by `BoxSound.comp` from
    conv / max-pool / dense / ReLU pieces — separation of the propagated output
    boxes on `x ∓ ε` certifies `x`: every perturbation with `|δ i| ≤ ε`
    coordinatewise leaves `y` the strict argmax.

    This is `ibp2_certified_at_eps` with the two-layer dense shape replaced by a
    hypothesis, which is what lets the same theorem serve a conv net. -/
theorem ibp_certified_of_boxSound {n k : Nat} {f : Vec n → Vec k}
    {Flo Fhi : Vec n → Vec n → Vec k} (hs : BoxSound f Flo Fhi)
    {x : Vec n} {ε : ℝ} {y : Fin k}
    (hsep : ∀ j, j ≠ y →
      Fhi (fun i => x i - ε) (fun i => x i + ε) j
        < Flo (fun i => x i - ε) (fun i => x i + ε) y) :
    CertifiedAtLinfV f ε x y := by
  intro δ hδ j hj
  have hbox : InBox (fun i => x i - ε) (fun i => x i + ε) (x + δ) := by
    intro i
    have h1 := abs_le.mp (hδ i)
    have hxi : (x + δ) i = x i + δ i := rfl
    exact ⟨by rw [hxi]; linarith [h1.1], by rw [hxi]; linarith [h1.2]⟩
  have hout := hs _ _ _ hbox
  exact lt_of_le_of_lt (hout j).2 (lt_of_lt_of_le (hsep j hj) (hout y).1)

-- ════════════════════════════════════════════════════════════════
-- § The tensor-space capstone (what a conv net actually instantiates)
-- ════════════════════════════════════════════════════════════════

/-! A conv classifier is `Tensor3 → Vec k`: a tensor body then a dense head that
reads the activation directly. Stating the certificate there keeps `flatten` out
of every per-image proof (it is an `Equiv` computation, and paying for it at each
of the head's inputs is the difference between a cheap instance and an
unaffordable one). The flat machinery above stays available for nets that need
it; nothing here depends on it. -/

/-- The dense head of a conv net: `Fin c → Fin h → Fin w → Fin k` weights read
    the rank-3 activation in place (this is "flatten then dense", fused). -/
noncomputable def denseT {c h w k : Nat} (W : Fin c → Fin h → Fin w → Fin k → ℝ)
    (b : Vec k) (x : Tensor3 c h w) : Vec k :=
  fun j => (∑ o, ∑ i, ∑ m, x o i m * W o i m j) + b j

noncomputable def denseTLo {c h w k : Nat} (W : Fin c → Fin h → Fin w → Fin k → ℝ)
    (b : Vec k) (lo hi : Tensor3 c h w) : Vec k :=
  fun j => (∑ o, ∑ i, ∑ m,
    if 0 ≤ W o i m j then lo o i m * W o i m j else hi o i m * W o i m j) + b j

noncomputable def denseTHi {c h w k : Nat} (W : Fin c → Fin h → Fin w → Fin k → ℝ)
    (b : Vec k) (lo hi : Tensor3 c h w) : Vec k :=
  fun j => (∑ o, ∑ i, ∑ m,
    if 0 ≤ W o i m j then hi o i m * W o i m j else lo o i m * W o i m j) + b j

/-- A sound interval transformer for a `Tensor3 → Vec` map (the head rung). -/
def BoxSound3V {c h w k : Nat} (f : Tensor3 c h w → Vec k)
    (Flo Fhi : Tensor3 c h w → Tensor3 c h w → Vec k) : Prop :=
  ∀ lo hi u, InBox3 lo hi u → InBox (Flo lo hi) (Fhi lo hi) (f u)

/-- Compose a tensor body into a head — the last link of a conv net's chain. -/
theorem BoxSound3V.comp3 {c h w c' h' w' k : Nat}
    {f : Tensor3 c h w → Tensor3 c' h' w'} {g : Tensor3 c' h' w' → Vec k}
    {Flo Fhi : Tensor3 c h w → Tensor3 c h w → Tensor3 c' h' w'}
    {Glo Ghi : Tensor3 c' h' w' → Tensor3 c' h' w' → Vec k}
    (hg : BoxSound3V g Glo Ghi) (hf : BoxSound3 f Flo Fhi) :
    BoxSound3V (g ∘ f) (fun lo hi => Glo (Flo lo hi) (Fhi lo hi))
                       (fun lo hi => Ghi (Flo lo hi) (Fhi lo hi)) :=
  fun lo hi u hu => hg _ _ _ (hf lo hi u hu)

theorem denseT_boxSound3V {c h w k : Nat} (W : Fin c → Fin h → Fin w → Fin k → ℝ) (b : Vec k) :
    BoxSound3V (denseT W b) (denseTLo W b) (denseTHi W b) := by
  intro lo hi u hu j
  have hlo : (∑ o, ∑ i, ∑ m,
        if 0 ≤ W o i m j then lo o i m * W o i m j else hi o i m * W o i m j)
      ≤ ∑ o, ∑ i, ∑ m, u o i m * W o i m j := by
    refine Finset.sum_le_sum fun o _ => Finset.sum_le_sum fun i _ =>
      Finset.sum_le_sum fun m _ => ?_
    by_cases hW : 0 ≤ W o i m j
    · rw [if_pos hW]; exact mul_le_mul_of_nonneg_right (hu o i m).1 hW
    · rw [if_neg hW]; exact mul_le_mul_of_nonpos_right (hu o i m).2 (le_of_not_ge hW)
  have hhi : (∑ o, ∑ i, ∑ m, u o i m * W o i m j)
      ≤ ∑ o, ∑ i, ∑ m,
          if 0 ≤ W o i m j then hi o i m * W o i m j else lo o i m * W o i m j := by
    refine Finset.sum_le_sum fun o _ => Finset.sum_le_sum fun i _ =>
      Finset.sum_le_sum fun m _ => ?_
    by_cases hW : 0 ≤ W o i m j
    · rw [if_pos hW]; exact mul_le_mul_of_nonneg_right (hu o i m).2 hW
    · rw [if_neg hW]; exact mul_le_mul_of_nonpos_right (hu o i m).1 (le_of_not_ge hW)
  simp only [denseT, denseTLo, denseTHi]
  exact ⟨by linarith [hlo], by linarith [hhi]⟩

/-- `f` is *certified at pixel `L∞` radius ε* on the image `x` with class `y`:
    every perturbation bounded by `ε` in EVERY pixel keeps `y` the strict argmax.
    The rank-3 (image-shaped) peer of `CertifiedAtLinfV`. -/
def CertifiedAtLinf3 {c h w k : Nat} (f : Tensor3 c h w → Vec k) (ε : ℝ)
    (x : Tensor3 c h w) (y : Fin k) : Prop :=
  ∀ δ : Tensor3 c h w, (∀ a b d, |δ a b d| ≤ ε) →
    ∀ j, j ≠ y → f (x + δ) j < f (x + δ) y

/-- Certification is monotone in the radius: a certificate at `ε` is a
    certificate at every `ε' ≤ ε`, directly from the statement. Instances
    therefore only need to prove (and carry the box data for) the LARGEST radius
    at which an image certifies — every smaller radius on the ε-grid is a
    one-line corollary, not another propagated box. -/
theorem CertifiedAtLinf3.mono {c h w k : Nat} {f : Tensor3 c h w → Vec k} {ε ε' : ℝ}
    {x : Tensor3 c h w} {y : Fin k} (hc : CertifiedAtLinf3 f ε x y) (hle : ε' ≤ ε) :
    CertifiedAtLinf3 f ε' x y :=
  fun δ hδ => hc δ fun a b d => (hδ a b d).trans hle

/-- The flat peer of `CertifiedAtLinf3.mono`. -/
theorem CertifiedAtLinfV.mono {n k : Nat} {f : Vec n → Vec k} {ε ε' : ℝ}
    {x : Vec n} {y : Fin k} (hc : CertifiedAtLinfV f ε x y) (hle : ε' ≤ ε) :
    CertifiedAtLinfV f ε' x y :=
  fun δ hδ => hc δ fun i => (hδ i).trans hle

/-- **IBP `L∞` certificate for a conv net.** Same statement as
    `ibp_certified_of_boxSound`, in the shape a convolutional classifier has:
    an arbitrary-depth tensor body (`conv`/`relu`/`pool`, composed by
    `BoxSound3.comp`) followed by a dense head. Separation of the propagated
    output boxes on the pixel box `x ∓ ε` certifies `x`. -/
theorem ibp3_certified_of_boxSound {c h w k : Nat} {f : Tensor3 c h w → Vec k}
    {Flo Fhi : Tensor3 c h w → Tensor3 c h w → Vec k} (hs : BoxSound3V f Flo Fhi)
    {x : Tensor3 c h w} {ε : ℝ} {y : Fin k}
    (hsep : ∀ j, j ≠ y →
      Fhi (fun a b d => x a b d - ε) (fun a b d => x a b d + ε) j
        < Flo (fun a b d => x a b d - ε) (fun a b d => x a b d + ε) y) :
    CertifiedAtLinf3 f ε x y := by
  intro δ hδ j hj
  have hbox : InBox3 (fun a b d => x a b d - ε) (fun a b d => x a b d + ε) (x + δ) := by
    intro a b d
    have h1 := abs_le.mp (hδ a b d)
    have hxi : (x + δ) a b d = x a b d + δ a b d := rfl
    exact ⟨by rw [hxi]; linarith [h1.1], by rw [hxi]; linarith [h1.2]⟩
  have hout := hs _ _ _ hbox
  exact lt_of_le_of_lt (hout j).2 (lt_of_lt_of_le (hsep j hj) (hout y).1)

-- ════════════════════════════════════════════════════════════════
-- § Depth, concretely: a six-layer conv stack, no new proof obligations
-- ════════════════════════════════════════════════════════════════

/-- A deeper stack than anything the dense `ibp2_certified_at_eps` could state:
    `conv → relu → conv → relu → max-pool → dense head`. Weights are arbitrary —
    this is a statement about the ENGINE, not about one trained net. -/
noncomputable def deepNet {ic c₁ c₂ h w k : Nat}
    (W₁ : Kernel4 c₁ ic 3 3) (b₁ : Vec c₁) (W₂ : Kernel4 c₂ c₁ 3 3) (b₂ : Vec c₂)
    (Wh : Fin c₂ → Fin h → Fin w → Fin k → ℝ) (bh : Vec k) :
    Tensor3 ic (2 * h) (2 * w) → Vec k :=
  denseT Wh bh ∘ maxPool2 ∘ reluT ∘ conv2d W₂ b₂ ∘ reluT ∘ conv2d W₁ b₁

/-- **Depth costs nothing.** The six-layer stack's sound interval transformer is
    five `.comp`s of the per-layer facts — no induction, no new lemma, and the
    proof term is one line. Feeding this to `ibp3_certified_of_boxSound` gives a
    pixel-`L∞` certificate for the deep net exactly as for the shallow one; the
    only thing that grows with depth is how loose the box gets. -/
theorem deepNet_boxSound {ic c₁ c₂ h w k : Nat}
    (W₁ : Kernel4 c₁ ic 3 3) (b₁ : Vec c₁) (W₂ : Kernel4 c₂ c₁ 3 3) (b₂ : Vec c₂)
    (Wh : Fin c₂ → Fin h → Fin w → Fin k → ℝ) (bh : Vec k) :
    BoxSound3V (deepNet W₁ b₁ W₂ b₂ Wh bh)
      (fun lo hi => denseTLo Wh bh
        (maxPool2 (reluT (convLo W₂ b₂ (reluT (convLo W₁ b₁ lo hi))
                                       (reluT (convHi W₁ b₁ lo hi)))))
        (maxPool2 (reluT (convHi W₂ b₂ (reluT (convLo W₁ b₁ lo hi))
                                       (reluT (convHi W₁ b₁ lo hi))))))
      (fun lo hi => denseTHi Wh bh
        (maxPool2 (reluT (convLo W₂ b₂ (reluT (convLo W₁ b₁ lo hi))
                                       (reluT (convHi W₁ b₁ lo hi)))))
        (maxPool2 (reluT (convHi W₂ b₂ (reluT (convLo W₁ b₁ lo hi))
                                       (reluT (convHi W₁ b₁ lo hi)))))) :=
  (denseT_boxSound3V Wh bh).comp3
    (maxPool2_boxSound3.comp (reluT_boxSound3.comp
      ((conv2d_boxSound3 W₂ b₂).comp (reluT_boxSound3.comp (conv2d_boxSound3 W₁ b₁)))))

-- ════════════════════════════════════════════════════════════════
-- § Uniform-box collapse for conv (the instance-cost lever)
-- ════════════════════════════════════════════════════════════════

/-- The entrywise absolute kernel. -/
noncomputable def absK {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW) : Kernel4 oc ic kH kW :=
  fun o c kh kw => |W o c kh kw|

/-- The all-ones input, used to read off each output position's *active* tap
    count (SAME padding drops taps at the border, so the `ℓ1` weight is
    position-dependent — `conv2d (absK W) 0 ones` computes exactly that). -/
noncomputable def onesT (ic h w : Nat) : Tensor3 ic h w := fun _ _ _ => 1

/-- A tap of the uniform box's LOWER face splits into `tap x − ε·tap 𝟙`. -/
theorem convTap_uniform_lo {ic h w kH kW : Nat} (x : Tensor3 ic h w) (ε : ℝ)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hI : Fin h) (wI : Fin w) :
    convTap (fun a b d => x a b d - ε) c kh kw hI wI
      = convTap x c kh kw hI wI - ε * convTap (onesT ic h w) c kh kw hI wI := by
  unfold convTap onesT
  by_cases hpad : ((kH - 1) / 2 ≤ kh.val + hI.val ∧ kh.val + hI.val - (kH - 1) / 2 < h ∧
      (kW - 1) / 2 ≤ kw.val + wI.val ∧ kw.val + wI.val - (kW - 1) / 2 < w)
  · rw [dif_pos hpad, dif_pos hpad, dif_pos hpad]; ring
  · rw [dif_neg hpad, dif_neg hpad, dif_neg hpad]; ring

/-- A tap of the uniform box's UPPER face splits into `tap x + ε·tap 𝟙`. -/
theorem convTap_uniform_hi {ic h w kH kW : Nat} (x : Tensor3 ic h w) (ε : ℝ)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hI : Fin h) (wI : Fin w) :
    convTap (fun a b d => x a b d + ε) c kh kw hI wI
      = convTap x c kh kw hI wI + ε * convTap (onesT ic h w) c kh kw hI wI := by
  unfold convTap onesT
  by_cases hpad : ((kH - 1) / 2 ≤ kh.val + hI.val ∧ kh.val + hI.val - (kH - 1) / 2 < h ∧
      (kW - 1) / 2 ≤ kw.val + wI.val ∧ kw.val + wI.val - (kW - 1) / 2 < w)
  · rw [dif_pos hpad, dif_pos hpad, dif_pos hpad]; ring
  · rw [dif_neg hpad, dif_neg hpad, dif_neg hpad]; ring

/-- **Uniform-box collapse, LOWER.** On the first layer's box `x ∓ ε` the conv
    sign split evaluates to `conv2d W b x − ε · conv2d |W| 0 𝟙` — one ordinary
    forward pass plus one absolute-kernel pass, instead of two full sign-split
    sums. The conv peer of `denseLo_uniform`. -/
theorem convLo_uniform {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w) (ε : ℝ)
    (o : Fin oc) (hI : Fin h) (wI : Fin w) :
    convLo W b (fun a b' d => x a b' d - ε) (fun a b' d => x a b' d + ε) o hI wI
      = conv2d W b x o hI wI - ε * conv2d (absK W) (fun _ => 0) (onesT ic h w) o hI wI := by
  rw [conv2d_eq_tap, conv2d_eq_tap]
  unfold convLo
  have hmul : ∀ (c : Fin ic) (kh : Fin kH) (kw : Fin kW),
      (if 0 ≤ W o c kh kw
        then W o c kh kw * convTap (fun a b' d => x a b' d - ε) c kh kw hI wI
        else W o c kh kw * convTap (fun a b' d => x a b' d + ε) c kh kw hI wI)
      = W o c kh kw * convTap x c kh kw hI wI
        - ε * (absK W o c kh kw * convTap (onesT ic h w) c kh kw hI wI) := by
    intro c kh kw
    by_cases hW : 0 ≤ W o c kh kw
    · rw [if_pos hW, convTap_uniform_lo, absK, abs_of_nonneg hW]; ring
    · rw [if_neg hW, convTap_uniform_hi, absK, abs_of_neg (lt_of_not_ge hW)]; ring
  calc b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          (if 0 ≤ W o c kh kw
            then W o c kh kw * convTap (fun a b' d => x a b' d - ε) c kh kw hI wI
            else W o c kh kw * convTap (fun a b' d => x a b' d + ε) c kh kw hI wI)
      = b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          (W o c kh kw * convTap x c kh kw hI wI
            - ε * (absK W o c kh kw * convTap (onesT ic h w) c kh kw hI wI)) := by
          refine congrArg _ (Finset.sum_congr rfl fun c _ =>
            Finset.sum_congr rfl fun kh _ => Finset.sum_congr rfl fun kw _ => hmul c kh kw)
    _ = _ := by
          simp only [Finset.sum_sub_distrib, ← Finset.mul_sum]
          ring

/-- **Uniform-box collapse, UPPER** (sibling of `convLo_uniform`). -/
theorem convHi_uniform {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w) (ε : ℝ)
    (o : Fin oc) (hI : Fin h) (wI : Fin w) :
    convHi W b (fun a b' d => x a b' d - ε) (fun a b' d => x a b' d + ε) o hI wI
      = conv2d W b x o hI wI + ε * conv2d (absK W) (fun _ => 0) (onesT ic h w) o hI wI := by
  rw [conv2d_eq_tap, conv2d_eq_tap]
  unfold convHi
  have hmul : ∀ (c : Fin ic) (kh : Fin kH) (kw : Fin kW),
      (if 0 ≤ W o c kh kw
        then W o c kh kw * convTap (fun a b' d => x a b' d + ε) c kh kw hI wI
        else W o c kh kw * convTap (fun a b' d => x a b' d - ε) c kh kw hI wI)
      = W o c kh kw * convTap x c kh kw hI wI
        + ε * (absK W o c kh kw * convTap (onesT ic h w) c kh kw hI wI) := by
    intro c kh kw
    by_cases hW : 0 ≤ W o c kh kw
    · rw [if_pos hW, convTap_uniform_hi, absK, abs_of_nonneg hW]; ring
    · rw [if_neg hW, convTap_uniform_lo, absK, abs_of_neg (lt_of_not_ge hW)]; ring
  calc b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          (if 0 ≤ W o c kh kw
            then W o c kh kw * convTap (fun a b' d => x a b' d + ε) c kh kw hI wI
            else W o c kh kw * convTap (fun a b' d => x a b' d - ε) c kh kw hI wI)
      = b o + ∑ c : Fin ic, ∑ kh : Fin kH, ∑ kw : Fin kW,
          (W o c kh kw * convTap x c kh kw hI wI
            + ε * (absK W o c kh kw * convTap (onesT ic h w) c kh kw hI wI)) := by
          refine congrArg _ (Finset.sum_congr rfl fun c _ =>
            Finset.sum_congr rfl fun kh _ => Finset.sum_congr rfl fun kw _ => hmul c kh kw)
    _ = _ := by
          simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
          ring

end IBP
end Proofs
