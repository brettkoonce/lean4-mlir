import LeanMlir.Proofs.Foundation.IntervalBoundConv

/-! # The conv IBP box in exact rationals — one kernel check per image

`IntervalBoundConv.lean` proves the box engine sound over `ℝ`. An instance still
has to evaluate the box on a concrete image, and over `ℝ` that is hundreds of
`simp`/`norm_num` goals per image — the generated conv scorecard spent ~250 s an
image doing it. Here the same layers are re-stated over `ℚ` as computable
functions (`convLoQ`, `convHiQ`, `reluTQ`, `maxPool2Q`, `denseTLoQ`,
`denseTHiQ`: the `ℝ` definitions verbatim, sums as `List.ofFn` sums), each with a
cast lemma back to its `ℝ` original. `convNetCheckQ` runs the box through
`conv → relu → max-pool → dense head` and compares the class scores;
`convNetCheckQ_sound` turns `convNetCheckQ … = true` into `CertifiedAtLinf3` for
the net built from the cast weights. An instance then proves each image with
one `decide +kernel` on the checker.

The weights and the image enter as `ℚ` data and the `ℝ` net is DEFINED as their
cast (`castK`, `castV`, `castW`, `castT`), so no per-tensor bridge is needed.
Closes under `propext / Classical.choice / Quot.sound`. -/

namespace Proofs
namespace IBP

open scoped BigOperators

-- ════════════════════════════════════════════════════════════════
-- § Casts from `ℚ` data
-- ════════════════════════════════════════════════════════════════

/-- A `ℚ` vector read as a `Vec`. -/
def castV {n : Nat} (v : Fin n → ℚ) : Vec n := fun i => (v i : ℝ)

/-- A `ℚ` rank-3 tensor read as a `Tensor3`. -/
def castT {c h w : Nat} (x : Fin c → Fin h → Fin w → ℚ) : Tensor3 c h w :=
  fun a b d => (x a b d : ℝ)

/-- A `ℚ` conv kernel read as a `Kernel4`. -/
def castK {oc ic kH kW : Nat} (W : Fin oc → Fin ic → Fin kH → Fin kW → ℚ) :
    Kernel4 oc ic kH kW :=
  fun o c i j => (W o c i j : ℝ)

/-- A `ℚ` in-place dense head read over `ℝ`. -/
def castW {c h w k : Nat} (W : Fin c → Fin h → Fin w → Fin k → ℚ) :
    Fin c → Fin h → Fin w → Fin k → ℝ :=
  fun o i m j => (W o i m j : ℝ)

-- ════════════════════════════════════════════════════════════════
-- § The layers over `ℚ`
-- ════════════════════════════════════════════════════════════════

/-- A `Fin n`-indexed sum the kernel can evaluate. -/
def sumQ {n : Nat} (f : Fin n → ℚ) : ℚ := (List.ofFn f).sum

theorem sumQ_cast {n : Nat} (f : Fin n → ℚ) : ((sumQ f : ℚ) : ℝ) = ∑ i, (f i : ℝ) := by
  rw [sumQ, List.sum_ofFn, Rat.cast_sum]

/-- `convTap` over `ℚ`. -/
def convTapQ {ic h w kH kW : Nat} (x : Fin ic → Fin h → Fin w → ℚ)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) : ℚ :=
  let pH := (kH - 1) / 2
  let pW := (kW - 1) / 2
  let hh := kh.val + hi.val
  let ww := kw.val + wi.val
  if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
    x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
  else 0

theorem convTapQ_cast {ic h w kH kW : Nat} (x : Fin ic → Fin h → Fin w → ℚ)
    (c : Fin ic) (kh : Fin kH) (kw : Fin kW) (hi : Fin h) (wi : Fin w) :
    ((convTapQ x c kh kw hi wi : ℚ) : ℝ) = convTap (castT x) c kh kw hi wi := by
  unfold convTapQ convTap convPad
  dsimp only
  split_ifs <;> simp [castT]

/-- `convLo` over `ℚ`. -/
def convLoQ {ic oc h w kH kW : Nat} (W : Fin oc → Fin ic → Fin kH → Fin kW → ℚ)
    (b : Fin oc → ℚ) (lo hi : Fin ic → Fin h → Fin w → ℚ) : Fin oc → Fin h → Fin w → ℚ :=
  fun o hI wI => b o + sumQ fun c => sumQ fun kh => sumQ fun kw =>
    if 0 ≤ W o c kh kw then W o c kh kw * convTapQ lo c kh kw hI wI
    else W o c kh kw * convTapQ hi c kh kw hI wI

/-- `convHi` over `ℚ`. -/
def convHiQ {ic oc h w kH kW : Nat} (W : Fin oc → Fin ic → Fin kH → Fin kW → ℚ)
    (b : Fin oc → ℚ) (lo hi : Fin ic → Fin h → Fin w → ℚ) : Fin oc → Fin h → Fin w → ℚ :=
  fun o hI wI => b o + sumQ fun c => sumQ fun kh => sumQ fun kw =>
    if 0 ≤ W o c kh kw then W o c kh kw * convTapQ hi c kh kw hI wI
    else W o c kh kw * convTapQ lo c kh kw hI wI

theorem convLoQ_cast {ic oc h w kH kW : Nat} (W : Fin oc → Fin ic → Fin kH → Fin kW → ℚ)
    (b : Fin oc → ℚ) (lo hi : Fin ic → Fin h → Fin w → ℚ) :
    castT (convLoQ W b lo hi) = convLo (castK W) (castV b) (castT lo) (castT hi) := by
  funext o hI wI
  simp only [castT, convLoQ, convLo, castK, castV, Rat.cast_add, sumQ_cast]
  congr 1; refine Finset.sum_congr rfl fun c _ => ?_
  refine Finset.sum_congr rfl fun kh _ => ?_
  refine Finset.sum_congr rfl fun kw _ => ?_
  by_cases h1 : 0 ≤ W o c kh kw <;> simp [h1, convTapQ_cast]

theorem convHiQ_cast {ic oc h w kH kW : Nat} (W : Fin oc → Fin ic → Fin kH → Fin kW → ℚ)
    (b : Fin oc → ℚ) (lo hi : Fin ic → Fin h → Fin w → ℚ) :
    castT (convHiQ W b lo hi) = convHi (castK W) (castV b) (castT lo) (castT hi) := by
  funext o hI wI
  simp only [castT, convHiQ, convHi, castK, castV, Rat.cast_add, sumQ_cast]
  congr 1; refine Finset.sum_congr rfl fun c _ => ?_
  refine Finset.sum_congr rfl fun kh _ => ?_
  refine Finset.sum_congr rfl fun kw _ => ?_
  by_cases h1 : 0 ≤ W o c kh kw <;> simp [h1, convTapQ_cast]

/-- `reluT` over `ℚ`. -/
def reluTQ {c h w : Nat} (x : Fin c → Fin h → Fin w → ℚ) : Fin c → Fin h → Fin w → ℚ :=
  fun o i j => max (x o i j) 0

theorem reluTQ_cast {c h w : Nat} (x : Fin c → Fin h → Fin w → ℚ) :
    castT (reluTQ x) = reluT (castT x) := by
  funext o i j
  simp [castT, reluTQ, reluT, Rat.cast_max]

/-- `maxPool2` over `ℚ`. -/
def maxPool2Q {c h w : Nat} (x : Fin c → Fin (2*h) → Fin (2*w) → ℚ) : Fin c → Fin h → Fin w → ℚ :=
  fun ch hi wi =>
    let i0 : Fin (2*h) := ⟨2*hi.val,     by have := hi.isLt; omega⟩
    let i1 : Fin (2*h) := ⟨2*hi.val + 1, by have := hi.isLt; omega⟩
    let j0 : Fin (2*w) := ⟨2*wi.val,     by have := wi.isLt; omega⟩
    let j1 : Fin (2*w) := ⟨2*wi.val + 1, by have := wi.isLt; omega⟩
    max (max (x ch i0 j0) (x ch i1 j0)) (max (x ch i0 j1) (x ch i1 j1))

theorem maxPool2Q_cast {c h w : Nat} (x : Fin c → Fin (2*h) → Fin (2*w) → ℚ) :
    castT (maxPool2Q x) = maxPool2 (castT x) := by
  funext o i j
  simp [castT, maxPool2Q, maxPool2, Rat.cast_max]

/-- `denseTLo` over `ℚ`. -/
def denseTLoQ {c h w k : Nat} (W : Fin c → Fin h → Fin w → Fin k → ℚ) (b : Fin k → ℚ)
    (lo hi : Fin c → Fin h → Fin w → ℚ) : Fin k → ℚ :=
  fun j => sumQ (fun o => sumQ fun i => sumQ fun m =>
    if 0 ≤ W o i m j then lo o i m * W o i m j else hi o i m * W o i m j) + b j

/-- `denseTHi` over `ℚ`. -/
def denseTHiQ {c h w k : Nat} (W : Fin c → Fin h → Fin w → Fin k → ℚ) (b : Fin k → ℚ)
    (lo hi : Fin c → Fin h → Fin w → ℚ) : Fin k → ℚ :=
  fun j => sumQ (fun o => sumQ fun i => sumQ fun m =>
    if 0 ≤ W o i m j then hi o i m * W o i m j else lo o i m * W o i m j) + b j

theorem denseTLoQ_cast {c h w k : Nat} (W : Fin c → Fin h → Fin w → Fin k → ℚ)
    (b : Fin k → ℚ) (lo hi : Fin c → Fin h → Fin w → ℚ) (j : Fin k) :
    ((denseTLoQ W b lo hi j : ℚ) : ℝ) = denseTLo (castW W) (castV b) (castT lo) (castT hi) j := by
  simp only [denseTLoQ, denseTLo, castW, castV, castT, Rat.cast_add, sumQ_cast]
  congr 1; refine Finset.sum_congr rfl fun o _ => ?_
  refine Finset.sum_congr rfl fun i _ => ?_
  refine Finset.sum_congr rfl fun m _ => ?_
  by_cases h1 : 0 ≤ W o i m j <;> simp [h1]

theorem denseTHiQ_cast {c h w k : Nat} (W : Fin c → Fin h → Fin w → Fin k → ℚ)
    (b : Fin k → ℚ) (lo hi : Fin c → Fin h → Fin w → ℚ) (j : Fin k) :
    ((denseTHiQ W b lo hi j : ℚ) : ℝ) = denseTHi (castW W) (castV b) (castT lo) (castT hi) j := by
  simp only [denseTHiQ, denseTHi, castW, castV, castT, Rat.cast_add, sumQ_cast]
  congr 1; refine Finset.sum_congr rfl fun o _ => ?_
  refine Finset.sum_congr rfl fun i _ => ?_
  refine Finset.sum_congr rfl fun m _ => ?_
  by_cases h1 : 0 ≤ W o i m j <;> simp [h1]

-- ════════════════════════════════════════════════════════════════
-- § The checker and its soundness
-- ════════════════════════════════════════════════════════════════

/-- The pooled lower box of `conv → relu → max-pool` on the pixel box `x ∓ ε`. -/
def convPoolLoQ {ic oc h w kH kW : Nat} (W : Fin oc → Fin ic → Fin kH → Fin kW → ℚ)
    (b : Fin oc → ℚ) (x : Fin ic → Fin (2*h) → Fin (2*w) → ℚ) (ε : ℚ) :
    Fin oc → Fin h → Fin w → ℚ :=
  maxPool2Q (reluTQ (convLoQ W b (fun a c d => x a c d - ε) (fun a c d => x a c d + ε)))

/-- The pooled upper box of `conv → relu → max-pool` on the pixel box `x ∓ ε`. -/
def convPoolHiQ {ic oc h w kH kW : Nat} (W : Fin oc → Fin ic → Fin kH → Fin kW → ℚ)
    (b : Fin oc → ℚ) (x : Fin ic → Fin (2*h) → Fin (2*w) → ℚ) (ε : ℚ) :
    Fin oc → Fin h → Fin w → ℚ :=
  maxPool2Q (reluTQ (convHiQ W b (fun a c d => x a c d - ε) (fun a c d => x a c d + ε)))

/-- **The per-image check.** Propagate the pixel box `x ∓ ε` through
    `conv → relu → max-pool → dense head` in exact rationals, and ask that every
    other class's upper score sit strictly below class `y`'s lower score. -/
def convNetCheckQ {ic oc h w kH kW k : Nat} (W : Fin oc → Fin ic → Fin kH → Fin kW → ℚ)
    (b : Fin oc → ℚ) (Wd : Fin oc → Fin h → Fin w → Fin k → ℚ) (bd : Fin k → ℚ)
    (x : Fin ic → Fin (2*h) → Fin (2*w) → ℚ) (ε : ℚ) (y : Fin k) : Bool :=
  (List.finRange k).all fun j =>
    j == y || decide (denseTHiQ Wd bd (convPoolLoQ W b x ε) (convPoolHiQ W b x ε) j
                        < denseTLoQ Wd bd (convPoolLoQ W b x ε) (convPoolHiQ W b x ε) y)

/-- **A passing check is a certificate.** For the net built from the cast
    weights, `convNetCheckQ … = true` gives `CertifiedAtLinf3` at radius `ε` on
    the cast image: `ibp3_certified_of_boxSound` over the composed box transformer,
    with the separation read off the check through the cast lemmas. -/
theorem convNetCheckQ_sound {ic oc h w kH kW k : Nat}
    (W : Fin oc → Fin ic → Fin kH → Fin kW → ℚ) (b : Fin oc → ℚ)
    (Wd : Fin oc → Fin h → Fin w → Fin k → ℚ) (bd : Fin k → ℚ)
    (x : Fin ic → Fin (2*h) → Fin (2*w) → ℚ) (ε : ℚ) (y : Fin k)
    (hc : convNetCheckQ W b Wd bd x ε y = true) :
    CertifiedAtLinf3
      (denseT (castW Wd) (castV bd) ∘ maxPool2 (c := oc) (h := h) (w := w) ∘ reluT
        ∘ conv2d (castK W) (castV b))
      (ε : ℝ) (castT x) y := by
  refine ibp3_certified_of_boxSound
    ((denseT_boxSound3V (castW Wd) (castV bd)).comp3
      ((maxPool2_boxSound3 (c := oc) (h := h) (w := w)).comp
        (reluT_boxSound3.comp (conv2d_boxSound3 (castK W) (castV b))))) ?_
  intro j hj
  have hbox : ∀ s : ℚ, (fun a c d => castT x a c d + (s : ℝ)) = castT (fun a c d => x a c d + s) :=
    fun s => by funext a c d; simp [castT]
  have hboxm : ∀ s : ℚ, (fun a c d => castT x a c d - (s : ℝ)) = castT (fun a c d => x a c d - s) :=
    fun s => by funext a c d; simp [castT]
  simp only [convNetCheckQ, List.all_eq_true, List.mem_finRange, true_implies, Bool.or_eq_true,
    beq_iff_eq, decide_eq_true_eq] at hc
  have hq := (hc j).resolve_left hj
  have hr : ((denseTHiQ Wd bd (convPoolLoQ W b x ε) (convPoolHiQ W b x ε) j : ℚ) : ℝ)
      < ((denseTLoQ Wd bd (convPoolLoQ W b x ε) (convPoolHiQ W b x ε) y : ℚ) : ℝ) :=
    Rat.cast_lt.mpr hq
  rw [denseTHiQ_cast, denseTLoQ_cast] at hr
  simp only [convPoolLoQ, convPoolHiQ, maxPool2Q_cast, reluTQ_cast, convLoQ_cast,
    convHiQ_cast] at hr
  simpa only [hbox, hboxm] using hr

end IBP
end Proofs
