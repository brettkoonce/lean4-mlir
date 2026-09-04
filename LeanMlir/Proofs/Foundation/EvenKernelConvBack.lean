import LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie

/-! # ⛔⛔ `convFlatBack` is NOT the adjoint at an EVEN kernel — and the one-line repair

`convFlatBack W = flatConv (reverseSwap W) 0` is the reversed-kernel forward conv every backward
in this repo runs, and `convFlatBack_eq_vjp_backward` (`Resnet34BackCertifiedTie.lean`) ties it to
the certified conv input-VJP **for ODD kernels only**. That hypothesis is not a convenience: it is
load-bearing, and the statement is FALSE without it.

`conv2d` pads by `pH = (kH-1)/2`, so `convFlatBack`'s coefficient of `dy[j]` at output `hi` is
`W[hi - j + (kH-1-pH)]`, while the adjoint's is `W[hi - j + pH]`. They agree iff
`kH - 1 - pH = pH`, i.e. iff `kH` is odd. At `kH = 4` the reversed-kernel conv is the adjoint of a
conv shifted one pixel.

⛔ **ConvNeXt-T is the only net in the repo with an even kernel**, and it has four: the 4×4/s4
patchify stem and the three 2×2/s2 downsamples. (ViT's 16×16 patch embed is NOT affected —
`patchEmbed_flat` is its own definition over non-overlapping patches, with no `conv2d` and no
padding convention.) Every other net is all-odd: R34 7×7/3×3/1×1, MobileNetV2 and
EfficientNet-B0 1×1/3×3/5×5.

⚠⚠ **The codegen tier already knew.** `StableHLO.lean`'s `.convStridedBack` emitter pads
ASYMMETRICALLY, `[[kH-1-pH, pH]]`, in both the per-example (:6120) and the batched (:8248) arms,
and the batched one says so in as many words — *"The symmetric `[[p,p],[p,p]]` this emitted AGREES
at every odd kernel and is WRONG at even ones (kH=2 ⇒ `[[0,0]]` where the VJP needs `[[1,0]]`) …
Found by the whole-net backward tie"*. Its `den` is `(flatConvStride2_has_vjp W b).backward`, the
certified VJP, so the EMITTED ConvNeXt backward is correct and nothing trained is affected. What
was never carried across is the third spelling of the same map — the float bridge's
`flatConvStride2Back` / `flatConvStride4Back`, which are `convFlatBack ∘ scatter` at the SYMMETRIC
pad. ⭐ That is `imagenet_specs_drift_from_twins` in its "a fix landed on one tier and its twin
kept the old spelling" form, for the third time
(`planning/float_budget_numbers.md` §3.10's pool and §3.16's head LayerNorm were the first two) —
and here the fix landed on TWO tiers and missed the third.

## The repair: spell the even kernel at an odd size

An even-kernel conv IS an odd-kernel conv on the kernel zero-extended at offset `(+1,+1)`:
`conv2d (padOdd W) b = conv2d W b` (`conv2d_padOdd_eq`). The padding bookkeeping is exactly the
emitter's: for even `kH`, a SYMMETRIC pad `[[pH', pH']]` at `kH+1` (where `pH' = kH/2 = pH+1`) on a
kernel whose leading tap is zero is the same program as the ASYMMETRIC `[[kH-1-pH, pH]]` at `kH`.
So this is not a third convention — it is the emitter's convention, expressed in the vocabulary the
float tier already has.

⭐ **The consequence is that no new conv machinery is needed anywhere.** The odd-kernel leaf tie
does all the work at `kH+1`; `|padOdd W| ≤ w'` is free (the new entries are `0`), so every
`floatBridges_convBack` / `Maps.convBack` transfers unchanged. ⚠ What DOES move is the fan-in a
budget numeral charges: `2×2 → 3×3` is 4 taps → 9, `4×4 → 5×5` is 16 → 25, at four sites. The extra
taps are identically zero, so the bound stays an upper bound for the emitted 4- and 16-tap program;
it is simply not tight there.
-/

namespace Proofs

open Classical

-- ════════════════════════════════════════════════════════════════
-- § The zero-extension, and the forward it does not change
-- ════════════════════════════════════════════════════════════════

/-- **Zero-extend a kernel to the next size, at offset `(+1, +1)`.** The `kh = 0` row and the
    `kw = 0` column are `0`; `padOdd W (kh+1) (kw+1) = W kh kw`.

    ⭐ Written with `Fin.cons` rather than a `dite` on `0 < kh.val` on purpose: `Fin.cons_zero`
    and `Fin.cons_succ` are `simp` lemmas that match `Fin.sum_univ_succ` head-on, which is the
    whole proof of `conv2d_padOdd_eq`. -/
noncomputable def padOdd {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW) :
    Kernel4 oc ic (kH + 1) (kW + 1) :=
  fun o c =>
    Fin.cons (α := fun _ => Fin (kW + 1) → ℝ) (fun _ => 0)
      (fun kh => Fin.cons (α := fun _ => ℝ) 0 (fun kw => W o c kh kw))

@[simp] theorem padOdd_zero_row {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (o : Fin oc) (c : Fin ic) (kw : Fin (kW + 1)) : padOdd W o c 0 kw = 0 := rfl

@[simp] theorem padOdd_zero_col {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) : padOdd W o c kh.succ 0 = 0 := rfl

@[simp] theorem padOdd_succ {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW)
    (o : Fin oc) (c : Fin ic) (kh : Fin kH) (kw : Fin kW) :
    padOdd W o c kh.succ kw.succ = W o c kh kw := rfl

/-- `padOdd` never exceeds the original's magnitude bound — the new entries are `0`. This is why
    the fix costs no float work: every `|W| ≤ w'` hypothesis transfers verbatim. -/
theorem padOdd_abs_le {oc ic kH kW : Nat} (W : Kernel4 oc ic kH kW) {w' : ℝ} (hw' : 0 ≤ w')
    (hW : ∀ o c kh kw, |W o c kh kw| ≤ w') :
    ∀ o c kh kw, |padOdd W o c kh kw| ≤ w' := by
  intro o c kh kw
  refine Fin.cases ?_ (fun kh' => ?_) kh
  · simpa using hw'
  · refine Fin.cases ?_ (fun kw' => ?_) kw
    · simpa using hw'
    · rw [padOdd_succ]; exact hW _ _ _ _

/-- **An even-kernel conv IS an odd-kernel conv on the zero-extended kernel.**
    `conv2d (padOdd W) b = conv2d W b` whenever `kH`, `kW` are EVEN.

    The pad bookkeeping is the whole content: at `kH + 1` the pad is `pH' = kH/2`, and evenness
    gives `pH' = (kH-1)/2 + 1 = pH + 1`, so the shifted tap `kh+1` reads
    `x[(kh+1) + hi - (pH+1)] = x[kh + hi - pH]` — the original's window, with the same guard.

    ⚠ Evenness is stated as `2 * (kH / 2) = kH` rather than `kH % 2 = 0` because that is the form
    `omega` consumes directly in the index arithmetic below. -/
theorem conv2d_padOdd_eq {ic oc h w kH kW : Nat}
    (hH : 2 * (kH / 2) = kH) (hW : 2 * (kW / 2) = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Tensor3 ic h w) :
    conv2d (padOdd W) b x = conv2d W b x := by
  funext o hi wi
  simp only [conv2d]
  congr 1
  refine Finset.sum_congr rfl (fun c _ => ?_)
  rw [Fin.sum_univ_succ]
  -- the `kh = 0` row of `padOdd` is zero, so its whole inner sum vanishes
  have hrow : (∑ kw : Fin (kW + 1), padOdd W o c 0 kw *
      (let pH := (kH + 1 - 1) / 2
       let pW := (kW + 1 - 1) / 2
       let hh := (0 : Fin (kH + 1)).val + hi.val
       let ww := kw.val + wi.val
       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
         x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
       else 0)) = 0 := by
    refine Finset.sum_eq_zero (fun kw _ => ?_)
    rw [padOdd_zero_row, zero_mul]
  rw [hrow, zero_add]
  refine Finset.sum_congr rfl (fun kh _ => ?_)
  rw [Fin.sum_univ_succ]
  have hcol : padOdd W o c kh.succ 0 *
      (let pH := (kH + 1 - 1) / 2
       let pW := (kW + 1 - 1) / 2
       let hh := (kh.succ).val + hi.val
       let ww := (0 : Fin (kW + 1)).val + wi.val
       if hpad : pH ≤ hh ∧ hh - pH < h ∧ pW ≤ ww ∧ ww - pW < w then
         x c ⟨hh - pH, hpad.2.1⟩ ⟨ww - pW, hpad.2.2.2⟩
       else 0) = 0 := by
    rw [padOdd_zero_col, zero_mul]
  rw [hcol, zero_add]
  refine Finset.sum_congr rfl (fun kw _ => ?_)
  rw [padOdd_succ]
  congr 1
  -- the two pad-evaluations agree: same guard, same index
  have hkh : (kh.succ : Fin (kH + 1)).val = kh.val + 1 := rfl
  have hkw : (kw.succ : Fin (kW + 1)).val = kw.val + 1 := rfl
  have hpH : (kH + 1 - 1) / 2 = (kH - 1) / 2 + 1 := by omega
  have hpW : (kW + 1 - 1) / 2 = (kW - 1) / 2 + 1 := by omega
  simp only [hkh, hkw, hpH, hpW]
  by_cases hg : (kH - 1) / 2 ≤ kh.val + hi.val ∧ kh.val + hi.val - (kH - 1) / 2 < h ∧
      (kW - 1) / 2 ≤ kw.val + wi.val ∧ kw.val + wi.val - (kW - 1) / 2 < w
  · rw [dif_pos (by omega : (kH - 1) / 2 + 1 ≤ kh.val + 1 + hi.val ∧
          kh.val + 1 + hi.val - ((kH - 1) / 2 + 1) < h ∧
          (kW - 1) / 2 + 1 ≤ kw.val + 1 + wi.val ∧
          kw.val + 1 + wi.val - ((kW - 1) / 2 + 1) < w),
       dif_pos hg]
    -- ⚠ NOT `congr 1 <;> Fin.ext (by omega)`: after `congr` the goal is
    -- `(⟨_, _⟩ : Fin h).val = (⟨_, _⟩).val` and `omega` treats `Fin.val ⟨·,·⟩` as opaque.
    -- Rewriting the Nat index expressions makes both sides syntactically identical instead.
    have e1 : kh.val + 1 + hi.val - ((kH - 1) / 2 + 1) = kh.val + hi.val - (kH - 1) / 2 := by
      omega
    have e2 : kw.val + 1 + wi.val - ((kW - 1) / 2 + 1) = kw.val + wi.val - (kW - 1) / 2 := by
      omega
    simp only [e1, e2]
  · rw [dif_neg (by omega), dif_neg hg]

/-- `flatConv` inherits it — the flat form is `flatten ∘ conv2d ∘ unflatten`. -/
theorem flatConv_padOdd_eq {ic oc h w kH kW : Nat}
    (hH : 2 * (kH / 2) = kH) (hW : 2 * (kW / 2) = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) :
    (flatConv (h := h) (w := w) (padOdd W) b : Vec (ic * h * w) → Vec (oc * h * w))
      = flatConv (h := h) (w := w) W b := by
  funext v
  simp only [flatConv, conv2d_padOdd_eq hH hW W b]

-- ════════════════════════════════════════════════════════════════
-- § Uniqueness without transport
-- ════════════════════════════════════════════════════════════════

/-- **Two VJP witnesses for EQUAL maps have the same backward.** The generalisation of
    `HasVJP.backward_unique` that a respelling needs: the witnesses have different TYPES
    (`HasVJP f` and `HasVJP g`), so the existing form does not apply and `hfg ▸ ·` would give an
    `Eq.mpr`-blocked `backward` — `FloatBridgesTo.ofEq`'s trap, one tier down
    (`planning/float_budget_numbers.md` §3.5.2 item 5). Going through `.correct` avoids transport
    entirely. -/
theorem HasVJP.backward_unique_of_eq {m n : Nat} {f g : Vec m → Vec n} (hfg : f = g)
    (h₁ : HasVJP f) (h₂ : HasVJP g) (x : Vec m) (dy : Vec n) :
    h₁.backward x dy = h₂.backward x dy := by
  funext i
  rw [h₁.correct, h₂.correct, hfg]

-- ════════════════════════════════════════════════════════════════
-- § The four even-kernel leaf ties
-- ════════════════════════════════════════════════════════════════

/-- **The even-kernel conv input-VJP leaf tie.** `convFlatBack (padOdd W)` — NOT
    `convFlatBack W` — is the certified input-VJP of `flatConv W b`.

    Two steps and no new mathematics: the ODD tie at `kH + 1` (which is odd exactly because `kH`
    is even), then `backward_unique_of_eq` across `flatConv_padOdd_eq`. -/
theorem convFlatBack_padOdd_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hH : 2 * (kH / 2) = kH) (hW : 2 * (kW / 2) = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * h * w)) :
    convFlatBack (h := h) (w := w) (padOdd W) = (flatConv_has_vjp W b).backward x := by
  funext dy
  rw [convFlatBack_eq_vjp_backward (W := padOdd W) (b := b) (x := x)
        (by omega) (by omega)]
  exact HasVJP.backward_unique_of_eq (flatConv_padOdd_eq hH hW W b)
    (flatConv_has_vjp (padOdd W) b) (flatConv_has_vjp W b) x dy

/-- **The even-kernel STRIDE-2 conv input-VJP leaf tie** — ConvNeXt's three 2×2/s2 downsamples.
    The even-kernel peer of `flatConvStride2Back_eq_vjp_backward`; same proof shape, with the
    conv leaf discharged by `convFlatBack_padOdd_eq_vjp_backward` instead of the odd tie. -/
theorem flatConvStride2Back_padOdd_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hH : 2 * (kH / 2) = kH) (hW : 2 * (kW / 2) = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (x : Vec (ic * (2 * h) * (2 * w))) :
    flatConvStride2Back (h := h) (w := w) (padOdd W)
      = (flatConvStride2_has_vjp W b).backward x := by
  funext dy
  show convFlatBack (h := 2*h) (w := 2*w) (padOdd W) (decimateBack oc h w dy) = _
  rw [convFlatBack_padOdd_eq_vjp_backward hH hW W b x]
  rfl

/-- **The even-kernel STRIDE-4 (patchify) conv input-VJP leaf tie** — ConvNeXt's 4×4/s4 stem.
    The stride-2 tie with one more exact scatter (`decimateOddBack`), matching
    `flatConvStride4 = decimateFlat ∘ decimateOddFlat ∘ flatConv`. -/
theorem flatConvStride4Back_padOdd_eq_vjp_backward {ic oc h w kH kW : Nat}
    (hH : 2 * (kH / 2) = kH) (hW : 2 * (kW / 2) = kW)
    (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (x : Vec (ic * (2 * (2 * h)) * (2 * (2 * w)))) :
    flatConvStride4Back (h := h) (w := w) (padOdd W)
      = (flatConvStride4_has_vjp W b).backward x := by
  funext dy
  show convFlatBack (h := 2*(2*h)) (w := 2*(2*w)) (padOdd W)
      (decimateOddBack oc (2*h) (2*w) (decimateBack oc h w dy)) = _
  rw [convFlatBack_padOdd_eq_vjp_backward hH hW W b x]
  rfl

end Proofs
