import LeanMlir.Proofs.Foundation.MobileNetV4BackB0
import LeanMlir.Proofs.Architectures.EfficientNetFullB0

/-! # EfficientNet's four remaining holes, closed — and they were all one shape

`planning/mnv4_verified.md` §8e swept the repo for **certified batched forwards with no BACKWARD
graph** and found five. One (`efficientnetForwardB`) is the whole-net forward, which no net has and
which is the artifact-tie item. The other four are EfficientNet's:

| forward | is | why it had no backward graph |
|---|---|---|
| `mbExpFwdB` | `projB ∘ seB ∘ dwbsB ∘ cbsB` | `mbBodyBackBatchedGraph` is the *same* chain, but its types bake in `ic = oc = c` for the residual — it could not serve the channel-changing form |
| `mbNoExpFwdB` | `projB ∘ seB ∘ dwbsB` | MBConv1: no expand stage, so nothing composed it |
| `mbStridedFwdB` | `projB ∘ seB ∘ dwbsSB ∘ cbsB@2h` | the downsample variant |
| `headFwdB` | `dense ∘ GAP ∘ cbsB` | same shape as MNv4's head, swish for relu |

⭐ **Every constituent stage was already certified AND already had a backward graph.** Nothing was
missing mathematically — what was missing was the *composition*, and before `CertLayer` each one
would have been a hand-written `den`-chain + `rfl` in the shape §8b describes. Here they are four
`comp` chains and the faithfulness is `CertLayer.faithful`.

⚠ **This is the argument for the fold, stated as a measurement.** These four holes existed for as
long as EfficientNet has, nothing was red, and no test failed — the theorems simply did not exist.
They became one afternoon's work only once composition was a combinator.

## Every stage here is GLOBAL

EfficientNet is swish/sigmoid throughout, both smooth, so every layer below has `ok = True`: the
backward graph denotes the VJP at **every** input, no side conditions. Contrast MNv4/R34/R50, whose
relu kinks force `_at`. That also means the four block layers compose with **no hypothesis
threading at all** — `(enetMbExp …).ok` is a conjunction of `True`s.
-/

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The EfficientNet stages, as CertLayers (all global)
-- ════════════════════════════════════════════════════════════════

/-- Batched **conv → bn → swish** (`cbsB`) as a `CertLayer`. -/
noncomputable def enetCbsLayer (N : Nat) {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) where
  fwd := cbsB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (cbsB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (cbsB_has_vjp N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => cbsBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => cbsBackBatchedGraph_faithful W b ε hε γ β x e

/-- Batched **depthwise → bn → swish** (`dwbsB`) as a `CertLayer`. -/
noncomputable def enetDwbsLayer (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := dwbsB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (dwbsB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (dwbsB_has_vjp N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => dwbsBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => dwbsBackBatchedGraph_faithful W b ε hε γ β x e

/-- Batched **STRIDE-2 depthwise → bn → swish** (`dwbsSB`) as a `CertLayer`. -/
noncomputable def enetDwbsSLayer (N : Nat) {c h w kH kW : Nat}
    (W : DepthwiseKernel c kH kW) (b : Vec c) (ε : ℝ) (hε : 0 < ε) (γ β : Vec c) :
    CertLayer (N * (c * (2 * h) * (2 * w))) (N * (c * h * w)) where
  fwd := dwbsSB N (h := h) (w := w) W b ε γ β
  ok := fun _ => True
  diff := fun x _ => (dwbsSB_differentiable N W b ε hε γ β) x
  vjp := fun x _ => (dwbsSB_has_vjp N W b ε hε γ β).toHasVJPAt x
  graph := fun x e => dwbsSBackBatchedGraph W b ε γ β x e
  faithful := fun x _ e => dwbsSBackBatchedGraph_faithful W b ε hε γ β x e

/-- The batched **squeeze-and-excitation** gate (`seB`) as a `CertLayer`. ⭐ Its backward token
    `.seBackBatched` ties `seB_has_vjp` by `rfl` — the multiplicative fan-in is definitional. -/
noncomputable def enetSeLayer (N : Nat) {c h w r : Nat}
    (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c) :
    CertLayer (N * (c * h * w)) (N * (c * h * w)) where
  fwd := seB N (h := h) (w := w) W₁ b₁ W₂ b₂
  ok := fun _ => True
  diff := fun x _ => (seB_differentiable N (h := h) (w := w) W₁ b₁ W₂ b₂) x
  vjp := fun x _ => (seB_has_vjp N (h := h) (w := w) W₁ b₁ W₂ b₂).toHasVJPAt x
  graph := fun x e =>
    .seBackBatched (N := N) "%seW1" "%seb1" "%seW2" "%seb2" "%seX" W₁ b₁ W₂ b₂ x e
  faithful := fun _ _ _ => rfl

-- ════════════════════════════════════════════════════════════════
-- § The four holes, as four `comp` chains
-- ════════════════════════════════════════════════════════════════

/-- **`mbExpFwdB`** — MBConv6: expand → depthwise → SE → project. ⭐ Unlike `mbBodyB` this is
    channel-changing (`ic ≠ oc` allowed), which is exactly why `mbBodyBackBatchedGraph` could not
    serve it: that graph's types bake in `ic = oc = c` for the residual it feeds. -/
noncomputable def enetMbExp (N : Nat) {ic mid oc h w : Nat}
    (expand : CertLayer (N * (ic * h * w)) (N * (mid * h * w)))
    (depthwise : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (se : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) :=
  expand.comp (depthwise.comp (se.comp project))

/-- **`mbNoExpFwdB`** — MBConv1: no expand stage, so the depthwise runs on the block input. -/
noncomputable def enetMbNoExp (N : Nat) {ic oc h w : Nat}
    (depthwise : CertLayer (N * (ic * h * w)) (N * (ic * h * w)))
    (se : CertLayer (N * (ic * h * w)) (N * (ic * h * w)))
    (project : CertLayer (N * (ic * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * h * w)) (N * (oc * h * w)) :=
  depthwise.comp (se.comp project)

/-- **`mbStridedFwdB`** — the downsample MBConv. ⚠ The expand runs at the INPUT resolution `2h` and
    the *depthwise* carries the stride, the same placement MNv4's post-strided UIB uses. -/
noncomputable def enetMbStrided (N : Nat) {ic mid oc h w : Nat}
    (expand : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * (2 * h) * (2 * w))))
    (depthwise : CertLayer (N * (mid * (2 * h) * (2 * w))) (N * (mid * h * w)))
    (se : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w))) :
    CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (oc * h * w)) :=
  expand.comp (depthwise.comp (se.comp project))

/-- **`headFwdB`** — EfficientNet's head: 1×1 conv-bn-**swish** → GAP → dense. ⭐ Structurally
    MNv4's head with swish for relu, so it reuses `mnv4GapLayer` and `mnv4DenseLayer` verbatim —
    those two are net-agnostic despite the name. -/
noncomputable def enetHead (N : Nat) {c oc h w nC : Nat}
    (headConv : CertLayer (N * (c * h * w)) (N * (oc * h * w)))
    (gap : CertLayer (N * (oc * h * w)) (N * oc))
    (cls : CertLayer (N * oc) (N * nC)) :
    CertLayer (N * (c * h * w)) (N * nC) :=
  headConv.comp (gap.comp cls)

-- ════════════════════════════════════════════════════════════════
-- § The four faithfulness theorems — each is `CertLayer.faithful`
-- ════════════════════════════════════════════════════════════════

theorem enetMbExp_faithful (N : Nat) {ic mid oc h w : Nat}
    (expand : CertLayer (N * (ic * h * w)) (N * (mid * h * w)))
    (depthwise se : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w)))
    (x : Vec (N * (ic * h * w))) (hx : (enetMbExp N expand depthwise se project).ok x)
    (e : SHlo (N * (oc * h * w))) :
    den ((enetMbExp N expand depthwise se project).graph x e)
      = ((enetMbExp N expand depthwise se project).vjp x hx).backward (den e) :=
  (enetMbExp N expand depthwise se project).faithful x hx e

theorem enetMbNoExp_faithful (N : Nat) {ic oc h w : Nat}
    (depthwise se : CertLayer (N * (ic * h * w)) (N * (ic * h * w)))
    (project : CertLayer (N * (ic * h * w)) (N * (oc * h * w)))
    (x : Vec (N * (ic * h * w))) (hx : (enetMbNoExp N depthwise se project).ok x)
    (e : SHlo (N * (oc * h * w))) :
    den ((enetMbNoExp N depthwise se project).graph x e)
      = ((enetMbNoExp N depthwise se project).vjp x hx).backward (den e) :=
  (enetMbNoExp N depthwise se project).faithful x hx e

theorem enetMbStrided_faithful (N : Nat) {ic mid oc h w : Nat}
    (expand : CertLayer (N * (ic * (2 * h) * (2 * w))) (N * (mid * (2 * h) * (2 * w))))
    (depthwise : CertLayer (N * (mid * (2 * h) * (2 * w))) (N * (mid * h * w)))
    (se : CertLayer (N * (mid * h * w)) (N * (mid * h * w)))
    (project : CertLayer (N * (mid * h * w)) (N * (oc * h * w)))
    (x : Vec (N * (ic * (2 * h) * (2 * w))))
    (hx : (enetMbStrided N expand depthwise se project).ok x)
    (e : SHlo (N * (oc * h * w))) :
    den ((enetMbStrided N expand depthwise se project).graph x e)
      = ((enetMbStrided N expand depthwise se project).vjp x hx).backward (den e) :=
  (enetMbStrided N expand depthwise se project).faithful x hx e

theorem enetHead_faithful (N : Nat) {c oc h w nC : Nat}
    (headConv : CertLayer (N * (c * h * w)) (N * (oc * h * w)))
    (gap : CertLayer (N * (oc * h * w)) (N * oc))
    (cls : CertLayer (N * oc) (N * nC))
    (x : Vec (N * (c * h * w))) (hx : (enetHead N headConv gap cls).ok x)
    (e : SHlo (N * nC)) :
    den ((enetHead N headConv gap cls).graph x e)
      = ((enetHead N headConv gap cls).vjp x hx).backward (den e) :=
  (enetHead N headConv gap cls).faithful x hx e

-- ════════════════════════════════════════════════════════════════
-- § ⭐⭐ CLOSING §8e's HOLES AGAINST THE *NAMED* FORWARDS
-- ════════════════════════════════════════════════════════════════

/-! ⚠⚠ **The `comp` chains above did NOT close §8e's holes, and re-running the sweep is what said
so.** They give the *capability* — a certified composition — but they are stated over abstract
`CertLayer` arguments, so nothing referenced `mbExpFwdB_has_vjp` and friends. The sweep reported
the same five holes afterwards. **A fix that the measurement does not confirm is not a fix.**

⭐ Probing the named forwards then found the real situation, which is **not** what §8e assumed:

| forward | verdict |
|---|---|
| `mbStridedFwdB` | ⭐ **never a hole** — `mbDownBodyB_has_vjp` is *definitionally the same object* (`rfl`), and it already has a certified graph. A duplicate NAME, not a missing proof. |
| `mbExpFwdB` | same shape as `mbBodyB_has_vjp`, which bakes in `ic = oc = c`; tied where the types meet |
| `mbNoExpFwdB` | genuine — nothing composed `projB ∘ seB ∘ dwbsB` |
| `headFwdB` | genuine — nothing composed `dense ∘ GAP ∘ cbsB` |

▶ So §8e over-counted: a sweep keyed on *names* cannot see that two names denote one object. The
lesson is the same one §4c(a) taught about the relu6 detector, in the opposite direction — there a
detector could not fire, here one fires spuriously. **Both are measurement bugs, and only re-running
the measurement after the fix catches either.** -/

/-- ⭐ `mbStridedFwdB`'s backward graph — the SAME graph `mbDownBodyB` already had, restated against
    the other name for that VJP. `rfl`-level, because the two `_has_vjp` defs are one object. -/
theorem mbStridedFwdBackBatchedGraph_faithful {N ic mid oc h w kHd kWd r : Nat}
    (We : Kernel4 mid ic 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (e : SHlo (N * (oc * h * w))) :
    den (mbDownBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂
          Wp bp εp γp βp x e)
      = (mbStridedFwdB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).backward x (den e) :=
  mbDownBodyBackBatchedGraph_faithful We be εe hεe γe βe Wd bd εd hεd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp x e

/-- `mbExpFwdB`'s backward graph, where its types meet `mbBodyB`'s (`ic = oc = c`, the residual
    MBConv6 that EfficientNet actually stacks). -/
theorem mbExpFwdBackBatchedGraph_faithful {N c mid h w kHd kWd r : Nat}
    (We : Kernel4 mid c 1 1) (be : Vec mid) (εe : ℝ) (hεe : 0 < εe) (γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd : Vec mid) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec mid)
    (Wz₁ : Mat mid r) (bz₁ : Vec r) (Wz₂ : Mat r mid) (bz₂ : Vec mid)
    (Wp : Kernel4 c mid 1 1) (bp : Vec c) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec c)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * (c * h * w))) :
    den (mbBodyBackBatchedGraph We be εe γe βe Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x e)
      = (mbExpFwdB_has_vjp N We be εe hεe γe βe Wd bd εd hεd γd βd
          Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp).backward x (den e) :=
  mbBodyBackBatchedGraph_faithful We be εe hεe γe βe Wd bd εd hεd γd βd
    Wz₁ bz₁ Wz₂ bz₂ Wp bp εp hεp γp βp x e

/-- **`mbNoExpFwdB`'s backward graph** — genuinely new: MBConv1 has no expand stage, so
    `dwbsB⁻¹ ∘ seB⁻¹ ∘ projB⁻¹` had never been chained. -/
noncomputable def mbNoExpBackBatchedGraph {N ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) : SHlo (N * (ic * h * w)) :=
  let xD := dwbsB N (h := h) (w := w) Wd bd εd γd βd x
  let xS := seB N (h := h) (w := w) Wz₁ bz₁ Wz₂ bz₂ xD
  dwbsBackBatchedGraph Wd bd εd γd βd x
    (.seBackBatched (N := N) "%seW1" "%seb1" "%seW2" "%seb2" "%seX" Wz₁ bz₁ Wz₂ bz₂ xD
      (projBackBatchedGraph Wp bp εp γp βp xS e))

theorem mbNoExpBackBatchedGraph_faithful {N ic oc h w kHd kWd r : Nat}
    (Wd : DepthwiseKernel ic kHd kWd) (bd : Vec ic) (εd : ℝ) (hεd : 0 < εd) (γd βd : Vec ic)
    (Wz₁ : Mat ic r) (bz₁ : Vec r) (Wz₂ : Mat r ic) (bz₂ : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp : Vec oc) (εp : ℝ) (hεp : 0 < εp) (γp βp : Vec oc)
    (x : Vec (N * (ic * h * w))) (e : SHlo (N * (oc * h * w))) :
    den (mbNoExpBackBatchedGraph Wd bd εd γd βd Wz₁ bz₁ Wz₂ bz₂ Wp bp εp γp βp x e)
      = (mbNoExpFwdB_has_vjp N Wd bd εd hεd γd βd Wz₁ bz₁ Wz₂ bz₂
          Wp bp εp hεp γp βp).backward x (den e) := by
  rw [mbNoExpBackBatchedGraph, dwbsBackBatchedGraph_faithful (hε := hεd),
      seBackBatched_faithful, projBackBatchedGraph_faithful (hε := hεp)]
  simp only [mbNoExpFwdB_has_vjp, vjp_comp, Function.comp_apply]
  rfl

/-- **`headFwdB`'s backward graph** — genuinely new: `cbsB⁻¹ ∘ GAP⁻¹ ∘ dense⁻¹`. The EfficientNet
    peer of MNv4's head, and the last stage-level hole in the repo. -/
noncomputable def headBackBatchedGraph {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * nC)) : SHlo (N * (c * h * w)) :=
  cbsBackBatchedGraph Wh bh εh γh βh x
    (.gapBackBatched (N := N) (c := oc) (h := h) (w := w)
      (.denseRowBack (N := N) (a := oc) (c := nC) "%Wfc" Wfc e))

theorem headBackBatchedGraph_faithful {N c oc h w nC : Nat}
    (Wh : Kernel4 oc c 1 1) (bh : Vec oc) (εh : ℝ) (hεh : 0 < εh) (γh βh : Vec oc)
    (Wfc : Mat oc nC) (bfc : Vec nC)
    (x : Vec (N * (c * h * w))) (e : SHlo (N * nC)) :
    den (headBackBatchedGraph Wh bh εh γh βh Wfc bfc x e)
      = (headFwdB_has_vjp N Wh bh εh hεh γh βh Wfc bfc).backward x (den e) := by
  rw [headBackBatchedGraph, cbsBackBatchedGraph_faithful (hε := hεh)]
  simp only [headFwdB_has_vjp, vjp_comp, Function.comp_apply]
  rfl

/-- **EfficientNet-B0's trunk, as a type-level check.** MBConv1 (no-expand) at full resolution,
    then a strided MBConv6, then a stride-1 MBConv6 — the `stem → MBConv1 → MBConv6/s2 → MBConv6`
    ladder `EfficientNetRenderPC` documents for its representative forward. -/
noncomputable def enetTrunk (N : Nat) {c₀ c₁ c₂ h w : Nat}
    (stem : CertLayer (N * (c₀ * (2 * (2 * h)) * (2 * (2 * w))))
                      (N * (c₁ * (2 * h) * (2 * w))))
    (mb1 : CertLayer (N * (c₁ * (2 * h) * (2 * w))) (N * (c₁ * (2 * h) * (2 * w))))
    (mb6s2 : CertLayer (N * (c₁ * (2 * h) * (2 * w))) (N * (c₂ * h * w)))
    (mb6 : CertLayer (N * (c₂ * h * w)) (N * (c₂ * h * w))) :
    CertLayer (N * (c₀ * (2 * (2 * h)) * (2 * (2 * w)))) (N * (c₂ * h * w)) :=
  stem.comp (mb1.comp (mb6s2.comp mb6))

end Proofs.StableHLO
