import LeanMlir.Proofs.Architectures.ConvNeXtChannelLN
import LeanMlir.Proofs.Float.BnPerChannelBackFloatBridge
import LeanMlir.Proofs.Float.LinBackFloatBridge

/-! # ℝ→Float32 bridge: ConvNeXt's REAL channel LayerNorm (§2n step 1)

`ConvNeXtChannelLN.lean` (§2m) replaced ConvNeXt's scalar-γ/β whole-map LayerNorm with the
reference's `channel_layer_norm`: `h·w` statistics per example, each over the `c` channels at ONE
spatial position, with a per-channel `[c]` affine. That is a different function at 21 of the net's
22 LN sites, and the float story still described the retired one. This file is its float peer.

**Route A pays off twice.** `chanLNTensor3` is built as a conjugation

  `reassocBack ∘ transposeFlat ∘ rowLNVecFlat ∘ transposeFlat ∘ reassocFwd`

and *every* outer piece is a coordinate permutation — exact in float, magnitude-stable, 1-Lipschitz
— so it is `floatBridges_gather` five times over with one genuinely-new middle. That middle is ViT's
per-token vector-LN lifted rowwise, i.e. `floatBridges_bn`'s rsqrt keystone read at each spatial
position. No new transcendental, no new budget shape: this is `floatBridges_bnPerChannelTensor3`'s
blueprint (relabel → block-diagonal → relabel back) with a transpose inserted and the per-row map
changed from per-channel BN to per-position LN.

Three new op-bridges are needed to say that:

* `floatBridges_transposeFlat` — the `[c,s] ↔ [s,c]` transpose IS a `gather` (definitionally: both
  read `v (finProdFinEquiv (idx.2, idx.1))`), so it inherits `floatClose_gather`;
* `floatBridges_biasAdd` — the `+β` translation of the vector-LN affine. One rounded add per
  coordinate: magnitude `(1+u)(A+Bb)`, modulus `e ↦ u·(A+Bb) + e`. The constant-vector cousin of
  `floatClose_addResidual`;
* `floatBridges_layerNormVec` — `layerNormVec = (+β) ∘ layerScale γ ∘ LN(1,0)` *definitionally*
  (`ViTVecLN.lean`), so the vector-γ/β LN is the supplied pure-normalise bridge, then
  `floatBridges_diagBack` at the exact stored `γ` (`es = 0`), then the translation.

The BACKWARD peer is the same conjugation with `rowLNVecFlatBack` in the middle — the row's
`diagBack γ` then the three-term `bn_grad_input` at `γ = 1`, `FloatClose.perRowIdx`-lifted exactly
as `floatBridges_bnPerChannelFlatBack` lifts the per-channel BN backward.

**What the backward chain is tied to.** `chanLNTensor3Back` is *written* as a hand-composed reverse
chain, in the same sense as `cnxBlockBodyBack` / `convnextInputGrad`
(`ConvNeXtBackFloatBridge.lean`): each factor is the adjoint of its forward factor (a permutation's
adjoint is its inverse permutation; the row map's is `bn_grad_input ∘ diagBack γ`). ✅ **That is now
PROVED, not asserted** — `ConvNeXtBackCertifiedTie.chanLNTensor3Back_eq_chanLN_vjp` shows the chain
equals `(chanLNTensor3_has_vjp …).backward`, so this file's closeness is closeness to the
**certified** gradient. The §B block ties (`cnxBodyWithChanLNBack_eq_vjp`, `cnxBlockChBack_eq_vjp`)
sit on top of it, and note where the LN slot's tie had to come from: the scalar block tie could pin
an abstract `lnB` to whatever it liked, while the channel-LN one had to earn it.
-/

namespace Proofs

open FloatModel
open Proofs.StableHLO (transposeFlat)

-- ════════════════════════════════════════════════════════════════
-- § The layout permutations (all four are gathers)
-- ════════════════════════════════════════════════════════════════

/-- The identity float-bridges — the `FloatBridges` peer of `floatClose_id`. Magnitude and modulus
    pass through. It is the `convNextStageK`/`convNextStageChK` depth-0 base case, and (§2n) the
    channel-LN net's HEAD LN slot: the reference has no head LN, so both the forward skeleton's
    `lnHead` and the backward's `lnBhead` are filled with `id`. Lives here rather than in
    `ConvNeXtWholeFloatBridge` because the forward and backward ConvNeXt bridge files both need it
    and this is the file they share. -/
theorem floatBridges_idVec {m : Nat} : FloatBridges (id : Vec m → Vec m) :=
  fun A hA => ⟨A, _, _, hA, ⟨fun _v hv i => ⟨hv i, hv i⟩, fun _ _ _ _ _ hd i => hd i⟩⟩

/-- The Mat transpose as an index `Equiv` — swap the two factors of the row-major split. -/
def transposeEquiv (m n : Nat) : Fin (n * m) ≃ Fin (m * n) :=
  finProdFinEquiv.symm.trans ((Equiv.prodComm (Fin n) (Fin m)).trans finProdFinEquiv)

/-- **The flat transpose IS a reindex.** `transposeFlat m n v idx = v (finProdFinEquiv (idx.2, idx.1))`
    on the row-major split, which is exactly `gather (transposeEquiv m n)` — no arithmetic, so exact
    in float (`transposeFlat_diff` uses the same reading on the differentiability side). -/
theorem transposeFlat_eq_gather (m n : Nat) :
    transposeFlat m n = gather (transposeEquiv m n) := by
  funext v idx; rfl

/-- **The flat transpose float-bridges** (magnitude-stable, modulus `id`). -/
theorem floatBridges_transposeFlat (m n : Nat) : FloatBridges (transposeFlat m n) := by
  rw [transposeFlat_eq_gather]
  exact floatBridges_gather (transposeEquiv m n)

/-- The Tensor3 → Mat-split re-association float-bridges (`reassocFwd = gather (reassocEquiv …)`). -/
theorem floatBridges_reassocFwd (oc h w : Nat) : FloatBridges (reassocFwd oc h w) :=
  floatBridges_gather (reassocEquiv oc h w)

/-- The Mat-split → Tensor3 re-association float-bridges (the inverse relabeling). -/
theorem floatBridges_reassocBack (oc h w : Nat) : FloatBridges (reassocBack oc h w) :=
  floatBridges_gather (reassocEquiv oc h w).symm

-- ════════════════════════════════════════════════════════════════
-- § The `+β` translation (the second half of the vector-LN affine)
-- ════════════════════════════════════════════════════════════════

/-- Float bias translation: one rounded add per coordinate, `fl(zₖ ⊕ βₖ)`. -/
noncomputable def FloatModel.biasAddF {n : Nat} (M : FloatModel) (β : Vec n) (z : Vec n) : Vec n :=
  fun k => M.add (z k) (β k)

/-- **The bias translation is `FloatClose`.** `z ↦ z + β` at a bounded constant vector (`|βₖ| ≤ Bb`):
    the real output is within `A + Bb`, the float one adds a single rounding, and the error modulus
    is that rounding plus the inherited shift (the translation is 1-Lipschitz and β is exact). The
    constant-vector cousin of `floatClose_addResidual`; the ViT/ConvNeXt vector-LN `+β` token. -/
theorem floatClose_biasAdd {n : Nat} (M : FloatModel) (β : Vec n) {Bb A : ℝ}
    (hβ : ∀ k, |β k| ≤ Bb) :
    FloatClose A (A + Bb + M.u * (A + Bb))
      (fun (z : Vec n) k => z k + β k) (M.biasAddF β)
      (fun e => M.u * (A + Bb) + e) := by
  have hu := M.u_nonneg
  refine ⟨fun v hv k => ?_, fun vt va e _hva hvt hd k => ?_⟩
  · have hA0 : 0 ≤ A := (abs_nonneg _).trans (hv k)
    have hB0 : 0 ≤ Bb := (abs_nonneg _).trans (hβ k)
    have hsum : |v k + β k| ≤ A + Bb := (abs_add_le _ _).trans (add_le_add (hv k) (hβ k))
    refine ⟨hsum.trans (le_add_of_nonneg_right (mul_nonneg hu (by linarith))), ?_⟩
    calc |M.biasAddF β v k| = |M.rnd (v k + β k)| := rfl
      _ ≤ |M.rnd (v k + β k) - (v k + β k)| + |v k + β k| := by
          simpa using abs_sub_le (M.rnd (v k + β k)) (v k + β k) 0
      _ ≤ M.u * |v k + β k| + |v k + β k| := add_le_add (M.err _) le_rfl
      _ ≤ M.u * (A + Bb) + (A + Bb) := add_le_add (mul_le_mul_of_nonneg_left hsum hu) hsum
      _ = A + Bb + M.u * (A + Bb) := by ring
  · have hsum : |vt k + β k| ≤ A + Bb := (abs_add_le _ _).trans (add_le_add (hvt k) (hβ k))
    have hshift : |(vt k + β k) - (va k + β k)| ≤ e := by
      simpa using hd k
    calc |M.biasAddF β vt k - (va k + β k)|
        ≤ |M.rnd (vt k + β k) - (vt k + β k)| + |(vt k + β k) - (va k + β k)| := abs_sub_le _ _ _
      _ ≤ M.u * (A + Bb) + e :=
          add_le_add ((M.err _).trans (mul_le_mul_of_nonneg_left hsum hu)) hshift

/-- The bias translation float-bridges. -/
theorem floatBridges_biasAdd {n : Nat} (M : FloatModel) (β : Vec n) {Bb : ℝ}
    (hn : 0 < n) (hβ : ∀ k, |β k| ≤ Bb) :
    FloatBridges (fun (z : Vec n) k => z k + β k) := fun A hA =>
  ⟨_, _, _, (floatClose_biasAdd M β hβ (A := A)).cod_nonneg hA hn, floatClose_biasAdd M β hβ⟩

-- ════════════════════════════════════════════════════════════════
-- § The vector-[D] LayerNorm (one row of the channel LN)
-- ════════════════════════════════════════════════════════════════

/-- **Vector-γ/β LayerNorm float-bridges.** `layerNormVec D ε γ β = (+β) ∘ layerScale γ ∘ LN(1,0)`
    definitionally (`ViTVecLN.lean` — the decomposition the render also emits, `lnRowF`(1,0) →
    `rowScaleF` → `rowBiasF`), so this is: the supplied pure-normalise bridge (the rsqrt keystone,
    discharged by `floatBridges_bn` since `layerNormForward = bnForward`), then `floatBridges_diagBack`
    at the exact stored `γ` (`es = 0`, `layerScale γ = diagBack γ`), then `floatBridges_biasAdd`. -/
theorem floatBridges_layerNormVec {D : Nat} (M : FloatModel) {ε : ℝ} (γ β : Vec D)
    {Gd Bb : ℝ} (hD : 0 < D) (hγ : ∀ i, |γ i| ≤ Gd) (hβ : ∀ i, |β i| ≤ Bb)
    (hln : FloatBridges (layerNormForward D ε 1 0)) :
    FloatBridges (layerNormVec D ε γ β) := by
  have heq : layerNormVec D ε γ β
      = (fun (z : Vec D) k => z k + β k) ∘ diagBack γ ∘ layerNormForward D ε 1 0 := rfl
  rw [heq]
  exact (hln.comp (floatBridges_diagBack (es := 0) M γ γ hD hγ (fun _ => by simp))).comp
    (floatBridges_biasAdd M β hD hβ)

/-- **The rowwise vector-LN float-bridges** — `rowLNVecFlat s c = perRowFlat s c (layerNormVec c …)`
    definitionally, so this is `FloatBridges.perRow` of the row bridge: `s` independent copies at
    one shared budget (all rows share the same `γ`/`β` and the same supplied stat accuracy). -/
theorem floatBridges_rowLNVecFlat {s c : Nat} (M : FloatModel) {ε : ℝ} (γ β : Vec c)
    {Gd Bb : ℝ} (hc : 0 < c) (hγ : ∀ i, |γ i| ≤ Gd) (hβ : ∀ i, |β i| ≤ Bb)
    (hln : FloatBridges (layerNormForward c ε 1 0)) :
    FloatBridges (rowLNVecFlat s c ε γ β) := by
  have heq : rowLNVecFlat s c ε γ β = perRowFlat s c (layerNormVec c ε γ β) := rfl
  rw [heq]
  exact FloatBridges.perRow s (floatBridges_layerNormVec M γ β hc hγ hβ hln)

-- ════════════════════════════════════════════════════════════════
-- § THE CHANNEL LAYERNORM (network Tensor3 layout)
-- ════════════════════════════════════════════════════════════════

/-- **ConvNeXt's channel LayerNorm float-bridges** — the op the shipped net actually contains
    (§2m). One `.comp` chain over the conjugation: re-associate to the Mat-split `[c, h·w]`,
    transpose to `[h·w, c]`, run the rowwise vector-LN (each row = one spatial position over its
    `c` channels), transpose and re-associate back. The four layout maps are permutations
    (`floatBridges_gather`, modulus `id`, magnitude-stable) so the whole budget is the row LN's —
    exactly the shape of `floatBridges_bnPerChannelTensor3`, with a transpose inserted.

    The only supplied fact is the pure-normalise `FloatBridges (layerNormForward c ε 1 0)`, i.e.
    the SAME rsqrt keystone the scalar-LN world supplied; the channel flip did not add a
    transcendental. Closes under `[propext, Classical.choice, Quot.sound]`. -/
theorem floatBridges_chanLNTensor3 {c h w : Nat} (M : FloatModel) {ε : ℝ} (γ β : Vec c)
    {Gd Bb : ℝ} (hc : 0 < c) (hγ : ∀ i, |γ i| ≤ Gd) (hβ : ∀ i, |β i| ≤ Bb)
    (hln : FloatBridges (layerNormForward c ε 1 0)) :
    FloatBridges (chanLNTensor3 c h w ε γ β) := by
  unfold chanLNTensor3
  exact ((((floatBridges_reassocFwd c h w).comp
      (floatBridges_transposeFlat c (h * w))).comp
      (floatBridges_rowLNVecFlat M γ β hc hγ hβ hln)).comp
      (floatBridges_transposeFlat (h * w) c)).comp
      (floatBridges_reassocBack c h w)

-- ════════════════════════════════════════════════════════════════
-- § The BACKWARD peer (same conjugation, the row backward in the middle)
-- ════════════════════════════════════════════════════════════════

/-- **The rowwise vector-LN input-VJP.** Per spatial row: scale the cotangent by the per-channel
    `γ` (`layerScale`'s adjoint is `diagBack γ`), then the consolidated three-term `bn_grad_input`
    at `γ = 1` over that row's `c` channels (LN's adjoint = BN's, `layerNormForward = bnForward`).
    The `+β` translation contributes the identity, so it does not appear. -/
noncomputable def rowLNVecFlatBack (s c : Nat) (ε : ℝ) (γ : Vec c) (X : Vec (s * c)) :
    Vec (s * c) → Vec (s * c) :=
  perRowIdxFlat s c (fun r => bn_grad_input c ε 1 (Mat.unflatten X r) ∘ diagBack γ)

/-- **The rowwise vector-LN backward float-bridges.** `FloatClose.perRowIdx` of the per-row
    composite `floatClose_diagBack` (the `γ` scale, at a supplied float `fγ` within `egam`) then
    `floatClose_bnBack` (the three-term input gradient at that row's saved activation, supplied
    float `istd`/`x̂` within `es`/`exh`). Uniform hypotheses across rows ⇒ one shared budget, which
    is what `perRowIdx` needs. The exact lift `floatBridges_bnPerChannelFlatBack` performs for the
    per-channel BN backward, one axis over. -/
theorem floatBridges_rowLNVecFlatBack {s c : Nat} (M : FloatModel) {ε : ℝ} (γ fγ : Vec c)
    (X : Vec (s * c)) (fs : Fin s → ℝ) (fxh : Fin s → Vec c)
    {Gd egam S Xh es exh : ℝ} (hs0 : 0 < s) (hc : 0 < c)
    (hγ : ∀ i, |γ i| ≤ Gd) (hfγ : ∀ i, |fγ i - γ i| ≤ egam)
    (hst : ∀ r, |fs r - bnIstd c (Mat.unflatten X r) ε| ≤ es)
    (hSabs : ∀ r, |bnIstd c (Mat.unflatten X r) ε| ≤ S)
    (hxh : ∀ r i, |bnXhat c ε (Mat.unflatten X r) i| ≤ Xh)
    (hfxh : ∀ r i, |fxh r i - bnXhat c ε (Mat.unflatten X r) i| ≤ exh) :
    FloatBridges (rowLNVecFlatBack s c ε γ X) := by
  intro A hA
  have hrow := fun r : Fin s =>
    (floatClose_diagBack M γ fγ hγ hfγ (A := A)).comp
      (floatClose_bnBack M (Mat.unflatten X r) (fxh r) (fs r) hc (le_refl |(1 : ℝ)|)
        (hst r) (hSabs r) (hxh r) (hfxh r))
  have hpr := FloatClose.perRowIdx (d := c) s hrow
  exact ⟨_, _, _, hpr.cod_nonneg hA (Nat.mul_pos hs0 hc), hpr⟩

/-- The saved activation as the row backward sees it: the `[h·w, c]` view of `x`, one row per
    spatial position holding its `c` channels. Naming this keeps the backward's operating-point
    hypotheses (`bnIstd`/`bnXhat` per row) readable — it is `chanLNTensor3`'s own first two factors. -/
noncomputable def chanLNRows (c h w : Nat) (x : Vec (c * h * w)) : Vec ((h * w) * c) :=
  transposeFlat c (h * w) (reassocFwd c h w x)

/-- **The channel-LN input-VJP** (as a function of the cotangent, at a saved input `x`) — the exact
    reverse of `chanLNTensor3`'s five factors. A permutation's adjoint is its inverse permutation,
    so the conjugation comes back unchanged and only the middle flips to `rowLNVecFlatBack`, read
    at the TRANSPOSED saved input (the row backward needs its own row's activation).

    Hand-composed in the sense of `cnxBlockBodyBack`, but **tied**: it equals
    `(chanLNTensor3_has_vjp …).backward` by `ConvNeXtBackCertifiedTie.chanLNTensor3Back_eq_chanLN_vjp`
    (§B). Note it takes no `β` — the `+β` translation's VJP is the identity, and the tie proves the
    certified backward is β-free too. -/
noncomputable def chanLNTensor3Back (c h w : Nat) (ε : ℝ) (γ : Vec c) (x : Vec (c * h * w)) :
    Vec (c * h * w) → Vec (c * h * w) :=
  reassocBack c h w ∘
    transposeFlat (h * w) c ∘
    rowLNVecFlatBack (h * w) c ε γ (chanLNRows c h w x) ∘
    transposeFlat c (h * w) ∘
    reassocFwd c h w

/-- **The channel-LN backward float-bridges** — the forward chain's shape with
    `floatBridges_rowLNVecFlatBack` in the middle. This is what discharges the abstract
    `FloatBridges lnB` hypotheses of `floatBridges_cnxBlockBodyBack` / `floatBridges_cnxDownBack` /
    `convnext_grad_floatBridges` for the channel-LN net — those three are already LN-abstract, so
    the §2m flip costs the whole-net backward exactly this one op. -/
theorem floatBridges_chanLNTensor3Back {c h w : Nat} (M : FloatModel) {ε : ℝ} (γ fγ : Vec c)
    (x : Vec (c * h * w))
    (fs : Fin (h * w) → ℝ) (fxh : Fin (h * w) → Vec c)
    {Gd egam S Xh es exh : ℝ} (hhw : 0 < h * w) (hc : 0 < c)
    (hγ : ∀ i, |γ i| ≤ Gd) (hfγ : ∀ i, |fγ i - γ i| ≤ egam)
    (hst : ∀ r, |fs r - bnIstd c (Mat.unflatten (chanLNRows c h w x) r) ε| ≤ es)
    (hSabs : ∀ r, |bnIstd c (Mat.unflatten (chanLNRows c h w x) r) ε| ≤ S)
    (hxh : ∀ r i, |bnXhat c ε (Mat.unflatten (chanLNRows c h w x) r) i| ≤ Xh)
    (hfxh : ∀ r i, |fxh r i - bnXhat c ε (Mat.unflatten (chanLNRows c h w x) r) i| ≤ exh) :
    FloatBridges (chanLNTensor3Back c h w ε γ x) := by
  unfold chanLNTensor3Back
  exact ((((floatBridges_reassocFwd c h w).comp
      (floatBridges_transposeFlat c (h * w))).comp
      (floatBridges_rowLNVecFlatBack M γ fγ _ fs fxh hhw hc hγ hfγ hst hSabs hxh hfxh)).comp
      (floatBridges_transposeFlat (h * w) c)).comp
      (floatBridges_reassocBack c h w)

end Proofs
