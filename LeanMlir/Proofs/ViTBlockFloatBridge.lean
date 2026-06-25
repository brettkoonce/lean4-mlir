import LeanMlir.Proofs.ViTFloatBridge
import LeanMlir.Proofs.ViTAttentionFloatBridge

/-!
# ℝ→Float32 bridge: ViT — the transformer-block fold (the `Mat`↔`Vec` seam)

The transformer block is `LN → MHSA → +x → LN → MLP → +x`. Its two halves live in
different spaces: **MHSA mixes across tokens** (`Mat n d`, §2c `sdpa_close`), while
**LayerNorm and the MLP act per token** (`Vec d`, §2a/§2b/§2d). The only new plumbing the
fold needs is the *seam* between them: a way to lift a per-token `Vec d → Vec d` bridge to
the whole-sequence `Mat n d ≅ Vec (n·d)` so it composes with the cross-token attention.

That seam is **`perRowFlat`** + **`FloatClose.perRow`/`FloatBridges.perRow`**: applying a
per-token map to every row is, on the flattened `Vec (n·d)`, float-close with the *same*
magnitude `B` and *same* error modulus `L` (the rows are independent, so a per-coordinate
input perturbation `e` stays `e` per row and the per-row modulus carries over verbatim).

With the seam, the block folds in one `FloatBridges.comp`:

  `vitBlock = perRowFlat (MLP-sublayer, §2d) ∘ attentionSublayer`

The MLP+LN₂ sublayer is fully discharged (`floatBridges_vitMlpResidual.perRow`). The
attention sublayer enters as a **supplied** `FloatBridges attnSub` hypothesis — exactly the
operating-point pattern by which BN/LN entered the MBConv/MLP folds. Honest accounting of
that hypothesis: its *fresh-input rounding* is `sdpa_close` (§2c, proved); its *input-
sensitivity* modulus (how `sdpa` responds to a perturbed `X` — softmax-through-`QKᵀ`
Lipschitz) is the remaining open analysis. So `floatBridges_vitBlock` is the whole block
*modulo* the one attention input-sensitivity constant, with everything else proved.
-/

namespace Proofs

open FloatModel

-- ════════════════════════════════════════════════════════════════
-- § The per-row seam: lift a per-token bridge to the whole sequence
-- ════════════════════════════════════════════════════════════════

/-- Apply a per-token map `f : Vec d → Vec d` to every row, on the flattened `Vec (n·d)`
    (`Mat.unflatten` → per-row `f` → `Mat.flatten`). The whole-sequence form of a per-token
    op (LayerNorm, the MLP sub-block), so it can `.comp` the cross-token attention. -/
noncomputable def perRowFlat (n d : Nat) (f : Vec d → Vec d) : Vec (n * d) → Vec (n * d) :=
  fun v => Mat.flatten (fun i => f (Mat.unflatten v i))

/-- `perRowFlat` reads coordinatewise as the per-row map at `(row, col) = finProdFinEquiv.symm idx`. -/
theorem perRowFlat_apply {n d : Nat} (f : Vec d → Vec d) (v : Vec (n * d)) (idx : Fin (n * d)) :
    perRowFlat n d f v idx
      = f (Mat.unflatten v (finProdFinEquiv.symm idx).1) (finProdFinEquiv.symm idx).2 := rfl

/-- **The seam — `FloatClose.perRow`.** A per-token `FloatClose A B f fF L` lifts to the
    whole sequence `FloatClose A B (perRowFlat n d f) (perRowFlat n d fF) L` with the SAME
    magnitude and SAME error modulus: each row is an independent copy of `f`, so a
    per-coordinate input bound/perturbation `≤ A`/`≤ e` restricts to each row and the per-row
    output bound `B` / modulus `L e` is exactly the per-flat-coordinate one. -/
theorem FloatClose.perRow {d : Nat} (n : Nat) {A B : ℝ} {f fF : Vec d → Vec d} {L : ℝ → ℝ}
    (hf : FloatClose A B f fF L) :
    FloatClose A B (perRowFlat n d f) (perRowFlat n d fF) L := by
  obtain ⟨hm, he⟩ := hf
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · -- magnitude: restrict the per-coordinate bound to the row, apply the per-row bound
    have hrow : ∀ j', |Mat.unflatten v (finProdFinEquiv.symm idx).1 j'| ≤ A :=
      fun j' => hv (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))
    have hboth := hm (Mat.unflatten v (finProdFinEquiv.symm idx).1) hrow (finProdFinEquiv.symm idx).2
    rw [perRowFlat_apply, perRowFlat_apply]
    exact hboth
  · -- error: the row perturbation is the inherited `e`, the per-row modulus is `L e`
    have hva' : ∀ j', |Mat.unflatten va (finProdFinEquiv.symm idx).1 j'| ≤ A :=
      fun j' => hva (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))
    have hvt' : ∀ j', |Mat.unflatten vt (finProdFinEquiv.symm idx).1 j'| ≤ A :=
      fun j' => hvt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))
    have hd' : ∀ j', |Mat.unflatten vt (finProdFinEquiv.symm idx).1 j'
                    - Mat.unflatten va (finProdFinEquiv.symm idx).1 j'| ≤ e :=
      fun j' => hd (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))
    rw [perRowFlat_apply, perRowFlat_apply]
    exact he (Mat.unflatten vt (finProdFinEquiv.symm idx).1)
      (Mat.unflatten va (finProdFinEquiv.symm idx).1) e hva' hvt' hd' (finProdFinEquiv.symm idx).2

/-- **`FloatBridges.perRow`** — the seam in bridge form (magnitudes threaded automatically). -/
theorem FloatBridges.perRow (n : Nat) {d : Nat} {f : Vec d → Vec d} (hf : FloatBridges f) :
    FloatBridges (perRowFlat n d f) := by
  intro A hA
  obtain ⟨B, L, fF, hB, hfc⟩ := hf A hA
  exact ⟨B, L, perRowFlat n d fF, hB, hfc.perRow n⟩

-- ════════════════════════════════════════════════════════════════
-- § The transformer-block fold
-- ════════════════════════════════════════════════════════════════

/-- **THE TRANSFORMER-BLOCK FOLD.** The ViT encoder block `LN → MHSA → +x → LN → MLP → +x`
    float-bridges, built by one `FloatBridges.comp` from its two sublayers:

    * the **MLP sublayer** `+x ∘ (MLP ∘ LN₂)` applied per token — fully discharged by
      `floatBridges_vitMlpResidual` (§2d) lifted across the sequence with `FloatBridges.perRow`;
    * the **attention sublayer** `attnSub : Vec (n·d) → Vec (n·d)` (= `+x ∘ MHSA ∘ LN₁`) —
      supplied as the hypothesis `hattn`, the way BN/LN entered earlier folds. Its fresh-input
      rounding is `sdpa_close` (§2c); its input-sensitivity modulus is the one remaining open
      piece. So this is the whole block proved *modulo* that single attention constant.

    Composes to depth: a 12-layer encoder is `floatClose_iterate`/`.comp` of this block. -/
theorem floatBridges_vitBlock {n d dff : Nat} (M : FloatModel)
    (W₁ : Mat d dff) (b₁ : Vec dff) (W₂ : Mat dff d) (b₂ : Vec d) (fgelu : ℝ → ℝ)
    {εln γln βln w' β egelu : ℝ}
    (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hegelu : 0 ≤ egelu) (hd : 0 < d) (hdff : 0 < dff)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β)
    (hln : FloatBridges (layerNormForward d εln γln βln))
    {attnSub : Vec (n * d) → Vec (n * d)} (hattn : FloatBridges attnSub) :
    FloatBridges
      (perRowFlat n d (Proofs.residual
          (Proofs.dense W₂ b₂ ∘ gelu dff ∘ Proofs.dense W₁ b₁ ∘ layerNormForward d εln γln βln))
        ∘ attnSub) :=
  FloatBridges.comp hattn
    (FloatBridges.perRow n
      (floatBridges_vitMlpResidual M W₁ b₁ W₂ b₂ fgelu hw' hβ hegelu hd hdff hg
        hW₁ hb₁ hW₂ hb₂ hln))

end Proofs
