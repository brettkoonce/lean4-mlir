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

-- ════════════════════════════════════════════════════════════════
-- § Discharging `hattn`: self-attention as a full FloatClose → UNCONDITIONAL block
-- ════════════════════════════════════════════════════════════════

/-- Self-attention `X ↦ sdpa(X,X,X)` (Q=K=V=X) on the flattened sequence `Vec (n·d)`. -/
noncomputable def sdpaSelfFlat (n d : Nat) (X : Vec (n * d)) : Vec (n * d) :=
  Mat.flatten (sdpa n d (Mat.unflatten X) (Mat.unflatten X) (Mat.unflatten X))

/-- Float self-attention `X ↦ sdpaF(X,X,X)`. -/
noncomputable def sdpaSelfFlatF (M : FloatModel) (fexp : ℝ → ℝ) (n d : Nat)
    (X : Vec (n * d)) : Vec (n * d) :=
  Mat.flatten (M.sdpaF fexp (Mat.unflatten X) (Mat.unflatten X) (Mat.unflatten X))

/-- **Self-attention is a full `FloatClose`** — the piece that discharges the attention
    sublayer's modulus with NOTHING supplied. Magnitude-STABLE (`B = A + rounding`, the real
    output `≤ A` by `sdpa_abs_le`, attention being a convex average — so the block composes to
    depth with a fixed `A`); the error modulus combines the fresh-input rounding `M.attnOutErr`
    (`sdpa_close`, §2c) with the input-sensitivity `attnOutInErr` (`sdpa_input_close`, the
    Lipschitz-through-softmax bound). Q=K=V=X (identity projections). -/
theorem floatClose_sdpaSelf (M : FloatModel) (fexp : ℝ → ℝ) {n d : Nat} {A scaleA eexp : ℝ}
    (hn : 0 < n) (hA : 0 ≤ A) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp n < 1) :
    FloatClose A (A + M.attnOutErr n d A A A scaleA eexp)
      (sdpaSelfFlat n d) (sdpaSelfFlatF M fexp n d)
      (fun e => M.attnOutErr n d A A A scaleA eexp + attnOutInErr n d A A A scaleA e) := by
  have hscaleA0 : 0 ≤ scaleA := (abs_nonneg _).trans hscaleA
  have hAttn0 : 0 ≤ M.attnOutErr n d A A A scaleA eexp :=
    M.attnOutErr_nonneg n d hA hA hA hscaleA0 heexp0 hρ1
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · -- magnitude: real ≤ A (convex), float ≤ A + rounding
    have hX : ∀ a b, |Mat.unflatten v a b| ≤ A := fun a b => hv (finProdFinEquiv (a, b))
    have hreal : |sdpa n d (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
                   (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| ≤ A :=
      sdpa_abs_le hn (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v) hX _ _
    have hround : |M.sdpaF fexp (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
                    (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
                  - sdpa n d (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
                    (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
                  ≤ M.attnOutErr n d A A A scaleA eexp :=
      M.sdpa_close fexp (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
        hA hA hA heexp0 heexp1 hfexp hscaleA hρ1 hX hX hX _ _
    refine ⟨le_trans hreal (by linarith), ?_⟩
    calc |sdpaSelfFlatF M fexp n d v idx|
        = |M.sdpaF fexp (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := rfl
      _ ≤ |M.sdpaF fexp (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
           - sdpa n d (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |sdpa n d (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := by
          simpa using abs_sub_le
            (M.sdpaF fexp (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
              (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2)
            (sdpa n d (Mat.unflatten v) (Mat.unflatten v) (Mat.unflatten v)
              (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2) 0
      _ ≤ M.attnOutErr n d A A A scaleA eexp + A := add_le_add hround hreal
      _ = A + M.attnOutErr n d A A A scaleA eexp := by ring
  · -- error: rounding at the float input (sdpa_close) + sensitivity float-vs-real input
    have hXt : ∀ a b, |Mat.unflatten vt a b| ≤ A := fun a b => hvt (finProdFinEquiv (a, b))
    have hXa : ∀ a b, |Mat.unflatten va a b| ≤ A := fun a b => hva (finProdFinEquiv (a, b))
    have hXe : ∀ a b, |Mat.unflatten vt a b - Mat.unflatten va a b| ≤ e :=
      fun a b => hd (finProdFinEquiv (a, b))
    have he0 : 0 ≤ e := (abs_nonneg _).trans (hd idx)
    have hround : |M.sdpaF fexp (Mat.unflatten vt) (Mat.unflatten vt) (Mat.unflatten vt)
                    (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
                  - sdpa n d (Mat.unflatten vt) (Mat.unflatten vt) (Mat.unflatten vt)
                    (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
                  ≤ M.attnOutErr n d A A A scaleA eexp :=
      M.sdpa_close fexp (Mat.unflatten vt) (Mat.unflatten vt) (Mat.unflatten vt)
        hA hA hA heexp0 heexp1 hfexp hscaleA hρ1 hXt hXt hXt _ _
    have hsens : |sdpa n d (Mat.unflatten vt) (Mat.unflatten vt) (Mat.unflatten vt)
                   (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
                 - sdpa n d (Mat.unflatten va) (Mat.unflatten va) (Mat.unflatten va)
                   (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
                 ≤ attnOutInErr n d A A A scaleA e :=
      sdpa_input_close (Mat.unflatten vt) (Mat.unflatten vt) (Mat.unflatten vt)
        (Mat.unflatten va) (Mat.unflatten va) (Mat.unflatten va)
        hA hA hA he0 hscaleA hXa hXa hXa hXe hXe hXe _ _
    calc |sdpaSelfFlatF M fexp n d vt idx - sdpaSelfFlat n d va idx|
        = |M.sdpaF fexp (Mat.unflatten vt) (Mat.unflatten vt) (Mat.unflatten vt)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
           - sdpa n d (Mat.unflatten va) (Mat.unflatten va) (Mat.unflatten va)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := rfl
      _ ≤ |M.sdpaF fexp (Mat.unflatten vt) (Mat.unflatten vt) (Mat.unflatten vt)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
           - sdpa n d (Mat.unflatten vt) (Mat.unflatten vt) (Mat.unflatten vt)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |sdpa n d (Mat.unflatten vt) (Mat.unflatten vt) (Mat.unflatten vt)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
           - sdpa n d (Mat.unflatten va) (Mat.unflatten va) (Mat.unflatten va)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := abs_sub_le _ _ _
      _ ≤ M.attnOutErr n d A A A scaleA eexp + attnOutInErr n d A A A scaleA e :=
          add_le_add hround hsens

/-- Self-attention float-bridges (magnitude-stable, `B = A + rounding`). -/
theorem floatBridges_sdpaSelf (M : FloatModel) (fexp : ℝ → ℝ) {n d : Nat} {scaleA eexp : ℝ}
    (hn : 0 < n) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp n < 1) :
    FloatBridges (sdpaSelfFlat n d) := by
  intro A hA
  have hscaleA0 : 0 ≤ scaleA := (abs_nonneg _).trans hscaleA
  exact ⟨A + M.attnOutErr n d A A A scaleA eexp, _, _,
    add_nonneg hA (M.attnOutErr_nonneg n d hA hA hA hscaleA0 heexp0 hρ1),
    floatClose_sdpaSelf M fexp hn hA heexp0 heexp1 hfexp hscaleA hρ1⟩

/-- **THE UNCONDITIONAL TRANSFORMER-BLOCK FOLD** (self-attention, Q=K=V=X). The block
    `LN → MHSA → +x → LN → MLP → +x` float-bridges with NO supplied attention hypothesis —
    `floatBridges_vitBlock`'s `hattn` is discharged by `floatBridges_sdpaSelf` (its rounding
    is `sdpa_close`, its input-sensitivity is `sdpa_input_close`, both proved). Every piece of
    the ViT encoder block is now proved in rounding, a-posteriori in the activation magnitude.
    Composes to depth: a 12-layer encoder is `FloatBridges.comp`/`floatClose_iterate` of this. -/
theorem floatBridges_vitBlockSelf {n d dff : Nat} (M : FloatModel)
    (W₁ : Mat d dff) (b₁ : Vec dff) (W₂ : Mat dff d) (b₂ : Vec d)
    (fgelu fexp : ℝ → ℝ)
    {εln γln βln w' β egelu scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hegelu : 0 ≤ egelu) (hd : 0 < d) (hdff : 0 < dff)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β)
    (hln : FloatBridges (layerNormForward d εln γln βln))
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp n < 1) :
    FloatBridges
      (perRowFlat n d (Proofs.residual
          (Proofs.dense W₂ b₂ ∘ gelu dff ∘ Proofs.dense W₁ b₁ ∘ layerNormForward d εln γln βln))
        ∘ Proofs.residual (sdpaSelfFlat n d)) :=
  floatBridges_vitBlock M W₁ b₁ W₂ b₂ fgelu hw' hβ hegelu hd hdff hg hW₁ hb₁ hW₂ hb₂ hln
    ((floatBridges_sdpaSelf M fexp hn heexp0 heexp1 hfexp hscaleA hρ1).residual M)

end Proofs
