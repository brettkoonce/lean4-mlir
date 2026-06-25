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

-- ════════════════════════════════════════════════════════════════
-- § Projected attention: Q=XWq, K=XWk, V=XWv (the real MHSA, single head)
--
-- Q/K/V are per-token denses of X (a three-way fan-in at X). Each projection's
-- float-vs-real drift threads into sdpa's Q/K/V slots: its rounding via `dense_close`
-- (≤ layerBudget), its magnitude via `dense_abs_le` (≤ layerAct). sdpa_close (rounding
-- at the float projections) + sdpa_input_close (sensitivity, projection drift as the
-- perturbation) then bound the projected attention. The output projection Wo is a plain
-- per-token dense AFTER sdpa, so it composes via `perRowFlat (dense Wo bo)`.
-- ════════════════════════════════════════════════════════════════

/-- Real per-token projection `X ↦ (Xᵢ·W + b)ᵢ` on the flattened sequence. -/
noncomputable def projR {n d : Nat} (W : Mat d d) (b : Vec d) (v : Vec (n * d)) : Mat n d :=
  fun i => Proofs.dense W b (Mat.unflatten v i)

/-- Float per-token projection. -/
noncomputable def projF (M : FloatModel) {n d : Nat} (W : Mat d d) (b : Vec d)
    (v : Vec (n * d)) : Mat n d :=
  fun i => M.dense W b (Mat.unflatten v i)

/-- Real projection magnitude: `|projR W b v i k| ≤ layerAct d w' β A`. -/
theorem projR_abs_le {n d : Nat} (W : Mat d d) (b : Vec d) {w' β A : ℝ}
    (hA : 0 ≤ A) (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b j| ≤ β)
    (v : Vec (n * d)) (hv : ∀ idx, |v idx| ≤ A) (i : Fin n) (k : Fin d) :
    |projR W b v i k| ≤ layerAct d w' β A :=
  dense_abs_le hA hW hb (fun l => hv (finProdFinEquiv (i, l))) k

/-- Float projection magnitude: `|projF M W b v i k| ≤ layerAct + layerBudget(e=0)`. -/
theorem projF_abs_le (M : FloatModel) {n d : Nat} (W : Mat d d) (b : Vec d) {w' β A : ℝ}
    (hw' : 0 ≤ w') (hA : 0 ≤ A) (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b j| ≤ β)
    (v : Vec (n * d)) (hv : ∀ idx, |v idx| ≤ A) (i : Fin n) (k : Fin d) :
    |projF M W b v i k| ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 := by
  have hround : |M.dense W b (Mat.unflatten v i) k - Proofs.dense W b (Mat.unflatten v i) k|
      ≤ layerBudget M.u d w' β A 0 :=
    (M.dense_close_fresh W b (Mat.unflatten v i) k).trans
      (M.denseErr_le_uniform hw' le_rfl hW hb (fun l => hv (finProdFinEquiv (i, l))) k)
  have hreal : |Proofs.dense W b (Mat.unflatten v i) k| ≤ layerAct d w' β A :=
    dense_abs_le hA hW hb (fun l => hv (finProdFinEquiv (i, l))) k
  show |M.dense W b (Mat.unflatten v i) k| ≤ layerAct d w' β A + layerBudget M.u d w' β A 0
  calc |M.dense W b (Mat.unflatten v i) k|
      ≤ |M.dense W b (Mat.unflatten v i) k - Proofs.dense W b (Mat.unflatten v i) k|
        + |Proofs.dense W b (Mat.unflatten v i) k| := by
        simpa using abs_sub_le (M.dense W b (Mat.unflatten v i) k)
          (Proofs.dense W b (Mat.unflatten v i) k) 0
    _ ≤ layerBudget M.u d w' β A 0 + layerAct d w' β A := add_le_add hround hreal
    _ = layerAct d w' β A + layerBudget M.u d w' β A 0 := by ring

/-- Float-projection-of-`vt` vs real-projection-of-`va`: `≤ layerBudget d w' β A e`
    (`dense_close` at the perturbed token row, uniform-magnitude `denseErr`). -/
theorem projFR_close (M : FloatModel) {n d : Nat} (W : Mat d d) (b : Vec d) {w' β A e : ℝ}
    (hw' : 0 ≤ w') (he : 0 ≤ e) (hW : ∀ i j, |W i j| ≤ w') (hb : ∀ j, |b j| ≤ β)
    (vt va : Vec (n * d)) (hva : ∀ idx, |va idx| ≤ A)
    (hd : ∀ idx, |vt idx - va idx| ≤ e) (i : Fin n) (k : Fin d) :
    |projF M W b vt i k - projR W b va i k| ≤ layerBudget M.u d w' β A e :=
  (M.dense_close W b (Mat.unflatten vt i) (Mat.unflatten va i) e he
    (fun l => hd (finProdFinEquiv (i, l))) k).trans
    (M.denseErr_le_uniform hw' he hW hb (fun l => hva (finProdFinEquiv (i, l))) k)

/-- Real projected attention `X ↦ sdpa(XWq+bq, XWk+bk, XWv+bv)` on the flattened sequence. -/
noncomputable def projAttnFlat {n d : Nat} (Wq Wk Wv : Mat d d) (bq bk bv : Vec d)
    (v : Vec (n * d)) : Vec (n * d) :=
  Mat.flatten (sdpa n d (projR Wq bq v) (projR Wk bk v) (projR Wv bv v))

/-- Float projected attention. -/
noncomputable def projAttnFlatF (M : FloatModel) (fexp : ℝ → ℝ) {n d : Nat}
    (Wq Wk Wv : Mat d d) (bq bk bv : Vec d) (v : Vec (n * d)) : Vec (n * d) :=
  Mat.flatten (M.sdpaF fexp (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v))

/-- Propagated magnitude: the float-projection bound `layerAct + layerBudget(0)` plus the
    sdpa rounding at that magnitude. -/
noncomputable def projAttnB (M : FloatModel) (n d : Nat) (w' β A scaleA eexp : ℝ) : ℝ :=
  (layerAct d w' β A + layerBudget M.u d w' β A 0)
    + M.attnOutErr n d (layerAct d w' β A + layerBudget M.u d w' β A 0)
        (layerAct d w' β A + layerBudget M.u d w' β A 0)
        (layerAct d w' β A + layerBudget M.u d w' β A 0) scaleA eexp

/-- Error modulus: sdpa rounding at the float-projection magnitude + sdpa input-sensitivity
    at the real-projection magnitude with the projection drift `layerBudget(e)` as δ. -/
noncomputable def projAttnL (M : FloatModel) (n d : Nat) (w' β A scaleA eexp e : ℝ) : ℝ :=
  M.attnOutErr n d (layerAct d w' β A + layerBudget M.u d w' β A 0)
      (layerAct d w' β A + layerBudget M.u d w' β A 0)
      (layerAct d w' β A + layerBudget M.u d w' β A 0) scaleA eexp
    + attnOutInErr n d (layerAct d w' β A) (layerAct d w' β A) (layerAct d w' β A)
        scaleA (layerBudget M.u d w' β A e)

/-- **Projected single-head attention is a `FloatClose`.** `X ↦ sdpa(XWq+bq, XWk+bk, XWv+bv)`
    — Q/K/V are per-token denses of the SAME `X` (the three-way fan-in). Each projection's
    float drift threads into `sdpa`'s slots: rounding via `dense_close` (≤ `layerBudget`),
    magnitude via `dense_abs_le` (≤ `layerAct`). The error modulus is `sdpa_close` (rounding at
    the float projections, magnitude `layerAct+layerBudget(0)`) + `sdpa_input_close` (sensitivity,
    the projection drift `layerBudget(e)` as the per-row logit shift). The genuine MHSA core. -/
theorem floatClose_projAttn (M : FloatModel) (fexp : ℝ → ℝ) {n d : Nat}
    (Wq Wk Wv : Mat d d) (bq bk bv : Vec d) {w' β A scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hA : 0 ≤ A)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d : ℝ)| ≤ scaleA) (hρ1 : smRho M.u eexp n < 1)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hbq : ∀ j, |bq j| ≤ β)
    (hWk : ∀ i j, |Wk i j| ≤ w') (hbk : ∀ j, |bk j| ≤ β)
    (hWv : ∀ i j, |Wv i j| ≤ w') (hbv : ∀ j, |bv j| ≤ β) :
    FloatClose A (projAttnB M n d w' β A scaleA eexp)
      (projAttnFlat (n := n) Wq Wk Wv bq bk bv) (projAttnFlatF (n := n) M fexp Wq Wk Wv bq bk bv)
      (fun e => projAttnL M n d w' β A scaleA eexp e) := by
  have hscaleA0 : 0 ≤ scaleA := (abs_nonneg _).trans hscaleA
  have hLa0 : 0 ≤ layerAct d w' β A := layerAct_nonneg hw' hβ hA
  have hLb00 : 0 ≤ layerBudget M.u d w' β A 0 := layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl
  have hqAF0 : 0 ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 := by linarith
  have hAttnF0 : 0 ≤ M.attnOutErr n d (layerAct d w' β A + layerBudget M.u d w' β A 0)
      (layerAct d w' β A + layerBudget M.u d w' β A 0)
      (layerAct d w' β A + layerBudget M.u d w' β A 0) scaleA eexp :=
    M.attnOutErr_nonneg n d hqAF0 hqAF0 hqAF0 hscaleA0 heexp0 hρ1
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · -- magnitudes
    have hVR : ∀ a b, |projR Wv bv v a b| ≤ layerAct d w' β A :=
      fun a b => projR_abs_le Wv bv hA hWv hbv v hv a b
    have hVF : ∀ a b, |projF M Wv bv v a b| ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 :=
      fun a b => projF_abs_le M Wv bv hw' hA hWv hbv v hv a b
    have hQF : ∀ a b, |projF M Wq bq v a b| ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 :=
      fun a b => projF_abs_le M Wq bq hw' hA hWq hbq v hv a b
    have hKF : ∀ a b, |projF M Wk bk v a b| ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 :=
      fun a b => projF_abs_le M Wk bk hw' hA hWk hbk v hv a b
    have hrealR : |sdpa n d (projR Wq bq v) (projR Wk bk v) (projR Wv bv v)
                   (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| ≤ layerAct d w' β A :=
      sdpa_abs_le hn (projR Wq bq v) (projR Wk bk v) (projR Wv bv v) hVR _ _
    have hrealF : |sdpa n d (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
                   (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
                  ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 :=
      sdpa_abs_le hn (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v) hVF _ _
    have hroundF : |M.sdpaF fexp (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
                     (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
                   - sdpa n d (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
                     (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
                  ≤ M.attnOutErr n d (layerAct d w' β A + layerBudget M.u d w' β A 0)
                      (layerAct d w' β A + layerBudget M.u d w' β A 0)
                      (layerAct d w' β A + layerBudget M.u d w' β A 0) scaleA eexp :=
      M.sdpa_close fexp (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
        hqAF0 hqAF0 hqAF0 heexp0 heexp1 hfexp hscaleA hρ1 hQF hKF hVF _ _
    refine ⟨le_trans hrealR (by unfold projAttnB; linarith), ?_⟩
    show |M.sdpaF fexp (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
         ≤ projAttnB M n d w' β A scaleA eexp
    calc |M.sdpaF fexp (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ |M.sdpaF fexp (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
           - sdpa n d (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |sdpa n d (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := by
          simpa using abs_sub_le
            (M.sdpaF fexp (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
              (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2)
            (sdpa n d (projF M Wq bq v) (projF M Wk bk v) (projF M Wv bv v)
              (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2) 0
      _ ≤ M.attnOutErr n d (layerAct d w' β A + layerBudget M.u d w' β A 0)
            (layerAct d w' β A + layerBudget M.u d w' β A 0)
            (layerAct d w' β A + layerBudget M.u d w' β A 0) scaleA eexp
          + (layerAct d w' β A + layerBudget M.u d w' β A 0) := add_le_add hroundF hrealF
      _ = projAttnB M n d w' β A scaleA eexp := by unfold projAttnB; ring
  · -- error
    have he0 : 0 ≤ e := (abs_nonneg _).trans (hd idx)
    have hLbe0 : 0 ≤ layerBudget M.u d w' β A e := layerBudget_nonneg M.u_nonneg hw' hβ hA he0
    have hQaR : ∀ a b, |projR Wq bq va a b| ≤ layerAct d w' β A :=
      fun a b => projR_abs_le Wq bq hA hWq hbq va hva a b
    have hKaR : ∀ a b, |projR Wk bk va a b| ≤ layerAct d w' β A :=
      fun a b => projR_abs_le Wk bk hA hWk hbk va hva a b
    have hVaR : ∀ a b, |projR Wv bv va a b| ≤ layerAct d w' β A :=
      fun a b => projR_abs_le Wv bv hA hWv hbv va hva a b
    have hQFt : ∀ a b, |projF M Wq bq vt a b| ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 :=
      fun a b => projF_abs_le M Wq bq hw' hA hWq hbq vt hvt a b
    have hKFt : ∀ a b, |projF M Wk bk vt a b| ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 :=
      fun a b => projF_abs_le M Wk bk hw' hA hWk hbk vt hvt a b
    have hVFt : ∀ a b, |projF M Wv bv vt a b| ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 :=
      fun a b => projF_abs_le M Wv bv hw' hA hWv hbv vt hvt a b
    have hQe : ∀ a b, |projF M Wq bq vt a b - projR Wq bq va a b| ≤ layerBudget M.u d w' β A e :=
      fun a b => projFR_close M Wq bq hw' he0 hWq hbq vt va hva hd a b
    have hKe : ∀ a b, |projF M Wk bk vt a b - projR Wk bk va a b| ≤ layerBudget M.u d w' β A e :=
      fun a b => projFR_close M Wk bk hw' he0 hWk hbk vt va hva hd a b
    have hVe : ∀ a b, |projF M Wv bv vt a b - projR Wv bv va a b| ≤ layerBudget M.u d w' β A e :=
      fun a b => projFR_close M Wv bv hw' he0 hWv hbv vt va hva hd a b
    have hroundF : |M.sdpaF fexp (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
                     (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
                   - sdpa n d (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
                     (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
                  ≤ M.attnOutErr n d (layerAct d w' β A + layerBudget M.u d w' β A 0)
                      (layerAct d w' β A + layerBudget M.u d w' β A 0)
                      (layerAct d w' β A + layerBudget M.u d w' β A 0) scaleA eexp :=
      M.sdpa_close fexp (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
        hqAF0 hqAF0 hqAF0 heexp0 heexp1 hfexp hscaleA hρ1 hQFt hKFt hVFt _ _
    have hsens : |sdpa n d (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
                   (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
                 - sdpa n d (projR Wq bq va) (projR Wk bk va) (projR Wv bv va)
                   (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
                 ≤ attnOutInErr n d (layerAct d w' β A) (layerAct d w' β A) (layerAct d w' β A)
                     scaleA (layerBudget M.u d w' β A e) :=
      sdpa_input_close (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
        (projR Wq bq va) (projR Wk bk va) (projR Wv bv va)
        hLa0 hLa0 hLa0 hLbe0 hscaleA hQaR hKaR hVaR hQe hKe hVe _ _
    show |M.sdpaF fexp (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - sdpa n d (projR Wq bq va) (projR Wk bk va) (projR Wv bv va)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
         ≤ projAttnL M n d w' β A scaleA eexp e
    calc |M.sdpaF fexp (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - sdpa n d (projR Wq bq va) (projR Wk bk va) (projR Wv bv va)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ |M.sdpaF fexp (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
           - sdpa n d (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |sdpa n d (projF M Wq bq vt) (projF M Wk bk vt) (projF M Wv bv vt)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
           - sdpa n d (projR Wq bq va) (projR Wk bk va) (projR Wv bv va)
             (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := abs_sub_le _ _ _
      _ ≤ M.attnOutErr n d (layerAct d w' β A + layerBudget M.u d w' β A 0)
            (layerAct d w' β A + layerBudget M.u d w' β A 0)
            (layerAct d w' β A + layerBudget M.u d w' β A 0) scaleA eexp
          + attnOutInErr n d (layerAct d w' β A) (layerAct d w' β A) (layerAct d w' β A)
              scaleA (layerBudget M.u d w' β A e) := add_le_add hroundF hsens
      _ = projAttnL M n d w' β A scaleA eexp e := rfl

/-- Projected attention float-bridges. -/
theorem floatBridges_projAttn (M : FloatModel) (fexp : ℝ → ℝ) {n d : Nat}
    (Wq Wk Wv : Mat d d) (bq bk bv : Vec d) {w' β scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d : ℝ)| ≤ scaleA) (hρ1 : smRho M.u eexp n < 1)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hbq : ∀ j, |bq j| ≤ β)
    (hWk : ∀ i j, |Wk i j| ≤ w') (hbk : ∀ j, |bk j| ≤ β)
    (hWv : ∀ i j, |Wv i j| ≤ w') (hbv : ∀ j, |bv j| ≤ β) :
    FloatBridges (projAttnFlat (n := n) Wq Wk Wv bq bk bv) := by
  intro A hA
  have hLa0 : 0 ≤ layerAct d w' β A := layerAct_nonneg hw' hβ hA
  have hLb00 : 0 ≤ layerBudget M.u d w' β A 0 := layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl
  have hqAF0 : 0 ≤ layerAct d w' β A + layerBudget M.u d w' β A 0 := by linarith
  have hscaleA0 : 0 ≤ scaleA := (abs_nonneg _).trans hscaleA
  refine ⟨projAttnB M n d w' β A scaleA eexp, _, _, ?_,
    floatClose_projAttn M fexp Wq Wk Wv bq bk bv hn hw' hβ hA heexp0 heexp1 hfexp hscaleA hρ1
      hWq hbq hWk hbk hWv hbv⟩
  unfold projAttnB
  exact add_nonneg hqAF0 (M.attnOutErr_nonneg n d hqAF0 hqAF0 hqAF0 hscaleA0 heexp0 hρ1)

/-- **Full single-head MHSA (with the output projection Wo) float-bridges.** The attention
    sublayer body `X ↦ Wo·sdpa(XWq, XWk, XWv)` — the projected attention (`floatBridges_projAttn`)
    composed with the per-token output projection `Wo` (`floatBridges_dense.perRow`). All four
    projections Wq/Wk/Wv/Wo are now genuine learned denses (no longer the identity-projection
    Q=K=V=X of `floatBridges_sdpaSelf`). -/
theorem floatBridges_mhsaProj (M : FloatModel) (fexp : ℝ → ℝ) {n d : Nat}
    (Wq Wk Wv Wo : Mat d d) (bq bk bv bo : Vec d) {w' β scaleA eexp : ℝ}
    (hn : 0 < n) (hd : 0 < d) (hw' : 0 ≤ w') (hβ : 0 ≤ β)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d : ℝ)| ≤ scaleA) (hρ1 : smRho M.u eexp n < 1)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hbq : ∀ j, |bq j| ≤ β)
    (hWk : ∀ i j, |Wk i j| ≤ w') (hbk : ∀ j, |bk j| ≤ β)
    (hWv : ∀ i j, |Wv i j| ≤ w') (hbv : ∀ j, |bv j| ≤ β)
    (hWo : ∀ i j, |Wo i j| ≤ w') (hbo : ∀ j, |bo j| ≤ β) :
    FloatBridges (perRowFlat n d (Proofs.dense Wo bo)
      ∘ projAttnFlat (n := n) Wq Wk Wv bq bk bv) :=
  FloatBridges.comp
    (floatBridges_projAttn M fexp Wq Wk Wv bq bk bv hn hw' hβ heexp0 heexp1 hfexp hscaleA hρ1
      hWq hbq hWk hbk hWv hbv)
    (FloatBridges.perRow n (floatBridges_dense M Wo bo hw' hβ hd hWo hbo))

/-- **THE FULLY-PROJECTED ViT BLOCK** (genuine Wq/Wk/Wv/Wo, single head). The encoder block
    `LN → MHSA → +x → LN → MLP → +x` with the real projected MHSA, float-bridges with nothing
    supplied — `floatBridges_vitBlock`'s `hattn` is discharged by `floatBridges_mhsaProj.residual`.
    The self-attention `floatBridges_vitBlockSelf` is the `Wq=Wk=Wv=I, Wo=I` special case;
    this is the deployed form. Every piece — projections, sdpa rounding, sdpa sensitivity, LN,
    GELU, MLP — proved in rounding, a-posteriori in the activation magnitude. -/
theorem floatBridges_vitBlockProj {n d dff : Nat} (M : FloatModel)
    (W₁ : Mat d dff) (b₁ : Vec dff) (W₂ : Mat dff d) (b₂ : Vec d)
    (Wq Wk Wv Wo : Mat d d) (bq bk bv bo : Vec d) (fgelu fexp : ℝ → ℝ)
    {εln γln βln w' β egelu scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hegelu : 0 ≤ egelu) (hd : 0 < d) (hdff : 0 < dff)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hbq : ∀ j, |bq j| ≤ β)
    (hWk : ∀ i j, |Wk i j| ≤ w') (hbk : ∀ j, |bk j| ≤ β)
    (hWv : ∀ i j, |Wv i j| ≤ w') (hbv : ∀ j, |bv j| ≤ β)
    (hWo : ∀ i j, |Wo i j| ≤ w') (hbo : ∀ j, |bo j| ≤ β)
    (hln : FloatBridges (layerNormForward d εln γln βln))
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (d : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp n < 1) :
    FloatBridges
      (perRowFlat n d (Proofs.residual
          (Proofs.dense W₂ b₂ ∘ gelu dff ∘ Proofs.dense W₁ b₁ ∘ layerNormForward d εln γln βln))
        ∘ Proofs.residual (perRowFlat n d (Proofs.dense Wo bo)
            ∘ projAttnFlat (n := n) Wq Wk Wv bq bk bv)) :=
  floatBridges_vitBlock M W₁ b₁ W₂ b₂ fgelu hw' hβ hegelu hd hdff hg hW₁ hb₁ hW₂ hb₂ hln
    ((floatBridges_mhsaProj M fexp Wq Wk Wv Wo bq bk bv bo hn hd hw' hβ heexp0 heexp1 hfexp
      hscaleA hρ1 hWq hbq hWk hbk hWv hbv hWo hbo).residual M)

-- ════════════════════════════════════════════════════════════════
-- § Multi-head: the reshape is a pure reindex (exact in float)
--
-- Multi-head attention is `h` independent single-head attentions over disjoint feature
-- slabs. In a HEAD-MAJOR layout (heads contiguous) that is exactly `perRowFlat` (heads =
-- blocks). The split/concat between the token-major `Vec (n·(h·dh))` and the head-major
-- `Vec (h·(n·dh))` is a coordinate PERMUTATION — exact in float, magnitude-stable, 1-Lip —
-- so it preserves FloatClose. Multi-head = reshape⁻¹ ∘ perRow(single-head) ∘ reshape.
-- ════════════════════════════════════════════════════════════════

/-- Reindex a vector by a bijection of index sets (`v ∘ e`). A pure permutation/reshape:
    no arithmetic, so exact in float. -/
noncomputable def gather {p q : Nat} (e : Fin p ≃ Fin q) (v : Vec q) : Vec p := fun i => v (e i)

/-- **A reindex is `FloatClose` with modulus `id`** — exact in float, magnitude-stable
    (`|v (e i)| ≤ A`), 1-Lipschitz on the inherited error. The reshape/split/concat backbone. -/
theorem floatClose_gather {p q : Nat} (e : Fin p ≃ Fin q) (A : ℝ) :
    FloatClose A A (gather e) (gather e) (id : ℝ → ℝ) :=
  ⟨fun _v hv i => ⟨hv (e i), hv (e i)⟩, fun _vt _va _e _ _ hd i => hd (e i)⟩

/-- A reindex float-bridges (magnitude-stable). -/
theorem floatBridges_gather {p q : Nat} (e : Fin p ≃ Fin q) : FloatBridges (gather e) :=
  fun A hA => ⟨A, _, _, hA, floatClose_gather e A⟩

/-- **The head-split reshape** `Vec (h·(n·dh)) ↔ Vec (n·(h·dh))` — the `(h,n,dh)` head-major
    order vs the `(n,h,dh)` token-major order (swap the head and token axes). Built from
    `finProdFinEquiv` + `prodAssoc`/`prodComm`; its exact coordinate action is irrelevant to
    the bridge (`floatClose_gather` holds for ANY equiv), only that it is a bijection. -/
def mhReshape (h n dh : Nat) : Fin (h * (n * dh)) ≃ Fin (n * (h * dh)) :=
  finProdFinEquiv.symm.trans
    ((Equiv.prodCongr (Equiv.refl (Fin h)) finProdFinEquiv.symm).trans
      ((Equiv.prodAssoc (Fin h) (Fin n) (Fin dh)).symm.trans
        ((Equiv.prodCongr (Equiv.prodComm (Fin h) (Fin n)) (Equiv.refl (Fin dh))).trans
          ((Equiv.prodAssoc (Fin n) (Fin h) (Fin dh)).trans
            ((Equiv.prodCongr (Equiv.refl (Fin n)) finProdFinEquiv).trans finProdFinEquiv)))))

/-- **Multi-head self-attention** on the token-major sequence: reshape to head-major, apply
    single-head `sdpaSelfFlat` (per-head scale `1/√dh`) to each head-block, reshape back. -/
noncomputable def mhSdpaSelfFlat (h n dh : Nat) : Vec (n * (h * dh)) → Vec (n * (h * dh)) :=
  gather (mhReshape h n dh).symm ∘ perRowFlat h (n * dh) (sdpaSelfFlat n dh)
    ∘ gather (mhReshape h n dh)

/-- **Multi-head self-attention float-bridges.** One `FloatBridges.comp` chain over the
    reshape (`floatBridges_gather`) / per-head (`FloatBridges.perRow` of `floatBridges_sdpaSelf`)
    / reshape-back — multi-head is `h` parallel single-heads, the reshape being exact in float.
    The `reshape` IS the "multi-head reshape"; no new analysis, just the layout permutation. -/
theorem floatBridges_mhSdpaSelf (M : FloatModel) (fexp : ℝ → ℝ) {h n dh : Nat}
    {scaleA eexp : ℝ} (hn : 0 < n) (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (dh : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp n < 1) :
    FloatBridges (mhSdpaSelfFlat h n dh) := by
  unfold mhSdpaSelfFlat
  exact ((floatBridges_gather (mhReshape h n dh)).comp
      (FloatBridges.perRow h
        (floatBridges_sdpaSelf M fexp hn heexp0 heexp1 hfexp hscaleA hρ1))).comp
    (floatBridges_gather (mhReshape h n dh).symm)

/-- **THE MULTI-HEAD ViT BLOCK** (`h` heads, self-attention per head). The encoder block with
    multi-head MHSA float-bridges unconditionally — `floatBridges_vitBlock`'s `hattn` is
    discharged by `floatBridges_mhSdpaSelf.residual`. The single-head `floatBridges_vitBlockSelf`
    is the `h = 1` case. The reshape contributed no new budget (exact in float); each head is the
    proved single-head attention at feature dim `dh`. -/
theorem floatBridges_vitBlockMH {h n dh dff : Nat} (M : FloatModel)
    (W₁ : Mat (h * dh) dff) (b₁ : Vec dff) (W₂ : Mat dff (h * dh)) (b₂ : Vec (h * dh))
    (fgelu fexp : ℝ → ℝ)
    {εln γln βln w' β egelu scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hegelu : 0 ≤ egelu)
    (hd : 0 < h * dh) (hdff : 0 < dff)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β)
    (hln : FloatBridges (layerNormForward (h * dh) εln γln βln))
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (dh : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp n < 1) :
    FloatBridges
      (perRowFlat n (h * dh) (Proofs.residual
          (Proofs.dense W₂ b₂ ∘ gelu dff ∘ Proofs.dense W₁ b₁
            ∘ layerNormForward (h * dh) εln γln βln))
        ∘ Proofs.residual (mhSdpaSelfFlat h n dh)) :=
  floatBridges_vitBlock M W₁ b₁ W₂ b₂ fgelu hw' hβ hegelu hd hdff hg hW₁ hb₁ hW₂ hb₂ hln
    ((floatBridges_mhSdpaSelf M fexp hn heexp0 heexp1 hfexp hscaleA hρ1).residual M)

-- ════════════════════════════════════════════════════════════════
-- § Projected multi-head: per-head Wq/Wk/Wv (block-diagonal) via indexed perRow
--
-- The `perRow` seam, but with a DIFFERENT per-token map per block — so each head can carry
-- its own projections. Multi-head projected attention is then `perRowIdx` of the single-head
-- `projAttn` (each head's `Wq/Wk/Wv : Mat dh dh` over its own slab), wrapped by the head
-- reshape. Uniform budget across heads (the bound depends on w'/β/A, not the specific weights),
-- exactly what `FloatClose.perRowIdx` needs.
-- ════════════════════════════════════════════════════════════════

/-- `perRow` with a per-block function `g : Fin n → (Vec d → Vec d)` (block hd gets `g hd`). -/
noncomputable def perRowIdxFlat (n d : Nat) (g : Fin n → (Vec d → Vec d)) :
    Vec (n * d) → Vec (n * d) :=
  fun v => Mat.flatten (fun i => g i (Mat.unflatten v i))

theorem perRowIdxFlat_apply {n d : Nat} (g : Fin n → (Vec d → Vec d)) (v : Vec (n * d))
    (idx : Fin (n * d)) :
    perRowIdxFlat n d g v idx
      = g (finProdFinEquiv.symm idx).1 (Mat.unflatten v (finProdFinEquiv.symm idx).1)
          (finProdFinEquiv.symm idx).2 := rfl

/-- **Indexed per-row seam.** If every block's map is `FloatClose A B (g i) (gF i) L` with the
    SAME `A`/`B`/`L`, the indexed per-row map is `FloatClose A B` with that same budget (blocks
    independent). The per-head version of `FloatClose.perRow`. -/
theorem FloatClose.perRowIdx {d : Nat} (n : Nat) {A B : ℝ}
    {g gF : Fin n → (Vec d → Vec d)} {L : ℝ → ℝ}
    (hg : ∀ i, FloatClose A B (g i) (gF i) L) :
    FloatClose A B (perRowIdxFlat n d g) (perRowIdxFlat n d gF) L := by
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · have hrow : ∀ j', |Mat.unflatten v (finProdFinEquiv.symm idx).1 j'| ≤ A :=
      fun j' => hv (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))
    have hboth := (hg (finProdFinEquiv.symm idx).1).1
      (Mat.unflatten v (finProdFinEquiv.symm idx).1) hrow (finProdFinEquiv.symm idx).2
    rw [perRowIdxFlat_apply, perRowIdxFlat_apply]
    exact hboth
  · have hva' : ∀ j', |Mat.unflatten va (finProdFinEquiv.symm idx).1 j'| ≤ A :=
      fun j' => hva (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))
    have hvt' : ∀ j', |Mat.unflatten vt (finProdFinEquiv.symm idx).1 j'| ≤ A :=
      fun j' => hvt (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))
    have hd' : ∀ j', |Mat.unflatten vt (finProdFinEquiv.symm idx).1 j'
                    - Mat.unflatten va (finProdFinEquiv.symm idx).1 j'| ≤ e :=
      fun j' => hd (finProdFinEquiv ((finProdFinEquiv.symm idx).1, j'))
    rw [perRowIdxFlat_apply, perRowIdxFlat_apply]
    exact (hg (finProdFinEquiv.symm idx).1).2 (Mat.unflatten vt (finProdFinEquiv.symm idx).1)
      (Mat.unflatten va (finProdFinEquiv.symm idx).1) e hva' hvt' hd' (finProdFinEquiv.symm idx).2

/-- **Multi-head PROJECTED attention** (per-head Wq/Wk/Wv : `Mat dh dh`, block-diagonal). Reshape
    to head-major, apply the single-head projected attention `projAttn` with head hd's own
    weights to each head-block, reshape back. The `h=1` case is `projAttnFlat`. -/
noncomputable def mhProjAttnFlat (h n dh : Nat) (Wq Wk Wv : Fin h → Mat dh dh)
    (bq bk bv : Fin h → Vec dh) : Vec (n * (h * dh)) → Vec (n * (h * dh)) :=
  gather (mhReshape h n dh).symm
    ∘ perRowIdxFlat h (n * dh)
        (fun hd => projAttnFlat (n := n) (Wq hd) (Wk hd) (Wv hd) (bq hd) (bk hd) (bv hd))
    ∘ gather (mhReshape h n dh)

/-- **Multi-head projected attention float-bridges.** `FloatClose.comp` over reshape /
    `FloatClose.perRowIdx` of the per-head `floatClose_projAttn` (uniform budget `projAttnB`
    at head dim `dh`) / reshape-back. Genuine per-head learned projections, multi-head. -/
theorem floatBridges_mhProjAttn (M : FloatModel) (fexp : ℝ → ℝ) {h n dh : Nat}
    (Wq Wk Wv : Fin h → Mat dh dh) (bq bk bv : Fin h → Vec dh) {w' β scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β)
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (dh : ℝ)| ≤ scaleA) (hρ1 : smRho M.u eexp n < 1)
    (hWq : ∀ hd i j, |Wq hd i j| ≤ w') (hbq : ∀ hd j, |bq hd j| ≤ β)
    (hWk : ∀ hd i j, |Wk hd i j| ≤ w') (hbk : ∀ hd j, |bk hd j| ≤ β)
    (hWv : ∀ hd i j, |Wv hd i j| ≤ w') (hbv : ∀ hd j, |bv hd j| ≤ β) :
    FloatBridges (mhProjAttnFlat h n dh Wq Wk Wv bq bk bv) := by
  intro A hA
  have hLa0 : 0 ≤ layerAct dh w' β A := layerAct_nonneg hw' hβ hA
  have hLb00 : 0 ≤ layerBudget M.u dh w' β A 0 := layerBudget_nonneg M.u_nonneg hw' hβ hA le_rfl
  have hqAF0 : 0 ≤ layerAct dh w' β A + layerBudget M.u dh w' β A 0 := by linarith
  have hscaleA0 : 0 ≤ scaleA := (abs_nonneg _).trans hscaleA
  have hnn : 0 ≤ projAttnB M n dh w' β A scaleA eexp := by
    unfold projAttnB
    exact add_nonneg hqAF0 (M.attnOutErr_nonneg n dh hqAF0 hqAF0 hqAF0 hscaleA0 heexp0 hρ1)
  have hfc := (floatClose_gather (mhReshape h n dh) A).comp
    ((FloatClose.perRowIdx h (fun hd =>
        floatClose_projAttn M fexp (Wq hd) (Wk hd) (Wv hd) (bq hd) (bk hd) (bv hd)
          hn hw' hβ hA heexp0 heexp1 hfexp hscaleA hρ1
          (hWq hd) (hbq hd) (hWk hd) (hbk hd) (hWv hd) (hbv hd))).comp
      (floatClose_gather (mhReshape h n dh).symm (projAttnB M n dh w' β A scaleA eexp)))
  exact ⟨projAttnB M n dh w' β A scaleA eexp, _, _, hnn, hfc⟩

/-- **THE PROJECTED-MULTI-HEAD ViT BLOCK** (per-head Wq/Wk/Wv, shared output Wo). The block
    with multi-head projected MHSA float-bridges unconditionally — `hattn` discharged by
    `floatBridges_mhProjAttn` composed with the per-token output projection `Wo`
    (`floatBridges_dense.perRow`), wrapped `residual`. This combines all three extensions —
    projections, multi-head reshape, and the unconditional block — into the deployed ViT layer. -/
theorem floatBridges_vitBlockMHProj {h n dh dff : Nat} (M : FloatModel)
    (W₁ : Mat (h * dh) dff) (b₁ : Vec dff) (W₂ : Mat dff (h * dh)) (b₂ : Vec (h * dh))
    (Wq Wk Wv : Fin h → Mat dh dh) (bq bk bv : Fin h → Vec dh)
    (Wo : Mat (h * dh) (h * dh)) (bo : Vec (h * dh)) (fgelu fexp : ℝ → ℝ)
    {εln γln βln w' β egelu scaleA eexp : ℝ}
    (hn : 0 < n) (hw' : 0 ≤ w') (hβ : 0 ≤ β) (hegelu : 0 ≤ egelu)
    (hd : 0 < h * dh) (hdff : 0 < dff)
    (hg : ∀ t, |fgelu t - geluScalar t| ≤ egelu)
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hb₁ : ∀ j, |b₁ j| ≤ β)
    (hW₂ : ∀ i j, |W₂ i j| ≤ w') (hb₂ : ∀ j, |b₂ j| ≤ β)
    (hWq : ∀ hd' i j, |Wq hd' i j| ≤ w') (hbq : ∀ hd' j, |bq hd' j| ≤ β)
    (hWk : ∀ hd' i j, |Wk hd' i j| ≤ w') (hbk : ∀ hd' j, |bk hd' j| ≤ β)
    (hWv : ∀ hd' i j, |Wv hd' i j| ≤ w') (hbv : ∀ hd' j, |bv hd' j| ≤ β)
    (hWo : ∀ i j, |Wo i j| ≤ w') (hbo : ∀ j, |bo j| ≤ β)
    (hln : FloatBridges (layerNormForward (h * dh) εln γln βln))
    (heexp0 : 0 ≤ eexp) (heexp1 : eexp ≤ 1)
    (hfexp : ∀ t, |fexp t - Real.exp t| ≤ eexp * Real.exp t)
    (hscaleA : |(1 : ℝ) / Real.sqrt (dh : ℝ)| ≤ scaleA)
    (hρ1 : smRho M.u eexp n < 1) :
    FloatBridges
      (perRowFlat n (h * dh) (Proofs.residual
          (Proofs.dense W₂ b₂ ∘ gelu dff ∘ Proofs.dense W₁ b₁
            ∘ layerNormForward (h * dh) εln γln βln))
        ∘ Proofs.residual (perRowFlat n (h * dh) (Proofs.dense Wo bo)
            ∘ mhProjAttnFlat h n dh Wq Wk Wv bq bk bv)) :=
  floatBridges_vitBlock M W₁ b₁ W₂ b₂ fgelu hw' hβ hegelu hd hdff hg hW₁ hb₁ hW₂ hb₂ hln
    ((FloatBridges.comp
      (floatBridges_mhProjAttn M fexp Wq Wk Wv bq bk bv hn hw' hβ heexp0 heexp1 hfexp hscaleA hρ1
        hWq hbq hWk hbk hWv hbv)
      (FloatBridges.perRow n (floatBridges_dense M Wo bo hw' hβ hd hWo hbo))).residual M)

end Proofs
