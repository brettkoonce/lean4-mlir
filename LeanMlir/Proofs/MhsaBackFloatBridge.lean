import LeanMlir.Proofs.SdpaBackFloatBridge
import LeanMlir.Proofs.ViTBlockFloatBridge
import LeanMlir.Proofs.LinBackFloatBridge
import LeanMlir.Proofs.Resnet34DownBackFloatBridge

/-! # ℝ→Float32 bridge: the FULL multi-head self-attention backward (A3 §1f — vit assembly)

A3 (planning/a3_backward_deepnet_assembly.md §1f): assemble the whole MHSA-sublayer input-gradient
backward as a composable `FloatBridges`, the backward peer of the forward `floatBridges_mhProjAttnFull`.
The certified MHSA backward at the input `X` (Attention.lean, `mhsa_layer`) is

  dconcat = dY · Woᵀ                                 -- output-projection backward (per-token dense)
  (dQ, dK, dV) = per-head sdpa_back(dconcat)         -- the multi-head sdpa core (SdpaBackFloatBridge)
  dX = dQ · Wqᵀ + dK · Wkᵀ + dV · Wvᵀ               -- Q/K/V projection backwards, three-way fan-in at X

KEY: **the projection backwards are FREE.** `dX_via_Q = dQ · Wqᵀ` is, per token, `linBack Wq`
(`dx = Wᵀ·dy`, `LinBackFloatBridge`); over the sequence it is `FloatBridges.perRow N (floatBridges_linBack Wq)`.
Likewise `Wo`/`Wk`/`Wv`. So the ONLY new work is the three sdpa cores as `FloatBridges`
(`floatBridges_core{V,Q,K}`): the flattened multi-head sdpa backward (`SdpaBackFloatBridge.mhsaSdpaBack*`),
lifted to a `FloatClose` over the cotangent. Each core is LINEAR in the cotangent (Q/K/V and the saved
softmax weights are fixed at the smooth point), so — like `bn_grad_input`/`softmaxBack`/`sdpaSelf` — its
modulus is the rounding budget (`mhsaSdpaBack*_close`) plus the real magnitude at the input-error `e`
(`mhsaSdpaBack*_{abs,sub}_le`). The `floatClose_sdpaSelf` template, in reverse.

`floatBridges_mhsaBack` then assembles everything with `comp` / `biPathSum` (the three-way fan-in) /
`perRow` — no new analysis, the budgets thread automatically. This is the backward peer of
`floatBridges_vitBlockMHFull`; pair with LN-back + residual + the Vec-space MLP-half backward for the
transformer-block / whole-net fold. A3 = gradient *closeness* at a smooth point (NOT descent).
-/

namespace Proofs

open FloatModel

variable {h N dh : Nat}

-- ════════════════════════════════════════════════════════════════
-- § The flattened multi-head sdpa cores (cotangent ↦ dV / dQ / dK)
-- ════════════════════════════════════════════════════════════════

/-- Flattened real multi-head sdpa backward w.r.t. V (saved projections `Q K V` fixed). -/
noncomputable def coreVFlat (Q K V : Mat N (h * dh)) (v : Vec (N * (h * dh))) : Vec (N * (h * dh)) :=
  Mat.flatten (mhsaSdpaBackV Q K V (Mat.unflatten v))

/-- Flattened real multi-head sdpa backward w.r.t. Q. -/
noncomputable def coreQFlat (Q K V : Mat N (h * dh)) (v : Vec (N * (h * dh))) : Vec (N * (h * dh)) :=
  Mat.flatten (mhsaSdpaBackQ Q K V (Mat.unflatten v))

/-- Flattened real multi-head sdpa backward w.r.t. K. -/
noncomputable def coreKFlat (Q K V : Mat N (h * dh)) (v : Vec (N * (h * dh))) : Vec (N * (h * dh)) :=
  Mat.flatten (mhsaSdpaBackK Q K V (Mat.unflatten v))

/-- Flattened float multi-head sdpa backward w.r.t. V (saved float weights `fp`). -/
noncomputable def FloatModel.coreVFlatF (M : FloatModel) (fp : Fin h → Mat N N)
    (v : Vec (N * (h * dh))) : Vec (N * (h * dh)) :=
  Mat.flatten (M.mhsaSdpaBackVF fp (Mat.unflatten v))

/-- Flattened float multi-head sdpa backward w.r.t. Q. -/
noncomputable def FloatModel.coreQFlatF (M : FloatModel) (fp : Fin h → Mat N N) (K V : Mat N (h * dh))
    (v : Vec (N * (h * dh))) : Vec (N * (h * dh)) :=
  Mat.flatten (M.mhsaSdpaBackQF fp K V (Mat.unflatten v))

/-- Flattened float multi-head sdpa backward w.r.t. K. -/
noncomputable def FloatModel.coreKFlatF (M : FloatModel) (fp : Fin h → Mat N N) (Q V : Mat N (h * dh))
    (v : Vec (N * (h * dh))) : Vec (N * (h * dh)) :=
  Mat.flatten (M.mhsaSdpaBackKF fp Q V (Mat.unflatten v))

-- ════════════════════════════════════════════════════════════════
-- § The cores are FloatClose / FloatBridges (linear-in-cotangent lift)
-- ════════════════════════════════════════════════════════════════

/-- **The V-core is `FloatClose`.** `dV = pᵀ·dconcat`, linear in the cotangent: magnitude `N·A`
    (`|p| ≤ 1`), modulus = the rounding `sdpaBackErr` (`mhsaSdpaBackV_close`) + the real magnitude at
    the input-error `e` (`mhsaSdpaBackV_sub_abs_le`). The reverse of `floatClose_sdpaSelf`. -/
theorem floatClose_coreV (M : FloatModel) (Q K V : Mat N (h * dh)) (fp : Fin h → Mat N N)
    {A ew : ℝ} (hA : 0 ≤ A) (hew : 0 ≤ ew)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew) :
    FloatClose A ((N : ℝ) * A + M.sdpaBackErr N 1 ew A)
      (coreVFlat Q K V) (M.coreVFlatF fp)
      (fun e => M.sdpaBackErr N 1 ew A + (N : ℝ) * e) := by
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · have hdOut : ∀ a b, |Mat.unflatten v a b| ≤ A := fun a b => hv (finProdFinEquiv (a, b))
    have hreal : |mhsaSdpaBackV Q K V (Mat.unflatten v)
        (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| ≤ (N : ℝ) * A :=
      mhsaSdpaBackV_abs_le Q K V (Mat.unflatten v) hdOut _ _
    have hround : |M.mhsaSdpaBackVF fp (Mat.unflatten v)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
        - mhsaSdpaBackV Q K V (Mat.unflatten v)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| ≤ M.sdpaBackErr N 1 ew A :=
      mhsaSdpaBackV_close M Q K V (Mat.unflatten v) fp hA hew hdOut hfp _ _
    have hB0 : 0 ≤ M.sdpaBackErr N 1 ew A := (abs_nonneg _).trans hround
    refine ⟨le_trans hreal (by linarith), ?_⟩
    calc |M.coreVFlatF fp v idx|
        = |M.mhsaSdpaBackVF fp (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := rfl
      _ ≤ |M.mhsaSdpaBackVF fp (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackV Q K V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |mhsaSdpaBackV Q K V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := by
          simpa using abs_sub_le (M.mhsaSdpaBackVF fp (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2)
            (mhsaSdpaBackV Q K V (Mat.unflatten v)
              (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2) 0
      _ ≤ M.sdpaBackErr N 1 ew A + (N : ℝ) * A := add_le_add hround hreal
      _ = (N : ℝ) * A + M.sdpaBackErr N 1 ew A := by ring
  · have hdOutt : ∀ a b, |Mat.unflatten vt a b| ≤ A := fun a b => hvt (finProdFinEquiv (a, b))
    have hde : ∀ a b, |Mat.unflatten vt a b - Mat.unflatten va a b| ≤ e :=
      fun a b => hd (finProdFinEquiv (a, b))
    have hround : |M.mhsaSdpaBackVF fp (Mat.unflatten vt)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
        - mhsaSdpaBackV Q K V (Mat.unflatten vt)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| ≤ M.sdpaBackErr N 1 ew A :=
      mhsaSdpaBackV_close M Q K V (Mat.unflatten vt) fp hA hew hdOutt hfp _ _
    have hsens : |mhsaSdpaBackV Q K V (Mat.unflatten vt)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
        - mhsaSdpaBackV Q K V (Mat.unflatten va)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| ≤ (N : ℝ) * e :=
      mhsaSdpaBackV_sub_abs_le Q K V (Mat.unflatten vt) (Mat.unflatten va) hde _ _
    calc |M.coreVFlatF fp vt idx - coreVFlat Q K V va idx|
        = |M.mhsaSdpaBackVF fp (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackV Q K V (Mat.unflatten va)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := rfl
      _ ≤ |M.mhsaSdpaBackVF fp (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackV Q K V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |mhsaSdpaBackV Q K V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackV Q K V (Mat.unflatten va)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := abs_sub_le _ _ _
      _ ≤ M.sdpaBackErr N 1 ew A + (N : ℝ) * e := add_le_add hround hsens

/-- **The Q-core is `FloatClose`** (head dim `dh`, scale `1/√dh`): magnitude `N·sdpaDScoresMag(A)·kA`,
    modulus the `sdpaBackErr` rounding + the real magnitude at `e`. -/
theorem floatClose_coreQ (M : FloatModel) (Q K V : Mat N (h * dh)) (fp : Fin h → Mat N N)
    {A kA vA scaleA ew : ℝ} (hA : 0 ≤ A) (hkA : 0 ≤ kA) (_hew : 0 ≤ ew)
    (hscaleA : |sdpa_scale dh| ≤ scaleA) (hK : ∀ i k, |K i k| ≤ kA) (hV : ∀ i k, |V i k| ≤ vA)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew) :
    FloatClose A
      ((N : ℝ) * sdpaDScoresMag N dh A vA scaleA * kA
        + M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) kA)
      (coreQFlat Q K V) (M.coreQFlatF fp K V)
      (fun e => M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) kA
        + (N : ℝ) * sdpaDScoresMag N dh e vA scaleA * kA) := by
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · have hdOut : ∀ a b, |Mat.unflatten v a b| ≤ A := fun a b => hv (finProdFinEquiv (a, b))
    have hreal : |mhsaSdpaBackQ Q K V (Mat.unflatten v)
        (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ (N : ℝ) * sdpaDScoresMag N dh A vA scaleA * kA :=
      mhsaSdpaBackQ_abs_le Q K V (Mat.unflatten v) hA hscaleA hK hV hdOut _ _
    have hround : |M.mhsaSdpaBackQF fp K V (Mat.unflatten v)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
        - mhsaSdpaBackQ Q K V (Mat.unflatten v)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) kA :=
      mhsaSdpaBackQ_close M Q K V (Mat.unflatten v) fp hkA hA hscaleA hK hV hdOut hfp _ _
    have hB0 : 0 ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA)
        (M.sdpaDScoresErr N dh A vA scaleA ew) kA := (abs_nonneg _).trans hround
    refine ⟨le_trans hreal (by linarith), ?_⟩
    calc |M.coreQFlatF fp K V v idx|
        = |M.mhsaSdpaBackQF fp K V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := rfl
      _ ≤ |M.mhsaSdpaBackQF fp K V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackQ Q K V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |mhsaSdpaBackQ Q K V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := by
          simpa using abs_sub_le (M.mhsaSdpaBackQF fp K V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2)
            (mhsaSdpaBackQ Q K V (Mat.unflatten v)
              (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2) 0
      _ ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) kA
          + (N : ℝ) * sdpaDScoresMag N dh A vA scaleA * kA := add_le_add hround hreal
      _ = (N : ℝ) * sdpaDScoresMag N dh A vA scaleA * kA
          + M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA)
            (M.sdpaDScoresErr N dh A vA scaleA ew) kA := by ring
  · have hdOutt : ∀ a b, |Mat.unflatten vt a b| ≤ A := fun a b => hvt (finProdFinEquiv (a, b))
    have hde : ∀ a b, |Mat.unflatten vt a b - Mat.unflatten va a b| ≤ e :=
      fun a b => hd (finProdFinEquiv (a, b))
    have hround : |M.mhsaSdpaBackQF fp K V (Mat.unflatten vt)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
        - mhsaSdpaBackQ Q K V (Mat.unflatten vt)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) kA :=
      mhsaSdpaBackQ_close M Q K V (Mat.unflatten vt) fp hkA hA hscaleA hK hV hdOutt hfp _ _
    have hsens : |mhsaSdpaBackQ Q K V (Mat.unflatten vt)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
        - mhsaSdpaBackQ Q K V (Mat.unflatten va)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ (N : ℝ) * sdpaDScoresMag N dh e vA scaleA * kA :=
      mhsaSdpaBackQ_sub_abs_le Q K V (Mat.unflatten vt) (Mat.unflatten va) hscaleA hK hV hde _ _
    calc |M.coreQFlatF fp K V vt idx - coreQFlat Q K V va idx|
        = |M.mhsaSdpaBackQF fp K V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackQ Q K V (Mat.unflatten va)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := rfl
      _ ≤ |M.mhsaSdpaBackQF fp K V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackQ Q K V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |mhsaSdpaBackQ Q K V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackQ Q K V (Mat.unflatten va)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := abs_sub_le _ _ _
      _ ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) kA
          + (N : ℝ) * sdpaDScoresMag N dh e vA scaleA * kA := add_le_add hround hsens

/-- **The K-core is `FloatClose`** (the transpose path; magnitude `N·sdpaDScoresMag(A)·qA`). -/
theorem floatClose_coreK (M : FloatModel) (Q K V : Mat N (h * dh)) (fp : Fin h → Mat N N)
    {A qA vA scaleA ew : ℝ} (hA : 0 ≤ A) (hqA : 0 ≤ qA) (_hew : 0 ≤ ew)
    (hscaleA : |sdpa_scale dh| ≤ scaleA) (hQ : ∀ i k, |Q i k| ≤ qA) (hV : ∀ i k, |V i k| ≤ vA)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew) :
    FloatClose A
      ((N : ℝ) * sdpaDScoresMag N dh A vA scaleA * qA
        + M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) qA)
      (coreKFlat Q K V) (M.coreKFlatF fp Q V)
      (fun e => M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) qA
        + (N : ℝ) * sdpaDScoresMag N dh e vA scaleA * qA) := by
  refine ⟨fun v hv idx => ?_, fun vt va e hva hvt hd idx => ?_⟩
  · have hdOut : ∀ a b, |Mat.unflatten v a b| ≤ A := fun a b => hv (finProdFinEquiv (a, b))
    have hreal : |mhsaSdpaBackK Q K V (Mat.unflatten v)
        (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ (N : ℝ) * sdpaDScoresMag N dh A vA scaleA * qA :=
      mhsaSdpaBackK_abs_le Q K V (Mat.unflatten v) hA hscaleA hQ hV hdOut _ _
    have hround : |M.mhsaSdpaBackKF fp Q V (Mat.unflatten v)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
        - mhsaSdpaBackK Q K V (Mat.unflatten v)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) qA :=
      mhsaSdpaBackK_close M Q K V (Mat.unflatten v) fp hqA hA hscaleA hQ hV hdOut hfp _ _
    have hB0 : 0 ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA)
        (M.sdpaDScoresErr N dh A vA scaleA ew) qA := (abs_nonneg _).trans hround
    refine ⟨le_trans hreal (by linarith), ?_⟩
    calc |M.coreKFlatF fp Q V v idx|
        = |M.mhsaSdpaBackKF fp Q V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := rfl
      _ ≤ |M.mhsaSdpaBackKF fp Q V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackK Q K V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |mhsaSdpaBackK Q K V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := by
          simpa using abs_sub_le (M.mhsaSdpaBackKF fp Q V (Mat.unflatten v)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2)
            (mhsaSdpaBackK Q K V (Mat.unflatten v)
              (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2) 0
      _ ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) qA
          + (N : ℝ) * sdpaDScoresMag N dh A vA scaleA * qA := add_le_add hround hreal
      _ = (N : ℝ) * sdpaDScoresMag N dh A vA scaleA * qA
          + M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA)
            (M.sdpaDScoresErr N dh A vA scaleA ew) qA := by ring
  · have hdOutt : ∀ a b, |Mat.unflatten vt a b| ≤ A := fun a b => hvt (finProdFinEquiv (a, b))
    have hde : ∀ a b, |Mat.unflatten vt a b - Mat.unflatten va a b| ≤ e :=
      fun a b => hd (finProdFinEquiv (a, b))
    have hround : |M.mhsaSdpaBackKF fp Q V (Mat.unflatten vt)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
        - mhsaSdpaBackK Q K V (Mat.unflatten vt)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) qA :=
      mhsaSdpaBackK_close M Q K V (Mat.unflatten vt) fp hqA hA hscaleA hQ hV hdOutt hfp _ _
    have hsens : |mhsaSdpaBackK Q K V (Mat.unflatten vt)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
        - mhsaSdpaBackK Q K V (Mat.unflatten va)
          (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
        ≤ (N : ℝ) * sdpaDScoresMag N dh e vA scaleA * qA :=
      mhsaSdpaBackK_sub_abs_le Q K V (Mat.unflatten vt) (Mat.unflatten va) hscaleA hQ hV hde _ _
    calc |M.coreKFlatF fp Q V vt idx - coreKFlat Q K V va idx|
        = |M.mhsaSdpaBackKF fp Q V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackK Q K V (Mat.unflatten va)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := rfl
      _ ≤ |M.mhsaSdpaBackKF fp Q V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackK Q K V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2|
          + |mhsaSdpaBackK Q K V (Mat.unflatten vt)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2
          - mhsaSdpaBackK Q K V (Mat.unflatten va)
            (finProdFinEquiv.symm idx).1 (finProdFinEquiv.symm idx).2| := abs_sub_le _ _ _
      _ ≤ M.sdpaBackErr N (sdpaDScoresMag N dh A vA scaleA) (M.sdpaDScoresErr N dh A vA scaleA ew) qA
          + (N : ℝ) * sdpaDScoresMag N dh e vA scaleA * qA := add_le_add hround hsens

/-- The V-core float-bridges. -/
theorem floatBridges_coreV (M : FloatModel) (Q K V : Mat N (h * dh)) (fp : Fin h → Mat N N)
    {ew : ℝ} (hN : 0 < N) (hhd : 0 < h * dh) (hew : 0 ≤ ew)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew) :
    FloatBridges (coreVFlat Q K V) := by
  intro A hA
  have hcod : 0 < N * (h * dh) := Nat.mul_pos hN hhd
  exact ⟨_, _, _, (floatClose_coreV M Q K V fp hA hew hfp).cod_nonneg hA hcod,
    floatClose_coreV M Q K V fp hA hew hfp⟩

/-- The Q-core float-bridges. -/
theorem floatBridges_coreQ (M : FloatModel) (Q K V : Mat N (h * dh)) (fp : Fin h → Mat N N)
    {kA vA scaleA ew : ℝ} (hN : 0 < N) (hhd : 0 < h * dh) (hkA : 0 ≤ kA) (hew : 0 ≤ ew)
    (hscaleA : |sdpa_scale dh| ≤ scaleA) (hK : ∀ i k, |K i k| ≤ kA) (hV : ∀ i k, |V i k| ≤ vA)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew) :
    FloatBridges (coreQFlat Q K V) := by
  intro A hA
  have hcod : 0 < N * (h * dh) := Nat.mul_pos hN hhd
  exact ⟨_, _, _, (floatClose_coreQ M Q K V fp hA hkA hew hscaleA hK hV hfp).cod_nonneg hA hcod,
    floatClose_coreQ M Q K V fp hA hkA hew hscaleA hK hV hfp⟩

/-- The K-core float-bridges. -/
theorem floatBridges_coreK (M : FloatModel) (Q K V : Mat N (h * dh)) (fp : Fin h → Mat N N)
    {qA vA scaleA ew : ℝ} (hN : 0 < N) (hhd : 0 < h * dh) (hqA : 0 ≤ qA) (hew : 0 ≤ ew)
    (hscaleA : |sdpa_scale dh| ≤ scaleA) (hQ : ∀ i k, |Q i k| ≤ qA) (hV : ∀ i k, |V i k| ≤ vA)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew) :
    FloatBridges (coreKFlat Q K V) := by
  intro A hA
  have hcod : 0 < N * (h * dh) := Nat.mul_pos hN hhd
  exact ⟨_, _, _, (floatClose_coreK M Q K V fp hA hqA hew hscaleA hQ hV hfp).cod_nonneg hA hcod,
    floatClose_coreK M Q K V fp hA hqA hew hscaleA hQ hV hfp⟩

-- ════════════════════════════════════════════════════════════════
-- § The full MHSA backward (cotangent dY ↦ input gradient dX)
-- ════════════════════════════════════════════════════════════════

/-- **The full multi-head self-attention input-gradient backward** (cotangent `dY ↦ dX`):
    output-projection backward (`linBack Wo`, per token) → the three sdpa cores → Q/K/V projection
    backwards (`linBack Wq/Wk/Wv`, per token), fanning in at `X` (the three paths add). -/
noncomputable def mhsaBackFlat (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (Q K V : Mat N (h * dh)) :
    Vec (N * (h * dh)) → Vec (N * (h * dh)) :=
  (fun dconcat j =>
      (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wq) (0 : Vec (h * dh))) ∘ coreQFlat Q K V)
        dconcat j
      + ((perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wk) (0 : Vec (h * dh))) ∘ coreKFlat Q K V)
          dconcat j
        + (perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wv) (0 : Vec (h * dh))) ∘ coreVFlat Q K V)
          dconcat j))
    ∘ perRowFlat N (h * dh) (Proofs.dense (Mat.transpose Wo) (0 : Vec (h * dh)))

/-- **THE FULL MHSA BACKWARD FLOAT-BRIDGES.** One `FloatBridges.comp` of the output-projection
    backward and the three-way fan-in (`biPathSum` twice) of the `projection-back ∘ sdpa-core` paths.
    The projection backwards are the free `linBack`s lifted per token (`FloatBridges.perRow`); the
    cores are `floatBridges_core{V,Q,K}`. No new analysis — the budgets thread automatically. The
    backward peer of `floatBridges_mhProjAttnFull`; the `hattn` discharger for the backward block. -/
theorem floatBridges_mhsaBack (M : FloatModel) (Wq Wk Wv Wo : Mat (h * dh) (h * dh))
    (Q K V : Mat N (h * dh)) (fp : Fin h → Mat N N)
    {w' qA kA vA scaleA ew : ℝ} (hN : 0 < N) (hhd : 0 < h * dh)
    (hw' : 0 ≤ w') (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) (hew : 0 ≤ ew)
    (hscaleA : |sdpa_scale dh| ≤ scaleA)
    (hQ : ∀ i k, |Q i k| ≤ qA) (hK : ∀ i k, |K i k| ≤ kA) (hV : ∀ i k, |V i k| ≤ vA)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hWk : ∀ i j, |Wk i j| ≤ w')
    (hWv : ∀ i j, |Wv i j| ≤ w') (hWo : ∀ i j, |Wo i j| ≤ w')
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew) :
    FloatBridges (mhsaBackFlat Wq Wk Wv Wo Q K V) := by
  unfold mhsaBackFlat
  exact FloatBridges.comp
    (FloatBridges.perRow N (floatBridges_linBack M Wo hw' hhd hWo))
    (FloatBridges.biPathSum M
      (FloatBridges.comp (floatBridges_coreQ M Q K V fp hN hhd hkA hew hscaleA hK hV hfp)
        (FloatBridges.perRow N (floatBridges_linBack M Wq hw' hhd hWq)))
      (FloatBridges.biPathSum M
        (FloatBridges.comp (floatBridges_coreK M Q K V fp hN hhd hqA hew hscaleA hQ hV hfp)
          (FloatBridges.perRow N (floatBridges_linBack M Wk hw' hhd hWk)))
        (FloatBridges.comp (floatBridges_coreV M Q K V fp hN hhd hew hfp)
          (FloatBridges.perRow N (floatBridges_linBack M Wv hw' hhd hWv)))))

-- ════════════════════════════════════════════════════════════════
-- § The transformer-block backward (the reverse of the ViT encoder block)
-- ════════════════════════════════════════════════════════════════

/-- **The ViT encoder-block input-gradient backward** — the reverse of `LN → MHSA → +x → LN → MLP → +x`.
    The block is `mlpResidual ∘ attnSub` (forward), so the backward is `attnSubBack ∘ mlpResidualBack`:

    * **MLP-residual backward** (per token): `residual (LN₂-back ∘ linBack W₁ ∘ geluBack ∘ linBack W₂)`
      — the reverse of `dense W₂ ∘ gelu ∘ dense W₁ ∘ LN₂`, lifted over the sequence (`perRow`);
    * **attention-sublayer backward**: `residual (LN₁-back ∘ mhsaBack)` — the residual skip's cotangent
      flows both through the MHSA backward (`floatBridges_mhsaBack`) and directly to `x`.

    The LN backwards (`lnB₁`/`lnB₂`) are supplied as `FloatBridges` (= the per-token BatchNorm backward,
    dischargeable by `floatBridges_bnBack`), exactly as the forward `floatBridges_vitBlock` supplies `hln`;
    `geluBack` is the saved-derivative `diagBack`. -/
noncomputable def vitBlockBack {dff : Nat} (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (Q K V : Mat N (h * dh))
    (lnB₁ : Vec (h * dh) → Vec (h * dh)) (W₁ : Mat (h * dh) dff) (W₂ : Mat dff (h * dh))
    (sgelu : Vec dff) (lnB₂ : Vec (h * dh) → Vec (h * dh)) :
    Vec (N * (h * dh)) → Vec (N * (h * dh)) :=
  Proofs.residual (perRowFlat N (h * dh) lnB₁ ∘ mhsaBackFlat Wq Wk Wv Wo Q K V)
    ∘ perRowFlat N (h * dh) (Proofs.residual
        (lnB₂ ∘ Proofs.dense (Mat.transpose W₁) (0 : Vec (h * dh)) ∘ diagBack sgelu
          ∘ Proofs.dense (Mat.transpose W₂) (0 : Vec dff)))

/-- **THE TRANSFORMER-BLOCK BACKWARD FLOAT-BRIDGES.** One `FloatBridges.comp` of the MLP-residual
    backward (per token) and the attention-sublayer backward (`residual` of `LN₁-back ∘ mhsaBack`).
    Pure assembly — `comp` / `residual` / `perRow` over `floatBridges_mhsaBack`, the free `linBack`s,
    the `geluBack` `diagBack`, and the supplied LN backwards. The backward peer of the forward
    `floatBridges_vitBlock`; a 12-layer encoder backward is `.comp` of this block (the whole-net fold,
    blocks supplied — the `r34_grad_floatBridges` blueprint). A3 = closeness at a smooth point. -/
theorem floatBridges_vitBlockBack {dff : Nat} (M : FloatModel)
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (Q K V : Mat N (h * dh)) (fp : Fin h → Mat N N)
    (lnB₁ : Vec (h * dh) → Vec (h * dh)) (W₁ : Mat (h * dh) dff) (W₂ : Mat dff (h * dh))
    (sgelu fsgelu : Vec dff) (lnB₂ : Vec (h * dh) → Vec (h * dh))
    {w' qA kA vA scaleA ew Sd es : ℝ} (hN : 0 < N) (hhd : 0 < h * dh) (hdff : 0 < dff)
    (hw' : 0 ≤ w') (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) (hew : 0 ≤ ew)
    (hscaleA : |sdpa_scale dh| ≤ scaleA)
    (hQ : ∀ i k, |Q i k| ≤ qA) (hK : ∀ i k, |K i k| ≤ kA) (hV : ∀ i k, |V i k| ≤ vA)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hWk : ∀ i j, |Wk i j| ≤ w')
    (hWv : ∀ i j, |Wv i j| ≤ w') (hWo : ∀ i j, |Wo i j| ≤ w')
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hW₂ : ∀ i j, |W₂ i j| ≤ w')
    (hsgelu : ∀ i, |sgelu i| ≤ Sd) (hfsgelu : ∀ i, |fsgelu i - sgelu i| ≤ es)
    (hlnB₁ : FloatBridges lnB₁) (hlnB₂ : FloatBridges lnB₂)
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew) :
    FloatBridges (vitBlockBack Wq Wk Wv Wo Q K V lnB₁ W₁ W₂ sgelu lnB₂) := by
  unfold vitBlockBack
  exact FloatBridges.comp
    (FloatBridges.perRow N (FloatBridges.residual M
      (FloatBridges.comp
        (FloatBridges.comp
          (FloatBridges.comp (floatBridges_linBack M W₂ hw' hhd hW₂)
            (floatBridges_diagBack M sgelu fsgelu hdff hsgelu hfsgelu))
          (floatBridges_linBack M W₁ hw' hdff hW₁))
        hlnB₂)))
    (FloatBridges.residual M
      (FloatBridges.comp
        (floatBridges_mhsaBack M Wq Wk Wv Wo Q K V fp hN hhd hw' hqA hkA hew hscaleA
          hQ hK hV hWq hWk hWv hWo hfp)
        (FloatBridges.perRow N hlnB₁)))

-- ════════════════════════════════════════════════════════════════
-- § The per-token-input-aware transformer-block backward (the tie-able form)
-- ════════════════════════════════════════════════════════════════

/-- **The per-token-input-aware ViT encoder-block backward** — the enrichment of
    `vitBlockBack` whose LayerNorm and GELU slots thread *each token's* saved activation
    (`perRowFlatPR` instead of `perRowFlat`). Structurally identical to `vitBlockBack`
    (residual MLP-sublayer back, then residual attention-sublayer back), but `lnB₁`/`lnB₂`
    are now per-token *families* `Fin N → (Vec → Vec)` and the GELU derivative `sgelu` is a
    per-token family `Fin N → Vec dff`. This is the form the certified per-token block VJP
    actually takes: `layerNorm_per_token_has_vjp_mat.backward A` runs the single-token LN
    backward at each token's own saved input `A r` (its Jacobian differs per token), which a
    single shared `lnB₁` cannot carry. `vitBlockBack` is the special case of all rows sharing
    one map; this is the general one the §B tie equals. -/
noncomputable def vitBlockBackPR {dff : Nat} (Wq Wk Wv Wo : Mat (h * dh) (h * dh))
    (Q K V : Mat N (h * dh))
    (lnB₁ : Fin N → (Vec (h * dh) → Vec (h * dh))) (W₁ : Mat (h * dh) dff) (W₂ : Mat dff (h * dh))
    (sgelu : Fin N → Vec dff) (lnB₂ : Fin N → (Vec (h * dh) → Vec (h * dh))) :
    Vec (N * (h * dh)) → Vec (N * (h * dh)) :=
  Proofs.residual (perRowFlatPR N (h * dh) lnB₁ ∘ mhsaBackFlat Wq Wk Wv Wo Q K V)
    ∘ perRowFlatPR N (h * dh) (fun r => Proofs.residual
        (lnB₂ r ∘ Proofs.dense (Mat.transpose W₁) (0 : Vec (h * dh)) ∘ diagBack (sgelu r)
          ∘ Proofs.dense (Mat.transpose W₂) (0 : Vec dff)))

/-- **THE PER-TOKEN-AWARE TRANSFORMER-BLOCK BACKWARD FLOAT-BRIDGES.** Same `comp`/`residual`
    assembly as `floatBridges_vitBlockBack`, with the per-token seams `FloatBridges.perRowPR`
    (each token's LN/GELU slot bridges independently — uniform `⊔` magnitude/modulus over the
    `N` rows). The bridge for the certified-tie-able block backward; the §B peer of the
    forward `floatBridges_vitBlock`. A3 = closeness at a smooth point. -/
theorem floatBridges_vitBlockBackPR {dff : Nat} (M : FloatModel)
    (Wq Wk Wv Wo : Mat (h * dh) (h * dh)) (Q K V : Mat N (h * dh)) (fp : Fin h → Mat N N)
    (lnB₁ : Fin N → (Vec (h * dh) → Vec (h * dh))) (W₁ : Mat (h * dh) dff) (W₂ : Mat dff (h * dh))
    (sgelu fsgelu : Fin N → Vec dff) (lnB₂ : Fin N → (Vec (h * dh) → Vec (h * dh)))
    {w' qA kA vA scaleA ew Sd es : ℝ} (hN : 0 < N) (hhd : 0 < h * dh) (hdff : 0 < dff)
    (hw' : 0 ≤ w') (hqA : 0 ≤ qA) (hkA : 0 ≤ kA) (hew : 0 ≤ ew)
    (hscaleA : |sdpa_scale dh| ≤ scaleA)
    (hQ : ∀ i k, |Q i k| ≤ qA) (hK : ∀ i k, |K i k| ≤ kA) (hV : ∀ i k, |V i k| ≤ vA)
    (hWq : ∀ i j, |Wq i j| ≤ w') (hWk : ∀ i j, |Wk i j| ≤ w')
    (hWv : ∀ i j, |Wv i j| ≤ w') (hWo : ∀ i j, |Wo i j| ≤ w')
    (hW₁ : ∀ i j, |W₁ i j| ≤ w') (hW₂ : ∀ i j, |W₂ i j| ≤ w')
    (hsgelu : ∀ r i, |sgelu r i| ≤ Sd) (hfsgelu : ∀ r i, |fsgelu r i - sgelu r i| ≤ es)
    (hlnB₁ : ∀ r, FloatBridges (lnB₁ r)) (hlnB₂ : ∀ r, FloatBridges (lnB₂ r))
    (hfp : ∀ hd i j, |fp hd i j - sdpa_weights N dh (mhSlab hd Q) (mhSlab hd K) i j| ≤ ew) :
    FloatBridges (vitBlockBackPR Wq Wk Wv Wo Q K V lnB₁ W₁ W₂ sgelu lnB₂) := by
  unfold vitBlockBackPR
  exact FloatBridges.comp
    (FloatBridges.perRowPR hN (fun r => FloatBridges.residual M
      (FloatBridges.comp
        (FloatBridges.comp
          (FloatBridges.comp (floatBridges_linBack M W₂ hw' hhd hW₂)
            (floatBridges_diagBack M (sgelu r) (fsgelu r) hdff (hsgelu r) (hfsgelu r)))
          (floatBridges_linBack M W₁ hw' hdff hW₁))
        (hlnB₂ r))))
    (FloatBridges.residual M
      (FloatBridges.comp
        (floatBridges_mhsaBack M Wq Wk Wv Wo Q K V fp hN hhd hw' hqA hkA hew hscaleA
          hQ hK hV hWq hWk hWv hWo hfp)
        (FloatBridges.perRowPR hN hlnB₁)))

-- ════════════════════════════════════════════════════════════════
-- § The encoder-tower backward (depth fold over distinct blocks)
-- ════════════════════════════════════════════════════════════════

/-- Identity float-bridges (the cotangent passes through unchanged — exact, modulus `id`). The base
    case of the tower fold and the positional-embedding `+pos` backward. -/
theorem floatBridges_id {m : Nat} : FloatBridges (id : Vec m → Vec m) :=
  fun A hA => ⟨A, _, _, hA, ⟨fun _v hv i => ⟨hv i, hv i⟩, fun _ _ _ _ _ hd i => hd i⟩⟩

/-- Compose a list of dim-preserving maps: `towerBack [g₁, …, gₖ] = gₖ ∘ … ∘ g₁` (the head is applied
    first). The ViT encoder backward is `towerBack` of the per-layer block backwards. -/
noncomputable def towerBack {m : Nat} : List (Vec m → Vec m) → (Vec m → Vec m)
  | [] => id
  | f :: fs => towerBack fs ∘ f

/-- **THE ENCODER-TOWER BACKWARD FLOAT-BRIDGES** — the whole `k`-layer encoder backward is the `.comp`
    fold of the per-block backwards (each `floatBridges_vitBlockBack`). Blocks have DISTINCT params, so
    this is the explicit list fold (the depth thread), not a uniform iterate — the `r34_grad_floatBridges`
    blueprint, generic in depth. Discharges every transformer tower (12 layers for ViT-Tiny). -/
theorem floatBridges_towerBack {m : Nat} (l : List (Vec m → Vec m))
    (hl : ∀ f ∈ l, FloatBridges f) : FloatBridges (towerBack l) := by
  induction l with
  | nil => exact floatBridges_id
  | cons f fs ih =>
      exact FloatBridges.comp (hl f (List.mem_cons.mpr (Or.inl rfl)))
        (ih (fun g hg => hl g (List.mem_cons.mpr (Or.inr hg))))

-- ════════════════════════════════════════════════════════════════
-- § The endpoints — cls-slice scatter + classifier head + the whole-net fold
-- ════════════════════════════════════════════════════════════════

/-- **The CLS-slice backward** — the adjoint of `cls_slice_flat` (gather row 0 of the `(N+1)×D`
    sequence): scatter the head cotangent `dy` back to row 0 (the CLS token), zero on the patch rows.
    The certified `cls_slice_flat_has_vjp.backward`; exact in float (a structural select/zero). -/
noncomputable def clsScatter (N D : Nat) (dy : Vec D) : Vec ((N + 1) * D) :=
  fun idx =>
    if (finProdFinEquiv.symm idx).1 = (0 : Fin (N + 1)) then dy (finProdFinEquiv.symm idx).2 else 0

/-- The CLS-slice scatter is `FloatClose` with modulus `id` — exact (real = float), magnitude-stable
    (`B = A`): row 0 carries `|dy| ≤ A`, every other row is `0 ≤ A`. -/
theorem floatClose_clsScatter (N D : Nat) {A : ℝ} (hA : 0 ≤ A) :
    FloatClose A A (clsScatter N D) (clsScatter N D) (fun e => e) := by
  refine ⟨fun v hv idx => ⟨?_, ?_⟩, fun vt va e _ _ hd idx => ?_⟩
  · by_cases h : (finProdFinEquiv.symm idx).1 = (0 : Fin (N + 1))
    · simpa only [clsScatter, if_pos h] using hv _
    · simpa only [clsScatter, if_neg h, abs_zero] using hA
  · by_cases h : (finProdFinEquiv.symm idx).1 = (0 : Fin (N + 1))
    · simpa only [clsScatter, if_pos h] using hv _
    · simpa only [clsScatter, if_neg h, abs_zero] using hA
  · by_cases h : (finProdFinEquiv.symm idx).1 = (0 : Fin (N + 1))
    · simpa only [clsScatter, if_pos h] using hd _
    · simp only [clsScatter, if_neg h, sub_zero, abs_zero]
      exact (abs_nonneg _).trans (hd (finProdFinEquiv.symm idx).2)

/-- The CLS-slice scatter float-bridges (magnitude-stable). -/
theorem floatBridges_clsScatter (N D : Nat) : FloatBridges (clsScatter N D) :=
  fun A hA => ⟨A, _, _, hA, floatClose_clsScatter N D hA⟩

/-- **The classifier-head backward** — `classifier_flat = dense Wcls ∘ cls_slice` reverses to
    `clsScatter ∘ linBack Wcls`: the head dense's input VJP (free `linBack`) then the CLS-slice scatter. -/
theorem floatBridges_vitHeadBack {N D nClasses : Nat} (M : FloatModel) (Wcls : Mat D nClasses)
    {w' : ℝ} (hw' : 0 ≤ w') (hnc : 0 < nClasses) (hWcls : ∀ i j, |Wcls i j| ≤ w') :
    FloatBridges (clsScatter N D ∘ Proofs.dense (Mat.transpose Wcls) (0 : Vec D)) :=
  FloatBridges.comp (floatBridges_linBack M Wcls hw' hnc hWcls) (floatBridges_clsScatter N D)

/-- **The whole ViT input-gradient backward** — the reverse of `vit_full = classifier ∘ (finalLN ∘
    tower) ∘ patchEmbed`: `patchEmbedBack ∘ towerBack ∘ finalLN-back ∘ (clsScatter ∘ linBack Wcls)`.
    The head + CLS-slice fold concretely; the encoder tower is `floatBridges_towerBack` over the
    per-block backwards; the patch-embed backward, the final LN backward, and the per-block backwards
    are supplied as `FloatBridges` (each separately dischargeable — `patchEmbedBack` by the
    strided-conv VJP machinery, the LNs by `bnBack`, the blocks by `floatBridges_vitBlockBack`),
    exactly as `r34_grad_floatBridges` supplies its 16 blocks around concrete endpoints. -/
noncomputable def vitGradFlat {N D nClasses imgDim : Nat} (Wcls : Mat D nClasses)
    (finalLNBack : Vec D → Vec D) (blockBacks : List (Vec ((N + 1) * D) → Vec ((N + 1) * D)))
    (patchEmbedBack : Vec ((N + 1) * D) → Vec imgDim) : Vec nClasses → Vec imgDim :=
  patchEmbedBack ∘ towerBack blockBacks ∘ perRowFlat (N + 1) D finalLNBack
    ∘ clsScatter N D ∘ Proofs.dense (Mat.transpose Wcls) (0 : Vec D)

/-- **THE WHOLE-NET ViT BACKWARD FLOAT-BRIDGES.** One `.comp` thread over the concrete head/CLS-slice
    endpoints, the encoder tower (`floatBridges_towerBack`), the supplied final-LN backward, and the
    supplied patch-embed backward — the `r34_grad_floatBridges` blueprint for ViT. -/
theorem vit_grad_floatBridges {N D nClasses imgDim : Nat} (M : FloatModel) (Wcls : Mat D nClasses)
    (finalLNBack : Vec D → Vec D) (blockBacks : List (Vec ((N + 1) * D) → Vec ((N + 1) * D)))
    (patchEmbedBack : Vec ((N + 1) * D) → Vec imgDim)
    {w' : ℝ} (hw' : 0 ≤ w') (hnc : 0 < nClasses) (hWcls : ∀ i j, |Wcls i j| ≤ w')
    (hFinalLN : FloatBridges finalLNBack) (hblocks : ∀ f ∈ blockBacks, FloatBridges f)
    (hPatch : FloatBridges patchEmbedBack) :
    FloatBridges (vitGradFlat Wcls finalLNBack blockBacks patchEmbedBack) := by
  unfold vitGradFlat
  exact FloatBridges.comp
    (FloatBridges.comp
      (FloatBridges.comp
        (FloatBridges.comp (floatBridges_linBack M Wcls hw' hnc hWcls) (floatBridges_clsScatter N D))
        (FloatBridges.perRow (N + 1) hFinalLN))
      (floatBridges_towerBack blockBacks hblocks))
    hPatch

end Proofs
