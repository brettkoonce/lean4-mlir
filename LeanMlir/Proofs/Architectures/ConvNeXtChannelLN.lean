import LeanMlir.Proofs.Architectures.ConvNeXtChainClose
import LeanMlir.Proofs.Architectures.ViTVecLN
import LeanMlir.Proofs.Codegen.StableHLO

/-! # ConvNeXt's REAL channel LayerNorm — the math side of §2m Route A

`convnextVerified`'s LN was `bnForward` over the whole flattened `c·h·w` map with a **scalar**
γ/β. ConvNeXt specifies `channel_layer_norm`: `h·w` statistics per example, each over the `c`
channels at ONE spatial position, with a per-channel `[c]` affine. That is a different function
on 21 of the net's 22 sites (the 22nd is the head, which runs after GAP where there is no
spatial extent left, so reducing "everything" already IS reducing over channels).

**Route A — no new `SHlo` op, and no new VJP.** ConvNeXt's channel-LN *is* ViT's row-LN under a
transpose: view one example as `[c, s]` with `s = h·w`, transpose to `[s, c]`, and each row is
one spatial position holding its `c` channels — exactly what ViT's `layerNormVec` normalises.
Every piece below is already proven and shipping:

| piece | from |
|---|---|
| `reassocFwd`/`reassocBack` + VJPs | `PerChannelBN.lean` (the per-channel BN layout bridge) |
| `transpose_has_vjp` | `Tensor.lean` |
| `layerNormVec` + `layerNormVec_per_token_has_vjp_mat` | `ViTVecLN.lean` (ViT's `[192]` LN) |
| `hasVJPMat_to_hasVJP` | `Tensor.lean` |

Settled on device before any of this was written (`lake build channel-ln`): the composition ties
the closed form at rel 0 forward and on all three backward pieces, the incumbent `.bnF` control
fires at rel 0.82, and the transposes measure free (Δ 0.00 ms on 16.1 ms of whole-net LN).

## ⚠ The seam this file closes

`Nat` multiplication is not definitionally associative: the ambient activation index is
`c*h*w = (c*h)*w` while the transpose needs `c*(h*w)`. The **render** spells that with a `▸`
transport (`ConvNeXtRender.reassoc`); the **math** spells it with `PerChannelBN`'s
`finProdFinEquiv` re-association, whose "row `c` is channel `c`" reading is what makes the
composition legibly a *channel* LN. Nothing forces those two to be the same map, and if they are
not, the math and the artifact are different functions with no gate between them — §2k's own sin
in a new place.

They ARE the same map, and `reassocFwdIdx_val` proves it: row-major `finProdFinEquiv` sends both
`((c,hi),wi)` and `(c,(hi,wi))` to the same linear offset, so the bridge preserves the underlying
natural and is therefore exactly the type-level cast. `den_reassocS` lifts that to the graph.
-/

namespace Proofs

open scoped BigOperators
open Proofs.StableHLO (transposeFlat)

-- ════════════════════════════════════════════════════════════════
-- § The two reindexes are ONE map (the seam)
-- ════════════════════════════════════════════════════════════════

/-- **The Mat-split bridge is the `Nat.mul_assoc` cast.** `finProdFinEquiv` is row-major, so
    `((c,hi),wi) ↦ wi + w·hi + w·h·c` and `(c,(hi,wi)) ↦ wi + w·hi + h·w·c` are the same offset;
    the re-association therefore preserves `Fin.val`. This is what lets the proof-side graph
    transport its index with `▸` while the denotation stays on `reassocFwd`. -/
theorem reassocFwdIdx_val (oc h w : Nat) (k : Fin (oc * (h * w))) :
    (reassocFwdIdx oc h w k).val = k.val := by
  unfold reassocFwdIdx
  simp [finProdFinEquiv]
  generalize (k : Nat) = K
  rw [Nat.mul_add, ← Nat.mul_assoc, Nat.mul_comm w h]
  have hdvd : w ∣ h * w := Dvd.intro_left h rfl
  have h1 := Nat.div_add_mod K (h * w)
  have h2 := Nat.div_add_mod (K % (h * w)) w
  have h3 : K % (h * w) % w = K % w := Nat.mod_mod_of_dvd _ hdvd
  omega

/-- The inverse direction, from `reassocFwdIdx_val` through the round-trip. -/
theorem reassocBackIdx_val (oc h w : Nat) (k : Fin (oc * h * w)) :
    (reassocBackIdx oc h w k).val = k.val := by
  have h1 := reassocFwdIdx_val oc h w (reassocBackIdx oc h w k)
  rw [reassocFwdIdx_reassocBackIdx] at h1
  omega

-- ════════════════════════════════════════════════════════════════
-- § The rowwise vector-LN, and the transpose, as `Vec → Vec` with VJPs
-- ════════════════════════════════════════════════════════════════

/-- **Rowwise vector-LN on the flat `[s, c]` layout** — `s` spatial rows, each normalised over
    its `c` channels and then given the per-channel affine. Literally ViT's per-token LN with
    "token" read as "spatial position"; that re-reading is the whole of Route A. -/
noncomputable def rowLNVecFlat (s c : Nat) (ε : ℝ) (γ β : Vec c) :
    Vec (s * c) → Vec (s * c) :=
  fun v => Mat.flatten ((fun X : Mat s c => fun r => layerNormVec c ε γ β (X r))
                          (Mat.unflatten v))

theorem rowLNVecFlat_diff (s c : Nat) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε) :
    Differentiable ℝ (rowLNVecFlat s c ε γ β) :=
  layerNormVec_per_token_flat_diff s c ε γ β hε

/-- ViT's per-token LN VJP, bridged to the flat layout. No new proof — `layerNormVec_has_vjp`
    is `(+β) ∘ layerScale γ ∘ LN(1,0)` and needs only `0 < ε`. -/
noncomputable def rowLNVecFlat_has_vjp (s c : Nat) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε) :
    HasVJP (rowLNVecFlat s c ε γ β) :=
  hasVJPMat_to_hasVJP (layerNormVec_per_token_has_vjp_mat s c ε γ β hε)

/-- `transposeFlat` is a coordinate permutation, hence a `reindexCLM`. -/
theorem transposeFlat_diff (m n : Nat) : Differentiable ℝ (transposeFlat m n) := by
  have h : transposeFlat m n = fun v : Vec (m * n) => fun idx : Fin (n * m) =>
      v (finProdFinEquiv ((finProdFinEquiv.symm idx).2, (finProdFinEquiv.symm idx).1)) := by
    funext v idx; rfl
  rw [h]
  exact (reindexCLM (fun idx : Fin (n * m) =>
    finProdFinEquiv ((finProdFinEquiv.symm idx).2, (finProdFinEquiv.symm idx).1))).differentiable

/-- `transposeFlat`'s VJP is `Tensor.lean`'s `transpose_has_vjp` through the flatten bijection —
    the flat form is definitionally the bridged Mat form, so this is a re-typing, not a proof. -/
noncomputable def transposeFlat_has_vjp (m n : Nat) : HasVJP (transposeFlat m n) :=
  hasVJPMat_to_hasVJP (transpose_has_vjp (m := m) (n := n))

-- ════════════════════════════════════════════════════════════════
-- § Channel LayerNorm at the network's Tensor3 layout
-- ════════════════════════════════════════════════════════════════

/-- **ConvNeXt's channel LayerNorm** on the activation layout the convolutions use
    (`Vec (c*h*w)`): re-associate to the Mat-split `[c, h·w]`, transpose to `[h·w, c]` so each
    row is one spatial position, normalise that row over its `c` channels with the per-channel
    `[c]` affine, then transpose and re-associate back.

    Contrast the incumbent `layerNormForward (c*h*w) ε γ β`, which takes ONE mean and ONE
    variance over all `c·h·w` values and applies two scalars — for a stage-1 site that is one
    statistic where ConvNeXt wants 3,136 of them. -/
noncomputable def chanLNTensor3 (c h w : Nat) (ε : ℝ) (γ β : Vec c) :
    Vec (c * h * w) → Vec (c * h * w) :=
  reassocBack c h w ∘
    transposeFlat (h * w) c ∘
    rowLNVecFlat (h * w) c ε γ β ∘
    transposeFlat c (h * w) ∘
    reassocFwd c h w

/-- Everywhere-differentiable given `0 < ε` — four permutations and one LN. -/
theorem chanLNTensor3_diff (c h w : Nat) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε) :
    Differentiable ℝ (chanLNTensor3 c h w ε γ β) := by
  unfold chanLNTensor3
  exact (reassocBack_differentiable c h w).comp
    ((transposeFlat_diff (h * w) c).comp
      ((rowLNVecFlat_diff (h * w) c ε γ β hε).comp
        ((transposeFlat_diff c (h * w)).comp (reassocFwd_differentiable c h w))))

/-- **Channel-LN VJP (global)** — `vjp_comp` over the five proven pieces. The only hypothesis
    is the LN positivity `0 < ε`, exactly as the scalar `layerNorm_has_vjp` it replaces. -/
noncomputable def chanLNTensor3_has_vjp (c h w : Nat) (ε : ℝ) (γ β : Vec c) (hε : 0 < ε) :
    HasVJP (chanLNTensor3 c h w ε γ β) := by
  unfold chanLNTensor3
  have d0 := reassocFwd_differentiable c h w
  have d1 := transposeFlat_diff c (h * w)
  have d2 := rowLNVecFlat_diff (h * w) c ε γ β hε
  have d3 := transposeFlat_diff (h * w) c
  have d4 := reassocBack_differentiable c h w
  have e1 := vjp_comp _ _ d0 d1 (reassocFwd_has_vjp c h w) (transposeFlat_has_vjp c (h * w))
  have f1 := d1.comp d0
  have e2 := vjp_comp _ _ f1 d2 e1 (rowLNVecFlat_has_vjp (h * w) c ε γ β hε)
  have f2 := d2.comp f1
  have e3 := vjp_comp _ _ f2 d3 e2 (transposeFlat_has_vjp (h * w) c)
  have f3 := d3.comp f2
  exact vjp_comp _ _ f3 d4 e3 (reassocBack_has_vjp c h w)

/-- **The emitted three-op affine tail IS the per-token vector-LN.** The chain normalises with
    `lnRowF` at scalar γ=1/β=0 and then applies the REAL `[c]` affine with `rowScaleF`/`rowBiasF`
    — ViT's spelling, and the reason ConvNeXt needs no new op. This is the lemma that lets the
    graph's five denotations collapse onto `chanLNTensor3`'s three. -/
theorem rowLN_affine_eq (s c : Nat) (ε : ℝ) (γ β : Vec c) (u : Vec (s * c)) :
    StableHLO.rowBiasFlat s c β
        (StableHLO.rowScaleFlat s c γ (StableHLO.rowLNFlat s c ε 1 0 u))
      = rowLNVecFlat s c ε γ β u := by
  unfold StableHLO.rowBiasFlat StableHLO.rowScaleFlat StableHLO.rowLNFlat
         rowLNVecFlat layerNormVec layerScale layerNormForward
  simp only [Mat.unflatten_flatten]


-- ════════════════════════════════════════════════════════════════
-- § The per-channel γ/β parameter Jacobians — ViT's `[D]` certs, conjugated
--
-- The render's γ/β tails (`ConvNeXtRender.lnGammaTail`/`lnBetaTail`) re-emit the two transposes
-- and then run ViT's `veclnGammaSgd`/`rowDenseBiasSgd` on the `[h·w, c]` view. So the emitted op
-- is certified by ViT's `vit_render_vecln{gamma,beta}_certified` *at that layout* — but the
-- theorem the net needs is about `chanLNTensor3` at the `c·h·w` activation layout, contracted
-- with the chain cotangent the block backward delivers there.
--
-- The bridge is one fact: `chanLNTensor3`'s pre- and post-conjugations are inverse PERMUTATIONS,
-- so the output-side one moves onto the cotangent as its inverse — which is exactly the
-- transposed cotangent the render feeds the op. No new analysis; a permutation's adjoint.
-- ════════════════════════════════════════════════════════════════

/-- The saved activation as the row backward sees it: the `[h·w, c]` view of `x`, one row per
    spatial position holding its `c` channels. Naming this keeps the backward's operating-point
    hypotheses (`bnIstd`/`bnXhat` per row) readable — it is `chanLNTensor3`'s own first two
    factors, and it is the value the render's re-emitted `transposeF` pair denotes. -/
noncomputable def chanLNRows (c h w : Nat) (x : Vec (c * h * w)) : Vec ((h * w) * c) :=
  transposeFlat c (h * w) (reassocFwd c h w x)

/-- `chanLNTensor3`'s conjugation as ONE index map: the activation index `j` of the `c·h·w` layout,
    read in the `[h·w, c]` row view. Composition of the Mat-split re-association and the
    transpose, both pure reindexes. -/
noncomputable def chanRowsIdx (c h w : Nat) (j : Fin (c * h * w)) : Fin ((h * w) * c) :=
  let ch := finProdFinEquiv.symm (reassocBackIdx c h w j)
  finProdFinEquiv (ch.2, ch.1)

/-- The inverse direction — the row-view index `o`, read back in the activation layout. -/
noncomputable def chanRowsIdxInv (c h w : Nat) (o : Fin ((h * w) * c)) : Fin (c * h * w) :=
  let sc := finProdFinEquiv.symm o
  reassocFwdIdx c h w (finProdFinEquiv (sc.2, sc.1))

theorem chanRowsIdxInv_chanRowsIdx (c h w : Nat) (j : Fin (c * h * w)) :
    chanRowsIdxInv c h w (chanRowsIdx c h w j) = j := by
  unfold chanRowsIdx chanRowsIdxInv
  simp only [Equiv.symm_apply_apply]
  rw [Prod.mk.eta, Equiv.apply_symm_apply, reassocFwdIdx_reassocBackIdx]

theorem chanRowsIdx_chanRowsIdxInv (c h w : Nat) (o : Fin ((h * w) * c)) :
    chanRowsIdx c h w (chanRowsIdxInv c h w o) = o := by
  unfold chanRowsIdx chanRowsIdxInv
  simp only [reassocBackIdx_reassocFwdIdx, Equiv.symm_apply_apply]
  rw [Prod.mk.eta, Equiv.apply_symm_apply]

/-- **The conjugation is a permutation.** Both directions are `finProdFinEquiv` round-trips, so
    the `[c·h·w] ↔ [h·w, c]` relabeling is a genuine bijection — the fact the two certs below
    turn into "the adjoint is the inverse". -/
noncomputable def chanRowsPerm (c h w : Nat) : Fin (c * h * w) ≃ Fin ((h * w) * c) :=
  ⟨chanRowsIdx c h w, chanRowsIdxInv c h w,
   chanRowsIdxInv_chanRowsIdx c h w, chanRowsIdx_chanRowsIdxInv c h w⟩

/-- The row view IS the reindex by the inverse permutation (definitional; stated so the two
    spellings are visibly one map rather than two that happen to agree). -/
theorem chanLNRows_eq_reindex (c h w : Nat) (v : Vec (c * h * w)) :
    chanLNRows c h w v = fun o => v ((chanRowsPerm c h w).symm o) := rfl

/-- And channel-LN is the row-LN read through the permutation. -/
theorem chanLNTensor3_eq_rows (c h w : Nat) (ε : ℝ) (γ β : Vec c) (x : Vec (c * h * w)) :
    chanLNTensor3 c h w ε γ β x
      = fun j => rowLNVecFlat (h * w) c ε γ β (chanLNRows c h w x) (chanRowsPerm c h w j) := rfl

/-- **An output-side permutation moves onto the cotangent as its inverse.** Generic: for any
    differentiable `f` and any bijection `σ` of output indices, contracting the Jacobian of
    `σ`-reindexed `f` with a cotangent is contracting `f`'s own Jacobian with the `σ⁻¹`-reindexed
    cotangent. `pdiv_comp` against `pdiv_reindex`'s indicator, then `Equiv.sum_comp`. -/
theorem pdiv_reindexOut_contract {m n n' : Nat} (f : Vec m → Vec n) (x : Vec m)
    (hf : DifferentiableAt ℝ f x) (σ : Fin n' ≃ Fin n) (i : Fin m) (cot : Vec n') :
    ∑ j : Fin n', pdiv (fun y : Vec m => fun k : Fin n' => f y (σ k)) x i j * cot j
      = ∑ o : Fin n, pdiv f x i o * cot (σ.symm o) := by
  have hstep : ∀ j : Fin n',
      pdiv (fun y : Vec m => fun k : Fin n' => f y (σ k)) x i j = pdiv f x i (σ j) := by
    intro j
    have hg : DifferentiableAt ℝ (fun z : Vec n => fun k : Fin n' => z (σ k)) (f x) :=
      (reindexCLM (fun k : Fin n' => σ k)).differentiableAt
    rw [show (fun y : Vec m => fun k : Fin n' => f y (σ k))
          = (fun z : Vec n => fun k : Fin n' => z (σ k)) ∘ f from rfl,
        pdiv_comp f (fun z : Vec n => fun k : Fin n' => z (σ k)) x hf hg i j]
    simp_rw [pdiv_reindex (fun k : Fin n' => σ k)]
    rw [Finset.sum_eq_single (σ j)
        (fun o _ hne => by rw [if_neg hne, mul_zero])
        (fun h => absurd (Finset.mem_univ (σ j)) h), if_pos rfl, mul_one]
  simp_rw [hstep]
  rw [← Equiv.sum_comp σ (fun o => pdiv f x i o * cot (σ.symm o))]
  exact Finset.sum_congr rfl (fun j _ => by rw [Equiv.symm_apply_apply])

/-- As a function of γ the row-LN is `x̂ ⊙ gather γ + β` — a masked gather plus a constant, hence
    differentiable. (ViT proves the Jacobian of this shape; the differentiability is what
    `pdiv_comp` needs and what it does not export.) -/
theorem rowLNVecFlat_gamma_diffAt (s c : Nat) (ε : ℝ) (β : Vec c) (X : Vec (s * c)) (γ : Vec c) :
    DifferentiableAt ℝ (fun γ' : Vec c => rowLNVecFlat s c ε γ' β X) γ := by
  have hmul : DifferentiableAt ℝ
      (fun (v : Vec c) (o : Fin (s * c)) =>
        layerNormForward c ε 1 0 (Mat.unflatten X (finProdFinEquiv.symm o).1)
            (finProdFinEquiv.symm o).2 * v (finProdFinEquiv.symm o).2) γ :=
    (differentiableAt_const _).mul
      (reindexCLM (fun o : Fin (s * c) => (finProdFinEquiv.symm o).2)).differentiableAt
  have hconst : DifferentiableAt ℝ
      (fun (_ : Vec c) (o : Fin (s * c)) => β (finProdFinEquiv.symm o).2) γ :=
    differentiableAt_const _
  have hshape : (fun γ' : Vec c => rowLNVecFlat s c ε γ' β X)
      = (fun γ' : Vec c => fun o : Fin (s * c) =>
          (fun (v : Vec c) (o' : Fin (s * c)) =>
            layerNormForward c ε 1 0 (Mat.unflatten X (finProdFinEquiv.symm o').1)
                (finProdFinEquiv.symm o').2 * v (finProdFinEquiv.symm o').2) γ' o
          + (fun (_ : Vec c) (o' : Fin (s * c)) => β (finProdFinEquiv.symm o').2) γ' o) := by
    funext γ' o
    unfold rowLNVecFlat layerNormVec Mat.flatten Mat.unflatten
    ring
  rw [hshape]
  exact hmul.add hconst

/-- The β peer: `const + gather β`. -/
theorem rowLNVecFlat_beta_diffAt (s c : Nat) (ε : ℝ) (γ : Vec c) (X : Vec (s * c)) (β : Vec c) :
    DifferentiableAt ℝ (fun β' : Vec c => rowLNVecFlat s c ε γ β' X) β := by
  have hconst : DifferentiableAt ℝ
      (fun (_ : Vec c) (o : Fin (s * c)) =>
        γ (finProdFinEquiv.symm o).2 *
          layerNormForward c ε 1 0 (Mat.unflatten X (finProdFinEquiv.symm o).1)
            (finProdFinEquiv.symm o).2) β := differentiableAt_const _
  have hgather : DifferentiableAt ℝ
      (fun (v : Vec c) (o : Fin (s * c)) => v (finProdFinEquiv.symm o).2) β :=
    (reindexCLM (fun o : Fin (s * c) => (finProdFinEquiv.symm o).2)).differentiableAt
  have hshape : (fun β' : Vec c => rowLNVecFlat s c ε γ β' X)
      = (fun β' : Vec c => fun o : Fin (s * c) =>
          (fun (_ : Vec c) (o' : Fin (s * c)) =>
            γ (finProdFinEquiv.symm o').2 *
              layerNormForward c ε 1 0 (Mat.unflatten X (finProdFinEquiv.symm o').1)
                (finProdFinEquiv.symm o').2) β' o
          + (fun (v : Vec c) (o' : Fin (s * c)) => v (finProdFinEquiv.symm o').2) β' o) := by
    funext β' o
    unfold rowLNVecFlat layerNormVec Mat.flatten Mat.unflatten
    ring
  rw [hshape]
  exact hconst.add hgather

/-- **The γ contraction, moved to the row layout.** The activation-layout Jacobian against the
    activation-layout cotangent equals the row-layout Jacobian against the TRANSPOSED cotangent —
    which is the operand `lnGammaTail` actually emits. -/
theorem chanLN_gamma_contract {c h w : Nat} (ε : ℝ) (β γ : Vec c) (x cot : Vec (c * h * w))
    (k : Fin c) :
    ∑ j : Fin (c * h * w),
        pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β x) γ k j * cot j
      = ∑ o : Fin ((h * w) * c),
          pdiv (fun γ' : Vec c => rowLNVecFlat (h * w) c ε γ' β (chanLNRows c h w x)) γ k o
            * chanLNRows c h w cot o := by
  rw [show (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β x)
        = (fun γ' : Vec c => fun j : Fin (c * h * w) =>
            (fun v : Vec c => rowLNVecFlat (h * w) c ε v β (chanLNRows c h w x)) γ'
              (chanRowsPerm c h w j)) from rfl,
      pdiv_reindexOut_contract _ γ (rowLNVecFlat_gamma_diffAt (h * w) c ε β _ γ)
        (chanRowsPerm c h w) k cot]
  rfl

/-- The β peer of `chanLN_gamma_contract`. -/
theorem chanLN_beta_contract {c h w : Nat} (ε : ℝ) (γ β : Vec c) (x cot : Vec (c * h * w))
    (k : Fin c) :
    ∑ j : Fin (c * h * w),
        pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' x) β k j * cot j
      = ∑ o : Fin ((h * w) * c),
          pdiv (fun β' : Vec c => rowLNVecFlat (h * w) c ε γ β' (chanLNRows c h w x)) β k o
            * chanLNRows c h w cot o := by
  rw [show (fun β' : Vec c => chanLNTensor3 c h w ε γ β' x)
        = (fun β' : Vec c => fun j : Fin (c * h * w) =>
            (fun v : Vec c => rowLNVecFlat (h * w) c ε γ v (chanLNRows c h w x)) β'
              (chanRowsPerm c h w j)) from rfl,
      pdiv_reindexOut_contract _ β (rowLNVecFlat_beta_diffAt (h * w) c ε γ _ β)
        (chanRowsPerm c h w) k cot]
  rfl

/-- **Channel-LN γ output, certified.** The rendered per-channel reduce — ViT's
    `vecLN_grad_gamma` on the two transposed views the tail emits — equals the certified Jacobian
    of `chanLNTensor3` in its `Vec c` γ, contracted with the activation-layout cotangent. The
    `Vec c` peer of `ConvNeXtClose.cnx_render_lngamma_certified`, and the `den` target of the
    render's `veclnGammaSgd` LN tail. -/
theorem cnx_render_chlngamma_certified {c h w : Nat} (ε : ℝ) (β γ : Vec c)
    (x cot : Vec (c * h * w)) (lr : ℝ) (k : Fin c) :
    γ k - lr * vecLN_grad_gamma (h * w) c ε (Mat.unflatten (chanLNRows c h w x))
                  (Mat.unflatten (chanLNRows c h w cot)) k
      = γ k - lr * ∑ j : Fin (c * h * w),
          pdiv (fun γ' : Vec c => chanLNTensor3 c h w ε γ' β x) γ k j * cot j := by
  rw [chanLN_gamma_contract ε β γ x cot k]
  exact congrArg (fun t => γ k - lr * t)
    (vit_veclnGamma_grad_bridge ε β γ (Mat.unflatten (chanLNRows c h w x))
      (chanLNRows c h w cot) k)

/-- **Channel-LN β output, certified.** The β grad is the plain reduce `Σ_rows dy`, so the same
    `rowDenseBiasSgd` op ViT's LN-β uses denotes it here too. -/
theorem cnx_render_chlnbeta_certified {c h w : Nat} (ε : ℝ) (γ β : Vec c)
    (x cot : Vec (c * h * w)) (lr : ℝ) (k : Fin c) :
    β k - lr * vecLN_grad_beta (h * w) c (Mat.unflatten (chanLNRows c h w cot)) k
      = β k - lr * ∑ j : Fin (c * h * w),
          pdiv (fun β' : Vec c => chanLNTensor3 c h w ε γ β' x) β k j * cot j := by
  rw [chanLN_beta_contract ε γ β x cot k]
  exact congrArg (fun t => β k - lr * t)
    (vit_veclnBeta_grad_bridge ε γ β (Mat.unflatten (chanLNRows c h w x))
      (chanLNRows c h w cot) k)

end Proofs

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The graph-side transport
-- ════════════════════════════════════════════════════════════════

/-- **`den` commutes with a type-level index transport.** Transporting the GRAPH along `m = n`
    reindexes its denotation by the val-preserving `Fin.cast`. Stated at variable `m`/`n` so
    `subst` applies — at `c*h*w = c*(h*w)` neither side is a variable and it would not. -/
theorem den_cast {m n : Nat} (heq : m = n) (e : SHlo m) :
    den (heq ▸ e) = fun k => den e (Fin.cast heq.symm k) := by
  subst heq; rfl

/-- **The graph's `▸` transport IS the math's Mat-split bridge** — `den_cast` composed with
    `reassocFwdIdx_val`. This is the lemma that keeps `ConvNeXtRender`'s `reassoc` and
    `chanLNTensor3` describing one function. -/
theorem den_reassocS {c h w : Nat} (e : SHlo (c * h * w)) :
    den ((Nat.mul_assoc c h w) ▸ e) = reassocFwd c h w (den e) := by
  rw [den_cast]
  funext k
  exact congrArg (den e) (Fin.ext (reassocFwdIdx_val c h w k).symm)

theorem den_unassocS {c h w : Nat} (e : SHlo (c * (h * w))) :
    den ((Nat.mul_assoc c h w).symm ▸ e) = reassocBack c h w (den e) := by
  rw [den_cast]
  funext k
  exact congrArg (den e) (Fin.ext (reassocBackIdx_val c h w k).symm)

end Proofs.StableHLO
