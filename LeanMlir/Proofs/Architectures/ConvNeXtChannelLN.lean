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
