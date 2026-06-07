import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.StridedConv

/-! # Toward real ResNet-34 — the deep-block chain (Chapter 6 Milestone B4)

A real ResNet-34 stacks **16 basic blocks** in four stages (3+4+6+3). Within a
stage every block is a self-map `Vec n → Vec n` (same channel count) but with its
**own** weights — so it is a *composition of a list* of distinct same-type maps,
not an `iterate` of one map.

This file proves the generic enabler: if every map in a list is differentiable
and has a VJP, their composition (`chainComp`) does too — by induction chaining
`vjp_comp`. That turns "16 blocks deep" into a `List.length`, no per-block
boilerplate. The full ResNet-34 forward (strided proj blocks via `flatConvStride2`
+ chained identity blocks + per-channel BN) is assembled on top of this.

Closes under `[propext, Classical.choice, Quot.sound]`.
-/

open Finset BigOperators

namespace Proofs

-- ════════════════════════════════════════════════════════════════
-- § Composition of a list of same-type self-maps
-- ════════════════════════════════════════════════════════════════

/-- Compose a list of self-maps left-to-right as data flows: `chainComp [f₁,…,fₖ]
    = f₁ ∘ … ∘ fₖ` (the last list element runs first, i.e. is the deepest). A
    ResNet stage is `chainComp` of its blocks. -/
noncomputable def chainComp {n : Nat} (fs : List (Vec n → Vec n)) : Vec n → Vec n :=
  fs.foldr (· ∘ ·) id

@[simp] theorem chainComp_nil {n : Nat} : chainComp ([] : List (Vec n → Vec n)) = id := rfl

@[simp] theorem chainComp_cons {n : Nat} (f : Vec n → Vec n) (fs : List (Vec n → Vec n)) :
    chainComp (f :: fs) = f ∘ chainComp fs := rfl

/-- A chain of differentiable maps is differentiable. -/
theorem chainComp_differentiable {n : Nat} (fs : List (Vec n → Vec n))
    (hdiff : ∀ f ∈ fs, Differentiable ℝ f) : Differentiable ℝ (chainComp fs) := by
  induction fs with
  | nil => exact differentiable_id
  | cons f fs ih =>
    rw [chainComp_cons]
    exact (hdiff f (List.mem_cons.2 (Or.inl rfl))).comp
      (ih (fun g hg => hdiff g (List.mem_cons.2 (Or.inr hg))))

/-- **Deep-chain VJP.** A composition of a list of differentiable maps that each
    have a VJP has a VJP — the backward runs each block's backward in reverse
    order. By induction chaining `vjp_comp`; the structural heart of a deep
    ResNet stage (k distinct-weight basic blocks). -/
noncomputable def vjp_chain {n : Nat} (fs : List (Vec n → Vec n))
    (hdiff : ∀ f ∈ fs, Differentiable ℝ f) (hvjp : ∀ f ∈ fs, HasVJP f) :
    HasVJP (chainComp fs) :=
  match fs with
  | [] => show HasVJP (id : Vec n → Vec n) from identity_has_vjp n
  | f :: rest =>
    show HasVJP (f ∘ chainComp rest) from
    vjp_comp (chainComp rest) f
      (chainComp_differentiable rest (fun g hg => hdiff g (List.mem_cons.2 (Or.inr hg))))
      (hdiff f (List.mem_cons.2 (Or.inl rfl)))
      (vjp_chain rest (fun g hg => hdiff g (List.mem_cons.2 (Or.inr hg)))
                      (fun g hg => hvjp g (List.mem_cons.2 (Or.inr hg))))
      (hvjp f (List.mem_cons.2 (Or.inl rfl)))

/-- **Deep-chain VJP correctness** (ℝ-headline): the chained backward equals the
    `pdiv`-contracted Jacobian of the whole composition. -/
theorem vjp_chain_correct {n : Nat} (fs : List (Vec n → Vec n))
    (hdiff : ∀ f ∈ fs, Differentiable ℝ f) (hvjp : ∀ f ∈ fs, HasVJP f)
    (x dy : Vec n) (i : Fin n) :
    (vjp_chain fs hdiff hvjp).backward x dy i
      = ∑ j : Fin n, pdiv (chainComp fs) x i j * dy j :=
  (vjp_chain fs hdiff hvjp).correct x dy i

-- ════════════════════════════════════════════════════════════════
-- § Deep-block chain at a smooth point (the conditional `_at` chain)
-- ════════════════════════════════════════════════════════════════

/-- Recursive hypothesis bundle for a chain of `HasVJPAt` blocks: each block is
    `DifferentiableAt` and `HasVJPAt` **at its running activation** — the point
    `chainComp rest x` feeding it (the deeper blocks run first). Residual identity
    blocks are only `HasVJPAt` at smooth points, so the chain must thread the
    point, not assume global differentiability. -/
def ChainData {n : Nat} (x : Vec n) : List (Vec n → Vec n) → Type
  | [] => PUnit
  | f :: rest =>
      -- `PProd` (not `×`): the first field `DifferentiableAt` is a `Prop`.
      PProd (DifferentiableAt ℝ f (chainComp rest x))
        (PProd (HasVJPAt f (chainComp rest x)) (ChainData x rest))

/-- The chain at a point both **has a VJP and is differentiable** there, from the
    per-block `ChainData`. The companion `DifferentiableAt` is carried alongside
    so the recursion can feed the inner-composition differentiability into each
    `vjp_comp_at` / `DifferentiableAt.comp`. -/
noncomputable def chain_vjp_diff_at {n : Nat} (x : Vec n) :
    (fs : List (Vec n → Vec n)) → ChainData x fs →
      PProd (HasVJPAt (chainComp fs) x) (DifferentiableAt ℝ (chainComp fs) x)
  | [], _ => ⟨(identity_has_vjp n).toHasVJPAt x, differentiable_id.differentiableAt⟩
  | f :: rest, d =>
      let ih := chain_vjp_diff_at x rest d.snd.snd
      ⟨vjp_comp_at (chainComp rest) f x ih.snd d.fst ih.fst d.snd.fst, d.fst.comp x ih.snd⟩

/-- **Deep-block chain VJP at a smooth point.** A composition of conditional
    (`HasVJPAt`) blocks — e.g. the k identity residual blocks of a ResNet stage —
    has a VJP at `x`, given each block is differentiable + has a VJP at its
    running activation (`ChainData`). The `_at` peer of `vjp_chain`. -/
noncomputable def vjp_chain_at {n : Nat} (x : Vec n) (fs : List (Vec n → Vec n))
    (hdata : ChainData x fs) : HasVJPAt (chainComp fs) x :=
  (chain_vjp_diff_at x fs hdata).fst

/-- **Deep-chain-at VJP correctness** (ℝ-headline): the chained backward at `x`
    equals the `pdiv`-Jacobian of the composition at `x`. -/
theorem vjp_chain_at_correct {n : Nat} (x : Vec n) (fs : List (Vec n → Vec n))
    (hdata : ChainData x fs) (dy : Vec n) (i : Fin n) :
    (vjp_chain_at x fs hdata).backward dy i = ∑ j : Fin n, pdiv (chainComp fs) x i j * dy j :=
  (vjp_chain_at x fs hdata).correct dy i

/-- **A full ResNet stage has a VJP at a point.** A stage is a downsample block
    `down : Vec m → Vec n` (channel/spatial change — `rblkPStrided`, or for the
    first stage the identity / stem-fed input) followed by a chain of `k` identity
    residual blocks `chainComp ids : Vec n → Vec n`. VJPAt by one `vjp_comp_at`
    gluing the downsample to the (deep-chained) identity blocks. The reusable
    composition pattern for assembling ResNet-34's four stages. -/
noncomputable def resStage_has_vjp_at {m n : Nat}
    (down : Vec m → Vec n) (ids : List (Vec n → Vec n)) (x : Vec m)
    (hdown_diff : DifferentiableAt ℝ down x) (hdown : HasVJPAt down x)
    (hids : ChainData (down x) ids) :
    HasVJPAt (chainComp ids ∘ down) x :=
  vjp_comp_at down (chainComp ids) x hdown_diff
    (chain_vjp_diff_at (down x) ids hids).snd hdown (vjp_chain_at (down x) ids hids)

/-- **ResNet-stage VJP correctness** (ℝ-headline): the stage's backward equals the
    `pdiv`-Jacobian of `(identity-block chain) ∘ downsample` at `x`. -/
theorem resStage_has_vjp_at_correct {m n : Nat}
    (down : Vec m → Vec n) (ids : List (Vec n → Vec n)) (x : Vec m)
    (hdown_diff : DifferentiableAt ℝ down x) (hdown : HasVJPAt down x)
    (hids : ChainData (down x) ids) (dy : Vec n) (i : Fin m) :
    (resStage_has_vjp_at down ids x hdown_diff hdown hids).backward dy i
      = ∑ j : Fin n, pdiv (chainComp ids ∘ down) x i j * dy j :=
  (resStage_has_vjp_at down ids x hdown_diff hdown hids).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § The whole ResNet-34 network VJP
-- ════════════════════════════════════════════════════════════════

/-- Compose two `HasVJPAt`-with-`DifferentiableAt` pairs (carried as `PProd` so the
    `DifferentiableAt` `Prop` is allowed). The fold step for the whole net. -/
noncomputable def vjp_comp_diff_at {m n p : Nat} (f : Vec m → Vec n) (g : Vec n → Vec p)
    (x : Vec m)
    (hf : PProd (HasVJPAt f x) (DifferentiableAt ℝ f x))
    (hg : PProd (HasVJPAt g (f x)) (DifferentiableAt ℝ g (f x))) :
    PProd (HasVJPAt (g ∘ f) x) (DifferentiableAt ℝ (g ∘ f) x) :=
  ⟨vjp_comp_at f g x hf.snd hg.snd hf.fst hg.fst, hg.snd.comp x hf.snd⟩

/-- **Whole-network ResNet-34 VJP.** The conditional VJP of a real ResNet-34-shaped
    network at an input `x`:

      `dense ∘ GAP ∘ stage₄ ∘ stage₃ ∘ stage₂ ∘ stage₁ ∘ maxpool ∘ stem`

    with `stageᵢ = (identity-block chain) ∘ downsampleᵢ` for the three downsampling
    stages (the 3+4+6+3 = 16 basic blocks live in the `idsᵢ` lists + the three
    `downᵢ` blocks; instantiate `down`/`ids`/`stem`/`gap`/`dense` with the verified
    `convBnReluStrided`/`rblkPStrided`/`rblk`/`globalAvgPoolFlat`/`dense` and
    `maxPoolFlat`). Parametric over the component functions and their per-component
    VJP+differentiability witnesses at the running activations — so depth is a
    `List.length`, not 100 explicit weight arguments. Folded from the verified
    `vjp_comp_at` / `vjp_chain_at` (`ChainData` threads each block's smooth point).

    This is the structural analogue of `cnn_has_vjp_at` scaled to 34 layers; the
    discharge of the smoothness/no-tie hypotheses for a concrete instance (à la
    `CnnConcrete`) plus per-channel BN and the GPU render remain. -/
noncomputable def resnet34_has_vjp_at
    {s0 s1 s2 s3 s4 s5 s6 s7 : Nat}
    (stem : Vec s0 → Vec s1) (mp : Vec s1 → Vec s2)
    (ids1 : List (Vec s2 → Vec s2))
    (down2 : Vec s2 → Vec s3) (ids2 : List (Vec s3 → Vec s3))
    (down3 : Vec s3 → Vec s4) (ids3 : List (Vec s4 → Vec s4))
    (down4 : Vec s4 → Vec s5) (ids4 : List (Vec s5 → Vec s5))
    (gap : Vec s5 → Vec s6) (dense : Vec s6 → Vec s7)
    (x : Vec s0)
    (hstem : PProd (HasVJPAt stem x) (DifferentiableAt ℝ stem x))
    (hmp : PProd (HasVJPAt mp (stem x)) (DifferentiableAt ℝ mp (stem x)))
    (hids1 : ChainData (mp (stem x)) ids1)
    (hdown2 : PProd (HasVJPAt down2 (chainComp ids1 (mp (stem x))))
                    (DifferentiableAt ℝ down2 (chainComp ids1 (mp (stem x)))))
    (hids2 : ChainData (down2 (chainComp ids1 (mp (stem x)))) ids2)
    (hdown3 : PProd (HasVJPAt down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x))))))
                    (DifferentiableAt ℝ down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))
    (hids3 : ChainData (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))) ids3)
    (hdown4 : PProd (HasVJPAt down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x))))))))
                    (DifferentiableAt ℝ down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))))
    (hids4 : ChainData (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))) ids4)
    (hgap : PProd (HasVJPAt gap (chainComp ids4 (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x))))))))))
                  (DifferentiableAt ℝ gap (chainComp ids4 (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))))))
    (hdense : PProd (HasVJPAt dense (gap (chainComp ids4 (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x)))))))))))
                    (DifferentiableAt ℝ dense (gap (chainComp ids4 (down4 (chainComp ids3 (down3 (chainComp ids2 (down2 (chainComp ids1 (mp (stem x))))))))))))
    : HasVJPAt
        (dense ∘ gap ∘ chainComp ids4 ∘ down4 ∘ chainComp ids3 ∘ down3 ∘
          chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem) x :=
  let p1 := vjp_comp_diff_at stem mp x hstem hmp
  let p2 := vjp_comp_diff_at (mp ∘ stem) (chainComp ids1) x p1 (chain_vjp_diff_at _ ids1 hids1)
  let p3 := vjp_comp_diff_at (chainComp ids1 ∘ mp ∘ stem) down2 x p2 hdown2
  let p4 := vjp_comp_diff_at (down2 ∘ chainComp ids1 ∘ mp ∘ stem) (chainComp ids2) x p3
              (chain_vjp_diff_at _ ids2 hids2)
  let p5 := vjp_comp_diff_at (chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem) down3 x p4 hdown3
  let p6 := vjp_comp_diff_at (down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              (chainComp ids3) x p5 (chain_vjp_diff_at _ ids3 hids3)
  let p7 := vjp_comp_diff_at (chainComp ids3 ∘ down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              down4 x p6 hdown4
  let p8 := vjp_comp_diff_at (down4 ∘ chainComp ids3 ∘ down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              (chainComp ids4) x p7 (chain_vjp_diff_at _ ids4 hids4)
  let p9 := vjp_comp_diff_at (chainComp ids4 ∘ down4 ∘ chainComp ids3 ∘ down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              gap x p8 hgap
  let p10 := vjp_comp_diff_at (gap ∘ chainComp ids4 ∘ down4 ∘ chainComp ids3 ∘ down3 ∘ chainComp ids2 ∘ down2 ∘ chainComp ids1 ∘ mp ∘ stem)
              dense x p9 hdense
  p10.fst

-- ════════════════════════════════════════════════════════════════
-- § Strided downsampling block — conv(stride 2) → BN → relu
-- ════════════════════════════════════════════════════════════════

/-- **conv(stride-2) → bn block VJP (no ReLU), everywhere.** The strided peer of
    `convBn_has_vjp`: `flatConvStride2` then `bnForward`, both differentiable
    everywhere, so a global `HasVJP`. The downsampling body of a stage-start
    block and its strided 1×1 projection skip. -/
noncomputable def convBnStrided_has_vjp {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε) :
    HasVJP (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b
      : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
  vjp_comp (flatConvStride2 W b) (bnForward (oc * h * w) ε γ β)
    (flatConvStride2_differentiable W b)
    (bnForward_differentiable (oc * h * w) ε γ β hε)
    (flatConvStride2_has_vjp W b)
    (bn_has_vjp (oc * h * w) ε γ β hε)

/-- **conv(stride-2) → bn is differentiable everywhere.** -/
theorem convBnStrided_differentiable {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε) :
    Differentiable ℝ (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b
      : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
  (bnForward_differentiable (oc * h * w) ε γ β hε).comp (flatConvStride2_differentiable W b)

/-- **conv(stride-2) → bn → relu block VJP at a smooth point.** The strided peer
    of `convBnRelu_has_vjp_at` (the workhorse opening each downsampling stage):
    two `vjp_comp_at`, with `flatConvStride2_has_vjp` for the conv and the ReLU
    smoothness hypothesis `h_smooth` (no post-BN activation hits the kink). -/
noncomputable def convBnReluStrided_has_vjp_at {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, bnForward (oc * h * w) ε γ β (flatConvStride2 W b v) k ≠ 0) :
    HasVJPAt (relu (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v := by
  have hconv_diff : Differentiable ℝ
      (flatConvStride2 W b : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W b
  have hbn_diff : Differentiable ℝ (bnForward (oc * h * w) ε γ β) :=
    bnForward_differentiable (oc * h * w) ε γ β hε
  have step1 : HasVJPAt (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v :=
    vjp_comp_at (flatConvStride2 W b) (bnForward (oc * h * w) ε γ β) v
      (hconv_diff v) (hbn_diff _)
      ((flatConvStride2_has_vjp W b).toHasVJPAt v)
      ((bn_has_vjp (oc * h * w) ε γ β hε).toHasVJPAt _)
  have step1_diff : DifferentiableAt ℝ
      (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v :=
    DifferentiableAt.comp v (hbn_diff _) (hconv_diff v)
  exact vjp_comp_at (bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b)
    (relu (oc * h * w)) v step1_diff
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth) step1
    (relu_has_vjp_at (oc * h * w) _ h_smooth)

/-- **Strided block VJP correctness** (ℝ-headline): the strided downsampling
    block's backward equals the `pdiv`-Jacobian of `relu ∘ bn ∘ conv_stride2`. -/
theorem convBnReluStrided_has_vjp_at_correct {ic oc h w kH kW : Nat}
    (W : Kernel4 oc ic kH kW) (b : Vec oc) (ε γ β : ℝ) (hε : 0 < ε)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth : ∀ k, bnForward (oc * h * w) ε γ β (flatConvStride2 W b v) k ≠ 0)
    (dy : Vec (oc * h * w)) (i : Fin (ic * (2 * h) * (2 * w))) :
    (convBnReluStrided_has_vjp_at W b ε γ β hε v h_smooth).backward dy i
      = ∑ j : Fin (oc * h * w),
          pdiv (relu (oc * h * w) ∘ bnForward (oc * h * w) ε γ β ∘ flatConvStride2 W b) v i j * dy j :=
  (convBnReluStrided_has_vjp_at W b ε γ β hε v h_smooth).correct dy i

-- ════════════════════════════════════════════════════════════════
-- § Strided residual-projection block (the stage-start downsampling block)
-- ════════════════════════════════════════════════════════════════

/-- **Strided basic-block body VJP** `F = convBn₂(stride 1) ∘ convBnRelu₁(stride 2)`
    (channels `ic → oc`, spatial `2h×2w → h×w`). The strided peer of
    `resblock_body_has_vjp_at`: inner downsampling conv→bn→relu (needs `h_smooth₁`),
    outer stride-1 conv→bn (everywhere); two `vjp_comp_at`. -/
noncomputable def resblock_bodyStrided_has_vjp_at {ic oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ : ℝ) (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0) :
    HasVJPAt
      ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v := by
  have hconv1_diff : Differentiable ℝ
      (flatConvStride2 W₁ b₁ : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W₁ b₁
  have step1 : HasVJPAt
      (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁) v :=
    convBnReluStrided_has_vjp_at W₁ b₁ ε₁ γ₁ β₁ hε₁ v h_smooth₁
  have step1_diff : DifferentiableAt ℝ
      (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁) v := by
    apply DifferentiableAt.comp
    · exact relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth₁
    · exact ((bnForward_differentiable (oc * h * w) ε₁ γ₁ β₁ hε₁).comp hconv1_diff) v
  exact vjp_comp_at
    (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)
    (bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) v
    step1_diff
    ((convBn_differentiable W₂ b₂ ε₂ γ₂ β₂ hε₂) _)
    step1
    ((convBn_has_vjp W₂ b₂ ε₂ γ₂ β₂ hε₂).toHasVJPAt _)

/-- **Strided basic-block body is `DifferentiableAt`** at a smooth point. -/
theorem resblock_bodyStrided_differentiableAt {ic oc h w kH₁ kW₁ kH₂ kW₂ : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ : ℝ) (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0) :
    DifferentiableAt ℝ
      ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
        (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v := by
  have hconv1_diff : Differentiable ℝ
      (flatConvStride2 W₁ b₁ : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w)) :=
    flatConvStride2_differentiable W₁ b₁
  apply DifferentiableAt.comp
  · exact (convBn_differentiable W₂ b₂ ε₂ γ₂ β₂ hε₂) _
  · apply DifferentiableAt.comp
    · exact relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth₁
    · exact ((bnForward_differentiable (oc * h * w) ε₁ γ₁ β₁ hε₁).comp hconv1_diff) v

/-- **Full strided residual-projection block VJP** — the block that opens each
    ResNet-34 downsampling stage: `relu( proj(x) + F(x) )` where both the body's
    first conv `W₁` and the 1×1 projection skip `Wp` are stride-2 (so `ic→oc`,
    `2h×2w → h×w`), and the body's second conv `W₂` is stride-1. Built via
    `residualProj_has_vjp_at` (fan-in of the strided proj `convBnStrided` and the
    strided body) then a final `vjp_comp_at` with the post-add ReLU. The strided
    peer of `resblockProj_has_vjp_at`. -/
noncomputable def rblkPStrided_has_vjp_at
    {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp : ℝ)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp) v k)
      + ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v k
        ≠ 0) :
    HasVJPAt
      (relu (oc * h * w) ∘
        residualProj
          (bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp)
          ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
            (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁))) v := by
  let proj : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
    bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp
  let F : Vec (ic * (2 * h) * (2 * w)) → Vec (oc * h * w) :=
    (bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
      (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)
  show HasVJPAt (relu (oc * h * w) ∘ residualProj proj F) v
  have hproj_diff : DifferentiableAt ℝ proj v :=
    (convBnStrided_differentiable Wp bp εp γp βp hεp) v
  have hproj_vjp : HasVJPAt proj v :=
    (convBnStrided_has_vjp Wp bp εp γp βp hεp).toHasVJPAt v
  have hF_diff : DifferentiableAt ℝ F v :=
    resblock_bodyStrided_differentiableAt W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hF : HasVJPAt F v :=
    resblock_bodyStrided_has_vjp_at W₁ b₁ W₂ b₂ ε₁ γ₁ β₁ ε₂ γ₂ β₂ hε₁ hε₂ v h_smooth₁
  have hres : HasVJPAt (residualProj proj F) v :=
    residualProj_has_vjp_at proj F v hproj_diff hF_diff hproj_vjp hF
  have hres_diff : DifferentiableAt ℝ (residualProj proj F) v :=
    DifferentiableAt.add hproj_diff hF_diff
  have h_smooth_res : ∀ k, residualProj proj F v k ≠ 0 := h_smooth_out
  exact vjp_comp_at (residualProj proj F) (relu (oc * h * w)) v
    hres_diff
    (relu_differentiableAt_of_smooth (oc * h * w) _ h_smooth_res)
    hres
    (relu_has_vjp_at (oc * h * w) _ h_smooth_res)

/-- **Strided residual-projection block VJP correctness** (ℝ-headline): the
    downsampling block's backward equals the `pdiv`-Jacobian of
    `relu ∘ residualProj (strided proj) (strided body)`. -/
theorem rblkPStrided_has_vjp_at_correct
    {ic oc h w kH₁ kW₁ kH₂ kW₂ kHp kWp : Nat}
    (W₁ : Kernel4 oc ic kH₁ kW₁) (b₁ : Vec oc)
    (W₂ : Kernel4 oc oc kH₂ kW₂) (b₂ : Vec oc)
    (Wp : Kernel4 oc ic kHp kWp) (bp : Vec oc)
    (ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp : ℝ)
    (hε₁ : 0 < ε₁) (hε₂ : 0 < ε₂) (hεp : 0 < εp)
    (v : Vec (ic * (2 * h) * (2 * w)))
    (h_smooth₁ : ∀ k, bnForward (oc * h * w) ε₁ γ₁ β₁ (flatConvStride2 W₁ b₁ v) k ≠ 0)
    (h_smooth_out : ∀ k,
      ((bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp) v k)
      + ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
          (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)) v k
        ≠ 0)
    (dy : Vec (oc * h * w)) (i : Fin (ic * (2 * h) * (2 * w))) :
    (rblkPStrided_has_vjp_at W₁ b₁ W₂ b₂ Wp bp ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp
        hε₁ hε₂ hεp v h_smooth₁ h_smooth_out).backward dy i
      = ∑ j : Fin (oc * h * w),
          pdiv (relu (oc * h * w) ∘
            residualProj
              (bnForward (oc * h * w) εp γp βp ∘ flatConvStride2 Wp bp)
              ((bnForward (oc * h * w) ε₂ γ₂ β₂ ∘ flatConv W₂ b₂) ∘
                (relu (oc * h * w) ∘ bnForward (oc * h * w) ε₁ γ₁ β₁ ∘ flatConvStride2 W₁ b₁)))
            v i j * dy j :=
  (rblkPStrided_has_vjp_at W₁ b₁ W₂ b₂ Wp bp ε₁ γ₁ β₁ ε₂ γ₂ β₂ εp γp βp
      hε₁ hε₂ hεp v h_smooth₁ h_smooth_out).correct dy i

end Proofs
