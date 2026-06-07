import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.MnistCNN
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

-- ════════════════════════════════════════════════════════════════
-- § Milestone B7 — concrete-instance discharge (non-vacuity)
--
-- `resnet34_has_vjp_at` is *conditional*: parametric over abstract
-- `stem`/`down`/`ids`/`gap`/`dense` with smoothness/no-tie hypotheses. This
-- section instantiates all of it at concrete 1-channel, 32×32 dims with the
-- verified components (`convBnReluStrided` stem, `rblkPStrided` downsamplers,
-- `resblock` identity blocks ×(3+4+6+3), `maxPoolFlat`, `globalAvgPoolFlat`,
-- `dense`) and discharges every hypothesis — the unconditional headline
-- `resnet34Concrete_has_vjp_correct`, the ResNet-34 peer of
-- `CnnConcrete.cnnConcrete_has_vjp_correct`.
--
-- TWO dimension-robust tricks make the 256/1024-element discharge tractable
-- (no `norm_num` over thousand-element BN sums):
--   1. `bnForward_lb` — `bn ≥ β − |γ|·√n` from `(vₖ−μ)² ≤ Σ(vⱼ−μ)² = n·σ²`, so
--      a large stem `β` (here 20 > √256) forces `bn > 0` (ReLU = id ⇒ injective).
--   2. zero-weight blocks: every non-stem conv has a zero kernel, so its BN input
--      is the constant `0` and `bnForward_const` collapses it to `β` — the body of
--      every residual block is the *constant* `β₂`, and `relu(β₂ + activation) > 0`.
-- ════════════════════════════════════════════════════════════════

/-- `istd = 1/√(σ²+ε) > 0` (variance ≥ 0, `ε > 0`). -/
theorem bnIstd_pos {n : Nat} (v : Vec n) (ε : ℝ) (hε : 0 < ε) : 0 < bnIstd n v ε := by
  unfold bnIstd
  have hvar : 0 ≤ bnVar n v := by
    unfold bnVar; exact div_nonneg (Finset.sum_nonneg (fun _ _ => mul_self_nonneg _)) (by positivity)
  exact one_div_pos.mpr (Real.sqrt_pos.mpr (by linarith))

/-- **BN of an injective vector is injective** when `γ ≠ 0`: `bn` is the strictly
    monotone affine map `γ·istd·(· − μ) + β` (`istd > 0`), so it preserves the
    distinctness needed for the stem's maxpool to have no ties. -/
theorem bnForward_injective {n : Nat} (ε γ β : ℝ) (hε : 0 < ε) (hγ : γ ≠ 0)
    {v : Vec n} (hv : Function.Injective v) :
    Function.Injective (bnForward n ε γ β v) := by
  intro a b hab
  have hist : bnIstd n v ε ≠ 0 := (bnIstd_pos v ε hε).ne'
  simp only [bnForward, bnXhat] at hab
  have h1 : γ * ((v a - bnMean n v) * bnIstd n v ε) = γ * ((v b - bnMean n v) * bnIstd n v ε) := by
    linarith
  have h2 := mul_right_cancel₀ hist (mul_left_cancel₀ hγ h1)
  exact hv (by linarith)

/-- Each normalized coordinate is bounded: `x̂ₖ² ≤ n`. Proof: `istd² = 1/(σ²+ε)` and
    `(vₖ−μ)² ≤ Σⱼ(vⱼ−μ)² = n·σ² ≤ n·(σ²+ε)`. -/
theorem bnXhat_sq_le {n : Nat} (ε : ℝ) (hε : 0 < ε) (v : Vec n) (k : Fin n) :
    (bnXhat n ε v k) ^ 2 ≤ (n : ℝ) := by
  have hn : 0 < n := Nat.lt_of_le_of_lt (Nat.zero_le _) k.isLt
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr hn.ne'
  set μ := bnMean n v with hμ
  have hvar_nonneg : 0 ≤ bnVar n v :=
    div_nonneg (Finset.sum_nonneg (fun _ _ => mul_self_nonneg _)) (by positivity)
  have hpos : 0 < bnVar n v + ε := by linarith
  have histd_sq : bnIstd n v ε * bnIstd n v ε = 1 / (bnVar n v + ε) := by
    unfold bnIstd; rw [div_mul_div_comm, one_mul, Real.mul_self_sqrt (le_of_lt hpos)]
  have hterm : (v k - μ) * (v k - μ) ≤ ∑ j : Fin n, (v j - μ) * (v j - μ) :=
    Finset.single_le_sum (f := fun j => (v j - μ) * (v j - μ))
      (fun j _ => mul_self_nonneg _) (Finset.mem_univ k)
  have hsum : (n : ℝ) * bnVar n v = ∑ j : Fin n, (v j - μ) * (v j - μ) := by
    unfold bnVar; rw [← hμ]; field_simp
  have hbound : (v k - μ) * (v k - μ) ≤ (n : ℝ) * (bnVar n v + ε) := by
    have : (v k - μ) * (v k - μ) ≤ (n : ℝ) * bnVar n v := by rw [hsum]; exact hterm
    nlinarith [Nat.cast_nonneg (α := ℝ) n, le_of_lt hε]
  have hxsq : (bnXhat n ε v k) ^ 2 = ((v k - μ) * (v k - μ)) * (1 / (bnVar n v + ε)) := by
    simp only [bnXhat, ← hμ]; rw [← histd_sq]; ring
  rw [hxsq, mul_one_div, div_le_iff₀ hpos]; exact hbound

/-- **Dimension-robust BN lower bound** `β − |γ|·√n ≤ bn`, with no mean/variance
    computation. Lets a large stem `β` force `bn > 0` over a 256-element BN. -/
theorem bnForward_lb {n : Nat} (ε γ β : ℝ) (hε : 0 < ε) (v : Vec n) (k : Fin n) :
    β - |γ| * Real.sqrt (n : ℝ) ≤ bnForward n ε γ β v k := by
  have hsq := bnXhat_sq_le ε hε v k
  have habs : |bnXhat n ε v k| ≤ Real.sqrt (n : ℝ) := by
    rw [← Real.sqrt_sq_eq_abs]; exact Real.sqrt_le_sqrt hsq
  have hmul : |γ * bnXhat n ε v k| ≤ |γ| * Real.sqrt (n : ℝ) := by
    rw [abs_mul]; exact mul_le_mul_of_nonneg_left habs (abs_nonneg γ)
  have hge : -(|γ| * Real.sqrt (n : ℝ)) ≤ γ * bnXhat n ε v k :=
    le_trans (neg_le_neg hmul) (neg_abs_le _)
  simp only [bnForward]; linarith

/-- **The strided decimation index is injective** (distinct output cells map to
    distinct even input cells) — so `decimateFlat` of an injective vector is
    injective, the keystone of the strided stem's maxpool no-tie discharge. -/
theorem decimateIdx_injective (oc h w : Nat) :
    Function.Injective (decimateIdx oc h w) := by
  intro k₁ k₂ heq
  simp only [decimateIdx] at heq
  obtain ⟨hA, hB⟩ := Prod.mk.inj (finProdFinEquiv.injective heq)
  -- hB : the doubled `w`-coordinates agree ⇒ the `w`-coordinates agree
  have hp2 : (finProdFinEquiv.symm k₁).2 = (finProdFinEquiv.symm k₂).2 := by
    have : 2 * (finProdFinEquiv.symm k₁).2.val = 2 * (finProdFinEquiv.symm k₂).2.val :=
      Fin.mk.inj_iff.mp hB
    exact Fin.ext (by omega)
  -- hA : finProdFinEquiv (q₁.1, double q₁.2) = finProdFinEquiv (q₂.1, double q₂.2)
  obtain ⟨hA1, hA2⟩ := Prod.mk.inj (finProdFinEquiv.injective hA)
  have hq2 : (finProdFinEquiv.symm (finProdFinEquiv.symm k₁).1).2
           = (finProdFinEquiv.symm (finProdFinEquiv.symm k₂).1).2 := by
    have : 2 * (finProdFinEquiv.symm (finProdFinEquiv.symm k₁).1).2.val
         = 2 * (finProdFinEquiv.symm (finProdFinEquiv.symm k₂).1).2.val :=
      Fin.mk.inj_iff.mp hA2
    exact Fin.ext (by omega)
  -- q₁ = q₂ (both components), so p₁.1 = p₂.1
  have hq : finProdFinEquiv.symm (finProdFinEquiv.symm k₁).1
          = finProdFinEquiv.symm (finProdFinEquiv.symm k₂).1 := Prod.ext hA1 hq2
  have hp1 : (finProdFinEquiv.symm k₁).1 = (finProdFinEquiv.symm k₂).1 :=
    finProdFinEquiv.symm.injective hq
  exact finProdFinEquiv.symm.injective (Prod.ext hp1 hp2)

/-- `decimateFlat` of an injective vector is injective. -/
theorem decimateFlat_injective (oc h w : Nat) {x : Vec (oc * (2 * h) * (2 * w))}
    (hx : Function.Injective x) : Function.Injective (decimateFlat oc h w x) := by
  intro a b hab
  exact decimateIdx_injective oc h w (hx hab)

-- ── Generic zero-weight building blocks (single channel) ──────────
-- Every non-stem conv in the concrete net has a zero kernel/bias, so its BN
-- input is constant `0` and `bnForward_const` collapses it to `β`. The residual
-- body is therefore the constant `β₂`, and `relu(β₂ + activation) ≥ 0`.

/-- The single-channel zero 1×1 kernel and zero bias. -/
noncomputable def Zk : Kernel4 1 1 1 1 := fun _ _ _ _ => 0
noncomputable def Zb : Vec 1 := fun _ => 0

/-- BN of a constant vector is its shift `β` (local copy of `MobileNetV2`'s
    `bnForward_const`, re-proved here to avoid an inter-architecture import). -/
theorem bnForward_const_eq {n : Nat} (hn : 0 < n) (ε γ β c : ℝ) :
    bnForward n ε γ β (fun _ => c) = (fun _ => β) := by
  have hmean : bnMean n (fun _ : Fin n => c) = c := by
    unfold bnMean
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_comm,
        mul_div_assoc, div_self (Nat.cast_ne_zero.mpr hn.ne'), mul_one]
  funext i; simp only [bnForward, bnXhat]; rw [hmean]; ring

/-- A conv with everywhere-zero kernel/bias maps anything to `0` (local copy of
    `MobileNetV2`'s `flatConv_eq_zero`). -/
theorem flatConv_zero {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (hW : ∀ o c kh kw, W o c kh kw = 0) (hb : ∀ o, b o = 0) (v : Vec (ic * h * w)) :
    flatConv (h := h) (w := w) W b v = (fun _ => (0:ℝ)) := by
  funext k; simp [flatConv, conv2d, Tensor3.flatten, hW, hb]

/-- ReLU output is always nonnegative. -/
theorem relu_nonneg (n : Nat) (v : Vec n) (k : Fin n) : 0 ≤ relu n v k := by
  simp only [relu]
  by_cases h : v k > 0
  · rw [if_pos h]; exact le_of_lt h
  · rw [if_neg h]

/-- ReLU is the identity on a positive constant vector. -/
theorem relu_const_pos (n : Nat) (c : ℝ) (hc : 0 < c) : relu n (fun _ => c) = (fun _ => c) := by
  funext i; simp only [relu]; rw [if_pos hc]

/-- A stride-2 conv with zero kernel/bias maps anything to `0` (decimate of `0`). -/
theorem flatConvStride2_eq_zero {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (hW : ∀ o c kh kw, W o c kh kw = 0) (hb : ∀ o, b o = 0) (a : Vec (ic * (2 * h) * (2 * w))) :
    flatConvStride2 W b a = (fun _ => (0:ℝ)) := by
  unfold flatConvStride2
  simp only [Function.comp_apply]
  rw [flatConv_zero W b hW hb]
  funext k; simp [decimateFlat]

-- ── The identity residual block (zero weights, BN (1,0,1)) ────────

/-- A single-channel identity residual block with zero weights and BN `(ε,γ,β)=(1,0,1)`:
    `relu( x + bn₂(conv₂(relu(bn₁(conv₁ x)))) )`. The body collapses to the constant
    `β₂ = 1`, so the block is `relu(1 + x)`. -/
noncomputable def idBlk (h w : Nat) : Vec (1 * h * w) → Vec (1 * h * w) :=
  relu (1 * h * w) ∘ residual
    ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb))

/-- The identity block's body is the constant `1` (every conv is zero ⇒ every BN is its
    shift; the outer zero conv ignores its input). -/
theorem idBlk_body_const (h w : Nat) (hhw : 0 < 1 * h * w) (a : Vec (1 * h * w)) :
    ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb)) a = (fun _ => (1:ℝ)) := by
  simp only [Function.comp_apply]
  rw [flatConv_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const_eq hhw]

/-- The identity block output is nonnegative (it is a ReLU). -/
theorem idBlk_nonneg (h w : Nat) (a : Vec (1 * h * w)) (k : Fin (1 * h * w)) :
    0 ≤ idBlk h w a k := relu_nonneg (1 * h * w) _ k

/-- The identity block has a VJP at any nonnegative activation: `bn₁`-input is constant
    (`β₁=1≠0`) and the post-add ReLU input is `1 + aₖ > 0` since `aₖ ≥ 0`. -/
noncomputable def idBlk_hasVJPAt (h w : Nat) (hhw : 0 < 1 * h * w)
    (a : Vec (1 * h * w)) (ha : ∀ k, 0 ≤ a k) : HasVJPAt (idBlk h w) a :=
  resblock_has_vjp_at (h := h) (w := w) Zk Zb Zk Zb 1 0 1 1 0 1 (by norm_num) (by norm_num) a
    (fun k => by
      rw [flatConv_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const_eq hhw]
      change (1:ℝ) ≠ 0; norm_num)
    (fun k => by
      rw [idBlk_body_const h w hhw a]; change (1:ℝ) + a k ≠ 0
      exact ne_of_gt (by linarith [ha k]))

/-- The identity block is differentiable at any nonnegative activation. -/
theorem idBlk_diffAt (h w : Nat) (hhw : 0 < 1 * h * w)
    (a : Vec (1 * h * w)) (ha : ∀ k, 0 ≤ a k) : DifferentiableAt ℝ (idBlk h w) a := by
  have hsm₁ : ∀ k, bnForward (1 * h * w) 1 0 1 (flatConv Zk Zb a) k ≠ 0 := fun k => by
    rw [flatConv_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const_eq hhw]
    change (1:ℝ) ≠ 0; norm_num
  have hF_diff : DifferentiableAt ℝ
      ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
        (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb)) a :=
    resblock_body_differentiableAt (h := h) (w := w) Zk Zb Zk Zb 1 0 1 1 0 1
      (by norm_num) (by norm_num) a hsm₁
  have hsm_res : ∀ k, residual
      ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
        (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb)) a k ≠ 0 := fun k => by
    show ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
        (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb)) a k + a k ≠ 0
    rw [idBlk_body_const h w hhw a]; change (1:ℝ) + a k ≠ 0
    exact ne_of_gt (by linarith [ha k])
  show DifferentiableAt ℝ (relu (1 * h * w) ∘ residual
    ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb))) a
  exact (relu_differentiableAt_of_smooth (1 * h * w) _ hsm_res).comp a
    (DifferentiableAt.add hF_diff differentiable_id.differentiableAt)

/-- A chain of identity blocks fed a nonnegative base stays nonnegative (each block is a
    ReLU; the base only matters for the empty chain). -/
theorem chainComp_replicate_idBlk_nonneg (h w : Nat) (base : Vec (1 * h * w))
    (hbase : ∀ k, 0 ≤ base k) :
    ∀ (j : Nat) (k : Fin (1 * h * w)), 0 ≤ chainComp (List.replicate j (idBlk h w)) base k
  | 0, k => hbase k
  | j + 1, k => by
      rw [List.replicate_succ, chainComp_cons]
      exact idBlk_nonneg h w _ k

/-- `ChainData` for `j` stacked identity blocks at a nonnegative base — every running
    activation is a ReLU output (or the base), so each block's smooth-point hypotheses hold. -/
noncomputable def idChainData (h w : Nat) (hhw : 0 < 1 * h * w) (base : Vec (1 * h * w))
    (hbase : ∀ k, 0 ≤ base k) :
    ∀ (j : Nat), ChainData base (List.replicate j (idBlk h w))
  | 0 => PUnit.unit
  | j + 1 => by
      rw [List.replicate_succ]
      refine ⟨idBlk_diffAt h w hhw _ ?_, idBlk_hasVJPAt h w hhw _ ?_, idChainData h w hhw base hbase j⟩
      · exact chainComp_replicate_idBlk_nonneg h w base hbase j
      · exact chainComp_replicate_idBlk_nonneg h w base hbase j

-- ── The strided downsampling/projection block (zero weights) ──────

/-- A single-channel strided projection block with zero weights and BN `(1,0,1)`:
    `relu( proj(x) + bn₂(conv₂(relu(bn₁(conv₁ x)))) )`, both `conv₁` and `proj` stride-2.
    Body and projection both collapse to the constant `1`, so the post-add ReLU input is
    `1 + 1 = 2` everywhere — unconditional (no activation-sign assumption). -/
noncomputable def downBlk (h w : Nat) : Vec (1 * (2 * h) * (2 * w)) → Vec (1 * h * w) :=
  relu (1 * h * w) ∘ residualProj
    (bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)
    ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb))

/-- The strided block's projection path collapses to the constant `1`. -/
theorem downBlk_proj_const (h w : Nat) (hhw : 0 < 1 * h * w) (a : Vec (1 * (2 * h) * (2 * w))) :
    (bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb) a = (fun _ => (1:ℝ)) := by
  simp only [Function.comp_apply]
  rw [flatConvStride2_eq_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]

/-- The strided block's residual body collapses to the constant `1`. -/
theorem downBlk_body_const (h w : Nat) (hhw : 0 < 1 * h * w) (a : Vec (1 * (2 * h) * (2 * w))) :
    ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)) a = (fun _ => (1:ℝ)) := by
  simp only [Function.comp_apply]
  rw [flatConv_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl), bnForward_const_eq hhw]

/-- The strided block output is nonnegative (a ReLU). -/
theorem downBlk_nonneg (h w : Nat) (a : Vec (1 * (2 * h) * (2 * w))) (k : Fin (1 * h * w)) :
    0 ≤ downBlk h w a k := relu_nonneg (1 * h * w) _ k

/-- The strided block has a VJP at every point (smoothness is unconditional: both paths are
    constant `1`, so the post-add ReLU input is `2 ≠ 0`). -/
noncomputable def downBlk_hasVJPAt (h w : Nat) (hhw : 0 < 1 * h * w)
    (a : Vec (1 * (2 * h) * (2 * w))) : HasVJPAt (downBlk h w) a :=
  rblkPStrided_has_vjp_at (h := h) (w := w) Zk Zb Zk Zb Zk Zb 1 0 1 1 0 1 1 0 1
    (by norm_num) (by norm_num) (by norm_num) a
    (fun k => by
      rw [flatConvStride2_eq_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
      change (1:ℝ) ≠ 0; norm_num)
    (fun k => by
      rw [downBlk_proj_const h w hhw a, downBlk_body_const h w hhw a]
      change (1:ℝ) + 1 ≠ 0; norm_num)

/-- The strided block is differentiable at every point. -/
theorem downBlk_diffAt (h w : Nat) (hhw : 0 < 1 * h * w)
    (a : Vec (1 * (2 * h) * (2 * w))) : DifferentiableAt ℝ (downBlk h w) a := by
  have hsm₁ : ∀ k, bnForward (1 * h * w) 1 0 1 (flatConvStride2 Zk Zb a) k ≠ 0 := fun k => by
    rw [flatConvStride2_eq_zero Zk Zb (fun _ _ _ _ => rfl) (fun _ => rfl) a, bnForward_const_eq hhw]
    change (1:ℝ) ≠ 0; norm_num
  have hproj_diff : DifferentiableAt ℝ (bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb) a :=
    convBnStrided_differentiable (h := h) (w := w) Zk Zb 1 0 1 (by norm_num) a
  have hF_diff : DifferentiableAt ℝ
      ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
        (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)) a :=
    resblock_bodyStrided_differentiableAt (h := h) (w := w) Zk Zb Zk Zb 1 0 1 1 0 1
      (by norm_num) (by norm_num) a hsm₁
  have hsm_res : ∀ k, residualProj
      (bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)
      ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
        (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)) a k ≠ 0 := fun k => by
    show (bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb) a k
        + ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
            (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)) a k ≠ 0
    rw [downBlk_proj_const h w hhw a, downBlk_body_const h w hhw a]
    change (1:ℝ) + 1 ≠ 0; norm_num
  show DifferentiableAt ℝ (relu (1 * h * w) ∘ residualProj
    (bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb)
    ((bnForward (1 * h * w) 1 0 1 ∘ flatConv Zk Zb) ∘
      (relu (1 * h * w) ∘ bnForward (1 * h * w) 1 0 1 ∘ flatConvStride2 Zk Zb))) a
  exact (relu_differentiableAt_of_smooth (1 * h * w) _ hsm_res).comp a
    (DifferentiableAt.add hproj_diff hF_diff)

-- ════════════════════════════════════════════════════════════════
-- § The concrete ResNet-34 instance (1 channel, 32×32 input)
-- ════════════════════════════════════════════════════════════════

namespace ResNet34Concrete

/-- Stem: a 1×1 **identity** conv (so `flatConvStride2` collapses to decimation),
    BN `(ε,γ,β) = (1,1,20)`, ReLU. `β = 20 > √256` forces `bn > 0` (`bnForward_lb`),
    so ReLU is the identity and the stem output stays injective — the maxpool no-tie. -/
noncomputable def Ws : Kernel4 1 1 1 1 := fun _ _ _ _ => 1
noncomputable def bs : Vec 1 := fun _ => 0
/-- Positional (hence injective) input: `X i = i`. -/
noncomputable def X : Vec (1 * (2 * 16) * (2 * 16)) := fun i => (i.val : ℝ)
noncomputable def Wd : Mat 1 2 := fun _ _ => 0
noncomputable def bd : Vec 2 := fun _ => 0

/-- The stem `relu ∘ bn ∘ conv_stride2` at 1ch, 16×16 output. -/
noncomputable def stem : Vec (1 * (2 * 16) * (2 * 16)) → Vec (1 * 16 * 16) :=
  relu (1 * 16 * 16) ∘ bnForward (1 * 16 * 16) 1 1 20 ∘ flatConvStride2 Ws bs

theorem X_inj : Function.Injective X := by
  intro a b hab
  simp only [X] at hab
  exact Fin.ext (by exact_mod_cast hab)

/-- The 1×1 identity stem conv is the identity on the flattened input. -/
theorem flatConv_id_X : flatConv (h := 2 * 16) (w := 2 * 16) Ws bs X = X := by
  have hc : conv2d Ws bs (Tensor3.unflatten X) = Tensor3.unflatten X := by
    funext o hi wi
    rw [conv2d_1x1]
    simp only [bs, Ws, Fin.sum_univ_one, one_mul, zero_add]
    congr 1
    exact (Fin.fin_one_eq_zero o).symm ▸ rfl
  simp only [flatConv, hc, Tensor3.flatten_unflatten]

/-- Stride-2 identity conv = decimation. -/
theorem stem_conv_eq : flatConvStride2 Ws bs X = decimateFlat 1 16 16 X := by
  unfold flatConvStride2
  simp only [Function.comp_apply]
  rw [flatConv_id_X]

/-- The stem's BN output is strictly positive: `bn ≥ 20 − √256 = 4 > 0`. -/
theorem stem_bn_pos : ∀ k, 0 < bnForward (1 * 16 * 16) 1 1 20 (flatConvStride2 Ws bs X) k := by
  intro k
  have hlb := bnForward_lb (n := 1 * 16 * 16) 1 1 20 (by norm_num) (flatConvStride2 Ws bs X) k
  have hsqrt : Real.sqrt ((1 * 16 * 16 : ℕ) : ℝ) = 16 := by
    rw [show ((1 * 16 * 16 : ℕ) : ℝ) = 256 by norm_num, show (256:ℝ) = 16 ^ 2 by norm_num,
        Real.sqrt_sq (by norm_num : (0:ℝ) ≤ 16)]
  rw [abs_one, hsqrt] at hlb
  linarith

/-- Hence the whole stem output is strictly positive (ReLU = identity). -/
theorem stem_pos : ∀ k, 0 < stem X k := by
  intro k
  have hbn := stem_bn_pos k
  show 0 < (relu (1 * 16 * 16) ∘ bnForward (1 * 16 * 16) 1 1 20 ∘ flatConvStride2 Ws bs) X k
  simp only [Function.comp_apply, relu]
  rw [if_pos hbn]; exact hbn

/-- The stem output is injective: `bn` of the injective decimated input is injective
    (`bnForward_injective`, `γ = 1 ≠ 0`) and the ReLU is the identity (`stem_pos`). -/
theorem stem_inj : Function.Injective (stem X) := by
  have hstemeq : stem X = bnForward (1 * 16 * 16) 1 1 20 (flatConvStride2 Ws bs X) := by
    show (relu (1 * 16 * 16) ∘ bnForward (1 * 16 * 16) 1 1 20 ∘ flatConvStride2 Ws bs) X = _
    simp only [Function.comp_apply]
    exact relu_id_of_pos stem_bn_pos
  rw [hstemeq, stem_conv_eq]
  exact bnForward_injective 1 1 20 (by norm_num) (by norm_num)
    (decimateFlat_injective 1 16 16 X_inj)

/-- The maxpool input (`= stem X`) is positionally injective ⇒ `MaxPool2Smooth`. -/
theorem stem_maxpool_smooth :
    MaxPool2Smooth (Tensor3.unflatten (stem X) : Tensor3 1 (2 * 8) (2 * 8)) := by
  apply maxPool2Smooth_of_injective
  intro ci r r' s s' heq
  simp only [Tensor3.unflatten] at heq
  have h2 := finProdFinEquiv.injective (stem_inj heq)
  have h5 := finProdFinEquiv.injective (congrArg Prod.fst h2)
  exact ⟨congrArg Prod.snd h5, congrArg Prod.snd h2⟩

/-- The maxpool output is strictly positive (max of positive stem outputs). -/
theorem mp_stem_pos : ∀ k, 0 < maxPoolFlat 1 8 8 (stem X) k := by
  intro k
  rw [maxPoolFlat]
  apply flatten_pos_of_pos
  intro ci hi wi
  apply maxPool2_pos
  intro c r s
  simp only [Tensor3.unflatten]
  exact stem_pos _

/-- The maxpool point bridge: `flatten ∘ unflatten = id` at the stem output. -/
theorem mp_point_eq :
    Tensor3.flatten (Tensor3.unflatten (stem X) : Tensor3 1 (2 * 8) (2 * 8)) = stem X :=
  Tensor3.flatten_unflatten (stem X)

/-- Maxpool VJP at the stem output (no ties via `stem_maxpool_smooth`). -/
noncomputable def hmp_vjp : HasVJPAt (maxPoolFlat 1 8 8) (stem X) := by
  have h := maxPoolFlat_has_vjp_at (Tensor3.unflatten (stem X) : Tensor3 1 (2 * 8) (2 * 8))
    stem_maxpool_smooth
  rwa [mp_point_eq] at h

/-- Maxpool differentiability at the stem output. -/
theorem hmp_diff : DifferentiableAt ℝ (maxPoolFlat 1 8 8) (stem X) := by
  have h := maxPoolFlat_differentiableAt (Tensor3.unflatten (stem X) : Tensor3 1 (2 * 8) (2 * 8))
    stem_maxpool_smooth (by norm_num) (by norm_num) (by norm_num)
  rwa [mp_point_eq] at h

/-- ResNet-34's four stages: `3 + 4 + 6 + 3 = 16` identity blocks. -/
noncomputable def ids1 : List (Vec (1 * 8 * 8) → Vec (1 * 8 * 8)) := List.replicate 3 (idBlk 8 8)
noncomputable def ids2 : List (Vec (1 * 4 * 4) → Vec (1 * 4 * 4)) := List.replicate 4 (idBlk 4 4)
noncomputable def ids3 : List (Vec (1 * 2 * 2) → Vec (1 * 2 * 2)) := List.replicate 6 (idBlk 2 2)
noncomputable def ids4 : List (Vec (1 * 1 * 1) → Vec (1 * 1 * 1)) := List.replicate 3 (idBlk 1 1)

/-- The concrete whole-network forward map: `dense ∘ gap ∘ (stage₄…₁) ∘ maxpool ∘ stem`,
    a real 34-layer ResNet (strided stem + 3 strided downsamplers + 16 identity blocks +
    GAP + dense) at 1 channel / 32×32. -/
noncomputable def fwd : Vec (1 * (2 * 16) * (2 * 16)) → Vec 2 :=
  dense Wd bd ∘ globalAvgPoolFlat 1 1 1 ∘ chainComp ids4 ∘ downBlk 1 1 ∘ chainComp ids3 ∘
    downBlk 2 2 ∘ chainComp ids2 ∘ downBlk 4 4 ∘ chainComp ids1 ∘ maxPoolFlat 1 8 8 ∘ stem

/-- **Whole-network VJP for a concrete ResNet-34** — every smoothness/no-tie hypothesis of
    `resnet34_has_vjp_at` discharged. The strided identity stem yields distinct positive BN
    outputs (so the maxpool has no ties via `stem_maxpool_smooth`); every residual block uses
    zero weights, so its body is the constant `1` (`bnForward_const_eq`) and the post-add ReLU
    input is `1 + activation > 0` (identity blocks, `activation ≥ 0`) or `2` (downsamplers). -/
noncomputable def resnet34Concrete_has_vjp_at : HasVJPAt fwd X :=
  resnet34_has_vjp_at stem (maxPoolFlat 1 8 8) ids1 (downBlk 4 4) ids2 (downBlk 2 2) ids3
    (downBlk 1 1) ids4 (globalAvgPoolFlat 1 1 1) (dense Wd bd) X
    ⟨convBnReluStrided_has_vjp_at Ws bs 1 1 20 (by norm_num) X (fun k => ne_of_gt (stem_bn_pos k)),
     DifferentiableAt.comp X
       (relu_differentiableAt_of_smooth (1 * 16 * 16) _ (fun k => ne_of_gt (stem_bn_pos k)))
       ((convBnStrided_differentiable Ws bs 1 1 20 (by norm_num)) X)⟩
    ⟨hmp_vjp, hmp_diff⟩
    (idChainData 8 8 (by norm_num) (maxPoolFlat 1 8 8 (stem X))
      (fun k => le_of_lt (mp_stem_pos k)) 3)
    ⟨downBlk_hasVJPAt 4 4 (by norm_num) _, downBlk_diffAt 4 4 (by norm_num) _⟩
    (idChainData 4 4 (by norm_num) _ (fun k => downBlk_nonneg 4 4 _ k) 4)
    ⟨downBlk_hasVJPAt 2 2 (by norm_num) _, downBlk_diffAt 2 2 (by norm_num) _⟩
    (idChainData 2 2 (by norm_num) _ (fun k => downBlk_nonneg 2 2 _ k) 6)
    ⟨downBlk_hasVJPAt 1 1 (by norm_num) _, downBlk_diffAt 1 1 (by norm_num) _⟩
    (idChainData 1 1 (by norm_num) _ (fun k => downBlk_nonneg 1 1 _ k) 3)
    ⟨(globalAvgPoolFlat_has_vjp 1 1 1).toHasVJPAt _, (globalAvgPoolFlat_differentiable 1 1 1) _⟩
    ⟨(dense_has_vjp Wd bd).toHasVJPAt _, (dense_differentiable Wd bd) _⟩

/-- **Public unconditional correctness theorem** — the concrete ResNet-34's backward equals
    the `pdiv`-Jacobian VJP, no hypotheses. The ResNet-34 peer of
    `CnnConcrete.cnnConcrete_has_vjp_correct`. -/
theorem resnet34Concrete_has_vjp_correct (dy : Vec 2) (i : Fin (1 * (2 * 16) * (2 * 16))) :
    resnet34Concrete_has_vjp_at.backward dy i = ∑ j : Fin 2, pdiv fwd X i j * dy j :=
  resnet34Concrete_has_vjp_at.correct dy i

end ResNet34Concrete

end Proofs
