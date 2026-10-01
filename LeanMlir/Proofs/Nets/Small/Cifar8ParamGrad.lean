import LeanMlir.Proofs.Nets.Small.Cifar8StepTieG
import LeanMlir.Proofs.Nets.Small.CifarParamGrad

/-! # The 8-conv CIFAR CNN — every gradient node IS the loss's derivative, up to pool twins

`cifar8_train_step_tiedG` states the 22 un-fused gradient nodes the packed `cifar8w_*` arms emit,
each at the cotangent the chain threads to it. `cifar8_net_lossGrad` states that each node, at the
chain cotangent, is the gradient of the loss in that parameter, for any loss `L` of the logits with
gradient `g` there; `cifar8_net_lossGrad_CE` instantiates it at the softmax cross-entropy the
render emits.

The four pools are handled as the 2-stage net's two (`CifarFold.cifar_net_lossGrad`): each pool's
clause allows ties between twins, cells equal at every weight upstream of that pool
(`CnnFold.CnnPoolTwin`, `Cifar8PoolTwin2`, `Cifar8PoolTwin3`, `Cifar8PoolTwin4`), and each pool's
backward routes a window's cotangent to the one cell a selection names, as the rendered
`select_and_scatter` does. A stage-`s` parameter sees pools `s`…4 move; its germ rewrites them
outermost first, at the true pre-activations (`germ4` … `germ1` in the proof). `cifar8Up` is the
step between two pools' pre-activations.

**Hypotheses.** Odd kernels, every ReLU off its kink, every pool window dead or tied only between
twins, each selection naming a maximum of every window (`Cifar8LossSmoothAt`).
**Scope.** One example (the emitted module batch-contracts; `den` is per-example).
-/

open Proofs Proofs.StableHLO Proofs.IR Proofs.SmallParamGrad Proofs.CnnFold Proofs.CifarFold

namespace Proofs.Cifar8TieG

open scoped BigOperators

/-- From one pool's pre-activation to the next pool's: ReLU, pool, conv, ReLU, conv. -/
noncomputable def cifar8Up {c c' H W kH kW : Nat} (Wc : Kernel4 c' c kH kW) (bc : Vec c')
    (Wd : Kernel4 c' c' kH kW) (bd : Vec c') (z : Vec (c * (2 * H) * (2 * W))) : Vec (c' * H * W) :=
  flatConv (h := H) (w := W) Wd bd (relu (c' * H * W) (flatConv (h := H) (w := W) Wc bc
    (maxPoolFlat c H W (relu (c * (2 * H) * (2 * W)) z))))

theorem cifar8Up_continuous {c c' H W kH kW : Nat} (Wc : Kernel4 c' c kH kW) (bc : Vec c')
    (Wd : Kernel4 c' c' kH kW) (bd : Vec c') :
    Continuous (cifar8Up (H := H) (W := W) Wc bc Wd bd) :=
  (flatConv_differentiable Wd bd).continuous.comp ((relu_continuous _).comp
    ((flatConv_differentiable Wc bc).continuous.comp
      ((maxPoolFlat_continuous _ _ _).comp (relu_continuous _))))

section Twins
variable {ic c1 c2 c3 c4 h w kH kW : Nat}

/-- The second pool's pre-activation (conv₄'s output). -/
noncomputable def cifar8Pre2 (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
    (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :
    Vec (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) :=
  cifar8Up W₃ b₃ W₄ b₄ (cnnPoolPre W₁ b₁ W₂ b₂ x)

/-- The third pool's pre-activation (conv₆'s output). -/
noncomputable def cifar8Pre3 (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
    (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :
    Vec (c3 * (2 * (2 * h)) * (2 * (2 * w))) :=
  cifar8Up W₅ b₅ W₆ b₆ (cifar8Pre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x)

/-- The fourth pool's pre-activation (conv₈'s output). -/
noncomputable def cifar8Pre4 (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
    (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w)))))) :
    Vec (c4 * (2 * h) * (2 * w)) :=
  cifar8Up W₇ b₇ W₈ b₈ (cifar8Pre3 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x)

/-- **Twins of the second pool**: equal at every weight of convs 1–4, in every channel. -/
def Cifar8PoolTwin2 (c1 c2 kH kW : Nat)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (p q : Fin (2 * (2 * (2 * h))) × Fin (2 * (2 * (2 * w)))) : Prop :=
  ∀ (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (ci : Fin c2),
    cifar8Pre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x (t3Idx ci p.1 p.2)
      = cifar8Pre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x (t3Idx ci q.1 q.2)

/-- **Twins of the third pool**: equal at every weight of convs 1–6, in every channel. -/
def Cifar8PoolTwin3 (c1 c2 c3 kH kW : Nat)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (p q : Fin (2 * (2 * h)) × Fin (2 * (2 * w))) : Prop :=
  ∀ (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (ci : Fin c3),
    cifar8Pre3 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x (t3Idx ci p.1 p.2)
      = cifar8Pre3 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x (t3Idx ci q.1 q.2)

/-- **Twins of the fourth pool**: equal at every weight of the eight convs, in every channel. -/
def Cifar8PoolTwin4 (c1 c2 c3 c4 kH kW : Nat)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (p q : Fin (2 * h) × Fin (2 * w)) : Prop :=
  ∀ (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1)
    (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (ci : Fin c4),
    cifar8Pre4 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ x (t3Idx ci p.1 p.2)
      = cifar8Pre4 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ x (t3Idx ci q.1 q.2)

end Twins

section Net
variable {ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat}

/-- **The smooth-point bundle the loss gradient needs.** Every ReLU off its kink; every window of
    each pool dead or tied only between that pool's twins; each selection naming a maximum of
    every window. -/
structure Cifar8LossSmoothAt (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1) (W₂ : Kernel4 c1 c1 kH kW)
    (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2) (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2)
    (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3) (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3)
    (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4) (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4)
    (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1) (Wa : Mat d1 d1) (ba : Vec d1)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (σ₁ : Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin c4 → Fin h → Fin w → Fin 2 × Fin 2) : Prop where
  z1 : ∀ k, flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₁ b₁ x k ≠ 0
  z2 : ∀ k, cnnPoolPre W₁ b₁ W₂ b₂ x k ≠ 0
  pool1 : MaxPool2SmoothUpTo (CnnPoolTwin c1 kH kW x)
    (Tensor3.unflatten (cnnPoolPre W₁ b₁ W₂ b₂ x) :
      Tensor3 c1 (2 * (2 * (2 * (2 * h)))) (2 * (2 * (2 * (2 * w)))))
  sel1 : PoolSelDom σ₁ (relu _ (cnnPoolPre W₁ b₁ W₂ b₂ x))
  z3 : ∀ k, flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ b₃
    (maxPoolFlat c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))) (relu _ (cnnPoolPre W₁ b₁ W₂ b₂ x))) k
      ≠ 0
  z4 : ∀ k, cifar8Pre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x k ≠ 0
  pool2 : MaxPool2SmoothUpTo (Cifar8PoolTwin2 c1 c2 kH kW x)
    (Tensor3.unflatten (cifar8Pre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x) :
      Tensor3 c2 (2 * (2 * (2 * h))) (2 * (2 * (2 * w))))
  sel2 : PoolSelDom σ₂ (relu _ (cifar8Pre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x))
  z5 : ∀ k, flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₅ b₅
    (maxPoolFlat c2 (2 * (2 * h)) (2 * (2 * w)) (relu _ (cifar8Pre2 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x))) k
      ≠ 0
  z6 : ∀ k, cifar8Pre3 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x k ≠ 0
  pool3 : MaxPool2SmoothUpTo (Cifar8PoolTwin3 c1 c2 c3 kH kW x)
    (Tensor3.unflatten (cifar8Pre3 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x) :
      Tensor3 c3 (2 * (2 * h)) (2 * (2 * w)))
  sel3 : PoolSelDom σ₃ (relu _ (cifar8Pre3 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x))
  z7 : ∀ k, flatConv (h := 2 * h) (w := 2 * w) W₇ b₇
    (maxPoolFlat c3 (2 * h) (2 * w)
      (relu _ (cifar8Pre3 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x))) k ≠ 0
  z8 : ∀ k, cifar8Pre4 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ x k ≠ 0
  pool4 : MaxPool2SmoothUpTo (Cifar8PoolTwin4 c1 c2 c3 c4 kH kW x)
    (Tensor3.unflatten (cifar8Pre4 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ x) :
      Tensor3 c4 (2 * h) (2 * w))
  sel4 : PoolSelDom σ₄ (relu _ (cifar8Pre4 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ x))
  z9 : ∀ k, dense W₉ b₉ (maxPoolFlat c4 h w
    (relu _ (cifar8Pre4 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ x))) k ≠ 0
  za : ∀ k, dense Wa ba (relu d1 (dense W₉ b₉ (maxPoolFlat c4 h w
    (relu _ (cifar8Pre4 W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ x))))) k ≠ 0

/-- **Every cifar8 gradient node is the gradient of `L`** in that parameter: the 22 un-fused nodes
    `cifar8_train_step_tiedG` states, each at the cotangent the chain threads to its layer (each
    pool routed at its selection), stated against `L` of `cifarCnn8Forward` with that one
    parameter varied. -/
def Cifar8NetLossTied (xN cotN : String) (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1)
    (Wa : Mat d1 d1) (ba : Vec d1) (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (σ₁ : Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin c4 → Fin h → Fin w → Fin 2 × Fin 2) (L : Vec nClasses → Vec 1) (g : Vec nClasses) :
    Prop :=
  let F := fun (W₁' : Kernel4 c1 ic kH kW) (b₁' : Vec c1) (W₂' : Kernel4 c1 c1 kH kW) (b₂' : Vec c1)
      (W₃' : Kernel4 c2 c1 kH kW) (b₃' : Vec c2) (W₄' : Kernel4 c2 c2 kH kW) (b₄' : Vec c2)
      (W₅' : Kernel4 c3 c2 kH kW) (b₅' : Vec c3) (W₆' : Kernel4 c3 c3 kH kW) (b₆' : Vec c3)
      (W₇' : Kernel4 c4 c3 kH kW) (b₇' : Vec c4) (W₈' : Kernel4 c4 c4 kH kW) (b₈' : Vec c4)
      (W₉' : Mat (c4 * h * w) d1) (b₉' : Vec d1) (Wa' : Mat d1 d1) (ba' : Vec d1)
      (Wb' : Mat d1 nClasses) (bb' : Vec nClasses) =>
    L (cifarCnn8Forward W₁' b₁' W₂' b₂' W₃' b₃' W₄' b₄' W₅' b₅' W₆' b₆' W₇' b₇' W₈' b₈'
      W₉' b₉' Wa' ba' Wb' bb' x)
  let z1 := flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₁ b₁ x
  let a1 := relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) z1
  let z2 := flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂ a1
  let pl1 := maxPoolFlat c1 (2 * (2 * (2 * h))) (2 * (2 * (2 * w)))
    (relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) z2)
  let z3 := flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ b₃ pl1
  let a3 := relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) z3
  let z4 := flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄ a3
  let pl2 := maxPoolFlat c2 (2 * (2 * h)) (2 * (2 * w))
    (relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) z4)
  let z5 := flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₅ b₅ pl2
  let a5 := relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) z5
  let z6 := flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆ a5
  let pl3 := maxPoolFlat c3 (2 * h) (2 * w) (relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) z6)
  let z7 := flatConv (h := 2 * h) (w := 2 * w) W₇ b₇ pl3
  let a7 := relu (c4 * (2 * h) * (2 * w)) z7
  let z8 := flatConv (h := 2 * h) (w := 2 * w) W₈ b₈ a7
  let pl4 := maxPoolFlat c4 h w (relu (c4 * (2 * h) * (2 * w)) z8)
  let h9 := dense W₉ b₉ pl4
  let ha := dense Wa ba (relu d1 h9)
  let cotHa := (mlpCotOut1 Wb ha).denote g
  let cotH9 := (mlpCotOut0 Wa Wb h9 ha).denote g
  let cotZ8 := cnnChainCotW2Sel σ₄ W₉ Wa Wb h9 ha z8 g
  let cotZ7 := cnnChainCotW1 W₈ z7 cotZ8
  let cotZ6 := cifarChainCotW2Sel σ₃ W₇ z6 cotZ7
  let cotZ5 := cnnChainCotW1 W₆ z5 cotZ6
  let cotZ4 := cifarChainCotW2Sel σ₂ W₅ z4 cotZ5
  let cotZ3 := cnnChainCotW1 W₄ z3 cotZ4
  let cotZ2 := cifarChainCotW2Sel σ₁ W₃ z2 cotZ3
  let cotZ1 := cnnChainCotW1 W₂ z1 cotZ2
  -- conv₁ … conv₈
  HasGradAt (fun θ => F (Kernel4.unflatten θ) b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₁)
      (den (SHlo.convWeightGrad xN b₁ (Tensor3.unflatten x) W₁ (.operand cotN cotZ1)))
  ∧ HasGradAt (fun θ => F W₁ θ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb) b₁
      (den (SHlo.convBiasGrad W₁ (Tensor3.unflatten x) b₁ (.operand cotN cotZ1)))
  ∧ HasGradAt (fun θ => F W₁ b₁ (Kernel4.unflatten θ) b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₂)
      (den (SHlo.convWeightGrad xN b₂ (Tensor3.unflatten a1) W₂ (.operand cotN cotZ2)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ θ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb) b₂
      (den (SHlo.convBiasGrad W₂ (Tensor3.unflatten a1) b₂ (.operand cotN cotZ2)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ (Kernel4.unflatten θ) b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₃)
      (den (SHlo.convWeightGrad xN b₃ (Tensor3.unflatten pl1) W₃ (.operand cotN cotZ3)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ θ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb) b₃
      (den (SHlo.convBiasGrad W₃ (Tensor3.unflatten pl1) b₃ (.operand cotN cotZ3)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ (Kernel4.unflatten θ) b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₄)
      (den (SHlo.convWeightGrad xN b₄ (Tensor3.unflatten a3) W₄ (.operand cotN cotZ4)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ θ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb) b₄
      (den (SHlo.convBiasGrad W₄ (Tensor3.unflatten a3) b₄ (.operand cotN cotZ4)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ (Kernel4.unflatten θ) b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₅)
      (den (SHlo.convWeightGrad xN b₅ (Tensor3.unflatten pl2) W₅ (.operand cotN cotZ5)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ θ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb) b₅
      (den (SHlo.convBiasGrad W₅ (Tensor3.unflatten pl2) b₅ (.operand cotN cotZ5)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ (Kernel4.unflatten θ) b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₆)
      (den (SHlo.convWeightGrad xN b₆ (Tensor3.unflatten a5) W₆ (.operand cotN cotZ6)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ θ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb) b₆
      (den (SHlo.convBiasGrad W₆ (Tensor3.unflatten a5) b₆ (.operand cotN cotZ6)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ (Kernel4.unflatten θ) b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₇)
      (den (SHlo.convWeightGrad xN b₇ (Tensor3.unflatten pl3) W₇ (.operand cotN cotZ7)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ θ W₈ b₈ W₉ b₉ Wa ba Wb bb) b₇
      (den (SHlo.convBiasGrad W₇ (Tensor3.unflatten pl3) b₇ (.operand cotN cotZ7)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ (Kernel4.unflatten θ) b₈ W₉ b₉ Wa ba Wb bb)
      (Kernel4.flatten W₈)
      (den (SHlo.convWeightGrad xN b₈ (Tensor3.unflatten a7) W₈ (.operand cotN cotZ8)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ θ W₉ b₉ Wa ba Wb bb) b₈
      (den (SHlo.convBiasGrad W₈ (Tensor3.unflatten a7) b₈ (.operand cotN cotZ8)))
  -- the dense head
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ (Mat.unflatten θ) b₉ Wa ba Wb bb)
      (Mat.flatten W₉) (den (SHlo.weightGrad xN pl4 (.operand cotN cotH9)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ θ Wa ba Wb bb) b₉
      (den (SHlo.biasGrad (.operand cotN cotH9)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ (Mat.unflatten θ) ba Wb bb)
      (Mat.flatten Wa) (den (SHlo.weightGrad xN (relu d1 h9) (.operand cotN cotHa)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa θ Wb bb) ba
      (den (SHlo.biasGrad (.operand cotN cotHa)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba (Mat.unflatten θ) bb)
      (Mat.flatten Wb) (den (SHlo.weightGrad xN (relu d1 ha) (.operand cotN g)))
  ∧ HasGradAt (fun θ => F W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb θ) bb
      (den (SHlo.biasGrad (.operand cotN g)))

/-- **Every cifar8 gradient node is the gradient of `L` in that parameter**, whenever `g` is `L`'s
    gradient at the logits.

    Hypotheses: odd kernels, and `Cifar8LossSmoothAt` — every ReLU off its kink, every window of
    each pool dead or tied only between cells that are the same function of the weights upstream
    of it, each selection naming a maximum of every window. -/
theorem cifar8_net_lossGrad (xN cotN : String) (hkH : 2 * ((kH - 1) / 2) + 1 = kH)
    (hkW : 2 * ((kW - 1) / 2) + 1 = kW) (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1)
    (Wa : Mat d1 d1) (ba : Vec d1) (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (σ₁ : Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin c4 → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : Cifar8LossSmoothAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba x
      σ₁ σ₂ σ₃ σ₄)
    {L : Vec nClasses → Vec 1} {g : Vec nClasses}
    (hL : HasGradAt L (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
      W₉ b₉ Wa ba Wb bb x) g) :
    Cifar8NetLossTied xN cotN W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x
      σ₁ σ₂ σ₃ σ₄ L g := by
  unfold Cifar8NetLossTied
  intro F z1 a1 z2 pl1 z3 a3 z4 pl2 z5 a5 z6 pl3 z7 a7 z8 pl4 h9 ha cotHa cotH9 cotZ8 cotZ7
    cotZ6 cotZ5 cotZ4 cotZ3 cotZ2 cotZ1
  -- the dense head
  have hLb : HasGradAt L (dense Wb bb (relu d1 ha)) g := hL
  have hHa : HasGradAt (fun y => L (dense Wb bb (relu d1 y))) ha cotHa :=
    (hasGradAt_relu ha hx.za (hasGradAt_dense Wb bb _ hLb)).of_eq (denote_subst _ _ g).symm
  have hH9 : HasGradAt (fun y => L (dense Wb bb (relu d1 (dense Wa ba (relu d1 y))))) h9 cotH9 :=
    (hasGradAt_relu h9 hx.z9 (hasGradAt_dense Wa ba _ hHa)).of_eq
      (by simp only [cotH9, mlpCotOut0, denote_subst]; rfl)
  -- the gather model: each pool frozen at its selection
  let Gp4 : Vec (c4 * h * w) → Vec 1 := fun u =>
    L (dense Wb bb (relu d1 (dense Wa ba (relu d1 (dense W₉ b₉ u)))))
  let G8 : Vec (c4 * (2 * h) * (2 * w)) → Vec 1 := fun y =>
    Gp4 (fun k => relu (c4 * (2 * h) * (2 * w)) y (poolSelIdx σ₄ k))
  let G6 : Vec (c3 * (2 * (2 * h)) * (2 * (2 * w))) → Vec 1 := fun y =>
    G8 (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈ (relu (c4 * (2 * h) * (2 * w))
      (flatConv (h := 2 * h) (w := 2 * w) W₇ b₇
        (fun k => relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) y (poolSelIdx σ₃ k)))))
  let G4 : Vec (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) → Vec 1 := fun y =>
    G6 (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
      (relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w))
        W₅ b₅ (fun k => relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) y
          (poolSelIdx σ₂ k)))))
  let G2 : Vec (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) → Vec 1 := fun y =>
    G4 (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
      (relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))
        (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ b₃
          (fun k => relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) y
            (poolSelIdx σ₁ k)))))
  have hpt4 : pl4 = fun k => relu (c4 * (2 * h) * (2 * w)) z8 (poolSelIdx σ₄ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₄ _ hx.sel4
  have hpt3 : pl3 = fun k => relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) z6 (poolSelIdx σ₃ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₃ _ hx.sel3
  have hpt2 : pl2 = fun k => relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) z4
      (poolSelIdx σ₂ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₂ _ hx.sel2
  have hpt1 : pl1 = fun k => relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) z2
      (poolSelIdx σ₁ k) := by
    rw [← poolGatherFlat_eq_sel]; exact maxPoolFlat_eq_poolGatherFlat σ₁ _ hx.sel1
  have hZ8 : HasGradAt G8 z8 cotZ8 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₄) z8 hx.z8
      ((hasGradAt_dense W₉ b₉ pl4 hH9).congr_point hpt4)).of_eq (by
      funext i
      simp only [cotZ8, cnnChainCotW2Sel, cnnDenseHeadCot, cotH9, mlpCotOut0, mlpCotOut1,
        denote_subst]
      rfl)
  have hZ7 : HasGradAt (fun y => G8 (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
      (relu (c4 * (2 * h) * (2 * w)) y))) z7 cotZ7 :=
    hasGradAt_relu z7 hx.z7 (hasGradAt_conv hkH hkW W₈ b₈ a7 hZ8)
  have hZ6 : HasGradAt G6 z6 cotZ6 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₃) z6 hx.z6
      ((hasGradAt_conv hkH hkW W₇ b₇ pl3 hZ7).congr_point hpt3)).of_eq rfl
  have hZ5 : HasGradAt (fun y => G6 (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
      (relu (c3 * (2 * (2 * h)) * (2 * (2 * w))) y))) z5 cotZ5 :=
    hasGradAt_relu z5 hx.z5 (hasGradAt_conv hkH hkW W₆ b₆ a5 hZ6)
  have hZ4 : HasGradAt G4 z4 cotZ4 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₂) z4 hx.z4
      ((hasGradAt_conv hkH hkW W₅ b₅ pl2 hZ5).congr_point hpt2)).of_eq rfl
  have hZ3 : HasGradAt (fun y => G4 (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w)))
      W₄ b₄ (relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))) y))) z3 cotZ3 :=
    hasGradAt_relu z3 hx.z3 (hasGradAt_conv hkH hkW W₄ b₄ a3 hZ4)
  have hZ2 : HasGradAt G2 z2 cotZ2 :=
    (hasGradAt_gatherRelu (poolSelIdx σ₁) z2 hx.z2
      ((hasGradAt_conv hkH hkW W₃ b₃ pl1 hZ3).congr_point hpt1)).of_eq rfl
  have hZ1 : HasGradAt (fun y => G2 (flatConv (h := 2 * (2 * (2 * (2 * h))))
      (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
      (relu (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))) y))) z1 cotZ1 :=
    hasGradAt_relu z1 hx.z1 (hasGradAt_conv hkH hkW W₂ b₂ a1 hZ2)
  -- along a parameter, the real net agrees with the gather model near the point: pool 4 alone
  -- (a stage-4 parameter), then each earlier pool under the later ones, outermost first
  have germ4 : ∀ {P : Nat} (Z : Vec P → Vec (c4 * (2 * h) * (2 * w))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = z8 →
      (∀ θ (ci : Fin c4) (p q : Fin (2 * h) × Fin (2 * w)), Cifar8PoolTwin4 c1 c2 c3 c4 kH kW x p q →
        Z θ (t3Idx ci p.1 p.2) = Z θ (t3Idx ci q.1 q.2)) →
      (fun θ => Gp4 (maxPoolFlat c4 h w (relu _ (Z θ)))) =ᶠ[nhds θ₀] fun θ => G8 (Z θ) := by
    intro P Z θ₀ hZc h0 hT
    have hg := maxPool_relu_eventuallyEq_sel Z σ₄ (Cifar8PoolTwin4 c1 c2 c3 c4 kH kW x) hT θ₀ hZc
      (by rw [h0]; exact hx.z8) (by rw [h0]; exact hx.pool4) (by rw [h0]; exact hx.sel4)
    filter_upwards [hg] with θ hθ
    exact congrArg Gp4 hθ
  have germ3 : ∀ {P : Nat} (Z : Vec P → Vec (c3 * (2 * (2 * h)) * (2 * (2 * w)))) (θ₀ : Vec P),
      ContinuousAt Z θ₀ → Z θ₀ = z6 →
      (∀ θ (ci : Fin c3) (p q : Fin (2 * (2 * h)) × Fin (2 * (2 * w))),
        Cifar8PoolTwin3 c1 c2 c3 kH kW x p q → Z θ (t3Idx ci p.1 p.2) = Z θ (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c4) (p q : Fin (2 * h) × Fin (2 * w)), Cifar8PoolTwin4 c1 c2 c3 c4 kH kW x p q →
        cifar8Up W₇ b₇ W₈ b₈ (Z θ) (t3Idx ci p.1 p.2)
          = cifar8Up W₇ b₇ W₈ b₈ (Z θ) (t3Idx ci q.1 q.2)) →
      (fun θ => Gp4 (maxPoolFlat c4 h w (relu _ (cifar8Up W₇ b₇ W₈ b₈ (Z θ)))))
        =ᶠ[nhds θ₀] fun θ => G6 (Z θ) := by
    intro P Z θ₀ hZc h0 hT3 hT4
    refine (germ4 (fun θ => cifar8Up W₇ b₇ W₈ b₈ (Z θ)) θ₀
      ((cifar8Up_continuous W₇ b₇ W₈ b₈).continuousAt.comp hZc) (by rw [h0]; rfl) hT4).trans ?_
    have hg := maxPool_relu_eventuallyEq_sel Z σ₃ (Cifar8PoolTwin3 c1 c2 c3 kH kW x) hT3 θ₀ hZc
      (by rw [h0]; exact hx.z6) (by rw [h0]; exact hx.pool3) (by rw [h0]; exact hx.sel3)
    filter_upwards [hg] with θ hθ
    exact congrArg (fun v => G8 (flatConv (h := 2 * h) (w := 2 * w) W₈ b₈
      (relu (c4 * (2 * h) * (2 * w)) (flatConv (h := 2 * h) (w := 2 * w) W₇ b₇ v)))) hθ
  have germ2 : ∀ {P : Nat} (Z : Vec P → Vec (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w)))))
      (θ₀ : Vec P), ContinuousAt Z θ₀ → Z θ₀ = z4 →
      (∀ θ (ci : Fin c2) (p q : Fin (2 * (2 * (2 * h))) × Fin (2 * (2 * (2 * w)))),
        Cifar8PoolTwin2 c1 c2 kH kW x p q → Z θ (t3Idx ci p.1 p.2) = Z θ (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c3) (p q : Fin (2 * (2 * h)) × Fin (2 * (2 * w))),
        Cifar8PoolTwin3 c1 c2 c3 kH kW x p q →
        cifar8Up W₅ b₅ W₆ b₆ (Z θ) (t3Idx ci p.1 p.2)
          = cifar8Up W₅ b₅ W₆ b₆ (Z θ) (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c4) (p q : Fin (2 * h) × Fin (2 * w)), Cifar8PoolTwin4 c1 c2 c3 c4 kH kW x p q →
        cifar8Up W₇ b₇ W₈ b₈ (cifar8Up W₅ b₅ W₆ b₆ (Z θ)) (t3Idx ci p.1 p.2)
          = cifar8Up W₇ b₇ W₈ b₈ (cifar8Up W₅ b₅ W₆ b₆ (Z θ)) (t3Idx ci q.1 q.2)) →
      (fun θ => Gp4 (maxPoolFlat c4 h w (relu _
          (cifar8Up W₇ b₇ W₈ b₈ (cifar8Up W₅ b₅ W₆ b₆ (Z θ))))))
        =ᶠ[nhds θ₀] fun θ => G4 (Z θ) := by
    intro P Z θ₀ hZc h0 hT2 hT3 hT4
    refine (germ3 (fun θ => cifar8Up W₅ b₅ W₆ b₆ (Z θ)) θ₀
      ((cifar8Up_continuous W₅ b₅ W₆ b₆).continuousAt.comp hZc) (by rw [h0]; rfl) hT3
      hT4).trans ?_
    have hg := maxPool_relu_eventuallyEq_sel Z σ₂ (Cifar8PoolTwin2 c1 c2 kH kW x) hT2 θ₀ hZc
      (by rw [h0]; exact hx.z4) (by rw [h0]; exact hx.pool2) (by rw [h0]; exact hx.sel2)
    filter_upwards [hg] with θ hθ
    exact congrArg (fun v => G6 (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆
      (relu (c3 * (2 * (2 * h)) * (2 * (2 * w)))
        (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₅ b₅ v)))) hθ
  have germ1 : ∀ {P : Nat}
      (Z : Vec P → Vec (c1 * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
      (θ₀ : Vec P), ContinuousAt Z θ₀ → Z θ₀ = z2 →
      (∀ θ (ci : Fin c1) (p q : Fin (2 * (2 * (2 * (2 * h)))) × Fin (2 * (2 * (2 * (2 * w))))),
        CnnPoolTwin c1 kH kW x p q → Z θ (t3Idx ci p.1 p.2) = Z θ (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c2) (p q : Fin (2 * (2 * (2 * h))) × Fin (2 * (2 * (2 * w)))),
        Cifar8PoolTwin2 c1 c2 kH kW x p q →
        cifar8Up W₃ b₃ W₄ b₄ (Z θ) (t3Idx ci p.1 p.2)
          = cifar8Up W₃ b₃ W₄ b₄ (Z θ) (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c3) (p q : Fin (2 * (2 * h)) × Fin (2 * (2 * w))),
        Cifar8PoolTwin3 c1 c2 c3 kH kW x p q →
        cifar8Up W₅ b₅ W₆ b₆ (cifar8Up W₃ b₃ W₄ b₄ (Z θ)) (t3Idx ci p.1 p.2)
          = cifar8Up W₅ b₅ W₆ b₆ (cifar8Up W₃ b₃ W₄ b₄ (Z θ)) (t3Idx ci q.1 q.2)) →
      (∀ θ (ci : Fin c4) (p q : Fin (2 * h) × Fin (2 * w)), Cifar8PoolTwin4 c1 c2 c3 c4 kH kW x p q →
        cifar8Up W₇ b₇ W₈ b₈ (cifar8Up W₅ b₅ W₆ b₆ (cifar8Up W₃ b₃ W₄ b₄ (Z θ)))
            (t3Idx ci p.1 p.2)
          = cifar8Up W₇ b₇ W₈ b₈ (cifar8Up W₅ b₅ W₆ b₆ (cifar8Up W₃ b₃ W₄ b₄ (Z θ)))
            (t3Idx ci q.1 q.2)) →
      (fun θ => Gp4 (maxPoolFlat c4 h w (relu _
          (cifar8Up W₇ b₇ W₈ b₈ (cifar8Up W₅ b₅ W₆ b₆ (cifar8Up W₃ b₃ W₄ b₄ (Z θ)))))))
        =ᶠ[nhds θ₀] fun θ => G2 (Z θ) := by
    intro P Z θ₀ hZc h0 hT1 hT2 hT3 hT4
    refine (germ2 (fun θ => cifar8Up W₃ b₃ W₄ b₄ (Z θ)) θ₀
      ((cifar8Up_continuous W₃ b₃ W₄ b₄).continuousAt.comp hZc) (by rw [h0]; rfl) hT2 hT3
      hT4).trans ?_
    have hg := maxPool_relu_eventuallyEq_sel Z σ₁ (CnnPoolTwin c1 kH kW x) hT1 θ₀ hZc
      (by rw [h0]; exact hx.z2) (by rw [h0]; exact hx.pool1) (by rw [h0]; exact hx.sel1)
    filter_upwards [hg] with θ hθ
    exact congrArg (fun v => G4 (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄
      (relu (c2 * (2 * (2 * (2 * h))) * (2 * (2 * (2 * w))))
        (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ b₃ v)))) hθ
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    denseW_hasGradAt xN cotN pl4 W₉ b₉ hH9, denseB_hasGradAt cotN W₉ pl4 b₉ hH9,
    denseW_hasGradAt xN cotN _ Wa ba hHa, denseB_hasGradAt cotN Wa _ ba hHa,
    denseW_hasGradAt xN cotN _ Wb bb hLb, denseB_hasGradAt cotN Wb _ bb hLb⟩
  -- stage 1: all four pools move
  · refine (convW_hasGradAt xN cotN b₁ x W₁ hZ1).congr_of_eventuallyEq (germ1
      (fun θ => flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
        (relu _ (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
          (Kernel4.unflatten θ) b₁ x))) _
      ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        (conv2d_weight_differentiable b₁ (Tensor3.unflatten x)).continuous)).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ W₂ b₂ ci)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ W₂ b₂ W₃ b₃ W₄ b₄ ci)
      (fun θ ci p q hpq => hpq (Kernel4.unflatten θ) b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ ci)
      (fun θ ci p q hpq =>
        hpq (Kernel4.unflatten θ) b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  · refine (convB_hasGradAt cotN W₁ x b₁ hZ1).congr_of_eventuallyEq (germ1
      (fun θ => flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ b₂
        (relu _ (flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₁ θ x)))
      _ ((flatConv_differentiable W₂ b₂).continuous.comp ((relu_continuous _).comp
        (conv2d_bias_differentiable W₁ (Tensor3.unflatten x)).continuous)).continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ θ W₂ b₂ ci)
      (fun θ ci p q hpq => hpq W₁ θ W₂ b₂ W₃ b₃ W₄ b₄ ci)
      (fun θ ci p q hpq => hpq W₁ θ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ ci)
      (fun θ ci p q hpq => hpq W₁ θ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₂ a1 W₂ hZ2).congr_of_eventuallyEq (germ1
      (fun θ => flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w))))
        (Kernel4.unflatten θ) b₂ a1) _
      (conv2d_weight_differentiable b₂ (Tensor3.unflatten a1)).continuous.continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ (Kernel4.unflatten θ) b₂ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ (Kernel4.unflatten θ) b₂ W₃ b₃ W₄ b₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ (Kernel4.unflatten θ) b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ ci)
      (fun θ ci p q hpq =>
        hpq W₁ b₁ (Kernel4.unflatten θ) b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  · refine (convB_hasGradAt cotN W₂ a1 b₂ hZ2).congr_of_eventuallyEq (germ1
      (fun θ => flatConv (h := 2 * (2 * (2 * (2 * h)))) (w := 2 * (2 * (2 * (2 * w)))) W₂ θ a1) _
      (conv2d_bias_differentiable W₂ (Tensor3.unflatten a1)).continuous.continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ θ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ θ W₃ b₃ W₄ b₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ θ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ θ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  -- stage 2: pools 2–4 move
  · refine (convW_hasGradAt xN cotN b₃ pl1 W₃ hZ3).congr_of_eventuallyEq (germ2
      (fun θ => flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄ (relu _
        (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) (Kernel4.unflatten θ) b₃ pl1)))
      _ ((flatConv_differentiable W₄ b₄).continuous.comp ((relu_continuous _).comp
        (conv2d_weight_differentiable b₃ (Tensor3.unflatten pl1)).continuous)).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ (Kernel4.unflatten θ) b₃ W₄ b₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ (Kernel4.unflatten θ) b₃ W₄ b₄ W₅ b₅ W₆ b₆ ci)
      (fun θ ci p q hpq =>
        hpq W₁ b₁ W₂ b₂ (Kernel4.unflatten θ) b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  · refine (convB_hasGradAt cotN W₃ pl1 b₃ hZ3).congr_of_eventuallyEq (germ2
      (fun θ => flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ b₄ (relu _
        (flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₃ θ pl1))) _
      ((flatConv_differentiable W₄ b₄).continuous.comp ((relu_continuous _).comp
        (conv2d_bias_differentiable W₃ (Tensor3.unflatten pl1)).continuous)).continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ θ W₄ b₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ θ W₄ b₄ W₅ b₅ W₆ b₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ θ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₄ a3 W₄ hZ4).congr_of_eventuallyEq (germ2
      (fun θ => flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w)))
        (Kernel4.unflatten θ) b₄ a3) _
      (conv2d_weight_differentiable b₄ (Tensor3.unflatten a3)).continuous.continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ (Kernel4.unflatten θ) b₄ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ (Kernel4.unflatten θ) b₄ W₅ b₅ W₆ b₆ ci)
      (fun θ ci p q hpq =>
        hpq W₁ b₁ W₂ b₂ W₃ b₃ (Kernel4.unflatten θ) b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  · refine (convB_hasGradAt cotN W₄ a3 b₄ hZ4).congr_of_eventuallyEq (germ2
      (fun θ => flatConv (h := 2 * (2 * (2 * h))) (w := 2 * (2 * (2 * w))) W₄ θ a3) _
      (conv2d_bias_differentiable W₄ (Tensor3.unflatten a3)).continuous.continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ θ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ θ W₅ b₅ W₆ b₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ θ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  -- stage 3: pools 3–4 move
  · refine (convW_hasGradAt xN cotN b₅ pl2 W₅ hZ5).congr_of_eventuallyEq (germ3
      (fun θ => flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆ (relu _
        (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) (Kernel4.unflatten θ) b₅ pl2))) _
      ((flatConv_differentiable W₆ b₆).continuous.comp ((relu_continuous _).comp
        (conv2d_weight_differentiable b₅ (Tensor3.unflatten pl2)).continuous)).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ (Kernel4.unflatten θ) b₅ W₆ b₆ ci)
      (fun θ ci p q hpq =>
        hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ (Kernel4.unflatten θ) b₅ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  · refine (convB_hasGradAt cotN W₅ pl2 b₅ hZ5).congr_of_eventuallyEq (germ3
      (fun θ => flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ b₆ (relu _
        (flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₅ θ pl2))) _
      ((flatConv_differentiable W₆ b₆).continuous.comp ((relu_continuous _).comp
        (conv2d_bias_differentiable W₅ (Tensor3.unflatten pl2)).continuous)).continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ θ W₆ b₆ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ θ W₆ b₆ W₇ b₇ W₈ b₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₆ a5 W₆ hZ6).congr_of_eventuallyEq (germ3
      (fun θ => flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) (Kernel4.unflatten θ) b₆ a5) _
      (conv2d_weight_differentiable b₆ (Tensor3.unflatten a5)).continuous.continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ (Kernel4.unflatten θ) b₆ ci)
      (fun θ ci p q hpq =>
        hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ (Kernel4.unflatten θ) b₆ W₇ b₇ W₈ b₈ ci)).symm
  · refine (convB_hasGradAt cotN W₆ a5 b₆ hZ6).congr_of_eventuallyEq (germ3
      (fun θ => flatConv (h := 2 * (2 * h)) (w := 2 * (2 * w)) W₆ θ a5) _
      (conv2d_bias_differentiable W₆ (Tensor3.unflatten a5)).continuous.continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ θ ci)
      (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ θ W₇ b₇ W₈ b₈ ci)).symm
  -- stage 4: pool 4 moves
  · refine (convW_hasGradAt xN cotN b₇ pl3 W₇ hZ7).congr_of_eventuallyEq (germ4
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) W₈ b₈ (relu _
        (flatConv (h := 2 * h) (w := 2 * w) (Kernel4.unflatten θ) b₇ pl3))) _
      ((flatConv_differentiable W₈ b₈).continuous.comp ((relu_continuous _).comp
        (conv2d_weight_differentiable b₇ (Tensor3.unflatten pl3)).continuous)).continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq =>
        hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ (Kernel4.unflatten θ) b₇ W₈ b₈ ci)).symm
  · refine (convB_hasGradAt cotN W₇ pl3 b₇ hZ7).congr_of_eventuallyEq (germ4
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) W₈ b₈ (relu _
        (flatConv (h := 2 * h) (w := 2 * w) W₇ θ pl3))) _
      ((flatConv_differentiable W₈ b₈).continuous.comp ((relu_continuous _).comp
        (conv2d_bias_differentiable W₇ (Tensor3.unflatten pl3)).continuous)).continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ θ W₈ b₈ ci)).symm
  · refine (convW_hasGradAt xN cotN b₈ a7 W₈ hZ8).congr_of_eventuallyEq (germ4
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) (Kernel4.unflatten θ) b₈ a7) _
      (conv2d_weight_differentiable b₈ (Tensor3.unflatten a7)).continuous.continuousAt
      (by simp only [Kernel4.unflatten_flatten]; rfl)
      (fun θ ci p q hpq =>
        hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ (Kernel4.unflatten θ) b₈ ci)).symm
  · exact (convB_hasGradAt cotN W₈ a7 b₈ hZ8).congr_of_eventuallyEq (germ4
      (fun θ => flatConv (h := 2 * h) (w := 2 * w) W₈ θ a7) _
      (conv2d_bias_differentiable W₈ (Tensor3.unflatten a7)).continuous.continuousAt
      rfl (fun θ ci p q hpq => hpq W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ θ ci)).symm

/-- **The artifact's loss**: every node is the gradient of the softmax cross-entropy at `label`,
    `g` the emitted loss cotangent. -/
theorem cifar8_net_lossGrad_CE (xN cotN nlogN ohN : String) (hkH : 2 * ((kH - 1) / 2) + 1 = kH)
    (hkW : 2 * ((kW - 1) / 2) + 1 = kW) (W₁ : Kernel4 c1 ic kH kW) (b₁ : Vec c1)
    (W₂ : Kernel4 c1 c1 kH kW) (b₂ : Vec c1) (W₃ : Kernel4 c2 c1 kH kW) (b₃ : Vec c2)
    (W₄ : Kernel4 c2 c2 kH kW) (b₄ : Vec c2) (W₅ : Kernel4 c3 c2 kH kW) (b₅ : Vec c3)
    (W₆ : Kernel4 c3 c3 kH kW) (b₆ : Vec c3) (W₇ : Kernel4 c4 c3 kH kW) (b₇ : Vec c4)
    (W₈ : Kernel4 c4 c4 kH kW) (b₈ : Vec c4) (W₉ : Mat (c4 * h * w) d1) (b₉ : Vec d1)
    (Wa : Mat d1 d1) (ba : Vec d1) (Wb : Mat d1 nClasses) (bb : Vec nClasses)
    (x : Vec (ic * (2 * (2 * (2 * (2 * h)))) * (2 * (2 * (2 * (2 * w))))))
    (σ₁ : Fin c1 → Fin (2 * (2 * (2 * h))) → Fin (2 * (2 * (2 * w))) → Fin 2 × Fin 2)
    (σ₂ : Fin c2 → Fin (2 * (2 * h)) → Fin (2 * (2 * w)) → Fin 2 × Fin 2)
    (σ₃ : Fin c3 → Fin (2 * h) → Fin (2 * w) → Fin 2 × Fin 2)
    (σ₄ : Fin c4 → Fin h → Fin w → Fin 2 × Fin 2)
    (hx : Cifar8LossSmoothAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba x
      σ₁ σ₂ σ₃ σ₄) (label : Fin nClasses) :
    Cifar8NetLossTied xN cotN W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x
      σ₁ σ₂ σ₃ σ₄ (fun z _ => crossEntropy nClasses z label)
      (den (SHlo.sub (SHlo.softmaxDiv (SHlo.expe (.operand nlogN
          (cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x))))
        (.operand ohN (oneHot nClasses label)))) := by
  rw [softmaxCELossCot_den]
  exact cifar8_net_lossGrad xN cotN hkH hkW W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈
    W₉ b₉ Wa ba Wb bb x σ₁ σ₂ σ₃ σ₄ hx (hasGradAt_crossEntropy label _)

end Net

end Proofs.Cifar8TieG
