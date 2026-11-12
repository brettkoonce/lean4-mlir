module

public import LeanMlir.Proofs.Foundation.CertifiedChain
public import LeanMlir.Proofs.Foundation.DropSites
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp

@[expose] public section

/-! # Optional drop sites as certified layers

`Foundation.DropSites` states stochastic depth and classifier dropout as `Option` sites
(`dropPathOpt`, `dropoutOpt`): `none` is the identity and emits no node, `some` is the
`dropPathB` / `dropoutB` node at a host-drawn mask. Here each site is a `CertLayer`, so a block
that carries one is a `CertLayer.comp` like any other and its backward graph and VJP come from the
combinators. A site is certified everywhere (it is linear), and its backward is the op itself at
the same mask (`dropPathOpt_vjp_is_self`, `dropoutOpt_vjp_is_self`).

Used by the step ties of the nets whose renders carry a site inside a certified block:
MobileNetV4 (`Proofs.Mnv4TieB`) and ResNet-50 (`Proofs.ResNet50TieB`).
-/

namespace Proofs.StableHLO

/-- **The classifier-dropout site as a certified layer**: `dropoutOpt cd`, certified everywhere,
    its backward graph the `dropoutB` node at the same mask (nothing at `none`). -/
noncomputable def dropoutOptLayer (N : Nat) {n : Nat} (cd : Option (Vec (N * n))) :
    CertLayer (N * n) (N * n) where
  fwd := dropoutOpt cd
  ok := fun _ => True
  diff := fun x _ => (dropoutOpt_differentiable cd) x
  vjp := fun x _ => (dropoutOptHasVJP cd).toHasVJPAt x
  graph := fun _ e => dropoutOptG "%do" cd e
  faithful := fun x _ e => by rw [den_dropoutOptG]; rfl

/-- **A stochastic-depth site as a certified layer**: `dropPathOpt N n s`, certified everywhere,
    its backward graph the `dropPathB` node at the same per-example scale (nothing at `none`). -/
noncomputable def dropPathOptLayer (N n : Nat) (s : Option (Vec N)) : CertLayer (N * n) (N * n) where
  fwd := dropPathOpt N n s
  ok := fun _ => True
  diff := fun x _ => (dropPathOpt_differentiable N n s) x
  vjp := fun x _ => (dropPathOptHasVJP N n s).toHasVJPAt x
  graph := fun _ e => dropPathOptG "%dp" s e
  faithful := fun x _ e => by rw [den_dropPathOptG]; rfl

end Proofs.StableHLO
