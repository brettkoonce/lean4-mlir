import LeanMlir.VerifiedSpec

/-! # Concrete verified architectures — the shared specs

Readable layer-list specs that are referenced by **both** a trainer (`Main*Verified`)
and a proof (`LeanMlir/Proofs/*`). Kept in this light module (no Mathlib) so the proof
side can import the *exact* object the trainer runs — there's then a single source of
truth, and "the spec the trainer runs is the proven one" is literally true, not a twin.

Specs with no proof importing them yet (e.g. `resnet34Verified`) stay in their own
`Main*Verified.lean`; a spec moves here the moment a proof needs to name it. -/

/-- The Chapter-2 linear classifier: a single dense 784→10. Trained by
    `MainMnistLinearVerified`; its math VJP is proven in `Proofs/SpecVJP.lean`
    (`linearVerified_has_vjp`) — both over *this* object. -/
def linearVerified : VerifiedNetSpec where
  name     := "MNIST-Linear"
  slug     := "linear"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.dense 784 10]
  blurb    := "MNIST-Linear via the VERIFIED renderer (pretty∘emit) → IREE FFI → GPU"

-- Shape tie: the derived param layout is W:[784,10] (He) + b:[10] (zeros).
#guard linearVerified.toSpecs == #[(#[784, 10], 0), (#[10], 2)]

/-- The Chapter-3 MLP: dense 784→512 → relu → dense 512→512 → relu → dense 512→10.
    Trained by `MainMnistMlpVerified`; its math VJP is proven in `Proofs/SpecVJP.lean`
    (`mlpVerified_has_vjp` / `_at`) — both over *this* object. -/
def mlpVerified : VerifiedNetSpec where
  name     := "MNIST-MLP"
  slug     := "mlp"
  inC      := 1
  imageH   := 28
  imageW   := 28
  nClasses := 10
  data     := .mnist
  layers   := [.dense 784 512, .relu, .dense 512 512, .relu, .dense 512 10]
  blurb    := "MNIST-MLP via the VERIFIED renderer (784→512→512→10) → IREE FFI → GPU"

-- Shape tie: W₀:[784,512] b₀:[512] | W₁:[512,512] b₁:[512] | W₂:[512,10] b₂:[10].
#guard mlpVerified.toSpecs ==
  #[(#[784, 512], 0), (#[512], 2), (#[512, 512], 0), (#[512], 2), (#[512, 10], 0), (#[10], 2)]
