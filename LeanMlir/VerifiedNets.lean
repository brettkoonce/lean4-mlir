import LeanMlir.VerifiedNetsCore
import LeanMlir.VerifiedTrain
import LeanMlir.VerifiedAttack
import LeanMlir.VerifiedSmoothing

/-! # The verified nets, runnable

`VerifiedNetsCore` (the specs — import-free apart from the DSL and the layout tables, and the
module the proofs import) plus the program side: `VerifiedTrain` (the driver), `VerifiedAttack`
(PGD attacks, spectral-norm studies) and `VerifiedSmoothing` (the randomized-smoothing
certificate). An entry point that imports this file can write `resnet34Verified.train cfg dir`. -/
