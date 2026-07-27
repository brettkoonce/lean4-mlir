import apps.cifar.Cifar8AdamCommon

/-! # `cifar8-verified-adam-xla` — the no-BN control for rung 2

Identical to `cifar8-bn-verified-adam-xla` except the net has no BatchNorm. Its
purpose is diagnostic: rung 2's G2 tie came in ~4 orders of magnitude looser than
rungs 0-1, and the two candidate causes — Adam's `g/(√v+ε)` normalisation and BN's
batch reductions — are separated by running this one. See
`planning/xla_pjrt_ladder.md` §8.
-/

def main (argv : List String) : IO Unit := runCifar8Adam argv
