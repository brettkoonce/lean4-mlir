import LeanMlir

/-! CIFAR-10 plain CNN (no batch norm) — phase-3 unified pipeline.
    32×32 RGB, 10 classes.

    **Heads-up if you got here from Chapter 5 of the blueprint.**
    Chapter 5's headline demonstration is that this exact spec
    (no-BN CIFAR CNN) **fails to train** under SGD lr=0.1 — loss
    sits at log(10) ≈ 2.302 (random guess on 10 classes) for the
    full run. That failure is what motivates batch normalization
    in the chapter. **This exe does NOT reproduce that failure** —
    it uses Adam at lr=1e-3 with cosine + warmup + augmentation,
    a much more forgiving optimizer that masks the LR-headroom
    issue BN was designed to solve. Under Adam the no-BN spec
    trains fine and reaches ~72% val accuracy, comparable to the
    BN variant under the same recipe.

    To reproduce Chapter 5's failure / lift, run the ablation
    cells `cifar-nobn-sgd` and `cifar-bn-sgd` instead — those
    use `s4tfBaseline` (SGD lr=0.1, no recipe tricks), which is
    the apples-to-apples comparison the chapter is built around.
    The smaller-LR pair `cifar-nobn-sgd002` / `cifar-bn-sgd002`
    shows the same architectures top out around the same accuracy
    (~72-73%) once the LR is tuned — confirming BN's lift is LR
    headroom, not capacity.

    This exe exists for the "modern recipe under Adam" datapoint
    and as a smoke test that the no-BN spec compiles + trains
    end-to-end. If you're chasing Chapter-5 pedagogy, use the
    ablation harness. -/

def cifarCnn : NetSpec where
  name := "CIFAR-10-CNN"
  imageH := 32
  imageW := 32
  layers := [
    .conv2d  3 32 3 .same .relu,
    .conv2d 32 32 3 .same .relu,
    .maxPool 2 2,
    .conv2d 32 64 3 .same .relu,
    .conv2d 64 64 3 .same .relu,
    .maxPool 2 2,
    .flatten,
    .dense 4096 512 .relu,
    .dense 512 512 .relu,
    .dense 512 10 .identity
  ]

def cifarCnnConfig : TrainConfig where
  learningRate := 0.001
  batchSize    := 128
  epochs       := 25
  useAdam      := true
  weightDecay  := 0.0001
  cosineDecay  := true
  warmupEpochs := 2
  augment      := true

def main (args : List String) : IO Unit :=
  cifarCnn.train cifarCnnConfig (args.head?.getD "data") .cifar10
