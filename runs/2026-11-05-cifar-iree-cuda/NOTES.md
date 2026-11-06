# Reading these CIFAR numbers

`results.tsv` scores one IREE/CUDA seed per arm against the XLA/CUDA **five-seed** envelope
(`runs/2026-09-01-cifar8w-6arm-constlr`). The per-arm `over/under by N images` is a literal
distance to that envelope's edge, not a pass/fail verdict — a single fresh seed lands outside a
five-seed envelope all the time, and three things make these envelopes a loose ruler here:

* **The envelopes are tight.** The BN arms span as little as 0.22 pt (AdamW 74.24–74.46). An
  independent seed — same lowerer, let alone a different one — misses a 0.22 pt target by chance
  far more often than not. The two BN edge-misses (SGD +0.24, AdamW −0.22) are ~22–24 images,
  i.e. inside seed noise.
* **The bbn f32 arms have no own five-seed set** and are measured against the plain-BN range (see
  the `results.tsv` header). `cifar8wb-bbn` is a stronger net than `cifar8w-BN`, so f32 AdamW
  landing at 77.94 vs the BN range's 74.46 top (+3.48 pt) is the *architecture*, not the lowerer.
* **bf16 at CIFAR shapes is noise on both sides** (see memory `bf16_useless_at_cifar_shapes`). One
  XLA bf16 seed collapsed to 10 %, which is why the bf16-Nesterov reference reads 10.00–74.47. The
  bf16 edge-misses carry no signal.

No arm shows a gap beyond what seed and bf16-precision noise explain, and the no-BN net — the one
arm family with a matched reference — lands all three inside. The well-powered comparison is MNIST
(30 seeds, `runs/2026-11-05-mnist-iree-cuda`), which passes clean on median and range.

The firm pass/fail for the IREE platform row is set in the final update, alongside the 80-epoch
imagenette finals, not here. This commit records the runs; it does not adjudicate them.
