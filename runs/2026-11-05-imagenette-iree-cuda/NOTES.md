# Reading these Imagenette numbers

`results.tsv` scores **one** IREE/CUDA seed per net (80 epochs, IREE 3.12.0, one RTX 4060 Ti each,
ares) against the XLA/CUDA **five-seed** range per net (`runs/2026-08-31-imagenette-n3`). The
`over/under by N images` column is a literal distance to that range's edge, not a pass/fail verdict:
a single fresh seed lands just outside a five-seed envelope all the time, and `N images` (out of the
3,925-image validation set) is the honest scale of any such miss.

Five of the seven land **inside** the five-seed range. The two that sit outside do so by a handful of
images, i.e. inside seed noise:

* **ResNet-50** 90.17 % is **8 images over** the top of a tight five-seed band (89.58–89.96). One
  seed clearing a 0.38 pt band by 0.20 pt is the expected spread, not a lowerer effect.
* **ConvNeXt-T** 82.17 % is **4 images under** a *single* reference run (82.27 %), not a range —
  its five seeds predate the LayerScale-1e-6 recipe, so the one post-recipe CUDA run
  (`runs/2026-09-13-convnext-imagenette-ls1e-6-resident`) is the only matched comparison there is.

One net carries a reference caveat rather than an edge-miss:

* **MobileNetV4** 85.83 % is scored against the **pre-rebuild** five seeds (85.15–86.90); those
  predate the timm conv_medium rebuild of 2026-09-25 and no post-rebuild 80-epoch CUDA run is on
  record. It lands inside them anyway, but the band is not a like-for-like reference.

Taken with MNIST (clean on the 30-seed median, `runs/2026-11-05-mnist-iree-cuda`) and CIFAR (within
seed + bf16 noise, `runs/2026-11-05-cifar-iree-cuda`), no net shows a gap beyond what seed noise
explains: IREE 3.12.0 reproduces the engine as a reference oracle. The firm platform-row verdict is
set in the final update that also carries the record into the book, not here; this commit records the
runs.
