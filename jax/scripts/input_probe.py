"""Input-pipeline throughput probe: images/sec out of build_imagenet_iter.

Settles whether repeated augmentation costs throughput. The deferral rationale
in planning/vit_imagenet.md assumed "3x CPU decode+RandAug". But flat_map sits
BEFORE .map(_pp) and steps_per_epoch is unchanged, so per epoch the trainer
pulls the same number of images through _pp -- just from 1/3 as many unique
records. Prediction: throughput is unchanged.

env: GEN=<generated trainer>  BATCH=<batch>  NBATCH=<batches to time>
"""
import importlib.util, os, time
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # input pipeline only, no GPU needed
GEN = os.environ["GEN"]
BATCH = int(os.environ.get("BATCH", "256"))
NB = int(os.environ.get("NBATCH", "12"))
WARM = 4

spec = importlib.util.spec_from_file_location("gen", GEN)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

it = iter(m.build_imagenet_iter("train", BATCH, True, True))  # trainer wraps in iter()
ts = []
for i in range(NB):
    t0 = time.time()
    x, y = next(it)
    dt = time.time() - t0
    if i >= WARM:
        ts.append(dt)

med = float(np.median(ts))
print(f"RESULT {os.path.basename(GEN)} batch={BATCH} "
      f"median={med*1000:.0f} ms/batch  {BATCH/med:.0f} img/s  "
      f"(n={len(ts)} timed, {WARM} warmup)")
