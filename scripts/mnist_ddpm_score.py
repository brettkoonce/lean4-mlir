#!/usr/bin/env python3
"""Score the unconditional MNIST DDPM with Chapter 3's verified CNN.

⭐ The image demos' metric problem, fixed the way planning/diffusion_2d_demo.md
§4 fixed it for 2-D: stop asking whether the grid looks like digits and push the
samples through a classifier whose math VJP is proven. `cnnVerified` maps an
image to ten numbers, and in THAT space every statistic the 2-D demo uses works
unchanged.

  1. Class coverage   — digits holding >= 1% of the mass. The analogue of the
                        8-gaussians' 8/8 mode recall.
  2. Per-class mass   — against a true ~10% each. On 2-D this is where the whole
                        residual lived once recall saturated.
  3. Confidence       — mean max-softmax.
  4. Energy distance  — 2E|X-Y| - E|X-X'| - E|Y-Y'| over the 10-d softmax,
                        against real MNIST.
  5. Pixel moments    — mean and sd against the data's own. The cheapest check
                        and, on the first run, the one that localised the fault.

⭐⭐ EVERY row is bracketed by two controls, because a metric that only ever
reports "bad" is indistinguishable from a broken metric:

  * real MNIST scored AS IF generated — the positive control, the best any
    model can do;
  * unstructured pixels carrying MNIST's own mean and sd — the negative
    control, what a model that learned nothing scores.

⚠ Coverage and confidence alone would be the checkerboard mistake (§5.7): a
model emitting one canonical seven scores perfect confidence. Per-class mass,
the energy distance and the pixel moments are what can see that.

Also writes a 10-row PPM: row d holds the generated samples the classifier is
most confident are a d, so coverage and per-class quality read off one figure.

Usage: python3 scripts/mnist_ddpm_score.py [outdir]
"""
import sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else ".lake/build"
B   = ".lake/build"
NC, PIX = 10, 784

gen_l   = np.fromfile(f"{B}/mnist_ddpm_logits_gen.bin",   dtype=np.float32).reshape(-1, NC)
real_l  = np.fromfile(f"{B}/mnist_ddpm_logits_real.bin",  dtype=np.float32).reshape(-1, NC)
noise_l = np.fromfile(f"{B}/mnist_ddpm_logits_noise.bin", dtype=np.float32).reshape(-1, NC)
real_y  = np.fromfile(f"{B}/mnist_ddpm_labels_real.bin",  dtype=np.int32)
samp    = np.fromfile(f"{B}/mnist_ddpm_samples.bin",      dtype=np.float32).reshape(-1, PIX)
real_px = np.fromfile("data/t10k-images-idx3-ubyte", dtype=np.uint8)[16:]
real_px = real_px.astype(np.float32).reshape(-1, PIX) / 255.0


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def energy_distance(x, y, cap=2048, seed=0):
    """The 2-D demo's own statistic (scripts/toy2d_metrics.py), applied to the
    classifier's output instead of a point in the plane."""
    rng = np.random.default_rng(seed)
    if len(x) > cap: x = x[rng.choice(len(x), cap, replace=False)]
    if len(y) > cap: y = y[rng.choice(len(y), cap, replace=False)]
    d = lambda a, b: np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(2 * d(x, y).mean() - d(x, x).mean() - d(y, y).mean())


gp, rp, np_ = softmax(gen_l), softmax(real_l), softmax(noise_l)
rpred = real_l.argmax(1)

# ⚠ Read this line FIRST. The classifier is the instrument; if it did not load
# correctly it scores at chance and every number below is measuring nothing.
acc = float((rpred == real_y).mean())
print(f"instrument: Chapter-3 cnnVerified on {len(real_y)} real test images "
      f"= {acc*100:.2f}% accuracy")
if acc < 0.9:
    sys.exit("  => the classifier is not working; every number below would be noise")

n = len(gen_l)
# The reference the arms are scored against, and the half held out to give the
# positive control something independent to be compared with.
ref_p, ref_px = rp[n:], real_px[n:]
floor = energy_distance(rp[n:n + n], rp[-n:])


def arm(name, probs, pixels):
    cover = int((np.bincount(probs.argmax(1), minlength=NC) / len(probs) >= 0.01).sum())
    mass  = np.bincount(probs.argmax(1), minlength=NC) / len(probs)
    return dict(name=name, cover=cover, mass=mass,
                conf=float(probs.max(1).mean()),
                ed=energy_distance(probs, ref_p),
                mu=float(pixels.mean()), sd=float(pixels.std()))


noise_px = np.fromfile(f"{B}/mnist_ddpm_noise.bin", dtype=np.float32).reshape(-1, PIX)
arms = [arm("real (positive control)",  rp[:n], real_px[:n]),
        arm("DDPM samples",             gp,     samp),
        arm("noise (negative control)", np_,    noise_px)]

print(f"generated: {n} samples   reference: {len(ref_p)} held-out real images")
print()
hdr = f"  {'arm':<26} {'cover':>6} {'conf':>8} {'energy':>9} {'xfloor':>7} {'px mean':>8} {'px sd':>7}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for a in arms:
    print(f"  {a['name']:<26} {a['cover']:>4}/10 {a['conf']*100:>7.2f}% "
          f"{a['ed']:>9.5f} {a['ed']/max(floor,1e-9):>6.0f}x {a['mu']:>8.4f} {a['sd']:>7.4f}")
print(f"  {'real-vs-real floor':<26} {'':>6} {'':>8} {floor:>9.5f} {1:>6}x "
      f"{real_px.mean():>8.4f} {real_px.std():>7.4f}")
print()
for a in arms:
    print(f"  per-class mass {a['name']:<26}: " + " ".join(f"{m*100:5.1f}" for m in a["mass"]))
print(f"  {'per-class mass real test split':<41}: " +
      " ".join(f"{m*100:5.1f}" for m in np.bincount(real_y, minlength=NC) / len(real_y)))
print()


def class_grid(path, cols=12, cell=28):
    """Row d = the generated samples the classifier is most confident are a d.
    Coverage is the number of non-empty rows and per-class quality is the row
    itself, so one figure carries both."""
    img = np.full((NC * cell, cols * cell), 18, dtype=np.uint8)
    conf, pred = gp.max(1), gp.argmax(1)
    for d in range(NC):
        idx = np.where(pred == d)[0]
        idx = idx[np.argsort(-conf[idx])][:cols]
        for c, i in enumerate(idx):
            tile = np.clip(samp[i].reshape(cell, cell), 0, 1) * 255
            img[d * cell:(d + 1) * cell, c * cell:(c + 1) * cell] = tile.astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode())
        f.write(img.tobytes())
    print(f"  wrote {path}  (row d = samples the classifier calls a d, most confident first)")
    try:
        from PIL import Image
        Image.fromarray(img).save(path.replace(".ppm", ".png"))
        print(f"  wrote {path.replace('.ppm', '.png')}")
    except ImportError:
        pass


class_grid(f"{OUT}/mnist_ddpm_class_grid.ppm")

# Gate, in the shape of the 2-D demo's, and the coverage half is the sharp one.
# ⚠ No energy threshold yet: it takes more than one measurement to set a bar that
# is not flaky in the direction that costs the most. See the note in
# scripts/toy2d_metrics.py.
d = arms[1]
ok = d["cover"] == NC
print()
print(f"  => gate: class coverage {d['cover']}/10 (need 10)  [{'PASS' if ok else 'FAIL'}]")
sys.exit(0 if ok else 1)
