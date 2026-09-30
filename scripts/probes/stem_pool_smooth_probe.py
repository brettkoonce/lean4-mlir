#!/usr/bin/env python3
"""Does the ResNet stem's smooth-point hypothesis hold on a real training batch?

`R34SmoothAtB` / `R50SmoothAtB` (the input VJPs) carry the stem pool's condition
`StemPoolSmoothAt` (LeanMlir/Proofs/Foundation/HeadLayers.lean): every 3x3/s2 window of the
post-ReLU stem activation has its maximum at one position, or is entirely zero.
`R34LossSmoothAtB` / `R50LossSmoothAtB` (the loss gradients) carry the weaker `StemPoolTwinAt`,
which also allows ties between cells that read identical input patches. This script evaluates it on real
ImageNet validation images at trained weights, with train-mode (batch) BatchNorm as a training
step sees it, and counts three kinds of window per channel and example:

  dead        every cell zero. Allowed now; the old condition (`MaxPool3s2Smooth` on the
              post-ReLU activation) rejected these.
  pos-tie     a positive maximum at two or more positions. Excluded by the input-VJP condition.
              Each is classified by whether the tied cells read IDENTICAL zero-padded 7x7x3 input
              patches (flat image regions), which is `StemConvTwin`: then the cells are the same
              function of the stem's weights, and the loss-gradient condition `StemPoolTwinAt`
              allows the tie.
  zero-pre    a pre-ReLU value exactly 0 (the stem ReLU's own clause, `R34StemSmoothAt`).

Arithmetic is float64, standing in for the reals the Lean statement is about. The stem's
output scale is free (a BatchNorm follows it), and trained checkpoints carry large gamma/beta, so
float32 would manufacture ties by rounding that the real-valued statement does not have.

Stem: conv 7x7/s2 pad 3 -> batch BN -> ReLU -> max-pool 3x3/s2 pad 1 (near edge clamped). The
conv bias cancels under batch BN, so only (W, gamma, beta) are read: leaves l0, l1, l2 of a
full-state checkpoint (`save_train_state`: params first, the stem's `(W, gamma, beta)` first).

  .venv/bin/python3 scripts/probes/stem_pool_smooth_probe.py \\
      --ckpt ~/resnet/r50_a3_rerun/ckpt_e100.state.npz \\
      --val-tar ~/imagenet-2024/ILSVRC2012_img_val.tar --res 160 --batches 4
"""
import argparse, io, tarfile
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406]) * 255
STD = np.array([0.229, 0.224, 0.225]) * 255


def images(tar_path, res):
    """Center-crop validation images at `res` (resize the short side to res/0.875, bicubic)."""
    with tarfile.open(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            im = Image.open(io.BytesIO(tf.extractfile(m).read())).convert('RGB')
            w, h = im.size
            sc = (res / 0.875) / min(w, h)
            im = im.resize((max(res, round(w * sc)), max(res, round(h * sc))), Image.BICUBIC)
            w, h = im.size
            l, t = (w - res) // 2, (h - res) // 2
            px = np.asarray(im.crop((l, t, l + res, t + res)), np.float64)
            yield ((px - MEAN) / STD).transpose(2, 0, 1)


def conv7s2(x, W):
    """[N,3,H,W] -> [N,oc,H/2,W/2], stride 2, zero pad 3."""
    xp = np.pad(x, ((0, 0), (0, 0), (3, 3), (3, 3)))
    ho = x.shape[2] // 2
    cols = np.lib.stride_tricks.sliding_window_view(xp, (7, 7), axis=(2, 3))[:, :, ::2, ::2]
    return np.einsum('ncijkl,ockl->noij', cols[:, :, :ho, :ho], W, optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--val-tar', required=True)
    ap.add_argument('--res', type=int, default=160)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--batches', type=int, default=4)
    a = ap.parse_args()
    d = np.load(a.ckpt)
    W, g, b = (d[k].astype(np.float64) for k in ('l0', 'l1', 'l2'))
    assert W.shape[1:] == (3, 7, 7), f'l0 is not a 7x7 stem kernel: {W.shape}'
    it = images(a.val_tar, a.res)
    tot = dict(windows=0, dead=0, pos_tie=0, pos_tie_same_patch=0, zero_pre=0, batches_with_pos_tie=0)
    for bi in range(a.batches):
        x = np.stack([next(it) for _ in range(a.batch)])
        c = conv7s2(x, W)
        mu, var = c.mean((0, 2, 3), keepdims=True), c.var((0, 2, 3), keepdims=True)
        z = g[None, :, None, None] * (c - mu) / np.sqrt(var + 1e-5) + b[None, :, None, None]
        y = np.maximum(z, 0)
        tot['zero_pre'] += int((z == 0).sum())
        n_ex, oc, hh, _ = y.shape
        ho = hh // 2
        # window i covers rows 2i-1 .. 2i+1, clamped at 0 (the -inf pad of a max): distinct positions
        rows = [sorted({max(2 * i + o - 1, 0) for o in range(3)}) for i in range(ho)]
        xp = np.pad(x, ((0, 0), (0, 0), (3, 3), (3, 3)))
        pos_tie_batch = 0
        for i in range(ho):
            for j in range(ho):
                win = y[:, :, rows[i]][:, :, :, rows[j]].reshape(n_ex, oc, -1)
                mx = win.max(-1)
                at_max = win == mx[..., None]
                tie = (at_max.sum(-1) >= 2) & (mx > 0)
                tot['windows'] += n_ex * oc
                tot['dead'] += int((mx == 0).sum())
                pos_tie_batch += int(tie.sum())
                cells = [(r, s) for r in rows[i] for s in rows[j]]
                for n, o in zip(*np.nonzero(tie)):
                    ks = np.nonzero(at_max[n, o])[0]
                    pats = [xp[n, :, 2 * cells[k][0]:2 * cells[k][0] + 7, 2 * cells[k][1]:2 * cells[k][1] + 7] for k in ks]
                    tot['pos_tie_same_patch'] += all(np.array_equal(pats[0], p) for p in pats[1:])
        tot['pos_tie'] += pos_tie_batch
        tot['batches_with_pos_tie'] += pos_tie_batch > 0
        print(f'batch {bi}: {tot}', flush=True)
    w = tot['windows']
    print(f"\nres {a.res}, {a.batches} batches of {a.batch}: {w} windows")
    print(f"  dead (allowed now, rejected by the old condition): {tot['dead']} ({tot['dead'] / w:.2%})")
    print(f"  positive ties (still excluded):                    {tot['pos_tie']} ({tot['pos_tie'] / w:.2%}),"
          f" {tot['pos_tie_same_patch']} of them identical input patches;"
          f" in {tot['batches_with_pos_tie']} of {a.batches} batches")
    print(f"  pre-ReLU exactly zero:                             {tot['zero_pre']}")
    bad_vjp = tot['pos_tie'] + tot['zero_pre']
    bad_loss = tot['pos_tie'] - tot['pos_tie_same_patch'] + tot['zero_pre']
    print(f"\n  input-VJP stem clauses (R34StemSmoothAt + StemPoolSmoothAt):        "
          f"{'hold' if bad_vjp == 0 else f'fail at {bad_vjp} windows/cells'}")
    print(f"  loss-gradient stem clauses (R34StemSmoothAt + StemPoolTwinAt at StemConvTwin): "
          f"{'hold' if bad_loss == 0 else f'fail at {bad_loss} windows/cells'}")


if __name__ == '__main__':
    main()
