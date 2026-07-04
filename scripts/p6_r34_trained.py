"""P6 — trained-weight gains for ResNet-34 (JAX/GPU).

Loads the trained r34 checkpoint (.lake/build/resnet34_adam_ckpt.bin, first nP of
[params, adam_m, adam_v] in ResNet34Layout.specs order — mapping verified: γ≈1,
β≈0), reconstructs the frozen eval-BN running stats via a true-batch-norm pass
over a real Imagenette calib batch (the stats aren't persisted for r34), then
re-measures the op-granularity + tree-reduction adjoint budget and per-op tail
gains on the TRAINED net at its real operating point — the §11 measurement on
trained weights + real images instead of He-init + noise.

Gate: the reconstructed-stats eval must recover the checkpoint's ~90.7% val acc
(else the operating point is wrong and the tail gains are meaningless).
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

from adjoint_chain_probe import U32, _rfac, _rexp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, ".lake/build/resnet34_adam_ckpt.bin")
VAL = "/home/skoonce/lean/claude_max/lean4-jax/data/imagenette/val.bin"
EPS = 1e-5
mags = lambda a: float(np.abs(np.asarray(a)).max())
stage_cfg = [(64, 64, 3, 1), (64, 128, 4, 2), (128, 256, 6, 2), (256, 512, 3, 2)]


def idBlk(c):
    return [([c, c, 3, 3], 0), ([c], 2), ([c], 1), ([c], 2)] * 2


def downBlk(cin, c):
    return ([([c, cin, 3, 3], 0), ([c], 2), ([c], 1), ([c], 2)]
            + [([c, c, 3, 3], 0), ([c], 2), ([c], 1), ([c], 2)]
            + [([c, cin, 3, 3], 0), ([c], 2), ([c], 1), ([c], 2)])


specs = [([64, 3, 7, 7], 0), ([64], 2), ([64], 1), ([64], 2)]
for _ in range(3):
    specs += idBlk(64)
specs += downBlk(64, 128)
for _ in range(3):
    specs += idBlk(128)
specs += downBlk(128, 256)
for _ in range(5):
    specs += idBlk(256)
specs += downBlk(256, 512)
for _ in range(2):
    specs += idBlk(512)
specs += [([512, 10], 0), ([10], 2)]
sizes = [int(np.prod(d)) for d, _ in specs]
nP = sum(sizes)

flat = np.fromfile(CKPT, dtype=np.float32, count=nP)
params, off = [], 0
for (d, _k) in specs:
    n = int(np.prod(d))
    params.append(jnp.asarray(flat[off:off + n].reshape(d)))
    off += n
it = iter(params)
Wstem, bstem, gstem, btstem = [next(it) for _ in range(4)]
blocks = []
for (ic, oc, n, s) in stage_cfg:
    for j in range(n):
        bic, bs = (ic, s) if j == 0 else (oc, 1)
        blk = {"ic": bic, "oc": oc, "stride": bs, "down": (bs != 1 or bic != oc)}
        blk["W1"], blk["b1"], blk["g1"], blk["bt1"] = [next(it) for _ in range(4)]
        blk["W2"], blk["b2"], blk["g2"], blk["bt2"] = [next(it) for _ in range(4)]
        if blk["down"]:
            blk["Wd"], blk["bd"], blk["gd"], blk["btd"] = [next(it)
                                                           for _ in range(4)]
        blocks.append(blk)
Whead, bhead = next(it), next(it)

# real Imagenette val (ImageNet-normalized). val.bin = 4-byte count header +
# records [label(1) + 224²·3 uint8, channel-first]; the set is class-SORTED, so
# stride-sample across it to cover all 10 classes.
REC = 1 + 3 * 224 * 224
NREC = 3925
allb = np.fromfile(VAL, dtype=np.uint8, offset=4).reshape(NREC, REC)
idx = np.arange(0, NREC, NREC // 320)[:320]          # spread across classes
recs = allb[idx]
labels = recs[:, 0].astype(np.int64)
mean = np.array([0.485, 0.456, 0.406], np.float32)[None, :, None, None]
std = np.array([0.229, 0.224, 0.225], np.float32)[None, :, None, None]
imgs = ((recs[:, 1:].reshape(-1, 3, 224, 224).astype(np.float32) / 255.0)
        - mean) / std


def jconv(a, W, s, p):
    return jax.lax.conv_general_dilated(
        a, W, (s, s), ((p, p), (p, p)),
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))


def jmaxpool2(a):
    N, C, H, Wd = a.shape
    return a.reshape(N, C, H // 2, 2, Wd // 2, 2).max(axis=(3, 5))


def convb(a, W, b, s, p):
    return jconv(a, W, s, p) + b[None, :, None, None]


gpu = jax.devices()[0]


def forward(x, stats, record):
    """stats: dict tag→(mu,var); record=True computes true-BN stats & stores."""
    def bn(z, tag, g, bt):
        if record:
            stats[tag] = (z.mean(axis=(0, 2, 3)), z.var(axis=(0, 2, 3)))
        mu, var = stats[tag]
        return (z - mu[None, :, None, None]) / jnp.sqrt(var[None, :, None, None]
                                                        + EPS) \
            * g[None, :, None, None] + bt[None, :, None, None]
    z = convb(x, Wstem, bstem, 2, 3)
    h = jmaxpool2(jnp.maximum(bn(z, 'st', gstem, btstem), 0))
    for i, b in enumerate(blocks):
        y1 = jnp.maximum(bn(convb(h, b["W1"], b["b1"], b["stride"], 1),
                            f'{i}n1', b["g1"], b["bt1"]), 0)
        y2 = bn(convb(y1, b["W2"], b["b2"], 1, 1), f'{i}n2', b["g2"], b["bt2"])
        sk = bn(convb(h, b["Wd"], b["bd"], b["stride"], 1), f'{i}np',
                b["gd"], b["btd"]) if b["down"] else h
        h = jnp.maximum(y2 + sk, 0)
    return h.mean(axis=(2, 3)) @ Whead + bhead


with jax.default_device(gpu):
    stats = {}
    forward(jnp.asarray(imgs[:256]), stats, True)     # freeze running stats
    stats = {k: (jax.device_get(mu), jax.device_get(v))
             for k, (mu, v) in stats.items()}
    stats = {k: (jnp.asarray(mu), jnp.asarray(v)) for k, (mu, v) in stats.items()}
    la = jax.device_get(forward(jnp.asarray(imgs[:64]), stats, False))
acc = float((la.argmax(1) == labels[:64]).mean())
print(f"GATE: reconstructed-stats eval top-1 on 64 val = {acc:.1%} "
      f"(checkpoint trained to 90.7%)")
print(f"logits magnitude = {mags(la):.3f}")
if acc < 0.5:
    raise SystemExit("gate failed — operating point wrong, aborting P6 measure")

# ═══════════════════════════════════════════════════════════════════════════
# P6 measurement: op-granularity + tree-reduction budget & tail gains on the
# TRAINED net at its real operating point (frozen eval-BN affines a=γ/√(σ²+ε),
# b=β−γμ/√(σ²+ε); real-image batch). Compare to §11 at-init (12.6 / P1+P2 0.17).
# ═══════════════════════════════════════════════════════════════════════════
dt = np.float64
B = 2
xb = imgs[:B]


def npget(a):
    return np.asarray(a, dt)


# frozen eval-BN affine (a, b) per BN from the reconstructed running stats
def affine(tag, g, bt):
    mu, var = stats[tag]
    a = npget(g) / np.sqrt(npget(var) + EPS)
    b = npget(bt) - npget(g) * npget(mu) / np.sqrt(npget(var) + EPS)
    return a, b


ab_stem = affine('st', gstem, btstem)
for i, blk in enumerate(blocks):
    blk["ab1"] = affine(f'{i}n1', blk["g1"], blk["bt1"])
    blk["ab2"] = affine(f'{i}n2', blk["g2"], blk["bt2"])
    if blk["down"]:
        blk["abd"] = affine(f'{i}np', blk["gd"], blk["btd"])

from adjoint_chain_probe import conv_np, maxpool2   # noqa: E402
Wn = {k: npget(v) for k, v in
      [('stem', Wstem)] + [(f'{i}W1', b["W1"]) for i, b in enumerate(blocks)]}


def convb_np(a, W, b, s, p):
    return conv_np(a, W, s, p, dt) + npget(b)[None, :, None, None]


def bnap(z, ab):
    a, b = ab
    return z * a[None, :, None, None] + b[None, :, None, None]


def conv_fresh0(x_act, W, b, s, p):
    m = W.shape[1] * W.shape[2] * W.shape[3]
    sumabs = float(conv_np(np.abs(x_act), np.abs(npget(W)), s, p, dt).max())
    return _rfac(m, 2) * (sumabs + float(np.abs(npget(b)).max()))


def bn_fresh0(ab, A):
    amax = float(np.abs(ab[0]).max())
    bmax = float(np.abs(ab[1]).max())
    return (2 * U32 + U32 * U32) * (amax * A + bmax)


# jax op fns (trained weights + frozen affines)
def jc(a, W, s, p):
    return jax.lax.conv_general_dilated(
        a, W, (s, s), ((p, p), (p, p)),
        dimension_numbers=('NCHW', 'OIHW', 'NCHW'))


def jbn(a, ab):
    return a * jnp.asarray(ab[0].astype(np.float32))[None, :, None, None] \
        + jnp.asarray(ab[1].astype(np.float32))[None, :, None, None]


jrelu = lambda a: jnp.maximum(a, 0.0)


def jxs(arr, smp):
    return jnp.asarray(np.asarray(arr[smp:smp + 1], np.float32))


def block_fwd(bi, h):
    b = blocks[bi]
    y1 = jrelu(jbn(jc(h, b["W1"], b["stride"], 1), b["ab1"]))
    y2 = jbn(jc(y1, b["W2"], 1, 1), b["ab2"])
    sk = jbn(jc(h, b["Wd"], b["stride"], 1), b["abd"]) if b["down"] else h
    return jrelu(y2 + sk)


def tail_from_blockout(bi, h):
    for bb in range(bi + 1, len(blocks)):
        h = block_fwd(bb, h)
    return h.mean(axis=(2, 3)) @ Whead + bhead


def jstem(t):
    N, C, H, W = t.shape
    return (jrelu(jbn(t, ab_stem))).reshape(N, C, H // 2, 2, W // 2,
                                            2).max(axis=(3, 5))


# ── numpy trajectory: record per-op outputs, fresh, tail closures ───────────
ops = []
a = npget(xb)
z = convb_np(a, Wstem, bstem, 2, 3)
ops.append(dict(name="stem-conv", out=z, fresh=conv_fresh0(a, Wstem, bstem,
                                                           2, 3),
                mk=lambda smp: (lambda t: tail_from_blockout(-1, jstem(t)))))
zb = bnap(z, ab_stem)
ops.append(dict(name="stem-bn", out=zb, fresh=bn_fresh0(ab_stem, mags(z)),
                mk=lambda smp: (lambda t: tail_from_blockout(
                    -1, (jrelu(t)).reshape(t.shape[0], t.shape[1],
                                           t.shape[2] // 2, 2,
                                           t.shape[3] // 2, 2).max(axis=(3, 5))))))
h = maxpool2(np.maximum(zb, 0))


def main_seq(bi):
    b = blocks[bi]
    return [lambda t: jc(t, b["W1"], b["stride"], 1),   # conv1
            lambda t: jbn(t, b["ab1"]),                 # bn1
            jrelu,                                      # relu (exact)
            lambda t: jc(t, b["W2"], 1, 1),             # conv2
            lambda t: jbn(t, b["ab2"])]                 # bn2


def skip_seq(bi):
    b = blocks[bi]
    return [lambda t: jc(t, b["Wd"], b["stride"], 1),   # conv_d
            lambda t: jbn(t, b["abd"])]                 # bn_d


def mk_main_tail(bi, j, sk_c):
    # j = branch index of the op that produced the state (c1=0,bn1=1,c2=3,bn2=4)
    seq = main_seq(bi)
    def tail(t, smp):
        for f in seq[j + 1:]:
            t = f(t)
        return tail_from_blockout(bi, jrelu(t + jxs(sk_c, smp)))
    return lambda smp: (lambda t: tail(t, smp))


def mk_skip_tail(bi, j, main_c):
    seq = skip_seq(bi)
    def tail(t, smp):
        for f in seq[j + 1:]:
            t = f(t)
        return tail_from_blockout(bi, jrelu(jxs(main_c, smp) + t))
    return lambda smp: (lambda t: tail(t, smp))


for bi, blk in enumerate(blocks):
    s = blk["stride"]
    dn = f"{bi+1}" + ("↓" if blk["down"] else "")
    z1 = convb_np(h, blk["W1"], blk["b1"], s, 1)
    y1 = np.maximum(bnap(z1, blk["ab1"]), 0)
    z2 = convb_np(y1, blk["W2"], blk["b2"], 1, 1)
    y2 = bnap(z2, blk["ab2"])
    if blk["down"]:
        zd = convb_np(h, blk["Wd"], blk["bd"], s, 1)
        sk = bnap(zd, blk["abd"])
    else:
        sk = h
    merged = y2 + sk
    ops.append(dict(name=f"b{dn} c1", out=z1,
                    fresh=conv_fresh0(h, blk["W1"], blk["b1"], s, 1),
                    mk=mk_main_tail(bi, 0, sk)))
    ops.append(dict(name=f"b{dn} bn1", out=bnap(z1, blk["ab1"]),
                    fresh=bn_fresh0(blk["ab1"], mags(z1)),
                    mk=mk_main_tail(bi, 1, sk)))
    ops.append(dict(name=f"b{dn} c2", out=z2,
                    fresh=conv_fresh0(y1, blk["W2"], blk["b2"], 1, 1),
                    mk=mk_main_tail(bi, 3, sk)))
    ops.append(dict(name=f"b{dn} bn2", out=y2,
                    fresh=bn_fresh0(blk["ab2"], mags(z2)),
                    mk=mk_main_tail(bi, 4, sk)))
    if blk["down"]:
        ops.append(dict(name=f"b{dn} cd", out=zd,
                        fresh=conv_fresh0(h, blk["Wd"], blk["bd"], s, 1),
                        mk=mk_skip_tail(bi, 0, y2)))
        ops.append(dict(name=f"b{dn} bnd", out=sk,
                        fresh=bn_fresh0(blk["abd"], mags(zd)),
                        mk=mk_skip_tail(bi, 1, y2)))
    ops.append(dict(name=f"b{dn} add", out=merged, fresh=U32 * mags(merged),
                    mk=(lambda bi: (lambda smp: (lambda t: tail_from_blockout(
                        bi, jrelu(t)))))(bi)))
    h = np.maximum(merged, 0)

hw = h.shape[2] * h.shape[3]
gap = h.mean(axis=(2, 3))
ops.append(dict(name="gap", out=gap,
                fresh=U32 * (1 + U32) ** _rexp(hw, 1) * mags(h)
                + _rfac(hw, 1) * mags(h),
                mk=lambda smp: (lambda t: t @ Whead + bhead)))
out = gap @ npget(Whead) + npget(bhead)
sumabs_h = float((np.abs(gap) @ np.abs(npget(Whead))).max())
ops.append(dict(name="dense (logits)", out=out,
                fresh=_rfac(512, 2) * (sumabs_h + float(np.abs(npget(bhead))
                                                        .max())), mk=None))

with jax.default_device(gpu):
    cb = 0.0
    rows = []
    for k, op in enumerate(ops):
        if op["mk"] is None:
            H = 1.0
        else:
            H = 0.0
            for smp in range(B):
                tf = op["mk"](smp)
                a1 = jnp.asarray(np.asarray(op["out"][smp:smp + 1], np.float32))
                J = jax.jacrev(lambda t: tf(t)[0])(a1)
                H = max(H, float(jnp.abs(J.reshape(10, -1)).sum(axis=1).max()))
        cb += H * op["fresh"]
        rows.append((op["name"], op["fresh"], H, H * op["fresh"]))

rows.sort(key=lambda r: -r[3])
print("\n  P6 — TRAINED r34 (90.7% net), op-gran + tree-reduce, real images")
print(f"\n{'top op (by H·b)':<16}{'fresh b':>12}{'H meas':>11}{'H·b':>11}")
for name, b, H, c in rows[:12]:
    print(f"{name:<16}{b:>12.3e}{H:>11.3e}{c:>11.3e}")
maxH = max(r[2] for r in rows)
print(f"\n  chainBudget2 (trained, op-gran+P2): {cb:.3e}")
print(f"  logits magnitude                  : {mags(out):.3f}")
print(f"  max tail gain (trained)           : {maxH:.1f}")
print(f"  vs §11 at-init: P1+P2 0.17, early tail gains up to ~1195")

