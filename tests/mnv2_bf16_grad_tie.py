#!/usr/bin/env python3
"""mnv2_bf16_grad_tie.py — is the verified MNv2 render's bf16 GRADIENT the JAX reference's bf16 gradient?

    python3 tests/mnv2_bf16_grad_tie.py                     # GPU, after the GPUs are free
    python3 tests/mnv2_bf16_grad_tie.py --theta init        # at He init instead of a trained θ
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
      python3 tests/mnv2_bf16_grad_tie.py --arms A,B        # plumbing smoke on 4 host devices

WHY IT EXISTS (2026-09-30). The 350-epoch MNv2 pair — JAX `mnv2-full-jax-4gpu` (71.634) against the
verified `mnv2-default-4gpu` (`rmsdp64wxdols0eps0001bf16`) — trails by a STEADY ~0.5 top-1 from
epoch ~126 on, and the verified TRAIN loss is +0.022–0.027 higher in every 25-epoch window from
epoch 26. So the gap is in TRAINING, not eval. A read-only audit matched everything structural:
the RMSProp update op for op, the loss scale (/64 per replica, gradient all-reduce ÷4), sync-BN
forward AND backward (`gdst`), XLA-SAME padding at all 5 stride-2 sites, BN ε, the LR staircase,
the shim, dropout. The render's own header says what is left OUTSIDE its proof: "this artifact's
bf16 conv twins, which round their operands per element, are not in that statement". JAX rounds
where `convdt`/`mm` put the casts and autodiffs through them; the render rounds where its bf16
twins sit, including every dgrad and wgrad. Same nominal precision, possibly different placement.

WHAT IT MEASURES. Four arms get the SAME θ, the SAME 256-image batch, dropout OFF (render mask = 1,
JAX drop_key = None) and zero-init optimizer state (buffer 0, mean-square 1.0), and each takes
ONE step:

    A  JAX reference, bf16 (as it trained)        C  JAX reference, f32 (DT/CONV_DT -> float32)
    B  verified render, bf16 (what is running)    D  verified render, f32 twin (same emitter, bf16 off)

From the momentum buffer b the gradient is recovered exactly (b = g'/sqrt(1.9 + 0.1 g'^2) at this
init; g' = g + wd·mask·θ, and the wd term is subtracted back out). The read is the ladder's G2
measured on the gradient, not θ (planning/archive/xla_pjrt_ladder.md: θ ties are uninformative).

⚠ AT THIS DEPTH A FIXED TOLERANCE IS MEANINGLESS. 52 BN layers amplify backend reassociation (R34's
36 floored at ~6e-3 against ITSELF under a sub-ULP nudge). So every distance is read against the
arm's own floor: A', B', C', D' rerun the arm at θ nudged by --nudge (default 1e-7, relative).
Distances are control-relative, per the tie-gate lesson: magnitude AND spread, not one number.

HOW TO READ IT
  d(D, C) ~ floor         f32 render and f32 JAX are the same function — the structural tie holds.
  d(B, D) vs d(A, C)      each side's OWN bf16 error. Comparable ⇒ precision is placed alike.
                          d(B, D) >> d(A, C) ⇒ the render's bf16 twins lose more than JAX's casts.
  bias β(X vs C)          <g_X, g_C>/<g_C, g_C> - 1 per depth group, against the f32 JAX arm C
                          ONLY. A consistent sign (e.g. verified gradients systematically
                          smaller) is a training-speed effect a rel-L2 can hide.
                          ⛔ NEVER read β against a NOISY reference (A or B): with
                          b = true + noise, <a,b>/<b,b> - 1 ≈ -rel² whatever a is — REGRESSION
                          DILUTION. The first CPU smoke (2026-09-30) read β(B,A) ≈ -0.1 through
                          b4..b17 as "verified gradients 10% smaller"; β(A',A) — JAX against
                          ITSELF — showed the same -0.1. C's floor is ~7e-4, so it is a clean
                          reference; A and B floor at ~7e-2.
  groups                  rel/β per depth group are taken over the CONCATENATED group vector,
                          not averaged per tensor: a conv W feeding a BN can have a near-zero
                          f32 gradient, and one such tensor's per-tensor ratio (~1e2) swamps a
                          group mean. A ramp with depth is reassociation, a spike is wiring.
  loss                    step-0 losses must agree to ~1e-5 rel in each precision (forward tie).

Read-only on the run: θ comes from a JAX checkpoint .bin (default e150, where the gap is already
established), the batch from the shim's VALIDATION split (deterministic center crop), cached under
.lake/build/mnv2bf16tie/. Nothing in the repo or any checkpoint is written.

The f32 twin (arm D) is a gate INPUT, never a committed artifact. Render it with

    lake build LeanMlir.Proofs.Codegen.MobileNetV2RenderB   # ⚠ the olean was stale on 2026-09-30,
                                                            # and this rewrites verified_mlir/ —
                                                            # `git status verified_mlir/` must be clean
    lake env lean <file with the #eval below>
      #eval IO.FS.writeFile ".lake/build/mnv2bf16tie/mobilenetv2in_rmsdp64wxdols0eps0001_train_step.mlir"
        (Proofs.StableHLO.mobilenetv2AdamTrainStepFaithfulB 64 1000 "1.0e-3" 4 false "mobilenetv2in"
          Proofs.StableHLO.OptKind.rmsprop false (wdExclude := true) (cd := true) (alpha := 0.0))

Without it the script runs A, B, C and says D is missing.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, ".lake/build/mnv2bf16tie")
JAX_REF = os.path.join(REPO, "jax/.lake/build/generated_mobilenet_v2_imagenet_full.py")
SHIM = os.path.join(REPO, "jax/.lake/build/generated_mobilenet_v2_imagenet_shim.py")
RENDER_BF16 = os.path.join(REPO, "verified_mlir/mobilenetv2in_rmsdp64wxdols0eps0001bf16_train_step.mlir")
RENDER_F32 = os.path.join(OUT_DIR, "mobilenetv2in_rmsdp64wxdols0eps0001_train_step.mlir")
CKPT_DIR = "/home/skoonce/mnv2_full350_relu6"
PY = "/home/skoonce/.venv-cuda/bin/python3"

REPLICAS, PER_REPLICA, NCLS, FLAT = 4, 64, 1000, 3 * 224 * 224
GB = REPLICAS * PER_REPLICA
RHO, MU, EPS, WD = 0.9, 0.9, 1.0, 4e-5          # the recipe; asserted against both sides below


# ── the batch ─────────────────────────────────────────────────────────────────────────────────
def load_batch():
    """First 256 images of the shim's VALIDATION stream: center crop, no aug, deterministic."""
    cache = os.path.join(OUT_DIR, "valbatch256.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["x"], d["y"]
    env = dict(os.environ, SHIM_SPLIT="validation", SHIM_BATCH=str(GB), SHIM_SEED="0",
               CUDA_VISIBLE_DEVICES="")
    env.pop("SHIM_NCLASSES", None)
    p = subprocess.Popen([PY, SHIM], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env)

    def rd(n):
        b = b""
        while len(b) < n:
            c = p.stdout.read(n - len(b))
            if not c:
                raise SystemExit("shim closed the pipe early")
            b += c
        return b
    magic = rd(4)
    ver, batch, flat = np.frombuffer(rd(12), np.int32)
    assert magic == b"LMSH" and ver == 3 and batch == GB and flat == FLAT, (magic, ver, batch, flat)
    rows = int(np.frombuffer(rd(4), np.int32)[0])
    assert rows == GB, rows
    y = np.frombuffer(rd(4 * rows), np.int32).copy()
    x = np.frombuffer(rd(4 * rows * FLAT), np.float32).reshape(rows, FLAT).copy()
    p.kill(); p.wait()
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(cache, x=x, y=y)
    return x, y


# ── the JAX reference, imported (not reimplemented) ───────────────────────────────────────────
def load_ref(precision):
    """The generated reference as a module. Its `Main` is under `if __name__ == "__main__"`, so
    importing runs no training. f32 = the same source with the two dtype knobs flipped."""
    src = open(JAX_REF).read()
    if precision == "f32":
        for knob in ("DT = jnp.bfloat16", "CONV_DT = jnp.bfloat16"):
            assert len(re.findall(rf"^{re.escape(knob)}$", src, re.M)) == 1, knob
            src = re.sub(rf"^{re.escape(knob)}$", knob.replace("bfloat16", "float32"), src, flags=re.M)
    for name, want in (("RHO", RHO), ("MOMENTUM", MU), ("EPS", EPS), ("WD", WD)):
        got = float(re.search(rf"^{name} = ([0-9.e-]+)$", src, re.M).group(1))
        assert abs(got - want) < 1e-12, (name, got, want)
    spec = importlib.util.spec_from_loader(f"mnv2ref_{precision}", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = JAX_REF
    exec(compile(src, JAX_REF, "exec"), mod.__dict__)
    return mod


def ref_step(M, params, x, y, lr):
    import jax
    import jax.numpy as jnp
    M.WD_MASK = M._wd_mask(params)          # train_step reads this global at trace time
    opt = (jax.tree.map(jnp.ones_like, params), jax.tree.map(jnp.zeros_like, params))
    p = jax.device_put(params, M.replicated_sharding)
    xs = jax.device_put(x, M.data_sharding)
    ys = jax.device_put(y, M.data_sharding)
    _, (_, buf), _, loss = M.train_step(p, opt, M.init_bn_state(), xs, ys, jnp.float32(lr), None)
    leaves = [np.asarray(jax.device_get(l)) for l in jax.tree.leaves(buf)]
    leaves[-2] = leaves[-2].T                       # dense W: JAX (1000,1280) -> render (1280,1000)
    return leaves, float(loss)


# ── the verified render, executed as-is over 4 replicas ───────────────────────────────────────
def parse_render(path):
    src = open(path).read()
    entry = re.search(r"func\.func @([a-zA-Z0-9_]+)\(", src).group(1)
    sig = re.search(r"func\.func @[a-zA-Z0-9_]+\((.*?)\)\s*->", src, re.S).group(1)
    args = [(n, [] if t.replace("f32", "").rstrip("x") == "" else
             [int(v) for v in t.replace("f32", "").rstrip("x").split("x")])
            for n, t in re.findall(r"%([a-zA-Z_0-9]+):\s*tensor<([^>]*)>", sig)]
    seg = src[src.rindex("return"):]
    rets = re.findall(r"%([a-zA-Z_0-9]+)", seg[:seg.index(" : tensor")])
    return src.replace("@" + entry, "@main"), args, rets


def render_step(path, theta, x, y, lr, _cache={}):
    import jax
    import jax.extend.backend
    from jax._src.lib import xla_client as xc
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    src, args, rets = parse_render(path)
    names = [n for n, _ in args]
    x_i, do_i, oh_i = names.index("x"), names.index("do"), names.index("onehot")
    nP = (names.index("lr") - 1) // 3             # x | θ | m | v | lr bc1 bc2 | BN stats | do onehot
    assert nP == len(theta) == 158, (nP, len(theta))
    for (n, shp), t in zip(args[1:1 + nP], theta):
        assert list(t.shape) == shp, (n, t.shape, shp)
    assert rets[nP] != rets[0] and len(rets) >= 3 * nP
    devs = jax.devices()[:REPLICAS]
    assert len(devs) == REPLICAS, f"need {REPLICAS} devices, have {jax.devices()}"
    mesh = Mesh(np.array(devs), ("r",))
    if path not in _cache:
        opts = xc.CompileOptions()
        ebo = opts.executable_build_options
        ebo.num_replicas, ebo.num_partitions = REPLICAS, 1
        ebo.device_assignment = xc.DeviceAssignment.create(np.array([[d.id] for d in devs]))
        t0 = time.time()
        _cache[path] = jax.extend.backend.get_backend().compile_and_load(
            src, xc.DeviceList(tuple(devs)), opts)
        print(f"    compiled {os.path.basename(path)} in {time.time() - t0:.0f}s", flush=True)
    exe = _cache[path]

    onehot = np.zeros((GB, NCLS), np.float32); onehot[np.arange(GB), y] = 1.0
    per = {x_i: x, oh_i: onehot, do_i: np.ones((GB, FLAT_DO), np.float32)}
    vals = [None] * len(args)
    for i, t in enumerate(theta):
        vals[1 + i] = t                                             # θ
        vals[1 + nP + i] = np.zeros_like(t)                         # momentum buffer 0
        vals[1 + 2 * nP + i] = np.ones_like(t)                      # mean-square 1.0 (TF)
    for i, (n, shp) in enumerate(args):
        if vals[i] is None and i not in per:
            vals[i] = (np.float32(lr) if n == "lr" else
                       np.float32(1.0) if n in ("bc1", "bc2") else
                       (np.ones if n.endswith("vari") else np.zeros)(shp, np.float32))
    arrays = []
    for i, v in enumerate(vals):
        if i in per:                                                # sharded on the batch axis
            g = per[i]
            shards = [jax.device_put(g[r * PER_REPLICA:(r + 1) * PER_REPLICA], d) for r, d in enumerate(devs)]
            arrays.append(jax.make_array_from_single_device_arrays(g.shape, NamedSharding(mesh, P("r")), shards))
        else:                                                       # replicated
            v = np.asarray(v, np.float32)
            arrays.append(jax.make_array_from_single_device_arrays(
                v.shape, NamedSharding(mesh, P()), [jax.device_put(v, d) for d in devs]))
    outs = exe.execute_sharded(arrays).disassemble_into_single_device_arrays()
    buf = []
    for i in range(nP):
        reps = [np.asarray(a) for a in outs[nP + i]]
        spread = max(float(np.abs(r - reps[0]).max()) for r in reps)
        assert spread == 0.0, f"replicas disagree on m'[{args[1 + i][0]}] by {spread}"
        buf.append(reps[0])
    loss = None
    if "loss" in rets:
        loss = float(np.mean([np.asarray(a) for a in outs[rets.index("loss")]]))
    return buf, loss, [n for n, _ in args[1:1 + nP]]


FLAT_DO = 1280


# ── analysis ──────────────────────────────────────────────────────────────────────────────────
def grad_from_buf(buf, theta, wdmask):
    """b = g'/sqrt(ρ·1 + (1-ρ)g'^2 + ε) with buffer 0, mean-square 1 ⇒ g'; then g = g' - wd·mask·θ."""
    out = []
    for b, t, m in zip(buf, theta, wdmask):
        b = b.astype(np.float64)
        g2 = (RHO + EPS) * b * b / (1.0 - (1.0 - RHO) * b * b)
        out.append(np.sign(b) * np.sqrt(g2) - WD * m * t.astype(np.float64))
    return out


def dist(a, b):
    fa = np.concatenate([t.ravel() for t in a]); fb = np.concatenate([t.ravel() for t in b])
    per = [float(np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30)) for x, y in zip(a, b)]
    beta = [float(np.dot(x.ravel(), y.ravel()) / (np.dot(y.ravel(), y.ravel()) + 1e-30) - 1.0)
            for x, y in zip(a, b)]
    return {"rel": float(np.linalg.norm(fa - fb) / np.linalg.norm(fb)),
            "sq": [[float(np.dot((x - y).ravel(), (x - y).ravel())), float(np.dot(y.ravel(), y.ravel())),
                    float(np.dot(x.ravel(), y.ravel()))] for x, y in zip(a, b)],
            "cos": float(np.dot(fa, fb) / (np.linalg.norm(fa) * np.linalg.norm(fb))),
            "beta_all": float(np.dot(fa, fb) / np.dot(fb, fb) - 1.0),
            "per_rel": per, "per_beta": beta}


def groups(names):
    """Depth groups off the render's names: s (stem), b1..b17, h (head conv), d (dense)."""
    out = []
    for n in names:
        m = re.match(r"(s|b\d+|h|W?d|bd)", n)
        g = m.group(1) if m else n
        out.append("dense" if g in ("Wd", "bd", "d") else g)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", default="e150", help="init | eNNN (JAX checkpoint epoch) | path.bin")
    ap.add_argument("--arms", default="A,B,C,D", help="subset of A,B,C,D")
    ap.add_argument("--nudge", type=float, default=1e-7, help="relative θ nudge for the floor arms")
    ap.add_argument("--no-floor", action="store_true", help="skip the nudged reruns")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "result.json"))
    a = ap.parse_args()
    arms = a.arms.split(",")
    os.makedirs(OUT_DIR, exist_ok=True)

    import jax
    print(f"jax {jax.__version__} on {jax.devices()}", flush=True)
    x, y = load_batch()
    print(f"batch: {x.shape} labels[:8]={y[:8].tolist()}", flush=True)

    MB = load_ref("bf16")
    if a.theta == "init":
        params = MB.init_params(jax.random.PRNGKey(314159)); lr = 0.045
    else:
        path = a.theta if a.theta.endswith(".bin") else f"{CKPT_DIR}/mobilenet_v2_imagenet_{a.theta}.bin"
        params = MB.init_params_from_file(path)
        ep = int(re.search(r"_e(\d+)\.bin$", path).group(1)) if re.search(r"_e(\d+)\.bin$", path) else 0
        lr = 0.045 * 0.98 ** ep          # the LR of the step AFTER epoch ep
        print(f"θ from {path} (lr {lr:.6g})", flush=True)
    leaves = [np.asarray(l, np.float32) for l in jax.tree.leaves(params)]
    leaves[-2] = leaves[-2].T
    wdmask = [np.float64(l.ndim > 1) for l in leaves]

    rng = np.random.default_rng(7)
    u = [rng.standard_normal(l.shape) for l in leaves]
    un = np.sqrt(sum(float((v * v).sum()) for v in u)); tn = np.sqrt(sum(float((l.astype(np.float64) ** 2).sum()) for l in leaves))
    nudged = [(l + a.nudge * tn * v / un).astype(np.float32) for l, v in zip(leaves, u)]

    def as_ref_params(ls):
        ls = list(ls); ls[-2] = ls[-2].T
        return jax.tree.unflatten(jax.tree.structure(params), [jax.numpy.asarray(l) for l in ls])

    MF = load_ref("f32") if "C" in arms else None
    runners = {
        "A": lambda th: ref_step(MB, as_ref_params(th), x, y, lr),
        "C": lambda th: ref_step(MF, as_ref_params(th), x, y, lr),
        "B": lambda th: render_step(RENDER_BF16, th, x, y, lr)[:2],
        "D": lambda th: render_step(RENDER_F32, th, x, y, lr)[:2],
    }
    if "D" in arms and not os.path.exists(RENDER_F32):
        print(f"⚠ arm D skipped: {RENDER_F32} not rendered (see the module docstring)")
        arms.remove("D")
    names = [n for n, _ in parse_render(RENDER_BF16)[1][1:159]]

    G, L = {}, {}
    for arm in arms:
        for tag, th in ((arm, leaves),) + (() if a.no_floor else ((arm + "'", nudged),)):
            t0 = time.time()
            buf, loss = runners[arm](th)
            G[tag] = grad_from_buf(buf, th, wdmask); L[tag] = loss
            print(f"  arm {tag:3s} loss {loss}  ({time.time() - t0:.0f}s)", flush=True)

    pairs = [("B", "A", "verified bf16 vs JAX bf16   ← the question"),
             ("D", "C", "verified f32 vs JAX f32     ← structural tie"),
             ("A", "C", "JAX's own bf16 error"),
             ("B", "D", "verified's own bf16 error"),
             ("B", "C", "verified bf16 vs f32 truth"),
             ("A'", "A", "floor A"), ("B'", "B", "floor B"), ("C'", "C", "floor C"), ("D'", "D", "floor D")]
    res = {"theta": a.theta, "nudge": a.nudge, "loss": L, "pairs": {}}
    print("\n  pair        rel L2     cos          β(all)     what")
    for p, q, what in pairs:
        if p in G and q in G:
            d = dist(G[p], G[q]); res["pairs"][f"{p}|{q}"] = d
            print(f"  {p:3s}vs{q:3s}  {d['rel']:.3e}  {d['cos']:.9f}  {d['beta_all']:+.3e}  {what}")
    def grp(key, idx):
        """(rel, β) over the concatenated group vector, from per-tensor sums of squares."""
        sq = res["pairs"][key]["sq"]
        dd = sum(sq[i][0] for i in idx); bb = sum(sq[i][1] for i in idx); ab = sum(sq[i][2] for i in idx)
        return np.sqrt(dd / bb), ab / bb - 1.0
    gs = groups(names); order = list(dict.fromkeys(gs))
    cols = [k for k in ("B|A", "A'|A", "B'|B", "A|C", "B|C", "D|C") if k in res["pairs"]]
    if cols:
        print("\n  depth profile — rel L2 per group (concatenated); β only against C (see docstring):")
        print("    group  " + "  ".join(f"{k:>10s}" for k in cols) + "   β(A,C)     β(B,C)")
        for g in order:
            idx = [i for i, k in enumerate(gs) if k == g]
            rels = "  ".join(f"{grp(k, idx)[0]:10.3e}" for k in cols)
            bet = "  ".join(f"{grp(k, idx)[1]:+.3e}" if k in res["pairs"] else "    n/a   "
                            for k in ("A|C", "B|C"))
            print(f"    {g:6s} {rels}   {bet}")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\n→ {a.out}")


if __name__ == "__main__":
    main()
