#!/usr/bin/env python
"""CPU gates for `generateVjp` / `generateApplyAdam` (planning/muzero_ttt_demo.md §2).

  1. lake build vjp-probe && .lake/build/bin/vjp-probe      # writes .lake/build/vjp_probe/
  2. <iree+jax venv>/bin/python scripts/probes/vjp_gate.py

Per probe spec, every module compiled with IREE for llvm-cpu and run in-process:

  fwd     the eval forward against a JAX re-implementation of the spec (the reference the
          other gates lean on).
  ∂θ      the SGD + momentum train step at lr 1, zero momentum buffer, no weight decay, on the MSE
          block, returns θ' = θ − g in f32 where g is its own parameter gradient. The VJP fed that
          loss's cotangent ḡ = 2(out − y)/N must give back g to the float: f32(θ − ∂θ_vjp) equals
          θ' bit for bit. (Comparing θ − θ' against ∂θ instead would bury a small gradient under
          θ's rounding.)
  ∂x      central finite differences of ⟨ḡ, net(x)⟩ in x along random directions, on the emitted
          eval module itself — no reference implementation involved.
  jax     ∂θ and ∂x against `jax.vjp` of the JAX forward at the same θ, x, ḡ.
  adam    the Adam apply on the VJP's ∂θ from zero moments at t = 1 against the Adam train step's
          θ', m', v' on the same batch, and against numpy's Adam.

GPU not required: CPU is the whole point. Needs iree.compiler + iree.runtime and jax in one
python (the IREE install must be one install; see scripts/lib/_iree.py), e.g.

  uv venv --python 3.12 ../venv-cpu && uv pip install --python ../venv-cpu/bin/python \
      jax==0.11.0 jaxlib==0.11.0 numpy==2.4.4 iree-base-compiler==3.12.0 iree-base-runtime==3.12.0
"""
import json
import os
import pathlib
import re
import sys
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
import _iree  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

PROBE_DIR = pathlib.Path(".lake/build/vjp_probe")
SPECS = ("convstack", "convhead", "mlp", "muz_h3", "muz_g3", "muz_f3", "muz_r3")

failures = []


def check(ok, what):
    print(f"    {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


# ------------------------------------------------------------------ modules
SIG = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*): tensor<([0-9x]*)x?(f32|i32)>")


FUNC = re.compile(r"func\.func @([A-Za-z0-9_]+)\(")


def signature(text):
    """The entry function's argument names and shapes, in order, from the emitted text."""
    head = text[FUNC.search(text).start():]
    head = head[: head.index(") ->")]
    return [(m.group(1), tuple(int(d) for d in m.group(2).split("x") if d)) for m in SIG.finditer(head)]


def load(path, work):
    text = path.read_text()
    mod = re.search(r"module @([A-Za-z0-9_]+)", text).group(1)
    vmfb = _iree.compile_mlir(path, f"{work}/{path.stem}.vmfb", what=path.name)
    fn = _iree.load_function(vmfb, mod, FUNC.search(text).group(1))

    def call(**named):
        args = [np.ascontiguousarray(named[nm], dtype=np.float32).reshape(sh) for nm, sh in signature(text)]
        out = fn(*args)
        out = out if isinstance(out, (tuple, list)) else [out]
        return [np.asarray(o.to_host() if hasattr(o, "to_host") else o) for o in out]
    return call, signature(text)


# ------------------------------------------------------------------ the JAX reference
def jax_forward(spec):
    B, H, W = spec["B"], spec["imageH"], spec["imageW"]

    def f(theta, x):
        it = iter(theta)
        a = x
        first = spec["layers"][0]
        if first["kind"] == "conv2d":
            a = a.reshape(B, first["ic"], H, W)
        for l in spec["layers"]:
            if l["kind"] == "conv2d":
                w, b = next(it), next(it)
                a = jax.lax.conv_general_dilated(a, w, (1, 1), "SAME",
                                                 dimension_numbers=("NCHW", "OIHW", "NCHW"),
                                                 precision=jax.lax.Precision.HIGHEST)
                a = a + b[None, :, None, None]
            elif l["kind"] == "dense":
                w, b = next(it), next(it)
                a = jnp.dot(a, w, precision=jax.lax.Precision.HIGHEST) + b
            elif l["kind"] == "flatten":
                a = a.reshape(B, -1)
                continue
            else:
                raise SystemExit(f"jax reference: no layer {l['kind']}")
            if l["act"] == "relu":
                a = jnp.maximum(a, 0.0)
        return a
    return f


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


# ------------------------------------------------------------------ one spec
def gate(name, work, rng):
    spec = json.loads((PROBE_DIR / f"{name}.json").read_text())
    print(f"{name}: {spec['layers'].__len__()} layers, in {spec['inDim']}, out {spec['outShape']}")
    vjp, vsig = load(PROBE_DIR / f"{name}_vjp.mlir", work)
    ev, _ = load(PROBE_DIR / f"{name}_eval.mlir", work)
    sgd, _ = load(PROBE_DIR / f"{name}_sgd.mlir", work)
    astep, _ = load(PROBE_DIR / f"{name}_adam_step.mlir", work)
    apply_adam, _ = load(PROBE_DIR / f"{name}_apply_adam.mlir", work)

    params = [(nm, sh) for nm, sh in vsig if nm not in ("x_flat", "gbar")]
    theta = {nm: (rng.standard_normal(sh) * (0.5 if nm.startswith("W") else 0.1)).astype(np.float32)
             for nm, sh in params}
    zeros = {f"{p}{nm}": np.zeros(sh, np.float32) for nm, sh in params for p in ("m_", "v_")}
    B, inDim = spec["B"], spec["inDim"]
    x = rng.standard_normal((B, inDim)).astype(np.float32)
    out_shape = tuple(spec["outShape"])

    # fwd: the eval module against the JAX forward
    out = ev(**theta, x=x)[0].reshape(out_shape)
    fwd = jax_forward(spec)
    tl = [jnp.asarray(theta[nm]) for nm, _ in params]
    out_j = np.asarray(fwd(tl, jnp.asarray(x)))
    check(rel(out, out_j) < 1e-5, f"fwd: eval vs JAX forward, rel {rel(out, out_j):.2e}")

    # the MSE block's cotangent, as the block computes it: (diff · 2) / N in f32
    y = (out - rng.standard_normal(out_shape).astype(np.float32)).astype(np.float32)
    N = np.float32(np.prod(out_shape))
    gbar = ((out - y) * np.float32(2.0)) / N
    y_arg = y.reshape(y.shape + (1, 1)) if len(out_shape) == 2 else y

    d = vjp(**theta, x_flat=x, gbar=gbar)
    dtheta = {nm: d[i] for i, (nm, _) in enumerate(params)}
    dx = d[-1]

    # ∂θ: SGD at lr 1 from a zero momentum buffer returns θ − g in f32
    one = np.float32(1.0)
    so = sgd(**theta, **zeros, x_flat=x, y_ddpm=y_arg, lr=one, t=one)
    n_bits = n_all = 0
    worst = 0.0
    for i, (nm, _) in enumerate(params):
        want = so[i]
        got = (theta[nm] - dtheta[nm]).astype(np.float32)
        n_bits += int(np.sum(got.view(np.uint32) == want.view(np.uint32)))
        n_all += got.size
        worst = max(worst, float(np.max(np.abs(got.astype(np.float64) - want))))
    check(n_bits == n_all, f"∂θ: f32(θ − ∂θ_vjp) == SGD θ' on {n_bits}/{n_all} elements (max |Δ| {worst:.1e})")

    # ∂x: central differences of <ḡ, eval(x)> on the emitted module
    def phi(xx):
        return float(np.sum(ev(**theta, x=xx)[0].reshape(out_shape).astype(np.float64) * gbar))
    # The step is a compromise on a piecewise-linear f32 net: at 1e-2 a direction crosses ReLU
    # kinks in convstack's three convs (rel errors 3–4e-2 on 2 of 4 directions, every one of which
    # jax.vjp matches to 2e-7), at 3e-4 f32 rounding of ⟨ḡ, net⟩ dominates (2–5e-3); 1e-3 sits
    # between, at ≲ 2e-3.
    worst_fd = 0.0
    for _ in range(16):
        u = rng.standard_normal(x.shape)
        u /= np.linalg.norm(u)
        eps = 1e-3
        fd = (phi((x + eps * u).astype(np.float32)) - phi((x - eps * u).astype(np.float32))) / (2 * eps)
        an = float(np.sum(dx.astype(np.float64) * u))
        # on the scale of ‖∂x‖ (|⟨∂x, u⟩| ≤ ‖∂x‖ for a unit u): a direction nearly orthogonal to
        # the gradient has a near-zero derivative, and its own size is no yardstick
        worst_fd = max(worst_fd, abs(fd - an) / np.linalg.norm(dx))
    check(worst_fd < 1e-2, f"∂x: central differences on 16 directions, worst |fd − ⟨∂x, u⟩| / ‖∂x‖ {worst_fd:.2e}")

    # jax.vjp at the same θ, x, ḡ
    _, pull = jax.vjp(fwd, tl, jnp.asarray(x))
    jt, jx = pull(jnp.asarray(gbar))
    r_t = max(rel(dtheta[nm], np.asarray(jt[i])) for i, (nm, _) in enumerate(params))
    r_x = rel(dx, np.asarray(jx))
    check(r_t < 1e-5, f"jax: ∂θ vs jax.vjp, worst tensor rel {r_t:.2e}")
    check(r_x < 1e-5, f"jax: ∂x vs jax.vjp, rel {r_x:.2e}")

    # adam: apply on ∂θ_vjp from zero moments at t = 1 vs the Adam train step on the same batch
    lr = np.float32(1e-3)
    ao = astep(**theta, **zeros, x_flat=x, y_ddpm=y_arg, lr=lr, t=one)
    po = apply_adam(**theta, **zeros, **{f"d_{nm}": dtheta[nm] for nm, _ in params}, lr=lr, t=one)
    k = len(params)
    r_step = max(rel(po[j], ao[j]) for j in range(3 * k))
    check(r_step < 1e-6, f"adam: apply_adam vs the Adam train step (θ', m', v'), worst rel {r_step:.2e}")
    # numpy Adam at t = 1 from zero moments, a second state at t = 2 from the first
    m = {nm: np.zeros_like(theta[nm]) for nm, _ in params}
    v = {nm: np.zeros_like(theta[nm]) for nm, _ in params}
    th = dict(theta)
    worst_np = 0.0
    for t in (1, 2):
        po = apply_adam(**th, **{f"m_{nm}": m[nm] for nm, _ in params}, **{f"v_{nm}": v[nm] for nm, _ in params},
                        **{f"d_{nm}": dtheta[nm] for nm, _ in params}, lr=lr, t=np.float32(t))
        for i, (nm, _) in enumerate(params):
            g = dtheta[nm].astype(np.float64)
            m[nm] = 0.9 * m[nm] + 0.1 * g
            v[nm] = 0.999 * v[nm] + 0.001 * g * g
            mh, vh = m[nm] / (1 - 0.9 ** t), v[nm] / (1 - 0.999 ** t)
            want = th[nm] - float(lr) * mh / (np.sqrt(vh) + 1e-8)
            worst_np = max(worst_np, rel(po[i] - th[nm], want - th[nm]))
            th[nm] = po[i]
            m[nm], v[nm] = po[k + i], po[2 * k + i]
    check(worst_np < 1e-4, f"adam: apply_adam vs numpy Adam over 2 steps, worst rel update {worst_np:.2e}")


def main():
    if not PROBE_DIR.is_dir():
        sys.exit(f"{PROBE_DIR} missing: lake build vjp-probe && .lake/build/bin/vjp-probe")
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as work:
        for name in SPECS:
            gate(name, work, rng)
    if failures:
        print(f"\n⛔ {len(failures)} gate(s) failed")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)
    print(f"\nall vjp gates pass ({len(SPECS)} specs)")


if __name__ == "__main__":
    main()
