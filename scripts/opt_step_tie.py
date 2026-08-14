#!/usr/bin/env python3
"""OPTIMIZER tie — one step of the verified rendered optimizer against the reference's own.

⭐⭐ `planning/verified_optimizer_parity.md` §5's gate, and the gap it closes is stated there:
*the reference and the verified path share a data pipeline BY CONSTRUCTION and share an optimizer
BY NOBODY'S CONSTRUCTION.* `tests/vjp_oracle` and `scripts/grad_tie.py` diff the two at the
GRADIENT; until this file nothing diffed them at the UPDATE. D1 is what a drift between them looks
like — timm's `Lamb` clips by default and every LAMB artifact rendered before 2026-08-14 did not.

## What is compared

For each variant, on ONE step from the same `(θ, g, m, v, G)`:

    θ'   the updated parameters
    m'   the first moment
    v'   the second moment
    G'   ⭐ the ACCUMULATOR, and it is the sharpest row here — see below

## ⚠⚠ Why `G'` is the row that matters

D1 forced a decision the reference never had to make. The reference clips the MEAN accumulated
gradient (`generated_*.py`: `grads = _gsum / _K` and only THEN the clip line), and its accumulator
carry `_gsum` is RAW — clipped only on the way into the optimizer. The verified render carries the
SUM and folds the `1/k` into `%ob1`/`%ob2` downstream, so it clips `Gt` with a `k`-scaled threshold
(`Proofs.clipFactor_accum`) and must return the UNCLIPPED `Gt` as its fourth region.

Returning the clipped total instead would compound the clip across the k micro-batches of every
cycle. That variant trains, descends, and passes every other check in this file. `G' == _gsum` is
what says it did not happen.

## ⚠⚠ The reference side is EXECUTED, not re-implemented

`scripts/grad_tie.py`'s standing rule — *`ref_py` must be the GENERATED reference, not a
re-implementation, because the whole point is to check against the code that produced the published
number* — applies with more force here, since an optimizer is short enough that a re-implementation
would look obviously right and could still differ in the placement that matters.

So this file EXTRACTS the optimizer lines from the generated Python and `exec`s them, verbatim, with
`(params, opt_state, grads/_gsum, lr, WD_MASK, _K)` supplied. **No line is edited, added or
skipped** — the variants in `tests/TestOptStepFixtures.lean` were chosen so that each maps onto a
reference that ships, including `_K = 1` for the no-accumulation row (`grads = _gsum / 1` IS the
non-accumulating case). `_ls` is supplied only because `loss = jnp.mean(_ls)` sits inside the
extracted span; the loss is not compared and is on no gradient path here.

## ⭐⭐ THE GATE HAS TEETH, MEASURED — not asserted

A green gate means nothing until something has been shown to turn it red, which is
`scripts/misplace_drop_sites.py`'s rule applied here: *an all-ones-mask gate is structurally blind
to placement.* Both of D1's genuinely new decisions were reverted, deliberately, and re-run
(2026-08-14). Numbers are worst relative disagreement; the noise floor is ~1e-7.

| counterfactual | `lamb` | `lambwxclip` | `lambacc4wxclip` | `lambacc8wxclip` |
|---|---|---|---|---|
| *(shipped)* | 8.1e-08 ✓ | 8.7e-08 ✓ | 1.1e-07 ✓ | 1.1e-07 ✓ |
| **① threshold not scaled by `k`** (bake `C`, `ε` instead of `k·C`, `k·ε`) | ✓ | ✓ | **1.8e-01 ✗** | **2.1e-01 ✗** |
| **② region 4 returns the CLIPPED total** (the compounding clip) | ✓ | ✓ | **9.4e-01 ✗** | **8.8e-01 ✗** |

▶ Both fire six orders of magnitude above the floor, and both fire **only on the accumulating
rows** — which is what the `k = 1` control is for: a failure in `lambacc*` and not in `lambwxclip`
localises to the accumulation arithmetic rather than to the clip itself. Under ② the reported
`worst 3` is `G'[0..2] (RAW)`, i.e. the gate names the accumulator region rather than leaving a
reader to bisect.

| **③ the AdamW row pointed at the LAMB reference** | — | — | — | `adamwxclipwd002` **1.1e+00 ✗** |
| **④ `wdStr` not reaching the constant block** (the `wd001` row run against the wd = 0.02 reference) | ✓ | ✓ | ✓ | ✓, and `wd001` **9.5e-05 ✗** |

▶ ④ localises perfectly on its own: the worst rows are `θ'[2]` and `θ'[0]`, the two DECAYING
parameters, while `θ'[1]` — the rank-1 one `wx` excludes from decay — is untouched. A decay knob
that failed to reach the graph could not produce that pattern by accident.

⚠ ① is ALSO caught statically, before this file runs — the `#guard`s under `clipNormStr` in
`ResNet34RenderB.lean` refuse to build. That is defence in depth and not redundancy: the guards pin
the two literals, this pins that the literals are used on the right tensor in the right order.

## ⚠ Coverage this does NOT have

✅ `.adamw` IS now gated (2026-08-14), via the `adam-probe` recipe — which bakes eps 1e-8 like the
render and wd 0.02 unlike it, so the row is only expressible because `wdStr` made the decay a render
PARAMETER. Before that, gating `.adamw` would have meant hand-writing a reference, which is the one
thing this file refuses to do.

⚠ Still uncovered: `.heavyBall` (no generated reference uses it — `2018` is SGD+momentum but its
update is not the render's coupled-L2 heavy ball) and `.adamwAccum` (no config composes AdamW with
accumulation; `.lambAccum` is the one RSB renders). Both want a config that generates them.

Usage:
    lake build opt-step-fixtures && .lake/build/bin/opt-step-fixtures
    .venv/bin/python scripts/opt_step_tie.py

⚠ Run under the PINNED interpreter (`.venv`, jax 0.11.0). Runs on CPU in seconds — the fixtures are
three tensors of 48/5/24 floats, not a net.

⚠⚠ **`jax/.lake/build/generated_*.py` IS A BUILD PRODUCT AND GOES STALE.** The local copies were
four commits behind `f8cd3a9` when this file was written — that particular drift was confined to
the eval pipeline and left the optimizer span byte-identical, but a tie measured against a stale
reference is worth nothing, and nothing on disk says which you have. Regenerate before trusting a
green run:

    (cd jax && lake build resnet50-imagenet &&
     for r in default a2-accum a1 adam-probe; do
       lake exe resnet50-imagenet $r /nonexistent >/dev/null 2>&1 || true; done)

The `optimizer` job in `.github/workflows/jax.yml` does exactly this, every run, for this reason.
"""
import os, re, sys
import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
REF_DIR = "jax/.lake/build"
FIX_DIR = ".lake/build"

# The fixture's parameters, and they must agree with `fixtureParams` in
# tests/TestOptStepFixtures.lean. ⚠ Three of them, two decaying and one rank-1: with a single leaf
# the GLOBAL norm and the PER-TENSOR norm coincide, so a per-parameter clip would pass every check
# below (`Proofs.clipFactor_shared` against `Proofs.lambScale_not_shared`).
SHAPES = [(4, 3, 2, 2), (5,), (6, 4)]

# (fixture slug, reference file, _K). See TestOptStepFixtures.lean's `variants` docstring for why
# each row maps onto the reference it does.
VARIANTS = [
    ("lamb",           "generated_resnet50_imagenet.py",          1, False),
    ("lambwxclip",     "generated_resnet50_imagenet_a2accum.py",  1, False),
    ("lambacc4wxclip", "generated_resnet50_imagenet_a2accum.py",  4, True),
    ("lambacc8wxclip", "generated_resnet50_imagenet_a2accum.py",  8, True),
    # ⭐ RSB-A1's decay: 0.01 against A3's 0.02, the `wdStr` knob. Its reference is a1.py, which
    # bakes WD = 0.010000 — so this row is what turns "the string reaches the constant block" from
    # a code-reading claim into a measurement. ⚠ Nothing else about the row differs from
    # `lambacc8wxclip`, so a failure here and not there localises to the decay alone.
    ("lambacc8wxclipwd001", "generated_resnet50_imagenet_a1.py",  8, True),
    # ⭐⭐ THE `.adamw` GAP, CLOSED 2026-08-14 — and `wdStr` is what closed it. The obstacle was that
    # no generated reference baked R50's AdamW constants; `adam-probe` bakes eps 1e-8 (matching the
    # render) but wd 0.02 (against the render's 1e-4 default), so before the decay was a render
    # PARAMETER this row could not have existed without hand-writing a reference. Now the fixture
    # renders `.adamw` at wdStr = "0.02" and the two agree by construction rather than by luck.
    ("adamwxclipwd002", "generated_resnet50_imagenet_adamprobe.py", 1, False),
]

# Tolerance. The two sides run the SAME arithmetic in the same f32 order for the most part, but not
# op-for-op: the render broadcasts a scalar and multiplies where the reference relies on numpy
# broadcasting, and the clip's `sqrt` is recomputed per parameter in the render and once in the
# reference. Both are f32-exact rearrangements in principle and reassociations in practice, so this
# is a tight relative tolerance rather than a bit-exactness claim.
RTOL = 2e-6


def extract_ref_optimizer(path):
    """Pull the optimizer span out of a generated `train_step`, verbatim.

    From the first line that produces `grads` WITHOUT calling the model — i.e. `grads = ... _gsum`
    under accumulation, or `m, v, t = opt_state` when there is none — through the `params =
    jax.tree.map(_lamb, ...)` that closes the update. Everything before it is the forward/backward,
    which this gate deliberately does not run.

    ⚠ Asserts the span is found exactly once and that it does NOT contain `value_and_grad`. A span
    that silently swallowed the model call would turn this into a very slow forward test that
    passes for the wrong reason.
    """
    src = open(os.path.join(REF_DIR, path)).read().split("\n")
    # The span STARTS at the earliest line that transforms `grads` without calling the model, so the
    # clip is included when there is one. ⚠ Getting this wrong is silent in one direction: starting
    # below the clip would tie a clipped render against an unclipped reference and report a real-
    # looking mismatch, which is §3.1's shape one file over.
    starts = ["grads = jax.tree.map(lambda _a: _a / _K",   # accumulation: the mean, then the clip
              "gn = jnp.sqrt(",                            # clip, no accumulation
              "m, v, t = opt_state"]                       # neither
    i0 = None
    for pat in starts:
        hit = [i for i, l in enumerate(src) if l.strip().startswith(pat)]
        if hit:
            i0 = hit[0]
            break
    assert i0 is not None, f"{path}: no optimizer-span start anchor matched"
    # …and ENDS at the parameter update. ⚠ `jax.tree.map(_lamb, …)` for LAMB, but AdamW inlines its
    # lambda (`jax.tree.map(lambda p, mi, vi, msk: …)`), so the anchor is the assignment, not the
    # callee — matched at or after i0 so an earlier `params = …` cannot be picked up.
    i1 = next(i for i, l in enumerate(src)
              if i >= i0 and l.strip().startswith("params = jax.tree.map("))
    span = src[i0:i1 + 1]
    assert not any("value_and_grad" in l for l in span), \
        f"{path}: the extracted span reached the model call — the anchors moved"
    assert sum(1 for l in span if l.strip().startswith("params = jax.tree.map(")) == 1
    # dedent the function body by its own indent so it can exec at module level
    pad = len(span[0]) - len(span[0].lstrip())
    body = "\n".join(l[pad:] if l.strip() else "" for l in span)
    # constants live at module level in the generated file; take them from the same source
    consts = {}
    for name in ("BETA1", "BETA2", "EPS", "WD", "_WD_POS_SHAPE"):
        for l in src:
            if l.startswith(name + " "):
                consts[name] = eval(l.split("=", 1)[1].strip(), {"None": None})
                break
    wd_mask_src = None
    if any("WD_MASK" in l for l in span):
        j0 = next(i for i, l in enumerate(src) if l.startswith("def _wd_mask"))
        j1 = next(i for i in range(j0 + 1, len(src)) if src[i].startswith("return ") or
                  (src[i] and not src[i].startswith((" ", "\t"))and i > j0 + 1))
        wd_mask_src = "\n".join(src[j0:j1])
    return body, consts, wd_mask_src


def run_ref(path, K, theta, m, v, gsum, lr, t):
    """Execute the extracted reference optimizer. Returns (θ', m', v')."""
    import jax, jax.numpy as jnp
    body, consts, wd_mask_src = extract_ref_optimizer(path)
    ns = dict(jax=jax, jnp=jnp, np=np, **consts)
    ns["params"] = [jnp.asarray(a) for a in theta]
    ns["opt_state"] = ([jnp.asarray(a) for a in m], [jnp.asarray(a) for a in v],
                       jnp.float32(t - 1))          # the span does `t = t + 1` itself
    ns["_gsum"] = [jnp.asarray(a) for a in gsum]
    ns["grads"] = [jnp.asarray(a) for a in gsum]    # the no-accumulation span reads `grads`
    ns["_K"] = K
    ns["_ls"] = jnp.zeros((1,), jnp.float32)        # only feeds the uncompared `loss`
    ns["lr"] = jnp.float32(lr)
    if wd_mask_src is not None:
        exec(wd_mask_src, ns)
        ns["WD_MASK"] = ns["_wd_mask"](ns["params"])
    exec(body, ns)
    return ([np.asarray(a) for a in ns["params"]],
            [np.asarray(a) for a in ns["m"]], [np.asarray(a) for a in ns["v"]])


def run_render(slug, arrays, n_out):
    """Execute the rendered optimizer through XLA — the lowerer the PJRT trainers use.

    ⚠ XLA requires the entry point to be called `main`, so the symbol is renamed on the way in.
    A pure rename; no other text changes. Same approach as `scripts/grad_tie.py:run_xla`.
    """
    import jax
    from jax._src import xla_bridge
    from jax._src.lib import xla_client as xc
    from jax._src.interpreters import mlir as jmlir
    import jaxlib.mlir.ir as ir
    path = os.path.join(FIX_DIR, f"opt_step_{slug}.mlir")
    if not os.path.exists(path):
        sys.exit(f"missing {path} — run: lake build opt-step-fixtures && "
                 f".lake/build/bin/opt-step-fixtures")
    txt = open(path).read().replace(f"func.func @opt_step_{slug}(", "func.func @main(", 1)
    backend = xla_bridge.get_backend()
    devices = xc.DeviceList(tuple(backend.local_devices()[:1]))
    ctx = jmlir.make_ir_context()
    with ctx, ir.Location.unknown(ctx):
        module = ir.Module.parse(txt)
        exe = backend.compile_and_load(module, executable_devices=devices,
                                       compile_options=xc.CompileOptions())
    outs = exe.execute([jax.device_put(a) for a in arrays])
    assert len(outs) == n_out, f"{slug}: {len(outs)} outputs, expected {n_out}"
    return [np.asarray(o) for o in outs]


def cmp(label, got, want, worst):
    """Relative max-abs difference, scaled by the reference's own magnitude."""
    d = float(np.max(np.abs(got - want)))
    s = float(np.max(np.abs(want))) or 1.0
    rel = d / s
    worst.append((rel, label))
    return rel


def main():
    rng = np.random.default_rng(20260814)
    f32 = lambda *s: rng.standard_normal(s).astype(np.float32)
    LR, T = 0.005, 3
    BETA1, BETA2 = 0.9, 0.999
    bc1 = np.float32(1.0 - BETA1 ** T)
    bc2 = np.float32(1.0 - BETA2 ** T)

    theta = [f32(*s) for s in SHAPES]
    m0 = [f32(*s) * 0.1 for s in SHAPES]
    v0 = [np.abs(f32(*s)) * 0.01 for s in SHAPES]
    # ⚠ The gradients are scaled UP so the global norm exceeds the threshold and the clip is ACTIVE.
    # Below the threshold the factor is the literal 1 (`Proofs.clipFactor_eq_one_below`) and every
    # clip variant would agree with an unclipped reference bit for bit — the identity-below gate is
    # structurally blind to placement, which is exactly the trap `GradClip.lean` warns about.
    G0 = [f32(*s) * 5.0 for s in SHAPES]
    dg = [f32(*s) * 5.0 for s in SHAPES]

    print("── one-step optimizer tie (planning/verified_optimizer_parity.md §5) ──")
    gnorm = np.sqrt(sum(float(np.sum((a + b) ** 2)) for a, b in zip(G0, dg)))
    print(f"  ‖Gt‖ = {gnorm:.3f}   (clip threshold k·C; the clip is ACTIVE in every clip row)")
    worst, failures = [], 0

    for slug, ref, K, accum in VARIANTS:
        if accum:
            gsum = [a + b for a, b in zip(G0, dg)]          # Gt = akeep·G + g at akeep = 1
            arrays = ([*theta, *m0, *v0, *G0,
                       np.float32(LR), bc1, bc2,
                       np.float32(1.0), np.float32(1.0),    # %aup, %akeep on the APPLY micro-batch
                       *dg])
            n_out = 12
        else:
            gsum = dg
            arrays = [*theta, *m0, *v0, np.float32(LR), bc1, bc2, *dg]
            n_out = 9

        outs = run_render(slug, arrays, n_out)
        rt, rm, rv = run_ref(ref, K, theta, m0, v0, gsum, LR, T)

        rels = []
        for i in range(3):
            rels.append(cmp(f"{slug}/θ'[{i}]", outs[i], rt[i], worst))
            rels.append(cmp(f"{slug}/m'[{i}]", outs[3 + i], rm[i], worst))
            rels.append(cmp(f"{slug}/v'[{i}]", outs[6 + i], rv[i], worst))
        if accum:
            # ⭐ THE ROW THAT CATCHES A COMPOUNDING CLIP. The fourth region must be the RAW `Gt` —
            # the reference's `_gsum`, before its clip line — not the clipped total the optimizer
            # tail consumed. See this file's header.
            for i in range(3):
                rels.append(cmp(f"{slug}/G'[{i}] (RAW)", outs[9 + i], gsum[i], worst))
        bad = max(rels)
        ok = bad <= RTOL
        failures += 0 if ok else 1
        tag = "✓" if ok else "✗"
        print(f"  {tag} {slug:<16} K={K}  ref={ref.replace('generated_resnet50_imagenet', 'r50')}"
              f"  worst rel {bad:.2e}")

    worst.sort(reverse=True)
    print(f"\n  worst 3: " + ", ".join(f"{l} {r:.2e}" for r, l in worst[:3]))
    if failures:
        print(f"✗ {failures}/{len(VARIANTS)} variants disagree beyond rtol {RTOL:g}")
        return 1
    print(f"✓ {len(VARIANTS)} variants agree with the shipped reference within rtol {RTOL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
