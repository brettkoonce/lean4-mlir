#!/usr/bin/env python3
"""CONVENTION AUDIT — diff the verified render against its JAX reference, statically.

⭐ **Why this exists.** Every whole-net tie run in this repo so far has found the render and the
reference to be different nets, and each one cost a session to discover:

    mnv4  stem padding          render wrong   fixed (convStridedXla)
    mnv2  padding, 5 sites      render wrong   fixed (switched + 2 re-runs)
    mnv2  stem/head relu6/relu  REFERENCE      open
    enet  stem/head swish/relu  REFERENCE      open
    enet  stem padding          render wrong   open
    r34   conv + pool padding   REFERENCE      open  (found 2026-08-10)

Those are not six bugs. They are **two gaps spelled twice**: the JAX `.convBn` layer emitter
hardcodes `jax.nn.relu` and has no activation parameter, and padding is stated independently on
each side — the generator hands XLA the *string* `'SAME'` while the renderer picks an op *token*
(`convStrided` vs `convStridedXla`). `VLayer` cannot express `Padding` at all, so on the render
side the convention is implicit in which token someone chose, which is exactly why it stays
invisible.

This script makes the comparison mechanical and cheap: **no GPU, no training, no gradients** — it
runs the reference's `forward()` once on a tiny input with `jax.lax` instrumented, reads the same
facts out of the committed `.mlir`, and diffs the two convention profiles. It would have caught all
six of the rows above in seconds.

⚠ **It compares CONVENTIONS, not values.** A green audit does not mean the nets agree numerically —
that is what `scripts/*_forward_tie.py` and `scripts/grad_tie.py` are for. It means the two sides
spell padding, activation and BN the same way, which is the precondition those ties kept failing.

Usage:
    JAX_PLATFORMS=cpu python3 scripts/convention_audit.py            # every matched pair
    JAX_PLATFORMS=cpu python3 scripts/convention_audit.py --net r34
    JAX_PLATFORMS=cpu python3 scripts/convention_audit.py --selftest # must reproduce the ledger
"""
import argparse, re, sys
import numpy as np

# ── the matched pairs. A net with no 10-class generated reference cannot be audited at all, which
#    is itself a finding and is reported rather than skipped silently. ──────────────────────────
#    `pad_src` is the forward that PAIRS WITH TRAINING, which is not always `<slug>_fwd`. On mnv2
#    it is `fwd_eval`: §3h switched the Adam path and the two eval forwards to XLA `SAME` and left
#    the SGD pair (`mobilenetv2_fwd`, `mobilenetv2_train_step`) symmetric on purpose, so those two
#    are a self-consistent DIFFERENT net. Auditing `mobilenetv2_fwd` reports a padding divergence
#    that is real for the SGD net and irrelevant to the number anyone quotes. Getting this wrong is
#    how an audit produces true-but-useless failures, so the pairing is explicit per net.
#    `split` names the two artifacts whose BN worlds must agree with EACH OTHER (§3d(b)).
NETS = {
    "r34":  dict(pad_src="verified_mlir/resnet34_fwd.mlir",
                 train="verified_mlir/resnet34_adam_train_step.mlir",
                 split=("verified_mlir/resnet34_fwd.mlir",
                        "verified_mlir/resnet34_adam_train_step.mlir"),
                 ref="jax/.lake/build/generated_resnet34.py", shape=(2, 3, 224, 224)),
    "mnv2": dict(pad_src="verified_mlir/mobilenetv2_fwd_eval.mlir",
                 train="verified_mlir/mobilenetv2_adam_train_step.mlir",
                 split=("verified_mlir/mobilenetv2_fwd.mlir",
                        "verified_mlir/mobilenetv2_adam_train_step.mlir"),
                 ref="jax/.lake/build/generated_mobilenet_v2.py", shape=(2, 3, 224, 224)),
    "mnv4": dict(pad_src="verified_mlir/mnv4_fwd.mlir",
                 train="verified_mlir/mnv4_adam_train_step.mlir",
                 split=("verified_mlir/mnv4_fwd.mlir",
                        "verified_mlir/mnv4_adam_train_step.mlir"),
                 ref="jax/.lake/build/generated_mobilenet_v4.py", shape=(2, 3, 224, 224)),
    "enet": dict(pad_src="verified_mlir/efficientnet_fwd.mlir",
                 train="verified_mlir/efficientnet_adam_train_step.mlir",
                 split=("verified_mlir/efficientnet_fwd.mlir",
                        "verified_mlir/efficientnet_adam_train_step.mlir"),
                 ref="jax/.lake/build/generated_efficientnet_b0.py", shape=(2, 3, 224, 224)),
}

# The ledger above, as an assertion. `--selftest` requires the audit to reproduce it: an audit that
# cannot rediscover the findings that motivated it is not working, and a green run would be the
# most expensive possible way to be wrong.
# ⚠ `enet` carries ONLY `activation`. Its stem padding was switched to XLA `SAME` and re-run on
# 2026-08-08 (`runs/enet_adam_80ep_xlapad_aug08.log`, 89.76%) — the audit found that row already
# fixed while a hand-kept ledger still called it open, which is the whole argument for having this
# script rather than a list in a doc.
KNOWN = {
    "r34":  {"padding"},
    "mnv2": {"activation"},
    "enet": {"activation"},
    "mnv4": set(),
}


def pad_kind(lo, hi):
    """Classify one spatial padding pair.

    At an EVEN input, XLA `'SAME'` puts the extra row on the high side (`hi == lo + 1`) while the
    symmetric convention splits evenly. That one bit is the entire difference between the two
    conventions, it is invisible to every shape and count, and it is what four of the six ledger
    rows are about."""
    if lo == hi:
        return "sym"
    if hi == lo + 1:
        return "same"
    return f"odd({lo},{hi})"


# ═══════════════════════════════════════════════════════════════════════════
# § The RENDER side — read the conventions out of the committed MLIR
# ═══════════════════════════════════════════════════════════════════════════

def render_profile(pad_src, train_path, split):
    txt = open(pad_src).read()
    prof = {}

    # convolutions, in file order: (stride, pad-kind, grouped?)
    convs = []
    for m in re.finditer(
            r"stablehlo\.convolution.*?window\s*=\s*\{stride\s*=\s*\[(\d+), (\d+)\],\s*"
            r"pad\s*=\s*\[\[(\d+), (\d+)\], \[(\d+), (\d+)\]\].*?feature_group_count\s*=\s*(\d+)",
            txt, re.S):
        sh, sw, plo, phi, qlo, qhi, fgc = (int(g) for g in m.groups())
        convs.append(dict(stride=sh, pad=pad_kind(plo, phi), grouped=fgc > 1))
    # ⚠ only STRIDED sites carry signal: at stride 1 with an odd kernel, XLA `SAME` and symmetric
    # are the same padding, so a stride-1 site can never distinguish the conventions. Counting them
    # would dilute the profile into always-agreeing mush — a vacuously green audit.
    prof["strided_conv_pads"] = sorted(c["pad"] for c in convs if c["stride"] > 1)
    prof["n_conv"] = len(convs)

    pools = []
    for m in re.finditer(
            r"reduce_window.*?window_strides\s*=\s*array<i64: \d+, \d+, (\d+), \d+>,\s*"
            r"padding\s*=\s*dense<\[\[\d+, \d+\], \[\d+, \d+\], \[(\d+), (\d+)\]",
            txt, re.S):
        st, lo, hi = (int(g) for g in m.groups())
        if st > 1:
            pools.append(pad_kind(lo, hi))
    prof["strided_pool_pads"] = sorted(pools)

    # activations. relu6 = a clamp on BOTH sides, so `minimum` is its fingerprint; `logistic` is
    # sigmoid, which is swish's (and SE's) fingerprint.
    # ⚠ a max-POOL's reduction body also emits `stablehlo.maximum`, but on a SCALAR. Counting it as
    # a relu overstates R34's relu sites by exactly one and would make the count check wrong on the
    # one net that has a maxpool. Filter on the operand type.
    relu_sites = [l for l in txt.splitlines()
                  if "stablehlo.maximum" in l and not l.rstrip().endswith(": tensor<f32>")]
    prof["acts"] = {
        "relu": len(relu_sites),
        "clamp_hi(relu6)": len(re.findall(r"stablehlo\.minimum", txt)),
        "logistic(swish/sigmoid)": len(re.findall(r"stablehlo\.logistic", txt)),
    }

    prof["bn_train"] = ("batch" if "dimensions = [0, 2, 3]" in open(train_path).read()
                        else "per-example")
    a, b = split
    prof["bn_split_a"] = ("batch" if "dimensions = [0, 2, 3]" in open(a).read() else "per-example")
    prof["bn_split_b"] = ("batch" if "dimensions = [0, 2, 3]" in open(b).read() else "per-example")
    prof["split_names"] = (a.split("/")[-1], b.split("/")[-1])
    return prof


# ═══════════════════════════════════════════════════════════════════════════
# § The REFERENCE side — instrument `jax.lax` and run forward() once
# ═══════════════════════════════════════════════════════════════════════════

def reference_profile(ref_py, shape):
    """⭐ RUNTIME instrumentation, not source parsing.

    The generated references reach `conv_general_dilated` through several helpers with different
    defaults, and callers override them per site — `mnv4`'s `uib_block` passes an explicit tuple
    while `mnv2`'s `sep_conv` does not. Any grep-level rule about "the default is 'SAME'" is
    therefore wrong for some net, and getting that exact reading wrong is what made the first R34
    patch inert (it targeted `conv2d`, but `conv_bn` calls `conv_general_dilated` DIRECTLY).

    So: wrap the lax primitives, run the real `forward()` once, and record what each site actually
    asked for. No heuristics, and it tracks the generator automatically."""
    import jax, jax.numpy as jnp

    src = open(ref_py).read().split("\n")
    cut = next((i for i, l in enumerate(src) if l.startswith("def loss_fn")), None)
    if cut is None:
        sys.exit(f"{ref_py}: no `def loss_fn` to cut at; the generator's shape changed")
    mod = {}
    exec("\n".join(src[:cut]), mod)
    for need in ("forward", "init_params"):
        if need not in mod:
            sys.exit(f"{ref_py}: prefix does not define {need}()")

    rec = dict(conv=[], pool=[], acts={})
    real_conv, real_rw = jax.lax.conv_general_dilated, jax.lax.reduce_window

    def resolve(padding, k, stride):
        """'SAME'/'VALID' are strings; everything else is an explicit pair sequence."""
        if isinstance(padding, str):
            return "same" if padding.upper() == "SAME" else "valid"
        (lo, hi) = tuple(padding)[0]
        return pad_kind(int(lo), int(hi))

    def conv_shim(lhs, rhs, window_strides, padding, **kw):
        st = int(tuple(window_strides)[0])
        if st > 1:
            rec["conv"].append(dict(stride=st, pad=resolve(padding, rhs.shape[2], st),
                                    grouped=kw.get("feature_group_count", 1) > 1))
        return real_conv(lhs, rhs, window_strides, padding, **kw)

    def rw_shim(operand, init, computation, window_dimensions, window_strides, padding, **kw):
        st = int(tuple(window_strides)[2])
        if st > 1:
            rec["pool"].append(resolve(padding, None, st))
        return real_rw(operand, init, computation, window_dimensions, window_strides, padding, **kw)

    def count(name, fn):
        def shim(*a, **k):
            rec["acts"][name] = rec["acts"].get(name, 0) + 1
            return fn(*a, **k)
        return shim

    saved = {}
    try:
        jax.lax.conv_general_dilated, jax.lax.reduce_window = conv_shim, rw_shim
        for nm in ("relu", "relu6", "sigmoid", "silu", "swish", "gelu"):
            if hasattr(jax.nn, nm):
                saved[nm] = getattr(jax.nn, nm)
                setattr(jax.nn, nm, count(nm, saved[nm]))
        for nm in ("swish", "hard_swish"):          # the generator's OWN helpers
            if nm in mod and callable(mod[nm]):
                mod[nm] = count(nm, mod[nm])
        key = jax.random.PRNGKey(0)
        params = mod["init_params"](key)
        mod["forward"](params, jnp.zeros(shape, jnp.float32))
    finally:
        jax.lax.conv_general_dilated, jax.lax.reduce_window = real_conv, real_rw
        for nm, fn in saved.items():
            setattr(jax.nn, nm, fn)

    return {
        "strided_conv_pads": sorted(c["pad"] for c in rec["conv"]),
        "strided_pool_pads": sorted(rec["pool"]),
        "acts": rec["acts"],
        "bn": "batch" if "axis=(0, 2, 3)" in "\n".join(src[:cut]) else "per-example",
    }


# ═══════════════════════════════════════════════════════════════════════════

def audit(net, verbose=True):
    cfg = NETS[net]
    r = render_profile(cfg["pad_src"], cfg["train"], cfg["split"])
    j = reference_profile(cfg["ref"], cfg["shape"])
    issues = set()

    def hist(xs):
        return {k: xs.count(k) for k in sorted(set(xs))} or {}

    rp, jp = hist(r["strided_conv_pads"]), hist(j["strided_conv_pads"])
    rq, jq = hist(r["strided_pool_pads"]), hist(j["strided_pool_pads"])
    if rp != jp or rq != jq:
        issues.add("padding")
    # activation: compare the SETS of activation kinds each side uses. Counts differ legitimately
    # (a swish is one call on the reference and logistic+multiply on the render), but a kind that
    # appears on one side and not the other is a real divergence — that is exactly mnv2's relu6
    # and enet's swish.
    ref_kinds = {k for k, v in j["acts"].items() if v}
    ren_relu6 = r["acts"]["clamp_hi(relu6)"] > 0
    if ren_relu6 != ("relu6" in ref_kinds):
        issues.add("activation")        # mnv2: render relu6, reference plain relu
    # ⭐ COUNT, not presence. EfficientNet's reference calls relu exactly TWICE — stem and head —
    # and swish everywhere else, while the render is swish throughout. Both sides therefore "have
    # swish", and a presence test calls that clean. Only the relu SITE COUNT sees it.
    if r["acts"]["relu"] != j["acts"].get("relu", 0):
        issues.add("activation")
    if r["bn_train"] != j["bn"]:
        issues.add("bn-world")
    if r["bn_split_a"] != r["bn_split_b"]:
        issues.add("bn-split")          # the §3d(b) hole: two artifacts of one net disagree

    if verbose:
        print(f"\n══ {net} ══  render {cfg['pad_src'].split('/')[-1]}  vs  {cfg['ref'].split('/')[-1]}")
        print(f"  strided conv padding   render {rp}   reference {jp}"
              f"   {'✓' if rp == jp else '✗'}")
        print(f"  strided pool padding   render {rq}   reference {jq}"
              f"   {'✓' if rq == jq else '✗'}")
        print(f"  activations            render relu={r['acts']['relu']} relu6={ren_relu6}"
              f" logistic={r['acts']['logistic(swish/sigmoid)']}"
              f"   reference {dict(sorted(j['acts'].items()))}"
              f"   {'✓' if 'activation' not in issues else '✗'}")
        print(f"  BN world               render train={r['bn_train']}   reference {j['bn']}"
              f"   {'✓' if 'bn-world' not in issues else '✗'}")
        na, nb = r["split_names"]
        print(f"  BN self-consistency    {na}={r['bn_split_a']}  {nb}={r['bn_split_b']}"
              f"   {'✓' if 'bn-split' not in issues else '✗'}")
        if "bn-split" in issues:
            print("     ⛔ two artifacts of ONE net are in different BN worlds — the net that scores")
            print("        is not the net that trains (§3d(b)); regen_verified_mlir.sh cannot see it")
        print(f"  ⇒ {'CLEAN' if not issues else 'DIVERGES: ' + ', '.join(sorted(issues))}")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", choices=sorted(NETS), default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="require the audit to reproduce the known ledger (see KNOWN)")
    args = ap.parse_args()

    nets = [args.net] if args.net else sorted(NETS)
    found = {n: audit(n) for n in nets}

    print("\n" + "═" * 72)
    bad = [n for n, v in found.items() if v]
    for n in nets:
        print(f"  {n:<6} {'clean' if not found[n] else ', '.join(sorted(found[n]))}")

    if args.selftest:
        print("\n  SELFTEST — the audit must rediscover the findings that motivated it:")
        ok = True
        for n in nets:
            want, got = KNOWN[n], {i for i in found[n] if i in ("padding", "activation")}
            mark = "✓" if want == got else "✗"
            if want != got:
                ok = False
            print(f"    {mark} {n:<6} expected {sorted(want) or ['clean']}, got {sorted(got) or ['clean']}")
        if not ok:
            print("\n  ✗ SELFTEST FAILED — the audit does not reproduce the ledger, so a green run")
            print("    from it means nothing. Fix the audit before trusting any row above.")
            return 2
        print("  ✓ selftest passes — the audit reproduces every known finding")

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
