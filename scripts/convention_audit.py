#!/usr/bin/env python3
"""CONVENTION AUDIT — diff the verified render against its JAX reference, statically.

⭐ **Why this exists.** Every whole-net tie run in this repo so far has found the render and the
reference to be different nets, and each one cost a session to discover:

    mnv4  stem padding          render wrong   fixed (convStridedXla)
    mnv2  padding, 5 sites      render wrong   fixed (switched + 2 re-runs)
    mnv2  stem/head relu6/relu  REFERENCE      fixed (NetSpec.convBnAct)
    enet  stem/head swish/relu  REFERENCE      fixed (NetSpec.convBnAct)
    enet  stem padding          render wrong   fixed (convStridedXla, re-run aug08)
    r34   conv + pool padding   REFERENCE      fixed (pool + helper defaults, then
    r50   stem padding          REFERENCE             NetSpec.convPadStyle for the stem)

⚠ **This script has itself been wrong.** Its `relu6` detector shimmed `jax.nn.relu6`, which this
generator never calls — it emits `jnp.minimum(jax.nn.relu(x), 6.0)`. So the reference side could
not report relu6 under any circumstances, and `mnv2:activation` stayed red for a day after the
generator was fixed. A false RED is the dangerous direction for a ratchet: the baseline gets
frozen against it and the row looks like live debt forever. `--selftest` is what caught it, by
failing on enet — see `min_shim`.

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
    # ⭐ R50 is audited against the 1000-CLASS reference, deliberately. The Imagenette R50 demo is
    # real (`resnet50Verified`, 89.86% in runs/r50_imagenette_adam_80ep.log) but there is no
    # `generated_resnet50.py` — only the ImageNet ones. Conventions do not depend on class count:
    # padding, activation and BN axis are identical between the two variants and only the head
    # width differs, so the ImageNet reference is a sound CONVENTION proxy. Checked, not assumed:
    # resnet50_fwd and resnet50in_fwd have byte-identical strided-pad profiles
    # (3x[[0,0]] 1x1 projections, 3x[[1,1]], 1x[[3,3]]).
    #
    # ⚠ What this CANNOT substitute for is a numeric tie: R50-Imagenette has no baseline number to
    # compare 89.86% against, the same "no matched pair" gap §3f records for EfficientNet.
    "r50":  dict(pad_src="verified_mlir/resnet50_fwd.mlir",
                 train="verified_mlir/resnet50_adam_train_step.mlir",
                 split=("verified_mlir/resnet50_fwd.mlir",
                        "verified_mlir/resnet50_adam_train_step.mlir"),
                 ref="jax/.lake/build/generated_resnet50_imagenet.py", shape=(2, 3, 224, 224),
                 # R50's reference threads running-stats BN: forward(params, x, bn, training)
                 # returning (logits, new_bn). Every other net is forward(params, x).
                 call=lambda mod, ps, xx: mod["forward"](ps, xx, mod["init_bn_state"](), True)[0]),
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
# The live ledger: what each net is EXPECTED to still show for padding/activation. All five are
# now clean on both axes (r34/mnv2 still carry `bn-split`, which is a separate defect and not
# checked here). ⚠ Because this is all-clean it no longer proves the detectors work — that job
# moved to REGRESSIONS below.
KNOWN = {
    "r50":  set(),            # padding closed 2026-08-10 (NetSpec.convPadStyle)
    "r34":  set(),            # padding closed 2026-08-10 (NetSpec.convPadStyle)
    "mnv2": set(),            # activation closed 2026-08-09 (NetSpec.convBnAct)
    "enet": set(),            # activation closed 2026-08-09 (NetSpec.convBnAct)
    "mnv4": set(),
}

# ⭐ THE HISTORICAL DIVERGENCES, as synthetic profile pairs. Each is a real finding this repo paid
# a session for, reduced to the two numbers that separate the sides. `--selftest` replays them
# through `compare()` and requires each to be caught.
#
# ⚠ This is the load-bearing half of the selftest now that the ledger is clean. A ledger-only
# selftest degenerates the moment the debt is paid: it asserts "clean == clean" and passes even if
# every detector is dead. That is not hypothetical — the `relu6` detector WAS dead (it shimmed
# `jax.nn.relu6`, which the generator never emits) and the ledger-based selftest happily called
# mnv2 a confirmed finding for it.
def _prof(pads=("sym",), pools=("sym",), relu=10, relu6=0, bn="batch", split=("batch", "batch")):
    return (dict(strided_conv_pads=list(pads), strided_pool_pads=list(pools),
                 acts={"relu": relu, "clamp_hi(relu6)": relu6, "logistic(swish/sigmoid)": 0},
                 bn_train=bn, bn_split_a=split[0], bn_split_b=split[1], split_names=("a", "b")),
            dict(strided_conv_pads=list(pads), strided_pool_pads=list(pools),
                 acts={"relu": relu, "relu6": relu6}, bn=bn))

REGRESSIONS = []
_r, _j = _prof(pads=("sym",) * 7)                       # r34/r50: render symmetric stem…
_j["strided_conv_pads"] = ["same"] + ["sym"] * 6        # …reference XLA 'SAME' at exactly one site
REGRESSIONS.append(("r34/r50 stem padding", _r, _j, "padding"))
_r, _j = _prof(relu=35, relu6=35)                       # mnv2: render relu6 at every site…
_j["acts"] = {"relu": 35, "relu6": 0}                   # …reference plain relu (the dead shim)
REGRESSIONS.append(("mnv2 stem/head relu6-vs-relu", _r, _j, "activation"))
_r, _j = _prof(relu=0)                                  # enet: render swish throughout…
_j["acts"] = {"relu": 2, "swish": 47}                   # …reference relu at exactly 2 of 49 sites
REGRESSIONS.append(("enet stem/head swish-vs-relu", _r, _j, "activation"))
_r, _j = _prof(split=("per-example", "batch"))          # mnv2/r34: fwd and train_step disagree
REGRESSIONS.append(("BN two-worlds split", _r, _j, "bn-split"))
_r, _j = _prof(bn="batch")                              # render trains batch, reference per-example
_j["bn"] = "per-example"
REGRESSIONS.append(("BN world mismatch", _r, _j, "bn-world"))
_r, _j = _prof(pads=("sym",) * 7, relu=35, relu6=35)    # the NEGATIVE control — see selftest
REGRESSIONS.append(("matched profiles (control)", _r, _j, None))


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

def reference_profile(ref_py, shape, call=None):
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

    # ⚠⚠ relu6 is NOT spelled `jax.nn.relu6` by this generator — it emits the idiom
    # `jnp.minimum(jax.nn.relu(x), 6.0)`. Shimming `jax.nn.relu6` therefore never fires, and the
    # reference side reported "relu, no relu6" no matter what the generator did. That made
    # `mnv2:activation` a divergence the audit could not clear even once the generator was fixed
    # (`NetSpec.convBnAct`, 2026-08-09) — a FALSE RED, which is worse than a false green here
    # because it is what a ratchet baseline gets frozen against.
    #
    # So detect the clamp itself: a `jnp.minimum` against the scalar 6.0. This deliberately counts
    # hard-swish's clamp too — and that is correct, because the RENDER side counts
    # `stablehlo.minimum` with exactly the same ambiguity. Both sides count "clamp-at-6 sites", so
    # the comparison stays sound even if a hard-swish net is ever audited.
    real_min = jnp.minimum

    def is_six(v):
        try:
            return np.ndim(v) == 0 and float(v) == 6.0
        except Exception:
            return False

    def min_shim(a, b, *rest, **kw):
        if is_six(a) or is_six(b):
            rec["acts"]["relu6"] = rec["acts"].get("relu6", 0) + 1
        return real_min(a, b, *rest, **kw)

    saved = {}
    try:
        jax.lax.conv_general_dilated, jax.lax.reduce_window = conv_shim, rw_shim
        jnp.minimum = min_shim
        for nm in ("relu", "relu6", "sigmoid", "silu", "swish", "gelu"):
            if hasattr(jax.nn, nm):
                saved[nm] = getattr(jax.nn, nm)
                setattr(jax.nn, nm, count(nm, saved[nm]))
        for nm in ("swish", "hard_swish"):          # the generator's OWN helpers
            if nm in mod and callable(mod[nm]):
                mod[nm] = count(nm, mod[nm])
        key = jax.random.PRNGKey(0)
        params = mod["init_params"](key)
        xin = jnp.zeros(shape, jnp.float32)
        (call or (lambda m, p, x: m["forward"](p, x)))(mod, params, xin)
    finally:
        jax.lax.conv_general_dilated, jax.lax.reduce_window = real_conv, real_rw
        jnp.minimum = real_min
        for nm, fn in saved.items():
            setattr(jax.nn, nm, fn)

    return {
        "strided_conv_pads": sorted(c["pad"] for c in rec["conv"]),
        "strided_pool_pads": sorted(rec["pool"]),
        "acts": rec["acts"],
        "bn": "batch" if "axis=(0, 2, 3)" in "\n".join(src[:cut]) else "per-example",
    }


# ═══════════════════════════════════════════════════════════════════════════

def hist(xs):
    return {k: xs.count(k) for k in sorted(set(xs))} or {}


def compare(r, j):
    """The whole verdict, as a PURE function of two profiles.

    ⭐ Factored out of `audit()` so `--selftest` can drive it with synthetic profiles. Once every
    ledger row is fixed, a selftest that only checks "the live corpus matches KNOWN" asserts
    nothing — every detector could be dead and it would still print green, which is the exact
    failure mode that let the broken `relu6` shim survive. Replaying the historical divergences
    through this function keeps the guarantee after the debt is paid."""
    issues = set()
    rp, jp = hist(r["strided_conv_pads"]), hist(j["strided_conv_pads"])
    rq, jq = hist(r["strided_pool_pads"]), hist(j["strided_pool_pads"])
    if rp != jp or rq != jq:
        issues.add("padding")
    # ⭐ COUNT, not presence, on both activation checks. EfficientNet's reference called relu
    # exactly TWICE — stem and head — and swish everywhere else, while the render is swish
    # throughout; both sides therefore "have swish" and a presence test calls that clean. Only the
    # relu SITE COUNT sees it. Same for relu6: mnv2's render clamps at 6 in all 35 sites, so "the
    # reference has SOME relu6" would pass with the stem and head still plain relu, which is
    # precisely the divergence the row exists to catch. The two sides are commensurable because
    # both count clamp-at-6 SITES (see `min_shim`).
    if r["acts"]["clamp_hi(relu6)"] != j["acts"].get("relu6", 0):
        issues.add("activation")        # mnv2 pre-fix: render relu6 at 35 sites, reference 0
    if r["acts"]["relu"] != j["acts"].get("relu", 0):
        issues.add("activation")        # enet pre-fix: render 0 relu, reference 2
    if r["bn_train"] != j["bn"]:
        issues.add("bn-world")
    if r["bn_split_a"] != r["bn_split_b"]:
        issues.add("bn-split")          # the §3d(b) hole: two artifacts of one net disagree
    return issues


def audit(net, verbose=True):
    cfg = NETS[net]
    r = render_profile(cfg["pad_src"], cfg["train"], cfg["split"])
    j = reference_profile(cfg["ref"], cfg["shape"], cfg.get("call"))
    issues = compare(r, j)
    rp, jp = hist(r["strided_conv_pads"]), hist(j["strided_conv_pads"])
    rq, jq = hist(r["strided_pool_pads"]), hist(j["strided_pool_pads"])
    ren_relu6 = r["acts"]["clamp_hi(relu6)"]

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
    ap.add_argument("--baseline", default=None, metavar="FILE",
                    help="RATCHET mode for CI: fail only on divergences not already recorded in "
                         "FILE. The list may SHRINK, never grow — a new entry means two sides of "
                         "one net drifted apart with CI green, which is the whole failure this "
                         "script exists to prevent. Same discipline as render_guard_baseline.txt.")
    ap.add_argument("--update-baseline", action="store_true",
                    help="rewrite the baseline to the current state (needs --baseline)")
    args = ap.parse_args()

    nets = [args.net] if args.net else sorted(NETS)
    found = {n: audit(n) for n in nets}

    print("\n" + "═" * 72)
    bad = [n for n, v in found.items() if v]
    for n in nets:
        print(f"  {n:<6} {'clean' if not found[n] else ', '.join(sorted(found[n]))}")

    ratchet_fail = False
    if args.baseline:
        import os
        current = sorted(f"{n}:{i}" for n in nets for i in found[n])
        if args.update_baseline:
            with open(args.baseline, "w") as fh:
                fh.write("# Render/reference convention divergences this repo currently carries.\n"
                         "# Regenerate with:\n"
                         "#   python3 scripts/convention_audit.py --baseline FILE --update-baseline\n"
                         "# This list may SHRINK, never grow. A new entry means the render and the\n"
                         "# JAX reference for one net drifted apart and CI stayed green.\n")
                fh.writelines(f"{e}\n" for e in current)
            print(f"\n  baseline updated: {len(current)} entr(ies) -> {args.baseline}")
            return 0
        known = set()
        if os.path.exists(args.baseline):
            known = {l.strip() for l in open(args.baseline)
                     if l.strip() and not l.startswith("#")}
        new = [e for e in current if e not in known]
        # ⚠ scope "resolved" to the nets actually AUDITED. With `--net r34` the baseline still
        # holds enet/mnv2 rows that this run never looked at, and reporting them as resolved would
        # invite deleting live debt on the strength of a run that never tested it.
        audited = {e for e in known if e.split(":")[0] in set(nets)}
        gone = sorted(audited - set(current))
        print(f"\n  RATCHET vs {args.baseline}: {len(current)} current, {len(known)} baselined")
        for e in gone:
            print(f"    ✓ RESOLVED (drop from the baseline): {e}")
        if new:
            ratchet_fail = True
            print("\n  ✗ NEW DIVERGENCE — the render and its reference drifted apart:")
            for e in new:
                print(f"      {e}")
            print("    Fix it, or — only if the gap is deliberate and budgeted — re-record with")
            print(f"    python3 scripts/convention_audit.py --baseline {args.baseline} --update-baseline")
        elif not gone:
            print("    ✓ no new divergences")

    if args.selftest:
        ok = True
        # (1) DETECTORS — replay every historical divergence through compare(). This is what keeps
        #     the selftest meaningful now that the live ledger is clean.
        print("\n  SELFTEST (1/2) — each historical finding must still be CAUGHT:")
        for label, rp_, jp_, want in REGRESSIONS:
            got = compare(rp_, jp_)
            good = (want is None and not got) or (want is not None and want in got)
            ok &= good
            expect = "clean (as expected)" if want is None else f"detected {want}"
            actual = "" if good else f"  ← GOT {sorted(got) or ['clean']}"
            print(f"    {'✓' if good else '✗'} {label:<32} {expect}{actual}")
        # (2) LEDGER — the live corpus must match what the ledger says it is.
        print("\n  SELFTEST (2/2) — the live corpus must match the ledger:")
        for n in nets:
            want, got = KNOWN[n], {i for i in found[n] if i in ("padding", "activation")}
            ok &= want == got
            print(f"    {'✓' if want == got else '✗'} {n:<6} expected "
                  f"{sorted(want) or ['clean']}, got {sorted(got) or ['clean']}")
        if not ok:
            print("\n  ✗ SELFTEST FAILED — the audit does not reproduce the ledger, so a green run")
            print("    from it means nothing. Fix the audit before trusting any row above.")
            return 2
        print("\n  ✓ selftest passes — every detector fires, and the corpus matches the ledger")

    # ⚠ In ratchet mode the EXIT CODE tracks new debt, not total debt: the three open rows are
    # known and tracked, and failing on them every run would make the signal worthless. Outside
    # ratchet mode any divergence is a failure, which is what a human wants interactively.
    if args.baseline:
        return 2 if ratchet_fail else 0
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
