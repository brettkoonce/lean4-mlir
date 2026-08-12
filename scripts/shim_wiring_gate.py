#!/usr/bin/env python3
"""shim_wiring_gate.py — gate the PER-NET ImageNet data shims (handoff §0.2 ▶1).

The defect this exists to hold closed: `spawnShim` hardcoded
`generated_resnet34_imagenet_shim.py` for **every** net, `$SHIM_SCRIPT` was set nowhere, and R34's
was the only generated shim on disk — so a "verified EfficientNet / ViT / ConvNeXt ImageNet run"
streamed ResNet-34's RandomResizedCrop+hflip and nothing else. Nothing failed. The recipe matrix
read ✅ on a CAPABILITY (`generateShim` honours every augmentation flag) rather than on the STATE.

Structure — four gates and two controls, run in this order:

  0  the wiring is a BIJECTION      every .imagenet net names its own shim, all distinct, all present
  1  each shim is the RIGHT net's   its generated banner names the reference that net is a port of
  2  the augmentation PARTITION     config flags vs generated CALL SITES, feature by feature
  3  the producer default           SHIM_MIX's baked default == the config's useMixup/useCutmix
  C1 the count-based control        a census that counts DEFINITIONS mis-classifies 4 of 7 nets
  C2 the stream, MEASURED (--stream) SHIM_HASH: determinism, mnv2≡r34 known answer, r34≠the rest

⚠ GATE 2 CHECKS CALL SITES, NOT DEFINITIONS, AND THAT IS THE WHOLE POINT. `generateShim` emits the
shared `_aa_*` op block whenever AutoAugment **or** geometric RandAugment is on, so ConvNeXt and ViT
both contain `def _autoaugment` while calling neither. A gate that greps for `_autoaugment` reports
AutoAugment on three nets when it is on one. C1 measures exactly that and is what licenses gate 2's
shape — the same reason `wdx-tie` gates the partition instead of the count.

  scripts/shim_wiring_gate.py              # gates 0-3 + C1: static, instant, no data needed
  scripts/shim_wiring_gate.py --stream     # + C2: runs each shim over tfds (minutes, no GPU)
  scripts/shim_wiring_gate.py --break      # + the negative control: a deliberately mis-wired net

Exit 0 = every gate green. Exit 1 = a gate fired (the message names which and why).
"""
import os, re, subprocess, sys, hashlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "jax", ".lake", "build")

# ── The ONE hand-written pairing in this file, and it is irreducible: which JAX reference each
#    verified net is a port of. Everything else — the config name, the shim filename, the flags —
#    is READ from the two sources rather than restated here (the mixup-λ lesson: recover a constant
#    by reading it, not by fitting it).
#    ⚠ The 4th column is the RECIPE the verified net ports, added 2026-08-06. It is "default" for
#    every net that existed before, which is what this file hardcoded — see `read_recipe`.
NETS = [
    # verified slug, the jax reference's Main file,          the reference name in its banner, recipe
    ("resnet34in", "MainResnetImagenet.lean",        "ResNet-34 (ImageNet)",         "default"),
    ("vitin",      "MainVitImagenet.lean",           "ViT-Tiny (ImageNet, bf16)",    "default"),
    ("mobilenetv2in",     "MainMobilenetV2Imagenet.lean",   "MobileNetV2 (ImageNet, bf16)", "default"),
    ("efficientnetin",     "MainEfficientNetImagenet.lean",  "EfficientNet-B0 (ImageNet, bf16)", "default"),
    ("convnextin",      "MainConvNeXtImagenet.lean",      "ConvNeXt-T (ImageNet, bf16)",  "default"),
    ("resnet50in", "MainResnet50Imagenet.lean",      "ResNet-50 (ImageNet)",         "default"),
    # ⭐ Same reference and same Main file as `resnet50in`; it is the RECIPE that differs. `short`
    # is timm's RSB-A3 — trainRes 160, testCropRatio 0.95, RandAugment m6, mixup/cutmix — and that
    # recipe's shim is the only one that can feed `resnet50Imagenet160Verified` (d0 = 76,800).
    ("resnet50in160", "MainResnet50Imagenet.lean",   "ResNet-50 (ImageNet)",         "short"),
    # ⭐ MNv4 joined 2026-08-12. ⚠⚠ THE ONE ROW WHERE THE VERIFIED NET IS A DIFFERENT SIZE FROM ITS
    # REFERENCE: `mnv4ImagenetVerified` is Conv-**S** (4.1M trunk), the reference here is
    # Conv-**M** (~9.7M). That is sound for a SHIM pairing, which is all this file gates — a shim
    # emits augmented batches and never weights, so what crosses over is the MNv4-family data
    # pipeline both sizes share. It is NOT sound for accuracy: never quote Conv-M's ImageNet
    # number against a Conv-S run. Stated here because this table is where someone looks to find
    # out what a verified net is a port of.
    ("mnv4in",     "MainMobilenetV4Imagenet.lean",   "MobileNetV4-Conv-M (ImageNet, bf16)", "default"),
]

FAILS = []
def check(ok, label, detail=""):
    print(("  ✅ " if ok else "  ❌ ") + label + (("  — " + detail) if detail else ""))
    if not ok:
        FAILS.append(label)
    return ok

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  Readers — each pulls facts out of ONE source of truth, never out of this file.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def read_shim_scripts():
    """slug -> shimScript, straight out of LeanMlir/VerifiedNets.lean.

    Reads the ACTUAL wiring the driver compiles against. A gate restating the map here would pass
    while the driver spawned something else — the shape of defect this whole thread is about."""
    src = open(os.path.join(ROOT, "LeanMlir", "VerifiedNets.lean")).read()
    out, imagenet_slugs = {}, []
    # Split on `def <name> : VerifiedNetSpec where` so a field is attributed to the right net.
    for block in re.split(r"^def\s+\w+\s*:\s*VerifiedNetSpec\s+where", src, flags=re.M)[1:]:
        m = re.search(r'^\s*slug\s*:=\s*"([^"]+)"', block, re.M)
        if not m:
            continue
        slug = m.group(1)
        if re.search(r"^\s*data\s*:=\s*\.imagenet\s*$", block, re.M):
            imagenet_slugs.append(slug)
            s = re.search(r'^\s*shimScript\s*:=\s*"([^"]*)"', block, re.M)
            out[slug] = s.group(1) if s else None      # None = the field is absent entirely
    return out, imagenet_slugs

def read_recipe(mainfile, recipe="default"):
    """(configName, shimFilename) for a Main*Imagenet.lean's `<recipe>` entry.

    The shim's name comes from the recipe's `out`, so it is DERIVED here rather than hardcoded —
    renaming a recipe moves the expectation and the wiring together or the gate fires.

    ⚠ `recipe` was a hardcoded "default" until 2026-08-06. It is a parameter because a net's
    RESOLUTION lives in its recipe's TrainConfig: `resnet50in160` is the same reference and the
    same Main file as `resnet50in`, differing only in that it ports `short` (RSB-A3, trainRes 160)
    instead of `default`. Reading the flags off the wrong recipe would compare a RandAugment shim
    against a no-RandAugment config and fire on a correct wiring."""
    src = open(os.path.join(ROOT, "jax", mainfile)).read()
    m = re.search(r'\{\s*name\s*:=\s*"' + re.escape(recipe) + r'"\s*,\s*cfg\s*:=\s*(\w+)\s*,\s*'
                  r'out\s*:=\s*"([^"]+)"', src, re.S)
    if not m:
        raise SystemExit(f"{mainfile}: could not find the `{recipe}` recipe entry")
    cfg, out = m.group(1), m.group(2)
    return cfg, (out[:-3] if out.endswith(".py") else out) + "_shim.py"

def _config_body(mainfile, cfgname):
    """(rawText, baseConfigName|None) for `def <cfgname> : TrainConfig ...`.

    ⚠ Handles BOTH declaration forms. Until 2026-08-06 this only matched `... TrainConfig where`,
    which silently excluded every config written as a structure UPDATE
    (`def X : TrainConfig := { Base with ... }`) — the form 20+ configs across the Main files use,
    including every `*ConfigShort`. A gate that cannot READ a config cannot check it, and this one
    was reachable the moment a verified net ported a non-`default` recipe."""
    src = open(os.path.join(ROOT, "jax", mainfile)).read()
    m = re.search(rf"^def\s+{cfgname}\s*:\s*TrainConfig\b", src, re.M)
    if not m:
        raise SystemExit(f"{mainfile}: no `def {cfgname} : TrainConfig`")
    body = []
    for line in src[m.end():].splitlines()[1:]:
        if line.strip() and not line.startswith(" "):
            break                                    # dedent ⇒ the next declaration
        body.append(line)
    body = "\n".join(body)
    mb = re.search(r"\{\s*(\w+)\s+with\b", body)
    return body, (mb.group(1) if mb else None)

# The RAW TrainConfig fields the partition is derived from, with their structure defaults
# (LeanMlir/Types.lean). ⚠ Inheritance is resolved HERE, at the FIELD level, never on the derived
# features: `randaugment` is `useRandAugment ∧ randAugmentGeometric`, so a derived config that
# overrides only one of the two is mis-read by any coarser merge.
_RAW_BOOL = ["useAutoAugment", "useRandAugment", "randAugmentGeometric",
             "randomErasing", "useMixup", "useCutmix"]
_RAW_NAT  = {"repeatedAug": 1}

def read_raw_config(mainfile, cfgname, _depth=0):
    """Raw TrainConfig fields, following `{ Base with ... }` chains base-first."""
    if _depth > 8:
        raise SystemExit(f"{mainfile}: the `with` chain from {cfgname} does not terminate")
    body, base = _config_body(mainfile, cfgname)
    raw = (read_raw_config(mainfile, base, _depth + 1) if base
           else {**{k: False for k in _RAW_BOOL}, **_RAW_NAT})
    for name in _RAW_BOOL:
        mm = re.search(rf"^\s*{name}\s*:=\s*(true|false)\b", body, re.M)
        if mm:
            raw[name] = (mm.group(1) == "true")
    for name in _RAW_NAT:
        mm = re.search(rf"^\s*{name}\s*:=\s*(\d+)\b", body, re.M)
        if mm:
            raw[name] = int(mm.group(1))
    return raw

def read_config_flags(mainfile, cfgname):
    """The augmentation flags of one TrainConfig. Absent field ⇒ INHERITED if the config is a
    structure update, else the structure's default (all off / 1 / 0, per LeanMlir/Types.lean)."""
    raw = read_raw_config(mainfile, cfgname)
    b = lambda name: raw[name]
    n = lambda name, dflt: raw.get(name, dflt)
    return {
        "autoaugment":  b("useAutoAugment"),
        # The full geometric sampler. `useRandAugment` alone is the colour-only "lite" path, which
        # emits inline tf.image calls rather than `_randaugment` — a different call site, so the
        # two are separated here instead of collapsed.
        "randaugment":  b("useRandAugment") and b("randAugmentGeometric"),
        "ra_lite":      b("useRandAugment") and not b("randAugmentGeometric"),
        "random_erase": b("randomErasing"),
        "repeated_aug": n("repeatedAug", 1) > 1,
        "mixup":        b("useMixup"),
        "cutmix":       b("useCutmix"),
    }

def shim_call_sites(path):
    """What the generated Python actually DOES — call sites, never definitions."""
    py = open(path).read()
    return {
        "autoaugment":  "img = _autoaugment(" in py,
        "randaugment":  "img = _randaugment(" in py,
        "ra_lite":      "img = tf.image.random_brightness(" in py,
        "random_erase": "img = _random_erase(" in py,
        # Repeated augmentation is a STREAM change (K copies before the shuffle), not a per-image
        # one — `flat_map` is its only fingerprint, and no other feature emits one.
        "repeated_aug": "flat_map" in py,
    }

def shim_defs(path):
    """The count-based census C1 exists to discredit: definitions, not calls."""
    py = open(path).read()
    return {
        "autoaugment":  "def _autoaugment" in py,
        "randaugment":  "def _randaugment" in py,
        "random_erase": "def _random_erase" in py,
    }

def shim_mix_default(path):
    m = re.search(r"os\.environ\.get\('SHIM_MIX',\s*'(\w+)'\)", open(path).read())
    return m.group(1) if m else None

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  Gate 0 — the wiring is a bijection
# ──────────────────────────────────────────────────────────────────────────────────────────────
print("── gate 0: every .imagenet net names its OWN shim, all distinct, all on disk ──")
wiring, imagenet_slugs = read_shim_scripts()
check(len(imagenet_slugs) == len(NETS),
      f"{len(NETS)} .imagenet nets in VerifiedNets.lean",
      f"found {len(imagenet_slugs)}: {imagenet_slugs}")
for slug, _, _, _ in NETS:
    s = wiring.get(slug)
    check(bool(s), f"{slug}: shimScript is set",
          "MISSING — the driver would refuse at spawn" if not s else s)
present = [wiring[s] for s, _, _, _ in NETS if wiring.get(s)]
# THE property the defect violated: five nets resolved to ONE shim. Distinctness is what says the
# fix landed, and a duplicated entry (copy-paste between two nets) is otherwise silent.
check(len(set(present)) == len(present), "all resolve to DISTINCT shims",
      f"{len(set(present))} distinct of {len(present)}")
for slug, _, _, _ in NETS:
    s = wiring.get(slug)
    if s:
        check(os.path.exists(os.path.join(BUILD, s)), f"{slug}: {s} exists",
              "" if os.path.exists(os.path.join(BUILD, s)) else "run scripts/gen_shims.sh")

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  Gate 1 — the shim named is the one generated from THAT net's reference
# ──────────────────────────────────────────────────────────────────────────────────────────────
print("\n── gate 1: each shim's banner names the reference its net is a port of ──")
recipes = {}
for slug, mainfile, refname, recipe in NETS:
    cfgname, expected_file = read_recipe(mainfile, recipe)
    recipes[slug] = (mainfile, cfgname, expected_file)
    # The filename is DERIVED from the recipe's `out`; this compares it to what Lean spawns.
    check(wiring.get(slug) == expected_file, f"{slug}: shimScript == {mainfile}'s `{recipe}` `out`",
          f"wired {wiring.get(slug)!r}, recipe writes {expected_file!r}")
    p = os.path.join(BUILD, expected_file)
    if os.path.exists(p):
        head = open(p).read(2000)
        check(refname in head, f"{slug}: banner names {refname!r}",
              "" if refname in head else "the file on disk was generated from a DIFFERENT recipe")

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  Gate 2 — the augmentation PARTITION: config flags vs generated call sites, feature by feature
# ──────────────────────────────────────────────────────────────────────────────────────────────
print("\n── gate 2: the augmentation partition — config vs the generated CALL SITES ──")
FEATURES = ["autoaugment", "randaugment", "ra_lite", "random_erase", "repeated_aug"]
print(f"    {'net':<10}" + "".join(f"{f:<14}" for f in FEATURES))
flags_by_slug, sites_by_slug = {}, {}
for slug, mainfile, _, _ in NETS:
    _, cfgname, shimfile = recipes[slug]
    flags = read_config_flags(mainfile, cfgname)
    p = os.path.join(BUILD, shimfile)
    if not os.path.exists(p):
        continue
    sites = shim_call_sites(p)
    flags_by_slug[slug], sites_by_slug[slug] = flags, sites
    row = "".join(("on" if sites[f] else "-").ljust(14) for f in FEATURES)
    print(f"    {slug:<10}{row}")
    for f in FEATURES:
        if flags[f] != sites[f]:
            check(False, f"{slug}: {f}",
                  f"config says {flags[f]}, the shim call site says {sites[f]}")
    if all(flags[f] == sites[f] for f in FEATURES):
        check(True, f"{slug}: all {len(FEATURES)} features match its config")

# ⚠ ANTI-VACUITY. Every feature above reads False on every net ⇒ the gate is comparing two
# constants and cannot fail. Require the partition to be NON-TRIVIAL: at least one net on and at
# least one net off, for at least one feature. (R34/mnv2 are the "off" side by construction.)
nontrivial = [f for f in FEATURES
              if any(s[f] for s in sites_by_slug.values())
              and any(not s[f] for s in sites_by_slug.values())]
check(len(nontrivial) >= 3, "the partition is non-trivial (≥3 features split the nets)",
      f"split features: {nontrivial}")

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  C1 — the control that licenses gate 2's shape: counting DEFINITIONS gets the answer wrong
# ──────────────────────────────────────────────────────────────────────────────────────────────
print("\n── control C1: a definition-census gate mis-classifies nets that gate 2 gets right ──")
wrong = []
for slug, _, _, _ in NETS:
    _, _, shimfile = recipes[slug]
    p = os.path.join(BUILD, shimfile)
    if not os.path.exists(p):
        continue
    defs, sites = shim_defs(p), sites_by_slug[slug]
    for f in ("autoaugment", "randaugment", "random_erase"):
        if defs[f] != sites[f]:
            wrong.append(f"{slug}/{f} (defined, not called)")
# `generateShim` emits the shared `_aa_*` block for geometric RandAugment too, so ViT and ConvNeXt
# DEFINE _autoaugment without calling it. If this ever reads 0 the emitter changed shape and gate 2
# no longer needs to be call-site-based — which is a finding, not a pass.
check(len(wrong) >= 1,
      "counting definitions is measurably wrong on ≥1 net (so gate 2 must read call sites)",
      f"{len(wrong)} mis-classifications: {wrong}")

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  Gate 3 — the producer default. ⚠ This one CHANGES BEHAVIOUR under SHIM_SOFT=1.
# ──────────────────────────────────────────────────────────────────────────────────────────────
print("\n── gate 3: the baked SHIM_MIX default == the config's useMixup/useCutmix ──")
for slug, _, _, _ in NETS:
    _, _, shimfile = recipes[slug]
    p = os.path.join(BUILD, shimfile)
    if not os.path.exists(p):
        continue
    fl = flags_by_slug[slug]
    want = ("both" if fl["mixup"] and fl["cutmix"] else
            "mixup" if fl["mixup"] else "cutmix" if fl["cutmix"] else "off")
    got = shim_mix_default(p)
    check(got == want, f"{slug}: SHIM_MIX default {got!r}", f"config implies {want!r}")
print("    ⚠ ViT and ConvNeXt default to 'both'. Before this wiring every net ran R34's shim, whose"
      "\n      default is 'off' — so `SHIM_SOFT=1` on those two now MIXES unless SHIM_MIX=off. That is"
      "\n      their reference's recipe, but it is a behaviour change, not just an augmentation one.")

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  C2 — the stream, measured. Optional: needs tfds imagenet2012 and a few minutes (no GPU).
# ──────────────────────────────────────────────────────────────────────────────────────────────
def shim_hash(path, split, batch, nbatches, seed, mix="off"):
    """Hash `nbatches` batches the way the DRIVER will produce them.

    ⚠ `SHIM_MIX=off` is not a convenience here — it is what `spawnShim` passes on wire v1, because
    ViT's and ConvNeXt's shims bake `both` as their default and a mixed target cannot ride int32
    labels. Hashing without it measures a stream the trainer never sees (and on those two nets it
    does not measure anything at all: the shim exits). C2b below is that failure, on purpose."""
    env = dict(os.environ, SHIM_BATCH=str(batch), SHIM_SPLIT=split,
               SHIM_SEED=str(seed), SHIM_HASH=str(nbatches), SHIM_MIX=mix)
    py = os.environ.get("SHIM_PYTHON", os.path.join(ROOT, ".venv", "bin", "python3"))
    r = subprocess.run([py, path], env=env, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{os.path.basename(path)} ({split}) failed:\n{r.stderr[-3000:]}")
    # Anchored on the shim's own `SHIM_HASH … : <digest>` line rather than "the last hex-looking
    # run", because TF logs timestamps that a loose pattern happily matches.
    m = re.search(r"SHIM_HASH[^\n]*?:\s*([0-9a-f]{16,})", r.stdout + r.stderr)
    if not m:
        raise SystemExit(f"{os.path.basename(path)}: no digest in SHIM_HASH output:\n"
                         f"{(r.stdout + r.stderr)[-2000:]}")
    return m.group(1)

if "--stream" in sys.argv:
    print("\n── control C2: the streams, MEASURED (SHIM_HASH over tfds) ──")
    B, N = 8, 2
    train, val = {}, {}
    for slug, _, _, _ in NETS:
        _, _, shimfile = recipes[slug]
        p = os.path.join(BUILD, shimfile)
        train[slug] = shim_hash(p, "train", B, N, 7)
        val[slug] = shim_hash(p, "validation", B, N, 7)
        print(f"    {slug:<10} train {train[slug][:16]}   val {val[slug][:16]}")
    # determinism — the property the shim already claimed; it was only ever measured on R34's.
    again = shim_hash(os.path.join(BUILD, recipes["vitin"][2]), "train", B, N, 7)
    check(again == train["vitin"], "determinism: same seed twice ⇒ same digest (ViT, the busiest)",
          f"{again[:16]} vs {train['vitin'][:16]}")
    # KNOWN ANSWER, predicted from the bytes: mnv2's config sets `useAutoAugment := false` and
    # nothing else, so its shim differs from R34's in exactly one comment line ⇒ the streams must be
    # IDENTICAL. This is the inertness half — it says the wiring did not perturb a net it shouldn't.
    check(train["mobilenetv2in"] == train["resnet34in"],
          "known answer: mnv2 ≡ R34 on train (their shims differ only in the banner)",
          f"{train['mobilenetv2in'][:16]} vs {train['resnet34in'][:16]}")
    # DISCRIMINATION — and it is the whole claim of this thread: the three nets whose references
    # ask for more augmentation now get a DIFFERENT stream than they did yesterday.
    for slug in ("vitin", "convnextin", "efficientnetin"):
        check(train[slug] != train["resnet34in"],
              f"discrimination: {slug}'s train stream ≠ R34's (it did NOT differ before this)",
              f"{train[slug][:16]} vs {train['resnet34in'][:16]}")
    # The val split takes the center-crop path on every net and no config here sets testCropRatio,
    # so all five must agree. A difference means an augmentation leaked into eval — which would be
    # the mixup split defect (§0.4) one layer down.
    check(len(set(val.values())) == 1,
          "val is untouched: every validation stream is IDENTICAL",
          f"{len(set(val.values()))} distinct digests")
    # C2b — the failure the driver's SHIM_MIX handling exists to prevent, measured rather than
    # argued. ViT's shim, left to its own baked default on a wire-v1 TRAIN stream, exits before
    # emitting anything; through the driver the symptom would be the useless "shim closed the pipe
    # after 0 of 16 bytes", since the child's stderr is not captured.
    env = dict(os.environ, SHIM_BATCH="8", SHIM_SPLIT="train", SHIM_SEED="7", SHIM_HASH="1")
    env.pop("SHIM_MIX", None)
    py = os.environ.get("SHIM_PYTHON", os.path.join(ROOT, ".venv", "bin", "python3"))
    r = subprocess.run([py, os.path.join(BUILD, recipes["vitin"][2])],
                       env=env, capture_output=True, text=True, cwd=ROOT)
    check(r.returncode != 0 and "needs SHIM_NCLASSES" in (r.stdout + r.stderr),
          "C2b: ViT's shim REFUSES a v1 train stream at its own SHIM_MIX default "
          "(so the driver must pass off — it does)",
          f"rc={r.returncode}")
    # ...and the same shim on the VALIDATION split does NOT refuse, because `_MIX_ON` is
    # `and training`. That asymmetry is the mixup split rule (§0.4), re-measured one layer up.
    env["SHIM_SPLIT"] = "validation"
    r2 = subprocess.run([py, os.path.join(BUILD, recipes["vitin"][2])],
                        env=env, capture_output=True, text=True, cwd=ROOT)
    check(r2.returncode == 0,
          "C2b': the same shim, same env, VAL split — does not refuse (mixing is train-only)",
          f"rc={r2.returncode}")
else:
    print("\n── control C2 SKIPPED (pass --stream to measure the actual byte streams) ──")

# ──────────────────────────────────────────────────────────────────────────────────────────────
#  --break — prove the gates go red, and for the reason they claim
# ──────────────────────────────────────────────────────────────────────────────────────────────
if "--break" in sys.argv:
    print("\n── --break: the negative controls ──")
    # (a) the pre-fix world: every net wired to R34's shim. Gate 0's distinctness must fire, and so
    #     must gate 2 on every net whose augmentation vanishes — i.e. every net whose recipe is
    #     NOT R34's. That count is 5 as of 2026-08-07: vitin, efficientnetin, convnextin, resnet50in and
    #     resnet50in160.
    #     ⚠ It was 3 until R50 was wired, then 4, and 5 once R50 gained its SECOND recipe — the
    #     `short` (RSB-A3) shim at trainRes 160. Both R50 rows share a Main file and a reference
    #     but differ in RECIPE, and the partition keys on the recipe, so they count separately.
    #     mobilenetv2in still does not count because its config IS R34's (no extra augmentation), which
    #     is why gate 2's partition treats them as one class.
    #     Bump this deliberately when a net joins — a wrong constant here makes control A red
    #     and the whole file useless, which is exactly what it did when R50 arrived.
    r34 = os.path.join(BUILD, "generated_resnet34_imagenet_shim.py")
    faux = {s: "generated_resnet34_imagenet_shim.py" for s, _, _, _ in NETS}
    distinct_fired = len(set(faux.values())) != len(faux)
    partition_fired = sum(
        1 for slug, _, _, _ in NETS
        if any(flags_by_slug[slug][f] != shim_call_sites(r34)[f] for f in FEATURES))
    print(f"    pre-fix wiring (every net → R34's shim): distinctness fires = {distinct_fired}, "
          f"partition fires on {partition_fired} nets")
    check(distinct_fired and partition_fired == 5,
          "control A: the pre-fix wiring is REJECTED, on exactly the 5 affected nets",
          f"distinct={distinct_fired}, partition fired on {partition_fired} (want 5)")
    # (b) swap two nets' shims — the counts stay right, only the PARTITION is wrong. This is the
    #     `swap1` control from wdx-tie: a gate checking "five distinct shims exist" passes it.
    swapped = dict(wiring)
    swapped["vitin"], swapped["convnextin"] = swapped["convnextin"], swapped["vitin"]
    still_distinct = len(set(swapped.values())) == len(swapped)
    swap_fired = 0
    for slug in ("vitin", "convnextin"):
        sites = shim_call_sites(os.path.join(BUILD, swapped[slug]))
        if any(flags_by_slug[slug][f] != sites[f] for f in FEATURES):
            swap_fired += 1
    print(f"    swapped ViT↔ConvNeXt: still all-distinct shims = {still_distinct}, "
          f"partition fires on {swap_fired} nets")
    check(still_distinct and swap_fired == 2,
          "control B: a SWAP is rejected by the partition though distinctness passes it",
          f"distinct={still_distinct}, partition fired on {swap_fired} (want 2)")

print()
if FAILS:
    print(f"❌ FAIL — {len(FAILS)} gate(s) fired:")
    for f in FAILS:
        print(f"     · {f}")
    sys.exit(1)
print("✅ every gate green")
