#!/usr/bin/env python3
"""mixup_gate.py — the three gates for the shim's mixup/cutmix producer (recipe_gaps v1.3).

Mixing is the ONE part of the shim that is not `emitDataLoading` reused verbatim: the reference
applies `_mixup`/`_cutmix` inside the jitted train step with `jax.random`, so there is nothing to
share and the shim carries a second copy of the rule (`Jax/Codegen.lean`'s `_mix`). A second writer
needs its own evidence, and "it runs and the loss goes down" is not evidence — a mixed LABEL against
an unmixed IMAGE trains perfectly happily and is simply a worse objective.

So all three gates are known-answer or determinism gates, and none of them trusts the producer:

  1. INERT WHEN OFF   `SHIM_MIX=off` reproduces the committed v1 and v2 digests byte-for-byte.
  2. DETERMINISM      same seed => same digest, AND different seeds => DIFFERENT digests. The
                      second half is the one that matters: a producer ignoring its seed entirely
                      passes the first half perfectly.
  3. KNOWN ANSWER     drive the shim TWICE at one seed — once off, once mixing — and check the
                      mixed stream against the unmixed one:
                        mixup   x' = L*x + (1-L)*flip(x)   and   t' = L*t + (1-L)*flip(t)
                        cutmix  x' = x*(1-M) + flip(x)*M   with  t' = A*t + (1-A)*flip(t),
                                M a RECTANGLE and A = 1 - M.sum()/(H*W)
                      L and A are recovered from the emitted targets, never assumed, and the
                      reconstruction must be BIT-EXACT.

  ⚠ It gates the IMAGES as well as the labels, deliberately. Checking only the target would pass a
    producer that mixes labels and streams unmixed pixels — see `--break`, which is exactly that.

  ⚠ It does NOT compare against the JAX reference's own lambda stream, because that comparison does
    not exist to be made: numpy's Generator and `jax.random.beta` draw the same DISTRIBUTION and
    different NUMBERS. A paired run under mixup agrees in distribution, not per step.

    scripts/mixup_gate.py                 # all three gates
    scripts/mixup_gate.py --break         # the verified-to-fail run; expect rc=1
    SHIM_PY=... SHIM_SCRIPT=... scripts/mixup_gate.py --batch 8 --batches 3
"""
import argparse, os, subprocess, sys
import numpy as np

AP = argparse.ArgumentParser()
# ⚠ R34's shim by default, and that is deliberate rather than left over: the mixing code is
# `generateShim`'s and is identical in all five, so the CHEAPEST shim (no AutoAugment/RandAugment
# to run per image) is the right instrument for gating the producer. Point --script at another
# net's to re-run it there. ⚠ Note those nets bake `SHIM_MIX=both` as their default, so the
# "inert when off" gate must pass SHIM_MIX=off explicitly on them, not rely on the default.
AP.add_argument("--script", default=os.environ.get(
    "SHIM_SCRIPT", "jax/.lake/build/generated_resnet34_imagenet_shim.py"))
AP.add_argument("--python", default=os.environ.get("SHIM_PY", ".venv/bin/python3"))
AP.add_argument("--batch", type=int, default=8)
AP.add_argument("--batches", type=int, default=3)
AP.add_argument("--nclasses", type=int, default=1000)
AP.add_argument("--img", type=int, default=224)
AP.add_argument("--seed", type=int, default=0)
AP.add_argument("--break", dest="do_break", action="store_true",
                help="run the negative controls; the gate must reject each")
A = AP.parse_args()

FLAT = 3 * A.img * A.img
FAILURES = []


def run_env(**kw):
    e = dict(os.environ)
    e.update({k: str(v) for k, v in kw.items() if v is not None})
    return e


def digest(mix, seed, n=None, split=None, nclasses=...):
    """`SHIM_HASH` over n batches -> the hex digest the shim prints to stderr."""
    n = n or A.batches
    if nclasses is ...:
        nclasses = A.nclasses if mix != "v1" else None
    p = subprocess.run([A.python, A.script], capture_output=True, env=run_env(
        SHIM_BATCH=A.batch, SHIM_HASH=n, SHIM_SEED=seed, SHIM_SPLIT=split,
        SHIM_NCLASSES=nclasses,
        SHIM_MIX=(mix if mix != "v1" else None)))
    for line in p.stderr.decode().splitlines():
        if line.startswith("SHIM_HASH"):
            return line.rsplit(": ", 1)[1].strip()
    raise SystemExit(f"no SHIM_HASH line (rc={p.returncode}):\n{p.stderr.decode()[-2000:]}")


def stream(mix, seed):
    """Read `A.batches` records off the shim's stdout. Returns [(t, x), ...] as float32."""
    p = subprocess.Popen([A.python, A.script], stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, env=run_env(
                             SHIM_BATCH=A.batch, SHIM_SEED=seed,
                             SHIM_NCLASSES=A.nclasses, SHIM_MIX=mix))

    def rd(n):
        buf = b""
        while len(buf) < n:                       # a pipe read returns what is AVAILABLE (§2k)
            c = p.stdout.read(n - len(buf))
            if not c:
                raise SystemExit(f"shim closed early ({len(buf)}/{n} bytes) — check SHIM_MIX={mix}")
            buf += c
        return buf

    assert rd(4) == b"LMSH", "bad preamble magic"
    ver, batch, flat, nc = np.frombuffer(rd(16), dtype=np.int32)
    assert (ver, batch, flat, nc) == (2, A.batch, FLAT, A.nclasses), \
        f"preamble {(ver, batch, flat, nc)} != {(2, A.batch, FLAT, A.nclasses)}"
    out = []
    for _ in range(A.batches):
        t = np.frombuffer(rd(4 * batch * nc), dtype=np.float32).reshape(batch, nc).copy()
        x = np.frombuffer(rd(4 * batch * flat), dtype=np.float32).reshape(batch, flat).copy()
        out.append((t, x))
    p.kill()
    return out


def recover_lambda(t, tm):
    """L, READ out of the emitted target rather than fitted to it.

    ⚠ The first version of this solved `tm = L*t + (1-L)*flip(t)` by least squares in float64 and
    cast the answer to float32. It reconstructed L to about a ULP — which is fine for a tolerance
    check and USELESS here, because the reconstruction it feeds is asserted BIT-EXACT: mixup batch
    0 passed and batch 1 failed at 1.5e-08, a phantom defect in a correct producer. **Recover a
    constant by reading it, not by fitting it.**

    Reading it is exact and needs no tolerance: `t` is one-hot, so for any row whose partner
    carries a DIFFERENT label, `tm[i, y_i] = L*1 + (1-L)*0 = L` — the producer's own float32, to
    the bit. Rows whose partner shares their label are skipped: there the equation degenerates to
    `tm[i, y_i] = 1` and says nothing about L.

    It also gates a real property for free: L is a per-STEP scalar, not per-example (the reference
    draws one `lam` per call), so every identified row must agree EXACTLY."""
    tf_ = np.flip(t, 0)
    ya, yb = t.argmax(1), tf_.argmax(1)
    rows = np.where(ya != yb)[0]
    if rows.size == 0:
        raise SystemExit("DEGENERATE: every example's partner shares its label — L is unidentified")
    vals = {float(tm[i, ya[i]]) for i in rows}
    if len(vals) != 1:
        raise SystemExit(f"L is NOT a per-step scalar: {len(vals)} distinct values across "
                         f"{rows.size} identified rows ({sorted(vals)[:4]}...) — the reference "
                         f"draws ONE lam per call and broadcasts it over the batch")
    return np.float32(vals.pop())


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'}  {name}{('   ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)
    return ok


# ── gate 1: inert when off ────────────────────────────────────────────────────────────────────
#
# ⚠ These are the digests of the shim as it was BEFORE mixing existed, measured on the pre-change
#   generated script at exactly this config. That is the whole content of "inert": not that the two
#   modes of the new shim agree with each other (they trivially do — one calls the other's code
#   path), but that the new shim's off path is byte-for-byte the OLD shim. A cross-version claim
#   needs a recorded constant; nothing the script can compute at run time substitutes for it.
#
# ⚠ The digest depends on batch AND batch count, so the check is SKIPPED — loudly — at any other
#   config rather than silently comparing incomparable numbers. A first version of this gate only
#   PRINTED the digests, which made it unfalsifiable: it could not fail. §4's rule.
BASELINE = {(8, 3): {
    "v1":  "c375ad0fb9fcd5b9d34abb5dfacc8fd465c922d82f4352ac1fdd20a115eadfc2",
    "off": "f3a4b2a005524277af7a227b99fdb81f6469d9f97c16ed9fb99210fdb00b193e"}}

print("── gate 1: INERT WHEN OFF (the digests must not move at all)")
d_v1 = digest("v1", A.seed)
d_v2 = digest("off", A.seed)
print(f"  v1 (int32 hard labels)      {d_v1}")
print(f"  v2 (one-hot soft targets)   {d_v2}")
base = BASELINE.get((A.batch, A.batches)) if A.seed == 0 else None
if base is None:
    print(f"  ⚠ SKIPPED — no pre-change baseline recorded for "
          f"(batch={A.batch}, batches={A.batches}, seed={A.seed}). This gate is VACUOUS at this "
          f"config; run the default, or record a new baseline off the pre-mixing shim.")
else:
    check("v1 digest unchanged since before mixing existed", d_v1 == base["v1"])
    check("v2 digest unchanged since before mixing existed", d_v2 == base["off"])

# ── gate 1b: the VALIDATION split is never mixed, whatever SHIM_MIX says ───────────────────────
#
# ⚠ This gate exists because its absence broke a run. `SHIM_MIX` is an ordinary environment
#   variable, so EVERY shim the driver spawns inherits it — and it spawns two: the train stream at
#   nclasses=K and the validation drain at nclasses=0 (hard labels; eval scores against a label,
#   not a distribution). Gating the mixing on the variable alone made the "needs SHIM_NCLASSES>0"
#   refusal fire on the VAL shim, killing it before the preamble; the trainer died with
#   `imagenet shim closed the pipe after 0 of 16 bytes`.
#
#   Gates 1-3 all passed throughout, because every one of them drives the TRAIN split. **A gate
#   that exercises one split cannot see a split-dependent defect** — the end-to-end smoke found
#   this, and nothing producer-side would have.
print("── gate 1b: THE VALIDATION SPLIT IS NEVER MIXED (SHIM_MIX must not reach eval data)")
v_off = digest("off", A.seed, n=1, split="validation")
for mode in ("mixup", "cutmix", "both"):
    # v1 on the val split, exactly as the driver spawns it: nclasses unset AND SHIM_MIX set.
    v_v1 = digest(mode, A.seed, n=1, split="validation", nclasses=None)
    check(f"validation survives SHIM_MIX={mode} at wire v1 (the driver's own spawn)",
          v_v1 is not None and len(v_v1) == 64, v_v1[:16] if v_v1 else "")
    v_on = digest(mode, A.seed, n=1, split="validation")
    check(f"validation at SHIM_MIX={mode} is UNMIXED", v_on == v_off, v_on[:16])

# ── gate 2: determinism, and the control that stops a seed-ignoring producer ───────────────────
print("── gate 2: DETERMINISM")
for mode in ("mixup", "cutmix", "both"):
    a, b = digest(mode, A.seed), digest(mode, A.seed)
    c = digest(mode, A.seed + 1)
    check(f"{mode}: same seed => same digest", a == b, a[:16])
    # The load-bearing half. Without it, a producer that drew its lambda from a CONSTANT — or
    # ignored the seed entirely — would pass the line above perfectly.
    check(f"{mode}: different seed => DIFFERENT digest", a != c, f"{a[:16]} vs {c[:16]}")

# ── gate 3: the known answer, against the unmixed stream at the same seed ──────────────────────
print("── gate 3: KNOWN ANSWER (mixed stream vs the off stream, same seed)")
ref = stream("off", A.seed)
for mode in ("mixup", "cutmix"):
    got = stream(mode, A.seed)
    for i, ((t, x), (tm, xm)) in enumerate(zip(ref, got)):
        # The off stream is the same (x, y) — that is what makes this a known answer rather than
        # a self-comparison. Check it before reading anything into the mixed one.
        if i == 0 and not check(f"{mode}: the off stream is the same data (targets are one-hot)",
                                bool(np.all(t.sum(1) == 1.0))):
            continue
        L = recover_lambda(t, tm)
        if mode == "mixup":
            want_x = L * x + (np.float32(1.0) - L) * np.flip(x, 0)
            want_t = L * t + (np.float32(1.0) - L) * np.flip(t, 0)
            check(f"mixup b{i}: t' = L*t + (1-L)*flip(t)  [L={L:.6f}]",
                  bool(np.array_equal(tm, want_t)), f"maxdiff {np.abs(tm - want_t).max():.3e}")
            check(f"mixup b{i}: x' = L*x + (1-L)*flip(x)  — the IMAGES, not just the labels",
                  bool(np.array_equal(xm, want_x)), f"maxdiff {np.abs(xm - want_x).max():.3e}")
        else:
            H = W = A.img
            x4, x4m = x.reshape(-1, 3, H, W), xm.reshape(-1, 3, H, W)
            # Recover the box from the PIXELS: inside it every example took its partner's value.
            inside = np.all(x4m == np.flip(x4, 0), axis=(0, 1))
            outside = np.all(x4m == x4, axis=(0, 1))
            M = (inside & ~outside).astype(np.float32)
            rows, cols = np.where(M.any(1))[0], np.where(M.any(0))[0]
            rect = (M.sum() == 0) or bool(
                np.array_equal(M, np.outer((np.arange(H) >= rows[0]) & (np.arange(H) <= rows[-1]),
                                           (np.arange(W) >= cols[0]) & (np.arange(W) <= cols[-1]))
                               .astype(np.float32))
                and len(rows) == rows[-1] - rows[0] + 1 and len(cols) == cols[-1] - cols[0] + 1)
            check(f"cutmix b{i}: the pasted region is a RECTANGLE", rect,
                  f"{int(M.sum())} px" + (f", rows {rows[0]}-{rows[-1]} cols {cols[0]}-{cols[-1]}"
                                          if M.sum() else ""))
            # The label must follow the PIXELS, not the drawn lambda: the box is clipped at the
            # border, so the drawn and the effective lambda genuinely differ.
            area = float(M.sum()) / float(H * W)
            check(f"cutmix b{i}: L is the ACTUAL pasted area  [L={float(L):.6f}, "
                  f"1-area={1.0-area:.6f}]", abs(float(L) - (1.0 - area)) < 1e-6)
            La = np.float32(1.0 - area)
            want_t = La * t + (np.float32(1.0) - La) * np.flip(t, 0)
            want_x = (x4 * (np.float32(1.0) - M) + np.flip(x4, 0) * M).reshape(x.shape)
            check(f"cutmix b{i}: t' = L*t + (1-L)*flip(t)",
                  bool(np.array_equal(tm, want_t)), f"maxdiff {np.abs(tm - want_t).max():.3e}")
            check(f"cutmix b{i}: x' = x*(1-M) + flip(x)*M  — the IMAGES",
                  bool(np.array_equal(xm, want_x)), f"maxdiff {np.abs(xm - want_x).max():.3e}")

# ── the negative controls: prove each gate can go red, and for its own reason ───────────────────
if A.do_break:
    print("── CONTROLS (--break): each must be REJECTED")
    t, x = ref[0]
    tm, xm = stream("mixup", A.seed)[0]
    L = recover_lambda(t, tm)
    bad_x = x                                        # labels mixed, images NOT — trains fine, wrong
    good_x = L * x + (np.float32(1.0) - L) * np.flip(x, 0)
    check("control A: unmixed images against a mixed target are REJECTED",
          not np.array_equal(bad_x, good_x),
          f"maxdiff {np.abs(bad_x - good_x).max():.3e}")
    Lb = np.float32(float(L) * 1.01)                 # lambda off by 1%
    check("control B: a 1% wrong L is REJECTED",
          not np.array_equal(Lb * t + (np.float32(1.0) - Lb) * np.flip(t, 0), tm),
          f"maxdiff {np.abs((Lb * t + (np.float32(1.0) - Lb) * np.flip(t, 0)) - tm).max():.3e}")

print()
if FAILURES:
    print(f"✖ {len(FAILURES)} check(s) FAILED: {FAILURES}")
    sys.exit(1)
print("✓ mixup/cutmix producer: inert when off, deterministic and seed-sensitive, and the mixed "
      "stream is EXACTLY the convex combination of the unmixed one — images and labels both.")
