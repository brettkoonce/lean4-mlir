#!/usr/bin/env python3
"""r50_gradcheck_stability.py — is `r50-gradcheck`'s VERDICT reproducible?

⭐⭐ `a3_paper_fidelity.md` §3.1 records that tier 1 B "does not survive BCE", with one sample of
each number: a passing site at 0.000757 against a weakest control violation of 0.000545, hence "no
tolerance separates the populations". **That framing is incomplete, and this script is what shows
it.** Run the same binary three times on the same seeded base point and the numbers move — the
weakest control violation swings by 5.9× under CE and 11× under BCE. A single sample cannot decide
whether tier 1 B is ill-conditioned, because a single sample is not reproducible.

⚠ And it is not ill-conditioned: tier 1 B's own value under BCE is one of the STABLEST numbers here
(0.000757…0.000771, a 2% spread). The instability was all in the control statistic — see below.

## What is actually nondeterministic

Every input is seeded (`heInit 555` for `%x`, `3131` for the BN stats, per-parameter seeds for θ),
so the BASE POINT is fixed. What moves is the GPU execution: the base LOSS itself differs run to
run in its 6th digit, and `‖g‖` by ~2e-3 relative. ⚠ `XLA_FLAGS=--xla_gpu_deterministic_ops=true`
does **not** fix it — measured, not assumed.

▶ 2e-3 relative noise on the gradient, against a tier-1-B residual of 7.6e-4 under BCE, is the
whole of §3.1: the quantity being measured is smaller than the run-to-run variation of the thing
measuring it. Under CE the residual is ~7e-5 against the same 2e-3 noise, and the gate passes —
but its MARGIN swings from 13.5× to 78×, so CE is lucky rather than sound.

## ⚠⚠ The second finding was a plain bug — ✅ FIXED 2026-08-14 (§3.1b)

`TestR50GradCheck.lean`'s control check carried this comment:

    ⚠ The control is a SEPARATION statement, not an order statistic on one site. Requiring the
    weakest of 21 violations to clear a fixed multiple is brittle …

and the line immediately below it was `if bC <= 5.0 * max wA wB`, where `bC` is `bestOf ctl` — the
MINIMUM over the 21 control sites, i.e. exactly the order statistic the comment rules out. It is the
least stable number in the report, and the verdict rested on it.

✅ The verdict now uses the **10th-percentile** violation, which is what that comment asks for.
Measured here, three reps per variant, before → after:

    adam64     separation  27.6×, 12.0×, 30.8×  (3/3)   →  34.3×, 88.6×, 88.6×  (3/3)
    adam64bce  separation   0.6×,  6.3×,  6.3×  (2/3!)  →  12.3×, 15.1×, 12.3×  (3/3)

▶ The BCE verdict is reproducible again, and CE's headroom over the 5× rule went from 2.4× to 6.9×.
⚠ BCE still FAILS the gate, on `tolExact` — that is §3.1(a), the CE-calibrated tolerance against a
stable 7.6e-4 residual, and it is deliberately left open. This script is how you would re-measure
if that gets recalibrated.

## What this script does NOT do

It does not change any verdict, tolerance or pass criterion — it runs the gate as committed, N
times, and reports the spread. Deciding what the gate ACCEPTS is a human's call with this data in
hand; see `a3_paper_fidelity.md` §3.1.

Usage:
    lake build r50-gradcheck
    scripts/r50_gradcheck_stability.py                    # CE and BCE, 3 reps each
    scripts/r50_gradcheck_stability.py --reps 5 --variants adam64 adam64bce lamb64bce

⚠ Needs one GPU. ~25 s per rep; the default is 6 reps, so about three minutes.
"""
import argparse, os, re, statistics, subprocess, sys

BIN = ".lake/build/bin/r50-gradcheck"

PATTERNS = {
    "loss":      re.compile(r"base loss (\S+)\s+‖g‖ (\S+)"),
    "tierA":     re.compile(r"A  ⟨g_W,W⟩ = 0.*worst \|cos∠\| (\S+) at (\S+)"),
    "tierB":     re.compile(r"B  ⟨g_γ,γ⟩\+⟨g_β,β⟩=0.*worst \|cos∠\| (\S+) at (\S+)"),
    "control":   re.compile(r"weakest violation (\S+) at (.+?);\s+10th pct (\S+);\s+median (\S+);"),
    "tier2":     re.compile(r"worst rel (\S+) at (\S+);"),
}


def one_run(variant, env_extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", R50_GC_VARIANT=variant,
               # ⚠ The tier-1 tolerance is lifted so a FAILING variant still reports its numbers.
               # This measures the STATISTICS, not the verdict; the verdict is recorded separately
               # from the process exit code, which is left exactly as the committed gate produces it.
               R50_GC_EXACT_U="999999", **env_extra)
    p = subprocess.run([BIN], capture_output=True, text=True, env=env, timeout=1800)
    out = p.stdout + p.stderr
    r = {"rc_relaxed": p.returncode}
    for k, rx in PATTERNS.items():
        m = rx.search(out)
        if not m:
            r[k] = None
            continue
        if k == "loss":
            r["loss"], r["gnorm"] = float(m.group(1)), float(m.group(2))
        elif k == "control":
            r["ctl_min"], r["ctl_at"] = float(m.group(1)), m.group(2)
            r["ctl_q10"], r["ctl_med"] = float(m.group(3)), float(m.group(4))
        else:
            r[k] = float(m.group(1))
            r[k + "_at"] = m.group(2)
    return r


def verdict_run(variant):
    """The gate exactly as committed — no relaxed tolerance. Records the real pass/fail."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", R50_GC_VARIANT=variant)
    p = subprocess.run([BIN], capture_output=True, text=True, env=env, timeout=1800)
    first = ""
    for line in (p.stdout + p.stderr).splitlines():
        if "FAILED" in line or "CONTROL DEAD" in line or "TOLERANCE TOO LOOSE" in line:
            first = line.strip()[:110]
            break
    return p.returncode, first


def spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    lo, hi = min(xs), max(xs)
    return f"{lo:.6g} … {hi:.6g}  ({hi/lo:.2f}×)" if lo > 0 else f"{lo:.6g} … {hi:.6g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=["adam64", "adam64bce"])
    a = ap.parse_args()
    if not os.path.exists(BIN):
        sys.exit(f"missing {BIN} — run: lake build r50-gradcheck")

    print("── r50-gradcheck stability (a3_paper_fidelity.md §3.1) ──")
    print(f"   {a.reps} reps per variant, same seeded base point every time\n")
    unstable = 0
    for v in a.variants:
        runs = [one_run(v, {}) for _ in range(a.reps)]
        print(f"  {v}")
        print(f"    base loss      {spread([r.get('loss') for r in runs])}")
        print(f"    ‖g‖            {spread([r.get('gnorm') for r in runs])}")
        print(f"    tier 1 A worst {spread([r.get('tierA') for r in runs])}")
        print(f"    tier 1 B worst {spread([r.get('tierB') for r in runs])}")
        ctl_mins = [r.get("ctl_min") for r in runs]
        print(f"    ⟂ weakest      {spread(ctl_mins)}   <- the MINIMUM; reported, NOT the verdict "
              f"(§3.1b)")
        print(f"    ⟂ 10th pct     {spread([r.get('ctl_q10') for r in runs])}   <- ⭐ THE VERDICT'S "
              f"STATISTIC since 2026-08-14")
        print(f"    ⟂ median       {spread([r.get('ctl_med') for r in runs])}   <- the robust one")
        print(f"    tier 2 worst   {spread([r.get('tier2') for r in runs])}")

        # The separation the verdict actually tests, computed per run so its own spread is visible.
        seps = [r["ctl_q10"] / r["tierB"] for r in runs
                if r.get("ctl_q10") and r.get("tierB")]
        meds = [r["ctl_med"] / r["tierB"] for r in runs
                if r.get("ctl_med") and r.get("tierB")]
        if seps:
            ok = sum(1 for s in seps if s > 5.0)
            print(f"    separation (10th pct/passing, gate wants > 5×): "
                  f"{', '.join(f'{s:.1f}×' for s in seps)}   → {ok}/{len(seps)} reps clear it")
            if 0 < ok < len(seps):
                unstable += 1
                print(f"      ⚠⚠ THE VERDICT IS NOT REPRODUCIBLE for this variant")
        if meds:
            okm = sum(1 for s in meds if s > 100.0)
            print(f"    separation (median/passing,  gate wants > 100×): "
                  f"{', '.join(f'{s:.0f}×' for s in meds)}   → {okm}/{len(meds)} reps clear it")

        rc, why = verdict_run(v)
        print(f"    committed gate (no relaxation): rc {rc}"
              f"{'  — ' + why if why else '  — PASSES'}")
        print()

    if unstable:
        print(f"⚠⚠ {unstable} variant(s) have a verdict that depends on which run you happened to "
              f"do.\n   That is the finding; see a3_paper_fidelity.md §3.1 for the options.")
    else:
        print("✓ every variant's verdict was stable across the reps measured "
              "(which is weaker than 'deterministic' — the NUMBERS still move).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
