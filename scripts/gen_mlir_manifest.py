#!/usr/bin/env python3
"""Generate `verified_mlir/MANIFEST.md` — an index of the committed artifacts.

`verified_mlir/` is ~250 files and ~210 MB, and until now the only way to know what one of them
was came from reading the renderer that wrote it. Two things made that worse than a long
directory listing:

  * **Nothing addresses these files by name.** `VerifiedTrain` builds the path at runtime as
    `{net.slug}_{variant}_train_step.mlir` from `LEAN_MLIR_VARIANT`, so grep finds no reader for
    any of them and "unreferenced" is meaningless. Reachability is a decision, not a measurement.
  * **The variant suffix is a grown DSL** — `lambaccdp8x64bce`, `emarms64dropdo`,
    `adamdpwxclipdrop` — whose markers have collided THREE times (see
    `tests/TestVariantPredicates.lean`; twice in production, once caught first).

⚠⚠ **This decoder is a SECOND statement of the variant convention, and second statements rot** —
that is how `grad_tie.py`'s reference patch list died (§4c(c) of `planning/archive/mnv4_verified.md`).
Two mitigations, and neither is optional:
  1. `tests/TestVariantPredicates.lean` is the AUTHORITY. If it and this file disagree, it wins
     and this file is the bug.
  2. `--selftest` replays the three historical marker collisions through this decoder. A decoder
     that cannot rediscover them is not tracking the Lean predicates.

⚠ **It deliberately does NOT check `func.func @<sym>` against the filename stem.** An earlier
draft of this file did, and claimed to be the only script doing so — that claim was wrong:
`scripts/regen_verified_mlir.sh` already runs exactly that check, with a documented `EXEMPT` set
(`cifar8_adam256_train_step` — a render nothing resolves by `{slug}_{variant}`). Duplicating it here would have meant two copies of the exempt list, which is
the same rot this file's own ⚠⚠ warns about, and the duplicate immediately produced two false
reds. **That invariant has an owner; leave it there.**

Usage:  python3 scripts/gen_mlir_manifest.py            # write verified_mlir/MANIFEST.md
        python3 scripts/gen_mlir_manifest.py --check    # verify only, non-zero if stale
        python3 scripts/gen_mlir_manifest.py --selftest # replay the marker collisions
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MLIR = ROOT / "verified_mlir"
MANIFEST = MLIR / "MANIFEST.md"

# ── The four independent axes, spelled exactly as `trainAdamSched` tests them ────────────────
# (LeanMlir/VerifiedTrain.lean; the collision history is in tests/TestVariantPredicates.lean)
#   EMA              variant.startsWith "ema"   — PREFIX, not substring
#   RMSProp          "rms" substring            — SUBSTRING: `emarms` does not START with rms
#   stochastic depth "drop" substring           — marker is "drop", NOT "sd": `rms`++`dp` = `rmsdp`
#   classifier drop  "do" substring             — marker is "do", NOT "dropout": that contains "drop"
#   accumulation     "acc" substring, k after   — SUBSTRING: `lamb`++`acc` puts it in the middle


def ema_on(v: str) -> bool:
    return v.startswith("ema")


def rms_on(v: str) -> bool:
    return "rms" in v


def sd_on(v: str) -> bool:
    return "drop" in v


def cd_on(v: str) -> bool:
    return "do" in v


def acc_on(v: str) -> bool:
    return "acc" in v


def acc_k(v: str) -> int | None:
    if not acc_on(v):
        return None
    after = v.split("acc", 1)[1]
    if after.startswith("dp"):
        after = after[2:]
    digits = ""
    for c in after:
        if c.isdigit():
            digits += c
        else:
            break
    return int(digits) if digits else None


def selftest() -> int:
    """Replay the three collisions that actually happened. A green here is the only reason to
    believe the decode column below."""
    checks = [
        # 1. ema.md — a PREFIX rms test misses the RMSProp+EMA spelling.
        ("emarms64", rms_on, True, "`emarms` is RMSProp; a startsWith test would say no"),
        ("emarms64", ema_on, True, "...and it is also EMA — the axes are independent"),
        # 2. stochastic_depth.md — the old "sd" marker fired on every RMSProp/DP variant.
        ("rmsdp64", lambda v: "sd" in v, True, "`rms`++`dp` CONTAINS sd — why the marker is `drop`"),
        ("rmsdp64", sd_on, False, "...and the real marker correctly does NOT fire there"),
        # 3. recipe_gaps.md gap C — "dropout" contains "drop".
        ("adamdropout", sd_on, True, "`dropout` contains `drop` — why the dropout marker is `do`"),
        ("adamdrop", cd_on, False, "...and plain stochastic depth is not classifier dropout"),
        # 4. lamb++acc puts the marker in the middle, so `startsWith` would miss it.
        ("lambaccdp8x64bce", acc_on, True, "`lamb`++`acc` — substring, not prefix"),
    ]
    bad = 0
    for v, pred, want, why in checks:
        got = pred(v)
        ok = got == want
        print(f"  {'✓' if ok else '⛔'} {v:<18} -> {got!s:<5} (want {want}) — {why}")
        if not ok:
            bad += 1
    # A negative control: the table must be able to fail.
    if sd_on("adam") is not False:
        print("  ⛔ control: sd_on('adam') should be False")
        bad += 1
    print(f"\n{'✅ selftest passed' if not bad else f'⛔ {bad} selftest failures'}"
          " — authority is tests/TestVariantPredicates.lean, not this file")
    return 1 if bad else 0


def decode(variant: str) -> str:
    """Human-readable decode of a variant suffix. Descriptive only — see the ⚠⚠ above."""
    if not variant:
        return "—"
    bits = []
    if ema_on(variant):
        # 4 regions [θ|m|v|ema], or 5 behind a gradient accumulator [θ|m|v|G|ema]
        bits.append(f"EMA shadow ({5 if acc_k(variant) else 4}-region blob)")
    if rms_on(variant):
        bits.append("RMSProp")
    elif "lamb" in variant:
        bits.append("LAMB")
    elif "adam" in variant or acc_k(variant):
        # `acc…` without `lamb` is `R34Opt.adamwAccum`: AdamW over k micro-batches
        bits.append("AdamW")
    elif "mom" in variant:
        bits.append("momentum")
    elif "sgd" in variant:
        bits.append("SGD")
    elif ema_on(variant):
        # ConvNeXt/ViT/EfficientNet spell `ema` IN PLACE OF `adam` (`cnxAdamVariant`): AdamW + shadow
        bits.append("AdamW")
    if "dp" in variant:
        bits.append("data-parallel")
    k = acc_k(variant)
    if k:
        bits.append(f"grad-accum ×{k}")
    if sd_on(variant):
        bits.append("stochastic depth")
    if cd_on(variant):
        bits.append("classifier dropout")
    if "wx" in variant:
        # timm `no_weight_decay`: decay OFF BN γ/β and biases
        bits.append("no decay on norm/bias")
    if "clip" in variant:
        bits.append("grad clip")
    if "bce" in variant:
        bits.append("BCE loss")
    if "bf16" in variant:
        bits.append("bf16")
    if "fp8" in variant:
        bits.append("fp8")
    # the precision markers carry digits of their own — strip them before reading a batch size
    m = re.search(r"(\d{2,4})(?:x(\d+))?", variant.replace("bf16", "").replace("fp8", ""))
    if m and not k:
        bits.append(f"batch {m.group(1)}")
    return ", ".join(bits) if bits else variant


# Slugs with committed artifacts but no `VerifiedNetsCore` entry: renders kept after their trainer
# was retired. `cifar8b` — the narrow-head CIFAR-8 trainers went on 2026-09-20; its
# `cifar8b{,_bf16,_fp8}` renders stay as `CnnRender` outputs (the §4.1/§5.2 provenance), per the
# note beside `cifar8Verified` in `VerifiedNetsCore.lean`.
RENDER_ONLY_SLUGS = {"cifar8b"}


def slugs_from_verified_nets() -> list[str]:
    src = (ROOT / "LeanMlir" / "VerifiedNetsCore.lean").read_text(encoding="utf-8")
    found = set(re.findall(r'slug\s*:=\s*"([^"]+)"', src)) | RENDER_ONLY_SLUGS
    return sorted(found, key=len, reverse=True)  # longest first, for prefix matching


def writers() -> dict[str, str]:
    # `git grep` over TRACKED files, so the manifest does not depend on what else is on disk.
    out = subprocess.run(
        ["git", "grep", "-n", 'IO.FS.writeFile "verified_mlir/', "--", "*.lean"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    w: dict[str, str] = {}
    for line in out.splitlines():
        path, _lineno, code = line.split(":", 2)
        if code.lstrip().startswith("--"):
            continue  # a comment mentioning the literal, not a writer
        m = re.search(r'IO\.FS\.writeFile "verified_mlir/([^"]+)\.mlir"', code)
        if not m:
            continue
        w.setdefault(m.group(1), path)
    return w


def run_refs() -> dict[str, list[str]]:
    # TRACKED run logs only: an untracked local run must not change a committed file (it made the
    # counts differ from machine to machine).
    logs = subprocess.run(["git", "ls-files", "runs/"], cwd=ROOT, capture_output=True,
                          text=True).stdout.split()
    pat = re.compile(r"compiled verified_mlir/([a-z0-9_]*)\.mlir")
    refs: dict[str, list[str]] = {}
    for rel in sorted(l for l in logs if l.endswith(".log")):
        try:
            txt = (ROOT / rel).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for name in set(pat.findall(txt)):
            refs.setdefault(name, []).append(rel)
    return refs


def build() -> str:
    slugs = slugs_from_verified_nets()
    w = writers()
    refs = run_refs()
    rows = []
    for p in sorted(MLIR.glob("*.mlir")):
        stem = p.stem
        slug = next((s for s in slugs if stem == s or stem.startswith(s + "_")), None)
        rest = stem[len(slug) + 1:] if slug else stem
        kind = ("fwd_eval" if rest.endswith("fwd_eval")
                else "fwd" if rest.endswith("fwd")
                else "train_step" if rest.endswith("train_step") else "?")
        variant = rest.removesuffix("_" + kind) if rest != kind else ""
        rows.append({
            "file": p.name, "slug": slug or "(unknown)", "kind": kind, "variant": variant,
            "mb": p.stat().st_size / 1048576, "writer": w.get(stem, "⛔ no writer"),
            "runs": refs.get(stem, []),
        })

    by_slug: dict[str, list[dict]] = {}
    for r in rows:
        by_slug.setdefault(r["slug"], []).append(r)

    total_mb = sum(r["mb"] for r in rows)
    L = [
        "# `verified_mlir/` — what is in here",
        "",
        "**Generated by `scripts/gen_mlir_manifest.py`. Do not hand-edit.**",
        "",
        f"{len(rows)} artifacts, {total_mb:.0f} MB, {len(by_slug)} slugs.",
        "",
        "## How these are addressed — read this before pruning",
        "",
        "⚠⚠ **Nothing references these files by name.** `VerifiedTrain` builds the path at runtime",
        "as `{net.slug}_{variant}_train_step.mlir` from `LEAN_MLIR_VARIANT`, so *every* artifact",
        "looks unreferenced to grep and *every* one is reachable by setting one env var. A",
        "\"nothing uses this\" measurement over this directory is meaningless; deciding what to drop",
        "is a judgement about which variants are still wanted, not something a script can answer.",
        "",
        "⚠ The **Runs** column is evidence of use, not of importance: an artifact with no run log",
        "may still back a book table (`cifar8w` does — memory `tests-dir-emitters`).",
        "",
        "## The variant suffix",
        "",
        "Four independent axes, each a string test in `trainAdamSched`. The markers have collided",
        "**three times** (twice in production) — `tests/TestVariantPredicates.lean` is the authority",
        "and `gen_mlir_manifest.py --selftest` replays those collisions through this decoder:",
        "",
        "| axis | marker | why this marker |",
        "|---|---|---|",
        "| EMA shadow | `ema` **prefix** | 4th blob region `[θ\\|m\\|v\\|ema]`, 5 scalars |",
        "| RMSProp | `rms` **substring** | `emarms` does not *start* with `rms` |",
        "| stochastic depth | `drop` | not `sd` — `rms`++`dp` spells `rmsdp`, which contains `sd` |",
        "| classifier dropout | `do` | not `dropout` — that contains `drop` |",
        "| grad accumulation | `acc<k>x<B>` **substring** | `lamb`++`acc` puts it mid-string |",
        "",
    ]

    for slug in sorted(by_slug):
        rs = sorted(by_slug[slug], key=lambda r: (r["kind"], r["variant"]))
        L += [f"## `{slug}` — {len(rs)} artifacts, {sum(r['mb'] for r in rs):.1f} MB", "",
              "| file | kind | variant | decoded | MB | writer | runs |",
              "|---|---|---|---|---|---|---|"]
        for r in rs:
            runs = f"{len(r['runs'])}" if r["runs"] else "—"
            L.append(f"| `{r['file']}` | {r['kind']} | `{r['variant'] or '—'}` | "
                     f"{decode(r['variant'])} | {r['mb']:.1f} | `{r['writer']}` | {runs} |")
        L.append("")

    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    text = build()
    if args.check:
        cur = MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else ""
        if cur != text:
            print("⛔ verified_mlir/MANIFEST.md is stale — re-run scripts/gen_mlir_manifest.py")
            return 1
        print(f"manifest OK ({len(text.splitlines())} lines)")
        return 0
    MANIFEST.write_text(text, encoding="utf-8")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")
    print("ℹ the func.func-symbol invariant is owned by scripts/regen_verified_mlir.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
