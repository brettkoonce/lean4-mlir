#!/usr/bin/env python3
"""An exact-GELU render is its tanh twin with the GELU sites swapped, and nothing else.

Every ImageNet ViT and ConvNeXt artifact in `verified_mlir/` exists at both forms of the GELU
(`GeluForm`): the tanh approximation the landed runs trained under, and the exact `x · Φ(x)` of
DeiT and ConvNeXt, whose names carry `erf` (`geluMarker`: after `drop` or the ε marker, before
`bf16`; `<slug>_erf_fwd` for a forward with no train-step variant in its name). The two writers of
a pair are one call at `.tanh` and at `.erf`, so the two files must be the same graph outside the
GELU sites. This gate checks the committed bytes for that:

  * every artifact of the six slugs has its twin at the other form;
  * the tanh file holds no `chlo.erfc`, the exact file no `stablehlo.tanh`, and the exact file has
    one `chlo.erfc` for each `stablehlo.tanh` of its twin;
  * with each GELU site collapsed to one placeholder line (`GELU x` / `GELU_BACK x, dy`) and the
    `%v<n>` names renumbered in order of definition, the two files are identical line for line.

The sites are recognised by their emitted shape (`Pretty.lean`: the tanh emit, `geluErfFwdText`,
`geluErfBackText`), so a change to either emit has to be made here too; the gate then fails on
every pair rather than passing.

Usage:  python3 scripts/gates/gelu_form_twins.py            # assert
        python3 scripts/gates/gelu_form_twins.py --control  # prove the gate can fail
"""
import re
import sys
from pathlib import Path

MLIR = Path(__file__).resolve().parents[2] / "verified_mlir"
SLUGS = ("vitin", "vitsin", "vitbin", "convnextin", "convnextsin", "convnextbin")
DEF = re.compile(r"^\s*(%v\d+) = (\S+) (.*)$")
# lines in a site, and where its transcendental sits from the site's first line
TANH_FWD, TANH_BACK, TANH_AT = 13, 23, 7
ERF_FWD, ERF_BACK, ERF_AT = 7, 18, 5


def tanh_stem(stem: str) -> str:
    """The tanh twin's name: the exact name without its one `erf` marker."""
    slug = next(s for s in SLUGS if stem.startswith(s + "_"))
    rest = stem[len(slug) + 1:]
    assert rest.count("erf") == 1, f"{stem}: expected one `erf` marker"
    rest = rest.replace("erf", "")
    return f"{slug}_{rest.lstrip('_')}"


def operands(line: str) -> list[str]:
    return re.findall(r"%[\w#]+", line.split(" = ", 1)[1].split(" : ")[0])


def name(line: str) -> str:
    return DEF.match(line).group(1)


def ty(line: str) -> str:
    return line.rsplit(" : ", 1)[1]


def canon(text: str, stem: str) -> list[str]:
    """`text` with each GELU site collapsed to a placeholder and its `%v<n>` names renumbered."""
    src = text.replace(f"@{stem}", "@ENTRY").split("\n")
    out, i = [], 0
    while i < len(src):
        is_tanh = i + TANH_AT < len(src) and " = stablehlo.tanh " in src[i + TANH_AT]
        is_erf = i + ERF_AT < len(src) and " = chlo.erfc " in src[i + ERF_AT]
        if is_tanh:
            x = operands(src[i])[0]
            t = name(src[i + TANH_AT])
            # the backward squares the tanh five lines on; the forward has ended by then
            back = i + 12 < len(src) and " = stablehlo.multiply " in src[i + 12] \
                and operands(src[i + 12]) == [t, t]
            n = TANH_BACK if back else TANH_FWD
        elif is_erf:
            x = operands(src[i + 1])[1]
            z = name(src[i + 4])
            # the backward squares erfc's argument on the next line; the forward ends there
            back = operands(src[i + 6]) == [z, z]
            n = ERF_BACK if back else ERF_FWD
        else:
            out.append(src[i])
            i += 1
            continue
        last = src[i + n - 1]
        if back:
            dy = operands(last)[0] if is_tanh else operands(src[i + 9])[1]
            out.append(f"    {name(last)} = GELU_BACK {x}, {dy} : {ty(last)}")
        else:
            out.append(f"    {name(last)} = GELU {x} : {ty(last)}")
        i += n
    ids: dict[str, str] = {}
    return [re.sub(r"%v\d+\b", lambda m: ids.setdefault(m.group(0), f"%c{len(ids)}"), l) for l in out]


def compare(tanh_text: str, tanh_name: str, erf_text: str, erf_name: str) -> str | None:
    nt, ne = tanh_text.count("stablehlo.tanh"), erf_text.count("chlo.erfc")
    if "chlo.erfc" in tanh_text:
        return "the tanh render holds a chlo.erfc"
    if "stablehlo.tanh" in erf_text:
        return "the exact render holds a stablehlo.tanh"
    if nt == 0 or nt != ne:
        return f"{nt} tanh site(s) against {ne} erfc site(s)"
    a, b = canon(tanh_text, tanh_name), canon(erf_text, erf_name)
    if sum(" = GELU" in l for l in a) != nt:
        return f"recognised {sum(' = GELU' in l for l in a)} of {nt} tanh sites — the emit moved"
    if sum(" = GELU" in l for l in b) != ne:
        return f"recognised {sum(' = GELU' in l for l in b)} of {ne} erfc sites — the emit moved"
    if a != b:
        k = next((j for j, (p, q) in enumerate(zip(a, b)) if p != q), min(len(a), len(b)))
        return (f"differs outside the GELU sites at canonical line {k}:\n"
                f"        tanh : {a[k][:120] if k < len(a) else '<eof>'}\n"
                f"        exact: {b[k][:120] if k < len(b) else '<eof>'}")
    return None


def pairs() -> tuple[list[tuple[str, str]], list[str]]:
    stems = {p.stem for p in MLIR.glob("*.mlir") if p.stem.startswith(tuple(s + "_" for s in SLUGS))}
    exact = sorted(s for s in stems if "erf" in s)
    twins = [(tanh_stem(e), e) for e in exact]
    lone = sorted(stems - set(exact) - {t for t, _ in twins}) + [e for t, e in twins if t not in stems]
    return [(t, e) for t, e in twins if t in stems], lone


def main() -> int:
    twins, lone = pairs()
    if "--control" in sys.argv:
        # One changed constant outside every GELU site, and a tanh site left in the exact render.
        t, e = next((t, e) for t, e in twins if t.endswith("train_step"))
        tt, et = (MLIR / f"{t}.mlir").read_text(), (MLIR / f"{e}.mlir").read_text()
        moved = et.replace("dense<0.9>", "dense<0.8>", 1)
        assert moved != et, "control: no β₁ constant to move"
        mixed = et.replace("chlo.erfc", "stablehlo.tanh", 1)
        ok = compare(tt, t, et, e) is None and compare(tt, t, moved, e) and compare(tt, t, mixed, e)
        print(f"{'✓' if ok else '✗'} control on {e}: the pair passes, a moved constant and a mixed "
              f"form each fail")
        return 0 if ok else 1
    bad = []
    for t, e in twins:
        why = compare((MLIR / f"{t}.mlir").read_text(), t, (MLIR / f"{e}.mlir").read_text(), e)
        if why:
            bad.append(f"{e}.mlir against {t}.mlir: {why}")
    for s in lone:
        bad.append(f"{s}.mlir has no twin at the other form of the GELU")
    for b in bad:
        print(f"  ✗ {b}")
    print(f"{'✓' if not bad else '✗'} {len(twins) - sum(1 for b in bad if 'against' in b)}/{len(twins)} "
          f"exact-GELU renders equal their tanh twins outside the GELU sites; "
          f"{len(lone)} artifact(s) without a twin")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
