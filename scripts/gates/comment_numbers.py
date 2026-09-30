#!/usr/bin/env python3
"""No numbers that nothing re-checks in comments (planning/rubric_review.md, decision 6).

Code comments and docstrings do not carry measured results (accuracy %, "N/100 certified",
timings, memory sizes, speedups, mIoU / mAP / Dice), counts of the codebase ("185 roots", "~250 lines"), `file:line`
references or commit hashes. Nothing re-checks them, so they go stale silently. A comment
names the source instead: the run directory, the theorem (which docstring-checkrefs
resolves) or the generator. Measured numbers live in run READMEs and the book, whose
ledgers are audited.

Numbers that ARE the specification stay (224, ε, the 0.1 smoothing, "50% per image"), as
does a count the adjacent statement proves. Those are listed in
`comment_numbers_allow.tsv` beside this script, one `path<TAB>match<TAB>reason` row per
occurrence. A row is keyed by (path, matched text), not by line, so edits that move a
line do not break it. `reason` is one of:
  spec       the number is part of what the code does or is stated at;
  statement  an adjacent theorem states it;
  citation   a published figure, attributed to its paper in the same comment;
  example    illustrative arithmetic, not a claim about this tree or a run.

Scanned: comments and docstrings of every tracked `.lean` file outside planning/ and
historical/ (lakefile.lean included), the `#` comments of formalization.yaml, and the
step-summary lines of .github/workflows/*.yml.

  python3 scripts/gates/comment_numbers.py          # check; exit 1 on a hit not allowed
  python3 scripts/gates/comment_numbers.py --list   # print every hit, allowed or not
"""

import collections
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOW = Path(__file__).resolve().parent / "comment_numbers_allow.tsv"
REASONS = {"spec", "statement", "citation", "example"}

PATTERNS = {
    "n/100": re.compile(r"\b\d+\s*/\s*100\b"),
    "percent": re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    "count": re.compile(
        r"\b\d[\d,]*\s+(?:roots|modules|lines|call sites|files|theorems|lemmas|"
        r"declarations|artifacts|exes|binaries|variants|targets|proof modules)\b"),
    "file:line": re.compile(r"[\w./-]+\.(?:lean|py|tex|md|c|h|yml|yaml|sh|mlir):\d+"),
    # 7–12 hex digits with at least one letter and one digit, not inside a word or 0x….
    "hash": re.compile(r"(?<![\w#])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,12}(?!\w)"),
    # Timings. A bare `h` counts only after a decimal or a multi-digit number, or with a space
    # (`7.8h`, `71 h`), so shape variables like `2h × 2w` do not match.
    "time": re.compile(
        r"(?:\b\d+(?:\.\d+)?\s?(?:ms|µs|ns|s/epoch|ms/step|s/step|min/epoch|sec|seconds|min|"
        r"minutes|hr|hrs|hours)\b|\b(?:\d+\.\d+|\d{2,})\s?h\b|\b\d+ h\b|\b\d+h\d+m\b)"),
    "size": re.compile(r"\b\d+(?:\.\d+)?\s?(?:GiB|MiB|KiB|TiB|GB|MB|KB|kB|TB)\b"),
    # Speedups and ratios: a decimal factor (`1.78×`), or an integer one with a comparative word.
    # Structural counts (`3× block`, `8 × unetUp`, `2h × 2w`) do not match.
    "speedup": re.compile(
        r"\b\d+\.\d+\s?[×x](?!\s?\d)|\b\d+\s?[×x]\s+(?:faster|slower|speedup|over|more|less|"
        r"fewer|noisier|larger|smaller|cheaper)\b"),
    "metric": re.compile(
        r"\b(?:mIoU|IoU|Dice|mAP(?:@[\d.]+)?|top-?[15]|acc(?:uracy)?|F1|AUC|PSNR|FID)\s*"
        r"(?:of|=|:|≈|~)?\s*\d*\.\d+"),
}


def lean_comments(text: str):
    """Yield (line, text) for every line of a `--` comment or a (nested) `/- … -/` block,
    docstrings included. String literals are skipped, so printed text is not scanned."""
    i, n, line = 0, len(text), 1
    while i < n:
        if text.startswith("--", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            yield line, text[i:j]
            i = j
        elif text.startswith("/-", i):
            start, first, depth = i, line, 0
            while i < n:
                if text.startswith("/-", i):
                    depth += 1
                    i += 2
                elif text.startswith("-/", i):
                    depth -= 1
                    i += 2
                    if depth == 0:
                        break
                else:
                    line += text[i] == "\n"
                    i += 1
            for k, seg in enumerate(text[start:i].split("\n")):
                yield first + k, seg
        elif text[i] == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            line += text[i:j].count("\n")
            i = j + 1
        elif text[i] == "'" and i + 2 < n and text[i + 2] == "'":
            i += 3  # a Char literal such as '"' must not open a string
        else:
            line += text[i] == "\n"
            i += 1


def yaml_comments(text: str):
    """`#` comments, whole-line or trailing."""
    for k, raw in enumerate(text.split("\n"), 1):
        s = raw.strip()
        if s.startswith("#"):
            yield k, s
            continue
        m = re.search(r"\s#\s", raw)
        if m and raw.count('"', 0, m.start()) % 2 == 0 and raw.count("'", 0, m.start()) % 2 == 0:
            yield k, raw[m.start():]


def step_summaries(text: str):
    """The lines that write a workflow's step summary (`echo …` inside a
    `{ … } >> $GITHUB_STEP_SUMMARY` group, or on a line that appends to it). A workflow's `#`
    comments are its incident history and are not scanned."""
    in_summary = False
    for k, raw in enumerate(text.split("\n"), 1):
        s = raw.strip()
        if s == "{":
            in_summary = True
        if (in_summary or "GITHUB_STEP_SUMMARY" in s) and s.startswith(("echo", "printf")):
            yield k, s
        if s.startswith("}"):
            in_summary = False


def sources():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split("\n")
    for p in tracked:
        if p.endswith(".lean") and not p.startswith(("planning/", "historical/")):
            yield p, lean_comments
        elif p == "formalization.yaml":
            yield p, yaml_comments
        elif p.startswith(".github/workflows/") and p.endswith(".yml"):
            yield p, step_summaries


def hits():
    for path, scan in sources():
        f = ROOT / path
        if not f.exists():
            continue
        for line, text in scan(f.read_text(errors="replace")):
            for kind, pat in PATTERNS.items():
                for m in pat.finditer(text):
                    yield path, line, kind, m.group(0).strip(), text.strip()


def load_allow():
    allow, bad = collections.Counter(), []
    if not ALLOW.exists():
        return allow, bad
    for k, row in enumerate(ALLOW.read_text().split("\n"), 1):
        if not row.strip() or row.startswith("#"):
            continue
        cols = row.split("\t")
        if len(cols) != 3 or cols[2] not in REASONS:
            bad.append(f"{ALLOW.name}:{k}: want path<TAB>match<TAB>{'|'.join(sorted(REASONS))}")
            continue
        allow[(cols[0], cols[1])] += 1
    return allow, bad


def main() -> int:
    listing = "--list" in sys.argv[1:]
    allow, bad = load_allow()
    for b in bad:
        print(f"✗ {b}")
    found = list(hits())
    seen, flagged = collections.Counter(), []
    for path, line, kind, match, text in found:
        seen[(path, match)] += 1
        ok = seen[(path, match)] <= allow[(path, match)]
        if listing:
            print(f"{'allowed' if ok else 'HIT'}\t{kind}\t{path}:{line}\t{match}\t{text[:120]}")
        elif not ok:
            flagged.append(f"✗ {path}:{line}: {kind} `{match}` in: {text[:120]}")
    stale = [f"✗ {ALLOW.name}: `{p}` / `{m}` allowed ×{c} but found ×{seen[(p, m)]}"
             for (p, m), c in sorted(allow.items()) if seen[(p, m)] < c]
    if listing:
        return 0
    for msg in flagged + stale:
        print(msg)
    if flagged or stale or bad:
        print(f"\n{len(flagged)} unchecked number(s) in comments, {len(stale)} stale allow row(s). "
              "Name the source (run dir, theorem, generator) instead of the number; a spec "
              f"constant or statement-backed count goes in {ALLOW.relative_to(ROOT)}.")
        return 1
    print(f"✓ comment numbers: {len(found)} numbers in comments, every one allowed "
          f"({sum(allow.values())} allow rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
