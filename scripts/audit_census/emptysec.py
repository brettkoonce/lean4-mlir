"""    python3 scripts/audit_census/emptysec.py [files...]   (default: files changed vs HEAD)

Report section headers (`-- ═══ / -- § …` blocks, or `/-! ## … -/` docs) followed by nothing
but another header, an `end`, or EOF."""
import re, sys, subprocess
files = sys.argv[1:] or subprocess.run(["git","diff","--name-only","HEAD","--","LeanMlir"],capture_output=True,text=True).stdout.split()
for f in files:
    try: L = open(f, encoding="utf-8").read().split("\n")
    except FileNotFoundError: continue
    i = 0; n = len(L)
    def is_hdr_start(k):
        return L[k].startswith("-- ═") or L[k].startswith("-- §") or (L[k].startswith("/-! ##"))
    while i < n:
        if is_hdr_start(i):
            start = i
            if L[i].startswith("/-!"):
                while i < n and "-/" not in L[i]: i += 1
                i += 1
            else:
                while i < n and L[i].startswith("--"): i += 1
            j = i
            while j < n and not L[j].strip(): j += 1
            nxt = L[j] if j < n else "<EOF>"
            if j >= n or is_hdr_start(j) or re.match(r"^end\b", nxt) or nxt.startswith("namespace "):
                title = next((x for x in L[start:i] if "§" in x or "##" in x), L[start])
                print(f"{f}:{start+1}: {title.strip()[:100]}   → next: {nxt.strip()[:50]}")
            continue
        i += 1
