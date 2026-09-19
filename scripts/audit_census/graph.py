"""Declaration-level use graph for the audit-only census.

Union of two evidence sources, so errors lean toward "used":
  * env: exact constant references from the elaborated environment (decls.tsv, Dump.lean);
  * tok: identifier tokens over every tracked .lean file, comments stripped. A token counts
    where the env cannot see it: outside any lib declaration (commands such as #guard /
    #eval / example, and non-lib files: tests, apps, demos, IRPrint, SDPFull, ...), or
    inside a lib declaration whose elaborated term does not reference the name, when that
    short name is unambiguous (a mention the term dropped, e.g. an unused simp lemma).
Audit files (tests/AuditAxioms*.lean) never count as users.
"""
import re, subprocess, sys, json, pickle, os
from collections import defaultdict
from pathlib import Path

DATA = Path(os.environ.get("CENSUS_DIR", "/tmp/audit_census"))
REPO = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True,
                           text=True).stdout.strip())
AUDIT_FILES = {"tests/AuditAxioms.lean", "tests/AuditAxiomsHeavy.lean"}

def mod_to_file(m):
    return m.replace(".", "/") + ".lean"

# ---------------------------------------------------------------- env dump
def load_decls():
    consts = {}   # name -> dict(kind, module, file, lo, hi, uses)
    for line in open(DATA / "decls.tsv", encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        kind, name, mod, lo, hi = parts[:5]
        uses = parts[5].split(" ") if len(parts) > 5 and parts[5] else []
        consts[name] = dict(kind=kind, module=mod, file=mod_to_file(mod),
                            lo=None if lo == "-" else int(lo),
                            hi=None if hi == "-" else int(hi), uses=uses)
    return consts

def split_name(n):
    # components, respecting «...»
    out, cur, i, depth = [], "", 0, 0
    while i < len(n):
        c = n[i]
        if c == "«": depth += 1
        elif c == "»": depth -= 1
        if c == "." and depth == 0:
            out.append(cur); cur = ""
        else:
            cur += c
        i += 1
    out.append(cur)
    return out

def owner_of(name, consts, cache={}):
    if name in cache: return cache[name]
    comps = split_name(name)
    res = None
    for k in range(len(comps), 0, -1):
        cand = ".".join(comps[:k])
        c = consts.get(cand)
        if c is not None and c["lo"] is not None:
            res = cand; break
    cache[name] = res
    return res

def user_name(n):
    # _private.LeanMlir.X.Y.0.Proofs.foo -> Proofs.foo
    if n.startswith("_private."):
        comps = split_name(n)
        if "0" in comps:
            i = comps.index("0")
            return ".".join(comps[i+1:])
    return n

# ---------------------------------------------------------------- lexer
TOK_RE = re.compile(r"(?:«[^»\n]*»|[^\W\d][\w'!?]*)(?:\.(?:«[^»\n]*»|[\w'!?]+))*")

def strip_comments(src):
    """Replace comments with spaces (newlines kept); strings kept verbatim."""
    out = []
    i, n = 0, len(src)
    special = re.compile(r'--|/-|"|\'')
    while i < n:
        m = special.search(src, i)
        if not m:
            out.append(src[i:]); break
        j = m.start()
        out.append(src[i:j])
        t = m.group()
        if t == "--":
            k = src.find("\n", j)
            if k < 0: k = n
            out.append(" " * (k - j)); i = k
        elif t == "/-":
            depth, k = 1, j + 2
            while k < n and depth:
                if src.startswith("/-", k): depth += 1; k += 2
                elif src.startswith("-/", k): depth -= 1; k += 2
                else: k += 1
            seg = src[j:k]
            out.append("".join("\n" if ch == "\n" else " " for ch in seg)); i = k
        elif t == '"':
            k = j + 1
            while k < n and src[k] != '"':
                k += 2 if src[k] == "\\" else 1
            out.append(src[j:k+1]); i = k + 1
        else:  # ' : char literal only if not part of an identifier
            prev = src[j-1] if j > 0 else " "
            if re.match(r"[\w'!?₀-₉]", prev):
                out.append("'"); i = j + 1
            else:
                mm = re.match(r"'(\\.[^']*|[^'\\])'", src[j:j+12])
                if mm:
                    out.append(mm.group()); i = j + len(mm.group())
                else:
                    out.append("'"); i = j + 1
    return "".join(out)

DECL_KW = re.compile(r"\b(theorem|lemma|def|abbrev|instance|structure|class|inductive|opaque|axiom)\s+$")

def scan_tokens(files):
    """comp -> list of (file, line, is_decl_header)."""
    occ = defaultdict(list)
    for f in files:
        try:
            src = (REPO / f).read_text(encoding="utf-8")
        except Exception:
            continue
        code = strip_comments(src)
        lines = code.split("\n")
        for ln, text in enumerate(lines, 1):
            for m in TOK_RE.finditer(text):
                tok = m.group()
                hdr = bool(DECL_KW.search(text[:m.start()]))
                seen = set()
                for c in split_name(tok):
                    c = c.strip("«»")
                    if c and c not in seen:
                        seen.add(c)
                        occ[c].append((f, ln, hdr))
    return occ

# ---------------------------------------------------------------- main build
def build():
    consts = load_decls()
    # user-written declarations (have a range), keyed by name
    decls = {n: c for n, c in consts.items() if c["lo"] is not None}
    # per-file interval index for enclosing-declaration lookup (innermost = shortest)
    by_file = defaultdict(list)
    for n, c in decls.items():
        by_file[c["file"]].append((c["lo"], c["hi"], n))
    paint = {}
    for f, ivs in by_file.items():
        mx = max(hi for _, hi, _ in ivs)
        arr = [None] * (mx + 2)
        for lo, hi, n in sorted(ivs, key=lambda t: -(t[1] - t[0])):
            for k in range(lo, hi + 1):
                arr[k] = n
        paint[f] = arr
    def enclosing(f, ln):
        arr = paint.get(f)
        if arr is None or ln >= len(arr): return None
        return arr[ln]

    # env edges between owners
    env_users = defaultdict(set)   # d -> {owner using d}
    env_uses = defaultdict(set)
    for n, c in consts.items():
        o = owner_of(n, consts)
        if o is None: continue
        for d in c["uses"]:
            od = owner_of(d, consts)
            if od is None or od == o: continue
            env_users[od].add(o); env_uses[o].add(od)

    files = subprocess.run(["git", "ls-files", "*.lean"], cwd=REPO, capture_output=True,
                           text=True).stdout.split()
    files = [f for f in files if f not in AUDIT_FILES]
    occ = scan_tokens(files)

    # short-name multiplicity among user-written decls
    short = defaultdict(list)
    for n in decls:
        short[split_name(user_name(n))[-1]].append(n)

    lib_files = {c["file"] for c in decls.values()}
    tok_users = defaultdict(set)   # d -> {"decl:<name>" | "cmd:<file>:<line>" | "file:<file>"}
    for n, c in decls.items():
        s = split_name(user_name(n))[-1]
        unique = len(short[s]) == 1
        for (f, ln, hdr) in occ.get(s, ()):
            if f == c["file"] and c["lo"] <= ln <= c["hi"]:
                continue                      # its own declaration
            if hdr and s in short and any(consts[h]["file"] == f and consts[h]["lo"] == ln
                                          for h in short[s]):
                continue                      # a homonym's own header
            if f in lib_files:
                e = enclosing(f, ln)
                if e is None:
                    tok_users[n].add(f"cmd:{f}:{ln}")
                elif e == n:
                    continue
                elif e in env_users[n]:
                    continue                  # already an env edge
                elif unique:
                    tok_users[n].add(f"mention:{e}")
            else:
                tok_users[n].add(f"file:{f}")
    return consts, decls, env_users, env_uses, tok_users, short

def load_pins(consts):
    pins, unresolved = [], []
    for ln, line in enumerate((REPO / "tests/AuditAxioms.lean").read_text().splitlines(), 1):
        m = re.match(r"#print axioms\s+(\S+)", line)
        if not m: continue
        raw = m.group(1)
        cands = [raw, "Proofs." + raw]
        hit = next((c for c in cands if c in consts), None)
        if hit is None:
            unresolved.append(raw)
        else:
            pins.append((hit, ln))
    return pins, unresolved

if __name__ == "__main__":
    consts, decls, env_users, env_uses, tok_users, short = build()
    pins, unresolved = load_pins(consts)
    with open(DATA / "graph.pkl", "wb") as fh:
        pickle.dump(dict(consts=consts, env_users=dict(env_users), env_uses=dict(env_uses),
                         tok_users=dict(tok_users), pins=pins), fh)
    print("decls", len(decls), "pins", len(pins), "unresolved", unresolved[:20], len(unresolved))
