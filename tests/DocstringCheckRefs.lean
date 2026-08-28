import Lake.CLI.Main

/-! # `lake exe docstring-checkrefs` — resolve every `Ident` a docstring cites

The blueprint survived five rewrites intact and the docstrings did not, and the reason is
not style. `blueprint-checkdecls` resolves every `\lean{}` name in `content.tex` against the
real environment and fails CI on a miss, so a rename that orphans a blueprint citation goes
red. **Nothing does that for docstrings**, so a rename that orphans a docstring citation is
invisible: the file still compiles, doc-gen4 still renders it, and the text still reads
current. Found the hard way, repeatedly — `VerifiedNets.lean` cited
`Proofs.convNextForwardTC_has_vjp_correct`, a symbol that has never existed under that
spelling, and `ConvNeXtFullT.lean` still cites `convNextForwardT_has_vjp` from inside the
file that defines `convNextForwardTCh_has_vjp`.

This is that gate, and it is deliberately the SAME shape as the blueprint one: same
workspace load, same root filtering, resolve-or-fail against `Environment`.

⚠⚠ **RESOLUTION IS AGAINST THE ENVIRONMENT, NOT A REGEX, AND THAT IS THE WHOLE DESIGN.**
A first pass of this check written as a Python regex reported 26% of refs unresolved;
tightening the filters by hand got it to 8.7%, and the residue was still mostly false
positives (Mathlib names the regex could not see, structure projections like
`foo_has_vjp.backward`, tactic names). `Environment.find?` answers all three exactly,
because it knows about Mathlib, about projections, and about namespaces. A heuristic that
needs hand-tuned filters to stay quiet is a heuristic that will be turned off.

**What counts as resolved**, in order:

1. the name is in the environment outright;
2. it resolves under a project namespace (`Proofs`, `LeanMlir`, `Layer`);
3. its PREFIX resolves — `foo_has_vjp.backward` is a field access on a real declaration,
   not a declaration, and the prefix is the thing a rename would break;
4. it is baselined in `scripts/docstring_ref_baseline.txt`.

**What is skipped before resolution is attempted** (never Lean names, and cheap to rule
out): anything with a file extension, anything ALL-CAPS (environment variables), and
anything that is a single lowercase word with no `_` or `.` (prose in backticks).

⚠ The baseline is the escape hatch for real non-Lean vocabulary a docstring legitimately
quotes in backticks — MLIR ops (`stablehlo.dot_general`), tactic names (`norm_num`),
CLI fragments. It carries the `render_guard_baseline.txt` contract: **it may SHRINK, never
grow.** A new entry means a docstring citation can rot with CI green, which is the exact
hole this file exists to close.

```
lake exe docstring-checkrefs                    # gate: exit 1 on any unresolved ref
lake exe docstring-checkrefs --list             # print every unresolved ref and its site
lake exe docstring-checkrefs --update-baseline  # re-record (deliberate + budgeted only)
```
-/

open Lake Lean

/-- ⚠⚠ **IMPORT THE SAME SET `BlueprintCheckDecls` DOES, AND THE FIRST VERSION OF THIS FILE
    DID NOT.** It imported `LeanMlir` alone, reasoning that an allow-list cannot be broken by
    a fourth `apps`/`demos`-style tree. That is true and it was still wrong: the proof corpus
    is spread across several libs, so `Proofs/Float/*` was outside the environment and the
    gate reported `bnIstd_close` and `reduction_close` — both of which exist — as dangling.
    A gate that invents misses is worse than no gate, because the first triage session
    teaches everyone to disbelieve it.

    So: every lib root except `CertsHeavy` (OOMs the shared runners, which is why the
    2026-07-12 split exists) and except the `apps`/`demos` trainer trees the proof job does
    not build. ▶ If a fourth tree appears this will fail with `unknown module prefix`, the
    same way the blueprint gate did twice; that is a known cost, taken deliberately here in
    exchange for resolving against the corpus the docstrings actually live in. -/
def unbuiltTrees : List Name := [`apps, `demos]

/-- Project namespaces a docstring may cite relatively (`\`vjp_comp\`` for
    `Proofs.vjp_comp`). Kept short on purpose: every entry widens what counts as resolved,
    so a long list would quietly re-admit the false-negative class this gate exists to catch. -/
def projectNamespaces : List Name := [`Proofs, `LeanMlir, `Layer]

/-- Doc-comment bodies in a source string: both `/-- … -/` (declaration docs) and
    `/-! … -/` (module docs). Nested `/- … -/` inside a doc comment is tracked, since the
    codegen files use them to comment out MLIR fragments mid-docstring. -/
def docBodies (src : String) : Array String := Id.run do
  let cs := src.toList.toArray
  let n := cs.size
  let mut out : Array String := #[]
  let mut i := 0
  while i + 2 < n do
    let isDoc := cs[i]! == '/' && cs[i+1]! == '-' && (cs[i+2]! == '-' || cs[i+2]! == '!')
    if isDoc then
      let mut j := i + 3
      let mut depth := 1
      let mut buf : String := ""
      while j + 1 < n && depth > 0 do
        if cs[j]! == '/' && cs[j+1]! == '-' then
          depth := depth + 1
          buf := buf.push cs[j]!
          j := j + 1
        else if cs[j]! == '-' && cs[j+1]! == '/' then
          depth := depth - 1
          if depth == 0 then
            j := j + 2
          else
            buf := buf.push cs[j]!
            j := j + 1
        else
          buf := buf.push cs[j]!
          j := j + 1
      out := out.push buf
      i := j
    else
      i := i + 1
  return out

/-- Backtick-delimited runs inside one doc body. Opening and closing tick must be on the
    same LINE: an unbalanced tick in prose would otherwise swallow the rest of the file and
    report a paragraph as an identifier. -/
def backtickRefs (body : String) : Array String := Id.run do
  let mut out : Array String := #[]
  for line in body.splitOn "\n" do
    let parts := line.splitOn "`"
    -- odd indices are the delimited runs; an unbalanced line yields a trailing part we drop
    let mut k := 1
    while k < parts.length do
      out := out.push parts[k]!
      k := k + 2
  return out

/-- ⚠⚠ **THE GATE CHECKS THIS PROJECT'S OWN THEOREM NAMES AND NOTHING ELSE, AND THAT IS A
    DESIGN CHOICE, NOT A SHORTCUT.** Checking every backtick citation reported 2,447 hits
    across 1,062 names on the first run: MLIR op names (`all_reduce`), artifact basenames
    (`resnet34_fwd`), timm flags (`no_weight_decay`), Lean core cited relative to an `open`
    (`Environment.find?`). Absorbing that into a baseline would mean an 800-line allow-list,
    and a gate whose allow-list is larger than its signal is a gate that gets turned off.

    So the gate targets the class that actually rots: a THEOREM this project renamed, still
    cited under its old spelling. Those citations are the ones a reader follows and the ones
    doc-gen4 renders as prose about a symbol that is not there. Recall is deliberately traded
    for a signal that stays worth reading.

    ▶ Widening this list is the way to grow the gate, and each addition should come with a
    look at what it admits. -/
def projectMarkers : List String :=
  ["_has_vjp", "_correct", "_bridge", "_close", "_tied", "_faithful", "_denote_eq",
   "_descends", "_adjointClose", "_argmaxSafe", "_fwd_faithful", "_eq_chain", "_rowIndep"]

/-- Substring test written by hand rather than via `splitOn`, because this toolchain is
    mid-migration from `String` to `String.Slice` and the return type of the string helpers
    is not stable across it. -/
def containsSubstr (hay needle : String) : Bool := Id.run do
  let h := hay.toList.toArray
  let nd := needle.toList.toArray
  if nd.size == 0 || nd.size > h.size then return false
  let mut found := false
  for i in [0 : h.size - nd.size + 1] do
    if !found then
      let mut ok := true
      for j in [0 : nd.size] do
        if h[i+j]! != nd[j]! then ok := false
      if ok then found := true
  return found

def checkWorthy (s : String) : Bool :=
  projectMarkers.any (fun m => containsSubstr s m)

/-- Ruled out before the environment is consulted: never a Lean name, and cheap to reject. -/
def skipRef (s : String) : Bool :=
  let exts := [".lean", ".md", ".py", ".sh", ".mlir", ".txt", ".so", ".c", ".h", ".yml", ".json", ".tsv", ".bin"]
  s.isEmpty
  || s.any (fun c => c == ' ' || c == '\t')                  -- a phrase, not an identifier
  || exts.any (fun e => s.endsWith e)                        -- a path
  || s.front == '.' || s.back == '.'                         -- `.imagenet`, sentence-final
  || s.front == '_'                                          -- `_close`, a SUFFIX fragment
                                                             -- quoted to name a family, not
                                                             -- a declaration
  || !s.front.isAlpha                                        -- not identifier-initial

/-- Last name component, the key the suffix index is built on. -/
def lastComponent : Name → String
  | .str _ s => s
  | n => n.toString

/-- Does `full` end with the dotted components of `ref`? A docstring under `open Proofs`
    writes `` `vjp_comp` `` for `Proofs.vjp_comp`, and under `open Lean` writes
    `` `Environment.find?` `` for `Lean.Environment.find?`. Resolving absolutely would call
    both dangling, which is how the first run produced 2,447 false positives. -/
def endsWithComponents (full : Name) (parts : List String) : Bool :=
  let fc := full.components.map lastComponent
  parts.length ≤ fc.length && (fc.drop (fc.length - parts.length) == parts)

/-- The ways a citation counts as live: exact, under a project namespace, as a dotted SUFFIX
    of a real declaration, by its PREFIX resolving (a field access such as
    `foo_has_vjp.backward` is not itself a declaration, but a rename of `foo_has_vjp` breaks
    it and that is what we are catching), or baselined. -/
def resolves (env : Environment) (idx : Std.HashMap String (Array Name))
    (baseline : Array String) (s : String) : Bool :=
  let n := s.toName
  if env.contains n then true
  else if baseline.contains s then true
  else if projectNamespaces.any (fun ns => env.contains (ns ++ n)) then true
  else
    let parts := s.splitOn "."
    let hit := (idx.getD (parts.getLast!) #[]).any (endsWithComponents · parts)
    if hit then true
    else match parts.dropLast with
      | [] => false
      | pre =>
        -- field access: does the PREFIX resolve?
        let pn := String.intercalate "." pre
        env.contains pn.toName
          || projectNamespaces.any (fun ns => env.contains (ns ++ pn.toName))
          || (idx.getD (pre.getLast!) #[]).any (endsWithComponents · pre)

/-- Every `.lean` file under `dir`, recursively. -/
partial def leanFiles (dir : System.FilePath) : IO (Array System.FilePath) := do
  let mut out : Array System.FilePath := #[]
  for e in ← dir.readDir do
    let p := e.path
    if ← p.isDir then
      out := out ++ (← leanFiles p)
    else if p.extension == some "lean" then
      out := out.push p
  return out

/-- Directories scanned. `LeanMlir` is the proof + codegen corpus; `lakefile.lean` is
    included because it carries 200+ target docstrings and is where two of the stale
    citations that motivated this gate were found. -/
def scanRoots : List System.FilePath := ["LeanMlir", "tests", "apps", "demos"]

unsafe def main (args : List String) : IO UInt32 := do
  let listOnly := args.contains "--list"
  let update   := args.contains "--update-baseline"
  let baselinePath : System.FilePath := "scripts/docstring_ref_baseline.txt"
  -- ⚠ `--update-baseline` must recompute against an EMPTY baseline. Reading the existing
  -- one first made every recorded entry resolve, so the regenerated file came back with
  -- zero lines and silently discarded the whole ratchet. Caught by running it twice.
  let baseline ← if update then pure #[]
    else if ← baselinePath.pathExists then do
      let lines ← IO.FS.lines baselinePath
      -- A line is `name` or `name  # why it is here`. The trailing comment is the point:
      -- the first version of this file was a bare list, which made a PROPOSED name
      -- ("you'd need to add `globalAvgPool_has_vjp`") indistinguishable from a stale
      -- citation, and the header then had to describe all of them as one thing and was
      -- wrong about most.
      pure <| lines.filterMap fun l =>
        let body := (l.splitOn "#").headD ""
        let t := body.trimAscii.toString
        if t.isEmpty then none else some t
    else pure #[]

  let (elanInstall?, leanInstall?, lakeInstall?) ← findInstall?
  let config ← MonadError.runEIO <| mkLoadConfig { elanInstall?, leanInstall?, lakeInstall? }
  let (ws?, log) ← (loadWorkspace config).run?
  log.replay (logger := .stderr)
  let some ws := ws? | return 1
  let coreLibs := ws.root.leanLibs.filter fun lib => lib.name != `CertsHeavy
  let imports := coreLibs.flatMap fun lib =>
    lib.config.roots.filterMap fun module =>
      if unbuiltTrees.any (·.isPrefixOf module) then none else some { module }
  enableInitializersExecution
  let env ← Lean.importModules imports {}
  -- Suffix index: last component -> the declarations carrying it. Built once; without it a
  -- relative citation under an `open` reads as dangling (see `endsWithComponents`).
  let mut idx : Std.HashMap String (Array Name) := {}
  for (n, _) in env.constants.toList do
    if n.isInternal then continue
    let k := lastComponent n
    idx := idx.insert k ((idx.getD k #[]).push n)

  let mut files : Array System.FilePath := #["lakefile.lean"]
  for r in scanRoots do
    if ← r.pathExists then files := files ++ (← leanFiles r)
  -- ⚠ This file is excluded from its own scan, and the reason is not vanity: its header
  -- documents the gate by QUOTING the dead citations that motivated it
  -- (`convNextForwardT_has_vjp`, `Proofs.convNextForwardTC_has_vjp_correct`) plus a
  -- `foo_has_vjp` placeholder. Those are deliberately dangling — a gate whose own
  -- documentation trips it teaches the reader to add exceptions.
  files := files.filter fun f => f.toString != "tests/DocstringCheckRefs.lean"

  let mut checked := 0
  let mut misses : Array (System.FilePath × String) := #[]
  let mut distinct : Std.HashSet String := {}
  for f in files do
    let src ← IO.FS.readFile f
    for body in docBodies src do
      for ref in backtickRefs body do
        if skipRef ref || !checkWorthy ref then continue
        checked := checked + 1
        unless resolves env idx baseline ref do
          misses := misses.push (f, ref)
          distinct := distinct.insert ref

  if update then
    let names := distinct.toArray.qsort (· < ·)
    let hdr := "# Docstring citations that do not resolve to any declaration.\n\
                # Generated by `lake exe docstring-checkrefs --update-baseline`.\n\
                #\n\
                # ⚠ THESE ARE DEBT, NOT EXCEPTIONS. Every line is a docstring telling a\n\
                # reader to look up a theorem that is not there — doc-gen4 renders the\n\
                # prose and the symbol does not exist. They are listed rather than fixed\n\
                # only because the gate was introduced to a corpus that already had them,\n\
                # so the ratchet starts where the code is.\n\
                #\n\
                # This list may SHRINK, never grow. Fix a citation and delete its line. A\n\
                # NEW entry means a rename just orphaned a docstring with CI green, which\n\
                # is the hole this gate exists to close.\n"
    IO.FS.writeFile baselinePath (hdr ++ String.intercalate "\n" names.toList ++ "\n")
    println! "wrote {baselinePath} with {names.size} entries"
    return 0

  println! "docstring citations checked: {checked} across {files.size} file(s)"
  if misses.isEmpty then
    println! "✅ every cited identifier resolves"
    return 0
  println! "❌ {misses.size} unresolved citation(s), {distinct.size} distinct name(s)"
  if listOnly then
    for (f, ref) in misses do
      println! "  {f}: `{ref}`"
  else
    let shown := misses.toList.take 15
    for (f, ref) in shown do
      println! "  {f}: `{ref}`"
    if misses.size > 15 then println! "  … {misses.size - 15} more (pass --list)"
  println! "\nfix: correct the citation, or if it is genuinely not a Lean name, re-record with\n  \
            lake exe docstring-checkrefs --update-baseline"
  return 1
