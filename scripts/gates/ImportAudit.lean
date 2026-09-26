import Lean

/-! # Import audit — the modules each module's constants need, read from the .oleans

`lake env lean --run scripts/gates/ImportAudit.lean <Module> …` loads the named modules and prints,
for each, one line `Module<TAB>Req₁ Req₂ …`: the modules (project or Mathlib, `Init` aside) that
define a constant some constant of `Module` uses. This is `Lean.Environment.requiredModules` (ImportGraph's
`#min_imports`) computed for every module at once from the compiled `.olean`s, rather than by
re-elaborating each file.

It sees constants only. A tactic, a notation, an `#eval` or an `example` body leaves no constant
behind, so an import this marks unneeded is a CANDIDATE, confirmed only when the file compiles
without it. `scripts/gates/import_audit.py unused` does that bookkeeping; run it, not this. -/

open Lean

def main (args : List String) : IO UInt32 := do
  let mods := args.map String.toName
  if mods.isEmpty then
    IO.eprintln "usage: lake env lean --run scripts/gates/ImportAudit.lean <Module> …"
    return 2
  initSearchPath (← findSysroot)
  let env ← importModules (mods.map ({ module := · })).toArray {} (trustLevel := 1024)
  for m in mods do
    let some idx := env.getModuleIdx? m
      | IO.eprintln s!"not loaded: {m}"; return 1
    let data := env.header.moduleData[idx.toNat]!
    let mut req : NameSet := {}
    for ci in data.constants do
      for n in ci.getUsedConstantsAsSet do
        if let some j := env.getModuleIdxFor? n then
          let r := env.header.moduleNames[j.toNat]!
          if r != m && !(`Init).isPrefixOf r then req := req.insert r
    IO.println s!"{m}\t{" ".intercalate (req.toList.map toString)}"
  return 0
