import Lean
open Lean

/-! Dump every project constant: kind, name, module, declaration range, project constants used.
    `lake env lean --run scripts/audit_census/Dump.lean <modules.txt> <decls.tsv>` (~2 min, ~7 GB).
    Driven by `scripts/audit_census/run.sh`. -/

def kindOf : ConstantInfo → String
  | .thmInfo _ => "thm" | .defnInfo _ => "def" | .axiomInfo _ => "axiom"
  | .opaqueInfo _ => "opaque" | .inductInfo _ => "induct" | .ctorInfo _ => "ctor"
  | .recInfo _ => "rec" | .quotInfo _ => "quot"

unsafe def main (args : List String) : IO Unit := do
  let modsFile := args[0]!
  let outFile := args[1]!
  initSearchPath (← findSysroot)
  let mods := (← IO.FS.lines modsFile).filter (· ≠ "")
  enableInitializersExecution
  let env ← importModules (mods.map fun m => { module := m.toName }) {} (loadExts := true)
  let isProj (n : Name) : Option Name := do
    let idx ← env.getModuleIdxFor? n
    let m := env.header.moduleNames[idx.toNat]!
    if (`LeanMlir).isPrefixOf m || m == `LeanMlir then some m else none
  let h ← IO.FS.Handle.mk outFile .write
  let ctx : Core.Context := { fileName := "<dump>", fileMap := default, options := {} }
  let st : Core.State := { env }
  let mut n := 0
  for (c, ci) in env.constants.toList do
    let some m := isProj c | continue
    let rng ← (do
        let r? ← findDeclarationRanges? c
        pure (match r? with
          | some r => s!"{r.range.pos.line}\t{r.range.endPos.line}"
          | none => "-\t-") : CoreM String).toIO ctx st
    let rng := rng.1
    let uses := ci.getUsedConstantsAsSet.toList.filter (fun d => d != c && (isProj d).isSome)
    let usesS := " ".intercalate (uses.map toString)
    h.putStrLn s!"{kindOf ci}\t{c}\t{m}\t{rng}\t{usesS}"
    n := n + 1
  IO.println s!"dumped {n} constants"
