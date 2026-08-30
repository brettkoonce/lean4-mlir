import LeanMlir.Types
import LeanMlir.Spec
import Jax.Codegen
/-! Runner: find Python, generate script, execute training. -/

def findPython : IO String := do
  let r ← IO.Process.output { cmd := "test", args := #["-f", ".venv/bin/python3"] }
  if r.exitCode == 0 then return ".venv/bin/python3"
  return "python3"

def runJax (spec : NetSpec) (cfg : TrainConfig) (ds : DatasetKind) (dataDir scriptName : String) : IO Unit := do
  IO.println s!"Lean 4 → JAX  {spec.name}"
  IO.println s!"  arch:   {spec.archStr}"
  IO.println s!"  params: {spec.totalParams}"
  IO.println s!"  data:   {dataDir}"
  IO.println ""

  let code := JaxCodegen.generate spec cfg ds dataDir
  let scriptPath := ".lake/build/" ++ scriptName
  IO.FS.createDirAll ".lake/build"
  IO.FS.writeFile scriptPath code
  IO.println s!"Generated: {scriptPath} ({code.length} chars)"

  -- ⭐ `LEAN_MLIR_EMIT_ONLY=1` stops here, before spawning python. Emitting and TRAINING were
  -- one indivisible action, which meant the only way to ask "is the committed artifact still
  -- what this source emits?" was to start a training run — so nobody asked, and
  -- `generated_mobilenet_v2_imagenet_full.py` sat three weeks behind its own source with
  -- labelSmoothing 0.1 after the source said 0.0. That artifact is what a 45 h published run
  -- trained on. Checked here rather than in `runRecipeMain` so EVERY caller gets it, including
  -- the mains that call `runJax` directly. See scripts/regen_jax_generated.sh.
  if (← IO.getEnv "LEAN_MLIR_EMIT_ONLY").isSome then
    IO.println s!"[emit-only] wrote {scriptPath}; NOT training."
    return

  IO.println "Running JAX training...\n"

  let python ← findPython
  let child ← IO.Process.spawn {
    cmd := python
    args := #[scriptPath]
    stdout := .piped
    stderr := .piped
    stdin  := .null
  }

  let stdout ← child.stdout.readToEnd
  IO.print stdout

  let stderr ← child.stderr.readToEnd
  let exitCode ← child.wait
  if exitCode != 0 then
    IO.eprintln s!"\nJAX process exited with code {exitCode}"
    IO.eprintln stderr

/-- A named training recipe: a `TrainConfig`, its generated-file name, and a
    one-line description. Selected by a positional CLI arg (see `runRecipeMain`). -/
structure Recipe where
  name : String
  cfg  : TrainConfig
  out  : String
  desc : String

/-- Shared CLI entry point for the ImageNet trainers. A positional `<recipe>` arg
    (matched against `recipes`; default `"default"`) picks the config, an optional
    `[data_dir]` (default `data/imagenet`) picks the dataset, and `--help`/`-h`
    lists the recipes. One uniform interface across every net. -/
def runRecipeMain (exe : String) (spec : NetSpec) (ds : DatasetKind)
    (recipes : List Recipe) (args : List String) : IO Unit := do
  if args.any (fun a => a == "--help" || a == "-h") then
    IO.println s!"usage: {exe} [recipe] [data_dir] [--shim|--emit]\n"
    IO.println "  --shim   write this recipe's ImageNet batch shim and stop"
    IO.println "  --emit   write this recipe's trainer and stop (do NOT run it)\n"
    IO.println "recipes (default if omitted: \"default\"):"
    let width := (recipes.map (·.name.length)).foldl Nat.max 0 + 2
    for r in recipes do
      let pad := String.ofList (List.replicate (width - r.name.length) ' ')
      IO.println s!"  {r.name}{pad}{r.desc}"
    IO.println "\ndata_dir defaults to \"data/imagenet\"."
    return
  -- Recipe: a CLI arg matching a known recipe wins; else "default". The first
  -- remaining non-flag arg is the data dir.
  let name := (args.find? (fun a => recipes.any (·.name == a))).getD "default"
  match recipes.find? (·.name == name) with
  | none   => IO.eprintln s!"unknown recipe '{name}' — run with --help for the list"
  | some r =>
    let dataDir := (args.filter (fun a => a != r.name && !a.startsWith "-")).head?
                     |>.getD "data/imagenet"
    IO.println s!"[{exe}] recipe '{r.name}': {r.desc}"
    -- `--shim`: emit the ImageNet batch shim for THIS recipe instead of generating and running the
    -- trainer. Same `spec`, same `r.cfg`, so the augmentation the verified path consumes is
    -- provably the augmentation this recipe trains on — one writer, no drift. See
    -- `JaxCodegen.generateShim`.
    if args.any (fun a => a == "--shim") then
      let out := (if r.out.endsWith ".py" then r.out.dropRight 3 else r.out) ++ "_shim.py"
      let code := JaxCodegen.generateShim spec r.cfg
      IO.FS.createDirAll ".lake/build"
      IO.FS.writeFile (".lake/build/" ++ out) code
      IO.println s!"[{exe}]   -> .lake/build/{out}  (SHIM, {code.length} chars; emits data only)"
      return
    -- `--emit`: write the trainer and STOP, the exact counterpart of `--shim` above. ⚠ It exists
    -- because `runJax` writes the script and then immediately spawns python on it, so the only way
    -- to refresh a `generated_*.py` after a spec change was to start a real ImageNet run and race a
    -- timeout against it. That is a bad way to regenerate a file, and it is how a regeneration
    -- would end up writing a checkpoint nobody asked for. Same `spec`, same `r.cfg`, byte-identical
    -- output to what `runJax` would have written — it is the same `JaxCodegen.generate` call.
    if args.any (fun a => a == "--emit") then
      let code := JaxCodegen.generate spec r.cfg ds dataDir
      IO.FS.createDirAll ".lake/build"
      IO.FS.writeFile (".lake/build/" ++ r.out) code
      IO.println s!"[{exe}]   -> .lake/build/{r.out}  (EMIT ONLY, {code.length} chars; not run)"
      return
    IO.println s!"[{exe}]   -> {r.out}  (data: {dataDir})"
    runJax spec r.cfg ds dataDir r.out
