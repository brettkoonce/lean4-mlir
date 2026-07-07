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
    IO.println s!"usage: {exe} [recipe] [data_dir]\n"
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
    IO.println s!"[{exe}]   -> {r.out}  (data: {dataDir})"
    runJax spec r.cfg ds dataDir r.out
