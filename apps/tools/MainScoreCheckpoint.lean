import LeanMlir.VerifiedNets

/-! # `score-checkpoint` — score a finished checkpoint, standalone

`planning/next_session_verified_trainer_code.md` §2. The JAX side has six `eval_*_full50k.py`;
the verified side had none, so a verified number could only be produced *in training*, and only
for the weights that happened to be live at that moment. This is the peer:

    .lake/build/bin/score-checkpoint <net> [dataDir]

with `$LEAN_MLIR_VARIANT` naming the render (as in every trainer here), `$LEAN_MLIR_CKPT`
overriding the checkpoint path, and `$LEAN_MLIR_REGION` ∈ `auto | live | ema` choosing which blob
region to score. The default path is `VerifiedNet.ckptPathFor`, i.e. **the file the training run
just wrote** — so the zero-argument case is "re-score what that run scored", which is the equality
gate below.

## ⭐ The gate, and it is an equality rather than a smoke test

For one checkpoint at one region, the number this prints must EQUAL the number the training run
printed for the epoch that wrote it. Same denominator, same batching, same graph, same metric.
Anything else means the eval half and its factoring disagree, and the factoring is wrong.

    LEAN_MLIR_VARIANT=<v> .lake/build/bin/score-checkpoint convnext data
    # → compare against that run's last `epoch N: test_acc = …` line

▶ It is available TODAY on ConvNeXt and ViT and on no other net, and that is a property of the
checkpoint format rather than of this tool: they have `nBnStats = 0`, so their whole eval state is
in the blob. The BN nets refuse — see `scoreCheckpoint`'s own docstring for why zeros in the
running-stat slots produce a plausible-looking percentage off garbage, and for the two exits.

## ⚠ Why a hand-written registry

There is no Lean-side enumeration of `VerifiedNetSpec`s — every trainer is its own `main` naming
its own net — so a tool that takes a net BY NAME has to pair the two somewhere. Kept here, next to
the only consumer, and deliberately covering every net rather than only the two that work: a net
that is missing from the table reads as "this tool does not support it", where a net that is
present and refuses reads as "the checkpoint format does not carry what scoring needs", which is
the true statement.
-/

/-- Every `VerifiedNetSpec` this tool can score, by the name you type. ⚠ The Imagenette peers are
    here too: they are the cheap way to exercise the tool at all (no 30 GB val drain, no shim), and
    `convnext`/`vit` among them have no BN either. -/
def scorableNets : List (String × VerifiedNetSpec) :=
  [ -- LayerNorm nets — no running statistics, so these SCORE today
    ("convnext",     convnextVerified)
  , ("convnext-in",  convnextImagenetVerified)
  , ("convnexts-in", convnextSImagenetVerified)
  , ("convnextb-in", convnextBImagenetVerified)
  , ("vit",          vitVerified)
  , ("vit-in",       vitImagenetVerified)
  , ("vits-in",      vitSImagenetVerified)
  , ("vitb-in",      vitBImagenetVerified)
    -- BatchNorm nets — present so the refusal is a message about the FORMAT rather than about
    -- this tool's coverage. Each throws with the `nBnStats` count and the two exits (§2b).
  , ("resnet34",     resnet34Verified)
  , ("resnet34-in",  resnet34ImagenetVerified)
  , ("resnet50",     resnet50Verified)
  , ("resnet50-in",  resnet50ImagenetVerified)
  , ("resnet50-in160", resnet50Imagenet160Verified)
  , ("mobilenetv2",  mobilenetv2Verified)
  , ("mobilenetv2-in", mobilenetv2ImagenetVerified)
  , ("efficientnet", efficientnetVerified)
  , ("efficientnet-in", efficientnetImagenetVerified)
  , ("mnv4",         mobilenetv4Verified)
  , ("mnv4-in",      mnv4ImagenetVerified) ]

def usage : String :=
  let names := String.intercalate " " (scorableNets.map (·.1))
  s!"score-checkpoint <net> [dataDir]\n\
  nets: {names}\n\
  env:  LEAN_MLIR_VARIANT (required — which render wrote the checkpoint)\n\
        LEAN_MLIR_CKPT    (default: the path the trainer writes, `ckptPathFor`)\n\
        LEAN_MLIR_REGION  auto | live | ema   (default auto = the shadow iff the variant has one)\n\
        LEAN_MLIR_CKPT_TAG, LEAN_MLIR_LOWERER — as in the trainers"

def main (argv : List String) : IO Unit := do
  let some netName := argv.head? | throw (IO.userError usage)
  let some spec := (scorableNets.find? (·.1 == netName)).map (·.2)
    | throw (IO.userError s!"unknown net '{netName}'\n{usage}")
  let net := spec.toNet
  -- ⚠ REQUIRED, with no default. The variant decides the blob's REGION COUNT, so a wrong guess is
  -- not a wrong graph — it is a size-guard refusal at best and a misaligned parameter walk at
  -- worst. The trainers can default it because they also render at it; this tool only reads.
  let some variant := ← IO.getEnv "LEAN_MLIR_VARIANT"
    | throw (IO.userError s!"LEAN_MLIR_VARIANT is unset, and it cannot be guessed: it names the \
render, the checkpoint file AND the blob's region count.\n{usage}")
  let ckpt ← match ← IO.getEnv "LEAN_MLIR_CKPT" with
    | some p => pure p
    | none   => net.ckptPathFor variant
  let region := (← IO.getEnv "LEAN_MLIR_REGION").getD "auto"
  net.scoreCheckpoint (argv.tail.head?.getD "data") variant ckpt region
