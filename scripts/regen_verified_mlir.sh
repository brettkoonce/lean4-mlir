#!/usr/bin/env bash
# Regenerate verified_mlir/ — the canonical entry point.
#
# Until now nothing rebuilt verified_mlir/; two lakefile.lean comments described it in prose, and
# you had to know which `lake env lean tests/Test*.lean` to run. That is how the committed
# resnet34_train_step.mlir got silently clobbered by a second writer
# (planning/xla_pjrt_handoff.md §2a).
#
# Two classes of writer, and the difference matters:
#
#   Proofs/Codegen/*.lean   `#eval` at ELABORATION time — `lake build <module>` writes the file.
#                           These are `pretty(provenGraph)`: the committed bytes ARE the
#                           certified render.
#   tests/Test*.lean        `#eval main` — `lake env lean <file>` writes the file (and tries
#                           iree-compile, which needs .venv/bin on PATH).
#                           These are hand-written string emitters: faithful per-op, NOT certified.
#
# Usage:
#   scripts/regen_verified_mlir.sh              # regenerate everything, then audit
#   scripts/regen_verified_mlir.sh proofs       # only the certified renders
#   scripts/regen_verified_mlir.sh tests        # only the hand-written renders
#   scripts/regen_verified_mlir.sh check        # write nothing; just run the writer audit
#
# After regenerating, `git diff verified_mlir/` should be EMPTY. A non-empty diff means either
# you changed a renderer (intended) or two writers disagree (not intended) — see `check`.
set -euo pipefail
cd "$(dirname "$0")/.."

WHAT="${1:-all}"

# ── the writer audit: every artifact should have exactly one writer ──
# An artifact with two writers is a silent last-writer-wins race: `lake build` and
# `lake env lean tests/...` produce different bytes and neither warns.
check_writers() {
  echo "── verified_mlir/ writer audit ──"
  local tmp rc
  tmp="$(mktemp)"
  grep -rn 'IO.FS.writeFile "verified_mlir/' --include='*.lean' . \
    | awk -F'verified_mlir/' '{
        split($2, a, "\"");
        split($1, b, ":");
        print a[1] "\t" b[1] ":" b[2]
      }' | sed 's|^\./||; s|\t\./|\t|' | sort -u > "$tmp"

  local total dupes
  total="$(cut -f1 "$tmp" | sort -u | wc -l)"
  dupes="$(cut -f1 "$tmp" | uniq -d || true)"
  rc=0
  if [ -z "$dupes" ]; then
    echo "  OK — $total artifacts, one writer each"
  else
    echo "  ⚠ $(echo "$dupes" | wc -l) of $total artifacts have MORE THAN ONE writer:"
    while IFS= read -r art; do
      [ -z "$art" ] && continue
      echo "    $art"
      awk -F'\t' -v a="$art" '$1 == a { print "      " $2 }' "$tmp"
    done <<< "$dupes"
    rc=1
  fi
  rm -f "$tmp"
  return $rc
}

# ── the train/eval tie: the eval forward must BE the forward the trainer differentiates ──
# A train step is (forward ++ backward), so a correctly-paired forward artifact is a byte-prefix
# of its train-step artifact. This is what caught the ResNet-34 BN skew: the committed
# resnet34_fwd.mlir normalised over the batch (reduce [0,2,3], n = B·H·W) while the certified
# resnet34_train_step.mlir normalises per example (reduce [2,3], n = H·W) — the SGD trainer was
# evaluating a different function than it trained. Only listed for nets whose forward AND train
# step both come from Proofs/Codegen; a hand-written forward has no reason to match textually.
check_fwd_prefix() {
  echo "── forward ⊂ train-step prefix check ──"
  python3 - <<'PY'
import sys
# Every net whose forward AND train step come from Proofs/Codegen. All five were hand-written
# until 2026-07-28; mobilenetv2_fwd was the second artifact this check would have caught —
# it rendered BATCH BN against a PER-EXAMPLE train step (measured: logits rel 1.86).
# ConvNeXt has no _fwd_eval and must not grow one: LayerNorm ⇒ train == eval.
PAIRS = [("resnet34_fwd.mlir",     "resnet34_train_step.mlir"),
         ("convnext_fwd.mlir",     "convnext_train_step.mlir"),
         ("efficientnet_fwd.mlir", "efficientnet_train_step.mlir"),
         ("mobilenetv2_fwd.mlir",  "mobilenetv2_train_step.mlir"),
         # Stochastic depth (planning/stochastic_depth.md). The SD variant gets its OWN pair
         # rather than reusing efficientnet_fwd, which is the whole point of §3's design: the drop
         # sites are emitted in the forward too (at an all-ones scale, exactly the identity), so
         # the SD train step keeps a prefix partner instead of the audit quietly not covering it.
         ("efficientnet_drop_fwd.mlir", "efficientnet_adamdrop_train_step.mlir"),
         ("enetin_drop_fwd.mlir",       "enetin_emarmsdrop64_train_step.mlir"),
         # ConvNeXt's SD pair, on the BATCHED chain (ConvNeXtRenderB) — where a per-example mask is
         # expressible at all. ⚠ These are the only ConvNeXt artifacts from that chain: the drop-free
         # batched render is tied but NOT swapped, so `convnext_fwd`/`convnext_train_step` above are
         # still the per-example renderer's. The two chains differ on 78 conv-VJP lines (commuting
         # transpose/reverse), all in the BACKWARD — so each pair is internally consistent and the
         # prefix property is unaffected by which chain produced it.
         ("convnext_drop_fwd.mlir", "convnext_adamdrop_train_step.mlir"),
         ("cnxin_drop_fwd.mlir",    "cnxin_adamwxclipdrop_train_step.mlir"),
         # ViT's SD pair, on the batched chain (ViTRenderB). ⚠ Unlike ConvNeXt's, ViT's batched
         # chain reproduces its committed artifacts byte-for-byte, so these are the same emitter
         # the drop-free vit_fwd/vit_adam_train_step above come from — one renderer, two flags.
         ("vit_drop_fwd.mlir",   "vit_adamdrop_train_step.mlir"),
         ("vitin_drop_fwd.mlir", "vitin_adamwxclipdrop_train_step.mlir")]

def body(lines, what):
    """The pretty(AST) body: everything from the first `%v0 = ` definition on.

    Skipping a fixed line count instead (as this did) only worked for ResNet-34: ConvNeXt's train
    step emits two header constants the forward does not need, so the two bodies start at different
    offsets. Anchoring on `%v0` is what makes the check net-independent — every `pretty` render
    starts its AST body there."""
    for i, l in enumerate(lines):
        if l.strip().startswith("%v0 = "):
            return lines[i:]
    raise ValueError(f"{what}: no `%v0 = ` line — not a pretty(AST) render?")

rc = 0
for fwd, ts in PAIRS:
    try:
        fl = open(f"verified_mlir/{fwd}").read().split("\n")
        tl = open(f"verified_mlir/{ts}").read().split("\n")
    except FileNotFoundError as e:
        print(f"  SKIP {fwd}: {e}"); continue
    fb = body(fl, fwd)
    fb = fb[: next(i for i, l in enumerate(fb) if l.strip().startswith("return"))]
    tb = body(tl, ts)[: len(fb)]
    if fb == tb:
        print(f"  OK — {fwd} is a byte-identical {len(fb)}-line prefix of {ts}")
    else:
        bad = next((i for i, (a, b) in enumerate(zip(fb, tb)) if a != b), len(fb))
        print(f"  ✗ {fwd} DIVERGES from {ts} at body line {bad}")
        print(f"      fwd: {fb[bad][:110] if bad < len(fb) else '<eof>'}")
        print(f"      ts : {tb[bad][:110] if bad < len(tb) else '<eof>'}")
        rc = 1
sys.exit(rc)
PY
}

# ── the silent-emit-failure audit ──
# `emitTok` ends in a catch-all that emits `// MALFORMED token stream`, and the batched
# tags fall back to `// ... render TODO`. Both are COMMENTS: a missing emit case does not
# fail the Lean build, does not fail iree-compile (the op is simply absent), and shows up
# only as a wrong numeric result much later. Adding an SHlo op without its `emitTok` case
# is the documented way to hit this, so it is checked here rather than remembered.
check_no_malformed() {
  echo "── silent-emit audit (MALFORMED / render TODO) ──"
  local hits
  hits="$(grep -rln 'MALFORMED\|render TODO' verified_mlir/ 2>/dev/null || true)"
  if [ -n "$hits" ]; then
    echo "  ✖ emit fell through to a catch-all in:"
    echo "$hits" | sed 's/^/      /'
    return 1
  fi
  echo "  OK — no artifact contains a catch-all emit marker"
  return 0
}

# ── the empty-SSA-slot audit ──
# A renderer that gates an op (`if convBias then pretty … else pure ("", "")`) but NOT the
# name list it feeds emits the empty string as an operand or a return value: `return %a, , %b`,
# `stablehlo.multiply %v5793,  :`. Both are malformed text the LOWERER rejects, so nothing in
# Lean catches them — and an arity `#guard` does not either, because the list keeps its full
# length. Hit twice on the mnv2/enet conv-bias drop (§2m): once in the AdamW tail, once in the
# SGD return list, where it rendered 210 names against 160 types.
check_no_empty_slot() {
  echo "── empty-SSA-slot audit (a gated op whose name list was not gated) ──"
  local hits
  hits="$(grep -rlE '(, ,|,  +[:=])|= [a-z_.]+ +,' verified_mlir/ 2>/dev/null || true)"
  if [ -n "$hits" ]; then
    echo "  ✖ an emitted operand or return slot is EMPTY in:"
    echo "$hits" | sed 's/^/      /'
    return 1
  fi
  echo "  OK — no artifact has an empty operand or return slot"
  return 0
}

# ── the zero-bias prelude audit ──
# A `convBias := false` render binds every dropped conv bias to a `%zb<c>` constant emitted by
# `zeroBiasPrelude`. Wire the operand (`biasName`) and forget the prelude and the artifact uses
# an SSA name nothing defines — `iree-compile`/XLA say "use of undeclared SSA value name" and
# nothing before them says anything. Measured 2026-07-31: all FOUR EfficientNet renders were in
# exactly that state (15 widths used, 0 declared). Vacuous while the committed bytes carry biases;
# it goes live the moment a net is swapped, which is when it is needed.
check_zero_bias_decls() {
  echo "── zero-bias prelude audit (%zb used but not declared) ──"
  python3 - <<'PY'
import re, sys, glob, os
rc = 0
for p in sorted(glob.glob("verified_mlir/*.mlir")):
    src = open(p).read()
    used = set(re.findall(r"%zb\d+", src))
    if not used: continue
    decl = set(re.findall(r"(%zb\d+) = stablehlo\.constant", src))
    miss = sorted(used - decl, key=lambda s: int(s[3:]))
    if miss:
        print(f"  ✖ {os.path.basename(p)}: {len(miss)} undeclared — {' '.join(miss)}")
        rc = 1
if rc == 0: print("  OK — every %zb an artifact uses is declared by its prelude")
sys.exit(rc)
PY
}

if [ "$WHAT" = "check" ]; then
  rc=0
  check_writers || rc=1
  echo
  check_fwd_prefix || rc=1
  echo
  check_no_malformed || rc=1
  echo
  check_no_empty_slot || rc=1
  echo
  check_zero_bias_decls || rc=1
  exit $rc
fi

# ── the certified renders (#eval fires during `lake build`) ──
if [ "$WHAT" = "all" ] || [ "$WHAT" = "proofs" ]; then
  echo "── Proofs/Codegen (pretty(provenGraph)) ──"
  for m in \
    LeanMlir.Proofs.Codegen.StableHLO \
    LeanMlir.Proofs.Codegen.MlpRender \
    LeanMlir.Proofs.Codegen.CnnRender \
    LeanMlir.Proofs.Codegen.ResNet34Render \
    LeanMlir.Proofs.Codegen.MobileNetV2Render \
    LeanMlir.Proofs.Codegen.EfficientNetRender \
    LeanMlir.Proofs.Codegen.ConvNeXtRender \
    LeanMlir.Proofs.Codegen.ConvNeXtRenderB \
    LeanMlir.Proofs.Codegen.ViTRender \
    LeanMlir.Proofs.Codegen.ViTRenderB
  do
    echo "  lake build $m"
    lake build "$m"
  done
fi

# ── the hand-written renders (`#eval main` at file elaboration) ──
if [ "$WHAT" = "all" ] || [ "$WHAT" = "tests" ]; then
  echo "── tests/ (hand-written emitters) ──"
  # NOTE most of these no longer WRITE anything — they are `iree-compile` smokes over the committed
  # bytes, kept here because that is the one step `lake build` cannot do (it needs the compiler on
  # PATH) and they throw if an artifact is missing. As of 2026-07-28 the only files below that still
  # emit an artifact are the two cifar8 ones; every `_fwd`/`_train_step` writer has moved to
  # Proofs/Codegen. Running them is still the right smoke — it just no longer risks a clobber.
  export PATH="$PWD/.venv/bin:$PATH"
  for f in \
    tests/TestResnet34Train.lean \
    tests/TestMobilenetV2Fwd.lean \
    tests/TestMobilenetV2Train.lean \
    tests/TestMobilenetV2TrainPC.lean \
    tests/TestEfficientNetFwd.lean \
    tests/TestEfficientNetTrain.lean \
    tests/TestConvNeXtFwd.lean \
    tests/TestConvNeXtTrain.lean \
    tests/TestViTFwd.lean \
    tests/TestViTTrain.lean \
    tests/TestCifar8AdamTrain.lean \
    tests/TestCifar8WideTrain.lean
  do
    echo "  lake env lean $f"
    lake env lean "$f"
  done
fi

echo
check_writers || true
echo
check_fwd_prefix || true
echo
check_no_malformed || true
echo
echo "── git diff verified_mlir/ (should be empty) ──"
git diff --stat verified_mlir/ || true
