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
PAIRS = [("resnet34_fwd.mlir", "resnet34_train_step.mlir")]
rc = 0
for fwd, ts in PAIRS:
    try:
        fl = open(f"verified_mlir/{fwd}").read().split("\n")
        tl = open(f"verified_mlir/{ts}").read().split("\n")
    except FileNotFoundError as e:
        print(f"  SKIP {fwd}: {e}"); continue
    # body = past "module @m {", the func header, and the leading provenance comment
    fb = fl[3:]
    fb = fb[: next(i for i, l in enumerate(fb) if l.strip().startswith("return"))]
    tb = tl[3 : 3 + len(fb)]
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

if [ "$WHAT" = "check" ]; then
  rc=0
  check_writers || rc=1
  echo
  check_fwd_prefix || rc=1
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
    LeanMlir.Proofs.Codegen.ViTRender
  do
    echo "  lake build $m"
    lake build "$m"
  done
fi

# ── the hand-written renders (`#eval main` at file elaboration) ──
if [ "$WHAT" = "all" ] || [ "$WHAT" = "tests" ]; then
  echo "── tests/ (hand-written emitters) ──"
  export PATH="$PWD/.venv/bin:$PATH"   # tests/* also try iree-compile
  for f in \
    tests/TestResnet34Fwd.lean \
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
echo "── git diff verified_mlir/ (should be empty) ──"
git diff --stat verified_mlir/ || true
