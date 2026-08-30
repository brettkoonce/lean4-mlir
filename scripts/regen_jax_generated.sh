#!/usr/bin/env bash
# regen_jax_generated.sh — the JAX half of what `regen_verified_mlir.sh` does for verified_mlir/.
#
#     scripts/regen_jax_generated.sh          # re-emit into jax/generated/ (the committed copies)
#     scripts/regen_jax_generated.sh check    # write nothing; drift vs jax/generated/ -> exit 1  [CI]
#     scripts/regen_jax_generated.sh box      # is THIS box's .lake/build stale?     -> exit 1  [PRECHECK]
#     scripts/regen_jax_generated.sh sync     # make this box's .lake/build match the committed copies
#
# ⛔ WHY THIS EXISTS. `verified_mlir/` commits 233 rendered artifacts and CI diffs them against
# their renderer. The JAX path emits the same CLASS of object — the file that actually trains —
# to `.lake/build/`, which is GITIGNORED, checked by nothing, and never re-emitted once warm.
#
# That is not hypothetical. `generated_mobilenet_v2_imagenet_full.py` sat dated Jun 22 while its
# source was fixed 2026-07-07 (`ae0f40bb`, labelSmoothing 0.1 -> 0.0). The blueprint's
# paper-faithful 350-epoch MNv2 result (71.44/90.34, `4c1cb1eb`, 2026-08-01) was almost certainly
# trained on the stale file — 45 h of GPU, at a label smoothing the source had stopped asking for
# three weeks earlier, published as "paper-faithful". Re-emitting on 2026-08-30 moved 175 lines:
# LS removed, RMSProp `g/(sqrt(s)+EPS)` -> `g/sqrt(s+EPS)` (TF's form; EPS=1.0 makes it large),
# `antialias=True` on both resizes, `take`-masked eval over all 50,000.
#
# ⭐ THE THREE MODES ANSWER THREE DIFFERENT QUESTIONS, and only the third is free:
#   check : does the SOURCE still emit what is COMMITTED?   (needs Lean; this is the CI gate)
#   box   : does THIS BOX's .lake/build match what is COMMITTED?  (a diff; no Lean, no GPU)
#   write : adopt the current source output as the new committed copy (review the diff!)
# `box` is the one a supervisor PRECHECK should call — it is instant and it is the exact question
# the MNv2 run needed asked.
#
# ⚠ SCOPE: the 11 ImageNet emitters and their shims — i.e. every artifact behind a published
# ImageNet number. NOT the Imagenette/MNIST demos or the vjp-oracles, which back no ImageNet
# result. A guard that silently claimed to cover them would be its own version of this bug.
#
# ⚠ `--emit` IS LOAD-BEARING. `runJax` writes the trainer and then SPAWNS TRAINING, so without
# it every mode here would start 30+ training runs. That flag is `runRecipeMain`'s (added upstream
# in the same week and for the same reason — regenerating an artifact used to mean racing a timeout
# against a real ImageNet run); this script uses it rather than adding a second way to do one job.
# The emitters write CWD-relative, so `check` runs them in a scratch CWD and never touches
# `.lake/build` — safe while a run is in flight, and it was, repeatedly.
set -u
cd "$(dirname "$0")/.." || exit 1
MODE="${1:-write}"
ROOT="$PWD"
GEN="$ROOT/jax/generated"
LIVE="$ROOT/jax/.lake/build"
BIN="$ROOT/jax/.lake/build/bin"

EXES="resnet34-imagenet resnet50-imagenet vit-tiny-imagenet vit-s-imagenet vit-b-imagenet
      mobilenet-v2-imagenet mobilenet-v4-imagenet efficientnet-b0-imagenet
      convnext-tiny-imagenet convnext-s-imagenet convnext-b-imagenet"

# ── mode `sync`: make the box match the committed copies. ────────────────────────
# ⚠ WITHOUT THIS THE GUARD IS A TRAP. `box` says "re-emit", but re-emitting through the
# emitters SPAWNS TRAINING (Jax/Runner.lean) — so the obvious fix starts 30+ runs. The
# committed jax/generated/ copies ARE the current source output (CI proves it), so copying
# them in is the same bytes with none of the hazard.
if [ "$MODE" = "sync" ]; then
  [ -d "$GEN" ] || { echo "⛔ $GEN missing"; exit 1; }
  mkdir -p "$LIVE"; n=0
  for f in "$GEN"/*.py; do
    b="$(basename "$f")"
    cmp -s "$f" "$LIVE/$b" || { cp "$f" "$LIVE/$b"; echo "  synced $b"; n=$((n+1)); }
  done
  echo "✅ synced $n artifact(s) into .lake/build"
  exit 0
fi

# ── mode `box`: pure diff, no Lean, no build. What a PRECHECK calls. ──────────────
if [ "$MODE" = "box" ]; then
  [ -d "$GEN" ] || { echo "⛔ $GEN missing — run scripts/regen_jax_generated.sh first"; exit 1; }
  bad=0; n=0
  for f in "$GEN"/*.py; do
    [ -e "$f" ] || continue
    b="$(basename "$f")"; n=$((n+1))
    if [ ! -f "$LIVE/$b" ]; then
      echo "⚠ $b: not built on this box yet (emit it before running)"; continue
    fi
    if ! cmp -s "$f" "$LIVE/$b"; then
      echo "⛔ STALE: $LIVE/$b differs from the committed jax/generated/$b"
      echo "   $(diff "$f" "$LIVE/$b" | grep -c '^[<>]') lines differ. Re-emit before trusting a run off it."
      bad=1
    fi
  done
  [ "$bad" = 0 ] && echo "✅ box in sync with jax/generated/ ($n artifacts)"
  exit $bad
fi

# ── modes `write` and `check`: re-emit from source in a scratch CWD ───────────────
for e in $EXES; do
  [ -x "$BIN/$e" ] || { echo "⛔ missing $BIN/$e — (cd jax && lake build $e)"; exit 1; }
done
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/.lake/build"

emitted=0
for e in $EXES; do
  # The recipe list is the exe's own --help, so a new recipe is covered without editing this file.
  recipes="$("$BIN/$e" --help 2>/dev/null | awk '/^recipes/{f=1;next} /^data_dir/{f=0} f && NF{print $1}')"
  [ -n "$recipes" ] || recipes="default"
  for r in $recipes; do
    for extra in "--emit" "--shim"; do
      ( cd "$TMP" && timeout 300 "$BIN/$e" "$r" $extra >/dev/null 2>&1 )
    done
    emitted=$((emitted+1))
  done
done
echo "re-emitted from source: $emitted (exe, recipe) pairs -> $(ls "$TMP/.lake/build"/*.py 2>/dev/null | wc -l) artifacts"

if [ "$MODE" = "check" ]; then
  bad=0
  for f in "$TMP/.lake/build"/*.py; do
    b="$(basename "$f")"
    if [ ! -f "$GEN/$b" ]; then echo "⛔ NEW, uncommitted: $b"; bad=1; continue; fi
    if ! cmp -s "$f" "$GEN/$b"; then
      echo "⛔ DRIFT: jax/generated/$b is not what the source emits ($(diff "$GEN/$b" "$f" | grep -c '^[<>]') lines)"
      bad=1
    fi
  done
  for f in "$GEN"/*.py; do
    b="$(basename "$f")"
    [ -f "$TMP/.lake/build/$b" ] || { echo "⛔ ORPHAN: jax/generated/$b is emitted by nothing"; bad=1; }
  done
  [ "$bad" = 0 ] && echo "✅ jax/generated/ matches what the source emits"
  exit $bad
fi

mkdir -p "$GEN"
cp "$TMP/.lake/build"/*.py "$GEN"/
echo "✅ wrote $(ls "$GEN"/*.py | wc -l) artifacts to jax/generated/ — REVIEW THE DIFF before committing."
