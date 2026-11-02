#!/usr/bin/env bash
# R50 A3 seeds 2 and 3: the paired test per seed, the 10-12 pair's method (runs/2026-10-12-r50-a3-jax/
# RESULTS.md): the JAX checkpoint re-scored through an f32 forward on all 50,000 val images with a
# per-image bitmap (jax/scripts/eval_full50k.py, PROTOCOL=train, DUMP_CORRECT), against the verified
# run's own epoch-100 bitmap, through scripts/demos/mcnemar.py. Seed 1's pair is that RESULTS.md's.
#
# Guards — a pairing is only as good as its two bitmaps:
#   * the re-scored JAX top-1 count must sit within 10 of the trainer's own epoch-100 val_top1 count
#     (same checkpoint, same f32 eval, same 50k; the 4-card sharded eval and this one-card batch-200
#     one differ by a few images: s1 38,803 vs 38,801, s2 38,833 vs 38,829) — a wrong checkpoint or
#     protocol is off by hundreds;
#   * the verified bitmap's set count must EQUAL its log's epoch-100 test_acc count.
# Two seeds score in parallel, one card each, after the A3 queue has left the GPUs.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
Q=runs/2026-10-30-r50-a3-seeds
PY=/home/skoonce/.venv-cuda/bin/python3
OUT="$Q/pairs"; mkdir -p "$OUT"

score() {   # seed gpu
  local s=$1 g=$2 ck=/home/skoonce/r50_a3_jax_2611s$1/resnet50_imagenet_rsbfaithful.state.npz
  [ -f "$ck" ] || { echo "⛔ s$s: no $ck"; return 1; }
  (cd jax && CUDA_VISIBLE_DEVICES=$g JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache \
    TFDS_DATA_DIR=/home/skoonce/tensorflow_datasets \
    GEN=generated/generated_resnet50_imagenet_rsbfaithful.py CKPT="$ck" BATCH=200 PROTOCOL=train \
    EVAL_DTYPE=f32 DUMP_CORRECT="../$OUT/jax_s${s}_f32" LABEL="r50-a3 jax s$s f32" \
    "$PY" -u scripts/eval_full50k.py) > "$OUT/jax_s${s}_score.log" 2>&1
}

score 2 0 & p2=$!
score 3 1 & p3=$!
wait $p2; r2=$?; wait $p3; r3=$?

rc=0
for s in 2 3; do
  echo "━━━ seed $s"
  jl=runs/2026-10-30-r50-a3-jax-s$s/full.log
  vl=runs/2026-10-30-r50-a3-ver-s$s/full.log
  jb="$OUT/jax_s${s}_f32.bin"
  vb=runs/2026-10-30-r50-a3-ver-s$s/bitmaps/a3_lambaccdp4x128wxclipbcebf16_e100.bin
  want_j=$(grep -a '^\[Epoch 100\]' "$jl" | tail -1 | sed -n 's/.*val_top1=\([0-9]*\)\/50000.*/\1/p')
  want_v=$(grep -a 'epoch 100: test_acc' "$vl" | tail -1 | sed -n 's/.*test_acc = \([0-9]*\)\/50000.*/\1/p')
  [ -f "$jb" ] || { echo "⛔ s$s: the JAX re-score wrote no bitmap — $OUT/jax_s${s}_score.log:"; tail -5 "$OUT/jax_s${s}_score.log"; rc=1; continue; }
  [ -f "$vb" ] || { echo "⛔ s$s: no verified e100 bitmap $vb"; rc=1; continue; }
  got_j=$("$PY" -c "import numpy as n,sys; print(int(n.fromfile(sys.argv[1],n.uint8).sum()))" "$jb")
  got_v=$("$PY" -c "import numpy as n,sys; print(int(n.fromfile(sys.argv[1],n.uint8).sum()))" "$vb")
  echo "  JAX      re-scored $got_j  trainer's e100 ${want_j:-?}"
  echo "  verified bitmap    $got_v  log's e100     ${want_v:-?}"
  if [ -z "$want_j" ] || [ $(( got_j > want_j ? got_j - want_j : want_j - got_j )) -gt 10 ] || [ "$got_v" != "$want_v" ]; then
    echo "⛔ s$s: a bitmap does not reproduce its run's own count — not pairing"; rc=1; continue; fi
  "$PY" scripts/demos/mcnemar.py "$jb" "$vb" --label-a "JAX s$s" --label-b "verified s$s" | tee "$OUT/mcnemar_s$s.txt"
done
exit $rc
