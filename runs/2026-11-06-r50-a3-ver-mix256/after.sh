#!/usr/bin/env bash
# After the verified A3 MIX_CHUNK=256 run: pair its e100 bitmap (McNemar) against the JAX per-chunk
# run and the per-record arms, then resume ViT-S. Runs as its own systemd unit; waits on the run's.
# Nothing launches unless the run is COMPLETE and its bitmap reproduces its log's e100 count.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2 || exit 1
R=runs/2026-11-06-r50-a3-ver-mix256
PY=/home/skoonce/.venv-cuda/bin/python3
log() { echo "[after] $(date -u '+%F %T') $*" | tee -a "$R/after.log"; }

while systemctl --user is-active --quiet r50-a3-ver-mix256; do sleep 120; done
grep -q 'COMPLETE — 100/100' "$R/master.log" || { log "⛔ run not COMPLETE — no pairing, ViT-S NOT resumed"; exit 1; }

vb=$R/bitmaps/a3_lambaccdp4x128wxclipbcebf16_e100.bin
want=$(grep -a 'epoch 100: test_acc' "$R/full.log" | tail -1 | sed -n 's/.*test_acc = \([0-9]*\)\/50000.*/\1/p')
[ -f "$vb" ] || { log "⛔ no e100 bitmap $vb — ViT-S NOT resumed"; exit 1; }
got=$("$PY" -c "import numpy as n,sys; print(int(n.fromfile(sys.argv[1],n.uint8).sum()))" "$vb")
log "verified mix256 e100: bitmap $got, log ${want:-?}"
[ -n "$want" ] && [ "$got" = "$want" ] || { log "⛔ bitmap does not reproduce the log's count — ViT-S NOT resumed"; exit 1; }

pair() {  # name bitmap
  "$PY" scripts/demos/mcnemar.py "$vb" "$2" --label-a "verified mix256 s1" --label-b "$1" > "$R/mcnemar_vs_$1.txt" 2>&1
  log "vs $1: $(grep -o 'p = [0-9.e-]*' "$R/mcnemar_vs_$1.txt")"
}
pair jax_mix8_s1 runs/2026-11-06-r50-a3mix8-jax/bitmaps/a3mix8_jax_f32.bin
pair jax_s2 runs/2026-10-30-r50-a3-seeds/pairs/jax_s2_f32.bin
pair jax_s3 runs/2026-10-30-r50-a3-seeds/pairs/jax_s3_f32.bin
pair ver_1009 runs/2026-10-09-r50-a3-verified-bf16-100ep/bitmaps/a3_lambaccdp4x128wxclipbcebf16_e100.bin
pair ver_s2 runs/2026-10-30-r50-a3-ver-s2/bitmaps/a3_lambaccdp4x128wxclipbcebf16_e100.bin
pair ver_s3 runs/2026-10-30-r50-a3-ver-s3/bitmaps/a3_lambaccdp4x128wxclipbcebf16_e100.bin

sleep 30
nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . && { log "⛔ GPUs not idle — ViT-S NOT resumed"; exit 1; }
systemctl --user reset-failed vits-jax 2>/dev/null
systemd-run --user --unit=vits-jax --working-directory="$PWD" --setenv=PATH="$PATH" \
  --setenv=RUNDIR=runs/2026-11-05-vits-jax-300ep --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 \
  scripts/supervise.sh vits-accum-jax-4gpu && log "ViT-S resumed (unit vits-jax)"
