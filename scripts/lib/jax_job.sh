# jax_job.sh — the JAX reference path's half of a `scripts/supervise.sh` job conf.
#
# A JAX-path conf sets two variables and sources this file:
#
#     CKPT_BASE=/home/skoonce/<run dir>/<trainer stem>    # checkpoints live OUTSIDE the repo
#     PY_REL=.lake/build/generated_<net>_imagenet[_<recipe>].py   # relative to jax/
#     . "$(dirname "${BASH_SOURCE[0]}")/../lib/jax_job.sh"
#
# and gets `CMD` (the launch), `epoch_now` (what supervise.sh resumes from) and `pc_jax_trainer`
# (the PRECHECK half every JAX conf needs). Used by the mnv2/mnv4/vit JAX confs.
#
# Resume semantics:
#   * `LEAN_MLIR_CKPT_EVERY=1` writes `<base>_e<N>.bin` + `<base>_e<N>.state.npz` every epoch. The
#     pruner keeps the 3 newest `.state.npz`, and the newest is complete (atomic rename).
#   * `LEAN_MLIR_RESUME=<newest .state.npz>` restores the full train state: weights, optimizer
#     moments, the EMA shadow, BN running stats and the global step, so the LR schedule and the
#     drop-path/dropout keys (`fold_in(_drop_base, _global_step)`) continue where they stopped. The
#     DATA stream does not: a fresh process restarts the `shuffle(8192, seed=42)` order from its
#     first permutation and draws augmentation unseeded (`AUG_SEED` unset), so a chunked run is the
#     same recipe on a different sample of crops and orderings, not the same bits as one launch.
#   * `JAX_COMPILATION_CACHE_DIR`: a cold compile is minutes (MNv4's UIB is ~15). The cache makes it
#     once per graph, not once per chunk.

: "${CKPT_BASE:?jax_job.sh needs CKPT_BASE}" "${PY_REL:?jax_job.sh needs PY_REL}"

CMD=(bash -c '
  set -u
  base='"$CKPT_BASE"'
  last=""; n=0
  for f in "${base}"_e*.state.npz; do
    [ -e "$f" ] || continue
    k="${f##*_e}"; k="${k%.state.npz}"
    [ "$k" -gt "$n" ] && { n="$k"; last="$f"; }
  done
  resume=()
  if [ -n "$last" ]; then echo "[jax] resuming from epoch $n ($last)"; resume=(LEAN_MLIR_RESUME="$last")
  else echo "[jax] fresh start"; fi
  cd jax || exit 1
  exec env JAX_COMPILATION_CACHE_DIR=/home/skoonce/.jax_cache \
    LEAN_MLIR_PARAMS_OUT="$base" LEAN_MLIR_CKPT_EVERY=1 \
    TFDS_DATA_DIR="${TFDS_DATA_DIR:-/home/skoonce/tensorflow_datasets}" \
    "${resume[@]}" ../.venv/bin/python -u '"$PY_REL"'
')

epoch_now() {
  local n=0 k f
  for f in "${CKPT_BASE}"_e*.state.npz; do
    [ -e "$f" ] || continue
    k="${f##*_e}"; k="${k%.state.npz}"
    [ "$k" -gt "$n" ] && n="$k"
  done
  echo "$n"
}

# pc_jax_trainer — the trainer exists and is the committed one, the data is there, and the venv is
# the pinned stack. Call from the conf's PRECHECK after sourcing scripts/lib/precheck.sh.
pc_jax_trainer() {
  local ok=0
  [ "${DRY_RUN:-0}" != "0" ] || mkdir -p "$(dirname "$CKPT_BASE")" /home/skoonce/.jax_cache
  [ -f "jax/$PY_REL" ] || {
    echo "⛔ missing jax/$PY_REL — scripts/regen_jax_generated.sh sync"; ok=1; }
  # the trainer this runs must be the committed one (a stale .lake/build copy trained the MNv2
  # 350-epoch run at a label smoothing its source had dropped)
  pc_jax_box || ok=1
  [ -d "${TFDS_DATA_DIR:-/home/skoonce/tensorflow_datasets}/imagenet2012" ] || {
    echo "⛔ no imagenet2012 under ${TFDS_DATA_DIR:-/home/skoonce/tensorflow_datasets}"; ok=1; }
  # the pinned stack only (other jax versions break bf16 convs); the pin is READ from the lock
  local pin; pin="$(sed -n 's/^jax==//p' jax/requirements-cuda-lock.txt)"
  .venv/bin/python -c "import jax,sys; sys.exit(0 if jax.__version__ == '$pin' else 1)" \
    2>/dev/null || { echo "⛔ .venv's jax is not the pinned $pin (jax/requirements-cuda-lock.txt)"; ok=1; }
  return $ok
}
