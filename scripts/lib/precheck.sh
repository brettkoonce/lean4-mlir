# Sourced by the ImageNet job confs (scripts/jobs/*.conf) for their PRECHECK, from the repo root.
#
# One copy of the checks every verified-path launch needs, so a fix to one lands in all of them.
# Before this file each conf carried its own copy, and they had drifted: only mnv4-half built its
# exe, three confs had no freshness check at all, and ConvNeXt's augmentation grep matched the
# `def _autoaugment(` line, so it passed on every shim (planning/imagenet_parity.md C7).
#
# Every pc_* function prints "⛔ …" and returns 1 on a failure, 0 otherwise; "⚠ …" lines are
# advice and never fail. A conf's PRECHECK chains them and ORs the statuses:
#
#     precheck_x() {
#       local ok=0
#       pc_box || ok=1
#       pc_exe mobilenetv2-imagenet-verified || ok=1
#       …
#       return $ok
#     }
#
# ⚠ supervise.sh runs PRECHECK under DRY_RUN too (`lake run <job> plan`). A plan must stay cheap and
# must not build, so under DRY_RUN `pc_exe` reports instead of building. Nothing here touches a GPU.

# pc_env_get KEY — the value ENV_EXTRA gives KEY (empty when absent). bash strips the quotes when it
# builds the array, so this is the unquoted value.
pc_env_get() { printf '%s\n' "${ENV_EXTRA[@]}" | sed -n "s/^$1=//p" | tail -1; }

# pc_env KEY=VAL [WHY] — ENV_EXTRA must carry exactly KEY=VAL. Most of these knobs are opt-in and
# SILENT when absent (residency, worker count, epoch count), so the conf names why each one matters.
pc_env() {
  printf '%s\n' "${ENV_EXTRA[@]}" | grep -qxF "$1" && return 0
  echo "⛔ $1 is missing from ENV_EXTRA${2:+ — $2}"; return 1
}

# pc_env_absent KEY [WHY] — ENV_EXTRA must NOT set KEY.
pc_env_absent() {
  printf '%s\n' "${ENV_EXTRA[@]}" | grep -q "^$1=" || return 0
  echo "⛔ ENV_EXTRA sets $1${2:+ — $2}"; return 1
}

# pc_box — the plugin and the shim python _box.sh selected must exist on this box.
pc_box() {
  local ok=0
  echo "box=$BOX plugin=$BOX_PLUG python=$BOX_PY"
  [ -f "$BOX_PLUG" ] || { echo "⛔ PJRT_PLUGIN not found: $BOX_PLUG"; ok=1; }
  [ -x "$BOX_PY" ]   || { echo "⛔ python for the shim missing: $BOX_PY"; ok=1; }
  return $ok
}

# pc_exe EXE [SRC...] — the trainer binary is current. `lake build EXE` is the authority: a no-op when
# fresh, and neither `lake build` of another target nor `lake build Apps` relinks it (the MNv2 run's
# binary was 11 days old at launch and carried the mimalloc leak; mnv4-half's first launch predated
# `mnv4ImagenetVerified.dropoutKeep` and died at step 1). mtime is not the authority — a cache replay
# restamps an identical binary, and a checkout restamps sources — so under DRY_RUN, where nothing is
# built, the SRC mtimes (default: the trainer core and the C helpers) only produce a ⚠.
pc_exe() {
  local exe="$1"; shift
  local bin=".lake/build/bin/$exe"
  local srcs=("$@")
  [ ${#srcs[@]} -gt 0 ] || srcs=(LeanMlir/Verified/Train.lean ffi/f32_helpers.c)
  if [ "${DRY_RUN:-0}" != "0" ]; then
    if [ ! -x "$bin" ]; then echo "⚠ $bin not built yet — a launch builds it"; return 0; fi
    local s
    for s in "${srcs[@]}"; do
      [ "$bin" -nt "$s" ] || echo "⚠ $bin is older than $s — a launch rebuilds it (lake build $exe)"
    done
    return 0
  fi
  local log; log="$(mktemp)"
  if ! lake build "$exe" > "$log" 2>&1; then
    echo "⛔ lake build $exe failed:"; tail -20 "$log" | sed 's/^/   /'; rm -f "$log"; return 1
  fi
  rm -f "$log"
  [ -x "$bin" ] || { echo "⛔ $bin missing after lake build $exe"; return 1; }
}

# pc_pjrt_so — ffi/libpjrt_ffi.so is what ffi/pjrt_ffi.c compiles to today. The .so is dlopen'd at run
# time and is not a lake target, so neither `lake build` nor the exe check above sees it (found
# 2026-09-18: the .so on the 3060 box was built 08-16, three commits of shim source behind).
# The check COMPILES the source to a scratch file and compares bytes (~0.6 s; the build is
# reproducible): mtime is not the authority, since a checkout restamps pjrt_ffi.c and made an
# up-to-date .so read as stale. Nothing in the repo is written.
pc_pjrt_so() {
  local fix="gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o ffi/libpjrt_ffi.so"
  [ -f ffi/libpjrt_ffi.so ] || { echo "⛔ ffi/libpjrt_ffi.so is missing:"; echo "   $fix"; return 1; }
  local tmp; tmp="$(mktemp --suffix=.so)"
  if ! gcc -fPIC -O2 -shared ffi/pjrt_ffi.c -ldl -o "$tmp" 2>/dev/null; then
    rm -f "$tmp"; echo "⛔ ffi/pjrt_ffi.c does not compile — cannot confirm the .so is current"; return 1
  fi
  if ! cmp -s "$tmp" ffi/libpjrt_ffi.so; then
    rm -f "$tmp"
    echo "⛔ ffi/libpjrt_ffi.so is not what ffi/pjrt_ffi.c builds to — the run would execute stale shim code:"
    echo "   $fix"; return 1
  fi
  rm -f "$tmp"
}

# pc_jax_box — this box's jax/.lake/build (where the trainers and the tfds shims are READ from) matches
# the committed jax/generated/. A warm cache serves a stale file silently. A diff: no Lean, no GPU.
pc_jax_box() {
  scripts/regen_jax_generated.sh box >/dev/null 2>&1 && return 0
  echo "⛔ this box's jax/.lake/build artifacts are STALE vs committed jax/generated/:"
  scripts/regen_jax_generated.sh box 2>&1 | grep -E 'STALE|not built' | sed 's/^/   /'
  echo "   scripts/regen_jax_generated.sh sync, then re-run."; return 1
}

# pc_shim NAME — jax/.lake/build/generated_NAME_shim.py exists and resizes with antialias (every shim
# emitted since C3 does).
pc_shim() {
  local f="jax/.lake/build/generated_$1_shim.py"
  [ -f "$f" ] || { echo "⛔ missing $f — scripts/regen_jax_generated.sh sync"; return 1; }
  grep -q 'antialias' "$f" || {
    echo "⛔ $f has no antialiased resize — it predates C3. scripts/regen_jax_generated.sh sync"; return 1; }
}

# The CALL lines of FILE that contain the fixed string PAT: `def` lines and comments are dropped, so
# a helper that is defined in every shim cannot satisfy the check.
_pc_calls() { grep -F -- "$2" "$1" 2>/dev/null | grep -vE '^[[:space:]]*(def |#)'; }

# pc_aug_call FILE PAT [WHAT] — FILE must CALL PAT, e.g. `_randaugment(img, 2, 9` (the call carries
# its arguments, so the magnitude is checked too).
pc_aug_call() {
  _pc_calls "$1" "$2" | grep -q . && return 0
  echo "⛔ $1 has no call to \`$2\`${3:+ ($3)} — a definition is not a call site."
  echo "   scripts/regen_jax_generated.sh sync"; return 1
}

# pc_aug_no_call FILE PAT [WHAT] — FILE must NOT call PAT (the other recipe's augmentation).
pc_aug_no_call() {
  _pc_calls "$1" "$2" | grep -q . || return 0
  echo "⛔ $1 calls \`$2\`${3:+ — $3}"; return 1
}

# pc_render SLUG VARIANT REPLICAS [ALL_REDUCES] — the train step the run executes exists, is rendered
# for REPLICAS replicas, and (when given) carries exactly ALL_REDUCES all-reduce ops. The count is how
# a sync-BN render is told from a per-replica one: MNv2's went 158 → 314 at cad811ac, and a relaunch
# off the old render would train different BN semantics with nothing in the log to say so.
pc_render() {
  local f="verified_mlir/$1_$2_train_step.mlir" ok=0
  [ -f "$f" ] || { echo "⛔ missing $f — scripts/regen_verified_mlir.sh proofs"; return 1; }
  grep -q "DATA-PARALLEL over $3 replicas" "$f" || {
    echo "⛔ $f is not a $3-replica render"; ok=1; }
  if [ -n "${4:-}" ]; then
    local n; n="$(grep -c 'stablehlo.all_reduce' "$f" || true)"
    [ "${n:-0}" = "$4" ] || {
      echo "⛔ $f has ${n:-0} all_reduce ops, expected $4 — not the render this conf was checked"
      echo "   against (sync-BN vs per-replica, or a re-render that moved the collectives)."; ok=1; }
  fi
  return $ok
}

# pc_ckpt SLUG VARIANT [TAG] — LEAN_MLIR_VARIANT and CKPT_EPOCH_FILE both name VARIANT. Checkpoints are
# `<slug>_<variant>_ckpt_xla.bin`, so an epoch file left at another variant's name reads 0 forever
# and the run never advances and never ends. Also warns about a resume and about other variants'
# checkpoints, which must not be resumed into this run (a different region layout).
# An optional TAG is `LEAN_MLIR_CKPT_TAG`, which the driver appends as `_ckpt_xla_<TAG>.bin`.
pc_ckpt() {
  local slug="$1" var="$2" tag="${3:-}" ok=0
  local ck=".lake/build/${slug}_${var}_ckpt_xla${tag:+_$tag}.bin"
  pc_env "LEAN_MLIR_VARIANT=$var" "the conf's arm" || ok=1
  [ "$CKPT_EPOCH_FILE" = "$ck.epoch" ] || {
    echo "⛔ CKPT_EPOCH_FILE ($CKPT_EPOCH_FILE) does not match the variant $var${tag:+ tag $tag}"; ok=1; }
  local stray; stray="$(ls .lake/build/"${slug}"_*_ckpt_xla*.bin 2>/dev/null \
    | grep -vx "$ck" || true)"
  if [ -n "$stray" ]; then
    echo "⚠ other $slug checkpoints present — they must NOT be resumed into this run:"
    printf '   %s\n' $stray
  fi
  if [ -f "$CKPT_EPOCH_FILE" ]; then
    local n; n="$(tr -cd '0-9' < "$CKPT_EPOCH_FILE")"
    echo "⚠ $CKPT_EPOCH_FILE exists (epoch ${n:-0}) — this run will RESUME from it, not start fresh."
  fi
  return $ok
}

# pc_gpu_idle — no compute process on any GPU. A leftover one fakes OOM and NCCL failures at step 0.
pc_gpu_idle() {
  local busy; busy="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)"
  [ -z "$busy" ] && return 0
  echo "⛔ GPUs are not idle — a leftover process will fake an OOM at step 0:"
  echo "$busy" | sed 's/^/   /'; return 1
}
