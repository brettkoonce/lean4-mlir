#!/usr/bin/env bash
# 4-GPU steady-state ms/step probe for every JAX (phase-2) ImageNet trainer, f32 and bf16 arm.
#
#     scripts/jax_probe_4gpu.sh /tmp/out.tsv
#
# The peer of `scripts/bf16_probe_4gpu.sh`, which does the same for the VERIFIED trainers. Same
# output shape, so the two tables sit side by side: `net|dtype|batch|ms|steps_per_epoch|secs|err`.
#
# ⚠⚠ THE JAX TRAINERS HAVE NO DTYPE FLAG. They hardcode `DT = jnp.bfloat16` and
#    `CONV_DT = jnp.bfloat16` as module-level constants. The f32 arm is produced by rewriting
#    those two lines into a scratch copy under $TMP; `jax/.lake/build/` is never modified.
# ⚠⚠ THEY PRINT A CUMULATIVE AVERAGE, not a steady-state median — `(time.time()-t0)/step`, so the
#    printed figure includes compile and is ~50% high at step 200. Every number here is
#    DIFFERENCED over steps 200->600. Reading the trainer's own line directly is the mistake.
# ⚠ Regenerate first (`lake exe <net>-imagenet default` in jax/). These are build products and
#   `planning/imagenet_rerun_sweep.md` §3.3 records them going stale silently, twice.
# ⚠ Box-specific: DEVS and TFDS_DATA_DIR below are this box's. The verified peer hardcodes ares'.
# ⛔ mnv4 carries §21.2's caveat unchanged: nothing has tied its collectives. Cost it, don't train it.
set -u
cd "$(dirname "$0")/.."
OUT="${1:?usage: scripts/jax_probe_4gpu.sh <out.tsv>}"
: > "$OUT"
PY="${JAX_PROBE_PY:-/home/skoonce/.venv-cuda/bin/python}"
DEVS="${JAX_PROBE_DEVS:-0,1,2,3}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-/home/skoonce/tensorflow_datasets}"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

wait_clean () {   # never start onto a box that still has a resident process: a stale one
  for i in $(seq 1 40); do   # produces a bogus OOM that reads like a hardware limit.
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    [ "$n" -eq 0 ] && return 0
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | while read -r p; do kill -9 "$p" 2>/dev/null; done
    sleep 5
  done
}

probe () {  # name generated-file dtype
  local name="$1" gen="jax/.lake/build/$2" dt="$3"
  local src="$gen" log="$TMP/$name.$dt.log" t0=$SECONDS
  if [ "$dt" = f32 ]; then
    src="$TMP/$name.f32.py"
    sed -e 's/^DT = jnp\.bfloat16/DT = jnp.float32/' \
        -e 's/^CONV_DT = jnp\.bfloat16/CONV_DT = jnp.float32/' "$gen" > "$src"
    grep -qE '^(DT|CONV_DT) = jnp\.float32' "$src" || {
      echo "$name|$dt|-|FAIL|-|0|no DT/CONV_DT constant to rewrite" >> "$OUT"; return; }
  fi
  [ -f "$gen" ] || { echo "$name|$dt|-|FAIL|-|0|missing $gen" >> "$OUT"; return; }
  wait_clean
  CUDA_VISIBLE_DEVICES="$DEVS" "$PY" -u "$src" > "$log" 2>&1 &
  local pid=$!
  for i in $(seq 1 240); do
    grep -qE '^  step 600/' "$log" 2>/dev/null && break
    grep -qiE 'Traceback|RESOURCE_EXHAUSTED|out of memory' "$log" 2>/dev/null && break
    kill -0 $pid 2>/dev/null || break
    sleep 5
  done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  local ms bs spe err
  ms=$(awk '/^  step /{split($2,a,"/");s=a[1];v=$5;gsub(/[()]|ms\/step/,"",v);
            if(s==200)x=s*v; if(s==600)y=s*v} END{if(y>0)printf "%.1f",(y-x)/400}' "$log")
  bs=$(grep -oE 'batch_size=[0-9]+' "$log" | head -1 | grep -oE '[0-9]+')
  spe=$(grep -oE '^  step 0/[0-9]+' "$log" | grep -oE '[0-9]+$' | head -1)
  err=$(grep -iEm1 'Traceback|RESOURCE_EXHAUSTED|out of memory' "$log" | cut -c1-110)
  echo "$name|$dt|${bs:--}|${ms:-FAIL}|${spe:--}|$((SECONDS-t0))|$err" >> "$OUT"
  echo "  done: $name $dt -> ${ms:-FAIL} ms/step (bs ${bs:-?}, ${spe:-?} steps/ep)"
  sleep 8
}

for arm in f32 bf16; do
  probe r34  generated_resnet34_imagenet.py        "$arm"
  probe r50  generated_resnet50_imagenet.py        "$arm"
  probe mnv2 generated_mobilenet_v2_imagenet.py    "$arm"
  probe enet generated_efficientnet_b0_imagenet.py "$arm"
  probe cnx  generated_convnext_tiny_imagenet.py   "$arm"
  probe vit  generated_vit_tiny_imagenet.py        "$arm"
  probe mnv4 generated_mobilenet_v4_imagenet.py    "$arm"
done
echo "ALL DONE"
