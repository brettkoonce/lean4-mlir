#!/usr/bin/env bash
# ConvNeXt-T @ 4x64 bf16 — the batch-64 rescope's viability probe.
#   * does the 64 render COMPILE and FIT on a 12 GB 3060? (the §1-A gate)
#   * fed vs synth ms/step -> the ETA, and fed-minus-synth -> the shim cost
# Peak per-GPU memory is sampled alongside, because the probe script does not record it.
set -u
cd /home/skoonce/lean/proof_verify_demo/verify-v2
D=runs/2026-09-17-cnx-sweep

# memory sampler: 2 s cadence, max per card, for the whole probe
( while :; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
    sleep 2
  done ) > "$D/mem_samples.txt" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

ROWS="cnx|convnext-imagenet-verified|adamdpwxclipdrop|adamdpwxclipdropbf16|64|" \
WORKERS=4 ARMS="fed synth" PRECS=bf16 NETS=cnx WARM=200 STEPS=600 \
CKPT_TAG=probe3060bs64 \
  scripts/bf16_probe_3060.sh "$D/sweep.tsv"
RC=$?

kill $SAMPLER 2>/dev/null
echo "── peak memory per GPU (MiB) during the probe ──"
awk -F', ' '{if($2>m[$1])m[$1]=$2} END{for(i in m) print "  GPU "i": "m[i]" MiB"}' "$D/mem_samples.txt" | sort
echo "probe rc=$RC"
