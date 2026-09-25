#!/usr/bin/env bash
# 2026-09-25: (a) fed arm at the job's SHIM_WORKERS=4 on the timm net; (b) the PRE-timm net's
# compute floor (synth) by swapping its renders (HEAD~1 = 17fb0c26) into verified_mlir/ for the
# duration, restored from HEAD afterwards. Same binary: the parameter shapes are identical.
set -u
cd "$(dirname "$0")/../.." || exit 1
R=runs/2026-09-25-mnv4-timm-probe
ROW='mnv4|mobilenetv4-imagenet-verified|adamdp64|adamdp64bf16|64|SHIM_WORKERS=4'
ROWS="$ROW" NETS=mnv4 ARMS=fed CKPT_TAG=probe-timm-w4 scripts/bf16_probe_3060.sh $R/probe_fed_w4.tsv
OLD=17fb0c26
for f in $(git ls-files 'verified_mlir/mnv4in_*'); do git show "$OLD:$f" > "$f"; done
ROWS="$ROW" NETS=mnv4 ARMS=synth CKPT_TAG=probe-old scripts/bf16_probe_3060.sh $R/probe_old_synth.tsv
git checkout HEAD -- verified_mlir/
git status --short verified_mlir/
echo RUN2 DONE
