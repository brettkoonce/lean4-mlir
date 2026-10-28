#!/bin/bash
# planning/part2_ship_audit.md §2.6: three more trainings of the flow arm (independent coupling,
# 20,000 steps, Adam 1e-3, batch 256 — runs/2026-09-11-boltzmann-generator/README.md) under
# LEAN_MLIR_SEED=1..3 (the knob added to demos/MainDiffusion2d.lean on 10-27), each sampled at
# NFE 50 with `logp` and `nll` as the 09-11 row was, then scored beside the 09-11 cloud (seed 0).
# The arm's checkpoint is the 09-11 model that `reuse` and the figure read, so it is put back after.
cd /home/skoonce/lean/klawd_max_power/lean4-jax-mlir || exit 1
G=${1:-0}; R=runs/2026-10-27-boltzmann-seeds
export LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$G LEAN_MLIR_MEM_FRACTION=0.1
log () { echo "$(date -u '+%F %T') $*" >> $R/queue.log; }
P=.lake/build/diffusion_2d_denoiser__muller_brown__flow_params.bin
S=.lake/build/diffusion2d_samples_muller_brown-flow_fm-euler_s20000_n50_e0
S0=runs/2026-09-11-boltzmann-generator/samples/$(basename $S)
cp $P $R/flow_s0_params.bin.keep
for x in bin logp.bin reflogp.bin; do cp $S0.$x $R/flow_s0.$x; done
for s in 1 2 3; do
  log "flow s$s start"
  LEAN_MLIR_SEED=$s .lake/build/bin/diffusion-2d muller_brown flow 20000 50 fm-euler logp nll > $R/flow_s$s.log 2>&1
  log "flow s$s exit $?"
  cp $P $R/flow_s${s}_params.bin
  for x in bin logp.bin reflogp.bin noise.bin; do cp $S.$x $R/flow_s$s.$x; done
done
cp $R/flow_s0_params.bin.keep $P && rm $R/flow_s0_params.bin.keep
python3 scripts/demos/boltzmann_metrics.py score flow_s0=$R/flow_s0.bin flow_s1=$R/flow_s1.bin \
  flow_s2=$R/flow_s2.bin flow_s3=$R/flow_s3.bin --gate --out=$R > $R/table1.txt 2>&1
log "DONE: $(grep -c . $R/table1.txt) table lines"
