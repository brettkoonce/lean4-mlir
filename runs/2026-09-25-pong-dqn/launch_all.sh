#!/bin/bash
# Table 1 (pixels k=4, k=1 at the default opponent) + Table 2 (opponent speed 1.0 / 2.0,
# pixels k=4 and the state twin; the 1.5 column is Table 1's). Two runs per card: the
# pixel loop is host-bound (GPU busy ~4 of ~24 ms per update), so they overlap.
cd "$(dirname "$0")/../.."
P=${PJRT_PLUGIN:-/home/skoonce/lean/klawd_max_power/lean4-jax-mlir/.venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so}
D=runs/2026-09-25-pong-dqn
run() {  # card tag args...
  local dev=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$dev PJRT_PLUGIN=$P LEAN_MLIR_MEM_FRACTION=0.3 \
    .lake/build/bin/pong-dqn "$@" tag=$tag 2>&1 | grep --line-buffered -v '^\[pjrt\|^I0925' > $D/$tag.log
}
run 3 px4_o15 mode=pixels k=4 opp=1.5 &
run 3 state_o10 mode=state opp=1.0 &
run 2 px1_o15 mode=pixels k=1 opp=1.5 &
run 2 px4_o20 mode=pixels k=4 opp=2.0 &
run 1 px4_o10 mode=pixels k=4 opp=1.0 &
run 1 state_o20 mode=state opp=2.0 &
wait
