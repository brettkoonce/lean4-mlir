# ConvNeXt-T JAX half, official recipe — how this run is launched (mars, 4× 3060 12 GB)

    systemd-run --user --unit=cnx-off-jax --working-directory="$PWD" --setenv=PATH="$PATH" \
      --setenv=RUNDIR=runs/2026-10-21-cnx-full-jax-bf16-300ep \
      --setenv=XLA_PYTHON_CLIENT_MEM_FRACTION=0.97 \
      scripts/supervise.sh cnx-default-jax-4gpu
    for w in edac_watch epoch_clock; do systemd-run --user --unit=cnxj-${w//_/-} --working-directory="$PWD" \
      bash runs/2026-10-21-cnx-full-jax-bf16-300ep/$w.sh; done
    systemd-run --user --unit=cnx-chain --working-directory="$PWD" --setenv=PATH="$PATH" \
      bash runs/2026-10-21-cnx-verified-bf16-300ep/chain_after_jax.sh

⛔ **`XLA_PYTHON_CLIENT_MEM_FRACTION=0.97` is required on this box.** The conf was smoked on ares
(4× 4060 Ti, 16 GB) at XLA's default 0.75 arena. On the 3060's 12 GB that arena is 9 GB, and the
accumulating trainer's first `jit_train_step` asked for one 5.92 GiB buffer and died
RESOURCE_EXHAUSTED twice (2026-10-21 13:23 UTC launch; logs in `oom_default_arena/`). A probe at
0.97 on mars trained 200 optimizer steps at 2.87 s a step (ares smoke: 2.83 s), 11.86 GB in use a
card. The conf itself is unchanged; the fraction rides on the systemd unit. **A relaunch after a stop
or a move must pass it again**, and the chain must be relaunched with it (it only fires after
✅ COMPLETE). The verified half does NOT need it (its probe peaks at 4.46 GiB) and the chain does
not set it.

Resume: `scripts/lib/jax_job.sh` restarts from the newest `<base>_e<N>.state.npz` in
`/home/skoonce/cnx_t_jax_2611/` (weights, AdamW moments, EMA, global step).
