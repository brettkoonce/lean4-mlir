# Gates for the MNv4 50-epoch pair's verified variant (2026-09-25)

`planning/mnv4_half_pair.md` § Gates has the table. These are the logs
and the smoke script:
- `dpcheck_recipe.log`: 4 × 128 against 1 × 128 on a duplicated batch.
- `dpcheck_control.log`: sum-not-mean, red.
- `dpcheck_adamdp64bf16.log`: regression.
- `smoke.sh` / `smoke.log`: the trainer end to end, 2 + 1 short epochs with a resume.
- `jax_smoke.log`: the JAX conf's launch wrapper, stopped at step 100+.
