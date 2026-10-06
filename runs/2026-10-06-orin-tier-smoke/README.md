# `lake run imagenette-orin` — the engine's relaunch loop, exercised on the desktop

`deploy/orin_imagenette.sh` is the row engine behind `lake run imagenette-orin`: the Orin's
Imagenette recipe (`LEAN_MLIR_IMAGENETTE_STREAM=1 LEAN_MLIR_PREALLOCATE=1`, the row's
`LEAN_MLIR_MEM_FRACTION`), the process under a `systemd-run --user --scope` memory cap, and a
relaunch that resumes from the checkpoint while each attempt completes an epoch. Before the
board runs it, the loop's three outcomes were forced here, on one RTX 4060 Ti under the box's
plugin, with `vit-verified-adam` at pool fraction 0.25 and the checkpoint tag `orinsmoke`:

1. **Completes.** `EPOCHS=3 LEAN_MLIR_G2_STEPS=40`: one attempt, `done` at epoch 3, rc 0, 24 s.
2. **Killed after a completed epoch → relaunch.** `EPOCHS=3`, full epochs; a watcher sent
   SIGTERM to the trainer three seconds after the epoch-2 checkpoint landed (during epoch 3).
   Attempt 1 ended rc 143 at epoch 2; attempt 2 printed `resuming from checkpoint at epoch 2`,
   trained epoch 3 and finished rc 0 — `runner.log` below.
3. **Killed before any epoch → stop.** `MEMCAP=700M EPOCHS=1 LEAN_MLIR_G2_STEPS=5`: the
   cgroup killed the trainer at start-up (rc 137, no `.epoch`), and the engine stopped after
   that one attempt with the log's tail, rather than relaunching into the same failure.

The accuracies are three-epoch numbers from a smoke and mean nothing; the test is the loop.

## runner.log of outcome 2

```
2026-10-06 03:49:54 start vit-verified-adam fraction 0.25 tag orinsmoke epochs 3 gpu 0 clocks n/a cap 5800M
2026-10-06 03:49:54 attempt 1 from epoch 0 → runs/…/attempt1.log
2026-10-06 03:50:42 attempt 1 ended rc 143 at epoch 2 (was 0) — relaunching from the checkpoint
2026-10-06 03:50:42 attempt 2 from epoch 2 → runs/…/attempt2.log
2026-10-06 03:51:10 ✓ done: rc 0 at epoch 3 after 2 attempt(s); val_acc = 1722/3925 = 43.872611%  top5 = 3428/3925 = 87.337580%  [95% CI 42.33–45.43]
```

## runner.log of outcome 3

```
2026-10-06 03:51:27 start vit-verified-adam fraction 0.25 tag orinsmoke3 epochs 1 gpu 0 clocks n/a cap 700M
2026-10-06 03:51:27 attempt 1 from epoch 0 → runs/…/attempt1.log
2026-10-06 03:51:30 ✗ attempt 1 ended rc 137 without completing an epoch (still 0) — stopping; tail of runs/…/attempt1.log:
I1006 03:51:27.289622  677126 gpu_helpers.cc:162] XLA backend allocating 3.89GiB (4181721088 bytes) on device 0 for BFCAllocator.
I1006 03:51:27.296392  677126 cuda_dnn.cc:440] Loaded cuDNN version 92302
[pjrt_ffi] XLA backend: PJRT 0.114, 1 device(s)
[pjrt_ffi] command buffers: enabled (CUDA default)
[pjrt_ffi] residency: on — parameters stay on the device between steps (default)
```
