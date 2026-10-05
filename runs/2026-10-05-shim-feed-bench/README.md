# 2026-10-05 — shim producer bench at the jobs' shape

CPU only, GPUs idle, DIMM fans in. `bench.py` spawns N generated shims with `spawnShim`'s env
(SHIM_BATCH, SHIM_SEED = 1+i, SHIM_SHARD = i/N, SHIM_NCLASSES for soft targets) and reads the wire.
`--mode rr` reads batch k from producer k mod N, one at a time (the trainer's round-robin, no
prefetch); `--mode free` gives each producer its own reader thread. 45 s warm, 90 s on the clock.
`matrix.sh` → `results.txt`.

| config | img/s | ms per 512 | job measured (§1) |
|---|---|---|---|
| ViT 128, 4, rr | 822 | 623 | |
| **ViT 512, 4, rr (the job)** | **831** | **616** | mean 609 |
| ViT 512, 4, free | 1,220 | 420 | |
| ViT 512, 4, rr, mix off | 1,113 | 460 | |
| ViT 128, 4, rr, mix off | 1,120 | 457 | |
| ViT 512, 6, rr | 795 | 644 | |
| MNv4 128, 4, rr | 1,764 | 290 | |
| **MNv4 512, 4, rr (the job)** | **1,758** | **291** | mean 329 |
| MNv4 512, 4, free | 1,637 | 313 | |

What it says:
- **The bench reproduces both jobs.** ViT-S verified is the producers' rate (616 vs 609). MNv4's
  producers give 291 against the job's 329: the rest is the trainer competing for the same cores.
- **Batch size is not the problem.** 128 and 512 read the same in every pairing.
- **ViT: mixup and the round-robin coupling are each about a third.** Mix off: 831 → 1,113 (+34%,
  §4.2's number again). Free drain: 831 → 1,220 (+47%). `_emit` runs mixup on the producer's main
  thread and then blocks in `write()` on a 308 MB record (the pipe holds 64 KB) until the trainer
  reaches that producer; tf.data's own threads keep going, the mixing does not. MNv4 does not mix,
  and rr ≈ free there.
- **MNv4 is plain CPU.** Every producer config sits at ~1,650–1,760 img/s with the box saturated (the
  ViT run held ~29 of 32 threads). Only less CPU per image moves it.
- **More producers make it worse** (n6 795 < n4 831).

Against §4.2 (09-28): every config reads 55–87% of that day's number on the same shim (unchanged
since before 09-28). Clocks are normal (4.45 GHz all-core at 85 °C) and memory is better than before
the re-slot (`membw.txt`: 1 stream 33.4 GB/s read+write, 8 streams 40.0 — 09-14 had 13.9 one-way
for 8, now 20). The 09-28 harness was in a scratchpad and is gone, so the gap is not explained;
today's numbers are the ones that match the jobs.

DIMMs are now in channels 0, 1, 6, 7 (EDAC; were 4–7), 0 corrected errors.

## The fix and after (same day)

`Jax/Codegen.lean`'s shim template: `_mix_rows` mixes in place a block of row pairs at a time (each
element the same float32 expression as the whole-array form), and `_writer` moves the pipe write to
its own thread behind a one-batch queue, writing `memoryview`s instead of `tobytes()` copies. All 37
shims re-emitted (`scripts/regen_jax_generated.sh`, then `sync`); the pre-change copies are in `old/`.

- `identity.py` → `identity.txt`: old and new SHIM_HASH digests and streamed wire bytes match on ViT
  (B = 63 odd, 64), ConvNeXt-S, R50 A2 (the BCE threshold path) and MNv4; control: mix on vs off
  differs.
- `gates.txt`: `mixup_gate.py` known-answer checks bit-exact, `bce_target_gate.py` passes. The mixup
  gate's two pinned "inert when off" digests fail on the OLD R34 shim with the same digests as the
  new one, so the baseline was already stale.
- `matrix2.sh` → `results2.txt`: ViT 512 × 4 rr 831 → **1,410 img/s** (616 → 363 ms per 512), free
  1,381; MNv4 rr 1,758 → 1,862.
- `reprobe.sh` (GPU, the §3a smoke): ViT-S verified mean **609 → 379** ms/step (starved 305 → 79 ms),
  MNv4 verified 329 → 318 (starved 109 ms: CPU, not the coupling).
