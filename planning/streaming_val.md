# Streaming ImageNet val — stop holding 30 GB of val inside the trainer

**Written 2026-09-22**, during the ConvNeXt-T verified run's last epochs. Nothing is built yet.
Brett's scope: **lossless only**, **2 val producers** (the cores sit idle during eval anyway).

---

## 0. Why, in numbers

* **The trainer holds the whole val split decoded to float32**: 50,000 × 150,528 × 4 B =
  **30.1 GB (28.0 GiB)**, drained once at startup (`LeanMlir/VerifiedTrain.lean`, `loadData`'s
  `.imagenet` branch, ~999–1083). Trainer `RssAnon` sits at 29–31 GiB, and nearly all of it is
  that buffer.
* **The box lost 16 GiB on 2026-09-21.** A failing DIMM crashed it, and at the next boot the
  firmware mapped out exactly 16,777,216 K (memory: `box-failing-dimm`). The page cache now tops
  out at ~114–117 GiB, but the **train split is 137 GiB** (val 6.3 GiB). So the loaders read
  **100–170 MB/s from disk every epoch** and ConvNeXt-T epochs went **1,096 → 1,178 s (+7.5%)**.
  That isn't iowait; it's feed cost.
* **Even at 188 GiB it was barely fitting.** The kernel was swapping ~8 GiB of the trainer's cold
  pages to squeeze the train split into cache, which was the "swap keeps growing" line in every
  status report.
* **Freeing the val buffer returns ~28 GiB to the cache**: ~114 + 28 ≈ 142 GiB against 137. It
  just about fits on the degraded box, and fits comfortably once the stick is replaced. Cheaper
  than DDR.

## 1. Constraints — decided, do not re-open

1. **Lossless.** No uint8, fp16 or bf16 storage. The shim's val path decodes the JPEG to uint8,
   then runs a **bicubic + antialias** resize, which leaves *fractional, unclipped* float32 pixels.
   Rounding those moves the eval off today's bytes and off the JAX reference's (same float path),
   in a pair built to read offsets of a few tenths. Brett: *"no lossy approaches … this is tricky
   enough to be consistent without making things harder for ourselves."*
2. **Same 50,000 images in the same order as today's single-producer drain.** Then every
   per-image bitmap (`LEAN_MLIR_DUMP_CORRECT`, McNemar, `sharded_eval_gate.sh`) stays comparable
   across the change. ⚠ The order is NOT free (see §3.1).
3. **2 val producers.** Brett, 2026-09-22.
4. **ImageNet only.** MNIST, CIFAR and Imagenette keep their in-memory eval (≤ ~2 GB and not
   shim-fed).
5. **The JAX reference trainers must regenerate BYTE-IDENTICAL.** This changes the verified
   path's val FEED, never the reference.
6. **No new always-on env var** (the eval-path rule from the sharded-eval handoff). Gate-only
   fault knobs are fine, in the `PJRT_FFI_FAULT` style.

## 2. Measure first — CPU-only, minutes, after the ConvNeXt run lands

* **Val producer throughput at 1, 2 and 4 producers**: stream `validation` to nowhere and time
  50,000 images. ⚠ Today's figure, **~1,250 img/s for ONE producer, is INFERRED**, not timed: the
  ViT/ImageNet gate's 4-replica arms took 55–57 s end to end, ≈ 9 s eval + ~5–10 s
  startup/compile ⇒ ~40 s drain. Eval consumes **~1,700 img/s** (ConvNeXt: 50k in ~29 s).
  **Prediction:** 1 producer ⇒ ~40 s eval (+~10 s/epoch); 2 ⇒ hidden (+0–2 s).
* **The pipe read of one 154 MB batch** (256 × 150,528 × 4). That decides whether eval needs
  read-ahead (§3.2). At a few GB/s it is ~30–50 ms × 196 batches ≈ 6–10 s/epoch if SERIAL with
  compute.

## 3. Design

### 3.1 Shim: batch-block sharding for the val split (emitter change)

⚠⚠ **The existing sharding would scramble the order.** `SHIM_SHARD=i/N` is ELEMENT-level
`ds.shard(num_shards=N, index=i)` (emitted at `jax/Jax/Codegen.lean:470`): producer i gets
elements i, i+N, i+2N, … The trainer reads round-robin BY BATCH, so val would come back in a
different order from the single-producer drain. That's harmless for train, which is shuffled
anyway, and breaks constraint 2 for val.

**Val-only batch-block shard, on the raw records, before the decode map:** producer k keeps
record j iff `(j // 256) % N == k`. For example: `ds.enumerate().filter(λ j,_: (j // 256) % N ==
k).map(λ _,ex: ex)` on the `SkipDecoding` dataset.
* **Order is reproduced exactly.** Reading global batch b from producer `b % N` gives
  single-producer order, including the tail. 50,000 = 195 × 256 + **80**; the 80-row batch 195
  goes to producer `195 % N` (producer 1 at N = 2, each producer carrying 98 batches).
* **Decode is split N ways.** Each producer READS all 6.3 GiB of raw records, which is cheap from
  cache or NVMe, and DECODES only 1/N.
* **Keep the map ordered.** The tf.data default is order-preserving, and nothing in the shim sets
  `deterministic=False`. Op determinism is irrelevant here because the center-crop path has no
  random ops.
* **Emit it from `generateShim` only** (`jax/Jax/Codegen.lean:3440`), NOT from the shared
  `build_imagenet_iter` fragment (`:451`). That fragment is also emitted into every
  `jax/generated/*_full.py`, and those must regenerate byte-identical (constraint 5). The
  reference's own val iterator (`generated_convnext_tiny_imagenet_full.py:1819`) passes no shard
  anyway.
* **The train path does not move.**
* **Regenerate** with `scripts/gen_shims.sh`, the ONE writer. Then run
  `scripts/regen_jax_generated.sh box` and `scripts/shim_wiring_gate.py`.

### 3.2 Trainer: stream val every epoch

* **Replace the drain** (`loadData`, `.imagenet`, ~1016–1083) with a per-epoch val stream. Keep
  the pieces that carry lessons:
  * the **50,000 denominator assert and announcement** (the 49,920 → 50,000 fix, 2026-08-14);
    it is now asserted EVERY epoch, and a short pass is a refusal;
  * the **eval-width** read off the artifact (`evalD0`, the RSB-A3 160/224 split);
  * the **`LEAN_MLIR_EVAL_BATCHSTATS` tail drop**, which moves into the reader;
  * **reaping** (kill then wait; the `<defunct>` lesson at ~1068).
* **Spawn per epoch, not long-lived.** Use `spawnShimSharded net.shimScript "validation" 256
  evalD0 0 2` with the §3.1 shard mode. Spawn during the last ~K train steps so TF/tfds startup
  (~5–10 s) and the first batches overlap training; the producers park in `write()` once their
  prefetch is full.
  * This keeps today's **closed pipe = end of pass** semantics.
  * Every eval gets fresh processes, so no leak can accumulate (the EfficientNet loader lesson).
  * Nothing interacts with the train-loader respawn.
  * ✗ The rejected alternative is long-lived `.repeat()` producers: they need epoch framing,
    tangle with the respawn, and accumulate.
* **ONE eval loop still.** `evalScore` must stay the single loop the trainer and
  `scoreCheckpoint` share, because the sharded-eval gate rests on it. Replace its `evalImg`/
  `evalLbl` arguments with a ROW SOURCE:
  * a held buffer for the small datasets (today's slicing, unchanged);
  * the stream for ImageNet, accumulating 256-row shim batches into the global eval batch
    `gB = R × evalBs`. That is 1 shim batch for ConvNeXt (4×64) and 4 for ViT (4×256). The tail is
    zero-padded and only the real rows are scored, exactly as now.
* **Read-ahead.** Pull batch k+1 while the invoke for k runs (`IO.asTask`, like train's depth-n
  prefetch). Without it, ~30 GB of pipe reads per epoch serialize with compute (§2).
* **`scoreCheckpoint`** gets the same single-pass stream.
* **Expected:** trainer RSS **~30 → ~2 GiB**; no 40 s drain at startup; eval window unchanged at
  **~29–31 s** with 2 producers.

## 4. The gate — before any run uses it

Same shape as `scripts/sharded_eval_gate.sh`: an EQUALITY, with controls that must go red.
Deterministic shim (`GATE_DET=1`), and `SHIM_DETERMINISM=1` per the shim's own rule ("a NEW
byte-identity gate MUST set it").

1. **Golden, BEFORE touching anything.** Using TODAY's binary, record the drained eval's line and
   per-image bitmap on:
   * the ViT/ImageNet e300 checkpoint: det shim 36176/45610; the production shim reproduces the
     run's own line, 36175/45608;
   * the ConvNeXt-T final checkpoint, once it lands.

   Keep them in `runs/<date>-streamed-val/golden/`. ⚠ The 09-18 gate bitmaps lived in `/tmp` and
   did not survive the 09-21 reboot.
2. **Test.** Streamed `score-checkpoint` at R = 1 and R = 4 must equal golden: count, top-5 and
   every bit.
3. **Controls (must FAIL):**
   * **order fault**: the round-robin starts at producer 1. The COUNT is equal and the BITMAP
     differs. This is exactly the defect a count-only gate cannot see, which is why the bitmap is
     the check.
   * **tail fault**: drop the 80-row batch. The count reads 49,920; that's the C4 bug class.
4. **Trainer end to end.** A one-epoch smoke with a tagged checkpoint, as on 2026-09-18: the
   in-run bitmap must equal `score-checkpoint`'s bitmap on that checkpoint. Measure the eval
   window and RSS there.

## 5. Payoff to measure, not assume

| quantity | today | expected |
|---|---|---|
| trainer `RssAnon` | 29–31 GiB | ~2 GiB |
| eval window (ConvNeXt-T) | 28–31 s | 29–31 s (2 producers) |
| page cache during train | 114–117 GiB | ~142 GiB |
| disk reads during train | 100–170 MB/s | ≈ 0 after epoch 1 warms the cache |
| ConvNeXt-T epoch, degraded box | 1,178 s | ~1,100 s |

## 6. Risks and open points

* **§2's throughput is inferred.** If 2 producers can't reach ~1,700 img/s, use 3; the box has 24
  threads and eval is when the train loaders are parked.
* **Spawn timing.** Spawning AT eval start costs TF startup (~5–10 s) every epoch. The pre-spawn
  point K needs one measurement.
* **Watchers.** `loader_rss.sh` pgreps the shim filename, so the short-lived val producers would
  pollute `loader_rss.tsv`. Filter on `SHIM_SPLIT`/argv in the template.
* **tfds val order** must be stable across versions. It is fixed by the dataset's file order, and
  `shuffle_files` defaults to False. The golden bitmap in §4 is what would catch a change.
* **The 142-vs-137 GiB margin is thin on the degraded box.** Streamed val also wants 6.3 GiB of
  cache each epoch, so expect *most* of the disk reads to vanish, not necessarily all. With the
  stick replaced there's ample headroom.

## 7. Files

`jax/Jax/Codegen.lean` (`generateShim`) · `jax/generated/*_shim.py` (regenerated, never
hand-edited) · `LeanMlir/VerifiedTrain.lean` (`loadData`, `evalScore`, the `trainAdamSched` eval
block, `scoreCheckpoint`) · a new gate script · `runs/<date>-streamed-val/`.
