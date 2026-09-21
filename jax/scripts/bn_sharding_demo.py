#!/usr/bin/env python3
"""BN-sharding side-by-side: the memory tax of a small GPU, made visible.

When a batch is too big for one GPU's memory you split it across devices (or,
here, across logical shards). BatchNorm then has a choice:

  * SYNCED  (GHOST_BN_GROUPS=1) — all-reduce the shards, normalize over the WHOLE
    batch. One big, low-variance BN group. This is what a big-memory GPU (A100)
    does for free: the whole batch lives on one device, no split, no sync.

  * SHARDED (GHOST_BN_GROUPS=k) — each of k shards normalizes over its OWN slice
    (batch/k images), no cross-shard reduce. This is "Ghost-BN": what a 16 GB
    card is FORCED into when the batch won't fit. k smaller, higher-variance
    groups, and on real multi-GPU it also costs the all-reduce you skipped.

The irony the demo makes concrete: more memory → keep the batch whole → *less*
math (no sync) AND *better* top-1 (bigger BN group). Sharding is a penalty you
pay for small memory, not a technique you reach for.

We emulate k shards on ONE device by reshaping the batch [N,C,H,W] into
[k, N/k, C,H,W] and reducing per group — bit-identical to what k physical shards
would compute, and identical whether run on a 4060 Ti or an A100. The single BN
chokepoint `conv_bn` is monkeypatched, so all 53 BN sites pick up the knob with
zero edits to the Lean-generated model.

Run one config (this is the exact A100 command — vary the number):

  CUDA_VISIBLE_DEVICES=0 GHOST_BN_GROUPS=1 python jax/scripts/bn_sharding_demo.py
  CUDA_VISIBLE_DEVICES=0 GHOST_BN_GROUPS=4 python jax/scripts/bn_sharding_demo.py

Or run the side-by-side driver (spawns one clean subprocess per config so each
gets a fresh JIT trace of its own BN), prints a comparison table:

  CUDA_VISIBLE_DEVICES=0 python jax/scripts/bn_sharding_demo.py

Env: GHOST_BN_GROUPS (set => single run), EPOCHS (default 80), GROUPS_SWEEP
(driver, default "1 4"). Keep it on ONE device — this is a BN-math demo, not a
throughput demo (see `lake run benchmark` for throughput).
"""
import importlib.util
import json
import os
import subprocess
import sys
import time

# ⚠⚠ WHICH PROBE — this decides what the measured gap MEANS (2026-09-20).
# The two probes differ by exactly one line: the `_noaug` one drops `augment_batch`
# (the pad-252/crop-224 random crop) and keeps only the horizontal flip. `conv_bn` is
# byte-identical between them. `_noaug` is the throughput model `probe/benchmark.py`
# uses, and R50's 23.5M params on 9,469 images without a crop OVERFIT: 38.7% top-1 at
# 80 epochs, eval loss 5.18 against ln 10 = 2.30. In that regime BN noise is the only
# regulariser in the room, so the BN-group gap it reports is inflated and does not
# transfer — the reference recipes all run `augment := true`, where augmentation is
# already doing that job. Default is therefore the AUGMENTED probe; PROBE_NOAUG=1
# reproduces the original demo.
PROBE = os.path.join(os.path.dirname(__file__), "..", "probe",
                     "probe_resnet50_imagenette_noaug.py"
                     if os.environ.get("PROBE_NOAUG")
                     else "probe_resnet50_imagenette.py")
AUGMENT = not os.environ.get("PROBE_NOAUG")
RESULT_TAG = "BN_SHARDING_RESULT"   # parseable line the driver greps for


def load_module():
    spec = importlib.util.spec_from_file_location("r50_probe", PROBE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_grouped_conv_bn(m, groups, train_batch):
    """conv_bn with the TRAINING batch split into `groups` per-shard BN groups.

    groups=1 reproduces the model's stock global BN (mean/var over axis (0,2,3)).
    groups=k reshapes [N,C,H,W] -> [k, N/k, C,H,W] and reduces over (1,3,4), i.e.
    each shard normalizes over its own N/k images — Ghost-BN with k shards.

    ⚠⚠ **EVAL IS HELD AT GLOBAL BN whatever the training grouping** (2026-09-20).
    The probe has no `training` flag — `eval_batch` -> `forward` -> this same
    `conv_bn`, at eval batch 512. Left alone, `groups=k` moved the EVAL group too
    (512/k), so each config differed from its neighbour in two places at once and
    the sweep measured train-group and eval-group together, which is not a number
    anyone can read. Both ImageNet paths score from running statistics that
    estimate the GLOBAL mean/var (the verified driver EMAs its shard's statistics
    over hundreds of steps), so global is the faithful eval and the TRAINING group
    is left as the one variable. `N` is a static shape at trace time, so this is a
    Python branch resolved during tracing, not a traced select.
    """
    import jax
    import jax.numpy as jnp

    def conv_bn(x, w, gamma, beta, stride=(1, 1), padding='SAME'):
        x = jax.lax.conv_general_dilated(
            m.convdt(x), m.convdt(w), stride, padding,
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')).astype(jnp.float32)
        N, C, H, W = x.shape
        g = groups if N == train_batch else 1
        if g > 1:
            assert N % g == 0, f"batch {N} not divisible by groups {g}"
            xg = x.reshape(g, N // g, C, H, W)
            mean = jnp.mean(xg, axis=(1, 3, 4), keepdims=True)
            var = jnp.var(xg, axis=(1, 3, 4), keepdims=True)
            xg = (xg - mean) / jnp.sqrt(var + 1e-5)
            x = xg.reshape(N, C, H, W)
        else:
            mean = jnp.mean(x, axis=(0, 2, 3), keepdims=True)
            var = jnp.var(x, axis=(0, 2, 3), keepdims=True)
            x = (x - mean) / jnp.sqrt(var + 1e-5)
        return x * gamma.reshape(1, -1, 1, 1) + beta.reshape(1, -1, 1, 1)

    return conv_bn


def run_single(groups, epochs):
    """Train Imagenette R50 with `groups` BN shards; return final metrics."""
    m = load_module()

    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax import random

    n_dev = m.n_devices
    print(f"[groups={groups}] devices={jax.devices()}  n_devices={n_dev}")
    # ⛔⛔ ABORT ON A SILENT CPU FALLBACK (2026-09-20). Seen for real: with four of these
    # starting at once, one lost the CUDA init ("a CUDA-enabled jaxlib is not installed.
    # Falling back to cpu"), then sat at 1316% CPU and 56 GB of host RAM — starving the
    # THREE GPU runs beside it down to 0% GPU, and heading for a "result" that was never
    # going to finish. A run that scores on CPU is not a slower run, it is a fake one.
    if jax.devices()[0].platform != "gpu" and not os.environ.get("ALLOW_CPU"):
        raise SystemExit(
            f"[groups={groups}] ⛔ backend is {jax.devices()[0].platform!r}, not 'gpu'. "
            "Refusing to run: CPU fallback starves the co-running GPU jobs and its "
            "number is not comparable. Re-run this config alone, or set ALLOW_CPU=1.")
    if n_dev != 1:
        print(f"[groups={groups}] WARNING: run on ONE device (CUDA_VISIBLE_DEVICES=0); "
              f"sharded reshape + P('batch') interact. Continuing anyway.")

    data_dir = "data/imagenette"
    tr_x, tr_y = m.load_imagenette(os.path.join(data_dir, "train.bin"))
    te_x, te_y = m.load_imagenette(os.path.join(data_dir, "val.bin"))

    BATCH = (192 // n_dev) * n_dev or n_dev
    assert BATCH % groups == 0, f"batch {BATCH} not divisible by groups {groups}"
    # Patch the single BN chokepoint. Still BEFORE any jit trace fires — decorating
    # `@jit` does not trace, and the first call is in the epoch loop below.
    m.conv_bn = make_grouped_conv_bn(m, groups, BATCH)
    LR = 0.001
    WARMUP = 3

    # Fixed init + fixed data order => the ONLY difference between configs is BN.
    # SEED shifts BOTH together, so configs stay paired within a seed and the
    # per-seed difference is the estimator; unset reproduces 314159/42 exactly.
    seed = int(os.environ.get("SEED", "0"))
    params = m.init_params(random.PRNGKey(314159 + seed))
    params = jax.device_put(params, m.replicated_sharding)
    opt_state = (jax.tree.map(jnp.zeros_like, params),
                 jax.tree.map(jnp.zeros_like, params), jnp.float32(0))
    rng = np.random.RandomState(42 + seed)

    t0 = time.time()
    acc1 = acc5 = 0.0
    for epoch in range(epochs):
        if epoch < WARMUP:
            lr = jnp.float32(LR * (epoch + 1) / WARMUP)
        else:
            lr = jnp.float32(LR * 0.5 * (1 + np.cos(np.pi * (epoch - WARMUP) / (epochs - WARMUP))))
        perm = rng.permutation(len(tr_x))
        sx, sy = tr_x[perm], tr_y[perm]
        flip = rng.random(len(sx)) > 0.5
        imgs = sx.reshape(-1, 3, 224, 224)
        imgs[flip] = imgs[flip, :, :, ::-1]
        sx = imgs.reshape(len(sx), -1)
        for i in range(0, len(tr_x) - BATCH + 1, BATCH):
            # `probe_resnet50_imagenette.py`'s own loop, verbatim: one crop offset per
            # BATCH (not per image), drawn from the same rng, on a copy.
            batch = (m.augment_batch(sx[i:i + BATCH].copy(), rng) if AUGMENT
                     else sx[i:i + BATCH])
            x = jax.device_put(batch, m.data_sharding)
            y = jax.device_put(sy[i:i + BATCH], m.data_sharding)
            params, opt_state, _ = m.train_step(params, opt_state, x, y, lr)
        c1, c5, total, tl = m.evaluate(params, te_x, te_y)
        acc1, acc5 = c1 / total, c5 / total
        print(f"[groups={groups}][Epoch {epoch+1}] top1={acc1:.4f} top5={acc5:.4f} "
              f"loss={tl:.4f} [{time.time()-t0:.1f}s]")

    secs = time.time() - t0
    bn_group = BATCH // groups
    print(f"{RESULT_TAG} " + json.dumps(dict(
        groups=groups, batch=BATCH, bn_group_size=bn_group, seed=seed,
        augment=AUGMENT,
        epochs=epochs, top1=round(acc1, 4), top5=round(acc5, 4),
        seconds=round(secs, 1))))
    return dict(groups=groups, batch=BATCH, bn_group_size=bn_group,
                top1=acc1, top5=acc5, seconds=secs)


def run_driver(epochs):
    sweep = [int(g) for g in os.environ.get("GROUPS_SWEEP", "1 4").split()]
    print(f"=== BN-sharding side-by-side: groups {sweep}, {epochs} epochs each ===\n")
    rows = []
    for g in sweep:
        env = dict(os.environ, GHOST_BN_GROUPS=str(g), EPOCHS=str(epochs))
        env.pop("GROUPS_SWEEP", None)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                              env=env, capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr)
            print(f"!! groups={g} failed (rc={proc.returncode})")
            continue
        line = next((l for l in proc.stdout.splitlines()
                     if l.startswith(RESULT_TAG)), None)
        if line:
            rows.append(json.loads(line[len(RESULT_TAG):].strip()))

    if not rows:
        print("no results")
        return
    print("\n" + "=" * 66)
    print("  BN sharding side-by-side  (Imagenette R50, single device)")
    print("=" * 66)
    print(f"  {'shards':>6} {'BN group':>9} {'top-1':>8} {'top-5':>8} {'sec':>7}   note")
    base1 = next((r["top1"] for r in rows if r["groups"] == 1), None)
    for r in rows:
        note = "synced / big-memory (A100)" if r["groups"] == 1 else "Ghost-BN / small-memory"
        delta = ""
        if base1 is not None and r["groups"] != 1:
            delta = f"  Δtop1={100*(r['top1']-base1):+.2f}pt vs synced"
        print(f"  {r['groups']:>6} {r['bn_group_size']:>9} {r['top1']:>8.4f} "
              f"{r['top5']:>8.4f} {r['seconds']:>7.1f}   {note}{delta}")
    print("=" * 66)
    print("  more memory -> keep batch whole -> no sync + bigger BN group -> faster & better")


if __name__ == "__main__":
    epochs = int(os.environ.get("EPOCHS", "80"))
    if "GHOST_BN_GROUPS" in os.environ:
        run_single(int(os.environ["GHOST_BN_GROUPS"]), epochs)
    else:
        run_driver(epochs)
