# trainium.md — AWS Trainium (trn1) as a platform: where it stands, what's next

Started 2026-10-30. Three sessions so far (2026-10-29/30 night, 2026-10-30 day, 2026-10-30 afternoon on
atlas). Work lives on branch `trainium`, which is not pushed; local `main` was moved back to da5a1404 on
2026-10-30, so the Trainium commits are on this branch only. Run records: `runs/2026-10-30-trainium-setup/README.md`,
`runs/2026-10-30-trainium-imagenette/README.md`, `runs/2026-10-30-trainium-upstream/README.md`,
`runs/2026-10-30-trainium-wgrad/README.md`.

## 1. Where it stands

- **Works, matches CUDA:** MNIST (92.10 / 97.82 / 98.73) and ViT-Tiny on Imagenette — with the rewritten
  patch-embed gradient, 80 epochs in 34.5 min at 69.27 % (the Arc B50: 68.46 %).
- **The convnets after the renderer fix (2026-10-30 evening, `runs/2026-10-30-trainium-wgrad/`):** ResNet-34
  b16 trains at 478 ms (was 491; stock JAX 349) and passes the CPU check — the dilation was not its problem;
  a device profile says it is DMA-bound (88 % DMA-active, 1.6× stock JAX's DMA cycles on the same bytes).
  ResNet-50 b32 f32 compiles and runs for the first time (659 ms) but fails the CPU check by a median 5 %,
  unexplained. ConvNeXt-T (8.6M), B0 (5.04M) and MobileNetV2 (5.05M) still exceed the 5M-instruction cap at
  batch 32; batch-16 renders are the next try.
- **Broken by Neuron:**
  - CIFAR and every CNN with max pooling: the max-pool gradient (`select_and_scatter`) is wrong, and so
    is a 2× upsample feeding a weight-gradient conv.
  - ResNet-34 and -50 at batch 32: the f32 graph fails `neuronx-cc` (`NCC_IBIR243`); the bf16 graph
    exceeds the runtime's DMA descriptor limit when loaded.
  - ResNet-34 at batch 16 (scratch renders `resnet34_adam16{,bf16}`, uncommitted) trains, but on wrong
    pool gradients.
- **Speed:** the verified ViT runs 1,281 ms/step (733 in bf16). **Stock JAX ViT on the same core runs 82.5
  (63.8 in bf16)**, about the 4060 Ti's 61. ResNet-34 b16: verified 494, stock JAX 349.
- **The 15× was one op, and it is fixed in a scratch render** (`runs/2026-10-30-trainium-wgrad/`): the
  patch-embed weight gradient was rendered as a conv whose kernel is the token cotangent padded
  `interior = [0, 0, 15, 15]` to 209×209 — 206 of the step's 325 GMAC against zeros, a 1 GB operand, 92 %
  of the compiled step's DMA blocks. Rewritten as a `rhs_dilate = [16, 16]` conv (form B) or an im2col
  matmul (form C), **the f32 step runs 71 ms on one NeuronCore (was 1,281; stock JAX 82.5), 54 ms with
  bf16 matmuls (JAX 63.8)**, and compiles in 400 s on atlas instead of 2,000. Neuron also computed the
  old conv wrong: the patch-embed Adam moment was 72 % off the CPU golden; the rewrite is within 1e-4.
  The scratch renders `verified_mlir/vit_adamwg{B,C}_train_step.mlir` are uncommitted; the renderer fix
  is item 1 below.
- **Compiler flags:** `-O1`, `-O3` and `--model-type=transformer` change nothing (bit-identical results
  and timing). `-O1` makes the DMA spills worse. tf32 auto-cast crashes the compiler. Only bf16 moves
  the step time.

## 2. Next session

Item 1 is the user's call (it moves pinned renders). Item 2 is the first thing to do once the instance is up.

1. ~~Fix the patch-embed weight gradient in the renderer~~ **done 2026-10-30, and wider than ViT**: every
   strided-conv weight and input gradient now carries its dilation on the conv (`rhs_dilate` / `lhs_dilate`)
   instead of a materialized pad — 16 templates in `Pretty.lean`, 252 renders re-emitted, no proof moved
   (`runs/2026-10-30-trainium-wgrad/README.md`, "The renderer change"). The grouped (depthwise) weight grads
   keep the pad: neuronx-cc rejects `rhs_dilate` with `batch_group_count > 1` (upstream issue 6).
   - Still to do for the corpus: the tier-2 goldens if `goldens --check` moved, `verified_mlir/MANIFEST.md`,
     the CUDA gate on a GPU box (the renders changed bytes; the math did not), and a CUDA timing of
     `vit_adam_train_step` before/after — XLA:CPU keeps the 209×209 kernel, so the 4060 Ti likely paid too.
   - Delete the scratch renders once the instance has tested the real ones: `vit_adamwg{B,C}`, `*dil`.
2. ~~ViT-Tiny on real Imagenette with the rewrite~~ **done 2026-10-30**: 80 epochs in 34.5 min on one
   NeuronCore, 69.27 % top-1 (the Arc B50: 39 min, 68.46 %). `runs/2026-10-30-trainium-wgrad/trn1/full.log`.
3. **File the six upstream issues, on the user's okay** (1–4 from the first sessions; 5, the dilated-kernel conv computing wrong values, and 6, the depthwise kernel-grad compile failure, from 10-30, both reproduced in plain JAX). Drafts and scripts are in
   `runs/2026-10-30-trainium-upstream/`. They go to `aws-neuron/aws-neuron-sdk`, which is public. Then
   wait about a week for a response.
4. **Benchmark a full stock-JAX run on real Imagenette, alone on the box.** This is the cheapest
   win: if the repo's own JAX trainer trains at ~80 ms/step on Trainium, the JAX path is the
   platform's fast path today. `runs/2026-10-30-trainium-imagenette/jax/generated_vit_tiny.py` with
   `JAX_BATCH=32 JAX_EPOCHS=1`, then a few epochs if it holds.
   - It was OOM-killed sharing the 30 GB host with another trainer, so nothing else runs alongside it.
   - Warm its graphs on atlas first. The trainer jits its own steps, so either add an emit-and-lower
     hook or let the instance compile once (about 6 min per graph there).
   - Record epoch time, accuracy and ms/step next to the verified ViT's 1,281 and the 4060 Ti's 61.
   - Then try both cores: the trainer shards over every visible device.
   - ViT only for now. The CNN trainers' max-pool backward is broken on Neuron (issue 1).
5. **Use both cores.** `neuron_warm.c` compiles for one device, so `vit_adamdp_train_step` (all_reduce)
   fails to parse there. Add a two-replica compile, warm it, and time ViT on 2 cores.
6. **CIFAR workaround (lower priority while the upstream bugs are open):** use the `dot_general`
   upsample only at small spatial sizes. The full v2 rewrite never finished compiling.
7. **ResNet-34: find the DMA shape.** `neuron-explorer capture` then `view --output-format json` on the
   dilated b16 NEFF (`MODULE_9809179191403821896` in `/var/tmp/neuron-compile-cache-dil`) and on stock JAX's
   (`MODULE_13503892209821715641` in `/var/tmp/neuron-jax-cache`): which HLO ops own the DMA time. The
   summaries are in `runs/2026-10-30-trainium-wgrad/trn1/`; the `.ntff` captures are in `~/wgrad/` on the
   instance. Suspects: the max-pool backward's nine shifted pads summed, the flat `[B, c·h·w]` relayouts
   around every pointwise op (`Pretty.lean` § "Pointwise ops at their 4-D shape"), BN reductions.
8. **ResNet-50 b32 numerics on Neuron.** Median 5e-2 against the CPU golden where ResNet-34 b16 is 4e-3.
   Bisect with `tier2.py`-style slices (`runs/2026-10-30-trainium-setup/{cut,multi,slice}.py`); first
   check whether `resnet50_acc2x16` (which trained) passes the same check.
9. **Batch-16 renders for ConvNeXt-T, B0 and MobileNetV2** so they fit under the 5M-instruction cap;
   B0 and MobileNetV2 are 1 % over at batch 32.
10. **ResNet-34 batch 16:** if Trainium stays, give `resnet34_adam16{,bf16}` real `#eval` lines in
   `ResNet34RenderB.lean`. Otherwise delete the scratch renders.

## 3. How to run a session

- **AWS login:** the session expires. The user runs `! aws login`. Region us-east-2. Quota is one
  trn1.2xlarge (8 vCPU, 30 GB RAM, 2 NeuronCores), `i-07ab710f38e5acdcf`, about $1.34/h.
- **Start:** `aws ec2 start-instances`, wait, read the new public IP, then `sudo shutdown -h +240` as the
  safety net. SSH from atlas only.
- **Stop:** always `aws ec2 stop-instances` and confirm `stopped`. A background watcher that pulls logs
  and then stops the instance worked well.
- **Compile on atlas first, never on the instance:**
  - Verified graphs: `scripts/platform/neuron_warm.sh <renders>`, with `NEURON_CC_FLAGS` and
    `NEURON_WARM_CACHE` per flag set. The flag string is part of the cache key.
  - JAX: `NEURON_FORCE_PJRT_PLUGIN_REGISTRATION=1 JAX_PLATFORMS=neuron,cpu` plus the fake-device
    environment from `runs/2026-10-30-trainium-upstream/fake_env.sh`, then
    `jit(f).lower(abstract args).compile()`. The load error at the end is expected.
  - Ship the tarball and `cp -a` it into `/var/tmp/neuron-compile-cache/` on the instance.
- **Run:** `runs/2026-10-30-trainium-imagenette/smoke.sh CORE NAME BIN ENV…` runs a 300-step epoch with
  eval, then a 160-step probe. Run one lane per core.
- **Gotchas:**
  - Trainers resume from `.lake/build/<variant>_ckpt_xla.bin` whatever the flags, so move checkpoints
    aside before each run.
  - Run one ResNet-50 (~16 GB RSS) or one in-RAM Imagenette JAX trainer at a time.
  - `pgrep -f` / `pkill -f` patterns match their own shell; use `'[n]ame'`.
