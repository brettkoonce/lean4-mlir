# trainium.md — AWS Trainium (trn1) as a platform: where it stands, what's next

Started 2026-10-30. Two sessions so far (2026-10-29/30 night, 2026-10-30 day). Work lives on branch
`trainium`, which is not pushed. `main` also carries the first two commits (3d96445f, a859a2c3), local only;
decide whether `main` keeps them. Run records: `runs/2026-10-30-trainium-setup/README.md`,
`runs/2026-10-30-trainium-imagenette/README.md`, `runs/2026-10-30-trainium-upstream/README.md`.

## 1. Where it stands

- **Works, matches CUDA:** MNIST (92.10 / 97.82 / 98.73) and ViT-Tiny on Imagenette (34.62 % after one
  epoch, against the 4060 Ti's 35.18 %).
- **Broken by Neuron:**
  - CIFAR and every CNN with max pooling: the max-pool gradient (`select_and_scatter`) is wrong, and so
    is a 2× upsample feeding a weight-gradient conv.
  - ResNet-34 and -50 at batch 32: the f32 graph fails `neuronx-cc` (`NCC_IBIR243`); the bf16 graph
    exceeds the runtime's DMA descriptor limit when loaded.
  - ResNet-34 at batch 16 (scratch renders `resnet34_adam16{,bf16}`, uncommitted) trains, but on wrong
    pool gradients.
- **Speed:** the verified ViT runs 1,281 ms/step (733 in bf16). **Stock JAX ViT on the same core runs 82.5
  (63.8 in bf16)**, about the 4060 Ti's 61. The cost is in how Neuron handles our graph, not the
  hardware. ResNet-34 b16: verified 494, stock JAX 349.
- **Compiler flags:** `-O1`, `-O3` and `--model-type=transformer` change nothing (bit-identical results
  and timing). `-O1` makes the DMA spills worse. tf32 auto-cast crashes the compiler. Only bf16 moves
  the step time.

## 2. Next session

Item 1 runs on atlas, so no instance is needed. Item 3 is the first thing to do once the instance is up.

1. **ViT: find the 15× (atlas only, no instance).** In op counts, the verified `vit_adam_train_step` has
   2,632 reshapes against stock JAX's 291, 148 slices against 3, and 146 pads against 1. That is likely
   the per-head split (slice + reshape) and its backward (pad). To check:
   - Lower a ViT block both ways and compare which ops Neuron turns into DMA. `neuronx-cc` can dump
     intermediate output, or use `neuron-profile` on the instance.
   - Then try a renderer change that writes the head split the way JAX does (one reshape + transpose,
     `dot_general` with batch dims) and confirm it is still the same math: bit-identical on XLA:CPU,
     like `sas_rewrite.py` does.
   - If the renderer changes, the proofs that cover it need to follow. Scope that before committing to it.
2. **File the four upstream issues, on the user's okay.** Drafts and scripts are in
   `runs/2026-10-30-trainium-upstream/`. They go to `aws-neuron/aws-neuron-sdk`, which is public. Then
   wait about a week for a response.
3. **Benchmark a full stock-JAX run on real Imagenette, alone on the box.** This is the cheapest
   win: if the repo's own JAX trainer trains at ~80 ms/step on Trainium, the JAX path is the
   platform's fast path today. `runs/2026-10-30-trainium-imagenette/jax/generated_vit_tiny.py` with
   `JAX_BATCH=32 JAX_EPOCHS=1`, then a few epochs if it holds.
   - It was OOM-killed sharing the 30 GB host with another trainer, so nothing else runs alongside it.
   - Warm its graphs on atlas first. The trainer jits its own steps, so either add an emit-and-lower
     hook or let the instance compile once (about 6 min per graph there).
   - Record epoch time, accuracy and ms/step next to the verified ViT's 1,281 and the 4060 Ti's 61.
   - Then try both cores: the trainer shards over every visible device.
   - ViT only for now. The CNN trainers' max-pool backward is broken on Neuron (issue 1).
4. **Use both cores.** `neuron_warm.c` compiles for one device, so `vit_adamdp_train_step` (all_reduce)
   fails to parse there. Add a two-replica compile, warm it, and time ViT on 2 cores.
5. **CIFAR workaround (lower priority while the upstream bugs are open):** use the `dot_general`
   upsample only at small spatial sizes. The full v2 rewrite never finished compiling.
6. **ResNet-34 batch 16:** if Trainium stays, give `resnet34_adam16{,bf16}` real `#eval` lines in
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
