# Trainium Imagenette — trn1.2xlarge, 2026-10-30 (Neuron SDK 2.33.0)

Smoke runs of the verified-MLIR trainers (`smoke.sh`: one 300-step epoch with eval, then a 160-step
timing probe), a neuronx-cc flag sweep (`flag_matrix.sh` warms on atlas, `flags_next.sh` runs on the
instance, one lane per NeuronCore), and the repo's own Lean-emitted JAX trainers as a reference
(`jax_bench.py`: `train_step` on random data). Fault repros for Neuron: `../2026-10-30-trainium-upstream/`.

## Verified graphs (ms/step = probe median)

| run | flags | ms/step | epoch-1 val (300 steps) |
|---|---|---|---|
| ViT-Tiny b32 f32 | `--auto-cast=none` (-O2) | 1,281 | 34.62 % |
| ViT-Tiny b32 f32 | -O1 / -O3 / `--model-type=transformer` | 1,288 / 1,281 / 1,281 | 34.62 % (bit-identical) |
| ViT-Tiny b32 | `--auto-cast=matmult --auto-cast-type=bf16` (± -O3, ± transformer) | 733–734 | 32.23 % |
| ResNet-34 b16 f32 (`adam16`) | -O2 / -O3 | 494 / 493 | 19.21 % |
| ResNet-34 b16 (`adam16bf16` render) | -O2 | 435 | 21.6 % |
| ResNet-34 b16 f32 | bf16 auto-cast -O3 | 434 | 13.78 % |
| ResNet-34 b16 f32 | -O1 | — | fails to load: DMA descriptor limit (16,777,200 per ring) |
| ResNet-34 b32 f32 | -O2 | — | fails neuronx-cc, `NCC_IBIR243` |
| ResNet-34 b32 f32 | -O1 | — | compiles, fails to load: DMA descriptor limit |
| ResNet-34 / -50 b32 | bf16 (render or auto-cast) | — | fails to load: DMA descriptor limit |
| ResNet-50 acc2x16 | f32 / bf16 auto-cast | (no probe on the acc path) | 15.3 % / 13.4 % |
| tf32 auto-cast | `--auto-cast=matmult --auto-cast-type=tf32` | — | neuronx-cc crashes (`NCC_IBCG901`, `NCC_IXCG863`) |

The optimization level and model type change neither the numbers nor the speed. Only bf16 moves the
step time. `smoke.sh` printed `EXIT 0` on failed loads (it took grep's status); fixed for next time.

## Reference: stock JAX on the same NeuronCore (`jax/bench.log`)

`jax/generated/generated_{vit_tiny,resnet34}_imagenet.py` (1000-class heads, otherwise the same nets),
`train_step` only, random data, `--auto-cast=none`, graphs warmed on atlas.

| net | dtype | batch | stock JAX ms/step | verified graph ms/step |
|---|---|---|---|---|
| ViT-Tiny | f32 | 32 | **82.5** | 1,281 |
| ViT-Tiny | bf16 matmuls | 32 | **63.8** | 733 (auto-cast) |
| ResNet-34 | f32 | 16 | 349 | 494 |
| ResNet-34 | bf16 convs | 16 | 405 | 435 |
| ResNet-34 | bf16 convs | 32 | fails to load | fails to load |

Stock JAX ViT is ~15× faster than the verified ViT graph on Neuron. On the 4060 Ti the verified ViT
runs 61 ms/step, so the cost is specific to how Neuron handles our graph. The op mix (`train_step`
lowered on CPU, `stablehlo` op counts) is the first lead. The verified graph has 2,632 reshapes
against 291, 148 slices against 3, and 146 pads against 1. Those are likely the per-head split and
its backward. Unprofiled. (Profiled later the same day in `../2026-10-30-trainium-wgrad/`: it is not
the head split; it is the patch-embed weight gradient's 209×209 dilated kernel, 92 % of the
compiled step's DMA blocks.) The JAX ResNet-34 numbers are timing only, because its max-pool backward
runs through Neuron's broken `select_and_scatter`.

The generated JAX ViT trainer on real Imagenette (`jax/vit-tiny-b32-e1.log`) was killed 16 s in.
It loads the whole dataset into host RAM while the other lane's trainer was running, so it is likely
host OOM on the 30 GB box. Not rerun.
