# `apps/` — the chapter trainers

One `Main` per `lake exe`; [`lakefile.lean`](../lakefile.lean) maps each exe name to its file.
The tour in the [top-level README](../README.md) runs these by tier. The Chapter 10 demos are
in [`demos/`](../demos/).

| directory | what runs from it | chapters |
|---|---|---|
| [`mnist/`](mnist/) | tier 1 (`lake run mnist`): `mnist-linear-verified`, `mnist-mlp-verified`, `mnist-cnn-verified`; also the grid, PGD, spectral, smoothing and E4M3 studies | 1–3 |
| [`cifar/`](cifar/) | tier 2 (`lake run cifar`): `cifar8w-ablation`, `cifar8w-bn-ablation`; also the verified CIFAR runners and the robustness studies | 4 |
| [`imagenette/`](imagenette/) | tier 3 (`lake run imagenette`): the seven `*-verified-adam` trainers; tier 4: the `*-imagenet-verified` runners, launched through [`scripts/jobs/`](../scripts/jobs/) | 5–9 |
| [`ablation/`](ablation/) | the optimizer, precision and recipe ablations | 4–5 |
| [`tools/`](tools/) | `score-checkpoint`: rescore a saved ConvNeXt or ViT checkpoint | — |

`lake build Apps` type-checks every entry point here and in `demos/` without linking them.
