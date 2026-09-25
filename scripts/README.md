# `scripts/` — gates, jobs, figures and probes

Every script says what it does in its first lines. Run them from the repo root.

| directory | what is in it |
|---|---|
| [`jobs/`](jobs/) | one `.conf` per ImageNet run; `lake run <job>` launches it through `supervise.sh`, and `lake run imagenet` prints the plan |
| [`gates/`](gates/) | the checks CI runs: target names, audit and render coverage, the `verified_mlir/` manifest, the comparator tier, the convention audit, the residency and eval gates |
| [`parity/`](parity/) | the verified render against its JAX reference (`*_forward_tie.py`), the JAX reference against timm's architecture on shared weights (`*_timm_parity.py`), and `score_timm.sh` for timm's validation protocol |
| [`certs/`](certs/) | the generators behind the scorecards in [`LeanMlir/Proofs/Certificates/`](../LeanMlir/Proofs/Certificates/) and the trained-net witnesses |
| [`demos/`](demos/) | one figure / metrics / score set per Chapter 10 demo |
| [`book/`](book/) | the book's dependency graphs, cross-reference and citation checks, plots from logs, the site map |
| [`datasets/`](datasets/) | `download_*.sh` and `preprocess_*.py` for every dataset |
| [`sweeps/`](sweeps/) | seed sweeps, ablation launchers and other multi-run drivers |
| [`probes/`](probes/) | one-off measurements behind a specific result: the detector and segmentation loss checks, bf16 probes, perturbation studies |
| [`lib/`](lib/) | helpers the others import or source: `_iree.py`, `_mnist_io.py`, `_stats.py`, `lean_graph.py`, the shell libraries |
| [`audit_census/`](audit_census/) | the declaration-level audit tooling |

The files left at this level are the ones running jobs call: `supervise.sh`, `gen_shims.sh`,
`det_shim.sh`, `shim_wiring_gate.py`, and the two regeneration scripts
`regen_verified_mlir.sh` and `regen_jax_generated.sh`.
