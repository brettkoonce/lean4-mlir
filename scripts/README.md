# `scripts/` — gates, jobs, figures and probes

Most files here are single-purpose and say what they do in their first lines. The ones worth
knowing:

| group | entry points |
|---|---|
| ImageNet jobs | [`jobs/`](jobs/) holds one `.conf` per run; `lake run <job>` launches it through `supervise.sh`, and `lake run imagenet` prints the plan |
| render gates | `regen_verified_mlir.sh` (regenerate or audit [`verified_mlir/`](../verified_mlir/)), `gen_mlir_manifest.py`, `check_render_coverage.py`, `regen_jax_generated.sh` (the JAX reference's emitted files) |
| name and citation gates | `check_target_names.sh` (every exe a script names exists), `check_audit_coverage.py`, `gen_comparator_tier.py`, `blueprint_uses.py`, `book_xrefs.py` |
| reference parity | `*_forward_tie.py`, `convention_audit.py`: the verified render against its JAX reference; `*_timm_parity.py`: the JAX reference against timm's architecture, on shared weights; `score_timm.sh`: score a checkpoint under timm's validation protocol |
| book figures | `blueprint_depgraph_tikz.py` (the dependency graphs), `log_to_pgfplots.py`, and one `*_figure.py` / `*_metrics.py` / `*_score.py` set per Chapter 10 demo |
| certificates | `lipschitz_cert_*`, `smooth_*_gen.py`, `crown_ibp_scorecard.py`: write the scorecards in [`LeanMlir/Proofs/Certificates/`](../LeanMlir/Proofs/Certificates/) |
| probes | `*_probe*.py`, `*_check.py`, `bf16_*`: one-off measurements behind a specific result |

Files starting with `_` are helper modules the others import. [`audit_census/`](audit_census/)
is the declaration-level audit tooling.
