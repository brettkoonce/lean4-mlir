# ViT-Ti pair: what the 2026-10-01 rerun closes, and what it still does not

Written 2026-10-01, ahead of the post-C6 rerun of both ViT-Ti arms (`planning/next_session_vit.md`
D1 = both; Brett: "if we can fix / bring anything else into parity right now we should try, then
throw anything not in this run into a TODO / missing-from-implementation note"). This file is that
note. The comparison targets are DeiT's own `main.py` / `datasets.py` (facebookresearch/deit,
`main`, read 2026-10-01), not timm's re-evaluation config.

Jobs: `vit-default-jax-4gpu` (JAX) and `vit-default-emabf16-4gpu`, which now runs
`emadp128x4wxclipdropeps0000001bf16`.

## 1. Closed in this rerun

| # | axis | before (JAX / verified) | now (both) | where |
|---|---|---|---|---|
| P1 | label smoothing | J 0 (the 72.31 run predates 4b765be7, H1) / V 0.1 | 0.1, folded into the mixed target | no code; the JAX rerun |
| P2 | data aug (C6) | both pre-C6 | bicubic geometry + `pixel` erasing, one shim (the pipeline is byte-identical between the JAX trainer and the shim) | no code; both reruns |
| P3 | CLS token + pos-embed init | J σ 0.02 / V **0** | σ 0.02 | init kind 5 (`mkParam`, `.param … 5` in the four ViT specs, `ViTLayout.specs`), gated on `vitInit`; `tests/init_parity_audit.py` knows kind 5 |
| P4 | patch embed + head precision | J **bf16** (`mm`) / V f32 (the render's carve-out) | f32 | `TrainConfig.f32StemHead` (JAX), set in `vitTinyImagenetConfig` |
| P5 | cosine progress | V one step ahead of J for the whole decay | J's 0-based `prog` | `trainAdamSched` (fleet-wide; moves every future verified cosine run by one step) |
| D1 | LayerNorm ε | 1e-5 both (PyTorch default) | **1e-6** (DeiT `partial(nn.LayerNorm, eps=1e-6)`) | `TrainConfig.lnEps` (JAX); `ViTRenderB`'s `eps` → `vitin_emadp128x4wxclipdropeps0000001bf16_{train_step,fwd}.mlir` |
| D2 | LR floor | 0 both | **1e-5** (DeiT `--min-lr`) | `TrainConfig.minLR` (JAX); `trainAdamSched (minLR := …)` |

Guards added with them: `checkLnEpsWorld` (a no-BN net refuses to score a 1e-6 train step through
a 1e-5 forward, in both `trainAdamSched` and `score-checkpoint`); the verified driver calls the
entry its chosen forward declares (`@<slug>_<variant>_fwd` for a per-variant forward); both confs'
prechecks assert every item above in the artifact or emitted trainer; `vit-ema-drop-render vitin
<variant>` gates the layout of the variant the conf launches.

Not a gap, checked: DeiT's cooldown. `--cooldown-epochs 10` is a default, but DeiT's
`main.py` discards `create_scheduler`'s epoch count (`lr_scheduler, _ = …`) and loops to
`args.epochs`, so DeiT trains 300 epochs with no cooldown. Also not a gap: the in-training eval crop.
DeiT's `--eval-crop-ratio` defaults to 0.875, which is what both paths use. timm's 0.9 is timm's
own re-evaluation protocol (`imagenet_parity.md` §5.6 S5), scored after the run.

## 2. Still different between the two arms (pair, not paper)

| # | gap | size of effect | fix |
|---|---|---|---|
| G1 | **Eval precision.** The verified path scores through an f32 forward (no bf16 forward is rendered, by design). The JAX trainer's `eval_batch` runs the training `forward`, so its block matmuls are bf16. | Small, probably <0.1, but unmeasured. | After the run, rescore the JAX final checkpoint with `DT` rebound to f32 (`jax/scripts/eval_full50k.py` has no such switch yet; it imports the trainer module, so it is one `mod.DT = jnp.float32` before tracing), or add an eval-f32 knob to the emitter. |
| G2 | **Mixup/CutMix granularity.** Verified: one λ, one flip partner and one cutmix box per 128-row replica shard (each producer mixes its own batch). JAX: one λ and one partner flip over the global 512. | Distributional; the gradient-noise structure differs. timm under DDP mixes per GPU batch, i.e. the verified way. | Make the JAX `_mixup`/`_cutmix` per-device (λ per 128-row slice), or accept it and say so. |
| G3 | **RNG streams.** Shim numpy λ vs `jax.random`; host-drawn drop masks vs `jax.random.bernoulli`; four seeded producers vs one pipeline. | Distributional only. | None needed. |
| G4 | **Init distribution.** `F32.heInit` is Bates-3 (≈normal), JAX draws `random.normal`. Variance matched. | Negligible. | A Box–Muller `F32` sampler if it is ever wanted. |
| G5 | **Repeated-aug placement.** Each of the 4 verified producers repeats ×3 within its own shard stream; JAX repeats ×3 in one stream. Both are stream-level approximations of timm's index-level `RASampler`. | Small. | See P-F below. |
| G6 | The fp8 trainer (`trainAdamSchedE4M3`) keeps the old one-step cosine offset. Not used by ViT. | None. | Port the P5 line if it is ever run again. |

## 3. Still different from DeiT on both arms (paper, not pair)

| # | gap | DeiT | ours | what fixing it takes |
|---|---|---|---|---|
| P-A | **GELU** | exact erf (`nn.GELU`) | tanh approximation | Multi-day. No erf op exists in the SHlo AST and Mathlib has no `Real.erf`, so this needs Φ as the Gaussian cdf plus FTC (~150–250 lines), about 20 AST/printer/parser sites (the parser round-trip is under `Certs`), and a whole-net tie that parametrises `blockV`/`vitBodyKVFlat` over the activation (ViTBackB0 recheck, ~11 min / 14 GB). Lowering works: `chlo.erf` compiled on PJRT CPU (agent probe 2026-10-01). Send one op through CUDA PJRT and IREE first. JAX side: `jax.nn.gelu(approximate=False)` behind a flag. ConvNeXt's paper is erf too. |
| P-B | **Gradient clipping** | none (`--clip-grad None`) | 1.0 | The render is trivial (`clip := false`, `emadp128x4wxdropeps0000001bf16`) and so is `gradClipNorm := 0` on the JAX side. The risk is the run: clip was "the unlock" for the LR-5e-4 collapse, but that was measured with Xavier init. A short JAX probe past the warmup peak (~3 epochs, ~30 min) on DeiT init settles whether the clip-off arm is safe. |
| P-C | **What is scored** | the live model (`evaluate(data_loader_val, model, …)`) | the EMA shadow | No training change. Score both final checkpoints on the live region too: `score-checkpoint` region `live` for verified, and the JAX final `.state.npz` `params`. Report both numbers. |
| P-D | **LR schedule shape** | timm `CosineLRScheduler` stepped per EPOCH, warmup from 1e-6 (`--warmup-lr`), cosine over [0, 300] including the warmup span (timm 0.3.2 has no `warmup_prefix`), `step(epoch)` called after each epoch | per STEP, warmup from ~0, cosine over the post-warmup span | Host-side on both paths (the emitter's LR block, `trainAdamSched`'s `lrt`). Small effect; a `timmEpochSchedule` flag on both if wanted. |
| P-E | **Mixup/CutMix switch** | random per batch, `switch_prob 0.5` | strictly alternating by step | Both emitted from `Jax/Codegen.lean` (trainer + shim); flag-gated so the mixup gate and other nets stay byte-identical. ConvNeXt and RSB recipes share it. |
| P-F | **Repeated augmentation** | timm `RASampler`: index-level, the 3 copies spread across GPUs | stream-level repeat + shuffle window | Index-level sampler in the shim (`ds.shard` ordering) and the JAX pipeline. |
| P-G | **Random resized crop** | torchvision `RandomResizedCrop` (scale 0.08–1, ratio 3/4–4/3, 10 tries, center-crop fallback) | TF `sample_distorted_bounding_box` (`min_object_covered=0.1`) | Fleet-wide pipeline change, every net's reference moves. |
| P-H | **Patch-embed bias init** | PyTorch Conv2d default `U(±1/√768)` | 0 both | A kind for the stem bias (verified) + the emitter's `vitInit` branch. Negligible. |
| P-I | **Batch** | 1024 (8 GPUs × 128), lr 1e-3 | 512, lr 5e-4 (DeiT's own linear scaling) | Hardware. |
| P-J | **timm-protocol eval resize** | `math.floor(224 / crop)` (248 at 0.9) | `round(…)` (249 at 0.9; 236 vs 235 at 0.95, which also affects the A3 and MNv4 timm-protocol scores) | One emitted line, fleet-wide. Only affects post-hoc timm-protocol scoring, not the in-training 0.875. |

## 4. Beyond ViT (found here, owned elsewhere)

* Every JAX reference runs its classifier head in bf16 and every verified render keeps it f32 (P4).
  `f32StemHead` exists now. Set it in the next rerun of each pair.
* P5 moves every verified cosine run by one step from today. Nothing landed is affected (no run in
  flight), but a resume across this commit fuses two schedules that differ by one step.
* P-J moves the R50-A3 / MNv4 timm-protocol rescoring by one pixel of resize when fixed.

## 5. Before launching (in addition to `next_session_vit.md` §4)

* The verified arm is a NEW variant, so it has a NEW checkpoint name. The 72.35 run's
  `vitin_emadp128x4wxclipdropbf16_ckpt_xla.bin` (marker 300) cannot be resumed into it, and
  `pc_ckpt` lists it as a stray. Leave it in place: it is the old pair's verified weights.
* Re-probe both arms (§5 of the handoff). The changes here add two f32 matmuls on the JAX side and
  change nothing in the verified render's cost; the ETAs in both confs are the 09-26 post-C6 probes.
* `scripts/regen_jax_generated.sh box` must print ✅ (the ViT-Ti trainer and its xavier sibling moved).
