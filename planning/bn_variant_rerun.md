# Handoff: R50 RSB-A3 "BN-fixed" rerun (chase the last ~1.4 pts)

**For a future Claude.** Self-contained runbook to launch the ResNet-50 RSB-A3
faithful run *with the grad-accum BN-momentum fix active*, and decide whether it
beats the current committed result.

## Why this run exists

The committed RSB-A3 faithful run (`resnet50ImagenetConfigRSBFaithful`: LAMB +
`gradAccumSteps=4` → effective bs2048) reached **76.66% top-1 / 93.03% top-5**,
~1.4 pts under timm's RSB-A3 paper (78.1%). See memory
`project_r50_a3_lowval_diagnostic` and `project_grad_accum_lever`.

**That 76.66% was measured BEFORE the BN-momentum fix.** In the accumulation
scan, `_bn` runs K=4× per optimizer step, so running-BN stats were decaying ~K×
too fast (`0.99^K/step`) vs a true single-forward bs2048 (`0.99/step`) — which
biases the eval-time stats (train uses batch stats, eval uses running stats).
Commit `cc265ba` fixed the codegen to emit `momentum = 0.99^(1/K) = 0.997491`
when `gradAccumSteps>1`, so K updates compose to ≈ one 0.99/step update.
Non-accum configs stay byte-identical (`momentum=0.99`).

**Hypothesis:** the BN fix reclaims ~0.5–1 pt → final lands ~77–77.5%. This run
tests it. It's the single highest-leverage remaining lever (the others are
RandAugment op-set parity + LAMB-impl details; see `planning/grad_accum.md`).

## Preconditions (verify first)

1. On `main` at/after `cc265ba`. Confirm the fix is in the source:
   `grep -n "0.99\^(1/K)" jax/Jax/Codegen.lean` (the compensation comment) and
   the emitter block near `def _bn`.
2. Hardware: run on **ares** (6× RTX 4060 Ti). Mask the AER-prone cards with
   `CUDA_VISIBLE_DEVICES=0,2,3,4` (idx1/idx5 are excluded). See memory
   `reference_ares_pcie_aer`. The supervise script already sets this.
3. venv python with jax+CUDA at
   `/home/skoonce/lean/klawd_max_power/lean4-jax/.venv/bin/python`.
4. ImageNet is TFDS `imagenet2012` at `~/tensorflow_datasets/imagenet2012/5.1.0`
   (~144G). The `data/imagenet` CLI arg is ignored by the tfds loader.

## Steps

```bash
cd /home/skoonce/lean/klawd_max_power/lean4-jax/jax

# 1. Build + regenerate the config so the .py picks up the BN fix.
#    (.lake/build is gitignored — a fresh clone must build+generate.)
lake build resnet50-imagenet
./.lake/build/bin/resnet50-imagenet rsb-faithful      # writes generated_..._rsbfaithful.py
#    NB: the binary then tries to auto-run training with the SYSTEM python
#    (no jaxlib) and errors — that's harmless; we only want the generated .py.
#    (Recipe selection is the positional arg: `resnet50-imagenet rsb-faithful`.)

# 2. Confirm the BN momentum is compensated (NOT 0.99):
grep -n "def _bn" .lake/build/generated_resnet50_imagenet_rsbfaithful.py
#    expect: momentum=0.997491   (0.99^(1/4))

# 3. (optional) 60s GPU smoke — random tensors, no data:
#    a saved smoke lives in the scratchpad from the original session; or mirror
#    smoke_r50_a3_gpu.py against the rsbfaithful module. Set m.WD_MASK = m._wd_mask(params)
#    before calling train_step (it's only set inside `if __name__=="__main__"`).

# 4. LAUNCH — CRITICAL: use a FRESH ckpt base so it does NOT resume the old
#    76.66% run (whose base is /home/skoonce/r50_rsb_a3_rsbfaithful).
CKPT_BASE=/home/skoonce/r50_rsb_a3_rsbfaithful_bnfix \
  bash scripts/supervise_r50_a3_rsbfaithful_100ep.sh
```

The supervisor handles: 4-GPU ares mask, lossless resume, thermal cooldowns
@ep25/50/75 (30 min each), and the AER watchdog (now fatal-only — benign
corrected DRAM ECC no longer false-trips it). Expect **~16 h wall** (~9.5
min/epoch × 100 + 3 cooldowns). Logs: `/tmp/r50_a3_rsbfaithful_master.log`
(supervisor) + `/tmp/r50_a3_rsbfaithful.log` (per-epoch `[Epoch N] ... val_top1`).

## Baseline to beat (per-epoch, the committed 76.66% run)

- Warmup climbs to ~40% by ep25 (the OLD broken LAMB@bs512 was 16.8% at ep25).
- ep50 ≈ 56.6% / 80.3%, ep75 ≈ 71.8% / 90.4%, **ep100 = 76.66% / 93.03%**.
- Healthy = tracking at or above this curve. Watch val@ep10 (~20% top1) and
  ep50 as early reads.

## Decision + what to update if it wins

- **If final top-1 > ~77.0%** (clearly beats 76.66 beyond val noise, ~±0.2):
  the BN fix helped. Update the recorded number in THREE places:
  1. `blueprint/src/content.tex` §10.2.1 ResNet entry (search `76.66`).
  2. `Bestiary/ResNet.lean` ("Reproduced here." note, search `76.66`).
  3. Memory `project_grad_accum_lever` + `project_r50_a3_lowval_diagnostic`.
  Rebuild the book PDF to preview (`cd blueprint/src && latexmk -xelatex print.tex`;
  `latexmk -c` after). Commit; ask before pushing.
- **If it lands ~the same (76.5–76.8%):** the BN-K× effect was negligible (as
  suspected — the run already beat the BN-once/step broken baseline decisively).
  Record the null result in `project_grad_accum_lever`; leave the book at 76.66%.
  Next lever would be RandAugment op-set parity (Identity vs SolarizeAdd, NEAREST
  geom interp) — smaller expected payoff.

## Gotchas

- **Fresh ckpt base** is the #1 footgun — reusing `r50_rsb_a3_rsbfaithful` resumes
  the finished old run and does nothing useful.
- Don't run the Lean binary directly for training — it uses the system anaconda
  python. Training runs the generated `.py` via the venv python, which the
  supervise script does (`$VENV_PY -u $PY`).
- `resnet50-imagenet rsb-faithful` (positional recipe arg) selects the config;
  the legacy `LEAN_MLIR_*` env flags have been retired.
