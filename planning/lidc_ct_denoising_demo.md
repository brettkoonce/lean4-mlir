# lidc_ct_denoising_demo.md — low-dose CT denoising on LIDC-IDRI (Image generation)

Scoped 2026-10-07 on ares from `planning/demo_slate.md` §3. Status log at the bottom (§9).
Branch `lidc-denoising` (rebased onto `origin/main` 026f9974 on 2026-10-10), checked out in the
worktree `../lean4-jax-mlir-lidc` (its `.lake/packages` and `data` are symlinks into the main
checkout): these scripts, the loader and this doc live there until the demo lands, so the shared
checkout stays clear for the ImageNet work.

## 0. The one-paragraph version

The chapter's image-generation slot has entries and no demo. This one asks a question a reader
can be wrong about: **does a denoiser keep the nodules?** An MSE UNet wins PSNR by regressing
to the mean and erases small, low-contrast nodules; a diffusion denoiser keeps texture and can
invent it. LIDC-IDRI gives the instrument the field's own paired set lacks — four radiologists'
outlines of every nodule, with size and characteristics — so the finding is contrast-to-noise by
nodule size beside PSNR/SSIM, not "looks sharper". The low-dose arm is simulated in the
projection domain, which is how the field's reference pairs (Mayo, TCIA) are made too, and the
simulator is checked against them (§3). Every net is one already in the kit; the only new code
is the data path (done, §2–§5) and a loader.

## 1. The question and the arms

| arm | what it is | role |
|---|---|---|
| full-dose FBP | the scanned image | the target and the ceiling |
| low-dose FBP | the scanned image + projection-domain noise at quarter dose (§3) | the floor |
| classical filter | BM3D or TV, CPU, no training | the pre-deep-learning bracket |
| MSE UNet | `unetBrats`'s layers at 1 in / 1 out channel, patch-trained, residual (predicts the noise) | the regression-to-the-mean arm |
| Noise2Noise | the same net trained low₁ → low₂ (two independent realisations are in every record) | the no-clean-target arm |
| conditional diffusion | the DDPM UNet's layers with the low-dose image as a conditioning channel (SR3 / Palette form), ε-prediction | the texture-keeping, texture-inventing arm |

Scored on val volumes: PSNR / SSIM (soft-tissue and lung windows), and **nodule
contrast-to-noise** — mean HU inside the reader-consensus outline minus the surrounding lung,
over the surrounding std — by nodule size bin (<3, 3–6, 6–10, >10 mm), with the four readers'
agreement count as a weight. The expected finding is the chapter's theme: MSE wins PSNR and
loses the <6 mm bins; diffusion keeps them and hallucinates some.

## 2. Data on disk (`data/lidc/`, gitignored)

LIDC-IDRI (TCIA, CC BY 3.0): 1,018 CT series over 1,010 patients, 243,958 slices, 128.4 GB
(the NBIA manifest's `FileSize`); also 237 DX + 53 CR radiographs, 6,949 DICOM SEG and
6,859 SR objects (the nodule annotations in standard form, 3.4 GB), and TCIA's
`LIDC-XML-only.zip` (8.6 MB; 1,036 `LidcReadMessage` files of 1,319 — the rest are the
radiograph reads). The public NBIA REST API needs no account: `getSeries?Collection=LIDC-IDRI`
for the manifest, `getImage?SeriesInstanceUID=…` for one ZIP per series.

| piece | script | what |
|---|---|---|
| fetch | `scripts/datasets/lidc_fetch.py` | manifest; one CT series per patient (the one with more slices for the eight with two); a manufacturer-stratified seeded subset (largest remainder); parallel `getImage`, idempotent on a `.done` record per patient |
| annotations | `scripts/datasets/download_lidc.sh` | the XML zip and TCIA's nodule-count sheet, then fetch, then preprocess |
| pairs | `scripts/datasets/preprocess_lidc.py` | §3–§5 |
| check | `scripts/probes/lidc_mayo_noise_check.py` | §3's Mayo comparison |

**The subset (seed 0, 300 patients):** GE 199 / Siemens 60 / Philips 22 / Toshiba 19 — the
collection's own 670/205/74/69 proportions — 74,440 slices, 39.2 GB. The API served it at
about 9 MB/s on six connections (≈ 70 min). The full set would be 128 GB; the disk had 92 GB
free, and the 300-patient draw was the user's call (2026-10-07).

Environment: `.venv-lidc` from `requirements-lidc-lock.txt` (pydicom 3.0.2, scikit-image
0.26.0, scipy 1.18.1, numpy 2.5.3; Python 3.12), never the pinned `.venv`.

## 3. The low-dose simulation

Low-dose CT noise is correlated streaks from photon starvation along long paths, not
image-domain Gaussian. The construction, per slice (`preprocess_lidc.py`'s docstring has the
equations): μ from HU at μ_water = 0.019 mm⁻¹; the parallel-beam Radon transform at 720 views
over 180° (scikit-image; 4.7 s per 512² slice on one core, round trip 0.39%); Poisson counts at
I0_add·exp(−p); the log-domain noise n; its filtered back-projection (Shepp-Logan apodisation,
3.4 s) is the noise image, and FBP being linear, the full-dose target is the scanned image
itself — the projector's blur never enters the pair. Added variance at I0_add on top of the
scan's own gives the low-dose total when I0_add = I0·dose/(1 − dose): I0/3 at quarter dose.

**I0 is calibrated per slice, not taken from a textbook.** Three measurements drove this:

1. At a fixed 10⁵ photons per ray (the usual simulation constant) the added noise was 70–450 HU
   over the reconstruction circle, 100–590 in soft tissue — a clinical quarter-dose chest adds
   40–60. The number is right for a scanner's geometry, not for a 512-pixel parallel beam.
2. One I0 per *scan* (median of three slices) put the mid-chest on target and left the shoulder
   slices, whose paths are longest, at 5–8× their own noise (539 HU on LIDC-IDRI-0007 z129).
   Scanners hold the noise level by modulating the tube current slice by slice; the simulation
   has to as well.
3. Per *slice*: the slice's own noise σ_full is the median 7×7 local std over homogeneous soft
   tissue (windows wholly within −200..300 HU, mean −50..150); the added noise must measure
   √(1/dose − 1)·σ_full = √3·σ_full by the same estimator; one realisation at a reference I0 gives
   σ_ref and I0_add = I0_ref·(σ_ref/target)² exactly (variance ∝ 1/I0), so the calibration costs
   one extra back-projection. On the four-patient smoke build: σ_full 20–69 HU (STANDARD kernels
   20–40, a LUNG kernel 50–69), I0_add 5×10³–9.6×10⁴, achieved/target 0.983–1.010 (median 0.999).
   Slices with under 500 measurable windows (apices, some bases) take their volume's median I0.

**The Mayo check.** TCIA's LDCT-and-Projection-data v7 (July 2026) opened its chest and abdomen
arms under CC BY 4.0 (200 patients, 698 CT series, 863 GB with projections; head remains
restricted). Its low-dose arm is itself *simulated*, by Siemens/GE noise insertion in the
projection domain at 10% (chest) / 25% (abdomen) of routine dose — so the slate's "why not real
pairs?" is answered by the reference set being simulated too, and it becomes the check. Both
Mayo arms are reconstructions of one scan, so low − full is exactly the inserted noise. On
C002 (Siemens, B50f, 273 mA, 1.5 mm; 20 mid-volume slices) by the estimator above: own 66.1 HU,
inserted 196.7, total 208.2; **inserted/own 2.99 against the 3.00 its dose predicts, total/own
3.15 against √10 = 3.16.** The "added variance = (1/dose − 1) × own" rule is the field's
construction to 1%, by this estimator. `scripts/probes/lidc_mayo_noise_check.py` reproduces it
(300 MB, `data/lidc/mayo/`).

Simplifications, named: parallel-beam geometry (not fan/helical); no bowtie filter, so the
periphery is noisier than a scanner's; one μ_water, no beam hardening; one apodisation where each
scanner had its own kernel; electronic noise off (I0 is an effective, calibrated quantity).

## 4. The instrument: nodules

The XML is parsed directly (pylidc is 0.2.3 from 2021): per reading session, each
`unblindedReadNodule`'s `roi`s (one per slice, `imageSOP_UID` joins the DICOM, `inclusion`
FALSE cuts a hole), polygon-filled with the outline pixels included; a <3 mm nodule is a single
point and gets a 3 mm disk. The per-slice mask stores **how many of the four readers** outlined
the pixel (0–4), so consensus is a threshold at scoring time. `nodules.json` has one row per
reader per nodule: slices, centroid, area and equivalent diameter at the largest outline, the
nine characteristics (subtlety … malignancy) when given, the <3 mm flag. Four readers matched on
every landed patient so far; `manifest_lidc.json` lists any patient whose XML did not match.

## 5. Record format and the Lean side

`train.bin` / `val.bin`: uint32 count, then per record `full` (S² uint16, HU + 1024 in 0..4095),
`low₁ … low_N` (N = 2 realisations), `mask` (S² uint8): (3 + 2N)·S² = 1,835,008 bytes at
S = 512. `*.idx` = uint32 volume count then one slice count per volume (val is volume by volume
in z order); `*.json` one row per record with patient, slice, z, spacing, thickness, kernel, kVp,
mA, maker, nodule pixel count, σ_full, I0_add, target and achieved σ_add.

Split: patient-wise, seed 0, 20% val. Train patients contribute 40 slices each — every
nodule-bearing slice first, evenly spaced slices to fill; val patients contribute every
nodule-bearing slice plus every 4th slice (the nodules are the instrument; the stride is enough
for PSNR/SSIM). 300 patients → ≈ 9,600 train + ≈ 5,000 val records ≈ 27 GB; the build is
≈ 12 s per slice per core, ≈ 1.7 h on 30 cores.

Loader (built 2026-10-10): `F32.loadCtPair path S N pairing r` (`lean_f32_load_ct_pair` in
`ffi/f32_helpers.c`, after the BraTS loader) returns the `DatasetIO` 3-tuple (input f32, target
f32, count), one S² image each: pairing 0 = (low copy r, scanned slice), pairing 1 = (low copy 0,
low copy 1) for Noise2Noise. Pixels are (HU + 1024)/4096: linear in HU, so MSE is HU² up to a
constant, and exact. `F32.loadCtMask` returns the reader-count masks for scoring. Both check
the file length against count·(3 + 2N)·S². Gate: `scripts/probes/lidc_loader_check.py <bin>`
runs `test-lidc-loader` and requires its integer statistics (Σ input, Σ target, Σ squared
difference, mask sums, all on the 1/4096 grid) to equal a numpy decode exactly; the data is also
a case in `test-dataset-record-sizes`. Memory: one pairing of the full train file is ≈ 19 GB
host-resident (9.6k records × 2 × 1 MiB).

Next: a `DatasetKind.lidc` beside
`brats224`; `unetCtDenoise (S : Nat)` — `unetBrats`'s layers at 1 → 1 channel with `imageH := S`
(zero new codegen); the DDPM trainer's per-pixel MSE for the regression arms (check
`Train.lean`: the ε-loss path is exactly this); patch training at 256², scoring at 512.

## 6. Plan

| phase | what | gate |
|---|---|---|
| 0 | data path (fetch, XML, simulator, calibration, Mayo check) | **done 2026-10-07**; the 300-patient build ran 2026-10-10 (§9) |
| 1 | loader + `unetCtDenoise` + `lidc-denoise-train` (**done 2026-10-10**, §9; `DatasetKind.lidc` not needed: the demo trainer drives the DDPM ε-loss step directly) + a render of `unetCtDenoise`; a 1-patient sanity fit | the net overfits one volume; low→full PSNR rises above the floor |
| 2 | MSE UNet, patch-trained on the 240 train patients; scoring script (PSNR/SSIM + nodule CNR by size) | beats the floor and BM3D on PSNR; the nodule table exists |
| 3 | Noise2Noise with the same net | within ~0.3 dB of phase 2 (the classic result) |
| 4 | conditional diffusion (the DDPM UNet, SR3 form), sampling strip | nodule CNR in the <6 mm bins beats MSE; the hallucination rate is measured (nodule-sized objects where no reader drew one) |
| 5 | the section: one figure (a nodule, five arms), one table (PSNR/SSIM + CNR by size) | the user's one-figure-one-table rule |

## 7. Open decisions

- Input scaling for the net (window vs. linear HU); residual vs. direct prediction.
- Classical arm: BM3D (needs a package) vs. TV (scikit-image has it, slower, weaker).
- Diffusion sampling budget (DDIM steps) and whether to report the posterior std as an
  uncertainty map beside the sample.
- Whether the <3 mm nodules (29 of 44 rows in the smoke build — most of LIDC) are scored on
  their 3 mm stand-in disks or dropped: they are the bin the question is about.

## 8. Disk

Before this demo `/` had 92 GB free of 1.9 TB. The DICOM subset is 39 GB, the pairs ≈ 27 GB,
Mayo's check 0.3 GB. The 2026-10-07 prune survey (two copies of the ImageNet tarballs at 145 GB
each; `~/hallway-2023` 129 GB; the old `../lean4-jax` checkout 55 GB; `../jax-orin-build` caches
59 GB; `.lake/build/distogram_*` 49 GB; …) is in the session's memory notes, not executed.

## 9. Status log

- 2026-10-07: scoped; `.venv-lidc`; fetch script + 300-patient draw launched; XML parsed (1,036
  reads); preprocessor written, smoke-tested on 4 patients three times — fixed I0 (far too
  noisy), per-scan I0 (shoulders 5–8×), per-slice I0 (0.983–1.010 of target); Mayo check 2.99 /
  3.15 against 3.00 / 3.16; two realisations per record and the val stride added. The full
  build is NOT launched: a 4-GPU job had the box at load 55 when the fetch finished; it runs
  with `./scripts/datasets/download_lidc.sh` (idempotent: skips the fetched patients, then
  `preprocess_lidc.py` with the §5 defaults) — `nice -n 19` and `--workers 8` beside a live
  run (≈ 6 h), 30 workers on an idle box (≈ 1.7 h).
- 2026-10-10: 145 GB freed (the duplicate ImageNet tarballs); the branch rebased onto
  `origin/main` and given its own worktree. The full build launched on the idle box with 30
  workers (`preprocess_lidc.py data/lidc/dicom data/lidc/xml data/lidc --workers 30`, log
  `data/lidc/preprocess.log`). Loader + gate written and passing on the 4-patient smoke file
  (92 val records, every statistic equal).
- 2026-10-10: **phase 1 gate passes.** `unetCtDenoise` (`ReferenceNets`, `unetBrats`'s layers at
  1 → 1 channel, 7,849,025 params) trained by `lidc-denoise-train` (`demos/MainLidcDenoiseTrain.lean`):
  input HU/1000, target the noise (low − full)/100 HU, the DDPM trainer's per-pixel MSE step
  (`useDdpm`), 8 × 256² crops cut by one seed from input and target, both realisations; scored on
  whole 512² slices by the eval forward at batch 1. 1,500 steps (≈ 0.65 s/step on one 4060 Ti
  beside the 30-worker data build) on the smoke file's one volume (LIDC-IDRI-0002, 92 slices),
  scored on that volume and on the 2 slices of an unseen patient (0007):

  | | body RMSE (HU) | PSNR (dB) | SSIM (soft tissue) |
  |---|---|---|---|
  | 0002, fitted: floor → denoised | 40.67 → 20.25 | 42.27 → 48.16 | 0.837 → 0.905 |
  | 0007, unseen: floor → denoised | 97.64 → 74.39 | 35.29 → 37.36 | 0.891 → 0.881 |

  `scripts/probes/lidc_denoise_score.py` recomputes the RMSE from the dumped outputs and
  matches the trainer's to 0.01 HU; figures and log in `runs/2026-10-10-lidc-denoise-smoke/`. The unseen
  patient is the 2-slice smoke train file, noisier (floor 98 HU) and over-smoothed in the lung
  by a one-volume net: SSIM dips. Not a result, only the gate. XLA logs "Can't reduce memory
  use below 11.8 GiB … only reduced to 24.4 GiB" compiling the 512² eval at batch 4 and at 1;
  it runs and scores at either, so the message is the autotuner's, not a failure.
- 2026-10-10: **runs prepped.** `lidc-denoise-train` takes `--arm=mse|n2n` (N2N: target = the
  other low copy, both directions via the loader's pairings 1 and 2; the scanned slice is never
  seen), `--seed`, `--cosine`, checkpoints `[θ, m, v, BN, step]` every 1,000 steps to
  `data/lidc/out/<arm>/` and resumes from them (tested: a 60-step run resumed at 60), writes the
  loss curve to `<out>/curve.csv`; the arm and seed are in the net's name, so the two arms'
  artifacts and checkpoints never collide. N2N smoke fit, same 1,500 steps on 0002: body RMSE
  40.67 → 19.68 (MSE 20.25), unseen 0007 97.64 → 74.11 (MSE 74.39) — the classic N2N ≈ MSE.
  `runs/2026-10-10-lidc-denoise/launch.sh`: loader gate on `val.bin`, both GPUs idle, then MSE on
  device 0 and N2N on device 1 (N2N after MSE's data is resident), 30,000 steps of 8 × 256²,
  Adam 5e-4 cosine, seed 0; each arm scores the val split and runs the Python scorer
  (`score.txt`, `figure.png`). Budget ≈ 3.3 h at 0.4 s/step, ≈ 5.4 h at 0.65 s/step.
