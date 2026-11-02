# lidc_ct_denoising_demo.md — full-dose CT synthesised from low dose, LIDC-IDRI + Mayo (Image generation)

Scoped 2026-10-07 on ares from `planning/demo_slate.md` §3; re-framed 2026-11-02 (§0, §10).
Status log at §9. Branch `lidc-denoising` (rebased onto `origin/main` 19c52c9a on 2026-11-02;
first pushed to `origin/lidc-denoising` the same day, before the rebase), checked out in the
worktree `../lean4-jax-mlir-lidc` (its `.lake/packages` and `data` are symlinks into the main
checkout): these scripts, the loader and this doc live there until the demo lands, so the shared
checkout stays clear for the ImageNet work.

## 0. The one-paragraph version

The chapter's image-generation slot has entries and no demo. This one is a real-world use of
image synthesis, not a new method: **given a low-dose chest CT, generate the full-dose scan it
stands in for** — less radiation for the patient, the image the radiologist is used to reading.
A conditional diffusion model is the generator. The baselines are what a reader would try first:
classical filters (Gaussian, TV, BM3D), an MSE-trained UNet and Noise2Noise. Their result (§9,
2026-10-11) sets up the generator's case: the nets keep ≥ 97% of nodule contrast where the
filters lose up to 31%, but the MSE net's output is the *average* of the full-dose images
consistent with its input, and an average is softer than any scan. A generator draws *one*
plausible full-dose image, so it should give the grain back — and the check every synthesis use
needs is whether it invents or drops findings doing so.

So the demo quantifies four things (§10): fidelity to the full dose (PSNR / SSIM / RMSE, which a
sampler must lose a little of — the full dose carries ≈ 36 HU of its own noise); **texture**
(the noise power spectrum against the full dose's — "too soft" as a number); **findings** (the
radiologists' nodules kept, blobs invented in unmarked lung, the spread across samples); and
**generalisation** to a held-out dataset the nets never see. LIDC-IDRI (CC BY 3.0) supplies the
training set and the instrument — four radiologists' outlines of every nodule. The low-dose arm is
simulated in the projection domain (§3), as the field's reference pairs are; those pairs, Mayo's
LDCT-and-Projection-data (CC BY 4.0, chest and abdomen open since July 2026), are the held-out
test: other patients, another institution, Siemens scanners, sharper kernels, and the vendor's
own noise insertion instead of ours. Every net is one already in the kit (the BraTS UNet's
layers, the MNIST DDPM's diffusion step); the new code is the data path (done, §2–§5), a loader
(done) and a sampler (§10).

## 1. The question and the arms

| arm | what it is | role |
|---|---|---|
| full-dose FBP | the scanned image | the target and the ceiling |
| low-dose FBP | the scanned image + projection-domain noise at quarter dose (§3) | the floor |
| classical filter | BM3D or TV, CPU, no training | the pre-deep-learning bracket |
| MSE UNet | `unetBrats`'s layers at 1 in / 1 out channel, patch-trained, residual (predicts the noise) | the regression-to-the-mean arm |
| Noise2Noise | the same net trained low₁ → low₂ (two independent realisations are in every record) | the no-clean-target arm |
| conditional diffusion | `unetCtDenoise`'s layers with the low-dose image and t as input channels, diffusing the residual (§10) | **the generator** — the demo's subject |

Scored on the LIDC val volumes and on the Mayo chest pairs (§10): PSNR / SSIM / body RMSE;
the noise power spectrum and local grain over flat soft tissue against the full dose's; nodule
**contrast kept** — (inside the reader-consensus outline − the surrounding lung ring) as a
fraction of the full dose's, by size and texture (LIDC only: Mayo has no reader outlines); blobs
in unmarked lung against the full dose's count; the spread across samples. (The 2026-10-07 plan
expected MSE to lose the < 6 mm nodules; it keeps them, §9 — the filters lose them.)

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
(300 MB, `data/lidc/mayo/`). The same collection is the held-out test set (§10.4): 50 chest
patients have both image arms (all Siemens; the 49 GE chest patients have full-dose images and
projections only), 17,313 slice pairs, 18.2 GB as DICOM, CC BY 4.0, no login.

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
| 2 | MSE UNet, patch-trained on the 240 train patients (**run 2026-10-10**, §9); scoring script (PSNR/SSIM + nodule CNR by size) | beats the floor and BM3D on PSNR; the nodule table exists |
| 3 | Noise2Noise with the same net (**done 2026-10-10**: 44.55 vs 44.60 dB, §9) | within ~0.3 dB of phase 2 (the classic result) |
| 4 | the generator: conditional residual diffusion + DDIM sampler, and the texture / invention scorers (§10.1–§10.3) | smoke fit: samples on the fitted volume score near the MSE net's RMSE, NPS nearer the full dose's than MSE's |
| 4a | the held-out set: Mayo chest pairs converted to the record format; NPS of our simulated noise vs. the vendor's (§10.4) | the two noise spectra agree in shape (peak frequency within ~10%) at matched σ |
| 4b | the 10%-dose LIDC build (Mayo chest is 10% of routine dose), MSE + diffusion retrained at 10% (§10.4) | the build's achieved/target σ as phase 0's |
| 5 | runs and scoring: diffusion at 25% and 10%, every arm on Mayo with no retraining (§10.5) | texture ratio nearer 1 than MSE's; nodule contrast kept ≥ 95%; invented blobs near the full dose's count; the ranking reported as found on Mayo |
| 6 | the section: one figure, one table (§10.6); a render of `unetCtDenoise`; land on main | the user's one-figure-one-table rule; the book's float gates |

## 7. Open decisions

- Input scaling for the net (window vs. linear HU); residual vs. direct prediction.
- Classical arm: ~~BM3D vs. TV~~ both run, with a Gaussian beside them, and BM3D with the
  measured noise spectrum (2026-10-11, §9). Its knob was chosen at its grid's edge (3× the
  spectrum); a wider grid would smooth more and keep less nodule contrast.
- ~~Diffusion form, η, step sweep~~: decided 2026-11-02 (§10.1) — residual, η = 1, 50 steps, no
  sweep. The per-pixel spread across samples is reported (§10.3), as an invention check rather
  than as an uncertainty product.
- ~~Held-out set~~: Mayo chest, all 50 paired patients, at its own 10% dose with nets trained on
  a 10% LIDC build (§10.4). Noise2Noise is not retrained at 10% (its 25% result stands; the
  10% build keeps one realisation to fit the disk, §8).
- ~~Whether the <3 mm nodules are scored on their 3 mm stand-in disks or dropped~~: scored, as a
  row of their own (242 of the 395 val nodules; 2026-10-11, §9).

## 8. Disk

Before this demo `/` had 92 GB free of 1.9 TB. The DICOM subset is 39 GB, the pairs ≈ 27 GB,
Mayo's check 0.3 GB. The 2026-10-07 prune survey (two copies of the ImageNet tarballs at 145 GB
each; `~/hallway-2023` 129 GB; the old `../lean4-jax` checkout 55 GB; `../jax-orin-build` caches
59 GB; `.lake/build/distogram_*` 49 GB; …) is in the session's memory notes, not executed.

2026-11-02: 48 GB free. The §10 additions: Mayo's 50 chest pairs are fetched one patient at a
time, converted to records (every 4th slice, ≈ 4,300 pairs in the N = 1 format, (3 + 2)·S² ≈ 1.3 MB,
≈ 5.6 GB), and their DICOM deleted; the 10%-dose LIDC build with one realisation is ≈ 19 GB
beside the 25% build's 27 GB. Together ≈ 25 GB, leaving ≈ 23 GB.

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
- 2026-10-10: **the 300-patient build** finished in 240 min (30 workers): 300 of 300 patients,
  no skips, no unmatched XML; `train.bin` 240 patients / 9,600 records (17.6 GB), `val.bin`
  60 patients / 4,282 records (7.9 GB); 13,623 of 13,882 slices calibrated, achieved / target
  noise median 1.000 (0.884–1.186). `launch.sh` started both arms at 21:20.
- 2026-10-10: **phases 2–3 trained.** Both arms 30,000 steps, ≈ 150 ms/step on an idle 4060 Ti
  (80 min each), scored on all 4,282 val slices (60 patients never trained on); the Lean and
  Python RMSEs agree:

  | | body RMSE (HU) | whole RMSE (HU) | PSNR (dB) | SSIM (soft tissue) |
  |---|---|---|---|---|
  | low-dose floor | 83.01 | 70.52 | 37.95 | 0.7722 |
  | MSE UNet | 43.27 | 36.90 | 44.60 | 0.8560 |
  | Noise2Noise | 42.94 | 37.24 | 44.55 | 0.8564 |

  Phase 3's gate passes (0.05 dB apart). The figures (`runs/2026-10-10-lidc-denoise/{mse,n2n}/figure.png`,
  val record 483, the split's largest nodule) are near-identical between the arms: the nodule's
  outline and the lung vessels kept, the soft-tissue grain and the nodule's inner texture
  smoothed below the full-dose image's. Phase 2's gate still needs the classical arm and the
  nodule table. Checkpoints in `data/lidc/out/{mse,n2n}/`.
- 2026-10-11: **the classical arm.** `scripts/probes/lidc_classical.py <train.bin> <val.bin>
  gauss|tv|bm3d <out.eps.f32>`: each filter's one knob chosen on 48 train slices by body RMSE
  (none at its grid's edge), then run on the val split's realisation 0 and written in the
  trainer's dump format, so `lidc_denoise_score.py` scores it unchanged. BM3D is `bm3d` 4.0.3
  (free for non-commercial use; `.venv-lidc` only, in its lock file), one thread per worker:
  ≈ 10 s a slice, 43 min on 30 cores. Runs in `runs/2026-10-11-lidc-classical/`:

  | | knob | body RMSE (HU) | whole RMSE (HU) | PSNR (dB) | SSIM (soft tissue) |
  |---|---|---|---|---|---|
  | low-dose floor | | 83.01 | 70.52 | 37.95 | 0.7722 |
  | Gaussian | σ 0.6 px | 57.44 | 49.68 | 41.18 | 0.8213 |
  | TV (Chambolle) | weight 0.08 | 55.63 | 48.05 | 41.51 | 0.8000 |
  | BM3D (white) | σ 160 HU | 58.64 | 50.35 | 41.78 | 0.7758 |
  | MSE UNet | | 43.27 | 36.90 | 44.60 | 0.8560 |
  | Noise2Noise | | 42.94 | 37.24 | 44.55 | 0.8564 |

  Every filter is 12–16 HU and ≥ 2.8 dB behind both nets. BM3D's tuned σ is twice the
  simulated noise's ≈ 80 HU: the projection-domain noise is streaked and correlated, which a
  white-noise σ under-describes, so the filter needs a larger one and smooths the image to a
  blur (its figure: the vessels and the nodule's texture gone). The pooled body RMSE and the
  per-slice-mean PSNR rank the three filters differently, so no one classical number is "the"
  baseline; BM3D with the measured noise PSD is the fair form and is still open (§7).
- 2026-10-11: **BM3D with the measured noise spectrum** (`lidc_classical.py … bm3dpsd`,
  `run_bm3dpsd.sh`): E|fft2(low₀ − full)|² over 128 held-out train slices (std 78 HU; the DC bin,
  a ≈ 6 HU slice-mean offset, replaced by its neighbours), its knob a multiplier on that
  spectrum, chosen 3.0 **at the grid's edge**. Val body RMSE 58.04 HU, PSNR 41.48, SSIM 0.768:
  no better than the white-noise σ (58.64). Tuned for body RMSE, every filter is pushed to
  smooth harder than the noise, which is what costs the nodules below.
- 2026-10-11: **the nodule table** (`scripts/demos/lidc_nodules.py`, `runs/2026-10-11-lidc-nodules/`).
  A nodule on a slice = a region of >= 2 readers' agreement, matched to `nodules.json` for size
  and texture; contrast = mean HU inside − mean of a 2–6 px lung ring; **kept** = the arm's
  contrast over the full dose's, pooled over the nodule's slices; medians over nodules, 95%
  bootstrap intervals. 395 val nodules (11 regions left out: 2 thin ring, 9 unmatched), 242 of
  them the < 3 mm marks. Contrast kept, median %:

  | | n | quarter | Gauss | TV | BM3D | BM3D-PSD | MSE | N2N |
  |---|---|---|---|---|---|---|---|---|
  | < 3 mm (stand-in disks) | 242 | 100.5 | 91.0 | 79.1 | 75.6 | 69.2 | 97.8 | 97.2 |
  | subsolid 3–6 / 6–10 / >10 mm | 9 / 10 / 3 | 96–100 | 91–96 | 77–88 | 82–95 | 78–93 | 98 / 98 / 93 | 97 / 98 / 92 |
  | solid 3–6 mm | 52 | 99.7 | 94.9 | 85.8 | 90.6 | 87.2 | 98.9 | 98.7 |
  | solid 6–10 mm | 52 | 100.0 | 96.7 | 90.0 | 94.3 | 92.8 | 99.4 | 99.4 |
  | solid > 10 mm | 27 | 100.0 | 98.6 | 95.6 | 98.5 | 98.0 | 99.8 | 99.8 |
  | all | 395 | 100.0 | 93.4 | 83.0 | 85.1 | 80.2 | 98.7 | 98.6 |

  The size gradient the question predicted is there — in the **filters**: TV keeps 96% of a
  > 10 mm solid nodule and 79% of a < 3 mm one, BM3D-PSD 98% and 69%. The nets keep ≥ 97% in
  every bin with more than 3 nodules; their one dip is the 3 large subsolid nodules (92–93%,
  intervals to 89–99). So §0's expectation — "an MSE UNet erases small, low-contrast nodules" —
  does not hold at this dose for this net: the regression-to-the-mean cost is the texture (the
  soft-tissue grain, a nodule's inside), not the nodule. CNR is in `table.txt` and ranks BM3D
  first — it rewards a flat ring, as the script's docstring warns; contrast kept is the column.
- 2026-10-11: **draft figure** (`scripts/demos/lidc_figure.py <nodules.json>`,
  `runs/2026-10-11-lidc-nodules/figure_draft.png`): (a) a full-dose slice, the nodule boxed;
  (b) a < 3 mm nodule (the row's most typical by contrast kept: TV 79%, BM3D 76%, nets 95%),
  a 10 mm ground-glass one, the split's largest (32 mm), in full / quarter / TV / BM3D / MSE /
  N2N; (c) HU profiles through the first two. Every nodule is chosen by a rule in the script.

- 2026-11-02: re-framed as a synthesis use case (§0, §10): the diffusion model is the generator,
  the other arms its baselines; measured by fidelity, texture (NPS), findings (kept / invented /
  unstable) and a held-out set, Mayo chest (50 paired patients, open under CC BY 4.0 since July
  2026). Decided: residual diffusion, η = 1, 50 steps, no sweep; a 10%-dose LIDC build to match
  Mayo. Branch rebased onto `origin/main` 19c52c9a (clean; PJRT shim rebuilt, the LIDC targets
  rebuilt, `lidc_loader_check.py` and `test-dataset-record-sizes` pass).

## 10. The generator, the measurements and the held-out set (2026-10-11, re-framed 2026-11-02)

**Why the generator is the subject.** The 2026-10-07 plan (§0 then) expected the MSE UNet to
erase small, faint nodules and diffusion to keep them. The nodule table (§9, 2026-10-11) says
the regression nets keep ≥ 97% of nodule contrast in every bin; the filters are what lose small
nodules. What the nets lose is texture: their output is the posterior **mean** of the full dose
given the low dose, softer than any scan (the user's read of the samples, 2026-11-02: "too
soft"). The full dose is not clean either — its own noise is ≈ 36 HU (median `sigma_full_hu`
over the val slices) beside the ≈ 62 HU the simulator adds. A diffusion model **samples** from
that posterior, so it should put the grain back, and possibly structure that is not there. The
demo is the real-world synthesis task — low dose in, a full-dose image out — with the checks
such a use needs. A sample must cost RMSE (its error to the target is ≈ √2 × the posterior
spread above the mean's); that is the expected result, so RMSE is reported, not ranked on.

### 10.1 The model (decided 2026-11-02; no new codegen)

- Diffuse the residual, x₀ = (full − low) / 100 HU (std ≈ 0.7), conditioned on the low dose: the
  net sees [x_t, low / 1000, 8 sin/cos channels of t] (`Ddpm.prependSinCosT`, nFreq 4, as
  `MainDiffusion2d`) and predicts ε. The sample is low + 100·x̂₀, written in the dump format, so
  every scorer reads it unchanged. The residual form makes the MSE arm exactly this sampler's
  mean: the comparison is sample against mean on one variable. (The SR3 / Palette image form,
  x₀ = full / 1000, is the alternative not taken.)
- Net: `unetCtDenoise`'s layers with a 10-channel stem (≈ 7.85M params), cosine ᾱ, T = 1000,
  `Ddpm.stepInputs` + `trainStepAdamF32Ddpm` — the MNIST DDPM's path at 256² patches, 60k steps
  (2× the regression arms' 30k), Adam 5e-4 cosine, as theirs.
- Sampler: DDIM with η = 1 (ancestral — every step adds fresh noise, so the grain is the model's,
  not suppressed), 50 steps, at 512² (`Ddpm.ddimCoefs` / `ddimStep`; η and the step count are
  flags, no sweep).
- New code: one C helper to concatenate channel blocks ([B,1,H,W] ⊕ [B,1,H,W]); `--arm=diff`
  in `lidc-denoise-train`; a sampler exe (`lidc-denoise-sample`) writing one dump per sample
  seed. Gate on the smoke volume first (the phase-1 recipe).

### 10.2 Texture — "too soft" as a number

- The 2-D noise power spectrum (NPS, the CT physics measure of noise texture), radially
  averaged, over flat soft-tissue ROIs (64², local std below a threshold in the full dose, no
  reader mark), of output − (the full dose's local mean); against the full dose's own NPS on the
  same ROIs. Reported: the integral ratio (grain magnitude) and the mean frequency ratio (grain
  coarseness — a soft output sits at low frequency). 1.0 / 1.0 = the scan's grain.
- The simpler local-std ratio (7 × 7 std over the same pixels, output / full) beside it.

### 10.3 Findings — kept, invented, unstable

- Kept: the contrast-kept table (§9), per sample, with its spread across samples.
- Invented: blobs (LoG at 1–3 mm scales, above a contrast threshold) in lung no reader marked,
  per slice, against the full dose's own count on the same pixels; reported as excess per slice.
- Unstable: the per-pixel std across 4 samples, inside nodules vs. in plain lung — where the
  generator is unsure is where invention would show.

### 10.4 The held-out set — Mayo chest

- **Data.** TCIA LDCT-and-Projection-data (CC BY 4.0; §3): 50 chest patients with both image
  arms, Siemens, 17,313 slice pairs, 18.2 GB as DICOM. A converter
  (`scripts/datasets/preprocess_mayo.py`) fetches one patient at a time, writes every 4th slice
  in the §5 record format with N = 1 (full, low, an all-zero mask) and deletes the DICOM
  (≈ 4,300 records, ≈ 5.6 GB, §8). No Mayo image is used for training or for choosing a knob.
- **What differs from training.** Patients, institution, scanners (Siemens vs LIDC's mix),
  reconstruction kernels (sharper), slice thickness, and the noise insertion (the vendor's,
  not our simulator) — the held-out claim is about all of these at once.
- **Dose.** Mayo chest is 10% of routine dose; the LIDC build is 25%. So a second LIDC build at
  `--dose 0.10` (one realisation, ≈ 19 GB, ≈ 4 h on 30 cores) trains the MSE net and the
  generator at Mayo's dose; the classical filters re-tune their knob on it. Mayo abdomen (25%)
  is not used: no lung, no nodules.
- **First check, no GPU.** The NPS of our simulated noise (low − full) against Mayo's inserted
  noise at matched σ: if the shapes agree, results on our pairs speak for a scanner's.
- **What Mayo can't do.** It has no reader outlines, so nodule contrast kept stays an LIDC
  result; Mayo reports fidelity, texture and invented blobs. Its low-dose arm is itself
  simulated (by the vendor) — it is the field's reference, not a second real low-dose scan.

### 10.5 Runs and compute (4× 4060 Ti; eval forward at 512² ≈ 0.11 s)

| run | where | cost |
|---|---|---|
| 10% LIDC build | CPU, 30 workers | ≈ 4 h (waits for the GPU job's data loader, or is paused with it) |
| MSE at 10% | 1 GPU, 30k steps | ≈ 1.3 h |
| diffusion at 25% and at 10% | 1 GPU each, 60k steps | ≈ 2.5 h each, in parallel |
| sampling, per dose: 1,205 nodule slices + 400 others, one sample | 50 DDIM steps ≈ 5.5 s a slice | ≈ 2.5 GPU-h |
| sampling, per dose: 4 samples of 200 slices (spread) | | ≈ 1.2 GPU-h |
| sampling on Mayo at 10%: 1,000 slices (20 a patient), one sample | | ≈ 1.5 GPU-h |
| regression nets + filters on Mayo, all ≈ 4,300 slices | GPU minutes; BM3D ≈ 45 min CPU | small |

Total ≈ 15 GPU-hours: ≈ 2.5 h wall for the three trainings side by side, then ≈ 2.5 h of
sampling over four cards. The user pauses the running GPU job when the slot is needed.

### 10.6 The section

One figure: (a) a Mayo low-dose slice and its generated full dose beside the real one (the
object first); (b) nodule crops (LIDC) in full / low / BM3D / MSE / two diffusion samples;
(c) the radial NPS of full dose, MSE and diffusion. One table: arms × {body RMSE, NPS integral
and mean-frequency ratio, nodule contrast kept (all / < 3 mm), excess blobs}, LIDC at 25% and
10%, and Mayo at 10%.
