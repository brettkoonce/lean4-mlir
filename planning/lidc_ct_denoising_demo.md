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
LDCT-and-Projection-data (CC BY 4.0, chest and abdomen open since v7, August 2025), are the held-out
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

**The Mayo check.** TCIA's LDCT-and-Projection-data v7 (2025-08-14; the July 2026 note on the page
is a clinical-data file refresh, checked 2026-11-02) opened its chest and abdomen arms under
CC BY 4.0 (199 patients, 698 CT series, 863 GB with projections; the 99 head patients remain
under controlled access). Its low-dose arm is itself *simulated*, by Siemens/GE noise insertion in the
projection domain at 10% (chest) / 25% (abdomen) of routine dose — so the slate's "why not real
pairs?" is answered by the reference set being simulated too, and it becomes the check. Both
Mayo arms are reconstructions of one scan, so low − full is exactly the inserted noise. On
C002 (Siemens, B50f, 273 mA, 1.5 mm; 20 mid-volume slices) by the estimator above: own 66.1 HU,
inserted 196.7, total 208.2; **inserted/own 2.99 against the 3.00 its dose predicts, total/own
3.15 against √10 = 3.16.** By this estimator, on that patient, the "added variance =
(1/dose − 1) × own" rule is the field's construction to 1%. `scripts/probes/lidc_mayo_noise_check.py`
reproduces it (300 MB, `data/mayo/dicom/C002/`). **Across all 50 paired patients (2026-11-02,
`runs/2026-11-02-lidc-mayo/dose_check.txt`) the median is 3.43 [IQR 3.07–3.76], total/own 3.57:**
C002 sits at the low end (per patient 2.73 … 3.48 … 4.60; 21 of 50 within ±10% of 3.00), and the
ratio rises with the patient's own noise (r = 0.68) and tube current (r = 0.51). So the vendor
inserts ≈ 14% more noise than the pure rule — the equivalent of ≈ 8% dose by the rule — and more in
the large, noisy patients; our 10% build (§10.4) follows the rule. The same collection is the
held-out test set (§10.4): 50 chest patients have both image arms (all Siemens; the 49 GE chest
patients have full-dose images and projections only), 17,313 slice pairs, 18.2 GB as DICOM,
CC BY 4.0, no login.

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
| 4 | the generator: conditional residual diffusion + DDIM sampler, and the texture / invention scorers (§10.1–§10.3) — **code done 2026-11-02** (§9): `--arm=diff`, `lidc-denoise-sample`, `lidc_texture.py`, plumbing smokes green; the 1,500-step smoke fit runs end to end but meets only half its gate (sample RMSE 46.9 ≈ the floor's 48.8, MSE 22.4; NPS integral 1.77 vs MSE's 0.71, mean frequency 1.36 vs 0.64 — too much grain at 2.5 % of the planned steps; §9) | smoke fit: samples on the fitted volume score near the MSE net's RMSE, NPS nearer the full dose's than MSE's |
| 4a | the held-out set: Mayo chest pairs converted to the record format; NPS of our simulated noise vs. the vendor's (§10.4) — **done 2026-11-02**: `data/mayo/val.bin`, 4,348 records; the spectrum check **fails** its gate (Mayo's inserted noise peaks at 0.425 /mm, ours at 0.330, ratio 1.29 — the kernel, §9; §7 has the choice) | the two noise spectra agree in shape (peak frequency within ~10%) at matched σ |
| 4b | the 10%-dose LIDC build (Mayo chest is 10% of routine dose), MSE + diffusion retrained at 10% (§10.4) — **4-patient smoke build 2026-11-02**: achieved/target median 0.998 (0.946–1.059); `runs/2026-11-02-lidc-diffusion/build_lidc10.sh` is the full build | the build's achieved/target σ as phase 0's |
| 5 | runs and scoring: diffusion at 25% and 10%, every arm on Mayo with no retraining (§10.5) — scripts written 2026-11-02 (`launch_train.sh`, `launch_sample.sh`, `score.sh`), nothing launched | texture ratio nearer 1 than MSE's; nodule contrast kept ≥ 95%; invented blobs near the full dose's count; the ranking reported as found on Mayo |
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
- ~~The blob threshold for "invented findings"~~: decided 2026-11-02 on the 25% val (§9) — the LoG
  response threshold 0.15 of the [−1000, −300] window (105 HU) is the headline, 0.25 (175 HU)
  beside it; at 0.08 the full dose alone carries 350 blobs a slice (vessel cross-sections and the
  grain), which is texture, not findings.
- **The kernel gap to Mayo (open, the user's call).** §6 phase 4a's gate fails: Mayo's inserted
  noise is reconstructed through B50f (a sharp kernel) and peaks at 0.425 /mm against our
  Shepp-Logan 0.330 /mm (ratio 1.29); Mayo's full-dose grain is sharper still against LIDC's
  STANDARD-kernel scans (peak 0.43 vs 0.10 /mm). Two readings: (a) leave it — §10.4 already names
  the kernel among the things that differ at once, and the held-out result is read with the
  measured gap beside it (the §10.6 table gets a "noise peak frequency" row for both sets); (b)
  close part of it — a sharper apodisation in `preprocess_lidc.py` (scikit-image's `ramp`, or a
  B50f-like custom boost of the mid frequencies) and a rebuild of the 10% set (≈ 4–6 h CPU),
  which moves the inserted noise's spectrum but not the scans' own grain. Not done unasked.
  **Taken 2026-11-03 (option b), after §11.6 left the kernel as the generator's open question: §12.**
- Sampling batch: the sampler draws independent per-record noise, so `--batch` changes
  throughput and nothing else; the launch scripts use batch 1 per shard (memory-safe at 512²),
  to be raised if the 512² forward leaves the card idle.

## 8. Disk

Before this demo `/` had 92 GB free of 1.9 TB. The DICOM subset is 39 GB, the pairs ≈ 27 GB,
Mayo's check 0.3 GB. The 2026-10-07 prune survey (two copies of the ImageNet tarballs at 145 GB
each; `~/hallway-2023` 129 GB; the old `../lean4-jax` checkout 55 GB; `../jax-orin-build` caches
59 GB; `.lake/build/distogram_*` 49 GB; …) is in the session's memory notes, not executed.

2026-11-02: 48 GB free; 190 GB after the user cleared old checkpoints the same day. The §10 additions: Mayo's 50 chest pairs are fetched one patient at a
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
  unstable) and a held-out set, Mayo chest (50 paired patients, open under CC BY 4.0 since v7,
  August 2025). Decided: residual diffusion, η = 1, 50 steps, no sweep; a 10%-dose LIDC build to match
  Mayo. Branch rebased onto `origin/main` 19c52c9a (clean; PJRT shim rebuilt, the LIDC targets
  rebuilt, `lidc_loader_check.py` and `test-dataset-record-sizes` pass).
- 2026-11-02: **phase 4 code, 4a converter, 4b smoke, 5 scripts — prep without a run** (the
  4-GPU MNv4 job kept the cards; everything below ran on CPU or on a 2.4–3.3 GB sliver of one
  card at `LEAN_MLIR_MEM_FRACTION`).
  - *Lean.* `lidc-denoise-train --arm=diff` (§10.1 as built): x₀ = (target − low)/100, the DDPM
    forward process per image (`Ddpm.stepInputs`), the net's input `[x_t, low]` by a new C helper
    `F32.concatChannels` then `Ddpm.prependSinCosT` (nFreq 4, T 1000) — `unetCtDenoise` takes
    `inC` (10 for this arm, 7,851,617 params; 1 otherwise), the one layer that differs. The
    realisation count is read off the pair file (`F32.ctRealisations`): N = 2 (25% build) gives
    two pairings an epoch, N = 1 (10% build, Mayo) one; `--arm=n2n` is refused at N = 1.
    `--patch`/`--batch` for small smokes; artifact names carry the arm, seed, a non-`lidc` data
    root (so the 25% and 10% generators never share a params file) and a non-default patch.
    `--eval-only --ckpt=<dir>` scores another file with a finished run (the 10% MSE net on Mayo).
    `lidc-denoise-sample`: DDIM over the cosine table, η = 1 default, 50 steps, per-record noise
    seeded by (k, record, step) so batch and shard change nothing; `--records=`, `--shard=i/n`;
    writes the trainer's dump layout at the record's offset (`F32.truncateFile`/`pwriteFile`,
    new C helpers — Lean's `Handle` has no seek; unwritten records are a hole) as
    `<dump>/<root>_val.k<K>.eps.f32`, −x̂₀ so denoised = low − 100·eps as for every other arm,
    plus a `.records` sidecar appended per record (resume, shards, and the scorers' subset).
    `test-lidc-loader` and `lidc_loader_check.py` take N from the file; `test-dataset-record-sizes`
    gains the N = 1 cases (skip until the files exist).
  - *Python.* `scripts/datasets/lidc_records.py` is the one reader (N from the file size,
    sidecars, `find_dumps`); `lidc_denoise_score.py`, `lidc_nodules.py`, `lidc_figure.py`,
    `lidc_classical.py` ported to it with `--records=` and the sidecar honoured — the 25% numbers
    reproduce exactly (`runs/2026-11-02-lidc-texture/regression.txt`: body 43.27, the "all" row
    93.4 / 83.0 / 85.1 / 80.2 / 98.7 / 98.6). `lidc_select_records.py` writes the §10.5 sets by
    rule: `records_val.txt` = the 1,205 nodule records + 400 evenly spaced others;
    `records_spread.txt` = the 100 nodule records with the most marked pixels + 100 of those
    others; `--per-patient=20` for Mayo. `lidc_texture.py` = §10.2–§10.3 (settings there).
    `scripts/datasets/preprocess_mayo.py` + `scripts/probes/lidc_noise_nps_check.py` (§10.4).
  - *Smokes* (`scratchpad/gpu_smoke.log`): 4-patient pair builds at 25%/N = 2 and 10%/N = 1
    (achieved/target 0.999 and 0.998 medians); loader gates PASS at both N; MSE arm end to end
    at 64² × 2 (train, 512² eval, dump); n2n refused at N = 1; MSE at N = 1; diffusion arm 60 steps
    (ε-MSE falls from the first step), checkpoint resume 60 → 70; the sampler at a 128² centre
    crop at batch 1 and 2, its resume, sparse dumps finite with zeros where the sidecar says.
  - *Texture of the regression arms* (`lidc_texture.py` on `records_val.txt`, 1,029 slices with a
    qualifying ROI, 9,960 ROIs; the full dose's residual 40.2 HU rms, mean frequency 0.246 /mm):

    | | body RMSE | NPS integral ratio | mean-frequency ratio | local-std ratio |
    |---|---|---|---|---|
    | low dose | 72.00 | 3.73 [3.64, 3.81] | 1.75 [1.69, 1.80] | 2.58 |
    | Gaussian | 50.51 | 1.59 | 1.12 | 1.54 |
    | TV | 48.21 | 0.59 | 0.62 | 0.50 |
    | BM3D / BM3D-PSD | 49.4 / 49.3 | 0.43 / 0.38 | 0.48 / 0.47 | 0.17 / 0.18 |
    | MSE UNet | 37.48 | 0.64 [0.62, 0.66] | 0.63 [0.63, 0.64] | 0.54 |
    | Noise2Noise | 37.13 | 0.64 [0.62, 0.66] | 0.63 [0.62, 0.64] | 0.53 |

    "Too soft" as numbers: the nets carry 64% of the scan's grain variance at 63% of its mean
    frequency. Two things the plan did not anticipate: the low dose's mean-frequency ratio is
    1.75 — the simulated projection noise (Shepp-Logan ramp) is *sharper* than the scanners' own
    kernel noise, which is exactly what §10.4's first check against Mayo's vendor noise measures;
    and the integral ratio reads 3.73, not ≈ 4, because the σ = 8 px local-mean subtraction takes
    the lowest frequencies out of both. Blobs a slice (threshold 0.15, the 200 spread records):
    full dose 110.9; low dose 296.7 (189 invented / 14 dropped); MSE 59.7 (11 / 60); N2N 53.5
    (9 / 64); Gaussian 75.2 (27 / 60); TV 24.0 (10 / 95); BM3D 23.4 (10 / 96) — the regression
    nets invent little and drop half; a generator should move both toward the full dose's own.
  - *Mayo (phase 4a).* `scripts/datasets/preprocess_mayo.py` fetched and converted all 50 chest
    pairs in 24.7 min (3 workers): `data/mayo/val.bin`, 4,348 records (every 4th of 17,313
    z-matched slices, none unmatched), 5.70 GB at N = 1, `val.idx` / `val.json` /
    `manifest_mayo.json`; the DICOM deleted but C002 (the check script's). The data root moved to
    `data/mayo/` (`dicom/<pid>/{full,low}`). All 50 are Siemens B50f, 120 kVp, 1.5 mm, 103–650 mA,
    pixel spacing 0.586–0.977 mm (median 0.742; recorded per row). Loader gate PASS at N = 1;
    `records_mayo.txt` = 20 a patient, 1,000 records. The dose check across the set: inserted/own
    median 3.43, not 3.00 (§3). `scripts/probes/lidc_noise_nps_check.py`
    (`runs/2026-11-02-lidc-mayo/`): our simulated noise peaks at 0.330 /mm (mean f 0.351), Mayo's
    inserted noise at 0.425 (0.398) — **ratio 1.29, the phase-4a gate fails**; the scanners' own
    grain differs more (LIDC 0.100 /mm, Mayo 0.430). §7 has the choice.
  - *The phase-4 smoke gate* (`scratchpad/gpu_gate.log`; a 3.3 GB sliver of GPU 3 beside the JAX
    job, ≈ 320 ms a step): the phase-1 recipe, 1,500 steps of 8 × 256² on the 4-patient build's one
    val volume (68 records, 136 pairs), both arms, then 24 samples (the nodule slices first) at
    50 DDIM steps, η = 1, batch 4 (2.1 s a 512² slice on the shared card — the plan's 5.5 s was
    conservative). MSE: body RMSE 44.6 → 21.3 on the fitted volume (the 2026-10-10 smoke's
    40.7 → 20.3 on another volume). Diffusion: ε-MSE 3.33 → 0.72 (step 100) → 0.19 (1,500), still
    falling. On the 24 records: floor 48.77 HU → MSE 22.40 → sample 46.86; PSNR 40.6 → 47.2 → 40.0;
    texture on 14 slices / 98 ROIs, full dose = 1: low dose 2.50 / 2.09 (integral / mean
    frequency), MSE 0.71 / 0.64, **sample 1.77 / 1.36**; blobs a slice full 22.5, low 39.9
    (19.5 invented), MSE 19.5 (4.0 / 6.9 dropped), sample 17.8 (4.8 / 9.1). So the pipe works —
    a sample is a slice with grain, not the MSE net's blur — but at 2.5 % of the planned steps the
    generator puts back more grain than the scan has and costs the floor's RMSE, not the
    √2 × 22 ≈ 32 HU a posterior sample should; the gate as written is half met. Read at 60,000
    steps before touching the recipe. One knob noted, not turned (§10.1 says no sweep): x₀ =
    (full − low)/100 has whole-slice std ≈ 0.36 (≈ 0.7 only over the body), below the unit
    variance the ᾱ schedule assumes; /50 would centre it.
- 2026-11-02 (evening): **the runs, launched** after the user paused the MNv4 job (≈ 17:00).
  - The 10% build finished at 19:14 (`data/lidc10/`): 240 / 60 patients, 9,600 / 4,282 records
    (the counts and the patient split identical to the 25% build's), 12.6 + 5.6 GB, 13,623 of
    13,882 slices calibrated, achieved/target median 1.001 (0.90–2.08; one outlier slice). Loader
    gates and the record-size test pass at N = 1. Wall time: the build is CPU-bound and the box is
    16 physical cores with SMT — 30 workers beside the GPU trainer were no faster than 14 and
    tripled the trainer's step time (553 ms against 160); 14 workers (one a core) is the setting.
    `preprocess_lidc.py --resume` (new) made the restarts free.
  - Training: `diff25` on GPU 0 from 17:09 (60k steps, ≈ 160–230 ms a step), `diff10` on GPU 1
    and `mse10` on GPU 2 from 19:16 (`ARMS=` selects arms in `launch_train.sh`).
  - The 25%-trained MSE and Noise2Noise nets scored on Mayo as they are (a dose-mismatch row, not
    in the plan; `data/mayo/out/{mse25,n2n25}/`): body RMSE 233.1 → 134.4 / 133.7 HU, whole
    202.1 → 115.9 / 118.6, SSIM 0.656 → 0.711 / 0.710 — nets trained at a quarter of the dose on
    other scanners still take 42% of the vendor's noise out.
  - Filters (`STAGES=filters score.sh`, knobs re-tuned on the 10% train split) on the 10% val and
    on Mayo, 10 CPU workers beside the trainers, from 19:17.
  - Mayo has two records (C158, slices 0 and 4) whose arms are identical — no noise inserted —
    found by the scorer's PSNR dividing by zero; §10.4 has the handling.
  - `mse10` done 20:35 (30k steps, 145 ms a step): 10% val body RMSE 140.72 → 52.17 HU, whole
    119.41 → 44.60, PSNR 33.32 → 42.82, SSIM 0.679 → 0.804 (Lean and Python agree). On Mayo
    (`--eval-only --ckpt`, `data/mayo/out/mse/`): body 233.07 → 109.40, whole 202.07 → 94.99 —
    25 HU better than the 25%-trained net's 134.4: the training dose matters on the held-out set.
  - **`diff25` done 21:05** (60k steps, 160–230 ms a step, final ε-MSE 0.20). Sampling on GPUs 0
    and 3 (two shards, batch 4, 1.1 s a 512² slice): the 1,605-record scoring set at k = 0 in
    16 min, k = 1–3 on the 200 spread records in 6 min. **The 25% generator's result**
    (`runs/2026-11-02-lidc-diffusion/score/{fidelity_lidc_*,texture_lidc,spread_lidc_*}`):

    | arm (1,605 records) | body RMSE | PSNR | SSIM | NPS integral | mean-freq | local-std | blobs / invented / dropped |
    |---|---|---|---|---|---|---|---|
    | low dose | 72.00 | 38.69 | 0.791 | 3.73 | 1.75 | 2.58 | 193 / 118 / 9 |
    | MSE UNet | 37.48 | 45.44 | 0.875 | 0.64 | 0.63 | 0.54 | 50 / 7.2 / 37 |
    | Noise2Noise | 37.13 | | | 0.64 | 0.63 | 0.53 | 47 / 6.2 / 38 |
    | BM3D-PSD | 49.34 | | | 0.38 | 0.47 | 0.18 | 7 / 2.4 / 76 |
    | **diffusion k0** | **54.10** | 42.19 | 0.819 | **0.87** [0.85, 0.88] | **0.81** [0.80, 0.83] | **0.92** | 57 / 15.5 / 38 |
    | full dose | 0 | | | 1.00 | 1.00 | 1.00 | 80 / — / — |

    The sample costs the RMSE a posterior draw should — √2 × 37.5 = 53.0 against 54.1 measured —
    and gives back 87% of the scan's grain variance at 81% of its mean frequency, against the MSE
    net's 64% and 63%; by local std it is at 92% of the scan's. Findings: it drops as many
    full-dose blobs as the MSE net (38 a slice against 37) and invents twice as many (15.5 against
    7.2; the low dose 118, the full dose's own count 80). Spread (200 records, 4 samples): per-pixel
    std 21 HU inside nodules, 10 in plain lung, 15 in soft tissue; the mean of the 4 samples
    scores 62.0 HU body RMSE against each sample's 74 and the MSE net's 49.2 on those records —
    the samples' mean sits above the regression net, so the sampler is not posterior-exact (4
    exact draws would average to ≈ 49 × √1.25 = 55). **Nodules** (`nodules_lidc.txt`, the 395
    val nodules as in §9's 10-11 table): contrast kept, median %, diffusion k0 beside the MSE net —
    all 98.7 [98.1, 99.3] against 98.7; < 3 mm 97.3 against 97.8; solid 3–6 mm 98.5 / 98.9; solid
    > 10 mm 99.8 / 99.8; the three large subsolid ones 95.4 / 92.5. The generator keeps the nodules
    as the regression net does — the phase-5 gate's "≥ 95%" holds in every row — while putting the
    grain back. On the 200 spread records (by construction the slices of the 15 largest nodules)
    the four samples keep 99.6–99.8 each and 99.7 averaged.
  - The 25%-trained generator on Mayo (a dose-mismatch row like `mse25`,
    `data/mayo/out/diff25/`, 1,000 records): body RMSE 235 → 227 HU — a conditional sampler trained
    on 62 HU of noise draws residuals at its training scale and leaves the rest of Mayo's 194 in
    place; the 25% MSE net took the same floor to 134. The 10% generator's Mayo pass is the one
    that counts.
  - **`diff10` done 22:05** (60k steps, 149–170 ms a step, final ε-MSE 0.23). Sampling on four
    shards: the 10% scoring set at k = 0 (8 min), k = 1–3 on the spread set, then Mayo at k = 0
    (1,000 records). On its own 10% val the pooled body RMSE goes 124 → 75 HU over one shard
    (the MSE net 52; √2 × 52 = 74 — the posterior-draw cost again). **The 10% tables**
    (`score/*lidc10*`, the 1,605-record scoring set, BM3D's knob at its grid's edge, 0.25):

    | arm | body RMSE | PSNR | SSIM | NPS integral | mean-freq | local-std | blobs / invented / dropped | nodules kept, all / < 3 mm |
    |---|---|---|---|---|---|---|---|---|
    | low dose (10%) | 122.84 | 34.06 | 0.704 | 9.17 | 1.91 | 4.16 | 497 / 415 / 8 | 99.9 / 100.0 |
    | Gaussian | 66.01 | 39.01 | 0.778 | 1.39 | 0.79 | 1.35 | 15 / 4 / 69 | 81.9 / 75.6 |
    | TV | 66.50 | 39.85 | 0.783 | 0.94 | 0.86 | 0.75 | 22 / 12 / 71 | 77.2 / 69.3 |
    | BM3D | 68.73 | 40.42 | 0.776 | 0.50 | 0.78 | 0.22 | 13 / 8 / 75 | 70.6 / 48.8 |
    | MSE UNet | **46.14** | 43.56 | 0.832 | 0.56 | 0.52 | 0.42 | 37 / 6.6 / 50 | 97.5 / 93.9 |
    | **diffusion k0** | 73.35 | 40.02 | 0.762 | **1.10** [1.09, 1.12] | **0.97** [0.95, 0.99] | 1.22 | 82 / 42 / 40 | 97.0 / 92.8 |
    | full dose | 0 | | | 1.00 | 1.00 | 1.00 | 80 / — / — | 100 |

    At a tenth of the dose the picture sharpens: the regression net is softer still (56% of the
    grain variance at 52% of its frequency) and the generator lands on the scan's texture — 110%
    of its variance at 97% of its mean frequency, and the full dose's own blob count (82 against
    80) — but now invents 42 blobs a slice to the MSE net's 7 and costs more than √2 in RMSE
    (73.4 against √2 × 46.1 = 65.3). Both nets keep ≥ 97% of nodule contrast over all 395 and
    about 93% on the sub-3 mm marks (the filters 49–76% there); the spread across four samples is
    32 HU inside nodules, 17 in plain lung, 23 in soft tissue; the mean of four samples scores
    78.1 HU against each sample's 101 on the spread set. BM3D's knob sat at its grid's edge (0.25)
    on this build, as BM3D-PSD's did on the 25% one.
  - **Mayo, the held-out set** (`score/{fidelity_mayo_*,texture_mayo.txt,mayo_diff10_tail.txt}`,
    1,000 records; BM3D landed 23:41, its knob at its grid's edge again):

    | arm | body RMSE | PSNR | SSIM | NPS integral | mean-freq | local-std | blobs / invented / dropped |
    |---|---|---|---|---|---|---|---|
    | low dose (vendor, 10%) | 235.0 | 27.2 | 0.654 | 13.95 | 1.10 | 3.67 | 2113 / 1903 / 11 |
    | Gaussian (10%-tuned) | **93.3** | 33.8 | 0.694 | 1.01 | 0.52 | 0.84 | 12 / 5 / 154 |
    | TV (10%-tuned) | 126.0 | 33.0 | 0.705 | 2.21 | 0.89 | 0.81 | 100 / 77 / 139 |
    | BM3D (10%-tuned, knob at the grid's edge) | 141.2 | 32.8 | 0.692 | 2.45 | 1.02 | 0.33 | 36 / 25 / 152 |
    | MSE UNet, 10% | 112.1 | 34.4 | **0.719** | 1.82 | 1.11 | 0.80 | 135 / 76 / 109 |
    | MSE / N2N, 25% | 137.5 / 137.1 | 32.8 / 32.5 | 0.710 / 0.709 | 3.2 / 3.3 | 1.11 / 1.12 | 1.25 / 1.23 | 317 / 221 / 79; 266 / 182 / 89 |
    | diffusion, 25% | 227.1 | 30.8 | 0.645 | 18.2 | 1.01 | 1.52 | 294 / 217 / 96 |
    | **diffusion, 10%** | **457.8** | 28.8 | 0.629 | 55 [14, 109] | 1.07 | 1.39 | 418 / 329 / 92 |
    | diffusion, 10%, clamped to the HU range each step (11-03) | 292 (median slice 122) | | | | | | 3.4% of pixel-steps clamped |
    | full dose | 0 | | | 1.00 (59 HU rms, f̄ 0.438 /mm) | 1.00 | 1.00 | 162 / — / — |

    The ranking on Mayo is the reverse of LIDC's. The generator trained at Mayo's dose **diverges**
    on the heavy slices: Mayo's insertion puts 300–800 HU of noise on the lower chest and
    shoulders of large patients, far outside the 10% build's range (median added σ 117 HU), and
    there the ancestral DDIM sampler runs away (|sample| up to 27,000 HU; 274 of 1,000 records
    worse than the floor; the median slice still improves, 202 → 132 HU against the MSE net's 82
    — `mayo_diff10_tail.txt` has the per-record picture). The generator trained at 25% does not
    diverge but draws residuals at its own 62 HU scale and leaves the rest of Mayo's noise in
    place (227). The regression nets degrade gracefully — the 10% MSE net is the best net on Mayo
    by every fidelity number and its texture sits near the scan's (integral 1.8, mean frequency
    1.11: on Mayo it is residual noise, not softness, that is left) — and a Gaussian blur with its
    one knob re-tuned on our 10% train split beats every net on RMSE, at the price of the lowest
    mean frequency (0.52) and 154 of the full dose's 162 blobs a slice dropped. So the held-out
    result, read with §3's and §7's gaps beside it (heavier and sharper noise than ours): the
    posterior-sampling story holds on the distribution it was fitted to and does not transfer
    across a 2× noise scale and a kernel; the regression nets transfer; a tuned blur transfers
    best on RMSE and worst on findings. By the slice's inserted-noise level (the training build's
    added σ is 117 HU at the median, 341 at most; medians of per-record body RMSE, floor / diff10 /
    mse10, and the share of slices the sample leaves worse than the floor): under 120 HU
    110 / 69 / 49, 0%; 120–180: 156 / 93 / 64, 1%; 180–250: 212 / 158 / 90, 24%; 250–350:
    286 / 400 / 132, 64%; over 350: 391 / 793 / 194, 100% — the generator is itself up to the
    training median and runs away from the top of its training range on, where the noise is
    also the wrong texture; the MSE net is never worse than the floor. The clamp (§11, run 2026-11-03
    00:08 as `data/mayo/out/diff_clamp/`): pooled 455 → 292 HU, p99 1,970 → 989, the median slice
    132 → 122, slices worse than the floor 274 → 261 of 1,000 — it bounds the tail (no value past
    3,071 HU) and fixes nothing; 3.4% of pixel-steps moved. The dose range is §11.

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
- As built (`scripts/demos/lidc_texture.py`, 2026-11-02): 64² tiles on a 32-px grid, every
  full-dose pixel in [−200, 300] HU, no reader mark, the tile's residual std below 2.5 × the
  slice's `sigma_full_hu` after subtracting the full dose's local mean (Gaussian σ = 8 px, the
  same mean for every arm); NPS in HU²·mm² at the file's pixel spacing; the two ratios pooled over
  slices with a 1,000-resample bootstrap over slices (seed 0); the radial curves on a 0.02 /mm
  grid for the figure (`--nps-csv`). Arms and record subsets through `lidc_records.py`.

### 10.3 Findings — kept, invented, unstable

- Kept: the contrast-kept table (§9), per sample, with its spread across samples.
- Invented: blobs (LoG at 1–3 mm scales, above a contrast threshold) in lung no reader marked,
  per slice, against the full dose's own count on the same pixels; reported as excess per slice.
- Unstable: the per-pixel std across 4 samples, inside nodules vs. in plain lung — where the
  generator is unsure is where invention would show.
- As built (2026-11-02): blobs by `skimage.feature.blob_log` in lung (full < −500, eroded 3 px,
  3 px clear of any reader mark) on the [−1000, −300] window, σ from 1 to 3 mm over √2, 5 scales,
  threshold 0.15 (§7; 0.25 beside it), matched to a full-dose blob within 3 px; the spread as the
  per-pixel std across `diff.k0 … k3` in nodules / plain lung / soft tissue plus the RMSE of the
  sample mean; the per-sample contrast kept is `lidc_nodules.py`'s extra columns. The spread
  records are a subset of the scoring set, so seed 0 serves both.

### 10.4 The held-out set — Mayo chest

- **Data.** TCIA LDCT-and-Projection-data (CC BY 4.0; §3): 50 chest patients with both image
  arms, Siemens, 17,313 slice pairs, 18.2 GB as DICOM. The converter
  (`scripts/datasets/preprocess_mayo.py`, run 2026-11-02) fetches one patient at a time, writes
  every 4th slice in the §5 record format with N = 1 (full, low, an all-zero mask) and deletes the
  DICOM: `data/mayo/val.bin`, 4,348 records, 5.70 GB (§9). Two of them (C158's first two sampled
  slices) have identical arms — the vendor inserted no noise there — so they are not pairs:
  `lidc_select_records.py` skips them and `lidc_denoise_score.py` leaves a zero-error slice out of
  the PSNR mean. No Mayo image is used for training or for choosing a knob.
- **What differs from training.** Patients, institution, scanners (Siemens vs LIDC's mix),
  reconstruction kernels (sharper), slice thickness, and the noise insertion (the vendor's,
  not our simulator) — the held-out claim is about all of these at once.
- **Dose.** Mayo chest is 10% of routine dose; the LIDC build is 25%. So a second LIDC build at
  `--dose 0.10` (one realisation, ≈ 19 GB, ≈ 4 h on 30 cores) trains the MSE net and the
  generator at Mayo's dose; the classical filters re-tune their knob on it. Mayo abdomen (25%)
  is not used: no lung, no nodules. Measured 2026-11-02 (§3): Mayo's insertion runs ≈ 14% above
  the (1/dose − 1) rule the build follows, so the held-out noise is a little heavier than the
  training noise as well as sharper.
- **First check, no GPU.** The NPS of our simulated noise (low − full) against Mayo's inserted
  noise at matched σ: if the shapes agree, results on our pairs speak for a scanner's. Done
  2026-11-02 (`scripts/probes/lidc_noise_nps_check.py`): they do not — peak 0.330 vs 0.425 /mm,
  the kernel (§7, §9). The result on Mayo is read with that gap beside it.
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

The run dir is `runs/2026-11-02-lidc-diffusion/`: `build_lidc10.sh` (phase 4b, CPU),
`launch_train.sh` (diff25 / diff10 / mse10 on GPUs 0–2, staggered, resumable), `launch_sample.sh`
(each sampling job as four shards of one dump over GPUs 0–3, in the order above, resumable),
`score.sh` (the 10% MSE net and the re-tuned filters on Mayo, then fidelity / nodules / texture
tables per data root into `score/`), and the record lists. Launch order: 4b, then the user
pauses the GPU job, `launch_train.sh`, `launch_sample.sh`, `score.sh`.

### 10.6 The section

One figure: (a) a Mayo low-dose slice and its generated full dose beside the real one (the
object first); (b) nodule crops (LIDC) in full / low / BM3D / MSE / two diffusion samples;
(c) the radial NPS of full dose, MSE and diffusion. One table: arms × {body RMSE, NPS integral
and mean-frequency ratio, nodule contrast kept (all / < 3 mm), excess blobs}, LIDC at 25% and
10%, and Mayo at 10%.

## 11. The generator across the dose range (proposed 2026-11-02, after the Mayo result)

Status 2026-11-03 01:40: the code of §11.3 is in (the trainer's and sampler's `--sigma`, the
sampler's `--clamp`, `preprocess_lidc.py --dose-range/--filter mix`, `ct_noise_level.py`, the
scorers' `--by=`, `runs/2026-11-03-lidc-range/{build,launch_train,launch_sample,score}.sh`);
the 4-patient smoke build and its sidecars exist and the σ̂ arm trains, resumes and samples at 64²
(7,851,905 params at inC 11); the clamp-only row is measured (§9); the range build is at
159/300 patients (launched 00:12, 14 workers, ≈ 1 h left; 160 s a patient against 60 s for the
fixed-dose builds, the five filters each costing their own FBP) and is chained to `build.sh`'s
gates, the five sidecars and `launch_train.sh` on GPUs 0–2. The estimator, measured on the smoke
build: the 3×3-MAD proxy of §11.2 is flat across the dose quartiles (0.54–0.58 of the total
noise) but reads the kernel as well as the level — 0.82 of the level on ramp-filtered noise, 0.39
on Hann — while a Gaussian-8 high-pass (`--estimator g8`) is flat across the filters (1.16–1.33)
and rises as the noise falls (1.07 at the heaviest quartile, 1.45 at the lightest: the anatomy
left under the blur). The `sig` arm takes the MAD proxy as specified: it is the same function
of the image at train and test, the mix build shows the net all five kernels so the proxy's
texture dependence is in its training distribution, and a sharp kernel at a given level is in
fact the harder slice; g8 would tell the net the light slices are noisier than they are. Next:
`launch_sample.sh`, `score.sh`, the §11 results here.

Status 2026-11-03 06:15: the build finished 02:39 (146 min, gates green); the three arms trained
02:41–05:38 (ε-MSE at 60k: diff-mix 0.2237, diff-mix-sig 0.2235, against diff10's 0.2265;
mse-mix body 158.3 → 54.8 HU on the range val, by dose bin from the lightest 62.5 / 54.6 / 51.1 /
50.8 / 47.6). **The first sampling pass ran with `--clamp` on and came out worse than the
low-dose floor** on the sampler's body statistic (full ≥ −950 HU, lungs in): diff-mix 124.8 vs
floor 72.8 on the 25 % val, 149.5 vs 124.0 on the 10 % val, 225.8 vs 232.4 on the first Mayo
shard; diff-mix-sig 95.7 vs 72.8 on the 25 % val. An A/B on 48 records of the 10 % val isolated
the clamp: §9's diff10 scores 40.4 with it and 42.9 without; diff-mix scores 165.4 with it and
61.1 without. The clamp as coded clips the diffusion *state* x_t after every step, not the clean
estimate x̂₀; stored air sits at exactly −1024 HU, where it forces x_t ≥ 0 at all 50 steps, and
the mixed models — trained where floor-clipped air is common at the heavy doses — react to the
rectified state and run away in lung and air, while the single-dose model is indifferent to it.
Under a body mask above −500 HU (5-px erosion) the clamped diff-mix was never worse than the
floor on the 10 % val (median 65.0 vs diff10's 44.0 vs floor 101.2 over 1,605 records): the
damage is in lung and air. So the clamp is withdrawn from every §11 run (`CLAMP=0`); the clamped
samples are kept aside, unscored, in `data/xtclamp_dumps/` and
`runs/2026-11-03-lidc-range/{sample_xtclamp,launch_sample_xtclamp.log}`; §9's clamp-only Mayo row
stands as what it is (diff10, which the clamp does not disturb). A clamp on x̂₀ — the usual
clipped-x₀ prediction — is the right form if a bound is ever wanted; not run. Early unclamped
numbers: diff-mix on its own range val 64.6 vs floor 106.1 (48 records). Sampling relaunched
unclamped 06:15, `score.sh` chained behind it; results in §11.6.

**The question.** §9's Mayo table says the generator is a posterior sampler only inside the noise
distribution it was fitted to: trained at one simulated dose it is itself up to that dose's
median noise and runs away from the top of its training range on (24 % of slices worse than the
floor at 180–250 HU, 64 % at 250–350, every slice above 350), while the MSE net never is. Mayo's
inserted noise spans 106–324 HU between its 5th and 95th percentiles with a tail to 800, and is
sharper than ours. Two changes are the standard answers, and the plan here measures both, apart
and together, against the fixed-dose generator on the same three test sets:

- **Train across the dose range** (data). The build draws each slice's dose, so the model sees
  added noise from well under Mayo's lightest slice to past its heaviest, and learns the
  conditional distribution as a function of the noise level rather than at one point.
- **Condition on the noise level** (model). A non-blind denoiser — one told σ — generalises
  across noise levels where a blind one trained on the same range has to infer σ from a 256²
  patch that may be mostly lung or air; the estimate must come from the low-dose image alone, so
  the same number is available on Mayo.

Neither touches the texture gap (§7: Shepp–Logan against B50f). A third, free change rides along
in the build: draw the apodisation filter per slice as well, so the training noise spans a range
of textures too (scikit-image's ramp, Shepp–Logan, cosine, Hamming, Hann — ramp the sharpest;
Mayo's peak at 0.425 /mm sits above even ramp's, so this covers part of the gap, not all of it).

### 11.1 The arms

| arm | trained on | sees σ̂ | steps | role |
|---|---|---|---|---|
| `mse-mix` | the range build | no | 30k | the regression baseline across the range |
| `diff-mix` | the range build | no | 60k | range-trained, blind |
| `diff-mix-sig` | the range build | yes | 60k | range-trained, noise-conditioned |
| `diff10` (have) | the 10 % build | no | 60k | the fixed-dose generator, §9's |
| `diff10 --clamp` (have, resample) | — | — | — | the sampler-side guard alone: does bounding the step remove the tail or only cap it |

Every generator is sampled on the three test sets the fixed-dose one was: the 25 % val
(`records_val.txt`), the 10 % val (same records) and Mayo (`records_mayo.txt`), at k = 0; the
spread seeds only for the arm the section ends up showing. Scoring is the §10.5 tables plus a
per-noise-bin view (the bins of §9's Mayo breakdown: < 120, 120–180, 180–250, 250–350, > 350 HU
of inserted noise), on every set.

### 11.2 Decisions (proposed; the user's call where marked)

- **Dose range** (user): log-uniform per slice in [0.03, 0.25]. Added σ = √(1/d − 1) · σ_full runs
  from 1.7× to 5.7× the slice's own noise; with σ_full at its median 39 HU that is 68–220 HU, at
  its maximum 114 HU up to 650 — Mayo's 3.43× median and its 2.7–4.6× patient spread sit inside,
  its 800 HU tail at the edge. Log-uniform so the light doses, where the sampler already works,
  do not take most of the budget. Drawn per slice from the (seed, patient, slice) stream the
  realisations use, recorded per row as `dose` beside the existing `sigma_add_*`.
- **Filter mix** (user): yes — per slice, uniform over the five filters, recorded per row as
  `filter`. The calibration (one trial back-projection per slice) already runs under whatever
  filter the slice gets, so the added-noise level stays exact per slice; only its texture varies.
  The cost is nil at build time. Against: the range model then differs from the fixed-dose ones
  in two things at once; the per-row `filter` lets the scorer split by it, which answers that.
- **One realisation**, N = 1: Noise2Noise is not retrained; the build is the 10 % build's size.
- **The noise estimate σ̂** is computed from the low-dose image alone, the same code at train and
  test: σ̂ = 1.4826 · median |low − median₃ₓ₃(low)| over body pixels (low > −500 HU after a
  5-px erosion) — the median-absolute-deviation estimator, which needs no flat region; the
  preprocessor's flat-soft-tissue estimator needs windows within −200..300 HU, which a 200 HU
  noise fails. One float per record, written beside the pair file
  as `<split>.bin.sigma.f32` by a new script, for every file (the range build, the 25 % and 10 %
  vals, Mayo); the script also prints its agreement with the known `sigma_add_hu` on the LIDC
  files, which is the estimator's gate (within ~10 % across the range). Fed to the net as one
  constant channel, log10(σ̂ / 100 HU) (−0.5 … 0.9 over 30–800 HU), after the time channels:
  `inC = 11`; the name carries `sig`.
- **The clamp** in the sampler (`--clamp`): after every DDIM step, clip x so that
  low + 100·x stays in [−1024, 3071] HU — the physically valid range, nothing tuned. *(As coded
  the x it clips is the diffusion state x_t, and that wrecks the mixed generators — withdrawn
  from every §11 run; the 06:15 status above has the A/B.)* Reported as
  its own row on the existing `diff10` (a resample, minutes) and left on for the range arms
  (their numbers reported with it; a without-clamp spot check on Mayo says whether it ever bites).
- **Steps, schedule, sampler** as §10.1: 60k steps, Adam 5e-4 cosine, 8 × 256², DDIM η = 1,
  50 steps. No sweep. The range model sees each dose a fifth as often as a fixed-dose model sees
  its one; if `diff-mix` trails `diff25` / `diff10` on their own vals by more than a few HU, that
  is the finding, not a reason to add steps.

### 11.3 Code

1. `preprocess_lidc.py`: `--dose-range LO HI` (log-uniform per slice, seeded; `--dose` stays the
   fixed form) and `--filter mix`; `dose` and `filter` per row; the manifest's spreads per dose
   quartile. Output root `data/lidcmix/`.
2. `scripts/datasets/ct_noise_level.py <pairs.bin>` → `<pairs.bin>.sigma.f32` + a line per file
   (median / spread of σ̂, and on LIDC files the ratio to `sigma_add_hu` by dose quartile).
3. `lidc-denoise-train --sigma`: loads the sidecar for the train file, one float per record,
   shuffled with the records (the shuffle moves whole records; carry σ̂ as a 1-float "label" of
   each record — `F32.shuffle`'s label argument does this), broadcast to the crop as a constant
   channel by a new C helper `F32.prependScalarChannel (x) (vals) (B C hw)` after
   `prependSinCosT`. `inC` 11, tag `sig`. `--eval-only` reads the eval file's sidecar.
4. `lidc-denoise-sample --sigma` (the pairs file's sidecar) and `--clamp` (a C helper
   `F32.clampResidual x low B hw lo hi`).
5. Scoring: `lidc_denoise_score.py --by=sigma_add_hu:120,180,250,350` (fidelity per bin, the
   bin from the row's field, falling back to the measured body std of low − full on Mayo) and the
   same `--by` in `lidc_texture.py`; `lidc_nodules.py` unchanged (LIDC only).
6. `runs/2026-11-0X-lidc-range/`: `build.sh`, `launch_train.sh` (three arms, three GPUs),
   `launch_sample.sh` (two generators × three sets, plus the `diff10 --clamp` resample),
   `score.sh`.
7. Smokes on the 4-patient build as §9's: the range build's per-row dose and filter land and
   calibrate (achieved/target ≈ 1 in every dose quartile); the σ̂ estimator against the known σ
   on the smoke files; the `sig` arm trains and samples at 64²; the clamp never triggers on an
   in-range slice.

### 11.4 Gates

- **R1, the range build**: achieved/target median within 1 % in every dose quartile and every
  filter; σ̂ within 10 % of `sigma_add_hu` (plus the scan's own) across the range.
- **R2, the fixed doses hold**: on the 25 % and 10 % vals, `diff-mix` and `diff-mix-sig` within
  ~5 HU of the fixed-dose generators' body RMSE and within 0.1 of their NPS ratios, nodule
  contrast kept ≥ 95 % in every row with n > 3.
- **R3, the range**: on the range build's own val, RMSE ≈ √2 × `mse-mix`'s and the NPS integral
  ratio in [0.9, 1.2] in every dose bin.
- **R4, Mayo**: no slice bin with more than 5 % of samples worse than the floor; the median below
  the floor in every bin; texture integral ratio in [0.8, 1.5]; then the ranking as found. The
  blind-vs-conditioned difference on Mayo's two heaviest bins is the result that decides which
  arm the section shows; the `--clamp` row says how much of §9's tail was the sampler and how
  much the model.

### 11.5 Cost and disk

| step | where | cost |
|---|---|---|
| the range build | CPU, 14 workers (one a core — §9) | ≈ 1 h; 18 GB |
| σ̂ sidecars, five files | CPU | minutes |
| `mse-mix` 30k / `diff-mix` 60k / `diff-mix-sig` 60k | GPUs 0–2, in parallel | ≈ 2.5 h wall |
| `diff10 --clamp` on Mayo (1,000) | 4 GPUs | ≈ 6 min, can run first |
| sampling: two generators × (1,605 + 1,605 + 1,000) at k = 0 | 4 GPUs, ≈ 1.1 s a slice | ≈ 40 min |
| spread seeds for the chosen arm | | ≈ 15 min |
| scoring, three roots + Mayo, per bin | CPU | ≈ 1.5 h, roots in parallel |

≈ 12 GPU-hours, ≈ 6 h wall after the code (half a day of agent work and the smokes), 18 GB of the
183 free. Everything lands in the same worktree beside §9's runs, nothing of §9's is rerun.

### 11.6 Results (run 2026-11-03)

What ran: the range build (02:39), `mse-mix` / `diff-mix` / `diff-mix-sig` on GPUs 0–2 (02:41–05:38), the
samples at k = 0 on the four sets and k = 1–3 for `diff-mix-sig` on the spread records, unclamped (the clamp
detour is in the status above), `mse-mix` evaluated on the fixed-dose vals and Mayo, and `score.sh` per root
with the noise bins. Tables in `runs/2026-11-03-lidc-range/score/`; the per-slice views below are
`scripts/…` one-offs over the same dumps (body = full ≥ −950 HU, the scorer's mask).

**Body RMSE, HU, pooled over the scoring records** (1,605 LIDC records a set, 1,000 Mayo; the low-dose
floor in the last row):

| arm | 25 % val | 10 % val | range val | Mayo |
|---|---|---|---|---|
| MSE UNet, trained at the set's dose (§9; 10 % net on Mayo) | 37.5 | 46.1 | — | 112.1 |
| `mse-mix` | 41.7 | 48.1 | 49.9 | **82.2** |
| generator, trained at the set's dose (§9: `diff25`, `diff10`; both on Mayo) | 54.1 | 73.3 | — | 227.1 / 457.8 |
| `diff-mix` | 61.6 | 82.8 | 85.6 | 347.2 |
| `diff-mix-sig` | 57.6 | 79.7 | 85.0 | 322.5 |
| Gaussian (§9) | 50.5 | 66.0 | — | 93.3 |
| low-dose floor | 72.0 | 122.8 | 140.0 | 235.0 |

On the range val the generators run at 1.7 × `mse-mix` (85.6 / 49.9), where the fixed-dose ones ran at
1.44 × and 1.59 × their MSE nets (§9) and a calibrated posterior sample runs at √2. By dose bin on the range
val, from the lightest (< 0.05) to the heaviest (≥ 0.2): `diff-mix` 108.9 / 85.6 / 74.5 / 66.7 / 63.3,
`diff-mix-sig` 107.6 / 85.9 / 74.1 / 65.5 / 61.4, `mse-mix` 59.5 / 49.2 / 46.2 / 43.1 / 42.5, floor 191 /
144 / 111 / 90 / 78.

**Mayo by inserted-noise bin, per slice** — the share of slices the arm leaves worse than the low-dose floor,
and the median slice RMSE (HU); n = 13 / 215 / 423 / 284 / 65:

| arm | < 120 | 120–180 | 180–250 | 250–350 | ≥ 350 | all 1,000 |
|---|---|---|---|---|---|---|
| floor, median | 98 | 139 | 191 | 266 | 384 | 202 |
| `diff10` (§9) | 0 % / 59 | 0 % / 81 | 14 % / 121 | 54 % / 288 | 97 % / 752 | 27 % / 132 |
| `diff25` (§9) | 0 % / 58 | 0 % / 83 | 1 % / 117 | 5 % / 175 | 22 % / 306 | 3 % / 125 |
| `diff-mix` | 0 % / 70 | 0 % / 93 | 3 % / 117 | 11 % / 153 | 32 % / 329 | 6 % / 121 |
| `diff-mix-sig` | 0 % / 67 | 0 % / 91 | 2 % / 116 | 7 % / 151 | 22 % / 230 | 4 % / 120 |
| MSE UNet, 10 % | 0 % / 43 | 0 % / 57 | 0 % / 78 | 0 % / 113 | 0 % / 186 | 0 % / 82 |
| `mse-mix` | 0 % / 43 | 0 % / 53 | 0 % / 66 | 0 % / 82 | 0 % / 111 | 0 % / 68 |
| Gaussian | 0 % / 66 | 0 % / 72 | 0 % / 84 | 0 % / 98 | 0 % / 128 | 0 % / 86 |

Pooled per bin (the squares, so the tail counts): `diff-mix` 72.5 / 93.8 / 140.4 / 325.1 / 1028.6,
`diff-mix-sig` 70.7 / 94.2 / 142.8 / 313.0 / 929.3, `diff25` 62.4 / 84.0 / 124.6 / 232.1 / 607.8; `mse-mix` 42.1 /
53.4 / 68.1 / 93.4 / 148.1, the 10 % MSE net 45.8 / 58.9 / 85.6 / 129.2 / 225.9, Gaussian 64.4 / 72.8 / 84.6 /
102.0 / 142.2; floor 95 / 139 / 195 / 275 / 416.

**Texture and findings** (flat soft-tissue ROIs; the full dose's own grain = 1.0; blobs are LoG 1–3 mm at
threshold 0.15, the full dose has 80.2 a slice on the LIDC sets):

| set | arm | NPS integral ratio | mean-frequency ratio | local-std ratio | blobs / slice | invented |
|---|---|---|---|---|---|---|
| 25 % val | `diff25` (§9) | 0.87 | 0.81 | 0.92 | 57.0 | 15.5 |
| | `diff-mix` | 1.87 | 1.42 | 1.90 | 81.5 | 33.3 |
| | `diff-mix-sig` | 1.65 | 1.24 | 1.59 | 80.5 | 31.6 |
| | MSE UNet | 0.64 | 0.63 | 0.54 | 49.6 | 7.2 |
| | `mse-mix` | 0.52 | 0.47 | 0.38 | 38.1 | 3.9 |
| 10 % val | `diff10` (§9) | 1.10 | 0.97 | 1.22 | 81.7 | 41.6 |
| | `diff-mix` | 2.89 | 1.49 | 2.58 | 131.7 | 84.2 |
| | `diff-mix-sig` | 2.92 | 1.44 | 2.29 | 151.2 | 97.3 |
| | MSE UNet | 0.56 | 0.52 | 0.42 | 36.6 | 6.6 |
| | `mse-mix` | 0.55 | 0.52 | 0.48 | 31.7 | 5.3 |
| range val | `diff-mix` | 2.79 | 1.39 | 2.40 | 141.2 | 102.1 |
| | `diff-mix-sig` | 2.85 | 1.34 | 2.28 | 157.5 | 113.9 |
| | `mse-mix` | 0.54 | 0.59 | 0.50 | 27.5 | 6.1 |
| Mayo (full dose 162.2 blobs) | `mse-mix` | 0.38 | 0.53 | 0.25 | 96.3 | 46.4 |
| | MSE UNet, 10 % | 1.82 | 1.11 | 0.80 | 135.1 | 75.8 |
| | Gaussian | 1.01 | 0.52 | 0.84 | 11.7 | 4.9 |

(The generators' Mayo integral ratios are the runaway slices' — `diff-mix` 46, `diff-mix-sig` 48, `diff25` 18,
`diff10` 55, intervals a factor of six wide — and say nothing about grain; `mse-mix`'s 0.38 is the regression
nets' usual over-smoothing, 115 of Mayo's 162 blobs a slice dropped.) On the range val by dose bin, from the
lightest, `diff-mix`'s integral ratio runs 3.98 / 2.83 / 2.36 / 2.08 / 1.87 and `diff-mix-sig`'s 4.08 / 2.98 /
2.34 / 2.07 / 1.86 — the lighter the dose, the more grain over; `mse-mix` 0.61 / 0.53 / 0.53 / 0.51 / 0.48. Spread across the four samples of
`diff-mix-sig` on the 200 spread records: nodule / lung / soft-tissue per-pixel std 28.1 / 15.7 / 22.7 HU against
`diff25`'s 21.4 / 9.8 / 15.4 on the 25 % val, 41.1 / 22.4 / 33.7 against `diff10`'s 32.2 / 16.7 / 23.2 at 10 % —
1.3–1.6 × the spread, the extra grain again; the mean of its four samples scores 61.1 against each sample's
74.5, as `diff25`'s 62.0 against 74.0 (at 10 %: 82.5 against 102.4; `diff10` 78.1 against 101.1).

**Nodule contrast kept** (median over the 395 nodules, % of the full dose's; all / the 242 under 3 mm), and
the CNR over all nodules (full dose 6.0): 25 % val — `diff25` 98.7 / 97.3, CNR 5.8; `diff-mix` 99.3 / 98.4,
5.1; `diff-mix-sig` 99.1 / 98.0, 5.4; MSE 98.7 / 97.8, 6.3. 10 % val — `diff10` 97.0 / 92.8, 5.3; `diff-mix`
97.7 / 94.0, 4.5; `diff-mix-sig` 98.0 / 95.0, 4.7; MSE 97.5 / 93.9, 6.5. Range val — `diff-mix` 93.6 / 87.1,
4.1; `diff-mix-sig` 94.0 / 89.5, 4.2; `mse-mix` 93.8 / 86.9, 6.1 (the drawn-dose low image itself 99.6 /
98.5, 3.4). Contrast is kept by every net; the mixed generators keep slightly more of it than the fixed-dose
ones because they smooth less, and pay in CNR.

**The gates.** R1 passes except the soft filters: achieved/target 1.003–1.008 by dose quartile; by filter
ramp 0.997, shepp-logan 1.000, cosine 1.012, hamming 1.021, hann 1.021 (2 % high, against the 1 % asked);
σ̂'s ratio to the known noise moves 0.47–0.55 across the quartiles (12 %, against the 10 % asked) and the
proxy reads the kernel (status above). R2 fails: `diff-mix-sig` is 3.5 HU behind `diff25` at 25 % but 6.4
behind `diff10` at 10 %, `diff-mix` 7.5 and 9.5 behind; the NPS ratios are 0.8–1.8 above, not within 0.1;
nodule contrast passes (≥ 97.7 % in every row with n > 3). R3 fails: 1.7 × `mse-mix`, not √2; NPS
integral 2.8 overall and 1.9–4.0 by dose bin, never in [0.9, 1.2]. R4 fails on its first two clauses — 22–32 % of the ≥ 350 HU slices
worse than the floor, 7–11 % of the 250–350 — and passes the third: the median is below the floor in every
bin for both mixed arms, which `diff10` could not do. The blind-vs-conditioned difference on the two
heaviest bins is 4 and 10 points of tail and 2 and 99 HU of median, in the conditioned arm's favour. `mse-mix`
passes R4's first two clauses outright (no slice worse than the floor, median 68 against the floor's 202) and
fails the texture band from below (0.38), as every regression net does.

**Reading.**

1. *The dose range costs the generator its calibration.* Trained across 0.03–0.25 the generator puts back
   1.9–2.9 × the scan's grain where the fixed-dose one put back 0.87–1.10 ×, invents twice the blobs, and
   samples at 1.7 × the MSE net's error instead of √2 — on the fixed-dose vals and on its own. The σ̂
   channel recovers a third of the RMSE gap at 25 % (61.6 → 57.6 against 54.1) and trims the grain there
   (1.87 → 1.65), a quarter of the gap at 10 % with no change in grain, and nothing on the range val, where
   the net can read the level from the image on its own training distribution; the proxy, which reads the
   kernel as much as the level, is not a dose label. Sixty thousand steps on a distribution five times wider
   is the plain explanation for the extra variance (an ε-predictor that is short of the posterior leaves
   noise in its samples), and it is not tested here.
2. *The range training rescues the regression net on Mayo and not the generator — and says which of §9's
   failures was the dose.* `mse-mix`, the MSE UNet trained across 0.03–0.25, is the best arm on Mayo by a
   margin: 82.2 HU pooled against the Gaussian's 93.3 and the 10 % net's 112.1, no slice worse than the floor,
   median 68 against the 10 % net's 82, and on the two heaviest bins 93 and 148 pooled where the 10 % net
   had 129 and 226 — its Mayo gap *was* the dose range, and training through the range closed it at a cost
   of 4 HU at 25 % and 2 HU at 10 % on the LIDC vals (41.7 / 48.1 against 37.5 / 46.1). The generator
   trained on the same data lowers its typical heavy slice two- to three-fold (median 288 → 153 HU at
   250–350, 752 → 329 / 230 at ≥ 350) and keeps the runaway tail — a fifth to a third of the heaviest
   slices still worse than the floor, 4–6 % overall — though slices with 250–400 HU of inserted noise were a
   quarter of what it saw. Its pooled RMSE, which squares that tail, stays above `diff25`'s and four times
   `mse-mix`'s. What is foreign to the generator on those slices is not the noise level, which it has seen;
   it is the scanner — the B50f kernel's grain (NPS peak 0.425 /mm against our 0.330, §9), the vendor's
   insertion, the anatomy at that noise — and §7's option (b), the kernel, is the open question the range
   run leaves for the generator alone.
3. *The tail lives on LIDC too.* The 25 % val's ≥ 120 HU bins are twelve patients, three of them (0661,
   0654, 0915, GE scans at low mA, I0 a sixth of the median) with every slice there, and **every generator,
   `diff25` included, is worse than the floor on a third of those 118 slices**, the MSE net on none. The
   generator fails on slices whose noise is unlike the bulk of what it was fitted to — by character, not
   level — and the regression net does not, because regression to the mean cannot run away.
4. *What the section shows.* The fixed-dose generator of §9 remains the demo's object: the mixed arms are
   behind it on every LIDC number and ahead of it on Mayo only by the median. §11 earns two rows in the
   section's Mayo table and a paragraph: `mse-mix` — "the regression net trained across the dose range is
   the best thing on the held-out scanner, 82 HU, no slice worse than the floor" — and `diff-mix-sig` —
   "trained across the range and told the noise level, the generator's typical Mayo slice improves 2.5 ×
   and one slice in four at the heaviest noise still runs away". The reading that the pair supports:
   regression to the mean generalises across dose once it has seen the range, because it cannot run
   away; a sampler that puts grain back does not, because the grain it puts back is its training
   scanner's. The recommendation is to leave the generator there and spend the next run on the kernel
   (§7 b), not on more conditioning. The user's call.

## 12. The kernel (§7 option b; run 2026-11-03)

§11.6 left the generator one open question. Training across the dose range did not take its Mayo
tail away, and the slices it fails on are foreign to it by the *character* of their noise, not its
level. Of what differs between Mayo and the training set at once (§10.4) — the scanner, its
reconstruction kernel, the vendor's insertion, the anatomy at that noise — the kernel is the one
axis the simulation can move: Mayo's inserted noise is reconstructed through Siemens's sharp B50f
and peaks at 0.425 /mm, ours through a Shepp-Logan apodisation at 0.330 (§9, the phase-4a gate
that failed at 1.29). §11's build already drew the apodisation per slice from scikit-image's five,
and its sharpest, the ramp, still sits below Mayo's peak, so a stock window does not get there. This
section fits one.

### 12.1 The kernel

`--filter b50f` in `scripts/datasets/preprocess_lidc.py`: the Shepp-Logan window times a gain in
cycles/mm, the gain being the square root of the ratio of Mayo's inserted-noise NPS to the 25 %
build's Shepp-Logan NPS, both radially averaged and normalised to unit integral
(`runs/2026-11-02-lidc-mayo/nps.csv`), 3-bin smoothed and divided by its 0.08–0.16 /mm plateau so
the gain is 1 at low frequency; it rises to 1.6 × at 0.56 /mm. Two choices outside the data, named:
below 0.08 /mm the measurement is the 64-px ROI's resolution and the detrend, and the gain is held
flat; past 0.64 /mm nothing is measured (Mayo's 0.742 mm grid has its Nyquist at 0.674) and the
gain is taken down as a sharp kernel's MTF goes, to 0.7 × and 0.3 × its 0.64 value at 0.80 and
1.00 /mm, so that the finest LIDC grids (0.50 mm, Nyquist 1.0 /mm) are not handed grain no scanner
makes. The knots are `B50F_GAIN`. It is applied by pre-filtering the sinogram with scikit-image's
own circle-to-square pad, power-of-two pad and ramp, the window at |f|, and back-projecting with the
filter off (`Projector.fbp_window`), which reproduces `iradon`'s ramp and Shepp-Logan to 1e-14, so
the only thing that changes between the builds is the window. The level does not move: the
per-slice I0 calibration runs under the kernel, and I0_add comes out twice the Shepp-Logan build's
(the kernel passes more noise power per photon, so the same σ_add takes twice the photons).

The gate, before the build (`scripts/probes/lidc_kernel_probe.py`: the full-dose slices of
`data/lidc10/val.bin`, 300 asked, 161 with a flat 64² ROI and a calibration, 1,417 ROIs, the
simulation and the calibration exactly as the build runs them, the NPS as `lidc_noise_nps_check.py`
measures it, against Mayo's curve from that script's run):

| filter on the 10 % val | NPS peak /mm | mean frequency /mm | achieved / target σ_add | Mayo / ours, peak | gate (10 %) |
|---|---|---|---|---|---|
| Shepp-Logan (the control; the builds so far) | 0.345 | 0.352 | 0.995 | 1.23 | FAIL |
| `b50f` | 0.425 | 0.397 | 0.992 | 1.00 | PASS |
| Mayo, vendor-inserted (B50f) | 0.425 | 0.398 | — | — | — |

Under `b50f` the per-frequency residual √(Mayo / ours) is within 2 % from 0.12 /mm up (5 % at
0.08). The run directories: `runs/2026-11-03-lidc-kernel/probe_{b50f,shepp-logan}/`.

What it does not do, stated up front: it moves the *inserted* noise's spectrum and nothing else. The
target images stay LIDC's STANDARD-kernel scans, whose own grain peaks at 0.10 /mm against Mayo's
0.43 (§9); the fan-beam geometry, the bowtie, the anatomy at that noise all stay as they were. So the
outcome space is: the Mayo tail shrinks, which says the kernel of the noise was the generator's
problem; or it does not, which says the generator needs the target scanner's own texture in training
and the held-out result stands as §11.6 read it. Either way the §10.6 table gets its noise-peak row
and the reading is clean.

### 12.2 The run

- **Build** `data/lidc10b50/`: the 300 patients, the seed-0 split, the slice selection and the
  per-slice calibration of `data/lidc10`, dose 0.10, one realisation, `--filter b50f`
  (`runs/2026-11-03-lidc-kernel/build.sh`, 24 workers at nice 19, ≈ 1.3 h). Gates, in order: the
  split equals `data/lidc10`'s; the (patient, slice) order of every val and train row equals
  `data/lidc10`'s, so `records_val.txt` / `records_spread.txt` apply unchanged and every comparison
  with the Shepp-Logan build is slice for slice; the loader tests at N = 1; the calibration
  (achieved / target within 2 %); and phase 4a on the *built* val against Mayo, the gate that failed
  under Shepp-Logan.
- **Arms**, the §10.1 recipe, steps and seed of §9's `diff10` / `mse10`, so the kernel is the only
  thing that moves: `diff10b50` (the generator, 60k, GPU 0) and `mse10b50` (the MSE UNet, 30k,
  GPU 1). Artifacts carry the root `lidc10b50`.
- **Samples and eval passes** (`launch_sample.sh`, DDIM η = 1, 50 steps, no clamp): on Mayo
  (`records_mayo.txt`); on the b50f val and, crossed, the b50f generator on the Shepp-Logan val and
  §9's generator on the b50f val — a 2 × 2 kernel matrix per net, train kernel against test kernel,
  on the same slices; spread seeds k = 1–3 on the b50f val; the MSE nets' three eval passes
  (`mse10b50` on Mayo and on the Shepp-Logan val, `mse10` on the b50f val); the Gaussian re-tuned
  on the b50f train split for that root's filter row.
- **Tables** (`score.sh`): fidelity per noise bin, the per-slice view (`lidc_per_slice.py`: share
  of slices worse than the floor and the median, the view that found the tail; it reproduces §11.6's
  Mayo rows), nodules, texture, spread; new dumps under `data/{mayo,lidc10}/out/{diff_b50,mse_b50}/`
  and `data/lidc10b50/out/{diff,mse,diff_sl,mse_sl,gauss}/`.
- **What decides it.** K1, the build gate above. K2, on Mayo: the generator's share of slices with
  ≥ 350 HU of inserted noise that are worse than the floor (97 % for `diff10`, 22 % for `diff25`
  and `diff-mix-sig`) and the pooled body RMSE (227 for `diff25`, 322 for `diff-mix-sig`, 82 for
  `mse-mix`): a kernel effect is a tail below `diff-mix-sig`'s with `diff10`'s dose coverage. K3,
  the MSE net on Mayo against `mse10`'s 112: whether the kernel also closes part of the regression
  net's gap, which §11 put down to the dose. K4, the cross-kernel rows on LIDC: what each net loses
  when the test kernel is not its training kernel, on the same scanner and slices — the size of the
  kernel sensitivity by itself. Cost: ≈ 1.3 h CPU, ≈ 4 h on two GPUs, ≈ 1 h of sampling and
  scoring; disk +18 GB build, ≈ 10 GB dumps (61 GB free at the start).

### 12.3 Status

- **2026-11-03 13:05 UTC** — `b50f` written and verified (the window path reproduces scikit-image's
  filters to 1e-14); the probe passes (table above) where the Shepp-Logan control fails at 1.23;
  the build launched at 24 workers; the chain (build gates → `launch_train.sh` → `launch_sample.sh`
  → `score.sh`) armed. Results in §12.4 when the tables land.
- **14:30 UTC** — the section drafted "from the end" (the user's ask): `planning/lidc_ct_synthesis_section.tex`,
  also placed in `blueprint/src/content.tex` after the DDPM section (uncommitted) so `scripts/book/blueprint_preview.py`
  renders the diff (served on :8765; 261 → 265 pages, no new errors or overfull boxes). One table (LIDC 10 % ×
  Mayo blocks, the two kernel rows dashed, the TODO marker after it) and one figure, drawn for real from the existing
  dumps by the new `scripts/demos/lidc_synthesis_figure.py`: (a) the Mayo slice at the median inserted noise with the
  range + σ̂ generator's draw (provisional: the kernel run decides which generator the section keeps), (b) the three
  rule-chosen nodules across scanned / tenth dose / BM3D / MSE / two samples (the second sample for the two records
  outside the spread set drawn ad hoc, 2 slices on GPU 2), (c) the NPS curves, log scale, LIDC and Mayo.
- **16:20 UTC** — build done 16:14 (3.1 h, not the 1.3 estimated: 24 workers on 16 physical cores and the window
  path's extra FFT, ≈ 16 s a slice). Every gate passed: split and (patient, slice) order identical to `data/lidc10`;
  calibration achieved/target median 0.995, `filters == ['b50f']`, I0_add median 1.98e4 (about twice the Shepp-Logan
  build's, as the probe predicted); loader tests at N = 1; **phase 4a on the built val passes** — added-noise NPS peak
  0.440 /mm against Mayo's 0.425 (ratio 0.97), mean frequency 0.396 against 0.398 (1.00), σ 97.8 HU median ROI. The
  full-dose grain is untouched (peak 0.100 /mm, the scanner's own; the kernel moves only what we add). Trainings
  launched 16:17 on free cards: `diff10b50` 171 ms/step (60k → ≈ 19:10 UTC), `mse10b50` 148 ms/step (30k → ≈ 17:30);
  the Gaussian re-tune on the b50f train split is done (chosen σ 1.0, `data/lidc10b50/out/gauss/`). Sampling and
  scoring follow the generator: tables ≈ 20:15 UTC.
- **16:25 UTC** — user: the section is the demo of **12.3 Image generation**; 12.2 becomes *Diffusion — demo: DDPM on
  MNIST* (swapped with the catalogue-only *Image generation* section), and that section's VAE / DCGAN / Pix2Pix /
  CycleGAN entries now close the CT section (title `Image generation --- demo: low-dose CT synthesis on LIDC, held out
  on Mayo`, label `sec:bestiary_ct_synthesis` unchanged; a bridge paragraph introduces the catalogue). Also:
  `scripts/demos/lidc_synthesis_table.py` prints `tab:ct_synthesis` rows from the score files (texture / nodules json,
  fidelity txt, per-slice csv) and reproduces every drafted cell; three cells were over-rounded from the text reports
  and are fixed (Gaussian blobs 3.9, BM3D 8.0, range + σ̂ 79.6 / 2.91). The kernel rows come from the same script.
- **16:32 UTC** — user: appendix D's *Demo datasets* order must match the book's. The LIDC/Mayo item and the two
  `tab:datasets_demo` rows moved from after BraTS to after tiny Shakespeare (12.1), before Müller–Brown (12.4); the rest
  already followed chapters 11–12 in order. (The list's opening sentence still names five task families and not
  generation or structure — re-written on the user's word: the families of chapters 11–12 in order, MNIST
  pointed at the core table.)
- **16:41 UTC** — user: 12.6 renamed *Representations --- demo: neural quantum states on the Ising chain* (user's second pick over
  *Fields and embeddings*; was
  *Beyond vision*; the section's own first sentence — a wavefunction, a radiance field, a shared embedding space — is
  the thread; label `sec:bestiary_beyond` kept; `Bestiary/README.md` + `demos/README.md` headings follow), and the
  Demo-datasets sentence trimmed to "Each has a dataset of its own:".
  The `demos/README.md` section *Image generation — low-dose CT synthesis on LIDC, held out on Mayo* (between
  *Diffusion* and *Physics*, the README mirrors the book's order) was written with the results, 11-03 evening.
- **20:05 UTC** — chain done: trainings 16:17–18:50 (`diff10b50` 60k at 152 ms/step, `mse10b50` 30k at 148), sampling
  18:52–20:01 on four cards (every dump complete: 1,000 Mayo, 1,605 × 3 on LIDC, 3 × 200 spread), scoring 20:01–20:05.
  Results below (§12.4). By 20:29: the section's two rows filled and grouped under their nets, the TODO paragraph gone,
  the figure redrawn (panel (a) both generators), `demos/README.md` gained its *Image generation* section between
  *Diffusion* and *Physics* (+ the layout and intro lists), the preview rebuilt: section p. 210, table 211, figure 212 of
  the proposed PDF, 265 pages, 0 errors, overfull back to the baseline 201 (the `2{,}026` cell and the Mayo URL in
  `tab:datasets_demo` were the two new boxes). Everything staged, nothing committed.

### 12.4 Results (run 2026-11-03; `runs/2026-11-03-lidc-kernel/score/`, rows by `table_rows.sh`)

**K1, the build gate: PASS.** On the built 10 % val the added noise's radial NPS peaks at 0.440 /mm against Mayo's
0.425 (ratio 0.97), mean frequency 0.396 against 0.398 (1.00); the level is unchanged (floor 122.1 HU on the 1,605
scored slices against the Shepp-Logan build's 122.8; calibration achieved/target median 0.995). The kernel moved the
spectrum and nothing else, on the same slices in the same order.

**Mayo, 1,000 slices** — pooled body RMSE (per-slice RMS; the book's table pools over pixels, 2,026 / 84.4 / 457.8 /
112.1 / 82.2 / 322.5 there), median slice, share of slices worse than the floor; per bin of inserted noise, % worse /
median (`lidc_per_slice.py`):

| arm | pooled | median | worse | <120 | 120–180 | 180–250 | 250–350 | ≥350 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| (bin) | | | | n 13, floor 98 | n 215, floor 139 | n 423, floor 191 | n 284, floor 266 | n 65, floor 384 |
| `diff10b50` (**kernel**) | 1982.0 | **100** | 5 % | 0 % / 55 | 0 % / 70 | 2 % / 94 | 10 % / 140 | 25 % / 277 |
| `diff10` (§9) | 455.4 | 132 | 27 % | 0 % / 59 | 0 % / 81 | 14 % / 121 | 54 % / 288 | 97 % / 752 |
| `diff25` (§9) | 225.7 | 125 | 3 % | 0 % / 58 | 0 % / 83 | 1 % / 117 | 5 % / 175 | 22 % / 306 |
| `diff-mix-sig` (§11) | 325.9 | 120 | 4 % | 0 % / 67 | 0 % / 91 | 2 % / 116 | 7 % / 150 | 22 % / 230 |
| `mse10b50` (**kernel**) | 83.4 | 69 | 0 % | 0 % / 43 | 0 % / 52 | 0 % / 66 | 0 % / 85 | 0 % / 126 |
| `mse10` (§9) | 110.7 | 82 | 0 % | 0 % / 43 | 0 % / 56 | 0 % / 78 | 0 % / 113 | 0 % / 186 |
| `mse-mix` (§11) | **81.6** | **68** | 0 % | 0 % / 43 | 0 % / 53 | 0 % / 66 | 0 % / 82 | 0 % / 111 |
| Gaussian | 92.3 | 86 | 0 % | 0 % / 66 | 0 % / 72 | 0 % / 84 | 0 % / 98 | 0 % / 128 |

Mayo texture (`lidc_texture.py`, flat soft-tissue ROIs; the scan's own grain = 1.0): `diff10b50` NPS integral 10.7×
[3.1, 21.6] at 1.12× the frequency, local std **0.92×** (`diff10` 55× / 1.07 / 1.39; `diff-mix-sig` 48× / 0.98 / 2.08);
invented blobs 187 a slice against the scan's 162 own (`diff10` 329, `diff-mix-sig` 324). `mse10b50` 0.58× / 0.68 / 0.35
(`mse10` **1.82× / 1.11 / 0.80**; `mse-mix` 0.38 / 0.53 / 0.25); invented 34 (`mse10` 76, `mse-mix` 46).

**The kernel generator's tail.** 52 slices worse than the floor (5 %), every one of them inside `diff10`'s 274 and 40
of them among `diff-mix-sig`'s 43 — the same heaviest slices fail for every generator. On them this one runs further:
18 slices above 1,000 HU, 10 above 5,000, the worst 51,454 (record 3831, 748 HU inserted, where `diff10` reaches 3,937
and `diff-mix-sig` 3,043). Without those 18 the pooled number is 151 HU; with them it is the four-digit cell in the
book's table. The median slice in every bin but the top is the best of any generator (55 / 70 / 94 / 140 against
`diff-mix-sig`'s 67 / 91 / 116 / 150), and at ≥ 350 HU the share worse (25 %) matches the range generators' 22 %
while the median (277) is worse than `diff-mix-sig`'s 230.

**LIDC, the 2 × 2 kernel matrix** (1,605 slices, pooled body RMSE / grain / mean-f / nodules kept all, < 3 mm / invented
blobs; floors 122.8 Shepp-Logan val, 122.1 b50f val):

| net \ val | Shepp-Logan val | b50f val |
|---|---|---|
| diffusion, Shepp-Logan (§9) | **73.4** / 1.10 / 0.97 / 97.0, 92.8 / 42 (home) | 90.8 / 1.17 / 1.24 / 95.5, 92.3 / 45; **29 % of ≥ 180 HU slices worse than the floor**, 3 % overall |
| diffusion, b50f | 67.6 / 1.28 / 1.01 / 98.1, 94.9 / 37; 0 % worse | **70.4** / 1.10 / 1.06 / 97.5, 93.9 / 35 (home) |
| MSE, Shepp-Logan (§9) | **46.1** / 0.56 / 0.52 / 97.5, 93.9 / 6.6 (home) | 47.2 / 0.62 / 0.74 / 96.2, 93.1 / 5.5 |
| MSE, b50f | 48.0 / 0.71 / 0.54 / 98.5, 96.0 / 9.3 | **45.2** / 0.58 / 0.51 / 97.7, 95.3 / 5.6 (home) |
| Gaussian, re-tuned (σ 1.0 px) | 66.0 | 62.5 |

Spread on the b50f val (200 records, k = 0–3): RMSE 98.2–98.6, grain 1.09–1.11, mean-f 1.04–1.06 — four seeds, one
distribution, as under Shepp-Logan.

**Verdicts.**
- **K2 (the generator's Mayo tail): half.** The kernel takes the share of slices worse than the input from 27 % to
  5 % and the median heavy slice from 132 to 100 HU — the best generator on Mayo by both — with `diff10`'s dose
  coverage; what it leaves is the fifty heaviest slices (≥ 300 HU inserted, two to four times anything in a 10 %
  LIDC build), on which it runs away harder than any other arm. The tail is the *amount* of noise; the kernel fixed
  the *character*.
- **K3 (the MSE net on Mayo): yes.** 84.4 HU against 112.1, within two of the range net's 82.2, bin for bin (85 vs 82
  at 250–350, 126 vs 111 at ≥ 350). The mechanism is in the texture row: the Shepp-Logan net leaves 1.8× the scan's
  grain variance in its Mayo output at 1.1× the scan's frequency — Mayo's sharp noise passed through as if it were
  anatomy — and the b50f net removes it (0.58×). §11's "the gap was the dose" becomes "the gap was the noise it had
  never seen, in character or in amount; either fix suffices for regression to the mean".
- **K4 (kernel sensitivity on LIDC): 2 HU for the regression nets, one-directional for the generator.** The MSE nets
  lose 1.9–2.0 HU (4 %) in either direction and keep the nodules. The b50f generator transfers *down* a kernel (67.6
  on the soft val, below the home generator's 73.4, with 28 % more grain than the scan has — it puts sharper grain
  back than the soft kernel made); the Shepp-Logan generator does not transfer *up*: 90.8 HU on the b50f val, worse
  than its input on 29 % of the slices above 180 HU — the Mayo failure reproduced on its own scanner, slices and
  level, with the kernel the only change. That is the cleanest statement the run makes.

**What the section does with it (done 11-03 evening).** The two kernel rows filled, LIDC block = each net at home on
the b50f build of the same slices (the caption says so), Mayo block = the 1,000 slices; the TODO paragraph gone; the
second-block paragraph closes on K3/K2/K4; panel (a) now shows the median-noise Mayo slice under both generators
(131 HU for §9's, 108 HU for the kernel's, against the vendor's 190); panel (c) overlays the b50f build's low-dose
spectrum on the LIDC panel and shows the kernel arms on Mayo. §9's generator stays the left block's generator.

**Not done, proposed.** One more build and generator would test the obvious combination — the dose range of §11
under the b50f kernel, with the σ̂ channel (≈ 3 h CPU + 3 h GPU): the kernel fixes the character, the range the
amount, and the two tails overlap on the same 40 slices. Not launched; the user's call. The runaway itself (a DDIM
chain at η = 1 on a residual far outside the training range) is a sampler property the §11 `--clamp` attempt handled
wrongly; a guard on the predicted residual rather than on x_t is untested.

## 13. The dose range under Mayo's kernel (run 2026-11-03 evening; the user's "yes please")

**Why.** §12.4: the kernel fixes the *character* of the noise (the generator's median on Mayo 132 → 100 HU, slices worse
than the input 27 % → 5 %) and §11's dose range its *amount* (4 % worse, median 120); each leaves the same ~40 heaviest
slices, and the regression net reaches 84 / 82 HU by either alone. One build and two nets test the combination.

### 13.1 The run (`runs/2026-11-03-lidc-rangekernel/`)

- **Build** `data/lidcmixb50`: §11's range (dose log-uniform in [0.03, 0.25] per slice, N = 1, the same 300 patients,
  split and slice selection) reconstructed through `--filter b50f` — one filter, not §11's five, since the five were all
  softer than Mayo and the kernel is now the known one. Gates as §12 (split + record order = `data/lidc10`'s, filters,
  calibration within 2 %, loader tests, phase 4a on the built val against Mayo), then the σ̂ sidecars
  (`ct_noise_level.py`) for the new files and for `data/lidc10b50/val.bin`, and the val's record list by the §10.5 rule.
- **Arms**: `msemixb50` (MSE UNet, 30k, GPU 0; its own val scored by dose bin, then the 4.3 GB dump removed) and
  `diffmixsigb50` (the generator told σ̂, 60k, GPU 1). No blind range generator: σ̂ was a small gain in §11 and one
  generator is the question.
- **Samples / eval passes**: the generator on Mayo (`records_mayo.txt`), on the b50f 10 % val (`records_val.txt`, the LIDC
  block the kernel rows use) and on its own val (gate R3); the MSE net's forward pass on Mayo and the b50f 10 % val.
- **Tables** (`score.sh`): the §12 set for the new arms; Mayo's per-slice csv merged with §12's
  (`score/perslice_mayo_all.csv`) so every arm is in one table; `table_rows.sh`-style rows from the score files.
- **Decides (K5).** The generator's Mayo tail with both fixes: share of slices worse than the input (5 % / 4 % with one
  fix), median (100 / 120), the ≥ 350 HU bin (25 % / 22 % worse; medians 277 / 230), pooled; whether the 40-slice tail
  yields to the combination or stays — if it stays, the runaway is the sampler's (DDIM at η = 1 far outside the training
  residuals), not the data's. And the regression net with both fixes against 84 / 82.
- **Disk.** 17 GB free at the start: removed the regenerable §12 MSE eval dumps (4 × 4.3 GB; the checkpoints stay, the
  scores are in `score/`) and the withdrawn `--clamp` dumps (`data/xtclamp_dumps`, 6.2 GB; §11.6 has their A/B) → 40 GB;
  the build takes ≈ 18, the new dumps ≈ 13 (two MSE eval passes are 8.6 of it) → ≈ 9 GB at the end.
- **Cost.** Build ≈ 3 h at 24 workers; trainings 2.7 h / 1.3 h in parallel; sampling ≈ 35 min on four cards; scoring
  ≈ 10 min → tables ≈ 03:30 UTC 11-04. Chain = scratchpad `chain_rangekernel.sh` (background), as §12's.

### 13.2 Status

- **21:00 UTC** — scripts written and staged, labels added (`diffmixsig_b50`, `msemix_b50`); build launching.
- **00:06 UTC 11-04** — build + gates PASS: split and (patient, slice) order = `data/lidc10`'s; manifest `b50f`,
  achieved/target within 2 %; loader tests; added-noise NPS peak Mayo/LIDC 0.96, mean frequency 1.00 (PASS); σ̂
  sidecars for train / val / the b50f 10 % val (medians 101 / 85 / 80 HU); 1,605-record val list.
- **01:26** — `msemixb50` trained (30k); own val body 161.4 → 52.7 HU (by dose: 59.6 / 52.0 / 51.0 / 47.9 / 46.5
  from < 5 % to ≥ 20 %); its 4.3 GB dump removed after the score.
- **02:59** — `diffmixsigb50` trained (60k, final ε-MSE 0.228, 166 ms/step). **03:38** sampling + the two MSE eval
  passes done (sampling was 35 min, as costed; the build and the trainings, not the sampling, set the clock).
  **03:44** scored. The two MSE eval dumps removed after the score (18 GB free); the three sample dumps kept.
- **04:30** — §13.3 written; the section, figure and README updated (below). Nothing committed.

### 13.3 Results (`runs/2026-11-03-lidc-rangekernel/score/`; rows by `table_rows.sh`)

**K5, the generator's Mayo tail with both fixes: NO — it is the worst generator in the table.** `diffmixsigb50` on
Mayo: pooled body RMSE 1.2 × 10²⁹ HU, median slice **168**, **49 %** of slices worse than the input (the kernel
alone 5 % / 100; the range + σ̂ alone 4 % / 120). Per bin of inserted noise, % worse / median, the same 1,000 slices
(`perslice_mayo_all.csv` carries every Mayo arm since §9 side by side):

| arm | pooled | median | worse | <120 (n 13) | 120–180 (215) | 180–250 (423) | 250–350 (284) | ≥350 (65) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `diffmixsigb50` (**both**) | 10²⁹ | 168 | **49 %** | 0 % / 64 | 1 % / 77 | 38 % / 96 | 92 % / 2·10⁹ | 100 % / 3·10¹⁷ |
| `diff10b50` (§12, kernel) | 2,026 | 100 | 5 % | 0 % / 55 | 0 % / 70 | 2 % / 94 | 10 % / 140 | 25 % / 277 |
| `diff-mix-sig` (§11, range + σ̂) | 322.5 | 120 | 4 % | 0 % / 67 | 0 % / 91 | 2 % / 116 | 7 % / 150 | 22 % / 230 |
| `diff10` (§9) | 457.8 | 132 | 27 % | 0 % / 59 | 0 % / 81 | 14 % / 121 | 54 % / 288 | 97 % / 752 |
| `msemixb50` (**both**) | 83.3 | 68 | 0 % | 0 % / 42 | 0 % / 52 | 0 % / 65 | 0 % / 83 | 0 % / 119 |
| `mse10b50` (§12) | 84.4 | 69 | 0 % | 0 % / 43 | 0 % / 52 | 0 % / 66 | 0 % / 85 | 0 % / 126 |
| `mse-mix` (§11) | 82.2 | 68 | 0 % | 0 % / 43 | 0 % / 53 | 0 % / 66 | 0 % / 82 | 0 % / 111 |

The failure is a *divergence*, not the drift of §9–§12: 449 of the 1,000 slices end above 1,000 HU, 338 above 10⁶,
the worst at 3.6 × 10³⁰ (float32 has room to 3 × 10³⁸). Where it fails, the slice does not go grainy — on the
figure's slice (record 2242, 222 HU inserted, 190 floor) a patch of the left lower lung saturates and the rest is
clean, 2,286 HU from the scan where `diff10b50` is 108 and `diff10` 131. The 40 worst slices are 28 of
`diff10b50`'s 40 and 31 of `diff10`'s: the same heavy slices as every generator, plus many that only this one loses.

**It fails at home too.** On its own val (`data/lidcmixb50`, 1,605 slices, dose bins): **11.8 % worse than the
floor**, median 55; by dose < 5 % 27 % / 64, 5–10 % 10 % / 58, 10–15 % 5 % / 50, 15–20 % 5 % / 44, ≥ 20 % 3 % / 40;
166 slices above 1,000 HU, max 3 × 10²⁴ — at doses it trained at. On the b50f 10 % val (the kernel rows' LIDC
block): 6.2 % worse, median 52 (`diff10b50` at home: 70.4 pooled); by inserted noise 0 % / 46, 1 % / 70, 25 % / 91,
97 % / 3·10¹⁰, 100 % / 3·10²⁰ (n 1071 / 354 / 108 / 62 / 10). §11's `diff-mix-sig` under Shepp-Logan on its own val
lost 22 % of the ≥ 350 HU slices to a few thousand HU; this one loses them to 10²⁰.

**Where it fails, by the level it was told.** σ̂ (MAD of a 3 × 3 high-pass over the body, the conditioning channel)
on Mayo: median 172 HU, p95 242 (the training set's median 101, p75 145, p95 225 — Mayo's median slice sits at the
training set's 85th percentile of the level, as in §11; σ̂ / √(σ_add² + σ_full²) is 0.73–0.78 on Mayo in every bin
but the top, as in the simulation, so the estimator reads Mayo as it reads LIDC). Share of Mayo slices worse than
the input by σ̂, `diffmixsigb50` against `diff10b50` (no level channel):

| σ̂ bin (HU) | <100 | 100–130 | 130–160 | 160–200 | 200–250 | ≥250 |
|---|---:|---:|---:|---:|---:|---:|
| n | 16 | 105 | 254 | 371 | 216 | 38 |
| both (told the level) | 0 % | 3 % | 22 % | **53 %** | **92 %** | **100 %** |
| kernel only | 0 % | 1 % | 7 % | 6 % | 4 % | 8 % |

At the same measured noise the kernel-only generator holds at one in twenty; the one told the level, trained on
that level, diverges on nine in ten above 200 HU. The top σ_add bins where σ̂ under-reads (ratio 0.45 at ≥ 350 HU
on Mayo, 0.16 at ≥ 350 on the b50f val: the heaviest slices have the least body for the estimator) fail 100 %
regardless — being told a lower level does not save them either. No blind range-under-b50f generator was trained
(§13.1), so range-under-the-sharp-kernel and the σ̂ channel are not separated by data; what is established is that
the combination is far less stable than either part, on Mayo and on its own validation set.

**The regression net with both fixes: 83.3 HU pooled, median 68, 0 % worse — the range net's 82.2 and the kernel
net's 84.4 again.** Bin for bin it is within 3 HU of `mse-mix` and never above `mse10b50`. Mayo texture (`texture_mayo`,
scan's own grain = 1.0): NPS integral **0.39×** at 0.52 the frequency, local std 0.31, invented blobs 40 a slice
(`mse-mix` 0.38 / 0.53 / 0.25 / 46; `mse10b50` 0.58 / 0.68 / 0.35 / 34; `mse10` 1.82 / 1.11 / 0.80 / 76). The two
fixes are one fix for regression to the mean: either shows the net noise it would otherwise pass through as
anatomy, and the second buys nothing more. At home on the b50f 10 % val: 45.4 / 0.59 / 0.53 / nodules 97.5, 95.8 /
invented 5.5 — the range costs 0.2 HU under the sharp kernel where it cost 2.0 under Shepp-Logan (§11), and on its
own range val 52.7 pooled (59.6 at < 5 % dose, 46.5 at ≥ 20 %).

**Nodules (b50f 10 % val, 395 / 242 under 3 mm):** `diffmixsigb50` 97.9 [96.7, 98.7] / 95.5; `msemixb50` 97.5
[96.9, 98.3] / 95.8; CNR all 4.8 / 6.5 (low dose 3.5, scan 6.0). The generator keeps the nodules on the slices it
does not lose — the median over nodules is blind to the 6 % it does.

**Verdict K5.** The tail does not yield to the combination; it widens from 50 slices to 490. The runaway is the
sampler's (fifty DDIM steps at η = 1 compounding the ε-net's error on the slices it is least sure of), not the
data's: a generator trained across the doses, under the test kernel, told the level, diverges on 12 % of its own
validation set. Of the four generators on Mayo the kernel-only one (§12) stays the best: median 100, 5 % worse.
The regression net is settled at 82–84 HU by any one of the three training sets that contain Mayo's noise in
amount or in character.

**What the section does with it (done 11-04 04:30).** Two rows, `\quad the same, both at once`, under each pair's
kernel row: MSE 45.4 / 0.59 / 0.53 / 97.5, 95.8 / 5.5 | 83.3 / **68** / 0 %; diffusion $10^{21}$ / — / — /
97.9, 95.5 / — | $10^{29}$ / 168 / 49 % (the table script prints the order of magnitude for a pooled RMSE ≥ 10⁵ and
dashes the texture cells; the caption says so and gives the medians). The data paragraph names the fourth build;
the right-block paragraph gains "Both fixes at once … settle it" (the regression net's 28 HU bought once; the
generator's divergence, the figure's patch, at home too, the kernel-only generator at the same σ̂; "the runaway is
the sampler's, not the data's"); the reading paragraph closes with "what is left to try is the sampler, not the
data". Panel (a) of the figure gains the both-at-once generator's draw of the same slice (2,286 HU, the saturated
patch), five columns with two-line titles; the README's table, figure text and findings follow. `table_rows.sh`
in this run dir prints the two rows.

**If the user wants the sampler probed (not launched; each is minutes on the existing checkpoint):** the same
generator at η = 0 (deterministic DDIM) and at 20 steps; the σ̂ channel clamped to the training median; x₀
clamping was withdrawn in §11.6 for wrecking the mixed generators and would need the same A/B here.

