# gw_detection_demo.md — a CNN against the matched filter on LIGO strain

Goal: the second entry of the Physics section. Train a chapter CNN to detect
binary-black-hole chirps in real Advanced LIGO noise, score it against the
matched filter — whose detection curve in Gaussian noise is a closed form, so
the ceiling is a theorem the way blackjack's is value iteration and
Müller-Brown's is quadrature — and then run it over the public catalogue and
print which events it finds.

Mock of the figure, computed from the physics alone on 2026-09-11:
`scripts/mock_gw_figure.py` (an analytic aLIGO noise curve, a Newtonian chirp
whitened by it, the Marcum-Q detection curve). The CNN curve in it is a
placeholder drawn one SNR unit below the theorem; the trained model's gap to
the theorem is this demo's headline.

Prerequisite reading: `planning/boltzmann_generator_demo.md` (the data-is-the-physics
pattern, the bracketed-table shape, the gates) and the GWOSC Open Data
Workshop notebooks (whitening, Q-transform, matched filter on GW150914).

## 0. The one-paragraph version

Gravitational-wave strain is a 1-D time series sampled at audio rates whose
signal band, 20–2000 Hz, is the band a chirp from two merging black holes
sweeps through in under a second. The Gravitational Wave Open Science Center
(GWOSC) publishes every observing run's strain under CC BY 4.0. Real events
are too few to train on, so the training set is real detector noise with
simulated chirps injected at chosen signal-to-noise ratios — the data is the
physics, as in the Boltzmann demo. Whitened windows become 64 × 128
spectrograms, one image channel per detector, and EfficientNet-B0 or the
CIFAR CNN trains on them with zero new codegen. The instrument is the
matched filter: in stationary Gaussian noise it is the Neyman–Pearson optimal
detector and its detection probability at threshold ρ* for a signal of
optimal SNR ρ is the Marcum Q-function Q₁(ρ, ρ*). Table 1 is that curve, the
same statistic run by PyCBC on the same windows, and the CNN below them;
Table 2 is what real, non-Gaussian noise costs each of them; Table 3 is the
catalogue: which of the ~90 O1–O3 events the trained model flags, beside
their published SNR.

## 1. Why this and not another audio demo

- The waveform-to-spectrogram-to-CNN pipeline is BirdNET's, ESC-50's and
  Speech Commands' too, but those score against a labelled test set. Here
  the ceiling is a theorem and every training example's SNR is a number we
  chose.
- It is the physics the reader has heard of. GW150914 was front-page news;
  "the chapter CNN finds it" is a sentence the book can print with a number.
- It settles a question the literature already asked: Gabbard et al. 2018
  (PRL 120, 141103, "Matching matched filtering with deep networks") showed a
  CNN matches the matched filter in Gaussian noise; George & Huerta 2018
  (Phys. Lett. B 778) did it first; Kaggle's G2Net 2021 was the spectrogram +
  EfficientNet recipe; MLGWSC-1 (Schäfer et al. 2023) is the mock-data
  challenge that ranks pipelines on the Gaussian-to-real-noise ladder this
  demo climbs.
- Nothing in the codegen moves. The spectrogram is host numpy; the network
  is a chapter net on a 2-channel image.

## 2. The data

**GWOSC** (gwosc.org), CC BY 4.0, cite Abbott et al. 2023 (SoftwareX 13,
"Open data from the first three observing runs of Advanced LIGO, Advanced
Virgo, KAGRA and GEO"). Two products:

- Bulk strain: every run (O1, O2, O3a, O3b, O4a) as HDF5 files of 4096 s
  per detector at 4 kHz and 16 kHz, with a data-quality bitmask per second
  (science-mode, and the hardware-injection flags `CBC_HW_INJ` /
  `BURST_HW_INJ` that mark seconds with a deliberately injected signal, to
  be excluded). 134 MB per 4 kHz file. Detectors H1, L1, V1.
- Event files: the 4096 s bulk file holding every catalogued event (GWTC-1/2/3,
  ~90 events through O3; GWTC-4 adds O4a), with the catalogue's network SNR,
  masses, distance. ⚠ The 32 s snippets stopped with GWTC-4: `get_event_urls`
  now returns the bulk file for every event, and the bulk file carries NaN gaps
  away from the event (GW150914's L1 file does), so cut the window and check it.

Access: `.venv-gw` from `requirements-gw-lock.txt` (⛔ never the pinned
`.venv`; 638 MB, pycbc 2.9.0 / lalsuite 7.26 / gwpy 3.0.14 / gwosc 0.8.3).
`scripts/gw_common.py` holds the fetch, the HDF5 read, the PSD and the
whitening, so every script shares one convention. Files are chosen from the
GWOSC timeline: 4096 s slots wholly inside H1 ∩ L1 of `DATA`, `CBC_CAT3` and
`NO_CBC_HW_INJ`, one slot per day across O3a (`pick_o3a_pairs`). Noise budget:
48k windows × 2 s of independent noise is 27 h, 24 files per detector, 3.2 GB
per detector. ⚠ Do not require `NO_CW_HW_INJ`: the continuous-wave hardware
injections ran for the whole of O3, the bit is clear in every second of every
file, and they are narrow lines the whitening removes with the instrument's
own. Exclude the four transient classes (CBC, BURST, DETCHAR, STOCH).

**What the strain looks like.** 10⁻²¹ in strain under coloured noise:
seismic below 10 Hz, quantum above a few hundred, lines at 60 Hz (mains),
~500 Hz (violin modes) and the calibration lines. Everything starts with a
median-averaged Welch PSD from the file itself (4 s segments; median, not
mean, so a glitch does not bias it), whitening by √S_n(f), and a 20–500 Hz
band-pass. After that a BBH merger is a chirp from ~25 Hz to 100–300 Hz in
0.2–1 s; a BNS (GW170817) is the same chirp over a minute, and out of scope.

**The window.** 2 s at 4096 Hz, H1 and L1 (V1 optional, third channel),
signal coalescence time uniform in [0.7, 1.8] s of the window. The two
detectors see the same chirp with ≤ 10 ms delay and different amplitude and
phase from their antenna patterns; that coincidence is what a 2-channel
input gives the CNN and a single-detector search does not have.

## 3. Data — the data is the physics

`preprocess_gw.py [--pairs=26] [--val-pairs=6] [--workers=13] [--out=data/gw]`
(repo root, beside `preprocess_boltzmann.py`; ~15 min on 13 cores once the
files are cached under `data/gw/raw/`):

- Fetch the O3a files, read the DQ mask, keep science-mode, injection-free
  seconds, and cut non-overlapping 2 s windows on the integer-second grid, 8 s
  in from each end of the file. Val is every 5th file pair (6 of 26) with a
  third of its windows injected; train is the rest at half.
- **Two noise sets.** (a) Gaussian: white noise coloured by the file's own
  PSD, the theorem's regime. (b) Real: the strain itself. Same windows, same
  injections. Both are whitened ONCE per file by one filter (1/√S_n
  inverse-spectrum-truncated to 4 s, times a raised-cosine 20–500 Hz mask);
  each injection is whitened by the same filter on a 16 s grid and ADDED in
  the whitened domain, so a window is exactly whitened noise + whitened
  signal and nothing leaks into its neighbour.
- **Injections** with PyCBC: `get_td_waveform(approximant='IMRPhenomD',
  mass1, mass2, delta_t=1/4096, f_lower=20)`, m₁, m₂ uniform in [10, 50] M☉,
  zero spin, random sky position and polarisation, projected onto each
  detector with `Detector.project_wave` (antenna pattern + light-travel
  delay). Scaled to a target optimal **network** SNR ρ = √(ρ_H1² + ρ_L1²),
  ρ uniform in [4, 20]. ρ is the optimal SNR of the signal INSIDE the
  window, in the whitened domain: `pycbc.filter.sigma` of the whitened
  signal against the measured (flat) PSD of the whitened Gaussian twin over
  20–500 Hz. A 10 + 10 M☉ chirp from 20 Hz lasts 6 s, so light systems are
  truncated at the window start with a 0.1 s half-Hann ramp and only the
  in-window part counts. The implied distance (1 Mpc / scale) is recorded.
- **Spectrogram**: a constant-Q filterbank, not an STFT — 64 Morlet bands
  with centres log-spaced over 20–500 Hz and σ_f = f/8 (Q = 8), hop 64 →
  128 frames, |analytic band signal|², divided by the band's median over
  time, log, floored at −5. (The plan's 256-sample STFT has 16 Hz bins; the
  chirp lives between 25 and 60 Hz for most of its length.) The decimation
  is done by folding each band's spectrum modulo the decimated length, 64
  short inverse FFTs per window instead of 64 full ones. Written as flat f32
  `[N, 2, 64, 128]` plus int32 labels and a sidecar `meta_<split>.npz` with
  every injection's parameters (masses, sky, per-detector arrival time,
  SNR per detector and network, distance) and the matched-filter statistics
  of §5, so the scorer bins by SNR without re-deriving anything.
- ⚠ The injection SNR and the whitening must use the SAME PSD. A mismatch
  shifts every row of Table 1 by the same factor and nothing else notices.
  The sidecar carries `sigma` of the RAW signal against the raw PSD beside
  the whitened-domain ρ; their ratio is 1.000 ± 0.001 on the smoke set, and
  `gw_metrics.py` prints it first.
- ⚠ The coalescence time is geocentric; each detector sees it up to 21 ms
  earlier or later (`time_delay_from_earth_center`). The per-detector
  `tc_h1` / `tc_l1` are what "known time" means below.
- `manifest.json`: GPS ranges, PSD segments, band, spectrogram geometry, the
  SNR grid.

Sizes: a 2-s window as a 2 × 64 × 128 f32 spectrogram is 64 KB; ~41k train
+ ~12k val = 3.4 GB per noise set, 7 GB raw, ~14 GB in all.

## 4. The model and the trainer — zero new codegen

`demos/MainGwDetect.lean`, `lake exe gw-detect [arm] [epochs]`:

- Net: EfficientNet-B0 with a 2-channel stem on 64 × 128 (the chapter net at
  a fifth of its ImageNet input, so a forward is milliseconds), or the CIFAR
  CNN as the small arm. Two-class head, CE loss, the ordinary
  `generateTrainStep` with int32 labels — the blackjack/2-D pattern of a
  host loop around the standard train step, no `DatasetKind`. Built
  2026-09-11: the first non-square input in the repo, and the codegen took
  it without a change. ⛔ Size the packed buffer from `heInitParams`, not
  `spec.totalParams`: on SE nets the two disagree (B0 4.0M vs 7.1M — Spec.lean
  counts SE off the block input, `paramShapes` and the graph off the expanded
  width, since ca6a655d 2026-06-02), and the wrong nP reads the loss from
  inside a weight tensor and segfaults in the BN EMA. `Train.lean:697` has
  the same exposure for any `useSE` spec, unverified. 150 ms/step at B=64 on
  a 4060 Ti, GPU 40 % busy (the host gather), ~60 s per epoch.
- Arms: `gauss` (trained on set a), `real` (trained on set b). Evaluated on
  both, the 2 × 2 of Table 2.
- Output: the logit per window; `score` mode writes the logits for the val
  set of both noise sets, and for the catalogue snippets (§6.4).

**The 1-D route** (Gabbard et al.) — whitened series straight into a 1-D
CNN — is the comparison arm, not the first build: a `1 × 8192` image with
square kernels pads to nothing useful and `unetDown`/pool2×2 collapse H = 1,
so it needs strided `convBn` with k × 1 kernels or a 1-D primitive. Phase 5
if the spectrogram arm leaves room.

## 5. The instrument — `scripts/gw_metrics.py`

Reads the logits, the sidecar and the matched-filter statistics; prints one
table per noise set:

| column | how |
|---|---|
| threshold | set EMPIRICALLY on the noise-only val windows so P_fa = 10⁻³ per window, separately for the CNN and for the matched filter |
| P_d per SNR bin | fraction of injected windows above threshold, bins of width 1 from 4 to 20, ± binomial error |
| SNR at P_d = 0.5 | interpolated; the one-number summary, "the SNR the detector needs" |

Rows the table always carries: the theorem at the empirical threshold,
evaluated at each injection's own SNR and averaged over the bin — Marcum
Q₁(ρ_det, ρ*) for one detector, Q₂(ρ_net, ρ*) for the network; the matched
filter with the TRUE template run by PyCBC on the same windows
(`pycbc.filter.matched_filter`, max |ρ| over the window per detector, and
for the network the COHERENT statistic max over t and the ±10 ms delay of
|z_H1(t)|² + |z_L1(t + δ)|² — adding two per-detector maxima in quadrature
hands a one-detector signal the other detector's noise maximum for free and
is not what Q₂ describes); the same at the known detector-frame time ± 5 ms,
which is the closed form's own regime; the matched filter with a small
template bank (~200 templates over the mass range, 3 % minimal match, the
statistic the real search runs — deferred to Phase 3, it is not a gate); the
CNN; and a "random" row. The fitted N_eff of the single-detector max
statistic (ρ* = √(2 ln(N_eff / P_fa))) is printed as a Gaussianity check.

Mock rows at ρ* = 8 (the search's conventional single-detector threshold),
from the closed form:

| ρ | theorem | bank, 3 % mismatch | placeholder CNN at ρ − 1 |
|---|---|---|---|
| 6 | 0.027 | 0.018 | 0.002 |
| 7 | 0.175 | 0.127 | 0.027 |
| 8 | 0.525 | 0.430 | 0.175 |
| 9 | 0.855 | 0.785 | 0.525 |
| 10 | 0.980 | 0.961 | 0.855 |
| 12 | 1.000 | 1.000 | 0.999 |

Regression gate: on Gaussian noise the CNN's SNR at P_d = 0.5 within 1.0 of
the true-template matched filter's, at P_fa = 10⁻². Gabbard reports
essentially zero gap; 1.0 is set loose for the reason the 2-D script gives —
nobody has measured retrain variance yet. P_fa is 10⁻² not 10⁻³ because
the val split has ~8k noise-only windows and 10⁻³ is its 8th-largest value;
the scorer takes `--pfa` and both are reported.

## 6. The physics claims, in order of cost

**6.1 The theorem, reproduced (Phase 1, CPU).** PyCBC's matched filter with
the true template on the Gaussian-noise windows must reproduce Q₁(ρ, ρ*)
within binomial error at every SNR bin. This is the calibration of the whole
instrument: if it does not, the injection SNR, the whitening PSD or the
threshold is wrong and nothing downstream means anything.

**6.2 Real noise costs the matched filter too.** The same rows on the real
windows. LIGO noise is non-stationary and non-Gaussian, and the matched
filter's own false-alarm tail is fatter than Rayleigh; the real search adds
a χ² veto for that reason. The CNN trained on real noise may lose less than
the filter does at the same P_fa; that is the finding the demo is fishing
for, and either answer is a row. The 2 × 2 (trained on / tested on) says
whether a model trained on Gaussian noise generalises to the real detector.
Measured 2026-09-11 (val, P_fa 1e-2): per detector the search threshold
rises from 4.90 to 5.55 (H1) and 5.27 (L1) and ρ₅₀ from 7.74 to 8.49 (H1),
6.03 to 6.66 (L1) — mild. The COHERENT network statistic's threshold jumps
from 5.86 to 10.40 and its ρ₅₀ from 5.27 to 10.32, because it unions both
detectors' glitch tails (L1's fatter one dominates). At 1e-3 the network
threshold is 417: eight val windows hold glitches that loud, and the
matched filter without a χ² veto has no row at all. The known-time rows are
untouched (ρ* 3.12 → 3.02), which is the point: glitches are elsewhere in
the window, and the search pays for not knowing when.

**6.3 The catalogue.** Fetch the 32 s H1+L1 snippets of every GWTC event,
whiten with the snippet's own off-source PSD, slide the 2 s window, take the
max CNN score. Table 3: event, published network SNR and masses, matched
filter ρ with the catalogue masses, CNN max score against the trained
threshold, found / missed. Every event above SNR ~10 should be found; the
marginal ones (SNR 8–10: GW151012, GW170729) are the interesting rows, and
the model never saw a real signal.

**6.4 Glitches.** Gravity Spy (Zenodo, CC) lists GPS times and classes of
labelled glitches; a "blip" is a sub-second broadband burst that mimics a
heavy-mass chirp. Cut windows at those times, score them: the CNN's and the
matched filter's false-alarm rate on blips against Gaussian noise is one
row each, and the reason the real search has a χ² veto.

**6.5 The 1-D arm** (§4), if built: same table, one more row.

## 7. Figure and section

`scripts/gw_figure.py`, grown from the mock: (a) a whitened 2 s window with
its injected chirp overlaid, (b) the two spectrograms the CNN must tell
apart, noise-only and with the chirp, (c) P_d against injected SNR with the
theorem, the PyCBC matched filter, the CNN, and the catalogue events as
ticks at their network SNR. Blue for the model, orange for the physics.

Section: a `\paragraph` in the Physics subsection after the Boltzmann
generator, in the demo shape — three things that change (the data is real
detector noise with the signal we chose; the instrument is the
Neyman–Pearson optimum; a real-world catalogue is the test set), Figure,
Table 1 with lead-in and one caveat, Table 3. Bestiary: `Bestiary/GwCnn.lean`
(Gabbard 2018's 1-D CNN and the G2Net spectrogram B0, zero new primitives
for the latter) and `Bestiary/PhaseNet.lean` (Zhu & Beroza 2019, the 1-D
UNet that picks P and S arrivals on STEAD — the seismology twin of the
segmentation UNet, one new primitive: 1-D pooling). Demo-datasets table: a
GWOSC row with `download_gwosc.sh`.

## 8. Phases

```
Phase 0 (½ session, CPU):   venv; fetch one O3a H1+L1 file pair and the GW150914
                             file; PSD, whiten; reproduce the published SNR
                             (H1 ≈ 20, L1 ≈ 13, network 24)
                             Gate 0: within 1 of the published values
                             ✅ 2026-09-11 `scripts/gw_gate0.py`: H1 19.83, L1 13.87,
                             network 24.20, L1 leads H1 by 7.1 ms (IMRPhenomD at
                             the detector-frame 39.2 + 31.6, 128 s median-Welch
                             PSD); the O3a pair at slot 1238777856 reads with
                             4096 clean seconds and no NaNs; runs/2026-09-11-gw-gate0/
Phase 1 (1 session, CPU):    §3 preprocess + §5 scorer; 6.1
                             Gate 1: PyCBC matched filter = closed form on Gaussian noise
                             ✅ 2026-09-11 PASS at P_fa 1e-2 and 1e-3 (runs/2026-09-11-gw-gate1/):
                             the single-sample known-time filter sits on Q₁: H1 ρ₅₀ 4.68 vs
                             4.75 (1e-2), 5.79 vs 5.89 (1e-3), L1 4.69 vs 4.68, every bin
                             inside 2σ; sigma(raw)/ρ(whitened) = 1.0001 ± 0.001 over 4051
                             injections. The search rows (max over the window) per
                             detector: ρ₅₀ 7.74 vs 7.84 (H1), 6.03 vs 6.26 (L1) at 1e-2.
                             ⚠ N_eff of the single-detector max is ~1280 (8192 samples ×
                             2 phases / ~13 correlated samples), so ρ* = 4.85 at 1e-2.
                             Dataset: 40,760 train (20,312 injected) / 12,228 val (4,051),
                             26 pairs, 14 GB, 11 min on 8 cores.
Phase 2 (1 session, GPU):    §4 trainer, `gauss` arm; Gate A (§5)
Phase 3 (½ session):         `real` arm, the 2 × 2, 6.2
                             ✅ 2026-09-11 both done; every arm below is a (gauss, real)
                             pair trained on the two noise sets and scored on both. SNR at
                             P_d = ½, P_fa 1e-2, val (4051 injected / 8177 noise-only):

                                                          trained on gauss   trained on real
                                                   params  ->gauss ->real    ->gauss ->real
                               B0, 2-ch stem, 3 ep          7.1M   7.46   7.74     7.67   7.48
                               B0, 20 ep (overfit)          7.1M   7.71   8.06     7.68   7.62
                               2 stacks + GAP head, 6 ep    0.3M   7.17   7.35     7.37   7.18
                               CIFAR-BN 2 stacks 512-512    17.1M   6.84   7.46     7.01   7.00
                               CIFAR 3 stacks 512-512       8.9M   6.64   7.28     6.81   6.75
                               CIFAR 4 stacks 512-512       5.6M   6.95   7.22     6.78   6.90
                               CIFAR-BN 2 stacks, ls 0.1    17.1M   6.89   7.32     7.00   7.15
                               PyCBC coherent search               5.27  10.32     5.27  10.32

                               chapter 4 CIFAR-CNN8-wide-BN  0.83M   6.80   7.17     6.65   6.91

                             ⭐ The demo's net is chapter 4's CIFAR-CNN8-wide-BN VERBATIM
                             (`cifar8w` in MainGwDetect.lean: 8 convs in 4 conv-conv-pool
                             stages at 16/16/32/32 into 512-512-out; only the 2-channel
                             stem and 2-way head differ) — it matches or beats every wider
                             or deeper variant and trains in 12 s/epoch. GAP and label
                             smoothing are chapter-5 machinery the section does not reach
                             for (smoothing moved nothing at 6 epochs anyway). ⚠ the
                             `apps/baselines` CIFAR-BN (2 stacks, 32/64) is NOT the
                             chapter's exhibit; the book's net is the 4-stage 16/16/32/32.
                             ⛔ Gate A FAILS on Gaussian noise for every arm — 1.53 for the
                             chapter net, best 1.38 (3 stacks) against the tolerance of 1.0 (Gabbard's ~0
                             gap was a 1-D CNN on whitened strain with ~10× the training
                             set; the constant-Q log-power front end discards phase).
                             ⭐ In real noise every CNN BEATS the matched filter by ~3,
                             because the search has no χ² veto and glitches set its
                             threshold; each net is best on the noise it trained on by
                             ~0.2–0.6. At 1e-3 the split is total: a gauss-trained CNN is
                             glitch-dominated in real noise like the filter (no ρ₅₀), the
                             real-trained 3-stack net still reads 7.58.
                             ⚠ B0 at 20 epochs OVERFITS 40k windows (train loss 1e-4, P_d
                             never reaches 1); 3 epochs is its schedule. Logs, tables and
                             figures: runs/2026-09-11-gw-*/, runs/2026-09-11-gw-gate1/
                             (compare_nets.log, gw_detect_{cifarbn3,cnn,e3}.png).
Phase 4 (½ session):         6.3 the catalogue table
Phase 5 (½ session):         6.4 glitches; the 1-D arm if cheap
Phase 6 (½ session):         §7 figure + section + two bestiary entries
```

Training is minutes on one card at B0 on 64 × 128; the download is a few
GB once. Nothing needs asking about.

## 9. Gates that fail loudly

- Gate 0 is the pipeline: a wrong PSD convention, sample rate or GPS offset
  puts GW150914's SNR off by a factor, not a percent.
- Gate 1 is the instrument: the empirical matched filter must sit on the
  closed-form curve in Gaussian noise. If it sits below it uniformly, the
  injection SNR and the whitening disagree on the PSD (§3 ⚠).
- The noise-only val set must give the target P_fa for BOTH thresholds; a
  threshold set on the training noise is a leak.
- The catalogue must be found. If GW150914 (SNR 24) is missed, the
  snippet's whitening does not match the training set's.

## 10. Out of scope

Parameter estimation (masses, distance: a regression demo of its own);
binary neutron stars (minute-long signals, a different window); continuous
waves and bursts; real-time streaming; Virgo/KAGRA as channels (a flag, not
a phase); glitch CLASSIFICATION as a demo (Gravity Spy is a row here).

## 11. Notes before starting

- Optimal SNR is ρ² = 4 ∫ |h̃(f)|² / S_n(f) df; PyCBC's `sigma` computes
  it. Network SNR adds in quadrature. Threshold ρ* = 8 is per detector in
  the real search and corresponds to a far lower false-alarm rate than the
  per-window 10⁻³ used here; the demo's thresholds are empirical.
- Whitened data have unit variance per sample only after the band-pass is
  accounted for; normalise the injected template's norm AFTER the same
  filter, or the SNR is off by the band-pass's variance loss.
- Hardware injections are real signals in the strain (deliberate, flagged);
  a training window containing one is a mislabelled positive.
- The 60 Hz line and its harmonics survive an imperfect PSD estimate as
  horizontal streaks in the spectrogram; the median-normalised display in
  the mock removes them, and so should the preprocess.
- `scripts/mock_gw_figure.py` is the template for `gw_figure.py`; its
  numbers are the closed form, not results.
