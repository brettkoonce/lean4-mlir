# part2_ship_audit.md — Part 2 (the bestiary) at HEAD 671e4aa8: what ships, what each demo owes, the runs in order

**Opened 2026-10-27** from a read of every demo section of `blueprint/src/content.tex` (chapters 11–13)
against its planning doc, its `runs/` directory and its code, after Part 1 converged. The order of
work agreed with the user: the runs first (the four-card box is freed for them), the copy later.
`content.tex` line numbers are at 671e4aa8. Companion: `planning/demo_slate.md` (the slate; §2 there
is the 2026-10-05 version of this list), `planning/book_review_2026-10-26.md` (the reviewer's items
for ch 12–13 at :694–734, 5 of 31 fixed by 0e9c6d9b).

## §1 The verdict

Sixteen demos are built. Nine are a prose pass from done; the real work sits in four sections
(TinyGPT, VisDrone, LIGO, NQS). One demo (denoising) does not exist.

| demo | ships? | between it and done |
|---|---|---|
| ArASL | yes | two low phrasings (§5) |
| PlantVillage / PlantDoc | yes | two low phrasings |
| Leduc Deep CFR | yes | two blank table cells, two phrasings |
| AlphaZero tic-tac-toe | yes | one sentence |
| CASP16 distogram | yes | one ambiguous phrase; nothing to run |
| Boltzmann generator | after an hour | two prose fixes, a README line, a seed study that runs in minutes |
| DDPM on MNIST | after an hour | question-first opener; introduce the 2-D toy before citing it |
| MuZero | after half a day | land `origin/muzero-ttt` (§4), prior-art check, five prose items |
| Remote sensing | after cutting one sentence | the change-detection promise; three low phrasings |
| NEU-DET | after ~3 GPU-h | the section says n = 1 itself; two seeds close it |
| Pong | after a short run | "not dissected here" (content.tex:19407); one knob + ~1.5 GPU-h |
| BraTS | after a session | third seed (~2.5 GPU-h); pooled lead table; the old voice; a published row |
| NQS | after a prose session | the Var column prints 0 (true 0.75 / 4.0); three overclaims; bestiary entry |
| LIGO | after a prose session | one false claim; four stale "half"s; the TODO; the README commands |
| VisDrone | after a session + ~4 GPU-h | n = 1 everywhere; mislabelled rows; compiled defaults ≠ recipe; no run cited |
| TinyGPT | after a session + ~2 GPU-h | both high review items; the measured ladder is unpublished |
| Denoising (Image generation) | not built | §6 |

## §2 The grind — runs in order

Budget: ~12 h × four 4060 Ti. The must-have rows total ~10 GPU-h; with every VisDrone row ~35.
Each run writes to a fresh `runs/<date>-<name>/` with a README in the usual shape (command, seeds,
numbers, what it says). Nothing in the book moves until §5.

**Before launch, on the four-card box.** (a) The detector and BraTS data (`data/visdrone_fpn`,
`data/visdrone448`, `data/neu_det_fpn`, `data/neu_det448`, `data/brats224`) and the R34 bootstrap
(`jax_r34_imagenet.bin` and its `_bn_stats.bin` companion) are not on the CPU box; confirm they
are on the GPU box before anything detector-shaped starts (`LeanMlir/Train.lean:795–808` is the
loader; the BN stats fall back to zeros silently, `planning/detector_v5_next.md` §1.6). (b) The
generic trainer's seed knob is `LEAN_MLIR_SEED=<k>` (`LeanMlir/Train.lean:651–654`); it prints
"seed offset: k" at launch — if that line is missing, the exe is not on the seeded path.
(c) `FPN_BACKBONE` defaults to r50; always pass it.

### 2.1 TinyGPT (card 0, first; ~4 GPU-h with everything)

Each job in its own directory with its own `.lake/build` and `data/`, `ffi/` symlinked — the
trainer always writes `.lake/build/tinygpt_shakespeare_params.bin` (`demos/MainTinyGptShakespeare.lean:584–586`),
so concurrent jobs of one config clobber each other. Pattern: `runs/2026-10-21-tinygpt-ngram-ladder/README.md:14–18`.
XLA is the default lowerer (leave `LEAN_MLIR_LOWERER` unset). Measured: nano 10k steps = 173 s on
one card (`runs/2026-09-09-tinygpt-nano-xla/timing.txt`); tiny is unmeasured on this stack —
budget 5–30 min per 10k until the first run prints its rate.

| # | run | command (per seed s ∈ {0,1,2}) | why |
|---|---|---|---|
| T1 | tiny 10k, scored | `LEAN_MLIR_SEED=$s LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$g ./tinygpt-shakespeare train tiny 10000 32 30` then `… score tiny .lake/build/tinygpt_shakespeare_params.bin <run>/tiny_s$s` | the empty ladder row (`tinygpt_ngram_ladder.md:176`) and the "val turns up" curve (`lm_demos_modernization.md:59–60`) |
| T2 | nano 20k and 40k, scored | `train nano 20000 32 30`, `train nano 40000 32 30`, each then `score nano …` | the "undertrained" follow-up by steps (`tinygpt_ngram_ladder.md:162–164`) |
| T3 | nano B = 64 | `train nano 10000 64 30` + score | more tokens per step, same wall |
| T4 | RoPE extrapolation gate | `train nano-rope-nopos 10000 32 30`; `xeval nano-rope-nopos 64`, `128`, `256`; contrast `xeval nano 128` on a copied nano checkpoint (must fail) | `lm_demos_modernization.md:76–81` §5 form; the plan's P2-at-T form needs §3 item 3 |
| T5 | tiny 5k | `train tiny 5000 32 30` + score | the historical best-val point (`historical/RESULTS.md:64–65`) |
| T6 | dropout arm | blocked on §3 item 1 | the plan's "indicated fix" (`lm_demos_modernization.md:36–38`) |

Then `python3 scripts/demos/ngram_ladder.py table --run <run> --nets nano,tiny` and
`scripts/demos/ngram_ladder_figure.py <run> <png> --nets nano,tiny` (CPU).

### 2.2 NEU-DET (~3 GPU-h): two more seeds on the three FPN arms and the grid

`runs/2026-09-17-neudet-fpn-run1/README.md:57–61` is the launch block; add `LEAN_MLIR_SEED=1` and
`=2` and a seed suffix on the tags, then `scripts/neudet_eval_sweep.sh` as there. FPN ~25 min per
30 epochs, grid ~14 min (`…run1/README.md:65`). Closes content.tex:16895–16897 ("no seed spread has
been measured on this detector").

### 2.3 BraTS (~2.5 GPU-h): a third seed on the three 10-epoch arms

`scripts/sweeps/run_brats_r34_ab.sh 10 data/brats224` (r34 + scratch, two cards) and
`CUDA_VISIBLE_DEVICES=$g ./.lake/build/bin/unet-brats-r34 data/brats224 10 scratch noskip`, each
under `LEAN_MLIR_SEED=2` (seed 1 is `runs/2026-09-25-brats-r34-xla`; check that README for whether a
second seed of the 2-D arms exists before launching — `tab:brats_3d` is the only two-seed table).
~4.5–5.5 min per epoch (`…brats-r34-xla/README.md:43–44`). Score with `scripts/brats_r34_ab.py`.

### 2.4 VisDrone (~4 GPU-h for the top row; ~20 for all five R34+FPN rows)

The top row first, seeds 1 and 2:

```
LEAN_MLIR_SEED=$s LD_LIBRARY_PATH=ffi CUDA_VISIBLE_DEVICES=$g \
  FPN_BACKBONE=r34 FPN_TOWER=0 FPN_TAG=aff30_s$s \
  FPN_AUG=1 FPN_CLSW=none FPN_CLSFOCAL=2 FPN_AFFINE=50 FPN_EPOCHS=30 \
  .lake/build/bin/yolov1-visdrone-fpn data/visdrone_fpn
```

then `infer` and `scripts/yolo_map_visdrone.py … --multilabel --topk 3000 --ml-k 3 --ml-floor 0.05`
exactly as `runs/2026-09-01-visdrone-affine/README.md:64–69`. ~2 h per 30-epoch run. If the cards
are free, the other four rows of `tab:visdrone` at two seeds each (the rows' env settings are in
`runs/2026-09-01-visdrone-affine/README.md` and `…-decode-sweep/README.md`). Note the 09-01 README
(:54–55) predates the seed knob; `Train.lean:653` has it now.

### 2.5 Pong (~1.5 GPU-h; needs §3 item 4 first)

Three seeds of the new arm beside the existing `runs/2026-09-25-pong-dqn/` ones
(`lake exe pong-dqn mode=<new> seed=$s tag=…`, XLA). State runs are 4.5–6 min, pixel runs 27 min.

### 2.6 Boltzmann seed study (minutes)

Three more trainings of the `flow` arm at 20,000 steps (`runs/2026-09-11-boltzmann-generator/README.md`
has the command; 30 s a run) and `scripts/demos/boltzmann_metrics.py score … --gate`. If the exe
has no training-seed knob (`demos/MainDiffusion2d.lean` seeds only the sampling points, :457),
that is a one-line knob first. Closes the single-checkpoint caveat (`…/README.md:81`).

### 2.7 Not runs: CASP, LIGO, MuZero

CASP owes nothing for the book as written; the plan's next levers (templates at inference, 1 day + a
run; recycling, ½ day after; tier B; the parked MSA route, ~3 days) are new work, not debts
(`casp16_distogram_demo.md:656–659, 667–736`). LIGO's unbuilt phases need code before any run (§3
item 7) and §5 proposes dropping them. MuZero's B50 runs already exist on the remote branch (§4).

## §3 Code bits

Blocking a run above:

1. **TinyGPT dropout.** `TrainConfig.dropout` is the classifier-head dropout only
   (`LeanMlir/Types.lean:1235–1240`); `.transformerEncoder` has no dropout argument (:153–157) and
   `MlirCodegen.lean` emits none. The plan calls dropout the fix for "undertrained"; T6 waits on it.
2. **TinyGPT checkpoint path.** Every run writes the same file (`MainTinyGptShakespeare.lean:584–586`,
   `heInitParams` always at :299; `lm_demos_modernization.md` §1.1). A prefix knob or a resume is the
   fix; the per-job-directory pattern is the workaround.
3. **TinyGPT `score` eval length.** `score` scores at the trained `seqLen` only (:487); the plan's
   RoPE gate wants P2 at T = 64/128/256 (`tinygpt_ngram_ladder.md:231–232`). `xeval` (:258–277) is the
   weak form: 20 random batches, and it exits on a size mismatch (:267–269).
4. **Pong ablation knob.** `mode=state|pixels` only (`demos/MainPongDqn.lean:145`); the owed arm is a
   normalised state input or the CNN on a rendered state (`demo_slate.md:67–68`).
5. **Boltzmann training seed** if absent (§2.6).

Not blocking a run:

6. **VisDrone compiled defaults.** `epochs := 12`, `augment := false` (`demos/MainYolov1VisdroneFpn.lean:129,136`),
   `backboneFromEnv` defaulting to r50 with a stale docstring (:197–210); the BN-stats fallback to
   zeros (`Train.lean:795–808`). Make the compiled defaults the shipped recipe (`detector_v5_next.md` §1.5–1.6).
   The T1+T2 assignment work on `yolo-v5-assignment` (commit 514c5f21) is gone from this clone and
   would have to be redone — §5 proposes archiving it.
7. **LIGO.** `README.md:74`'s bare `lake exe gw-detect` trains arm=gauss for 20 epochs
   (`demos/MainGwDetect.lean:134–136`), not the quoted arm=real 6-epoch row; `demos/README.md:717–722`
   calls `preprocess_gw.py` and the figure scripts with `python3` where `gw_common.py:31` imports pycbc
   (needs `.venv-gw`). The 1-D strain arm needs a strided k×1 convBn or a 1-D primitive
   (`gw_detection_demo.md:177–178`).
8. **`scripts/book/log_to_pgfplots.py`** parses only `val accuracy` lines (:24); the TinyGPT curve needs
   a bits/char regex.
9. **Bestiary entries that the plans name and the tree lacks:** `Bestiary/DQN.lean`
   (`pong_dqn_demo.md:124`), `Bestiary/NeuralQuantumState.lean` (`transformer_wavefunction_demo.md:189–194`),
   `Bestiary/GwCnn.lean` and `PhaseNet.lean` (`gw_detection_demo.md:284–288`).
10. **The 2-D toy's CI gate** (`planning/archive/diffusion_2d_demo.md:513–525`): `scripts/demos/toy2d_metrics.py`
    exists, nothing under `.github/` calls it. The η sweep on the fixed noise is also unrun
    (`runs/2026-08-28-toy2d-strip-and-targets/README.md:56–57`).
11. **PJRT RESIDENT banner** prints on every reseed (`ffi/pjrt_ffi.c:1729`).

## §4 MuZero: the branch

The work is on main (f63e6f87 … ec01a890). Local `muzero-ttt` (82371542) is fully superseded —
every non-run line is on main in the same or a later form — and can be deleted. **`origin/muzero-ttt`
(0a6230cb) is three commits ahead of local and main lacks them:**

- bfd4a4cf `rl(muzero): scoreLines stops copying its buffers; searchOnly= and scoreN= for 4×4 scores`
  — `acc := { acc with … }` copied the whole ByteArray per push, so exhaustive 4×4 scoring was
  quadratic and never finished; main still has that form (`demos/MainMuZeroTtt.lean:780–822`). The
  commit also adds `searchOnly=` and a Fisher–Yates `scoreN=`, where main grew a stride-sampled
  `scoreN=` in parallel.
- 1528970b, 0a6230cb — `runs/2026-10-23-muzero-ttt-b50/` and `runs/2026-10-24-muzero-ttt-b50-it150/`
  (~68 files): 4×4 on the Arc Pro B50 at 33 s an iteration; at 150 iterations inits 101 and 201
  play perfectly, init 301's dynamics collapse at 63%. The book does not cite them;
  `deploy/ARC_B50.md:229` does ("Evidence on the `muzero-ttt` branch").

`git merge-tree` of main and origin/muzero-ttt conflicts in `MainMuZeroTtt.lean` (add/add) and
`planning/muzero_ttt_demo.md`. Landing it: cherry-pick bfd4a4cf by hand (keep main's `scoreN`
design or the branch's, not both), cherry-pick the two run commits (new directories, no conflict),
re-run the 4×4 `mode=score` smoke on XLA, then delete both refs. Half a day. Also owed: the
prior-art re-check of arXiv 2411.04580 (`muzero_ttt_demo.md:30, 34–35, 286, 401`; under an hour;
the book claims no novelty and never cites it).

## §5 The copy, per section (later)

In order of value. Every item here is prose, a table cell or a figure; none needs a card.

- **TinyGPT** (content.tex:17676–17800). Land the ladder table and figure
  (`tinygpt_ngram_ladder.md:166–183, 205–206`; KN and nano numbers in `runs/2026-10-21-tinygpt-ngram-ladder/README.md`,
  tiny from §2.1). The bigram "3.56" (:17752) is `bigram-shakespeare`'s train loss; KN bigram on val is
  3.583. "`useSeg` in `TrainConfig`" (:17773) names a field that does not exist (the demo sets
  `lossKind := some .perPixelCE`, `MainTinyGptShakespeare.lean:144`). Drop `\paragraph{TinyGPT, in passing.}`
  (:17747); cite `fig:tinygpt_causal_mask` before the float (:17744 vs :17787); question-first, table,
  figure, "what this is not" (`tinygpt_ngram_ladder.md:213–217`).
- **LIGO** (:18596–18740). "Each network is best on the noise it trained on" (:18723) is false on the
  Gaussian column (6.65 real-trained vs 6.80 Gaussian-trained). "The filter sits on its closed form"
  (:18694): the search is at 5.27 against Q₂'s 5.63. "Neyman–Pearson optimum" (:18635) holds for a
  known signal only; "PyCBC's coherent search" is a quadrature sum. "Half" still at :21377 (Appendix A),
  `demos/README.md:735`, `MainGwDetect.lean:12`, `scripts/datasets/preprocess_gw.py:4`. Delete
  `% TODO Phase 4` (:18730–18732). Soften "the spectrogram discards phase" (:18698–18700) to a
  hypothesis. **Proposal:** archive the catalogue table, Gravity Spy row and 1-D arm
  (`gw_detection_demo.md` §6.3–6.5, Phases 4–5) rather than build them; the section's bracket works.
- **VisDrone** (:16592–16685). Rows labelled "+photometric aug" though the pack has hflip, and
  argmax / multilabel decodes mixed with no column saying so (:16627–16628); no `runs/` cited, no data
  paragraph, no "what this is not" (review :660). Seeds from §2.4. **Proposal:** archive
  `detector_v5_next.md` §1.1–1.3 (T1+T2, per-level balance, the dose question) as research.
- **NQS** (:18791–18970). Mean-field Var(E_loc) prints 0 at :18904, :18911 (true 0.75 and 4.0; the run
  README says 0.75). "10⁻⁵ to 10⁻⁴ at every field but the critical point" (~:18933): GPT at h = 2.0 is
  9.6e-4 unless the lr 1e-3 rerun (`runs/2026-09-11-nqs-ising/n64/reruns`, README :138–145) replaces
  it in the table and panel (c). "Standard error below five percent of every entry" (:18892): the GPT
  entry is 11%. "Variance three times lower" (~:18951). "Lanczos ceiling" (:18968) is unexplained. Close
  `demo_slate.md` §1's open row: NQS stays in Beyond vision, which the book already does; the plan
  header (`transformer_wavefunction_demo.md:23`) still says Phase 5 is not started.
- **BraTS** (:17302–17528). The pooled 0.741 / 0.910 table (:17321–17337) still leads; the
  "Three things change" bullets (:17362–17396); no "what this is not"; no published row
  (:17464–17465 promises one). Third seed from §2.3. Stages D–F (the Lean 3-D port) stay a decision,
  not a debt (`brats_25d_3d.md:280–283`).
- **Remote sensing** (:17163–17300). Cut :17297–17300 ("the first half of the instrument": change
  detection was never built, planned or costed). "Rondônia in June" (:17186) is 4 July; "reflectance
  doubles" (:17251); "every arm recovers about 90" (:17258–17260) — three arms were fine-tuned.
- **Pong** (:19339–19411). "An update was 24 ms … it is 14.6" (:19408–19411); the speed-sweep +15.60 is
  seed 1, not a mean (:19405–19407); "not dissected here" closes with §2.5's arm. Optional: the Q-value
  strip and Grad-CAM panels (`pong_dqn_demo.md:112–119`).
- **MuZero** (:19648–19804). "Sixty iterations is simply short" and the 99.1% arm absent from the table
  (:19775–19781); a "Figure … is" sentence before the float (:19727 vs :19787); "run end" (:19738–19739);
  timing out of the caption (:19744); `\href` on Grimm 2020 and He et al. 2023 (:19666–19668).
- **DDPM** (:18090–18230). Question-first opener (:18103–18104); "the 2-D toy demo's MLP" (:18321–18323,
  in the Boltzmann section) is introduced nowhere; positional words (:18154, :18163, :18218); the
  `fig:ddpm_mnist` caption argues (:18126–18128); "archived" variants (:18099, :18116).
- **Boltzmann** (:18305–18470). "The independent coupling needs ten" (~:18460; the table says 8.6× at
  NFE 10); `README.md:72` still says "to the last digit" (the book now says "to within 0.001"); the
  §7 reverse-process strip for the flow path is accepted (`MainDiffusion2d.lean:450`) but never
  rendered; the seed study from §2.6 and its one-sentence caveat.
- **NEU-DET** (:16805–16907). "Both arms are still rising at epoch 30" (:16902–16903; both peak at e28);
  the bold "Table~\ref{tab:neudet_visdrone_heads} is the result" (:16881) where the numbers are in
  `tab:neudet_results`; seeds from §2.2.
- **Leduc** (:19413–19533). Blank cells: strategy-net bluff, ES-MCCFR head-to-head and bluff
  (:19500, :19502); "three games around it" (:19437–19438), "first game in this section" (:19421);
  timing out of the caption (:19489).
- **AlphaZero** (:19535–19646). "Was expected to take a day" (:19632–19634); timing out of the caption (:19615).
- **Plant** (:17033–17161). "2,578 photographs … 28 classes" (:17047–17050; the paper gives 2,598 / 27);
  "leaves on black 60" is a 3-seed mean over 49–72 with no spread (:17101, :17148).
- **ArASL** (:16909–17031). Hand-typed "The chapter-4 CNN (arasl-signs)" in the data appendix (:21335); no `runs/` cited.
- **CASP16** (:19053–19250). "Is worth nothing" (:19137–19138). `casp16_distogram_demo.md:23` still says
  the ESMFold `--window` run is unrun; it ran 2026-10-08.
- **Catalogue entries.** DCGAN "eight" guidelines, five listed (:17994–17995); Stable Diffusion "three
  components … (CLIP)" (:18277–18279); `Ch~\ref`, ASCII "32x32", a Unicode em dash (:17687, :17777,
  :18233–18234, :19902); "novelty lives in the training" about five times (:18239, :18478, :19002).
- **Chapter-wide, one shape.** No section in :16590–18600 cites its `runs/` directory; ArASL, Plant and
  remote sensing have the question / table / "what this is not" shape, detection, inspection and BraTS
  do not (review :660, :219). Any row the prose compares at n ≥ 3. Captions without timings or recipe.
  Positional words out (:17796, :17959, :18154, :18163, :18218, :18310, :18505, :19249, :19808, :19850).

## §6 The denoising demo (Image generation): a decision

Nothing exists beyond `demo_slate.md` §3: no plan, no data, no scripts, no scoring, and the Image
generation section (:17955) is catalogue-only. It is the one Part 2 item that is a build, not a
finish, and the biggest schedule risk in the part.

The cheapest honest version, two sessions rather than four: TCIA's Mayo LDCT-and-Projection data
(real quarter-dose pairs, so there is no noise model to defend; the slate's own "check before
committing to simulation"), a patient subset; three arms — low-dose FBP as the floor, the MSE UNet,
the DDPM UNet conditioned on the low-dose image — with full-dose FBP as the ceiling; PSNR / SSIM
beside nodule contrast-to-noise by size. Reuse: the BraTS UNet spec under the DDPM trainer's
per-pixel MSE for the MSE arm; the MNIST DDPM UNet (`ReferenceNets.tinyDdpmUnet`, a 2-channel input
already) with the low-dose image as a third channel for the diffusion arm; Noise2Noise is a data-side
arm if a second noisy realisation exists. New code: the conditioning channel in the NetSpec, an
image-target data path, the preprocess and the scoring script; BM3D / TV as a Python bracket. Disk is
no longer the constraint (1.4 TB free on the CPU box, 2026-10-27). Gate 0 is CPU and disk work (data
down, the floor and ceiling numbers) and can run beside the §2 grind.

If Part 2 ships first, the section stays the catalogue it is and the slate paragraph stays the plan.

## §7 Decisions for the user

1. §6: build the two-session denoising demo now, or ship Part 2 without it.
2. §5 LIGO: archive Phases 4–5 (catalogue, Gravity Spy, 1-D arm) or keep them owed.
3. §5 VisDrone: archive `detector_v5_next.md` §1.1–1.3 or redo the lost T1+T2 work.
4. §2.4: the top VisDrone row only (~4 GPU-h) or all five rows (~20 GPU-h).
5. §4: which `scoreN=` design survives when bfd4a4cf lands.

## §8 Log

- 2026-10-27: opened on branch `part2-ship` from the audit (four read-only passes over the demos,
  the runs and the 10-26 review backlog). Nothing in the book or the code changed.
