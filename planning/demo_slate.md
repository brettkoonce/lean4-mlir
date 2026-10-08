# demo_slate.md — the chapter-10 demos: what stays, what swaps, what each section still owes

**Opened 2026-10-05**, from a read of every demo section of `blueprint/src/content.tex` (§10)
against its planning doc, with the user. The approach is last in, last out: a new demo
replaces an old one only once it is in the book. This doc is the slate and the backlog; each
demo that gets built gets its own plan.

## §1 The slate

| slot | today | proposed | state |
|---|---|---|---|
| Beyond vision | NQS on the Ising chain, then CASP16 as §10.3.7 (landed 2026-10-08) | — | `casp16_distogram_demo.md`; the Evoformer entry moved under §10.3.7; whether NQS stays here or moves to Physics is the open row below |
| Player of games | blackjack, Pong, tic-tac-toe | Pong, **Leduc poker**, tic-tac-toe | `leduc_deep_cfr_demo.md`; blackjack is deleted once poker lands (its §8) |
| Image generation | no demo (VAE/DCGAN/Pix2Pix/CycleGAN entries only) | LIDC-IDRI low-dose CT denoising | idea, §3 |
| Quantum computing (new subsection) | — | surface-code decoder: a CNN against matching and the exact optimum | idea, §4 |
| Physics | Boltzmann generator | Boltzmann, and NQS if it is kept | open: NQS moves here or lives only in the repo |
| someday | — | SAT / GNN | parked, §5 |

Decisions taken 2026-10-05:
- **One DQN, Pong.** DQN exists for state spaces a table cannot hold, and blackjack's own
  table has tabular Q beating the DQN (−0.0440 against −0.0476). Pong is Mnih et al.'s
  setting and has the better finding (pixels beat the six-number state). The solved-game
  instrument stays in the section through tic-tac-toe, and poker adds exploitability.
- **The third game is imperfect information** (Leduc, Deep CFR), not a cart-pole-style control task
  (PPO against LQR was considered; the user: "have seen a few times").
- **CASP16 joins Beyond vision** — 2026-10-08: added as §10.3.7 after NQS rather than in its place
  (Brett: "10.3.7 is where I'll have you put it"). Its section leads with the bracket — the 3B
  contact head, the net, ESMFold, the CASP16 field — as GW leads with the matched filter, because
  the net sits at 0.646 against ESMFold's 0.778. Whether NQS then moves to Physics is still open.

Cross-references the swaps break (`content.tex` lines as of 2026-10-05): NQS is cited at
:17216 (tic-tac-toe) and :17699; NQS cites blackjack at :16757. The host-weight trick (target
written so the squared-error block's cotangent is a chosen g) is introduced by blackjack and
generalised by NQS; after both go, Pong introduces it and tic-tac-toe and poker cite Pong.

## §2 What the existing sections owe

In order of value per cost.

1. **BraTS: the stalest section.** The book has the pooled 10-epoch table (mIoU 0.741, WT
   0.910); the per-patient protocol, the tail and the two-seed 3D verdict are only in
   `brats_25d_3d.md` §4d, which says "the verdict goes in the book". The section is also
   in the old voice (tutorial bullets, no published row, no "what this is not").
   *2026-10-05:* the 20k-step, two-seed 3D verdict is in the section (the per-patient table's 3D
   row, and a tail / false-alarm table at min-ET 200 in the through-plane paragraph). Still owed:
   the pooled lead table, the old voice, a published row.
2. **TinyGPT: the weakest section.** "In passing", no question, brackets only bigram and
   uniform. `lm_demos_modernization.md`: a rerun overwrites the checkpoint (resume is
   documented, not implemented), no dropout while `tiny` overfits, the train/val curve is
   logged and not drawn. A Kneser–Ney n-gram ladder is its matched filter; the RoPE
   length-extrapolation gate (§5 there) is a ready-made result.
3. **VisDrone and NEU-DET: every row is n = 1.** NEU-DET's prose cites "the noise floor the
   VisDrone table established", which was never measured. `LEAN_MLIR_SEED` exists now (it
   landed for BraTS); the top rows need two more seeds. `detector_v5_next.md` §1 has the
   rest: T1+T2 unmerged on `yolo-v5-assignment`, compiled defaults that train r50 / 12
   epochs / no augmentation, bootstrap BN stats falling back to zeros.
   *2026-10-05:* the "noise floor" sentence now says the +0.007 is one run against one run.
4. **GW.** A live `% TODO Phase 4` in the tex (catalogue table, Gravity Spy row). The
   section blames the 1.5-SNR gap to the matched filter on the spectrogram discarding phase
   and does not test it; a 1-D arm on the whitened strain would.
5. **Remote sensing.** The last paragraph promises change detection ("the first half of the
   instrument"). Build it from the Bahia September/March pairs or cut the sentence.
6. **Pong.** "Pixels beat the state … not dissected here" is the open question; one cheap
   ablation (normalised state inputs, or the CNN on a rendered state).
7. ArASL, PlantVillage, Boltzmann, DDPM, tic-tac-toe: nothing owed.

Chapter-wide:
- **n ≥ 3 for any row the prose compares.** ArASL, Plant, remote sensing and Pong do; the
  detectors, BraTS's book table and blackjack do not.
- **"Three things change relative to …"** opens about eight sections (BraTS, TinyGPT, DDPM,
  Boltzmann, GW, NQS, blackjack, tic-tac-toe). The newer sections lead with the question
  and end with "what this is not"; bring the older ones to that shape.
- **Not covered anywhere:** self-supervised / contrastive learning (CLIP has an entry, no
  demo). A SimCLR-on-Imagenette linear probe against R34's 89.99 would be the cheap version.
  Only if adding; nothing above needs it.

## §3 Idea: LIDC-IDRI low-dose CT denoising (Image generation)

- **Question:** does the denoiser keep the nodules? PSNR/SSIM beside nodule contrast-to-noise
  by nodule size, with LIDC's four-reader annotations as the instrument. The expected
  finding is the chapter's theme: an MSE UNet wins PSNR and erases small nodules; a diffusion
  denoiser (the DDPM UNet reused) keeps texture and can invent it.
- **Noise model:** Poisson noise in the sinogram, then filtered back-projection. Not
  image-domain Gaussian, which is not what low dose looks like (real low-dose noise is
  correlated streaks).
- **Arms:** full-dose FBP (ceiling), low-dose FBP (floor), a classical filter (BM3D or TV),
  MSE UNet, Noise2Noise (free with simulated noise: no clean target), diffusion.
- **Check before committing to simulation:** TCIA's LDCT-and-Projection-data (Mayo, real
  paired quarter-dose). If usable, it answers "why simulated?"
- **Disk:** about 93 GB free on `/` (2026-10-05) against roughly 125 GB for all of
  LIDC-IDRI. Use a patient subset.

## §4 Idea: a surface-code decoder (Quantum computing)

- **The net:** the chapter-4 CNN on the syndrome grid (one bit out: was the logical qubit
  flipped), repeated measurement rounds as input channels. Zero new codegen.
- **The ladder:** minimum-weight perfect matching (PyMatching; Edmonds' blossom, the
  graph-algorithm bracket); the exact optimal decoder, tabulated in C for independent
  bit-flip noise at d = 3 and 5 (25 data qubits, 2²⁵ patterns); at larger d the
  thresholds as the theory states them, about 10.3% for matching and about 10.9% optimal
  under code-capacity bit-flip noise (the random-bond Ising model at the Nishimori point;
  Dennis et al. 2002, Wang, Harrington & Preskill 2003).
- **Question:** does a net trained on syndromes beat matching, and how much of the gap to the
  optimum does it close? Under depolarizing noise matching decodes X and Y separately; the
  CNN sees both.
- **Data:** Stim, simulated, nothing to download. Context: AlphaQubit (Nature 2024).
- **Caveat, the same as blackjack's and Leduc's:** at d = 5 the optimum is a 4,096-entry
  table. The rows that matter are d = 7, 9, 11, where there is no table and the net must
  generalise against matching.
- **Gate 0:** our Stim + PyMatching pipeline reproduces matching's published threshold
  before any net trains.

## §5 Parked: SAT with a graph network

NeuroSAT-style (Selsam et al. 2019): predict satisfiability of random 3-SAT near the
clause-ratio-4.27 transition; a CDCL solver (kissat) is the exact answer, survey propagation
the classical inference bracket. Parked because it needs message passing on the
variable–clause graph, which the kit does not have (`bestiary_candidates.md` prices it as one
new primitive): codegen plus a proof obligation, where the decoder needs none. Message passing
would also open MPNN, GraphCast and NequIP for the bestiary.
