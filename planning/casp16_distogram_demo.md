# casp16_distogram_demo.md — a verified distogram ResNet scored against the CASP16 field

Goal: the Structure entry of the Bestiary (Chapter 10, "beyond recognition"). A small 2D
ResNet reads a protein sequence through a frozen protein language model, predicts a distance
distribution for every residue pair, is folded by gradient descent into a CA trace, and is
scored on real CASP16 targets with the scorer CASP used, beside every group that predicted the
same targets blind. The gap to the field is the story; the number being right is the deliverable.

Origin: Brett's plan of 2026-10-01 (PIPELINE.md, the plate at
https://claude.ai/artifact/TFoTN9GQvcEHe5eBkwYBvK, panel g already real). This document is that
plan after the feasibility pass against the repo, with the four changes of §1 and the scoring
recipe of §5 validated.

Status 2026-10-01 (evening): §5 scoring validated (G0); §3 steps 1 and 3 done (G1); step 6
done — 84 EU files with the ESM-2 contact-head baseline (easy 0.533 / medium 0.478 / hard
0.233 top-L/5 long-range); steps 2 and 4 done (27,690 chains, 0 fetch failures; 91.5 % of
residues observed, 4.2 % of pairs under 8 Å); step 5 (embeddings) running; `casp16_pack.py`
written and waits on it. Open: the §4 input route (host-tiled 1×1 conv vs a `pairTile` op,
recommendation below), then the trainer, training and the fold.

## 0. The one-paragraph version

A protein is a string over 20 letters whose fold is set by which residues end up near each
other. Before AlphaFold 2, the state of the art (AlphaFold 1, trRosetta, RaptorX, 2018–2020)
was exactly this demo's shape: per-residue features tiled into an L×L pair tensor, a deep 2D
residual network predicting a histogram of Cβ–Cβ distances per pair (the distogram), and a
fold obtained by minimizing the distogram's negative log-likelihood over coordinates. Those
methods fed multiple-sequence alignments; here a frozen ESM-2 35M embedding stands in for the
MSA, so the input is sequence-only and the preprocessing is one forward pass outside the
proof. The net is the chapter ResNet body with its strides removed and one new layer in front
(the pair tile); the loss is the per-pixel cross-entropy the segmentation demo already proves.
Training data is every PDB chain released before CASP16 opened (2024-05-01), clustered and
purged of anything homologous to the targets — the same information cutoff the competitors
had. Scoring is lDDT and TM-score from OpenStructure and US-align, which reproduce CASP16's
official table to the printed digit, so our model's dot goes on the same axis as the 80 groups.

## 1. Decisions (and what they replace)

1. CASP16, Phase 1 (`T1xxx`). The prefixes are phases, not stages: Phase 0 withheld
   stoichiometry, Phase 1 is "the traditional CASP routine" and "the primary phase" in the
   assessors' words, Phase 2 handed out MassiveFold models. Phase 1 = 74 EUs in the paper; the
   domain table has 85 rows (§2 reconciles). CASP17 has empty folders until Dec 2026.
2. No dilation in v1. Sixteen plain 3×3 residual units have receptive field 65, which covers a
   64×64 crop; dilation buys nothing the crop can use. Should a later ablation want it, a
   dilated 3×3 at rate d is a d² phase-split plus an ordinary 3×3 — existing ops, no new VJP.
   (PIPELINE.md and the plate list "dilated conv VJP" as the new proof item; it is not.)
3. The new op is the pair tile. `pairTile` takes the two per-residue blocks of a crop,
   projects each (two dense maps, the two halves of the 1×1 conv on an outer concatenation)
   and outer-sums them into an `[B, C, 64, 64]` map. Its VJP is the dense VJP composed with two
   reduce-sums. One forward tie, one VJP lemma. Relative position needs no one-hot: appending
   each residue's index (as a scalar and a few sinusoids) to its feature vector lets the first
   layer form j − i linearly, since the outer sum is linear in both inputs.
4. Inference by tiling. The full L×L map is assembled from 64×64 windows at stride 32,
   logits averaged over overlaps and symmetrized — the same compiled graph, the same shape, so
   the trained artifact is the verified artifact and nothing is re-rendered per target length.
5. Missing residues are masked exactly with `perPixelWeightedCE`: 66 classes (64 distance
   bins, one ">22 Å" bin, one "unobserved" bin) with weight 0 on the last. The reduction
   divides by Σw, so masked pairs drop out of loss and gradient with no new code.
6. Pseudo-Cβ fold, scored on that atom set on both sides. The net predicts Cβ–Cβ distances
   (the CASP contact convention), so the fold (`scripts/demos/casp16_fold.py`) optimizes one
   pseudo-Cβ point per residue; scoring reduces the field's models and the reference to the
   same atom (`casp16_score.py --pseudo-cb`: Cβ, Cα for glycine, written as CA) and reports
   Cβ-lDDT (the numpy `lddt()`, which equals OST's `--bb-lddt` on CA) and the TM-score over
   those atoms (US-align `-TMscore 1`). CASP's all-atom lDDT would count every missing side
   chain against a trace, so the official column stays the field's number and ours is never
   compared to it directly. Check 2026-10-01: folding T1235-D1's *true* distogram gives TM
   0.9999 and Cβ-lDDT 1.00.
7. One sentence the book owes: lDDT is a distance score and cannot see a mirror image;
   TM-score superposes with a proper rotation and can. A distance-geometry fold has that
   ambiguity, so the fold step runs both hands, picks by the sign of the i…i+3 dihedral over
   helical stretches, and writes both. Measured on T1235-D1's true distogram: the mirror
   scores Cβ-lDDT 1.00 and TM 0.34.
8. Headline on the date-cut-only list, purge as an ablation (decided 2026-10-01). The 82
   chains the purge drops are templates the CASP16 field was allowed to use — the "easy" class
   is defined by their existence — so the purged set is stricter than the field's conditions
   on 15 targets, and the date-cut-only set (`train_full.csv`, 26,310 chains) is exactly the
   information the field had. `train.csv` (26,228) becomes the "no templates" row of the
   table; both share `val.csv`.
9. Book shape: one figure of three panels (the object: the target's true contact map / fold;
   the check: predicted against true distogram for one target; the result: our dot on the
   CASP16 strip), one table (the ladder of §6). The seven-panel plate stays a design document.

## 2. Data on disk (`data/casp16/`, gitignored)

| Path | Contents | Source |
|---|---|---|
| `raw/casp16.T1.seq.txt` | 62 Phase-1 target sequences | download_area/CASP16/sequences/ |
| `raw/dom/*.pdb` | 159 domain-trimmed experimental structures | targets/casp16.targets_monomer_trimmed2domains.tgz |
| `raw/CASP16_prot_domains.scores.csv` | 50,157 rows: group × model × EU; GDT_TS, GDT_HA, LDDT, TMscore, RMS_CA, … | results/tables/ |
| `raw/domains_summary.html` | EU boundaries, length, difficulty, PDB id | casp16/domains_summary.cgi |
| `raw/groups.html` | group number → name, kind | casp16/docs.cgi?view=groupsbyname |
| `raw/predictions/T1235, T1267s1, T1226` | every group's five models for the featured targets | predictions/regular/ |
| `tools/USalign`, `tools/mmseqs/` | scorers; OpenStructure is the docker image | pylelab/USalign, mmseqs.com, scicore registry |
| `eu_list.csv`, `group_names.json` | derived by `casp16_score.py eus / groups` | |
| `train/entities.txt`, `train/entities.jsonl` | 230,942 pre-cutoff protein entities; sequence, chains, resolution, method, release date | RCSB search + data API, `casp16_chain_list.py search / fetch` |
| `train/clusters-by-entity-40.txt` | RCSB's 40 % identity clusters (324,163) | cdn.rcsb.org/resources/sequence/clusters |
| `train/reps.csv`, `train/reps.fasta` | 27,690 cluster representatives | `casp16_chain_list.py cluster` |
| `train/train_full.csv`, `train/train.csv`, `train/val.csv`, `train/target_hits.m8` | 26,310 date-cut-only (headline) / 26,228 purged (ablation) / 1,380 val; the MMseqs2 hits | `casp16_chain_list.py purge` |
| `pdb/<entity>.npz` | backbone + Cβ of the entity's first chain, keyed by label_seq_id (step 2) | RCSB CDN + gemmi, `casp16_fetch_chains.py` |
| `labels/<entity>.npz` | 66-class pair labels, observed mask, Cβ coordinates (step 4) | `casp16_labels.py` |
| `emb/<entity>.npy` | ESM-2 35M final-layer representation, f16 [L, 480] (step 5) | `casp16_embed.py` |
| `targets/<EU>.npz`, `targets/summary.csv` | the 84 scorable EUs: sequence, embedding, labels, ESM-2 contact head (step 6) | `casp16_targets.py` |

`scripts/datasets/download_casp16.sh` fetches all of it (idempotent, ~60 MB). The Python side
runs in `.venv-casp` (numpy, pandas, gemmi; torch-cpu + fair-esm to be added for step 5).

EU census: 85 EUs in the domain table (30 easy / 47 medium / 8 hard; length 37–1693, median
193; 31 with a public PDB id); 83 of them carry Phase-1 rows in the score table (T1214-D1 and
T1249v2-D1 were never scored), which is the plate's "all 83". The assessment paper's 74 is
Phase 1 at paper time. The 83 scored EUs are what we score against.

Featured EUs (model 1, official all-atom lDDT): T1235-D1 easy 106 aa, field median 0.87;
T1267s1-D1 medium 157 aa, median 0.65, MULTICOM best at 0.76; T1226-D1 hard 123 aa, median
0.37, ColabFold baseline 0.58 against AF3-server 0.36. Placeholders until the final pick (§9).

The same three at pseudo-Cβ (`casp16_score.py field <EU> --pseudo-cb`, model 1, our scoring
of the field; `data/casp16/work/field_<EU>_m1_cb.csv`), Cβ-lDDT / TM: T1235-D1 median
0.899 / 0.939, best 0.968 / 0.985 (AF3 0.882 / 0.919, MULTICOM 0.893 / 0.927, ColabFold
0.963 / 0.981); T1267s1-D1 median 0.671 / 0.757, best 0.785 / 0.895 (MULTICOM); T1226-D1
median 0.362 / 0.357, best 0.676 / 0.760 (ColabFold 0.596 / 0.668, AF3 0.357 / 0.358). On
the nine validation models Cβ-lDDT sits within 0.03 of the official all-atom lDDT and the
Cβ TM-score 0.01–0.02 under the official CA one.

## 3. Pipeline

```
 1  rcsb_query.py    RCSB search: released < 2024-05-01, X-ray/EM ≤ 3.0 Å, protein entities
                     40–512 aa; one representative per RCSB 40 % sequence cluster (the
                     precomputed clusters-by-entity-40.txt, best resolution wins) → chain list
 2  fetch_chains.py  per-chain coordinates (RCSB ModelServer, one chain per request; fallback
                     files.rcsb.org mmCIF.gz) → data/casp16/pdb/
 3  purge.sh         MMseqs2 easy-search: 85 EU sequences vs the chain list; drop any train
                     chain at ≥ 30 % identity over ≥ 50 % coverage to a target; val = 5 % of
                     the remaining clusters
 4  labels.py        per chain: Cβ (Cα for Gly), L×L distance → 64 bins over 2–22 Å + ">22"
                     + "unobserved" → uint8 .npz
 5  embed.py         ESM-2 35M (esm2_t12_35M_UR50D, frozen, CPU is enough) → f16 [L, 480]
                     per chain; the representations, not the model's contact head
 6  targets.py       the same labels and embeddings for the EUs from raw/dom/*.pdb, sliced by
                     eu_list.csv segments; ESM-2 on the full target sequence; the v2 EUs are the
                     v1 sequence in a second conformation; T1249v2-D1 has no structure → 84 EUs
 7  Lean trainer     demos/MainDistogramCasp.lean: pairTile → residualBlock stack → 1×1 head,
                     perPixelWeightedCE; random 64×64 crops anywhere in L×L
 8  predict          tiled inference (§1.4); contacts P(d < 8 Å) = Σ bins below 8 Å;
                     top-L/5 long-range precision (|i − j| ≥ 24)
 9  casp16_fold.py   pseudo-Cβ coordinates by Adam on the Gaussian-smoothed −log p_ij(d) +
                     chain (5.4 Å) + clash terms, classical-MDS start + random restarts, both
                     hands → PDB (one CA-named pseudo-atom per residue, target numbering)
10  casp16_score.py  `--pseudo-cb`: Cβ-lDDT / TM-score for ours and, with `field <EU>`, for
                     every group's model the same way; the official LDDT column stays the
                     field's all-atom number
```

Sizes (step 1 run 2026-10-01): 230,942 entities pass the search; 27,756 of RCSB's 40 % clusters
contain one, 27,690 representatives survive the unknown-residue filter (median length 214,
mean 233, 6.4 M residues; 90 % X-ray, median 1.9 Å; released 1979–2024-04-24). The purge
(MMseqs2 `-s 7.5`, ≥ 30 % identity over ≥ 50 % of the *train* chain, `--cov-mode 1`) finds 84
hits for 15 of the 62 targets — templates the field also had, e.g. 3BVF at 100 % over a third
of T1295, 7L9U at 53 % over 97 % of T1243 — and drops 82 representatives with their clusters;
5 % of the remaining clusters are val: 26,228 train / 1,380 val. G1: none of the 21 PDB entries
behind the EU table's PDB ids was released before the cutoff, so the date cut alone excludes
every target structure. Per-chain coordinates ~130 KB → ~3.6 GB; labels < 2 GB; embeddings
~6 GB at f16. 80 GB free after the 2026-10-01 cleanup.

## 4. The net

Route decided 2026-10-01 (Brett): `pairTile` as a new `Layer`, not a host-tiled 1×1 conv.
(The alternative, recorded for the ablation that may want it: outer-concat on the host + a
16-bin |i−j| one-hot = 994 channels into `conv2d 994 64 1`, the same function, 16 MB per crop.)
`Layer.pairTile seqLen inDim outDim`: the flat host input `[B, 2·L·D]` (the crop's i rows then
its j rows) → two weights `W`, `Wj` (`[D, C]`, He) → outer sum → `[B, C, L, L]`. No bias: the
`convBn` after it subtracts any per-channel constant, and a pairTile bias had gradient exactly
0 in the FD check. Backward = the two broadcast adjoints (reduce over j, over i) and the dense
weight rule; no input gradient. Touch points: `Types.lean` (constructor), `Spec.lean`
(`paramSlots` one group W + Wj, `archStr`, `outChannels`), `MlirCodegen.lean`
(`inputFlatDim` = 2·L·D, `emitPairTileForward` shared by eval and train, `fwdSigParts`,
`bnLayers` pidx +1, `FwdRec.{isPairTile, ptPidx, ptUSSA, ptVSSA}`, the backward arm), and
the header comment of `generateTrainStep` now stays on one line (a 66-weight `repr` wrapped
onto lines the MLIR parser read as code). Gradient check (`lake exe distogram-casp smoke`,
L = 8, D = 5, 4 ch, B = 2): Adam's first moment ×10 against central differences on W, Wj, a
body weight and the head bias — every coordinate within 2 % + 2e-4 (|diff| 1e-6 … 1.2e-4).
The relative-position signal rides on the per-residue features: `casp16_pack.py` appends
i/512 and four sin/cos pairs, and the outer sum is linear in both inputs, so the first layer
forms j − i directly. Zero-weight class: the demo's own loop calls `generateTrainStep`
directly, so `Train.lean`'s all-positive check is not on this path.

```
pairTile 64 489 64                 -- W, Wj: 489→64 on the i and j rows, outer sum → [B, 64, 64, 64]
convBn 64 64 1 1 .same             -- the stem's normalization (its β is the pair map's bias)
residualBlock 64 64 16 1           -- the chapter body, stride 1 throughout
conv2d 64 66 1 .same .identity     -- 66-class head, perPixelWeightedCE
```

`demos/MainDistogramCasp.lean` (`lake exe distogram-casp smoke | train | predict`), with the
crop gatherer, val metrics and the inference accumulator in `ffi/f32_helpers.c`
(`lean_casp_gather`, `lean_casp_val_metrics`, `lean_casp_accumulate`). Input per crop: two
per-residue blocks `[64, 489]` (ESM-2 480 + index scalar + 8 sinusoids) f32, gathered from the
f16 pool; labels `[64, 64]` int32. 1,254,850 parameters at 64 channels / 16 units; 155 ms per
step of 32 crops on one 4060 Ti (≈ 2 min per epoch of one crop per chain); 128 channels and
32 units are affordable (§8) and are the first ablation. Init loss 11.8 against ln 66 = 4.19
(sixteen residual units with γ = 1 inflate the head's logits); the first epoch's mean is 3.18.
Inference: every 64 × 64 window at stride 32, logits summed per target in C, averaged and
symmetrized by `scripts/demos/casp16_predict.py` (84 EUs, 10,703 windows, 55 s).

Proof item (done 2026-10-01): `LeanMlir/Proofs/Foundation/PairTile.lean` — `tileWHasVJP` and
`tileWjHasVJP`, the VJP witnesses of the pair map as a function of each weight (the other
block's term is the constant), by `pdiv_of_affine` in the shape of `pdiv_dense_W`; their
backwards `gradW` / `gradWj` are the cotangent summed over the broadcast axis then
contracted with the block — the two `reduce` + `dot_general` pairs the emitter writes. No
input gradient exists to prove: the input is the host's feature block. Everything downstream
of the tile is the chapter's. A whole-net step tie is not attempted; the demo's gradient
evidence is the FD smoke (§4 above) and the eval ≡ train identity.

Ablation rows the table wants (each one run): ESM-2's own contact head (no training by us;
measured 2026-10-01: top-L/5 long-range 0.533 easy / 0.478 medium / 0.233 hard, 0.474 over
the 84 EUs — the number the ResNet has to beat);
one-hot input instead of ESM-2 (what the ResNet alone can do without an MSA); the purged
training list (§1.8, no templates); 128 ch / 32 units.

## 5. Scoring (validated 2026-10-01)

Recipe, `scripts/demos/casp16_score.py`:
- prep: CASP files have a `PFRMAT` header and blank chain IDs, which OpenStructure refuses;
  rewrite ATOM-only, chain A, trimmed to the EU's residue ranges.
- lDDT (all-atom, the official number): `ost compare-structures --lddt` (OST 2.12.0 in docker;
  CASP16 used 2.9).
- CA-lDDT (our headline): `--bb-lddt`, which OST computes on CA atoms only; `lddt()` in the
  script is a 20-line numpy lDDT that equals it to three decimals on all nine models below.
- TM-score: `USalign model ref -TMscore 1` (residue-index superposition, normalized by the
  target length) — this, not TM-align, is CASP's TMscore column; TM-align runs 0.03 high on
  poor models.
- GDT_TS: OST `--rigid-scores` (`oligo_gdtts` × 100); within one point of LGA's, labelled "not LGA".

| model | lDDT ours / official | CA-lDDT | TM ours / official | GDT_TS ours / official |
|---|---|---|---|---|
| T1235TS304_1-D1 (AF3-server) | 0.842 / 0.842 | 0.905 | 0.930 / 0.930 | 92.9 / 92.92 |
| T1235TS051_1-D1 (MULTICOM) | 0.854 / 0.854 | 0.913 | 0.936 / 0.936 | 94.1 / 93.87 |
| T1235TS145_1-D1 (ColabFold) | 0.934 / 0.934 | 0.978 | 0.987 / 0.987 | 99.5 / 99.53 |
| T1267s1TS304_1-D1 | 0.645 / 0.645 | 0.714 | 0.774 / 0.774 | 72.2 / 72.89 |
| T1267s1TS051_1-D1 | 0.755 / 0.755 | 0.834 | 0.916 / 0.916 | 87.5 / 87.83 |
| T1267s1TS145_1-D1 | 0.645 / 0.645 | 0.714 | 0.772 / 0.772 | 72.1 / 73.05 |
| T1226TS304_1-D1 | 0.363 / 0.363 | 0.388 | 0.378 / 0.378 | 34.2 / 34.43 |
| T1226TS051_1-D1 | 0.361 / 0.361 | 0.388 | 0.378 / 0.378 | 34.2 / 34.22 |
| T1226TS145_1-D1 | 0.577 / 0.577 | 0.643 | 0.695 / 0.695 | 62.9 / 62.70 |

lDDT, CA-lDDT and TM-score agree to the printed digit in all nine cases; GDT_TS agrees within
one point (OST's rigid superposition is not LGA's search).

## 6. Figure and table

Figure (three panels): (a) the featured target as the reader meets it — its sequence and the
experimental contact map; (b) predicted distogram against truth for that target (expected
distance map, two pair histograms); (c) the CASP16 strip for the three EUs with every group's
model 1 at CA-lDDT and our dot. Table: per EU, top-L/5 long-range precision, CA-lDDT,
TM-score, GDT_TS for ours; field median and best; the three ablation rows of §4.

## 7. Gates

- G0 (done): scoring reproduces the official table on nine models (§5).
- G1: step 1 returns a chain list of plausible size and the purge removes every chain
  homologous to a CASP16 EU (spot-check the 31 EUs with PDB ids: none in train).
- G2 (done 2026-10-01): FD gradient check on `pairTile` passes (§4); the one-epoch probe runs
  train → val → checkpoint → predict → assemble end to end; loss 11.8 → 3.18, val CE 2.54.
  Added after the label-offset bug: the packed pools are checked against the per-chain files
  (300 random chains, all targets) before any run, and a 64-chain memorization probe must
  reach high train-set contact precision — a plateau at the separation prior (loss ≈ 2.47,
  precision ≈ 5 %) means the labels are not the features' labels.
- G3: full training; top-L/5 long-range precision on val reported before any CASP target is
  touched.
- G4: fold + score on the 85 EUs; strip plot; the field rescored at CA-lDDT on the featured
  EUs.

## 8. Budget

GPU: ~29 GFLOP per crop forward+backward at 64 ch; ~90 s per epoch over 30k chains on one
4060 Ti, so 100 epochs is an afternoon and the 128-ch / 32-unit arm an evening. ESM-2 35M
embedding on CPU ~1 h. Disk ~15 GB (§3). Downloads are the long pole (hours, unattended).
Every launch is asked for first.

## 9. Open questions

- Final featured EUs: the three placeholders tell a good story; confirm after G4 with the
  full 85-EU sweep (and keep ≤ 170 aa so the plate's maps are legible).
- CASP14 rematch (Brett's course-era predictions, DeepDist 5th of 30 servers): the same
  pipeline at cutoff 2020-05-18 against `CASP14/predictions/contacts/Contacts.ALL.tar`
  (1.85 GB). Second, once the CASP16 run lands.
- Whether `lddt()` becomes a Lean function in the book (a verified scorer is a natural coda).
- CASP17 after the December 2026 conference.

## 10. Work log

- 2026-10-01: feasibility pass; data downloaded; US-align built, MMseqs2 unpacked, OST image
  pulled; `casp16_score.py` + `download_casp16.sh` written; G0 passed (§5 table); committed
  bc09c154.
- 2026-10-01: `casp16_chain_list.py` search / fetch (230,942 entities, 18 min) / cluster /
  purge → 26,228 / 1,380; G1 passed; `casp16_fetch_chains.py` rewritten CDN + gemmi (4/s per
  core; ModelServer alone was 0.9/s), 200-chain smoke, label_seq alignment check 12 / 37,672;
  decided headline = date cut only, purge = ablation; full fetch launched; committed 3485efa9.
- 2026-10-01 (night): Brett picked `pairTile`; layer + emitters + FD smoke; `casp16_pack.py`
  (8.2 GB in 19 s), the demo trainer / predictor, `casp16_predict.py`; one-epoch probe end to
  end (155 ms/step). G2 passed; committed fcff9ce1. Four 30-epoch runs launched (GPU 0
  headline train_full; 1 purged list; 2 seed 2; 3 128 channels). `casp16_fold.py` written;
  `casp16_score.py --pseudo-cb`; truth-fold check passes (TM 0.9999 / mirror 0.34).
- 2026-10-01 (late): the four runs sat at loss 2.47 / chance precision through epoch 3 — a
  packing bug: `casp16_labels.py` sized each matrix from reps.csv's `length`
  (rcsb_sample_sequence_length), the packer from the embedding (len(seq)); 19 entities
  differ (8Q79_1: 234 vs 236), the first at pool position 444, and every later chain read
  shifted labels (296 of 300 sampled). Labels now sized from the sequence, the packer
  asserts L×L, both pools re-verified (0 of 300; targets 0 of 84). Runs restarted ~21:50 with
  a 64-chain memorization probe alongside (`val=tiny`). The smoke now also checks eval ≡
  train (the eval forward's masked CE at one batch's BN statistics equals that step's loss,
  5.210230 both). On the fixed pool the restarted runs learn from the first epoch: val
  top-L/5 long-range precision 19.0 % after epoch 1 (purged and seed-2 arms alike; the
  ESM-2 head's number on the EUs is 47 %), val CE 2.90; epoch 3: 24.5–24.8 %, loss 2.28.
  `Proofs/Foundation/PairTile.lean` (two HasVJP witnesses) builds clean, registered in the
  Proofs and Certs roots; committed c46f5a81.
- 2026-10-01 22:16: the three 64-ch arms finished at val top-L/5 LR 35.2 / 35.6 / 35.7 %
  (headline / purged / seed 2). On the 84 EUs: top-L/5 long-range precision 0.585 / 0.599 /
  0.597 against the ESM-2 head's 0.474 (easy 0.63 vs 0.53, medium 0.62–0.64 vs 0.48, hard
  0.23–0.27 vs 0.23). First folds were weak (T1235-D1 Cβ-lDDT 0.32 / TM 0.37, T1267s1-D1
  0.29 / 0.33, T1226-D1 0.16 / 0.15): no reference state, and a hand rule blind to all-β
  folds. Fixed both — `casp16_fold.py` subtracts the per-separation background of the
  training labels (P(far) 0.00 / 0.14 / 0.43 / 0.81 at separation 1 / 8 / 24 / >128), and the
  chirality score weights helical quads + and extended quads − (84/84 true traces positive,
  min +0.39) — giving 0.388 / 0.391, 0.450 / 0.477, 0.346 / 0.230 with every mirror lower.
  On the hard EU our 0.346 is the field's pseudo-Cβ median (0.362). 128-ch arm at epoch 15:
  35.9 % (64-ch at 15: 33.7 %). The purged and seed-2 arms fold the same three to 0.396 /
  0.415, 0.435 / 0.424, 0.346 / 0.250 and 0.401 / 0.414, 0.442 / 0.472, 0.349 / 0.274 — the
  fold numbers are stable to ±0.01 lDDT across seeds and the 82 purged chains.
- 2026-10-02 00:02: the 128-ch arm finished at val 37.4 % (64-ch: 35.2–35.7); on the 84 EUs
  top-L/5 long-range 0.629 (easy 0.668 / medium 0.667 / hard 0.261) against 0.585–0.599 for
  64 ch and 0.474 for the ESM-2 head — capacity matters, so the book run is 128 ch. CPU
  folding was the bottleneck (10 of 78 EUs in 90 min); `casp16_fold.py` gained `--device`
  (CUDA torch in .venv-casp) and resume. Fold hyperparameters benched OFF the test set
  (`casp16_valfold.py`: 24 val chains of 80–191 residues packed as `valsub`, predicted with the
  64-ch headline, folded against their own true pseudo-Cβ traces): reference state on/off =
  0.518 / 0.415 Cβ-lDDT and 0.477 / 0.387 TM; chain weight 0.3–3, clash 1–10 and σ × 2 all
  within ±0.01 — the defaults stand. The 128-ch featured folds (0.403 / 0.423, 0.459 / 0.454,
  0.339 / 0.275) sit within ±0.02 of the 64-ch ones: the fold step, not the distogram, is the
  limit now. Book run launched 00:20: 128 ch × 100 epochs on GPU 3
  (runs/2026-10-02-distogram-r16x128-e100/, ~9 h).
- 2026-10-01: `casp16_labels.py` (183 chains/s; adjacent Cβ–Cβ 5.39 Å, 4.0 % of pairs < 8 Å,
  93 % residues observed), `casp16_embed.py` (ESM-2 35M on CPU, ~1,700 residues/s at 16
  threads), `casp16_targets.py` (84 EUs, 0 residue-name mismatches, 98 % observed; ESM-2
  contact-head baseline above). Full embedding run launched. Fetch finished (27,690, 0 failed,
  40 min); labels for all 27,690 in 145 s (652 MB); `casp16_pack.py` written; code read for
  the input route (see §4).
