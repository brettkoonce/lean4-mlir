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

Status 2026-10-02 18:40: the whole pipeline runs and the ablation table (§10) is filled
(§11 is the handoff: what is running, where, and what to do when it finishes; §10a the night's
findings). Data, `pairTile` + its VJP lemma, trainer, fold, scoring all landed (commits bc09c154
… 77254aaa); the night's additions (orientation heads, the fold's angular restraints, the
scoring and table scripts, the 150M / 650M feature sets) are in the working tree. The language
model is the lever: at 64 ch × 30 epochs the 84-EU top-L/5 long-range precision runs 35M 0.585
→ 150M 0.735 → **650M 0.838** (each LM's own contact head: 0.474 / 0.623 / 0.754, so the
verified ResNet adds +0.08–0.11 on every LM); the fold follows, Cβ-lDDT / TM 0.423 / 0.389 →
0.512 / 0.498 → 0.569 / 0.572 over 78 EUs. Against the CASP16 field our TM still sits at the
~2nd percentile per EU (field medians 0.945 easy / 0.883 medium / 0.656 hard): the gap to the
AF3-era groups is not a contact-precision gap. The orientation heads leave the distance head
unchanged and their ω/φ restraints lift the fold by +0.02 on 73 of 78 EUs and halve wrong-hand
picks. At 650M each of width (128 ch: 0.848, fold 0.590 / 0.590), crop 96 (0.850, 0.583 / 0.589) and
100 epochs (0.847, 0.587 / 0.588) is worth +0.01 on the EUs and +0.02 on the fold, against a seed
gap of 0.002 / 0.001, and width and crop add: 128 ch × crop 96 is **0.858** (0.866 / 0.885 / 0.674)
with the plain fold at **0.600 / 0.598**, the best single arm on every column; ensembles of these
heads add nothing (the LM bounds the map). The book run as launched (128 ch × 100 ep, 35M: 0.647,
fold 0.466 / 0.443) finished at 09:27 and is the wrong feature set — the 650M book run is 128 ch ×
crop 96 × 100 ep with `orient=1`, ~20 h on one card, for Brett to schedule (§11, §11a item 6). Figure:
`scripts/demos/casp16_figure.py` → https://claude.ai/artifact/CyZfnLRWnjkpByMqdHyzTo.

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
  full 85-EU sweep (and keep ≤ 170 aa so the plate's maps are legible). After the night's 650M
  arm (06:30): T1226-D1 is out (our precision 0.00 on it at every LM; its fold score is the
  prior's), and T1235-D1 is our weakest easy EU (P@L/5 0.71 where most easy EUs are > 0.9; fold
  TM 0.38 against the field's 0.95 — an interesting failure, not a representative one).
  Candidates at ≤ 170 aa from `casp16_table.py --field`: easy T1271s6-D1 (93 aa, P 1.00, TM
  0.821 vs field median 0.954); medium T1295-D3 (156 aa, P 1.00, TM 0.879 vs 0.942) or
  T1267s1-D1 (157 aa, P 0.97, TM 0.734 vs 0.775); hard T1228v1-D3 (169 aa, P 0.91 — 0.44 at
  35M — TM 0.644 vs 0.751). Brett's call.
- CASP14 rematch (Brett's course-era predictions, DeepDist 5th of 30 servers): the same
  pipeline at cutoff 2020-05-18 against `CASP14/predictions/contacts/Contacts.ALL.tar`
  (1.85 GB). Second, once the CASP16 run lands.
- Whether `lddt()` becomes a Lean function in the book (a verified scorer is a natural coda).
- CASP17 after the December 2026 conference.
- The MSA route (§11a, parked 2026-10-04): go or no-go after its half-day probe; a source for the alignments of
  T1214, T1228 and T1239, which MassiveFold's release does not cover.

## 10. The ablation table (decided 2026-10-02 with Brett: "poke at it a few different ways")

One change per row; columns val top-L/5 LR · 84-EU top-L/5 LR (easy / medium / hard) · mean
fold Cβ-lDDT / TM over the folded EUs · GPU-hours. Rows and their state:

| row | isolates | state |
|---|---|---|
| ESM-2 35M contact head | the LM without us | done: 0.474 |
| ESMFold v1 (`scripts/demos/casp16_esmfold.py`) | the same LM (ESM-2 3B) under its published folding trunk and structure module: the single-sequence reference for everything downstream of the LM | done 2026-10-04: fold **0.751 / 0.778** over the 78 against the book run's 0.627 / 0.646 — paired TM +0.132 [+0.101, +0.165], better on 71 of 78. On the 73 units folded from the full target sequence 0.770 against 0.647, +0.123 [+0.093, +0.157]; T1218 and T1269 (over 1,000 residues) do not fit a 16 GB card at any chunk size, and their five units are folded from an 800-residue window of the target (a sensitivity run outside the script: 0.84 / 0.91 / 0.91 / 0.88 / 0.93) |
| one-hot residues, no LM (`fs=onehot`, dim 30) | the ResNet without the LM | done: val 17.1 %, EUs 0.252 (0.27 / 0.24 / 0.17), fold 0.319 / 0.228, hand a coin flip (39 of 78 mirrors better) |
| 64 ch · headline / purged / seed 2 | noise, templates | done: 0.585 / 0.599 / 0.597 |
| 128 ch · 30 ep | capacity | done: 0.629 |
| 128 ch · 100 ep | schedule length — the book run as launched | done 09:27 (9.2 h): val 39.0 %, EUs 0.647 (0.710 / 0.672 / 0.271), fold 0.466 / 0.443 over 78 EUs, 11 wrong hands — +0.018 EUs over 30 ep at the same width; the wrong LM for the book (650M is 0.838 at 64 ch × 30 ep) |
| crop 96 (`crop=96 batch=16`) | long-range context beyond 64 | done: EUs 0.617 (crop 64: 0.585); val numbers at crop 96 are over a different pair set — only the 84-EU line compares |
| ESM-2 150M (`fs=esm150`, dim 649) | LM scale | done: val 45.5 %, EUs **0.735** (0.767 / 0.762 / 0.463), fold **0.512 / 0.498** over 78 EUs, 6 wrong hands; the 150M head alone is 0.623, so the net adds +0.11 on either LM; field TM percentile still 0.02 |
| ESM-2 150M × 128 ch | both levers | done: val 46.8 %, EUs 0.742 (0.78 / 0.77 / 0.48), fold 0.528 / 0.504 — width is worth +0.007 once the LM is strong (4.9 M params) |
| ESM-2 150M, seed 2 | the row's error bar | dropped in favour of the 650M chain (queue04 stopped it at launch) |
| crops = 3 per chain per epoch | data passes | done: val 37.0 %, EUs 0.598 (one crop: 0.585), fold 0.446 / 0.419 (0.423 / 0.389) — three passes buy +0.013 precision, less than 128 ch (+0.044) or crop 96 (+0.032) |
| ESM-2 650M (`fs=esm650`, dim 1289) | LM scale, next point | Brett 04:10 "let's do the 650m model": done — val 51.5 %, EUs **0.838** (0.851 / 0.868 / 0.620), fold **0.569 / 0.572** (median TM 0.574), 4 wrong hands; the 650M head alone 0.754 (median 0.832); embeddings streamed into the packed pool (`casp16_embed.py --pool-out`, 1,644 s on one card, 16.6 GB), EUs on CPU |
| orientation heads × 650M | do the angular heads sharpen with a strong LM, and the ω/φ fold gain with them | done: distance head unchanged (EUs 0.838 = the plain 650M arm); heads sharper (ω/θ/φ ±1 bin 0.27 / 0.41 / 0.58 vs 0.18 / 0.26 / 0.47 at 35M); ω/φ fold 0.574 / 0.591 vs plain 0.564 / 0.577 — +0.010 lDDT on 66 of 78, +0.014 TM on 60, wrong hands 4 → 2; smaller than at 35M (+0.019 / +0.022): the better distogram already resolves most of what the angles add |
| crop 96 × 650M | long-range context on the strong LM | done 11:00: EUs **0.850** (0.854 / 0.880 / 0.660), fold 0.583 / 0.589, 3 wrong hands — +0.012 EUs, +0.014 / +0.017 fold over the 64-crop arm |
| 650M × 64 ch × 100 ep | the book-run candidate on the right LM (schedule length at 650M) | done 12:23 (3.9 h): val 52.5 %, EUs 0.847 (0.854 / 0.873 / **0.677**), fold **0.587 / 0.588**, 2 wrong hands — +0.009 EUs, +0.018 / +0.016 fold over 30 ep |
| 650M × 128 ch × 30 ep | width at the top LM | done 11:37 (2.8 h): val 52.8 %, EUs 0.848 (0.856 / 0.879 / 0.641), fold **0.590 / 0.590**, 4 wrong hands — +0.010 EUs, +0.021 / +0.018 fold: the largest single step after the LM |
| 650M, seed 2 | the error bar at the top LM | done 12:07: val 51.3 %, EUs 0.836 (0.849 / 0.863 / 0.632), fold 0.570 / 0.571 — seed noise 0.002 on the EUs, 0.001 on the fold; every lever above is 5–20× it |
| ensembles (`casp16_ensemble.py`) | do differently-trained heads on one LM add information | 650M plain + orient: EUs 0.842 (members 0.838 / 0.838), fold 0.569 / 0.575 — nothing; ω/φ fold 0.579 / 0.592 vs 0.574 / 0.591. Five members (seeds 1–2, 128 ch, crop 96, 100 ep): EUs 0.852 vs the best member's 0.850, fold running GPU 3 (queue12.sh). The heads agree on what they miss: the LM bounds the map |
| crop 96 × 650M × orientation heads | do the crop gain and the ω/φ fold gain stack | trained 13:41 (2.6 h): EUs 0.850 (= the plain crop-96 arm, the distance head unchanged a third time); plain fold 0.576 / 0.586 (plain crop 96: 0.583 / 0.589, within the arms' spread); **ω/φ fold 0.585 / 0.606** (TM median 0.644), wrong hands 2 — the best fold of the table: the angular restraints add +0.009 / +0.020 on top of crop 96, so the two gains stack |
| 650M × 128 ch × crop 96 | do width and crop stack (the config a 650M book run would use) | done 18:09 (6.1 h train): EUs **0.858** (0.866 / 0.885 / 0.674), fold **0.600 / 0.598** (TM median 0.626), 7 wrong hands — +0.020 EUs, +0.031 / +0.026 fold over the 64-ch arm: the two levers add (128 ch alone +0.010 / +0.021 / +0.018, crop 96 alone +0.011 / +0.014 / +0.017); the best single arm on every column, and its plain fold beats the crop 96 × orient ω/φ fold on lDDT (0.600 vs 0.585) while trailing it on TM (0.598 vs 0.606) — the book run takes both |
| 650M × 64 ch, `pair=1` (the 650M contact head's logit plane as a pair-tile channel; §11a item 1, tier A) | the pair-input op's first run: does the LM's own contact map, fed as a channel, lift the ResNet above what it learns from the embeddings | done 22:28 (4,199 s train — the plane costs nothing per step): val 52.7 % (48.2 % after one epoch; the baseline needed six to reach 47.0 %), EUs **0.856** (0.858 / 0.880 / **0.710**), fold **0.584 / 0.582** (medians 0.610 / 0.621), 8 wrong hands — +0.018 EUs (+0.090 on the hard class), +0.015 / +0.010 fold over the baseline; at 64 ch and 70 min it matches the 128 ch × crop 96 arm's 0.858 (6 h). The lever the plan ranked first |
| 3B × 64 ch, `pair=1` (the 3B contact head's plane; queue16) | do the two levers stack | done 23:49 (4,646 s): val **54.3 %**, EUs **0.866** (0.876 / 0.885 / **0.724**), fold **0.603 / 0.612** (medians 0.618 / 0.684), 7 wrong hands — the best arm on every column, above 128 ch × crop 96 (0.858 / 0.600 / 0.598) at a fifth of the training time: 0.838 → +plane 0.856 → +3B 0.866 |
| 650M × 64 ch, `pair=1 orient=1` (queue18) | do the plane and the ω/φ fold stack | done 00:23: EUs 0.849 (0.855 / — / 0.678) against the plane alone's 0.856 (the heads cost the distance head 0.007 here, where without the plane they cost nothing); plain fold 0.577 / 0.585, **ω/φ fold 0.588 / 0.605** (medians 0.611 / 0.663) — +0.011 / +0.020 over its plain fold, and against the plane alone +0.004 lDDT / **+0.023 TM**: the heads remain a TM lever through the fold |
| 3B × 64 ch, `pair=1 orient=1` (queue19) | the book run's heads on the best arm | EUs **0.864** (0.876 / 0.880 / 0.722) against 3B + plane's 0.866 — the heads cost 0.002 at 3B (seed-noise level; 0.007 at 650M); distance head 54.2 % val = 3B + plane's 54.3 %; plain fold 0.594 / 0.611 (4 wrong hands), **ω/φ fold 0.602 / 0.628** (medians 0.619 / 0.692, 2 wrong hands) — the best TM of any arm: against 3B + plane the heads are −0.002 / −0.001 / **+0.016 TM**. `orient=1` stays in the book run |
| 3B × 64 ch, `pair=1`, seed 2 (queue20) | the error bar on the best arm | done 02:52: EUs **0.866**, fold 0.605 / 0.614 against seed 1's 0.866 / 0.603 / 0.612 — a seed gap of 0.000 / 0.002 / 0.002, the 650M arm's size; every lever above is 5–10× it |
| 3B × 128 ch × crop 96 (queue14's second arm) | 3B at the wide config, without the plane | trained 03:00 (22,972 s): val 75.1 % against the 650M twin's 72.9 %, EUs **0.866** (0.884 / 0.884 / 0.696) against 0.858 — at the wide config 3B is worth +0.008 on precision (+0.002 at 64 ch), and it ties 3B + plane at 64 ch (0.866, 77 min); fold **0.609 / 0.611** (medians 0.629 / 0.666, 7 wrong hands) against 0.600 / 0.598 — +0.009 / +0.013 |
| **3B × 128 ch × crop 96, `pair=1`** (queue17) | the book-run config at 30 epochs, without the heads | trained 06:16 (23,104 s): val **75.8 %** (3B wide without the plane 75.1 %, 650M wide 72.9 %), EUs **0.880** (0.880 / **0.903** / **0.746**) — +0.014 over 3B wide, +0.022 over 650M wide (0.858), +0.042 over the baseline; fold **0.622 / 0.632** (medians 0.640 / 0.690, **1 wrong hand**) — the best fold of any arm, plain or ω/φ. **Best on every column: 0.880 / 0.622 / 0.632** |
| 650M × 128 ch × crop 96, `pair=1` (queue21) | the 650M twin of queue17: does the wide config need 3B? | trained 08:33 (21,893 s): val 74.0 %, EUs **0.865** (0.871 / 0.892 / 0.693) — +0.007 over 650M wide (0.858), 0.015 under 3B wide + plane (0.880): at the wide config the plane is +0.007, 3B +0.008, both +0.022 — additive, and the book run needs the 3B features; fold 0.607 / 0.605 (medians 0.622 / 0.651) against 0.600 / 0.598 and 3B wide + plane's 0.622 / 0.632 |
| **3B × 128 ch × crop 96, `pair=1 orient=1`** (queue22) | **the book-run config at 30 epochs** | trained 09:44 (22,650 s): distance head **76.0 %** val (the highest of any arm; queue17 without the heads 75.8 %), EUs **0.877** (0.880 / 0.895 / **0.764**) against queue17's 0.880 (0.880 / 0.903 / 0.746) — the heads −0.003 overall, within noise, and the best hard-class number; plain fold 0.621 / 0.630 (= queue17's), **ω/φ fold 0.628 / 0.645** (medians 0.639 / 0.698, 1 wrong hand) — the best fold of any arm. **The book-run config: 0.877 / 0.628 / 0.645 against the baseline's 0.838 / 0.569 / 0.572** |
| 3B × 64 ch × 100 ep, `pair=1` (queue23) | schedule length with the plane (the book run's 100 epochs) | trained 11:01 (15,523 s): val 54.8 % against 54.3 % at 30 ep, EUs **0.868** (0.870 / 0.889 / 0.739) against 0.866 — +0.002: with the plane the long schedule is nearly flat on precision (+0.009 at 650M without it); fold 0.612 / 0.617 (medians 0.630 / 0.678) against 0.603 / 0.612 — +0.009 / +0.005, half the 650M schedule gain. 100 epochs with the plane: +0.002 / +0.009 / +0.005 |
| ESM-2 3B × 64 ch (`fs=esm3b`, dim 2569; §11a item 3) | the ladder's next rung after 35M 0.585 → 150M 0.735 → 650M 0.838 | done 22:19 (4,632 s train; the pool 33.1 GB in fp16, 7 min over four cards): val 53.0 % (+1.5 pt from epoch 10 on; val CE 2.110 vs 2.138), EUs **0.840** (0.872 / 0.849 / 0.664) — a tie with 0.838 — fold **0.581 / 0.583** (+0.012 / +0.011), 3 wrong hands. The LM step shows in the calibration, not the ranking: the ladder is flat on precision past 650M |
| 650M, purged list (`list=train`) | templates at the top LM | done 14:11: EUs 0.841 (vs 0.838 with the 82 template chains), fold 0.569 / 0.567 (vs 0.569 / 0.572), 7 wrong hands — templates in the training set are worth nothing at 650M, as at 35M (0.599 vs 0.585) |
| fold: steps × lr on 200–500-residue val chains | the fold's ceiling on the chains it fails | done (650M arm, restarts 0, GPU): long chains 0.629 / 0.708 at 1500 steps, 0.629 / 0.709 at 4000; lr 0.2 / 0.5 / 1.0 → 0.627 / 0.629 / 0.630; short chains 0.617 / 0.593, 4000 steps 0.617 / 0.588, lr 0.2 0.621 / 0.603 — the optimizer is converged; and the long val chains fold BETTER than the short ones, so the EU failures are not a length effect |
| fold: no reference state / confidence weighting | the fold's levers | reference state done (0.415 → 0.518); confidence weighting closed by the energy-gap diagnostic (§10a): under every re-weighting the true trace scores worse than our fold, so no weighting of this distogram reaches it |
| orientation heads (`orient=1`, below) | the AlphaFold-1 / trRosetta angular restraints | done: distance head unchanged (EUs 0.595 vs 0.585, plain fold 0.427 / 0.400 vs 0.423 / 0.389); heads weak but calibrated (ω/θ/φ ±1 bin on contacts 0.24 / 0.27 / 0.46, chance 0.125 / 0.125 / 0.25; 89–96 % right where > 0.5 confident); **the ω/φ fold: 0.446 / 0.422, +0.019 lDDT on 73 of 78 EUs, +0.022 TM on 62, wrong hands 15 → 7** (val bench 0.519 / 0.487 / hand 96 % vs 0.498 / 0.444 / 83 %) |

Columns added 2026-10-04 (`casp16_table.py`, cached per arm in `<dir>/map_scores.csv`): **map lDDT** (the lDDT of
the distogram's own mean distances, over the pairs `lddt()` scores), **recall** (true long-range contacts given
P > 0.5) and **top-L** long-range precision, all over the 84 units. Book run 0.605 / 0.476 / 0.623; the 650M × 64 ch
baseline 0.549 / 0.387 / 0.557; one-hot 0.327 / 0.001 / 0.135. Top-L/5 precision is at 0.95 or above on 54 of the 78
folded units, so it no longer separates arms; these three do. Rows for the 650M and 3B contact heads (0.754, 0.759).

Not yet in the table: ESM-2 attention maps as pair channels (the largest known lever; needs a
second host input into the pair map — a day of codegen).

### 10a. The night of 10-01/02: findings on the 30-epoch arms, and the orientation heads

Precision at several depths and ranges, 84 EUs, from the finished arms (`scratchpad
analyze_preds.py`; the 150M head from `casp16_targets.py --model esm2_t30_150M_UR50D`):

| arm | L/5 LR | L/2 LR | L LR | L/5 medium (12–23) | easy / medium / hard |
|---|---|---|---|---|---|
| ESM-2 35M head | 0.474 | 0.355 | 0.267 | 0.428 | 0.533 / 0.478 / 0.233 |
| ESM-2 150M head | 0.623 | 0.504 | 0.377 | 0.569 | 0.664 / 0.636 / 0.394 |
| 64 ch (35M) | 0.585 | 0.468 | 0.356 | 0.601 | 0.629 / 0.620 / 0.225 |
| 128 ch (35M) | 0.629 | 0.508 | 0.387 | 0.641 | 0.668 / 0.667 / 0.261 |
| ensemble of the four arms | 0.624 | 0.496 | 0.379 | 0.620 | 0.663 / 0.658 / 0.282 |

- By separation band the 128-ch net beats the 150M head only at |i − j| < 48 (0.583 vs 0.552)
  and trails it in every longer band (0.119 vs 0.148 at 128–256); by length it wins under 100
  residues (0.77 vs 0.49) and loses over 400 (0.51 vs 0.60). The 64-crop is the ceiling.
- Ensembling the arms does not beat the 128-ch arm. P(contact) is calibrated below 0.3 and
  under-confident above it (predicted 0.55 → empirical 0.65, 0.75 → 0.83).
- T1226-D1, the featured hard EU: every arm is at 0.00–0.04 top-L/5 precision there (ESM-2
  head 0.12). Its fold score, 0.346 against the field's 0.362, is what a compact chain + clash
  prior scores on that fold and is not evidence of a prediction; the featured-EU pick must not
  lean on it.
- Fold restarts (`casp16_valfold.py`, 24 val chains): MDS start alone 0.513 / 0.464, two
  restarts 0.517 / 0.476 Cβ-lDDT / TM — restarts are worth ≤ 0.01. `casp16_fold.py` now folds
  the starts × hands copies as one batch (the step was launch-bound): 4–5× faster on a shared
  card, same energies.
- `scripts/demos/casp16_fold_score.py <dir>` scores every fold and its mirror (Cβ-lDDT / TM;
  mirror TM above the fold's = a wrong-hand pick); `casp16_table.py` builds this section's table
  from disk; `runs/2026-10-02-distogram-ablations/finish_run.sh <gpu> <predict args>` runs
  predict → assemble → fold → score for a finished run.

**The fold is the bottleneck.** At 650M, 11 of 78 EUs have top-L/5 long-range precision
above 0.85 yet fold to TM < 0.5 (T1218-D2, 370 aa, P 1.00 → TM 0.37; T1299-D1, 168 aa, P 1.00
→ TM 0.43; T1212-D1, 466 aa, P 0.99 → 0.38): a near-perfect contact map that the distance-only
Adam fold does not realize. Diagnosed 08:30: (i) on a 200–500-residue val subset (`valsub_long`,
24 chains, 650M arm) 4,000 steps and learning rates 0.2 / 0.5 / 1.0 change nothing (0.629 /
0.708 → 0.629 / 0.709; the long chains fold better than the 80–200 ones, 0.617 / 0.593), so the
optimizer is converged and the failures are not a length effect; (ii) under the fold's own
potential the TRUE pseudo-Cβ trace has higher energy than the fold on 78 of 78 EUs (gap 7–34 per
residue), and no cheap re-weighting changes that — without the reference state the truth wins on
19 of 78, weighting pairs by P(d < 22 Å) on 0, both together on 19 (scratchpad
`energy_gap{,_variants}.py`). The fold is finding the minimum of what we predict; the ceiling is
the distogram beyond its top-L/5 contacts (the mid-range and "far" mass), and the orientation
restraints are the one lever that moved it (+0.02 at 35M, +0.01 at 650M). The field's medians
(TM 0.945 easy) say the structure is recoverable from much less than a 650M distogram — with a
real folding engine and MSAs.

**Orientation heads — built.** trRosetta's three per-pair heads as planned below, as one
`perPixelMultiCE` loss (`LeanMlir/Types.lean`, the `.multiCE` arm of `emitSegLossBlock`): K
weighted per-pixel CE heads over one `[B, Σ NC_k, H, W]` output, the int32 label carrying the K
labels mixed-radix (`y = d + 66·(ω + 26·(θ + 26·φ))`), so the segmentation ABI and the
`perPixelWeightedCE` emitter are unchanged (it gained an SSA-prefix argument; the default output
is byte-identical). Labels: `casp16_labels.py --orient` (ω, θ, φ planes for all 27,650 chains in
211 s: 27 % of pairs binned, 58 % beyond 20 Å, 15 % unobserved; Cβ deposited where present,
virtual from N–Cα–C otherwise — the distance labels keep Cα for glycine), `casp16_targets.py
--orient` for the EUs, `casp16_pack.py --orient-only` → `pool_orient.bin` (5.67 GB, three bytes
per label byte). `lean_casp_gather_orient` packs the label, `lean_casp_val_metrics` takes the
distance head's width. `lake exe distogram-casp smoke` runs both heads: the plain one unchanged
(eval ≡ train, 5.210230), the four-head gradient within tolerance on 14 coordinates including a
bias in each head. `casp16_fold.py --orient` adds the ω and φ restraints (θ needs N, not folded):
Cα is folded as a unit direction from Cβ at 1.53 Å, a Cα–Cα 3.8 Å term joins the chain term,
each angle head is read through a one-bin Gaussian (wrapped for ω) against its own
per-separation reference state; ω is chiral, so the hand is the lower-energy one. On
T1235-D1's true labels the orientation fold recovers the structure (Cβ-lDDT 1.00 / TM 1.00) and
the wrong hand is excluded by energy (+23,000). `casp16_valfold.py --orient` benches it on val.

**Orientation / torsion heads — the design (Brett: "that was key in AF2").** trRosetta's
three per-pair orientation heads next to the distance head: ω (dihedral Cα_i–Cβ_i–Cβ_j–Cα_j,
24 bins + "no contact"), θ (dihedral N_i–Cα_i–Cβ_i–Cβ_j, 24 + 1, asymmetric), φ (angle
Cα_i–Cβ_i–Cβ_j, 12 + 1, asymmetric), defined only for pairs with Cβ–Cβ < 20 Å. Labels are
free: `pdb/<entity>.npz` already holds N, CA, C (and Cβ, or the virtual Cβ from N, CA, C by the
standard formula) per residue. What it needs: (1) `casp16_labels.py` writing four class planes
per chain and the packer carrying them (u8 × 4 per pair); (2) a loss kind for K per-pixel
softmax heads over one output tensor — `perPixelMultiCE (sizes : List Nat)`: the head conv
emits `Σ_k NC_k` channels, the labels come as `[B, K, H, W]` int32, the loss is the sum of the
K masked CEs (each head's "unobserved" class at weight 0), its gradient the concatenation of
the K per-head seeds — one new emitter block beside `emitPerPixelCEBlock`, FD-checked the same
way; (3) `casp16_fold.py` adding the three angular potentials over pseudo-Cβ + Cα (which means
folding Cα and Cβ per residue, 6 coordinates, with the Cα–Cβ bond fixed at 1.53 Å); (4) the
`field` pseudo-Cβ scoring is unchanged. trRosetta reported TM +0.1–0.2 from the orientations
on CASP13 FM targets; that is the gap between our fold (0.46 TM on T1267s1-D1) and the
contact precision (0.67) suggests is available.

## 11. Handoff — 2026-10-03 07:00: the night stacked three levers; the book run's config is settled

**The book run landed 2026-10-04 14:59 — and the 100-epoch schedule bought nothing.** queue24
(the book config at `epochs=100`; 164,400 steps in 21.0 h on GPU 1): distance head 75.7 % val
(flat from epoch ~98), EUs **0.873** (0.878 / 0.896 / 0.723) over 84, plain fold 0.619 / 0.625,
**ω/φ fold 0.627 / 0.646** (medians 0.642 / 0.706, 2 wrong hands) over 78 — against the 30-epoch
row's 0.877 / 0.628 / 0.645. Paired over units, 100 minus 30 epochs: precision −0.004 [−0.017,
+0.007], Cβ-lDDT −0.001 [−0.006, +0.003], TM +0.001 [−0.005, +0.007] (95 % bootstrap). The
expectation below (~0.88 / 0.635 / 0.65, from queue23's +0.002 / +0.009 / +0.005 at 64 ch) did not
hold: at 128 ch × crop 96 the 30-epoch net has already used what the schedule offers. Either row
can carry the book; the 100-epoch run is the headline, the 30-epoch row its schedule check (the
two agree within 0.004 on every column). Figures stay on queue22's fold for now. Outputs:
`.lake/build/distogram_r16x128_esm3b_orient_pair1_train_full_e100-esm3b-crop96-pair1-orient_targets/`.

**The review, 2026-10-04 evening** (Brett: "doesn't seem to have moved the needle … any thoughts on what to do
next"). All from the book run's outputs; nothing trained.
- Saturation, not under-training: validation precision is level from about epoch 20 in both runs; validation CE is
  lowest at epoch 41 (1.932; the 30-epoch run ends at 1.935) and rises to 1.967 while the training loss falls 4.29 →
  4.03.
- The fold realises the map: map lDDT 0.615 against the ω/φ fold's 0.627 over the 78 (the fold is ahead on 52, most
  at separations of 48 and more); corr(TM, map lDDT) = 0.90. Item 7 of §11a (a better fold) is not next.
- Breadth: long-range recall at P > 0.5 is 0.49 and top-L precision 0.62 over the 78; the error on pairs under 12 Å
  is 3.2 Å.
- The net's lift over the 3B contact head (0.864 against 0.747 top-L/5 on the 78) is the same in every separation
  band: +0.14 to +0.17 precision at the true contact count from 6–12 residues out to 96 and beyond. Both decay
  together, so the decay at long range is the LM's and the receptive field is not the evident limit (item 5).
- The distance to the field (0.646 against a median model of 0.892): the 16 units under 0.85 precision fold to 0.34
  against the field's 0.83 and are 41 % of the gap; the other 62 fold to 0.72 against 0.91. Units over 200 residues
  0.59 against 0.93.
- ESMFold (§10): +0.132 TM over our head on the same LM. On the 62 units where our top contacts are right it is 0.84
  against 0.72 and never under 0.5 (we are on 7); on the 16 where they are not, 0.53 against 0.34 — it lifts 4 above
  0.8 (T1298-D2, T1284-D1, T1279-D2, T1272s8-D1) and is under 0.5 on 8. So the head is the larger gap, and half of
  the low-precision units are blind for the LM itself.
- Closed: longer schedules, seeds, ensembles, the fold engine, dilations. Parked: the MSA route (§11a, "The MSA
  route — parked": OpenProteinSet for the alignments, the steps in order, the estimate). Open: the headline row
  (100 epochs or its 30-epoch twin); the featured units. The plate carries all of it (version 8).

**The morning version.** Day 1 of §11a ran as a night: the pair-input op (`Layer.pairTile`'s `pairIn`,
the LM's own contact-head logits as one pair-tile channel) and ESM-2 3B both landed, and they stack
with width and crop. The arms, 30 epochs each, EUs top-L/5 LR precision over 84 / fold Cβ-lDDT / TM
over 78 (the 650M × 64 ch baseline 0.838 / 0.569 / 0.572; §10 has every row, §12 the log):

| arm | EUs | fold |
|---|---|---|
| 650M × 64 ch + plane | 0.856 (hard 0.710) | 0.584 / 0.582 |
| 3B × 64 ch | 0.840 | 0.581 / 0.583 |
| 3B × 64 ch + plane | 0.866 (seed 2: 0.866) | 0.603 / 0.612 (seed 2: 0.605 / 0.614) |
| 3B × 64 ch + plane + heads, ω/φ fold | 0.864 | 0.602 / **0.628** |
| 3B × 128 ch × crop 96 | 0.866 | 0.609 / 0.611 |
| 650M × 128 ch × crop 96 + plane (queue21, 08:58) | 0.865 | 0.607 / 0.605 |
| **3B × 128 ch × crop 96 + plane + heads, ω/φ fold** (queue22, 10:26) — **the book config** | **0.877** (0.880 / 0.895 / 0.764) | 0.621 / 0.630 plain, **0.628 / 0.645** ω/φ, 1 wrong hand |
| **3B × 128 ch × crop 96 + plane** (queue17) | **0.880** (0.880 / 0.903 / 0.746) | **0.622 / 0.632**, 1 wrong hand |

Against the baseline the book config (ω/φ fold) is +0.039 [+0.021, +0.058] precision, +0.059
[+0.050, +0.069] lDDT, +0.073 [+0.056, +0.091] TM (95 % bootstrap over units); the same without the
heads +0.041 / +0.053 / +0.060 — twice the 128 ch × crop 96 arm's gains. The plane costs
nothing per step and gives the hard class most (+0.09 at 64 ch); 3B is worth +0.002 on precision at
64 ch and +0.008 at the wide config, +0.012 on the fold; the orientation heads cost 0.002 precision
at 3B and buy +0.016 TM through the ω/φ fold. The field's TM percentile moves for the first time, 0.02 → 0.03 (Cβ-lDDT 0.06; field medians
0.945 / 0.883 / 0.656 against the best arm's 0.707 / 0.597 / 0.499 by class): the fold's distance
to the field is still the field's templates and MSAs. The figures are on
the book config's ω/φ fold (queue22): the trio T1271s6-D1 / T1295-D3 / T1228v1-D3 at 0.84 / 0.92 /
0.70 TM (field medians 0.93 / 0.93 / 0.73; the night started at 0.85 / 0.91 / 0.65), all 78 units in
the fourth row, `casp16_ablation.png` with the night's rows on top.

**At 11:30 everything has landed and all four cards are idle.** queue22 — the book-run config at
30 epochs — is the row above: the heads cost 0.003 precision against queue17 and buy +0.015 TM
through the ω/φ fold; the config is settled. queue23 (3B × 64 ch × 100 ep × pair=1, 0.868 / 0.612 /
0.617 against 0.866 / 0.603 / 0.612 at 30 epochs) says the 100-epoch schedule is worth +0.002 /
+0.009 / +0.005 with the plane — half what it gave at 650M.

**The book run** — queue22's config with `epochs=100`: `fs=esm3b dim=2569 ch=128 crop=96 pair=1
orient=1 batch=16`, ~21 h of training (755 s/epoch) + the finish and the ω/φ fold; from the 30-epoch
row (0.877 / 0.628 / 0.645) and queue23's schedule gains (+0.002 / +0.009 / +0.005), expect ~0.88 on
the EUs and a fold near 0.635 / 0.65 — the book run buys the headline's schedule, not a new lever.
`queue24_bookrun.sh <gpu>` is written; not launched — Brett's call. The levers left (§11a):
templates at inference (item 2, the field's actual advantage on the easy class), recycling (item 4)
and tier B, all on the pair-input op that now exists. The
rest of §11a: item 2 (templates at inference) is the lever still untouched; item 4 (recycling) and
tier B (the top-K attention heads as K planes) are next on the op that now exists.

**Hygiene for the morning (stale — kept):** the night's rows, figures and queue17–23 landed in
`77fd1a12`, pushed with `73b2333c` to origin/main 2026-10-03. Rules learned: the contact-head pass needs a 400 k-pair batch
cap (fair-esm keeps ~5 copies of the attention maps) and the 3B pass fp16; the EUs' planes come from
the fp32 CPU npz; at most three 3B-pool trainers at once (the fourth dies at `cuInit`).

### The 2026-10-02 18:40 handoff (kept)


Everything the night and the day produced is in §10 (table), §10a (findings), §11a (the week
plan) and §12 (log); the one-line version: **the LM is the lever** (35M 0.585 → 150M 0.735 → 650M
0.838 on the 84 EUs at 64 ch × 30 ep), at 650M width and crop 96 add (128 ch × crop 96: 0.858,
fold 0.600 / 0.598, the best single arm on every column), and the ω/φ restraints add +0.02 TM on
top of crop 96. The field TM percentile is 0.02 at every arm: the fold's distance to the field is
the LM's. The night is committed (49f6993b, on origin/main); the last row, both figures and this
section are the commit after it. All four cards idle since 18:09.

**2026-10-02 20:40 — two queues ahead of the book run** (Brett 20:00: use the cards this
afternoon, not the 20-hour run out of the gate). Both written, neither launched (the session's
launcher was blocked; each is one line):

    setsid -f nohup runs/2026-10-02-distogram-ablations/queue14_esm3b.sh > runs/2026-10-02-distogram-ablations/queue14.log 2>&1 < /dev/null
    setsid -f nohup runs/2026-10-02-distogram-ablations/queue15_pair.sh  > runs/2026-10-02-distogram-ablations/queue15.log 2>&1 < /dev/null

`queue14_esm3b.sh` is §11a item 3: the ESM-2 3B pool in fp16 as four shards (one per card,
~15 min), the EUs on the CPU meanwhile, pack, then 3B × 64 ch × 30 ep on GPU 0 (the ladder's
next rung after 0.585 / 0.735 / 0.838, ~75 min + finish) and 3B × 128 ch × crop 96 × 30 ep on
GPU 1 (the best 650M config, ~6.5 h + finish). `queue15_pair.sh` is item 1, tier A: the 650M
contact head's logit plane for the pool in two shards on GPUs 2–3 (it waits for queue14's embed
phase if that log exists), the EUs' and val subset's planes packed, then 650M × 64 ch × 30 ep
with `pair=1` on GPU 2 and finish — against the 64-ch baseline 0.838 / 0.569 / 0.572. The book
run (below) waits for both: its config may change (3B, the pair plane). §12, 20:40, has what
was built.

**Next: the 650M book run** — 128 ch × crop 96 × 100 epochs with the orientation heads, the one
config every stacking result points at. Written, not launched (Brett 18:40: "point the handoff at
the 20 hour run and we'll come back to it"); it is one command on an idle card:

    setsid -f nohup runs/2026-10-02-distogram-ablations/queue13_bookrun.sh 1 \
      > runs/2026-10-02-distogram-ablations/queue13.log 2>&1 < /dev/null

The argument is the card. `queue13_bookrun.sh` is queue11 with `epochs=100 orient=1` and queue10's
tail: it trains into `runs/<launch date>-distogram-r16x128-e100-esm650-crop96-orient/train.log`
(one line per epoch, ~727 s each over 1,644 steps → ~20 h; the 35M book run's 100 epochs ran
without a hitch), then `finish_run.sh` (predict → assemble → plain fold → score, ~45 min, into
`finish_bookrun.log`), then the ω/φ fold and `casp16_fold_score.py --suffix orient` (~40 min), and
ends with `queue13 done` in `queue13.log`. Disk: 140 GB free; the run keeps a few GB (params +
predictions) and has an 8 GB transient. Kill: `pkill -f "[d]istogram-casp"` (bracket the pattern,
and mention nothing else with the target's name in the same command — §12, 08:05).

What it should land, from the 30-epoch arm (0.858, fold 0.600 / 0.598) and the 64-ch
schedule-length row (+0.009 / +0.018 / +0.016): ~0.865 on the EUs, a plain fold near 0.62 / 0.61,
the ω/φ fold near 0.62 / 0.63. When it lands:
1. `casp16_table.py --field` reads the row (`r16x128_esm650_orient_train_full_e100-esm650-crop96-orient`)
   with the `(orient)` fold column beside the plain one; §10 gets the row, the status paragraph and
   §0 the number.
2. `casp16_ablation_figure.py` already lists the run as its top row ("the book run, ω/φ fold") and
   skips it until the directory exists; re-run it.
3. The figure, on the ω/φ fold (`--fold orient` puts `<EU>.orient.fold.pdb` and
   `fold_scores_orient.csv` on panel (c)):
   `casp16_figure.py .lake/build/distogram_r16x128_esm650_orient_train_full_e100-esm650-crop96-orient_targets
   demos/figures/casp16_distogram.png --metric tm --all-units --fold orient --eus T1271s6-D1
   T1295-D3 T1228v1-D3 --panel-a T1271s6-D1 --label "ours (650M, 128 ch, crop 96, 100 ep)"`.
4. Then the section (§6, §9): the book run as the headline row, the §10 table as the ablation and
   `casp16_ablation.png` beside it.

Open for Brett besides the launch:
- **Featured EUs (§9).** The committed figure carries the proposed trio T1271s6-D1 / T1295-D3 /
  T1228v1-D3 (0.85 / 0.91 / 0.65 TM against field medians 0.93 / 0.93 / 0.73) with every folded
  unit in the fourth row. The old trio (T1235-D1 / T1267s1-D1 / T1226-D1) is one `--eus` away;
  T1226-D1's precision is 0.00–0.08 at every LM, so its dot at the field median was the prior's.
- **The week plan (§11a)**, in order: attention maps as pair input, templates at inference, ESM-2
  3B, recycling, dilations — each about a day plus a 70-minute run, all of them on the three cards
  the book run leaves free.

## 11a. Where to go from here — the week plan (Brett, 2026-10-02 14:20: "throw it in the planning doc")

### What the easy class is telling us

650M × 128 ch, the 29 folded easy units (field TM median 0.945):

| signal | easy | medium | hard |
|---|---|---|---|
| top-L/5 long-range precision | 0.85 | 0.88 | 0.62 |
| top-L long-range precision | 0.67 | 0.60 | 0.40 |
| recall of true long-range contacts at p > 0.5 | 0.49 | 0.40 | 0.19 |
| mean distance error on pairs under 12 Å | 2.9 Å | 3.6 Å | 5.6 Å |
| fold TM (ours) | 0.66 | 0.56 | 0.46 |
| corr(TM, top-L precision) / corr(TM, length) | 0.77 / 0.03 | 0.68 / −0.45 | — |

Two readings. (1) **Four "easy" units are blind at every LM** — T1206-D1 (P@L/5 0.00), T1231-D1
(0.04), T1276-D1 (0.31), T1245s2-D1 (0.35) — and cost the class ~0.09 TM by themselves. They are
easy because a template exists in the PDB; a single-sequence LM cannot see one, and the purged-list
arm (above) shows templates in the *training set* are worth nothing — the field's advantage is
templates at *inference*. (2) **Breadth, not the top.** The top L/5 pairs are right, the next L are
half wrong and recall is under 0.5; the fold tracks top-L precision (0.77), not length (0.03). The
long easy units (T1234-D1 377 aa, the T1208 pair ~310 aa) have precision 1.00 at the top with 3 Å
errors underneath — the 65-residue receptive field running out. So the lever is more correct pairs
per unit, not a better optimizer (the energy-gap test, §10a, says the same from the other side).

### The experiments, ranked by expected gain per day

| # | experiment | what it needs | cost | expected |
|---|---|---|---|---|
| 1 | **ESM-2 attention maps as pair-input channels.** ESM's own contact head is a logistic regression on its (symmetrized, APC-corrected) attention maps; our ResNet sees only the per-residue embeddings. Tier A: the contact head's logit map, 1–2 channels. Tier B: the top-K heads by the contact head's weight, K = 16, u8. | one new op — a host pair input concatenated onto the pair map after `pairTile` (`Layer.pairConcat`); no input gradient, so the VJP is the identity on the net's own channels (a short proof item); the packer writes a pair pool: Σ L² = 1.80 G pairs → tier A 3.6 GB f16, tier B 29 GB u8 | DONE 10-02 20:30 — op (`Layer.pairTile`'s `pairIn`, not a new layer), gather, packer, proof item (§12); the run is queue15 | the largest known lever for single-sequence contact prediction; attacks the 0.49 recall directly: +0.03–0.05 on the units, more on the fold |
| 2 | **Template distograms as masked pair channels.** MMseqs2 (already in the pipeline for the purge) run the other way: for every chain and every EU, the best pre-cutoff PDB hit's Cβ distogram, aligned, one-hot into ~8 coarse bins + a "no template / unaligned" mask. | the same pair-input op as #1; a day of data engineering; the EU side must respect the CASP cutoff (templates released before 2024-05-01) | 1 day + a run | this is why "easy" is easy for the field; the four blind units should fold; easy TM 0.66 → ~0.8; nothing on hard |
| 3 | **ESM-2 3B** (`esm2_t36_3B_UR50D`): 36 layers, 40 heads, embedding dim 2560, 2.8 B parameters. Checkpoint ~11 GB in fp32, ~5.6 GB in fp16 → fits a 16 GB card in fp16 for sequences ≤ 1,024 (`casp16_embed.py` runs the LM in fp32 today; it needs a `--half` flag). Pool: 6.12 M residues × 2,569 × 2 B = 31.4 GB (disk now 142 GB free). Time: the 650M pool took 1,644 s on one card; 3B is ~4.3× the FLOPs per token → ~2 h on one card, ~30 min across four (per-chain `.done` resume makes sharding by chain list trivial). ESM-2 15B (48 layers, dim 5120, 30 GB fp16) does not fit a card without 8-bit or sharding — out of scope. | `--half`, the model name in `MODELS`, `fs=esm3b dim=2569` | scripted 10-02 (queue14: `--half --shard k/4`, four cards, then two arms) | the ladder went +0.15 then +0.10 per step (35M → 150M → 650M); the next step should buy +0.04–0.06 on the units |
| 4 | **Recycling.** A second pass that sees the first pass's distogram (softmax probabilities, or P(< 8 Å) + expected distance) as pair channels. | the pair-input op from #1; at train time the first pass runs under `stop-gradient` (the AF2 recipe); predict does two passes | half a day once #1 exists | AF2 found 3 recycles worth several lDDT points; here the honest expectation is +0.01–0.02 on the fold |
| 5 | **Receptive field.** Dilated residual units (trRosetta: 1, 2, 4, 8 cycling) or whole-map training at batch 1 for the long units. | dilation on `convBn` (check the emitter), or `crop=0` meaning whole map | half a day + a run | crop 96 was +0.012; the long easy units are the target |
| 6 | **The 650M book run**: 128 ch × crop 96 × 100 ep with `orient=1` — queue11 answered the stacking question (0.858, fold 0.600 / 0.598 at 30 ep; the two levers add), queue10 that the ω/φ fold stacks on crop 96. | nothing new; waits for queue14 / queue15 (the config may change) | ~20 h on one card (64 ch: 8 h) | the headline row: ~0.865 and a fold near 0.62 / 0.62 if the 64-ch schedule-length gain carries |
| 7 | **The fold, last.** A full-backbone build from distances + ω/θ/φ (trRosetta style: rigid residue frames, spline potentials, both hands by energy), scored with the all-atom-ish lDDT the field is scored on. | a torch rigid-body model; the orientation heads already exist | 2 days | after #1–#3 the distogram stops being the ceiling and this becomes it |

Cheap and low: test-time stride 16 / multi-crop averaging (+0.005 at best); ensembles (closed: nothing);
more seeds (0.002); longer schedules alone (+0.009 at 650M).

### The MSA route — parked 2026-10-04

Brett, 2026-10-04: "i would use OpenProteinSet for the alignments / nothing custom", then "we'll come back to it".
**Status: nothing downloaded, nothing built, nothing launched.** Decided: the training alignments come from
OpenProteinSet and nothing is searched locally. Open: go or no-go, and the three targets with no public alignment.

**What it is.** One alignment per chain; MSA Transformer (`esm_msa1b_t12_100M_UR50S`, already in `.venv-casp`'s
fair-esm) over each gives a 768-wide per-residue embedding and a contact-logit plane — the two things the net
already takes from ESM-2. So the Lean side is done: a wider `dim=` and `pair=2` on the existing
`Layer.pairTile … (pairIn := K)`; no new op, no new proof item.

**What to expect — an estimate, not a measurement** (the plate's chart, version 8): fold TM ≈ 0.71 (0.67–0.74) from
0.646, top-L/5 ≈ 0.90–0.93 from 0.873. Assumptions: (1) an MSA un-blinds a low-precision unit only where the
ColabFold baseline group's model (group 145; the field median where it has none) is at TM 0.8 or better — 10 of
the 16; (2) those reach what our head gets where its top contacts are right, 0.72; (3) the 62 units the LM already
sees gain +0.02. Low = half of the 10 and nothing else; high = +0.05 on the seen units and +0.10 on the rest. That
is about a quarter of the distance to the field's median model (0.892), and it stays under OpenComplex (0.768, the
lowest of the 30 groups with all 78 units) and under ESMFold without an MSA (0.778): the head, not the input, is
the larger gap (§11, the review).

**The data, as measured 2026-10-04.**
- Training side, OpenProteinSet: `https://openfold.s3.amazonaws.com/pdb/<pdb>_<chain>/a3m/` holds
  `uniref90_hits.a3m`, `bfd_uniclust_hits.a3m` and `mgnify_hits.a3m` per chain (public, no sign-in; 131,487 chain
  directories; `duplicate_pdb_chains.txt` at the bucket root lists identical chains, one of which carries the
  files). Our entity maps to `<entry, lower case>_<auth chain>`, directly or through its duplicate group. Coverage:
  22,063 of the 26,310 `train_full` chains — all but 29 of those released through 2021, none of the 4,218 released
  2022–24 — and 1,149 of the 1,380 val chains; 84 % of the residues. From a 60-chain sample: 0.43 MB per chain for
  the BFD / UniClust30 file (≈ 10 GB for the 22,063), 1.51 MB for UniRef90 (≈ 33 GB), 0.49 MB for MGnify.
- Target side: the targets postdate OpenProteinSet. MassiveFold's CASP16 release
  (github.com/GBLille/CASP16-CAPRI_MassiveFold_Data; files on entrepot.recherche.data.gouv.fr, one `.tar.gz` per
  target, URLs in `dataset_download/casp_massivefold_files_{monomers,multimers}.csv`) says each archive holds
  "predictions as well as pickle files, sequence alignments, rankings and plots". It has an archive for 54 of our
  59 targets (28 monomeric, 26 inside multimer archives) and none for T1214, T1228 and T1239: 17 of the 84 units,
  16 of the 78 folded, so an MSA arm is scored on 62 folded units unless those three get a source.
- Not checked: which alignment files a MassiveFold archive holds (tools, databases, formats) and how large an
  archive is (each carries up to 8,040 models; a HEAD request on the file URL is refused).
- Not taken, being the custom route: a local MMseqs2 search (UniRef50 is an 8.8 GB download; ColabFold's UniRef30
  is 103 GB plus 118 GB for the environmental set) and the ColabFold server for the targets.

**When we come back, in this order.**
1. *Checks, about an hour, before anything is built.* List one monomeric MassiveFold archive without keeping it
   (stream it through `tar tz`): which alignment files, which databases, how many GB. Pick the alignment kind to
   feed MSA Transformer — it has to be the same kind on both sides, so the choice is whichever of OpenProteinSet's
   three files MassiveFold also ships (the BFD / UniClust30 file is the smallest and the closest to what MSA
   Transformer was trained on). Confirm on a sample that an OpenProteinSet alignment's query row is our entity's
   sequence, residue for residue.
2. *The probe, half a day, no training.* The target alignments only; MSA Transformer's own contact head on each,
   scored like the ESM-2 heads (`casp16_targets.py`'s top-L/5 column), unit by unit against the 3B head (0.759 over
   the 84). It tests assumption (1): go if it lifts the low-precision units the estimate counts on. If it does not,
   stop here.
3. *Training alignments.* A fetch script beside `casp16_fetch_chains.py` (to write): per covered chain, stream the
   chosen file, subsample to a fixed depth (256 at most, diversity-maximising, as in the ESM examples), keep only
   the filtered alignment (about 2 GB in all), resume from a `.done` list. It writes the covered lists
   (`train_msa.csv` 22,063, `val_msa.csv` 1,149).
4. *Encoder pass.* `casp16_embed.py` grows an MSA mode (to write): fp16, `--shard k/n` over the four cards, the
   query row's embedding into a pool and the contact-logit plane beside the 3B plane. Estimated 2.5–5 h on four
   cards at depth 128–256 (scaled from the 650M pool's 1,644 s; not measured). Pool ≈ 7.9 GB (5.1 M residues × 768
   in fp16), plane ≈ 1.5 GB. MSA Transformer takes 1,024 positions: the training pool's longest chain is 512, but
   five targets are longer and need windows.
5. *Arms at 64 ch × 30 ep, 77 min each,* all on the covered lists: the no-MSA twin (3B + its plane — the baseline
   the other two are paired against, since the list is 16 % shorter), MSA only (`dim=777 pair=1`), 3B + MSA
   (`dim=3337 pair=2`). The wide config (about 6 h) only if 3B + MSA clears the twin by more than seed noise
   (0.002 on precision, 0.002 on the fold).
6. *Score* with `finish_run.sh` on the units that have a target alignment, paired against the book run on the same
   units; `casp16_table.py` reads the row.

**Constraints to plan around.** Disk: about 60 GB free, and a combined 3B + MSA feature file for the covered
chains would be about 34 GB next to the 33 GB 3B pool it duplicates — better that the gatherer
(`lean_casp_gather`) reads two pools side by side. Memory: at most three 3B-pool trainers at once (§11, hygiene).
About three days end to end, most of it steps 3–4.

### The week's shape (four cards)

Day 1: the pair-input op + its VJP lemma while the 3B embeddings run on GPUs 0–3 (30 min) and
the tier-A attention pool packs. Day 2: 650M + attention (tier A, tier B), 3B plain, 3B + attention;
the finish pipeline (`finish_run.sh`) scores each in 40 min. Day 3: templates (data day; run
overnight). Day 4: recycling; dilations. Day 5: pick the config; launch the book run (20 h).
Day 6–7: the fold; the section + figure + wiring (§6, §9) with the proposed featured trio.

## 12. Work log

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
- 2026-10-02 00:30–01:30 (night, autonomous): the 00:11 fold sweeps had died silently with their
  session (empty logs; `queue01.sh` then launched the one-hot and crop-96 trainers at 00:40);
  relaunched. ESM-2 150M chain ran (embedding 503 s on GPU; head 0.623 on the EUs); its 64-ch
  run passed every 35M arm by epoch 6. Analysis of the four finished arms (§10a). Fold:
  restarts bench, batched copies, `casp16_fold_score.py`, `casp16_table.py`, `finish_run.sh`.
  Orientation heads built end to end and smoke-tested (§10a); first run queued. Disk is at
  17 GB free after the orientation pool — ESM-2 650M waits on a decision (the four `_targets`
  dirs' `.acc.bin` intermediates are 17 GB and regenerable from the kept checkpoints).
- 2026-10-02 01:30–03:00: fold sweeps done and scored (64 ch 0.423 / 0.389 over 81 EUs, 128 ch
  0.444 / 0.416 over 79; hand rule wrong on 21 % / 14 %); against the field's model-1 rows our
  TM sits at the 1st–3rd percentile on every class (`casp16_table.py --field`). ESM-2 150M arm:
  EUs 0.735, fold 0.512 / 0.498 — the night's largest move; one-hot 0.252 / fold 0.319. Orient
  run finished training (val 34.7 %, the plain run's 35.2); its finish, the val bench of the
  restraint weight and the ω/φ fold are running (queue03.sh). Disk fell to 8.8 GB free from the
  per-arm `.acc.bin` accumulators (4.2 GB each); the six assembled arms' accumulators were
  deleted (regenerable by `predict` in ~1 min each; `finish_run.sh` now drops them after
  assembly) — 25 GB free. The `queue02.sh` wait on the one-hot trainer hung 40 min on a
  `pgrep -f` ghost (the launching shell's heredoc text); killed the wrapper shell, noted in memory.
- 2026-10-02 03:00–04:50: orient run finished (val 34.6 %; EUs 0.595; heads weak but calibrated);
  val bench of the restraint weight (w 0.3 / 1 / 3 / 10 → 0.515 / 0.519 / 0.517 / 0.490 lDDT, hand
  96–100 %; plain at the same restarts 0.498 / 0.444 / 83 %); the 84-EU ω/φ fold at restarts 0 (the
  restarts-4 version ran 5 min per EU and was stopped at 11): 0.446 / 0.422, better on 73 / 62 of
  78 EUs. Crop-96 EUs 0.617. Brett (04:10): "let's do the 650m model at some point" → queue04.sh
  on GPU 2 (its esm150 seed-2 slot dropped); `casp16_embed.py --pool-out` streams the pool (checked
  against the 35M pool on 20 chains: ≤ 1 f16 ulp), `casp16_pack.py --sets`. Disk: the orient dir's
  accumulators and `emb150/*.npy` (7.8 GB, packed already) deleted → 16 GB free before the 650M pool
  fills (16.6 GB, ends ~6 GB free; `emb/` 5.9 GB is the next regenerable cache). queue05.sh adds the
  finish steps for esm150 × 128 ch (GPU 0, ~06:10) and crops=3 (GPU 1, ~07:00).
- 2026-10-02 04:50–06:10: 650M chain — pool embedded on one card in 1,644 s (3.9 k tokens/s), the EUs on
  CPU in 2 min (650M head 0.754 / median 0.832 on the 84 EUs, hard 0.532), 64-ch run to val 51.5 %
  (35M 35.2, 150M 45.5). 150M × 128 ch finished: EUs 0.742, fold 0.528 / 0.504. GPU 0 then took the
  orientation heads on 650M features and, after it, crop 96 on 650M (queue06.sh). `finish_run.sh`
  now serializes predict + assemble across queues (flock) because the window accumulators are a
  4 GB transient on a disk with ~10 GB free; the pool's membership is frozen in
  `packed/pool_order.txt` (both the packer and the streaming embedder read it), so `emb/`
  (5.9 GB of per-chain 35M features, all in the pool already) can be dropped if the disk demands.
- 2026-10-02 06:26: 650M arm scored — EUs 0.838, fold 0.569 / 0.572; the LM ladder is the table's
  spine (35M / 150M / 650M = 0.585 / 0.735 / 0.838 at 64 ch × 30 ep). Field TM percentile unmoved
  (~0.02 median): the fold's ceiling is the fold, not the contacts.
- 2026-10-02 07:18: disk hit 580 MB free. Cause: the orientation-head arm's window accumulators are
  132 channels wide (7.8 GB for 84 EUs; the two > 1,600-residue EUs are 1.5 GB each), and the
  flock only serializes finishes, it does not budget them. `casp16_predict.py` died with ENOSPC
  at 50 of 84 EUs; the fold started on the partial set. Freed `emb/*.npy` (5.9 GB; the pool and
  `pool_order.txt` make it redundant), stopped queue06, and queue07.sh redoes the assembly, drops
  the accumulators (+7.8 GB), folds and scores both ways, then runs crop 96 × 650M. The book run
  wrote its 07:19 checkpoint through the squeeze (19.5 MB, intact). crops=3 finished: EUs 0.598.
  Lesson for the predictor: write the accumulators per EU and assemble as it goes, or in f16 —
  a 4–8 GB transient per arm is the wrong shape for this disk.
- 2026-10-02 08:00–08:15: orient × 650M re-assembled and folded both ways (+0.010 / +0.014, heads
  sharper); crops=3 fold 0.446 / 0.419. Found that `casp16_valfold.py` never selected the GPU, so
  every fold bench of the night ran on CPU (same numbers, 5–20× slower); `--device` added, the
  long-chain bench relaunched on GPU 2. queue08.sh takes the idle cards: 650M × 64 ch × 100 ep on
  GPU 1 (the book-run candidate on the right LM) and 650M × 128 ch × 30 ep on GPU 2 after the bench.
- 2026-10-02 08:25: fold bench on the long val set (above): steps and learning rate are flat, the
  long chains fold better than the short — the optimizer is not the ceiling; next diagnostic is the
  energy of the true trace under the fold's own potential (optimizer gap vs potential gap).
- 2026-10-02 08:40: energy-gap diagnostic (above): the true trace scores worse than our fold under
  our own potential on every EU, under every re-weighting tried — the distogram, not the optimizer,
  is the fold's ceiling. Running: book run GPU 3 (epoch 90, ends ~09:35), crop 96 × 650M GPU 0,
  650M × 64 ch × 100 ep GPU 1 (~12:15), 650M × 128 ch × 30 ep GPU 2 (~11:15); each finishes itself.
- 2026-10-02 09:27: the book run (35M × 128 ch × 100 ep) finished: 82,200 steps in 9.2 h, val
  top-L/5 long-range 38.98 % (30 ep at the same width: 37.4 %; the 100-epoch schedule is +1.6 val
  points), val CE 2.397. Its finish (predict 74 s, assemble, fold, score) runs on GPU 3
  (`finish_book_e100.log`). It is the wrong LM for the book (650M is 0.838 to 35M's 0.585 on the
  EUs); the 650M × 64 ch × 100 ep run on GPU 1 is the candidate replacement — Brett's call.
- 2026-10-02 09:34: ensemble arm. `scripts/demos/casp16_ensemble.py <name> <dirs…>` averages the
  members' class probabilities into `.lake/build/distogram_ens-<name>_targets/` (the ω/θ/φ planes
  averaged over the members that carry them), writes table.csv, and the fold / score / table
  scripts read it like any arm. 650M plain + 650M orient (both 64 ch × 30 ep): EUs 0.842 against
  0.838 for either alone (easy 0.850 / medium 0.876 / hard 0.619) — +0.004 at the contact level;
  folding both ways on GPU 3 (`ens_fold.sh`, `ens_esm650x2.log`) to see what the fold makes of it.
  More members (650M × 128, crop 96, 100 ep) arrive through the morning.
- 2026-10-02 09:50: disk at 7.6 GB free with four finishes queued (each a 4 GB transient): deleted
  `packed/pool_esm150_feat.bin` (7.8 GB; both 150M arms are done and scored; regenerate with
  `casp16_embed.py --model esm2_t30_150M_UR50D --pool-out …` — 8 min on a GPU — then
  `casp16_pack.py --sets pool`). The book run's 84-EU precision: 0.647 (128 ch × 30 ep 0.629,
  64 ch × 30 ep 0.585). queue09.sh (GPU 3 after the folds): 650M × 64 ch × 30 ep at seed 2 — the
  seed-noise bar at the headline LM and a third ensemble member.
- 2026-10-02 10:14: book run (35M × 128 ch × 100 ep) folded and scored: Cβ-lDDT 0.466 / TM 0.443 over
  78 EUs (easy 0.473 / 0.476, medium 0.477 / 0.445, hard 0.330 / 0.244; mirror better on 11). The
  35M ladder is now 64 ch 0.423 / 0.389 → 128 ch 0.444 / 0.416 → 128 ch × 100 ep 0.466 / 0.443,
  and 650M × 64 ch × 30 ep sits at 0.569 / 0.572 — the LM is worth four of these steps.
- 2026-10-02 10:32: the two-member 650M ensemble folded both ways: plain Cβ-lDDT 0.569 / TM 0.575
  (members 0.569 / 0.572 and 0.564 / 0.577), with ω/φ 0.579 / 0.592 (the orient member alone
  0.574 / 0.591); wrong hands 5 → 4. The +0.004 at the contact level becomes nothing at the fold:
  two nets on the same LM, data and crops are too correlated to add information, which is the
  energy-gap finding again from the other side. Not a table row; the script stays for the
  128-ch / 100-ep / seed-2 members in case a less correlated set does better. queue09 started the
  650M seed-2 run on GPU 3 at 10:32; crop 96 × 650M at epoch 29/30.
- 2026-10-02 11:00: crop 96 × 650M (64 ch × 30 ep) finished: EUs 0.850 (easy 0.854 / medium 0.880 /
  hard 0.660) against the 64-crop arm's 0.838 / 0.851 / 0.868 / 0.620; fold Cβ-lDDT 0.583 / TM
  0.589 against 0.569 / 0.572, wrong hands 3. The crop lever carries to the top LM (+0.012 on the
  EUs, +0.014 / +0.017 on the fold; it was +0.032 at 35M), though at 650M the gain is the size of
  the 35M seed gap (0.585 vs 0.599) — the 650M seed-2 arm (GPU 3, ~12:30) sets that bar. GPU 0
  takes orientation heads on the crop-96 650M config next, to see whether the two gains stack.
- 2026-10-02 11:37: 650M × 128 ch × 30 ep finished (24,660 steps, 2.8 h): EUs 0.848 (easy 0.856 /
  medium 0.879 / hard 0.641); fold Cβ-lDDT 0.590 / TM 0.590 against 0.569 / 0.572 at 64 ch, wrong
  hands 4. Width is the second lever at the top LM: +0.010 on the EUs, +0.021 / +0.018 on the fold
  (crop 96: +0.012, +0.014 / +0.017). GPU 2 takes the stacking question — 128 ch × crop 96 × 650M,
  30 ep at batch 16 (~6 h; the config a 650M book run would use) — as queue11.sh.
- 2026-10-02 11:52: queue11 (650M × 128 ch × crop 96, batch 16) runs at 724 s per epoch — 6 h for
  30 epochs, done ~17:45 plus the finish. Seed 2 at 650M (64 ch × 30 ep): EUs 0.836 against seed
  1's 0.838 — the seed gap at the top LM is 0.002, so the width (+0.010) and crop (+0.012) steps
  are five times the noise; fold pending.
- 2026-10-02 12:07: seed 2 at 650M folded: Cβ-lDDT 0.570 / TM 0.571 against seed 1's 0.569 / 0.572
  — the fold's seed noise is 0.001, so the width and crop steps (+0.02) are ten times it. The
  650M × 64 ch × 100 ep run (GPU 1, 3.9 h) finished: EUs 0.847 (hard 0.677) against 0.838 at 30
  epochs; fold running. GPU 3 next: a five-member 650M ensemble (seed 1, seed 2, 128 ch, crop 96,
  100 ep — less correlated than the pair that gained nothing), then the no-template ablation
  (list=train) at 650M, the table's "purged" row on the right LM.
- 2026-10-02 12:15: five-member 650M ensemble (seed 1 0.838, seed 2 0.836, 128 ch 0.848, crop 96
  0.850, 100 ep 0.847): EUs 0.852 (easy 0.855 / medium 0.883 / hard 0.658) — +0.002 over the best
  member. Five differently-trained nets on one LM still agree on what they get wrong; the LM's
  representation, not the head, bounds the contact map. Folding it anyway on GPU 3 (queue12.sh),
  then the no-template ablation at 650M on the same card.
- 2026-10-02 12:23: 650M × 64 ch × 100 ep folded: Cβ-lDDT 0.587 / TM 0.588 (easy 0.618 / 0.658,
  medium 0.582 / 0.555, hard 0.449 / 0.473), wrong hands 2 — against 0.569 / 0.572 at 30 epochs.
  At the top LM the three training levers are each worth about the same on the fold: width
  +0.021 / +0.018, schedule +0.018 / +0.016, crop +0.014 / +0.017; seed noise 0.001 / 0.001. On the
  EUs: width +0.010, crop +0.012, schedule +0.009, seed 0.002. The field TM percentile is 0.02 at
  every one of them: the fold's distance to the field is the LM's, not the head's.
- 2026-10-02 12:35: five-member 650M ensemble folded: Cβ-lDDT 0.584 / TM 0.585 — the members'
  average (0.569–0.590 / 0.571–0.590), under the best of them (128 ch, 0.590 / 0.590). Ensembling
  closed: nothing at either level. Dropped both ensemble dirs' `.pred.npz` (3.8 GB; `casp16_ensemble.py`
  rebuilds them in two minutes; table.csv, folds and scores kept) ahead of the crop-96 orient
  finish, whose 132-channel accumulators need a 7.8 GB transient; `casp16_table.py` now skips the
  orientation column for a dir without `.pred.npz`. The purged-list 650M run started on GPU 3 at 12:35.
- 2026-10-02 14:10: Brett awake ("wrap up where things are at / let the current runs finish; longer
  runs scheduled later") — no new launches. Fetched the field's models for the proposed featured
  targets (T1271s6, T1295, T1228v1; `CASP16_TARGETS=… download_casp16.sh`, 295 MB) and rescored
  their first models at pseudo-Cβ (`casp16_score.py field <EU> --pseudo-cb`): T1271s6-D1 62
  groups, Cβ-lDDT median 0.892 / TM 0.928; T1295-D3 61, 0.857 / 0.930; T1228v1-D3 69, 0.710 /
  0.731. The figure on the 650M × 128 ch arm with that trio: panel (b) precision 1.00 on
  T1271s6-D1; our dots 0.81 / 0.82 / 0.61 against the medians 0.89 / 0.86 / 0.71 — within 0.1 of
  the field on units where the distogram is right, against 0.42 / 0.65 / 0.32 vs 0.90 / 0.67 /
  0.36 on the current trio. Plate (both figures + the ladder + decisions):
  https://claude.ai/artifact/CyZfnLRWnjkpByMqdHyzTo (`scratchpad/build_plate.py` rebuilds it).
- 2026-10-02 14:11: purged-list 650M arm done: EUs 0.841, fold 0.569 / 0.567 — indistinguishable
  from the headline list (0.838, 0.569 / 0.572). The 82 purged chains are the templates the
  field was allowed; at 650M they move nothing, so the "easy" class's advantage in the field is
  templates at inference, not in training. crop 96 × orient: EUs 0.850, plain fold 0.576 / 0.586;
  ω/φ fold running. Easy-class diagnosis (650M × 128 ch, 29 units): top-L/5 0.85 but top-L 0.67
  and recall of true long-range contacts at p > 0.5 only 0.49; TM correlates with top-L
  precision (0.77), not with length (0.03); four "easy" units (T1206-D1, T1231-D1, T1276-D1,
  T1245s2-D1) are blind at every LM (P@L/5 0.00–0.35) and cost the class ~0.09 TM — they are easy
  because a template exists, which a single-sequence LM cannot see. Week plan given to Brett:
  (1) ESM-2 attention maps as pair-input channels (one new host-pair-input op), (2) template
  distograms as masked pair channels (MMseqs2 the other way), (3) ESM-2 3B, (4) recycling,
  (5) dilated units / whole-map training, (6) the 650M book run, (7) the full-backbone fold last.
- 2026-10-02 14:22: crop 96 × 650M × orient, ω/φ fold: Cβ-lDDT 0.585 / TM 0.606 (median 0.644),
  2 wrong hands — the table's best fold (128 ch plain 0.590 / 0.590; crop 96 plain 0.583 / 0.589;
  this arm's own plain fold 0.576 / 0.586). The orientation gain (+0.009 / +0.020 here, +0.010 /
  +0.014 at crop 64) stacks on the crop gain, which argues for `orient=1` in the 650M book run.
  Only queue11 (128 ch × crop 96, GPU 2, ~18:30) is still running.
- 2026-10-02 15:00: two figure changes, both from what is on disk. `casp16_figure.py --metric tm
  --all-units`: panel (c) on TM-score (the fair column against the official table; it sees
  chirality) with a fourth row holding all 78 folded units — each unit's official field median
  in grey, ours in blue, the featured units ringed; on the 650M × 128 ch arm ours median 0.64 vs
  the field's 0.94, the three ringed dots inside the field's cloud. That row is the honest
  companion to the trio: "in the neighbourhood where the distogram is right, and how often".
  `casp16_ablation_figure.py`: the AF2 Fig. 4a shape — one row per arm, mean paired difference
  to the 650M baseline over units with a 95 % bootstrap interval, three columns (precision over
  84, fold Cβ-lDDT and TM over 78), broken x-axis. Numbers: crop 96 + orient ω/φ +0.011 / +0.017 /
  **+0.034**; 128 ch +0.010 / +0.021 / +0.018; crop 96 +0.011 / +0.014 / +0.017; 100 ep +0.009 /
  +0.018 / +0.016; orient alone −0.001 / +0.006 / +0.019; ensemble +0.013 / +0.015 / +0.013;
  seed 2 −0.003 / +0.001 / −0.001; purged +0.003 / −0.000 / −0.005; 150M −0.103 / −0.056 / −0.074;
  35M −0.253 / −0.138 / −0.177; one-hot −0.586 / −0.250 / −0.344. The 128 ch × crop 96 row joins
  when queue11 lands. Brett: the trio reads as cherry-picked without the all-units row; the book
  wants "in the neighbourhood, a stepping stone", not SOTA. Plate rebuilt with both.
- 2026-10-02 18:20: queue11 landed (650M × 128 ch × crop 96 × 30 ep, 49,320 steps in 6.1 h on
  GPU 2): EUs 0.858 (0.866 / 0.885 / 0.674), fold Cβ-lDDT / TM 0.600 / 0.598 over 78 EUs (medians
  0.621 / 0.626), 7 wrong hands. Width and crop add: +0.020 / +0.031 / +0.026 over the 64-ch arm
  against +0.010 / +0.021 / +0.018 (128 ch) and +0.011 / +0.014 / +0.017 (crop 96); 95 % intervals
  [+0.010, +0.030] / [+0.027, +0.036] / [+0.018, +0.036]. The best single arm on every column; the
  field's TM percentile is still 0.02 (easy 0.671 vs 0.945, medium 0.566 vs 0.883, hard 0.462 vs
  0.656). Both figures regenerated on this arm: `casp16_distogram.png` (the trio 0.85 / 0.91 / 0.65
  TM against field medians 0.93 / 0.93 / 0.73; all 78 units median 0.63 vs 0.94) and
  `casp16_ablation.png` (the row on top). `casp16_figure.py`: the legend is kept out of
  `tight_layout` (`set_in_layout(False)`) and centred a little left of (c), because the longer
  "ours" label pushed it off the page. All four cards idle; the 650M book run (§11a item 6) is
  Brett's to schedule.
- 2026-10-02 18:40: Brett: "point the handoff at the 20 hour run and we'll come back to it, commit".
  §11 rewritten around the book run: `queue13_bookrun.sh <gpu>` (queue11 with `epochs=100
  orient=1` plus queue10's ω/φ tail; ~22 h end to end), the expected numbers, the four steps when
  it lands. `casp16_figure.py --fold orient` puts the ω/φ-restrained fold on panel (c) and the
  all-units row (checked on the crop 96 × orient arm: the strip moves 0.794 / 0.798 / 0.592 →
  0.798 / 0.802 / 0.602 Cβ-lDDT). `casp16_ablation_figure.py` lists the book run as its top row and
  skips it until the directory exists. Committed; nothing launched.
- 2026-10-02 20:40 (Brett 20:00: "let you bang on it some more this afternoon, use the GPUs;
  maybe not the 20 h run out of the gate; anything we can code in the meantime?"): plan day 1,
  both halves. (a) ESM-2 3B (§11a item 3). `casp16_embed.py` gained `--half` (fp16 weights,
  5.7 GB on a card; checked against fp32 on the CPU on two pool chains: embedding cosine
  1.0000, contact map max |Δ| 0.024, the same 624 / 602 pairs over 0.5), `--shard k/n` (one
  process per card over the one pool file, created without O_TRUNC and only grown, per-shard
  `.done.k`), the 3B entry (36 layers, 2560 → `dim=2569`); the packer's `esm3b` feature set;
  the checkpoint (5.7 GB + its contact head) is in the torch hub cache. `queue14_esm3b.sh` as
  §11 describes. (b) The pair-input op (§11a item 1), as an extension of the pair tile rather
  than a new layer: `Layer.pairTile … (pairIn := K)`. The host row becomes `[2·L·D | K·L·L]`;
  the emitter slices the feature blocks off its head, reshapes the tail to `[B, K, L, L]` and
  concatenates it behind the tile's channels (`[B, C + K, L, L]`; the convBn after it takes
  `C + K`); the backward slices the cotangent's first `C` channels and is otherwise the tile's
  (no input gradient; `pairIn = 0` emits byte-identical graphs). `lean_casp_gather_pair`: the
  planes ride behind the feature blocks of every row, u8 `[L, L, K]` per chain at `K` times its
  label offset, byte v ↦ (v − 128)/64, zero past the chain end; demo knob `pair=K` (train, val,
  predict; the prefix gets `-pairK`). Checked exactly rather than by finite differences: a
  pass-through net (the tile and a 1×1 head whose weight picks the plane channels) returns the
  planes to 0.0 over 256 values, and the pair net at zero planes and zero plane weights is the
  plain net — the same loss and the same first Adam moment on all 698 shared coordinates
  (max |Δ| 0.000000). Three train steps at full width (650M features, crop 64, batch 32,
  `pair=1`: 1,357,314 params = the 64-ch net + 64) ran on GPU 3. Proof item,
  `Proofs/Foundation/PairTile.lean`: `tileWPair` / `tileWjPair` (the tile with `K` constant
  planes appended, `finSumFinEquiv` layout), `pdiv_tileWPair` / `pdiv_tileWjPair` (the tile's
  Jacobian on the tile block, zero on the plane block) and `tileWPairHasVJP` /
  `tileWjPairHasVJP` (backward = `gradW` / `gradWj` on the cotangent's tile block,
  `tileBlock`), each closed by the tile's own witness. (c) The smoke's finite-difference check,
  on the way: at seeds other than 7 the unchanged plain and orientation variants failed the
  same way the pair variant first did (a 2–5 % miss on the stem weights, body and head at
  1e-5) — the loss is ReLU on batch-statistics BN and the stem's FD straddles kinks. The check
  now takes the central difference at ε/2, adds |FD(ε) − FD(ε/2)| (the FD's own uncertainty)
  to the tolerance, and reports a coordinate whose two FDs disagree by more than the tolerance
  as a kink instead of comparing it (at most half may be); `seed=` and `eps=` are arguments.
  (d) Tier A data: `casp16_embed.py --pair-out` writes the model's contact-head logit planes
  for the pool (byte = clip(128 + 16·logit, 0, 255); `--max-pairs` caps B·L² for the attention
  maps the head keeps; a three-chain probe reproduced a CPU re-encode byte for byte, and the
  file is 1.89 GB, the label pool's size); `casp16_pack.py --pair-only` writes the EUs' planes
  from the `esm_contacts` of their npz and the val subset's out of the pool file.
  `queue15_pair.sh` as §11 describes. Nothing launched; the work is staged, not committed.
- 2026-10-02 20:29–20:55: Brett launched both queues (20:29); the work is origin/main 346be238
  (rebased on the Orin commits). queue14: the 3B pool in 425–433 s per shard (27,690 chains,
  33.1 GB), the EUs on the CPU alongside, pack, both arms training from 20:37 (64 ch: 1,521,090
  params, loss 12.12 at step 0; 128 ch × crop 96: 5,409,602, 11.13). queue15's first plane pass
  ran out of card memory at chain 6,531 on both shards: fair-esm's contact head keeps ~5 copies of
  the 33 × 20 attention maps (2.6 kB per pair per copy in fp32), so the 1.5 M-pair batch cap was
  ~4× too loose once chains passed ~130 residues. Cap 400 k (~6 GB; the pool's longest chain is
  512 residues), still fp32 to match the EUs' planes; relaunched 20:41, the remaining 21,160 chains
  in 11 min, 1.89 GB = the label pool's size. The packer's `--pair-only` then failed on a name
  (`sfx` was `pack()`'s local) — fixed and run by hand (targets 11.6 MB over 84 EUs, valsub 0.5 MB
  over 24 chains, each the size of its label file) while the `pair=1` arm, which needs only the
  pool's planes, had already started on GPU 2 (20:52; 1,357,314 params).
- 2026-10-02 22:20: the first two arms land. **3B × 64 ch × 30 ep** (queue14, 4,632 s): val top-L/5
  53.0 % against the 650M arm's 51.5 % (+1.5 pt from epoch 10 on; val CE 2.110 vs 2.138), but on the
  EUs **0.840** (0.872 / 0.849 / 0.664) against 0.838 (0.851 / 0.868 / 0.620) — a tie on precision —
  and the fold **0.581 / 0.583** against 0.569 / 0.572 (+0.012 / +0.011; medians 0.592 / 0.617, 3
  wrong hands): the LM step shows up in the calibration, not the ranking. The 3B × 128 ch × crop 96
  arm tracks its 650M twin at +2 pt on val (71.4 % vs 69.4 % at epoch 5). **650M × 64 ch × 30 ep
  pair=1** (queue15, 4,199 s — the plane costs nothing per step): val 52.7 % (48.2 % after ONE epoch,
  where the baseline needed six to reach 47.0 %), EUs **0.856** (0.858 / 0.880 / **0.710**) against
  0.838 — +0.018 overall and +0.090 on the hard class, at 64 ch matching the 128 ch × crop 96 arm's
  0.858 at a fifth of the training time; fold pending. The plan's ranking held: tier A (item 1) is
  the lever, 3B (item 3) is a fold-only gain. queue16 (3B + the 3B head's plane) training on GPU 3
  from 22:07; `queue17_pair_wide.sh` (650M × 128 ch × crop 96 × pair=1 × 30 ep, GPU 0, ~6 h) written
  as the book-run config test, awaiting Brett's go.
- 2026-10-02 23:50: **3B × 64 ch × 30 ep pair=1** (queue16, the 3B head's plane; 4,646 s): val
  54.3 % (the highest of any 64-ch arm), EUs **0.866** (0.876 / 0.885 / **0.724**) — above every arm
  including 128 ch × crop 96 (0.858); the levers stack (0.838 → +plane 0.856 → +3B 0.866; 3B alone
  0.840). Fold (23:49) **0.603 / 0.612**, medians 0.618 / **0.684**, 7 wrong hands — the best fold of
  any arm (128 ch × crop 96: 0.600 / 0.598; the ω/φ fold's TM 0.606), so 3B + plane at 64 ch is the
  best arm on every column, in 77 min. queue18 (650M × 64 ch, plane + orientation heads): distance
  head 52.9 % val, EUs 0.849 (0.855 / — / 0.678) against the plane alone's 0.856 — the heads cost
  0.007 here where at 650M without the plane they cost nothing; folds pending. Launched 23:50 with
  the permission Brett gave at 21:35: queue17 = 3B × 128 ch × crop 96 × pair=1 × 30 ep on GPU 0
  (~6.5 h, the book-run candidate at the wide config) and queue19 = queue18 on 3B features (GPU 3,
  ~2.5 h): the head question for the book run.
- 2026-10-03 00:30: queue18 lands — plane + orientation heads at 650M × 64 ch: plain fold 0.577 /
  0.585, ω/φ fold **0.588 / 0.605** (medians 0.611 / 0.663): the ω/φ fold adds +0.011 / +0.020 over
  the plain one, so against the plane alone the heads are −0.007 precision, +0.004 lDDT, +0.023 TM
  — the same shape as every earlier orient arm. queue19 (3B + plane + heads) tracks 3B + plane
  exactly on the distance head (53.4 % / 54.2 % at epochs 10 / 20). The 3B × 128 ch × crop 96 arm
  (no plane) is at 75.1 % against 72.7 % for its 650M twin at epoch 20; queue17 (with the plane) at
  73.6 % at epoch 4. queue20 (3B + plane, seed 2) launched on GPU 2 for the headline row's error bar
  — and died at `cuInit` (CUDA_ERROR_NOT_INITIALIZED, the NVRM host-memory symptom): three 3B-pool
  trainers (33 GB each, 36–44 GB RSS) and 108 GB of page cache left 23 GB free. At most three 3B
  trainers at once; relaunched 00:55 gated on queue19's trainer exiting (it started 01:10).
- 2026-10-03 01:15: queue19 (3B + plane + orientation heads, 64 ch; 4,683 s): distance head 54.2 %
  val, EUs **0.864** (0.876 / 0.880 / 0.722) against 3B + plane's 0.866 — at 3B the heads cost
  0.002, seed-noise level, where at 650M + plane they cost 0.007. Plain fold 0.594 / 0.611, ω/φ fold
  **0.602 / 0.628** (medians 0.619 / 0.692, 2 wrong hands; done 01:51): the best TM of any arm —
  the heads are −0.002 / −0.001 / +0.016 TM on the best arm, so `orient=1` stays in the book run.
  queue21 (650M × 128 ch × crop 96 × pair=1, the 650M twin of queue17) launched gated on queue20's
  trainer exiting (~02:30), GPU 3, ~6 h: whether the wide config needs 3B or the plane alone carries it.
- 2026-10-03 02:55 (Brett 02:35: "carry on, will be back in the morning"): queue20 — seed 2 of 3B +
  plane: val 54.2 %, EUs **0.866**, fold 0.605 / 0.614 against seed 1's 0.866 / 0.603 / 0.612: a seed
  gap of 0.000 / 0.002 / 0.002. queue21 (650M × 128 ch × crop 96 × pair=1) started 02:27 on GPU 3 the
  minute queue20's trainer exited; queue22 — the book-run config at 30 epochs, 3B × 128 ch × crop 96
  × pair=1 × orient=1 with the ω/φ fold — written and parked for GPU 1 behind queue14's finish (~03:30;
  lands ~11:30). GPU 2 idle on purpose: three trainers is the host's limit. Overnight rule: log, stage,
  no commits, no book run.
- 2026-10-03 03:10: the 3B × 128 ch × crop 96 arm (no plane; 6.4 h): val 75.1 % against 72.9 %,
  EUs **0.866** (0.884 / 0.884 / 0.696) against the 650M twin's 0.858 — 3B is worth +0.008 at the
  wide config; fold running. queue17 (with the plane) is at 75.5 % val at epoch 15, already past
  this arm's final; queue21 (650M wide + plane) at 71.0 % at epoch 3. 03:26: its fold 0.609 / 0.611
  (medians 0.629 / 0.666) against the twin's 0.600 / 0.598; queue14 done, queue22 (the book-run
  config at 30 ep) takes GPU 1.
- 2026-10-03 06:20: **queue17 lands on precision — 3B × 128 ch × crop 96 × pair=1 × 30 ep: 0.880**
  (0.880 / 0.903 / 0.746), val 75.8 %: +0.014 over the same without the plane (0.866), +0.022 over
  650M wide (0.858), +0.042 over the baseline. Every lever stacks at the wide config. Fold running.
  queue21 (650M wide + plane) at 73.9 % val at epoch 20 (650M wide 72.7 %, 3B wide 75.1 %); queue22
  (the book config with the heads) at 75.5 % at epoch 12, tracking queue17. 06:41: queue17's fold
  **0.622 / 0.632** (medians 0.640 / 0.690, 1 wrong hand) — the best fold of any arm; the wide 3B +
  plane arm is best on every column, 0.880 / 0.622 / 0.632. queue23 (3B × 64 ch × 100 ep pair=1, the
  schedule-length question with the plane, GPU 0, ~5 h) launched; GPU 2 idle under the trainer limit.
- 2026-10-03 08:36: queue21 (650M × 128 ch × crop 96 × pair=1; 6.1 h): val 74.0 %, EUs **0.865**
  (0.871 / 0.892 / 0.693) — +0.007 over 650M wide, 0.015 under 3B wide + plane: at the wide config
  the plane and 3B are each worth ~+0.007 and add to +0.022; the book run keeps the 3B features. Fold
  running. queue22 (the book config) at 76.0 % on the distance head at epoch 22; queue23 (100 ep) at
  54.7 % at epoch 35, past the 30-epoch arm's final. 08:58: queue21's fold 0.607 / 0.605 (medians
  0.622 / 0.651) against 650M wide's 0.600 / 0.598; queue21 done, GPU 3 left free for the book run.
- 2026-10-03 09:50: **queue22, the book-run config at 30 epochs** (3B × 128 ch × crop 96 × pair=1 ×
  orient=1; 6.3 h): distance head 76.0 % val, EUs **0.877** (0.880 / 0.895 / 0.764) against
  queue17's 0.880 without the heads — −0.003, within noise, hard class +0.018. Folds running (plain
  ~10:15, ω/φ ~10:55). queue23 (100 ep) at 54.8 % at epoch 70. 10:26: queue22's plain fold 0.621 /
  0.630 (= queue17's), **ω/φ fold 0.628 / 0.645** (medians 0.639 / 0.698, 1 wrong hand) — the best
  fold of any arm; the book-run config at 30 epochs is **0.877 / 0.628 / 0.645**, +0.039 / +0.059 /
  +0.073 over the baseline. `queue24_bookrun.sh` written (queue22 with epochs=100); not launched.
  Figures regenerated on queue22's ω/φ fold; the ablation figure gains the queue21/22 rows.
- 2026-10-03 11:05: queue23 (3B × 64 ch × 100 ep × pair=1; 4.3 h): val 54.8 % (30 ep: 54.3 %),
  EUs **0.868** (0.870 / 0.889 / 0.739) against 0.866 at 30 ep — +0.002: with the plane the
  100-epoch schedule is nearly flat on precision; fold running (at 650M the schedule was +0.018 /
  +0.016 on the fold). The book run's expectation on precision is therefore ~0.88, not 0.89.
  11:26: its fold 0.612 / 0.617 (medians 0.630 / 0.678) against 0.603 / 0.612 at 30 epochs — the
  schedule is +0.009 / +0.005 on the fold with the plane, half the 650M gain. queue23 done; every
  arm of the night has landed; all four cards idle. The night's rows, figures and queue17–24 are
  staged, not committed; `73b2333c` committed, not pushed.
- 2026-10-04 14:59: **queue24, the book run** (launched by Brett 2026-10-03 17:18, GPU 1; training
  ended 14:17 after 21.0 h): val 75.7 %, EUs **0.873** (0.878 / 0.896 / 0.723), plain fold 0.619 /
  0.625, ω/φ fold **0.627 / 0.646** (medians 0.642 / 0.706, 2 wrong hands) — the 30-epoch row's
  0.877 / 0.628 / 0.645 to within 0.004; paired 100 − 30: −0.004 / −0.001 / +0.001, every CI
  across zero. The schedule is flat at the wide config. All four cards idle.
- 2026-10-04 evening: review of the book run (§11), no training. `casp16_table.py` gains map lDDT / recall / top-L
  (cached `map_scores.csv`), the 650M and 3B head rows and an ESMFold row; `casp16_esmfold.py` (new; `transformers`
  in `.venv-casp`, `facebook/esmfold_v1`) folds 54 of the 59 targets on one card in 37 min — the five over 1,000
  residues do not fit — and scores 75 units through `casp16_score.score`: 0.770 TM against our 0.647 on the 73 folded
  units it covers; with T1218's and T1269's five units from an 800-residue window, 0.778 against 0.646 over the 78.
  `casp16_ablation_figure.py`: the top row is the landed book run and the three longest labels no longer clip.
  OpenProteinSet coverage measured and MassiveFold's CASP16 archives located (§11a, the MSA route); Brett:
  OpenProteinSet for the alignments, nothing custom. Plate version 8 (the ladder on one TM axis with the ESMFold
  row and the MSA estimate; figures re-rendered on the 100-epoch run into the session scratchpad, `demos/figures/`
  untouched). Working tree only: nothing staged, nothing committed. All four cards idle.
- 2026-10-04 (late): the MSA route parked (Brett: "we'll come back to it"); §11a carries it as a resumable
  plan — the decision, the measured coverage, the unchecked items, six steps in order, the constraints.
