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

Status 2026-10-01: §5 scoring validated on nine field models (table in §5); §3 steps 1 and 3
done (26,228 train / 1,380 val chains, G1 passed). Steps 2, 4–6, the Lean op (§4), training and
the fold not started.

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
6. CA-only fold, CA-lDDT on both sides. Our fold is a CA trace. CASP's lDDT column is all-atom
   and counts an atom the model lacks as lost, so it would punish a CA trace for not having
   side chains. The headline is therefore CA-lDDT (OpenStructure `--bb-lddt`, which is CA-only;
   §5), computed for the field's models and ours by the same command; the all-atom column is
   kept for the field as the official number. TM-score is CA-based already.
7. One sentence the book owes: lDDT is a distance score and cannot see a mirror image;
   TM-score superposes with a proper rotation and can. A distance-geometry fold has that
   ambiguity, so the fold step scores both hands and keeps the right-handed one.
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
| `pdb/<entity>.cif.gz` | per-chain coordinates (step 2) | RCSB ModelServer, `casp16_fetch_chains.py` |

`scripts/datasets/download_casp16.sh` fetches all of it (idempotent, ~60 MB). The Python side
runs in `.venv-casp` (numpy, pandas, gemmi; torch-cpu + fair-esm to be added for step 5).

EU census: 85 EUs in the domain table (30 easy / 47 medium / 8 hard; length 37–1693, median
193; 31 with a public PDB id); 83 of them carry Phase-1 rows in the score table (T1214-D1 and
T1249v2-D1 were never scored), which is the plate's "all 83". The assessment paper's 74 is
Phase 1 at paper time. The 83 scored EUs are what we score against.

Featured EUs (model 1, official all-atom lDDT): T1235-D1 easy 106 aa, field median 0.87;
T1267s1-D1 medium 157 aa, median 0.65, MULTICOM best at 0.76; T1226-D1 hard 123 aa, median
0.37, ColabFold baseline 0.58 against AF3-server 0.36. Placeholders until the final pick (§9).

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
 6  targets.py       the same labels and embeddings for the 85 EUs from raw/dom/*.pdb, sliced
                     by eu_list.csv segments
 7  Lean trainer     demos/MainDistogramCasp.lean: pairTile → residualBlock stack → 1×1 head,
                     perPixelWeightedCE; random 64×64 crops anywhere in L×L
 8  predict          tiled inference (§1.4); contacts P(d < 8 Å) = Σ bins below 8 Å;
                     top-L/5 long-range precision (|i − j| ≥ 24)
 9  fold.py          CA coordinates by Adam on −Σ log p_ij(‖x_i − x_j‖) + clash + chain
                     terms, both hands, 5 restarts → PDB CA records
10  casp16_score.py  CA-lDDT / TM-score / GDT_TS for ours; `field <EU>` rescoring every group
                     the same way; join with the official LDDT column for the strip plot
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

```
pairTile (L := 64) (inDim := 489) (dim := 64)        -- 2 × dense 489→64 + outer sum, [B, 64, 64, 64]
residualBlock 64 64 16 1                             -- the chapter body, stride 1 throughout
conv2d 64 66 1 .same .identity                       -- 66-bin head, perPixelWeightedCE
```

Input per crop: two per-residue blocks `[64, 489]` (ESM-2 480 + index scalar + 8 sinusoids),
host-built from the cached embeddings. Labels `[64, 64]` int32. ~1.2 M parameters at 64
channels; 128 channels and 32 units are affordable (§8) and are the first ablation.

Proof items: `pairTile` forward tie + VJP (dense ∘ reduce-sum), and the whole-net step tie at
the demo bar the other Chapter-10 demos meet. Everything downstream of the tile is already
proven. The gradient check for the new op goes through the existing vjp_oracle path.

Ablation rows the table wants (each one run): ESM-2's own contact head (no training by us);
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
- G2: Lean smoke on 1k chains: loss falls, FD gradient check on `pairTile` passes, the
  per-pixel mask verified by a crop with unobserved residues contributing zero gradient.
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
  decided headline = date cut only, purge = ablation; full fetch launched.
