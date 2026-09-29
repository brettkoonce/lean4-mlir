# remote_sensing_wavelengths_demo.md — which wavelengths travel: chapter 4's CNN from EuroSAT to the Amazon and the Cerrado

Goal: a *Remote sensing* entry for the bestiary and the book's first input that
is not a photograph — thirteen Sentinel-2 bands, 443 nm to 2190 nm, ten of them
outside what a camera sees. Train chapter 4's CIFAR-CNN8-wide-BN on EuroSAT
(27,000 Sentinel-2 chips over 34 European countries, 10 land-cover classes,
all 13 bands) five times, each with a different subset of the bands as its
stem; reproduce the literature's result that in Europe the ten invisible
bands buy nothing over RGB; then score the same five nets, unchanged, on
Sentinel-2 chips cut over Rondônia and western Bahia and labelled from
MapBiomas, and ask whether the bands that carry chlorophyll and leaf water
(red edge, NIR, SWIR) hold up where the visible ones collapse — and whether
they hold up across the Cerrado's wet and dry seasons on the very same
chips. Written 2026-09-29; nothing built; no data fetched. One net, one
scorer, one preprocessor per side of the Atlantic. Zero new codegen.

Prerequisite reading: `planning/plant_lab_to_field_demo.md` §2–§4 (lab→field
is the shape: the field's number, then the fixes, each on the same
denominator), `demos/MainGwDetect.lean` (the trainer this one copies: cifar8w
with a `C`-channel stem on flat f32 records, batches gathered by index),
`planning/gw_detection_demo.md` §8 (the phase log format), and
`scripts/demos/plant_shapley.py` (the two-player exact Shapley, which becomes
a five-player one here).

## 0. The one-paragraph version

EuroSAT (Helber, Bischke, Dengel & Borth, IEEE JSTARS 2019, arXiv:1709.00029;
Zenodo 10.5281/zenodo.7711810; MIT) is 27,000 64×64 patches of Sentinel-2A
Level-1C imagery at 10 m — a 640 m square each — with all 13 bands, the
20 m and 60 m bands cubic-spline upsampled to 10 m, no atmospheric
correction, 2,000–3,000 patches in each of ten classes: AnnualCrop, Forest,
HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop,
Residential, River, SeaLake. Helber's Table IV, ResNet-50 on an 80/20 random
split: **RGB 98.57%, colour-infrared (B8/B4/B3) 98.30%, SWIR 97.05%** — the
invisible bands are not needed in Europe. The demo trains chapter 4's
0.83M-parameter CNN (the GW demo's `cifar8w`, `C := 13`, `H := W := 64`) with
five stems — RGB; RGB+NIR; the ten 10 m/20 m bands; all 13; and the seven
infrared bands alone with no visible light at all — reproduces the flat
in-domain table, then scores every arm on three parts it never saw: Amazon
chips (Rondônia, dry season), Cerrado chips (western Bahia, dry season) and
the *same* Cerrado chips in the wet season, cut on the same 640 m grid from
Sentinel-2 L1C tiles and labelled by majority vote from MapBiomas's 10 m
Sentinel collection, through a seven-class map that EuroSAT's ten classes
collapse into. Table 1 is arm × part; Table 2 is the two Brazilian classes
Europe does not have (savanna formation, forest plantation) and what each
European net calls them; the figure is one Sentinel-2 scene of the Rondônia
fishbone with the chip grid coloured by the RGB net, by the 13-band net, and
by MapBiomas.

## 1. Why this and not another remote-sensing demo

- **Nothing in the codegen moves.** `.convBn 13 16 3 1 .same` is the GW
  demo's `.convBn 2 16 …` with a different literal; the GW build (2026-09-11)
  took a 2-channel non-square stem without a change. Flat f32 records, the
  index gather, `scoreSet`, the recipe (Adam 1e-3, one-epoch warmup, cosine,
  batch 64) are `MainGwDetect.lean` verbatim. The input is 13 × 64 × 64; the
  CIFAR net is the CIFAR-family op set at CIFAR shapes.
- **The wavelength question is a physics question with a control.** Every
  arm is the same net on the same chips with bands removed at the stem, so
  the only variable is which part of the spectrum the net was allowed to see.
  In-domain the literature says it does not matter; out of domain nobody has
  the number. The Cerrado wet/dry pair is a second control inside Brazil:
  the labels are annual and the chips are the same pixels, so a swing is the
  input and nothing else.
- **It is the plant move one step further.** PlantVillage → PlantDoc changed
  the photography; here the photography is identical (one instrument, one
  processing level, one chip geometry on both sides) and the *world* changed:
  latitude, phenology, vegetation type, sun angle, atmosphere. The section
  can say what the plant section could not — the gap is the scene, not the
  camera.
- **Both datasets are open, no account for EuroSAT and MapBiomas** (Zenodo;
  a public Google Cloud Storage bucket, CC BY 4.0). The Sentinel-2 tiles for
  Brazil come from the Copernicus Data Space Ecosystem (free registration;
  STAC search is open; 4 concurrent downloads, 12 TB/month on the free tier —
  this needs ~8 GB). ⚠ The user makes the CDSE account; a session does not
  register anywhere on the user's behalf.
- Cost: EuroSAT is 16,200 training chips of 13 × 4,096 floats — the GW set
  was 40,760 of 2 × 8,192 and trained cifar8w in 12 s/epoch, so an arm is
  ~2–3 min for 20 epochs; the five arms × three seeds are under an hour on
  one card. Nothing needs asking about except the Brazil download's size
  (once) and the Phase 4 ladder.

## 2. The data

**EuroSAT MS** — `EuroSAT_MS.zip` from Zenodo 7711810 (~2 GB; torchgeo mirrors
it as `EuroSATallBands.zip`, SHA256 `751f070f…df59`), 27,000 GeoTIFFs in ten
class folders, each 64×64×13 uint16. Band order in the tif, per torchgeo's
`all_band_names`: **B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A** —
B8A is the *thirteenth* plane, not the ninth. ⚠ Gate 0 verifies the order
from the data, not the docs: over vegetation chips B08 ≈ B8A ≫ B04, B10
(cirrus) is near zero on clear chips, B01 is the lowest visible-range plane.
Value range: L1C TOA reflectance × 10,000 as uint16 (GDAL quicklook in the
README scales 0–2,750); no radiometric offset (the chips predate processing
baseline 04.00, see §3.3). Licence: MIT, plus the Copernicus Sentinel data
terms (free, full and open access, attribution "contains modified Copernicus
Sentinel data"). Split: EuroSAT ships none; use torchgeo's lists
(`eurosat-{train,val,test}.txt`, from Neumann et al. 2019, arXiv:1911.06721,
16,200 / 5,400 / 5,400) so the number is comparable to the published
transfer-learning line, and run the ArASL 16×16 nearest-neighbour audit on
the split before training — EuroSAT chips were cut from ~a few hundred tiles
and adjacent chips of one field are the leak candidate. The audit's number
goes in the runs README either way.

| plane | band | centre | native res | what it sees |
|---|---|---|---|---|
| 0 | B01 | 443 nm | 60 m | coastal aerosol |
| 1 | B02 | 490 nm | 10 m | blue |
| 2 | B03 | 560 nm | 10 m | green |
| 3 | B04 | 665 nm | 10 m | red — chlorophyll absorbs |
| 4 | B05 | 705 nm | 20 m | red edge 1 |
| 5 | B06 | 740 nm | 20 m | red edge 2 |
| 6 | B07 | 783 nm | 20 m | red edge 3 |
| 7 | B08 | 842 nm | 10 m | NIR — leaf structure reflects |
| 8 | B09 | 945 nm | 60 m | water vapour |
| 9 | B10 | 1375 nm | 60 m | cirrus |
| 10 | B11 | 1610 nm | 20 m | SWIR 1 — leaf water absorbs |
| 11 | B12 | 2190 nm | 20 m | SWIR 2 |
| 12 | B8A | 865 nm | 20 m | narrow NIR |

**The five arms**, as plane lists into the 13-plane record:

| arm | bands | planes | C | published tie |
|---|---|---|---|---|
| `rgb` | B04 B03 B02 | 3 2 1 | 3 | Helber RGB 98.57 (R50) |
| `rgbn` | + B08 | 3 2 1 7 | 4 | Helber CI 98.30 is B08/B04/B03 |
| `ms10` | the 10 m + 20 m bands: B02–B08, B8A, B11, B12 | 1 2 3 4 5 6 7 12 10 11 | 10 | — |
| `all` | all 13 | 0…12 | 13 | — |
| `ir` | B05 B06 B07 B08 B8A B11 B12 — no visible light | 4 5 6 7 12 10 11 | 7 | Helber SWIR 97.05 is the nearest |

`ir` is the physics arm: red edge, NIR and SWIR, the bands where chlorophyll
and leaf water are, and nothing a camera sees. If it travels and `rgb` does
not, the section has its sentence.

**Sentinel-2 L1C over Brazil** — from the Copernicus Data Space Ecosystem
(STAC `https://stac.dataspace.copernicus.eu/v1/`, collection `sentinel-2-l1c`;
assets via S3 `eodata` with free keys, or HTTPS with a token). Three
region–season parts, each from two or three 110 km tiles chosen at Phase 0
by lat/lon box and cloud cover < 5%:

| part | where | when | why |
|---|---|---|---|
| `amazon_dry` | Rondônia arc of deforestation, around Ariquemes / Ji-Paraná (~10°S 62°W) | Jul–Aug | forest / pasture edges at the fishbone; dry season is the only cloud-free window |
| `cerrado_dry` | western Bahia, MATOPIBA soy frontier around Luís Eduardo Magalhães (~12°S 46°W) | Jul–Aug | savanna, pasture, centre-pivot annual crop, gallery forest |
| `cerrado_wet` | the same tiles | Feb–Apr, least cloudy scene | the same pixels green; the wet/dry control |

Each L1C SAFE is ~700 MB; ~8 tiles ≈ 6–8 GB once. Chips are cut on a 640 m
grid aligned to the tile's UTM grid (171 × 171 per tile), the 20 m and 60 m
bands upsampled to 10 m with the same cubic-spline interpolation EuroSAT
used (⚠ not nearest, not bilinear: a resampling mismatch is a texture
domain shift in B01/B05–B07/B09–B12), the radiometric offset read from
`MTD_MSIL1C.xml` and removed (§3.3), the cloud probability from `s2cloudless`
(Sentinel Hub's pixel classifier, trained on L1C TOA reflectance, one pip;
takes B01 B02 B04 B05 B08 B8A B09 B10 B11 B12 at reflectance 0–1); a chip is
kept if its max cloud probability < 0.2.

**MapBiomas** — Brazil's annual land-use map. Two products in the same
public bucket, same legend: Collection 11 at 30 m from Landsat, 1985–2025
(`https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/collection11/lulc/coverage/brazil_coverage/brazil_coverage-col11_<year>.tif`)
and **Collection 4 at 10 m from Sentinel-2, 2017–2025**
(`…/initiatives/brasil/lulc_10m/collection4/coverage/brazil_coverage/brazil_coverage-col4_10m_<year>.tif`).
Use the 10 m one — the label grid is the chip grid. Whole-Brazil GeoTIFFs;
read the tile's window through `rasterio` with `/vsicurl/`, never download
the file (the 10 m one is tens of GB). Licence CC BY 4.0; cite Souza et al.
2020, *Remote Sensing* 12(17):2735. Year: the chip's acquisition year
(2023 or 2024, see §10). A chip's label is the majority class over its 64×64
= 4,096 label pixels, kept only if that class covers ≥ 80% (purity); the
purity and the full class histogram are stored per chip.

**The class map**, EuroSAT → MapBiomas, spelled once in
`scripts/datasets/preprocess_rs_brazil.py` and printed by Gate 2:

| scored class | EuroSAT (train) | MapBiomas codes (score) |
|---|---|---|
| annual crop | AnnualCrop | 19 Temporary crop = 39 soybean, 20 sugar cane, 40 rice, 62 cotton, 41 other |
| perennial crop | PermanentCrop | 36 Perennial crop = 46 coffee, 47 citrus, 35 palm oil, 48 other |
| forest | Forest | 3 Forest formation |
| herbaceous | HerbaceousVegetation | 12 Grassland |
| pasture | Pasture | 15 Pasture |
| built | Residential ∪ Industrial | 24 Urban area |
| water | River ∪ SeaLake | 33 River, lake and ocean |

Highway has no MapBiomas class and is excluded from the argmax on the
Brazil parts (the plant demo's restricted argmax); 21 Mosaic of uses, 11
Wetland, 30 Mining, 6 Floodable forest, 5 Mangrove and 25 Other non-vegetated
chips are dropped from the scored parts and counted. Two MapBiomas classes
Europe does not have are kept as **diagnostic parts, never scored as
right or wrong**: **4 Savanna formation** (the Cerrado itself — woody and
herbaceous mixed; does a European net call it forest, herbaceous or
pasture, and does the answer change with the season?) and **9 Forest
plantation** (eucalyptus and pine rows; forest or crop?). Table 2 is what
each arm calls them.

**Records.** `scripts/datasets/preprocess_rs_eurosat.py` and
`preprocess_rs_brazil.py` write the GW format so `MainGwDetect`'s loader reads
them unchanged:

    data/rs/eurosat_{train,val,test}.bin      f32 [N, 13, 64, 64], reflectance (DN / 10000), all 13 planes
    data/rs/labels_eurosat_{train,val,test}.bin   int32, EuroSAT's 10 classes in torchgeo's alphabetical order
    data/rs/{amazon_dry,cerrado_dry,cerrado_wet}.bin        f32 [N, 13, 64, 64], same planes, same scaling
    data/rs/labels_{part}.bin                 int32 in the 7-class map; savanna = 7, plantation = 8 (diagnostic)
    data/rs/meta_{part}.npz                   per chip: tile, date, UTM origin, lat/lon, purity, class histogram,
                                              cloud probability, processing baseline, offset applied
    data/rs/manifest_rs.json                  band order check, per-band mean/std of EuroSAT train, censuses

Sizes: EuroSAT f32 all bands 27,000 × 13 × 4,096 × 4 B = 5.75 GB resident
(251 GB of host RAM; the plant demo's u8 route is not needed); a Brazil part
capped at 2,000 chips per class is ≤ 3 GB. Band selection is at gather time:
`gather` extracts each arm's planes per chip with `F32.sliceImages` (a
`ByteArray.extract`, memcpy, no Lean push loop — `lean_host_push_cost`), 13 × 64
extracts per batch. One 13-plane file per part serves every arm.

Normalisation: per-band mean/std of the EuroSAT *training* split, applied
to every part including Brazil. The mismatch is part of what is measured;
re-standardising on the target is a separate row if anyone wants it (§9).

## 3. What is and is not "just a dataloader"

1. **The stem is the arm.** `NetSpec` with `.convBn C 16 3 1 .same` for
   C ∈ {3, 4, 7, 10, 13}; the rest of cifar8w verbatim (four conv-conv-pool
   stages at 16/16/32/32, flatten 32 × 4 × 4 = 512, dense 512-512-10). Ten-way
   head on EuroSAT always; the 7-class collapse happens in the scorer (built =
   max of the Residential and Industrial logits, water = max of River and
   SeaLake, Highway masked out), so every part is scored by the same trained
   net through one function.
2. **One processing level.** EuroSAT is L1C. CDSE also serves L2A; the
   Planetary Computer serves *only* L2A. ⛔ Pull L1C. An L2A chip is surface
   reflectance after atmospheric correction, and the difference from TOA is
   largest in exactly the bands the demo is about (B01, B02, the SWIR pair).
3. **The +1000.** Processing baseline 04.00 (25 Jan 2022) added a radiometric
   offset so dark pixels do not clip: L1C DN = reflectance × 10,000 + 1,000,
   declared per band as `RADIO_ADD_OFFSET` in `MTD_MSIL1C.xml`. ⛔ Read it
   from the metadata and subtract; do not decide by date — CDSE's
   Collection-1 reprocessing (2023–24) rewrote the 2015–2021 archive under
   baseline 05.xx *with* the offset, so a 2017 tile may carry it too. Gate 2
   overlays the per-band histograms of open water in Brazil on EuroSAT's
   SeaLake chips: a wrong offset is a 0.1-reflectance shift in every band.
4. **Resampling.** Cubic-spline from 20 m/60 m to 10 m as EuroSAT did
   (`scipy.ndimage.zoom(order=3)` on the tile window before cutting; verify
   at Gate 0 that EuroSAT's B01 plane has the smooth, ringing-free texture
   of a spline and not the blocks of nearest).
5. **Cloud and cloud shadow.** L1C has no scene classification. `s2cloudless`
   per chip; shadow is not caught — the dry-season parts are near-cloud-free
   anyway, the wet-season Cerrado part is the exposed one, and its cloud
   histogram goes in the README. A cloudy `cerrado_wet` chip is dropped
   *together with its dry twin*, so the pair stays the same pixels.
6. **Chip purity and the 30 m ghost.** Collection 4 is 10 m so a 64×64 chip
   has 4,096 label pixels; purity ≥ 0.8 keeps edges out. ⚠ If Collection 4
   turns out unavailable for the chosen year, fall back to Collection 11 at
   30 m (455 label pixels per chip) with the same rule, and say so.
7. **Seasons are the same chips.** `cerrado_wet` and `cerrado_dry` are cut
   from the same tile grid and joined on chip id; the scorer refuses a
   wet/dry comparison whose chip sets differ.
8. **The Brazil-trained ceiling.** A cifar8w trained on Brazil chips
   themselves (5-fold over the union of the three parts, all 13 bands) says
   whether the 7-class task is learnable from these labels at all; if that
   arm is under ~85%, the labels are the problem and the zero-shot numbers
   mean nothing. It is Gate 3, and a row of Table 1.
9. **No Brazil chip touches European training** in Table 1's zero-shot rows;
   only the Phase 4 fine-tune sees them, five-fold, never the ones it is
   scored on; the scorer asserts it from the fold ids as the plant scorer
   does.

The Lean side is one file, `demos/MainRsBands.lean` (`lake exe rs-bands
[arm=rgb|rgbn|ms10|all|ir] [train=eurosat|brazil] [fold=0..4] [epochs=20]
[batch=64] [lr=0.001] [seed=1] [init=<prefix>] [labels=<N>] [tag=] [out=]
[eval]`), copied from `MainGwDetect.lean`: `C`, `H`, `W` from the arm, the
plane list in `gather`, `scoreSet` over every part in `data/rs/`, writing
`[N, 10]` logits per part. Registered in `lakefile.lean` beside `gw-detect`;
`scripts/gates/check_target_names.sh` after.

## 4. The arms, as tables

**Table 1 — accuracy (restricted 7-way argmax, Wilson) and macro-F1, arm ×
part; the EuroSAT column is the 10-way test accuracy:**

| arm | C | EuroSAT test (10-way) | amazon_dry | cerrado_dry | cerrado_wet | wet − dry |
|---|---|---|---|---|---|---|
| `rgb` | 3 | | | | | |
| `rgbn` | 4 | | | | | |
| `ms10` | 10 | | | | | |
| `all` | 13 | | | | | |
| `ir` | 7 | | | | | |
| Brazil-trained ceiling, `all`, 5-fold | 13 | — | | | | |
| published: Helber 2019 R50, 80/20 random | | RGB 98.57 / CI 98.30 / SWIR 97.05 | — | — | — | |

Three seeds on `rgb` and `all` (the two rows the section's sentence rests
on), one elsewhere. Per-class recall per part in the runs README; in the
section, the forest and pasture rows, because that pair is the Cerrado's
question (a European net that calls dry-season pasture "herbaceous" and
gallery forest "forest" is doing physics; one that calls savanna "forest" in
March and "pasture" in August is doing colour).

**Table 2 — what Europe calls the classes it does not have (fraction of
chips per predicted class, the diagnostic parts):**

| part | arm | forest | herbaceous | pasture | annual crop | perennial | built | water |
|---|---|---|---|---|---|---|---|---|
| savanna formation, dry | `rgb` / `all` / `ir` | | | | | | | |
| savanna formation, wet | `rgb` / `all` / `ir` | | | | | | | |
| forest plantation | `rgb` / `all` / `ir` | | | | | | | |

No right answer exists for these rows and the section says so; the claim is
about *stability*: an arm whose savanna row moves between the seasons is
reading the season.

**Table 3 (Phase 4, optional) — fine-tune from the European checkpoint on N
Brazil labels, 5-fold, scored on held-out Brazil chips:**

| arm | N = 0 (Table 1) | N = 100 | N = 300 | N = 1,000 | all (ceiling) |
|---|---|---|---|---|---|
| `rgb` | | | | | |
| `all` | | | | | |
| `ir` | | | | | |

The question: does the infrared arm need fewer labels to adapt, i.e. is its
European feature space closer to Brazil's.

## 5. The instrument

`scripts/demos/rs_score.py <logits.bin> --part eurosat_test|amazon_dry|cerrado_dry|cerrado_wet
[--pair <other logits>] [--json]`: 10-way accuracy on EuroSAT; on a Brazil
part the 7-way collapse, restricted argmax, accuracy with Wilson, macro-F1,
per-class recall, the confusion matrix and its top pairs, the diagnostic
rows for savanna and plantation chips; `--pair` joins two parts on chip id
and reports the per-chip agreement and the per-class swing. Fold hygiene
from `meta_*.npz` for the Phase 4 rows.

`scripts/demos/rs_shapley.py`: the plant demo's exact Shapley with **five
players = band groups** — visible (B02–B04), red edge (B05–B07), NIR (B08,
B8A), SWIR (B11, B12), atmospheric (B01, B09, B10) — on the `all` arm: 32
coalitions, a removed group set to its EuroSAT training mean, 32 forward
passes per chip through the eval graph (a probe part of masked twins scored
by `rs-bands eval`), φ per group for the true-class logit, efficiency to the
float. Run on 500 chips per part: which group carried the decision in
Europe, and which in Brazil. The 13-player exact version (8,192 coalitions)
is feasible at these shapes (~4 M forwards for 500 chips, minutes) and is
the runs-README extra if the group answer wants a finer one.

`scripts/demos/rs_figure.py`: the figure from the tile, MapBiomas and the
logits.

## 6. Figure and section

Figure, two rows. (a) A 10 × 10 km window of the Rondônia tile: true colour
(B04/B03/B02) | the 640 m chip grid coloured by the `rgb` net | by the `all`
net | by MapBiomas — the fishbone is the object itself, the reader sees which
net found the pasture between the forest ribs. (b) The same for a 10 × 10 km
window of the Cerrado tile in the dry season and the wet season side by
side, the `rgb` and `ir` nets, with the false-colour (B08/B04/B03) rendering
of the wet and dry scene between them so the reader sees what the infrared
arm sees. `demos/figures/remote_sensing_wavelengths.jpg` and a copy to
`blueprint/src/figures/demos/`.

Section: *Remote sensing — demo: which wavelengths travel, EuroSAT to the
Amazon and the Cerrado*, a `\subsection` after *Agriculture* (content.tex
line ~14374 as of 2026-09-29; the same move again — the chapter net on the
field's dataset, then the field's other field). Shape: two lead paragraphs
(the instrument and the 13 bands, four sentences on why a leaf is bright at
842 nm and dark at 1610 nm and that colour-infrared film found camouflage on
the same principle in the 1940s; what changes in the net and what does not —
the stem literal, nothing else), Table 1, the figure, Table 2 and the
paragraph that reads it, one closing paragraph on what travelled and what
did not, one sentence on what a chip classifier is not (a map: no boundaries,
no change date, no area). No acts, no gates, no plan in the book. Data
appendix: three rows (EuroSAT, Sentinel-2 L1C via CDSE, MapBiomas
Collection 4) and one "Building the EuroSAT and Brazil chip sets" entry — the
Zenodo download, the split lists, the audit, the tiles and dates, the
offset, the resampling, the cloud rule, the purity rule, the class map, the
censuses. Bestiary: no remote-sensing row exists in
`planning/bestiary_candidates.md`; this adds *Remote sensing* as a category
with EuroSAT as its dataset row.

## 7. Phases

```
Phase 0 (½ session, CPU):   .venv-rs (numpy, rasterio, scipy, pystac-client, s2cloudless, Pillow);
                            scripts/datasets/download_rs.sh: EuroSAT_MS.zip from Zenodo (~2 GB),
                            torchgeo's three split lists; preprocess_rs_eurosat.py: the 13-plane
                            records, per-band mean/std, the ArASL nearest-neighbour audit, a
                            true-colour and a false-colour quicklook strip
                            Gate 0: 27,000 tifs, 10 folders, 2,000–3,000 each, 64×64×13 uint16;
                                    band order verified from the data (B8A last: B08 ≈ B8A over
                                    Forest chips, B10 ≈ 0 everywhere); no offset (SeaLake B12
                                    median < 500 DN); split lists partition the 27,000 exactly;
                                    audit printed; quicklook eyeballed (Residential looks like
                                    Residential)
                            ✅ 2026-09-29 PASS (runs/2026-09-29-rs-gate0/gate0.log, 78 s): 27,000 tifs,
                            3,000/3,000/3,000/2,500/2,500/2,000/2,500/3,000/2,500/3,000 in class order;
                            (13, 64, 64) uint16; from the chips: forest B08/B8A 0.87, B08/B04 6.4, B10
                            median 11 DN, B08's most-correlated plane is 12 (B8A), SeaLake B12 median
                            34 DN, global min 0 → the torchgeo order with B8A last, no offset. The
                            torchgeo lists partition exactly (16,200 / 5,400 / 5,400). Train mean /
                            std per band in manifest_rs.json (B08 0.230 ± 0.112, B10 0.0012 ±
                            0.0005). ⚠ The ArASL 16×16 grey audit is uninformative here: 32.9% of
                            val chips have a train chip within 6 grey levels but the nearest is the
                            same class only 25% of the time and 1-NN scores 25.8% — 640 m land
                            patches are low-contrast at 16×16, not duplicates; reported, not a leak.
                            The archive came from torchgeo's HF mirror (sha256 751f070f…, 2 GB in
                            ~4 min; Zenodo served 0.4 MB/s). MapBiomas Collection 4 (10 m) 2023
                            exists at the URL pattern (4.6 GB, tiled 256², LZW, EPSG:4326 at
                            8.98e-5°), downloaded to data/rs/mapbiomas/; the CDSE STAC search is
                            open (no key) and every 2023 scene over both boxes is baseline 05.10 —
                            offsets come from MTD_MSIL1C.xml as planned.
Phase 1 (½ session, GPU):   demos/MainRsBands.lean from MainGwDetect; the five arms, seed 1, 20 ep
                            2026-09-29: one epoch is 2.6 s on a 4060 Ti (253 steps), so every arm
                            runs at three seeds. ⚠ The GW gather has no augmentation and the plain
                            ladder memorises (train loss 0.03): seed 1 plain = rgb 92.24 / rgbn 95.33
                            / ms10 96.85 / all 95.96 / ir 95.09 (runs/2026-09-29-rs-phase1/noaug/) —
                            under Gate 1 for rgb, and already the opposite of Helber's pretrained-R50
                            ordering: from scratch, every invisible-band arm beats RGB in Europe,
                            the no-visible-light arm included. Fix: `F32.dihedralGather` (ffi/
                            f32_helpers.c), each training chip under a random one of the eight
                            symmetries of the square (`aug=1`, default; `aug=0` is the plain row).
                            Augmented, 20 ep, three seeds (ep20/): rgb 94.30/94.39/94.30, rgbn
                            96.46/96.41/96.50, ms10 97.04/97.19/97.09, all 97.30/97.09/97.13, ir
                            96.20/96.02/96.06. ⚠ RGB was not converged at 20 epochs: rgb at 40 /
                            80 / 160 epochs = 96.20 / 97.19 / 97.30 (e40/, e80/, e160/), all at 40
                            / 80 = 97.70 / 98.26 — so the schedule is 80 epochs for every arm
                            (RGB has converged there, +0.1 at 160) and the in-domain gap that
                            remains (~1 point) is spectrum, not schedule. Gate 1's "rgb ≥ 97" is
                            met at 80 epochs; the published tie it was written against is a
                            pretrained ResNet-50 and the 13-band arm from scratch lands on it
                            (98.26 vs 98.57). The ladder at 80 epochs × three seeds is the table.
                            Gate 1: rgb ≥ 97.0 on the test list (Helber 98.57 at R50 on a random
                                    80/20; a 0.83M-param net a point under is expected, five
                                    under is a wrong plane list); all within 1 of rgb; ir ≥ 95
                                    (Helber SWIR 97.05); a shuffled plane list reads as rgb ≈ ir
Phase 2 (1 session, CPU):   preprocess_rs_brazil.py: tile choice by box + cloud + date (STAC),
                            SAFE download (user's CDSE keys), offset from MTD, spline upsample,
                            640 m grid, s2cloudless, MapBiomas Collection 4 window reads, majority
                            + purity, class map, caps, meta; the three parts + two diagnostic parts
                            Gate 2: per-band histograms of Brazil water chips overlay EuroSAT
                                    SeaLake's (offset and level right); B01 texture matches
                                    (resampling right); class census per part printed with the
                                    dropped codes; wet/dry chip sets identical; quicklooks of ten
                                    chips per class per part eyeballed against MapBiomas
                            ✅ 2026-09-29 (runs/2026-09-29-rs-phase2/, README.md = the dataset notes):
                            18 scenes, all baseline 05.10/05.11 with offset −1000 read from MTD;
                            amazon_dry 3,826 chips (forest 2,000 capped, pasture 1,775, crop 41,
                            one water chip — the water sentinel is a note), cerrado_dry/wet 3,620
                            each on identical chip ids (savanna 2,000 capped as the diagnostic
                            row, pasture 884, crop 520, gallery forest 198); purity median 0.985.
                            Amazon forest band means sit on EuroSAT's forest medians (level and
                            offset right); the same Cerrado chips read red 0.121 → 0.065, SWIR1
                            0.322 → 0.217, NIR 0.229 → 0.294 from September to March. ⚠ The first
                            pass lost two Rondônia tiles to fire-season smoke at ESA cloud 0.0%
                            (s2cloudless flagged every window): now June first for the Amazon,
                            September for the Cerrado dry, the STAC search paged (351 scenes,
                            not 200), a next-date fallback, candidates drawn inside the swath.
Phase 3 (½ session, GPU):   score the five European arms on the three parts; the Brazil-trained
                            ceiling (5-fold); Tables 1 and 2
                            2026-09-29, seed 1 at 80 epochs (runs/2026-09-29-rs-phase3/, rs_table.py):
                              arm    EuroSAT  amazon_dry  cerrado_dry  cerrado_wet   pasture recall (amazon)
                              rgb     97.06     87.01       33.66        37.92        73.6
                              rgbn    97.89     87.87       35.15        38.42        74.9
                              ms10    98.52     96.31       35.45        59.91        93.6
                              all     98.20     96.65       34.28        47.99        94.8
                              ir      98.00     93.57       29.15        42.80        87.1
                            ⭐ In the Amazon the invisible bands travel: +9.6 over rgb, pasture
                            recall 74 → 95; NIR alone (rgbn) does not do it, red edge + SWIR do.
                            ⭐ The dry Cerrado breaks every arm on the 7-way map (pasture recall
                            1–16%: a September Cerrado pasture is "herbaceous" to a European net;
                            with herbaceous ∪ pasture merged rgb reads 74.8) — partly taxonomy.
                            ⛔ Season stability REFUTED: same prediction in September and March
                            on 39% (rgb) / 37% (ir) / 31% (ms10) / 22% (all) of chips — the
                            multispectral arms are better in both seasons, not stabler; savanna
                            is "herbaceous" in September and 34–42% "forest" in March for every
                            arm.
                            ✅ FINAL, three seeds (runs/2026-09-29-rs-phase3/README.md, tables_final.txt):
                              arm    EuroSAT        Amazon June    Cerrado Sept   Cerrado March  same answer
                              rgb    97.09 ± 0.05   87.37 ± 0.92   33.15 ± 0.46   37.66 ± 0.34   38%
                              rgbn   98.04 ± 0.12   89.30 ± 1.35   34.61 ± 0.55   39.04 ± 0.53   41%
                              ms10   98.40 ± 0.08   96.95 ± 0.55   32.76 ± 2.25   59.21 ± 0.50   27%
                              all    98.33 ± 0.11   97.02 ± 0.46   35.39 ± 2.45   53.82 ± 4.27   24%
                              ir     97.94 ± 0.09   91.71 ± 1.48   29.03 ± 2.52   39.84 ± 3.05   39%
                            Ceiling from scratch on brazil_all (5 folds): rgb 94.19 ± 0.72, ms10
                            95.65 ± 0.91, all 95.83 ± 0.64. Fine-tune from EuroSAT s1: 300 labels
                            rgb 90.0 / all 91.0 / ir 89.2; all labels 94.3 / 95.1 / 94.9.
Phase 4 (½ session, GPU):   seeds on rgb and all; Table 3's ladder; rs_shapley.py on 500 chips
                            per part
                            ✅ 2026-09-29 (runs/2026-09-29-rs-phase4/): five-group exact Shapley on the
                            all arm (300 chips per part, residual ≤ 2e-6): visible is the largest
                            player everywhere (32–35% of |φ|), SWIR 13% in Europe → 21–22% in
                            Brazil, red edge 15% → 20% in the dry Cerrado, and the atmospheric
                            trio has NEGATIVE mean φ in Brazil (−3.0 Amazon, −1.5 March) — it
                            carries Europe's atmosphere; hence ms10 (no B01/B09/B10) is the best
                            zero-shot arm abroad and equals all in-domain (ceiling 95.65 / 95.83).
                            Gate 3: ceiling ≥ 85 (else labels, not physics); every European arm's
                                    water recall > 90 on amazon_dry (the Madeira and the Ji-Paraná
                                    are in the tiles; a net that cannot find a river has a
                                    plumbing fault, not a domain gap); rgb on cerrado_wet vs
                                    cerrado_dry differ (if not, the seasons were not what the
                                    dates said — check the scenes)
Phase 5 (½ session):        figure, section, appendix, demos/README, runs README, lakefile,
                            check_target_names.sh
                            2026-09-29: figure = demos/figures/remote_sensing_wavelengths.jpg (three
                            rows × scene | rgb | ir | all | MapBiomas, seed-1 weights; the windows
                            are the most mixed forest/pasture 10 km of 20LPQ and the most mixed
                            mosaic of 23LKF, chosen from MapBiomas inside the swath); section
                            written after Agriculture; appendix rows; READMEs under runs/.
```

## 8. Gates that fail loudly

- The band order. Every arm is a plane list; a wrong list makes `ir` see
  visible light and `rgb` see cirrus. Gate 0 derives the order from the
  physics of the chips (B08 ≈ B8A over forest, B10 ≈ 0), and Gate 1's
  "rgb ≈ ir means shuffled" catches the rest.
- The offset. A Brazil chip with the +1000 still in it is 0.1 brighter in
  every band; over water that is the whole signal. Gate 2's histogram
  overlay is the check; the meta records the baseline and the offset applied
  per chip so the fault is attributable.
- The level. An L2A tile downloaded by mistake passes every shape check and
  fails Gate 2's B01/B02 histogram (atmospheric correction removes most of
  the blue path radiance).
- The class map is the whole of the zero-shot table. A MapBiomas code with
  no row is dropped and counted, never silently mapped; the two diagnostic
  codes are never scored.
- Water recall is the plumbing sentinel (Gate 3): a river is the easiest
  class on Earth in SWIR; if a European net misses it, look at the chips
  before the physics.
- Fold hygiene on Table 3: the scorer reads fold ids and refuses to score a
  training chip.
- A season swing in `rgb` and none in `ir` is the headline; a swing in both
  is a result; a swing in neither is Gate 3's last clause.

## 9. Out of scope

- Per-pixel segmentation (the UNet on chips; MapBiomas would be the mask).
  A later session if the chip table wants a map.
- Time series and crop typing (DENETHOR, Sen4AgriNet, PASTIS): a temporal
  model, not this net.
- Sentinel-1 SAR as extra channels: a plane list away once a SAR part exists
  (CerraData-4MM ships it); a flag, not a phase.
- Pretrained Sentinel-2 encoders (SSL4EO, Prithvi, the foundation-model
  route): the section's one sentence on what the field does instead.
- Deforestation *change* detection against PRODES / DETER: two dates and a
  difference; a different demo. The Cerrado wet/dry pair (same windows, two
  dates, per-chip agreement in `rs_score.py --pair`) is the first half of it;
  the user flagged this 2026-09-29 as the next trick worth a section.
- Re-standardising on the target's band statistics: one row if a reader
  asks; the plan measures the shift, it does not paper over it.
- BigEarthNet (590k chips, multi-label, 66 GB) and So2Sat LCZ42: EuroSAT is
  the 2 GB cousin and the box is shared.
- Deploy.
- Redistributing any of it: the scripts fetch from Zenodo, CDSE and the
  MapBiomas bucket.

## 10. Notes before starting

- ⛔ `lake exe rs-bands` is not a `lake run` job. Runs are minutes; the Phase
  2 download is ~8 GB once and the Phase 4 ladder is ~an hour; ask before
  those two, not before Phase 1.
- **The year.** MapBiomas Collection 4 (10 m) covers 2017–2025; choose the
  chip year to match a label year, and take the Cerrado wet and dry scenes
  from the *same* year so the annual label is the same. 2023 is the default
  (Collection 4's map for it is final, S2A+S2B both flying, both scenes
  under baseline 05.xx). ⚠ Verify the Collection 4 GeoTIFF for that year
  exists at the URL pattern before cutting anything; fall back to
  Collection 11 (30 m) per §3.6.
- **The tiles.** Named by lat/lon box, not tile id, until Phase 0 looks at
  cloud cover; the Rondônia box should include the BR-364 fishbones between
  Ariquemes and Ji-Paraná and a stretch of the Madeira or Ji-Paraná river
  (Gate 3's water sentinel); the Bahia box should include centre pivots and
  gallery forest along the Rio Grande.
- **CerraData-4MM** (Miranda et al. 2025, arXiv:2502.00083; 30,291 128×128
  patches, 12 S2 bands (no B01) + 2 S1, per-pixel 7/14-class masks, Bico do
  Papagaio in Tocantins — the Cerrado–Amazon ecotone — 2022; CC BY-NC-SA 4.0,
  Kaggle `cerranet/cerradata-4mm`) is a ready-made Brazilian target with
  *human-checked* labels. Its processing level is not stated in the paper
  (GEE-sourced, likely L2A) and its B01 is missing, so it cannot be scored
  by the `all` arm as-is. It is the cross-check for Phase 2's label rule (cut
  its patches into 64×64 chips, majority-label them, compare the European
  nets' numbers on it against `cerrado_dry`'s) if its level checks out on
  download; not the primary target.
- **TreeSatAI** (Lower Saxony, 50k patches, S2 + S1 + 20 cm aerial, 20 tree
  species) is the "German forest, Sentinel-2" dataset by name; its S2 patches
  are 6×6 pixels and its label space does not cross the Atlantic. A side
  panel on which band tells spruce from pine, if the section wants a
  European second act; not this plan.
- Colour-infrared film (Kodak Aerochrome's ancestors, 1940s) is the
  four-sentence physics: healthy leaves reflect NIR strongly because of the
  spongy mesophyll, absorb red for chlorophyll, and absorb at 1.45 and
  1.94 µm for water, so the SWIR pair reads leaf water. Vegetation indices
  (NDVI, Rouse et al. 1974, Landsat-1) are ratios of exactly these planes;
  the net is being asked to find them itself.
- Every published number in §0 and §4 is from Helber et al.'s Table IV as
  read on 2026-09-29 (RGB 98.57, CI 98.30, SWIR 97.05, ResNet-50, 80/20);
  Neumann et al.'s EuroSAT numbers on the 60/20/20 lists go in the table
  with their split named when the section is written.
- ⚠ s2cloudless takes reflectance 0–1 in *its* band order (B01 B02 B04 B05
  B08 B8A B09 B10 B11 B12); feed it after the offset and before the
  standardisation.
- The demo's parts are u16-sized data stored as f32 for the loader's sake;
  if disk becomes the constraint (121 GB free on 2026-09-29), the
  preprocessor keeps uint16 and the C batch helper converts, the plant
  demo's route — not the first build.
