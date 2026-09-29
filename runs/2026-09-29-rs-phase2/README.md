# Brazil chips, 2026-09-29 — how the Amazon and Cerrado parts were made

Plan: `planning/remote_sensing_wavelengths_demo.md` §2–§3 (Gate 2). Script:
`scripts/datasets/preprocess_rs_brazil.py --tiles 6 --cands 1500 --cap 2000` (second pass,
`phase2.log`, 34 min; the first pass is `phase2_firstpass.log`, see §5). Everything below
is read from `data/rs/manifest_rs_brazil.json` and the per-chip `meta_<part>.npz`.

## 1. Sources and licences

- **Sentinel-2 Level-1C** (top-of-atmosphere reflectance, the level EuroSAT was cut from), from
  the Copernicus Data Space Ecosystem: STAC search at `https://stac.dataspace.copernicus.eu/v1/`
  (open, paged; collection `sentinel-2-l1c`), the 13 band JPEG2000s and `MTD_MSIL1C.xml` per
  scene over S3 (`eodata`, free keys from the S3 keys manager, read from `CDSE_S3_ACCESS_KEY` /
  `CDSE_S3_SECRET_KEY`; never on disk). Licence: Copernicus Sentinel data terms — free, full and
  open; "contains modified Copernicus Sentinel data 2023". ~700 MB per scene, 18 scenes cached in
  `data/rs/s2/<scene id>/` (not in the repo).
- **MapBiomas Collection 4 (Sentinel, 10 m)**, land-cover map of Brazil for 2023, one GeoTIFF
  (4.6 GB, tiled 256², LZW, EPSG:4326 at 8.98e-5° ≈ 10 m) from
  `https://storage.googleapis.com/mapbiomas-public/initiatives/brasil/lulc_10m/collection4/coverage/brazil_coverage/brazil_coverage-col4_10m_2023.tif`.
  CC BY 4.0; cite Souza et al. 2020, *Remote Sensing* 12(17):2735 and the MapBiomas project.
- **EuroSAT** supplies the geometry and the normalisation: 64 × 64 chips at 10 m (640 m
  squares), all bands at 10 m, per-band mean / std of its training split
  (`data/rs/manifest_rs.json`) applied to every Brazilian chip unchanged.

## 2. Where and when

Two boxes, scenes chosen per MGRS tile as the least-cloudy of the season (ESA `eo:cloud_cover`),
ties broken by date — earliest for the Amazon (the later the dry season, the thicker the fire
smoke), latest for the Cerrado's dry part (September is the driest month), earliest for the
wet part; a scene yielding under 300 clear windows would have been replaced by the next date
(none needed this pass). Tiles 20LPQ / 20LPP (the Rondônia fishbones) are pinned for the figure.

| box | lon, lat | note |
|---|---|---|
| amazon | −64.0…−61.0, −11.5…−9.0 | Rondônia arc of deforestation: Ariquemes, Jaru, Machadinho d'Oeste, the BR-364 fishbones |
| cerrado | −47.0…−44.5, −13.5…−11.0 | western Bahia (MATOPIBA): Luís Eduardo Magalhães, Barreiras, the Rio Grande; one window per tile, cut in both seasons |

| part | tile | date | ESA cloud % | baseline | offset (DN) | candidate windows | cloudy (s2cloudless) | s2cloudless median | scene |
|---|---|---|---|---|---|---|---|---|---|
| amazon_dry | 20LPQ | 2023-06-16 | 0.0 | 05.10 | -1000 | 1500 | 45 | 0.01 | `S2B_MSIL1C_20230616T141719_N0510_R010_T20LPQ_20240926T072904` |
| amazon_dry | 20LPP | 2023-06-16 | 0.0 | 05.10 | -1000 | 1500 | 72 | 0.01 | `S2B_MSIL1C_20230616T141719_N0510_R010_T20LPP_20240926T072904` |
| amazon_dry | 20LMP | 2023-07-04 | 0.0 | 05.10 | -1000 | 1500 | 101 | 0.01 | `S2A_MSIL1C_20230704T142721_N0510_R053_T20LMP_20240912T073543` |
| amazon_dry | 20LMQ | 2023-06-17 | 0.0 | 05.11 | -1000 | 524 | 159 | 0.01 | `S2A_MSIL1C_20230617T143731_N0511_R096_T20LMQ_20250618T173843` |
| amazon_dry | 20LNQ | 2023-06-19 | 0.0 | 05.10 | -1000 | 1500 | 146 | 0.01 | `S2B_MSIL1C_20230619T142719_N0510_R053_T20LNQ_20240926T025751` |
| amazon_dry | 20LLP | 2023-06-19 | 0.0 | 05.10 | -1000 | 1500 | 120 | 0.01 | `S2B_MSIL1C_20230619T142719_N0510_R053_T20LLP_20240926T025751` |
| cerrado_dry | 23LKF | 2023-09-11 | 0.0 | 05.10 | -1000 | 1500 | 209 | 0.01 | `S2A_MSIL1C_20230911T132241_N0510_R038_T23LKF_20241104T064627` |
| cerrado_wet | 23LKF | 2023-03-25 | 0.0 | 05.10 | -1000 | 1500 | 129 | 0.01 | `S2A_MSIL1C_20230325T132231_N0510_R038_T23LKF_20240831T002851` |
| cerrado_dry | 23LNF | 2023-09-13 | 0.0 | 05.10 | -1000 | 1500 | 78 | 0.01 | `S2B_MSIL1C_20230913T131249_N0510_R138_T23LNF_20241105T082013` |
| cerrado_wet | 23LNF | 2023-04-01 | 0.0 | 05.10 | -1000 | 1500 | 69 | 0.01 | `S2A_MSIL1C_20230401T131241_N0510_R138_T23LNF_20240903T140328` |
| cerrado_dry | 23LNG | 2023-09-13 | 0.0 | 05.10 | -1000 | 1500 | 111 | 0.01 | `S2B_MSIL1C_20230913T131249_N0510_R138_T23LNG_20241105T082013` |
| cerrado_wet | 23LNG | 2023-02-10 | 0.0 | 05.10 | -1000 | 1500 | 108 | 0.01 | `S2A_MSIL1C_20230210T131241_N0510_R138_T23LNG_20240814T002704` |
| cerrado_dry | 23LKG | 2023-09-11 | 0.0 | 05.10 | -1000 | 1500 | 158 | 0.01 | `S2A_MSIL1C_20230911T132241_N0510_R038_T23LKG_20241104T064627` |
| cerrado_wet | 23LKG | 2023-04-27 | 0.0 | 05.10 | -1000 | 1500 | 100 | 0.00 | `S2A_MSIL1C_20230427T133151_N0510_R081_T23LKG_20240902T181131` |
| cerrado_dry | 23LNH | 2023-09-13 | 0.0 | 05.10 | -1000 | 1500 | 83 | 0.01 | `S2B_MSIL1C_20230913T131249_N0510_R138_T23LNH_20241105T082013` |
| cerrado_wet | 23LNH | 2023-01-31 | 0.0 | 05.10 | -1000 | 1500 | 94 | 0.01 | `S2A_MSIL1C_20230131T131241_N0510_R138_T23LNH_20240808T160108` |
| cerrado_dry | 23LLF | 2023-09-11 | 0.0 | 05.10 | -1000 | 1500 | 73 | 0.01 | `S2A_MSIL1C_20230911T132241_N0510_R038_T23LLF_20241104T064627` |
| cerrado_wet | 23LLF | 2023-03-10 | 0.0 | 05.10 | -1000 | 1500 | 64 | 0.01 | `S2B_MSIL1C_20230310T132239_N0510_R038_T23LLF_20240824T014713` |

Every scene is processing baseline 05.10 / 05.11 — the archive was reprocessed in 2024 — and
carries `RADIO_ADD_OFFSET = −1000` in every band; the offset is read from `MTD_MSIL1C.xml`
per scene and removed before anything else (reflectance = (DN + offset) / 10000, clipped
at 0), never decided by date. 20LMQ has 524 candidates rather than 1,500 because most of
that scene is outside the swath (candidates are drawn inside the valid area only).

## 3. The chip rules, in the order they run

1. **Bands to 10 m.** The 10 m bands as they are; the 20 m bands (B05 B06 B07 B8A B11 B12)
   and the 60 m bands (B01 B09 B10) cubic-spline upsampled to 10 m with pixel edges aligned
   (`scipy.ndimage.zoom(order=3, grid_mode=True)`), EuroSAT's own resampling. Planes stored
   in EuroSAT's order: B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A.
2. **Windows.** 64 × 64 chips whose origins sit on the 60 m grid (multiples of 6 pixels), so
   every band's native pixels tile the chip exactly; 1,500 origins per tile drawn without
   replacement from a seed derived from the tile id — so a Cerrado tile's September and March
   scenes are cut at the same windows; a window touching the swath's nodata (B02 = 0) is
   never drawn.
3. **Clouds.** `s2cloudless` (Sentinel Hub's pixel classifier, trained on L1C reflectance;
   threshold 0.4, average over 4 px, dilation 2) on each chip; kept if the maximum
   probability over the chip is under 0.2. A Cerrado window is kept only if it is clear in
   *both* seasons, so the two parts hold the same chips (asserted: identical `chip_id`
   arrays). Median max-probability of the kept chips: 0.03 dry, 0.03 wet.
4. **Labels.** The chip's UTM bounds → EPSG:4326 (grid convergence < 1.5°, bounding-box
   overshoot < 2%) → the MapBiomas window, ~64 × 64 label pixels; the majority code wins,
   kept at purity ≥ 0.8 (median purity of the kept chips: 0.985). The class map, EuroSAT
   → MapBiomas, is `CODE_MAP` in the script:

   | scored class | MapBiomas codes |
   |---|---|
   | 0 annual crop | 19 temporary crop, 39 soybean, 20 sugar cane, 40 rice, 62 cotton, 41 other |
   | 1 perennial crop | 36, 46 coffee, 47 citrus, 35 palm oil, 48 other |
   | 2 forest | 3 forest formation |
   | 3 herbaceous | 12 grassland |
   | 4 pasture | 15 pasture |
   | 5 built | 24 urban area |
   | 6 water | 33 river, lake and ocean |
   | 7 savanna formation (diagnostic, never scored) | 4 |
   | 8 forest plantation (diagnostic, never scored) | 9 |

   Every other code (mosaic of uses, wetland, mining, floodable forest, …) drops the chip.
   None of them occurred in these boxes at purity ≥ 0.8: the codes before capping were
   Amazon — forest formation 3,889, pasture 1,775, temporary crop 41, grassland 7, urban 2,
   river 1; Cerrado — savanna formation 2,520, pasture 884, temporary crop 520, forest
   formation 198, grassland 17, forest plantation 1.
5. **Caps.** At most 2,000 chips per class per part, a seeded draw, applied identically to a
   region's parts.
6. **Records.** `(reflectance − EuroSAT train mean_b) / std_b` per band, f32 `[N, 13, 64,
   64]`; labels int32; `meta_<part>.npz` with chip id (`<tile>_<row>_<col>`), scene, date,
   window, lon/lat, purity, majority code, the full code histogram, s2cloudless max and mean,
   baseline and offset.

## 4. What came out

| part | chips | before caps | annual crop | perennial | forest | herbaceous | pasture | built | water | savanna (diag) | plantation (diag) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amazon_dry | 3,826 | 5,715 | 41 | 0 | 2,000 | 7 | 1,775 | 2 | 1 | 0 | 0 |
| cerrado_dry | 3,620 | 4,140 | 520 | 0 | 198 | 17 | 884 | 0 | 0 | 2,000 | 1 |
| cerrado_wet | 3,620 | 4,140 | the same chips | | | | | | | | |

The Amazon part is a forest/pasture world with a handful of crop chips; the Cerrado's
scored classes are pasture, annual crop and gallery forest, with savanna formation — the
biome itself, a class Europe does not have — as the largest diagnostic row. A 640 m chip of
pure river is rare in these tiles (one), so the plan's water sentinel is a note, not a gate.
`brazil_all` (`scripts/datasets/rs_folds.py`) is the union of the scored chips of the three
parts with a five-way fold id from a hash of the chip id (a wet/dry twin shares a fold), for
the Brazil-trained ceiling and the fine-tune rows.

Mean reflectance per band, the physics in numbers (EuroSAT train for reference):

| | B02 blue | B04 red | B08 NIR | B11 SWIR1 | B12 SWIR2 |
|---|---|---|---|---|---|
| EuroSAT train | 0.112 | 0.095 | 0.230 | 0.182 | 0.112 |
| amazon_dry (June) | 0.087 | 0.054 | 0.258 | 0.174 | 0.073 |
| cerrado_dry (September) | 0.109 | 0.121 | 0.229 | 0.322 | 0.196 |
| cerrado_wet (Feb–Apr) | 0.089 | 0.065 | 0.294 | 0.217 | 0.104 |

The same Cerrado chips are twice as bright in red and SWIR in September as in March and
darker in NIR: senescent grass and dry leaves. Amazon forest chips sit on EuroSAT's forest
medians in every ground band (B08 0.262 vs 0.262, B12 0.055 vs 0.053 on the one-tile smoke),
which is Gate 2's level-and-offset check passed.

## 5. What the first pass taught

- **Smoke is not cloud.** ESA's `eo:cloud_cover` was 0.0% on 20LPP (2023-08-18) and 20LMN,
  and s2cloudless flagged every window (median probability 0.30 / 0.5): August in Rondônia
  is fire season and the scenes were uniformly hazy (B01/B02 40% brighter than a clear
  scene, cirrus band unchanged). Hence the date ranking (June first for the Amazon), the
  paged STAC search (the first pass saw only the first 200 scenes), and the next-date
  fallback. Both August scenes stay in the cache; the smoke is a possible side row, not this
  demo's.
- **Partial swaths.** 20LPQ's August scene was 43% nodata and the first candidate draw did
  not know it; candidates are now drawn inside the swath.
- The Brazil quicklook PNGs of this pass were rendered by the running process before the
  render was corrected to use offset-removed reflectance, so they look washed out; the
  records are right (the band means above are from the records).

## 6. Rebuilding

```bash
export CDSE_S3_ACCESS_KEY=… CDSE_S3_SECRET_KEY=…          # https://eodata-s3keysmanager.dataspace.copernicus.eu/
.venv-rs/bin/python scripts/datasets/preprocess_rs_brazil.py --dry-run          # the scene choice, no fetch
.venv-rs/bin/python scripts/datasets/preprocess_rs_brazil.py --tiles 6 --cands 1500 --cap 2000
.venv-rs/bin/python scripts/datasets/rs_folds.py
```

The MapBiomas GeoTIFF must be at `data/rs/mapbiomas/brazil_coverage-col4_10m_2023.tif`
(`curl -L -o … <the URL in §1>`), and `data/rs/manifest_rs.json` must exist (EuroSAT first:
`scripts/datasets/download_rs.sh`).
