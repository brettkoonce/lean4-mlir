#!/usr/bin/env python3
"""Cut the Brazil side of the remote-sensing demo — planning/remote_sensing_wavelengths_demo.md §2–§3, Gate 2.

Sentinel-2 Level-1C scenes over two boxes — the Rondônia arc of deforestation (Amazon) and
the western-Bahia soy frontier (Cerrado) — found through the Copernicus Data Space STAC
(open), fetched per band from its S3 store (`CDSE_S3_ACCESS_KEY` / `CDSE_S3_SECRET_KEY`
from https://eodata-s3keysmanager.dataspace.copernicus.eu/) or over HTTPS with a Keycloak
token (`CDSE_USERNAME` / `CDSE_PASSWORD`), and cut into 64×64 chips on EuroSAT's geometry:
the 20 m and 60 m bands cubic-spline upsampled to 10 m, chip origins on the 60 m grid, the
radiometric offset READ FROM `MTD_MSIL1C.xml` (never decided by date — the archive was
reprocessed under baseline 05.xx with the +1000 in it), reflectance = (DN + offset) /
QUANTIFICATION_VALUE, then standardised with the EuroSAT training mean/std from
`manifest_rs.json`, so a Brazilian chip and a European one are the same function of
reflectance. Labels are the majority MapBiomas Collection 4 (Sentinel, 10 m, CC BY 4.0)
class over the chip's footprint, kept at purity ≥ 0.8, through the seven-class map
(0 annual crop, 1 perennial crop, 2 forest, 3 herbaceous, 4 pasture, 5 built, 6 water) plus
the two diagnostic codes (7 savanna formation, 8 forest plantation); every other code is
dropped and counted. Clouds: s2cloudless on the chip, kept at max probability < 0.2. The
Cerrado's wet and dry parts are the SAME windows of the same tiles (drawn once per tile
from a seed), a chip entering both parts or neither.

  data/rs/{amazon_dry,cerrado_dry,cerrado_wet}.bin   f32 [N, 13, 64, 64], EuroSAT plane order, standardised
  data/rs/labels_{part}.bin                          int32 (0–6 scored, 7–8 diagnostic)
  data/rs/meta_{part}.npz                            chip_id, scene, tile, date, row, col, lon, lat, purity,
                                                     code, hist[76], cloud_max, cloud_mean, baseline, offset
  data/rs/{part}_quicklook.png                       ten chips per class, true colour over false colour
  data/rs/manifest_rs_brazil.json                    scenes, censuses, drops, the class map

  .venv-rs/bin/python scripts/datasets/preprocess_rs_brazil.py [--year 2023] [--tiles 6] [--cands 1500] [--cap 2000]
                                                                [--parts amazon_dry,cerrado_dry,cerrado_wet] [--dry-run]
"""
import argparse
import glob
import json
import os
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import requests
from PIL import Image

STAC = "https://stac.dataspace.copernicus.eu/v1/search"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
S3_ENDPOINT = "https://eodata.dataspace.copernicus.eu"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B10", "B11", "B12", "B8A"]   # EuroSAT plane order
RES = {"B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20, "B07": 20, "B08": 10,
       "B09": 60, "B10": 60, "B11": 20, "B12": 20, "B8A": 20}
# MTD_MSIL1C.xml band_id → band name (the metadata lists B8A eighth)
MTD_BAND_ID = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
S2C_BANDS = ["B01", "B02", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]              # s2cloudless input order
CHIP = 64
ALIGN = 6                 # chip origins on the 60 m grid (6 × 10 m)

SHARED = ["annual crop", "perennial crop", "forest", "herbaceous", "pasture", "built", "water"]
DIAG = {7: "savanna formation", 8: "forest plantation"}
# MapBiomas code → shared class (Collection 4 / 11 legend)
CODE_MAP = {19: 0, 39: 0, 20: 0, 40: 0, 62: 0, 41: 0,        # temporary crop and its members
            36: 1, 46: 1, 47: 1, 35: 1, 48: 1,               # perennial crop and its members
            3: 2,                                            # forest formation
            12: 3,                                           # grassland
            15: 4,                                           # pasture
            24: 5,                                           # urban area
            33: 6,                                           # river, lake and ocean
            4: 7,                                            # savanna formation (diagnostic)
            9: 8}                                            # forest plantation (diagnostic)
CODE_NAMES = {3: "forest formation", 4: "savanna formation", 5: "mangrove", 6: "floodable forest", 9: "forest plantation",
              11: "wetland", 12: "grassland", 15: "pasture", 19: "temporary crop", 20: "sugar cane", 21: "mosaic of uses",
              23: "beach/dune", 24: "urban area", 25: "other non-vegetated", 29: "rocky outcrop", 30: "mining",
              31: "aquaculture", 33: "river/lake/ocean", 35: "palm oil", 36: "perennial crop", 39: "soybean", 40: "rice",
              41: "other temporary crops", 46: "coffee", 47: "citrus", 48: "other perennial crops", 49: "wooded sandbank",
              50: "herbaceous sandbank", 62: "cotton", 0: "no data"}

REGIONS = {
    # lon_min, lat_min, lon_max, lat_max
    "amazon": dict(bbox=[-64.0, -11.5, -61.0, -9.0], note="Rondônia arc of deforestation: Ariquemes, Jaru, Ji-Paraná, the BR-364 fishbones"),
    "cerrado": dict(bbox=[-47.0, -13.5, -44.5, -11.0], note="western Bahia (MATOPIBA): Luís Eduardo Magalhães, Barreiras, the Rio Grande"),
}
SEASONS = {  # month-day windows within the chip year; the search cloud ceiling (per chip s2cloudless decides after)
    "dry": ("06-15", "09-15", 5.0),
    "wet": ("01-15", "04-30", 30.0),
}
PARTS = {"amazon_dry": ("amazon", "dry"), "cerrado_dry": ("cerrado", "dry"), "cerrado_wet": ("cerrado", "wet")}


# ────────────────────────── CDSE: search and fetch ──────────────────────────
def stac_search(bbox, start, end, cloud_max, limit=200):
    body = {"collections": ["sentinel-2-l1c"], "bbox": bbox, "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
            "limit": limit, "filter-lang": "cql2-json",
            "filter": {"op": "<", "args": [{"property": "eo:cloud_cover"}, cloud_max]}}
    feats = []
    url, payload = STAC, body
    while url and len(feats) < 2000:                      # follow the paging links: one bbox-season can exceed a page
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        feats += j["features"]
        nxt = [l for l in j.get("links", []) if l.get("rel") == "next"]
        url = nxt[0]["href"] if nxt else None
        payload = nxt[0].get("body", body) if nxt else None
    out = []
    for f in feats:
        p = f["properties"]
        out.append(dict(id=f["id"], tile=p["grid:code"].replace("MGRS-", ""), cloud=float(p["eo:cloud_cover"]),
                        date=p["datetime"][:10], baseline=p.get("processing:version"), sun_elev=p.get("view:sun_elevation"),
                        assets={k: dict(s3=a["href"], https=(a.get("alternate") or {}).get("https", {}).get("href"))
                                for k, a in f["assets"].items() if k in BANDS or k == "product_metadata"}))
    return out


def ranked_per_tile(scenes, late=False):
    """Per tile, its scenes ranked by cloud then date. Earliest first by default: in the
    Amazon's dry season the later the date the thicker the fire smoke, which ESA's cloud
    mask does not count and s2cloudless does (20LPP on 2023-08-18 flagged every window at
    0% cloud). `late` ranks the latest first — the Cerrado's dry part wants September, the
    driest month, not the June shoulder."""
    by = {}
    for s in scenes:
        by.setdefault(s["tile"], []).append(s)
    return {t: sorted(sorted(v, key=lambda s: s["date"], reverse=late), key=lambda s: s["cloud"]) for t, v in by.items()}


class Fetcher:
    """Per-band downloads into data/rs/s2/<scene>/, by S3 keys if present, else by Keycloak token."""

    def __init__(self, root):
        self.root = root
        self.s3 = None
        self.tok = None
        self.tok_t = 0
        ak, sk = os.environ.get("CDSE_S3_ACCESS_KEY"), os.environ.get("CDSE_S3_SECRET_KEY")
        if ak and sk:
            import boto3
            self.s3 = boto3.session.Session().client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=ak,
                                                     aws_secret_access_key=sk, region_name="default")
            self.route = "s3"
        elif os.environ.get("CDSE_USERNAME") and os.environ.get("CDSE_PASSWORD"):
            self.route = "https"
        else:
            sys.exit("no CDSE credentials: export CDSE_S3_ACCESS_KEY + CDSE_S3_SECRET_KEY "
                     "(https://eodata-s3keysmanager.dataspace.copernicus.eu/) or CDSE_USERNAME + CDSE_PASSWORD")

    def token(self):
        if self.tok is None or time.time() - self.tok_t > 480:
            r = requests.post(TOKEN_URL, data=dict(client_id="cdse-public", grant_type="password",
                                                   username=os.environ["CDSE_USERNAME"], password=os.environ["CDSE_PASSWORD"]), timeout=60)
            r.raise_for_status()
            self.tok, self.tok_t = r.json()["access_token"], time.time()
        return self.tok

    def fetch(self, scene, key, local):
        if os.path.exists(local) and os.path.getsize(local) > 0:
            return
        os.makedirs(os.path.dirname(local), exist_ok=True)
        tmp = local + ".part"
        a = scene["assets"][key]
        if self.route == "s3":
            self.s3.download_file("eodata", a["s3"].replace("s3://eodata/", ""), tmp)
        else:
            with requests.get(a["https"], headers={"Authorization": f"Bearer {self.token()}"}, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
        os.replace(tmp, local)

    def scene_dir(self, scene):
        return os.path.join(self.root, scene["id"])

    def fetch_scene(self, scene):
        d = self.scene_dir(scene)
        t = time.time()
        self.fetch(scene, "product_metadata", os.path.join(d, "MTD_MSIL1C.xml"))
        for b in BANDS:
            self.fetch(scene, b, os.path.join(d, f"{b}.jp2"))
        return time.time() - t


# ────────────────────────── the scene as 13 planes at 10 m ──────────────────────────
def read_offsets(mtd_path):
    root = ET.parse(mtd_path).getroot()
    quant = None
    offs = {b: 0.0 for b in BANDS}
    baseline = None
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag == "QUANTIFICATION_VALUE":
            quant = float(el.text)
        elif tag == "RADIO_ADD_OFFSET":
            offs[MTD_BAND_ID[int(el.attrib["band_id"])]] = float(el.text)
        elif tag == "PROCESSING_BASELINE":
            baseline = el.text
    if quant is None:
        sys.exit(f"{mtd_path}: no QUANTIFICATION_VALUE")
    return quant, offs, baseline


def load_scene(d):
    """Every band at 10 m as uint16 DN [13, 10980, 10980] in EuroSAT plane order, plus the
    10 m transform/CRS and the nodata mask from B02 (DN 0)."""
    import rasterio
    from scipy.ndimage import zoom
    planes = []
    transform = crs = None
    for b in BANDS:
        with rasterio.open(os.path.join(d, f"{b}.jp2")) as src:
            x = src.read(1)
            if RES[b] == 10:
                transform, crs = src.transform, src.crs
        f = RES[b] // 10
        if f > 1:
            # cubic spline to 10 m, pixel edges aligned (EuroSAT's own resampling)
            x = zoom(x.astype(np.float32), f, order=3, grid_mode=True, mode="grid-constant")
            x = np.clip(np.rint(x), 0, 65535).astype(np.uint16)
        planes.append(x)
    n = min(p.shape[0] for p in planes)
    X = np.stack([p[:n, :n] for p in planes])
    return X, transform, crs


def draw_windows(n_side, n_cand, seed, valid=None):
    """Chip origins on the 60 m grid, drawn once per tile from a seed (the wet/dry pairing);
    with `valid` (a B02 > 0 mask), only windows fully inside the swath, so a partial
    tile still yields `n_cand` candidates."""
    rng = np.random.default_rng(seed)
    n_pos = (n_side - CHIP) // ALIGN
    pos = rng.permutation(n_pos * n_pos)
    win = np.stack([(pos // n_pos) * ALIGN, (pos % n_pos) * ALIGN], axis=1)
    if valid is not None:
        # a window is inside the swath if its four corners and centre are (cheap, then exact below)
        ok = np.array([valid[r, c] and valid[r + CHIP - 1, c] and valid[r, c + CHIP - 1] and valid[r + CHIP - 1, c + CHIP - 1]
                       and valid[r + CHIP // 2, c + CHIP // 2] for r, c in win[:min(len(win), 20 * n_cand)]])
        win = win[:len(ok)][ok]
    return win[:n_cand]


def cloud_probs(chips_refl):
    """s2cloudless on [n, 13, 64, 64] reflectance (offset applied): per-chip max and mean probability."""
    from s2cloudless import S2PixelCloudDetector
    idx = [BANDS.index(b) for b in S2C_BANDS]
    x = chips_refl[:, idx].transpose(0, 2, 3, 1)
    det = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2, all_bands=False)
    out_max, out_mean = [], []
    for a in range(0, len(x), 256):
        p = det.get_cloud_probability_maps(x[a:a + 256])
        out_max.append(p.reshape(len(p), -1).max(axis=1))
        out_mean.append(p.reshape(len(p), -1).mean(axis=1))
    return np.concatenate(out_max), np.concatenate(out_mean)


class Labeller:
    """Majority MapBiomas code over a chip's footprint (its UTM bounds transformed to EPSG:4326;
    the grid convergence at these longitudes is under 1.5°, so the bounding box overshoots
    the rotated footprint by < 2%)."""

    def __init__(self, path):
        import rasterio
        self.src = rasterio.open(path)
        self.year = path

    def label(self, crs, transform, r0, c0):
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
        x0, y0 = transform * (c0, r0)
        x1, y1 = transform * (c0 + CHIP, r0 + CHIP)
        b = transform_bounds(crs, self.src.crs, min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        w = from_bounds(*b, transform=self.src.transform)
        arr = self.src.read(1, window=w, boundless=True, fill_value=0)
        hist = np.bincount(arr.ravel(), minlength=76)[:76]
        hist[0] = 0                      # no-data never wins
        code = int(hist.argmax())
        purity = float(hist[code] / max(hist.sum(), 1))
        lon = (b[0] + b[2]) / 2
        lat = (b[1] + b[3]) / 2
        return code, purity, hist, lon, lat


def quicklook(X_dn, labels, path, names):
    rng = np.random.default_rng(0)
    rows_tc, rows_fc = [], []
    for c in range(len(names)):
        ids = np.where(labels == c)[0]
        if len(ids) == 0:
            continue
        pick = rng.choice(ids, min(10, len(ids)), replace=False)
        for planes, dst in (([3, 2, 1], rows_tc), ([7, 3, 2], rows_fc)):
            tiles = [np.clip(X_dn[i][planes].transpose(1, 2, 0) / 2750.0 * 255, 0, 255).astype(np.uint8) for i in pick]
            row = np.concatenate(tiles, axis=1)
            if row.shape[1] < 10 * CHIP:
                row = np.pad(row, ((0, 0), (0, 10 * CHIP - row.shape[1]), (0, 0)))
            dst.append(row)
    ql = np.concatenate([np.concatenate(rows_tc, axis=0), np.concatenate(rows_fc, axis=0)], axis=1)
    Image.fromarray(ql).resize((ql.shape[1] * 2, ql.shape[0] * 2), Image.NEAREST).save(path)


# ────────────────────────── main ──────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/rs")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--tiles", type=int, default=6, help="tiles per region")
    ap.add_argument("--include", default="20LPQ,20LPP", help="tiles kept whenever a usable scene exists (the Rondônia fishbones for the figure)")
    ap.add_argument("--cands", type=int, default=1500, help="candidate windows per scene")
    ap.add_argument("--cap", type=int, default=2000, help="chips per class per part")
    ap.add_argument("--purity", type=float, default=0.8)
    ap.add_argument("--cloud", type=float, default=0.2, help="max s2cloudless probability over the chip")
    ap.add_argument("--min-clear", type=int, default=300, help="clear windows a scene must yield, else the next date is tried")
    ap.add_argument("--tries", type=int, default=3, help="dates tried per tile and season")
    ap.add_argument("--parts", default="amazon_dry,cerrado_dry,cerrado_wet")
    ap.add_argument("--mapbiomas", default=None, help="local Collection 4 GeoTIFF (default data/rs/mapbiomas/brazil_coverage-col4_10m_<year>.tif)")
    ap.add_argument("--dry-run", action="store_true", help="search and print the scene choice; fetch nothing")
    args = ap.parse_args()
    parts = [p for p in args.parts.split(",") if p]
    for p in parts:
        if p not in PARTS:
            sys.exit(f"unknown part {p}: {list(PARTS)}")
    mb_path = args.mapbiomas or os.path.join(args.out, "mapbiomas", f"brazil_coverage-col4_10m_{args.year}.tif")
    manifest = json.load(open(os.path.join(args.out, "manifest_rs.json")))
    mean = np.array(manifest["band_mean"], dtype=np.float32)[:, None, None]
    std = np.array(manifest["band_std"], dtype=np.float32)[:, None, None]
    assert manifest["band_order"] == BANDS
    t0 = time.time()

    # ── scene choice: per region, the least-cloudy scene per tile per season; the Cerrado's
    #    tiles are those with a usable scene in BOTH seasons ──
    choice = {}      # part → {tile: scene}
    for region, spec in REGIONS.items():
        need = {PARTS[p][1] for p in parts if PARTS[p][0] == region}
        if not need:
            continue
        per_season = {}
        for season in need:
            d0, d1, cmax = SEASONS[season]
            scenes = stac_search(spec["bbox"], f"{args.year}-{d0}", f"{args.year}-{d1}", cmax)
            per_season[season] = ranked_per_tile(scenes, late=(region == "cerrado" and season == "dry"))
            print(f"{region}/{season}: {len(scenes)} scenes under {cmax:g}% cloud, {len(per_season[season])} tiles", flush=True)
        tiles = set.intersection(*[set(v) for v in per_season.values()])
        must = [t for t in args.include.split(",") if t in tiles]
        ranked = must + [t for t in sorted(tiles, key=lambda t: max(per_season[s][t][0]["cloud"] for s in need)) if t not in must]
        ranked = ranked[:max(args.tiles, len(must))]
        for season in need:
            choice[f"{region}_{season}"] = {t: per_season[season][t][:args.tries] for t in ranked}
        for t in ranked:
            print(f"   {t}: " + "  ".join(f"{s} " + " / ".join(f"{sc['date']} {sc['cloud']:.1f}%" for sc in per_season[s][t][:args.tries])
                                          + f" ({per_season[s][t][0]['baseline']})" for s in need), flush=True)
    if args.dry_run:
        return

    fetcher = Fetcher(os.path.join(args.out, "s2"))
    labeller = Labeller(mb_path)
    print(f"fetch route: {fetcher.route}; labels: {mb_path}", flush=True)

    # ── cut: tile by tile so the Cerrado's seasons share windows and a keep mask ──
    results = {p: [] for p in parts}     # per part: list of dicts per kept chip (before caps)
    scene_log = {}
    by_region = {}
    for p in parts:
        by_region.setdefault(PARTS[p][0], []).append(p)
    for region, rparts in by_region.items():
        tiles = list(choice[rparts[0]])
        for tile in tiles:
            per_part = {}
            for p in rparts:
                for attempt, scene in enumerate(choice[p][tile]):
                    dt = fetcher.fetch_scene(scene)
                    d = fetcher.scene_dir(scene)
                    quant, offs, baseline = read_offsets(os.path.join(d, "MTD_MSIL1C.xml"))
                    t1 = time.time()
                    X, transform, crs = load_scene(d)
                    win = draw_windows(X.shape[1], args.cands, seed=int.from_bytes(tile.encode(), "little") % (2 ** 31), valid=X[1] > 0)
                    chips = np.stack([X[:, r:r + CHIP, c:c + CHIP] for r, c in win])           # [n, 13, 64, 64] DN
                    nodata = (chips[:, 1] == 0).reshape(len(chips), -1).any(axis=1)             # B02 == 0
                    off = np.array([offs[b] for b in BANDS], dtype=np.float32)[None, :, None, None]
                    refl = np.clip((chips.astype(np.float32) + off) / quant, 0, None)
                    cmax, cmean = cloud_probs(refl)
                    keep = (~nodata) & (cmax < args.cloud)
                    del X
                    per_part[p] = dict(scene=scene, chips=chips, refl=refl, keep=keep, cmax=cmax, cmean=cmean, win=win,
                                       transform=transform, crs=crs, baseline=baseline, offset=float(offs["B02"]), quant=quant)
                    scene_log[scene["id"]] = dict(part=p, tile=tile, date=scene["date"], cloud=scene["cloud"], baseline=baseline,
                                                  offset={b: offs[b] for b in BANDS}, quant=quant, fetch_s=round(dt), load_s=round(time.time() - t1),
                                                  cands=int(len(win)), nodata=int(nodata.sum()), cloudy=int(((~nodata) & (cmax >= args.cloud)).sum()),
                                                  attempt=attempt, used=bool(keep.sum() >= args.min_clear or attempt == len(choice[p][tile]) - 1),
                                                  s2cloudless_mean_median=float(np.median(cmean)))
                    print(f"  {p} {tile} {scene['date']}: fetched in {dt:.0f} s, loaded in {time.time() - t1:.0f} s; offset {offs['B02']:g}, "
                          f"baseline {baseline}; {len(win)} windows, {nodata.sum()} nodata, {((~nodata) & (cmax >= args.cloud)).sum()} cloudy "
                          f"(s2cloudless mean prob median {np.median(cmean):.2f})", flush=True)
                    if keep.sum() >= args.min_clear:
                        break
                    if attempt < len(choice[p][tile]) - 1:
                        print(f"  {p} {tile} {scene['date']}: only {keep.sum()} clear windows < {args.min_clear} — trying the next date", flush=True)
            keep = np.logical_and.reduce([per_part[p]["keep"] for p in rparts])
            # labels once per window (the annual map is the same in both seasons)
            ref = per_part[rparts[0]]
            labs = [labeller.label(ref["crs"], ref["transform"], int(r), int(c)) if k else None for (r, c), k in zip(ref["win"], keep)]
            n_kept = 0
            for i, ((r, c), k) in enumerate(zip(ref["win"], keep)):
                if not k:
                    continue
                code, purity, hist, lon, lat = labs[i]
                if code not in CODE_MAP or purity < args.purity:
                    continue
                n_kept += 1
                for p in rparts:
                    pp = per_part[p]
                    results[p].append(dict(chip_id=f"{tile}_{r}_{c}", scene=pp["scene"]["id"], tile=tile, date=pp["scene"]["date"],
                                           row=int(r), col=int(c), lon=lon, lat=lat, purity=purity, code=code, hist=hist,
                                           label=CODE_MAP[code], cloud_max=float(pp["cmax"][i]), cloud_mean=float(pp["cmean"][i]),
                                           baseline=pp["baseline"], offset=pp["offset"], dn=pp["chips"][i], refl=pp["refl"][i]))
            print(f"  {tile}: {keep.sum()} windows clear in every season, {n_kept} labelled and pure  ({time.time() - t0:.0f} s)", flush=True)
            del per_part

    # ── caps (identical across a region's parts, since the chips are identical), records, quicklooks ──
    summary = dict(year=args.year, purity=args.purity, cloud=args.cloud, cands=args.cands, cap=args.cap, class_map=CODE_MAP,
                   shared=SHARED, diagnostic=DIAG, scenes=scene_log, parts={})
    for region, rparts in by_region.items():
        ref = results[rparts[0]]
        labels = np.array([r["label"] for r in ref], dtype=np.int32)
        rng = np.random.default_rng(1)
        sel = np.concatenate([rng.permutation(np.where(labels == c)[0])[:args.cap] for c in range(9)]) if len(ref) else np.array([], dtype=int)
        sel = np.sort(sel)
        for p in rparts:
            rows = [results[p][i] for i in sel]
            if not rows:
                print(f"{p}: NO chips kept", flush=True)
                continue
            X = np.stack([r["refl"] for r in rows]).astype(np.float32)
            xs = (X - mean[None]) / std[None]
            xs.tofile(os.path.join(args.out, f"{p}.bin"))
            lab = np.array([r["label"] for r in rows], dtype=np.int32)
            lab.tofile(os.path.join(args.out, f"labels_{p}.bin"))
            np.savez_compressed(os.path.join(args.out, f"meta_{p}.npz"),
                                chip_id=np.array([r["chip_id"] for r in rows]), scene=np.array([r["scene"] for r in rows]),
                                tile=np.array([r["tile"] for r in rows]), date=np.array([r["date"] for r in rows]),
                                row=np.array([r["row"] for r in rows]), col=np.array([r["col"] for r in rows]),
                                lon=np.array([r["lon"] for r in rows]), lat=np.array([r["lat"] for r in rows]),
                                purity=np.array([r["purity"] for r in rows]), code=np.array([r["code"] for r in rows]),
                                hist=np.stack([r["hist"] for r in rows]), cloud_max=np.array([r["cloud_max"] for r in rows]),
                                cloud_mean=np.array([r["cloud_mean"] for r in rows]), baseline=np.array([r["baseline"] for r in rows]),
                                offset=np.array([r["offset"] for r in rows]))
            quicklook(np.stack([r["refl"] * 10000.0 for r in rows]), lab, os.path.join(args.out, f"{p}_quicklook.png"), SHARED + list(DIAG.values()))
            census = {(SHARED + list(DIAG.values()))[c]: int((lab == c).sum()) for c in range(9)}
            codes_all = np.bincount([r["code"] for r in results[p]], minlength=76)
            summary["parts"][p] = dict(n=int(len(rows)), n_before_cap=int(len(results[p])), census=census,
                                       codes_before_cap={CODE_NAMES.get(i, str(i)): int(n) for i, n in enumerate(codes_all) if n},
                                       band_mean_refl=X.mean(axis=(0, 2, 3)).tolist(), water_B12_median_DN=float(np.median(
                                           np.stack([r["dn"][11] for r in rows if r["label"] == 6])) if (lab == 6).any() else -1))
            print(f"{p}: {len(rows)} chips ({len(results[p])} before caps) {census}  -> {p}.bin ({xs.nbytes / 1e9:.2f} GB)", flush=True)
    with open(os.path.join(args.out, "manifest_rs_brazil.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"done ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
