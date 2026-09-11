#!/usr/bin/env python3
"""Turn a `bf16_probe_*.sh` TSV into the `ETA=` string each job conf should carry.

    scripts/probe_to_eta.py runs/<date>-bf16-probe/fed.tsv --box "4x 4060 Ti"

## The formula, and the one judgement call in it

    hours = (steps_per_epoch * ms_step / 1000 + PER_EPOCH_OVERHEAD_S) * epochs / 3600

`PER_EPOCH_OVERHEAD_S` is eval + checkpoint, 37.5 s, measured on ResNet-34 and carried
across nets — it is under 2% of every row here, so a per-net figure would not move any
ETA. The one-time val drain (~30 GB) is NOT in it: paid once, not per epoch, and at
90-350 epochs it rounds away.

⚠⚠ **The ms/step column is `mean`, not `med`, and that is the whole point.** Commit
4a0a2781 measured a net whose median sat within 5 ms of its compute floor at every
worker count while its p90 was 1906 ms: shim starvation is BURSTY, so the trainer eats
the prefetch queue and then stalls. The median hides that entirely, and ranking on it
prefers the wrong arm — determinism ON reads a better median (166 vs 202) and a worse
mean (333 vs 203), and mean is what sets wall clock. A conf whose ETA came off a median
is quoting the compute floor and calling it a schedule. Seen live in `runs/probe3060.tsv`:
EfficientNet bf16 fed is med 99 / mean 193 / p90 502, so on the median it looks 1.66x
faster than f32 and on the mean it is not faster at all.

▶ Rows where mean > 1.15 * med are flagged STARVING and their ETA is annotated: that ETA
is a feed result, not a graph result, and it moves when `SHIM_WORKERS` or the tf.data
determinism default moves.

## Matching a measurement to a job

⛔ **Keyed by (net, variant) — a variant name alone is NOT enough.** `adamdpwxclipdrop`
is ConvNeXt-T's, ConvNeXt-S's AND ConvNeXt-B's; `momdp64` is both ResNet-34's and
ResNet-50's. An earlier draft of this script keyed on variant and cheerfully wrote
ResNet-34's ms/step into ResNet-50's conf.

⛔ And the conf's variant is checked against the probe's: `mnv2-default-4gpu.conf` trains
`rmsdp64` (RMSProp, MobileNetV2's reference optimizer) while `bf16_probe_4gpu.sh`'s row
list measured `adamdp64`. Those are different graphs, and a number carried from one to
the other is exactly the failure `enet-default-4gpu.conf`'s header warns about.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

PER_EPOCH_OVERHEAD_S = 37.5

# Fallback worker counts for TSVs written BEFORE 2026-09-11, when `bf16_probe_3060.sh`
# recorded the sweep-wide `$WORKERS` even on rows whose `extra` field overrode it. Since
# then the TSV's `workers` column is the count the row actually ran at and is preferred;
# this map is consulted only when that column is empty.
ROW_WORKERS = {
    "r34": 8, "r50": 8, "enetema": 8, "vits": 8, "vitb": 8, "vit": 8,
    "mnv2": 4, "mnv4": 4, "cnx": 4, "cnxs": 4, "cnxb": 4,
    "vitema": 4, "r50a3": 4, "r50a3w8": 8,
}

# probe `net` column -> (conf basename, precision that conf trains)
# A conf appears once per precision it has a job for; nets with only an f32 conf get one row.
NET_CONF = {
    ("r34", "f32"): "r34-default-4gpu",
    ("r50", "f32"): "r50-2018-4gpu",
    ("r50", "bf16"): "r50-2018-bf16-4gpu",
    ("r50a3", "f32"): "r50-a3-wxclip-4gpu",
    # ⚠ the bf16 conf runs SHIM_WORKERS=8 where its f32 twin runs 4, so it is quoted from
    # the `r50a3w8` row (same graph, the conf's own producer count), not from `r50a3`'s
    # bf16 row — that one exists so the f32 conf's "bf16 twin" compares like with like.
    ("r50a3w8", "bf16"): "r50-a3-wxclip-bf16-4gpu",
    ("mnv2", "f32"): "mnv2-default-4gpu",
    ("mnv4", "f32"): "mnv4-default-4gpu",
    ("enetema", "f32"): "enet-default-4gpu",
    ("cnx", "f32"): "cnx-default-4gpu",
    ("cnxs", "f32"): "cnxs-default-4gpu",
    ("cnxb", "f32"): "cnxb-default-4gpu",
    ("vit", "f32"): "vit-default-4gpu",
    ("vitema", "bf16"): "vit-default-emabf16-4gpu",
    ("vits", "f32"): "vits-default-g512-4gpu",
    ("vitb", "f32"): "vitb-default-g512-4gpu",
}

# A row whose f32/bf16 twin lives under another net key (the RSB-A3 bf16 conf's 8-producer row
# pairs with the f32 conf's 4-producer row — same graph, the conf's own producer count each).
TWIN_ALIAS = {"r50a3w8": "r50a3"}


def conf_facts(name, jobs_dir):
    """(epochs, variant) as the conf actually sets them — comment lines ignored."""
    p = Path(jobs_dir) / f"{name}.conf"
    if not p.exists():
        return None, None
    live = [l for l in p.read_text().splitlines() if not l.lstrip().startswith("#")]
    ep = var = None
    for l in live:
        m = re.search(r'^EPOCHS=[\'"]?(?:\$\{EPOCHS:-)?(\d+)', l.strip())
        if m and ep is None:
            ep = int(m.group(1))
        m = re.search(r'LEAN_MLIR_VARIANT=([A-Za-z0-9_]+)', l)
        if m and var is None:
            var = m.group(1)
    return ep, var


def hours(steps_ep, ms, epochs):
    return (steps_ep * ms / 1000.0 + PER_EPOCH_OVERHEAD_S) * epochs / 3600.0


def fmt(h):
    """Hours, plus a day figure once the number stops being plannable in hours.

    Above ~100 h nobody reads "431" as two and a half weeks, and these runs are
    scheduled around other work on the same four cards."""
    base = f"{h:.0f}" if h >= 10 else f"{h:.1f}"
    return f"{base} h ({h/24:.1f} d)" if h > 100 else f"{base} h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--box", default="4x 4060 Ti")
    ap.add_argument("--jobs-dir", default="scripts/jobs")
    ap.add_argument("--stat", choices=["mean", "med"], default="mean")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the results table as markdown for a runs/ README")
    ap.add_argument("--epochs", action="append", default=[], metavar="CONF=N",
                    help="quote a conf's ETA at N epochs instead of the count it "
                         "carries, e.g. --epochs mnv4-default-4gpu=500. ⚠ This changes "
                         "only the ESTIMATE: the schedule length lives in the Lean "
                         "config (`<net>ImagenetConfig.epochs`) with no env override, "
                         "and the conf's own EPOCHS is only supervise.sh's stop point, "
                         "so a genuine N-epoch run needs both of those changed too.")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.tsv), delimiter="\t")
            if r.get("med_ms") not in (None, "", "FAIL")]
    if not rows:
        sys.exit("no successful probe rows in that TSV")

    def ms_of(r):
        return int(r[args.stat + "_ms"] or r["med_ms"])

    fed = {}
    if args.markdown:
        print("| net | variant | prec | workers | med | mean | p90 | steps/ep | note |")
        print("|---|---|---|---|---|---|---|---|---|")
    else:
        print(f"{'net':<8} {'variant':<30} {'prec':<5} {'arm':<6} "
              f"{'med':>6} {'mean':>6} {'p90':>6} {'st/ep':>6}  flag")
    for r in sorted(rows, key=lambda r: (r["net"], r["prec"], r["arm"])):
        med, mean = int(r["med_ms"]), int(r["mean_ms"] or r["med_ms"])
        flag = "**starving**" if mean > 1.15 * med else ""
        w = r.get("workers") or ROW_WORKERS.get(r["net"], "?")
        if args.markdown:
            print(f"| {r['net']} | `{r['variant']}` | {r['prec']} | {w} | {med} | "
                  f"**{mean}** | {r.get('p90_ms') or '-'} | {r['steps_ep']} | {flag} |")
        else:
            print(f"{r['net']:<8} {r['variant']:<30} {r['prec']:<5} {r['arm']:<6} "
                  f"{med:>6} {mean:>6} {str(r.get('p90_ms') or '-'):>6} "
                  f"{r['steps_ep']:>6}  {flag}")
        if r["arm"] == "fed":
            fed[(r["net"], r["prec"])] = r

    overrides = {}
    for spec in args.epochs:
        k, _, v = spec.partition("=")
        overrides[k] = int(v)

    print("\n--- ETA strings (fed arm) ---")
    for (net, prec), conf in sorted(NET_CONF.items(), key=lambda kv: kv[1]):
        r = fed.get((net, prec))
        if r is None:
            print(f"  {conf:<26} — no fed {prec} probe row for '{net}'; ETA left alone")
            continue
        ep, cvar = conf_facts(conf, args.jobs_dir)
        if ep is None:
            print(f"  {conf:<26} — conf not found; skipped")
            continue
        forced = overrides.get(conf)
        if forced is not None and forced != ep:
            print(f"  {conf:<26} ⚠ quoting at {forced} ep, not the conf's {ep}")
            ep = forced
        if cvar and cvar != r["variant"]:
            print(f"  {conf:<26} ⛔ conf trains '{cvar}' but the probe measured "
                  f"'{r['variant']}' — DIFFERENT GRAPHS, not writing an ETA")
            continue
        ms, spe = ms_of(r), int(r["steps_ep"])
        med, mean = int(r["med_ms"]), int(r["mean_ms"] or r["med_ms"])
        label = "median" if args.stat == "med" else "mean"
        stall_note = "periodic stalls, see runs/2026-09-11-imagenet-probe-postfix"
        # ⭐ bf16 FIRST, f32 after (the user's call, 2026-09-11): the string reads
        # "bf16 ~55 h / f32 ~57 h on … (249 / 259 ms/step median, 300 ep)" whichever precision
        # the conf trains — the conf's own precision is in its variant, the string is the plan.
        # ⚠ Only a twin that is genuinely THIS graph in the other precision qualifies:
        # `bf16_probe_3060.sh`'s `vitema` row fills its f32 slot with the non-EMA graph, so a
        # naive same-net lookup would pit EMA bf16 against a different graph.
        other = "bf16" if prec == "f32" else "f32"
        twin = fed.get((net, other)) or fed.get((TWIN_ALIAS.get(net, "∅"), other))
        if twin is not None:
            a, b = (r["variant"], twin["variant"]) if prec == "f32" else (twin["variant"], r["variant"])
            if b != a + "bf16":
                twin = None
        if twin is not None:
            rb, rf = (r, twin) if prec == "bf16" else (twin, r)
            hb = hours(int(rb["steps_ep"]), ms_of(rb), ep)
            hf = hours(int(rf["steps_ep"]), ms_of(rf), ep)
            s = (f'bf16 ~{fmt(hb)} / f32 ~{fmt(hf)} on {args.box} '
                 f'({ms_of(rb)} / {ms_of(rf)} ms/step {label}, {ep} ep)')
            mb, mf = int(rb["mean_ms"] or rb["med_ms"]), int(rf["mean_ms"] or rf["med_ms"])
            stalls = mb > 1.15 * int(rb["med_ms"]) or mf > 1.15 * int(rf["med_ms"])
            if stalls and args.stat == "med":
                s += f" ⚠ today's means {mb} / {mf}: {stall_note}"
            elif stalls:
                s += f" ⚠ mean ≫ median ({rb['med_ms']} / {rf['med_ms']}): {stall_note}"
        else:
            h = hours(spe, ms, ep)
            s = f'{prec} ~{fmt(h)} on {args.box} ({ms} ms/step {label}, {ep} ep; no {other} twin for this graph)'
            if mean > 1.15 * med:
                s += (f" ⚠ today's mean is {mean}: {stall_note}" if args.stat == "med"
                      else f" ⚠ mean ≫ median ({med}): {stall_note}")
        print(f'  {conf:<26} ETA="{s}"')


if __name__ == "__main__":
    main()
