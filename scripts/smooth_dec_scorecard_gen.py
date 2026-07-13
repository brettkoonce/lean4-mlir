#!/usr/bin/env python3
"""Generate the smoothing DECIMAL-radius scorecard (gaussian_smoothing_next.md,
the prefix-scan corpus pass).

The CP scorecard (SmoothingCPScorecard.lean) certifies per image the largest
4-decimal q0 = a/10000 with binomTail n k q0 <= alpha; its radius column
sigma*Phi^-1(q0) is display-only float. This script closes that gap for the
whole corpus via the ONE-kernel-evaluation prefix scan:

  * recompute phiScanRev (1/1000) 3300 — the descending list
    [phiGridUB 3300, ..., phiGridUB 0] — in EXACT rational arithmetic
    (mirrors ratExpLB / ratPdfUB / ratCeil9 / phiGridUB from
    SmoothingPhiBounds.lean; any drift fails the emitted `decide +kernel`
    equation at build, so the mirror is untrusted);
  * emit the literal + `phiScanLit_eq : phiScanRev (1/1000) 3300 = phiScanLit`
    (the single expensive kernel evaluation, ~2 min);
  * per certified image, the LARGEST grid index m with
    phiGridUB (1/1000) m <= a/10000: one cheap `decide +kernel` list lookup
    then certifies radius m/2000 <= (1/2)*Phi^-1(a/10000)
    (Proofs.smooth_radius_dec).

Image set and a-values are identical to the CP scorecard (same CSVs, same
largest_a search — imported from smooth_scorecard_gen.py).

Output: LeanMlir/Proofs/SmoothingDecScorecard.lean
"""

import csv
import math
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from smooth_scorecard_gen import NETS, RUNS, largest_a, phi_inv  # noqa: E402

D = 10000                 # q0 denominator (4 decimals)
H = Fraction(1, 1000)     # grid step
N_GRID = 3300             # grid extent: phiGridUB(3.3) ~ 0.99990 > max q0 0.9993
D12 = 10**12              # common denominator of every grid value
# Chunk boundaries for the scan verification: ONE whole-grid kernel eval
# peaks at ~15 GB of retained kernel-cache bignums (OOM on 16 GB CI
# runners), and memory is NOT reclaimed between declarations within one
# lean process (a 6-chunk single file still peaked 12.9 GB) — so each
# chunk gets its OWN MODULE/process (heaviest: chunk 6 at ~5.2 GB).
CHUNKS = [550, 1100, 1650, 2200, 2750, 3300]
CHUNK_OUT = "LeanMlir/Proofs/SmoothingDecChunk{c}.lean"
OUT = Path("LeanMlir/Proofs/SmoothingDecScorecard.lean")


# ── exact mirror of the SmoothingPhiBounds.lean kernel functions ──

def rat_exp_lb(x: Fraction) -> Fraction:
    """ratExpLB: 32-term Taylor lower bound for exp on [0, oo)."""
    s, t = Fraction(0), Fraction(1)
    for i in range(32):
        if i > 0:
            t = t * x / i
        s += t
    return s


def rat_pdf_ub(a: Fraction) -> Fraction:
    """ratPdfUB: 1/(2.5066282 * ratExpLB(a^2/2)) >= phi(a)."""
    return 1 / (Fraction(25066282, 10000000) * rat_exp_lb(a * a / 2))


def rat_ceil9(q: Fraction) -> Fraction:
    """ratCeil9: round up to denominator 10^9."""
    return Fraction(math.ceil(q * 10**9), 10**9)


def phi_scan(n: int) -> list[Fraction]:
    """ASCENDING [phiGridUB H 0, ..., phiGridUB H n] (phiScanRev reversed)."""
    scan = [Fraction(1, 2)]
    for m in range(n):
        scan.append(scan[-1] + H * rat_ceil9(rat_pdf_ub(m * H)))
    return scan


def largest_m(scan: list[Fraction], a: int) -> int:
    """Largest grid index m with scan[m] <= a/10000 (scan strictly increasing)."""
    q0 = Fraction(a, D)
    lo, hi = 0, len(scan) - 1
    assert scan[0] <= q0, f"scan[0] > {a}/10000"
    assert scan[hi] > q0, f"grid too short for a={a} — raise N_GRID"
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if scan[mid] <= q0:
            lo = mid
        else:
            hi = mid - 1
    return lo


def main() -> None:
    print(f"computing exact scan to {N_GRID} panels ...", flush=True)
    scan = phi_scan(N_GRID)

    per_net = {}
    for slug, _name in NETS:
        path = RUNS / f"smooth_{slug}_scorecard.csv"
        if not path.exists():
            sys.exit(f"missing {path} — run ./run_smooth_scorecard.sh first")
        entries = []   # (img_idx, label, pred, a, m, driver_radius, correct)
        with open(path) as f:
            for r in csv.DictReader(f):
                if int(r["abstain"]) == 1:
                    continue
                assert float(r["sigma"]) == 0.5, "protocol fixes sigma=0.5"
                a = largest_a(int(r["n"]), int(r["count"]))
                if a is None:
                    continue
                m = largest_m(scan, a)
                entries.append((int(r["img_idx"]), int(r["label"]),
                                int(r["pred"]), a, m, 0.5 * phi_inv(a / D),
                                r["label"] == r["pred"]))
        per_net[slug] = entries
        certified = [e for e in entries if e[4] > 0]
        loss = [e[5] - e[4] / 2000 for e in entries]
        print(f"{slug}: {len(entries)} images, {len(certified)} with positive "
              f"decimal radius; float-vs-certified gap avg {sum(loss)/len(loss):.4f} "
              f"max {max(loss):.4f}", flush=True)

    L = []
    L.append(f"import LeanMlir.Proofs.SmoothingDecChunk{len(CHUNKS)}\n")
    L.append("/-! # The smoothing DECIMAL-radius scorecard — Φ⁻¹ leaves symbolic-land, corpus-wide\n")
    L.append("GENERATED by scripts/smooth_dec_scorecard_gen.py from the same fixed-protocol")
    L.append("driver runs as SmoothingCPScorecard.lean (first-100 test images, σ = 0.5,")
    L.append("n = 10112, α = 1/1000; identical per-image q₀ = a/10000). The CP scorecard's")
    L.append("radius column σ·Φ⁻¹(q₀) was display-only float; here each certified image gets")
    L.append("a PROVED decimal lower bound: `m/2000 ≤ σ·Φ⁻¹(a/10000)` via `smooth_radius_dec`.")
    L.append("")
    L.append("The trick that makes 279 images affordable: `phiScanLit_eq` kernel-evaluates the")
    L.append("ENTIRE `h = 1/1000` upper-Riemann grid (`phiScanRev`, 3300 panels, descending,")
    L.append("~2 min total); each per-image check is then a single O(index) list lookup —")
    L.append("into the SMALLEST verified prefix covering its m, so a `decide +kernel` walks")
    L.append("one 551-entry chunk, never the whole grid (milliseconds each, and the module")
    L.append("stays ~2 GB where full-literal lookups accumulated 11.3 GB — kernel memory is")
    L.append("not reclaimed between declarations). The grid is verified in CHUNKS of 550 panels,")
    L.append("one MODULE each (SmoothingDecChunk1–6: `phiScanRevFrom` continued from literal")
    L.append("checkpoints, glued by `phiScanRevFrom_append`): one whole-grid declaration")
    L.append("retains ~15 GB of kernel-cache bignums and OOMs 16 GB CI runners, and memory")
    L.append("is not reclaimed between declarations within a lean process — per-module")
    L.append("processes cap the worst chunk at ~5 GB. Per image, m is the LARGEST grid index with")
    L.append("`phiGridUB (1/1000) m ≤ q₀`, so the bound is grid-optimal; the intrinsic")
    L.append("upper-sum slack costs ~0.003–0.036 vs the driver's float printout (largest at")
    L.append("the q₀ = 0.9993 unanimous-count images).")
    L.append("")
    L.append("Honest scope: the net-semantics hypotheses (C = a net's argmax + `hp`")
    L.append("interiority) are discharged for a concrete trained net in")
    L.append("SmoothingNetWitness.lean; the scorecard's own 784-dim driver checkpoints")
    L.append("remain untied (the same witness-generator pass at full width). -/")
    L.append("")
    L.append("namespace Proofs")
    L.append("")
    def num_of(q: Fraction) -> int:
        assert D12 % q.denominator == 0, q
        return q.numerator * (D12 // q.denominator)

    # ── chunk MODULES: chunk c covers grid indices bounds[c-1]..bounds[c],
    # descending, INCLUSIVE of the checkpoint bounds[c-1] as its last entry.
    # One module (= one lean process = one RSS budget) per chunk: memory is
    # not reclaimed between declarations inside a process. ──
    bounds = [0] + CHUNKS
    assert CHUNKS[-1] == N_GRID
    acc = "phiChunkLit1"
    for c in range(1, len(bounds)):
        k, kc = bounds[c - 1], bounds[c]
        seg = [num_of(scan[i]) for i in range(kc, k - 1, -1)]
        C = []
        if c == 1:
            C.append("import LeanMlir.Proofs.SmoothingPhiBounds\n")
        else:
            C.append(f"import LeanMlir.Proofs.SmoothingDecChunk{c-1}\n")
        C.append(f"/-! # Decimal-radius scorecard, scan chunk {c}/{len(CHUNKS)} "
                 f"(panels {k}…{kc})\n")
        C.append("GENERATED by scripts/smooth_dec_scorecard_gen.py — see")
        C.append("SmoothingDecScorecard.lean for the corpus story. Each chunk lives in its")
        C.append("own module because (a) one whole-grid kernel evaluation retains ~15 GB of")
        C.append("kernel-cache bignums (OOM on 16 GB CI runners) and (b) memory is NOT")
        C.append("reclaimed between declarations within one lean process — per-module")
        C.append("processes cap the worst chunk at ~5 GB. -/")
        C.append("")
        C.append("namespace Proofs")
        C.append("")
        C.append("set_option maxRecDepth 20000 in")
        C.append(f"/-- Grid values {kc}…{k} (descending) as NUMERATORS over the common")
        C.append("    denominator `10¹²` (every grid value is `1/2 + Σ (1/1000)·(k/10⁹)`).")
        C.append("    ℕ literals elaborate in ~1 s where flat `ℚ` division literals price")
        C.append("    out in pending-mvar instance synthesis (>10 min). Untrusted input —")
        C.append("    verified by the kernel checks below. -/")
        C.append(f"def phiChunkNum{c} : List ℕ := [{', '.join(map(str, seg))}]")
        C.append("")
        C.append(f"def phiChunkLit{c} : List ℚ := "
                 f"phiChunkNum{c}.map (fun n => (n:ℚ)/1000000000000)")
        C.append("")
        if c == 1:
            C.append("set_option maxRecDepth 20000 in")
            C.append("set_option maxHeartbeats 16000000 in")
            C.append(f"/-- Chunk 1: grid panels 0…{kc}, kernel-evaluated. -/")
            C.append(f"lemma phiScanEq{kc} : phiScanRev (1/1000) {kc} = phiChunkLit1 := by")
            C.append("  decide +kernel")
        else:
            v = f"{num_of(scan[k])}/1000000000000"
            C.append(f"/-- Checkpoint: the grid value at panel {k}, read off the verified scan. -/")
            C.append(f"lemma phiCp{k} : phiGridUB (1/1000) {k} = {v} := by")
            C.append(f"  rw [← phiScanRev_headI (1/1000) {k}, phiScanEq{k}]")
            C.append("  decide +kernel")
            C.append("")
            C.append("set_option maxRecDepth 20000 in")
            C.append("set_option maxHeartbeats 16000000 in")
            C.append(f"/-- Chunk {c}: panels {k}…{kc}, kernel-evaluated FROM the checkpoint. -/")
            C.append(f"lemma phiChunkEq{c} :")
            C.append(f"    phiScanRevFrom (1/1000) {k} ({v}) {kc - k} = phiChunkLit{c} := by")
            C.append("  decide +kernel")
            C.append("")
            C.append(f"/-- Panels 0…{kc}: chunk {c} glued onto the verified prefix. -/")
            C.append(f"lemma phiScanEq{kc} : phiScanRev (1/1000) {kc}")
            C.append(f"    = phiChunkLit{c} ++ ({acc}).tail := by")
            C.append(f"  rw [show ({kc}:ℕ) = {k} + {kc - k} by norm_num,")
            C.append(f"    ← phiScanRevFrom_append (1/1000) {k} {kc - k},")
            C.append(f"    phiScanEq{k}, phiCp{k}, phiChunkEq{c}]")
            acc = f"phiChunkLit{c} ++ ({acc}).tail"
        C.append("")
        C.append("end Proofs")
        Path(CHUNK_OUT.format(c=c)).write_text("\n".join(C) + "\n")

    nums = [num_of(q) for q in reversed(scan)]
    L.append("set_option maxRecDepth 20000 in")
    L.append(f"/-- The FULL scan literal's numerators: entry `i` is `phiGridUB (1/1000)")
    L.append(f"    ({N_GRID} − i)` (descending) over `10¹²`. -/")
    L.append(f"def phiScanNum : List ℕ := [{', '.join(map(str, nums))}]")
    L.append("")
    L.append("/-- The scan literal: the ℕ numerators over `10¹²`. -/")
    L.append("def phiScanLit : List ℚ := phiScanNum.map (fun n => (n:ℚ)/1000000000000)")
    L.append("")
    L.append("set_option maxRecDepth 20000 in")
    L.append("set_option maxHeartbeats 16000000 in")
    L.append("/-- The whole grid scan against the flat literal (chunks reassembled;")
    L.append("    the final decide is literal-vs-literal, no panel arithmetic). -/")
    L.append(f"lemma phiScanLit_eq : phiScanRev (1/1000) {N_GRID} = phiScanLit := by")
    L.append(f"  rw [phiScanEq{N_GRID}]")
    L.append("  decide +kernel")
    L.append("")
    for slug, name in NETS:
        entries = per_net[slug]
        L.append(f"-- ── {name}: certified decimal radii, {len(entries)} images ──")
        L.append("")
        for (idx, label, pred, a, m, radius, correct) in entries:
            tag = "" if correct else "  (misclassified)"
            # The check runs against the SMALLEST verified prefix covering m:
            # `getD` with index < chunk size never forces the appended tail,
            # so each decide walks one 551-entry chunk, not the whole grid
            # (279 full-literal decides accumulated to 11.3 GB in one module).
            kc = min(b for b in CHUNKS if b >= m)
            L.append(f"/-- img {idx}: label {label}, pred {pred}, q₀ = {a}/10000 → "
                     f"radius ≥ {m}/2000 = {m/2000:.4f} (driver float: {radius:.3f}){tag} -/")
            L.append(f"lemma smooth_dec_{slug}_i{idx} :")
            L.append(f"    ({m}:ℝ)/2000 ≤ (1/2:ℝ) * stdNormalQuantile (({a}:ℝ)/10000) :=")
            L.append(f"  smooth_radius_dec phiScanEq{kc} {m} {a} (by norm_num) (by norm_num)")
            L.append(f"    (by norm_num) (by decide +kernel)")
            L.append("")
        pairs = ", ".join(f"({e[4]}, {e[3]})" for e in entries)
        L.append(f"/-- {name} aggregate: (m, a) per certified image — decimal radius "
                 f"m/2000 for q₀ = a/10000. -/")
        L.append(f"def smoothDec{slug.capitalize()}Entries : List (ℕ × ℕ) := [{pairs}]")
        L.append("")
        L.append(f"/-- Every {name} scorecard image's decimal radius is a theorem: "
                 f"m/2000 ≤ σ·Φ⁻¹(a/10000). -/")
        L.append(f"theorem smoothDec{slug.capitalize()}_certified :")
        L.append(f"    ∀ e ∈ smoothDec{slug.capitalize()}Entries,")
        L.append(f"      (e.1:ℝ)/2000 ≤ (1/2:ℝ) * stdNormalQuantile ((e.2:ℝ)/10000) :=")
        names = [f"smooth_dec_{slug}_i{e[0]}" for e in entries]
        if len(names) == 1:
            L.append(f"  List.forall_iff_forall_mem.mp {names[0]}")
        else:
            L.append("  List.forall_iff_forall_mem.mp")
            for i in range(0, len(names), 5):
                chunk = ", ".join(names[i:i + 5])
                prefix = "    ⟨" if i == 0 else "     "
                suffix = "⟩" if i + 5 >= len(names) else ","
                L.append(f"{prefix}{chunk}{suffix}")
        L.append("")
    L.append("end Proofs")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
