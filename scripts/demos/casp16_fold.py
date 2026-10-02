#!/usr/bin/env python3
"""Step 9 of planning/casp16_distogram_demo.md §3: fold a predicted distogram into coordinates.

The net predicts Cβ–Cβ distances (Cα for glycine), so what is folded is the pseudo-Cβ trace:
one point per residue, x ∈ ℝ^{L×3}, minimizing

    E(x) = Σ_{i<j} V_ij(‖x_i − x_j‖) + w_chain Σ_i (‖x_{i+1} − x_i‖ − 5.4)² + w_clash Σ_{|i−j|≥2} relu(3.6 − d_ij)²

where V_ij(d) = −log( Σ_k p_ij(k)·exp(−(d − c_k)²/2σ²) + p_ij(far)·sigmoid((d − 22)/σ) + ε ) is the
distogram smoothed by a Gaussian of one bin width over the bin centres (the AlphaFold-1
potential without its reference-state term), the "unobserved" class dropped and the rest
renormalized. Adam from a classical-MDS start (expected distances) plus random restarts, in
both hands — a distance potential cannot tell a structure from its mirror; the hand is picked
by the sign of the i, i+1, i+2, i+3 dihedral over helical stretches (right-handed α-helices),
and both are written. Output: <dir>/<EU>.fold.pdb (one ATOM named CA per residue at the
pseudo-Cβ position, target numbering), <EU>.fold_mirror.pdb, <EU>.fold.json (energies).
`--from-truth` folds the one-hot distogram of the experimental structure instead, the check
that the machinery recovers a known fold. CPU torch; a 300-residue target takes ~20 s."""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from casp16_labels import EDGES, NBINS, FAR, UNOBS

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
NC = NBINS + 2
DEV = torch.device("cpu")                                                        # set by --device
CENTRES = torch.tensor((EDGES[:-1] + EDGES[1:]) / 2, dtype=torch.float32)   # 64 bin centres
SIGMA = float(EDGES[1] - EDGES[0])


def set_device(name):
    """`auto` = CUDA when available. The potential is O(L²·64) per step: a 500-residue target
    is minutes on CPU threads and seconds on a GPU."""
    global DEV, CENTRES
    DEV = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                       ("cpu" if name == "auto" else name))
    CENTRES = CENTRES.to(DEV)
    return DEV
AA1 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS",
       "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
       "Y": "TYR", "V": "VAL"}


SEP_EDGES = np.array(list(range(1, 33)) + [48, 64, 96, 128, 10**9])   # separation bins of the reference


def sep_bin(sep):
    """Index into SEP_EDGES: separations 1..32 one each, then 33-48, 49-64, 65-96, 97-128, >128."""
    return np.searchsorted(SEP_EDGES, sep, side="left")


def build_reference(n_chains=3000, seed=0):
    """p_ref(class | separation): the class histogram of the training labels by |i - j|, the
    background a prediction is read against (AlphaFold 1's reference state) — without it the
    'far' class, which most pairs share, dominates the potential and inflates the fold."""
    import csv, random
    reps = list(csv.DictReader(open(ROOT / "train" / "train_full.csv")))
    random.Random(seed).shuffle(reps)
    counts = np.zeros((len(SEP_EDGES), NC - 1), np.float64)
    for r in reps[:n_chains]:
        cls = np.load(ROOT / "labels" / f"{r['id']}.npz")["cls"]
        L = cls.shape[0]
        iu = np.triu_indices(L, 1)
        c = cls[iu]; ok = c < UNOBS
        np.add.at(counts, (sep_bin(iu[1][ok] - iu[0][ok]), c[ok]), 1.0)
    counts += 1.0
    ref = counts / counts.sum(1, keepdims=True)
    np.save(ROOT / "packed" / "ref_by_sep.npy", ref.astype(np.float32))
    return ref


def potential(probs, ref=None):
    """probs [L, L, 66] -> (p_bins [L, L, 64], p_far [L, L]) renormalized without 'unobserved',
    and, with a reference, the same for p_ref broadcast to every pair by its separation."""
    p = torch.as_tensor(np.asarray(probs, np.float32), device=DEV)
    p = p[:, :, :NC - 1]
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-8)
    out = [p[:, :, :NBINS].contiguous(), p[:, :, FAR].contiguous()]
    if ref is not None:
        L = p.shape[0]
        sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
        r = torch.as_tensor(ref[sep_bin(np.maximum(sep, 1))], dtype=torch.float32, device=DEV)   # [L, L, 65]
        out += [r[:, :, :NBINS].contiguous(), r[:, :, FAR].contiguous()]
    else:
        out += [None, None]
    return tuple(out)


def smoothed(p_bins, p_far, iu, dij, g, sigma):
    return (p_bins[iu] * g).sum(-1) + p_far[iu] * torch.sigmoid((dij - EDGES[-1]) / sigma)


def energy(x, p_bins, p_far, iu, r_bins=None, r_far=None, w_chain=1.0, w_clash=3.0, eps=1e-4,
           sigma=SIGMA):
    d = torch.cdist(x, x)
    dij = d[iu]                                                   # [P]
    g = torch.exp(-0.5 * ((dij[:, None] - CENTRES[None, :]) / sigma) ** 2)   # [P, 64]
    e_dist = -torch.log(smoothed(p_bins, p_far, iu, dij, g, sigma) + eps).sum()
    if r_bins is not None:   # reference state: V = -log p + log p_ref
        e_dist = e_dist + torch.log(smoothed(r_bins, r_far, iu, dij, g, sigma) + eps).sum()
    chain = d.diagonal(1)
    e_chain = ((chain - 5.4) ** 2).sum()
    far = iu[1] - iu[0] >= 2
    e_clash = (torch.relu(3.6 - dij[far]) ** 2).sum()
    return e_dist + w_chain * e_chain + w_clash * e_clash, (e_dist.item(), e_chain.item(), e_clash.item())


def mds_init(p_bins, p_far):
    """Classical MDS on the expected distance matrix (far class at 25 Å)."""
    ed = (p_bins * CENTRES).sum(-1) + p_far * 25.0
    ed = 0.5 * (ed + ed.T); ed.fill_diagonal_(0)
    D2 = ed ** 2
    n = D2.shape[0]
    J = torch.eye(n, device=ed.device) - torch.full((n, n), 1.0 / n, device=ed.device)
    B = -0.5 * J @ D2 @ J
    w, v = torch.linalg.eigh(B)
    top = w[-3:].clamp_min(0).sqrt()
    return v[:, -3:] * top


def dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / b1.norm(dim=-1, keepdim=True)
    v = b0 - (b0 * b1n).sum(-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdim=True) * b1n
    xx = (v * w).sum(-1)
    yy = (torch.cross(b1n, v, dim=-1) * w).sum(-1)
    return torch.atan2(yy, xx)


def handedness(x):
    """Chirality score of a pseudo-Cβ trace from the sign of its i…i+3 dihedrals: helical
    quads (d(i, i+3) < 7 Å) count +sign, extended ones (≥ 8 Å) count −sign, the band between
    nothing. Calibrated on the 84 CASP16 EUs' true traces (2026-10-01): helical quads run
    +0.85 … +0.94, extended −0.26 … −0.49, and every EU scores positive — so the native hand is
    the positive one even for an all-β fold, and the mirror scores the exact negative."""
    d3 = (x[3:] - x[:-3]).norm(dim=-1)
    phi = dihedral(x[:-3], x[1:-2], x[2:-1], x[3:])
    w = torch.where(d3 < 7.0, 1.0, torch.where(d3 >= 8.0, -1.0, 0.0))
    return (torch.sign(phi) * w).mean().item()


def fold(probs, restarts=4, steps=1500, seed=0, verbose=False, ref=None,
         w_chain=1.0, w_clash=3.0, sigma=SIGMA):
    torch.manual_seed(seed)
    kw = dict(w_chain=w_chain, w_clash=w_clash, sigma=sigma)
    p_bins, p_far, r_bins, r_far = potential(probs, ref)
    L = p_bins.shape[0]
    iu = torch.triu_indices(L, L, 1, device=DEV)
    iu = (iu[0], iu[1])
    starts = [mds_init(p_bins, p_far)] + [torch.randn(L, 3, device=DEV) * (2.0 * L ** (1 / 3)) for _ in range(restarts)]
    results = []
    for si, x0 in enumerate(starts):
        for hand in (1.0, -1.0):
            x = (x0 * torch.tensor([1.0, 1.0, hand], device=DEV)).clone().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=0.5)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.01)
            for t in range(steps):
                opt.zero_grad()
                e, _ = energy(x, p_bins, p_far, iu, r_bins, r_far, **kw)
                e.backward()
                opt.step(); sched.step()
            e, parts = energy(x.detach(), p_bins, p_far, iu, r_bins, r_far, **kw)
            results.append(dict(start=si, hand=hand, energy=e.item(), parts=parts, x=x.detach().cpu(),
                                handedness=handedness(x.detach().cpu())))
            if verbose:
                print(f"    start {si} hand {hand:+.0f}: E {e.item():.1f} (dist {parts[0]:.1f}, chain {parts[1]:.1f}, clash {parts[2]:.1f}) handedness {results[-1]['handedness']:+.2f}")
    results.sort(key=lambda r: r["energy"])
    best = results[0]
    mirror = best["x"] * torch.tensor([1.0, 1.0, -1.0])
    # the two hands of the best basin have the same energy; the chirality score picks
    chosen, other = (best["x"], mirror) if handedness(best["x"]) >= 0 else (mirror, best["x"])
    return chosen.numpy(), other.numpy(), results


def write_pdb(path, x, seq, resnum):
    with open(path, "w") as f:
        for k, (s, n) in enumerate(zip(seq, resnum)):
            f.write(f"ATOM  {k + 1:5d}  CA  {AA1.get(s, 'UNK')} A{int(n):4d}    {x[k, 0]:8.3f}{x[k, 1]:8.3f}{x[k, 2]:8.3f}  1.00  0.00           C\n")
        f.write("END\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", help="prediction directory with <EU>.pred.npz, or any directory with --from-truth")
    ap.add_argument("eus", nargs="*", help="EUs to fold (default: every .pred.npz in dir)")
    ap.add_argument("--from-truth", action="store_true", help="fold the experimental structure's one-hot distogram")
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=0, help="skip EUs longer than this (0 = fold all)")
    ap.add_argument("--no-ref", action="store_true", help="no reference-state term (the plain −log p)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--force", action="store_true", help="refold EUs that already have a .fold.pdb")
    ap.add_argument("--build-ref", action="store_true", help="(re)build packed/ref_by_sep.npy from the training labels")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    print(f"device {set_device(a.device)}")
    d = Path(a.dir); d.mkdir(parents=True, exist_ok=True)
    ref_path = ROOT / "packed" / "ref_by_sep.npy"
    if a.build_ref or (not a.no_ref and not ref_path.exists()):
        ref = build_reference()
        print(f"reference state from the training labels -> {ref_path}  (P(far) at sep 1, 8, 24, >128: "
              f"{ref[sep_bin(1), FAR]:.2f} {ref[sep_bin(8), FAR]:.2f} {ref[sep_bin(24), FAR]:.2f} {ref[-1, FAR]:.2f})")
    ref = None if a.no_ref else np.load(ref_path)
    eus = a.eus or sorted(f.name[:-9] for f in d.glob("*.pred.npz"))
    for eu in eus:
        t = np.load(ROOT / "targets" / f"{eu}.npz")
        seq, resnum = str(t["seq"]), t["resnum"]
        if a.max_len and len(seq) > a.max_len:
            print(f"{eu}: L {len(seq)} > {a.max_len}, skipped"); continue
        tag = ".truth" if a.from_truth else ""
        if not a.force and (d / f"{eu}{tag}.fold.pdb").exists():
            continue
        if a.from_truth:
            cls = t["cls"].astype(np.int64)
            probs = np.zeros(cls.shape + (NC,), np.float32)
            np.put_along_axis(probs, cls[:, :, None], 1.0, axis=2)
            probs[cls == UNOBS] = 1.0 / (NC - 1); probs[cls == UNOBS, UNOBS] = 0.0   # no information
            tag = ".truth"
        else:
            probs = np.load(d / f"{eu}.pred.npz")["probs"]
            tag = ""
        t0 = time.time()
        x, xm, results = fold(probs, a.restarts, a.steps, verbose=a.verbose, ref=ref)
        write_pdb(d / f"{eu}{tag}.fold.pdb", x, seq, resnum)
        write_pdb(d / f"{eu}{tag}.fold_mirror.pdb", xm, seq, resnum)
        json.dump(dict(eu=eu, L=len(seq), best_energy=results[0]["energy"], parts=results[0]["parts"],
                       handedness=handedness(torch.as_tensor(x)), energies=[r["energy"] for r in results]),
                  open(d / f"{eu}{tag}.fold.json", "w"), indent=1)
        print(f"{eu}: L {len(seq)}, best E {results[0]['energy']:.1f} of {len(results)} (hand {handedness(torch.as_tensor(x)):+.2f}), "
              f"{time.time() - t0:.0f} s -> {d / (eu + tag + '.fold.pdb')}")
