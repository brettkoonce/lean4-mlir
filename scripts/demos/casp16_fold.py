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
that the machinery recovers a known fold. The starts × hands copies are folded as one batch (the step is launch-bound, not FLOP-bound); a 300-residue target takes seconds on a GPU, ~20 s on CPU threads."""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "datasets"))
from casp16_labels import EDGES, NBINS, FAR, UNOBS, OBINS, PBINS, ORIENT_CUT

ROOT = Path(os.environ.get("CASP16_DIR", Path(__file__).resolve().parents[2] / "data" / "casp16"))
NC = NBINS + 2
DEV = torch.device("cpu")                                                        # set by --device
CENTRES = torch.tensor((EDGES[:-1] + EDGES[1:]) / 2, dtype=torch.float32)   # 64 bin centres
SIGMA = float(EDGES[1] - EDGES[0])
# orientation restraints (`--orient`): trRosetta's ω (dihedral Cα_i–Cβ_i–Cβ_j–Cα_j) and φ (angle
# Cα_i–Cβ_i–Cβ_j) over the predicted 15° bins; θ needs N, which is not folded. Cα is folded as a
# unit direction from Cβ at 1.53 Å, so the bond is exact and the only new geometry term is the
# Cα–Cα virtual bond (3.8 Å). ω is chiral: with it the two hands no longer tie, and the hand is
# picked by energy (the dihedral rule is reported beside it).
OM_CENTRES = torch.tensor(-np.pi + (np.arange(OBINS) + 0.5) * 2 * np.pi / OBINS, dtype=torch.float32)
PH_CENTRES = torch.tensor((np.arange(PBINS) + 0.5) * np.pi / PBINS, dtype=torch.float32)
SIG_O, SIG_P = 2 * np.pi / OBINS, np.pi / PBINS
CA_CB, CA_CA = 1.53, 3.8


def set_device(name):
    """`auto` = CUDA when available. The potential is O(L²·64) per step: a 500-residue target
    is minutes on CPU threads and seconds on a GPU."""
    global DEV, CENTRES, OM_CENTRES, PH_CENTRES
    DEV = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else
                       ("cpu" if name == "auto" else name))
    CENTRES, OM_CENTRES, PH_CENTRES = CENTRES.to(DEV), OM_CENTRES.to(DEV), PH_CENTRES.to(DEV)
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
    c_om = np.zeros((len(SEP_EDGES), OBINS), np.float64)      # ω over i < j, φ over both orderings
    c_ph = np.zeros((len(SEP_EDGES), PBINS), np.float64)
    for r in reps[:n_chains]:
        lab = np.load(ROOT / "labels" / f"{r['id']}.npz")
        cls = lab["cls"]
        L = cls.shape[0]
        iu = np.triu_indices(L, 1)
        sb = sep_bin(iu[1] - iu[0])
        c = cls[iu]; ok = c < UNOBS
        np.add.at(counts, (sb[ok], c[ok]), 1.0)
        if "omega" in lab:
            o = lab["omega"][iu]; ok = o < OBINS
            np.add.at(c_om, (sb[ok], o[ok]), 1.0)
            for f in (lab["phi"][iu], lab["phi"][iu[1], iu[0]]):
                ok = f < PBINS
                np.add.at(c_ph, (sb[ok], f[ok]), 1.0)
    counts += 1.0
    ref = counts / counts.sum(1, keepdims=True)
    np.save(ROOT / "packed" / "ref_by_sep.npy", ref.astype(np.float32))
    if c_om.sum() > 0:
        c_om += 1.0; c_ph += 1.0
        np.savez(ROOT / "packed" / "ref_orient_by_sep.npz", omega=(c_om / c_om.sum(1, keepdims=True)).astype(np.float32),
                 phi=(c_ph / c_ph.sum(1, keepdims=True)).astype(np.float32))
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


def wrap(x):
    return torch.remainder(x + np.pi, 2 * np.pi) - np.pi


def angle(a, b, c):
    """Angle at b, over the last axis."""
    u, w = a - b, c - b
    cos = (u * w).sum(-1) / (u.norm(dim=-1) * w.norm(dim=-1)).clamp_min(1e-8)
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))


def orient_potential(pred, ref=None):
    """pred["omega"] [L, L, 26], pred["phi"] [L, L, 14] -> per pair the bin distribution renormalized
    over the angle bins and its weight (the mass on the bins, i.e. P(Cβ–Cβ < 20 Å and observed));
    with `ref` (the training-label background by separation) the same for p_ref."""
    om = torch.as_tensor(np.asarray(pred["omega"], np.float32), device=DEV)
    ph = torch.as_tensor(np.asarray(pred["phi"], np.float32), device=DEV)
    c_om, c_ph = om[:, :, :OBINS].sum(-1), ph[:, :, :PBINS].sum(-1)
    op = dict(p_om=om[:, :, :OBINS] / c_om.clamp_min(1e-8)[..., None], c_om=c_om,
              p_ph=ph[:, :, :PBINS] / c_ph.clamp_min(1e-8)[..., None], c_ph=c_ph, r_om=None, r_ph=None)
    if ref is not None:
        L = om.shape[0]
        sb = sep_bin(np.maximum(np.abs(np.arange(L)[:, None] - np.arange(L)[None, :]), 1))
        op["r_om"] = torch.as_tensor(ref["omega"][sb], dtype=torch.float32, device=DEV)
        op["r_ph"] = torch.as_tensor(ref["phi"][sb], dtype=torch.float32, device=DEV)
    return op


def orient_energy(xb, xa, iu, op, eps=1e-4):
    """xb, xa [K, L, 3] (Cβ, Cα): −Σ c_ij log(smoothed p(ω_ij)) over i < j and the same for φ over
    both orderings, each smoothed by a Gaussian of one bin (wrapped for ω) and read against the
    reference state when given. Returns [K]."""
    i, j = iu
    om = dihedral(xa[:, i], xb[:, i], xb[:, j], xa[:, j])
    g = torch.exp(-0.5 * (wrap(om[:, :, None] - OM_CENTRES) / SIG_O) ** 2)
    e = -(op["c_om"][i, j] * torch.log((op["p_om"][i, j] * g).sum(-1) + eps)).sum(-1)
    if op["r_om"] is not None:
        e = e + (op["c_om"][i, j] * torch.log((op["r_om"][i, j] * g).sum(-1) + eps)).sum(-1)
    for a, b in ((i, j), (j, i)):
        ph = angle(xa[:, a], xb[:, a], xb[:, b])
        g = torch.exp(-0.5 * ((ph[:, :, None] - PH_CENTRES) / SIG_P) ** 2)
        e = e - (op["c_ph"][a, b] * torch.log((op["p_ph"][a, b] * g).sum(-1) + eps)).sum(-1)
        if op["r_ph"] is not None:
            e = e + (op["c_ph"][a, b] * torch.log((op["r_ph"][a, b] * g).sum(-1) + eps)).sum(-1)
    return e


def ca_from(xb, n):
    """Cα = Cβ + 1.53 Å along the free direction n (the bond is exact by construction)."""
    return xb + CA_CB * n / n.norm(dim=-1, keepdim=True).clamp_min(1e-6)


def energy(x, p_bins, p_far, iu, r_bins=None, r_far=None, w_chain=1.0, w_clash=3.0, eps=1e-4,
           sigma=SIGMA, xa=None, op=None, w_orient=1.0):
    """x [K, L, 3]: K independent copies (starts × hands) folded in one batch — the energy is a sum
    over copies, so one Adam step moves each exactly as it would alone, and the launch-bound
    per-step cost is paid once. Returns the per-copy energies [K] and parts [K, 3]."""
    d = torch.cdist(x, x)                                         # [K, L, L]
    dij = d[:, iu[0], iu[1]]                                      # [K, P]
    g = torch.exp(-0.5 * ((dij[:, :, None] - CENTRES[None, None, :]) / sigma) ** 2)   # [K, P, 64]
    e_dist = -torch.log(smoothed(p_bins, p_far, iu, dij, g, sigma) + eps).sum(-1)
    if r_bins is not None:   # reference state: V = -log p + log p_ref
        e_dist = e_dist + torch.log(smoothed(r_bins, r_far, iu, dij, g, sigma) + eps).sum(-1)
    chain = d.diagonal(offset=1, dim1=1, dim2=2)
    e_chain = ((chain - 5.4) ** 2).sum(-1)
    far = iu[1] - iu[0] >= 2
    e_clash = (torch.relu(3.6 - dij[:, far]) ** 2).sum(-1)
    e_or = torch.zeros_like(e_dist)
    if op is not None:
        ca_chain = (xa[:, 1:] - xa[:, :-1]).norm(dim=-1)
        e_or = w_orient * orient_energy(x, xa, iu, op, eps) + w_chain * ((ca_chain - CA_CA) ** 2).sum(-1)
    return e_dist + w_chain * e_chain + w_clash * e_clash + e_or, torch.stack([e_dist, e_chain, e_clash, e_or], -1)


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
         w_chain=1.0, w_clash=3.0, sigma=SIGMA, orient=None, orient_ref=None, w_orient=1.0, lr=0.5):
    """`orient`: the prediction's omega/phi planes (see `orient_potential`) — then Cα directions
    are folded too, the hand is the lower-energy one, and results carry `ca`."""
    torch.manual_seed(seed)
    kw = dict(w_chain=w_chain, w_clash=w_clash, sigma=sigma)
    p_bins, p_far, r_bins, r_far = potential(probs, ref)
    op = orient_potential(orient, orient_ref) if orient is not None else None
    L = p_bins.shape[0]
    iu = torch.triu_indices(L, L, 1, device=DEV)
    iu = (iu[0], iu[1])
    starts = [mds_init(p_bins, p_far)] + [torch.randn(L, 3, device=DEV) * (2.0 * L ** (1 / 3)) for _ in range(restarts)]
    hands = torch.tensor([1.0, -1.0], device=DEV)
    # copies ordered (start 0, hand +), (start 0, hand −), (start 1, hand +), …
    x = torch.stack([x0 * torch.tensor([1.0, 1.0, h], device=DEV) for x0 in starts for h in (1.0, -1.0)]).clone().requires_grad_(True)
    params = [x]
    n = None
    if op is not None:
        n = torch.randn(x.shape, device=DEV).requires_grad_(True)
        params.append(n)
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=0.01)
    for t in range(steps):
        opt.zero_grad()
        e, _ = energy(x, p_bins, p_far, iu, r_bins, r_far, **kw,
                      xa=ca_from(x, n) if op is not None else None, op=op, w_orient=w_orient)
        e.sum().backward()
        opt.step(); sched.step()
    with torch.no_grad():
        xa = ca_from(x, n) if op is not None else None
        e, parts = energy(x.detach(), p_bins, p_far, iu, r_bins, r_far, **kw, xa=xa, op=op, w_orient=w_orient)
    results = []
    for k in range(x.shape[0]):
        xk = x[k].detach().cpu()
        results.append(dict(start=k // 2, hand=float(hands[k % 2]), energy=e[k].item(), parts=tuple(parts[k].tolist()),
                            x=xk, ca=xa[k].detach().cpu() if xa is not None else None, handedness=handedness(xk)))
        if verbose:
            r = results[-1]
            print(f"    start {r['start']} hand {r['hand']:+.0f}: E {r['energy']:.1f} (dist {r['parts'][0]:.1f}, chain {r['parts'][1]:.1f}, "
                  f"clash {r['parts'][2]:.1f}, orient {r['parts'][3]:.1f}) handedness {r['handedness']:+.2f}")
    results.sort(key=lambda r: r["energy"])
    best = results[0]
    mirror = best["x"] * torch.tensor([1.0, 1.0, -1.0])
    if op is None:
        # the two hands of the best basin have the same energy; the chirality score picks
        chosen, other = (best["x"], mirror) if handedness(best["x"]) >= 0 else (mirror, best["x"])
    else:
        chosen, other = best["x"], mirror      # ω is chiral: the hand is the lower-energy one
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
    ap.add_argument("--orient", action="store_true", help="add the ω and φ restraints from the prediction's orientation heads (fold Cα too)")
    ap.add_argument("--w-orient", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=0.5, help="Adam step (cosine to 0.01)")
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
    oref_path = ROOT / "packed" / "ref_orient_by_sep.npz"
    if a.orient and not a.no_ref and not oref_path.exists():
        build_reference()
    oref = None if (a.no_ref or not a.orient) else np.load(oref_path)
    eus = a.eus or sorted(f.name[:-9] for f in d.glob("*.pred.npz"))
    for eu in eus:
        t = np.load(ROOT / "targets" / f"{eu}.npz")
        seq, resnum = str(t["seq"]), t["resnum"]
        if a.max_len and len(seq) > a.max_len:
            print(f"{eu}: L {len(seq)} > {a.max_len}, skipped"); continue
        tag = (".truth" if a.from_truth else "") + (".orient" if a.orient else "")
        if not a.force and (d / f"{eu}{tag}.fold.pdb").exists():
            continue
        orient = None
        if a.from_truth:
            cls = t["cls"].astype(np.int64)
            probs = np.zeros(cls.shape + (NC,), np.float32)
            np.put_along_axis(probs, cls[:, :, None], 1.0, axis=2)
            probs[cls == UNOBS] = 1.0 / (NC - 1); probs[cls == UNOBS, UNOBS] = 0.0   # no information
            if a.orient:
                orient = {}
                for name, nb in (("omega", OBINS + 2), ("phi", PBINS + 2)):
                    k = t[name].astype(np.int64)
                    pl = np.zeros(k.shape + (nb,), np.float32)
                    np.put_along_axis(pl, k[:, :, None], 1.0, axis=2)
                    orient[name] = pl
        else:
            pred = np.load(d / f"{eu}.pred.npz")
            probs = pred["probs"]
            if a.orient:
                if "omega" not in pred:
                    print(f"{eu}: no orientation heads in the prediction, skipped"); continue
                orient = dict(omega=pred["omega"], phi=pred["phi"])
        t0 = time.time()
        x, xm, results = fold(probs, a.restarts, a.steps, verbose=a.verbose, ref=ref,
                              orient=orient, orient_ref=oref, w_orient=a.w_orient, lr=a.lr)
        write_pdb(d / f"{eu}{tag}.fold.pdb", x, seq, resnum)
        write_pdb(d / f"{eu}{tag}.fold_mirror.pdb", xm, seq, resnum)
        if results[0]["ca"] is not None:
            write_pdb(d / f"{eu}{tag}.fold_ca.pdb", results[0]["ca"].numpy(), seq, resnum)
        hand = handedness(torch.as_tensor(x))
        # copies change hands during the descent, so "the other hand" is by final chirality
        other_e = min((r["energy"] for r in results if (r["handedness"] >= 0) != (results[0]["handedness"] >= 0)), default=None)
        json.dump(dict(eu=eu, L=len(seq), best_energy=results[0]["energy"], parts=results[0]["parts"],
                       handedness=hand, energies=[r["energy"] for r in results], other_hand_energy=other_e),
                  open(d / f"{eu}{tag}.fold.json", "w"), indent=1)
        print(f"{eu}: L {len(seq)}, best E {results[0]['energy']:.1f} of {len(results)} (hand {hand:+.2f}"
              + (f", other hand's best E {other_e:.1f}" if a.orient and other_e is not None else "")
              + f"), {time.time() - t0:.0f} s -> {d / (eu + tag + '.fold.pdb')}")
