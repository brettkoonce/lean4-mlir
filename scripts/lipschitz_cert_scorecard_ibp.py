"""IBP (interval bound propagation) L-infinity scorecard for the full-input nets.

Produces LeanMlir/Proofs/Certificates/LipschitzCertScorecardIBP.lean (capped sigma<=2 net)
and LipschitzCertScorecardIBPUncon.lean: per-image certificates in the
literature-standard PIXEL L-INFINITY model, eps in {1,2,4,8}/255 —
`forall delta, (forall i, |delta_i| <= eps) -> argmax fixed` — via exact
rational interval propagation (IntervalBound.lean: sign-split dense boxes,
endpoint-max ReLU, `ibp2_certified_at_eps`).

IBP is LINEAR in width (unlike the Gram/LipSDP certificates), and the first
layer sees a uniform box, so its interval image is `<w,x> -/+ eps*||w||_1`
(`denseLo_uniform`) — the certificate reuses the committed scorecard's kernel
`dotZ` facts for `<w,x>` (hpre*) and needs just ONE new kernel fact per
weight row: `absSumZ w = A` (the l1 norm, `decide +kernel`).

Reuses the base generator's nets/images by importing it (byte-identical
base re-emission; harmless).
"""
import numpy as np
from fractions import Fraction
from pathlib import Path

import lipschitz_cert_scorecard_full as base

ROOT, OUTDIR = base.ROOT, base.OUTDIR
H, K, DIM, DEN, PIX = base.H, base.K, base.DIM, base.DEN, base.PIX
N_IMG = base.N_IMG
yte = base.yte
frac, zlist, rrow = base.frac, base.zlist, base.rrow
EPS_GRID = [(1, "e1"), (2, "e2"), (4, "e4"), (8, "e8")]   # eps = num/255

def logits_pre(W1q, i):
    x = base.Xte_raw[i].astype(object)
    return [int(v) for v in W1q.astype(object) @ x]      # /(DEN*PIX)

def ibp_box(W1q, W2q, i, num):
    """exact rational output box at eps = num/PIX; returns (y, outLo, outHi) numerators /(DEN^2*PIX)."""
    A1 = [int(v) for v in np.abs(W1q).sum(1)]
    pre = logits_pre(W1q, i)
    lo1 = [pre[k] - num * A1[k] * 1 for k in range(H)]   # /(DEN*PIX) with eps=num/PIX: eps*A/DEN = num*A/(DEN*PIX)
    hi1 = [pre[k] + num * A1[k] * 1 for k in range(H)]
    rl = [max(v, 0) for v in lo1]
    rh = [max(v, 0) for v in hi1]
    outLo, outHi = [], []
    for c in range(K):
        lo = sum(int(W2q[c][t]) * (rl[t] if W2q[c][t] >= 0 else rh[t]) for t in range(H))
        hi = sum(int(W2q[c][t]) * (rh[t] if W2q[c][t] >= 0 else rl[t]) for t in range(H))
        outLo.append(lo)
        outHi.append(hi)
    return int(yte[i]), outLo, outHi

def pgd_linf(W1q, W2q, eps, steps=100, restarts=4):
    W1 = W1q / DEN; W2 = W2q / DEN
    X = base.Xte_raw / 255.0
    robust = 0
    rng = np.random.default_rng(1)
    for i in range(N_IMG):
        x0 = X[i]; y = int(yte[i])
        if np.argmax(W2 @ np.maximum(W1 @ x0, 0)) != y:
            continue
        broken = False
        for r in range(restarts):
            d = (rng.uniform(-eps, eps, DIM) if r else np.zeros(DIM))
            for _ in range(steps):
                pre = W1 @ (x0 + d); hr = np.maximum(pre, 0)
                zz = W2 @ hr; zz -= zz.max(); p = np.exp(zz); p /= p.sum()
                g = p.copy(); g[y] -= 1
                gx = W1.T @ ((W2.T @ g) * (pre > 0))
                d = np.clip(d + (2.5 * eps / steps) * np.sign(gx), -eps, eps)
            if np.argmax(W2 @ np.maximum(W1 @ (x0 + d), 0)) != y:
                broken = True; break
        robust += not broken
    return robust

def emit_net(tag, W1q, W2q, out_path):
    A1 = [int(v) for v in np.abs(W1q).sum(1)]
    # which images certify at which eps (exact)
    certs = {}       # en -> list of images
    for num, en in EPS_GRID:
        certs[en] = []
        for i in range(N_IMG):
            y, outLo, outHi = ibp_box(W1q, W2q, i, num)
            if all(outHi[j] < outLo[y] for j in range(K) if j != y):
                certs[en].append(i)
    # empirical bracket + L2-Lipschitz-implied comparison (||d||2 <= 28*eps_inf)
    pgd = {}
    l2impl = {}
    L = base.info[tag]["L"]
    mp = base.info[tag]["mp"]
    PGD_CACHE = Path("/tmp") / f"ibp_pgd_{tag}_h{H}.npz"
    if PGD_CACHE.exists():
        zc = np.load(PGD_CACHE)
        pgd = {en: int(zc[en]) for _, en in EPS_GRID}
    else:
        for num, en in EPS_GRID:
            pgd[en] = pgd_linf(W1q, W2q, num / 255)
        np.savez(PGD_CACHE, **{en: pgd[en] for _, en in EPS_GRID})
    for num, en in EPS_GRID:
        eps = Fraction(num, 255)
        l2impl[en] = sum(1 for i in range(N_IMG)
                         if mp[i][0] > 0 and base.SQRT2_UB * L * 28 * eps <= mp[i][0])
    print(f"[{tag}] IBP certified:", {en: len(certs[en]) for _, en in EPS_GRID},
          "PGD-Linf:", pgd, "L2-Lipschitz-implied:", l2impl, flush=True)

    base_hpre = {i for i, tags in base.need.items() if tag in tags}
    base_imgs = set(base.need.keys())
    all_certified = sorted({i for _, en in EPS_GRID for i in certs[en]})

    Lb = []
    A = Lb.append
    A("import LeanMlir.Proofs.Foundation.IntervalBound")
    A("import LeanMlir.Proofs.Certificates.LipschitzCertScorecardFullImgsA")
    A("import LeanMlir.Proofs.Certificates.LipschitzCertScorecardFullImgsB")
    A("")
    netdesc = ("spectrally-capped σ≤2 net (`mlpSF`)" if tag == "SF"
               else "unconstrained net (`mlpTF`)")
    A(f"/-! # IBP L∞ scorecard, full 784-dim input — {netdesc}")
    A("")
    A(f"Pixel-L∞ certificates by exact interval bound propagation over the SAME")
    A(f"first-{N_IMG} MNIST test subset as the L2 scorecard: at ε = 1/255, 2/255,")
    A(f"4/255, 8/255 the box certificate proves")
    cline = ", ".join(f"**{len(certs[en])}/{N_IMG}**" for _, en in EPS_GRID)
    A(f"{cline} predictions robust (PGD-L∞ bracket: "
      + ", ".join(str(pgd[en]) for _, en in EPS_GRID) + ").")
    A("For comparison, pushing the L2 Lipschitz certificate through")
    A(f"`‖δ‖₂ ≤ √784·ε∞` certifies only "
      + ", ".join(str(l2impl[en]) for _, en in EPS_GRID)
      + " — at small L∞ radii the box beats the ball.")
    A("")
    A("Engine: `IntervalBound.lean`. The first layer is a uniform box, so its")
    A("interval image is `⟨w,x⟩ ∓ ε·‖w‖₁` — `⟨w,x⟩` reuses the committed `dotZ`")
    A("pre-activation facts (`hpre*`), `‖w‖₁` is ONE new kernel fact per row")
    A("(`absSumZ`, `decide +kernel`); the 16-wide second layer is sign-split")
    A("rational arithmetic. Counts are lower bounds (a loose box cannot prove")
    A("an image UNcertifiable). Generated by")
    A("`scripts/lipschitz_cert_scorecard_ibp.py`; weights/images are DATA. -/")
    A("")
    A(base.HEADER_OPTS)
    A(f"-- ════════ per-row ℓ1 norms: one kernel `absSumZ` fact each ════════")
    A("")
    for k in range(H):
        A(f"theorem az{tag}{k} : absSumZ w1z{tag}{k} = {A1[k]} := by decide +kernel")
    A("")
    A(f"/-- `absr{tag} t = ‖row t of W1{tag}‖₁` (data, verified below). -/")
    A(f"noncomputable def absr{tag} : Fin {H} → ℝ :=")
    A("  ![" + ", ".join(f"(({A1[k]} : ℝ)/{DEN})" for k in range(H)) + "]")
    A("")
    for k in range(H):
        A(f"theorem absrow{tag}_{k} : (∑ j, |W1{tag} {k} j|) = absr{tag} {k} := by")
        A(f"  rw [show absr{tag} {k} = (({A1[k]} : ℝ)/{DEN}) from rfl, W1{tag}r{k}]")
        A(f"  simp only [w1r{tag}{k}]")
        A(f"  rw [sum_getD_abs_div w1z{tag}{k}_len az{tag}{k} (by norm_num)]")
        A("  norm_num")
        A("")
    leaves = "; ".join(f"exact absrow{tag}_{k}" for k in range(H))
    A(f"theorem absrow{tag} : ∀ t, (∑ j, |W1{tag} t j|) = absr{tag} t := by")
    A("  intro t")
    A("  fin_cases t <;>")
    A(f"    [ {leaves} ]")
    A("")
    A("-- ════════ per-image certificates ════════")
    A("")
    for i in all_certified:
        y = int(yte[i])
        # image + hpre fallback for images the base scorecard didn't certify
        if i not in base_imgs:
            base.emit_image(A, i, [])
        if i in base_hpre:
            hpre = f"hpre{tag}{i}"
        else:
            hpre = f"hpreI{tag}{i}"
            pre = logits_pre(W1q, i)
            A(f"-- net {tag} pre-activations for #{i} (not in the base L2-certified set)")
            for k in range(H):
                A(f"theorem pzI{tag}_{i}_{k} : dotZ w1z{tag}{k} imgz{i} = {int(pre[k])} := by decide +kernel")
            A("")
            A(f"noncomputable def {hpre} : Fin {H} → ℝ :=")
            A("  " + rrow(pre, DEN * PIX))
            A("")
            for k in range(H):
                A(f"theorem {hpre}_eval_{k} : denseE W1{tag} imgF{i} {k} = {hpre} {k} := by")
                A(f"  rw [show {hpre} {k} = (({int(pre[k])} : ℝ)/{DEN * PIX}) from rfl,")
                A(f"      denseE_apply, W1{tag}r{k}]")
                A(f"  simp only [w1r{tag}{k}, imgF{i}_apply]")
                A(f"  rw [sum_getD_div w1z{tag}{k}_len imgz{i}_len pzI{tag}_{i}_{k} {DEN} {PIX}]")
                A(f"  norm_num")
                A("")
            lv = "; ".join(f"exact {hpre}_eval_{k}" for k in range(H))
            A(f"theorem {hpre}_eval : ∀ k : Fin {H}, denseE W1{tag} imgF{i} k = {hpre} k := by")
            A("  intro k")
            A("  fin_cases k <;>")
            A(f"    [ {lv} ]")
            A("")
        A(f"theorem {hpre}_sum : ∀ t, (∑ j, W1{tag} t j * imgF{i} j) = {hpre} t := by")
        A("  intro t")
        A("  rw [← denseE_apply]")
        A(f"  exact {hpre}_eval t")
        A("")
        for num, en in EPS_GRID:
            if i not in certs[en]:
                continue
            eps = f"(({num} : ℝ)/255)"
            box = (f"(fun q => imgF{i} q - {eps}) (fun q => imgF{i} q + {eps})")
            A("set_option maxHeartbeats 6400000 in")
            A(f"theorem hb{tag}{en}_{i} : ∀ j : Fin {K}, j ≠ {y} →")
            A(f"    denseHi W2{tag} (reluLo (denseLo W1{tag} {box}))")
            A(f"                (reluHi (denseHi W1{tag} {box})) j")
            A(f"      < denseLo W2{tag} (reluLo (denseLo W1{tag} {box}))")
            A(f"                    (reluHi (denseHi W1{tag} {box})) {y} := by")
            A("  intro j hj")
            A("  rw [denseLo2_eval, denseHi2_eval]")
            A(f"  simp only [{hpre}_sum, absrow{tag}]")
            A("  fin_cases j <;>")
            A("    first")
            A("    | exact absurd rfl hj")
            A(f"    | · simp [W2{tag}, {hpre}, absr{tag}, Fin.sum_univ_succ, max_def]")
            A("        try norm_num")
            A("")
            A(f"/-- Test #{i} (digit {y}): IBP-certified at pixel-L∞ ε = {num}/255. -/")
            A(f"theorem certIBP{tag}{en}_{i} (δ : EuclideanSpace ℝ (Fin {DIM}))")
            A(f"    (hδ : ∀ q, |δ q| ≤ {eps}) :")
            A(f"    ∀ j, j ≠ {y} → mlp{tag} (imgF{i} + δ) j < mlp{tag} (imgF{i} + δ) {y} :=")
            A(f"  ibp2_certified_at_eps W1{tag} W2{tag} hb{tag}{en}_{i} δ hδ")
            A("")
    A("-- ════════ aggregates (peers of `scorecardFull`) ════════")
    A("")
    stem = "ibpCapped" if tag == "SF" else "ibpUncon"
    parts = []
    for num, en in EPS_GRID:
        if not certs[en]:
            continue
        lname = f"{stem}Certs{en}"
        eps = f"(({num} : ℝ)/255)"
        A(f"noncomputable def {lname} : List (ℕ × EuclideanSpace ℝ (Fin {DIM}) × Fin {K}) :=")
        A("  [" + ",\n   ".join(f"({i}, imgF{i}, {int(yte[i])})" for i in certs[en]) + "]")
        A("")
        A(f"theorem {lname}_certified :")
        A(f"    ∀ p ∈ {lname}, CertifiedAtLinf mlp{tag} {eps} p.2.1 p.2.2 :=")
        A("  List.forall_iff_forall_mem.mp")
        A("    ⟨" + ", ".join(f"certIBP{tag}{en}_{i}" for i in certs[en]) + "⟩")
        A("")
        parts.append((lname, en, eps, len(certs[en])))
    A(f"/-- **The IBP L∞ scorecard, {netdesc}**: "
      + ", ".join(f"{len(certs[en])}/{N_IMG} @ {num}/255" for num, en in EPS_GRID)
      + f" (PGD-L∞ bracket " + "/".join(str(pgd[en]) for _, en in EPS_GRID) + "). -/")
    A(f"theorem scorecard_ibp{'' if tag == 'SF' else '_uncon'} :")
    A("    " + " ∧\n    ".join(
        f"({ln}.length = {cnt} ∧ ∀ p ∈ {ln}, CertifiedAtLinf mlp{tag} {ep} p.2.1 p.2.2)"
        for ln, en, ep, cnt in parts) + " :=")
    A("  ⟨" + ", ".join(f"⟨rfl, {ln}_certified⟩" for ln, en, ep, cnt in parts) + "⟩")
    A("")
    A("end LipschitzCertDemo")
    A("end Proofs")
    out_path.write_text("\n".join(Lb) + "\n")
    print(f"[{tag}] wrote {out_path}: {len(Lb)} lines")

emit_net("SF", *base.nets["SF"], OUTDIR / "LipschitzCertScorecardIBP.lean")
emit_net("TF", *base.nets["TF"], OUTDIR / "LipschitzCertScorecardIBPUncon.lean")
