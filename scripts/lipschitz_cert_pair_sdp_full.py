"""Per-pair LipSDP pass for the FULL-INPUT (784-dim) scorecard.

Produces LeanMlir/Proofs/Certificates/LipschitzCertScorecardSDPFull.lean (capped sigma<=2
net) and LipschitzCertScorecardSDPFullUncon.lean: for each ordered class pair
needed by a certified image, a LipSDP-Neuron certificate (Fazlyab 2019, one
hidden layer)

    (g x - g x')^2 <= rho * ||x - x'||^2,   g = <W2_i - W2_j, relu(W1 .)>

witnessed by an exact rational LDL^T factorization of
S = 2 diag(T) - vv^T - (1/rho) T G1 T, discharged in Lean by the pooled
files' recipe -- expand the quadratic form once, then linarith with the
LDL^T column squares as hints. (MEASURED faster than the "deterministic"
lipsdp_slack_of_cert entrywise route at BOTH widths; the exact-LDL
fractions hurt 512 separate norm_num goals far more than one linarith
goal.) Everything pair-level is 16x16 (Schur:
the input dimension never appears); the 784-dim ingredients (G1*_eq Gram
wrappers, per-image hpre evals) are REUSED from the committed full-input
scorecard (LipschitzCertScorecardFull*.lean, kernel dotZ engine).

Certification per image at eps in {1/10, 3/10}: for every j != y,
Lp_{y,j} * eps <= logit_y - logit_j (exact rationals).

Reuses the base generator's trained nets/caches by importing it (module
import re-emits the base files byte-identically; harmless).
"""
import numpy as np
from fractions import Fraction
from math import ceil, sqrt
from pathlib import Path
from scipy.linalg import sqrtm
from scipy.optimize import minimize

import lipschitz_cert_scorecard_full as base

ROOT = base.ROOT
OUTDIR = base.OUTDIR
H, K, DIM, DEN, PIX = base.H, base.K, base.DIM, base.DEN, base.PIX
N_IMG = base.N_IMG
EPS10, EPS30 = base.EPS10, base.EPS30
yte = base.yte
frac, zlist, rrow = base.frac, base.zlist, base.rrow

# ── LipSDP per pair: numeric solve + exact rational certificate ──
def rho_of_theta(theta, Q, G1f):
    T = np.diag(np.exp(theta))
    A = 2 * T - Q
    ew = np.linalg.eigvalsh(A)
    if ew.min() <= 1e-9:
        return 1e12 * (1 - ew.min())
    Ah = np.real(sqrtm(np.linalg.inv(A)))
    Mm = Ah @ T @ G1f @ T @ Ah
    return np.linalg.eigvalsh(Mm).max()

def solve_pair(vf, G1f):
    Q = np.outer(vf, vf)
    best = None
    for s in (max(np.abs(vf).max() ** 2, 1e-2), 2 * float(vf @ vf) + 1e-2, 10.0):
        r = minimize(rho_of_theta, np.log(np.full(H, s)), args=(Q, G1f),
                     method="Nelder-Mead",
                     options={"maxiter": 12000, "xatol": 1e-11, "fatol": 1e-13})
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    return best

def ldl_exact(S):
    Lm = [[Fraction(0)] * H for _ in range(H)]
    dm = [Fraction(0)] * H
    for i in range(H):
        s = S[i][i] - sum(Lm[i][k] ** 2 * dm[k] for k in range(i))
        if s < 0:
            return None, None
        dm[i] = s
        Lm[i][i] = Fraction(1)
        for jj in range(i + 1, H):
            num = S[jj][i] - sum(Lm[jj][k] * Lm[i][k] * dm[k] for k in range(i))
            if s == 0:
                if num != 0:
                    return None, None
                Lm[jj][i] = Fraction(0)
            else:
                Lm[jj][i] = num / s
    return Lm, dm

def rational_cert(a, b, W2q, G1q):
    """Certificate for the unordered pair {a,b}; S depends only on {a,b}."""
    W2f = W2q / DEN
    vf = W2f[a] - W2f[b]
    G1f = np.array([[float(Fraction(int(G1q[x][y]), DEN ** 2)) for y in range(H)]
                    for x in range(H)])
    rho_f, theta = solve_pair(vf, G1f)
    TQ = [max(Fraction(round(np.exp(t) * 1024), 1024), Fraction(0)) for t in theta]
    vQ = [Fraction(int(W2q[a][k] - W2q[b][k]), DEN) for k in range(H)]
    GQ = [[Fraction(int(G1q[x][y]), DEN ** 2) for y in range(H)] for x in range(H)]
    for bump in (1.01, 1.02, 1.05, 1.1, 1.25, 1.5):
        rhoQ = Fraction(ceil(rho_f * bump * 10 ** 6), 10 ** 6)
        S = [[2 * (TQ[x] if x == y else Fraction(0)) - vQ[x] * vQ[y]
              - Fraction(1) / rhoQ * (TQ[x] * GQ[x][y] * TQ[y])
              for y in range(H)] for x in range(H)]
        Lm, dm = ldl_exact(S)
        if Lm is not None:
            break
    assert Lm is not None, f"LDL failed for pair ({a},{b})"
    LpQ = Fraction(ceil(sqrt(float(rhoQ)) * 10 ** 4), 10 ** 4)
    while LpQ ** 2 < rhoQ:
        LpQ += Fraction(1, 10 ** 4)
    for x in range(H):
        for y in range(H):
            assert S[x][y] == sum(Lm[x][i] * dm[i] * Lm[y][i] for i in range(H))
    assert all(d >= 0 for d in dm) and all(t >= 0 for t in TQ)
    return dict(rho=rhoQ, T=TQ, v=vQ, S=S, L=Lm, d=dm, Lp=LpQ)

def logits_exact(W1q, W2q, i):
    x = base.Xte_raw[i].astype(object)
    pre = [int(v) for v in W1q.astype(object) @ x]
    hid = np.array([max(v, 0) for v in pre], dtype=object)
    logit = [int(v) for v in W2q.astype(object) @ hid]   # /(DEN^2*PIX)
    return pre, logit

def fvec(vals):
    return "![" + ", ".join(frac(v) for v in vals) + "]"

def fmat(M):
    return "![" + ",\n    ".join(fvec(r) for r in M) + "]"

HEADER_OPTS = base.HEADER_OPTS

def emit_net(tag, W1q, W2q, out_path):
    """tag: 'SF' (capped) or 'TF' (uncon); reuses W1{tag}/W2{tag}/G1{tag}/
    G1{tag}_eq/mlp{tag}/imgF<i>/hpre{tag}<i>_eval from the base files."""
    G1q = W1q.astype(object) @ W1q.astype(object).T
    den_l = DEN * DEN * PIX
    pair_cache = {}
    def cert_of(a, b):
        key = (min(a, b), max(a, b))
        if key not in pair_cache:
            pair_cache[key] = rational_cert(key[0], key[1], W2q, G1q)
        return pair_cache[key]

    # exact per-image facts + which images certify at which eps
    facts = {}
    for i in range(N_IMG):
        pre, logit = logits_exact(W1q, W2q, i)
        y = int(yte[i])
        ms = {j: Fraction(logit[y] - logit[j], den_l) for j in range(K) if j != y}
        facts[i] = (y, ms, pre, logit)
    cert10, cert30 = [], []
    for i in range(N_IMG):
        y, ms, _, _ = facts[i]
        if min(ms.values()) <= 0:
            continue
        if all(cert_of(y, j)["Lp"] * EPS10 <= ms[j] for j in ms):
            cert10.append(i)
        if all(cert_of(y, j)["Lp"] * EPS30 <= ms[j] for j in ms):
            cert30.append(i)
    ordered = sorted({(facts[i][0], j) for i in cert10 + cert30
                      for j in range(K) if j != facts[i][0]})
    unordered = sorted({(min(a, b), max(a, b)) for a, b in ordered})
    print(f"[{tag}] SDP certified @0.1: {len(cert10)}/{N_IMG}, @0.3: {len(cert30)}/{N_IMG}; "
          f"{len(unordered)} unordered pairs", flush=True)

    # base-file coverage: which images already have imgF/hpre (union of base cert10 sets)
    base_imgs = set(base.need.keys())            # imgF<i> exists
    base_hpre = {i for i, tags in base.need.items() if tag in tags}

    L = []
    A = L.append
    A("import LeanMlir.Proofs.Certificates.LipschitzCertPairSDP")
    A("import LeanMlir.Proofs.Certificates.LipschitzCertScorecardFullImgsA")
    A("import LeanMlir.Proofs.Certificates.LipschitzCertScorecardFullImgsB")
    A("")
    netdesc = ("spectrally-capped σ≤2 net (`mlpSF`)" if tag == "SF"
               else "unconstrained net (`mlpTF`)")
    b10 = len(base.info[tag]["cert10"]); b30 = len(base.info[tag]["cert30"])
    p10 = base.info[tag]["pgd10"]; p30 = base.info[tag]["pgd30"]
    A(f"/-! # Per-pair LipSDP scorecard, FULL 784-dim input — {netdesc}")
    A("")
    A(f"The tighter-Lipschitz-constant pass over the SAME first-{N_IMG} MNIST test")
    A(f"subset as `LipschitzCertScorecardFull.lean`, at BOTH ε = 1/10 and 3/10:")
    A(f"per-pair LipSDP constants lift the counts from **{b10}→{len(cert10)}/{N_IMG} @ ε=0.1**")
    A(f"(PGD bracket {p10}) and **{b30}→{len(cert30)}/{N_IMG} @ ε=0.3** (PGD {p30}) — no")
    A("retraining, no new data, a less lossy constant per pairwise logit gap.")
    A("")
    A("Everything pair-level is hidden-width-sized (16×16, Schur — the 784-dim")
    A("input never appears); the PSD witness is an exact rational LDLᵀ, checked")
    A("as one `linarith` goal per pair from the column squares (`hS*` — measured")
    A("faster than the entrywise `lipsdp_slack_of_cert` route at both widths).")
    A("The 784-term work — Gram wrappers `G1*_eq`, the per-image `hpre*_eval` —")
    A("is reused from the kernel-dotZ scorecard files.")
    A("")
    A("Generated by `scripts/lipschitz_cert_pair_sdp_full.py`; weights/images/")
    A("certificates are DATA (SDP solved off-line, verified exactly here). -/")
    A("")
    A(HEADER_OPTS)
    A("-- ════════════════════════════════════════════════════════════")
    A(f"-- § Pair certificates ({len(unordered)} unordered class pairs)")
    A("-- ════════════════════════════════════════════════════════════")
    A("")
    for (a, b) in unordered:
        c = cert_of(a, b)
        nm = f"{a}{b}{tag}"
        A(f"/-- Pair ({a},{b}): ρ = {float(c['rho']):.3f}, Lp = {float(c['Lp']):.4f} "
          f"(the global √2·L criterion charges ≈ {float(base.SQRT2_UB * base.info[tag]['L']):.2f} to every pair). -/")
        A(f"noncomputable def vP{nm} : Fin {H} → ℝ := {fvec(c['v'])}")
        A("")
        A(f"noncomputable def tP{nm} : Fin {H} → ℝ := {fvec(c['T'])}")
        A("")
        A(f"theorem vP{nm}_eq : ∀ t, vP{nm} t = W2{tag} {a} t - W2{tag} {b} t := by")
        A("  intro t")
        A("  fin_cases t <;>")
        A(f"    · simp [vP{nm}, W2{tag}]")
        A("      try norm_num")
        A("")
        A(f"theorem tP{nm}_nonneg : ∀ k, 0 ≤ tP{nm} k := by")
        A("  intro k")
        A("  fin_cases k <;>")
        A(f"    · simp [tP{nm}]")
        A("      try norm_num")
        A("")
        # the PSD slack: expand the sums once, then linarith with the 16 LDLᵀ
        # column squares as hints (the pooled files' recipe). MEASURED faster
        # than the lipsdp_slack_of_cert route at BOTH widths (h=8: 26 vs 62
        # CPU-min file-level; h=16: ~14 s vs ~40-60 s per pair) — the exact-LDL
        # fractions (denominators to ~230 digits at h=16) hurt 512 separate
        # entrywise norm_num goals far more than one linarith goal.
        hints = []
        for col in range(H):
            terms = " + ".join(
                (f"z {r}" if c["L"][r][col] == 1 else f"{frac(c['L'][r][col])} * z {r}")
                for r in range(H) if c["L"][r][col] != 0)
            if terms:
                hints.append(f"sq_nonneg ({terms})")
        A("set_option maxHeartbeats 64000000 in")
        A(f"theorem hS{nm} : ∀ z : Fin {H} → ℝ,")
        A(f"    (∑ k, vP{nm} k * z k) ^ 2")
        A(f"      + (1/{frac(c['rho'])}) * (∑ a, ∑ b, (tP{nm} a * z a) * (G1{tag} a b * (tP{nm} b * z b)))")
        A(f"      ≤ 2 * ∑ k, tP{nm} k * z k ^ 2 := by")
        A("  intro z")
        A(f"  simp [vP{nm}, tP{nm}, G1{tag}, Fin.sum_univ_succ]")
        A("  linarith [" + ",\n    ".join(hints) + "]")
        A("")
        A(f"theorem pairSq{tag}_{a}_{b} : ∀ u u' : EuclideanSpace ℝ (Fin {DIM}),")
        A(f"    ((mlp{tag} u {a} - mlp{tag} u {b}) - (mlp{tag} u' {a} - mlp{tag} u' {b})) ^ 2")
        A(f"      ≤ {frac(c['rho'])} * ‖u - u'‖ ^ 2 := by")
        A("  intro u u'")
        A(f"  have e : ∀ w : EuclideanSpace ℝ (Fin {DIM}), mlp{tag} w {a} - mlp{tag} w {b}")
        A(f"      = ∑ t, vP{nm} t * max (denseE W1{tag} w t) 0 := by")
        A("    intro w")
        A(f"    have h : mlp{tag} w {a} - mlp{tag} w {b}")
        A(f"        = ∑ t, (W2{tag} {a} t - W2{tag} {b} t) * max (denseE W1{tag} w t) 0 :=")
        A(f"      mlp_gap_eq W1{tag} W2{tag} {a} {b} w")
        A("    rw [h]")
        A(f"    exact Finset.sum_congr rfl fun t _ => by rw [vP{nm}_eq t]")
        A("  rw [e u, e u']")
        A(f"  exact pair_sq_bound W1{tag} G1{tag} G1{tag}_eq vP{nm} tP{nm} tP{nm}_nonneg (by norm_num)")
        A(f"    hS{nm} u u'")
        A("")
    for (a, b) in ordered:
        if a < b:
            continue
        lo, hi = b, a
        c = cert_of(lo, hi)
        A(f"theorem pairSq{tag}_{a}_{b} : ∀ u u' : EuclideanSpace ℝ (Fin {DIM}),")
        A(f"    ((mlp{tag} u {a} - mlp{tag} u {b}) - (mlp{tag} u' {a} - mlp{tag} u' {b})) ^ 2")
        A(f"      ≤ {frac(c['rho'])} * ‖u - u'‖ ^ 2 := by")
        A("  intro u u'")
        A(f"  have e : (mlp{tag} u {a} - mlp{tag} u {b}) - (mlp{tag} u' {a} - mlp{tag} u' {b})")
        A(f"      = -((mlp{tag} u {lo} - mlp{tag} u {hi}) - (mlp{tag} u' {lo} - mlp{tag} u' {hi})) := by")
        A("    ring")
        A("  rw [e, neg_sq]")
        A(f"  exact pairSq{tag}_{lo}_{hi} u u'")
        A("")
    A("-- ════════════════════════════════════════════════════════════")
    A(f"-- § Per-image certificates ({len(cert10)} @ ε=1/10, {len(cert30)} @ ε=3/10)")
    A("-- ════════════════════════════════════════════════════════════")
    A("")
    imgs_emitted = set()
    for i in sorted(set(cert10) | set(cert30)):
        y, ms, pre, logit = facts[i]
        # image data + hpre if the base files don't have them for this net
        if i not in base_imgs and i not in imgs_emitted:
            imgs_emitted.add(i)
            base.emit_image(A, i, [])          # imgz/imgv/imgF only (no tags)
        if i not in base_hpre:
            A(f"-- net {tag} pre-activations for #{i} (not in the base scorecard set)")
            for k in range(H):
                A(f"theorem pz{tag}_{i}_{k} : dotZ w1z{tag}{k} imgz{i} = {int(pre[k])} := by decide +kernel")
            A("")
            A(f"noncomputable def hpre{tag}{i} : Fin {H} → ℝ :=")
            A("  " + rrow(pre, DEN * PIX))
            A("")
            for k in range(H):
                A(f"theorem hpre{tag}{i}_eval_{k} : denseE W1{tag} imgF{i} {k} = hpre{tag}{i} {k} := by")
                A(f"  rw [show hpre{tag}{i} {k} = (({int(pre[k])} : ℝ)/{DEN * PIX}) from rfl,")
                A(f"      denseE_apply, W1{tag}r{k}]")
                A(f"  simp only [w1r{tag}{k}, imgF{i}_apply]")
                A(f"  rw [sum_getD_div w1z{tag}{k}_len imgz{i}_len pz{tag}_{i}_{k} {DEN} {PIX}]")
                A(f"  norm_num")
                A("")
            leaves = "; ".join(f"exact hpre{tag}{i}_eval_{k}" for k in range(H))
            A(f"theorem hpre{tag}{i}_eval : ∀ k : Fin {H}, denseE W1{tag} imgF{i} k = hpre{tag}{i} k := by")
            A("  intro k")
            A("  fin_cases k <;>")
            A(f"    [ {leaves} ]")
            A("")
        lg = f"logit{tag}{i}"
        A(f"noncomputable def {lg} : Fin {K} → ℝ :=")
        A("  " + rrow(logit, den_l))
        A("")
        A("set_option maxHeartbeats 3200000 in")
        A(f"theorem {lg}_eval : ∀ jj : Fin {K}, mlp{tag} imgF{i} jj = {lg} jj := by")
        A(f"  have hout : ∀ jj : Fin {K}, mlp{tag} imgF{i} jj = ∑ k, W2{tag} jj k * max (hpre{tag}{i} k) 0 := by")
        A("    intro jj")
        A(f"    show denseE W2{tag} (reluE (denseE W1{tag} imgF{i})) jj = _")
        A("    rw [denseE_apply]")
        A("    refine Finset.sum_congr rfl fun k _ => ?_")
        A(f"    rw [reluE_apply, hpre{tag}{i}_eval k]")
        A("  intro jj")
        A("  rw [hout jj]")
        A("  fin_cases jj <;>")
        A(f"    · simp [W2{tag}, hpre{tag}{i}, {lg}, Fin.sum_univ_succ, max_def]")
        A("      try norm_num")
        A("")
        for epsname, eps, certs in (("10", EPS10, cert10), ("30", EPS30, cert30)):
            if i not in certs:
                continue
            A("set_option maxHeartbeats 3200000 in")
            A(f"/-- Test #{i} (digit {y}): LipSDP-per-pair certified at ε = {eps} — each")
            A("    of the 9 margins clears its own `Lp·ε`. -/")
            A(f"theorem certifiedS{tag}{epsname}_{i} (δ : EuclideanSpace ℝ (Fin {DIM})) (hδ : ‖δ‖ < {frac(eps)}) :")
            A(f"    ∀ j, j ≠ {y} → mlp{tag} (imgF{i} + δ) j < mlp{tag} (imgF{i} + δ) {y} := by")
            A("  intro j hj")
            A("  fin_cases j")
            for j in range(K):
                if j == y:
                    A("  · exact absurd rfl hj")
                else:
                    c = cert_of(y, j)
                    A(f"  · refine certified_at_eps_pair (Lp := {frac(c['Lp'])}) pairSq{tag}_{y}_{j}")
                    A("      (by norm_num) (by norm_num) ?_ δ hδ")
                    A(f"    rw [{lg}_eval, {lg}_eval]")
                    A(f"    simp [{lg}]")
                    A("    try norm_num")
            A("")
    A("-- ════════════════════════════════════════════════════════════")
    A("-- § Aggregates — mechanized (peers of `scorecardFull`)")
    A("-- ════════════════════════════════════════════════════════════")
    A("")
    stem = "sdpCappedFull" if tag == "SF" else "sdpUnconFull"
    for epsname, eps, certs in (("10", EPS10, cert10), ("30", EPS30, cert30)):
        lname = f"{stem}Certs{epsname}"
        A(f"noncomputable def {lname} : List (ℕ × EuclideanSpace ℝ (Fin {DIM}) × Fin {K}) :=")
        trips = ",\n   ".join(f"({i}, imgF{i}, {facts[i][0]})" for i in certs)
        A(f"  [{trips}]")
        A("")
        A(f"theorem {lname}_certified :")
        A(f"    ∀ p ∈ {lname}, CertifiedAt mlp{tag} {frac(eps)} p.2.1 p.2.2 :=")
        A("  List.forall_iff_forall_mem.mp")
        A("    ⟨" + ", ".join(f"certifiedS{tag}{epsname}_{i}" for i in certs) + "⟩")
        A("")
    A(f"/-- **The LipSDP full-input scorecard, {netdesc}**: {b10}→{len(cert10)}/{N_IMG} @ ε=0.1")
    A(f"    (PGD {p10}) and {b30}→{len(cert30)}/{N_IMG} @ ε=0.3 (PGD {p30}) — same net, same images;")
    A("    the constant was the bottleneck, not the network. Lower bounds only. -/")
    A(f"theorem scorecard_sdp_full{'' if tag == 'SF' else '_uncon'} :")
    A(f"    ({stem}Certs10.length = {len(cert10)} ∧")
    A(f"      ∀ p ∈ {stem}Certs10, CertifiedAt mlp{tag} {frac(EPS10)} p.2.1 p.2.2) ∧")
    A(f"    ({stem}Certs30.length = {len(cert30)} ∧")
    A(f"      ∀ p ∈ {stem}Certs30, CertifiedAt mlp{tag} {frac(EPS30)} p.2.1 p.2.2) :=")
    A(f"  ⟨⟨rfl, {stem}Certs10_certified⟩, ⟨rfl, {stem}Certs30_certified⟩⟩")
    A("")
    A("end LipschitzCertDemo")
    A("end Proofs")
    out_path.write_text("\n".join(L) + "\n")
    print(f"[{tag}] wrote {out_path}: {len(L)} lines")
    return cert10, cert30

c10, c30 = emit_net("SF", *base.nets["SF"], OUTDIR / "LipschitzCertScorecardSDPFull.lean")
u10, u30 = emit_net("TF", *base.nets["TF"], OUTDIR / "LipschitzCertScorecardSDPFullUncon.lean")
