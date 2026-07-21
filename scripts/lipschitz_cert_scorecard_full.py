"""FULL-INPUT certified-accuracy scorecard generator (post-audit gap #3:
lift the Lipschitz scorecard off the 4x4-pooled 49-dim reduction).

Produces LeanMlir/Proofs/LipschitzCertScorecardFull{Nets,ImgsA,ImgsB,}.lean:
over the first 100 MNIST test images at FULL 784-dim input (exact k/255
pixels), per-image Tsuzuku certificates at pixel-L2 eps = 1/10 AND 3/10, on
two 784->16->10 nets:

* UNCONSTRAINED: 12 epochs plain SGD, /256-rationalized;
* SPECTRALLY CAPPED: same recipe + projected SGD onto sigma_max <= 2
  (host-side rescale each step, as mnist-mlp-spectral), 36 epochs.

The 49-dim pooled scorecard (lipschitz_cert_scorecard.py) evaluated its dot
products by simp/norm_num -- quadratic in sum length, unusable at 784. Here
every 784-term dot is ONE kernel evaluation (`dotZ ... = v := by decide
+kernel` on List-int data, GMP-fast, propext-only) bridged to the real-level
`Fin 784` sums by Proofs/ListDot.lean's `sum_getD_div` (proved once).
Negative list literals are emitted as `Int.negSucc` to dodge a ~15ms/element
Neg-elaboration tax.

Also runs L2-PGD (empirical, not proof) on both nets at both eps for the
cert <= TRUE <= PGD sandwich.
"""
import numpy as np, struct
from fractions import Fraction
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data"
OUTDIR = ROOT / "LeanMlir" / "Proofs"
N_IMG = 100
H, K, DIM = 16, 10, 784
DEN = 256
PIX = 255
BS, LR = 64, 0.15
CAP, EP_CAP, EP_UNC = 2.0, 36, 12
EPS10, EPS30 = Fraction(1, 10), Fraction(3, 10)
SQRT2_UB = Fraction(14143, 10000)   # >= sqrt 2; the factor certified_at_eps uses

def load_images(fn):
    with open(fn, "rb") as f:
        _, n, r, c = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r * c)

def load_labels(fn):
    with open(fn, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)

Xtr_raw = load_images(D / "train-images-idx3-ubyte"); ytr = load_labels(D / "train-labels-idx1-ubyte")
Xte_raw = load_images(D / "t10k-images-idx3-ubyte"); yte = load_labels(D / "t10k-labels-idx1-ubyte")
Xtr = Xtr_raw / 255.0; Xte = Xte_raw / 255.0

def train(cap, epochs, seed=0):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0, np.sqrt(2.0 / DIM), (H, DIM))
    W2 = rng.normal(0, np.sqrt(2.0 / H), (K, H))
    for _ in range(epochs):
        idx = rng.permutation(len(Xtr))
        for b in range(0, len(Xtr), BS):
            xb = Xtr[idx[b:b+BS]]; yb = ytr[idx[b:b+BS]]
            hp = xb @ W1.T; hr = np.maximum(hp, 0); z = hr @ W2.T
            z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
            g = p.copy(); g[np.arange(len(yb)), yb] -= 1; g /= len(yb)
            W1 -= LR * ((g @ W2) * (hp > 0)).T @ xb; W2 -= LR * g.T @ hr
            if cap is not None:
                s1 = np.linalg.svd(W1, compute_uv=False)[0]
                s2 = np.linalg.svd(W2, compute_uv=False)[0]
                if s1 > cap: W1 *= cap / s1
                if s2 > cap: W2 *= cap / s2
    return W1, W2

def s8_B(Wq, den):
    """rational B with B^8 >= tr((Wq Wq^T)^4)/den^8, i.e. B >= Schatten-8 >= sigma_1."""
    G = Wq.astype(object) @ Wq.astype(object).T
    Hm = G @ G
    S = Fraction(int((Hm ** 2).sum()), den ** 8)
    B = Fraction(ceil((float(S) ** 0.125) * 1000), 1000)
    while B ** 8 < S:
        B += Fraction(1, 1000)
    return B

def margins_exact(W1q, W2q):
    """(margin Fraction, hidden pre-activation numerators /(DEN*PIX)) per image, exact."""
    out = []
    for i in range(N_IMG):
        x = Xte_raw[i].astype(object)
        pre = [int(v) for v in W1q.astype(object) @ x]          # /(DEN*PIX)
        hid = np.array([max(v, 0) for v in pre], dtype=object)
        logit = [int(v) for v in W2q.astype(object) @ hid]      # /(DEN^2*PIX)
        y = int(yte[i])
        m = Fraction(logit[y] - max(logit[c] for c in range(K) if c != y),
                     DEN * DEN * PIX)
        out.append((m, pre))
    return out

def pgd_robust(W1f, W2f, eps, steps=100, restarts=4):
    robust = 0
    rng = np.random.default_rng(1)
    for i in range(N_IMG):
        x0 = Xte[i]; y = int(yte[i])
        if np.argmax(W2f @ np.maximum(W1f @ x0, 0)) != y:
            continue
        broken = False
        for r in range(restarts):
            d = rng.normal(size=DIM) if r else np.zeros(DIM)
            if r: d *= eps / np.linalg.norm(d)
            for _ in range(steps):
                pre = W1f @ (x0 + d); hr = np.maximum(pre, 0)
                z = W2f @ hr; z = z - z.max()
                p = np.exp(z); p /= p.sum()
                gz = p.copy(); gz[y] -= 1
                g = W1f.T @ ((W2f.T @ gz) * (pre > 0))
                gn = np.linalg.norm(g)
                if gn > 0: d += (2.5 * eps / steps) * g / gn
                dn = np.linalg.norm(d)
                if dn > eps: d *= eps / dn
            if np.argmax(W2f @ np.maximum(W1f @ (x0 + d), 0)) != y:
                broken = True; break
        robust += not broken
    return robust

# ── nets (cached: training is deterministic, PGD/margins cheap to redo) ──
CACHE = Path("/tmp") / f"lipschitz_full_w_h{H}_cap{CAP}_{EP_CAP}_{EP_UNC}.npz"
if CACHE.exists():
    z = np.load(CACHE)
    W1c, W2c, W1u, W2u = z["W1c"], z["W2c"], z["W1u"], z["W2u"]
else:
    W1c, W2c = train(CAP, EP_CAP)
    W1u, W2u = train(None, EP_UNC)
    np.savez(CACHE, W1c=W1c, W2c=W2c, W1u=W1u, W2u=W2u)
nets = {}
nets["SF"] = (np.round(W1c * DEN).astype(np.int64), np.round(W2c * DEN).astype(np.int64))
nets["TF"] = (np.round(W1u * DEN).astype(np.int64), np.round(W2u * DEN).astype(np.int64))

info = {}
for tag, (W1q, W2q) in nets.items():
    W1f, W2f = W1q / DEN, W2q / DEN
    acc = (np.argmax(np.maximum(Xte @ W1f.T, 0) @ W2f.T, 1) == yte).mean()
    B1, B2 = s8_B(W1q, DEN), s8_B(W2q, DEN)
    L = B1 * B2
    mp = margins_exact(W1q, W2q)
    cert10 = [i for i in range(N_IMG) if mp[i][0] > 0 and SQRT2_UB * L * EPS10 <= mp[i][0]]
    cert30 = [i for i in range(N_IMG) if mp[i][0] > 0 and SQRT2_UB * L * EPS30 <= mp[i][0]]
    PGD_CACHE = Path("/tmp") / f"lipschitz_full_pgd_{tag}_h{H}_cap{CAP}_{EP_CAP}_{EP_UNC}.npz"
    if PGD_CACHE.exists():
        zc = np.load(PGD_CACHE)
        pgd10, pgd30 = int(zc["p10"]), int(zc["p30"])
    else:
        pgd10 = pgd_robust(W1f, W2f, float(EPS10))
        pgd30 = pgd_robust(W1f, W2f, float(EPS30))
        np.savez(PGD_CACHE, p10=pgd10, p30=pgd30)
    info[tag] = dict(acc=acc, B1=B1, B2=B2, L=L, mp=mp,
                     cert10=cert10, cert30=cert30, pgd10=pgd10, pgd30=pgd30)
    print(f"{tag}: qacc={acc:.4f} B1={float(B1):.3f} B2={float(B2):.3f} L={float(L):.3f} "
          f"cert@0.1={len(cert10)} cert@0.3={len(cert30)} PGD@0.1={pgd10} PGD@0.3={pgd30}",
          flush=True)

# ═══ Lean emission helpers ═══
def zlit(x):
    x = int(x)
    return str(x) if x >= 0 else f"Int.negSucc {-x - 1}"

def zlist(vals):
    return "[" + ", ".join(zlit(v) for v in vals) + "]"

def frac(q):
    q = Fraction(q)
    return f"(({q.numerator} : ℝ)/{q.denominator})" if q.denominator != 1 else f"({q.numerator} : ℝ)"

def rrow(vals, den):
    return "![" + ", ".join(f"(({int(v)} : ℝ)/{den})" for v in vals) + "]"

def rmat(M, den):
    return "![" + ",\n    ".join(rrow(r, den) for r in M) + "]"

HEADER_OPTS = """set_option maxRecDepth 100000
set_option maxHeartbeats 3200000

namespace Proofs
namespace LipschitzCertDemo

open scoped BigOperators
"""

def emit_net(A, tag, W1q, W2q, B1, B2, L):
    """weights, Gram-entry lemmas via dotZ, Schatten-8 Lipschitz theorems."""
    den2, den4 = DEN**2, DEN**4
    G1 = W1q.astype(object) @ W1q.astype(object).T
    H1 = G1 @ G1
    G2 = W2q.astype(object) @ W2q.astype(object).T
    H2 = G2 @ G2
    A(f"-- ════════ net {tag}: rows as ℤ-lists, layer-1 Gram via kernel dotZ ════════")
    A("")
    for k in range(H):
        A(f"def w1z{tag}{k} : List ℤ := {zlist(W1q[k])}")
    A("")
    for k in range(H):
        A(f"theorem w1z{tag}{k}_len : (w1z{tag}{k}).length = {DIM} := by decide +kernel")
    A("")
    for k in range(H):
        A(f"noncomputable def w1r{tag}{k} : Fin {DIM} → ℝ := fun j => ((w1z{tag}{k}).getD j 0 : ℝ)/{DEN}")
    A("")
    A(f"/-- net {tag} hidden weights ({H}×{DIM}), rows `k/{DEN}`. -/")
    A(f"noncomputable def W1{tag} : Fin {H} → Fin {DIM} → ℝ :=")
    A("  ![" + ", ".join(f"w1r{tag}{k}" for k in range(H)) + "]")
    A("")
    A(f"/-- net {tag} output weights ({K}×{H}), entries `k/{DEN}`. -/")
    A(f"noncomputable def W2{tag} : Fin {K} → Fin {H} → ℝ :=")
    A("  " + rmat(W2q, DEN))
    A("")
    A(f"/-- The trained {tag} MLP: dense → ReLU → dense, full 784-dim input. -/")
    A(f"noncomputable def mlp{tag} : EuclideanSpace ℝ (Fin {DIM}) → EuclideanSpace ℝ (Fin {K}) :=")
    A(f"  denseE W2{tag} ∘ reluE ∘ denseE W1{tag}")
    A("")
    A(f"-- row-extraction lemmas: pure `rfl` (no vecCons-at-numeral simp indexing)")
    for k in range(H):
        A(f"theorem W1{tag}r{k} : W1{tag} {k} = w1r{tag}{k} := rfl")
    A("")
    # layer-1 Gram: kernel dot facts (upper triangle) + all-pairs entry lemmas
    A(f"-- layer-1 Gram entries: one kernel `dotZ` evaluation per pair (`decide +kernel`)")
    for a in range(H):
        for b in range(a, H):
            A(f"theorem gz{tag}_{a}_{b} : dotZ w1z{tag}{a} w1z{tag}{b} = {int(G1[a][b])} := by decide +kernel")
    A("")
    A(f"/-- `G1{tag} = W1{tag}·W1{tag}ᵀ` ({H}×{H}, denominators {DEN}² = {den2}). -/")
    A(f"noncomputable def G1{tag} : Fin {H} → Fin {H} → ℝ :=")
    A("  " + rmat(G1, den2))
    A("")
    for a in range(H):
        for b in range(H):
            lo, hi = min(a, b), max(a, b)
            fact = (f"gz{tag}_{lo}_{hi}" if a <= b
                    else f"((dotZ_comm w1z{tag}{a} w1z{tag}{b}).trans gz{tag}_{lo}_{hi})")
            rows = f"W1{tag}r{a}" if a == b else f"W1{tag}r{a}, W1{tag}r{b}"
            A(f"theorem g{tag}_{a}_{b} : G1{tag} {a} {b} = ∑ j, W1{tag} {a} j * W1{tag} {b} j := by")
            A(f"  rw [show G1{tag} {a} {b} = (({int(G1[a][b])} : ℝ)/{den2}) from rfl, {rows}]")
            A(f"  simp only [w1r{tag}{a}, w1r{tag}{b}]")
            A(f"  rw [sum_getD_div w1z{tag}{a}_len w1z{tag}{b}_len {fact} {DEN} {DEN}]")
            A(f"  norm_num")
            A("")
    inner = ";\n      ".join(
        "(fin_cases b <;> [" + "; ".join(f"exact g{tag}_{a}_{b}" for b in range(H)) + "])"
        for a in range(H))
    A(f"theorem G1{tag}_eq : ∀ a b, G1{tag} a b = ∑ j, W1{tag} a j * W1{tag} b j := by")
    A("  intro a b")
    A("  fin_cases a <;>")
    A(f"    [ {inner} ]")
    A("")
    A(f"/-- `H1{tag} = G1{tag}²` (denominators {DEN}⁴ = {den4}). -/")
    A(f"noncomputable def H1{tag} : Fin {H} → Fin {H} → ℝ :=")
    A("  " + rmat(H1, den4))
    A("")
    A("set_option maxHeartbeats 12800000 in")
    A(f"theorem H1{tag}_eq : ∀ a b, H1{tag} a b = ∑ c, G1{tag} c a * G1{tag} c b := by")
    A("  intro a b")
    A("  fin_cases a <;> fin_cases b <;>")
    A(f"    · simp [H1{tag}, G1{tag}, Fin.sum_univ_succ]")
    A("      norm_num")
    A("")
    A("set_option maxHeartbeats 12800000 in")
    A(f"/-- Schatten-8 bound for {tag} layer 1: B₁ = {B1}. -/")
    A(f"theorem W1{tag}_lip : LipschitzL2 {frac(B1)} (denseE W1{tag}) := by")
    A(f"  refine denseE_lipschitzL2_gram2 W1{tag} G1{tag} H1{tag} (by norm_num) G1{tag}_eq H1{tag}_eq ?_")
    A(f"  simp [H1{tag}, Fin.sum_univ_succ]")
    A("  norm_num")
    A("")
    # layer 2: small (10×16 over 16) — the pooled file's recipe verbatim
    A(f"noncomputable def G2{tag} : Fin {K} → Fin {K} → ℝ :=")
    A("  " + rmat(G2, den2))
    A("")
    A(f"noncomputable def H2{tag} : Fin {K} → Fin {K} → ℝ :=")
    A("  " + rmat(H2, den4))
    A("")
    A("set_option maxHeartbeats 12800000 in")
    A(f"theorem G2{tag}_eq : ∀ a b, G2{tag} a b = ∑ j, W2{tag} a j * W2{tag} b j := by")
    A("  intro a b")
    A("  fin_cases a <;> fin_cases b <;>")
    A(f"    · simp [G2{tag}, W2{tag}, Fin.sum_univ_succ]")
    A("      norm_num")
    A("")
    A("set_option maxHeartbeats 12800000 in")
    A(f"theorem H2{tag}_eq : ∀ a b, H2{tag} a b = ∑ c, G2{tag} c a * G2{tag} c b := by")
    A("  intro a b")
    A("  fin_cases a <;> fin_cases b <;>")
    A(f"    · simp [H2{tag}, G2{tag}, Fin.sum_univ_succ]")
    A("      norm_num")
    A("")
    A("set_option maxHeartbeats 12800000 in")
    A(f"theorem W2{tag}_lip : LipschitzL2 {frac(B2)} (denseE W2{tag}) := by")
    A(f"  refine denseE_lipschitzL2_gram2 W2{tag} G2{tag} H2{tag} (by norm_num) G2{tag}_eq H2{tag}_eq ?_")
    A(f"  simp [H2{tag}, Fin.sum_univ_succ]")
    A("  norm_num")
    A("")
    A(f"/-- {tag} Schatten-8 product: L = {float(L):.3f}. -/")
    A(f"theorem mlp{tag}_lip : LipschitzL2 {frac(L)} mlp{tag} := by")
    A(f"  have h := W2{tag}_lip.comp (reluE_lipschitzL2.comp W1{tag}_lip (by norm_num)) (by norm_num)")
    A(f"  have e : {frac(B2)} * (1 * {frac(B1)}) = {frac(L)} := by norm_num")
    A("  rw [e] at h; exact h")
    A("")

def emit_image(A, i, tags_for_i):
    y = int(yte[i])
    A(f"/-- MNIST test image #{i} (digit {y}), exact pixels k/{PIX}. -/")
    A(f"def imgz{i} : List ℤ := {zlist(Xte_raw[i])}")
    A("")
    A(f"theorem imgz{i}_len : (imgz{i}).length = {DIM} := by decide +kernel")
    A("")
    A(f"noncomputable def imgv{i} : Fin {DIM} → ℝ := fun j => ((imgz{i}).getD j 0 : ℝ)/{PIX}")
    A("")
    A(f"noncomputable def imgF{i} : EuclideanSpace ℝ (Fin {DIM}) := WithLp.toLp 2 imgv{i}")
    A("")
    A(f"theorem imgF{i}_apply : ∀ j, imgF{i} j = ((imgz{i}).getD j 0 : ℝ)/{PIX} := fun _ => rfl")
    A("")
    for tag in tags_for_i:
        m, pre = info[tag]["mp"][i]
        A(f"-- net {tag} on image {i}: hidden pre-activations by kernel dotZ, then the margin")
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
        A(f"theorem margin{tag}{i} : ∀ j : Fin {K}, j ≠ {y} →")
        A(f"    {frac(m)} ≤ mlp{tag} imgF{i} {y} - mlp{tag} imgF{i} j := by")
        A(f"  have hout : ∀ jj : Fin {K}, mlp{tag} imgF{i} jj =")
        A(f"      ∑ k : Fin {H}, W2{tag} jj k * max (hpre{tag}{i} k) 0 := by")
        A("    intro jj")
        A(f"    show denseE W2{tag} (reluE (denseE W1{tag} imgF{i})) jj = _")
        A("    rw [denseE_apply]")
        A("    refine Finset.sum_congr rfl fun k _ => ?_")
        A(f"    rw [reluE_apply, hpre{tag}{i}_eval k]")
        A("  intro j hj")
        A("  fin_cases j <;>")
        A("    first")
        A("    | exact absurd rfl hj")
        A(f"    | · rw [hout, hout]")
        A(f"        simp [W2{tag}, hpre{tag}{i}, Fin.sum_univ_succ, max_def]")
        A("        norm_num")
        A("")

# ═══ file 1: nets ═══
L1 = []
A = L1.append
A("import LeanMlir.Proofs.LipschitzCertScorecard")
A("import LeanMlir.Proofs.Foundation.ListDot")
A("")
A("/-! # Full-input scorecard, part 1/4: the two 784→16→10 nets")
A("")
A(f"Weights of the spectrally-capped (`SF`: σ ≤ {CAP:g} projected SGD, {EP_CAP} epochs,")
A(f"q-acc {info['SF']['acc']:.3f}) and unconstrained (`TF`: {EP_UNC} epochs, q-acc {info['TF']['acc']:.3f})")
A(f"nets, /{DEN}-rationalized, at FULL 784-dim input — plus their in-kernel Schatten-8")
A(f"Lipschitz bounds (SF: L = {float(info['SF']['L']):.3f}, TF: L = {float(info['TF']['L']):.3f}).")
A("Layer-1 Gram entries are 784-term dots: one kernel `dotZ` evaluation each")
A("(`decide +kernel`, GMP; see `ListDot.lean`). Generated by")
A("`scripts/lipschitz_cert_scorecard_full.py`; weights are DATA. -/")
A("")
A(HEADER_OPTS)
for tag in ("SF", "TF"):
    W1q, W2q = nets[tag]
    emit_net(A, tag, W1q, W2q, info[tag]["B1"], info[tag]["B2"], info[tag]["L"])
A("end LipschitzCertDemo")
A("end Proofs")
(OUTDIR / "LipschitzCertScorecardFullNets.lean").write_text("\n".join(L1) + "\n")

# ═══ files 2+3: images (split halves) ═══
need = {}
for i in range(N_IMG):
    tags = [t for t in ("SF", "TF") if i in info[t]["cert10"]]
    if tags:
        need[i] = tags

for part, rng_ in (("A", range(0, 50)), ("B", range(50, 100))):
    L2 = []
    A = L2.append
    A("import LeanMlir.Proofs.LipschitzCertScorecardFullNets")
    A("")
    A(f"/-! # Full-input scorecard, part {'2' if part == 'A' else '3'}/4: images {rng_.start}–{rng_.stop - 1}")
    A("")
    A("Exact k/255 pixel data, per-net hidden pre-activations (one kernel `dotZ`")
    A("per row) and exact rational margin lemmas, for every image certified at")
    A("ε = 1/10 by either net. Generated by `scripts/lipschitz_cert_scorecard_full.py`. -/")
    A("")
    A(HEADER_OPTS)
    for i in rng_:
        if i in need:
            emit_image(A, i, need[i])
    A("end LipschitzCertDemo")
    A("end Proofs")
    (OUTDIR / f"LipschitzCertScorecardFullImgs{part}.lean").write_text("\n".join(L2) + "\n")

# ═══ file 4: certificates + aggregate ═══
L4 = []
A = L4.append
A("import LeanMlir.Proofs.LipschitzCertScorecardFullImgsA")
A("import LeanMlir.Proofs.LipschitzCertScorecardFullImgsB")
A("")
A("/-! # Full-input certified-accuracy scorecard (4/4): the certificates")
A("")
A(f"The pooled 49-dim scorecard, lifted to FULL 784-dim input (the 2026-07 audit's")
A(f"gap #3): first {N_IMG} MNIST test images, per-image `∀ δ, ‖δ‖ < ε → argmax fixed`")
A(f"theorems at pixel-L2 ε = 1/10 and 3/10, two 784→16→10 nets:")
A("")
A(f"* **spectrally capped** (σ ≤ {CAP:g} projected SGD, q-acc {info['SF']['acc']:.3f}, Schatten-8")
A(f"  L = {float(info['SF']['L']):.2f}): **{len(info['SF']['cert10'])}/{N_IMG} certified at ε = 0.1** (L2-PGD leaves {info['SF']['pgd10']}/{N_IMG}")
A(f"  robust — the certificate is within {info['SF']['pgd10'] - len(info['SF']['cert10'])} image(s) of the attack bound), and")
A(f"  **{len(info['SF']['cert30'])}/{N_IMG} at ε = 0.3** (PGD: {info['SF']['pgd30']}/{N_IMG});")
A(f"* **unconstrained** (q-acc {info['TF']['acc']:.3f}, L = {float(info['TF']['L']):.2f}): {len(info['TF']['cert10'])}/{N_IMG} at ε = 0.1")
A(f"  (PGD: {info['TF']['pgd10']}/{N_IMG}), collapsing to **{len(info['TF']['cert30'])}/{N_IMG} at ε = 0.3** (PGD: {info['TF']['pgd30']}/{N_IMG}) —")
A("  at the larger radius the σ-projection is what keeps the certificate alive.")
A("")
A("ε here is FULL-pixel-space L2 (pixels in [0,1]), not the pooled-feature L2 of")
A("`LipschitzCertScorecard.lean` — a strictly stronger, directly comparable-to-")
A("the-literature perturbation model. Width 16 (vs the canonical 512) is the")
A("honest cost: 512-wide Gram certificates are ~10⁵× this kernel work.")
A("Aggregates are lower bounds only — an upper-bound L cannot prove an image")
A("UNcertifiable. Generated by `scripts/lipschitz_cert_scorecard_full.py`. -/")
A("")
A(HEADER_OPTS)
names = {}
for tag in ("SF", "TF"):
    for epsname, eps, certs in (("10", EPS10, info[tag]["cert10"]),
                                ("30", EPS30, info[tag]["cert30"])):
        for i in certs:
            y = int(yte[i])
            m = info[tag]["mp"][i][0]
            A(f"/-- Test #{i} (digit {y}), net {tag}: certified at ε = {eps} — margin {float(m):.3f} ≥ √2·L·ε. -/")
            A(f"theorem cert{tag}{epsname}_{i} (δ : EuclideanSpace ℝ (Fin {DIM})) (hδ : ‖δ‖ < {frac(eps)}) :")
            A(f"    ∀ j, j ≠ {y} → mlp{tag} (imgF{i} + δ) j < mlp{tag} (imgF{i} + δ) {y} :=")
            A(f"  certified_at_eps mlp{tag}_lip (by norm_num) margin{tag}{i} (by norm_num)")
            A("    (by norm_num) δ hδ")
            A("")
        lname = ("cappedFull" if tag == "SF" else "unconFull") + "Certs" + epsname
        names[(tag, epsname)] = lname
        A(f"/-- Certificate witnesses (subset index, image, class), net {tag}, ε = {eps}. -/")
        A(f"noncomputable def {lname} : List (ℕ × EuclideanSpace ℝ (Fin {DIM}) × Fin {K}) :=")
        A("  [" + ",\n   ".join(f"({i}, imgF{i}, {int(yte[i])})" for i in certs) + "]")
        A("")
        A(f"theorem {lname}_certified :")
        A(f"    ∀ p ∈ {lname}, CertifiedAt mlp{tag} {frac(eps)} p.2.1 p.2.2 :=")
        A("  List.forall_iff_forall_mem.mp")
        A("    ⟨" + ", ".join(f"cert{tag}{epsname}_{i}" for i in certs) + "⟩")
        A("")
A("/-- **The full-input scorecard, as a theorem** — counts tied to the per-image")
A("    `CertifiedAt` proofs, both nets, both radii. Lower bounds only. -/")
A("theorem scorecardFull :")
A(f"    ({names[('SF','10')]}.length = {len(info['SF']['cert10'])} ∧")
A(f"      ∀ p ∈ {names[('SF','10')]}, CertifiedAt mlpSF {frac(EPS10)} p.2.1 p.2.2) ∧")
A(f"    ({names[('SF','30')]}.length = {len(info['SF']['cert30'])} ∧")
A(f"      ∀ p ∈ {names[('SF','30')]}, CertifiedAt mlpSF {frac(EPS30)} p.2.1 p.2.2) ∧")
A(f"    ({names[('TF','10')]}.length = {len(info['TF']['cert10'])} ∧")
A(f"      ∀ p ∈ {names[('TF','10')]}, CertifiedAt mlpTF {frac(EPS10)} p.2.1 p.2.2) ∧")
A(f"    ({names[('TF','30')]}.length = {len(info['TF']['cert30'])} ∧")
A(f"      ∀ p ∈ {names[('TF','30')]}, CertifiedAt mlpTF {frac(EPS30)} p.2.1 p.2.2) :=")
A(f"  ⟨⟨rfl, {names[('SF','10')]}_certified⟩, ⟨rfl, {names[('SF','30')]}_certified⟩,")
A(f"   ⟨rfl, {names[('TF','10')]}_certified⟩, ⟨rfl, {names[('TF','30')]}_certified⟩⟩")
A("")
A("end LipschitzCertDemo")
A("end Proofs")
(OUTDIR / "LipschitzCertScorecardFull.lean").write_text("\n".join(L4) + "\n")

tot = sum(len(open(OUTDIR / f"LipschitzCertScorecardFull{s}.lean").readlines())
          for s in ("Nets", "ImgsA", "ImgsB", ""))
print(f"wrote 4 files, {tot} lines total; images with theorems: {len(need)}")
