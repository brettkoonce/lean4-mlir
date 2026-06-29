# Planning — Natural gradient: the missing rungs that finish the optimizer-as-metric ladder

**The thesis.** The Muon ladder (`MuonGeometry.lean`, [[muon-geometry-proof-ladder]]) proved
*"every optimizer is steepest descent under a norm"* for three points: SGD (Euclidean), sign/Adam-ish
(`‖·‖∞`), Muon (operator norm). The table in `math_threads.md` Thread 1 lists more rungs — and the two
that complete it both come from a *metric* `M`: **Adam = steepest descent under a diagonal metric**,
**natural gradient = steepest descent under the Fisher metric `F = JᵀJ`**. The jewel: Adam, Shampoo,
and Muon are all **approximations to natural gradient with different metric structure** (diagonal /
Kronecker / spectral). And `F = JᵀJ` is built from the **VJP machinery** (`J` = the cotangent
Jacobian), so this rung *welds the optimizer thread to the autodiff thread* — the cotangent layer
constructs the metric the optimizer descends under.

## 0. The one idea — steepest descent under a metric `M`

Generalize `steepest_l2_*` (which is `M = I`). For an SPD metric `M`, the steepest-ascent direction
over the `M`-unit-ball is
```
argmax_{ dᵀ M d ≤ 1 } ⟨g, d⟩  =  M⁻¹ g / ‖g‖_{M⁻¹},   optimal value (dual norm)  ‖g‖_{M⁻¹} = √(gᵀ M⁻¹ g).
```
Pure Cauchy–Schwarz in the `M`-inner-product `⟨a,b⟩_M = aᵀ M b` (`‖·‖_M`-Cauchy–Schwarz is the
existing L1 proof, conjugated by `M^{1/2}`). Every optimizer is this with a different `M`:

| optimizer | metric `M` | `d⋆ = M⁻¹g` | repo |
|---|---|---|---|
| SGD | `I` | `g` | `steepest_l2_*` (done) |
| Adam / RMSprop | `diag(v̂)` | `g / √v̂` | **NG2** |
| Shampoo | Kronecker `L⊗R` | `(GGᵀ)^{-1/4}G(GᵀG)^{-1/4}` | `shampoo_eq_muon` (done) |
| Muon | operator-norm geometry | polar `UVᵀ` | `muon_polar_steepest` (done) |
| **natural gradient** | **Fisher `F = JᵀJ`** | `F⁻¹ g` | **NG3** |

## 1. Layered plan

New file `LeanMlir/Proofs/NaturalGradient.lean`, `Proofs` lib root, audited, target 3-axiom clean.

- ⬜ **NG1 — steepest descent under a general SPD metric.** `steepest_metric_bound` /
  `steepest_metric_attained`: over `{d : dᵀ M d ≤ 1}`, `⟨g,d⟩ ≤ √(gᵀ M⁻¹ g)` with equality at
  `d = M⁻¹g / √(gᵀM⁻¹g)`. Proof = L1's Cauchy–Schwarz in the `M`-inner-product (substitute
  `u = M^{1/2}d`, `w = M^{-1/2}g`, reduce to `⟨w,u⟩ ≤ ‖w‖‖u‖`). The `bound`+`attained` shape of every
  rung. The whole ladder becomes a *corollary* of NG1 (SGD = `M=I`; the others = specific `M`).
- ⬜ **NG2 — Adam/RMSprop = the diagonal metric.** Instantiate NG1 with `M = diag(v̂)` (the running
  second-moment preconditioner) → `d⋆ = g / √v̂`, the RMSprop/Adam-without-momentum update. Fills the
  "diagonal" rung the table has been missing. Cheap (diagonal `M⁻¹`, `M^{1/2}` are pointwise).
- ⬜ **NG3 — natural gradient / Gauss–Newton = the Fisher metric.** `F = JᵀJ` where `J` is the model
  Jacobian (Gauss–Newton approx of the Fisher; exact Fisher for an exponential-family output head).
  Natural-gradient step `= F⁻¹ g` is NG1 with `M = F`. **The bridge to VJP:** `J` and `Jᵀ` are exactly
  JVP/VJP, so `F = JᵀJ = VJP ∘ JVP` — the Fisher is *assembled from the cotangent machinery*. State
  `naturalGrad_steepest_fisher` as the NG1 instance, and note `F` SPD ⟸ `J` full-column-rank (else
  PSD + a damping `F + εI`, the practical Tikhonov form).

## 2. Honest scoping (the through-line, not a defect)

- **NG4 — the approximation hierarchy (the jewel, stated honestly).** Prove the *exact* relationships
  you can: Adam = NG1 with **diagonal** `F`; Shampoo = NG1 with **Kronecker-factored** `F`
  (`shampoo_eq_muon` already gives the single-step form). Then be candid: **the practical optimizers
  exist *because* `F⁻¹` is intractable at scale** — full natural gradient is `O(n²)` storage / `O(n³)`
  inverse, impossible for real nets. So Adam (diagonal), Shampoo (Kronecker), Muon (spectral) are
  *cheap structured approximations* to the same `F⁻¹g`. **The gap between them is the content of the
  chapter, not something to hide** ([[validation-ladder-method]]: the thing that breaks at scale —
  here `F⁻¹` — is the signal; every named optimizer is a different bet on how to approximate it).
- Don't claim "we compute the true natural gradient." Claim: the *geometry* (steepest descent under
  `F`) is the unifying frame, and the zoo are its tractable shadows.

## 3. Mathlib support / files

- Reuse: the `steepest_l2_*` / Cauchy–Schwarz pattern (`MuonGeometry.lean`), `fInner`, SPD/PosDef
  matrix API (`Matrix.PosDef`, `IsHermitian`), `M^{1/2}` via the spectral `√` already used in
  `svd_of_isUnit`. NG1's `M`-inner-product Cauchy–Schwarz: `inner_mul_le_norm_mul_norm` conjugated, or
  `Finset.inner_mul_le_norm_mul_norm`.
- Files: `LeanMlir/Proofs/NaturalGradient.lean` (imports `MuonGeometry`), root in `lakefile.lean`,
  `#print axioms` in `tests/AuditAxioms.lean`. Companion to `planning/muon_geometry.md`,
  `planning/shampoo.md`, `math_threads.md` Thread 1.

## 4. Sequencing

This is the **optimizer-thread capstone**: it subsumes the existing SGD/sign/Muon/Shampoo rungs as
instances of NG1 and adds Adam (NG2) + natural gradient (NG3), then NG4 names the whole zoo as
Fisher-approximations. Do NG1+NG2 first (clean linear algebra, immediate "Adam = diagonal metric"
payoff); NG3+NG4 are the deeper capstone that fuses with the VJP machinery. Pairs naturally with
`planning/power_iteration_lipschitz.md` (both reuse the spectral `√`/eigen engine) — together they are
the two "muon ∩ {VJP, Lipschitz}" bridges.
