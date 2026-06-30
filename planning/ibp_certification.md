# Planning — Verified Interval Bound Propagation (IBP): a deterministic *local* robustness cert

**Where this sits.** The robustness thread (`planning/robustness_ladder.md`, [[robustness-ladder-thread]])
has two corners: the **Lipschitz-margin cert** (`LipschitzCert.lean` — deterministic but the *global*
`∏‖Wᵢ‖₂` is loose/vacuous deep) and **randomized smoothing** (Cohen — probabilistic, scalable, L2). IBP
is the missing **deterministic *local*** corner — the crudest member of the CROWN/LiRPA
bound-propagation family, but the one that is **sound by construction** (no relaxation gap to bound, so
the soundness proof is pure interval monotonicity + induction — far easier to formalize than CROWN). It
gives non-vacuous deterministic certs exactly where the global Lipschitz product is hopeless, and it
would be genuinely novel as a *machine-checked* bound-propagation cert applied to *codegen-faithful*
nets. **Threat model: L∞** (the box) — complements (does not replace) the L2 Lipschitz/Cohen certs.

## 0. What IBP certifies

Input box `x ∈ [x₀−ε, x₀+ε]` (L∞ ball). Propagate per-coordinate bounds `[l,u]` through every layer to
bounds `[l_out, u_out]` on the **logits** (certify pre-softmax — softmax is monotone, argmax-preserving).
The certificate is one line:
```
robust at (x₀, ε)  ⟸  l_out[y] > max_{j≠y} u_out[j]
```
the lower bound of the true logit beats the upper bound of every other ⟹ no input in the box flips the
argmax.

## 1. The core — per-op interval transformers + soundness (the only real proofs)

Predicate `Contains (l u x : Vec n) := ∀ i, l i ≤ x i ∧ x i ≤ u i`. For each *verified* forward op, an
interval transformer + a one-line containment lemma `Contains l u x → Contains l' u' (op x)`:

- **Affine (linear / conv / BN-eval all reduce to this — the key one).** Positive/negative weight split
  (≡ center/radius `c'=Wc+b, r'=|W|r`): `l' = W⁺·l + W⁻·u + b`, `u' = W⁺·u + W⁻·l + b`. Soundness =
  sign-split of the dot product `(Wx)_i = Σ W_ij x_j` (a `Finset.sum` / triangle-inequality argument).
  **The only nontrivial lemma, and it's small.** Tie it to the verified `Proofs.dense`.
- **ReLU (any monotone activation).** `[l,u] → [relu l, relu u]`; soundness is *just monotonicity*
  (`l ≤ x ≤ u ⟹ relu l ≤ relu x ≤ relu u`). Tight, trivial.
- **Conv** = affine over the receptive field (`|W|·r` summed over the window) — same lemma as affine,
  tie to the verified conv (`CifarFaithfulPoC.convW_den` / `CNN.lean`).
- **BN-eval** = per-channel affine — tie to `bnEvalAffine` (`BnEvalFloatBridge.lean`); same lemma.
- **Residual add** `l' = l_a + l_b`, `u' = u_a + u_b`. **Avg-pool** = affine; **max-pool** = monotone.

## 2. Composition + the cert

- ⬜ **`ibpNet`** — the transformers sequenced to mirror the *verified* forward (the same fold the
  forward-faithfulness work already uses).
- ⬜ **`ibpNet_sound`** — induction over layers: each transformer preserves `Contains`, so
  `∀ x, Contains (x₀−ε) (x₀+ε) x → Contains l_out u_out (netForward x)`. **Sound by construction** — no
  gap term (this is what makes IBP tractable vs CROWN).
- ⬜ **`ibp_margin_certified`** — `l_out[y] > max_{j≠y} u_out[j] → ∀ x ∈ box, argmax (netForward x) = y`.
  A finite-max argument; **reuse `FloatModel.argmax_preserved`** (the margin⟹same-argmax lemma already
  built for the fp8/E4M3 cert, `Binary32Instance.lean`).

## 3. Phase plan (the repo's "start at MLP" rhythm)

- ⬜ **I1 — MLP (hello-world).** `Interval`/`Contains` + `ibpAffine_sound` + `ibpRelu_sound` → compose →
  `mlp_ibp_certified`. Fully sound-by-construction; smallest end-to-end story.
- ⬜ **I2 — the margin cert.** `ibp_margin_certified` via `argmax_preserved` → an end-to-end L∞ cert on
  a real net.
- ⬜ **I3 — CNN.** conv-as-affine transformer + pool + BN-eval → `cnn_ibp_certified`. Conv is the only
  new transformer (same dot-product bound, summed over the window).
- ⬜ **I4 — ResNet.** add the residual-add transformer → the deeper conv nets.
- ⬜ **I5 — ViT (hard frontier; flag, don't promise).** Two snags: **GELU is non-monotone** (dips ~−0.17
  near `x≈−0.75`), so its transformer needs the interval min/max at the critical point, not just
  endpoints; and **attention** (a bilinear `QKᵀ` of two intervals, then softmax) is where IBP goes
  notoriously loose — exactly the work LiRPA/CROWN add. So IBP is natively a **CNN/MLP** cert;
  transformers are CROWN territory.

## 4. Honest scoping (tier discipline — mirror the NS quintic / smoothing tiers)

- **Sound-by-construction ⟹ tractable** — the whole proof is monotonicity + triangle inequality +
  induction; *no relaxation-gap bounding* (CROWN's hard part). Strictly easier to formalize than CROWN.
- **But loose** — interval bounds drop inter-coordinate correlations ⟹ blow up with depth ("wrapping
  effect"). Non-vacuous for *shallow* nets, *small* ε, or **IBP-trained** nets; loose vs CROWN on a
  normally-trained deep net. Honest claim: strictly better than the global `∏‖Wᵢ‖₂` (it's *local*), the
  *loose* end of the deterministic family. Don't overclaim tightness.
- **L∞, not L2** — complements the Cohen/Lipschitz L2 certs (different threat model), not a replacement.

## 5. The bonus — IBP *certified training* (drops into the trainer-ablation engine)

IBP's sweet spot is **certified training** (Gowal et al. 2018): make the IBP worst-case margin a
*training loss* ⟹ the net is verifiably IBP-robust *by construction*, lifting I3/I4 from "loose on a
normal net" to "non-vacuous on a net trained for it." That's an **IBP-loss trainer** — it slots straight
into the trainer-ablation engine (clean vs IBP-trained → *certified* accuracy as the ablation metric),
and it's the L∞ sibling of the "tight by construction" / Lipschitz-net theme ([[robustness-ladder-thread]],
the Muon/spectral orthogonality through-line in `math_threads.md`).

## 6. Files / wiring

`LeanMlir/Proofs/IBP.lean` (the `Interval`/`Contains` core + transformers + soundness + margin cert),
per-net `<Net>IbpCert.lean` (tie the transformers to the verified forward of each net), root in
`lakefile.lean`, `#print axioms` in `tests/AuditAxioms.lean`, target 3-axiom clean. Reuses:
`Proofs.dense` / `bnEvalAffine` / the conv semantics (tie targets), `FloatModel.argmax_preserved` (the
margin cert), the forward-fold composition pattern. Companion to `planning/robustness_ladder.md`,
`LeanMlir/Proofs/LipschitzCert.lean`, and `planning/power_iteration_lipschitz.md` (the other deterministic
spectral-norm route). Reference: Gowal et al. 2018 (IBP), Zhang et al. 2018 (CROWN), Xu et al. 2020
(auto_LiRPA), Mirman et al. 2018.

## 7. Session handoff — start at I1

1. `LeanMlir/Proofs/IBP.lean`: `Contains` predicate, `ibpAffine`/`ibpRelu` transformers + the two
   soundness lemmas (affine = sign-split dot product; ReLU = monotonicity).
2. Compose `ibpMlp` + `ibpMlp_sound` (induction over layers) + `ibp_margin_certified` via
   `argmax_preserved` → `mlp_ibp_certified`. Keep 3-axiom clean; add the audit line.
3. Then I3 (conv transformer, tie to the CNN forward) — the first *non-vacuous-where-Lipschitz-is-hopeless*
   demonstration. ViT (I5) is the honest open frontier; certified training (§5) is the trainer tie-in.
