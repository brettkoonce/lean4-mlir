# Closing the last two trusted links: Tier 2–3 float numerics + syntactic (lexer) faithfulness

Planning doc for the two remaining items in the **TRUSTED column** that sit between
"proven SHlo graph over ℝ" and "the float bytes `iree-compile` actually consumes":

- **Part A (`den → Float32`, Tier 2–3).** The §1a ties certify every emitted train-step
  node's `den` (over `ℝ`) for CIFAR + the five Imagenette nets. The `ℝ → Float32` rounding
  bridge that makes that meaningful for the *deployed* float net is closed only for **Tier 1
  (MNIST linear/MLP/CNN)**. Push it up.
- **Part B (syntactic faithfulness).** `den (emit g) = fderiv` is the *semantic* half of R4.
  The *syntactic* half — that the emitted `.mlir` **text** is a faithful, recoverable encoding
  of `g` — has its structural core proven (`parse (toToks (skel a)) = some (skel a)`,
  `StableHLOParse.lean:239`) but the `String ↔ Tok` lexical step is still audited, not proven.

These are deliberately paired: they are the two trusted edges of the *same* boundary. Closing
A says "the float numbers are within budget of the proven `ℝ` graph"; closing B says "the text
IREE reads *is* that proven graph." Neither needs new architecture.

Reference style/state: `planning/floatbridge_certificate_gaps.md` (the TRUSTED/MEASURED/PROVEN
ledger), `planning/floatbridge_descent_pass.md`, `planning/floatbridge_descent_cnn.md`.

---

## Part A — `den → Float32` for Tier 2–3

### A0. Where we actually are (read before planning)

The per-op forward bridge is **architecture-complete** and the composition machinery exists:

- `FloatClose A B f fF L` (`FloatComposeBridge.lean:29`) — magnitude bound (`A → B`) **+** a
  forward error modulus (`|vt − va| ≤ e ⇒ |fF vt − f va| ≤ L e`). Composes via
  `FloatClose.comp` (`:36`), threading `A→B→C`, moduli `Lg ∘ Lf`.
- `FloatBridges f` (`:479`) — the existential closure (`∀ A, ∃ B L fF, FloatClose …`), chained
  by `FloatBridges.comp` (`:484`); this is the one-line whole-net assembly backbone.
- Per-op instances exist for **every** Tier 2–3 op: conv (`floatClose_flatConv`), ReLU
  (`floatClose_relu`), BN forward (`floatClose_bn`, keystone `bnIstd_close`/`rsqrt_lipschitz`
  in `BnFloatBridge.lean`), depthwise (`DepthwiseFloatBridge`), Swish/sigmoid/SE
  (`EnetFloatBridge`), LayerNorm/GELU (`ViTFloatBridge`), attention
  (`ViTAttentionFloatBridge`), and the **fully-assembled ViT block** `floatBridges_vitBlock`
  (`ViTBlockFloatBridge.lean`). Eval-mode BN is a bare affine (`floatClose_bnEval`,
  `BnEvalFloatBridge.lean`). Subnormals are handled (`FloatSubnormalBridge.lean`).

So three things — and **only** these three — are still open. The first two are tractable; the
third is the honest stop.

### A1. Whole-net **forward** float-close capstones — PARTLY DONE; pattern validated

**Corrected baseline (2026-06-25 survey + prototype).** Whole-net *forward* capstones already
exist for the shallow nets — `FloatModel.cifar_float_close` (4-conv CIFAR, monolithic budget
style), plus `cnn_float_close` / `mlp_float_close` / `linear_float_close` (MNIST). The earlier
"no per-net capstone exists" was a **grep miss**: they are named `<net>_float_close` (a
`FloatModel.` method), not `*forward*`/`*net*`. Two distinct assembly styles are available:

- **Monolithic budget** (`cifar_float_close`): one `|netF − net| ≤ <net>FwdBudget` theorem with
  explicit per-layer `set Aᵢ/Eᵢ` threading. Needs a hand-written float-forward `F` **and** a
  budget term per net. ~130+ lines. Tight, but heavy to author.
- **Existential `FloatBridges.comp`** (the backbone, designed for one-line assembly): no `F`
  def, no budget term — `FloatBridges` packages `B`/`L`/`fF` and `.comp` threads magnitudes.

**Prototype landed this session — `LeanMlir/Proofs/Cifar8FloatBridge.lean`:**
`Proofs.cifar8_floatBridges : FloatBridges cifarCnn8Forward` (the deeper 8-conv no-BN CIFAR),
assembled as a 25-op `FloatBridges.comp` chain over `floatBridges_{flatConv,relu,maxPool,dense}`,
dimension positivity discharged by `positivity`. **3-axiom-clean.** This **validates the
`FloatBridges.comp` path end-to-end** — prefer it over the monolith for the rest.

**BN keystone + `cifarBn` — DONE (2026-06-26).** The "do-it-once" BN infra landed in
`LeanMlir/Proofs/BnPerChannelFloatBridge.lean` (3-axiom-clean), three rungs:
- `floatBridges_bn` — flat/global BN in `FloatBridges` form: packages `floatClose_bn`, discharging
  the two generic operating-point facts (`D = 2A` via `bn_centered_le`/`bnMean_abs_le`;
  `S = 1/√ε` via `bnIstd_abs_le`/`bnVar ≥ 0`). The only remaining inputs are the *supplied*
  float-stat accuracy moduli `emean`/`eistd` (rsqrt/exp have no IEEE spec — modelled, not derived,
  like `fexp`/`fsig`). **This directly discharges EfficientNet's deferred `hbnE/hbnD/hbnP`** (those
  are flat/global `bnForward`, not per-channel).
- `floatBridges_bnPerChannelFlat` — block-diagonal per-channel lift via `FloatClose.perRowIdx`
  (key: `bnPerChannelFlat = perRowIdxFlat oc m (fun c => bnForward m ε (γ c) (β c))`
  *definitionally* — both are `Mat.flatten ∘ per-row-bnForward ∘ Mat.unflatten`); uniform budget
  from uniform `G`/`Bbnd` bounds.
- `floatBridges_bnPerChannelTensor3` — conjugates to the network's Tensor3 activation layout, the
  `reassocFwd`/`reassocBack` permutations being `gather E` / `gather E.symm` for the re-association
  `Equiv` (round-trips already proven), `floatBridges_gather` + two `FloatBridges.comp`. The BN op
  the CIFAR-BN and ResNet-34 forwards *actually contain*.

The CIFAR-BN capstone is `Proofs.cifarBn_floatBridges` (`CifarBnFloatBridge.lean`, 3-axiom-clean):
the four per-channel BNs are taken as `FloatBridges (bnPerChannelTensor3 …)` hypotheses (each
discharged by `floatBridges_bnPerChannelTensor3`), exactly as `floatBridges_mbconvBody` takes its
three — so the capstone is the pure `.comp` assembly. **A1 CIFAR family now complete** (no-BN
4-conv `cifar_float_close` + 8-conv `cifar8_floatBridges` + BN `cifarBn_floatBridges`). All wired
into `lakefile.lean` roots + `tests/AuditAxioms.lean`.

**Remaining A1 gap — ✅ CLOSED (deep-net forward capstones all exist).** Survey 2026-06-27: the
distinct-per-block fold was already done for every deep net. Whole-net *forward* `FloatBridges`
capstones exist and are audited: `r34Forward_floatBridges` (`Resnet34WholeFloatBridge.lean`),
`mnv2Forward_floatBridges` (`MobileNetV2WholeFloatBridge.lean`), `efficientnetForwardB_floatBridges`
(`EfficientNetWholeFloatBridge.lean`), `convnextForward…floatBridges` (`ConvNeXtWholeFloatBridge.lean`),
`vit_floatBridges_concrete` (`PatchEmbedFloatBridge.lean`). Each is the `.comp` chain over distinct
per-block instances + stem/patchEmbed + classifier/finalLN, with the per-block / BN / LN bridges
**supplied** as `FloatBridges` hypotheses (the `cifar8_floatBridges` discipline — each separately
dischargeable). So A1 is complete for all nets; nothing deep-net forward remains.

Pair each finished capstone with a measured-vs-proven row (`scripts/cifar_bn_margin_probe.py`
style) so the worst-case `L` sits next to the silicon number.

**Effort: CIFAR family DONE (no-BN 4/8-conv + BN); medium–large remaining for the deep nets
(volume + the per-block fold).** **Risk: low** — no new mathematics, the hard per-op moduli are
done; the per-channel-BN lift is built (`BnPerChannelFloatBridge.lean`); the rest is assembly.

### A2. BN backward + CIFAR float-gradient closeness — sub-steps 1–2 ✅ DONE; only the descent probe remains

**Corrected status (2026-06-27).** Sub-steps 1 and 2 below were absorbed into the A3 backward
sweep and are **done, 3-axiom-clean**:
1. **BN backward bridge — DONE.** `bnGradInput_close` (keystone) + `floatClose_bnBack`/`floatBridges_bnBack`
   (`BnBackFloatBridge.lean` + `BnBackComposeBridge.lean`). The `rsqrt`-in-the-backward terms are
   handled exactly as planned (forward `bnIstd_close` operating point + the stays-normal invariant).
2. **CIFAR float-gradient closeness — DONE.** `cifar8_grad_floatBridges` (no-BN 8-conv) and
   `cifarBn_grad_floatBridges` (BN) in `CnnBackFloatBridge.lean` — the whole-net input-gradient VJP
   float-bridges (the float backward ≈ the certified gradient). The honest stop short of descent.

**So the ONLY open A2 item is the descent probe (sub-step 3).** This is the genuinely-uncertain
research question, and the reason it's still worth doing: **CIFAR is the last shallow-enough net where
descent might still be provable.** Today `*_float_sgd_descends` exists ONLY for MNIST (linear / MLP /
CNN — `SgdDescentLinear`/`SgdDescentMlp`/`SgdDescentCnn`). For the deep Imagenette nets descent is
off-limits BY DESIGN (compounding operator norms ⇒ vanishing admissible `lr`). CIFAR sits in between.

**✅ PROBE DONE (2026-06-27) — the first non-MNIST provable descent. `SgdDescentCifar.lean`, 3-axiom-clean.**
The key discovery: **CIFAR-8's tail is byte-for-byte `cnn_conv2_sgd_descends`'s architecture** — last conv
`W₈` (c4→c4) → relu → maxpool → three denses → CE is exactly conv→relu→maxpool→3×dense. So descent at the
LAST conv layer reaches CIFAR-8 *for free*, with the SAME non-vacuous `lr` as MNIST. Two theorems:
- `cifarCnn8Forward_factor` (`rfl`): the committed net factors `head ∘ (relu ∘ flatConv W₈) ∘ prefix7`
  (`Function.comp` is definitionally associative).
- `cifar8_lastConv_sgd_descends`: one SGD step on `W₈` (earlier 7 layers frozen — their output on `image`
  is the feature map `x₁`) decreases the CIFAR-8 cross-entropy by `≥ lr·‖∇‖²/2`. Proved by reducing the
  loss-as-a-function-of-`W₈` to the `cnn_conv2` program at `x₁` (via the factor lemma +
  `flatConv = flatten∘conv2d∘unflatten`) and applying `cnn_conv2_sgd_descends`.

**The honest stop (logged, NOT forced):** descent through the DEPTH of all 8 conv layers stays open by
design. `cnn_conv2_sgd_descends`'s admissible-`lr` condition `hsmall` is a PRODUCT of per-layer
operator-norm factors; each extra conv layer multiplies another `(spatial · weight-bound)` factor in, so
the admissible `lr` shrinks geometrically with depth ⇒ vacuous in any realistic regime — the SAME
compounding mechanism that puts deep-net descent off-limits. CIFAR-BN (BN-in-the-backward Lipschitz) was
not attempted: last-conv descent already answers "does descent reach CIFAR" (yes), and the full-depth
stop is independent of BN. So **last-conv descent is the honest reach of provable descent for CIFAR.**

**Result: medium Lean (reused the MNIST-CNN scaffold via an instantiation, not a reproof); the
"compounding kills full-depth descent" outcome is the documented stop, exactly as the deep nets.**

### A3. Deep-net float **gradient** closeness (forward → + backward) — TRACTABLE, HIGH VALUE

**Gap.** The §1a ties (`r34_net_tied_certified`, `vit_net_tied_certified`, …) prove the emitted
backward denotes the certified gradient **over `ℝ`**. There is **no float story on the deep-net
backward at all** — so for the deployed Imagenette nets, "the float gradient ≈ the real
gradient" is currently unproven (only the forward bridges).

**BN-backward keystone — DONE (2026-06-26), `LeanMlir/Proofs/BnBackFloatBridge.lean`.** The
"other side" mirror started where the forward did: with the BatchNorm backward, the one
genuinely-new backward op every deep net's gradient folds (the §1a ties denote
`bn_grad_{gamma,beta,input}` over `ℝ`; these bridge them to float). All 3-axiom-clean:
- **Parameter grads** (the easy reductions): `bnBetaGrad_close` (`Σ dy`, pure `sum_close`) and
  `bnGammaGrad_close` (`Σ dy·x̂`, `sum_close` + `mul_close` at the supplied float `x̂`).
- **Input gradient** (the new op): `bnGradInput_close` — the three-term
  `dx = (1/n)·s·(n·dx̂ − Σdx̂ − x̂·Σ(x̂·dx̂))` (`dx̂ = γ·dy`, `s = istd`) bridged by threading
  `mul_close`/`sum_close`/`M.err` through the assembly. Reusable helpers `reduction_close`
  (float `Σ` of close terms) + `sub_close'`/`sub_mag` (rounded subtraction). Float `istd`/`x̂`
  supplied close (`es`/`exh` — discharged by the forward `bnIstd_close` + centered closeness at
  instantiation, exactly the forward's supplied-stats discipline).

**Plan (remaining).** This is the float analogue of the §1a ties, stopping at *closeness* (not
descent). The per-block backward graphs are already proven faithful over `ℝ`
(`*BackGraph_faithful`/`*BackBatchedGraph_faithful`, `vitNetBackGraph_faithful`,
`convBnBackGraph_faithful`, `mhsaBackGraph_faithful`, …). Mirror the forward `FloatBridges`
assembly on the backward:
1. A `FloatClose`-style certificate for each **backward** block op — **DONE** for BN-grad
   (`bnGradInput_close`), the **linear input-VJP** (`floatBridges_linBack`: `dx = Wᵀ·dy` is a
   bias-free dense over the transpose, so it *reuses `floatBridges_dense`*), and the **ReLU-back**
   (`floatBridges_reluMaskBack`: the `selectPos` mask, exact in float, modulus `id`, the backward
   peer of `floatClose_relu`). *Remaining: conv input-VJP (`convBackDenote`, a reversed-kernel
   conv — its own bridge, not a `flatConv` reuse), depthwise/SE/attention grads, and the loss-head
   cotangent (`softmax_ce_cot_close`, exists per-entry; lift to a `FloatBridges` map).*
2. Compose them into a per-net `<net>_grad_floatBridges` — **DONE for all 5 deep nets + the cifar
   pair** (2026-06-27 survey + the EfficientNet capstone): `r34_grad_floatBridges`
   (`Resnet34WholeBackFloatBridge.lean`), `mnv2_grad_floatBridges` (`MobileNetV2BackFloatBridge.lean`),
   `convnext_grad_floatBridges` (`ConvNeXtBackFloatBridge.lean`), `vit_grad_floatBridges(_concrete)`
   (`MhsaBackFloatBridge`/`PatchEmbedBackFloatBridge.lean`), **`efficientnet_grad_floatBridges`
   (`EfficientNetWholeBackFloatBridge.lean`, the last one — added 2026-06-27)**, plus
   `cifar8/cifarBn_grad_floatBridges` (`CnnBackFloatBridge.lean`) and `mlpInputGrad_floatBridges`
   (`LinBackFloatBridge.lean`). Each is the `.comp` fold over concrete endpoints + supplied
   per-block / BN / activation backwards (the `mnv2_grad_floatBridges` discipline; each dischargeable).
   **So the whole-net `FloatBridges` matrix is COMPLETE — all 5 nets × {forward, backward}.** What
   stays open is genuinely-distinct: the §B *certified-VJP* whole-net fold (gated by toy-only certified
   terms), and deep-net descent (open by design).

**→ The remaining deep-net backward assembly has its own pick-up plan:
`planning/a3_backward_deepnet_assembly.md`** (per-op backward bridges still needed, per-net
assembly table, suggested order `cifar8 → cifarBn → r34 → …`, and the gotchas carried forward).

This makes the §1a den-ties **mean something at float** for the deep nets, which is the single
biggest credibility jump for the Imagenette tier.

**Effort: large Lean** (the deep-net backward is the most surface), but **fully tractable** —
no new math, it's the forward bridge mirrored. **Risk: medium**, mostly volume.

### A-stop. What NOT to target — deep-net **descent** (open BY DESIGN)

Do **not** attempt `resnet34_float_sgd_descends` (or vit/cnx/enet/mnv2). Descent needs a
loss-gradient Lipschitz constant that compounds operator norms with depth ⇒ the admissible `lr`
vanishes; no `*_sgd_descends` exists for any deep net and that is a *considered* stop, not an
oversight (`floatbridge_certificate_gaps.md` §3, honesty-pass flag F1). A3 (gradient *closeness*)
is the correct ceiling for the deep nets. Keep "loss-descent **step**" (the certified update, all
nets) distinct from "the loss provably **decreases**" (linear end-to-end; MLP/CNN per-layer; deep
nets: never).

---

## Part B — Syntactic faithfulness: the verified lexer

### B0. Where we are

`StableHLOParse.lean` already lands the **structural core** of `parse (pretty a) = a`:

- `Raw` — the renderable skeleton of an `SHlo` (opcodes + shapes + leaf SSA names; the `ℝ`
  operand values and shape index **correctly erased** — the text never carries runtime values,
  they arrive as IREE function arguments).
- `skel : SHlo n → Raw`, `toToks : Raw → List Tok` (postorder, the order `pretty` emits),
  `parse : List Tok → Option Raw` (stack reconstructor).
- `roundtrip : parse (toToks (skel a)) = some (skel a)` (`:239`), by structural induction.

So the **op-graph structure is proven recoverable from its token serialization.** Because the
runtime values live in IREE arguments, not the text, skeleton-level recovery is the *right*
target for the text — there is no missing `den`-of-operands obligation here.

### B1. The missing link — `lex : String → List Tok` with `lex (pretty g) = toToks (skel g)`

**Corrected, much-narrower gap (`StableHLO.lean:3994-4009`).** `pretty` is **not** independent
of `toToks` — it already factors through it:
```
pretty B g = serializeToks B (toToks (skel g))            -- serializeToks = foldl emitTok
serializeToks B = List.foldl (emitTok B)                  -- one line of text per token
```
So the text is a **transparent, ordered, per-token rendering** of `toToks (skel g)`: walk the
token list, `emitTok` prints one SSA assignment per token and pushes/pops the result-name stack
**exactly as `parseStack` pushes/pops `Raw`s**. The whole-graph string parser the original plan
imagined is unnecessary. The *only* residue is inverting **one token's worth** of text:
the per-op `Tok ↔ line` map (e.g. that `emitTok`'s `"stablehlo.dot_general … = [1] x [0]"` line
is what a `dotIn` token prints) — today audited by inspection, validated only indirectly by
`iree-compile` accepting the bytes.

**Plan.**
1. **Write `lexTok : String → Option (Tok)` (per emitted line/op), not a whole-text parser.**
   `serializeToks` is `foldl emitTok`; the lexer is its per-token inverse over the finite op
   vocabulary. Operand references resolve **by stack position** (same discipline `emitTok` and
   `parseStack` already use) — no symbol table.
2. **Prove `emitTok_lexTok : lexTok (emitTok B t st).1 = some t`** per op (a finite case-split),
   then lift by `foldl` induction to `lex (pretty g) = toToks (skel g)`. Compose with the
   existing `roundtrip` (`StableHLOParse.lean:239`) for the **end-to-end syntactic theorem**:
   ```
   parse (lex (pretty g)) = some (skel g)
   ```
   "tokenize-then-parse the emitted text recovers exactly the proven op-graph skeleton" — moving
   the per-op lexical map **out of the trusted surface** into the kernel.
3. **Pin the byte tie.** The CI drift guard already byte-checks every committed
   `verified_mlir/<net>_train_step.mlir` against its renderer (`proofs.yml`); `pretty` *is* what
   produces those bytes, so `emitTok_lexTok` is anchored to the literal deployed text by
   construction — just add an assertion that the rendered `pretty` equals the committed file.

### B2. SSA names are already nameless — a non-issue, not a subtlety

This is the answer to "should I think in SSA terms / something more ordered": **yes, and the
repo already does.** `pretty`/`emitTok` allocate `%0, %1, …` from a `StateM Nat` counter in
**emission (postorder) order**, and `parseStack` reconstructs by **stack position**. So SSA
names carry no information beyond the order already in `toToks` — they are De-Bruijn-style
*indices*, not a free-form symbol table. The standard verified-compiler move (CompCert's
numbered temporaries; named-vs-nameless λ-terms) is to cut at exactly this nameless layer, and
the repo is already on that side. Concretely: `lexTok` never parses a name into a symbol table —
it pops operands off the reconstruction stack, mirroring `emitTok`'s push. No α-renaming, no
freshness bookkeeping. The earlier "name resolution is the hard part" worry is dissolved by the
`pretty = serializeToks ∘ toToks ∘ skel` factoring.

### B-stop. Out of scope (needs a verified compiler)

`lex_pretty` closes the **lexer**. It does **not** close: per-op StableHLO *spec* conformance
(that `stablehlo.dot_general` with these attrs means matmul to MLIR), IREE lowering, or
`float32 ≈ ℝ`. Those require a verified MLIR/IREE and stay validated by `iree-compile` + the GPU
runs. State this residue explicitly wherever `lex_pretty` is cited — same discipline as the
honesty pass.

**Effort: medium Lean** — revised *down* from the docstring's "large separate build". Because
`pretty = serializeToks ∘ toToks ∘ skel` already holds, the obligation is a per-token
`emitTok`/`lexTok` inverse (a finite case-split) lifted by `foldl` induction — **not** a
free-form whole-text parser with a symbol table. Self-contained (string/list induction, no
Mathlib analysis). **Risk: low** — B2 (names) is dissolved by the factoring; the residual work
is the finite per-op lexer and its inverse lemma.

---

## Effort / risk / value summary

| Item | What it delivers | Effort | Risk | Value |
|---|---|---|---|---|
| **A1** per-net forward capstones | "deployed float forward ≈ certified `ℝ` forward", named per net | ✅ DONE (all 5 deep nets + cifar family) | low | medium |
| **A2** BN-backward + CIFAR grad-close (+opt descent) | float gradient closeness one rung above MNIST; maybe Tier-2 descent | M | med | medium |
| **A3** deep-net grad-close (backward float) | makes the §1a ties mean something at float for Imagenette | ✅ DONE (all 5 `<net>_grad_floatBridges` + cifar/mlp; the 5×2 whole-net matrix is complete) | med | **high** |
| **B1/B2** per-op `lexTok` inverse → `parse (lex (pretty g)) = skel g` | text→graph faithfulness; retires the per-op lexical audit | M | low | **high** |

*A1 CIFAR family DONE (2026-06-26): `cifar8_floatBridges` (`Cifar8FloatBridge.lean`) + the BN
keystone `floatBridges_bn`/`floatBridges_bnPerChannelFlat`/`floatBridges_bnPerChannelTensor3`
(`BnPerChannelFloatBridge.lean`) + the BN-net capstone `cifarBn_floatBridges`
(`CifarBnFloatBridge.lean`), all 3-axiom-clean, wired into lakefile roots + AuditAxioms. The BN
keystone also discharges EfficientNet's deferred MBConv `hbnE/hbnD/hbnP`. Only deep-net forward
capstones (the per-block fold) remain in A1.*

## Suggested order

1. **A1** first — cheap, it's assembly, and it turns "every op bridges" into a per-net headline
   that A3 then upgrades to the backward. Immediate credibility, unblocks the magnitude-domain
   plumbing A3 reuses.
2. **A3** — highest single jump for the Imagenette tier (the deep-net float gradient story).
   Largest, but pure mirror of the forward assembly.
3. **A2** — the CIFAR rung; do alongside/after A3 since it shares the BN-backward bridge with the
   deep nets. Take the optional CIFAR descent only if the admissible `lr` survives BN.
4. **B1/B2** — independent of A; can run in parallel (different files, no shared lemmas). Schedule
   when a string-induction build is wanted rather than analysis.

**One-line recommendation (updated 2026-06-27):** ~~A1 → A3~~ **both DONE** — the whole-net
`FloatBridges` matrix is complete for all 5 nets × {forward, backward} (the last entry,
`efficientnet_grad_floatBridges`, landed 2026-06-27). The remaining tractable items are **B1/B2 (the
lexer / syntactic faithfulness — the other provable axis, self-contained, the SSA-names worry already
dissolved)** and **A2 (the CIFAR grad-closeness rung + optional Tier-2 descent)**. Take B for the
"deployed artifact faithful both numerically AND syntactically" headline; A2 is a smaller numeric rung.
**Deep-net descent stays open by design — do not target it.** The remaining §B *certified-VJP*
whole-net folds stay gated by toy-only certified terms (honest stop).
