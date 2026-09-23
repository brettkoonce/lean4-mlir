# certlayer_nets.md — whole nets as `CertLayer` compositions

Started 2026-09-23, after `proof_cleanup.md` §1(t). That thread cut lines and compile time; this one
is about COMPLEXITY: the same composition argument is restated once per net, at that net's shapes.
⚠ **Re-scoped 2026-09-23** after the §4.1 prototype (§6): whole-net CertLayers are cheap, but the
prefix vocabulary they would retire is named by public statements, so the thread is cut down to the
no-statement-change part (§4.2). Everything else is parked.

## 0. Rules for this thread

* Everything in `proof_cleanup.md` §0 applies (measure before/after, strip-and-compile, generated
  files via their generators, peel with `rw` never `simp only` over `rfl` lemmas, root-file lemmas
  parked in leaves). Its landing recipe and gate list (▶ Start here) apply unchanged.
* ⛔ The known failure is recorded in `MobileNetV4FullB.lean`'s header: composing CertLayer GROUPS
  into one at LITERAL resolutions elaborates, but every later peel of `CertLayer.comp` to `.fwd`
  costs ~10 min and then `(kernel) deterministic timeout` (four spellings tried). Any prototype
  must measure this first, before building on it.
* Statement changes to anything in the comparator tier → regenerate with
  `scripts/gen_comparator_tier.py` and run `tests/comparator/run.sh` locally (~5 min).

## 1. The problem

Each ImageNet net carries roughly the same file stack (counts from `ls LeanMlir/Proofs/Nets/*/`):

| stem | nets | what it states |
|---|---|---|
| `FullB` | 4 | the batch-BN forward + its typed graph (T2) |
| `FullBVJP` | 4 | the whole-net `HasVJPAt` (T1) |
| `FullBSeal` | 4 | non-vacuity at a witness |
| `BackB0` / `BackChains` | 7 / 5 | per-block backward graphs |
| `WholeBackCertifiedTieB` | 5 | input-gradient graph = VJP |
| `StepTieB` / `SyncB` / `SyncStepTieB` | 4 / 5 / 4 | training-step ties, DP / sync-BN |

`Nets/` is 112 files / ~46k lines. Inside `FullBVJP` alone, R34, R50 and MNv2 each hand-write
(§1(t) left them tidy but still per-net):

* 17–18 prefix definitions (`r34Pre0…16`, `r50Pre0…16`, `mnv2PreB0…17`) and their `_apply` lemmas;
* a positivity bundle (`R34PosB`, `R50PosB`, `MNV2PosB`) and a smoothness bundle (`…SmoothAtB`);
* a `vjp_comp_diff_at` chain (`r34ChainB`, …) and a chain equation (`…_eq_chain`).

## 2. What already exists

`Foundation/CertifiedChain.lean` (209 lines): `CertLayer m n` bundles `fwd`, `ok` (the smoothness
predicate), `diff`, `vjp`, `graph` and `faithful` (graph denotes the VJP), with `comp`, `residual`,
`residualProj`, `reluOut`, `chain`, and `comp_fwd` / `comp_fwd_apply` (proved between variables).

It is half-adopted. BLOCKS are CertLayers already — `r34BasicBlockLayer` / `r34DownBlockLayer`
(`ResNet34BackB0.lean:197, 420`), the R50 bottlenecks (`ResNet50BackB0`), `mnv2BodyLayer`
(`MobileNetV2BackB0.lean:314`), MNv4's groups, ViT's blocks. Only the TOP LEVEL is hand-rolled.
MNv4 went furthest: seven groups, each one CertLayer, chained by hand at seven prefixes; its
positivity lives inside the weight records (`UibParams` carries `hq he hd hz`), so no `0 < ε`
hypothesis appears in its statements.

If a whole net were ONE CertLayer, the apex (`.vjp`), `…_differentiableAt` (`.diff`), the
smoothness hypothesis (`.ok`, conjoined by `comp` at the right activations) and backward-graph
faithfulness (`.faithful`) would all be projections — the prefix chains, both bundles, the chain
equation and most of `FullBVJP` would not be written at all.

## 3. The hypothesis: the timeout is about numerals, not `comp`

ResNet-50 is stated at a resolution BINDER `q` (`N q` variables; `q = 7` → 224 px, `q = 5` → 160 px)
and its prefix chain elaborates in ~3 s. MNv4 is at literal 224/112/56/28/14/7, and its composed
CertLayer timed out. `proof_cleanup.md`'s rule "collapse lemmas at VARIABLE shapes, never at the
net's numerals" is the same observation at lemma scale. So the likely enabler is: **state every net
at a binder resolution and instantiate numerals only at the edges** (the renders, the seals).

Untested. If R50 composed as one CertLayer also times out, the limit is in `CertLayer.comp`
itself (e.g. the `ok` conjunction or `graph` unfolding) and §4's plan changes.

## 4. Plan

### 4.1 Prototype on ResNet-50 (first, and decides the rest)

In a scratch module (not wired into any lib):

1. `r50NetLayer N q w hp : CertLayer _ _` := stem `.comp` the sixteen existing bottleneck layers
   `.comp` the head (the head is global — build a CertLayer with `ok := True`).
2. Prove `(r50NetLayer …).fwd = resnet50ForwardB_full N q w` — by `rw` with `comp_fwd` one level
   at a time, never a closing `rfl` through the concrete chain.
3. Derive the existing statements from it: `resnet50ForwardB_full_has_vjp_at_correct` and
   `resnet50ForwardB_full_differentiableAt`. The hypothesis becomes `(r50NetLayer …).ok x`; decide
   whether to keep `R50SmoothAtB` as the public statement (prove `R50SmoothAtB → .ok`) so the
   comparator tier does not move, or to switch the tier to `.ok`.
4. **Measure**: elaboration + kernel time of steps 1–3 under `-Dtrace.profiler=true`, and
   a downstream use (the seal) against today's ~3 s.

Go/no-go: all of 4.1 within a few seconds per declaration, no bumps. If it fails, record where
(which peel, which decl) in §6 and stop — do not try MNv4-style workarounds before understanding it.

### 4.2 Re-scoped: one CertLayer per net, prefixes kept (no statement changes)

Only if it comes out smaller, measured per net; the prototype says R50 alone is about line-neutral.

1. Move the generic layers to one shared file: the stem pool (new), GAP and dense (today's
   `mnv4GapLayer` / `mnv4DenseLayer`), and the ResNet stem and head built from them. MNv4 switches
   to the shared GAP / dense.
2. R50: `r50NetLayer` replaces `r50ChainB` and the six block delegation lemmas. The apex
   keeps its name and type, and `R50SmoothAtB → .ok` goes through `comp_ok_of'`, one `refine` per
   block. `R50PosB`, `R50SmoothAtB`, `r50PreK`, their `_apply` lemmas and `…_eq_chain` stay: T3,
   sync-T3, the whole-back tie and the seal name them.
3. R34 the same way (same stem and head). Survey its consumers first.
4. Do not change `ChallengeTier.lean`. Run `gen_comparator_tier.py --check` to confirm it still matches.

### 4.3 Parked

* **Retiring the prefixes:** restating T3 / sync-T3 / seal over layer activations. This is a
  comparator-tier change, and the only thing that would make §2's prediction come true.
* **Binder resolutions for R34 / MNv2** (the old §4.2), **positivity in the weight records** (the old
  §4.3), **sync / DP ties over a chain** and **the suffix-tier audit** (the old §4.4). None of them is
  needed for 4.2.

## 5. Where things are

| what | where |
|---|---|
| CertLayer + combinators | `LeanMlir/Proofs/Foundation/CertifiedChain.lean` |
| the recorded timeout | `LeanMlir/Proofs/Nets/MobileNet/MobileNetV4FullB.lean` header, `MobileNetV4FullBVJP.lean` header |
| R50 apex (prototype target) | `LeanMlir/Proofs/Nets/ResNet/ResNet50FullBVJP.lean` |
| R50 block layers | `LeanMlir/Proofs/Nets/ResNet/ResNet50BackB0.lean` |
| the template the prototype replaces | `proof_cleanup.md` §1(o), §1(t) |

## 6. Log

### 2026-09-23 — §4.1 R50 prototype: GO on cost, SMALLER payoff than §2 predicted (at 52dceb3e)

Scratch module (outside the repo), importing `ResNet50WholeBackCertifiedTieB`; `lake env lean
-Dtrace.profiler=true`, threshold 100 ms.

| what | result |
|---|---|
| `r50NetLayer N q hq0 w hp` — stem (`cbReluStridedLayer.comp` a new pool layer) `.comp` 16 blocks `.comp` head (GAP + dense layers) | elaborates, < 100 ms |
| pool layer's `faithful` — the batched `maxPool3s2BackB` | `rw [den_maxPool3s2BackB_eq_flatBackB]; rfl`, 2 lines (the stem is no longer outside the chain) |
| `(r50NetLayer …).fwd x = resnet50ForwardB_full N q w x` — `rw [comp_fwd_apply]` ×18 then the per-block `_fwd` (`rfl` at variable shapes) | < 100 ms |
| `_has_vjp_at_correct`, `_differentiableAt`, whole-net backward-graph faithfulness | one-line projections, < 100 ms each |
| `R50SmoothAtB → (r50NetLayer …).ok` (keeps the comparator tier fixed) | 0.16 s, see ⚠ below |
| the same peel at the LITERAL `q = 7`, and the theorems at `q = 7` / `q = 5` | < 100 ms; no timeout |
| whole file, including import load | 2.8 s wall, 2.8 GB — same as today's `ResNet50FullBVJP.lean` (3.2 s) |
| bumps | none |

⚠ **The `ok` bridge fails as one anonymous constructor** (`maximum recursion depth` at block 5):
matching `hx.s2b1 : … (r50Pre4 N q w x)` against the nested layer forwards is a defeq check whose
depth grows with the block index. It works as one `refine` per block through
`comp_ok_of' (h₁) (y := r50PreK N q w x) rfl ?_`, which names the intermediate activation so the goal
stays flat. Any long chain whose hypothesis is stated at named prefixes needs this.

**§3's hypothesis, refined.** R50 at `q = 7` never hits MNv4's failure, even at the literal: its
widths stay syntactic `2 * (2 * (2 * 7))` on both sides of every `comp`. MNv4's types agree only up
to numeral evaluation (`2 * 28` vs `56`), and its timeout was in T2 (forward-graph `den`), which a
CertLayer does not touch. So the risk is literal widths that must be unified by arithmetic, not
numerals themselves.

**⛔ The payoff §2 predicted is wrong for R50.** `r50Pre0…16` are not only `FullBVJP` scaffolding:
they name activations in public statements. `ResNet50StepTieB` (T3, in `ChallengeTier.lean`),
`ResNet50SyncStepTieB`, `ResNet50WholeBackCertifiedTieB` and the seal's `nnK` positivity lemmas all
use them, and the seal also uses the `_apply` lemmas and `resnet50ForwardB_full_eq_chain`. So
`R50SmoothAtB`, `R50PosB`, the prefixes and their `_apply` lemmas **stay**. What goes is `r50ChainB`
(~60 lines) and the six block delegation lemmas (~45 lines); what comes in is the generic pool / GAP
/ dense / stem / head layers, the three record-level block layers and the bridge (~90 lines).
That's roughly line-neutral for R50 alone. The generic layers are ResNet-34's too (same stem and
head), and GAP / dense duplicate MNv4's `mnv4GapLayer` / `mnv4DenseLayer`, which should move to a
shared file.

**What is new rather than smaller:** a whole-net ResNet-50 backward graph *including the stem pool*,
proven to denote the VJP. `ViTBackNet.lean` records the stem as blocked on exactly that pool proof.
⚠ It is a CertLayer graph (SSA names such as `%psW` / `%stemR`), not the render's emitted backward,
so it does not replace T6.

Open decision before §4.2: whether retiring the prefix vocabulary is worth restating T3 / sync-T3 /
seal statements over layer activations (a comparator-tier change), or whether §4.2 should be
re-scoped to "one CertLayer per net + shared head/stem layers" and keep the prefixes as public
names.
