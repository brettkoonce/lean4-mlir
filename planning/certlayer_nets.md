# certlayer_nets.md — whole nets as `CertLayer` compositions

Started 2026-09-23, after `proof_cleanup.md` §1(t). That thread cut lines and compile time; this one
is about COMPLEXITY: the same composition argument is restated once per net, at that net's shapes.
Nothing below has been built or measured yet — every "replaces" is a prediction to test.

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

### 4.2 If 4.1 holds: R34 and MNv2 at binder resolutions

* Restate `resnet34ForwardB_full` / `mobilenetv2ForwardB_full` over a resolution binder (as R50's
  `q`), with the 224-px net as the instance. ⚠ This touches renders' faithfulness statements and
  every `…_full` consumer — survey consumers first (`grep -rn`), as §1(t) did.
* Then repeat 4.1 for each. Expected to retire `r34Pre*`, `mnv2PreB*`, their `_apply` lemmas and
  the chain equations.

### 4.3 Positivity in one place

Pick MNv4's convention everywhere: `0 < ε` carried by the weight records, so `R34PosB` / `R50PosB`
/ `MNV2PosB` and every block `…Pos` disappear from statements. Mechanical, but a statement change
across the tier. Independent of 4.1; could go first as a warm-up, but it collides with 4.2's
statement changes — do them in one tier regeneration.

### 4.4 Later, only if 4.1–4.2 land

* **Sync / DP ties as one theorem over a CertLayer chain.** The five `SyncB` + four
  `SyncStepTieB` files look like one proof shape over "a chain of per-example layers". Unverified —
  read two side by side before planning.
* **Suffix tiers (`B`, `G`, `GB`, `B0`, `full`).** Audit which are still cited from the book,
  `formalization.yaml`, the comparator tier or the blueprint; retire tiers cited only by their
  successors. Unverified guess that some exist.

## 5. Where things are

| what | where |
|---|---|
| CertLayer + combinators | `LeanMlir/Proofs/Foundation/CertifiedChain.lean` |
| the recorded timeout | `LeanMlir/Proofs/Nets/MobileNet/MobileNetV4FullB.lean` header, `MobileNetV4FullBVJP.lean` header |
| R50 apex (prototype target) | `LeanMlir/Proofs/Nets/ResNet/ResNet50FullBVJP.lean` |
| R50 block layers | `LeanMlir/Proofs/Nets/ResNet/ResNet50BackB0.lean` |
| the template the prototype replaces | `proof_cleanup.md` §1(o), §1(t) |

## 6. Log

(empty — record each measurement here with its commit)
