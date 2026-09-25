# Proof-quality audit — `LeanMlir/Proofs/Nets/MobileNet/*.lean` (26 files, 14,125 lines)

Static reading only (no builds). None of the 26 files is in `gen.txt`. Toolchain v4.34.0; every
Mathlib name suggested below was grepped in `.lake/packages/mathlib/Mathlib`.

Signal scan for this directory: `set_option maxHeartbeats` 7 sites (1 of them FILE-WIDE),
`maxRecDepth` 6, `change` 0, `show` 27 (15 in `MobileNetV4FullBSeal`), `native_decide` 0.
Declarations over 50 lines: 18. Of those, only 3 are single-net necessities; the rest are the
copy-paste skeletons listed under **Recurring patterns**.

Two things the files already get right and that the fixes below reuse: `MobileNetV4FullBVJP`'s
`Mnv4SmoothAt` structure plus `CertLayer` groups (the V2 apex is the long-hand version of this),
and `ConvNeXtWholeBackCertifiedTieB.lean:398` `vjpCompDiffAt_fst_backward` (the rw-peel that
avoids a whole-chain closing `rfl`).

---

## MobileNetV2WholeBackCertifiedTieB.lean

### MobileNetV2WholeBackCertifiedTieB.lean:207 — `mnv2InputGradB_eq_mobilenetv2B_full_vjp`

**Smell:** heartbeats (1,000,000 = 5× default, plus `maxRecDepth 800000`)
**Current:**
```lean
set_option maxRecDepth 800000 in
set_option maxHeartbeats 1000000 in
theorem mnv2InputGradB_eq_mobilenetv2B_full_vjp (N : Nat) {nCls : Nat}
    (Ws : Kernel4 32 3 3 3) ... (Wh : Kernel4 1280 320 1 1) ...
    (b1 : Vec (N * (32 * 112 * 112)) → Vec (N * (16 * 112 * 112))) ...
  unfold mnv2InputGradB
  rw [mnv2StemBBack_eq_vjp_backward (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      cbrBBack_eq_vjp_backward (by decide) (by decide) Wh bh εh hεh γh βh _ h_head,
      dense_transpose_eq_vjp_backward Wfc bfc (fun _ => 0)]
  rfl
```
**Why it breaks:** the closing `rfl` has to unfold all twenty `vjpCompDiffAt` levels of
`mobilenetv2PaperPCHasVJPAt` (a `let`-chain of `PProd`s, lines 129–149) and match them against
`mnv2InputGradB`'s `∘`-chain. That is the kernel re-deriving the whole composition by unfolding —
the known 48 GB hazard in its `rfl` form — and it happens at literal widths (`32`, `112`, `1280`,
`7`), which is why `maxRecDepth` had to go to 800k. Any change to how `vjpCompDiffAt` is
spelled (or to `mnv2InputGradB`'s association) silently moves the cost.
**Suggested:** peel with a lemma proved *between variables*, then `rw` — the ConvNeXt method.
Add, next to `opaqueA*` in `Foundation/OpaquePrefix.lean` (4 direct importers — a leaf), or move
the apex there with it:
```lean
theorem mobilenetv2PaperPC_backward {s0 … s21 : Nat} (stem : Vec s0 → Vec s1) … (dns : Vec s20 → Vec s21)
    (x : Vec s0) (hstem …) (hb1 …) … (hdns …) :
    (mobilenetv2PaperPCHasVJPAt stem b1 … dns x hstem hb1 … hdns).backward
      = hstem.fst.backward ∘ hb1.fst.backward ∘ … ∘ hdns.fst.backward := rfl
```
(`rfl` is free here: every term is a variable.) The tie becomes
`unfold mnv2InputGradB; rw [mobilenetv2PaperPC_backward, <stem tie>, <head tie>, <dense tie>]`
with no closing `rfl`. Also generalise the tie's stem/head widths (`Kernel4 oc ic kH kW`, `h w`)
and take `hkH hkW` as hypotheses instead of `(by decide)`; instantiate at 32/112/1280/7 in a
one-line corollary. Then drop both `set_option`s. (Do NOT turn the `rw` into `simp only [...]`:
these peel lemmas are `rfl`-lemmas, so `simp` would record a `dsimp` and hand the kernel the same
unfolding.)

### MobileNetV2WholeBackCertifiedTieB.lean:321 — `mnv2InputGradB_correct`

**Smell:** heartbeats + repetition (119-line declaration for a 2-line proof)
**Current:**
```lean
set_option maxRecDepth 800000 in
set_option maxHeartbeats 1000000 in
theorem mnv2InputGradB_correct (N : Nat) {nCls : Nat} ... -- all 40 binders of the tie, retyped
  rw [congrFun (mnv2InputGradB_eq_mobilenetv2B_full_vjp N Ws bs εs hεs γs βs Wh bh εh hεh γh βh
    Wfc bfc b1 b2 … b17 x h_stem hb1 … hb17 h_head) dy]
  exact (mobilenetv2PaperPCHasVJPAt (mnv2StemB N 112 112 Ws bs εs γs βs) b1 … x
          ⟨…⟩ hb1 … ⟨…⟩ ⟨…⟩ ⟨…⟩).correct dy i
```
**Why it breaks:** the statement re-spells the 21-slot apex term from the tie (lines 425–436 are a
verbatim copy of 303–314) and the `rw` makes the elaborator re-check the tie's statement at
literal widths — the heartbeat bump is paying for elaborating the same term a second time. Any
edit to the tie's binder list must be replicated here by hand. The V4 file has the identical
pair (`MobileNetV4WholeBackCertifiedTieB.lean:393`, `2000000` heartbeats, 150 lines).
**Suggested:** one generic lemma, e.g. in `Foundation/OpaquePrefix.lean`:
```lean
theorem HasVJPAt.correct_of_eq {m n : Nat} {f : Vec m → Vec n} {x : Vec m}
    (hf : HasVJPAt f x) {B : Vec n → Vec m} (hB : B = hf.backward) (dy : Vec n) (i : Fin m) :
    B dy i = ∑ j, pdiv f x i j * dy j := hB ▸ hf.correct dy i
```
Then each `_correct` is `(apex …).correct_of_eq (tie …) dy i`, or better, is deleted and callers
use `.correct_of_eq` directly. (Not in `Foundation/Tensor.lean` even though `HasVJPAt` lives
there: 423 downstream modules.)

### MobileNetV2WholeBackCertifiedTieB.lean:65 — `mobilenetv2PaperPCHasVJPAt`

**Smell:** repetition (97 lines; third copy of one construction)
**Current:**
```lean
  let p1 := vjpCompDiffAt stem b1 x hstem hb1
  let p2 := vjpCompDiffAt (b1 ∘ stem) b2 x p1 hb2
  …
  let p20 := vjpCompDiffAt (gap ∘ head ∘ b17 ∘ … ∘ b1 ∘ stem) dns x p19 hdns
  p20.fst
```
**Why it breaks:** the same N-stage apex exists at 18 stages (`ResNet34BackCertifiedTieB.lean:145`
`r34BFullHasVJPAt`), 21 (here) and 26 (`MobileNetV4WholeBackCertifiedTieB.lean:131`
`mnv4BFullHasVJPAt`, whose own docstring says "MobileNetV4 needs its own because Conv-M's
ladder is longer, not because anything differs"). Every prefix is spelled out explicitly — the
`vjpCompDiffAt _ g x` trap is avoided, good — but the three copies must be kept in lockstep
with `OpaquePrefix`'s `opaqueA*` by hand.
**Suggested:** move all three apexes into `Foundation/OpaquePrefix.lean` beside the `opaqueA*` they
are stated over (that file's header already records collapsing four private copies of the
prefixes into one), each with its `_backward` rfl-peel lemma from the first finding. If a fourth
net appears, generate them from the same script as `opaqueA*`.

### MobileNetV2WholeBackCertifiedTieB.lean:460 — `mobilenetv2ForwardBFull_eq_slots`

**Smell:** fragile-simpa (dsimp-only `simp only` over 19 defs) + repetition
**Current:**
```lean
  rw [mobilenetv2ForwardBFull_eq_chain N w x]
  simp only [mnv2HeadB, mnv2PreB17, mnv2PreB16, …, mnv2PreB1, mnv2PreB0]
  rw [comp3_assoc]
```
and `private theorem comp3_assoc … : (f ∘ g ∘ h) ∘ k = f ∘ g ∘ h ∘ k := rfl` (line 446).
**Why it breaks:** all 19 names are definitions, so this `simp only` is a pure `dsimp` unfold; the
kernel re-derives it, and the step only closes because the unfolded forms happen to reassociate
into exactly the target (hence the trailing `comp3_assoc`). The V4 peer
(`MobileNetV4WholeBackCertifiedTieB.lean:574,625`) does the same job with a generic
`mnv4Chain_apply` (`rfl` between variables) and a plain `rw`, and records that it takes 2 s.
`comp3_assoc` is `Function.comp_assoc f (g ∘ h) k` (Mathlib `Logic/Function/Defs.lean:28`).
**Suggested:** mirror V4: add `chain21_apply` (the 21-stage composition applied equals the nested
application, `rfl` at variable stages) next to the apex, and prove the shape check by
`rw [chain21_apply, mobilenetv2ForwardBFull_eq_chain, mnv2PreB17_apply, …]` or, after the
`MobileNetV2FullBVJP` refactor below, by the group-level `rfl` V4 uses. Replace the private
`comp3_assoc` by `Function.comp_assoc` with its arguments given explicitly (same statement, same
kernel cost), or delete it if `chain21_apply` removes the need.

---

## MobileNetV4WholeBackCertifiedTieB.lean

### MobileNetV4WholeBackCertifiedTieB.lean:243 — `mnv4InputGradB_eq_mnv4B_full_vjp`

**Smell:** heartbeats (2,000,000 = 10× default, plus `maxRecDepth 800000`)
**Current:**
```lean
  unfold mnv4InputGradB
  rw [mnv4StemBBack_eq_vjp_backward (N := N) (h := 112) (w := 112)
        (by decide) (by decide) Ws bs εs hεs γs βs x h_stem,
      cbReluBBack_eq_vjp_backward … (opaqueA22 (mnv4StemB N 112 112 Ws bs εs γs βs) fused b1 … b21 x) h_h1,
      cbReluBBack_eq_vjp_backward … (opaqueA23 …) h_h2,
      r34HeadBBack_eq_vjp_backward … (opaqueA24 …)]
  rfl
```
**Why it breaks:** same cause as the V2 tie, at 25 levels instead of 20, so twice the budget. The
file header (lines 61–66) says "Everything generic in its widths, instantiated only in the
capstone" — true of the stage ties and the apex, but this tie itself binds `Kernel4 32 3 3 3`,
`Kernel4 960 256 1 1`, `Vec (N * (48 * 56 * 56))` … and closes with a whole-chain `rfl`.
**Suggested:** exactly the V2 fix: an `mnv4B_full_backward` rfl-peel lemma between variables,
`rw` it, drop the `rfl`; restate at variable widths and instantiate. `mnv4InputGradB_correct`
(line 393, 150 lines, same bumps) becomes `.correct_of_eq`.
Separately, `mobilenetv4ForwardBFull_eq_slots` (line 609) carries `maxRecDepth 800000` although
its docstring says it is seven rewrites of variable-proved lemmas taking two seconds — the bump is
likely vestigial; try deleting it.

---

## MobileNetV2FullBVJP.lean

### MobileNetV2FullBVJP.lean:59 — file-wide `set_option maxHeartbeats 1000000`

**Smell:** heartbeats (file-scoped, 5×)
**Current:**
```lean
namespace Proofs
open scoped BigOperators
set_option maxHeartbeats 1000000
```
**Why it breaks:** no `in`, so every one of the file's ~55 declarations — including one-line
delegations like `mnv2ExpOnlyBHasVJPAt` — runs at 5× budget, and a future regression anywhere
in the file is masked. Mathlib forbids file-level heartbeat options for this reason. The docstring
also claims "this tier carries no numerals", but `mnv2PreB0 … mnv2PreB17` and the apex are stated
at `112/56/28/14/7`.
**Suggested:** delete the line; if `mobilenetv2ForwardBFullHasVJPAt` (the only plausible
consumer) then fails, scope it `set_option maxHeartbeats … in` on that one declaration and state
the measured value.

### MobileNetV2FullBVJP.lean:277 — `mobilenetv2ForwardBFullHasVJPAt`

**Smell:** long-proof (127 lines) + repetition
**Current:**
```lean
    (qb1 : IVNoExpPos w.b1) (qb2 : IVPos w.b2) … (qb17 : IVPos w.b17)
    (x : …) (h_stem : …) (sb1 : IVNoExpSmoothAtB N 112 112 w.b1 (mnv2PreB0 N w x)) … (sb17 : …) (h_head : …) :
  have d1 := mnv2NoExpB_differentiableAt N 112 112 w.b1 qb1 _ sb1
  have e1 : HasVJPAt (mnv2PreB1 N w) x :=
    vjpCompAt _ _ x dS d1 vS (mnv2NoExpBHasVJPAt N 112 112 w.b1 qb1 _ sb1)
  have f1 : DifferentiableAt ℝ (mnv2PreB1 N w) x := d1.comp x dS
  … (×17)
```
**Why it breaks:** 38 hypothesis binders, 51 `have`s, and the same 38 binders re-typed in
`mobilenetv2ForwardBFullHasVJPAt_correct` (line 492). Every consumer must pass 38 arguments
(the seal needs 19 helper lemmas `sc_stem, sc1 … sc17, sc_head` just to produce them, and
`seal_differentiableAt` re-derives the 18 `DifferentiableAt` steps — see below). The V4 peer
(`MobileNetV4FullBVJP.lean:92,135`) does the same job with one `Mnv4SmoothAt` structure and seven
`CertLayer` group joins.
**Suggested:** port V4's architecture: an `MNV2SmoothAtB N w x : Prop` structure (stem, one field
per block, head) and `CertLayer` block layers (`MobileNetV2BackB0.lean` already has
`mnv2BodyLayer`/`mnv2DownBodyLayer`). Short of that, fold each `d/e/f` triple into one
`vjpCompDiffAt` (which already returns the `PProd` of both), halving the proof, and export
`mobilenetv2ForwardBFull_differentiableAt` under the same hypotheses so the seal can reuse it.

### MobileNetV2FullBVJP.lean:203 — `mnv2PreB0 … mnv2PreB17` and `mnv2PreB*_apply` (line 404)

**Smell:** repetition (36 declarations + a 19-step `rw` chain)
**Current:**
```lean
theorem mnv2PreB5_apply … : mnv2PreB5 N w x = mnv2ResidB N 28 28 w.b5 (mnv2PreB4 N w x) := by
  rw [mnv2PreB5, Function.comp_apply]
…
  rw [mobilenetv2ForwardBFull, Function.comp_apply, mnv2PreB17_apply, mnv2PreB16_apply, …, mnv2PreB0_apply]
```
**Why it breaks:** 18 prefix defs duplicate `Foundation/OpaquePrefix.lean`'s `opaqueA*` (which
were introduced precisely to stop per-net copies) and 18 `_apply` lemmas are each `rfl`. The
19-name `rw` in `mobilenetv2ForwardBFull_eq_chain` must be kept in reverse block order by hand.
**Suggested:** one generic `chain18_apply` (`rfl` at variable stages, like `mnv4Chain_apply`), and
`mobilenetv2ForwardBFull_eq_chain := by rw [mobilenetv2ForwardBFull, chain18_apply]`. If the
prefix names are kept for the hypothesis bundles, define them as `abbrev mnv2PreBk N w :=
opaqueAk (stem) (b1) … ` so the two vocabularies coincide.

### MobileNetV2FullBVJP.lean:142 — `mnv2ResidB_differentiableAt`

**Smell:** undocumented-defeq
**Current:**
```lean
  show DifferentiableAt ℝ (biPath _ (fun y => y)) v
  exact (mnv2ExpOnlyB_differentiableAt N h w p hq v hs).add differentiable_id.differentiableAt
```
**Why it breaks:** relies on `residual f` being defeq to `biPath f id` (two definitions in two
files, `Architectures/Residual.lean:46`, `Foundation/Tensor.lean:291`). Same `show` at
`MobileNetV2.lean:315`; the same unfolding is done by `show … + v k = v k` in
`MobileNetV2FullBSeal.lean:213` and `MobileNetV4FullBSeal.lean:252`, and by
`unfold residual biPath; rfl` in `MobileNetV2FullB.lean:274`. There is `residualHasVJPAt`
but no `residual_differentiableAt` and no `residual_apply`.
**Suggested:** add to `Architectures/Residual.lean` (beside `residualHasVJPAt`, line 174):
```lean
@[simp] theorem residual_apply {n} (f : Vec n → Vec n) (v : Vec n) (k : Fin n) :
    residual f v k = f v k + v k := rfl
theorem residual_differentiableAt {n} {f : Vec n → Vec n} {x : Vec n}
    (hf : DifferentiableAt ℝ f x) : DifferentiableAt ℝ (residual f) x :=
  hf.add differentiable_id.differentiableAt
```
and replace all five sites. (Check the transitive rebuild count of `Residual.lean` first; if it is
high, a new leaf beside `CertifiedChain.lean` works too.)

---

## MobileNetV2StepTieB.lean

### MobileNetV2StepTieB.lean:673 — `mnv2_net_tiedB`

**Smell:** heartbeats (1,600,000 = 8×)
**Current:**
```lean
set_option maxHeartbeats 1600000 in
theorem mnv2_net_tiedB (N : Nat) … (x : …) (t : Vec (N * (1 * nCls))) :
    let g : Vec (N * nCls) :=
      unrowB N nCls (den (smoothedLossCotGraph N nCls α B aStr negAK bStr logN ohN
        (rowB N nCls (mobilenetv2ForwardBFull N w x)) t))
    let dy17 := …
    …
  intro g dy17 … cotStem
  exact ⟨mnv2_stem_tiedB …, mnv2_noexp_tiedB …, … , mnv2_head_tiedB …⟩
```
**Why it breaks:** the proof is 19 closed instances of `∀ cot` lemmas; nothing in it needs work.
The budget goes into elaborating/defeq-checking the statement, whose `g` embeds the whole literal
forward `mobilenetv2ForwardBFull N w x`. The V4 peer `mnv4_net_tiedB`
(`MobileNetV4StepTieB.lean:852`) takes `g : Vec (N * nCls)` as a **binder** and puts the loss in a
separate `mnv4_lossCot_is_smoothedCE_grad` — and has no bump. (Hypothesis from the diff between
the two files; confirm by removing the bump after the change.)
**Suggested:** make `g` a binder as V4 does (the smoothed-CE reading is already its own theorem at
line 770). Same fix for `MobileNetV2SyncStepTieB.lean:925` `mnv2_net_syncTiedB` (1,600,000,
164 lines): V4's `mnv4NetSyncTiedB` def + free `G gs hgs` capstone + `…_smoothedCE` corollary
(`MobileNetV4SyncStepTieB.lean:1190,1309,1442`) carries no bump.

### MobileNetV2StepTieB.lean:156 (and 240, 263, 341) — `*CotIn_eq_vjp`

**Smell:** repetition + redundant hypothesis
**Current:**
```lean
theorem mnv2NoExpCotIn_eq_vjp … (hs : IVNoExpSmoothAtB N h w p xin) (cotN : String) :
    mnv2NoExpCotIn N h w p xin dyOut = (mnv2NoExpBHasVJPAt N h w p hq xin hs).backward dyOut := by
  have h := mnv2NoExpBackGraph_faithful p hq xin (.operand cotN dyOut) hs
  have hd : den (SHlo.operand cotN dyOut) = dyOut := rfl
  rw [hd] at h
  rw [← h]
  rfl
```
**Why it breaks:** `hd` re-proves `den_operand` (`Codegen/StableHLO.lean:2410`, `@[simp]`) locally
at all four sites. `cotN : String` does not occur in the statement: every caller must invent a
string for nothing.
**Suggested:** drop `cotN` (use `(.operand "" dyOut)` inside) and write
`have h := …faithful … (.operand "" dyOut) hs; rw [den_operand] at h; rw [← h]; rfl`. The residual
variant (line 263) already documents why it needs `Eq.trans`; keep that comment.

### MobileNetV2StepTieB.lean:83 — `relu6MaskB`

**Smell:** repetition (V2 vs V4 cotangent vocabularies)
**Current:**
```lean
noncomputable def relu6MaskB (n : Nat) (pre dy : Vec n) : Vec n :=
  fun i => if 0 < pre i ∧ pre i < 6 then dy i else 0
```
alongside `mnv2CotPc/Dn/Dc/En/Ec/InBody`, `mnv2SCot*`, `mnv2NoExpCot*`, and in V4
`mnv4CotPc/Dn/Dc/En/Ec/Qn/Qc`, `mnv4SCot*`, each with a `_smul`, a `Sync` twin and a `_shard`.
**Why it breaks:** MobileNetV2's inverted residual is the UIB body with no pre-DW slot and a 3×3
post-DW slot — exactly what `UibSpec` (`MobileNetV4BackB0.lean`) parametrises — and the two
chains differ only in the mask (`relu6MaskB` vs ResNet's `reluMaskB`). The result is ~60
declarations per net (cot defs × {plain, `_smul`, `Sync`, `Sync_shard`, `_scaled`}) that are
the same up to the mask.
**Suggested:** (longer-term, highest payoff in the directory) a mask-generic chain:
`maskB (p : ℝ → Prop) [DecidablePred p] (pre dy : Vec n)` with `maskB_smul`/`maskB_shard` once,
and the IB cotangent chain stated over `(mask, UibSpec-like slots)`; `relu6MaskB := maskB (· ∈ Ioo 0 6)`,
`reluMaskB := maskB (0 < ·)`. Put the mask kit in a new leaf beside `ResNet34SyncStepTieB`'s
`reluMaskB_smul` (it currently lives in a net file).

---

## MobileNetV2SyncStepTieB.lean / MobileNetV4SyncStepTieB.lean

### MobileNetV2SyncStepTieB.lean:76–219 (26 lemmas), MobileNetV4SyncStepTieB.lean:104–276 (27) — `*_smul`

**Smell:** brittle-chain + repetition
**Current:**
```lean
theorem mnv2CotDn_smul … :
    mnv2CotDn N h w p xin (fun i => s * dy i) = fun i => s * mnv2CotDn N h w p xin dy i := by
  unfold mnv2CotDn; rw [mnv2CotPc_smul, cInB_smul, relu6MaskB_smul]
theorem mnv2CotDc_smul … := by unfold mnv2CotDc; rw [mnv2CotDn_smul, bnInB_smul]
…
```
**Why it breaks:** each lemma is "a composite of homogeneous maps is homogeneous", re-proved by
hand for every link, with the `rw` order tied to the definition's nesting. Repo-wide this is 134
lemmas (R34 33, V4 27, V2 26, B0 25, R50 23). `fun i => s * dy i` is the non-mainstream spelling
of `s • dy` on `Fin n → ℝ` (`Pi.smul_apply`, `smul_eq_mul`).
**Suggested:** a composition-closed predicate, stated once in a leaf next to the primitive
`bnInB_smul`/`cInB_smul` (currently `ResNet34SyncStepTieB.lean:93,113`):
```lean
def Homog {m n : Nat} (f : Vec m → Vec n) : Prop := ∀ (s : ℝ) (dy : Vec m), f (s • dy) = s • f dy
theorem Homog.comp {f : Vec m → Vec n} {g : Vec n → Vec p} (hg : Homog g) (hf : Homog f) :
    Homog (g ∘ f) := fun s dy => by simp [hf s dy, hg s (f dy)]
```
then `mnv2CotDn_smul := (relu6MaskB_homog _).comp ((cInB_homog …).comp (bnInB_homog …))`.
The mainstream end state is to build the cotangent maps as `Vec m →ₗ[ℝ] Vec n` and get
`map_smul` for free; `Homog` is the cheap intermediate.

### MobileNetV2SyncStepTieB.lean:390 (and 23 more sites across both Sync files) — `*_shard` ending in bare `rfl`

**Smell:** undocumented-defeq
**Current:**
```lean
theorem mnv2SyncCotDn_shard (r : Fin R) : … := by
  unfold mnv2SyncCotDn
  rw [mnv2SyncCotPc_shard R hR N h w hN hh hw p XIN dys DY hdys, cInB_shard]
  rfl
```
**Why it breaks:** the final `rfl` is `relu6MaskB (shard PRE) (shard DY) = shard (relu6MaskB PRE DY)`
— which is exactly `relu6MaskB_shard` (line 234), declared `:= rfl` and never used (grep: its
only mention is the module docstring). If either mask or `batchShard` changes representation the
24 `rfl`s fail with no pointer to why. Also every shard/tie lemma re-derives
`(nhw_ne_zero hN hh hw) (nhw_ne_zero (Nat.mul_pos hR hN) hh hw)`.
**Suggested:** end with `exact relu6MaskB_shard _ _ r` (V4: `reluMaskB_shard`); add one section
`variable`-level lemma `nhw_pair hR hN hh hw : _ ≠ 0 ∧ _ ≠ 0` and destructure it.

---

## MobileNetV2FullBSeal.lean / MobileNetV4FullBSeal.lean (29 s CI for V2)

### MobileNetV2FullBSeal.lean:797–1060 (`ed0 … eDiff_dH`, 22 lemmas); MobileNetV4FullBSeal.lean:698–818 (15)

**Smell:** repetition
**Current:**
```lean
theorem ed2e (nCls : Nat) (t : ℝ) : EDiff (d2e nCls t) (A2e nCls t) := by
  refine eDiff_bn 96 (2 * 56) (2 * 56) 1 (kv 96 (1 / 64)) (kv 96 3) (fun _ => 1 * d1p nCls t 0)
    (d2e nCls t) (Z2e nCls t) ?_ ?_
  · exact eDiff_conv (h := (2 * 56)) (w := (2 * 56)) (0 : Fin 16) rfl (by norm_num) (by norm_num) 1
      (kv 96 0) (d1p nCls t) _ (mnv2PreB1 2 (sealW nCls) (sealX t)) (ed1 nCls t) (fun o => rfl)
  · intro ci
    simp only [d2e, rf, kv_apply]
    ring
```
**Why it breaks:** the same three steps at 46 sites repo-wide (V2 22, V4 15, R50 5, R34 4), each
at literal shapes with two `(by norm_num)` kernel-size discharges and a `simp only … ; ring` that
re-proves the same scalar identity `1/64 * δ * istd = δ * rf …`. This is also the bulk of the V2
seal's 29 s (not measured per-declaration — static reading).
**Suggested:** two kit lemmas in `Training/BatchSealKit.lean` (beside `eDiff_bn`, line 783), stated
at variable shapes with the carrier update baked in:
```lean
theorem EDiff_ctConvBn {ic oc h w : Nat} (γ0 β0 : ℝ) {δ : Fin ic → ℝ} {v : Vec (2 * (ic * h * w))}
    (hv : EDiff δ v) :
    EDiff (fun o => δ 0 * (γ0 * bnIstd (2 * (h * w))
              (bnRowLA 2 oc h w (batchMap 2 (flatConv (ctK oc ic 1 1 1) (kv oc 0)) v) o) 1))
      (bnBatchLA 2 oc h w 1 (kv oc γ0) (kv oc β0) (batchMap 2 (flatConv (ctK oc ic 1 1 1) (kv oc 0)) v))
theorem EDiff_ctDwBn …  -- same with `ctDW c 3 3 1`, carrier `fun c => δ c * …`
```
(plus strided/XLA variants). Each `ed*` becomes one term; the per-block `Z*`/`A*`/`d*` defs can
then be replaced by one per-block carrier lemma at variable `(h, w, ic, mid, oc)`.

### MobileNetV2FullBSeal.lean:1380–1418; MobileNetV4FullBSeal.lean:1221–1259 — seal tail

**Smell:** repetition (verbatim across 4 nets)
**Current:**
```lean
theorem sealX_nonconstant … := by
  intro heq
  have h1 := gd_ray nCls hn 1
  have h0 := gd_ray nCls hn 0
  rw [heq] at h1
  have hz : (1 : ℝ) * Rr nCls 1 = 0 * Rr nCls 0 := by rw [← h1, ← h0]
  rw [one_mul, zero_mul] at hz
  linarith [Rr_pos nCls 1]
theorem sealX_jacobian_nonzero … := by
  refine fderiv_ne_zero_of_ray sealV (seal_differentiableAt nCls 0) (fun y => y (…) - y (…)) (by fun_prop) … ?_
  have heq : (fun t => F (sealX 0 + t • sealV) (…) - F (sealX 0 + t • sealV) (…)) = fun t => t * Rr nCls t := …
```
identical in `ResNet34FullBSeal.lean:1017` and R50.
**Why it breaks:** four hand-maintained copies of one argument.
**Suggested:** one lemma in `Training/JacobianSeal.lean` (beside `fderiv_ne_zero_of_ray`, line 93),
taking `F`, the ray `x0 v`, the readout indices, `g : ℝ → ℝ` with
`hg : ∀ t, F (x0 + t • v) i₀ - F (x0 + t • v) i₁ = g t`, `HasDerivAt g c 0`, `c ≠ 0`, `g 1 ≠ g 0`,
and `DifferentiableAt ℝ F x0`, returning the conjunction (nonconstant ∧ `fderiv ≠ 0`); each seal
supplies `gd_ray` and its slope.

### MobileNetV2FullBSeal.lean:1333 — `seal_differentiableAt` (also MobileNetV4FullBSeal.lean:1201)

**Smell:** repetition
**Current:**
```lean
  have f0 : DifferentiableAt ℝ (mnv2PreB0 2 (sealW nCls)) (sealX t) := mnv2StemB_differentiableAt …
  have f1 : … := (mnv2NoExpB_differentiableAt 2 112 112 (sealW nCls).b1 (seal_noExp_pos 32 16) _ (sc1 nCls t)).comp (sealX t) f0
  … (×18)
```
**Why it breaks:** a second copy of the `f1 … f17` chain already inside
`mobilenetv2ForwardBFullHasVJPAt`; any change to block order must be made in both.
**Suggested:** export `mobilenetv2ForwardBFull_differentiableAt` from `MobileNetV2FullBVJP.lean`
(same hypotheses as the apex; or have the apex return the `PProd`) and make `seal_differentiableAt` a single
application. Same for V4 with `mobilenetv4ForwardBFull_differentiableAt … hx`.

### MobileNetV2FullBSeal.lean:220, 238, 258 — `sealExpB_eq` / `sealStridedB_eq` / `sealNoExpB_eq`

**Smell:** undocumented-defeq
**Current:**
```lean
  show projB N (h := h) (w := w) (ctK oc mid 1 1 1) (kv oc 0) 1 (kv oc (1 / 64)) (kv oc 0)
      (StableHLO.dwbrB N (h := h) (w := w) (ctDW mid 3 3 1) (kv mid 0) 1 (kv mid (1 / 64)) (kv mid 3)
        (StableHLO.cbrB N (h := h) (w := w) (ctK mid ic 1 1 1) (kv mid 0) 1 (kv mid (1 / 64)) (kv mid 3) v)) = _
  rw [cbrB_eq _ _ hm, dwbrB_eq _ _ hm]
  rfl
```
**Why it breaks:** a 5-line hand-unfolded restatement of `mnv2ExpOnlyB` (`MobileNetV2FullB.lean`)
through the `IVW` projections — any change to the block definition or the witness record breaks
the `show` with a defeq failure far from the cause. Same idiom at `MobileNetV4FullBSeal.lean:279`
and in 13 `show (mnv4Res*Layer 2 (sealW nCls)).fwd (mnv4Pre* …) = _` steps (lines 998–1066; those
at least carry a comment at 990).
**Suggested:** `mnv2ExpOnlyB_apply`/`mnv2StridedB_apply`/`mnv2NoExpB_apply` lemmas (each `rfl`,
at variable shapes) in `MobileNetV2FullB.lean`, then `rw [mnv2ExpOnlyB_apply, cbrB_eq …, dwbrB_eq …]`.
For V4, `mnv4Pre{k}_apply` lemmas (as V2 has `mnv2PreB*_apply`) replace the `show`s.

### MobileNetV4FullBSeal.lean:300–342 — `sealUib_ok` / `sealUibStrided_ok`; :950 `Rr_pos`

**Smell:** repetition + brittle-chain
**Current:**
```lean
  · show (mnv4PostDWSlot (h := s.h) (w := s.h) N s.postDWk _ _ _ _ _ _).ok _
    unfold mnv4PostDWSlot
    by_cases hk : s.postDWk = 0
    · simp only [hk, ↓reduceIte]; trivial
    · simp only [hk, ↓reduceIte]; exact bne N (s.ic * s.expand) s.h s.h hm _
```
(3 copies), and
```lean
theorem Rr_pos (t : ℝ) : 0 < Rr t := by
  unfold Rr
  exact mul_pos (rf_pos _ _) (mul_pos (rf_pos _ _) (… 15 deep …))
```
(V2 `Rr_pos` 22 deep; `Rr_continuous` the same shape with `.mul`).
**Why it breaks:** the slot discharge is written out per slot; the 15/22-deep `mul_pos` nest must
match `Rr`'s parenthesisation exactly (the comment says `repeat' apply mul_pos` fails because it
splits inside `rf`).
**Suggested:** `theorem preDWSlot_ok`/`postDWSlot_ok (N k …) (hm : Mg N h h) (x) : (slot).ok x`
proved once by the `by_cases`. For `Rr`, define it as a product
`∏ i : Fin 15, rf … (Zi t)` or `List.prod [...]` and use `Finset.prod_pos` / `List.prod_pos`
(both in Mathlib `Algebra/Order/BigOperators/GroupWithZero/`), or mark `rf` `@[irreducible]` so
`repeat' apply mul_pos` stops at it, then close with `rf_pos`. Either removes the hand-nested term.

---

## MobileNetV2.lean

### MobileNetV2.lean:60 — `relu6LinearPart_apply`; :100 `pdiv_relu6`; :115 `relu6HasVJPAt`

**Smell:** undocumented-defeq + brittle-chain
**Current:**
```lean
  show (ContinuousLinearMap.pi (fun k' =>
          if 0 < x k' ∧ x k' < 6 then ContinuousLinearMap.proj k' else (0 : Vec n →L[ℝ] ℝ))) y k = _
  rw [ContinuousLinearMap.pi_apply]
  by_cases hxk : 0 < x k ∧ x k < 6
  · rw [ite_eq_left hxk, ite_eq_left hxk]; rfl
  · rw [ite_eq_right hxk, ite_eq_right hxk]; rfl
…
  · subst hij; rw [ite_eq_left rfl, ite_eq_left rfl]
  · rw [ite_eq_right (fun h : j = i => hij h.symm), ite_eq_right hij]
    by_cases hxj : 0 < x j ∧ x j < 6
    · rw [ite_eq_left hxj]
    · rw [ite_eq_right hxj]
```
**Why it breaks:** 24 `ite_eq_left/ite_eq_right` rewrites in this directory, each depending on the
exact `if` nesting and on the decidability instance matching syntactically; the `show` re-spells
the definition. The relu peer (`Foundation/MLP.lean:160`) is already idiomatic:
`rw [reluLinearPart, ContinuousLinearMap.pi_apply]; split_ifs <;> rfl`.
**Suggested:** `relu6LinearPart_apply := by rw [relu6LinearPart, ContinuousLinearMap.pi_apply]; split_ifs <;> rfl`;
`pdiv_relu6`'s tail `by split_ifs <;> simp_all` (or `by_cases hij : i = j <;> simp [hij]`);
`relu6HasVJPAt.correct` via `Finset.sum_ite_eq'` then `split_ifs <;> ring`. Longer-term, relu
and relu6 are both "diagonal 0/1 mask at a smooth point": one `pdiv_of_hasFDerivAt_diagMask`
lemma (new leaf, not `Foundation/MLP.lean` if that is widely imported) would serve both.

---

## MobileNetV2BackB0.lean

### MobileNetV2BackB0.lean:94 — `bnRelu6StageHasVJPAt`; :188, :209, :235 `*BackBatchedGraph_faithful`

**Smell:** repetition
**Current:**
```lean
theorem cbrBackBatchedGraph_faithful … := by
  rw [cbrBackBatchedGraph, convBackBatched_faithful (v := x),
      bnBatchLABack_faithful (β := β) (hε := hε),
      selectMid_faithful _ _ h_smooth]
  simp only [cbrBHasVJPAt, bnRelu6StageHasVJPAt, stageHasVJPAt, vjpCompAt,
    HasVJP.toHasVJPAt, Function.comp_apply]
```
(×3 here with `depthwise`/`depthwiseStridedXla` swapped in; ×2 in `MobileNetV4BackB0.lean`, ×2 in
`ResNet34BackB0.lean`), and `bnRelu6StageHasVJPAt` is `ResNet34BackB0.lean:90`
`bnReluStageHasVJPAt` with `relu6` for `relu`.
**Why it breaks:** seven copies of one skeleton; the trailing `simp only` is a pure `dsimp` unfold
of VJP defs. At stage level and variable shapes that is cheap and acceptable (this is NOT the
whole-net hazard — keep it `simp only`, not `rw`, here), but it is repeated rather than proved once.
**Suggested:** `bnActStageHasVJPAt (act) (hact : HasVJPAt act (bn (batchMap op x)))
(hactd : DifferentiableAt …)` and one `bnActStageBackGraph_faithful` parametric in the op's
`*BackBatched` token and its faithfulness lemma, in a new leaf imported by the three BackB0 files.

---

## MobileNetV2FullPaperEval.lean

### MobileNetV2FullPaperEval.lean:301 — `mobilenetv2FwdGraphPaperEval_faithful`

**Smell:** heartbeats (`maxRecDepth 20000`) + fragile-simpa
**Current:**
```lean
set_option maxRecDepth 20000 in
theorem mobilenetv2FwdGraphPaperEval_faithful … (x : Vec (3 * 224 * 224)) :
    den (mobilenetv2FwdGraphPaperEval epsStr ε w x) = mobilenetv2ForwardPaperEval ε w x := by
  simp only [mobilenetv2FwdGraphPaperEval, denseF_faithful, gapF_faithful, relu6F_faithful,
             bnPerChannelEvalF_faithful, flatConvF_faithful, flatConvStridedXlaF_faithful,
             ivExpOnlyGraphEvalW_faithful, ivResidGraphEvalW_faithful,
             ivStridedGraphEvalW_faithful, ivNoExpGraphEvalW_faithful, den_operand]
  rfl
```
**Why it breaks:** after the rewrites, the closing `rfl` compares two whole-net terms at literal
224/112/… shapes — the numeral-shape `rfl` the project lessons warn about, hence the recursion
bump. The batched twin (`MobileNetV2FullB.lean:353`) closes with rewrites only.
**Suggested:** add `stemGraphEvalW`/`headGraphEvalW` with `_faithful` lemmas at variable shapes (as
the block kinds already have), `unfold` both sides, and close with the rewrite chain alone.

---

## MobileNetV2FullB.lean

### MobileNetV2FullB.lean:353 — `mobilenetv2FwdGraphBFull_faithful`

**Smell:** brittle-chain
**Current:**
```lean
  unfold mobilenetv2FwdGraphBFull mobilenetv2ForwardBFull
  rw [mnv2HeadGraphB_faithful, mnv2ExpOnlyGraphB_faithful, mnv2ResidGraphB_faithful,
      mnv2ResidGraphB_faithful, mnv2StridedGraphB_faithful, … (19 entries, 6 distinct names) …,
      mnv2StemGraphB_faithful]
```
**Why it breaks:** the list must mirror the 19-deep nesting outside-in; inserting or reordering a
block breaks it with a motive error. These six are propositional lemmas proved by tactics (not
`rfl`-lemmas), and this is a forward faithfulness, not a VJP tie — so the dsimp hazard does not
apply.
**Suggested:** `simp only [mnv2HeadGraphB_faithful, mnv2ExpOnlyGraphB_faithful,
mnv2ResidGraphB_faithful, mnv2StridedGraphB_faithful, mnv2NoExpGraphB_faithful,
mnv2StemGraphB_faithful]` (6 names, order-free). Same for `mnv4FwdGraphBFull_faithful`
(`MobileNetV4FullB.lean:848`). Measure once; if the kernel cost rises, keep `rw`.

---

## Minor (one line each)

- `MobileNetV2SyncB.lean:308–312`, `MobileNetV2SyncStepTieB.lean:1022–1026`,
  `MobileNetV4SyncStepTieB.lean:1318–1320`: `have h112 : 0 < 112 := by norm_num` ×5 per capstone —
  `by decide` or `Nat.succ_pos _`, or take `Mg`-style bundled positivity once.
- `MobileNetV2Fold.lean:58,68`: `show depthwiseWeightSgdDen b x W lr cot idx = _` relies on `den`'s
  equation-compiler unfolding; `simp only [den]` (as `MobileNetV2FoldPaperG.lean:104` does) or a
  comment.
- `MobileNetV2Fold.lean:43`: `congr 1; congr 1` → `congr 2`.

---

## Recurring patterns (fix these, not the instances)

1. **One construction, one copy per net (and per variant).** The dominant cost of this
   directory. Counted sites:
   - N-stage opaque apex + its `_correct` restatement + tie restatement: 3 apexes (R34 18, V2 21,
     V4 26) and 4 `_correct`-style restatements of 119–150 lines each in these two files. Fix:
     apexes into `Foundation/OpaquePrefix.lean` with rfl-peel `_backward` lemmas;
     `HasVJPAt.correct_of_eq` for every `_correct`.
   - Homogeneity `_smul` lemmas: 53 here, 134 repo-wide. Fix: a `Homog` predicate with `.comp`
     (or `→ₗ[ℝ]` cotangent maps and `map_smul`).
   - Seal carrier steps `refine eDiff_bn … · exact eDiff_conv/dw … · simp only […]; ring`: 37 here,
     46 repo-wide. Fix: `EDiff_ctConvBn` / `EDiff_ctDwBn` kit lemmas at variable shapes. Plus the
     seal tail verbatim ×4 nets → one `JacobianSeal` lemma.
   - V2 vs V4 inverted-residual cotangent chain (plain / `_smul` / `Sync` / `_shard` / `_scaled`):
     ~60 declarations per net differing only in `relu6MaskB` vs `reluMaskB` and the UIB slot
     dispatch. Fix: a mask-generic chain (longer-term, largest single payoff).
   - Stage VJP + back-graph faithfulness skeleton: 7 copies (V2 3, V4 2, R34 2).

2. **Whole-chain closing `rfl` / numeral-shape statements behind heartbeat bumps.** 5 bumped
   declarations (V2 tie + `_correct` at 1M, V4 tie + `_correct` at 2M, PaperEval `maxRecDepth
   20000`) plus 1 file-wide 1M bump and 1 apparently vestigial `maxRecDepth 800000`. Each closes a
   statement at literal widths with an `rfl` that re-derives a composition. Fix: peel with
   `rw` by lemmas proved between variables (ConvNeXt's `vjpCompDiffAt_fst_backward` is the
   in-repo template), state the tie at variable widths, instantiate. Do NOT substitute
   `simp only [rfl-lemmas]` in these ties. Separately, the two V2 train-step capstones (1.6M each)
   look to be paying for embedding the literal forward in the statement; V4 takes the cotangent as
   a binder and needs no bump.

3. **V2 is the long-hand version of V4.** Where V4 uses a hypothesis structure (`Mnv4SmoothAt`),
   `CertLayer` groups, a generic `*Chain_apply`, and a named-`Prop` capstone with a free
   cotangent, V2 uses 38 positional binders, 18 prefix defs + 18 `_apply` lemmas, a 19-def `simp
   only` unfold, and inline `let`-telescopes. 9 of the findings above are "port V4's shape to V2".
   Porting it removes `sc1…sc17`, the duplicated `seal_differentiableAt` chain, both V2 capstone bumps, and
   `comp3_assoc`.
   Smaller recurring idiom: `show`-based defeq unfolding of `residual`/`biPath` (5 sites, no
   `residual_apply` / `residual_differentiableAt` lemma exists) and bare `rfl` closing shard
   lemmas (24 sites) while the named `rfl`-lemma `relu6MaskB_shard` goes unused.
