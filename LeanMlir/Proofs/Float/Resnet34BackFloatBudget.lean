import LeanMlir.Proofs.Float.FloatBudgetEnvBack

/-! # A NUMBER for the ResNet-34 BACKWARD: the whole-net input-gradient VJP

⭐⭐ **The first whole-net float number in this repo at TRAINING-mode BatchNorm, and the first
that is a FOLD rather than a cap since EfficientNet-B0.** For the certified input-gradient VJP of
the `[3,4,6,3]` ResNet-34 at `224²` — the exact reverse of `resnet34Forward_full_pc`, folded over
the loss cotangent — at the profile measured per parameter KIND on the trained checkpoint, for
any rounding model at binary32 accuracy:

    output window  ≤ 8.857·10²⁴⁵      (`r34GradBridge_mag_le`)
    fresh budget   ≤ 6.894·10²⁴⁴      (`r34GradBridge_fresh_le`)

and hence, per input-pixel gradient, `|float − real| ≤ 6.894·10²⁴⁴` (`r34_grad_float_le`), on
cotangents of magnitude `≤ 1` — which is what softmax cross-entropy gives (`|p − y| ≤ 1`).

⭐ **`budget / window = 0.0778`, and that ratio is the point.** ConvNeXt-T's and ViT-Tiny's
forward numbers come out at `2.00` because every normalisation site is discharged by
`FloatBridgesTo.capped` — the triangle inequality. This one is not capped anywhere: it is the
interval fold, at a net whose FORWARD has no number at all in this mode (§0.1's `10⁷⁴¹⁷`).

⭐⭐ **Why a backward can do what its own forward cannot.** §0.1's quadratic came from the
statistics MOVING with the input: perturb a training-mode BatchNorm's input and you move its mean
and its variance. A VJP does not perturb them — it reads them off the SAVED forward activations,
and the cotangent is a separate argument. So `floatClose_bnBack`'s modulus is
`bnGradInputBudget(A) + bnGradInputReMag(e)` with `bnGradInputReMag n G e S Xh = (1/n)·S·G·e·(2 +
Xh²)` — **linear in the inherited error**, because a VJP at a fixed point is a linear map and
this is its Lipschitz constant. No cap is available and none is needed.

⛔⛔ **THREE HYPOTHESES, AND THEY ARE THE WHOLE CAVEAT.** Two are the saved activations' float
accuracies — `es` on the inverse-stddev and `exh` on the normalised `x̂`, both at `10⁻²`, as
`DeviceRsqrt`'s and `DeviceGelu`'s accuracies are elsewhere in this cone. ⚠ Unlike those, they are
quantities the repo's own forward fold speaks about, and at training-mode BatchNorm it says
`10⁷⁴¹⁷`, not `10⁻²`. **The third is the operating point**, and it is new in this file:
`R34BnBack.hS` requires every site's inverse standard deviation to satisfy `|istd| ≤ 16` — i.e.
`σ ≥ 1/16` at the saved activations. So the theorem is an honest fold *given* a forward-accuracy
hypothesis its own forward cannot discharge, and *at an operating point*. Quote it that way.

⛔ **WHY THE OPERATING POINT, AND NOT THE `ε`-FLOOR.** At the unconditional bound
`|istd| ≤ 1/√ε ≤ 317` the same fold is **1.345·10²⁸⁸** — comfortably under the `10³⁰⁰` this repo
had recorded as `norm_num`'s ceiling, and NOT provable. ⭐⭐ **That ceiling is wrong, and the
correction is the finding of this file: `norm_num`'s wall is shape-dependent and bites around
`10²⁵³` for these expressions.** Bisected on the leaf shape itself:

    21/10 * 10^500                                ≤ 3 * 10^500      ✓
    4271 * 10^400                                 ≤ 4272 * 10^400   ✓
    21/10 * (4271 * 10^250)                       ≤ 8971 * 10^250   ✓
    21/10 * (4271 * 10^260)                       ≤ 8971 * 10^260   ✗
    317/12544 * (12544 * (21/10 * (4271*10^250))) ≤ 3571 * 10^257   ✗   ← same VALUE as row 3

so it is neither a magnitude nor a digit ceiling: it depends on the operation tree. Raising
`maxHeartbeats` or `maxRecDepth` does not move it, and neither `ring_nf`, `nlinarith`, nor
`simp only` + `norm_num` closes what `norm_num` alone will not. ConvNeXt-T's `10²²⁷` and
ViT-Tiny's `10²¹⁸` sit under it, which is why it had never been hit. ⭐ `S` is what buys the room
— `S` multiplies at all 33 BatchNorm sites, so `317 → 16` is worth ~43 orders — and `S` is
available as a lever precisely because `R34BnBack.hS` is a HYPOTHESIS about the saved
activations, not a consequence of `ε`. That is §0.1's escape 2, finally used.

⭐⭐ **The load-bearing lemma is `bnXhat_sq_le`, and it was already in the repo.** `Xh` bounds the
NORMALISED activation `x̂`, and it enters the fold as `Xh²`. Bounding it by the standardisation
`|x̂| ≤ √n` (exact at a conv net's square feature maps: `h·w = 56² ⇒ Xh = 56`) is what makes the
fold exist at all; deriving it from the FORWARD's certified window (`2·A·S`) gives `10⁷²⁷¹`. That
lemma sits in `Foundation/ResNet34.lean`, written for the realistic-seal work and named after
neither the float tier nor the backward. ⭐ Every place the activations enter an input-gradient
they enter normalised, or through `istd` — so **the backward's fold does not inherit the
forward's window at all**, which is the structural reason it is so much smaller than one would
guess.

**The profile, measured per parameter KIND** on `/home/skoonce/resnet/r34_imagenet_bf16_e79.bin`
(21,797,672 f32): conv and dense kernels max at **1.1007** over 21,779,648 entries, BN γ at
2.0741, BN β at 0.8118, the dense bias at 0.0475. ⚠ The forward's uniform `21/10` is a BN **γ**
and is 1.9× loose on exactly the entries every conv fan-in multiplies; splitting is worth 8
orders. ⭐ **The backward carries no bias anywhere** (`convFlatBack` and `linBack` are both at
bias `0`), so β and the dense bias enter no numeral.

**What it is stated about.** `r34InputGrad` with every abstract slot pinned: the 16 block
backwards are `r34IdBlockBack` / `r34DownBlockBack` at the record's real kernels and the
certified per-channel BatchNorm backwards, and `Resnet34BackCertifiedTie.lean` already ties each
of those — and every endpoint leaf — to its certified VJP
(`r34IdBlockBack_eq_rblkPC_vjp`, `r34DownBlockBack_eq_rblkPStridedPC_vjp`,
`dense_transpose_eq_vjp_backward`, `gapBack_eq_vjp_backward`,
`maxPool3s2FlatBack_eq_vjp_backward`, `cbrStridedPCBack_eq_vjp_backward`,
`convFlatBack_eq_vjp_backward`). ⭐⭐ **And the whole-net FOLD of those ties is now CLOSED too**
(`r34InputGrad_eq_resnet34_vjp`, 2026-09-03): `r34InputGrad` with every slot pinned to the
certified per-op backward IS `(resnet34_has_vjp_at …).backward` at the full `224²` dims. So the
reading is no longer "every piece of this chain is the certified gradient" — the chain IS the
certified whole-net gradient.

⛔⛔ **AND CLOSING IT MOVED THIS NUMBER, because it found a drift.** `r34InputGrad` pooled with
`maxPoolFlatBack` — the **2×2** pool's backward — while the committed forward
`resnet34Forward_full_pc` uses `maxPool3s2Flat`, He et al.'s 3×3/s2 stem pool (restored
2026-08-03). `MaxPool3s2.lean` warns in its header that the two share a TYPE and are different
functions; nothing forced the two statements to unify until the whole-net tie needed them to be
about one net. `maxPool3s2FlatBack` (`MaxPool3s2BackFloatBridge.lean`) is the missing leaf, and
because 3×3/s2 windows OVERLAP its backward ACCUMULATES — window `4A`, not the 2×2 peer's `A`.
The number moved `2.188·10²⁴⁵ → 8.857·10²⁴⁵`. ⚠ The `4` is proved from `win3Row_mem_le_two`, not
assumed: the trivial `c·h·w` fibre bound would put this at `10²⁵¹`, four orders under (a)'s
shape-dependent `norm_num` wall.

⚠ **Two mechanical lessons this file cost, both about ELABORATION ORDER and both new.**
(a) The whole-net bridge could not be rebuilt as one 22-stage `.comp` term: that makes the
elaborator compare the composition against `r34InputGrad` with every block CONCRETE, and the
`whnf` does not terminate in 4 000 000 heartbeats. Reusing `r34_grad_floatBridgesTo`, whose
blocks are hypotheses, makes the same comparison happen between opaque variables and cost
nothing. (b) **Pin all FOUR numerals on every leaf** — `(Ā := …) (Ē := …) (Ā' := …) (Ē' := …)`.
§3.3's lesson 3 said to pin the input window; that is not enough. Any `by norm_num` inside a
`have` runs before `Maps.comp` unifies anything, so an unpinned OUTPUT window is a metavariable
too, and the failure reads `⊢ <huge numeral> ≤ ?m.2125` rather than as a missing argument.

Provenance for the 254 numerals: `scripts/float_budget_envelope.py`'s `r34_back_chain`, which
folds 91 stages in exactly these lemmas' semantics with exact rationals and rounds every stage
UP to four significant figures, and `verify_r34_back`, which re-asserts each rounded inequality
before any of them is emitted.
-/

namespace Proofs

open FloatModel
open Classical

-- ════════════════════════════════════════════════════════════════
-- § The numeric profile, and the net's stored parameters and saved state
-- ════════════════════════════════════════════════════════════════

/-- The numeric profile the backward fold runs at. ⚠ Two of these are the caveat, not the
    machinery: `es` and `exh` are the deployed float inverse-stddev's and normalised
    activation's accuracies, SUPPLIED because a device `rsqrt` has no IEEE specification and
    because the saved activations come from a forward pass this theorem does not fold. -/
structure R34BackProfile (M : FloatModel) (ε wk gl S es exh q : ℝ) : Prop where
  /-- Conv and dense kernels — the backward has no bias anywhere, so this is the only weight
      bound it uses besides the BatchNorm γ. -/
  hwk : 0 ≤ wk
  /-- BatchNorm γ. -/
  hgl : 0 ≤ gl
  /-- ⛔ The float inverse-stddev's accuracy. SUPPLIED. -/
  hes : 0 ≤ es
  /-- ⛔ The float normalised activation's accuracy. SUPPLIED — see the file header. -/
  hexh : 0 ≤ exh
  hS0 : 0 ≤ S
  hε : 0 < ε
  hq : M.u ≤ q

/-- A conv kernel with its magnitude bound. The backward's peer of `R34Conv`, without the bias:
    `convFlatBack` and `linBack` are both stated at bias `0`. -/
structure R34KerB (oc ic kH kW : Nat) (wk : ℝ) where
  W : Kernel4 oc ic kH kW
  hW : ∀ o c kh kw, |W o c kh kw| ≤ wk

/-- The classifier kernel with its bound (the head's input-gradient is `Wᵀ·dy`). -/
structure R34HeadB (m n : Nat) (wk : ℝ) where
  W : Mat m n
  hW : ∀ i j, |W i j| ≤ wk

/-- ⭐ **One per-channel BatchNorm BACKWARD site.** Where the forward's `R34Bn` carries γ, β and
    the frozen running statistics, a backward site carries γ, the **saved forward activation**
    `x`, and the deployed float statistics computed from it — with their two SUPPLIED accuracies.
    ⛔ `x` is the input the forward pass saw; nothing here bounds it, and nothing needs to:
    `istd` is under `1/√ε` and `x̂` is under `√(h·w)` by standardisation, whatever `x` was. -/
structure R34BnBack (c h w : Nat) (ε gl S es exh : ℝ) where
  γ : Vec c
  /-- The SAVED forward activation this BatchNorm normalised. -/
  x : Vec (c * h * w)
  /-- The deployed float inverse-stddev, per channel. -/
  fs : Fin c → ℝ
  /-- The deployed float normalised activation, per channel. -/
  fxh : Fin c → Vec (h * w)
  hγ : ∀ k, |γ k| ≤ gl
  hs : ∀ k, |fs k - bnIstd (h * w) (Mat.unflatten (reassocFwd c h w x) k) ε| ≤ es
  hS : ∀ k, |bnIstd (h * w) (Mat.unflatten (reassocFwd c h w x) k) ε| ≤ S
  hfxh : ∀ k i, |fxh k i - bnXhat (h * w) ε (Mat.unflatten (reassocFwd c h w x) k) i| ≤ exh

/-- An identity basic block's backward data: two 3×3 kernels, two BatchNorm sites, and the two
    ReLU kink masks the smooth-point VJP reads. 13 of the 16 blocks. -/
structure R34IdBlkBack (c h w : Nat) (ε wk gl S es exh : ℝ) where
  k1 : R34KerB c c 3 3 wk
  k2 : R34KerB c c 3 3 wk
  bn1 : R34BnBack c h w ε gl S es exh
  bn2 : R34BnBack c h w ε gl S es exh
  mout : Fin (c * h * w) → Prop
  mmid : Fin (c * h * w) → Prop

/-- A downsample block's backward data: the two body kernels, the 1×1 option-B projection, three
    BatchNorm sites and the two masks. -/
structure R34DownBlkBack (ic oc h w : Nat) (ε wk gl S es exh : ℝ) where
  k1 : R34KerB oc ic 3 3 wk
  k2 : R34KerB oc oc 3 3 wk
  kp : R34KerB oc ic 1 1 wk
  bn1 : R34BnBack oc h w ε gl S es exh
  bn2 : R34BnBack oc h w ε gl S es exh
  bnp : R34BnBack oc h w ε gl S es exh
  mout : Fin (oc * h * w) → Prop
  mmid : Fin (oc * h * w) → Prop

/-- **The whole net's backward data** — 37 kernels, 34 BatchNorm backward sites (each with its
    saved activation), the max-pool argmax witness and the stem's ReLU mask. The backward peer of
    `R34Weights`, nested the same way so the chain stays at block granularity. -/
structure R34BackWeights (ε wk gl S es exh : ℝ) where
  stemK : R34KerB 64 3 7 7 wk
  stemBn : R34BnBack 64 112 112 ε gl S es exh
  head : R34HeadB 512 10 wk
  /-- The saved pre-pool activation the max-pool backward scatters through. -/
  xmp : Tensor3 64 112 112
  mstem : Fin (64 * 112 * 112) → Prop
  a0 : R34IdBlkBack 64 56 56 ε wk gl S es exh
  a1 : R34IdBlkBack 64 56 56 ε wk gl S es exh
  a2 : R34IdBlkBack 64 56 56 ε wk gl S es exh
  d2 : R34DownBlkBack 64 128 28 28 ε wk gl S es exh
  b0 : R34IdBlkBack 128 28 28 ε wk gl S es exh
  b1 : R34IdBlkBack 128 28 28 ε wk gl S es exh
  b2 : R34IdBlkBack 128 28 28 ε wk gl S es exh
  d3 : R34DownBlkBack 128 256 14 14 ε wk gl S es exh
  c0 : R34IdBlkBack 256 14 14 ε wk gl S es exh
  c1 : R34IdBlkBack 256 14 14 ε wk gl S es exh
  c2 : R34IdBlkBack 256 14 14 ε wk gl S es exh
  c3 : R34IdBlkBack 256 14 14 ε wk gl S es exh
  c4 : R34IdBlkBack 256 14 14 ε wk gl S es exh
  d4 : R34DownBlkBack 256 512 7 7 ε wk gl S es exh
  e0 : R34IdBlkBack 512 7 7 ε wk gl S es exh
  e1 : R34IdBlkBack 512 7 7 ε wk gl S es exh

-- ════════════════════════════════════════════════════════════════
-- § The real net, its float peer, and the closed bridge
-- ════════════════════════════════════════════════════════════════

section Net

variable {ε wk gl S es exh q : ℝ}

/-- The certified per-channel BatchNorm backward at this site. -/
noncomputable def R34BnBack.real {c h w : Nat} (s : R34BnBack c h w ε gl S es exh) :
    Vec (c * h * w) → Vec (c * h * w) :=
  fun dy => bnPerChannelTensor3_grad_input c h w ε s.γ s.x dy

/-- Its deployed float peer, at the supplied float statistics. -/
noncomputable def R34BnBack.float {c h w : Nat} (s : R34BnBack c h w ε gl S es exh)
    (M : FloatModel) : Vec (c * h * w) → Vec (c * h * w) :=
  bnPerChannelTensor3BackFV M s.γ s.fs s.fxh

/-- The site's bridge. ⭐ `Xh` is closed by `bnXhat_abs_le_num` off the standardisation bound;
    the caller supplies the perfect square (`h·w ≤ Xh²`), which for a conv net is exact. -/
noncomputable def R34BnBack.bridge {c h w : Nat} (s : R34BnBack c h w ε gl S es exh)
    (M : FloatModel) (P : R34BackProfile M ε wk gl S es exh q) (hc : 0 < c) (hhw : 0 < h * w)
    {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo s.real (s.float M) :=
  floatBridgesTo_bnPerChannelBack M s.γ s.x s.fs s.fxh hc hhw s.hγ s.hs s.hS
    (fun _k i => bnXhat_abs_le_num (X := Xh) P.hε _ hXh0 hnX i) s.hfxh

/-- One identity block's certified input-gradient VJP at the record's real data. -/
noncomputable def R34IdBlkBack.real {c h w : Nat} (b : R34IdBlkBack c h w ε wk gl S es exh) :
    Vec (c * h * w) → Vec (c * h * w) :=
  r34IdBlockBack b.k1.W b.k2.W b.bn1.real b.bn2.real b.mout b.mmid

/-- Its deployed float peer. -/
noncomputable def R34IdBlkBack.float {c h w : Nat} (b : R34IdBlkBack c h w ε wk gl S es exh)
    (M : FloatModel) : Vec (c * h * w) → Vec (c * h * w) :=
  r34IdBlockBackF M b.k1.W b.k2.W (b.bn1.float M) (b.bn2.float M) b.mout b.mmid

/-- The identity block's bridge, closed at real weights. -/
noncomputable def R34IdBlkBack.bridge {c h w : Nat} (b : R34IdBlkBack c h w ε wk gl S es exh)
    (M : FloatModel) (P : R34BackProfile M ε wk gl S es exh q) (hc : 0 < c) (hhw : 0 < h * w)
    (hn : 0 < c * h * w) {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo b.real (b.float M) :=
  floatBridgesTo_r34IdBlockBack M b.k1.W b.k2.W b.mout b.mmid P.hwk hn b.k1.hW b.k2.hW
    (b.bn1.bridge M P hc hhw hXh0 hnX) (b.bn2.bridge M P hc hhw hXh0 hnX)

/-- One downsample block's certified input-gradient VJP at the record's real data. -/
noncomputable def R34DownBlkBack.real {ic oc h w : Nat}
    (b : R34DownBlkBack ic oc h w ε wk gl S es exh) :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  r34DownBlockBack b.k1.W b.k2.W b.kp.W b.bn1.real b.bn2.real b.bnp.real b.mout b.mmid

/-- Its deployed float peer. -/
noncomputable def R34DownBlkBack.float {ic oc h w : Nat}
    (b : R34DownBlkBack ic oc h w ε wk gl S es exh) (M : FloatModel) :
    Vec (oc * h * w) → Vec (ic * (2 * h) * (2 * w)) :=
  r34DownBlockBackF M b.k1.W b.k2.W b.kp.W (b.bn1.float M) (b.bn2.float M) (b.bnp.float M)
    b.mout b.mmid

/-- The downsample block's bridge, closed at real weights. -/
noncomputable def R34DownBlkBack.bridge {ic oc h w : Nat}
    (b : R34DownBlkBack ic oc h w ε wk gl S es exh) (M : FloatModel)
    (P : R34BackProfile M ε wk gl S es exh q) (hoc : 0 < oc) (hh : 0 < h) (hw : 0 < w)
    (hhw : 0 < h * w) {Xh : ℝ} (hXh0 : 0 ≤ Xh) (hnX : ((h * w : ℕ) : ℝ) ≤ Xh ^ 2) :
    FloatBridgesTo b.real (b.float M) :=
  floatBridgesTo_r34DownBlockBack M b.k1.W b.k2.W b.kp.W b.mout b.mmid P.hwk P.hwk P.hwk
    b.k1.hW b.k2.hW b.kp.hW hoc hh hw
    (b.bn1.bridge M P hoc hhw hXh0 hnX) (b.bn2.bridge M P hoc hhw hXh0 hnX)
    (b.bnp.bridge M P hoc hhw hXh0 hnX)

/-- **The committed ResNet-34 input-gradient VJP at the record's real weights and saved state** —
    `r34InputGrad`'s body with every abstract slot pinned to the certified per-op backward, which
    is exactly the object `Resnet34BackCertifiedTie.lean` ties to the certified VJP.
    ⚠ Written with the stem's three stages GROUPED, exactly as `r34InputGrad` writes them. -/
noncomputable def r34GradR (w : R34BackWeights ε wk gl S es exh) :
    Vec 10 → Vec (3 * 224 * 224) :=
  (flatConvStride2Back (h := 112) (w := 112) w.stemK.W ∘ w.stemBn.real ∘ reluMaskBack w.mstem)
  ∘ maxPool3s2FlatBack (c := 64) (h := 56) (w := 56) w.xmp
  ∘ w.a0.real ∘ w.a1.real ∘ w.a2.real ∘ w.d2.real
  ∘ w.b0.real ∘ w.b1.real ∘ w.b2.real ∘ w.d3.real
  ∘ w.c0.real ∘ w.c1.real ∘ w.c2.real ∘ w.c3.real ∘ w.c4.real ∘ w.d4.real
  ∘ w.e0.real ∘ w.e1.real
  ∘ gapBack 512 7 7
  ∘ dense (Mat.transpose w.head.W) (0 : Vec 512)

/-- **The deployed float ResNet-34 input-gradient** — the same shape with every stage the float
    map its bridge names; the two structural scatters are unchanged (a select rounds nothing). -/
noncomputable def r34GradF (M : FloatModel) (w : R34BackWeights ε wk gl S es exh) :
    Vec 10 → Vec (3 * 224 * 224) :=
  ((M.flatConvF (h := 2 * 112) (w := 2 * 112) (IR.reverseSwap w.stemK.W) (fun _ => 0)
      ∘ decimateBack 64 112 112) ∘ w.stemBn.float M ∘ reluMaskBack w.mstem)
  ∘ M.maxPool3s2FlatBackF (c := 64) (h := 56) (w := 56) w.xmp
  ∘ w.a0.float M ∘ w.a1.float M ∘ w.a2.float M ∘ w.d2.float M
  ∘ w.b0.float M ∘ w.b1.float M ∘ w.b2.float M ∘ w.d3.float M
  ∘ w.c0.float M ∘ w.c1.float M ∘ w.c2.float M ∘ w.c3.float M ∘ w.c4.float M ∘ w.d4.float M
  ∘ w.e0.float M ∘ w.e1.float M
  ∘ gapBackF M 512 7 7
  ∘ M.dense (Mat.transpose w.head.W) (0 : Vec 512)

set_option maxRecDepth 400000 in
set_option maxHeartbeats 2000000 in
/-- ⭐ **The whole ResNet-34 input-gradient VJP float-bridges TO its float peer, CLOSED** — the
    16 block backwards and the stem's BatchNorm all discharged at the record's real data.

    ⚠⚠ **The GROUPING is load-bearing, and it is the expensive lesson of this file.** `.comp`
    builds `g ∘ f`, so a chain's type is right-nested — a FLAT `∘`-chain. `r34InputGrad` groups
    its stem, `(convBack ∘ bnBack ∘ mask) ∘ maxPool ∘ …`, and matching a flat chain against that
    forces the elaborator to unfold every leaf: it does not terminate in 4 000 000 heartbeats.
    Grouped the same way, the two agree STRUCTURALLY and cost nothing. ⭐ The `Maps` chain below
    must then be grouped identically — `.comp` is not associative as a bridge. -/
noncomputable def r34GradBridge (M : FloatModel) (P : R34BackProfile M ε wk gl S es exh q)
    (w : R34BackWeights ε wk gl S es exh) : FloatBridgesTo (r34GradR w) (r34GradF M w) :=
  (((((((((((((((((((floatBridgesTo_linBack M w.head.W P.hwk (by norm_num) w.head.hW).comp
      (floatBridgesTo_gapBack M 512 7 7 (by norm_num) (by norm_num) (by norm_num))).comp
      (w.e1.bridge M P (Xh := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.e0.bridge M P (Xh := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.d4.bridge M P (Xh := 7) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.c4.bridge M P (Xh := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.c3.bridge M P (Xh := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.c2.bridge M P (Xh := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.c1.bridge M P (Xh := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.c0.bridge M P (Xh := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.d3.bridge M P (Xh := 14) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.b2.bridge M P (Xh := 28) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.b1.bridge M P (Xh := 28) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.b0.bridge M P (Xh := 28) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.d2.bridge M P (Xh := 28) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.a2.bridge M P (Xh := 56) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.a1.bridge M P (Xh := 56) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (w.a0.bridge M P (Xh := 56) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (by norm_num))).comp
      (floatBridgesTo_maxPool3s2Back M (c := 64) (h := 56) (w := 56) w.xmp
        (by norm_num) (by norm_num) (by norm_num))).comp
    (((floatBridgesTo_reluMaskBack w.mstem).comp
        (w.stemBn.bridge M P (Xh := 112) (by norm_num) (by norm_num) (by norm_num)
          (by norm_num))).comp
      (floatBridgesTo_flatConvStride2Back (h := 112) (w := 112) M w.stemK.W P.hwk
        (by norm_num) w.stemK.hW))

end Net

-- ════════════════════════════════════════════════════════════════
-- § The committed profile, and the number
-- ════════════════════════════════════════════════════════════════

/-- **The committed profile**, measured per parameter KIND on the 79-epoch ImageNet run
    (`/home/skoonce/resnet/r34_imagenet_bf16_e79.bin`, 21,797,672 f32): conv and dense kernels
    within `12/10` (global max `1.1007` over 21,779,648 entries) and BatchNorm γ within `21/10`
    (max `2.0741`). ⚠ The forward's uniform `21/10` is a γ and is 1.9× loose on exactly the
    entries every conv fan-in multiplies. ⭐ β and the dense bias appear nowhere: the backward
    is stated at bias `0` throughout. `ε ≥ 10⁻⁵` puts every inverse-stddev under `317`; the float
    inverse-stddev and normalised activation are taken accurate to `10⁻²` — ⛔ SUPPLIED, and the
    file header is about exactly that. -/
theorem r34BackProfile_committed (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ}
    (hε5 : 1 / 100000 ≤ ε) :
    R34BackProfile M ε (12/10) (21/10) 16 (1/100) (1/100) u32 where
  hwk := by norm_num
  hgl := by norm_num
  hes := by norm_num
  hexh := by norm_num
  hS0 := by norm_num
  hε := by linarith
  hq := hMu

set_option maxRecDepth 4000000 in
set_option maxHeartbeats 8000000 in
/-- ⭐ **The envelope, kernel-checked.** 91 numeric stages, 254 rational inequalities, built
    bottom-up at block granularity (16 block steps rather than 90 leaf steps), with every γ-term
    bounded through `FloatModel.gamma_num` so `norm_num` never evaluates a big power.

    ⭐ Of the 254, NONE is a cap: `budget / window = 0.0778`, not `2.00`. Every one is the
    interval fold, at a net whose forward has no number in this mode at all. -/
theorem r34GradBridge_maps (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : R34BackWeights ε (12/10) (21/10) 16 (1/100) (1/100)) :
    (r34GradBridge M (r34BackProfile_committed M hMu hε5) w).Maps 1 0
      (8857 * 10 ^ 242) (6894 * 10 ^ 241) := by
  have P := r34BackProfile_committed M hMu hε5
  -- ⭐ the five per-resolution gain constants, each proved once
  have K49r : bnGradInputReMag (7 * 7) (21/10) 1 16 7
      ≤ 1714 := by norm_num [bnGradInputReMag]
  have K49b : bnGradInputBudgetG u32 (2981 / 10 ^ 9) (7 * 7) (21/10) 1 16 7
      (1/100) (1/100) ≤ 5787 / 10 ^ 3 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
    bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have K196r : bnGradInputReMag (14 * 14) (21/10) 1 16 14
      ≤ 6653 := by norm_num [bnGradInputReMag]
  have K196b : bnGradInputBudgetG u32 (1175 / 10 ^ 8) (14 * 14) (21/10) 1 16 14
      (1/100) (1/100) ≤ 1366 / 10 ^ 2 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
    bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have K784r : bnGradInputReMag (28 * 28) (21/10) 1 16 28
      ≤ 2641 * 10 ^ 1 := by norm_num [bnGradInputReMag]
  have K784b : bnGradInputBudgetG u32 (4680 / 10 ^ 8) (28 * 28) (21/10) 1 16 28
      (1/100) (1/100) ≤ 3659 / 10 ^ 2 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
    bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have K3136r : bnGradInputReMag (56 * 56) (21/10) 1 16 56
      ≤ 1055 * 10 ^ 2 := by norm_num [bnGradInputReMag]
  have K3136b : bnGradInputBudgetG u32 (1871 / 10 ^ 7) (56 * 56) (21/10) 1 16 56
      (1/100) (1/100) ≤ 1234 / 10 ^ 1 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
    bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  have K12544r : bnGradInputReMag (112 * 112) (21/10) 1 16 112
      ≤ 4216 * 10 ^ 2 := by norm_num [bnGradInputReMag]
  have K12544b : bnGradInputBudgetG u32 (7483 / 10 ^ 7) (112 * 112) (21/10) 1 16 112
      (1/100) (1/100) ≤ 6547 / 10 ^ 1 := by
    norm_num [bnGradInputBudgetG, bgMTr, bgEP, bgE2, bgM1, bgMXSf, bgE1, bgEXS,
    bgESXD, bgEXD, bgMND, bgEND, bgMSD, bgESD, bgED, FloatModel.mulErr, u32]
  -- the stem's three stages, grouped as `r34_grad_floatBridgesTo` groups them
  have hstem := ((FloatBridgesTo.Maps.reluMaskBack w.mstem (Ā := 5571 * 10 ^ 233) (Ē := 4245 * 10 ^ 232)).comp (by norm_num)
    (FloatBridgesTo.Maps.bnPerChannelBackGain M w.stemBn.γ w.stemBn.x w.stemBn.fs w.stemBn.fxh
      (by norm_num) (by norm_num) w.stemBn.hγ w.stemBn.hs w.stemBn.hS
      (fun c i => bnXhat_abs_le_num (X := 112) P.hε _ (by norm_num) (by norm_num) i)
      w.stemBn.hfxh (q := u32) (gn := 7483 / 10 ^ 7) P.hq
      (M.gamma_num (k := 112 * 112 + 1) (q := 7483 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
      P.hgl P.hS0 (by norm_num) P.hes P.hexh K12544r K12544b
      (by norm_num) (by norm_num)
      (Ā := 5571 * 10 ^ 233) (Ē := 4245 * 10 ^ 232) (Ā' := 2353 * 10 ^ 239) (Ē' := 1827 * 10 ^ 238)
      (by norm_num) (by norm_num))
    ).comp (by norm_num)
    (FloatBridgesTo.Maps.flatConvStride2Back (h := 112) (w := 112) M w.stemK.W
      P.hwk (by norm_num) w.stemK.hW
      (M.gamma_num (k := 64 * 7 * 7 + 2) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
      (Ā' := 8857 * 10 ^ 242) (Ē' := 6894 * 10 ^ 241) (by norm_num [u32]) (by norm_num [u32]))
  -- the loss head, then the trunk in the same groups
  have h0 := (FloatBridgesTo.Maps.linBack M w.head.W P.hwk (by norm_num) w.head.hW
    (M.gamma_num (k := 10 + 2) (q := 7153 / 10 ^ 10) hMu (by norm_num [u32]) (by norm_num [u32]))
    (Ā := 1) (Ē := 0) (Ā' := 1201 / 10 ^ 2) (Ē' := 8584 / 10 ^ 9)
    (by norm_num [u32]) (by norm_num [u32])).comp (by norm_num)
    (FloatBridgesTo.Maps.gapBack M 512 7 7 (by norm_num) (by norm_num) (by norm_num)
      P.hq (Ā' := 2452 / 10 ^ 4) (Ē' := 1898 / 10 ^ 10)
      (by norm_num [FloatModel.mulErr, u32]) (by norm_num [FloatModel.mulErr, u32]))
  have hE := (h0.comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 7) (w := 7)
        M w.e1.k1.W w.e1.k2.W w.e1.mout w.e1.mmid P.hwk (by norm_num)
        w.e1.k1.hW w.e1.k2.hW _ _
        (q := u32) (g := 2749 / 10 ^ 7) (Ā := 2452 / 10 ^ 4) (Ē := 1898 / 10 ^ 10)
        (Ā' := 2221 * 10 ^ 10) (Ē' := 1615 * 10 ^ 8)
        (A1 := 4217 / 10 ^ 1) (E1 := 1420 / 10 ^ 3)
        (A2 := 2333 * 10 ^ 3) (E2 := 8496)
        (A3 := 4013 * 10 ^ 6) (E3 := 2807 * 10 ^ 4)
        (A4 := 2220 * 10 ^ 10) (E4 := 1614 * 10 ^ 8)
        P.hq
        (M.gamma_num (k := 512 * 3 * 3 + 2) (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.e1.bn2.γ w.e1.bn2.x w.e1.bn2.fs w.e1.bn2.fxh
          (by norm_num) (by norm_num) w.e1.bn2.hγ w.e1.bn2.hs w.e1.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
          w.e1.bn2.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
          (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K49r K49b
          (by norm_num) (by norm_num)
          (Ā := 2452 / 10 ^ 4) (Ē := 1898 / 10 ^ 10) (Ā' := 4217 / 10 ^ 1) (Ē' := 1420 / 10 ^ 3)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.e1.bn1.γ w.e1.bn1.x w.e1.bn1.fs w.e1.bn1.fxh
          (by norm_num) (by norm_num) w.e1.bn1.hγ w.e1.bn1.hs w.e1.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
          w.e1.bn1.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
          (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K49r K49b
          (by norm_num) (by norm_num)
          (Ā := 2333 * 10 ^ 3) (Ē := 8496) (Ā' := 4013 * 10 ^ 6) (Ē' := 2807 * 10 ^ 4)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
    ).comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 7) (w := 7)
        M w.e0.k1.W w.e0.k2.W w.e0.mout w.e0.mmid P.hwk (by norm_num)
        w.e0.k1.hW w.e0.k2.hW _ _
        (q := u32) (g := 2749 / 10 ^ 7) (Ā := 2221 * 10 ^ 10) (Ē := 1615 * 10 ^ 8)
        (Ā' := 2012 * 10 ^ 24) (Ē' := 2915 * 10 ^ 22)
        (A1 := 3820 * 10 ^ 13) (E1 := 4054 * 10 ^ 11)
        (A2 := 2113 * 10 ^ 17) (E2 := 2301 * 10 ^ 15)
        (A3 := 3634 * 10 ^ 20) (E3 := 5167 * 10 ^ 18)
        (A4 := 2011 * 10 ^ 24) (E4 := 2914 * 10 ^ 22)
        P.hq
        (M.gamma_num (k := 512 * 3 * 3 + 2) (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.e0.bn2.γ w.e0.bn2.x w.e0.bn2.fs w.e0.bn2.fxh
          (by norm_num) (by norm_num) w.e0.bn2.hγ w.e0.bn2.hs w.e0.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
          w.e0.bn2.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
          (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K49r K49b
          (by norm_num) (by norm_num)
          (Ā := 2221 * 10 ^ 10) (Ē := 1615 * 10 ^ 8) (Ā' := 3820 * 10 ^ 13) (Ē' := 4054 * 10 ^ 11)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.e0.bn1.γ w.e0.bn1.x w.e0.bn1.fs w.e0.bn1.fxh
          (by norm_num) (by norm_num) w.e0.bn1.hγ w.e0.bn1.hs w.e0.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
          w.e0.bn1.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
          (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K49r K49b
          (by norm_num) (by norm_num)
          (Ā := 2113 * 10 ^ 17) (Ē := 2301 * 10 ^ 15) (Ā' := 3634 * 10 ^ 20) (Ē' := 5167 * 10 ^ 18)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
  have hD4 := hE.comp (by norm_num)
      (FloatBridgesTo.Maps.r34DownBlockBack (h := 7) (w := 7)
        M w.d4.k1.W w.d4.k2.W w.d4.kp.W w.d4.mout w.d4.mmid
        P.hwk P.hwk P.hwk w.d4.k1.hW w.d4.k2.hW w.d4.kp.hW
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) _ _ _
        (q := u32) (gp := 3064 / 10 ^ 8) (g1 := 2749 / 10 ^ 7) (g2 := 2749 / 10 ^ 7)
        (Ā := 2012 * 10 ^ 24) (Ē := 2915 * 10 ^ 22)
        (Ā' := 1823 * 10 ^ 38) (Ē' := 3946 * 10 ^ 36)
        (P1 := 3461 * 10 ^ 27) (F1 := 6161 * 10 ^ 25)
        (P2 := 2127 * 10 ^ 30) (F2 := 3792 * 10 ^ 28)
        (A1 := 3461 * 10 ^ 27) (E1 := 6161 * 10 ^ 25)
        (A2 := 1915 * 10 ^ 31) (E2 := 3461 * 10 ^ 29)
        (A3 := 3294 * 10 ^ 34) (E3 := 7041 * 10 ^ 32)
        (A4 := 1822 * 10 ^ 38) (E4 := 3945 * 10 ^ 36)
        P.hq
        (M.gamma_num (k := 512 * 1 * 1 + 2) (q := 3064 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 512 * 3 * 3 + 2) (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 512 * 3 * 3 + 2) (q := 2749 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.d4.bnp.γ w.d4.bnp.x w.d4.bnp.fs w.d4.bnp.fxh
          (by norm_num) (by norm_num) w.d4.bnp.hγ w.d4.bnp.hs w.d4.bnp.hS
          (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
          w.d4.bnp.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
          (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K49r K49b
          (by norm_num) (by norm_num)
          (Ā := 2012 * 10 ^ 24) (Ē := 2915 * 10 ^ 22) (Ā' := 3461 * 10 ^ 27) (Ē' := 6161 * 10 ^ 25)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.d4.bn2.γ w.d4.bn2.x w.d4.bn2.fs w.d4.bn2.fxh
          (by norm_num) (by norm_num) w.d4.bn2.hγ w.d4.bn2.hs w.d4.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
          w.d4.bn2.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
          (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K49r K49b
          (by norm_num) (by norm_num)
          (Ā := 2012 * 10 ^ 24) (Ē := 2915 * 10 ^ 22) (Ā' := 3461 * 10 ^ 27) (Ē' := 6161 * 10 ^ 25)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.d4.bn1.γ w.d4.bn1.x w.d4.bn1.fs w.d4.bn1.fxh
          (by norm_num) (by norm_num) w.d4.bn1.hγ w.d4.bn1.hs w.d4.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 7) P.hε _ (by norm_num) (by norm_num) i)
          w.d4.bn1.hfxh (q := u32) (gn := 2981 / 10 ^ 9) P.hq
          (M.gamma_num (k := 7 * 7 + 1) (q := 2981 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K49r K49b
          (by norm_num) (by norm_num)
          (Ā := 1915 * 10 ^ 31) (Ē := 3461 * 10 ^ 29) (Ā' := 3294 * 10 ^ 34) (Ē' := 7041 * 10 ^ 32)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
  have hC := ((((hD4.comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 14) (w := 14)
        M w.c4.k1.W w.c4.k2.W w.c4.mout w.c4.mmid P.hwk (by norm_num)
        w.c4.k1.hW w.c4.k2.hW _ _
        (q := u32) (g := 1375 / 10 ^ 7) (Ā := 1823 * 10 ^ 38) (Ē := 3946 * 10 ^ 36)
        (Ā' := 6201 * 10 ^ 52) (Ē' := 1608 * 10 ^ 51)
        (A1 := 1216 * 10 ^ 42) (E1 := 2875 * 10 ^ 40)
        (A2 := 3363 * 10 ^ 45) (E2 := 7997 * 10 ^ 43)
        (A3 := 2242 * 10 ^ 49) (E3 := 5780 * 10 ^ 47)
        (A4 := 6200 * 10 ^ 52) (E4 := 1607 * 10 ^ 51)
        P.hq
        (M.gamma_num (k := 256 * 3 * 3 + 2) (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c4.bn2.γ w.c4.bn2.x w.c4.bn2.fs w.c4.bn2.fxh
          (by norm_num) (by norm_num) w.c4.bn2.hγ w.c4.bn2.hs w.c4.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c4.bn2.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 1823 * 10 ^ 38) (Ē := 3946 * 10 ^ 36) (Ā' := 1216 * 10 ^ 42) (Ē' := 2875 * 10 ^ 40)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c4.bn1.γ w.c4.bn1.x w.c4.bn1.fs w.c4.bn1.fxh
          (by norm_num) (by norm_num) w.c4.bn1.hγ w.c4.bn1.hs w.c4.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c4.bn1.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 3363 * 10 ^ 45) (Ē := 7997 * 10 ^ 43) (Ā' := 2242 * 10 ^ 49) (Ē' := 5780 * 10 ^ 47)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
    ).comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 14) (w := 14)
        M w.c3.k1.W w.c3.k2.W w.c3.mout w.c3.mmid P.hwk (by norm_num)
        w.c3.k1.hW w.c3.k2.hW _ _
        (q := u32) (g := 1375 / 10 ^ 7) (Ā := 6201 * 10 ^ 52) (Ē := 1608 * 10 ^ 51)
        (Ā' := 2111 * 10 ^ 67) (Ē' := 6368 * 10 ^ 65)
        (A1 := 4134 * 10 ^ 56) (E1 := 1155 * 10 ^ 55)
        (A2 := 1144 * 10 ^ 60) (E2 := 3210 * 10 ^ 58)
        (A3 := 7627 * 10 ^ 63) (E3 := 2292 * 10 ^ 62)
        (A4 := 2110 * 10 ^ 67) (E4 := 6367 * 10 ^ 65)
        P.hq
        (M.gamma_num (k := 256 * 3 * 3 + 2) (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c3.bn2.γ w.c3.bn2.x w.c3.bn2.fs w.c3.bn2.fxh
          (by norm_num) (by norm_num) w.c3.bn2.hγ w.c3.bn2.hs w.c3.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c3.bn2.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 6201 * 10 ^ 52) (Ē := 1608 * 10 ^ 51) (Ā' := 4134 * 10 ^ 56) (Ē' := 1155 * 10 ^ 55)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c3.bn1.γ w.c3.bn1.x w.c3.bn1.fs w.c3.bn1.fxh
          (by norm_num) (by norm_num) w.c3.bn1.hγ w.c3.bn1.hs w.c3.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c3.bn1.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 1144 * 10 ^ 60) (Ē := 3210 * 10 ^ 58) (Ā' := 7627 * 10 ^ 63) (Ē' := 2292 * 10 ^ 62)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
    ).comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 14) (w := 14)
        M w.c2.k1.W w.c2.k2.W w.c2.mout w.c2.mmid P.hwk (by norm_num)
        w.c2.k1.hW w.c2.k2.hW _ _
        (q := u32) (g := 1375 / 10 ^ 7) (Ā := 2111 * 10 ^ 67) (Ē := 6368 * 10 ^ 65)
        (Ā' := 7180 * 10 ^ 81) (Ē' := 2471 * 10 ^ 80)
        (A1 := 1408 * 10 ^ 71) (E1 := 4525 * 10 ^ 69)
        (A2 := 3894 * 10 ^ 74) (E2 := 1257 * 10 ^ 73)
        (A3 := 2596 * 10 ^ 78) (E3 := 8895 * 10 ^ 76)
        (A4 := 7179 * 10 ^ 81) (E4 := 2470 * 10 ^ 80)
        P.hq
        (M.gamma_num (k := 256 * 3 * 3 + 2) (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c2.bn2.γ w.c2.bn2.x w.c2.bn2.fs w.c2.bn2.fxh
          (by norm_num) (by norm_num) w.c2.bn2.hγ w.c2.bn2.hs w.c2.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c2.bn2.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 2111 * 10 ^ 67) (Ē := 6368 * 10 ^ 65) (Ā' := 1408 * 10 ^ 71) (Ē' := 4525 * 10 ^ 69)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c2.bn1.γ w.c2.bn1.x w.c2.bn1.fs w.c2.bn1.fxh
          (by norm_num) (by norm_num) w.c2.bn1.hγ w.c2.bn1.hs w.c2.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c2.bn1.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 3894 * 10 ^ 74) (Ē := 1257 * 10 ^ 73) (Ā' := 2596 * 10 ^ 78) (Ē' := 8895 * 10 ^ 76)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
    ).comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 14) (w := 14)
        M w.c1.k1.W w.c1.k2.W w.c1.mout w.c1.mmid P.hwk (by norm_num)
        w.c1.k1.hW w.c1.k2.hW _ _
        (q := u32) (g := 1375 / 10 ^ 7) (Ā := 7180 * 10 ^ 81) (Ē := 2471 * 10 ^ 80)
        (Ā' := 2442 * 10 ^ 96) (Ē' := 9437 * 10 ^ 94)
        (A1 := 4787 * 10 ^ 85) (E1 := 1743 * 10 ^ 84)
        (A2 := 1324 * 10 ^ 89) (E2 := 4838 * 10 ^ 87)
        (A3 := 8827 * 10 ^ 92) (E3 := 3400 * 10 ^ 91)
        (A4 := 2441 * 10 ^ 96) (E4 := 9436 * 10 ^ 94)
        P.hq
        (M.gamma_num (k := 256 * 3 * 3 + 2) (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c1.bn2.γ w.c1.bn2.x w.c1.bn2.fs w.c1.bn2.fxh
          (by norm_num) (by norm_num) w.c1.bn2.hγ w.c1.bn2.hs w.c1.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c1.bn2.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 7180 * 10 ^ 81) (Ē := 2471 * 10 ^ 80) (Ā' := 4787 * 10 ^ 85) (Ē' := 1743 * 10 ^ 84)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c1.bn1.γ w.c1.bn1.x w.c1.bn1.fs w.c1.bn1.fxh
          (by norm_num) (by norm_num) w.c1.bn1.hγ w.c1.bn1.hs w.c1.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c1.bn1.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 1324 * 10 ^ 89) (Ē := 4838 * 10 ^ 87) (Ā' := 8827 * 10 ^ 92) (Ē' := 3400 * 10 ^ 91)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
    ).comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 14) (w := 14)
        M w.c0.k1.W w.c0.k2.W w.c0.mout w.c0.mmid P.hwk (by norm_num)
        w.c0.k1.hW w.c0.k2.hW _ _
        (q := u32) (g := 1375 / 10 ^ 7) (Ā := 2442 * 10 ^ 96) (Ē := 9437 * 10 ^ 94)
        (Ā' := 8303 * 10 ^ 110) (Ē' := 3561 * 10 ^ 109)
        (A1 := 1628 * 10 ^ 100) (E1 := 6613 * 10 ^ 98)
        (A2 := 4502 * 10 ^ 103) (E2 := 1835 * 10 ^ 102)
        (A3 := 3002 * 10 ^ 107) (E3 := 1283 * 10 ^ 106)
        (A4 := 8302 * 10 ^ 110) (E4 := 3560 * 10 ^ 109)
        P.hq
        (M.gamma_num (k := 256 * 3 * 3 + 2) (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c0.bn2.γ w.c0.bn2.x w.c0.bn2.fs w.c0.bn2.fxh
          (by norm_num) (by norm_num) w.c0.bn2.hγ w.c0.bn2.hs w.c0.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c0.bn2.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 2442 * 10 ^ 96) (Ē := 9437 * 10 ^ 94) (Ā' := 1628 * 10 ^ 100) (Ē' := 6613 * 10 ^ 98)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.c0.bn1.γ w.c0.bn1.x w.c0.bn1.fs w.c0.bn1.fxh
          (by norm_num) (by norm_num) w.c0.bn1.hγ w.c0.bn1.hs w.c0.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.c0.bn1.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 4502 * 10 ^ 103) (Ē := 1835 * 10 ^ 102) (Ā' := 3002 * 10 ^ 107) (Ē' := 1283 * 10 ^ 106)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
  have hD3 := hC.comp (by norm_num)
      (FloatBridgesTo.Maps.r34DownBlockBack (h := 14) (w := 14)
        M w.d3.k1.W w.d3.k2.W w.d3.kp.W w.d3.mout w.d3.mmid
        P.hwk P.hwk P.hwk w.d3.k1.hW w.d3.k2.hW w.d3.kp.hW
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) _ _ _
        (q := u32) (gp := 1538 / 10 ^ 8) (g1 := 1375 / 10 ^ 7) (g2 := 1375 / 10 ^ 7)
        (Ā := 8303 * 10 ^ 110) (Ē := 3561 * 10 ^ 109)
        (Ā' := 2825 * 10 ^ 125) (Ē' := 1330 * 10 ^ 124)
        (P1 := 5536 * 10 ^ 114) (F1 := 2483 * 10 ^ 113)
        (P2 := 1701 * 10 ^ 117) (F2 := 7631 * 10 ^ 115)
        (A1 := 5536 * 10 ^ 114) (E1 := 2483 * 10 ^ 113)
        (A2 := 1531 * 10 ^ 118) (E2 := 6887 * 10 ^ 116)
        (A3 := 1021 * 10 ^ 122) (E3 := 4792 * 10 ^ 120)
        (A4 := 2824 * 10 ^ 125) (E4 := 1329 * 10 ^ 124)
        P.hq
        (M.gamma_num (k := 256 * 1 * 1 + 2) (q := 1538 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 256 * 3 * 3 + 2) (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 256 * 3 * 3 + 2) (q := 1375 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.d3.bnp.γ w.d3.bnp.x w.d3.bnp.fs w.d3.bnp.fxh
          (by norm_num) (by norm_num) w.d3.bnp.hγ w.d3.bnp.hs w.d3.bnp.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.d3.bnp.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 8303 * 10 ^ 110) (Ē := 3561 * 10 ^ 109) (Ā' := 5536 * 10 ^ 114) (Ē' := 2483 * 10 ^ 113)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.d3.bn2.γ w.d3.bn2.x w.d3.bn2.fs w.d3.bn2.fxh
          (by norm_num) (by norm_num) w.d3.bn2.hγ w.d3.bn2.hs w.d3.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.d3.bn2.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 8303 * 10 ^ 110) (Ē := 3561 * 10 ^ 109) (Ā' := 5536 * 10 ^ 114) (Ē' := 2483 * 10 ^ 113)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.d3.bn1.γ w.d3.bn1.x w.d3.bn1.fs w.d3.bn1.fxh
          (by norm_num) (by norm_num) w.d3.bn1.hγ w.d3.bn1.hs w.d3.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 14) P.hε _ (by norm_num) (by norm_num) i)
          w.d3.bn1.hfxh (q := u32) (gn := 1175 / 10 ^ 8) P.hq
          (M.gamma_num (k := 14 * 14 + 1) (q := 1175 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K196r K196b
          (by norm_num) (by norm_num)
          (Ā := 1531 * 10 ^ 118) (Ē := 6887 * 10 ^ 116) (Ā' := 1021 * 10 ^ 122) (Ē' := 4792 * 10 ^ 120)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
  have hB := ((hD3.comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 28) (w := 28)
        M w.b2.k1.W w.b2.k2.W w.b2.mout w.b2.mmid P.hwk (by norm_num)
        w.b2.k1.hW w.b2.k2.hW _ _
        (q := u32) (g := 6879 / 10 ^ 8) (Ā := 2825 * 10 ^ 125) (Ē := 1330 * 10 ^ 124)
        (Ā' := 3783 * 10 ^ 140) (Ē' := 1886 * 10 ^ 139)
        (A1 := 7472 * 10 ^ 129) (E1 := 3616 * 10 ^ 128)
        (A2 := 1034 * 10 ^ 133) (E2 := 5007 * 10 ^ 131)
        (A3 := 2735 * 10 ^ 137) (E3 := 1361 * 10 ^ 136)
        (A4 := 3782 * 10 ^ 140) (E4 := 1885 * 10 ^ 139)
        P.hq
        (M.gamma_num (k := 128 * 3 * 3 + 2) (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b2.bn2.γ w.b2.bn2.x w.b2.bn2.fs w.b2.bn2.fxh
          (by norm_num) (by norm_num) w.b2.bn2.hγ w.b2.bn2.hs w.b2.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b2.bn2.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 2825 * 10 ^ 125) (Ē := 1330 * 10 ^ 124) (Ā' := 7472 * 10 ^ 129) (Ē' := 3616 * 10 ^ 128)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b2.bn1.γ w.b2.bn1.x w.b2.bn1.fs w.b2.bn1.fxh
          (by norm_num) (by norm_num) w.b2.bn1.hγ w.b2.bn1.hs w.b2.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b2.bn1.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 1034 * 10 ^ 133) (Ē := 5007 * 10 ^ 131) (Ā' := 2735 * 10 ^ 137) (Ē' := 1361 * 10 ^ 136)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
    ).comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 28) (w := 28)
        M w.b1.k1.W w.b1.k2.W w.b1.mout w.b1.mmid P.hwk (by norm_num)
        w.b1.k1.hW w.b1.k2.hW _ _
        (q := u32) (g := 6879 / 10 ^ 8) (Ā := 3783 * 10 ^ 140) (Ē := 1886 * 10 ^ 139)
        (Ā' := 5063 * 10 ^ 155) (Ē' := 2664 * 10 ^ 154)
        (A1 := 1001 * 10 ^ 145) (E1 := 5120 * 10 ^ 143)
        (A2 := 1384 * 10 ^ 148) (E2 := 7088 * 10 ^ 146)
        (A3 := 3661 * 10 ^ 152) (E3 := 1923 * 10 ^ 151)
        (A4 := 5062 * 10 ^ 155) (E4 := 2663 * 10 ^ 154)
        P.hq
        (M.gamma_num (k := 128 * 3 * 3 + 2) (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b1.bn2.γ w.b1.bn2.x w.b1.bn2.fs w.b1.bn2.fxh
          (by norm_num) (by norm_num) w.b1.bn2.hγ w.b1.bn2.hs w.b1.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b1.bn2.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 3783 * 10 ^ 140) (Ē := 1886 * 10 ^ 139) (Ā' := 1001 * 10 ^ 145) (Ē' := 5120 * 10 ^ 143)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b1.bn1.γ w.b1.bn1.x w.b1.bn1.fs w.b1.bn1.fxh
          (by norm_num) (by norm_num) w.b1.bn1.hγ w.b1.bn1.hs w.b1.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b1.bn1.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 1384 * 10 ^ 148) (Ē := 7088 * 10 ^ 146) (Ā' := 3661 * 10 ^ 152) (Ē' := 1923 * 10 ^ 151)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
    ).comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 28) (w := 28)
        M w.b0.k1.W w.b0.k2.W w.b0.mout w.b0.mmid P.hwk (by norm_num)
        w.b0.k1.hW w.b0.k2.hW _ _
        (q := u32) (g := 6879 / 10 ^ 8) (Ā := 5063 * 10 ^ 155) (Ē := 2664 * 10 ^ 154)
        (Ā' := 6773 * 10 ^ 170) (Ē' := 3750 * 10 ^ 169)
        (A1 := 1339 * 10 ^ 160) (E1 := 7221 * 10 ^ 158)
        (A2 := 1852 * 10 ^ 163) (E2 := 9996 * 10 ^ 161)
        (A3 := 4898 * 10 ^ 167) (E3 := 2708 * 10 ^ 166)
        (A4 := 6772 * 10 ^ 170) (E4 := 3749 * 10 ^ 169)
        P.hq
        (M.gamma_num (k := 128 * 3 * 3 + 2) (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b0.bn2.γ w.b0.bn2.x w.b0.bn2.fs w.b0.bn2.fxh
          (by norm_num) (by norm_num) w.b0.bn2.hγ w.b0.bn2.hs w.b0.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b0.bn2.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 5063 * 10 ^ 155) (Ē := 2664 * 10 ^ 154) (Ā' := 1339 * 10 ^ 160) (Ē' := 7221 * 10 ^ 158)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.b0.bn1.γ w.b0.bn1.x w.b0.bn1.fs w.b0.bn1.fxh
          (by norm_num) (by norm_num) w.b0.bn1.hγ w.b0.bn1.hs w.b0.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.b0.bn1.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 1852 * 10 ^ 163) (Ē := 9996 * 10 ^ 161) (Ā' := 4898 * 10 ^ 167) (Ē' := 2708 * 10 ^ 166)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
  have hD2 := hB.comp (by norm_num)
      (FloatBridgesTo.Maps.r34DownBlockBack (h := 28) (w := 28)
        M w.d2.k1.W w.d2.k2.W w.d2.kp.W w.d2.mout w.d2.mmid
        P.hwk P.hwk P.hwk w.d2.k1.hW w.d2.k2.hW w.d2.kp.hW
        (by norm_num) (by norm_num) (by norm_num) (by norm_num) _ _ _
        (q := u32) (gp := 7749 / 10 ^ 9) (g1 := 6879 / 10 ^ 8) (g2 := 6879 / 10 ^ 8)
        (Ā := 6773 * 10 ^ 170) (Ē := 3750 * 10 ^ 169)
        (Ā' := 9062 * 10 ^ 185) (Ē' := 5271 * 10 ^ 184)
        (P1 := 1792 * 10 ^ 175) (F1 := 1016 * 10 ^ 174)
        (P2 := 2753 * 10 ^ 177) (F2 := 1561 * 10 ^ 176)
        (A1 := 1792 * 10 ^ 175) (E1 := 1016 * 10 ^ 174)
        (A2 := 2478 * 10 ^ 178) (E2 := 1407 * 10 ^ 177)
        (A3 := 6554 * 10 ^ 182) (E3 := 3807 * 10 ^ 181)
        (A4 := 9061 * 10 ^ 185) (E4 := 5270 * 10 ^ 184)
        P.hq
        (M.gamma_num (k := 128 * 1 * 1 + 2) (q := 7749 / 10 ^ 9) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 128 * 3 * 3 + 2) (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (M.gamma_num (k := 128 * 3 * 3 + 2) (q := 6879 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.d2.bnp.γ w.d2.bnp.x w.d2.bnp.fs w.d2.bnp.fxh
          (by norm_num) (by norm_num) w.d2.bnp.hγ w.d2.bnp.hs w.d2.bnp.hS
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.d2.bnp.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 6773 * 10 ^ 170) (Ē := 3750 * 10 ^ 169) (Ā' := 1792 * 10 ^ 175) (Ē' := 1016 * 10 ^ 174)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.d2.bn2.γ w.d2.bn2.x w.d2.bn2.fs w.d2.bn2.fxh
          (by norm_num) (by norm_num) w.d2.bn2.hγ w.d2.bn2.hs w.d2.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.d2.bn2.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 6773 * 10 ^ 170) (Ē := 3750 * 10 ^ 169) (Ā' := 1792 * 10 ^ 175) (Ē' := 1016 * 10 ^ 174)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.d2.bn1.γ w.d2.bn1.x w.d2.bn1.fs w.d2.bn1.fxh
          (by norm_num) (by norm_num) w.d2.bn1.hγ w.d2.bn1.hs w.d2.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 28) P.hε _ (by norm_num) (by norm_num) i)
          w.d2.bn1.hfxh (q := u32) (gn := 4680 / 10 ^ 8) P.hq
          (M.gamma_num (k := 28 * 28 + 1) (q := 4680 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K784r K784b
          (by norm_num) (by norm_num)
          (Ā := 2478 * 10 ^ 178) (Ē := 1407 * 10 ^ 177) (Ā' := 6554 * 10 ^ 182) (Ē' := 3807 * 10 ^ 181)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
  have hA := ((hD2.comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 56) (w := 56)
        M w.a2.k1.W w.a2.k2.W w.a2.mout w.a2.mmid P.hwk (by norm_num)
        w.a2.k1.hW w.a2.k2.hW _ _
        (q := u32) (g := 3446 / 10 ^ 8) (Ā := 9062 * 10 ^ 185) (Ē := 5271 * 10 ^ 184)
        (Ā' := 4833 * 10 ^ 201) (Ē' := 2922 * 10 ^ 200)
        (A1 := 9572 * 10 ^ 190) (E1 := 5673 * 10 ^ 189)
        (A2 := 6617 * 10 ^ 193) (E2 := 3924 * 10 ^ 192)
        (A3 := 6990 * 10 ^ 198) (E3 := 4222 * 10 ^ 197)
        (A4 := 4832 * 10 ^ 201) (E4 := 2921 * 10 ^ 200)
        P.hq
        (M.gamma_num (k := 64 * 3 * 3 + 2) (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.a2.bn2.γ w.a2.bn2.x w.a2.bn2.fs w.a2.bn2.fxh
          (by norm_num) (by norm_num) w.a2.bn2.hγ w.a2.bn2.hs w.a2.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.a2.bn2.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 9062 * 10 ^ 185) (Ē := 5271 * 10 ^ 184) (Ā' := 9572 * 10 ^ 190) (Ē' := 5673 * 10 ^ 189)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.a2.bn1.γ w.a2.bn1.x w.a2.bn1.fs w.a2.bn1.fxh
          (by norm_num) (by norm_num) w.a2.bn1.hγ w.a2.bn1.hs w.a2.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.a2.bn1.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 6617 * 10 ^ 193) (Ē := 3924 * 10 ^ 192) (Ā' := 6990 * 10 ^ 198) (Ē' := 4222 * 10 ^ 197)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
    ).comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 56) (w := 56)
        M w.a1.k1.W w.a1.k2.W w.a1.mout w.a1.mmid P.hwk (by norm_num)
        w.a1.k1.hW w.a1.k2.hW _ _
        (q := u32) (g := 3446 / 10 ^ 8) (Ā := 4833 * 10 ^ 201) (Ē := 2922 * 10 ^ 200)
        (Ā' := 2578 * 10 ^ 217) (Ē' := 1618 * 10 ^ 216)
        (A1 := 5105 * 10 ^ 206) (E1 := 3143 * 10 ^ 205)
        (A2 := 3529 * 10 ^ 209) (E2 := 2174 * 10 ^ 208)
        (A3 := 3728 * 10 ^ 214) (E3 := 2338 * 10 ^ 213)
        (A4 := 2577 * 10 ^ 217) (E4 := 1617 * 10 ^ 216)
        P.hq
        (M.gamma_num (k := 64 * 3 * 3 + 2) (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.a1.bn2.γ w.a1.bn2.x w.a1.bn2.fs w.a1.bn2.fxh
          (by norm_num) (by norm_num) w.a1.bn2.hγ w.a1.bn2.hs w.a1.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.a1.bn2.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 4833 * 10 ^ 201) (Ē := 2922 * 10 ^ 200) (Ā' := 5105 * 10 ^ 206) (Ē' := 3143 * 10 ^ 205)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.a1.bn1.γ w.a1.bn1.x w.a1.bn1.fs w.a1.bn1.fxh
          (by norm_num) (by norm_num) w.a1.bn1.hγ w.a1.bn1.hs w.a1.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.a1.bn1.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 3529 * 10 ^ 209) (Ē := 2174 * 10 ^ 208) (Ā' := 3728 * 10 ^ 214) (Ē' := 2338 * 10 ^ 213)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
    ).comp (by norm_num)
      (FloatBridgesTo.Maps.r34IdBlockBack (h := 56) (w := 56)
        M w.a0.k1.W w.a0.k2.W w.a0.mout w.a0.mmid P.hwk (by norm_num)
        w.a0.k1.hW w.a0.k2.hW _ _
        (q := u32) (g := 3446 / 10 ^ 8) (Ā := 2578 * 10 ^ 217) (Ē := 1618 * 10 ^ 216)
        (Ā' := 1376 * 10 ^ 233) (Ē' := 8944 * 10 ^ 231)
        (A1 := 2723 * 10 ^ 222) (E1 := 1739 * 10 ^ 221)
        (A2 := 1883 * 10 ^ 225) (E2 := 1203 * 10 ^ 224)
        (A3 := 1989 * 10 ^ 230) (E3 := 1293 * 10 ^ 229)
        (A4 := 1375 * 10 ^ 233) (E4 := 8943 * 10 ^ 231)
        P.hq
        (M.gamma_num (k := 64 * 3 * 3 + 2) (q := 3446 / 10 ^ 8) hMu (by norm_num [u32]) (by norm_num [u32]))
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.a0.bn2.γ w.a0.bn2.x w.a0.bn2.fs w.a0.bn2.fxh
          (by norm_num) (by norm_num) w.a0.bn2.hγ w.a0.bn2.hs w.a0.bn2.hS
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.a0.bn2.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 2578 * 10 ^ 217) (Ē := 1618 * 10 ^ 216) (Ā' := 2723 * 10 ^ 222) (Ē' := 1739 * 10 ^ 221)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32])
        (FloatBridgesTo.Maps.bnPerChannelBackGain M w.a0.bn1.γ w.a0.bn1.x w.a0.bn1.fs w.a0.bn1.fxh
          (by norm_num) (by norm_num) w.a0.bn1.hγ w.a0.bn1.hs w.a0.bn1.hS
          (fun c i => bnXhat_abs_le_num (X := 56) P.hε _ (by norm_num) (by norm_num) i)
          w.a0.bn1.hfxh (q := u32) (gn := 1871 / 10 ^ 7) P.hq
          (M.gamma_num (k := 56 * 56 + 1) (q := 1871 / 10 ^ 7) hMu (by norm_num [u32]) (by norm_num [u32]))
          P.hgl P.hS0 (by norm_num) P.hes P.hexh K3136r K3136b
          (by norm_num) (by norm_num)
          (Ā := 1883 * 10 ^ 225) (Ē := 1203 * 10 ^ 224) (Ā' := 1989 * 10 ^ 230) (Ē' := 1293 * 10 ^ 229)
          (by norm_num) (by norm_num))
        (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]) (by norm_num [u32]))
  have hMP := hA.comp (by norm_num)
    (FloatBridgesTo.Maps.maxPool3s2Back M (c := 64) (h := 56) (w := 56) w.xmp
      (by norm_num) (by norm_num) (by norm_num) (g := 1211 / 10 ^ 5) (by norm_num)
      (M.gamma_num (k := 64 * 56 * 56 + 1) (q := 1211 / 10 ^ 5) hMu
        (by norm_num [u32]) (by norm_num [u32]))
      (Ā := 1376 * 10 ^ 233) (Ē := 8944 * 10 ^ 231)
      (Ā' := 5571 * 10 ^ 233) (Ē' := 4245 * 10 ^ 232) (by norm_num) (by norm_num))
  exact hMP.comp (by norm_num) hstem

/-- The deployed r34 backward bridge's certified output window at the committed profile. -/
theorem r34GradBridge_mag_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : R34BackWeights ε (12/10) (21/10) 16 (1/100) (1/100)) :
    (r34GradBridge M (r34BackProfile_committed M hMu hε5) w).mag 1 ≤ 8857 * 10 ^ 242 :=
  (r34GradBridge_maps M hMu hε5 w).mag_le 1 (by norm_num) le_rfl

/-- ⭐ The deployed r34 backward bridge's fresh budget — `0.0778 ×` the certified window, and that
    ratio is the tell that this is the FOLD and not the triangle inequality (§9). -/
theorem r34GradBridge_fresh_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : R34BackWeights ε (12/10) (21/10) 16 (1/100) (1/100)) :
    (r34GradBridge M (r34BackProfile_committed M hMu hε5) w).fresh 1 ≤ 6894 * 10 ^ 241 :=
  (r34GradBridge_maps M hMu hε5 w).mod_le 1 0 (by norm_num) le_rfl le_rfl le_rfl

/-- ⭐⭐ **The deployed ResNet-34 input-gradient is within `6.894·10²⁴⁴` of the certified real
    input-gradient, per pixel**, on loss cotangents of magnitude `≤ 1` (which is what softmax
    cross-entropy gives, `|p − y| ≤ 1`), at the measured per-kind parameter profile, for
    `ε ≥ 10⁻⁵`, any float inverse-stddev and normalised activation accurate to `10⁻²`, and any
    rounding model at binary32 accuracy.

    ⭐ **It is the interval FOLD, at TRAINING-mode BatchNorm** — `budget / window = 0.0778`, not
    `2.00` — where the forward of this same net in this same mode has no statable number at all.
    ⛔ **And read the file header before quoting it**: `es` and `exh` are supplied, and they are
    the two quantities the forward's own training-mode fold cannot discharge. -/
theorem r34_grad_float_le (M : FloatModel) (hMu : M.u ≤ u32) {ε : ℝ} (hε5 : 1 / 100000 ≤ ε)
    (w : R34BackWeights ε (12/10) (21/10) 16 (1/100) (1/100))
    (dy : Vec 10) (hdy : ∀ k, |dy k| ≤ 1) (j : Fin (3 * 224 * 224)) :
    |r34GradF M w dy j - r34GradR w dy j| ≤ 6894 * 10 ^ 241 :=
  (r34GradBridge_maps M hMu hε5 w).budget_le (by norm_num) le_rfl dy hdy j

end Proofs
