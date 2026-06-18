import LeanMlir.Proofs.EfficientNetFaithfulPoC
import LeanMlir.Proofs.EfficientNetFullB0

/-! # PoC: the full-16 EfficientNet-B0 train step §1a TIE (whole-net thread) — IN PROGRESS

The EfficientNet peer of `MobileNetV2TiePoCPaper` (the §1a tie). The §1 fold
(`EfficientNetFaithfulPoC`) gives every batched param-SGD op `den = certified ∀ cotangent`; this
file pins each cotangent to the **actual loss-driven backward chain** of the rendered net — threading
the real forward activations through every param op and composing the backward cotangent from the
loss down through all 16 MBConv blocks (with the residual fan-in at every stride-1 skip AND the SE
gate fan-in), so each output's `den = certified` becomes a single composed theorem with the forward
= the proven `efficientnetForwardB`.

**What is NEW vs mnv2's tie** (the harder content, hence a dedicated effort):
* **swish** masks instead of relu6 — the cotangent crosses `swishBack` (smooth, no two-kink
  `selectMid`) at every conv-bn-swish / depthwise-bn-swish stage.
* the **SE multiplicative gate fan-in** — the cotangent at an SE input is `gate ⊙ dyOut` (the
  fused `seBackBatched` value) and the SE dense param cotangents come from `seReduceB` (the gate
  cotangent `Σ_{h,w}(x⊙dy)`) threaded back through `sigmoidBack → denseRowBack(W₂) → swishBack`.
* **true batch-norm** backward (`bnBatchBack`) — batch-coupled, vs mnv2's per-example BN.
* `EfficientNetChainClose`'s whole-net backward is **HasVJP-composition** style (`vjp_comp` of the
  per-block `_has_vjp`), NOT explicit cotangent-vector defs like mnv2's `invresCot*` — so the tie
  must BUILD explicit chain-cot constructors (the bulk of the remaining work).

## Landed so far (all 3-axiom clean — `[propext, Classical.choice, Quot.sound]`)
* `efficientnetLossCot_den` — the emitted loss-cotangent graph (`softmaxRowF − %onehot`, the batched
  per-row softmax-CE gradient) denotes `rowSoftmax(logits) − onehot`. The top of the cotangent chain.
* **All five per-block-type tie lemmas — every one of the 262 params' SGD ops** denotes the certified
  batched `Σ_n` loss-descent step at the REAL loss-driven backward cotangent:
  - `enet_exp_tied` (16 params) — stride-1 expand block; covers the 9 residual blocks (`ic=oc`) AND the
    2 no-skip widenings (`ic≠oc`, b9/b16); the param ops are skip-agnostic (the fan-in lives in the thread).
  - `enet_strided_tied` (16) — strided downsample (b2/4/6/12): expand at `2h×2w`, strided depthwise.
  - `enet_noexp_tied` (12) — b1 (t=1, no expand; depthwise on `ic` → SE → project).
  - `enet_stem_tied` (4) — 3×3/s2 conv-bn-swish stem.
  - `enet_head_tied` (6) — 1×1 conv-bn-swish head + dense (Wfc/bfc tied at the loss cotangent `g`).
* The genuinely-new content vs mnv2 is PROVEN here: **swish** masks (`swBackB`), the **SE gate fan-in**
  (`gateCotB`=`den seReduceB` → `sigBackB` → `rowDenseBackFlat` → `swBackB` → SE dense ops), **true
  batch-norm** backward (`bnBackB`=`bnBatchLA` VJP), the strided depthwise back (`dStridedInB`), all at
  the **batched index** `N·(c·h·w)`. `reassocB` bridges the conv/swish `(oc·h·w)` ↔ BN `(oc·(h·w))` index.
  Each per-block tie is a pure delegation to the §1-fold generics `EnetPoC.*` at the chain cotangents.

## Remaining
1. The whole-net thread `efficientnet_net_tied_certified` over the 16 blocks + stem + head, threading
   the full forward `efficientnetForwardB_full` (FullB0): block inputs are its prefixes, the per-block
   `dyOut`s are composed top-down by the proven block VJPs (`mb{Resid,Strided,Exp,NoExp}W_has_vjp` and
   `headFwdB_has_vjp`), with `@[irreducible]` `FwdO`/`CotInAt`/`TiedAt` wrappers to dodge the heartbeat
   blowup (the r34/mnv2 lesson, more acute at 16 blocks + SE). Mechanical enumeration over the bricks above.
2. (Optional refinement) the dense-head total-loss fold (`Wfc → ∂CE/∂Wfc`) — the batched-`Σ_n` analogue
   of `mlp_output_total_loss_grad`; today the head dense ties at the loss cotangent `g` directly. -/

open Proofs Proofs.StableHLO

namespace Proofs.EnetTiePoC

open scoped BigOperators

/-- **The emitted loss-cotangent graph denotes the (batched, per-row) softmax-CE gradient at the
    logits.** The renderer's `sub (softmaxRowF logits) %onehot`; `rowSoftmaxFlat` is the per-row
    softmax over the `n=10` classes. The top of the §1a cotangent chain (generic in the batch unit
    `m`; the renderer instantiates `m=1`). -/
theorem efficientnetLossCot_den {m : Nat} (nlogN ohN : String) (logits oh : Vec (m * 10)) :
    den (SHlo.sub (SHlo.softmaxRowF (m := m) (n := 10) (.operand nlogN logits)) (.operand ohN oh))
      = fun idx => rowSoftmaxFlat m 10 logits idx - oh idx := by
  funext idx
  simp only [den_sub, softmaxRowF_faithful, den_operand]

/-! ## Chain-cotangent helpers — the per-op batched backward steps (built fresh, HasVJP-style)

`EfficientNetChainClose` proves the per-block VJPs by `vjp_comp` of the per-op VJPs but exposes no
explicit cotangent-vector defs (unlike mnv2's `invresCot*`). So the tie BUILDS the chain cotangents
from the proven per-op backwards: `bnBackB` (true-BN, the batch-coupled `bnBatchLA` VJP), `swBackB`
(swish, smooth), `cInB`/`dInB` (the batched conv/depthwise input-VJP = `den convBackBatched`/
`depthwiseBackBatched`), `seInB` (the fused SE input-cot = `den seBackBatched`), `gateCotB` (the SE
gate cotangent = `den seReduceB`), `sigBackB`, `rowDenseBackFlat` (the SE excite/reduce backs). Every
helper IS a `.backward` of a proven VJP (or the exact `den` of the emitted backward op), so the
cotangents are the genuine loss-driven backward, not a free `∀c`. `reassocB` bridges the conv/swish
index `(oc·h·w)` to the BN param-op index `(oc·(h·w))`. -/

/-- `(oc·h·w) → (oc·(h·w))` batched reassociation reindex — bridges the conv/swish chain index to the
    BN γ/β + conv-bias op index (`EnetPoC.bn{Gamma,Beta}B_den` consume `Vec (N·(oc·(h·w)))`). -/
noncomputable def reassocB (N oc h w : Nat) (v : Vec (N * (oc * h * w))) : Vec (N * (oc * (h * w))) :=
  fun i => v (Fin.cast (congrArg (N * ·) (Nat.mul_assoc oc h w)).symm i)

/-- Batched **true-BN** input-cotangent (`bnBatchLA` VJP — batch-coupled). -/
noncomputable def bnBackB (N oc h w : Nat) (ε : ℝ) (hε : 0 < ε) (γ β : Vec oc)
    (x dy : Vec (N * (oc * h * w))) : Vec (N * (oc * h * w)) :=
  (bnBatchLA_has_vjp N oc h w ε hε γ β).backward x dy

/-- Batched **swish** mask-back (smooth, no kink). -/
noncomputable def swBackB (n : Nat) (x dy : Vec n) : Vec n := (swish_has_vjp n).backward x dy

/-- Batched **sigmoid** back (the SE gate excite-dense output cotangent). -/
noncomputable def sigBackB (n : Nat) (x dy : Vec n) : Vec n := (sigmoid_has_vjp n).backward x dy

/-- Batched **1×1/conv input-VJP** (= `den convBackBatched`; conv is linear, `x` unused). -/
noncomputable def cInB (N : Nat) {ic oc h w kH kW : Nat} (W : Kernel4 oc ic kH kW) (b : Vec oc)
    (dy : Vec (N * (oc * h * w))) : Vec (N * (ic * h * w)) :=
  batchMap N (fun d => (flatConv_has_vjp W b).backward (fun _ => 0) d) dy

/-- Batched **depthwise input-VJP** (= `den depthwiseBackBatched`). -/
noncomputable def dInB (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (dy : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  batchMap N (fun d => (depthwiseFlat_has_vjp W b).backward (fun _ => 0) d) dy

/-- Batched **strided depthwise input-VJP** (= `den depthwiseStridedBackBatched`; upsamples `h→2h`). -/
noncomputable def dStridedInB (N : Nat) {c h w kH kW : Nat} (W : DepthwiseKernel c kH kW) (b : Vec c)
    (dy : Vec (N * (c * h * w))) : Vec (N * (c * (2 * h) * (2 * w))) :=
  batchMap N (fun d => (depthwiseStride2Flat_has_vjp W b).backward (fun _ => 0) d) dy

/-- Batched **GAP input-VJP** (= `den gapBackBatched`; the head's GAP backward, broadcast÷(h·w)). -/
noncomputable def gapInB (N c h w : Nat) (dy : Vec (N * c)) : Vec (N * (c * h * w)) :=
  batchMap N (fun d => (globalAvgPoolFlat_has_vjp c h w).backward (fun _ => 0) d) dy

/-- Batched **fused SE input-cotangent** (= `den seBackBatched`, the `x⊙gate` VJP). -/
noncomputable def seInB (N : Nat) {c h w r : Nat} (W₁ : Mat c r) (b₁ : Vec r) (W₂ : Mat r c) (b₂ : Vec c)
    (x dy : Vec (N * (c * h * w))) : Vec (N * (c * h * w)) :=
  (seB_has_vjp N (h := h) (w := w) W₁ b₁ W₂ b₂).backward x dy

/-- Batched **SE gate cotangent** `dgate[n,c] = Σ_{h,w}(x⊙dy)` (= `den seReduceB`, the broadcast-adjoint
    of `x ⊙ dy` — the FIRST step of the SE gate backward, feeding the SE dense param grads). -/
noncomputable def gateCotB (N c h w : Nat) (x dy : Vec (N * (c * h * w))) : Vec (N * c) :=
  fun idx => ∑ q : Fin (c * h * w),
    if flatChannel c h w q = (finProdFinEquiv.symm idx).2 then
      batchSlice N (c * h * w) x (finProdFinEquiv.symm idx).1 q
        * batchSlice N (c * h * w) dy (finProdFinEquiv.symm idx).1 q
    else 0

/-! ## Residual stride-1 MBConv block — all 16 params tied (expand → dw → SE → project + skip)

The centerpiece: exercises the genuinely-new content vs mnv2 — swish masks (smooth), the SE gate
fan-in (`gateCotB → sigBackB → {zW₂,zb₂} → rowDenseBackFlat → swBackB → {zW₁,zb₁}`), and true
batch-norm backward (`bnBackB`), all at the batched index `N·(c·h·w)`. Backward from `dyOut` (cot at
project-BN out): project-BN-back → project-conv-back (cot at SE out) → SE backward (fused `dx` for the
depthwise side; un-fused gate-cot for the SE params) → depthwise swish/BN/conv backs → expand
swish/BN/conv backs. Residual (`ic=oc=c`): the block-input cotangent fan-in `+ dyOut` lives in the
whole-net thread, not here (the param ops are skip-agnostic — identical to the no-skip widenings). -/

/-- **Residual stride-1 MBConv block, tied.** All 16 params (expand/project 1×1 conv W+b, depthwise
    W+b, SE reduce/excite dense W₁/b₁/W₂/b₂, three true-BN γ/β) denote the certified batched Σ_n
    loss-descent step at the real block forward activations + the chain cotangents driven by `dyOut`. -/
def enetExpTied {N ic mid oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
  -- forward activations
  let ec : Vec (N * (mid * h * w)) := batchMap N (flatConv We be) xin
  let en : Vec (N * (mid * h * w)) := bnBatchLA N mid h w εe γe βe ec
  let er : Vec (N * (mid * h * w)) := swish (N * (mid * h * w)) en
  let dc : Vec (N * (mid * h * w)) := batchMap N (depthwiseFlat Wd bd) er
  let dn : Vec (N * (mid * h * w)) := bnBatchLA N mid h w εd γd βd dc
  let dr : Vec (N * (mid * h * w)) := swish (N * (mid * h * w)) dn
  let s  : Vec (N * mid) := batchMap N (globalAvgPoolFlat mid h w) dr
  let e1 : Vec (N * r) := batchMap N (dense Wz1 bz1) s
  let z  : Vec (N * r) := swish (N * r) e1
  let e2 : Vec (N * mid) := batchMap N (dense Wz2 bz2) z
  let se : Vec (N * (mid * h * w)) := seB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr
  let pc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wp bp) se
  -- backward chain cotangents (composed from dyOut)
  let cotPbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εp hεp γp βp pc dyOut
  let cotSeOut : Vec (N * (mid * h * w)) := cInB N Wp bp cotPbn
  let dgate : Vec (N * mid) := gateCotB N mid h w dr cotSeOut
  let cotE2 : Vec (N * mid) := sigBackB (N * mid) e2 dgate
  let cotZ : Vec (N * r) := rowDenseBackFlat N r mid Wz2 cotE2
  let cotE1 : Vec (N * r) := swBackB (N * r) e1 cotZ
  let cotDxSe : Vec (N * (mid * h * w)) := seInB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr cotSeOut
  let cotDn : Vec (N * (mid * h * w)) := swBackB (N * (mid * h * w)) dn cotDxSe
  let cotDc : Vec (N * (mid * h * w)) := bnBackB N mid h w εd hεd γd βd dc cotDn
  let cotEr : Vec (N * (mid * h * w)) := dInB N Wd bd cotDc
  let cotEn : Vec (N * (mid * h * w)) := swBackB (N * (mid * h * w)) en cotEr
  let cotEc : Vec (N * (mid * h * w)) := bnBackB N mid h w εe hεe γe βe ec cotEn
  -- expand 1×1 conv (c → mid), cot = cotEc
  (∀ idx : Fin (mid * ic * 1 * 1),
        den (SHlo.convWeightSgdB xN wN lrStr be xin We lr (.operand cotN cotEc)) idx
          = Kernel4.flatten We idx - lr * ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') be
                        (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                   (Kernel4.flatten We) idx j * batchSlice N (mid * h * w) cotEc n j)
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr be lr (.operand cotN (reassocB N mid h w cotEc))) o
          = be o - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εe (fun _ => 0) β' (fun _ => 0))
                   be o j * bnchwFwd N mid h w (reassocB N mid h w cotEc) j)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εe γe (reassocB N mid h w ec) lr
              (.operand cotN (reassocB N mid h w cotEn))) idx
          = γe idx - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun γ' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εe γ' βe
                      (bnchwFwd N mid h w (reassocB N mid h w ec)))
                   γe idx j * bnchwFwd N mid h w (reassocB N mid h w cotEn) j)
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr βe lr (.operand cotN (reassocB N mid h w cotEn))) o
          = βe o - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εe (fun _ => 0) β' (fun _ => 0))
                   βe o j * bnchwFwd N mid h w (reassocB N mid h w cotEn) j)
  -- depthwise (stride-1, kHd×kWd), cot = cotDc
  ∧ (∀ idx : Fin (mid * kHd * kWd),
        den (SHlo.depthwiseWeightSgdB xN wN lrStr bd er Wd lr (.operand cotN cotDc)) idx
          = Tensor3.flatten Wd idx - lr * ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * kHd * kWd) =>
                      Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') bd
                        (Tensor3.unflatten (batchSlice N (mid * h * w) er n))))
                   (Tensor3.flatten Wd) idx j * batchSlice N (mid * h * w) cotDc n j)
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr bd lr (.operand cotN (reassocB N mid h w cotDc))) o
          = bd o - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εd (fun _ => 0) β' (fun _ => 0))
                   bd o j * bnchwFwd N mid h w (reassocB N mid h w cotDc) j)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εd γd (reassocB N mid h w dc) lr
              (.operand cotN (reassocB N mid h w cotDn))) idx
          = γd idx - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun γ' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εd γ' βd
                      (bnchwFwd N mid h w (reassocB N mid h w dc)))
                   γd idx j * bnchwFwd N mid h w (reassocB N mid h w cotDn) j)
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr βd lr (.operand cotN (reassocB N mid h w cotDn))) o
          = βd o - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εd (fun _ => 0) β' (fun _ => 0))
                   βd o j * bnchwFwd N mid h w (reassocB N mid h w cotDn) j)
  -- SE reduce dense W₁/b₁ (mid → r), cot = cotE1; excite dense W₂/b₂ (r → mid), cot = cotE2
  ∧ (∀ i : Fin mid, ∀ j : Fin r,
        den (SHlo.denseWeightSgdB xN wN lrStr s Wz1 lr (.operand cotN cotE1)) (finProdFinEquiv (i, j))
          = Wz1 i j - lr * ∑ n : Fin N, ∑ k : Fin r,
              pdiv (fun v : Vec (mid * r) => dense (Mat.unflatten v) bz1 (batchSlice N mid s n))
                   (Mat.flatten Wz1) (finProdFinEquiv (i, j)) k * batchSlice N r cotE1 n k)
  ∧ (∀ j : Fin r,
        den (SHlo.denseBiasSgdB bN lrStr bz1 lr (.operand cotN cotE1)) j
          = bz1 j - lr * ∑ n : Fin N, ∑ k : Fin r,
              pdiv (fun b' : Vec r => dense (0 : Mat r r) b' (0 : Vec r))
                   bz1 j k * batchSlice N r cotE1 n k)
  ∧ (∀ i : Fin r, ∀ j : Fin mid,
        den (SHlo.denseWeightSgdB xN wN lrStr z Wz2 lr (.operand cotN cotE2)) (finProdFinEquiv (i, j))
          = Wz2 i j - lr * ∑ n : Fin N, ∑ k : Fin mid,
              pdiv (fun v : Vec (r * mid) => dense (Mat.unflatten v) bz2 (batchSlice N r z n))
                   (Mat.flatten Wz2) (finProdFinEquiv (i, j)) k * batchSlice N mid cotE2 n k)
  ∧ (∀ j : Fin mid,
        den (SHlo.denseBiasSgdB bN lrStr bz2 lr (.operand cotN cotE2)) j
          = bz2 j - lr * ∑ n : Fin N, ∑ k : Fin mid,
              pdiv (fun b' : Vec mid => dense (0 : Mat mid mid) b' (0 : Vec mid))
                   bz2 j k * batchSlice N mid cotE2 n k)
  -- project 1×1 conv (mid → oc), cot = cotPbn
  ∧ (∀ idx : Fin (oc * mid * 1 * 1),
        den (SHlo.convWeightSgdB xN wN lrStr bp se Wp lr (.operand cotN cotPbn)) idx
          = Kernel4.flatten Wp idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') bp
                        (Tensor3.unflatten (batchSlice N (mid * h * w) se n))))
                   (Kernel4.flatten Wp) idx j * batchSlice N (oc * h * w) cotPbn n j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bp lr (.operand cotN (reassocB N oc h w cotPbn))) o
          = bp o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp (fun _ => 0) β' (fun _ => 0))
                   bp o j * bnchwFwd N oc h w (reassocB N oc h w cotPbn) j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εp γp (reassocB N oc h w pc) lr
              (.operand cotN (reassocB N oc h w dyOut))) idx
          = γp idx - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp γ' βp
                      (bnchwFwd N oc h w (reassocB N oc h w pc)))
                   γp idx j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr βp lr (.operand cotN (reassocB N oc h w dyOut))) o
          = βp o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp (fun _ => 0) β' (fun _ => 0))
                   βp o j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)

theorem enet_exp_tied {N ic mid oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetExpTied xN wN bN gN vN epsStr lrStr cotN εe hεe εd hεd εp hεp
      We be γe βe Wd bd γd βd Wz1 bz1 Wz2 bz2 Wp bp γp βp xin dyOut lr := by
  unfold enetExpTied
  intro ec en er dc dn dr s e1 z e2 se pc cotPbn cotSeOut dgate cotE2 cotZ cotE1
        cotDxSe cotDn cotDc cotEr cotEn cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN be xin We cotEc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εe (fun _ => 0) be (fun _ => 0) (reassocB N mid h w cotEc) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εe γe βe (reassocB N mid h w ec) (reassocB N mid h w cotEn) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εe (fun _ => 0) βe (fun _ => 0) (reassocB N mid h w cotEn) lr o
  · intro idx; exact EnetPoC.depthwiseWB_den xN wN lrStr cotN bd er Wd cotDc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εd (fun _ => 0) bd (fun _ => 0) (reassocB N mid h w cotDc) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εd γd βd (reassocB N mid h w dc) (reassocB N mid h w cotDn) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εd (fun _ => 0) βd (fun _ => 0) (reassocB N mid h w cotDn) lr o
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN s Wz1 bz1 cotE1 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 lr j
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN z Wz2 bz2 cotE2 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2 lr j
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN bp se Wp cotPbn lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εp (fun _ => 0) bp (fun _ => 0) (reassocB N oc h w cotPbn) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εp γp βp (reassocB N oc h w pc) (reassocB N oc h w dyOut) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εp (fun _ => 0) βp (fun _ => 0) (reassocB N oc h w dyOut) lr o

/-! ## Strided downsampling MBConv block — all 16 params tied (b2/b4/b6/b12)

Same as the expand block EXCEPT the expand stage lives at the block-input grid `2h×2w` and the
depthwise is strided (`depthwiseStridedWeightSgdB`, the expand-side cotangent `cotEr` upsamples `h→2h`
via `dStridedInB`). No skip (spatial+channels change). -/

/-- **Strided downsampling MBConv block, tied.** All 16 params at the real forward (expand at `2h×2w`,
    strided depthwise `2h→h`) + the chain cotangents driven by `dyOut`. -/
def enetStridedTied {N ic mid oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
  -- forward activations (expand at 2h×2w, depthwise downsamples to h×w)
  let ec : Vec (N * (mid * (2 * h) * (2 * w))) := batchMap N (flatConv We be) xin
  let en : Vec (N * (mid * (2 * h) * (2 * w))) := bnBatchLA N mid (2 * h) (2 * w) εe γe βe ec
  let er : Vec (N * (mid * (2 * h) * (2 * w))) := swish (N * (mid * (2 * h) * (2 * w))) en
  let dc : Vec (N * (mid * h * w)) := batchMap N (depthwiseStride2Flat Wd bd) er
  let dn : Vec (N * (mid * h * w)) := bnBatchLA N mid h w εd γd βd dc
  let dr : Vec (N * (mid * h * w)) := swish (N * (mid * h * w)) dn
  let s  : Vec (N * mid) := batchMap N (globalAvgPoolFlat mid h w) dr
  let e1 : Vec (N * r) := batchMap N (dense Wz1 bz1) s
  let z  : Vec (N * r) := swish (N * r) e1
  let e2 : Vec (N * mid) := batchMap N (dense Wz2 bz2) z
  let se : Vec (N * (mid * h * w)) := seB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr
  let pc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wp bp) se
  -- backward chain cotangents
  let cotPbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εp hεp γp βp pc dyOut
  let cotSeOut : Vec (N * (mid * h * w)) := cInB N Wp bp cotPbn
  let dgate : Vec (N * mid) := gateCotB N mid h w dr cotSeOut
  let cotE2 : Vec (N * mid) := sigBackB (N * mid) e2 dgate
  let cotZ : Vec (N * r) := rowDenseBackFlat N r mid Wz2 cotE2
  let cotE1 : Vec (N * r) := swBackB (N * r) e1 cotZ
  let cotDxSe : Vec (N * (mid * h * w)) := seInB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr cotSeOut
  let cotDn : Vec (N * (mid * h * w)) := swBackB (N * (mid * h * w)) dn cotDxSe
  let cotDc : Vec (N * (mid * h * w)) := bnBackB N mid h w εd hεd γd βd dc cotDn
  let cotEr : Vec (N * (mid * (2 * h) * (2 * w))) := dStridedInB N Wd bd cotDc
  let cotEn : Vec (N * (mid * (2 * h) * (2 * w))) := swBackB (N * (mid * (2 * h) * (2 * w))) en cotEr
  let cotEc : Vec (N * (mid * (2 * h) * (2 * w))) := bnBackB N mid (2 * h) (2 * w) εe hεe γe βe ec cotEn
  -- expand 1×1 conv (ic → mid, at 2h×2w), cot = cotEc
  (∀ idx : Fin (mid * ic * 1 * 1),
        den (SHlo.convWeightSgdB xN wN lrStr be xin We lr (.operand cotN cotEc)) idx
          = Kernel4.flatten We idx - lr * ∑ n : Fin N, ∑ j : Fin (mid * (2 * h) * (2 * w)),
              pdiv (fun v' : Vec (mid * ic * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') be
                        (Tensor3.unflatten (batchSlice N (ic * (2 * h) * (2 * w)) xin n))))
                   (Kernel4.flatten We) idx j * batchSlice N (mid * (2 * h) * (2 * w)) cotEc n j)
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr be lr (.operand cotN (reassocB N mid (2 * h) (2 * w) cotEc))) o
          = be o - lr * ∑ j : Fin (mid * (N * ((2 * h) * (2 * w)))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * ((2 * h) * (2 * w))) εe (fun _ => 0) β' (fun _ => 0))
                   be o j * bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) cotEc) j)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εe γe (reassocB N mid (2 * h) (2 * w) ec) lr
              (.operand cotN (reassocB N mid (2 * h) (2 * w) cotEn))) idx
          = γe idx - lr * ∑ j : Fin (mid * (N * ((2 * h) * (2 * w)))),
              pdiv (fun γ' : Vec mid => bnPerChannelFlat mid (N * ((2 * h) * (2 * w))) εe γ' βe
                      (bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) ec)))
                   γe idx j * bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) cotEn) j)
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr βe lr (.operand cotN (reassocB N mid (2 * h) (2 * w) cotEn))) o
          = βe o - lr * ∑ j : Fin (mid * (N * ((2 * h) * (2 * w)))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * ((2 * h) * (2 * w))) εe (fun _ => 0) β' (fun _ => 0))
                   βe o j * bnchwFwd N mid (2 * h) (2 * w) (reassocB N mid (2 * h) (2 * w) cotEn) j)
  -- strided depthwise (kHd×kWd, 2h→h), cot = cotDc
  ∧ (∀ idx : Fin (mid * kHd * kWd),
        den (SHlo.depthwiseStridedWeightSgdB xN wN lrStr bd er Wd lr (.operand cotN cotDc)) idx
          = Tensor3.flatten Wd idx - lr * ∑ n : Fin N, ∑ j : Fin (mid * h * w),
              pdiv (fun v' : Vec (mid * kHd * kWd) =>
                      depthwiseStride2Flat (Tensor3.unflatten v') bd (batchSlice N (mid * (2 * h) * (2 * w)) er n))
                   (Tensor3.flatten Wd) idx j * batchSlice N (mid * h * w) cotDc n j)
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr bd lr (.operand cotN (reassocB N mid h w cotDc))) o
          = bd o - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εd (fun _ => 0) β' (fun _ => 0))
                   bd o j * bnchwFwd N mid h w (reassocB N mid h w cotDc) j)
  ∧ (∀ idx : Fin mid,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εd γd (reassocB N mid h w dc) lr
              (.operand cotN (reassocB N mid h w cotDn))) idx
          = γd idx - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun γ' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εd γ' βd
                      (bnchwFwd N mid h w (reassocB N mid h w dc)))
                   γd idx j * bnchwFwd N mid h w (reassocB N mid h w cotDn) j)
  ∧ (∀ o : Fin mid,
        den (SHlo.bnBetaSgdB bN lrStr βd lr (.operand cotN (reassocB N mid h w cotDn))) o
          = βd o - lr * ∑ j : Fin (mid * (N * (h * w))),
              pdiv (fun β' : Vec mid => bnPerChannelFlat mid (N * (h * w)) εd (fun _ => 0) β' (fun _ => 0))
                   βd o j * bnchwFwd N mid h w (reassocB N mid h w cotDn) j)
  -- SE reduce/excite dense (mid → r → mid)
  ∧ (∀ i : Fin mid, ∀ j : Fin r,
        den (SHlo.denseWeightSgdB xN wN lrStr s Wz1 lr (.operand cotN cotE1)) (finProdFinEquiv (i, j))
          = Wz1 i j - lr * ∑ n : Fin N, ∑ k : Fin r,
              pdiv (fun v : Vec (mid * r) => dense (Mat.unflatten v) bz1 (batchSlice N mid s n))
                   (Mat.flatten Wz1) (finProdFinEquiv (i, j)) k * batchSlice N r cotE1 n k)
  ∧ (∀ j : Fin r,
        den (SHlo.denseBiasSgdB bN lrStr bz1 lr (.operand cotN cotE1)) j
          = bz1 j - lr * ∑ n : Fin N, ∑ k : Fin r,
              pdiv (fun b' : Vec r => dense (0 : Mat r r) b' (0 : Vec r))
                   bz1 j k * batchSlice N r cotE1 n k)
  ∧ (∀ i : Fin r, ∀ j : Fin mid,
        den (SHlo.denseWeightSgdB xN wN lrStr z Wz2 lr (.operand cotN cotE2)) (finProdFinEquiv (i, j))
          = Wz2 i j - lr * ∑ n : Fin N, ∑ k : Fin mid,
              pdiv (fun v : Vec (r * mid) => dense (Mat.unflatten v) bz2 (batchSlice N r z n))
                   (Mat.flatten Wz2) (finProdFinEquiv (i, j)) k * batchSlice N mid cotE2 n k)
  ∧ (∀ j : Fin mid,
        den (SHlo.denseBiasSgdB bN lrStr bz2 lr (.operand cotN cotE2)) j
          = bz2 j - lr * ∑ n : Fin N, ∑ k : Fin mid,
              pdiv (fun b' : Vec mid => dense (0 : Mat mid mid) b' (0 : Vec mid))
                   bz2 j k * batchSlice N mid cotE2 n k)
  -- project 1×1 conv (mid → oc), cot = cotPbn
  ∧ (∀ idx : Fin (oc * mid * 1 * 1),
        den (SHlo.convWeightSgdB xN wN lrStr bp se Wp lr (.operand cotN cotPbn)) idx
          = Kernel4.flatten Wp idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * mid * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') bp
                        (Tensor3.unflatten (batchSlice N (mid * h * w) se n))))
                   (Kernel4.flatten Wp) idx j * batchSlice N (oc * h * w) cotPbn n j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bp lr (.operand cotN (reassocB N oc h w cotPbn))) o
          = bp o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp (fun _ => 0) β' (fun _ => 0))
                   bp o j * bnchwFwd N oc h w (reassocB N oc h w cotPbn) j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εp γp (reassocB N oc h w pc) lr
              (.operand cotN (reassocB N oc h w dyOut))) idx
          = γp idx - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp γ' βp
                      (bnchwFwd N oc h w (reassocB N oc h w pc)))
                   γp idx j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr βp lr (.operand cotN (reassocB N oc h w dyOut))) o
          = βp o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp (fun _ => 0) β' (fun _ => 0))
                   βp o j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)

theorem enet_strided_tied {N ic mid oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εe : ℝ) (hεe : 0 < εe) (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (We : Kernel4 mid ic 1 1) (be γe βe : Vec mid)
    (Wd : DepthwiseKernel mid kHd kWd) (bd γd βd : Vec mid)
    (Wz1 : Mat mid r) (bz1 : Vec r) (Wz2 : Mat r mid) (bz2 : Vec mid)
    (Wp : Kernel4 oc mid 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * (2 * h) * (2 * w)))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetStridedTied xN wN bN gN vN epsStr lrStr cotN εe hεe εd hεd εp hεp
      We be γe βe Wd bd γd βd Wz1 bz1 Wz2 bz2 Wp bp γp βp xin dyOut lr := by
  unfold enetStridedTied
  intro ec en er dc dn dr s e1 z e2 se pc cotPbn cotSeOut dgate cotE2 cotZ cotE1
        cotDxSe cotDn cotDc cotEr cotEn cotEc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN be xin We cotEc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εe (fun _ => 0) be (fun _ => 0) (reassocB N mid (2 * h) (2 * w) cotEc) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εe γe βe (reassocB N mid (2 * h) (2 * w) ec) (reassocB N mid (2 * h) (2 * w) cotEn) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εe (fun _ => 0) βe (fun _ => 0) (reassocB N mid (2 * h) (2 * w) cotEn) lr o
  · intro idx; exact EnetPoC.depthwiseStridedWB_den xN wN lrStr cotN bd er Wd cotDc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εd (fun _ => 0) bd (fun _ => 0) (reassocB N mid h w cotDc) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εd γd βd (reassocB N mid h w dc) (reassocB N mid h w cotDn) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εd (fun _ => 0) βd (fun _ => 0) (reassocB N mid h w cotDn) lr o
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN s Wz1 bz1 cotE1 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 lr j
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN z Wz2 bz2 cotE2 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat mid mid) (0 : Vec mid) bz2 cotE2 lr j
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN bp se Wp cotPbn lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εp (fun _ => 0) bp (fun _ => 0) (reassocB N oc h w cotPbn) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εp γp βp (reassocB N oc h w pc) (reassocB N oc h w dyOut) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εp (fun _ => 0) βp (fun _ => 0) (reassocB N oc h w dyOut) lr o

/-! ## No-expand MBConv block (b1, t=1) — all 12 params tied (depthwise on `ic` → SE → project)

NO expand conv: the depthwise runs directly on the block input (`ic` channels). 12 params (4 depthwise+BN,
4 SE, 4 project). The SE squeeze/excite is on `ic` channels (`ic → r → ic`). -/

/-- **No-expand MBConv block, tied.** All 12 params at the real forward + chain cotangents. -/
def enetNoExpTied {N ic oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (Wd : DepthwiseKernel ic kHd kWd) (bd γd βd : Vec ic)
    (Wz1 : Mat ic r) (bz1 : Vec r) (Wz2 : Mat r ic) (bz2 : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
  -- forward activations (depthwise on the block input ic, no expand)
  let dc : Vec (N * (ic * h * w)) := batchMap N (depthwiseFlat Wd bd) xin
  let dn : Vec (N * (ic * h * w)) := bnBatchLA N ic h w εd γd βd dc
  let dr : Vec (N * (ic * h * w)) := swish (N * (ic * h * w)) dn
  let s  : Vec (N * ic) := batchMap N (globalAvgPoolFlat ic h w) dr
  let e1 : Vec (N * r) := batchMap N (dense Wz1 bz1) s
  let z  : Vec (N * r) := swish (N * r) e1
  let e2 : Vec (N * ic) := batchMap N (dense Wz2 bz2) z
  let se : Vec (N * (ic * h * w)) := seB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr
  let pc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wp bp) se
  -- backward chain cotangents
  let cotPbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εp hεp γp βp pc dyOut
  let cotSeOut : Vec (N * (ic * h * w)) := cInB N Wp bp cotPbn
  let dgate : Vec (N * ic) := gateCotB N ic h w dr cotSeOut
  let cotE2 : Vec (N * ic) := sigBackB (N * ic) e2 dgate
  let cotZ : Vec (N * r) := rowDenseBackFlat N r ic Wz2 cotE2
  let cotE1 : Vec (N * r) := swBackB (N * r) e1 cotZ
  let cotDxSe : Vec (N * (ic * h * w)) := seInB N (h := h) (w := w) Wz1 bz1 Wz2 bz2 dr cotSeOut
  let cotDn : Vec (N * (ic * h * w)) := swBackB (N * (ic * h * w)) dn cotDxSe
  let cotDc : Vec (N * (ic * h * w)) := bnBackB N ic h w εd hεd γd βd dc cotDn
  -- depthwise (stride-1, kHd×kWd, on ic), cot = cotDc
  (∀ idx : Fin (ic * kHd * kWd),
        den (SHlo.depthwiseWeightSgdB xN wN lrStr bd xin Wd lr (.operand cotN cotDc)) idx
          = Tensor3.flatten Wd idx - lr * ∑ n : Fin N, ∑ j : Fin (ic * h * w),
              pdiv (fun v' : Vec (ic * kHd * kWd) =>
                      Tensor3.flatten (depthwiseConv2d (Tensor3.unflatten v') bd
                        (Tensor3.unflatten (batchSlice N (ic * h * w) xin n))))
                   (Tensor3.flatten Wd) idx j * batchSlice N (ic * h * w) cotDc n j)
  ∧ (∀ o : Fin ic,
        den (SHlo.bnBetaSgdB bN lrStr bd lr (.operand cotN (reassocB N ic h w cotDc))) o
          = bd o - lr * ∑ j : Fin (ic * (N * (h * w))),
              pdiv (fun β' : Vec ic => bnPerChannelFlat ic (N * (h * w)) εd (fun _ => 0) β' (fun _ => 0))
                   bd o j * bnchwFwd N ic h w (reassocB N ic h w cotDc) j)
  ∧ (∀ idx : Fin ic,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εd γd (reassocB N ic h w dc) lr
              (.operand cotN (reassocB N ic h w cotDn))) idx
          = γd idx - lr * ∑ j : Fin (ic * (N * (h * w))),
              pdiv (fun γ' : Vec ic => bnPerChannelFlat ic (N * (h * w)) εd γ' βd
                      (bnchwFwd N ic h w (reassocB N ic h w dc)))
                   γd idx j * bnchwFwd N ic h w (reassocB N ic h w cotDn) j)
  ∧ (∀ o : Fin ic,
        den (SHlo.bnBetaSgdB bN lrStr βd lr (.operand cotN (reassocB N ic h w cotDn))) o
          = βd o - lr * ∑ j : Fin (ic * (N * (h * w))),
              pdiv (fun β' : Vec ic => bnPerChannelFlat ic (N * (h * w)) εd (fun _ => 0) β' (fun _ => 0))
                   βd o j * bnchwFwd N ic h w (reassocB N ic h w cotDn) j)
  -- SE reduce/excite dense (ic → r → ic)
  ∧ (∀ i : Fin ic, ∀ j : Fin r,
        den (SHlo.denseWeightSgdB xN wN lrStr s Wz1 lr (.operand cotN cotE1)) (finProdFinEquiv (i, j))
          = Wz1 i j - lr * ∑ n : Fin N, ∑ k : Fin r,
              pdiv (fun v : Vec (ic * r) => dense (Mat.unflatten v) bz1 (batchSlice N ic s n))
                   (Mat.flatten Wz1) (finProdFinEquiv (i, j)) k * batchSlice N r cotE1 n k)
  ∧ (∀ j : Fin r,
        den (SHlo.denseBiasSgdB bN lrStr bz1 lr (.operand cotN cotE1)) j
          = bz1 j - lr * ∑ n : Fin N, ∑ k : Fin r,
              pdiv (fun b' : Vec r => dense (0 : Mat r r) b' (0 : Vec r))
                   bz1 j k * batchSlice N r cotE1 n k)
  ∧ (∀ i : Fin r, ∀ j : Fin ic,
        den (SHlo.denseWeightSgdB xN wN lrStr z Wz2 lr (.operand cotN cotE2)) (finProdFinEquiv (i, j))
          = Wz2 i j - lr * ∑ n : Fin N, ∑ k : Fin ic,
              pdiv (fun v : Vec (r * ic) => dense (Mat.unflatten v) bz2 (batchSlice N r z n))
                   (Mat.flatten Wz2) (finProdFinEquiv (i, j)) k * batchSlice N ic cotE2 n k)
  ∧ (∀ j : Fin ic,
        den (SHlo.denseBiasSgdB bN lrStr bz2 lr (.operand cotN cotE2)) j
          = bz2 j - lr * ∑ n : Fin N, ∑ k : Fin ic,
              pdiv (fun b' : Vec ic => dense (0 : Mat ic ic) b' (0 : Vec ic))
                   bz2 j k * batchSlice N ic cotE2 n k)
  -- project 1×1 conv (ic → oc), cot = cotPbn
  ∧ (∀ idx : Fin (oc * ic * 1 * 1),
        den (SHlo.convWeightSgdB xN wN lrStr bp se Wp lr (.operand cotN cotPbn)) idx
          = Kernel4.flatten Wp idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * ic * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') bp
                        (Tensor3.unflatten (batchSlice N (ic * h * w) se n))))
                   (Kernel4.flatten Wp) idx j * batchSlice N (oc * h * w) cotPbn n j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bp lr (.operand cotN (reassocB N oc h w cotPbn))) o
          = bp o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp (fun _ => 0) β' (fun _ => 0))
                   bp o j * bnchwFwd N oc h w (reassocB N oc h w cotPbn) j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εp γp (reassocB N oc h w pc) lr
              (.operand cotN (reassocB N oc h w dyOut))) idx
          = γp idx - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp γ' βp
                      (bnchwFwd N oc h w (reassocB N oc h w pc)))
                   γp idx j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr βp lr (.operand cotN (reassocB N oc h w dyOut))) o
          = βp o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εp (fun _ => 0) β' (fun _ => 0))
                   βp o j * bnchwFwd N oc h w (reassocB N oc h w dyOut) j)

theorem enet_noexp_tied {N ic oc h w r kHd kWd : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String)
    (εd : ℝ) (hεd : 0 < εd) (εp : ℝ) (hεp : 0 < εp)
    (Wd : DepthwiseKernel ic kHd kWd) (bd γd βd : Vec ic)
    (Wz1 : Mat ic r) (bz1 : Vec r) (Wz2 : Mat r ic) (bz2 : Vec ic)
    (Wp : Kernel4 oc ic 1 1) (bp γp βp : Vec oc)
    (xin : Vec (N * (ic * h * w))) (dyOut : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetNoExpTied xN wN bN gN vN epsStr lrStr cotN εd hεd εp hεp
      Wd bd γd βd Wz1 bz1 Wz2 bz2 Wp bp γp βp xin dyOut lr := by
  unfold enetNoExpTied
  intro dc dn dr s e1 z e2 se pc cotPbn cotSeOut dgate cotE2 cotZ cotE1 cotDxSe cotDn cotDc
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.depthwiseWB_den xN wN lrStr cotN bd xin Wd cotDc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εd (fun _ => 0) bd (fun _ => 0) (reassocB N ic h w cotDc) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εd γd βd (reassocB N ic h w dc) (reassocB N ic h w cotDn) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εd (fun _ => 0) βd (fun _ => 0) (reassocB N ic h w cotDn) lr o
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN s Wz1 bz1 cotE1 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat r r) (0 : Vec r) bz1 cotE1 lr j
  · intro i j; exact EnetPoC.denseWB_den xN wN lrStr cotN z Wz2 bz2 cotE2 lr i j
  · intro j;   exact EnetPoC.denseBB_den bN lrStr cotN (0 : Mat ic ic) (0 : Vec ic) bz2 cotE2 lr j
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN bp se Wp cotPbn lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εp (fun _ => 0) bp (fun _ => 0) (reassocB N oc h w cotPbn) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εp γp βp (reassocB N oc h w pc) (reassocB N oc h w dyOut) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εp (fun _ => 0) βp (fun _ => 0) (reassocB N oc h w dyOut) lr o

/-! ## Stem — the 3×3/s2 conv-bn-swish (4 params), feeding block 1

`swish(bn(convStride2 Ws bs x))`, 3→32 at 224→112. The cotangent block 1 delivers at the stem swish
output (`dyStem`) lifts through swish-back + true-BN-back to the conv-out cotangent (the
`convStridedWeightSgdB` consumes it; NO conv-back past `%x`). 4 params. -/

/-- **Stem, tied.** The 3×3/s2 conv (`Ws`/`bs`) + its true-BN (`γs`/`βs`) at the real stem forward +
    the cotangent through the stem swish (no maxpool, no conv-back). -/
def enetStemTied {N ic oc h w kHs kWs : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (εs : ℝ) (hεs : 0 < εs)
    (Ws : Kernel4 oc ic kHs kWs) (bs γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) (lr : ℝ) : Prop :=
  let stc : Vec (N * (oc * h * w)) := batchMap N (flatConvStride2 Ws bs) x
  let stn : Vec (N * (oc * h * w)) := bnBatchLA N oc h w εs γs βs stc
  let cotBnS : Vec (N * (oc * h * w)) := swBackB (N * (oc * h * w)) stn dyStem
  let cotStc : Vec (N * (oc * h * w)) := bnBackB N oc h w εs hεs γs βs stc cotBnS
  (∀ idx : Fin (oc * ic * kHs * kWs),
        den (SHlo.convStridedWeightSgdB xN wN lrStr bs x Ws lr (.operand cotN cotStc)) idx
          = Kernel4.flatten Ws idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * ic * kHs * kWs) =>
                      flatConvStride2 (Kernel4.unflatten v') bs (batchSlice N (ic * (2 * h) * (2 * w)) x n))
                   (Kernel4.flatten Ws) idx j * batchSlice N (oc * h * w) cotStc n j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bs lr (.operand cotN (reassocB N oc h w cotStc))) o
          = bs o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εs (fun _ => 0) β' (fun _ => 0))
                   bs o j * bnchwFwd N oc h w (reassocB N oc h w cotStc) j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εs γs (reassocB N oc h w stc) lr
              (.operand cotN (reassocB N oc h w cotBnS))) idx
          = γs idx - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εs γ' βs
                      (bnchwFwd N oc h w (reassocB N oc h w stc)))
                   γs idx j * bnchwFwd N oc h w (reassocB N oc h w cotBnS) j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr βs lr (.operand cotN (reassocB N oc h w cotBnS))) o
          = βs o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εs (fun _ => 0) β' (fun _ => 0))
                   βs o j * bnchwFwd N oc h w (reassocB N oc h w cotBnS) j)

theorem enet_stem_tied {N ic oc h w kHs kWs : Nat}
    (xN wN bN gN vN epsStr lrStr cotN : String) (εs : ℝ) (hεs : 0 < εs)
    (Ws : Kernel4 oc ic kHs kWs) (bs γs βs : Vec oc)
    (x : Vec (N * (ic * (2 * h) * (2 * w)))) (dyStem : Vec (N * (oc * h * w))) (lr : ℝ) :
    enetStemTied xN wN bN gN vN epsStr lrStr cotN εs hεs Ws bs γs βs x dyStem lr := by
  unfold enetStemTied
  intro stc stn cotBnS cotStc
  refine ⟨?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.convStridedWB_den xN wN lrStr cotN bs x Ws cotStc lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εs (fun _ => 0) bs (fun _ => 0) (reassocB N oc h w cotStc) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εs γs βs (reassocB N oc h w stc) (reassocB N oc h w cotBnS) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εs (fun _ => 0) βs (fun _ => 0) (reassocB N oc h w cotBnS) lr o

/-! ## Head — the 1×1 conv-bn-swish (4 params) → GAP → dense (Wfc/bfc), + the loss cotangent

`dense(GAP(swish(bn(conv Wh bh)))))` (320→1280 conv, GAP, 1280→nClasses dense), then the batched
per-row softmax-CE gradient `g = rowSoftmax(logits) − onehot`. The head conv params tie at the chain
cotangent (loss → dense-back → GAP-back → swish/BN-back); the dense Wfc/bfc tie at the loss cotangent
`g` directly (the `efficientnetLossCot_den` graph denotes `g`). -/

/-- **Head, tied.** The 4 head conv-bn params + the 2 dense params (Wfc/bfc) denote the certified step
    at the real head forward + the loss-driven cotangent `g = rowSoftmax(logits) − onehot`. -/
def enetHeadTied {N c oc h w nC : Nat}
    (xN wN bN gN vN epsStr lrStr cotN dN : String) (εh : ℝ) (hεh : 0 < εh)
    (Wh : Kernel4 oc c 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (xhead : Vec (N * (c * h * w))) (onehot : Vec (N * nC)) (lr : ℝ) : Prop :=
  let hc : Vec (N * (oc * h * w)) := batchMap N (flatConv Wh bh) xhead
  let hn : Vec (N * (oc * h * w)) := bnBatchLA N oc h w εh γh βh hc
  let hr : Vec (N * (oc * h * w)) := swish (N * (oc * h * w)) hn
  let a_gap : Vec (N * oc) := batchMap N (globalAvgPoolFlat oc h w) hr
  let logits : Vec (N * nC) := batchMap N (dense Wfc bfc) a_gap
  let g : Vec (N * nC) := fun idx => rowSoftmaxFlat N nC logits idx - onehot idx
  let cotGapIn : Vec (N * oc) := rowDenseBackFlat N oc nC Wfc g
  let cotHr : Vec (N * (oc * h * w)) := gapInB N oc h w cotGapIn
  let cotHsw : Vec (N * (oc * h * w)) := swBackB (N * (oc * h * w)) hn cotHr
  let cotHbn : Vec (N * (oc * h * w)) := bnBackB N oc h w εh hεh γh βh hc cotHsw
  -- head 1×1 conv (c → oc), cot = cotHbn
  (∀ idx : Fin (oc * c * 1 * 1),
        den (SHlo.convWeightSgdB xN wN lrStr bh xhead Wh lr (.operand cotN cotHbn)) idx
          = Kernel4.flatten Wh idx - lr * ∑ n : Fin N, ∑ j : Fin (oc * h * w),
              pdiv (fun v' : Vec (oc * c * 1 * 1) =>
                      Tensor3.flatten (conv2d (Kernel4.unflatten v') bh
                        (Tensor3.unflatten (batchSlice N (c * h * w) xhead n))))
                   (Kernel4.flatten Wh) idx j * batchSlice N (oc * h * w) cotHbn n j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr bh lr (.operand cotN (reassocB N oc h w cotHbn))) o
          = bh o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εh (fun _ => 0) β' (fun _ => 0))
                   bh o j * bnchwFwd N oc h w (reassocB N oc h w cotHbn) j)
  ∧ (∀ idx : Fin oc,
        den (SHlo.bnGammaSgdB gN vN epsStr lrStr εh γh (reassocB N oc h w hc) lr
              (.operand cotN (reassocB N oc h w cotHsw))) idx
          = γh idx - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun γ' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εh γ' βh
                      (bnchwFwd N oc h w (reassocB N oc h w hc)))
                   γh idx j * bnchwFwd N oc h w (reassocB N oc h w cotHsw) j)
  ∧ (∀ o : Fin oc,
        den (SHlo.bnBetaSgdB bN lrStr βh lr (.operand cotN (reassocB N oc h w cotHsw))) o
          = βh o - lr * ∑ j : Fin (oc * (N * (h * w))),
              pdiv (fun β' : Vec oc => bnPerChannelFlat oc (N * (h * w)) εh (fun _ => 0) β' (fun _ => 0))
                   βh o j * bnchwFwd N oc h w (reassocB N oc h w cotHsw) j)
  -- dense classifier (oc → nC), cot = g (the batched softmax-CE gradient)
  ∧ (∀ i : Fin oc, ∀ j : Fin nC,
        den (SHlo.denseWeightSgdB dN wN lrStr a_gap Wfc lr (.operand cotN g)) (finProdFinEquiv (i, j))
          = Wfc i j - lr * ∑ n : Fin N, ∑ k : Fin nC,
              pdiv (fun v : Vec (oc * nC) => dense (Mat.unflatten v) bfc (batchSlice N oc a_gap n))
                   (Mat.flatten Wfc) (finProdFinEquiv (i, j)) k * batchSlice N nC g n k)
  ∧ (∀ j : Fin nC,
        den (SHlo.denseBiasSgdB dN lrStr bfc lr (.operand cotN g)) j
          = bfc j - lr * ∑ n : Fin N, ∑ k : Fin nC,
              pdiv (fun b' : Vec nC => dense (0 : Mat nC nC) b' (0 : Vec nC))
                   bfc j k * batchSlice N nC g n k)

theorem enet_head_tied {N c oc h w nC : Nat}
    (xN wN bN gN vN epsStr lrStr cotN dN : String) (εh : ℝ) (hεh : 0 < εh)
    (Wh : Kernel4 oc c 1 1) (bh γh βh : Vec oc) (Wfc : Mat oc nC) (bfc : Vec nC)
    (xhead : Vec (N * (c * h * w))) (onehot : Vec (N * nC)) (lr : ℝ) :
    enetHeadTied xN wN bN gN vN epsStr lrStr cotN dN εh hεh Wh bh γh βh Wfc bfc xhead onehot lr := by
  unfold enetHeadTied
  intro hc hn hr a_gap logits g cotGapIn cotHr cotHsw cotHbn
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro idx; exact EnetPoC.convWB_den xN wN lrStr cotN bh xhead Wh cotHbn lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εh (fun _ => 0) bh (fun _ => 0) (reassocB N oc h w cotHbn) lr o
  · intro idx; exact EnetPoC.bnGammaB_den gN vN epsStr lrStr cotN εh γh βh (reassocB N oc h w hc) (reassocB N oc h w cotHsw) lr idx
  · intro o;   exact EnetPoC.bnBetaB_den bN lrStr cotN εh (fun _ => 0) βh (fun _ => 0) (reassocB N oc h w cotHsw) lr o
  · intro i j; exact EnetPoC.denseWB_den dN wN lrStr cotN a_gap Wfc bfc g lr i j
  · intro j;   exact EnetPoC.denseBB_den dN lrStr cotN (0 : Mat nC nC) (0 : Vec nC) bfc g lr j

end Proofs.EnetTiePoC
