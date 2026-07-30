import LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie
import LeanMlir.Proofs.Foundation.ResNet34TiePoC
import LeanMlir.Proofs.Foundation.ResNet34ChainClose
import LeanMlir.Proofs.Foundation.SpecVJP

/-! # §2l — the R34 downsample layer, INSTANTIATED at a 1×1 projection

The generalization check. §2l's plan says the proof side is "parameterised, so this is arguments not
proofs"; as of 2026-07-30 that was true of `rblkPStridedPC` and false of everything stated about it
— the block faithfulness theorem, the certified VJP, the chain-close param ties and the `SpecVJP`
weight record all pinned `Wp : Kernel4 oc ic 3 3`. They are now generic in `kHp kWp`.

This file is what makes that claim evidence rather than an assertion: it instantiates every
generalized declaration at **`kHp = kWp = 1`** — the paper's option-B shortcut — and at **3×3**,
the shape that ships today. If a binder were still pinned, the 1×1 line would not elaborate.

⚠ It proves *elaboration*, not accuracy: these are the same theorems, applied. What it rules out is
the failure mode that would otherwise surface halfway through re-rendering 13 artifacts — a proof
that cannot even be stated at the new kernel.

The emitted side is `tests/TestStrided1x1.lean` (the four ops, type-valid + known answer on device).
Together they are the two halves of "1×1 is arguments": the text and the proofs.

Run: `lake build LeanMlir.Proofs.Foundation.Resnet34BackCertifiedTie`
     `lake env lean tests/TestProj1x1Generic.lean`   (silent = pass; §4's rebuild trap applies)
-/

open Proofs Proofs.StableHLO Proofs.ResNet34PoC

namespace Proj1x1Check

-- A small but non-degenerate downsample block: 8→16 channels, 6×6 → 3×3.
private def IC : Nat := 8
private def OC : Nat := 16
private def HH : Nat := 3

private noncomputable def zk3  : Kernel4 OC IC 3 3  := fun _ _ _ _ => 0
private noncomputable def zk33 : Kernel4 OC OC 3 3  := fun _ _ _ _ => 0
private noncomputable def zkp1 : Kernel4 OC IC 1 1  := fun _ _ _ _ => 0   -- ▶ the paper's shortcut
private noncomputable def zkp3 : Kernel4 OC IC 3 3  := fun _ _ _ _ => 0   -- the committed shape
private noncomputable def zc   : Vec OC             := fun _ => 0
private noncomputable def zin  : Vec (IC*(2*HH)*(2*HH)) := fun _ => 0
private noncomputable def zout : Vec (OC*HH*HH)     := fun _ => 0

-- ── the ℝ forward, at both projection kernels ────────────────────────────────
noncomputable example : Vec (IC*(2*HH)*(2*HH)) → Vec (OC*HH*HH) :=
  rblkPStridedPC (h := HH) (w := HH) zk3 zc 1 zc zc zk33 zc 1 zc zc zkp1 zc 1 zc zc
noncomputable example : Vec (IC*(2*HH)*(2*HH)) → Vec (OC*HH*HH) :=
  downFwd (h := HH) (w := HH) 1 zk3 zc zc zc zk33 zc zc zc zkp1 zc zc zc

-- ── the SHlo graph + its faithfulness, at 1×1 ────────────────────────────────
example (e : SHlo (IC*(2*HH)*(2*HH))) :
    den (downBlockGraphPC "d" "1.0e-05" (h := HH) (w := HH)
          zk3 zc 1 zc zc zk33 zc 1 zc zc zkp1 zc 1 zc zc e)
      = rblkPStridedPC zk3 zc 1 zc zc zk33 zc 1 zc zc zkp1 zc 1 zc zc (den e) :=
  downBlockGraphPC_faithful "d" "1.0e-05" zk3 zc 1 zc zc zk33 zc 1 zc zc zkp1 zc 1 zc zc e

-- ── the certified VJP, at 1×1. The odd-kernel side condition the strided-conv leaf tie needs is
--    `2·((k−1)/2)+1 = k`: TRUE at 1 and 3, FALSE at 2 — which is exactly ConvNeXt's even-kernel
--    exclusion (§2f-bis) now stated instead of hidden. `by decide` discharges it at both. ──
example : 2 * ((1 - 1) / 2) + 1 = 1 := by decide
example : 2 * ((3 - 1) / 2) + 1 = 3 := by decide

noncomputable example
    (h1 : ∀ k, bnPerChannelTensor3 OC HH HH 1 zc zc (flatConvStride2 zk3 zc zin) k ≠ 0)
    (hout : ∀ k, (bnPerChannelTensor3 OC HH HH 1 zc zc ∘ flatConvStride2 zkp1 zc) zin k
      + ((bnPerChannelTensor3 OC HH HH 1 zc zc ∘ flatConv zk33 zc) ∘
          (relu (OC*HH*HH) ∘ bnPerChannelTensor3 OC HH HH 1 zc zc ∘ flatConvStride2 zk3 zc)) zin k
        ≠ 0) :
    HasVJPAt (rblkPStridedPC (h := HH) (w := HH)
      zk3 zc 1 zc zc zk33 zc 1 zc zc zkp1 zc 1 zc zc) zin :=
  rblkPStridedPC_has_vjp_at zk3 zc 1 zc zc zk33 zc 1 zc zc zkp1 zc 1 zc zc
    one_pos one_pos one_pos zin h1 hout

-- ▶ The one that carries the NEW hypotheses: the certified-VJP tie, whose proof rewrites the
--   projection's strided-conv leaf and so needs the parity fact for `kHp kWp` specifically.
--   `by decide` discharging it here IS the check — at an even kernel it does not (measured: at
--   `k = 2` `decide` reports the proposition FALSE, which is ConvNeXt's exclusion, §2f-bis).
example
    (h1 : ∀ k, bnPerChannelTensor3 OC HH HH 1 zc zc (flatConvStride2 zk3 zc zin) k ≠ 0)
    (hout : ∀ k, (bnPerChannelTensor3 OC HH HH 1 zc zc ∘ flatConvStride2 zkp1 zc) zin k
      + ((bnPerChannelTensor3 OC HH HH 1 zc zc ∘ flatConv zk33 zc) ∘
          (relu (OC*HH*HH) ∘ bnPerChannelTensor3 OC HH HH 1 zc zc ∘ flatConvStride2 zk3 zc)) zin k
        ≠ 0) : True := by
  have := r34DownBlockBack_eq_rblkPStridedPC_vjp (by decide) (by decide)
    zk3 zc 1 zc zc zk33 zc 1 zc zc zkp1 zc 1 zc zc one_pos one_pos one_pos zin h1 hout
  trivial

-- ── the chain-close parameter ties, at a 1×1 projection ──────────────────────
example (bp : Vec OC) (ε : ℝ) (γp : Vec OC) (a cp dyOut : Vec (OC*HH*HH))
    (v : Vec (OC*IC*1*1)) (lr : ℝ) (i : Fin (OC*IC*1*1)) :
    v i - lr * (flatConvStride2_weight_grad_has_vjp bp zin).backward v
        (idBlockCotC2 ε γp a cp dyOut) i
      = v i - lr * ∑ j : Fin (OC*HH*HH),
          pdiv (fun v' : Vec (OC*IC*1*1) =>
            flatConvStride2 (Kernel4.unflatten v') bp zin) v i j
            * idBlockCotC2 ε γp a cp dyOut j :=
  downBlock_render_convWp_chain_certified bp zin ε γp a cp dyOut v lr i

example (bp : Vec OC) (ε : ℝ) (γp : Vec OC) (a cp dyOut : Vec (OC*HH*HH)) (lr : ℝ) (o : Fin OC) :
    bp o - lr * (flatConvStride2_bias_grad_has_vjp zkp1 zin).backward bp
        (idBlockCotC2 ε γp a cp dyOut) o
      = bp o - lr * ∑ j : Fin (OC*HH*HH),
          pdiv (fun b' : Vec OC => flatConvStride2 zkp1 b' zin) bp o j
            * idBlockCotC2 ε γp a cp dyOut j :=
  downBlock_render_convbp_chain_certified zkp1 zin bp ε γp a cp dyOut lr o

-- ── the §1a-style block tie, at 1×1 (the 12-parameter downsample statement) ──
example (dyOut : Vec (OC*HH*HH)) (lr : ℝ) :
    downblockTiedAt "%x" "%W" "%b" "%g" "%v" "1.0e-05" "0.1" "%cot" 1
      zk3 zc zc zc zk33 zc zc zc zkp1 zc zc zc zin dyOut lr :=
  r34_downblock_tiedAt "%x" "%W" "%b" "%g" "%v" "1.0e-05" "0.1" "%cot" 1
    zk3 zc zc zc zk33 zc zc zc zkp1 zc zc zc zin dyOut lr

-- ── the SpecVJP weight record, at 1×1 ────────────────────────────────────────
noncomputable example : R34DownW IC OC 1 1 :=
  { W1 := zk3, b1 := zc, g1 := zc, t1 := zc, W2 := zk33, b2 := zc, g2 := zc, t2 := zc,
    Wp := zkp1, bp := zc, gp := zc, tp := zc }

-- ── and the SAME statements still hold at 3×3, so nothing was traded away ────
noncomputable example : Vec (IC*(2*HH)*(2*HH)) → Vec (OC*HH*HH) :=
  downFwd (h := HH) (w := HH) 1 zk3 zc zc zc zk33 zc zc zc zkp3 zc zc zc
noncomputable example : R34DownW IC OC 3 3 :=
  { W1 := zk3, b1 := zc, g1 := zc, t1 := zc, W2 := zk33, b2 := zc, g2 := zc, t2 := zc,
    Wp := zkp3, bp := zc, gp := zc, tp := zc }
example (dyOut : Vec (OC*HH*HH)) (lr : ℝ) :
    downblockTiedAt "%x" "%W" "%b" "%g" "%v" "1.0e-05" "0.1" "%cot" 1
      zk3 zc zc zc zk33 zc zc zc zkp3 zc zc zc zin dyOut lr :=
  r34_downblock_tiedAt "%x" "%W" "%b" "%g" "%v" "1.0e-05" "0.1" "%cot" 1
    zk3 zc zc zc zk33 zc zc zc zkp3 zc zc zc zin dyOut lr

end Proj1x1Check
