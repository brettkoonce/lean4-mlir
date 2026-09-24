import LeanMlir.Proofs.Architectures.BatchNorm
import LeanMlir.Proofs.Certificates.LipschitzCert
import LeanMlir.Proofs.Certificates.LipschitzCertScorecardSDP
import LeanMlir.Proofs.Certificates.SmoothingGaussian
import LeanMlir.Proofs.Float.FloatBridge
import LeanMlir.Proofs.Foundation.DataParallel
import LeanMlir.Proofs.Foundation.DataParallelNode
import LeanMlir.Proofs.Foundation.DataParallelSync
import LeanMlir.Proofs.Foundation.DataParallelSyncBf16
import LeanMlir.Proofs.Foundation.MuonGeometry
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTieGB
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTieB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullWholeBackCertifiedTie
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBSeal
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBSeal
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullB
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTieB
import LeanMlir.Proofs.Foundation.GradNodesB
import LeanMlir.Proofs.Nets.ResNet.ResNet50FullBVJP
import LeanMlir.Proofs.Nets.ResNet.ResNet50StepTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncB
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncStepTieB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2SyncStepTieB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetSyncStepTieG
import LeanMlir.Proofs.Nets.ResNet.ResNet50SyncStepTieB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncStepTieB
import LeanMlir.Proofs.Nets.ViT.ViTDepthK
import LeanMlir.Proofs.Nets.ViT.ViTStepTie
import LeanMlir.Proofs.Training.TrainedLinearDescent

universe u_1

open Proofs
open scoped Real

set_option maxHeartbeats 8000000
-- The statements are pretty-printer output, and the printer names binders the declarations
-- never use (`fun (x : Fin n) => (0 : ℝ)`, `[inst : ...]` under `pp.explicit`).
set_option linter.unusedVariables false

/-! # Solution to the tier challenge

Solution to `ChallengeTier.lean`. Each proof is the project theorem itself:
the statement IS that theorem's type, so the delegation is a bare constant and nothing can
be weakened between the two files without failing to elaborate.

⚠ **MACHINE-GENERATED — do not hand-edit.** Every statement is the project declaration's
own type as Lean prints it (`scripts/gen_comparator_tier.py`), so this file and its
`ChallengeTier.lean` carry the same text by construction rather than by review. Regenerate after any
statement change; the generator verifies that what it wrote still elaborates.
-/

/-- `Proofs.bn_input_grad_correct` -/
theorem chk_bn_input_grad_correct :
    ∀ (n : ℕ) (ε γ β : ℝ),
      (0 : ℝ) < ε →
        ∀ (x dy : Proofs.Vec n) (i : Fin n),
          Proofs.bn_grad_input n ε γ x dy i = ∑ j : Fin n, Proofs.pdiv (Proofs.bnForward n ε γ β) x i j * dy j :=
  Proofs.bn_input_grad_correct

/-- `Proofs.resnet50ForwardB_full_has_vjp_at_correct` -/
theorem chk_resnet50ForwardB_full_has_vjp_at_correct :
    ∀ (N q : ℕ) (hq0 : (0 : ℕ) < q) {nCls : ℕ}
      (w : Proofs.R50BWeights nCls) (hp : Proofs.R50PosB w)
      (x :
        Proofs.Vec
          (N *
            ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
              ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))))))
      (hx : Proofs.R50SmoothAtB N q w x) (dy : Proofs.Vec (N * nCls))
      (i :
        Fin
          (N *
            ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
              ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q)))))))),
      (Proofs.resnet50ForwardB_full_has_vjp_at N q hq0 w hp x hx).backward dy i =
        ∑ j : Fin (N * nCls), Proofs.pdiv (Proofs.resnet50ForwardB_full N q w) x i j * dy j :=
  Proofs.resnet50ForwardB_full_has_vjp_at_correct

/-- `Proofs.vitTiny_has_vjp_correct` -/
theorem chk_vitTiny_has_vjp_correct :
    ∀ (W_conv : Proofs.Kernel4 ((3 : ℕ) * (64 : ℕ)) (3 : ℕ) (16 : ℕ) (16 : ℕ))
      (b_conv cls_token : Proofs.Vec ((3 : ℕ) * (64 : ℕ)))
      (pos_embed : Proofs.Mat ((196 : ℕ) + (1 : ℕ)) ((3 : ℕ) * (64 : ℕ))) (ε : ℝ) (hε : (0 : ℝ) < ε)
      (ps : Fin (12 : ℕ) → Proofs.BlockParamsV ((3 : ℕ) * (64 : ℕ)) (768 : ℕ)) (γF βF : Proofs.Vec ((3 : ℕ) * (64 : ℕ)))
      (Wcls : Proofs.Mat ((3 : ℕ) * (64 : ℕ)) (10 : ℕ)) (bcls : Proofs.Vec (10 : ℕ))
      (x : Proofs.Vec ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))) (dy : Proofs.Vec (10 : ℕ))
      (i : Fin ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))),
      (Proofs.vitForwardKV_has_vjp (3 : ℕ) (224 : ℕ) (224 : ℕ) (16 : ℕ) (196 : ℕ) (768 : ℕ) (3 : ℕ) (64 : ℕ) (10 : ℕ)
              (12 : ℕ) W_conv b_conv cls_token pos_embed ε hε ps γF βF Wcls bcls).backward
          x dy i =
        ∑ j : Fin (10 : ℕ),
          Proofs.pdiv
              (Proofs.vitForwardKV (3 : ℕ) (224 : ℕ) (224 : ℕ) (16 : ℕ) (196 : ℕ) (768 : ℕ) (3 : ℕ) (64 : ℕ) (10 : ℕ)
                (12 : ℕ) W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
              x i j *
            dy j :=
  Proofs.vitTiny_has_vjp_correct

/-- `Proofs.StableHLO.mnv4FwdGraphB_full_faithful` -/
theorem chk_mnv4FwdGraphB_full_faithful :
    ∀ (N : ℕ) (epsStr : String) {nCls : ℕ}
      (w : Proofs.StableHLO.Mnv4BWeights nCls) (e : Proofs.StableHLO.SHlo (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
      Proofs.StableHLO.den (Proofs.StableHLO.mnv4FwdGraphB_full N epsStr w e) =
        Proofs.StableHLO.mobilenetv4ForwardB_full N w (Proofs.StableHLO.den e) :=
  Proofs.StableHLO.mnv4FwdGraphB_full_faithful

/-- `Proofs.Mnv2FullBSeal.sealX_nonconstant` -/
theorem chk_sealX_nonconstant :
    ∀ (nCls : ℕ),
      (0 : ℕ) < nCls →
        Proofs.mobilenetv2ForwardB_full (2 : ℕ) (Proofs.Mnv2FullBSeal.sealW nCls) (Proofs.Mnv2FullBSeal.sealX (1 : ℝ)) ≠
          Proofs.mobilenetv2ForwardB_full (2 : ℕ) (Proofs.Mnv2FullBSeal.sealW nCls) (Proofs.Mnv2FullBSeal.sealX (0 : ℝ)) :=
  Proofs.Mnv2FullBSeal.sealX_nonconstant

/-- `Proofs.Mnv4FullBSeal.sealX_backward_nontrivial` -/
theorem chk_sealX_backward_nontrivial :
    ∀ (nCls : ℕ),
      (0 : ℕ) < nCls →
        ∃ (j₀ : Fin ((2 : ℕ) * nCls)) (i₀ : Fin ((2 : ℕ) * ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ))))),
          (Proofs.Mnv4FullBSeal.sealVJP nCls (0 : ℝ)).backward (Proofs.basisVec j₀) i₀ ≠ (0 : ℝ) :=
  Proofs.Mnv4FullBSeal.sealX_backward_nontrivial

/-- `Proofs.ResNet34PoCB.convStridedWGradB_den` -/
theorem chk_convStridedWGradB_den :
    ∀ {N ic oc h w kH kW : ℕ} (xN cotN : String) (b : Proofs.Vec oc)
      (x : Proofs.Vec (N * (ic * ((2 : ℕ) * h) * ((2 : ℕ) * w)))) (W : Proofs.Kernel4 oc ic kH kW)
      (cot : Proofs.Vec (N * (oc * h * w))) (idx : Fin (oc * ic * kH * kW)),
      Proofs.StableHLO.den (Proofs.StableHLO.SHlo.convStridedWeightGradB xN b x W (Proofs.StableHLO.SHlo.operand cotN cot))
          idx =
        ∑ n : Fin N,
          ∑ j : Fin (oc * h * w),
            Proofs.pdiv
                (fun (v' : Proofs.Vec (oc * ic * kH * kW)) =>
                  Proofs.flatConvStride2 (Proofs.Kernel4.unflatten v') b
                    (Proofs.StableHLO.batchSlice N (ic * ((2 : ℕ) * h) * ((2 : ℕ) * w)) x n))
                W.flatten idx j *
              Proofs.StableHLO.batchSlice N (oc * h * w) cot n j :=
  Proofs.ResNet34PoCB.convStridedWGradB_den

/-- `Proofs.smoothedCE_grad` -/
theorem chk_smoothedCE_grad :
    ∀ (K : ℕ),
      (0 : ℕ) < K →
        ∀ (α : ℝ) (t z : Proofs.Vec K),
          ∑ k : Fin K, t k = (1 : ℝ) →
            ∀ (j : Fin K),
              Proofs.pdiv (fun (z' : Proofs.Vec K) (x : Fin (1 : ℕ)) => Proofs.softCE K (Proofs.smoothTarget K α t) z') z j
                  (0 : Fin (1 : ℕ)) =
                Proofs.softmax K z j - t j + α * t j - α / ↑K :=
  Proofs.smoothedCE_grad

/-- `Proofs.ResNet50TieB.r50_net_tiedB` -/
theorem chk_r50_net_tiedB :
    ∀ (N q : ℕ) {nCls : ℕ} (xN cotN vN epsStr : String) (w : Proofs.R50BWeights nCls)
      (x :
        Proofs.Vec
          (N *
            ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
              ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))))))
      (g : Proofs.Vec (N * nCls)),
      have dy16 := Proofs.ResNet34TieB.r34HeadCotBlk N q q w.Wd w.bd (Proofs.r50Pre16 N q w x) g;
      have dy15 := Proofs.ResNet50TieB.r50IdCotIn N q q w.s4b2 (Proofs.r50Pre15 N q w x) dy16;
      have dy14 := Proofs.ResNet50TieB.r50IdCotIn N q q w.s4b1 (Proofs.r50Pre14 N q w x) dy15;
      have dy13 := Proofs.ResNet50TieB.r50DownCotIn N q q w.s4b0 (Proofs.r50Pre13 N q w x) dy14;
      have dy12 := Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b5 (Proofs.r50Pre12 N q w x) dy13;
      have dy11 := Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b4 (Proofs.r50Pre11 N q w x) dy12;
      have dy10 := Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b3 (Proofs.r50Pre10 N q w x) dy11;
      have dy9 := Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b2 (Proofs.r50Pre9 N q w x) dy10;
      have dy8 := Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b1 (Proofs.r50Pre8 N q w x) dy9;
      have dy7 := Proofs.ResNet50TieB.r50DownCotIn N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b0 (Proofs.r50Pre7 N q w x) dy8;
      have dy6 :=
        Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) w.s2b3 (Proofs.r50Pre6 N q w x)
          dy7;
      have dy5 :=
        Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) w.s2b2 (Proofs.r50Pre5 N q w x)
          dy6;
      have dy4 :=
        Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) w.s2b1 (Proofs.r50Pre4 N q w x)
          dy5;
      have dy3 :=
        Proofs.ResNet50TieB.r50DownCotIn N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) w.s2b0
          (Proofs.r50Pre3 N q w x) dy4;
      have dy2 :=
        Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) w.s1b2
          (Proofs.r50Pre2 N q w x) dy3;
      have dy1 :=
        Proofs.ResNet50TieB.r50IdCotIn N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) w.s1b1
          (Proofs.r50Pre1 N q w x) dy2;
      have cotPool :=
        Proofs.ResNet50TieB.r50ProjCotIn N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q)))
          w.s1b0 (Proofs.r50Pre0 N q w x) dy1;
      Proofs.ResNet34TieB.r34StemTiedB N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) xN cotN
          vN epsStr w.sW w.sb w.sε w.sγ w.sβ x cotPool ∧
        Proofs.ResNet50TieB.r50ProjTiedB N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) xN
            cotN vN epsStr w.s1b0 (Proofs.r50Pre0 N q w x) dy1 ∧
          Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) xN
              cotN vN epsStr w.s1b1 (Proofs.r50Pre1 N q w x) dy2 ∧
            Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) xN
                cotN vN epsStr w.s1b2 (Proofs.r50Pre2 N q w x) dy3 ∧
              Proofs.ResNet50TieB.r50DownTiedB N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) xN cotN vN epsStr
                  w.s2b0 (Proofs.r50Pre3 N q w x) dy4 ∧
                Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) xN cotN vN epsStr
                    w.s2b1 (Proofs.r50Pre4 N q w x) dy5 ∧
                  Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) xN cotN vN epsStr
                      w.s2b2 (Proofs.r50Pre5 N q w x) dy6 ∧
                    Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) xN cotN vN epsStr
                        w.s2b3 (Proofs.r50Pre6 N q w x) dy7 ∧
                      Proofs.ResNet50TieB.r50DownTiedB N ((2 : ℕ) * q) ((2 : ℕ) * q) xN cotN vN epsStr w.s3b0
                          (Proofs.r50Pre7 N q w x) dy8 ∧
                        Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * q) ((2 : ℕ) * q) xN cotN vN epsStr w.s3b1
                            (Proofs.r50Pre8 N q w x) dy9 ∧
                          Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * q) ((2 : ℕ) * q) xN cotN vN epsStr w.s3b2
                              (Proofs.r50Pre9 N q w x) dy10 ∧
                            Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * q) ((2 : ℕ) * q) xN cotN vN epsStr w.s3b3
                                (Proofs.r50Pre10 N q w x) dy11 ∧
                              Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * q) ((2 : ℕ) * q) xN cotN vN epsStr w.s3b4
                                  (Proofs.r50Pre11 N q w x) dy12 ∧
                                Proofs.ResNet50TieB.r50IdTiedB N ((2 : ℕ) * q) ((2 : ℕ) * q) xN cotN vN epsStr w.s3b5
                                    (Proofs.r50Pre12 N q w x) dy13 ∧
                                  Proofs.ResNet50TieB.r50DownTiedB N q q xN cotN vN epsStr w.s4b0 (Proofs.r50Pre13 N q w x)
                                      dy14 ∧
                                    Proofs.ResNet50TieB.r50IdTiedB N q q xN cotN vN epsStr w.s4b1 (Proofs.r50Pre14 N q w x)
                                        dy15 ∧
                                      Proofs.ResNet50TieB.r50IdTiedB N q q xN cotN vN epsStr w.s4b2
                                          (Proofs.r50Pre15 N q w x) dy16 ∧
                                        Proofs.ResNet34TieB.r34HeadTiedB N q q xN cotN w.Wd w.bd (Proofs.r50Pre16 N q w x) g :=
  Proofs.ResNet50TieB.r50_net_tiedB

/-- `Proofs.ViTTiePoC.vit_net_tied_certified` -/
theorem chk_vit_net_tied_certified :
    ∀ (xN wN bN gN aN clsN pN epsStr lrStr cotN : String) (ε : Real)
      (w : Proofs.ViTTiePoC.ViTTieWeights (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))))
      (img :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat)
            (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
              (@OfNat.ofNat Nat (nat_lit 224) (instOfNatNat (nat_lit 224))))
            (@OfNat.ofNat Nat (nat_lit 224) (instOfNatNat (nat_lit 224)))))
      (label : Fin (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10)))) (lr : Real),
      have ib1 :=
        Proofs.patchEmbed_flat (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 224) (instOfNatNat (nat_lit 224)))
          (@OfNat.ofNat Nat (nat_lit 224) (instOfNatNat (nat_lit 224)))
          (@OfNat.ofNat Nat (nat_lit 16) (instOfNatNat (nat_lit 16)))
          (@OfNat.ofNat Nat (nat_lit 196) (instOfNatNat (nat_lit 196)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@Proofs.ViTTiePoC.ViTTieWeights.Wc (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTiePoC.ViTTieWeights.bc (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTiePoC.ViTTieWeights.cls (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTiePoC.ViTTieWeights.pos (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) img;
      have ib2 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b1 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib1;
      have ib3 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b2 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib2;
      have ib4 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b3 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib3;
      have ib5 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b4 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib4;
      have ib6 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b5 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib5;
      have ib7 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b6 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib6;
      have ib8 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b7 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib7;
      have ib9 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b8 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib8;
      have ib10 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b9 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib9;
      have ib11 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b10 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib10;
      have ib12 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b11 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib11;
      have b12out :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b12 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib12;
      have fl :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.Mat.flatten (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          fun (r : Fin (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))) =>
          Proofs.layerNormVec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) ε
            (@Proofs.ViTTiePoC.ViTTieWeights.γF (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
            (@Proofs.ViTTiePoC.ViTTieWeights.βF (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
            (@Proofs.Mat.unflatten (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
              (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) b12out r);
      have hn :=
        Proofs.StableHLO.clsSliceFlat (@OfNat.ofNat Nat (nat_lit 196) (instOfNatNat (nat_lit 196)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) fl;
      have logits :=
        @Proofs.dense (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10)))
          (@Proofs.ViTTiePoC.ViTTieWeights.Wcls (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTiePoC.ViTTieWeights.bcls (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) hn;
      have g : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) :=
        fun (c : Fin (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10)))) =>
        @HSub.hSub Real Real Real (@instHSub Real Real.instSub)
          (Proofs.softmax (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) logits c)
          (Proofs.oneHot (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) label c);
      have dy12 :=
        Proofs.vitCotB2outV (@OfNat.ofNat Nat (nat_lit 196) (instOfNatNat (nat_lit 196)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) ε
          (@Proofs.ViTTiePoC.ViTTieWeights.γF (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTiePoC.ViTTieWeights.Wcls (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) b12out g;
      have dy11 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b12 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib12 dy12;
      have dy10 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b11 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib11 dy11;
      have dy9 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b10 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib10 dy10;
      have dy8 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b9 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib9 dy9;
      have dy7 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b8 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib8 dy8;
      have dy6 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b7 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib7 dy7;
      have dy5 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b6 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib6 dy6;
      have dy4 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b5 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib5 dy5;
      have dy3 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b4 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib4 dy4;
      have dy2 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b3 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib3 dy3;
      have dy1 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b2 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib2 dy2;
      have dyEmbed :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b1 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib1 dy1;
      And
        (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTiePoC.ViTTieWeights.b1 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN gN
          epsStr lrStr cotN ε ib1 dy1 lr)
        (And
          (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
            (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
            (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
            (@Proofs.ViTTiePoC.ViTTieWeights.b2 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN gN
            epsStr lrStr cotN ε ib2 dy2 lr)
          (And
            (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
              (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
              (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
              (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
              (@Proofs.ViTTiePoC.ViTTieWeights.b3 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN gN
              epsStr lrStr cotN ε ib3 dy3 lr)
            (And
              (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                (@Proofs.ViTTiePoC.ViTTieWeights.b4 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN
                gN epsStr lrStr cotN ε ib4 dy4 lr)
              (And
                (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                  (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                  (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                  (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                  (@Proofs.ViTTiePoC.ViTTieWeights.b5 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN
                  bN gN epsStr lrStr cotN ε ib5 dy5 lr)
                (And
                  (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                    (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                    (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                    (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                    (@Proofs.ViTTiePoC.ViTTieWeights.b6 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN
                    bN gN epsStr lrStr cotN ε ib6 dy6 lr)
                  (And
                    (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                      (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                      (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                      (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                      (@Proofs.ViTTiePoC.ViTTieWeights.b7 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN
                      wN bN gN epsStr lrStr cotN ε ib7 dy7 lr)
                    (And
                      (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                        (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                        (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                        (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                        (@Proofs.ViTTiePoC.ViTTieWeights.b8 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                        xN wN bN gN epsStr lrStr cotN ε ib8 dy8 lr)
                      (And
                        (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                          (@Proofs.ViTTiePoC.ViTTieWeights.b9 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                          xN wN bN gN epsStr lrStr cotN ε ib9 dy9 lr)
                        (And
                          (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                            (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                            (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                            (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                            (@Proofs.ViTTiePoC.ViTTieWeights.b10 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10)))
                              w)
                            xN wN bN gN epsStr lrStr cotN ε ib10 dy10 lr)
                          (And
                            (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                              (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                              (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                              (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                              (@Proofs.ViTTiePoC.ViTTieWeights.b11
                                (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                              xN wN bN gN epsStr lrStr cotN ε ib11 dy11 lr)
                            (And
                              (@Proofs.BlockParamsV.TiedAt (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                                (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                                (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                                (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                                (@Proofs.ViTTiePoC.ViTTieWeights.b12
                                  (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                xN wN bN gN epsStr lrStr cotN ε ib12 dy12 lr)
                              (And
                                (Proofs.ViTTiePoC.vitFinalLNTied gN xN bN epsStr lrStr cotN ε
                                  (@Proofs.ViTTiePoC.ViTTieWeights.γF
                                    (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                  (@Proofs.ViTTiePoC.ViTTieWeights.βF
                                    (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                  (@Proofs.ViTTiePoC.ViTTieWeights.Wcls
                                    (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                  b12out g lr)
                                (And
                                  (Proofs.ViTTiePoC.vitHeadTied aN wN bN lrStr cotN hn
                                    (@Proofs.ViTTiePoC.ViTTieWeights.Wcls
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    (@Proofs.ViTTiePoC.ViTTieWeights.bcls
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    g lr)
                                  (Proofs.ViTTiePoC.vitEmbedTied wN xN bN clsN pN lrStr cotN
                                    (@Proofs.ViTTiePoC.ViTTieWeights.Wc
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    (@Proofs.ViTTiePoC.ViTTieWeights.bc
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    (@Proofs.ViTTiePoC.ViTTieWeights.cls
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    (@Proofs.ViTTiePoC.ViTTieWeights.pos
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    img dyEmbed lr)))))))))))))) :=
  Proofs.ViTTiePoC.vit_net_tied_certified

/-- `Proofs.CnxTiePoCGB.cnx_net_tiedGB` -/
theorem chk_cnx_net_tiedGB :
    ∀ (N : ℕ) {nC : ℕ} (xN epsStr cotN dN aStr negAK bStr logN ohN : String) (ε α B : ℝ)
      (w : Proofs.CnxTiePoC.CnxTieWeights nC) (xstem : Proofs.Vec (N * ((3 : ℕ) * (56 : ℕ) * (56 : ℕ))))
      (x : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))) (t : Proofs.Vec (N * nC)),
      have ib1 : Proofs.Vec (N * ((96 : ℕ) * (56 : ℕ) * (56 : ℕ))) :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxStemFwdO ε w.sW w.sb w.sγ w.sβ) x;
      have ib2 := Proofs.StableHLO.batchMap N (w.b1.fwdO ε) ib1;
      have ib3 := Proofs.StableHLO.batchMap N (w.b2.fwdO ε) ib2;
      have ibD0 := Proofs.StableHLO.batchMap N (w.b3.fwdO ε) ib3;
      have ib4 : Proofs.Vec (N * ((192 : ℕ) * (28 : ℕ) * (28 : ℕ))) := Proofs.StableHLO.batchMap N (w.d0.fwdO ε) ibD0;
      have ib5 := Proofs.StableHLO.batchMap N (w.b4.fwdO ε) ib4;
      have ib6 := Proofs.StableHLO.batchMap N (w.b5.fwdO ε) ib5;
      have ibD1 := Proofs.StableHLO.batchMap N (w.b6.fwdO ε) ib6;
      have ib7 : Proofs.Vec (N * ((384 : ℕ) * (14 : ℕ) * (14 : ℕ))) := Proofs.StableHLO.batchMap N (w.d1.fwdO ε) ibD1;
      have ib8 := Proofs.StableHLO.batchMap N (w.b7.fwdO ε) ib7;
      have ib9 := Proofs.StableHLO.batchMap N (w.b8.fwdO ε) ib8;
      have ib10 := Proofs.StableHLO.batchMap N (w.b9.fwdO ε) ib9;
      have ib11 := Proofs.StableHLO.batchMap N (w.b10.fwdO ε) ib10;
      have ib12 := Proofs.StableHLO.batchMap N (w.b11.fwdO ε) ib11;
      have ib13 := Proofs.StableHLO.batchMap N (w.b12.fwdO ε) ib12;
      have ib14 := Proofs.StableHLO.batchMap N (w.b13.fwdO ε) ib13;
      have ib15 := Proofs.StableHLO.batchMap N (w.b14.fwdO ε) ib14;
      have ibD2 := Proofs.StableHLO.batchMap N (w.b15.fwdO ε) ib15;
      have ib16 : Proofs.Vec (N * ((768 : ℕ) * (7 : ℕ) * (7 : ℕ))) := Proofs.StableHLO.batchMap N (w.d2.fwdO ε) ibD2;
      have ib17 := Proofs.StableHLO.batchMap N (w.b16.fwdO ε) ib16;
      have ib18 := Proofs.StableHLO.batchMap N (w.b17.fwdO ε) ib17;
      have xhead := Proofs.StableHLO.batchMap N (w.b18.fwdO ε) ib18;
      have gapB := Proofs.StableHLO.batchMap N (Proofs.globalAvgPoolFlat (768 : ℕ) (7 : ℕ) (7 : ℕ)) xhead;
      have hnB := Proofs.StableHLO.batchMap N (Proofs.rowLNVecFlat (1 : ℕ) (768 : ℕ) ε w.hG w.hT) gapB;
      have logitsB := Proofs.StableHLO.batchMap N (Proofs.dense w.Wfc w.bfc) hnB;
      have g := Proofs.StableHLO.den (Proofs.smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t);
      have dyO18 := Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoCGB.cnxHeadDyXheadChN ε w.hG w.hT w.Wfc w.bfc) xhead g;
      have dyO17 := Proofs.StableHLO.batchMapAux N (w.b18.cotIn ε) ib18 dyO18;
      have dyO16 := Proofs.StableHLO.batchMapAux N (w.b17.cotIn ε) ib17 dyO17;
      have dyD2 := Proofs.StableHLO.batchMapAux N (w.b16.cotIn ε) ib16 dyO16;
      have dyO15 := Proofs.StableHLO.batchMapAux N (w.d2.cotIn ε) ibD2 dyD2;
      have dyO14 := Proofs.StableHLO.batchMapAux N (w.b15.cotIn ε) ib15 dyO15;
      have dyO13 := Proofs.StableHLO.batchMapAux N (w.b14.cotIn ε) ib14 dyO14;
      have dyO12 := Proofs.StableHLO.batchMapAux N (w.b13.cotIn ε) ib13 dyO13;
      have dyO11 := Proofs.StableHLO.batchMapAux N (w.b12.cotIn ε) ib12 dyO12;
      have dyO10 := Proofs.StableHLO.batchMapAux N (w.b11.cotIn ε) ib11 dyO11;
      have dyO9 := Proofs.StableHLO.batchMapAux N (w.b10.cotIn ε) ib10 dyO10;
      have dyO8 := Proofs.StableHLO.batchMapAux N (w.b9.cotIn ε) ib9 dyO9;
      have dyO7 := Proofs.StableHLO.batchMapAux N (w.b8.cotIn ε) ib8 dyO8;
      have dyD1 := Proofs.StableHLO.batchMapAux N (w.b7.cotIn ε) ib7 dyO7;
      have dyO6 := Proofs.StableHLO.batchMapAux N (w.d1.cotIn ε) ibD1 dyD1;
      have dyO5 := Proofs.StableHLO.batchMapAux N (w.b6.cotIn ε) ib6 dyO6;
      have dyO4 := Proofs.StableHLO.batchMapAux N (w.b5.cotIn ε) ib5 dyO5;
      have dyD0 := Proofs.StableHLO.batchMapAux N (w.b4.cotIn ε) ib4 dyO4;
      have dyO3 := Proofs.StableHLO.batchMapAux N (w.d0.cotIn ε) ibD0 dyD0;
      have dyO2 := Proofs.StableHLO.batchMapAux N (w.b3.cotIn ε) ib3 dyO3;
      have dyO1 := Proofs.StableHLO.batchMapAux N (w.b2.cotIn ε) ib2 dyO2;
      have dyStem := Proofs.StableHLO.batchMapAux N (w.b1.cotIn ε) ib1 dyO1;
      Proofs.CnxTiePoCGB.cnxStemChTiedGBAt N xN epsStr cotN ε w.sW w.sb w.sγ w.sβ x xstem dyStem ∧
        w.b1.TiedGB N xN epsStr cotN ε ib1 dyO1 ∧
          w.b2.TiedGB N xN epsStr cotN ε ib2 dyO2 ∧
            w.b3.TiedGB N xN epsStr cotN ε ib3 dyO3 ∧
              w.d0.TiedGB N xN epsStr cotN ε ibD0 dyD0 ∧
                w.b4.TiedGB N xN epsStr cotN ε ib4 dyO4 ∧
                  w.b5.TiedGB N xN epsStr cotN ε ib5 dyO5 ∧
                    w.b6.TiedGB N xN epsStr cotN ε ib6 dyO6 ∧
                      w.d1.TiedGB N xN epsStr cotN ε ibD1 dyD1 ∧
                        w.b7.TiedGB N xN epsStr cotN ε ib7 dyO7 ∧
                          w.b8.TiedGB N xN epsStr cotN ε ib8 dyO8 ∧
                            w.b9.TiedGB N xN epsStr cotN ε ib9 dyO9 ∧
                              w.b10.TiedGB N xN epsStr cotN ε ib10 dyO10 ∧
                                w.b11.TiedGB N xN epsStr cotN ε ib11 dyO11 ∧
                                  w.b12.TiedGB N xN epsStr cotN ε ib12 dyO12 ∧
                                    w.b13.TiedGB N xN epsStr cotN ε ib13 dyO13 ∧
                                      w.b14.TiedGB N xN epsStr cotN ε ib14 dyO14 ∧
                                        w.b15.TiedGB N xN epsStr cotN ε ib15 dyO15 ∧
                                          w.d2.TiedGB N xN epsStr cotN ε ibD2 dyD2 ∧
                                            w.b16.TiedGB N xN epsStr cotN ε ib16 dyO16 ∧
                                              w.b17.TiedGB N xN epsStr cotN ε ib17 dyO17 ∧
                                                w.b18.TiedGB N xN epsStr cotN ε ib18 dyO18 ∧
                                                  Proofs.CnxTiePoCGB.cnxHeadChTiedGBAt N xN epsStr cotN dN ε w.hG w.hT w.Wfc
                                                    w.bfc xhead g :=
  Proofs.CnxTiePoCGB.cnx_net_tiedGB

/-- `Proofs.dpMeanGrad_ne_globalBatchGrad` -/
theorem chk_dpMeanGrad_ne_globalBatchGrad :
    ∀ (θ : Proofs.Vec (1 : ℕ)),
      (Proofs.dpMean (R := (2 : ℕ)) fun (r : Fin (2 : ℕ)) => Proofs.lossGrad (Proofs.bnToyLoss (Proofs.dpToyShard r)) θ) ≠
        Proofs.lossGrad (Proofs.bnToyLoss Proofs.dpToyBatch) θ :=
  Proofs.dpMeanGrad_ne_globalBatchGrad

/-- `Proofs.dpSyncGrad_eq_globalBatchGrad` -/
theorem chk_dpSyncGrad_eq_globalBatchGrad :
    ∀ {R N P : ℕ} (e : Fin R × Fin N ≃ Fin (R * N))
      (c : Fin (R * N) → Proofs.Vec P),
      Eq (α := Proofs.Vec P)
        (Proofs.dpMean (R := R) fun (r : Fin R) (i : Fin P) => (1 : ℝ) / ↑N * ∑ n : Fin N, c (e (r, n)) i)
        fun (i : Fin P) => (1 : ℝ) / ↑(R * N) * ∑ m : Fin (R * N), c m i :=
  Proofs.dpSyncGrad_eq_globalBatchGrad

/-- `Proofs.den_bnSyncBack_allReduce` -/
theorem chk_den_bnSyncBack_allReduce :
    ∀ {N oc h w : ℕ} (R : ℕ) (hR : (0 : ℕ) < R),
      N * (h * w) ≠ (0 : ℕ) →
        R * N * (h * w) ≠ (0 : ℕ) →
          ∀ (gN xN es t t' t'' : String) (ds ds' ds'' : List ℕ) (ε : ℝ) (γ : Proofs.Vec oc)
            (x : Fin R → Proofs.StableHLO.SHlo (N * (oc * (h * w)))) (xv : Fin R → Proofs.Vec (N * (oc * (h * w))))
            (dy : Fin R → Proofs.StableHLO.SHlo (N * (oc * (h * w)))) (X DY : Proofs.Vec (R * N * (oc * (h * w)))),
            (∀ (r : Fin R), Proofs.StableHLO.den (x r) = Proofs.batchShard R N (oc * (h * w)) X r) →
              (∀ (r : Fin R), xv r = Proofs.batchShard R N (oc * (h * w)) X r) →
                (∀ (r : Fin R), Proofs.StableHLO.den (dy r) = Proofs.batchShard R N (oc * (h * w)) DY r) →
                  ∀ (r : Fin R),
                    Proofs.StableHLO.den
                        (Proofs.StableHLO.SHlo.bnSyncBack gN xN es ε γ (xv r) (dy r)
                          (Proofs.StableHLO.SHlo.allReduceMeanF R hR t'' ds'' fun (r' : Fin R) =>
                            Proofs.StableHLO.SHlo.bnSyncDyStatsB gN xN es ε γ (xv r') (dy r')
                              (Proofs.syncStats R hR t t' ds ds' x))) =
                      Proofs.batchShard R N (oc * (h * w)) (Proofs.bnBatchTensor4_grad_input (R * N) oc h w ε γ X DY) r :=
  Proofs.den_bnSyncBack_allReduce

/-- `Proofs.den_allReduceMeanF_convWeightGradBBf16_sub_global` -/
theorem chk_den_allReduceMeanF_convWeightGradBBf16_sub_global :
    ∀ {N ic oc h w kH kW : ℕ} (R : ℕ) (hR : (0 : ℕ) < R)
      (rnd : ℝ → ℝ) (t xN cotN : String) (ds : List ℕ) (b : Proofs.Vec oc) (W : Proofs.Kernel4 oc ic kH kW)
      (X : Proofs.Vec (R * N * (ic * h * w))) (DY : Proofs.Vec (R * N * (oc * h * w)))
      (dy : Fin R → Proofs.StableHLO.SHlo (N * (oc * h * w))),
      (∀ (r : Fin R), Proofs.StableHLO.den (dy r) = Proofs.batchShard R N (oc * h * w) DY r) →
        ∀ (idx : Fin (oc * ic * kH * kW)),
          Proofs.StableHLO.den
                (Proofs.StableHLO.SHlo.allReduceMeanF R hR t ds fun (r : Fin R) =>
                  Proofs.StableHLO.SHlo.convWeightGradBBf16 rnd xN b (Proofs.batchShard R N (ic * h * w) X r) W (dy r))
                idx -
              (1 : ℝ) / ↑R *
                Proofs.StableHLO.den
                  (Proofs.StableHLO.SHlo.convWeightGradBBf16 rnd xN b X W (Proofs.StableHLO.SHlo.operand cotN DY)) idx =
            (1 : ℝ) / ↑R *
              (∑ r : Fin R, rnd (Proofs.convWGradShardSum rnd xN cotN b X W DY r idx) -
                rnd (∑ r : Fin R, Proofs.convWGradShardSum rnd xN cotN b X W DY r idx)) :=
  Proofs.den_allReduceMeanF_convWeightGradBBf16_sub_global

/-- `Proofs.StableHLO.resnet34FwdGraphSync_full_shard` -/
theorem chk_resnet34FwdGraphSync_full_shard :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (epsStr : String) {nCls : ℕ} (w : Proofs.R34BWeights nCls)
          (e :
            Fin R →
              Proofs.StableHLO.SHlo (N * ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))))))
          (X : Proofs.Vec (R * N * ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ)))))),
          (∀ (r : Fin R),
              Proofs.StableHLO.den (e r) =
                Proofs.batchShard R N ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ)))) X r) →
            ∀ (r : Fin R),
              Proofs.StableHLO.den (Proofs.StableHLO.resnet34FwdGraphSync_full R hR N epsStr w e r) =
                Proofs.batchShard R N nCls (Proofs.resnet34ForwardB_full (R * N) w X) r :=
  Proofs.StableHLO.resnet34FwdGraphSync_full_shard

/-- `Proofs.ResNet34SyncTieB.r34_net_syncTiedB` -/
theorem chk_r34_net_syncTiedB :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ {nCls : ℕ} (xN cotN vN epsStr : String) (w : Proofs.R34BWeights nCls)
          (X : Proofs.Vec (R * N * ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))))))
          (G : Proofs.Vec (R * N * nCls)) (gs : Fin R → Proofs.Vec (N * nCls)),
          (∀ (r : Fin R), gs r = Proofs.batchShard R N nCls (fun (i : Fin (R * N * nCls)) => ↑R * G i) r) →
            Proofs.ResNet34SyncTieB.r34NetSyncTiedB R hR N xN cotN vN epsStr w X G gs :=
  Proofs.ResNet34SyncTieB.r34_net_syncTiedB

/-- `Proofs.StableHLO.mobilenetv2FwdGraphSync_full_shard` -/
theorem chk_mobilenetv2FwdGraphSync_full_shard :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (epsStr : String) {nCls : ℕ} (w : Proofs.MNV2BWeights nCls)
          (e : Fin R → Proofs.StableHLO.SHlo (N * ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ)))))
          (X : Proofs.Vec (R * N * ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ))))),
          (∀ (r : Fin R),
              Proofs.StableHLO.den (e r) =
                Proofs.batchShard R N ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ))) X r) →
            ∀ (r : Fin R),
              Proofs.StableHLO.den (Proofs.StableHLO.mobilenetv2FwdGraphSync_full R hR N epsStr w e r) =
                Proofs.batchShard R N nCls (Proofs.mobilenetv2ForwardB_full (R * N) w X) r :=
  Proofs.StableHLO.mobilenetv2FwdGraphSync_full_shard

/-- `Proofs.MobileNetV2SyncTieB.mnv2_net_syncTiedB` -/
theorem chk_mnv2_net_syncTiedB :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ {nCls : ℕ} (xN cotN vN epsStr : String) (w : Proofs.MNV2BWeights nCls)
          (X : Proofs.Vec (R * N * ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ)))))
          (G : Proofs.Vec (R * N * nCls)) (gs : Fin R → Proofs.Vec (N * nCls)),
          (∀ (r : Fin R), gs r = Proofs.batchShard R N nCls (fun (i : Fin (R * N * nCls)) => ↑R * G i) r) →
            Proofs.MobileNetV2SyncTieB.mnv2NetSyncTiedB R hR N xN cotN vN epsStr w X G gs :=
  Proofs.MobileNetV2SyncTieB.mnv2_net_syncTiedB

/-- `Proofs.StableHLO.efficientnetFwdGraphSync_full_shard` -/
theorem chk_efficientnetFwdGraphSync_full_shard :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (epsStr : String) (w : Proofs.B0Weights)
          (e : Fin R → Proofs.StableHLO.SHlo (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))))
          (X : Proofs.Vec (R * N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
          (∀ (r : Fin R), Proofs.StableHLO.den (e r) = Proofs.batchShard R N ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)) X r) →
            ∀ (r : Fin R),
              Proofs.StableHLO.den (Proofs.StableHLO.efficientnetFwdGraphSync_full R hR N epsStr w e r) =
                Proofs.batchShard R N (10 : ℕ) (Proofs.efficientnetForwardB_full (R * N) w X) r :=
  Proofs.StableHLO.efficientnetFwdGraphSync_full_shard

/-- `Proofs.EnetSyncTieG.efficientnet_net_syncTiedG` -/
theorem chk_efficientnet_net_syncTiedG :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (xN vN epsStr cotN dN : String) (w : Proofs.B0Weights) (hεw : w.EpsPos)
          (x : Proofs.Vec (R * N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))) (g : Proofs.Vec (R * N * (10 : ℕ)))
          (gs : Fin R → Proofs.Vec (N * (10 : ℕ))),
          (∀ (r : Fin R), gs r = Proofs.batchShard R N (10 : ℕ) (fun (i : Fin (R * N * (10 : ℕ))) => ↑R * g i) r) →
            Proofs.EnetSyncTieG.enetNetSyncTiedG R hR N xN vN epsStr cotN dN w hεw x g gs :=
  Proofs.EnetSyncTieG.efficientnet_net_syncTiedG

/-- `Proofs.StableHLO.resnet50FwdGraphSync_full_shard` -/
theorem chk_resnet50FwdGraphSync_full_shard :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (q : ℕ),
          (0 : ℕ) < q →
            ∀ (epsStr : String) {nCls : ℕ} (w : Proofs.R50BWeights nCls)
              (e :
                Fin R →
                  Proofs.StableHLO.SHlo
                    (N *
                      ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
                        ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))))))
              (X :
                Proofs.Vec
                  (R * N *
                    ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
                      ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q)))))))),
              (∀ (r : Fin R),
                  Proofs.StableHLO.den (e r) =
                    Proofs.batchShard R N
                      ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
                        ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))))
                      X r) →
                ∀ (r : Fin R),
                  Proofs.StableHLO.den (Proofs.StableHLO.resnet50FwdGraphSync_full R hR N q epsStr w e r) =
                    Proofs.batchShard R N nCls (Proofs.resnet50ForwardB_full (R * N) q w X) r :=
  Proofs.StableHLO.resnet50FwdGraphSync_full_shard

/-- `Proofs.ResNet50SyncTieB.r50_net_syncTiedB` -/
theorem chk_r50_net_syncTiedB :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (q : ℕ),
          (0 : ℕ) < q →
            ∀ {nCls : ℕ} (xN cotN vN epsStr : String) (w : Proofs.R50BWeights nCls)
              (X :
                Proofs.Vec
                  (R * N *
                    ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
                      ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))))))
              (G : Proofs.Vec (R * N * nCls)) (gs : Fin R → Proofs.Vec (N * nCls)),
              (∀ (r : Fin R), gs r = Proofs.batchShard R N nCls (fun (i : Fin (R * N * nCls)) => ↑R * G i) r) →
                Proofs.ResNet50SyncTieB.r50NetSyncTiedB R hR N q xN cotN vN epsStr w X G gs :=
  Proofs.ResNet50SyncTieB.r50_net_syncTiedB

/-- `Proofs.StableHLO.mnv4FwdGraphSync_full_shard` -/
theorem chk_mnv4FwdGraphSync_full_shard :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (epsStr : String) {nCls : ℕ} (w : Proofs.StableHLO.Mnv4BWeights nCls)
          (e : Fin R → Proofs.StableHLO.SHlo (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))))
          (X : Proofs.Vec (R * N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
          (∀ (r : Fin R), Proofs.StableHLO.den (e r) = Proofs.batchShard R N ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)) X r) →
            ∀ (r : Fin R),
              Proofs.StableHLO.den (Proofs.StableHLO.mnv4FwdGraphSync_full R hR N epsStr w e r) =
                Proofs.batchShard R N nCls (Proofs.StableHLO.mobilenetv4ForwardB_full (R * N) w X) r :=
  Proofs.StableHLO.mnv4FwdGraphSync_full_shard

/-- `Proofs.MobileNetV4SyncTieB.mnv4_net_syncTiedB` -/
theorem chk_mnv4_net_syncTiedB :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ {nCls : ℕ} (xN cotN vN epsStr : String) (w : Proofs.StableHLO.Mnv4BWeights nCls)
          (X : Proofs.Vec (R * N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))) (G : Proofs.Vec (R * N * nCls))
          (gs : Fin R → Proofs.Vec (N * nCls)),
          (∀ (r : Fin R), gs r = Proofs.batchShard R N nCls (fun (i : Fin (R * N * nCls)) => ↑R * G i) r) →
            Proofs.MobileNetV4SyncTieB.mnv4NetSyncTiedB R hR N xN cotN vN epsStr w X G gs :=
  Proofs.MobileNetV4SyncTieB.mnv4_net_syncTiedB

/-- `Proofs.adamW_at_allReduceMeanF` -/
theorem chk_adamW_at_allReduceMeanF :
    ∀ {n : ℕ} (θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN : String) (ds : List ℕ)
      (β₁ β₂ ε lr wd bc₁ bc₂ : ℝ) (θ m v : Proofs.Vec n) (R : ℕ) (hR : (0 : ℕ) < R) (t : String) (ds' : List ℕ)
      (g : Fin R → Proofs.StableHLO.SHlo n),
      (Proofs.StableHLO.den
            (Proofs.StableHLO.SHlo.adamWParamF θN mN vN b1N ob1N b2N ob2N bc1N bc2N lrN epsN wdN ds β₁ β₂ ε lr wd bc₁ bc₂ θ
              m v (Proofs.StableHLO.SHlo.allReduceMeanF R hR t ds' g)),
          Proofs.StableHLO.den
            (Proofs.StableHLO.SHlo.adamMNextF mN b1N ob1N ds β₁ m (Proofs.StableHLO.SHlo.allReduceMeanF R hR t ds' g)),
          Proofs.StableHLO.den
            (Proofs.StableHLO.SHlo.adamVNextF vN b2N ob2N ds β₂ v (Proofs.StableHLO.SHlo.allReduceMeanF R hR t ds' g))) =
        Proofs.adamWStep β₁ β₂ ε lr wd bc₁ bc₂ θ m v (Proofs.dpMean (R := R) fun (r : Fin R) => Proofs.StableHLO.den (g r)) :=
  Proofs.adamW_at_allReduceMeanF

/-- `Proofs.r34InputGradB_eq_r34B_full_vjp` -/
theorem chk_r34InputGradB_eq_r34B_full_vjp :
    ∀ (N : ℕ) {nCls : ℕ} (Ws : Proofs.Kernel4 (64 : ℕ) (3 : ℕ) (7 : ℕ) (7 : ℕ))
      (bs : Proofs.Vec (64 : ℕ)) (εs : ℝ) (hεs : (0 : ℝ) < εs) (γs βs : Proofs.Vec (64 : ℕ))
      (Wd : Proofs.Mat (512 : ℕ) nCls) (bd : Proofs.Vec nCls)
      (b1 b2 b3 : Proofs.Vec (N * ((64 : ℕ) * (56 : ℕ) * (56 : ℕ))) → Proofs.Vec (N * ((64 : ℕ) * (56 : ℕ) * (56 : ℕ))))
      (b4 : Proofs.Vec (N * ((64 : ℕ) * (56 : ℕ) * (56 : ℕ))) → Proofs.Vec (N * ((128 : ℕ) * (28 : ℕ) * (28 : ℕ))))
      (b5 b6 b7 : Proofs.Vec (N * ((128 : ℕ) * (28 : ℕ) * (28 : ℕ))) → Proofs.Vec (N * ((128 : ℕ) * (28 : ℕ) * (28 : ℕ))))
      (b8 : Proofs.Vec (N * ((128 : ℕ) * (28 : ℕ) * (28 : ℕ))) → Proofs.Vec (N * ((256 : ℕ) * (14 : ℕ) * (14 : ℕ))))
      (b9 b10 b11 b12 b13 :
        Proofs.Vec (N * ((256 : ℕ) * (14 : ℕ) * (14 : ℕ))) → Proofs.Vec (N * ((256 : ℕ) * (14 : ℕ) * (14 : ℕ))))
      (b14 : Proofs.Vec (N * ((256 : ℕ) * (14 : ℕ) * (14 : ℕ))) → Proofs.Vec (N * ((512 : ℕ) * (7 : ℕ) * (7 : ℕ))))
      (b15 b16 : Proofs.Vec (N * ((512 : ℕ) * (7 : ℕ) * (7 : ℕ))) → Proofs.Vec (N * ((512 : ℕ) * (7 : ℕ) * (7 : ℕ))))
      (x : Proofs.Vec (N * ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))))))
      (h_stem : Proofs.R34StemSmoothAt N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs x)
      (h_pool : Proofs.R34PoolSmoothAt N (56 : ℕ) (56 : ℕ) (Proofs.StableHLO.cbReluStridedB N Ws bs εs γs βs x))
      (hb1 : Proofs.HasVJPDiffAt b1 (Proofs.opaqueA0 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) x))
      (hb2 : Proofs.HasVJPDiffAt b2 (Proofs.opaqueA1 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 x))
      (hb3 : Proofs.HasVJPDiffAt b3 (Proofs.opaqueA2 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 x))
      (hb4 : Proofs.HasVJPDiffAt b4 (Proofs.opaqueA3 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 x))
      (hb5 : Proofs.HasVJPDiffAt b5 (Proofs.opaqueA4 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 x))
      (hb6 : Proofs.HasVJPDiffAt b6 (Proofs.opaqueA5 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
      (hb7 :
        Proofs.HasVJPDiffAt b7 (Proofs.opaqueA6 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
      (hb8 :
        Proofs.HasVJPDiffAt b8
          (Proofs.opaqueA7 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
      (hb9 :
        Proofs.HasVJPDiffAt b9
          (Proofs.opaqueA8 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
      (hb10 :
        Proofs.HasVJPDiffAt b10
          (Proofs.opaqueA9 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
      (hb11 :
        Proofs.HasVJPDiffAt b11
          (Proofs.opaqueA10 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
      (hb12 :
        Proofs.HasVJPDiffAt b12
          (Proofs.opaqueA11 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
      (hb13 :
        Proofs.HasVJPDiffAt b13
          (Proofs.opaqueA12 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 x))
      (hb14 :
        Proofs.HasVJPDiffAt b14
          (Proofs.opaqueA13 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13
            x))
      (hb15 :
        Proofs.HasVJPDiffAt b15
          (Proofs.opaqueA14 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13
            b14 x))
      (hb16 :
        Proofs.HasVJPDiffAt b16
          (Proofs.opaqueA15 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12 b13
            b14 b15 x)),
      (Proofs.r34InputGradB N Ws Wd
          (Proofs.HasVJP.backward
            (Proofs.bnBatchLA_has_vjp N (64 : ℕ) ((2 : ℕ) * (56 : ℕ)) ((2 : ℕ) * (56 : ℕ)) εs hεs γs βs)
            (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2 Ws bs) x))
          (Proofs.StableHLO.cbReluStridedB N Ws bs εs γs βs x) hb16.fst.backward hb15.fst.backward hb14.fst.backward
          hb13.fst.backward hb12.fst.backward hb11.fst.backward hb10.fst.backward hb9.fst.backward hb8.fst.backward
          hb7.fst.backward hb6.fst.backward hb5.fst.backward hb4.fst.backward hb3.fst.backward hb2.fst.backward
          hb1.fst.backward fun (i : Fin (N * ((64 : ℕ) * ((2 : ℕ) * (56 : ℕ)) * ((2 : ℕ) * (56 : ℕ))))) =>
          Proofs.StableHLO.bnBatchLA N (64 : ℕ) ((2 : ℕ) * (56 : ℕ)) ((2 : ℕ) * (56 : ℕ)) εs γs βs
              (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2 Ws bs) x) i >
            (0 : ℝ)) =
        (Proofs.r34B_full_has_vjp_at (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11
            b12 b13 b14 b15 b16 (Proofs.r34HeadB N (7 : ℕ) (7 : ℕ) Wd bd) x
            ⟨Proofs.r34StemB_has_vjp_at N (56 : ℕ) (56 : ℕ) Ws bs εs hεs γs βs
                (Mathlib.Meta.NormNum.isNat_lt_true (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (0 : ℕ)))
                  (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (64 : ℕ))) (Eq.refl false))
                (Mathlib.Meta.NormNum.isNat_lt_true (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (0 : ℕ)))
                  (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (56 : ℕ))) (Eq.refl false))
                (Mathlib.Meta.NormNum.isNat_lt_true (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (0 : ℕ)))
                  (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (56 : ℕ))) (Eq.refl false))
                x h_stem h_pool,
              Proofs.r34StemB_differentiableAt N (56 : ℕ) (56 : ℕ) Ws bs εs hεs γs βs
                (Mathlib.Meta.NormNum.isNat_lt_true (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (0 : ℕ)))
                  (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (64 : ℕ))) (Eq.refl false))
                (Mathlib.Meta.NormNum.isNat_lt_true (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (0 : ℕ)))
                  (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (56 : ℕ))) (Eq.refl false))
                (Mathlib.Meta.NormNum.isNat_lt_true (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (0 : ℕ)))
                  (Mathlib.Meta.NormNum.isNat_ofNat ℕ (Eq.refl (56 : ℕ))) (Eq.refl false))
                x h_stem h_pool⟩
            hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16
            ⟨(Proofs.r34HeadB_has_vjp N (7 : ℕ) (7 : ℕ) Wd bd).toHasVJPAt
                (Proofs.opaqueA16 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11
                  b12 b13 b14 b15 b16 x),
              Proofs.r34HeadB_differentiable N (7 : ℕ) (7 : ℕ) Wd bd
                (Proofs.opaqueA16 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11
                  b12 b13 b14 b15 b16 x)⟩).backward :=
  Proofs.r34InputGradB_eq_r34B_full_vjp

/-- `Proofs.efficientnetInputGradB_full_correct` -/
theorem chk_efficientnetInputGradB_full_correct :
    ∀ (N : ℕ) (w : Proofs.B0Weights) (hεw : w.EpsPos)
      (x : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))) (dy : Proofs.Vec (N * (10 : ℕ)))
      (i : Fin (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
      Proofs.efficientnetInputGradB_full N w.sW w.hW w.fcW
          (Proofs.HasVJP.backward (f := Proofs.StableHLO.bnBatchLA N (32 : ℕ) (112 : ℕ) (112 : ℕ) w.sε w.sγ w.sβ)
            (Proofs.bnBatchLA_has_vjp N (32 : ℕ) (112 : ℕ) (112 : ℕ) w.sε hεw.s w.sγ w.sβ)
            (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2Xla w.sW w.sb) x))
          (Proofs.HasVJP.backward (Proofs.swish_has_vjp (N * ((32 : ℕ) * (112 : ℕ) * (112 : ℕ))))
            (Proofs.StableHLO.bnBatchLA N (32 : ℕ) (112 : ℕ) (112 : ℕ) w.sε w.sγ w.sβ
              (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2Xla w.sW w.sb) x)))
          (Proofs.HasVJP.backward (f := Proofs.StableHLO.bnBatchLA N (1280 : ℕ) (7 : ℕ) (7 : ℕ) w.hε w.hγ w.hβ)
            (Proofs.bnBatchLA_has_vjp N (1280 : ℕ) (7 : ℕ) (7 : ℕ) w.hε hεw.h w.hγ w.hβ)
            (Proofs.StableHLO.batchMap N (Proofs.flatConv w.hW w.hb)
              (Proofs.opaqueA16 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
                (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
                (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
                (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
                (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
                (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
                (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13)
                (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b14) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b15)
                (Proofs.mbExpW N (7 : ℕ) (7 : ℕ) w.b16) x)))
          (Proofs.HasVJP.backward (Proofs.swish_has_vjp (N * ((1280 : ℕ) * (7 : ℕ) * (7 : ℕ))))
            (Proofs.StableHLO.bnBatchLA N (1280 : ℕ) (7 : ℕ) (7 : ℕ) w.hε w.hγ w.hβ
              (Proofs.StableHLO.batchMap N (Proofs.flatConv w.hW w.hb)
                (Proofs.opaqueA16 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
                  (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
                  (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
                  (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
                  (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
                  (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
                  (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13)
                  (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b14) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b15)
                  (Proofs.mbExpW N (7 : ℕ) (7 : ℕ) w.b16) x))))
          ((Proofs.mbNoExpW_has_vjp N (112 : ℕ) (112 : ℕ) w.b1 hεw.b1.d hεw.b1.p).backward
            (Proofs.opaqueA0 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) x))
          ((Proofs.mbStridedW_has_vjp N (56 : ℕ) (56 : ℕ) w.b2 hεw.b2.e hεw.b2.d hεw.b2.p).backward
            (Proofs.opaqueA1 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1) x))
          ((Proofs.mbResidW_has_vjp N (56 : ℕ) (56 : ℕ) w.b3 hεw.b3.e hεw.b3.d hεw.b3.p).backward
            (Proofs.opaqueA2 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) x))
          ((Proofs.mbStridedW_has_vjp N (28 : ℕ) (28 : ℕ) w.b4 hεw.b4.e hεw.b4.d hεw.b4.p).backward
            (Proofs.opaqueA3 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3) x))
          ((Proofs.mbResidW_has_vjp N (28 : ℕ) (28 : ℕ) w.b5 hεw.b5.e hεw.b5.d hεw.b5.p).backward
            (Proofs.opaqueA4 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) x))
          ((Proofs.mbStridedW_has_vjp N (14 : ℕ) (14 : ℕ) w.b6 hεw.b6.e hεw.b6.d hεw.b6.p).backward
            (Proofs.opaqueA5 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5) x))
          ((Proofs.mbResidW_has_vjp N (14 : ℕ) (14 : ℕ) w.b7 hεw.b7.e hεw.b7.d hεw.b7.p).backward
            (Proofs.opaqueA6 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) x))
          ((Proofs.mbResidW_has_vjp N (14 : ℕ) (14 : ℕ) w.b8 hεw.b8.e hεw.b8.d hεw.b8.p).backward
            (Proofs.opaqueA7 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7) x))
          ((Proofs.mbExpW_has_vjp N (14 : ℕ) (14 : ℕ) w.b9 hεw.b9.e hεw.b9.d hεw.b9.p).backward
            (Proofs.opaqueA8 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) x))
          ((Proofs.mbResidW_has_vjp N (14 : ℕ) (14 : ℕ) w.b10 hεw.b10.e hεw.b10.d hεw.b10.p).backward
            (Proofs.opaqueA9 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9) x))
          ((Proofs.mbResidW_has_vjp N (14 : ℕ) (14 : ℕ) w.b11 hεw.b11.e hεw.b11.d hεw.b11.p).backward
            (Proofs.opaqueA10 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) x))
          ((Proofs.mbStridedW_has_vjp N (7 : ℕ) (7 : ℕ) w.b12 hεw.b12.e hεw.b12.d hεw.b12.p).backward
            (Proofs.opaqueA11 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11) x))
          ((Proofs.mbResidW_has_vjp N (7 : ℕ) (7 : ℕ) w.b13 hεw.b13.e hεw.b13.d hεw.b13.p).backward
            (Proofs.opaqueA12 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) x))
          ((Proofs.mbResidW_has_vjp N (7 : ℕ) (7 : ℕ) w.b14 hεw.b14.e hεw.b14.d hεw.b14.p).backward
            (Proofs.opaqueA13 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13) x))
          ((Proofs.mbResidW_has_vjp N (7 : ℕ) (7 : ℕ) w.b15 hεw.b15.e hεw.b15.d hεw.b15.p).backward
            (Proofs.opaqueA14 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13)
              (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b14) x))
          ((Proofs.mbExpW_has_vjp N (7 : ℕ) (7 : ℕ) w.b16 hεw.b16.e hεw.b16.d hεw.b16.p).backward
            (Proofs.opaqueA15 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13)
              (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b14) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b15) x))
          dy i =
        ∑ j : Fin (N * (10 : ℕ)), Proofs.pdiv (Proofs.efficientnetForwardB_full N w) x i j * dy j :=
  Proofs.efficientnetInputGradB_full_correct

/-- `Proofs.convnextImagenetInputGradB_eq_vjp` -/
theorem chk_convnextImagenetInputGradB_eq_vjp :
    ∀ (B : ℕ) (w : Proofs.CnxTWeightsCh (1000 : ℕ)) (hsε : (0 : ℝ) < w.sε)
      (h1 : ∀ (i : Fin (3 : ℕ)), (0 : ℝ) < (w.s1 i).εn) (hd1 : (0 : ℝ) < w.d1.ε)
      (h2 : ∀ (i : Fin (3 : ℕ)), (0 : ℝ) < (w.s2 i).εn) (hd2 : (0 : ℝ) < w.d2.ε)
      (h3 : ∀ (i : Fin (9 : ℕ)), (0 : ℝ) < (w.s3 i).εn) (hd3 : (0 : ℝ) < w.d3.ε)
      (h4 : ∀ (i : Fin (3 : ℕ)), (0 : ℝ) < (w.s4 i).εn) (hhε : (0 : ℝ) < w.hε)
      (x : Proofs.Vec (B * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
      Proofs.convnextInputGradB B w.Wd (Proofs.padOdd w.sW) (Proofs.chanLNTensor3Back (96 : ℕ) (56 : ℕ) (56 : ℕ) w.sε w.sγ)
          (Proofs.cnxSavedB0 B w x) (Proofs.rowLNVecFlatBack (1 : ℕ) (768 : ℕ) w.hε w.hγ) (Proofs.cnxSavedB9 B w x)
          (Proofs.cnxStageChKBack (3 : ℕ) w.s1) (Proofs.cnxSavedB1 B w x)
          (fun (u : Proofs.Vec ((96 : ℕ) * (56 : ℕ) * (56 : ℕ))) =>
            Proofs.cnxDownBack (Proofs.padOdd w.d1.W) (Proofs.chanLNTensor3Back (96 : ℕ) (56 : ℕ) (56 : ℕ) w.d1.ε w.d1.γ u))
          (Proofs.cnxSavedB2 B w x) (Proofs.cnxStageChKBack (3 : ℕ) w.s2) (Proofs.cnxSavedB3 B w x)
          (fun (u : Proofs.Vec ((192 : ℕ) * (28 : ℕ) * (28 : ℕ))) =>
            Proofs.cnxDownBack (Proofs.padOdd w.d2.W)
              (Proofs.chanLNTensor3Back (192 : ℕ) (28 : ℕ) (28 : ℕ) w.d2.ε w.d2.γ u))
          (Proofs.cnxSavedB4 B w x) (Proofs.cnxStageChKBack (9 : ℕ) w.s3) (Proofs.cnxSavedB5 B w x)
          (fun (u : Proofs.Vec ((384 : ℕ) * (14 : ℕ) * (14 : ℕ))) =>
            Proofs.cnxDownBack (Proofs.padOdd w.d3.W)
              (Proofs.chanLNTensor3Back (384 : ℕ) (14 : ℕ) (14 : ℕ) w.d3.ε w.d3.γ u))
          (Proofs.cnxSavedB6 B w x) (Proofs.cnxStageChKBack (3 : ℕ) w.s4) (Proofs.cnxSavedB7 B w x) =
        Proofs.HasVJP.backward
          (Proofs.batchMap_has_vjp
            (Proofs.dense w.Wd w.bd ∘
              Proofs.rowLNVecFlat (1 : ℕ) (768 : ℕ) w.hε w.hγ w.hβ ∘
                Proofs.globalAvgPoolFlat (768 : ℕ) (7 : ℕ) (7 : ℕ) ∘
                  Proofs.convNextStageChK (3 : ℕ) w.s4 ∘
                    Proofs.cnxDownChW (7 : ℕ) (7 : ℕ) w.d3 ∘
                      Proofs.convNextStageChK (9 : ℕ) w.s3 ∘
                        Proofs.cnxDownChW (14 : ℕ) (14 : ℕ) w.d2 ∘
                          Proofs.convNextStageChK (3 : ℕ) w.s2 ∘
                            Proofs.cnxDownChW (28 : ℕ) (28 : ℕ) w.d1 ∘
                              Proofs.convNextStageChK (3 : ℕ) w.s1 ∘
                                Proofs.chanLNTensor3 (96 : ℕ) (56 : ℕ) (56 : ℕ) w.sε w.sγ w.sβ ∘
                                  Proofs.flatConvStride4 w.sW w.sb)
            (Proofs.convNextForwardTCh_has_vjp w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
            (Proofs.convNextForwardTCh_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε))
          x :=
  Proofs.convnextImagenetInputGradB_eq_vjp

/-- `Proofs.FloatModel.linear_e4m3_argmax_preserved` -/
theorem chk_linear_e4m3_argmax_preserved :
    ∀ (M L : Proofs.FloatModel),
      M.u ≤ Proofs.u32 →
        L.u ≤ Proofs.u_e4m3 →
          ∀ {n : ℕ} {W : Proofs.Mat (784 : ℕ) n} {b : Proofs.Vec n} {x : Proofs.Vec (784 : ℕ)},
            (∀ (i : Fin (784 : ℕ)) (j : Fin n), |W i j| ≤ (3 / 5 : ℝ)) →
              (∀ (j : Fin n), |b j| ≤ (1 : ℝ)) →
                (∀ (i : Fin (784 : ℕ)), |x i| ≤ (1 : ℝ)) →
                  ∀ (k : Fin n),
                    (∀ (i : Fin n), i ≠ k → (122 : ℝ) < Proofs.dense W b x k - Proofs.dense W b x i) →
                      ∀ (i : Fin n), i ≠ k → M.denseMixed L W b x i < M.denseMixed L W b x k :=
  Proofs.FloatModel.linear_e4m3_argmax_preserved

/-- `Proofs.TrainedLinearDescent.trained_linear_sgd_strictly_descends` -/
theorem chk_trained_linear_sgd_strictly_descends :
    Proofs.crossEntropy (10 : ℕ)
        (Proofs.dense
          (Proofs.Mat.unflatten
            (Proofs.TrainedLinearDescent.Wd.flatten -
              (1 / 8192 : ℝ) •
                Proofs.binary32.linearFloatGrad Proofs.TrainedLinearDescent.Wd Proofs.TrainedLinearDescent.bd
                  Proofs.TrainedLinearDescent.xd Real.exp Proofs.TrainedLinearDescent.lblD))
          Proofs.TrainedLinearDescent.bd Proofs.TrainedLinearDescent.xd)
        Proofs.TrainedLinearDescent.lblD <
      Proofs.crossEntropy (10 : ℕ)
        (Proofs.dense (Proofs.Mat.unflatten Proofs.TrainedLinearDescent.Wd.flatten) Proofs.TrainedLinearDescent.bd
          Proofs.TrainedLinearDescent.xd)
        Proofs.TrainedLinearDescent.lblD :=
  Proofs.TrainedLinearDescent.trained_linear_sgd_strictly_descends

/-- `Proofs.lipschitz_margin_certified_radius` -/
theorem chk_lipschitz_margin_certified_radius :
    ∀ {k : ℕ} {E : Type u_1} [inst : NormedAddCommGroup E]
      {f : E → EuclideanSpace ℝ (Fin k)} {L : ℝ},
      Proofs.LipschitzL2 L f →
        (0 : ℝ) < L →
          ∀ {x δ : E} {i : Fin k} {m : ℝ},
            (∀ (j : Fin k), j ≠ i → m ≤ (f x).ofLp i - (f x).ofLp j) →
              ‖δ‖ < m / (√(2 : ℝ) * L) → ∀ (j : Fin k), j ≠ i → (f (x + δ)).ofLp j < (f (x + δ)).ofLp i :=
  Proofs.lipschitz_margin_certified_radius

/-- `Proofs.LipschitzCertDemo.scorecard_sdp` -/
theorem chk_scorecard_sdp :
    Proofs.LipschitzCertDemo.sdpCappedCerts.length = (8 : ℕ) ∧
      ∀ p ∈ Proofs.LipschitzCertDemo.sdpCappedCerts,
        Proofs.LipschitzCertDemo.CertifiedAt Proofs.LipschitzCertDemo.mlpS (1 / 10 : ℝ) p.2.1 p.2.2 :=
  Proofs.LipschitzCertDemo.scorecard_sdp

/-- `Proofs.smoothing_certified_radius_classifier` -/
theorem chk_smoothing_certified_radius_classifier :
    ∀ {n k : ℕ} {σ : ℝ},
      (0 : ℝ) < σ →
        ∀ {C : EuclideanSpace ℝ (Fin (n + (1 : ℕ))) → Fin k},
          Measurable C →
            (∀ (c : Fin k) (x : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))),
                (@MeasureTheory.integral _ _ _ _
                    (WithLp.measurableSpace (2 : ENNReal) ((i : Fin (n + (1 : ℕ))) → (fun (x : Fin (n + (1 : ℕ))) => ℝ) i))
                    (ProbabilityTheory.stdGaussian (EuclideanSpace ℝ (Fin (n + (1 : ℕ)))))
                    fun (z : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))) => if C (x + σ • z) = c then (1 : ℝ) else (0 : ℝ)) ∈
                  Set.Ioo (0 : ℝ) (1 : ℝ)) →
              ∀ {x δ : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))} {i : Fin k},
                ‖δ‖ <
                    σ *
                      Proofs.stdNormalQuantile
                        (@MeasureTheory.integral _ _ _ _
                          (WithLp.measurableSpace (2 : ENNReal)
                            ((i : Fin (n + (1 : ℕ))) → (fun (x : Fin (n + (1 : ℕ))) => ℝ) i))
                          (ProbabilityTheory.stdGaussian (EuclideanSpace ℝ (Fin (n + (1 : ℕ)))))
                          fun (z : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))) =>
                          if C (x + σ • z) = i then (1 : ℝ) else (0 : ℝ)) →
                  ∀ (j : Fin k),
                    j ≠ i →
                      LT.lt (α := ℝ)
                        (@MeasureTheory.integral _ _ _ _
                          (WithLp.measurableSpace (2 : ENNReal)
                            ((i : Fin (n + (1 : ℕ))) → (fun (x : Fin (n + (1 : ℕ))) => ℝ) i))
                          (ProbabilityTheory.stdGaussian (EuclideanSpace ℝ (Fin (n + (1 : ℕ)))))
                          fun (z : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))) =>
                          if C (x + δ + σ • z) = j then (1 : ℝ) else (0 : ℝ))
                        (@MeasureTheory.integral _ _ _ _
                          (WithLp.measurableSpace (2 : ENNReal)
                            ((i : Fin (n + (1 : ℕ))) → (fun (x : Fin (n + (1 : ℕ))) => ℝ) i))
                          (ProbabilityTheory.stdGaussian (EuclideanSpace ℝ (Fin (n + (1 : ℕ)))))
                          fun (z : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))) =>
                          if C (x + δ + σ • z) = i then (1 : ℝ) else (0 : ℝ)) :=
  Proofs.smoothing_certified_radius_classifier

/-- `Proofs.MuonGeometry.shampoo_eq_muon` -/
theorem chk_shampoo_eq_muon :
    ∀ {n : ℕ} (U V : Matrix (Fin n) (Fin n) ℝ) (s : Fin n → ℝ),
      U.transpose * U = (1 : Matrix (Fin n) (Fin n) ℝ) →
        V.transpose * V = (1 : Matrix (Fin n) (Fin n) ℝ) →
          (∀ (i : Fin n), (0 : ℝ) < s i) →
            ((V * Matrix.diagonal fun (i : Fin n) => (√(s i))⁻¹) * V.transpose) ^ (4 : ℕ) *
                  ((U * Matrix.diagonal s * V.transpose).transpose * (U * Matrix.diagonal s * V.transpose)) =
                (1 : Matrix (Fin n) (Fin n) ℝ) ∧
              ((U * Matrix.diagonal fun (i : Fin n) => (√(s i))⁻¹) * U.transpose) ^ (4 : ℕ) *
                    (U * Matrix.diagonal s * V.transpose * (U * Matrix.diagonal s * V.transpose).transpose) =
                  (1 : Matrix (Fin n) (Fin n) ℝ) ∧
                (U * Matrix.diagonal fun (i : Fin n) => (√(s i))⁻¹) * U.transpose * (U * Matrix.diagonal s * V.transpose) *
                    ((V * Matrix.diagonal fun (i : Fin n) => (√(s i))⁻¹) * V.transpose) =
                  U * V.transpose :=
  Proofs.MuonGeometry.shampoo_eq_muon
