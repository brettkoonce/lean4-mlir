import LeanMlir.Proofs.Architectures.BatchNorm
import LeanMlir.Proofs.Certificates.LipschitzCert
import LeanMlir.Proofs.Certificates.LipschitzCertScorecardSDP
import LeanMlir.Proofs.Certificates.SmoothingGaussian
import LeanMlir.Proofs.Float.FloatBridge
import LeanMlir.Proofs.Foundation.DataParallel
import LeanMlir.Proofs.Foundation.DataParallelNode
import LeanMlir.Proofs.Foundation.MuonGeometry
import LeanMlir.Proofs.Foundation.SmoothedLossCot
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtStepTieGB
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtWholeBackCertifiedTieB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetFullWholeBackCertifiedTie
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2FullBSeal
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullBSeal
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4FullB
import LeanMlir.Proofs.Nets.ResNet.ResNet34BackCertifiedTieB
import LeanMlir.Proofs.Nets.ResNet.ResNet34FoldB
import LeanMlir.Proofs.Nets.ResNet.ResNet50FullBVJP
import LeanMlir.Proofs.Nets.ResNet.ResNet50StepTieB
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
      (w : Proofs.R50BWeights nCls) (hsε : (0 : ℝ) < w.sε) (qs1b0 : Proofs.R50ProjPos w.s1b0)
      (qs1b1 : Proofs.R50IdPos w.s1b1) (qs1b2 : Proofs.R50IdPos w.s1b2) (qs2b0 : Proofs.R50ProjPos w.s2b0)
      (qs2b1 : Proofs.R50IdPos w.s2b1) (qs2b2 : Proofs.R50IdPos w.s2b2) (qs2b3 : Proofs.R50IdPos w.s2b3)
      (qs3b0 : Proofs.R50ProjPos w.s3b0) (qs3b1 : Proofs.R50IdPos w.s3b1) (qs3b2 : Proofs.R50IdPos w.s3b2)
      (qs3b3 : Proofs.R50IdPos w.s3b3) (qs3b4 : Proofs.R50IdPos w.s3b4) (qs3b5 : Proofs.R50IdPos w.s3b5)
      (qs4b0 : Proofs.R50ProjPos w.s4b0) (qs4b1 : Proofs.R50IdPos w.s4b1) (qs4b2 : Proofs.R50IdPos w.s4b2)
      (x :
        Proofs.Vec
          (N *
            ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
              ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))))))
      (h_stem :
        Proofs.R34StemSmoothAt N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) w.sW w.sb w.sε
          w.sγ w.sβ x)
      (h_pool :
        Proofs.R34PoolSmoothAt N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q)))
          (Proofs.StableHLO.cbReluStridedB N w.sW w.sb w.sε w.sγ w.sβ x))
      (ss1b0 :
        Proofs.R50ProjSmoothAt N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) w.s1b0
          (Proofs.r50Pre0 N q w x))
      (ss1b1 :
        Proofs.R50IdSmoothAt N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) w.s1b1
          (Proofs.r50Pre1 N q w x))
      (ss1b2 :
        Proofs.R50IdSmoothAt N ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))) w.s1b2
          (Proofs.r50Pre2 N q w x))
      (ss2b0 : Proofs.R50DownSmoothAt N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) w.s2b0 (Proofs.r50Pre3 N q w x))
      (ss2b1 : Proofs.R50IdSmoothAt N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) w.s2b1 (Proofs.r50Pre4 N q w x))
      (ss2b2 : Proofs.R50IdSmoothAt N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) w.s2b2 (Proofs.r50Pre5 N q w x))
      (ss2b3 : Proofs.R50IdSmoothAt N ((2 : ℕ) * ((2 : ℕ) * q)) ((2 : ℕ) * ((2 : ℕ) * q)) w.s2b3 (Proofs.r50Pre6 N q w x))
      (ss3b0 : Proofs.R50DownSmoothAt N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b0 (Proofs.r50Pre7 N q w x))
      (ss3b1 : Proofs.R50IdSmoothAt N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b1 (Proofs.r50Pre8 N q w x))
      (ss3b2 : Proofs.R50IdSmoothAt N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b2 (Proofs.r50Pre9 N q w x))
      (ss3b3 : Proofs.R50IdSmoothAt N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b3 (Proofs.r50Pre10 N q w x))
      (ss3b4 : Proofs.R50IdSmoothAt N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b4 (Proofs.r50Pre11 N q w x))
      (ss3b5 : Proofs.R50IdSmoothAt N ((2 : ℕ) * q) ((2 : ℕ) * q) w.s3b5 (Proofs.r50Pre12 N q w x))
      (ss4b0 : Proofs.R50DownSmoothAt N q q w.s4b0 (Proofs.r50Pre13 N q w x))
      (ss4b1 : Proofs.R50IdSmoothAt N q q w.s4b1 (Proofs.r50Pre14 N q w x))
      (ss4b2 : Proofs.R50IdSmoothAt N q q w.s4b2 (Proofs.r50Pre15 N q w x)) (dy : Proofs.Vec (N * nCls))
      (i :
        Fin
          (N *
            ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
              ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q)))))))),
      (Proofs.resnet50ForwardB_full_has_vjp_at N q hq0 w hsε qs1b0 qs1b1 qs1b2 qs2b0 qs2b1 qs2b2 qs2b3 qs3b0 qs3b1 qs3b2
              qs3b3 qs3b4 qs3b5 qs4b0 qs4b1 qs4b2 x h_stem h_pool ss1b0 ss1b1 ss1b2 ss2b0 ss2b1 ss2b2 ss2b3 ss3b0 ss3b1
              ss3b2 ss3b3 ss3b4 ss3b5 ss4b0 ss4b1 ss4b2).backward
          dy i =
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
      (Wc :
        Proofs.Kernel4 (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 16) (instOfNatNat (nat_lit 16)))
          (@OfNat.ofNat Nat (nat_lit 16) (instOfNatNat (nat_lit 16))))
      (bc cls : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (pos :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (γF βF : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (Wcls :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))))
      (bcls : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))))
      (lnG1_1 lnB1_1 lnG2_1 lnB2_1 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_1 mWk_1 mWv_1 mWo_1 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_1 mbk_1 mbv_1 mbo_1 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_1 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_1 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_1 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_1 lnG1_2 lnB1_2 lnG2_2 lnB2_2 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_2 mWk_2 mWv_2 mWo_2 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_2 mbk_2 mbv_2 mbo_2 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_2 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_2 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_2 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_2 lnG1_3 lnB1_3 lnG2_3 lnB2_3 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_3 mWk_3 mWv_3 mWo_3 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_3 mbk_3 mbv_3 mbo_3 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_3 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_3 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_3 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_3 lnG1_4 lnB1_4 lnG2_4 lnB2_4 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_4 mWk_4 mWv_4 mWo_4 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_4 mbk_4 mbv_4 mbo_4 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_4 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_4 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_4 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_4 lnG1_5 lnB1_5 lnG2_5 lnB2_5 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_5 mWk_5 mWv_5 mWo_5 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_5 mbk_5 mbv_5 mbo_5 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_5 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_5 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_5 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_5 lnG1_6 lnB1_6 lnG2_6 lnB2_6 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_6 mWk_6 mWv_6 mWo_6 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_6 mbk_6 mbv_6 mbo_6 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_6 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_6 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_6 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_6 lnG1_7 lnB1_7 lnG2_7 lnB2_7 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_7 mWk_7 mWv_7 mWo_7 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_7 mbk_7 mbv_7 mbo_7 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_7 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_7 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_7 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_7 lnG1_8 lnB1_8 lnG2_8 lnB2_8 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_8 mWk_8 mWv_8 mWo_8 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_8 mbk_8 mbv_8 mbo_8 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_8 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_8 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_8 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_8 lnG1_9 lnB1_9 lnG2_9 lnB2_9 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_9 mWk_9 mWv_9 mWo_9 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_9 mbk_9 mbv_9 mbo_9 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_9 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_9 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_9 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_9 lnG1_10 lnB1_10 lnG2_10 lnB2_10 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_10 mWk_10 mWv_10 mWo_10 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_10 mbk_10 mbv_10 mbo_10 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_10 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_10 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_10 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_10 lnG1_11 lnB1_11 lnG2_11 lnB2_11 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_11 mWk_11 mWv_11 mWo_11 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_11 mbk_11 mbv_11 mbo_11 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_11 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_11 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_11 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_11 lnG1_12 lnB1_12 lnG2_12 lnB2_12 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mWq_12 mWk_12 mWv_12 mWo_12 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (mbq_12 mbk_12 mbv_12 mbo_12 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fW1_12 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fb1_12 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))))
      (fW2_12 :
        Proofs.Mat (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
      (fb2_12 : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))))
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
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) Wc bc cls pos img;
      have ib2 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1
          mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1;
      have ib3 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2
          mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2;
      have ib4 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3
          mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3;
      have ib5 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4
          mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4;
      have ib6 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5
          mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5;
      have ib7 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6
          mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6;
      have ib8 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7
          mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7;
      have ib9 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8
          mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8;
      have ib10 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9
          mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 fb2_9 ib9;
      have ib11 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10
          mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 fb2_10 ib10;
      have ib12 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11
          mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 fb2_11 ib11;
      have b12out :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockFwdOMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12
          mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 fb2_12 ib12;
      have fl :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.Mat.flatten (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          fun (r : Fin (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))) =>
          Proofs.layerNormVec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) ε γF βF
            (@Proofs.Mat.unflatten (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
              (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) b12out r);
      have hn :=
        Proofs.StableHLO.clsSliceFlat (@OfNat.ofNat Nat (nat_lit 196) (instOfNatNat (nat_lit 196)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) fl;
      have logits :=
        @Proofs.dense (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) Wcls bcls hn;
      have g : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) :=
        fun (c : Fin (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10)))) =>
        @HSub.hSub Real Real Real (@instHSub Real Real.instSub)
          (Proofs.softmax (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) logits c)
          (Proofs.oneHot (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) label c);
      have dy12 :=
        Proofs.vitCotB2outV (@OfNat.ofNat Nat (nat_lit 196) (instOfNatNat (nat_lit 196)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) ε γF Wcls b12out g;
      have dy11 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12
          mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12 fW1_12 fb1_12 fW2_12 ib12 dy12;
      have dy10 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11
          mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11 fb1_11 fW2_11 ib11 dy11;
      have dy9 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10
          mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10 fb1_10 fW2_10 ib10 dy10;
      have dy8 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9
          mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9 ib9 dy9;
      have dy7 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_8 lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8
          mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 ib8 dy8;
      have dy6 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_7 lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7
          mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 ib7 dy7;
      have dy5 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_6 lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6
          mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 ib6 dy6;
      have dy4 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_5 lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5
          mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 ib5 dy5;
      have dy3 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_4 lnB1_4 lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4
          mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 ib4 dy4;
      have dy2 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_3 lnB1_3 lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3
          mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 ib3 dy3;
      have dy1 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_2 lnB1_2 lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2
          mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 ib2 dy2;
      have dyEmbed :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.ViTTiePoC.vitBlockCotInAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) ε lnG1_1 lnB1_1 lnG2_1 lnB2_1 mWq_1 mWk_1 mWv_1
          mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 ib1 dy1;
      And
        (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε lnG1_1 lnB1_1 lnG2_1
          lnB2_1 mWq_1 mWk_1 mWv_1 mWo_1 mbq_1 mbk_1 mbv_1 mbo_1 fW1_1 fb1_1 fW2_1 fb2_1 ib1 dy1 lr)
        (And
          (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
            (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
            (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε lnG1_2 lnB1_2
            lnG2_2 lnB2_2 mWq_2 mWk_2 mWv_2 mWo_2 mbq_2 mbk_2 mbv_2 mbo_2 fW1_2 fb1_2 fW2_2 fb2_2 ib2 dy2 lr)
          (And
            (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
              (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
              (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
              (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε lnG1_3 lnB1_3
              lnG2_3 lnB2_3 mWq_3 mWk_3 mWv_3 mWo_3 mbq_3 mbk_3 mbv_3 mbo_3 fW1_3 fb1_3 fW2_3 fb2_3 ib3 dy3 lr)
            (And
              (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε lnG1_4 lnB1_4
                lnG2_4 lnB2_4 mWq_4 mWk_4 mWv_4 mWo_4 mbq_4 mbk_4 mbv_4 mbo_4 fW1_4 fb1_4 fW2_4 fb2_4 ib4 dy4 lr)
              (And
                (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                  (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                  (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                  (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε lnG1_5
                  lnB1_5 lnG2_5 lnB2_5 mWq_5 mWk_5 mWv_5 mWo_5 mbq_5 mbk_5 mbv_5 mbo_5 fW1_5 fb1_5 fW2_5 fb2_5 ib5 dy5 lr)
                (And
                  (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                    (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                    (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                    (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε lnG1_6
                    lnB1_6 lnG2_6 lnB2_6 mWq_6 mWk_6 mWv_6 mWo_6 mbq_6 mbk_6 mbv_6 mbo_6 fW1_6 fb1_6 fW2_6 fb2_6 ib6 dy6 lr)
                  (And
                    (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                      (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                      (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                      (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε lnG1_7
                      lnB1_7 lnG2_7 lnB2_7 mWq_7 mWk_7 mWv_7 mWo_7 mbq_7 mbk_7 mbv_7 mbo_7 fW1_7 fb1_7 fW2_7 fb2_7 ib7 dy7
                      lr)
                    (And
                      (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                        (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                        (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                        (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε lnG1_8
                        lnB1_8 lnG2_8 lnB2_8 mWq_8 mWk_8 mWv_8 mWo_8 mbq_8 mbk_8 mbv_8 mbo_8 fW1_8 fb1_8 fW2_8 fb2_8 ib8 dy8
                        lr)
                      (And
                        (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε
                          lnG1_9 lnB1_9 lnG2_9 lnB2_9 mWq_9 mWk_9 mWv_9 mWo_9 mbq_9 mbk_9 mbv_9 mbo_9 fW1_9 fb1_9 fW2_9
                          fb2_9 ib9 dy9 lr)
                        (And
                          (@Proofs.ViTTiePoC.vitBlockTiedAtMHV (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                            (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                            (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                            (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε
                            lnG1_10 lnB1_10 lnG2_10 lnB2_10 mWq_10 mWk_10 mWv_10 mWo_10 mbq_10 mbk_10 mbv_10 mbo_10 fW1_10
                            fb1_10 fW2_10 fb2_10 ib10 dy10 lr)
                          (And
                            (@Proofs.ViTTiePoC.vitBlockTiedAtMHV
                              (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                              (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                              (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                              (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN ε
                              lnG1_11 lnB1_11 lnG2_11 lnB2_11 mWq_11 mWk_11 mWv_11 mWo_11 mbq_11 mbk_11 mbv_11 mbo_11 fW1_11
                              fb1_11 fW2_11 fb2_11 ib11 dy11 lr)
                            (And
                              (@Proofs.ViTTiePoC.vitBlockTiedAtMHV
                                (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                                (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                                (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                                (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768))) xN wN bN gN epsStr lrStr cotN
                                ε lnG1_12 lnB1_12 lnG2_12 lnB2_12 mWq_12 mWk_12 mWv_12 mWo_12 mbq_12 mbk_12 mbv_12 mbo_12
                                fW1_12 fb1_12 fW2_12 fb2_12 ib12 dy12 lr)
                              (And (Proofs.ViTTiePoC.vitFinalLNTied gN xN bN epsStr lrStr cotN ε γF βF Wcls b12out g lr)
                                (And (Proofs.ViTTiePoC.vitHeadTied aN wN bN lrStr cotN hn Wcls bcls g lr)
                                  (Proofs.ViTTiePoC.vitEmbedTied wN xN bN clsN pN lrStr cotN Wc bc cls pos img dyEmbed
                                    lr)))))))))))))) :=
  Proofs.ViTTiePoC.vit_net_tied_certified

/-- `Proofs.CnxTiePoCGB.cnx_net_tiedGB` -/
theorem chk_cnx_net_tiedGB :
    ∀ (N : ℕ) {nC : ℕ} (xN epsStr cotN dN aStr negAK bStr logN ohN : String) (ε α B : ℝ)
      (Wst : Proofs.Kernel4 (96 : ℕ) (3 : ℕ) (4 : ℕ) (4 : ℕ)) (psb psng psnbt : Proofs.Vec (96 : ℕ))
      (xstem : Proofs.Vec (N * ((3 : ℕ) * (56 : ℕ) * (56 : ℕ)))) (aW1 : Proofs.DepthwiseKernel (96 : ℕ) (7 : ℕ) (7 : ℕ))
      (aB1 nG1 nB1 : Proofs.Vec (96 : ℕ)) (eW1 : Proofs.Kernel4 (384 : ℕ) (96 : ℕ) (1 : ℕ) (1 : ℕ))
      (eB1 : Proofs.Vec (384 : ℕ)) (pW1 : Proofs.Kernel4 (96 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (pB1 sL1 : Proofs.Vec (96 : ℕ))
      (aW2 : Proofs.DepthwiseKernel (96 : ℕ) (7 : ℕ) (7 : ℕ)) (aB2 nG2 nB2 : Proofs.Vec (96 : ℕ))
      (eW2 : Proofs.Kernel4 (384 : ℕ) (96 : ℕ) (1 : ℕ) (1 : ℕ)) (eB2 : Proofs.Vec (384 : ℕ))
      (pW2 : Proofs.Kernel4 (96 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (pB2 sL2 : Proofs.Vec (96 : ℕ))
      (aW3 : Proofs.DepthwiseKernel (96 : ℕ) (7 : ℕ) (7 : ℕ)) (aB3 nG3 nB3 : Proofs.Vec (96 : ℕ))
      (eW3 : Proofs.Kernel4 (384 : ℕ) (96 : ℕ) (1 : ℕ) (1 : ℕ)) (eB3 : Proofs.Vec (384 : ℕ))
      (pW3 : Proofs.Kernel4 (96 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (pB3 sL3 dG0 dT0 : Proofs.Vec (96 : ℕ))
      (dW0 : Proofs.Kernel4 (192 : ℕ) (96 : ℕ) (2 : ℕ) (2 : ℕ)) (dB0 : Proofs.Vec (192 : ℕ))
      (aW4 : Proofs.DepthwiseKernel (192 : ℕ) (7 : ℕ) (7 : ℕ)) (aB4 nG4 nB4 : Proofs.Vec (192 : ℕ))
      (eW4 : Proofs.Kernel4 (768 : ℕ) (192 : ℕ) (1 : ℕ) (1 : ℕ)) (eB4 : Proofs.Vec (768 : ℕ))
      (pW4 : Proofs.Kernel4 (192 : ℕ) (768 : ℕ) (1 : ℕ) (1 : ℕ)) (pB4 sL4 : Proofs.Vec (192 : ℕ))
      (aW5 : Proofs.DepthwiseKernel (192 : ℕ) (7 : ℕ) (7 : ℕ)) (aB5 nG5 nB5 : Proofs.Vec (192 : ℕ))
      (eW5 : Proofs.Kernel4 (768 : ℕ) (192 : ℕ) (1 : ℕ) (1 : ℕ)) (eB5 : Proofs.Vec (768 : ℕ))
      (pW5 : Proofs.Kernel4 (192 : ℕ) (768 : ℕ) (1 : ℕ) (1 : ℕ)) (pB5 sL5 : Proofs.Vec (192 : ℕ))
      (aW6 : Proofs.DepthwiseKernel (192 : ℕ) (7 : ℕ) (7 : ℕ)) (aB6 nG6 nB6 : Proofs.Vec (192 : ℕ))
      (eW6 : Proofs.Kernel4 (768 : ℕ) (192 : ℕ) (1 : ℕ) (1 : ℕ)) (eB6 : Proofs.Vec (768 : ℕ))
      (pW6 : Proofs.Kernel4 (192 : ℕ) (768 : ℕ) (1 : ℕ) (1 : ℕ)) (pB6 sL6 dG1 dT1 : Proofs.Vec (192 : ℕ))
      (dW1 : Proofs.Kernel4 (384 : ℕ) (192 : ℕ) (2 : ℕ) (2 : ℕ)) (dB1 : Proofs.Vec (384 : ℕ))
      (aW7 : Proofs.DepthwiseKernel (384 : ℕ) (7 : ℕ) (7 : ℕ)) (aB7 nG7 nB7 : Proofs.Vec (384 : ℕ))
      (eW7 : Proofs.Kernel4 (1536 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (eB7 : Proofs.Vec (1536 : ℕ))
      (pW7 : Proofs.Kernel4 (384 : ℕ) (1536 : ℕ) (1 : ℕ) (1 : ℕ)) (pB7 sL7 : Proofs.Vec (384 : ℕ))
      (aW8 : Proofs.DepthwiseKernel (384 : ℕ) (7 : ℕ) (7 : ℕ)) (aB8 nG8 nB8 : Proofs.Vec (384 : ℕ))
      (eW8 : Proofs.Kernel4 (1536 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (eB8 : Proofs.Vec (1536 : ℕ))
      (pW8 : Proofs.Kernel4 (384 : ℕ) (1536 : ℕ) (1 : ℕ) (1 : ℕ)) (pB8 sL8 : Proofs.Vec (384 : ℕ))
      (aW9 : Proofs.DepthwiseKernel (384 : ℕ) (7 : ℕ) (7 : ℕ)) (aB9 nG9 nB9 : Proofs.Vec (384 : ℕ))
      (eW9 : Proofs.Kernel4 (1536 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (eB9 : Proofs.Vec (1536 : ℕ))
      (pW9 : Proofs.Kernel4 (384 : ℕ) (1536 : ℕ) (1 : ℕ) (1 : ℕ)) (pB9 sL9 : Proofs.Vec (384 : ℕ))
      (aW10 : Proofs.DepthwiseKernel (384 : ℕ) (7 : ℕ) (7 : ℕ)) (aB10 nG10 nB10 : Proofs.Vec (384 : ℕ))
      (eW10 : Proofs.Kernel4 (1536 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (eB10 : Proofs.Vec (1536 : ℕ))
      (pW10 : Proofs.Kernel4 (384 : ℕ) (1536 : ℕ) (1 : ℕ) (1 : ℕ)) (pB10 sL10 : Proofs.Vec (384 : ℕ))
      (aW11 : Proofs.DepthwiseKernel (384 : ℕ) (7 : ℕ) (7 : ℕ)) (aB11 nG11 nB11 : Proofs.Vec (384 : ℕ))
      (eW11 : Proofs.Kernel4 (1536 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (eB11 : Proofs.Vec (1536 : ℕ))
      (pW11 : Proofs.Kernel4 (384 : ℕ) (1536 : ℕ) (1 : ℕ) (1 : ℕ)) (pB11 sL11 : Proofs.Vec (384 : ℕ))
      (aW12 : Proofs.DepthwiseKernel (384 : ℕ) (7 : ℕ) (7 : ℕ)) (aB12 nG12 nB12 : Proofs.Vec (384 : ℕ))
      (eW12 : Proofs.Kernel4 (1536 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (eB12 : Proofs.Vec (1536 : ℕ))
      (pW12 : Proofs.Kernel4 (384 : ℕ) (1536 : ℕ) (1 : ℕ) (1 : ℕ)) (pB12 sL12 : Proofs.Vec (384 : ℕ))
      (aW13 : Proofs.DepthwiseKernel (384 : ℕ) (7 : ℕ) (7 : ℕ)) (aB13 nG13 nB13 : Proofs.Vec (384 : ℕ))
      (eW13 : Proofs.Kernel4 (1536 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (eB13 : Proofs.Vec (1536 : ℕ))
      (pW13 : Proofs.Kernel4 (384 : ℕ) (1536 : ℕ) (1 : ℕ) (1 : ℕ)) (pB13 sL13 : Proofs.Vec (384 : ℕ))
      (aW14 : Proofs.DepthwiseKernel (384 : ℕ) (7 : ℕ) (7 : ℕ)) (aB14 nG14 nB14 : Proofs.Vec (384 : ℕ))
      (eW14 : Proofs.Kernel4 (1536 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (eB14 : Proofs.Vec (1536 : ℕ))
      (pW14 : Proofs.Kernel4 (384 : ℕ) (1536 : ℕ) (1 : ℕ) (1 : ℕ)) (pB14 sL14 : Proofs.Vec (384 : ℕ))
      (aW15 : Proofs.DepthwiseKernel (384 : ℕ) (7 : ℕ) (7 : ℕ)) (aB15 nG15 nB15 : Proofs.Vec (384 : ℕ))
      (eW15 : Proofs.Kernel4 (1536 : ℕ) (384 : ℕ) (1 : ℕ) (1 : ℕ)) (eB15 : Proofs.Vec (1536 : ℕ))
      (pW15 : Proofs.Kernel4 (384 : ℕ) (1536 : ℕ) (1 : ℕ) (1 : ℕ)) (pB15 sL15 dG2 dT2 : Proofs.Vec (384 : ℕ))
      (dW2 : Proofs.Kernel4 (768 : ℕ) (384 : ℕ) (2 : ℕ) (2 : ℕ)) (dB2 : Proofs.Vec (768 : ℕ))
      (aW16 : Proofs.DepthwiseKernel (768 : ℕ) (7 : ℕ) (7 : ℕ)) (aB16 nG16 nB16 : Proofs.Vec (768 : ℕ))
      (eW16 : Proofs.Kernel4 (3072 : ℕ) (768 : ℕ) (1 : ℕ) (1 : ℕ)) (eB16 : Proofs.Vec (3072 : ℕ))
      (pW16 : Proofs.Kernel4 (768 : ℕ) (3072 : ℕ) (1 : ℕ) (1 : ℕ)) (pB16 sL16 : Proofs.Vec (768 : ℕ))
      (aW17 : Proofs.DepthwiseKernel (768 : ℕ) (7 : ℕ) (7 : ℕ)) (aB17 nG17 nB17 : Proofs.Vec (768 : ℕ))
      (eW17 : Proofs.Kernel4 (3072 : ℕ) (768 : ℕ) (1 : ℕ) (1 : ℕ)) (eB17 : Proofs.Vec (3072 : ℕ))
      (pW17 : Proofs.Kernel4 (768 : ℕ) (3072 : ℕ) (1 : ℕ) (1 : ℕ)) (pB17 sL17 : Proofs.Vec (768 : ℕ))
      (aW18 : Proofs.DepthwiseKernel (768 : ℕ) (7 : ℕ) (7 : ℕ)) (aB18 nG18 nB18 : Proofs.Vec (768 : ℕ))
      (eW18 : Proofs.Kernel4 (3072 : ℕ) (768 : ℕ) (1 : ℕ) (1 : ℕ)) (eB18 : Proofs.Vec (3072 : ℕ))
      (pW18 : Proofs.Kernel4 (768 : ℕ) (3072 : ℕ) (1 : ℕ) (1 : ℕ)) (pB18 sL18 hG hT : Proofs.Vec (768 : ℕ))
      (Wfc : Proofs.Mat (768 : ℕ) nC) (bfc : Proofs.Vec nC) (x : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))))
      (t : Proofs.Vec (N * nC)),
      have ib1 : Proofs.Vec (N * ((96 : ℕ) * (56 : ℕ) * (56 : ℕ))) :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxStemFwdO ε Wst psb psng psnbt) x;
      have ib2 := Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1) ib1;
      have ib3 := Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2) ib2;
      have ibD0 := Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3) ib3;
      have ib4 : Proofs.Vec (N * ((192 : ℕ) * (28 : ℕ) * (28 : ℕ))) :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxDownFwdChO ε dG0 dT0 dW0 dB0) ibD0;
      have ib5 := Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4) ib4;
      have ib6 := Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5) ib5;
      have ibD1 := Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6) ib6;
      have ib7 : Proofs.Vec (N * ((384 : ℕ) * (14 : ℕ) * (14 : ℕ))) :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxDownFwdChO ε dG1 dT1 dW1 dB1) ibD1;
      have ib8 := Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7) ib7;
      have ib9 := Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8) ib8;
      have ib10 := Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9) ib9;
      have ib11 :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10) ib10;
      have ib12 :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11) ib11;
      have ib13 :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12) ib12;
      have ib14 :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13) ib13;
      have ib15 :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14) ib14;
      have ibD2 :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15) ib15;
      have ib16 : Proofs.Vec (N * ((768 : ℕ) * (7 : ℕ) * (7 : ℕ))) :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxDownFwdChO ε dG2 dT2 dW2 dB2) ibD2;
      have ib17 :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16) ib16;
      have ib18 :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17) ib17;
      have xhead :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTiePoC.cnxBlockFwdChO ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18) ib18;
      have gapB := Proofs.StableHLO.batchMap N (Proofs.globalAvgPoolFlat (768 : ℕ) (7 : ℕ) (7 : ℕ)) xhead;
      have hnB := Proofs.StableHLO.batchMap N (Proofs.rowLNVecFlat (1 : ℕ) (768 : ℕ) ε hG hT) gapB;
      have logitsB := Proofs.StableHLO.batchMap N (Proofs.dense Wfc bfc) hnB;
      have g := Proofs.StableHLO.den (Proofs.smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t);
      have dyO18 := Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoCGB.cnxHeadDyXheadChN ε hG hT Wfc bfc) xhead g;
      have dyO17 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW18 aB18 nG18 nB18 eW18 eB18 pW18 pB18 sL18)
          ib18 dyO18;
      have dyO16 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW17 aB17 nG17 nB17 eW17 eB17 pW17 pB17 sL17)
          ib17 dyO17;
      have dyD2 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW16 aB16 nG16 nB16 eW16 eB16 pW16 pB16 sL16)
          ib16 dyO16;
      have dyO15 := Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxDownCotInChAt ε dG2 dT2 dW2 dB2) ibD2 dyD2;
      have dyO14 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW15 aB15 nG15 nB15 eW15 eB15 pW15 pB15 sL15)
          ib15 dyO15;
      have dyO13 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW14 aB14 nG14 nB14 eW14 eB14 pW14 pB14 sL14)
          ib14 dyO14;
      have dyO12 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW13 aB13 nG13 nB13 eW13 eB13 pW13 pB13 sL13)
          ib13 dyO13;
      have dyO11 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW12 aB12 nG12 nB12 eW12 eB12 pW12 pB12 sL12)
          ib12 dyO12;
      have dyO10 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW11 aB11 nG11 nB11 eW11 eB11 pW11 pB11 sL11)
          ib11 dyO11;
      have dyO9 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW10 aB10 nG10 nB10 eW10 eB10 pW10 pB10 sL10)
          ib10 dyO10;
      have dyO8 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9) ib9 dyO9;
      have dyO7 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8) ib8 dyO8;
      have dyD1 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7) ib7 dyO7;
      have dyO6 := Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxDownCotInChAt ε dG1 dT1 dW1 dB1) ibD1 dyD1;
      have dyO5 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6) ib6 dyO6;
      have dyO4 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5) ib5 dyO5;
      have dyD0 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4) ib4 dyO4;
      have dyO3 := Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxDownCotInChAt ε dG0 dT0 dW0 dB0) ibD0 dyD0;
      have dyO2 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3) ib3 dyO3;
      have dyO1 :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2) ib2 dyO2;
      have dyStem :=
        Proofs.StableHLO.batchMapAux N (Proofs.CnxTiePoC.cnxBlockCotInChAt ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1) ib1 dyO1;
      Proofs.CnxTiePoCGB.cnxStemChTiedGBAt N xN epsStr cotN ε Wst psb psng psnbt x xstem dyStem ∧
        Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW1 aB1 nG1 nB1 eW1 eB1 pW1 pB1 sL1 ib1 dyO1 ∧
          Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW2 aB2 nG2 nB2 eW2 eB2 pW2 pB2 sL2 ib2 dyO2 ∧
            Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW3 aB3 nG3 nB3 eW3 eB3 pW3 pB3 sL3 ib3 dyO3 ∧
              Proofs.CnxTiePoCGB.cnxDownChTiedGBAt N xN epsStr cotN ε dG0 dT0 dW0 dB0 ibD0 dyD0 ∧
                Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW4 aB4 nG4 nB4 eW4 eB4 pW4 pB4 sL4 ib4 dyO4 ∧
                  Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW5 aB5 nG5 nB5 eW5 eB5 pW5 pB5 sL5 ib5 dyO5 ∧
                    Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW6 aB6 nG6 nB6 eW6 eB6 pW6 pB6 sL6 ib6 dyO6 ∧
                      Proofs.CnxTiePoCGB.cnxDownChTiedGBAt N xN epsStr cotN ε dG1 dT1 dW1 dB1 ibD1 dyD1 ∧
                        Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW7 aB7 nG7 nB7 eW7 eB7 pW7 pB7 sL7 ib7
                            dyO7 ∧
                          Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW8 aB8 nG8 nB8 eW8 eB8 pW8 pB8 sL8 ib8
                              dyO8 ∧
                            Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW9 aB9 nG9 nB9 eW9 eB9 pW9 pB9 sL9 ib9
                                dyO9 ∧
                              Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW10 aB10 nG10 nB10 eW10 eB10 pW10
                                  pB10 sL10 ib10 dyO10 ∧
                                Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW11 aB11 nG11 nB11 eW11 eB11 pW11
                                    pB11 sL11 ib11 dyO11 ∧
                                  Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW12 aB12 nG12 nB12 eW12 eB12
                                      pW12 pB12 sL12 ib12 dyO12 ∧
                                    Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW13 aB13 nG13 nB13 eW13 eB13
                                        pW13 pB13 sL13 ib13 dyO13 ∧
                                      Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW14 aB14 nG14 nB14 eW14 eB14
                                          pW14 pB14 sL14 ib14 dyO14 ∧
                                        Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW15 aB15 nG15 nB15 eW15
                                            eB15 pW15 pB15 sL15 ib15 dyO15 ∧
                                          Proofs.CnxTiePoCGB.cnxDownChTiedGBAt N xN epsStr cotN ε dG2 dT2 dW2 dB2 ibD2
                                              dyD2 ∧
                                            Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW16 aB16 nG16 nB16
                                                eW16 eB16 pW16 pB16 sL16 ib16 dyO16 ∧
                                              Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW17 aB17 nG17 nB17
                                                  eW17 eB17 pW17 pB17 sL17 ib17 dyO17 ∧
                                                Proofs.CnxTiePoCGB.cnxBlockChTiedGBAt N xN epsStr cotN ε aW18 aB18 nG18 nB18
                                                    eW18 eB18 pW18 pB18 sL18 ib18 dyO18 ∧
                                                  Proofs.CnxTiePoCGB.cnxHeadChTiedGBAt N xN epsStr cotN dN ε hG hT Wfc bfc
                                                    xhead g :=
  Proofs.CnxTiePoCGB.cnx_net_tiedGB

/-- `Proofs.dpMeanGrad_ne_globalBatchGrad` -/
theorem chk_dpMeanGrad_ne_globalBatchGrad :
    ∀ (θ : Proofs.Vec (1 : ℕ)),
      (Proofs.dpMean (R := (2 : ℕ)) fun (r : Fin (2 : ℕ)) => Proofs.lossGrad (Proofs.bnToyLoss (Proofs.dpToyShard r)) θ) ≠
        Proofs.lossGrad (Proofs.bnToyLoss Proofs.dpToyBatch) θ :=
  Proofs.dpMeanGrad_ne_globalBatchGrad

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
      (hb1 :
        Proofs.HasVJPAt b1 (Proofs.opaqueA0 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) x) ×'
          DifferentiableAt ℝ b1 (Proofs.opaqueA0 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) x))
      (hb2 :
        Proofs.HasVJPAt b2 (Proofs.opaqueA1 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 x) ×'
          DifferentiableAt ℝ b2 (Proofs.opaqueA1 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 x))
      (hb3 :
        Proofs.HasVJPAt b3 (Proofs.opaqueA2 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 x) ×'
          DifferentiableAt ℝ b3 (Proofs.opaqueA2 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 x))
      (hb4 :
        Proofs.HasVJPAt b4 (Proofs.opaqueA3 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 x) ×'
          DifferentiableAt ℝ b4 (Proofs.opaqueA3 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 x))
      (hb5 :
        Proofs.HasVJPAt b5 (Proofs.opaqueA4 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 x) ×'
          DifferentiableAt ℝ b5 (Proofs.opaqueA4 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 x))
      (hb6 :
        Proofs.HasVJPAt b6 (Proofs.opaqueA5 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 x) ×'
          DifferentiableAt ℝ b6 (Proofs.opaqueA5 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 x))
      (hb7 :
        Proofs.HasVJPAt b7 (Proofs.opaqueA6 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x) ×'
          DifferentiableAt ℝ b7 (Proofs.opaqueA6 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 x))
      (hb8 :
        Proofs.HasVJPAt b8 (Proofs.opaqueA7 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x) ×'
          DifferentiableAt ℝ b8
            (Proofs.opaqueA7 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 x))
      (hb9 :
        Proofs.HasVJPAt b9
            (Proofs.opaqueA8 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x) ×'
          DifferentiableAt ℝ b9
            (Proofs.opaqueA8 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 x))
      (hb10 :
        Proofs.HasVJPAt b10
            (Proofs.opaqueA9 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x) ×'
          DifferentiableAt ℝ b10
            (Proofs.opaqueA9 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 x))
      (hb11 :
        Proofs.HasVJPAt b11
            (Proofs.opaqueA10 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x) ×'
          DifferentiableAt ℝ b11
            (Proofs.opaqueA10 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 x))
      (hb12 :
        Proofs.HasVJPAt b12
            (Proofs.opaqueA11 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x) ×'
          DifferentiableAt ℝ b12
            (Proofs.opaqueA11 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 x))
      (hb13 :
        Proofs.HasVJPAt b13
            (Proofs.opaqueA12 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
              x) ×'
          DifferentiableAt ℝ b13
            (Proofs.opaqueA12 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
              x))
      (hb14 :
        Proofs.HasVJPAt b14
            (Proofs.opaqueA13 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
              b13 x) ×'
          DifferentiableAt ℝ b14
            (Proofs.opaqueA13 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
              b13 x))
      (hb15 :
        Proofs.HasVJPAt b15
            (Proofs.opaqueA14 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
              b13 b14 x) ×'
          DifferentiableAt ℝ b15
            (Proofs.opaqueA14 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
              b13 b14 x))
      (hb16 :
        Proofs.HasVJPAt b16
            (Proofs.opaqueA15 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
              b13 b14 b15 x) ×'
          DifferentiableAt ℝ b16
            (Proofs.opaqueA15 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
              b13 b14 b15 x)),
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
    ∀ (N : ℕ) (w : Proofs.B0Weights) (hsε : (0 : ℝ) < w.sε)
      (hb1d : (0 : ℝ) < w.b1.dε) (hb1p : (0 : ℝ) < w.b1.pε) (hb2e : (0 : ℝ) < w.b2.eε) (hb2d : (0 : ℝ) < w.b2.dε)
      (hb2p : (0 : ℝ) < w.b2.pε) (hb3e : (0 : ℝ) < w.b3.eε) (hb3d : (0 : ℝ) < w.b3.dε) (hb3p : (0 : ℝ) < w.b3.pε)
      (hb4e : (0 : ℝ) < w.b4.eε) (hb4d : (0 : ℝ) < w.b4.dε) (hb4p : (0 : ℝ) < w.b4.pε) (hb5e : (0 : ℝ) < w.b5.eε)
      (hb5d : (0 : ℝ) < w.b5.dε) (hb5p : (0 : ℝ) < w.b5.pε) (hb6e : (0 : ℝ) < w.b6.eε) (hb6d : (0 : ℝ) < w.b6.dε)
      (hb6p : (0 : ℝ) < w.b6.pε) (hb7e : (0 : ℝ) < w.b7.eε) (hb7d : (0 : ℝ) < w.b7.dε) (hb7p : (0 : ℝ) < w.b7.pε)
      (hb8e : (0 : ℝ) < w.b8.eε) (hb8d : (0 : ℝ) < w.b8.dε) (hb8p : (0 : ℝ) < w.b8.pε) (hb9e : (0 : ℝ) < w.b9.eε)
      (hb9d : (0 : ℝ) < w.b9.dε) (hb9p : (0 : ℝ) < w.b9.pε) (hb10e : (0 : ℝ) < w.b10.eε) (hb10d : (0 : ℝ) < w.b10.dε)
      (hb10p : (0 : ℝ) < w.b10.pε) (hb11e : (0 : ℝ) < w.b11.eε) (hb11d : (0 : ℝ) < w.b11.dε) (hb11p : (0 : ℝ) < w.b11.pε)
      (hb12e : (0 : ℝ) < w.b12.eε) (hb12d : (0 : ℝ) < w.b12.dε) (hb12p : (0 : ℝ) < w.b12.pε) (hb13e : (0 : ℝ) < w.b13.eε)
      (hb13d : (0 : ℝ) < w.b13.dε) (hb13p : (0 : ℝ) < w.b13.pε) (hb14e : (0 : ℝ) < w.b14.eε) (hb14d : (0 : ℝ) < w.b14.dε)
      (hb14p : (0 : ℝ) < w.b14.pε) (hb15e : (0 : ℝ) < w.b15.eε) (hb15d : (0 : ℝ) < w.b15.dε) (hb15p : (0 : ℝ) < w.b15.pε)
      (hb16e : (0 : ℝ) < w.b16.eε) (hb16d : (0 : ℝ) < w.b16.dε) (hb16p : (0 : ℝ) < w.b16.pε) (hhε : (0 : ℝ) < w.hε)
      (x : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))) (dy : Proofs.Vec (N * (10 : ℕ)))
      (i : Fin (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
      Proofs.efficientnetInputGradB_full N w.sW w.hW w.fcW
          (Proofs.HasVJP.backward (f := Proofs.StableHLO.bnBatchLA N (32 : ℕ) (112 : ℕ) (112 : ℕ) w.sε w.sγ w.sβ)
            (Proofs.bnBatchLA_has_vjp N (32 : ℕ) (112 : ℕ) (112 : ℕ) w.sε hsε w.sγ w.sβ)
            (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2Xla w.sW w.sb) x))
          (Proofs.HasVJP.backward (Proofs.swish_has_vjp (N * ((32 : ℕ) * (112 : ℕ) * (112 : ℕ))))
            (Proofs.StableHLO.bnBatchLA N (32 : ℕ) (112 : ℕ) (112 : ℕ) w.sε w.sγ w.sβ
              (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2Xla w.sW w.sb) x)))
          (Proofs.HasVJP.backward (f := Proofs.StableHLO.bnBatchLA N (1280 : ℕ) (7 : ℕ) (7 : ℕ) w.hε w.hγ w.hβ)
            (Proofs.bnBatchLA_has_vjp N (1280 : ℕ) (7 : ℕ) (7 : ℕ) w.hε hhε w.hγ w.hβ)
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
          ((Proofs.mbNoExpW_has_vjp N (112 : ℕ) (112 : ℕ) w.b1 hb1d hb1p).backward
            (Proofs.opaqueA0 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) x))
          ((Proofs.mbStridedW_has_vjp N (56 : ℕ) (56 : ℕ) w.b2 hb2e hb2d hb2p).backward
            (Proofs.opaqueA1 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1) x))
          ((Proofs.mbResidW_has_vjp N (56 : ℕ) (56 : ℕ) w.b3 hb3e hb3d hb3p).backward
            (Proofs.opaqueA2 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) x))
          ((Proofs.mbStridedW_has_vjp N (28 : ℕ) (28 : ℕ) w.b4 hb4e hb4d hb4p).backward
            (Proofs.opaqueA3 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3) x))
          ((Proofs.mbResidW_has_vjp N (28 : ℕ) (28 : ℕ) w.b5 hb5e hb5d hb5p).backward
            (Proofs.opaqueA4 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) x))
          ((Proofs.mbStridedW_has_vjp N (14 : ℕ) (14 : ℕ) w.b6 hb6e hb6d hb6p).backward
            (Proofs.opaqueA5 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5) x))
          ((Proofs.mbResidW_has_vjp N (14 : ℕ) (14 : ℕ) w.b7 hb7e hb7d hb7p).backward
            (Proofs.opaqueA6 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) x))
          ((Proofs.mbResidW_has_vjp N (14 : ℕ) (14 : ℕ) w.b8 hb8e hb8d hb8p).backward
            (Proofs.opaqueA7 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7) x))
          ((Proofs.mbExpW_has_vjp N (14 : ℕ) (14 : ℕ) w.b9 hb9e hb9d hb9p).backward
            (Proofs.opaqueA8 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) x))
          ((Proofs.mbResidW_has_vjp N (14 : ℕ) (14 : ℕ) w.b10 hb10e hb10d hb10p).backward
            (Proofs.opaqueA9 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9) x))
          ((Proofs.mbResidW_has_vjp N (14 : ℕ) (14 : ℕ) w.b11 hb11e hb11d hb11p).backward
            (Proofs.opaqueA10 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) x))
          ((Proofs.mbStridedW_has_vjp N (7 : ℕ) (7 : ℕ) w.b12 hb12e hb12d hb12p).backward
            (Proofs.opaqueA11 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11) x))
          ((Proofs.mbResidW_has_vjp N (7 : ℕ) (7 : ℕ) w.b13 hb13e hb13d hb13p).backward
            (Proofs.opaqueA12 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) x))
          ((Proofs.mbResidW_has_vjp N (7 : ℕ) (7 : ℕ) w.b14 hb14e hb14d hb14p).backward
            (Proofs.opaqueA13 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13) x))
          ((Proofs.mbResidW_has_vjp N (7 : ℕ) (7 : ℕ) w.b15 hb15e hb15d hb15p).backward
            (Proofs.opaqueA14 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13)
              (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b14) x))
          ((Proofs.mbExpW_has_vjp N (7 : ℕ) (7 : ℕ) w.b16 hb16e hb16d hb16p).backward
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
