import LeanMlir.Proofs.Architectures.BatchNorm
import LeanMlir.Proofs.Certificates.LipschitzCert.Basic
import LeanMlir.Proofs.Certificates.LipschitzCert.ScorecardSDP
import LeanMlir.Proofs.Certificates.Smoothing.Gaussian
import LeanMlir.Proofs.Float.FloatBridge
import LeanMlir.Proofs.Float.MlpFloatBridge
import LeanMlir.Proofs.Foundation.DataParallel.Basic
import LeanMlir.Proofs.Foundation.DataParallel.Node
import LeanMlir.Proofs.Foundation.DataParallel.Sync
import LeanMlir.Proofs.Foundation.DataParallel.SyncBf16
import LeanMlir.Proofs.Foundation.Muon.Geometry
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
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2SyncB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetSyncB
import LeanMlir.Proofs.Nets.ResNet.ResNet50SyncB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncB
import LeanMlir.Proofs.Nets.ResNet.ResNet34SyncStepTieB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2SyncStepTieB
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetSyncStepTieG
import LeanMlir.Proofs.Nets.ResNet.ResNet50SyncStepTieB
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4SyncStepTieB
import LeanMlir.Proofs.Nets.ViT.ViTWholeBackCertifiedTie
import LeanMlir.Proofs.Nets.ViT.ViTStepTie
import LeanMlir.Proofs.Training.Trained.LinearDescent
import LeanMlir.Proofs.Training.Trained.CnnDescent
import LeanMlir.Proofs.Training.Trained.CnnDescentConv1
import LeanMlir.Proofs.Nets.ResNet.ResNet34ParamGrad
import LeanMlir.Proofs.Nets.ResNet.ResNet50ParamGrad
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV2ParamGrad
import LeanMlir.Proofs.Nets.MobileNet.MobileNetV4ParamGrad
import LeanMlir.Proofs.Nets.EfficientNet.EfficientNetParamGrad
import LeanMlir.Proofs.Nets.ConvNeXt.ConvNeXtParamGrad
import LeanMlir.Proofs.Nets.ViT.ViTParamGrad
import LeanMlir.Proofs.Nets.Small.LinearParamGrad
import LeanMlir.Proofs.Nets.Small.MlpParamGrad
import LeanMlir.Proofs.Nets.Small.CnnParamGrad
import LeanMlir.Proofs.Nets.Small.CifarParamGrad
import LeanMlir.Proofs.Nets.Small.Cifar8ParamGrad
import LeanMlir.Proofs.Nets.Small.Cifar8BnParamGrad

universe u_1

open Proofs
open scoped Real

set_option maxHeartbeats 8000000

/-! # Solution to the tier challenge

Solution to `ChallengeTier.lean`. Each proof is the project theorem itself:
the statement IS that theorem's type, so the delegation is a bare constant and nothing can
be weakened between the two files without failing to elaborate.

⚠ **MACHINE-GENERATED — do not hand-edit.** Every statement is the project declaration's
own type as Lean prints it (`scripts/gates/gen_comparator_tier.py`), so this file and its
`ChallengeTier.lean` carry the same text by construction rather than by review. Regenerate after any
statement change; the generator verifies that what it wrote still elaborates.
-/

/-- `Proofs.bn_input_grad_correct` -/
theorem chk_bn_input_grad_correct :
    ∀ (n : ℕ) (ε γ β : ℝ),
      (0 : ℝ) < ε →
        ∀ (x dy : Proofs.Vec n) (i : Fin n),
          Proofs.bnGradInput n ε γ x dy i = ∑ j : Fin n, Proofs.pdiv (Proofs.bnForward n ε γ β) x i j * dy j :=
  Proofs.bn_input_grad_correct

/-- `Proofs.resnet50ForwardBFullHasVJPAt_correct` -/
theorem chk_resnet50ForwardBFullHasVJPAt_correct :
    ∀ (N q : ℕ) {nCls : ℕ} (w : Proofs.R50BWeights nCls)
      (hp : Proofs.R50PosB w)
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
      (Proofs.resnet50ForwardBFullHasVJPAt N q w hp x hx).backward dy i =
        ∑ j : Fin (N * nCls), Proofs.pdiv (Proofs.resnet50ForwardBFull N q w) x i j * dy j :=
  Proofs.resnet50ForwardBFullHasVJPAt_correct

/-- `Proofs.vitTinyHasVJP_correct` -/
theorem chk_vitTinyHasVJP_correct :
    ∀ {gf : Proofs.GeluForm} {nCls : ℕ}
      (W_conv : Proofs.Kernel4 ((3 : ℕ) * (64 : ℕ)) (3 : ℕ) (16 : ℕ) (16 : ℕ))
      (b_conv cls_token : Proofs.Vec ((3 : ℕ) * (64 : ℕ)))
      (pos_embed : Proofs.Mat ((196 : ℕ) + (1 : ℕ)) ((3 : ℕ) * (64 : ℕ))) (ε : ℝ),
      (0 : ℝ) < ε →
        ∀ (ps : Fin (12 : ℕ) → Proofs.BlockParamsV ((3 : ℕ) * (64 : ℕ)) (768 : ℕ)) (γF βF : Proofs.Vec ((3 : ℕ) * (64 : ℕ)))
          (Wcls : Proofs.Mat ((3 : ℕ) * (64 : ℕ)) nCls) (bcls : Proofs.Vec nCls)
          (x : Proofs.Vec ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))) (dy : Proofs.Vec nCls)
          (i : Fin ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))),
          Proofs.vitInputGradK gf (3 : ℕ) (224 : ℕ) (224 : ℕ) (16 : ℕ) (196 : ℕ) (768 : ℕ) (3 : ℕ) (64 : ℕ) nCls (12 : ℕ)
              W_conv b_conv cls_token pos_embed ε ps γF Wcls x dy i =
            ∑ j : Fin nCls,
              Proofs.pdiv
                  (Proofs.vitForwardKV gf (3 : ℕ) (224 : ℕ) (224 : ℕ) (16 : ℕ) (196 : ℕ) (768 : ℕ) (3 : ℕ) (64 : ℕ) nCls
                    (12 : ℕ) W_conv b_conv cls_token pos_embed ε ps γF βF Wcls bcls)
                  x i j *
                dy j :=
  Proofs.vitTinyHasVJP_correct

/-- `Proofs.StableHLO.mnv4FwdGraphBFull_faithful` -/
theorem chk_mnv4FwdGraphBFull_faithful :
    ∀ (N : ℕ) (epsStr : String) {nCls : ℕ}
      (w : Proofs.StableHLO.Mnv4BWeights nCls) (e : Proofs.StableHLO.SHlo (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
      Proofs.StableHLO.den (Proofs.StableHLO.mnv4FwdGraphBFull N epsStr w e) =
        Proofs.StableHLO.mobilenetv4ForwardBFull N w (Proofs.StableHLO.den e) :=
  Proofs.StableHLO.mnv4FwdGraphBFull_faithful

/-- `Proofs.Mnv2FullBSeal.sealX_nonconstant` -/
theorem chk_sealX_nonconstant :
    ∀ (nCls : ℕ),
      (0 : ℕ) < nCls →
        Proofs.mobilenetv2ForwardBFull (2 : ℕ) (Proofs.Mnv2FullBSeal.sealW nCls) (Proofs.Mnv2FullBSeal.sealX (1 : ℝ)) ≠
          Proofs.mobilenetv2ForwardBFull (2 : ℕ) (Proofs.Mnv2FullBSeal.sealW nCls) (Proofs.Mnv2FullBSeal.sealX (0 : ℝ)) :=
  Proofs.Mnv2FullBSeal.sealX_nonconstant

/-- `Proofs.Mnv4FullBSeal.sealX_backward_nontrivial` -/
theorem chk_sealX_backward_nontrivial :
    ∀ (nCls : ℕ),
      (0 : ℕ) < nCls →
        ∃ (j₀ : Fin ((2 : ℕ) * nCls)) (i₀ : Fin ((2 : ℕ) * ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ))))),
          (Proofs.Mnv4FullBSeal.sealVJP nCls (0 : ℝ)).backward (Proofs.basisVec j₀) i₀ ≠ (0 : ℝ) :=
  Proofs.Mnv4FullBSeal.sealX_backward_nontrivial

/-- `Proofs.GradNodeB.convStridedWGradB_den` -/
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
  Proofs.GradNodeB.convStridedWGradB_den

/-- `Proofs.smoothedCE_grad` -/
theorem chk_smoothedCE_grad :
    ∀ (K : ℕ),
      (0 : ℕ) < K →
        ∀ (α : ℝ) (t z : Proofs.Vec K),
          ∑ k : Fin K, t k = (1 : ℝ) →
            ∀ (j : Fin K),
              Proofs.pdiv (fun (z' : Proofs.Vec K) (_ : Fin (1 : ℕ)) => Proofs.softCE K (Proofs.smoothTarget K α t) z') z j
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

/-- `Proofs.ViTTie.vit_net_tied_certified` -/
theorem chk_vit_net_tied_certified :
    ∀ {gf : Proofs.GeluForm} (xN wN bN gN aN clsN pN epsStr lrStr cotN : String)
      (ε : Real) (w : Proofs.ViTTie.ViTTieWeights (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))))
      (img :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat)
            (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
              (@OfNat.ofNat Nat (nat_lit 224) (instOfNatNat (nat_lit 224))))
            (@OfNat.ofNat Nat (nat_lit 224) (instOfNatNat (nat_lit 224)))))
      (label : Fin (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10)))) (lr : Real),
      have ib1 :=
        Proofs.patchEmbedFlat (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 224) (instOfNatNat (nat_lit 224)))
          (@OfNat.ofNat Nat (nat_lit 224) (instOfNatNat (nat_lit 224)))
          (@OfNat.ofNat Nat (nat_lit 16) (instOfNatNat (nat_lit 16)))
          (@OfNat.ofNat Nat (nat_lit 196) (instOfNatNat (nat_lit 196)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@Proofs.ViTTie.ViTTieWeights.Wc (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTie.ViTTieWeights.bc (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTie.ViTTieWeights.cls (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTie.ViTTieWeights.pos (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) img;
      have ib2 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b1 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib1;
      have ib3 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b2 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib2;
      have ib4 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b3 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib3;
      have ib5 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b4 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib4;
      have ib6 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b5 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib5;
      have ib7 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b6 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib6;
      have ib8 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b7 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib7;
      have ib9 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b8 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib8;
      have ib10 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b9 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib9;
      have ib11 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b10 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib10;
      have ib12 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b11 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib11;
      have b12out :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.fwdO gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b12 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib12;
      have fl :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.Mat.flatten (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          fun (r : Fin (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))) =>
          Proofs.layerNormVec (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) ε
            (@Proofs.ViTTie.ViTTieWeights.γF (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
            (@Proofs.ViTTie.ViTTieWeights.βF (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
            (@Proofs.Mat.unflatten (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
              (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) b12out r);
      have hn :=
        Proofs.StableHLO.clsSliceFlat (@OfNat.ofNat Nat (nat_lit 196) (instOfNatNat (nat_lit 196)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192))) fl;
      have logits :=
        @Proofs.dense (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10)))
          (@Proofs.ViTTie.ViTTieWeights.Wcls (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTie.ViTTieWeights.bcls (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) hn;
      have g : Proofs.Vec (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) :=
        fun (c : Fin (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10)))) =>
        @HSub.hSub Real Real Real (@instHSub Real Real.instSub)
          (Proofs.softmax (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) logits c)
          (Proofs.oneHot (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) label c);
      have dy12 :=
        Proofs.vitCotTowerOutV (@OfNat.ofNat Nat (nat_lit 196) (instOfNatNat (nat_lit 196)))
          (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))
          (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) ε
          (@Proofs.ViTTie.ViTTieWeights.γF (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
          (@Proofs.ViTTie.ViTTieWeights.Wcls (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) b12out g;
      have dy11 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b12 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib12 dy12;
      have dy10 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b11 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib11 dy11;
      have dy9 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b10 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib10 dy10;
      have dy8 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b9 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib9 dy9;
      have dy7 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b8 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib8 dy8;
      have dy6 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b7 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib7 dy7;
      have dy5 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b6 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib6 dy6;
      have dy4 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b5 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib5 dy5;
      have dy3 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b4 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib4 dy4;
      have dy2 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b3 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib3 dy3;
      have dy1 :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b2 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib2 dy2;
      have dyEmbed :
        Proofs.Vec
          (@HMul.hMul Nat Nat Nat (@instHMul Nat instMulNat) (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 192) (instOfNatNat (nat_lit 192)))) :=
        @Proofs.BlockParamsV.cotIn gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b1 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) ε ib1 dy1;
      And
        (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
          (@Proofs.ViTTie.ViTTieWeights.b1 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN gN epsStr
          lrStr cotN ε ib1 dy1 lr)
        (And
          (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
            (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
            (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
            (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
            (@Proofs.ViTTie.ViTTieWeights.b2 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN gN
            epsStr lrStr cotN ε ib2 dy2 lr)
          (And
            (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
              (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
              (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
              (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
              (@Proofs.ViTTie.ViTTieWeights.b3 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN gN
              epsStr lrStr cotN ε ib3 dy3 lr)
            (And
              (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                (@Proofs.ViTTie.ViTTieWeights.b4 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN gN
                epsStr lrStr cotN ε ib4 dy4 lr)
              (And
                (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                  (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                  (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                  (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                  (@Proofs.ViTTie.ViTTieWeights.b5 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN
                  gN epsStr lrStr cotN ε ib5 dy5 lr)
                (And
                  (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                    (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                    (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                    (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                    (@Proofs.ViTTie.ViTTieWeights.b6 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN bN
                    gN epsStr lrStr cotN ε ib6 dy6 lr)
                  (And
                    (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                      (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                      (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                      (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                      (@Proofs.ViTTie.ViTTieWeights.b7 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN wN
                      bN gN epsStr lrStr cotN ε ib7 dy7 lr)
                    (And
                      (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                        (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                        (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                        (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                        (@Proofs.ViTTie.ViTTieWeights.b8 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN
                        wN bN gN epsStr lrStr cotN ε ib8 dy8 lr)
                      (And
                        (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                          (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                          (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                          (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                          (@Proofs.ViTTie.ViTTieWeights.b9 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w) xN
                          wN bN gN epsStr lrStr cotN ε ib9 dy9 lr)
                        (And
                          (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                            (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                            (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                            (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                            (@Proofs.ViTTie.ViTTieWeights.b10 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                            xN wN bN gN epsStr lrStr cotN ε ib10 dy10 lr)
                          (And
                            (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                              (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                              (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                              (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                              (@Proofs.ViTTie.ViTTieWeights.b11 (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10)))
                                w)
                              xN wN bN gN epsStr lrStr cotN ε ib11 dy11 lr)
                            (And
                              (@Proofs.BlockParamsV.TiedAt gf (@OfNat.ofNat Nat (nat_lit 197) (instOfNatNat (nat_lit 197)))
                                (@OfNat.ofNat Nat (nat_lit 3) (instOfNatNat (nat_lit 3)))
                                (@OfNat.ofNat Nat (nat_lit 64) (instOfNatNat (nat_lit 64)))
                                (@OfNat.ofNat Nat (nat_lit 768) (instOfNatNat (nat_lit 768)))
                                (@Proofs.ViTTie.ViTTieWeights.b12
                                  (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                xN wN bN gN epsStr lrStr cotN ε ib12 dy12 lr)
                              (And
                                (Proofs.ViTTie.vitFinalLNTied gN xN bN epsStr lrStr cotN ε
                                  (@Proofs.ViTTie.ViTTieWeights.γF
                                    (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                  (@Proofs.ViTTie.ViTTieWeights.βF
                                    (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                  (@Proofs.ViTTie.ViTTieWeights.Wcls
                                    (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                  b12out g lr)
                                (And
                                  (Proofs.ViTTie.vitHeadTied aN wN bN lrStr cotN hn
                                    (@Proofs.ViTTie.ViTTieWeights.Wcls
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    (@Proofs.ViTTie.ViTTieWeights.bcls
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    g lr)
                                  (Proofs.ViTTie.vitEmbedTied wN xN bN clsN pN lrStr cotN
                                    (@Proofs.ViTTie.ViTTieWeights.Wc
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    (@Proofs.ViTTie.ViTTieWeights.bc
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    (@Proofs.ViTTie.ViTTieWeights.cls
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    (@Proofs.ViTTie.ViTTieWeights.pos
                                      (@OfNat.ofNat Nat (nat_lit 10) (instOfNatNat (nat_lit 10))) w)
                                    img dyEmbed lr)))))))))))))) :=
  Proofs.ViTTie.vit_net_tied_certified

/-- `Proofs.CnxTieGB.cnx_net_tiedGB` -/
theorem chk_cnx_net_tiedGB :
    ∀ {gf : Proofs.GeluForm} (N : ℕ) {nC : ℕ}
      (xN epsStr cotN dN aStr negAK bStr logN ohN : String) (ε α B : ℝ) (w : Proofs.CnxTie.CnxTieWeights nC)
      (x : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))) (t : Proofs.Vec (N * nC)),
      have ib1 : Proofs.Vec (N * ((96 : ℕ) * (56 : ℕ) * (56 : ℕ))) :=
        Proofs.StableHLO.batchMap N (Proofs.CnxTie.cnxStemFwdO ε w.sW w.sb w.sγ w.sβ) x;
      have ib2 := Proofs.StableHLO.batchMap N (w.b1.fwdO gf ε) ib1;
      have ib3 := Proofs.StableHLO.batchMap N (w.b2.fwdO gf ε) ib2;
      have ibD0 := Proofs.StableHLO.batchMap N (w.b3.fwdO gf ε) ib3;
      have ib4 : Proofs.Vec (N * ((192 : ℕ) * (28 : ℕ) * (28 : ℕ))) := Proofs.StableHLO.batchMap N (w.d0.fwdO ε) ibD0;
      have ib5 := Proofs.StableHLO.batchMap N (w.b4.fwdO gf ε) ib4;
      have ib6 := Proofs.StableHLO.batchMap N (w.b5.fwdO gf ε) ib5;
      have ibD1 := Proofs.StableHLO.batchMap N (w.b6.fwdO gf ε) ib6;
      have ib7 : Proofs.Vec (N * ((384 : ℕ) * (14 : ℕ) * (14 : ℕ))) := Proofs.StableHLO.batchMap N (w.d1.fwdO ε) ibD1;
      have ib8 := Proofs.StableHLO.batchMap N (w.b7.fwdO gf ε) ib7;
      have ib9 := Proofs.StableHLO.batchMap N (w.b8.fwdO gf ε) ib8;
      have ib10 := Proofs.StableHLO.batchMap N (w.b9.fwdO gf ε) ib9;
      have ib11 := Proofs.StableHLO.batchMap N (w.b10.fwdO gf ε) ib10;
      have ib12 := Proofs.StableHLO.batchMap N (w.b11.fwdO gf ε) ib11;
      have ib13 := Proofs.StableHLO.batchMap N (w.b12.fwdO gf ε) ib12;
      have ib14 := Proofs.StableHLO.batchMap N (w.b13.fwdO gf ε) ib13;
      have ib15 := Proofs.StableHLO.batchMap N (w.b14.fwdO gf ε) ib14;
      have ibD2 := Proofs.StableHLO.batchMap N (w.b15.fwdO gf ε) ib15;
      have ib16 : Proofs.Vec (N * ((768 : ℕ) * (7 : ℕ) * (7 : ℕ))) := Proofs.StableHLO.batchMap N (w.d2.fwdO ε) ibD2;
      have ib17 := Proofs.StableHLO.batchMap N (w.b16.fwdO gf ε) ib16;
      have ib18 := Proofs.StableHLO.batchMap N (w.b17.fwdO gf ε) ib17;
      have xhead := Proofs.StableHLO.batchMap N (w.b18.fwdO gf ε) ib18;
      have gapB := Proofs.StableHLO.batchMap N (Proofs.globalAvgPoolFlat (768 : ℕ) (7 : ℕ) (7 : ℕ)) xhead;
      have hnB := Proofs.StableHLO.batchMap N (Proofs.rowLNVecFlat (1 : ℕ) (768 : ℕ) ε w.hG w.hT) gapB;
      have logitsB := Proofs.StableHLO.batchMap N (Proofs.dense w.Wfc w.bfc) hnB;
      have g := Proofs.StableHLO.den (Proofs.smoothedLossCotGraphDiv N nC α B aStr negAK bStr logN ohN logitsB t);
      have dyO18 := Proofs.StableHLO.batchMapAux N (Proofs.CnxTieGB.cnxHeadDyXheadChN ε w.hG w.hT w.Wfc w.bfc) xhead g;
      have dyO17 := Proofs.StableHLO.batchMapAux N (w.b18.cotIn gf ε) ib18 dyO18;
      have dyO16 := Proofs.StableHLO.batchMapAux N (w.b17.cotIn gf ε) ib17 dyO17;
      have dyD2 := Proofs.StableHLO.batchMapAux N (w.b16.cotIn gf ε) ib16 dyO16;
      have dyO15 := Proofs.StableHLO.batchMapAux N (w.d2.cotIn ε) ibD2 dyD2;
      have dyO14 := Proofs.StableHLO.batchMapAux N (w.b15.cotIn gf ε) ib15 dyO15;
      have dyO13 := Proofs.StableHLO.batchMapAux N (w.b14.cotIn gf ε) ib14 dyO14;
      have dyO12 := Proofs.StableHLO.batchMapAux N (w.b13.cotIn gf ε) ib13 dyO13;
      have dyO11 := Proofs.StableHLO.batchMapAux N (w.b12.cotIn gf ε) ib12 dyO12;
      have dyO10 := Proofs.StableHLO.batchMapAux N (w.b11.cotIn gf ε) ib11 dyO11;
      have dyO9 := Proofs.StableHLO.batchMapAux N (w.b10.cotIn gf ε) ib10 dyO10;
      have dyO8 := Proofs.StableHLO.batchMapAux N (w.b9.cotIn gf ε) ib9 dyO9;
      have dyO7 := Proofs.StableHLO.batchMapAux N (w.b8.cotIn gf ε) ib8 dyO8;
      have dyD1 := Proofs.StableHLO.batchMapAux N (w.b7.cotIn gf ε) ib7 dyO7;
      have dyO6 := Proofs.StableHLO.batchMapAux N (w.d1.cotIn ε) ibD1 dyD1;
      have dyO5 := Proofs.StableHLO.batchMapAux N (w.b6.cotIn gf ε) ib6 dyO6;
      have dyO4 := Proofs.StableHLO.batchMapAux N (w.b5.cotIn gf ε) ib5 dyO5;
      have dyD0 := Proofs.StableHLO.batchMapAux N (w.b4.cotIn gf ε) ib4 dyO4;
      have dyO3 := Proofs.StableHLO.batchMapAux N (w.d0.cotIn ε) ibD0 dyD0;
      have dyO2 := Proofs.StableHLO.batchMapAux N (w.b3.cotIn gf ε) ib3 dyO3;
      have dyO1 := Proofs.StableHLO.batchMapAux N (w.b2.cotIn gf ε) ib2 dyO2;
      have dyStem := Proofs.StableHLO.batchMapAux N (w.b1.cotIn gf ε) ib1 dyO1;
      Proofs.CnxTieGB.cnxStemChTiedGBAt N xN epsStr cotN ε w.sW w.sb w.sγ w.sβ x dyStem ∧
        Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b1 N xN epsStr cotN ε ib1 dyO1 ∧
          Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b2 N xN epsStr cotN ε ib2 dyO2 ∧
            Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b3 N xN epsStr cotN ε ib3 dyO3 ∧
              w.d0.TiedGB N xN epsStr cotN ε ibD0 dyD0 ∧
                Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b4 N xN epsStr cotN ε ib4 dyO4 ∧
                  Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b5 N xN epsStr cotN ε ib5 dyO5 ∧
                    Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b6 N xN epsStr cotN ε ib6 dyO6 ∧
                      w.d1.TiedGB N xN epsStr cotN ε ibD1 dyD1 ∧
                        Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b7 N xN epsStr cotN ε ib7 dyO7 ∧
                          Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b8 N xN epsStr cotN ε ib8 dyO8 ∧
                            Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b9 N xN epsStr cotN ε ib9 dyO9 ∧
                              Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b10 N xN epsStr cotN ε ib10 dyO10 ∧
                                Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b11 N xN epsStr cotN ε ib11 dyO11 ∧
                                  Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b12 N xN epsStr cotN ε ib12 dyO12 ∧
                                    Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b13 N xN epsStr cotN ε ib13 dyO13 ∧
                                      Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b14 N xN epsStr cotN ε ib14 dyO14 ∧
                                        Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b15 N xN epsStr cotN ε ib15 dyO15 ∧
                                          w.d2.TiedGB N xN epsStr cotN ε ibD2 dyD2 ∧
                                            Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b16 N xN epsStr cotN ε ib16 dyO16 ∧
                                              Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b17 N xN epsStr cotN ε ib17 dyO17 ∧
                                                Proofs.CnxTie.CnxTieBlk.TiedGB gf w.b18 N xN epsStr cotN ε ib18 dyO18 ∧
                                                  Proofs.CnxTieGB.cnxHeadChTiedGB N xN epsStr cotN dN ε w.hG w.hT w.Wfc
                                                    w.bfc xhead g :=
  Proofs.CnxTieGB.cnx_net_tiedGB

/-- `Proofs.ResNet34TieB.r34_net_lossGrad` -/
theorem chk_r34_net_lossGrad :
    ∀ (N : ℕ) {nCls : ℕ} (xN cotN vN epsStr : String) (w : Proofs.R34BWeights nCls),
      Proofs.R34PosB w →
        ∀ (x : Proofs.Vec (N * ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ)))))),
          Proofs.ResNet34TieB.R34LossSmoothAtB N w x →
            ∀ {L : Proofs.Vec (N * nCls) → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec (N * nCls)},
              Proofs.HasGradAt L (Proofs.resnet34ForwardBFull N w x) g →
                Proofs.ResNet34TieB.R34NetLossTiedB N xN cotN vN epsStr w x L g :=
  Proofs.ResNet34TieB.r34_net_lossGrad

/-- `Proofs.ResNet50TieB.r50_net_lossGrad` -/
theorem chk_r50_net_lossGrad :
    ∀ (N q : ℕ) {nCls : ℕ} (xN cotN vN epsStr : String)
      (w : Proofs.R50BWeights nCls),
      Proofs.R50PosB w →
        ∀
          (x :
            Proofs.Vec
              (N *
                ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q))))) *
                  ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * q)))))))),
          Proofs.ResNet50TieB.R50LossSmoothAtB N q w x →
            ∀ {L : Proofs.Vec (N * nCls) → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec (N * nCls)},
              Proofs.HasGradAt L (Proofs.resnet50ForwardBFull N q w x) g →
                Proofs.ResNet50TieB.R50NetLossTiedB N q xN cotN vN epsStr w x L g :=
  Proofs.ResNet50TieB.r50_net_lossGrad

/-- `Proofs.MobileNetV2TieB.mnv2_net_lossGrad` -/
theorem chk_mnv2_net_lossGrad :
    ∀ (N : ℕ) {nCls : ℕ} (xN cotN vN epsStr : String)
      (w : Proofs.MNV2BWeights nCls),
      Proofs.MNV2PosB w →
        ∀ (x : Proofs.Vec (N * ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ))))),
          Proofs.MNV2SmoothAtB N w x →
            ∀ {L : Proofs.Vec (N * nCls) → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec (N * nCls)},
              Proofs.HasGradAt L (Proofs.mobilenetv2ForwardBFull N w x) g →
                Proofs.MobileNetV2TieB.MNV2NetLossTiedB N xN cotN vN epsStr w x L g :=
  Proofs.MobileNetV2TieB.mnv2_net_lossGrad

/-- `Proofs.Mnv4TieB.mnv4_net_lossGrad` -/
theorem chk_mnv4_net_lossGrad :
    ∀ (N : ℕ) {nCls : ℕ} (xN cotN vN epsStr : String)
      (w : Proofs.StableHLO.Mnv4BWeights nCls) (x : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
      Proofs.StableHLO.Mnv4SmoothAt N w x →
        ∀ {L : Proofs.Vec (N * nCls) → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec (N * nCls)},
          Proofs.HasGradAt L (Proofs.StableHLO.mobilenetv4ForwardBFull N w x) g →
            Proofs.Mnv4TieB.Mnv4NetLossTiedB N xN cotN vN epsStr w x L g :=
  Proofs.Mnv4TieB.mnv4_net_lossGrad

/-- `Proofs.EnetTieG.enet_net_lossGrad` -/
theorem chk_enet_net_lossGrad :
    ∀ (xN vN epsStr cotN dN : String) (N : ℕ) {nCls : ℕ} (w : Proofs.B0Weights nCls)
      (hεw : w.EpsPos) (x : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))))
      {L : Proofs.Vec (N * nCls) → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec (N * nCls)},
      Proofs.HasGradAt L (Proofs.efficientnetForwardBFull N w x) g →
        Proofs.EnetTieG.EnetNetLossTiedG xN vN epsStr cotN dN N w hεw x L g :=
  Proofs.EnetTieG.enet_net_lossGrad

/-- `Proofs.CnxTieGB.cnx_net_lossGrad` -/
theorem chk_cnx_net_lossGrad :
    ∀ {gf : Proofs.GeluForm} (xN epsStr cotN dN : String) (N : ℕ) {nC : ℕ} (ε : ℝ),
      (0 : ℝ) < ε →
        ∀ (w : Proofs.CnxTie.CnxTieWeights nC) (x : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))))
          {L : Proofs.Vec (N * nC) → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec (N * nC)},
          Proofs.HasGradAt L (Proofs.CnxTieGB.cnxNetB gf N ε w x) g →
            Proofs.CnxTieGB.CnxNetLossTiedGB gf xN epsStr cotN dN N ε w x L g :=
  Proofs.CnxTieGB.cnx_net_lossGrad

/-- `Proofs.ViTTieGB.vit_net_lossGrad` -/
theorem chk_vit_net_lossGrad :
    ∀ {gf : Proofs.GeluForm} (xN aN epsStr cotN : String) (N : ℕ) {nC : ℕ} (ε : ℝ),
      (0 : ℝ) < ε →
        ∀ (w : Proofs.ViTTie.ViTTieWeights nC) (img : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))))
          {L : Proofs.Vec (N * nC) → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec (N * nC)},
          Proofs.HasGradAt L (Proofs.ViTTieGB.vitNetB gf N ε w img) g →
            Proofs.ViTTieGB.ViTNetLossTiedGB gf xN aN epsStr cotN N ε w img L g :=
  Proofs.ViTTieGB.vit_net_lossGrad

/-- `Proofs.LinFold.linear_net_lossGrad` -/
theorem chk_linear_net_lossGrad :
    ∀ {m n : ℕ} (aN cotN : String) (W : Proofs.Mat m n) (b : Proofs.Vec n)
      (x : Proofs.Vec m) {L : Proofs.Vec n → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec n},
      Proofs.HasGradAt L (Proofs.mnistLinear W b x) g → Proofs.LinFold.LinNetLossTied aN cotN W b x L g :=
  Proofs.LinFold.linear_net_lossGrad

/-- `Proofs.MlpFold.mlp_net_lossGrad` -/
theorem chk_mlp_net_lossGrad :
    ∀ {d₀ d₁ d₂ d₃ : ℕ} (aN cotN : String) (W₀ : Proofs.Mat d₀ d₁) (b₀ : Proofs.Vec d₁)
      (W₁ : Proofs.Mat d₁ d₂) (b₁ : Proofs.Vec d₂) (W₂ : Proofs.Mat d₂ d₃) (b₂ : Proofs.Vec d₃) (x : Proofs.Vec d₀),
      (∀ (k : Fin d₁), Proofs.dense W₀ b₀ x k ≠ (0 : ℝ)) →
        (∀ (k : Fin d₂), Proofs.dense W₁ b₁ (Proofs.relu d₁ (Proofs.dense W₀ b₀ x)) k ≠ (0 : ℝ)) →
          ∀ {L : Proofs.Vec d₃ → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec d₃},
            Proofs.HasGradAt L (Proofs.mlpForward W₀ b₀ W₁ b₁ W₂ b₂ x) g →
              Proofs.MlpFold.MlpNetLossTied aN cotN W₀ b₀ W₁ b₁ W₂ b₂ x L g :=
  Proofs.MlpFold.mlp_net_lossGrad

/-- `Proofs.CnnFold.cnn_net_lossGrad` -/
theorem chk_cnn_net_lossGrad :
    ∀ {ic c h w d1 nClasses kH kW : ℕ} (xN cotN : String),
      (2 : ℕ) * ((kH - (1 : ℕ)) / (2 : ℕ)) + (1 : ℕ) = kH →
        (2 : ℕ) * ((kW - (1 : ℕ)) / (2 : ℕ)) + (1 : ℕ) = kW →
          ∀ (W₁ : Proofs.Kernel4 c ic kH kW) (b₁ : Proofs.Vec c) (W₂ : Proofs.Kernel4 c c kH kW) (b₂ : Proofs.Vec c)
            (W₃ : Proofs.Mat (c * h * w) d1) (b₃ : Proofs.Vec d1) (W₄ : Proofs.Mat d1 d1) (b₄ : Proofs.Vec d1)
            (W₅ : Proofs.Mat d1 nClasses) (b₅ : Proofs.Vec nClasses) (x : Proofs.Vec (ic * ((2 : ℕ) * h) * ((2 : ℕ) * w)))
            (σ : Fin c → Fin h → Fin w → Fin (2 : ℕ) × Fin (2 : ℕ)),
            Proofs.CnnFold.CnnLossSmoothAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ x σ →
              ∀ {L : Proofs.Vec nClasses → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec nClasses},
                Proofs.HasGradAt L (Proofs.mnistCnnNoBnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x) g →
                  Proofs.CnnFold.CnnNetLossTied xN cotN W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ x σ L g :=
  Proofs.CnnFold.cnn_net_lossGrad

/-- `Proofs.CifarFold.cifar_net_lossGrad` -/
theorem chk_cifar_net_lossGrad :
    ∀ {ic c1 c2 h w d1 nClasses kH kW : ℕ} (xN cotN : String),
      (2 : ℕ) * ((kH - (1 : ℕ)) / (2 : ℕ)) + (1 : ℕ) = kH →
        (2 : ℕ) * ((kW - (1 : ℕ)) / (2 : ℕ)) + (1 : ℕ) = kW →
          ∀ (W₁ : Proofs.Kernel4 c1 ic kH kW) (b₁ : Proofs.Vec c1) (W₂ : Proofs.Kernel4 c1 c1 kH kW) (b₂ : Proofs.Vec c1)
            (W₃ : Proofs.Kernel4 c2 c1 kH kW) (b₃ : Proofs.Vec c2) (W₄ : Proofs.Kernel4 c2 c2 kH kW) (b₄ : Proofs.Vec c2)
            (W₅ : Proofs.Mat (c2 * h * w) d1) (b₅ : Proofs.Vec d1) (W₆ : Proofs.Mat d1 d1) (b₆ : Proofs.Vec d1)
            (W₇ : Proofs.Mat d1 nClasses) (b₇ : Proofs.Vec nClasses)
            (x : Proofs.Vec (ic * ((2 : ℕ) * ((2 : ℕ) * h)) * ((2 : ℕ) * ((2 : ℕ) * w))))
            (σ₁ : Fin c1 → Fin ((2 : ℕ) * h) → Fin ((2 : ℕ) * w) → Fin (2 : ℕ) × Fin (2 : ℕ))
            (σ₂ : Fin c2 → Fin h → Fin w → Fin (2 : ℕ) × Fin (2 : ℕ)),
            Proofs.CifarFold.CifarLossSmoothAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ x σ₁ σ₂ →
              ∀ {L : Proofs.Vec nClasses → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec nClasses},
                Proofs.HasGradAt L (Proofs.cifarCnnForward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x) g →
                  Proofs.CifarFold.CifarNetLossTied xN cotN W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ x σ₁ σ₂ L g :=
  Proofs.CifarFold.cifar_net_lossGrad

/-- `Proofs.Cifar8TieG.cifar8_net_lossGrad` -/
theorem chk_cifar8_net_lossGrad :
    ∀ {ic c1 c2 c3 c4 h w d1 nClasses kH kW : ℕ} (xN cotN : String),
      (2 : ℕ) * ((kH - (1 : ℕ)) / (2 : ℕ)) + (1 : ℕ) = kH →
        (2 : ℕ) * ((kW - (1 : ℕ)) / (2 : ℕ)) + (1 : ℕ) = kW →
          ∀ (W₁ : Proofs.Kernel4 c1 ic kH kW) (b₁ : Proofs.Vec c1) (W₂ : Proofs.Kernel4 c1 c1 kH kW) (b₂ : Proofs.Vec c1)
            (W₃ : Proofs.Kernel4 c2 c1 kH kW) (b₃ : Proofs.Vec c2) (W₄ : Proofs.Kernel4 c2 c2 kH kW) (b₄ : Proofs.Vec c2)
            (W₅ : Proofs.Kernel4 c3 c2 kH kW) (b₅ : Proofs.Vec c3) (W₆ : Proofs.Kernel4 c3 c3 kH kW) (b₆ : Proofs.Vec c3)
            (W₇ : Proofs.Kernel4 c4 c3 kH kW) (b₇ : Proofs.Vec c4) (W₈ : Proofs.Kernel4 c4 c4 kH kW) (b₈ : Proofs.Vec c4)
            (W₉ : Proofs.Mat (c4 * h * w) d1) (b₉ : Proofs.Vec d1) (Wa : Proofs.Mat d1 d1) (ba : Proofs.Vec d1)
            (Wb : Proofs.Mat d1 nClasses) (bb : Proofs.Vec nClasses)
            (x :
              Proofs.Vec
                (ic * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * h)))) *
                  ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * w))))))
            (σ₁ :
              Fin c1 →
                Fin ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * h))) →
                  Fin ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * w))) → Fin (2 : ℕ) × Fin (2 : ℕ))
            (σ₂ : Fin c2 → Fin ((2 : ℕ) * ((2 : ℕ) * h)) → Fin ((2 : ℕ) * ((2 : ℕ) * w)) → Fin (2 : ℕ) × Fin (2 : ℕ))
            (σ₃ : Fin c3 → Fin ((2 : ℕ) * h) → Fin ((2 : ℕ) * w) → Fin (2 : ℕ) × Fin (2 : ℕ))
            (σ₄ : Fin c4 → Fin h → Fin w → Fin (2 : ℕ) × Fin (2 : ℕ)),
            Proofs.Cifar8TieG.Cifar8LossSmoothAt W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba x σ₁ σ₂ σ₃ σ₄ →
              ∀ {L : Proofs.Vec nClasses → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec nClasses},
                Proofs.HasGradAt L
                    (Proofs.cifarCnn8Forward W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb bb x) g →
                  Proofs.Cifar8TieG.Cifar8NetLossTied xN cotN W₁ b₁ W₂ b₂ W₃ b₃ W₄ b₄ W₅ b₅ W₆ b₆ W₇ b₇ W₈ b₈ W₉ b₉ Wa ba Wb
                    bb x σ₁ σ₂ σ₃ σ₄ L g :=
  Proofs.Cifar8TieG.cifar8_net_lossGrad

/-- `Proofs.Cifar8BnTieG.cifar8Bn_net_lossGrad` -/
theorem chk_cifar8Bn_net_lossGrad :
    ∀ {ic c1 c2 c3 c4 h w d1 nClasses kH kW : ℕ} (xN vN epsStr cotN : String),
      (2 : ℕ) * ((kH - (1 : ℕ)) / (2 : ℕ)) + (1 : ℕ) = kH →
        (2 : ℕ) * ((kW - (1 : ℕ)) / (2 : ℕ)) + (1 : ℕ) = kW →
          ∀ (W₁ : Proofs.Kernel4 c1 ic kH kW) (b₁ : Proofs.Vec c1) (ε₁ : ℝ) (γ₁ β₁ : Proofs.Vec c1)
            (W₂ : Proofs.Kernel4 c1 c1 kH kW) (b₂ : Proofs.Vec c1) (ε₂ : ℝ) (γ₂ β₂ : Proofs.Vec c1)
            (W₃ : Proofs.Kernel4 c2 c1 kH kW) (b₃ : Proofs.Vec c2) (ε₃ : ℝ) (γ₃ β₃ : Proofs.Vec c2)
            (W₄ : Proofs.Kernel4 c2 c2 kH kW) (b₄ : Proofs.Vec c2) (ε₄ : ℝ) (γ₄ β₄ : Proofs.Vec c2)
            (W₅ : Proofs.Kernel4 c3 c2 kH kW) (b₅ : Proofs.Vec c3) (ε₅ : ℝ) (γ₅ β₅ : Proofs.Vec c3)
            (W₆ : Proofs.Kernel4 c3 c3 kH kW) (b₆ : Proofs.Vec c3) (ε₆ : ℝ) (γ₆ β₆ : Proofs.Vec c3)
            (W₇ : Proofs.Kernel4 c4 c3 kH kW) (b₇ : Proofs.Vec c4) (ε₇ : ℝ) (γ₇ β₇ : Proofs.Vec c4)
            (W₈ : Proofs.Kernel4 c4 c4 kH kW) (b₈ : Proofs.Vec c4) (ε₈ : ℝ) (γ₈ β₈ : Proofs.Vec c4)
            (W₉ : Proofs.Mat (c4 * h * w) d1) (b₉ : Proofs.Vec d1) (Wa : Proofs.Mat d1 d1) (ba : Proofs.Vec d1)
            (Wb : Proofs.Mat d1 nClasses) (bb : Proofs.Vec nClasses),
            Proofs.Cifar8BnTieG.Cifar8BnPos ε₁ ε₂ ε₃ ε₄ ε₅ ε₆ ε₇ ε₈ →
              ∀
                (x :
                  Proofs.Vec
                    (ic * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * h)))) *
                      ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * w))))))
                (σ₁ :
                  Fin c1 →
                    Fin ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * h))) →
                      Fin ((2 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * w))) → Fin (2 : ℕ) × Fin (2 : ℕ))
                (σ₂ : Fin c2 → Fin ((2 : ℕ) * ((2 : ℕ) * h)) → Fin ((2 : ℕ) * ((2 : ℕ) * w)) → Fin (2 : ℕ) × Fin (2 : ℕ))
                (σ₃ : Fin c3 → Fin ((2 : ℕ) * h) → Fin ((2 : ℕ) * w) → Fin (2 : ℕ) × Fin (2 : ℕ))
                (σ₄ : Fin c4 → Fin h → Fin w → Fin (2 : ℕ) × Fin (2 : ℕ)),
                Proofs.Cifar8BnTieG.Cifar8BnLossSmoothAt W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅
                    ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba x σ₁ σ₂ σ₃ σ₄ →
                  ∀ {L : Proofs.Vec nClasses → Proofs.Vec (1 : ℕ)} {g : Proofs.Vec nClasses},
                    Proofs.HasGradAt L
                        (Proofs.cifarCnnBn8Forward W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃ W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅
                          β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x)
                        g →
                      Proofs.Cifar8BnTieG.Cifar8BnNetLossTied xN vN epsStr cotN W₁ b₁ ε₁ γ₁ β₁ W₂ b₂ ε₂ γ₂ β₂ W₃ b₃ ε₃ γ₃ β₃
                        W₄ b₄ ε₄ γ₄ β₄ W₅ b₅ ε₅ γ₅ β₅ W₆ b₆ ε₆ γ₆ β₆ W₇ b₇ ε₇ γ₇ β₇ W₈ b₈ ε₈ γ₈ β₈ W₉ b₉ Wa ba Wb bb x σ₁ σ₂
                        σ₃ σ₄ L g :=
  Proofs.Cifar8BnTieG.cifar8Bn_net_lossGrad

/-- `Proofs.dpMeanGrad_ne_globalBatchGrad` -/
theorem chk_dpMeanGrad_ne_globalBatchGrad :
    ∀ (θ : Proofs.Vec (1 : ℕ)),
      (Proofs.dpMean (R := (2 : ℕ)) fun (r : Fin (2 : ℕ)) => Proofs.lossGrad (Proofs.bnToyLoss (Proofs.dpToyShard r)) θ) ≠
        Proofs.lossGrad (Proofs.bnToyLoss Proofs.dpToyBatch) θ :=
  Proofs.dpMeanGrad_ne_globalBatchGrad

/-- `Proofs.dpMeanGrad_eq_globalBatchGrad_contiguous` -/
theorem chk_dpMeanGrad_eq_globalBatchGrad_contiguous :
    ∀ {R N P : ℕ} (ℓ : Fin (R * N) → Proofs.Vec P → ℝ)
      (θ : Proofs.Vec P),
      (∀ (k : Fin (R * N)), Proofs.LossDifferentiableAt (ℓ k) θ) →
        (Proofs.dpMean (R := R) fun (r : Fin R) =>
            Proofs.lossGrad (Proofs.meanLoss (M := N) fun (n : Fin N) => ℓ (finProdFinEquiv (r, n))) θ) =
          Proofs.lossGrad (Proofs.meanLoss ℓ) θ :=
  Proofs.dpMeanGrad_eq_globalBatchGrad_contiguous

/-- `Proofs.dpSyncGrad_eq_globalBatchGrad` -/
theorem chk_dpSyncGrad_eq_globalBatchGrad :
    ∀ {R N P : ℕ} (e : Fin R × Fin N ≃ Fin (R * N))
      (c : Fin (R * N) → Proofs.Vec P),
      Eq (α := Proofs.Vec P)
        (Proofs.dpMean (R := R) fun (r : Fin R) (i : Fin P) => (1 : ℝ) / ↑N * ∑ n : Fin N, c (e (r, n)) i)
        fun (i : Fin P) => (1 : ℝ) / ↑(R * N) * ∑ m : Fin (R * N), c m i :=
  Proofs.dpSyncGrad_eq_globalBatchGrad

/-- `Proofs.bnSyncTensor4_shard_eq_global` -/
theorem chk_bnSyncTensor4_shard_eq_global :
    ∀ (R N oc h w : ℕ),
      R ≠ (0 : ℕ) →
        N * (h * w) ≠ (0 : ℕ) →
          ∀ (ε : ℝ) (γ β : Proofs.Vec oc) (X : Proofs.Vec (R * N * (oc * (h * w)))) (r : Fin R),
            Proofs.bnSyncTensor4 N oc h w ε γ β
                (fun (c : Fin oc) =>
                  (1 : ℝ) / ↑R *
                    ∑ r' : Fin R,
                      Proofs.bnMean (N * (h * w))
                        (Proofs.Mat.unflatten (Proofs.bnchwFwd N oc h w (Proofs.batchShard R N (oc * (h * w)) X r')) c))
                (fun (c : Fin oc) =>
                  (1 : ℝ) / ↑R *
                    ∑ r' : Fin R,
                      Proofs.bnMeanSq (N * (h * w))
                        (Proofs.Mat.unflatten (Proofs.bnchwFwd N oc h w (Proofs.batchShard R N (oc * (h * w)) X r')) c))
                (Proofs.batchShard R N (oc * (h * w)) X r) =
              Proofs.batchShard R N (oc * (h * w)) (Proofs.bnBatchTensor4 (R * N) oc h w ε γ β X) r :=
  Proofs.bnSyncTensor4_shard_eq_global

/-- `Proofs.den_bnSyncBack_allReduce` -/
theorem chk_den_bnSyncBack_allReduce :
    ∀ {N oc h w : ℕ} (R : ℕ) (hR : (0 : ℕ) < R),
      N * (h * w) ≠ (0 : ℕ) →
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
                    Proofs.batchShard R N (oc * (h * w)) (Proofs.bnBatchTensor4GradInput (R * N) oc h w ε γ X DY) r :=
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

/-- `Proofs.StableHLO.resnet34FwdGraphSyncFull_shard` -/
theorem chk_resnet34FwdGraphSyncFull_shard :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (epsStr : String) {nCls : ℕ} (w : Proofs.R34BWeights nCls) (bf16 : Bool)
          (e :
            Fin R →
              Proofs.StableHLO.SHlo (N * ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))))))
          (X : Proofs.Vec (R * N * ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ)))))),
          (∀ (r : Fin R),
              Proofs.StableHLO.den (e r) =
                Proofs.batchShard R N ((3 : ℕ) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ))) * ((2 : ℕ) * ((2 : ℕ) * (56 : ℕ)))) X r) →
            ∀ (r : Fin R),
              Proofs.StableHLO.den (Proofs.StableHLO.resnet34FwdGraphSyncFull R hR N epsStr w bf16 e r) =
                Proofs.batchShard R N nCls (Proofs.resnet34ForwardBFull (R * N) w X) r :=
  Proofs.StableHLO.resnet34FwdGraphSyncFull_shard

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

/-- `Proofs.StableHLO.mobilenetv2FwdGraphSyncFull_shard` -/
theorem chk_mobilenetv2FwdGraphSyncFull_shard :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (epsStr : String) {nCls : ℕ} (w : Proofs.MNV2BWeights nCls)
          (e : Fin R → Proofs.StableHLO.SHlo (N * ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ)))))
          (X : Proofs.Vec (R * N * ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ))))),
          (∀ (r : Fin R),
              Proofs.StableHLO.den (e r) =
                Proofs.batchShard R N ((3 : ℕ) * ((2 : ℕ) * (112 : ℕ)) * ((2 : ℕ) * (112 : ℕ))) X r) →
            ∀ (r : Fin R),
              Proofs.StableHLO.den (Proofs.StableHLO.mobilenetv2FwdGraphSyncFull R hR N epsStr w e r) =
                Proofs.batchShard R N nCls (Proofs.mobilenetv2ForwardBFull (R * N) w X) r :=
  Proofs.StableHLO.mobilenetv2FwdGraphSyncFull_shard

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

/-- `Proofs.StableHLO.efficientnetFwdGraphSyncFull_shard` -/
theorem chk_efficientnetFwdGraphSyncFull_shard :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (epsStr : String) {nCls : ℕ} (w : Proofs.B0Weights nCls)
          (e : Fin R → Proofs.StableHLO.SHlo (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))))
          (X : Proofs.Vec (R * N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
          (∀ (r : Fin R), Proofs.StableHLO.den (e r) = Proofs.batchShard R N ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)) X r) →
            ∀ (r : Fin R),
              Proofs.StableHLO.den (Proofs.StableHLO.efficientnetFwdGraphSyncFull R hR N epsStr w e r) =
                Proofs.batchShard R N nCls (Proofs.efficientnetForwardBFull (R * N) w X) r :=
  Proofs.StableHLO.efficientnetFwdGraphSyncFull_shard

/-- `Proofs.EnetSyncTieG.efficientnet_net_syncTiedG` -/
theorem chk_efficientnet_net_syncTiedG :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (xN vN epsStr cotN dN : String) {nCls : ℕ} (w : Proofs.B0Weights nCls) (hεw : w.EpsPos)
          (x : Proofs.Vec (R * N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))) (g : Proofs.Vec (R * N * nCls))
          (gs : Fin R → Proofs.Vec (N * nCls)),
          (∀ (r : Fin R), gs r = Proofs.batchShard R N nCls (fun (i : Fin (R * N * nCls)) => ↑R * g i) r) →
            Proofs.EnetSyncTieG.enetNetSyncTiedG R hR N xN vN epsStr cotN dN w hεw x g gs :=
  Proofs.EnetSyncTieG.efficientnet_net_syncTiedG

/-- `Proofs.StableHLO.resnet50FwdGraphSyncFull_shard` -/
theorem chk_resnet50FwdGraphSyncFull_shard :
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
                  Proofs.StableHLO.den (Proofs.StableHLO.resnet50FwdGraphSyncFull R hR N q epsStr w e r) =
                    Proofs.batchShard R N nCls (Proofs.resnet50ForwardBFull (R * N) q w X) r :=
  Proofs.StableHLO.resnet50FwdGraphSyncFull_shard

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

/-- `Proofs.StableHLO.mnv4FwdGraphSyncFull_shard` -/
theorem chk_mnv4FwdGraphSyncFull_shard :
    ∀ (R : ℕ) (hR : (0 : ℕ) < R) (N : ℕ),
      (0 : ℕ) < N →
        ∀ (epsStr : String) {nCls : ℕ} (w : Proofs.StableHLO.Mnv4BWeights nCls)
          (e : Fin R → Proofs.StableHLO.SHlo (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ))))
          (X : Proofs.Vec (R * N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
          (∀ (r : Fin R), Proofs.StableHLO.den (e r) = Proofs.batchShard R N ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)) X r) →
            ∀ (r : Fin R),
              Proofs.StableHLO.den (Proofs.StableHLO.mnv4FwdGraphSyncFull R hR N epsStr w e r) =
                Proofs.batchShard R N nCls (Proofs.StableHLO.mobilenetv4ForwardBFull (R * N) w X) r :=
  Proofs.StableHLO.mnv4FwdGraphSyncFull_shard

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
      (h_pool : Proofs.StemPoolSmoothAt N (56 : ℕ) (56 : ℕ) (Proofs.StableHLO.cbReluStridedB N Ws bs εs γs βs x))
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
          (Proofs.HasVJP.backward (Proofs.bnBatchLAHasVJP N (64 : ℕ) ((2 : ℕ) * (56 : ℕ)) ((2 : ℕ) * (56 : ℕ)) εs hεs γs βs)
            (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2 Ws bs) x))
          (Proofs.StableHLO.cbReluStridedB N Ws bs εs γs βs x) hb16.fst.backward hb15.fst.backward hb14.fst.backward
          hb13.fst.backward hb12.fst.backward hb11.fst.backward hb10.fst.backward hb9.fst.backward hb8.fst.backward
          hb7.fst.backward hb6.fst.backward hb5.fst.backward hb4.fst.backward hb3.fst.backward hb2.fst.backward
          hb1.fst.backward fun (i : Fin (N * ((64 : ℕ) * ((2 : ℕ) * (56 : ℕ)) * ((2 : ℕ) * (56 : ℕ))))) =>
          Proofs.StableHLO.bnBatchLA N (64 : ℕ) ((2 : ℕ) * (56 : ℕ)) ((2 : ℕ) * (56 : ℕ)) εs γs βs
              (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2 Ws bs) x) i >
            (0 : ℝ)) =
        (Proofs.r34BFullHasVJPAt (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11 b12
            b13 b14 b15 b16 (Proofs.r34HeadB N (7 : ℕ) (7 : ℕ) Wd bd) x
            ⟨Proofs.r34StemBHasVJPAt N (56 : ℕ) (56 : ℕ) Ws bs εs hεs γs βs x h_stem h_pool,
              Proofs.r34StemB_differentiableAt N (56 : ℕ) (56 : ℕ) Ws bs εs hεs γs βs x h_stem h_pool⟩
            hb1 hb2 hb3 hb4 hb5 hb6 hb7 hb8 hb9 hb10 hb11 hb12 hb13 hb14 hb15 hb16
            ⟨(Proofs.r34HeadBHasVJP N (7 : ℕ) (7 : ℕ) Wd bd).toHasVJPAt
                (Proofs.opaqueA16 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11
                  b12 b13 b14 b15 b16 x),
              Proofs.r34HeadB_differentiable N (7 : ℕ) (7 : ℕ) Wd bd
                (Proofs.opaqueA16 (Proofs.r34StemB N (56 : ℕ) (56 : ℕ) Ws bs εs γs βs) b1 b2 b3 b4 b5 b6 b7 b8 b9 b10 b11
                  b12 b13 b14 b15 b16 x)⟩).backward :=
  Proofs.r34InputGradB_eq_r34B_full_vjp

/-- `Proofs.efficientnetInputGradBFull_correct` -/
theorem chk_efficientnetInputGradBFull_correct :
    ∀ (N : ℕ) {nCls : ℕ} (w : Proofs.B0Weights nCls) (hεw : w.EpsPos)
      (x : Proofs.Vec (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))) (dy : Proofs.Vec (N * nCls))
      (i : Fin (N * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
      Proofs.efficientnetInputGradBFull N w.sW w.hW w.fcW
          (Proofs.HasVJP.backward (f := Proofs.StableHLO.bnBatchLA N (32 : ℕ) (112 : ℕ) (112 : ℕ) w.sε w.sγ w.sβ)
            (Proofs.bnBatchLAHasVJP N (32 : ℕ) (112 : ℕ) (112 : ℕ) w.sε hεw.s w.sγ w.sβ)
            (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2Xla w.sW w.sb) x))
          (Proofs.HasVJP.backward (Proofs.swishHasVJP (N * ((32 : ℕ) * (112 : ℕ) * (112 : ℕ))))
            (Proofs.StableHLO.bnBatchLA N (32 : ℕ) (112 : ℕ) (112 : ℕ) w.sε w.sγ w.sβ
              (Proofs.StableHLO.batchMap N (Proofs.flatConvStride2Xla w.sW w.sb) x)))
          (Proofs.HasVJP.backward (f := Proofs.StableHLO.bnBatchLA N (1280 : ℕ) (7 : ℕ) (7 : ℕ) w.hε w.hγ w.hβ)
            (Proofs.bnBatchLAHasVJP N (1280 : ℕ) (7 : ℕ) (7 : ℕ) w.hε hεw.h w.hγ w.hβ)
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
          (Proofs.HasVJP.backward (Proofs.swishHasVJP (N * ((1280 : ℕ) * (7 : ℕ) * (7 : ℕ))))
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
          ((Proofs.mbNoExpWHasVJP N (112 : ℕ) (112 : ℕ) w.b1 hεw.b1.d hεw.b1.p).backward
            (Proofs.opaqueA0 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) x))
          ((Proofs.mbStridedWHasVJP N (56 : ℕ) (56 : ℕ) w.b2 hεw.b2.e hεw.b2.d hεw.b2.p).backward
            (Proofs.opaqueA1 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1) x))
          ((Proofs.mbResidWHasVJP N (56 : ℕ) (56 : ℕ) w.b3 hεw.b3.e hεw.b3.d hεw.b3.p).backward
            (Proofs.opaqueA2 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) x))
          ((Proofs.mbStridedWHasVJP N (28 : ℕ) (28 : ℕ) w.b4 hεw.b4.e hεw.b4.d hεw.b4.p).backward
            (Proofs.opaqueA3 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3) x))
          ((Proofs.mbResidWHasVJP N (28 : ℕ) (28 : ℕ) w.b5 hεw.b5.e hεw.b5.d hεw.b5.p).backward
            (Proofs.opaqueA4 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) x))
          ((Proofs.mbStridedWHasVJP N (14 : ℕ) (14 : ℕ) w.b6 hεw.b6.e hεw.b6.d hεw.b6.p).backward
            (Proofs.opaqueA5 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5) x))
          ((Proofs.mbResidWHasVJP N (14 : ℕ) (14 : ℕ) w.b7 hεw.b7.e hεw.b7.d hεw.b7.p).backward
            (Proofs.opaqueA6 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) x))
          ((Proofs.mbResidWHasVJP N (14 : ℕ) (14 : ℕ) w.b8 hεw.b8.e hεw.b8.d hεw.b8.p).backward
            (Proofs.opaqueA7 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7) x))
          ((Proofs.mbExpWHasVJP N (14 : ℕ) (14 : ℕ) w.b9 hεw.b9.e hεw.b9.d hεw.b9.p).backward
            (Proofs.opaqueA8 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) x))
          ((Proofs.mbResidWHasVJP N (14 : ℕ) (14 : ℕ) w.b10 hεw.b10.e hεw.b10.d hεw.b10.p).backward
            (Proofs.opaqueA9 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9) x))
          ((Proofs.mbResidWHasVJP N (14 : ℕ) (14 : ℕ) w.b11 hεw.b11.e hεw.b11.d hεw.b11.p).backward
            (Proofs.opaqueA10 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) x))
          ((Proofs.mbStridedWHasVJP N (7 : ℕ) (7 : ℕ) w.b12 hεw.b12.e hεw.b12.d hεw.b12.p).backward
            (Proofs.opaqueA11 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11) x))
          ((Proofs.mbResidWHasVJP N (7 : ℕ) (7 : ℕ) w.b13 hεw.b13.e hεw.b13.d hεw.b13.p).backward
            (Proofs.opaqueA12 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) x))
          ((Proofs.mbResidWHasVJP N (7 : ℕ) (7 : ℕ) w.b14 hεw.b14.e hεw.b14.d hεw.b14.p).backward
            (Proofs.opaqueA13 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13) x))
          ((Proofs.mbResidWHasVJP N (7 : ℕ) (7 : ℕ) w.b15 hεw.b15.e hεw.b15.d hεw.b15.p).backward
            (Proofs.opaqueA14 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13)
              (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b14) x))
          ((Proofs.mbExpWHasVJP N (7 : ℕ) (7 : ℕ) w.b16 hεw.b16.e hεw.b16.d hεw.b16.p).backward
            (Proofs.opaqueA15 (Proofs.stemB N w.sW w.sb w.sε w.sγ w.sβ) (Proofs.mbNoExpW N (112 : ℕ) (112 : ℕ) w.b1)
              (Proofs.mbStridedW N (56 : ℕ) (56 : ℕ) w.b2) (Proofs.mbResidW N (56 : ℕ) (56 : ℕ) w.b3)
              (Proofs.mbStridedW N (28 : ℕ) (28 : ℕ) w.b4) (Proofs.mbResidW N (28 : ℕ) (28 : ℕ) w.b5)
              (Proofs.mbStridedW N (14 : ℕ) (14 : ℕ) w.b6) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b7)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b8) (Proofs.mbExpW N (14 : ℕ) (14 : ℕ) w.b9)
              (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b10) (Proofs.mbResidW N (14 : ℕ) (14 : ℕ) w.b11)
              (Proofs.mbStridedW N (7 : ℕ) (7 : ℕ) w.b12) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b13)
              (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b14) (Proofs.mbResidW N (7 : ℕ) (7 : ℕ) w.b15) x))
          dy i =
        ∑ j : Fin (N * nCls), Proofs.pdiv (Proofs.efficientnetForwardBFull N w) x i j * dy j :=
  Proofs.efficientnetInputGradBFull_correct

/-- `Proofs.convnextImagenetInputGradB_eq_vjp` -/
theorem chk_convnextImagenetInputGradB_eq_vjp :
    ∀ {gf : Proofs.GeluForm} (B : ℕ) (w : Proofs.CnxTWeightsCh (1000 : ℕ))
      (hsε : (0 : ℝ) < w.sε) (h1 : ∀ (i : Fin (3 : ℕ)), (0 : ℝ) < (w.s1 i).εn) (hd1 : (0 : ℝ) < w.d1.ε)
      (h2 : ∀ (i : Fin (3 : ℕ)), (0 : ℝ) < (w.s2 i).εn) (hd2 : (0 : ℝ) < w.d2.ε)
      (h3 : ∀ (i : Fin (9 : ℕ)), (0 : ℝ) < (w.s3 i).εn) (hd3 : (0 : ℝ) < w.d3.ε)
      (h4 : ∀ (i : Fin (3 : ℕ)), (0 : ℝ) < (w.s4 i).εn) (hhε : (0 : ℝ) < w.hε)
      (x : Proofs.Vec (B * ((3 : ℕ) * (224 : ℕ) * (224 : ℕ)))),
      Proofs.convnextInputGradB B w.Wd (Proofs.padOdd w.sW) (Proofs.chanLNTensor3Back (96 : ℕ) (56 : ℕ) (56 : ℕ) w.sε w.sγ)
          (Proofs.cnxSavedB0 B w x) (Proofs.rowLNVecFlatBack (1 : ℕ) (768 : ℕ) w.hε w.hγ) (Proofs.cnxSavedB9 gf B w x)
          (Proofs.cnxStageChKBack gf (3 : ℕ) w.s1) (Proofs.cnxSavedB1 B w x)
          (fun (u : Proofs.Vec ((96 : ℕ) * (56 : ℕ) * (56 : ℕ))) =>
            Proofs.cnxDownBack (Proofs.padOdd w.d1.W) (Proofs.chanLNTensor3Back (96 : ℕ) (56 : ℕ) (56 : ℕ) w.d1.ε w.d1.γ u))
          (Proofs.cnxSavedB2 gf B w x) (Proofs.cnxStageChKBack gf (3 : ℕ) w.s2) (Proofs.cnxSavedB3 gf B w x)
          (fun (u : Proofs.Vec ((192 : ℕ) * (28 : ℕ) * (28 : ℕ))) =>
            Proofs.cnxDownBack (Proofs.padOdd w.d2.W)
              (Proofs.chanLNTensor3Back (192 : ℕ) (28 : ℕ) (28 : ℕ) w.d2.ε w.d2.γ u))
          (Proofs.cnxSavedB4 gf B w x) (Proofs.cnxStageChKBack gf (9 : ℕ) w.s3) (Proofs.cnxSavedB5 gf B w x)
          (fun (u : Proofs.Vec ((384 : ℕ) * (14 : ℕ) * (14 : ℕ))) =>
            Proofs.cnxDownBack (Proofs.padOdd w.d3.W)
              (Proofs.chanLNTensor3Back (384 : ℕ) (14 : ℕ) (14 : ℕ) w.d3.ε w.d3.γ u))
          (Proofs.cnxSavedB6 gf B w x) (Proofs.cnxStageChKBack gf (3 : ℕ) w.s4) (Proofs.cnxSavedB7 gf B w x) =
        Proofs.HasVJP.backward
          (Proofs.batchMapHasVJP
            (Proofs.dense w.Wd w.bd ∘
              Proofs.rowLNVecFlat (1 : ℕ) (768 : ℕ) w.hε w.hγ w.hβ ∘
                Proofs.globalAvgPoolFlat (768 : ℕ) (7 : ℕ) (7 : ℕ) ∘
                  Proofs.convNextStageChK gf (3 : ℕ) w.s4 ∘
                    Proofs.cnxDownChW (7 : ℕ) (7 : ℕ) w.d3 ∘
                      Proofs.convNextStageChK gf (9 : ℕ) w.s3 ∘
                        Proofs.cnxDownChW (14 : ℕ) (14 : ℕ) w.d2 ∘
                          Proofs.convNextStageChK gf (3 : ℕ) w.s2 ∘
                            Proofs.cnxDownChW (28 : ℕ) (28 : ℕ) w.d1 ∘
                              Proofs.convNextStageChK gf (3 : ℕ) w.s1 ∘
                                Proofs.chanLNTensor3 (96 : ℕ) (56 : ℕ) (56 : ℕ) w.sε w.sγ w.sβ ∘
                                  Proofs.flatConvStride4 w.sW w.sb)
            (Proofs.convNextForwardTChHasVJP gf w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε)
            (Proofs.convNextForwardTCh_differentiable w hsε h1 hd1 h2 hd2 h3 hd3 h4 hhε))
          x :=
  Proofs.convnextImagenetInputGradB_eq_vjp

/-- `Proofs.FloatModel.linear_e4m3_argmax_preserved` -/
theorem chk_linear_e4m3_argmax_preserved :
    ∀ (M L : Proofs.FloatModel),
      M.u ≤ Proofs.u32 →
        L.u ≤ Proofs.uE4M3 →
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

/-- `Proofs.TrainedCnnDescent.trained_cnn_conv2_sgd_descends_concrete` -/
theorem chk_trained_cnn_conv2_sgd_descends_concrete :
    Proofs.cnnConv2KernelLoss Proofs.TrainedCnnDescent.b2
        Proofs.TrainedCnnDescent.x1V Proofs.TrainedCnnDescent.W3 Proofs.TrainedCnnDescent.b3 Proofs.TrainedCnnDescent.W4
        Proofs.TrainedCnnDescent.b4 Proofs.TrainedCnnDescent.W5 Proofs.TrainedCnnDescent.b5 Proofs.TrainedCnnDescent.lbl
        (Proofs.TrainedCnnDescent.W2.flatten -
          ((1 : ℝ) / (2 : ℝ) ^ (42 : ℕ)) •
            Proofs.gradAt
              (Proofs.cnnConv2KernelLoss Proofs.TrainedCnnDescent.b2 Proofs.TrainedCnnDescent.x1V
                Proofs.TrainedCnnDescent.W3 Proofs.TrainedCnnDescent.b3 Proofs.TrainedCnnDescent.W4
                Proofs.TrainedCnnDescent.b4 Proofs.TrainedCnnDescent.W5 Proofs.TrainedCnnDescent.b5
                Proofs.TrainedCnnDescent.lbl)
              Proofs.TrainedCnnDescent.W2.flatten) ≤
      Proofs.cnnConv2KernelLoss Proofs.TrainedCnnDescent.b2 Proofs.TrainedCnnDescent.x1V Proofs.TrainedCnnDescent.W3
          Proofs.TrainedCnnDescent.b3 Proofs.TrainedCnnDescent.W4 Proofs.TrainedCnnDescent.b4 Proofs.TrainedCnnDescent.W5
          Proofs.TrainedCnnDescent.b5 Proofs.TrainedCnnDescent.lbl Proofs.TrainedCnnDescent.W2.flatten -
        ((1 : ℝ) / (2 : ℝ) ^ (42 : ℕ) *
            ∑ idx : Fin ((2 : ℕ) * (2 : ℕ) * (3 : ℕ) * (3 : ℕ)),
              Proofs.gradAt
                  (Proofs.cnnConv2KernelLoss Proofs.TrainedCnnDescent.b2 Proofs.TrainedCnnDescent.x1V
                    Proofs.TrainedCnnDescent.W3 Proofs.TrainedCnnDescent.b3 Proofs.TrainedCnnDescent.W4
                    Proofs.TrainedCnnDescent.b4 Proofs.TrainedCnnDescent.W5 Proofs.TrainedCnnDescent.b5
                    Proofs.TrainedCnnDescent.lbl)
                  Proofs.TrainedCnnDescent.W2.flatten idx ^
                (2 : ℕ)) /
          (2 : ℝ) :=
  Proofs.TrainedCnnDescent.trained_cnn_conv2_sgd_descends_concrete

/-- `Proofs.TrainedCnnDescent.trained_cnn_conv2_bias_sgd_descends_concrete` -/
theorem chk_trained_cnn_conv2_bias_sgd_descends_concrete :
    Proofs.cnnConv2BiasLoss
        Proofs.TrainedCnnDescent.W2 Proofs.TrainedCnnDescent.x1V Proofs.TrainedCnnDescent.W3 Proofs.TrainedCnnDescent.b3
        Proofs.TrainedCnnDescent.W4 Proofs.TrainedCnnDescent.b4 Proofs.TrainedCnnDescent.W5 Proofs.TrainedCnnDescent.b5
        Proofs.TrainedCnnDescent.lbl
        (Proofs.TrainedCnnDescent.b2 -
          ((1 : ℝ) / (2 : ℝ) ^ (36 : ℕ)) •
            Proofs.gradAt
              (Proofs.cnnConv2BiasLoss Proofs.TrainedCnnDescent.W2 Proofs.TrainedCnnDescent.x1V Proofs.TrainedCnnDescent.W3
                Proofs.TrainedCnnDescent.b3 Proofs.TrainedCnnDescent.W4 Proofs.TrainedCnnDescent.b4
                Proofs.TrainedCnnDescent.W5 Proofs.TrainedCnnDescent.b5 Proofs.TrainedCnnDescent.lbl)
              Proofs.TrainedCnnDescent.b2) ≤
      Proofs.cnnConv2BiasLoss Proofs.TrainedCnnDescent.W2 Proofs.TrainedCnnDescent.x1V Proofs.TrainedCnnDescent.W3
          Proofs.TrainedCnnDescent.b3 Proofs.TrainedCnnDescent.W4 Proofs.TrainedCnnDescent.b4 Proofs.TrainedCnnDescent.W5
          Proofs.TrainedCnnDescent.b5 Proofs.TrainedCnnDescent.lbl Proofs.TrainedCnnDescent.b2 -
        ((1 : ℝ) / (2 : ℝ) ^ (36 : ℕ) *
            ∑ o : Fin (2 : ℕ),
              Proofs.gradAt
                  (Proofs.cnnConv2BiasLoss Proofs.TrainedCnnDescent.W2 Proofs.TrainedCnnDescent.x1V
                    Proofs.TrainedCnnDescent.W3 Proofs.TrainedCnnDescent.b3 Proofs.TrainedCnnDescent.W4
                    Proofs.TrainedCnnDescent.b4 Proofs.TrainedCnnDescent.W5 Proofs.TrainedCnnDescent.b5
                    Proofs.TrainedCnnDescent.lbl)
                  Proofs.TrainedCnnDescent.b2 o ^
                (2 : ℕ)) /
          (2 : ℝ) :=
  Proofs.TrainedCnnDescent.trained_cnn_conv2_bias_sgd_descends_concrete

/-- `Proofs.TrainedCnnDescentConv1.trained_cnn_conv1_sgd_descends_concrete` -/
theorem chk_trained_cnn_conv1_sgd_descends_concrete :
    Proofs.cnnConv1KernelLoss
        Proofs.TrainedCnnDescentConv1.b1 Proofs.TrainedCnnDescentConv1.T0 Proofs.TrainedCnnDescentConv1.W2
        Proofs.TrainedCnnDescentConv1.b2 Proofs.TrainedCnnDescentConv1.W3 Proofs.TrainedCnnDescentConv1.b3
        Proofs.TrainedCnnDescentConv1.W4 Proofs.TrainedCnnDescentConv1.b4 Proofs.TrainedCnnDescentConv1.W5
        Proofs.TrainedCnnDescentConv1.b5 Proofs.TrainedCnnDescentConv1.lbl
        (Proofs.TrainedCnnDescentConv1.W1.flatten -
          ((1 : ℝ) / (2 : ℝ) ^ (48 : ℕ)) •
            Proofs.gradAt
              (Proofs.cnnConv1KernelLoss Proofs.TrainedCnnDescentConv1.b1 Proofs.TrainedCnnDescentConv1.T0
                Proofs.TrainedCnnDescentConv1.W2 Proofs.TrainedCnnDescentConv1.b2 Proofs.TrainedCnnDescentConv1.W3
                Proofs.TrainedCnnDescentConv1.b3 Proofs.TrainedCnnDescentConv1.W4 Proofs.TrainedCnnDescentConv1.b4
                Proofs.TrainedCnnDescentConv1.W5 Proofs.TrainedCnnDescentConv1.b5 Proofs.TrainedCnnDescentConv1.lbl)
              Proofs.TrainedCnnDescentConv1.W1.flatten) ≤
      Proofs.cnnConv1KernelLoss Proofs.TrainedCnnDescentConv1.b1 Proofs.TrainedCnnDescentConv1.T0
          Proofs.TrainedCnnDescentConv1.W2 Proofs.TrainedCnnDescentConv1.b2 Proofs.TrainedCnnDescentConv1.W3
          Proofs.TrainedCnnDescentConv1.b3 Proofs.TrainedCnnDescentConv1.W4 Proofs.TrainedCnnDescentConv1.b4
          Proofs.TrainedCnnDescentConv1.W5 Proofs.TrainedCnnDescentConv1.b5 Proofs.TrainedCnnDescentConv1.lbl
          Proofs.TrainedCnnDescentConv1.W1.flatten -
        ((1 : ℝ) / (2 : ℝ) ^ (48 : ℕ) *
            ∑ idx : Fin ((2 : ℕ) * (1 : ℕ) * (3 : ℕ) * (3 : ℕ)),
              Proofs.gradAt
                  (Proofs.cnnConv1KernelLoss Proofs.TrainedCnnDescentConv1.b1 Proofs.TrainedCnnDescentConv1.T0
                    Proofs.TrainedCnnDescentConv1.W2 Proofs.TrainedCnnDescentConv1.b2 Proofs.TrainedCnnDescentConv1.W3
                    Proofs.TrainedCnnDescentConv1.b3 Proofs.TrainedCnnDescentConv1.W4 Proofs.TrainedCnnDescentConv1.b4
                    Proofs.TrainedCnnDescentConv1.W5 Proofs.TrainedCnnDescentConv1.b5 Proofs.TrainedCnnDescentConv1.lbl)
                  Proofs.TrainedCnnDescentConv1.W1.flatten idx ^
                (2 : ℕ)) /
          (2 : ℝ) :=
  Proofs.TrainedCnnDescentConv1.trained_cnn_conv1_sgd_descends_concrete

/-- `Proofs.TrainedCnnDescentConv1.trained_cnn_conv1_bias_sgd_descends_concrete` -/
theorem chk_trained_cnn_conv1_bias_sgd_descends_concrete :
    Proofs.cnnConv1BiasLoss
        Proofs.TrainedCnnDescentConv1.W1 Proofs.TrainedCnnDescentConv1.T0 Proofs.TrainedCnnDescentConv1.W2
        Proofs.TrainedCnnDescentConv1.b2 Proofs.TrainedCnnDescentConv1.W3 Proofs.TrainedCnnDescentConv1.b3
        Proofs.TrainedCnnDescentConv1.W4 Proofs.TrainedCnnDescentConv1.b4 Proofs.TrainedCnnDescentConv1.W5
        Proofs.TrainedCnnDescentConv1.b5 Proofs.TrainedCnnDescentConv1.lbl
        (Proofs.TrainedCnnDescentConv1.b1 -
          ((1 : ℝ) / (2 : ℝ) ^ (45 : ℕ)) •
            Proofs.gradAt
              (Proofs.cnnConv1BiasLoss Proofs.TrainedCnnDescentConv1.W1 Proofs.TrainedCnnDescentConv1.T0
                Proofs.TrainedCnnDescentConv1.W2 Proofs.TrainedCnnDescentConv1.b2 Proofs.TrainedCnnDescentConv1.W3
                Proofs.TrainedCnnDescentConv1.b3 Proofs.TrainedCnnDescentConv1.W4 Proofs.TrainedCnnDescentConv1.b4
                Proofs.TrainedCnnDescentConv1.W5 Proofs.TrainedCnnDescentConv1.b5 Proofs.TrainedCnnDescentConv1.lbl)
              Proofs.TrainedCnnDescentConv1.b1) ≤
      Proofs.cnnConv1BiasLoss Proofs.TrainedCnnDescentConv1.W1 Proofs.TrainedCnnDescentConv1.T0
          Proofs.TrainedCnnDescentConv1.W2 Proofs.TrainedCnnDescentConv1.b2 Proofs.TrainedCnnDescentConv1.W3
          Proofs.TrainedCnnDescentConv1.b3 Proofs.TrainedCnnDescentConv1.W4 Proofs.TrainedCnnDescentConv1.b4
          Proofs.TrainedCnnDescentConv1.W5 Proofs.TrainedCnnDescentConv1.b5 Proofs.TrainedCnnDescentConv1.lbl
          Proofs.TrainedCnnDescentConv1.b1 -
        ((1 : ℝ) / (2 : ℝ) ^ (45 : ℕ) *
            ∑ o : Fin (2 : ℕ),
              Proofs.gradAt
                  (Proofs.cnnConv1BiasLoss Proofs.TrainedCnnDescentConv1.W1 Proofs.TrainedCnnDescentConv1.T0
                    Proofs.TrainedCnnDescentConv1.W2 Proofs.TrainedCnnDescentConv1.b2 Proofs.TrainedCnnDescentConv1.W3
                    Proofs.TrainedCnnDescentConv1.b3 Proofs.TrainedCnnDescentConv1.W4 Proofs.TrainedCnnDescentConv1.b4
                    Proofs.TrainedCnnDescentConv1.W5 Proofs.TrainedCnnDescentConv1.b5 Proofs.TrainedCnnDescentConv1.lbl)
                  Proofs.TrainedCnnDescentConv1.b1 o ^
                (2 : ℕ)) /
          (2 : ℝ) :=
  Proofs.TrainedCnnDescentConv1.trained_cnn_conv1_bias_sgd_descends_concrete

/-- `Proofs.lipschitz_margin_certified_radius` -/
theorem chk_lipschitz_margin_certified_radius :
    ∀ {k : ℕ} {E : Type u_1} [NormedAddCommGroup E]
      {f : E → EuclideanSpace ℝ (Fin k)} {L : ℝ},
      Proofs.LipschitzL2 L f →
        (0 : ℝ) < L →
          ∀ {x δ : E} {i : Fin k} {m : ℝ},
            (∀ (j : Fin k), j ≠ i → m ≤ (f x).ofLp i - (f x).ofLp j) →
              ‖δ‖ < m / (√(2 : ℝ) * L) → ∀ (j : Fin k), j ≠ i → (f (x + δ)).ofLp j < (f (x + δ)).ofLp i :=
  Proofs.lipschitz_margin_certified_radius

/-- `Proofs.Robustness.scorecard_sdp` -/
theorem chk_scorecard_sdp :
    Proofs.Robustness.sdpCappedCerts.length = (8 : ℕ) ∧
      ∀ p ∈ Proofs.Robustness.sdpCappedCerts, Proofs.Robustness.CertifiedAt Proofs.Robustness.mlpS (1 / 10 : ℝ) p.2.1 p.2.2 :=
  Proofs.Robustness.scorecard_sdp

/-- `Proofs.smoothing_certified_radius_classifier` -/
theorem chk_smoothing_certified_radius_classifier :
    ∀ {n k : ℕ} {σ : ℝ},
      (0 : ℝ) < σ →
        ∀ {C : EuclideanSpace ℝ (Fin (n + (1 : ℕ))) → Fin k},
          Measurable C →
            (∀ (c : Fin k) (x : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))),
                (@MeasureTheory.integral _ _ _ _
                    (WithLp.measurableSpace (2 : ENNReal) ((i : Fin (n + (1 : ℕ))) → (fun (_ : Fin (n + (1 : ℕ))) => ℝ) i))
                    (ProbabilityTheory.stdGaussian (EuclideanSpace ℝ (Fin (n + (1 : ℕ)))))
                    fun (z : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))) => if C (x + σ • z) = c then (1 : ℝ) else (0 : ℝ)) ∈
                  Set.Ioo (0 : ℝ) (1 : ℝ)) →
              ∀ {x δ : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))} {i : Fin k},
                ‖δ‖ <
                    σ *
                      Proofs.stdNormalQuantile
                        (@MeasureTheory.integral _ _ _ _
                          (WithLp.measurableSpace (2 : ENNReal)
                            ((i : Fin (n + (1 : ℕ))) → (fun (_ : Fin (n + (1 : ℕ))) => ℝ) i))
                          (ProbabilityTheory.stdGaussian (EuclideanSpace ℝ (Fin (n + (1 : ℕ)))))
                          fun (z : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))) =>
                          if C (x + σ • z) = i then (1 : ℝ) else (0 : ℝ)) →
                  ∀ (j : Fin k),
                    j ≠ i →
                      LT.lt (α := ℝ)
                        (@MeasureTheory.integral _ _ _ _
                          (WithLp.measurableSpace (2 : ENNReal)
                            ((i : Fin (n + (1 : ℕ))) → (fun (_ : Fin (n + (1 : ℕ))) => ℝ) i))
                          (ProbabilityTheory.stdGaussian (EuclideanSpace ℝ (Fin (n + (1 : ℕ)))))
                          fun (z : EuclideanSpace ℝ (Fin (n + (1 : ℕ)))) =>
                          if C (x + δ + σ • z) = j then (1 : ℝ) else (0 : ℝ))
                        (@MeasureTheory.integral _ _ _ _
                          (WithLp.measurableSpace (2 : ENNReal)
                            ((i : Fin (n + (1 : ℕ))) → (fun (_ : Fin (n + (1 : ℕ))) => ℝ) i))
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
