import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.GradcheckHelpers
import LeanMlir.ViTRender
import LeanMlir.SpecHelpers
import LeanMlir.Train
import LeanMlir.VerifiedTrain
import LeanMlir.VerifiedSpec
import LeanMlir.VerifiedNets
import LeanMlir.Ddpm
import LeanMlir.Cam
-- VJP proofs (Attention pulls in Tensor/MLP/Residual/SE/LayerNorm/BatchNorm
-- transitively; CNN + Depthwise need explicit imports).
import LeanMlir.Proofs.Architectures.Attention
import LeanMlir.Proofs.Architectures.CNN
import LeanMlir.Proofs.Architectures.Depthwise
-- End-to-end whole-network VJP compositions (each builds on the CNN/Depthwise
-- machinery; their own file imports pull in everything transitively).
import LeanMlir.Proofs.Architectures.MobileNetV2
import LeanMlir.Proofs.Architectures.ConvNeXt
import LeanMlir.Proofs.Architectures.EfficientNet
import LeanMlir.Proofs.Architectures.MobileNetV2Close
import LeanMlir.Proofs.Codegen.MobileNetV2RenderPC
import LeanMlir.Proofs.Architectures.MobileNetV2ChainClose
import LeanMlir.Proofs.Foundation.ConvLossFold
import LeanMlir.Proofs.Architectures.EfficientNetClose
import LeanMlir.Proofs.Codegen.EfficientNetRenderPC
import LeanMlir.Proofs.Architectures.EfficientNetChainClose
import LeanMlir.Proofs.Architectures.EfficientNetFullB0
import LeanMlir.Proofs.Foundation.ResNet34Close
import LeanMlir.Proofs.Codegen.ResNet34RenderPC
import LeanMlir.Proofs.Foundation.ResNet34ChainClose
import LeanMlir.Proofs.Architectures.ConvNeXtClose
import LeanMlir.Proofs.Architectures.ConvNeXtChainClose
import LeanMlir.Proofs.Architectures.ViTFwdGraph
import LeanMlir.Proofs.Architectures.ViTClose
import LeanMlir.Proofs.Architectures.ViTChainClose
import LeanMlir.Proofs.Architectures.ViTVecLN
import LeanMlir.Proofs.Architectures.ViTMultiHead
import LeanMlir.Proofs.Architectures.ViTDepthK
import LeanMlir.Proofs.Architectures.MobileNetV2FullPaper
import LeanMlir.Proofs.Architectures.ConvNeXtFullT
-- ℝ→Float32 bridge, Tier 1: standard-model rounding bounds for the toy nets.
import LeanMlir.Proofs.Float.FloatBridge
-- Inexact-gradient descent over ℝ: the keystone the float budgets plug into.
import LeanMlir.Proofs.SgdDescent
import LeanMlir.Proofs.SgdDescentLinear
import LeanMlir.Proofs.SgdDescentMlp
import LeanMlir.Proofs.SgdDescentCnn
-- Robustness certificate: the Lipschitz-margin certified radius (cert ≤ TRUE ≤ PGD).
import LeanMlir.Proofs.LipschitzCert
-- The real Gaussian probit: Φ/Φ⁻¹ facts + the smoothing radius at the true quantile.
import LeanMlir.Proofs.SmoothingGaussian
-- Verified-codegen bridges (denoted IR + per-op bridge theorems) so doc-gen4
-- documents them. IRPrint.lean is deliberately left out: its file-writing
-- #evals run at elaboration time (use `lake env lean …/IRPrint.lean`).
import LeanMlir.Proofs.Foundation.IR
-- Spec→math ties (rungs B/C/E). Also a Certs root + audited in
-- tests/AuditAxioms.lean since 2026-07-07: it rotted while orphaned
-- from every target (the mnv2 6→17-block spec promotion broke its rfl tie).
import LeanMlir.Proofs.Foundation.SpecVJP
