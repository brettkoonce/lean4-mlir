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
import LeanMlir.Proofs.Attention
import LeanMlir.Proofs.CNN
import LeanMlir.Proofs.Depthwise
-- End-to-end whole-network VJP compositions (each builds on the CNN/Depthwise
-- machinery; their own file imports pull in everything transitively).
import LeanMlir.Proofs.MobileNetV2
import LeanMlir.Proofs.ConvNeXt
import LeanMlir.Proofs.EfficientNet
import LeanMlir.Proofs.MobileNetV2Close
import LeanMlir.Proofs.MobileNetV2RenderPC
import LeanMlir.Proofs.MobileNetV2ChainClose
import LeanMlir.Proofs.ConvLossFold
import LeanMlir.Proofs.ResNet34Close
import LeanMlir.Proofs.ResNet34RenderPC
import LeanMlir.Proofs.ResNet34ChainClose
-- Verified-codegen bridges (denoted IR + per-op bridge theorems) so doc-gen4
-- documents them. IRPrint.lean is deliberately left out: its file-writing
-- #evals run at elaboration time (use `lake env lean …/IRPrint.lean`).
import LeanMlir.Proofs.IR
