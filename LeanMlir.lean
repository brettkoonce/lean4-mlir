import LeanMlir.Types
import LeanMlir.Spec
import LeanMlir.F32Array
import LeanMlir.IreeRuntime
import LeanMlir.MlirCodegen
import LeanMlir.GradcheckHelpers
import LeanMlir.ViTRender
import LeanMlir.SpecHelpers
import LeanMlir.Train
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
-- Verified-codegen bridges (denoted IR + per-op bridge theorems) so doc-gen4
-- documents them. IRPrint.lean is deliberately left out: its file-writing
-- #evals run at elaboration time (use `lake env lean …/IRPrint.lean`).
import LeanMlir.Proofs.IR
