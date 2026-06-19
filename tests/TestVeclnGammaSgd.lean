import LeanMlir.Proofs.StableHLO

/-! Standalone iree-validation of the new `veclnGammaSgd` core op (ViT vector-[D] LayerNorm γ SGD).
    Renders a one-op module at the real ViT-Tiny shapes (B=32, N=197 tokens, D=192) and writes it to
    `/tmp/vecln_gamma_sgd.mlir` for an iree-compile smoke test (rocm/gfx1100). NOT a proof — just
    confirms the `emitTok` text is well-formed StableHLO at production shapes. -/

open Proofs Proofs.StableHLO

private def zV {n : Nat} : Vec n := fun _ => 0

/-- One `veclnGammaSgd` op (B=32, N=197, D=192), wrapped as a `func.func` returning the updated
    `tensor<192xf32>` γ (rank-1 — so `renderModule`'s batched `[B,retLen]` wrapper does NOT apply). -/
def veclnGammaSgdSample : String :=
  let op : SHlo 192 :=
    .veclnGammaSgd "%g" "%x" "0.00001" "0.1" (0 : ℝ)
      (zV : Vec (197*192)) (zV : Vec 192) 0 (.operand "%dy" (zV : Vec (197*192)))
  let (body, res) := (pretty 32 op).run' 0
  "module @m {\n" ++
  "  func.func @vecln_gamma_sgd(%x: tensor<32x197x192xf32>, %g: tensor<192xf32>, " ++
  "%dy: tensor<32x197x192xf32>) -> tensor<192xf32> {\n" ++
  body ++ s!"    return {res} : tensor<192xf32>\n" ++ "  }\n}\n"

#eval IO.FS.writeFile "/tmp/vecln_gamma_sgd.mlir" veclnGammaSgdSample

/-- One `patchEmbedWeightSgd` op (ViT-Tiny: ic=3, H=W=224, P=16, N=196, D=192), wrapped as a
    `func.func` returning the updated patch kernel `tensor<192x3x16x16xf32>`. Validates the strided
    patchifyWGrad emit at production shapes. -/
def patchEmbedWeightSgdSample : String :=
  let op : SHlo (192*3*16*16) :=
    .patchEmbedWeightSgd "%wConv" "%ximg" "0.1"
      (zV : Vec (3*224*224)) (fun _ _ _ _ => 0 : Kernel4 192 3 16 16) 0
      (.operand "%dy" (zV : Vec ((196+1)*192)))
  let (body, res) := (pretty 32 op).run' 0
  "module @m {\n" ++
  "  func.func @patch_w_sgd(%ximg: tensor<32x3x224x224xf32>, %wConv: tensor<192x3x16x16xf32>, " ++
  "%dy: tensor<32x37824xf32>) -> tensor<192x3x16x16xf32> {\n" ++
  body ++ s!"    return {res} : tensor<192x3x16x16xf32>\n" ++ "  }\n}\n"

#eval IO.FS.writeFile "/tmp/patch_w_sgd.mlir" patchEmbedWeightSgdSample
