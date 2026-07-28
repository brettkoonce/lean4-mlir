import LeanMlir.Proofs.Codegen.StableHLO

/-! # The batched pointwise/row forms emit exactly what their per-example peers emit

`planning/xla_pjrt_handoff.md` §2b moved the batched renderers off the `N := 1` batch-unit
convention onto the honest batched index `N := B`. That required batched peers for every op whose
emitter reads its width off the SHlo index — the pointwise ops and the two row ops — because at
index `N·n` those emitted `tensor<B×(N·n)>`, a type that does not match their own operand.

Each batched form exists ONLY to move the batch out of the emit width. So the whole design claim is:
**the batched form renders byte-for-byte what the per-example form renders.** For EfficientNet that
was witnessed by `verified_mlir/efficientnet_train_step.mlir` coming back byte-identical, but that
is a whole-net check on one net that happens to use eight of the nine forms. This file pins each
form individually, including `relu`/`selectPos`, which ResNet-34 needs and EfficientNet never
exercises — so the tie is nailed down BEFORE the batched R34 render depends on it.

What this does NOT check is the `den` side: the render is value-independent (`skel` erases values),
so a form with the wrong denotation emits identical bytes. That half is
`den_batchOp_swish_eq_swishF` / `den_batchOp_relu_eq_reluF` / `selectPosB_faithful` and the `rfl`
faithfulness lemmas in `StableHLO.lean`. Both halves are needed; neither implies the other.

    lake env lean tests/TestBatchedEmitTie.lean
-/

open Proofs Proofs.StableHLO

private def BS : Nat := 32
private def n : Nat := 12
private def rows : Nat := 3
private def a : Nat := 5
private def c : Nat := 7
-- pool dims (input is 2h×2w) and conv dims for the bias-grad cases
private def pc : Nat := 2
private def ph : Nat := 3
private def ic : Nat := 2
private def oc : Nat := 3
private def ch : Nat := 4
private def kk : Nat := 3
-- ViT-shaped stand-ins for the transformer gradient cases: `tk` tokens (so the token axis is
-- `tk+1` with CLS), `dm` model dim, `pp × pp` patches over a `pi × pH × pW` image. `pH/pp = 2` and
-- `pW/pp = 2` gives tk = 4 patch tokens, which is what makes the patch-embed shapes consistent.
private def tk : Nat := 4
private def dm : Nat := 6
private def pp : Nat := 2
private def pi : Nat := 3
private def pH : Nat := 4
private def pW : Nat := 4

private def render (g : StateM Nat (String × String)) : String := (g.run' 0).1

/-- Per-example peer (SHlo index `n`) vs batched form (SHlo index `BS*n`), same `pretty BS`. -/
private def cases : List (String × String × String) :=
  let zv  : Vec n := fun _ => 0
  let zvb : Vec (BS*n) := fun _ => 0
  let zr  : Vec (rows*c) := fun _ => 0
  let zrb : Vec (BS*(rows*c)) := fun _ => 0
  let zW  : Mat a c := fun _ _ => 0
  [ ("swish",
     render (pretty BS (.swishF (.operand "%x" zv))),
     render (pretty BS (.batchOp (N := BS) (.swish (n := n)) (.operand "%x" zvb))))
  , ("relu",
     render (pretty BS (.reluF (.operand "%x" zv))),
     render (pretty BS (.batchOp (N := BS) (.relu (n := n)) (.operand "%x" zvb))))
  , ("swishBack",
     render (pretty BS (.swishBack "%s" zv (.operand "%x" zv))),
     render (pretty BS (.swishBackB "%s" zvb (.operand "%x" zvb))))
  , ("sigmoidBack",
     render (pretty BS (.sigmoidBack "%s" zv (.operand "%x" zv))),
     render (pretty BS (.sigmoidBackB "%s" zvb (.operand "%x" zvb))))
  , ("selectPos",
     render (pretty BS (.selectPos "%s" zv (.operand "%x" zv))),
     render (pretty BS (.selectPosB "%s" zvb (.operand "%x" zvb))))
  , ("addV",
     render (pretty BS (.addV (.operand "%x" zv) (.operand "%y" zv))),
     render (pretty BS (.addVB (.operand "%x" zvb) (.operand "%y" zvb))))
  , ("sub",
     render (pretty BS (.sub (.operand "%x" zv) (.operand "%y" zv))),
     render (pretty BS (.subB (.operand "%x" zvb) (.operand "%y" zvb))))
  , ("softmaxRow",
     render (pretty BS (.softmaxRowF (m := rows) (n := c) (.operand "%x" zr))),
     render (pretty BS (.batchOp (N := BS) (.softmaxRow (m := rows) (n := c))
                          (.operand "%x" zrb))))
  , ("denseRowBack",
     render (pretty BS (.denseRowBack (N := rows) (a := a) (c := c) "%W" zW (.operand "%x" zr))),
     render (pretty BS (.batchOp (N := BS)
                          (.denseRowBack (rows := rows) (a := a) (c := c) "%W" zW)
                          (.operand "%x" zrb))))
  ] ++
  -- ── step 3: max-pool + the conv bias param grads ──
  (let zp   : Vec (pc*(2*ph)*(2*ph)) := fun _ => 0
   let zpb  : Vec (BS*(pc*(2*ph)*(2*ph))) := fun _ => 0
   let zq   : Vec (pc*ph*ph) := fun _ => 0
   let zqb  : Vec (BS*(pc*ph*ph)) := fun _ => 0
   let zK   : Kernel4 oc ic kk kk := fun _ _ _ _ => 0
   let zT   : Tensor3 ic ch ch := fun _ _ _ => 0
   let zXb  : Vec (BS*(ic*ch*ch)) := fun _ => 0
   let zS   : Vec (ic*(2*ch)*(2*ch)) := fun _ => 0
   let zSb  : Vec (BS*(ic*(2*ch)*(2*ch))) := fun _ => 0
   let zB   : Vec oc := fun _ => 0
   let zdy  : Vec (oc*ch*ch) := fun _ => 0
   let zdyb : Vec (BS*(oc*ch*ch)) := fun _ => 0
   [ ("maxPool",
      render (pretty BS (.maxPoolF (c := pc) (h := ph) (w := ph) (.operand "%x" zp))),
      render (pretty BS (.batchOp (N := BS) (.maxPool (c := pc) (h := ph) (w := ph))
                           (.operand "%x" zpb))))
   , ("maxPoolBack",
      render (pretty BS (.maxPoolBack (c := pc) (h := ph) (w := ph) "%s" zp (.operand "%x" zq))),
      render (pretty BS (.maxPoolBackB (c := pc) (h := ph) (w := ph) "%s" zpb
                           (.operand "%x" zqb))))
   , ("convBiasSgd",
      render (pretty BS (.convBiasSgd (h := ch) (w := ch) "%b" "0.05" zK zT zB 0
                           (.operand "%x" zdy))),
      render (pretty BS (.convBiasSgdB (h := ch) (w := ch) "%b" "0.05" zK zXb zB 0
                           (.operand "%x" zdyb))))
   , ("convStridedBiasSgd",
      render (pretty BS (.convStridedBiasSgd (h := ch) (w := ch) "%b" "0.05" zK zS zB 0
                           (.operand "%x" zdy))),
      render (pretty BS (.convStridedBiasSgdB (h := ch) (w := ch) "%b" "0.05" zK zSb zB 0
                           (.operand "%x" zdyb)))) ])

/-- **The un-fused batched gradients** (`*GradB`) are their `*SgdB` peers with the SGD tail
    (const lr / multiply / subtract) cut off, so each render must be a byte-PREFIX of the fused
    one. Prefix rather than equality is the honest statement: the shared fresh-name counter means
    the gradient's lines carry the *same* SSA names in the fused render, and the tail is pure
    suffix. Denotationally this is the `*SgdB_eq_grad` set (`den (xSgdB …) = θ − lr·den (xGradB …)`,
    all `rfl`); this is its emit-side twin. -/
private def gradPrefixCases : List (String × String × String) :=
  let zK   : Kernel4 oc ic kk kk := fun _ _ _ _ => 0
  let zXb  : Vec (BS*(ic*ch*ch)) := fun _ => 0
  let zSb  : Vec (BS*(ic*(2*ch)*(2*ch))) := fun _ => 0
  let zB   : Vec oc := fun _ => 0
  let zdyb : Vec (BS*(oc*ch*ch)) := fun _ => 0
  let zbn  : Vec (BS*(oc*(ch*ch))) := fun _ => 0
  let zG   : Vec oc := fun _ => 0
  let zda  : Vec (BS*a) := fun _ => 0
  let zdc  : Vec (BS*c) := fun _ => 0
  let zWd  : Mat a c := fun _ _ => 0
  [ ("convWeightGradB",
     render (pretty BS (.convWeightGradB "%a" zB zXb zK (.operand "%x" zdyb))),
     render (pretty BS (.convWeightSgdB "%a" "%W" "0.05" zB zXb zK 0 (.operand "%x" zdyb))))
  , ("convStridedWeightGradB",
     render (pretty BS (.convStridedWeightGradB "%a" zB zSb zK (.operand "%x" zdyb))),
     render (pretty BS (.convStridedWeightSgdB "%a" "%W" "0.05" zB zSb zK 0 (.operand "%x" zdyb))))
  , ("convBiasGradB",
     render (pretty BS (.convBiasGradB (h := ch) (w := ch) zK zXb zB (.operand "%x" zdyb))),
     render (pretty BS (.convBiasSgdB (h := ch) (w := ch) "%b" "0.05" zK zXb zB 0
                          (.operand "%x" zdyb))))
  , ("convStridedBiasGradB",
     render (pretty BS (.convStridedBiasGradB (h := ch) (w := ch) zK zSb zB (.operand "%x" zdyb))),
     render (pretty BS (.convStridedBiasSgdB (h := ch) (w := ch) "%b" "0.05" zK zSb zB 0
                          (.operand "%x" zdyb))))
  , ("bnGammaGradB",
     render (pretty BS (.bnGammaGradB (oc := oc) (h := ch) (w := ch) "%v" "1.0e-5" 0 zbn
                          (.operand "%x" zbn))),
     render (pretty BS (.bnGammaSgdB (oc := oc) (h := ch) (w := ch) "%g" "%v" "1.0e-5" "0.05" 0 zG
                          zbn 0 (.operand "%x" zbn))))
  , ("bnBetaGradB",
     render (pretty BS (.bnBetaGradB (N := BS) (oc := oc) (h := ch) (w := ch) (.operand "%x" zbn))),
     render (pretty BS (.bnBetaSgdB (oc := oc) (h := ch) (w := ch) "%b" "0.05" zG 0
                          (.operand "%x" zbn))))
  , ("denseWeightGradB",
     render (pretty BS (.denseWeightGradB (c := c) "%a" zda (.operand "%x" zdc))),
     render (pretty BS (.denseWeightSgdB "%a" "%W" "0.05" zda zWd 0 (.operand "%x" zdc))))
  , ("denseBiasGradB",
     render (pretty BS (.denseBiasGradB (N := BS) (.operand "%x" zdc))),
     render (pretty BS (.denseBiasSgdB "%b" "0.05" (fun _ => 0 : Vec c) 0 (.operand "%x" zdc))))
  -- ── the TRANSFORMER family (§2a-quinquies follow-on: the ViT AdamW render needs these) ──
  -- Small stand-in shapes: `tk` tokens, `dm` model dim, `pp` patch, on a `pi × pH × pW` image.
  -- The property under test is textual, so the numbers only have to be consistent.
  , ("rowDenseWeightGrad",
     render (pretty BS (.rowDenseWeightGrad (N := tk) (a := a) (c := c) "%h"
                          (fun _ => 0 : Vec (tk*a)) (.operand "%x" (fun _ => 0 : Vec (tk*c))))),
     render (pretty BS (.rowDenseWeightSgd (N := tk) (a := a) (c := c) "%h" "%W" "0.05"
                          (fun _ => 0 : Vec (tk*a)) (fun _ _ => 0 : Mat a c) 0
                          (.operand "%x" (fun _ => 0 : Vec (tk*c))))))
  , ("rowDenseBiasGrad",
     render (pretty BS (.rowDenseBiasGrad (N := tk) (c := c) (.operand "%x" (fun _ => 0 : Vec (tk*c))))),
     render (pretty BS (.rowDenseBiasSgd (N := tk) (c := c) "%b" "0.05" (fun _ => 0 : Vec c) 0
                          (.operand "%x" (fun _ => 0 : Vec (tk*c))))))
  , ("veclnGammaGrad",
     render (pretty BS (.veclnGammaGrad (N := tk) (D := dm) "%h" "1.0e-5" 0
                          (fun _ => 0 : Vec (tk*dm)) (.operand "%x" (fun _ => 0 : Vec (tk*dm))))),
     render (pretty BS (.veclnGammaSgd (N := tk) (D := dm) "%g" "%h" "1.0e-5" "0.05" 0
                          (fun _ => 0 : Vec (tk*dm)) (fun _ => 0 : Vec dm) 0
                          (.operand "%x" (fun _ => 0 : Vec (tk*dm))))))
  , ("patchEmbedBiasGrad",
     render (pretty BS (.patchEmbedBiasGrad (N := tk) (c := c)
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*c))))),
     render (pretty BS (.patchEmbedBiasSgd (N := tk) (c := c) "%b" "0.05" (fun _ => 0 : Vec c) 0
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*c))))))
  , ("posEmbedGrad",
     render (pretty BS (.posEmbedGrad (N := tk) (D := dm)
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))),
     render (pretty BS (.posEmbedSgd (N := tk) (D := dm) "%p" "0.05"
                          (fun _ _ => 0 : Mat (tk+1) dm) 0
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))))
  -- the depthwise weight grads (EfficientNet's last blockers; mnv2/convnext reuse the shape)
  , ("depthwiseWeightGradB",
     render (pretty BS (.depthwiseWeightGradB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk) "%a" zB
                          (fun _ => 0 : Vec (BS*(oc*ch*ch)))
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))),
     render (pretty BS (.depthwiseWeightSgdB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk) "%a" "%W" "0.05" zB
                          (fun _ => 0 : Vec (BS*(oc*ch*ch)))
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk) 0
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))))
  , ("depthwiseStridedWeightGradB",
     render (pretty BS (.depthwiseStridedWeightGradB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk) "%a" zB
                          (fun _ => 0 : Vec (BS*(oc*(2*ch)*(2*ch))))
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))),
     render (pretty BS (.depthwiseStridedWeightSgdB (N := BS) (c := oc) (h := ch) (w := ch)
                          (kH := kk) (kW := kk) "%a" "%W" "0.05" zB
                          (fun _ => 0 : Vec (BS*(oc*(2*ch)*(2*ch))))
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk) 0
                          (.operand "%x" (fun _ => 0 : Vec (BS*(oc*ch*ch)))))))
  , ("patchEmbedWeightGrad",
     render (pretty BS (.patchEmbedWeightGrad (ic := pi) (H := pH) (W := pW) (P := pp)
                          (N := tk) (D := dm) "%img" (fun _ => 0 : Vec (pi*pH*pW))
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))),
     render (pretty BS (.patchEmbedWeightSgd (ic := pi) (H := pH) (W := pW) (P := pp)
                          (N := tk) (D := dm) "%W" "%img" "0.05" (fun _ => 0 : Vec (pi*pH*pW))
                          (fun _ _ _ _ => 0 : Kernel4 dm pi pp pp) 0
                          (.operand "%x" (fun _ => 0 : Vec ((tk+1)*dm))))))
  -- ── the ConvNeXt five (§2f). These ride the generic `.batched` tag, where an unmatched name
  --    falls through to `// MALFORMED` SILENTLY — these five cases are what catches that. ──
  , ("depthwiseWeightGrad (per-example)",
     render (pretty BS (.depthwiseWeightGrad (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          "%a" zB (fun _ _ _ => 0 : Tensor3 oc ch ch)
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.depthwiseWeightSgd (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          "%a" "%W" "0.05" zB (fun _ _ _ => 0 : Tensor3 oc ch ch)
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk) 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  , ("depthwiseBiasGrad",
     render (pretty BS (.depthwiseBiasGrad (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (fun _ _ _ => 0 : Tensor3 oc ch ch) zB
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.depthwiseBiasSgd (c := oc) (h := ch) (w := ch) (kH := kk) (kW := kk)
                          "%b" "0.05" (fun _ _ _ => 0 : DepthwiseKernel oc kk kk)
                          (fun _ _ _ => 0 : Tensor3 oc ch ch) zB 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  , ("lnGammaGrad",
     render (pretty BS (.lnGammaGrad (n := oc*ch*ch) "%a" "1.0e-6" 0
                          (fun _ => 0 : Vec (oc*ch*ch))
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.lnGammaSgd (n := oc*ch*ch) "%g" "%a" "1.0e-6" "0.05" 0
                          (fun _ => 0 : Vec (oc*ch*ch)) (fun _ => 0 : Vec 1) 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  , ("lnBetaGrad",
     render (pretty BS (.lnBetaGrad (n := oc*ch*ch)
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.lnBetaSgd (n := oc*ch*ch) "%bt" "0.05" (fun _ => 0 : Vec 1) 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))))
  , ("layerScaleChGammaGrad",
     render (pretty BS (.layerScaleChGammaGrad (c := oc) (h := ch) (w := ch) "%a"
                          (fun _ => 0 : Vec (oc*ch*ch))
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch))))),
     render (pretty BS (.layerScaleChGammaSgd (c := oc) (h := ch) (w := ch) "%g" "%a" "0.05"
                          (fun _ => 0 : Vec (oc*ch*ch)) (fun _ => 0 : Vec oc) 0
                          (.operand "%x" (fun _ => 0 : Vec (oc*ch*ch)))))) ]

/-- Fail loudly. NOT `IO.Process.exit 1`: under `#eval` the elaborator buffers the eval's output
    and prints it only after the eval returns, so `exit` kills the process with **every diagnostic
    discarded** — you get a bare non-zero status and no idea which form broke. (Verified against a
    deliberately broken `relu` emit case.) `throw` surfaces the message as an elaboration error and
    still makes `lake env lean` exit non-zero. Several older `tests/*.lean` use the `exit` form and
    have the same blind-failure problem. -/
private def die (msg : String) : IO α := throw (IO.userError msg)

def main : IO Unit := do
  let mut bad := 0
  for (name, perExample, batched) in cases do
    -- A form whose emit case is missing falls through to a COMMENT, not an error, so an empty
    -- or marker-bearing render must fail here rather than tie trivially against another one.
    if perExample.isEmpty || (perExample.splitOn "MALFORMED").length != 1 then
      die s!"DEGENERATE: {name} per-example render is empty or fell through"
    if perExample == batched then
      IO.println s!"  ✓ {name}"
    else
      bad := bad + 1
      IO.eprintln s!"  ✖ {name}\n    per-example:\n{perExample}    batched:\n{batched}"
  if bad != 0 then
    die s!"MISMATCH: {bad} batched form(s) do not emit their per-example peer's text"
  IO.println s!"✓ all {cases.length} batched forms emit their per-example peer's text byte-for-byte"
  IO.println "── un-fused gradients ⊂ their fused *SgdB peers ──"
  for (name, grad, sgd) in gradPrefixCases do
    if grad.isEmpty || grad.length >= sgd.length then
      die s!"DEGENERATE: {name} gradient render is empty or not shorter than the fused op"
    if sgd.startsWith grad then
      IO.println s!"  ✓ {name}  ({grad.length} of {sgd.length} bytes, tail = the SGD update)"
    else
      bad := bad + 1
      IO.eprintln s!"  ✖ {name} is NOT a prefix of its fused peer\n    grad:\n{grad}    sgd:\n{sgd}"
  if bad != 0 then
    die s!"MISMATCH: {bad} gradient render(s) are not a prefix of their fused peer"
  IO.println s!"✓ all {gradPrefixCases.length} un-fused gradients are byte-prefixes of their *SgdB peers"

#eval main
