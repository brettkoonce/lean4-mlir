import LeanMlir.Proofs.Codegen.StableHLO
import LeanMlir.Proofs.Codegen.ResNet34Render

/-! # ResNet-34 AdamW train step rendered from the verified AST, at the BATCHED index

The §2b peer of `ResNet34Render.lean`. Same net, different two things, and both are the point:

* **BatchNorm is `bnBatchF`** — μ/var reduced over `[0,2,3]`, coupling the batch — not
  `bnPerChannelF`'s per-example `[2,3]`. That is the semantics the AdamW trainer has always run
  (`tests/TestResnet34Train.lean`), and §2b's decision was to keep it rather than move the trainer
  onto the per-example chain. `ResNet34Render.lean` renders the per-example net; this file renders
  the batch-BN one. **They are different functions** — that is exactly the divergence §2a found
  between the two `resnet34_fwd` writers, and the reason these are two files rather than a flag.
* **The whole graph sits at `N := B`**, so every batch-coupled `den` here is honest: `bnBatchF`,
  `bnBatchBack`, and the whole `*GradB` family reduce over the batch, and at `N = 1` they would
  each describe a one-example function while the emitted text reduces over all `B` (§2b).

The optimizer is the proven `adamMNextF`/`adamVNextF`/`adamWParamF` triple applied to the un-fused
`*GradB` gradients — the fusion `θ − lr·g`, not Adam, was what kept every `_adam_train_step` in
`tests/` (§2a). β₁/β₂/ε/wd are baked; `%lr`/`%bc1`/`%bc2` arrive as runtime `tensor<f32>` args.

**The cotangent is composed from kit ops, not fused.** The hand-written render inlines label
smoothing (α = 0.1, K = 10) into one `[B,10]` block; here it is
`softmaxRow → subB → scaleB → addVB → shiftB → divConstB`, every line `pretty` of a verified node.
Same function, different graph — so the render does NOT match the hand-written artifact op-for-op,
and the tie against it has to be numeric. `%loss` is report-only and stays outside the AST, exactly
as `cifar8_adam_train_step`'s does.

Render is value-independent (`skel` erases values), so placeholder zeros and `lr := 0`/`ε := 0` are
passed; the emitted literals carry the real values.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-- A trainable parameter: emitted name (no `%`), gradient SSA name, and shape. The AdamW tail is
    a fold over this list, so the θ/m/v output order cannot drift from the signature order. -/
structure PGrad where
  nm   : String          -- parameter name without `%` (`%{nm}`, `%{nm}m`, `%{nm}v` are the args)
  grad : String          -- SSA name of its un-fused gradient
  ds   : List Nat        -- parameter shape, for the emitted Adam ops
deriving Inhabited

/-- Saved forward SSA names a block's backward + gradient passes reference. -/
structure BFwdB where
  code : String
  xin : String       -- block input (the merged dx flows back to this)
  o  : String        -- block output (post-relu)
  a  : String        -- pre-output-relu sum
  c1 : String        -- conv1 output (= BN1 input)
  n1 : String        -- BN1 output (= relu1 pre-activation)
  r1 : String        -- relu1 output (= conv2 input)
  c2 : String        -- conv2 output (= BN2 input)
  cp : String        -- projection conv output (downsample only; "" for identity)
deriving Inhabited

/-- Backward result: code, the dx cotangent to the previous block, and the block's parameter
    gradients in func-arg order. -/
structure BBackB where
  code : String
  dx : String
  ps : List PGrad

-- ════════════════════════════════════════════════════════════════
-- § Block forward (batch BN)
-- ════════════════════════════════════════════════════════════════

/-- Identity block forward: `conv1→BN1→relu1→conv2→BN2→(+x)→relu`, all at `N := B`. -/
private def idFwdB (B c hh : Nat) (epsStr p xName : String) : StateM Nat BFwdB := do
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn : Vec (B*(c*(hh*ww))) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W1" s!"%{p}b1" zk zc) (.operand xName zin))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zc zc (.operand nC1 zin))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nN1 zin))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W2" s!"%{p}b2" zk zc) (.operand nR1 zin))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zc zc (.operand nC2 zin))
  let (cA,  nA)  ← pretty B (.addVB (.operand nN2 zin) (.operand xName zin))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nA zin))
  let _ := zbn
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cA ++ cO, xin := xName,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := "" }

/-- Downsample block forward: strided body + strided projection skip. `cin→c`, `2hh→hh`. -/
private def downFwdB (B cin c hh : Nat) (epsStr p xName : String) : StateM Nat BFwdB := do
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zinS : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zout : Vec (B*(c*hh*ww)) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (.convStrided (h := hh) (w := ww) s!"%{p}W1" s!"%{p}b1" zk1 zc) (.operand xName zinS))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zc zc (.operand nC1 zout))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nN1 zout))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W2" s!"%{p}b2" zk2 zc) (.operand nR1 zout))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zc zc (.operand nC2 zout))
  let (cCp, nCp) ← pretty B (.batchOp (N := B) (.convStrided (h := hh) (w := ww) s!"%{p}Wp" s!"%{p}bp" zk1 zc) (.operand xName zinS))
  let (cNp, nNp) ← pretty B (.bnBatchF (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}gp" s!"%{p}btp" epsStr 0 zc zc (.operand nCp zout))
  let (cA,  nA)  ← pretty B (.addVB (.operand nN2 zout) (.operand nNp zout))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nA zout))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cCp ++ cNp ++ cA ++ cO, xin := xName,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := nCp }

-- ════════════════════════════════════════════════════════════════
-- § Block backward + UN-FUSED parameter gradients
-- ════════════════════════════════════════════════════════════════

/-- Identity block backward + its 8 parameter gradients. -/
private def idBackGradB (B c hh : Nat) (epsStr p : String) (f : BFwdB) (dyName : String) :
    StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn : Vec (B*(c*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zin (.operand dyName zin))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zc zbn (.operand nDa zbn))
  let (cDc2, nDc2) ← pretty B (.convBackBatched (N := B) (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" zk zc (.operand nDn2 zin))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zin (.operand nDc2 zin))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zc zbn (.operand nDr1 zbn))
  let (cDc1, nDc1) ← pretty B (.convBackBatched (N := B) (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W1" zk zc (.operand nDn1 zin))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zin) (.operand nDa zin))
  -- parameter gradients, func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2
  let (cW1, nW1) ← pretty B (.convWeightGradB xName zc zin zk (.operand nDn1 zin))
  let (cb1, nb1) ← pretty B (.convBiasGradB (h := hh) (w := ww) zk zin zc (.operand nDn1 zin))
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbn (.operand nDr1 zbn))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDr1 zbn))
  let (cW2, nW2) ← pretty B (.convWeightGradB f.r1 zc zin zk (.operand nDn2 zin))
  let (cb2, nb2) ← pretty B (.convBiasGradB (h := hh) (w := ww) zk zin zc (.operand nDn2 zin))
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbn (.operand nDa zbn))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  pure { code := cDa ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDx ++
                 cW1 ++ cb1 ++ cg1 ++ ct1 ++ cW2 ++ cb2 ++ cg2 ++ ct2,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [c,c,3,3]⟩, ⟨s!"{p}b1", nb1, [c]⟩,
                ⟨s!"{p}g1", ng1, [c]⟩, ⟨s!"{p}bt1", nt1, [c]⟩,
                ⟨s!"{p}W2", nW2, [c,c,3,3]⟩, ⟨s!"{p}b2", nb2, [c]⟩,
                ⟨s!"{p}g2", ng2, [c]⟩, ⟨s!"{p}bt2", nt2, [c]⟩] }

/-- Downsample block backward + its 12 parameter gradients. -/
private def downBackGradB (B cin c hh : Nat) (epsStr p : String) (f : BFwdB) (dyName : String) :
    StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zinS : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zout : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn  : Vec (B*(c*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zout (.operand dyName zout))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zc zbn (.operand nDa zbn))
  let (cDc2, nDc2) ← pretty B (.convBackBatched (N := B) (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" zk2 zc (.operand nDn2 zout))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zout (.operand nDc2 zout))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zc zbn (.operand nDr1 zbn))
  let (cDc1, nDc1) ← pretty B (.convStridedBackBatched (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}W1" zk1 zc (.operand nDn1 zout))
  let (cDnp, nDnp) ← pretty B (.bnBatchBack (N := B) (oc := c) (h := hh) (w := ww) s!"%{p}gp" f.cp epsStr 0 zc zbn (.operand nDa zbn))
  let (cDcp, nDcp) ← pretty B (.convStridedBackBatched (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}Wp" zk1 zc (.operand nDnp zout))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zinS) (.operand nDcp zinS))
  -- parameter gradients, func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2 Wp bp gp btp
  let (cW1, nW1) ← pretty B (.convStridedWeightGradB xName zc zinS zk1 (.operand nDn1 zout))
  let (cb1, nb1) ← pretty B (.convStridedBiasGradB (h := hh) (w := ww) zk1 zinS zc (.operand nDn1 zout))
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbn (.operand nDr1 zbn))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDr1 zbn))
  let (cW2, nW2) ← pretty B (.convWeightGradB f.r1 zc zout zk2 (.operand nDn2 zout))
  let (cb2, nb2) ← pretty B (.convBiasGradB (h := hh) (w := ww) zk2 zout zc (.operand nDn2 zout))
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbn (.operand nDa zbn))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  let (cWp, nWp) ← pretty B (.convStridedWeightGradB xName zc zinS zk1 (.operand nDnp zout))
  let (cbp, nbp) ← pretty B (.convStridedBiasGradB (h := hh) (w := ww) zk1 zinS zc (.operand nDnp zout))
  let (cgp, ngp) ← pretty B (.bnGammaGradB f.cp epsStr 0 zbn (.operand nDa zbn))
  let (ctp, ntp) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  pure { code := cDa ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDnp ++ cDcp ++ cDx ++
                 cW1 ++ cb1 ++ cg1 ++ ct1 ++ cW2 ++ cb2 ++ cg2 ++ ct2 ++ cWp ++ cbp ++ cgp ++ ctp,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [c,cin,3,3]⟩, ⟨s!"{p}b1", nb1, [c]⟩,
                ⟨s!"{p}g1", ng1, [c]⟩, ⟨s!"{p}bt1", nt1, [c]⟩,
                ⟨s!"{p}W2", nW2, [c,c,3,3]⟩, ⟨s!"{p}b2", nb2, [c]⟩,
                ⟨s!"{p}g2", ng2, [c]⟩, ⟨s!"{p}bt2", nt2, [c]⟩,
                ⟨s!"{p}Wp", nWp, [c,cin,3,3]⟩, ⟨s!"{p}bp", nbp, [c]⟩,
                ⟨s!"{p}gp", ngp, [c]⟩, ⟨s!"{p}btp", ntp, [c]⟩] }

end Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The AdamW tail — one proven triple per parameter, folded in signature order
-- ════════════════════════════════════════════════════════════════

/-- `(θ', m', v')` for one parameter, from its un-fused gradient. The three ops are the proven
    `adamMNextF`/`adamVNextF`/`adamWParamF` (`adamW_triple_faithful` bundles their `den`s into
    `Proofs.adamWStep` by `rfl`). β₁/β₂/ε/wd are baked literals; `%lr`/`%bc1`/`%bc2` are runtime
    `tensor<f32>` args, so one render serves a whole LR schedule. -/
private def adamOne (B : Nat) (g : PGrad) : StateM Nat (String × String × String × String) := do
  let n := g.ds.foldl (· * ·) 1
  let z : Vec n := fun _ => 0
  let gr : SHlo n := .operand g.grad z
  let (cM, nM) ← pretty B (.adamMNextF s!"%{g.nm}m" "%b1" "%ob1" g.ds 0 z gr)
  let (cV, nV) ← pretty B (.adamVNextF s!"%{g.nm}v" "%b2" "%ob2" g.ds 0 z gr)
  let (cT, nT) ← pretty B (.adamWParamF s!"%{g.nm}" s!"%{g.nm}m" s!"%{g.nm}v" "%b1" "%ob1"
                    "%b2" "%ob2" "%bc1" "%bc2" "%lr" "%eps" "%wd" g.ds 0 0 0 0 0 0 0 z z z gr)
  pure (cM ++ cV ++ cT, nT, nM, nV)

/-- β₁/β₂/ε/wd as graph constants — the committed ResNet-34 AdamW recipe. -/
private def adamConstsB : String :=
  "    %b1 = stablehlo.constant dense<0.9> : tensor<f32>\n" ++
  "    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>\n" ++
  "    %b2 = stablehlo.constant dense<0.999> : tensor<f32>\n" ++
  "    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>\n" ++
  "    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>\n" ++
  "    %wd = stablehlo.constant dense<0.0001> : tensor<f32>\n"

-- ════════════════════════════════════════════════════════════════
-- § The whole-net batched AdamW train step
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000000 in
/-- **ResNet-34 `[3,4,6,3]` AdamW train step, batch-BN, rendered from the verified AST at `N := B`.**
    515 inputs (`%x`, 146 θ, 146 m, 146 v, `%lr`/`%bc1`/`%bc2`, 72 running-stat slots, `%onehot`)
    and 513 outputs (146 θ', 146 m', 146 v', `%loss`/`%bc1`/`%bc2`, 72 batch stats) — the interface
    `tests/TestResnet34Train.lean`'s hand-written render already presents, so the driver is
    unchanged. Parameter ORDER comes from `r34SigList`, the same single source the per-example
    render and both forwards use, so the arity/order contract cannot drift between them. -/
def resnet34AdamTrainStepFaithfulB (B nClasses : Nat) (epsStr : String) : String :=
  let go : StateM Nat String := do
    -- ═══ stem: 7×7/s2 conv → batch BN → relu → 2×2 maxpool ═══
    let zx    : Vec (B*(3*224*224)) := fun _ => 0
    let zSk   : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
    let z64   : Vec 64 := fun _ => 0
    let z112  : Vec (B*(64*112*112)) := fun _ => 0
    let z112b : Vec (B*(64*(112*112))) := fun _ => 0
    let z56   : Vec (B*(64*56*56)) := fun _ => 0
    let (cStc, nStc) ← pretty B (.batchOp (N := B) (.convStrided (h := 112) (w := 112) "%sW" "%sbi" zSk z64) (.operand "%x" zx))
    let (cStn, nStn) ← pretty B (.bnBatchF (N := B) (oc := 64) (h := 112) (w := 112) "%sg" "%sbt" epsStr 0 z64 z64 (.operand nStc z112))
    let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu (n := 64*112*112)) (.operand nStn z112))
    let (cStp, nStp) ← pretty B (.batchOp (N := B) (.maxPool (c := 64) (h := 56) (w := 56)) (.operand nStr z112))
    -- ═══ 16 blocks ═══
    let f1  ← idFwdB   B 64 56 epsStr "s1b0" nStp
    let f2  ← idFwdB   B 64 56 epsStr "s1b1" f1.o
    let f3  ← idFwdB   B 64 56 epsStr "s1b2" f2.o
    let f4  ← downFwdB B 64 128 28 epsStr "d2" f3.o
    let f5  ← idFwdB   B 128 28 epsStr "s2b0" f4.o
    let f6  ← idFwdB   B 128 28 epsStr "s2b1" f5.o
    let f7  ← idFwdB   B 128 28 epsStr "s2b2" f6.o
    let f8  ← downFwdB B 128 256 14 epsStr "d3" f7.o
    let f9  ← idFwdB   B 256 14 epsStr "s3b0" f8.o
    let f10 ← idFwdB   B 256 14 epsStr "s3b1" f9.o
    let f11 ← idFwdB   B 256 14 epsStr "s3b2" f10.o
    let f12 ← idFwdB   B 256 14 epsStr "s3b3" f11.o
    let f13 ← idFwdB   B 256 14 epsStr "s3b4" f12.o
    let f14 ← downFwdB B 256 512 7 epsStr "d4" f13.o
    let f15 ← idFwdB   B 512 7 epsStr "s4b0" f14.o
    let f16 ← idFwdB   B 512 7 epsStr "s4b1" f15.o
    -- ═══ head: GAP(7×7) → dense(512→nClasses) ═══
    let zL    : Vec (B*(512*7*7)) := fun _ => 0
    let z512  : Vec (B*512) := fun _ => 0
    let zWd   : Mat 512 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    let zNCb  : Vec (B*(1*nClasses)) := fun _ => 0
    let zNCp  : Vec (B*nClasses) := fun _ => 0
    let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 512) (h := 7) (w := 7)) (.operand f16.o zL))
    let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC) (.operand nGap z512))
    -- ═══ label-smoothed softmax-CE cotangent, COMPOSED from kit ops (α = 0.1, K = nClasses):
    --     dy = (softmax(logits) − onehot + α·onehot − α/K) / B. Every line is a verified node;
    --     the hand-written render fuses this into one [B,K] block, so the two graphs differ. ═══
    let (cSm,  nSm)  ← pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses)) (.operand nLog zNCb))
    let (cD0,  nD0)  ← pretty B (.subB (.operand nSm zNCb) (.operand "%onehot" zNCb))
    let (cLsa, nLsa) ← pretty B (.scaleB "0.100000" 0 (.operand "%onehot" zNCb))
    let (cD1,  nD1)  ← pretty B (.addVB (.operand nD0 zNCb) (.operand nLsa zNCb))
    let (cD2,  nD2)  ← pretty B (.shiftB "-0.010000" 0 (.operand nD1 zNCb))
    let (cDy,  nDy)  ← pretty B (.divConstB s!"{B}.0" 0 (.operand nD2 zNCb))
    -- ═══ head backward + dense grads ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (N := B) (.denseRowBack (rows := 1) (a := 512) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    let (cWd,  nWd)  ← pretty B (.denseWeightGradB (c := nClasses) nGap z512 (.operand nDy zNCp))
    let (cbd,  nbd)  ← pretty B (.denseBiasGradB (N := B) (.operand nDy zNCp))
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 512) (h := 7) (w := 7) (.operand nDgi z512))
    -- ═══ 16 block backwards ═══
    let b16 ← idBackGradB   B 512 7 epsStr "s4b1" f16 nDgp
    let b15 ← idBackGradB   B 512 7 epsStr "s4b0" f15 b16.dx
    let b14 ← downBackGradB B 256 512 7 epsStr "d4" f14 b15.dx
    let b13 ← idBackGradB   B 256 14 epsStr "s3b4" f13 b14.dx
    let b12 ← idBackGradB   B 256 14 epsStr "s3b3" f12 b13.dx
    let b11 ← idBackGradB   B 256 14 epsStr "s3b2" f11 b12.dx
    let b10 ← idBackGradB   B 256 14 epsStr "s3b1" f10 b11.dx
    let b9  ← idBackGradB   B 256 14 epsStr "s3b0" f9  b10.dx
    let b8  ← downBackGradB B 128 256 14 epsStr "d3" f8 b9.dx
    let b7  ← idBackGradB   B 128 28 epsStr "s2b2" f7 b8.dx
    let b6  ← idBackGradB   B 128 28 epsStr "s2b1" f6 b7.dx
    let b5  ← idBackGradB   B 128 28 epsStr "s2b0" f5 b6.dx
    let b4  ← downBackGradB B 64 128 28 epsStr "d2" f4 b5.dx
    let b3  ← idBackGradB   B 64 56 epsStr "s1b2" f3 b4.dx
    let b2  ← idBackGradB   B 64 56 epsStr "s1b1" f2 b3.dx
    let b1  ← idBackGradB   B 64 56 epsStr "s1b0" f1 b2.dx
    -- ═══ stem backward: maxpool-back → relu mask → BN back, then the 4 stem grads ═══
    let (cDmp, nDmp) ← pretty B (.maxPoolBackB (N := B) (c := 64) (h := 56) (w := 56) nStr z112 (.operand b1.dx z56))
    let (cDsr, nDsr) ← pretty B (.selectPosB nStn z112 (.operand nDmp z112))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := B) (oc := 64) (h := 112) (w := 112) "%sg" nStc epsStr 0 z64 z112b (.operand nDsr z112b))
    let (csW, nsW) ← pretty B (.convStridedWeightGradB "%x" z64 zx zSk (.operand nDsn z112))
    let (csb, nsb) ← pretty B (.convStridedBiasGradB (h := 112) (w := 112) zSk zx z64 (.operand nDsn z112))
    let (csg, nsg) ← pretty B (.bnGammaGradB nStc epsStr 0 z112b (.operand nDsr z112b))
    let (cst, nst) ← pretty B (.bnBetaGradB (N := B) (oc := 64) (h := 112) (w := 112) (.operand nDsr z112b))
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT ═══
    let bnStat (oc hh : Nat) (xn : String) : StateM Nat (String × String × String) := do
      let zb : Vec (B*(oc*(hh*hh))) := fun _ => 0
      let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
      let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
      pure (cM ++ cV, nM, nV)
    let idStats (oc hh : Nat) (f : BFwdB) : StateM Nat (String × List String) := do
      let (c1, m1, v1) ← bnStat oc hh f.c1
      let (c2, m2, v2) ← bnStat oc hh f.c2
      pure (c1 ++ c2, [m1, v1, m2, v2])
    let downStats (oc hh : Nat) (f : BFwdB) : StateM Nat (String × List String) := do
      let (c1, m1, v1) ← bnStat oc hh f.c1
      let (c2, m2, v2) ← bnStat oc hh f.c2
      let (cp, mp, vp) ← bnStat oc hh f.cp
      pure (c1 ++ c2 ++ cp, [m1, v1, m2, v2, mp, vp])
    let (cSt0, st0) ← bnStat 64 112 nStc
    let (cSt1, st1) ← idStats 64 56 f1
    let (cSt2, st2) ← idStats 64 56 f2
    let (cSt3, st3) ← idStats 64 56 f3
    let (cSt4, st4) ← downStats 128 28 f4
    let (cSt5, st5) ← idStats 128 28 f5
    let (cSt6, st6) ← idStats 128 28 f6
    let (cSt7, st7) ← idStats 128 28 f7
    let (cSt8, st8) ← downStats 256 14 f8
    let (cSt9, st9) ← idStats 256 14 f9
    let (cSt10, st10) ← idStats 256 14 f10
    let (cSt11, st11) ← idStats 256 14 f11
    let (cSt12, st12) ← idStats 256 14 f12
    let (cSt13, st13) ← idStats 256 14 f13
    let (cSt14, st14) ← downStats 512 7 f14
    let (cSt15, st15) ← idStats 512 7 f15
    let (cSt16, st16) ← idStats 512 7 f16
    -- ═══ the 146 parameter gradients in func-arg order ═══
    let stemPs : List PGrad :=
      [⟨"sW", nsW, [64,3,7,7]⟩, ⟨"sbi", nsb, [64]⟩, ⟨"sg", nsg, [64]⟩, ⟨"sbt", nst, [64]⟩]
    let headPs : List PGrad := [⟨"Wd", nWd, [512, nClasses]⟩, ⟨"bd", nbd, [nClasses]⟩]
    let allPs : List PGrad := stemPs ++
      b1.ps ++ b2.ps ++ b3.ps ++ b4.ps ++ b5.ps ++ b6.ps ++ b7.ps ++ b8.ps ++
      b9.ps ++ b10.ps ++ b11.ps ++ b12.ps ++ b13.ps ++ b14.ps ++ b15.ps ++ b16.ps ++ headPs
    -- ═══ AdamW: one proven triple per parameter ═══
    let mut adamCode := ""
    let mut thetaN : List String := []
    let mut mNames : List String := []
    let mut vNames : List String := []
    for g in allPs do
      let (c, nT, nM, nV) ← adamOne B g
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]
      mNames := mNames ++ [nM]
      vNames := vNames ++ [nV]
    -- ═══ assemble ═══
    let statCode := cSt0 ++ cSt1 ++ cSt2 ++ cSt3 ++ cSt4 ++ cSt5 ++ cSt6 ++ cSt7 ++ cSt8 ++
      cSt9 ++ cSt10 ++ cSt11 ++ cSt12 ++ cSt13 ++ cSt14 ++ cSt15 ++ cSt16
    let statNames := st0.1 :: st0.2 :: (st1 ++ st2 ++ st3 ++ st4 ++ st5 ++ st6 ++ st7 ++ st8 ++
      st9 ++ st10 ++ st11 ++ st12 ++ st13 ++ st14 ++ st15 ++ st16)
    -- `%loss` is REPORT-ONLY: mean smoothed-CE for logging, on no gradient path. It is NOT
    -- `pretty` of an AST node and says so in the emitted text — the same carve-out
    -- `cifar8_adam_train_step`'s `%loss` takes (§5 of the handoff).
    let lossCode :=
      "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
      s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %llog = stablehlo.log {nSm} : {ty [B, nClasses]}\n" ++
      s!"    %lohll = stablehlo.multiply %onehot, %llog : {ty [B, nClasses]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [0, 1] : ({ty [B, nClasses]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbn = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
      s!"    %lmean = stablehlo.divide %lsum2, %lbn : tensor<f32>\n" ++
      s!"    %loss = stablehlo.negate %lmean : tensor<f32>\n"
    let body := cStc ++ cStn ++ cStr ++ cStp ++
      f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++ f8.code ++
      f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++ f15.code ++ f16.code ++
      cGap ++ cLog ++ cSm ++ cD0 ++ cLsa ++ cD1 ++ cD2 ++ cDy ++
      cDgi ++ cWd ++ cbd ++ cDgp ++
      b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++ b10.code ++ b9.code ++
      b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDmp ++ cDsr ++ cDsn ++ csW ++ csb ++ csg ++ cst ++ statCode
    let pTypes : List String := allPs.map (fun g => ty g.ds)
    let statTypes : List String := (r34StatSigList.map (·.2))
    let retVals := thetaN ++ mNames ++ vNames ++ ["%loss", "%bc1", "%bc2"] ++ statNames
    let retTys  := pTypes ++ pTypes ++ pTypes ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statTypes
    pure <|
      "    // ── ResNet-34 batch-BN AdamW train step: every line is pretty(verified AST node) ──\n" ++
      body ++ adamConstsB ++ adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let sigList : List (String × String) := r34SigList nClasses
  let pSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let mSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}m: {t}"))
  let vSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}v: {t}"))
  let statSig := String.intercalate ", " (r34StatSigList.map (fun (n, t) => s!"{n}i: {t}"))
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, " ++ statSig ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (·.2)
  let outSig := String.intercalate ", "
    (pTy ++ pTy ++ pTy ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ (r34StatSigList.map (·.2)))
  let inner : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @resnet34_adam_train_step({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- Regenerate `verified_mlir/resnet34_adam_train_step_b.mlir` — the BATCHED (`N := B`) AdamW train
-- step as `pretty(provenGraph)`. B=32, nClasses=10, ε=1e-5.
--
-- Deliberately a SEPARATE path from `resnet34_adam_train_step.mlir`, which the AdamW driver runs
-- today off the hand-written emitter in `tests/TestResnet34Train.lean`. Same interface (515 in /
-- 513 out, types positionally identical) but NOT the same graph — the cotangent is composed from
-- kit ops rather than fused — so the two are swapped only once the numeric tie passes. Pointing
-- both writers at one path before then is precisely the last-writer-wins race §2a found.
#eval IO.FS.writeFile "verified_mlir/resnet34_adam_train_step_b.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepFaithfulB 32 10 "1.0e-05")
