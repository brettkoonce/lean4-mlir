import LeanMlir.Proofs.StableHLO

/-! # ResNet-34 train step rendered ENTIRELY from the verified AST

The Chapter-5 peer of `cifar8BnTrainStepFaithfulV` (`CnnRender.lean`), scaled to the full
`[3,4,6,3]` ResNet-34 (146 params): a 7×7/s2 stem, 16 residual blocks (3 downsample, 13 identity),
global-average-pool + final dense. `MainResnet34Verified` trains on
`verified_mlir/resnet34_train_step.mlir`; this renderer emits that file as `pretty(provenGraph)` —
every line is `pretty` of a verified `SHlo` node, so the committed bytes ARE the certified render.

**Two new core ops, the rest pure reuse.** The strided convolutions (7×7 stem + 3×3 downsample
bodies/projections) use `convStridedWeightSgd`/`convStridedBiasSgd` (StableHLO.lean); everything
else reuses the existing op kit (`flatConvF`/`flatConvStridedF`/`bnPerChannelF`/`reluF`/`maxPoolF`/
`gapF`/`denseF` forward; `convBack`/`convStridedBack`/`bnPerChannelBack`/`selectPos`/`maxPoolBack`/
`gapBack`/`dotOut`/`addV` backward; `convWeightSgd`/`convBiasSgd`/`bnGammaSgd`/`bnBetaSgd`/
`weightSgd`/`biasSgd` param SGD). `ResNet34FaithfulPoC` proves each param output's `den` = certified.

**The residual wrinkle.** A skip-add sends its output cotangent to BOTH branches; where two paths
reconverge the cotangents SUM. So the backward of each block ends in an `addV` of (body-branch dx)
and (skip-branch dx) — identity skip = the masked cotangent passes through verbatim; downsample
skip = the strided projection's backward.

Render is value-independent (`skel` erases values), so the renderer passes placeholder zeros and
`lr := 0`/`ε := 0`; the emitted `lrStr`/`epsStr` literals carry the real values, and the `den`
theorems (ResNet34FaithfulPoC) use the real values.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-- Saved forward SSA names a block's backward + SGD passes reference. -/
structure BFwd where
  code : String
  o  : String        -- block output (post-relu)
  a  : String        -- pre-output-relu sum (the add result)
  c1 : String        -- conv1 output (= BN1 input)
  n1 : String        -- BN1 output (= relu1 pre-activation)
  r1 : String        -- relu1 output (= conv2 input activation)
  c2 : String        -- conv2 output (= BN2 input)
  cp : String        -- projection conv output (downsample only; "" for identity)

/-- Backward result: code, the dx cotangent to the previous block, and the block's param-update
    output SSA names in func-arg order. -/
structure BBack where
  code : String
  dx : String
  names : List String

-- ════════════════════════════════════════════════════════════════
-- § Block forward
-- ════════════════════════════════════════════════════════════════

/-- Identity block forward: `conv1→BN1→relu1→conv2→BN2→(+x)→relu`. `c` channels, `hh×ww` spatial. -/
private def idFwd (B c hh : Nat) (epsStr p xName : String) : StateM Nat BFwd := do
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (c*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvF (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W1" s!"%{p}b1" zk zc (.operand xName zin))
  let (cN1, nN1) ← pretty B (.bnPerChannelF (oc := c) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zc zc (.operand nC1 zin))
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zin))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" s!"%{p}b2" zk zc (.operand nR1 zin))
  let (cN2, nN2) ← pretty B (.bnPerChannelF (oc := c) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zc zc (.operand nC2 zin))
  let (cA,  nA)  ← pretty B (.addV (.operand nN2 zin) (.operand xName zin))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zin))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cA ++ cO,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := "" }

/-- Downsample block forward: strided `conv1→BN1→relu1→conv2→BN2` body + strided projection
    `convp→BNp` skip, `add`, `relu`. `cin→c` channels, input `2hh×2ww`, output `hh×ww`. -/
private def downFwd (B cin c hh : Nat) (epsStr p xName : String) : StateM Nat BFwd := do
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zinS : Vec (cin*(2*hh)*(2*ww)) := fun _ => 0
  let zout : Vec (c*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvStridedF (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}W1" s!"%{p}b1" zk1 zc (.operand xName zinS))
  let (cN1, nN1) ← pretty B (.bnPerChannelF (oc := c) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zc zc (.operand nC1 zout))
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zout))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" s!"%{p}b2" zk2 zc (.operand nR1 zout))
  let (cN2, nN2) ← pretty B (.bnPerChannelF (oc := c) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zc zc (.operand nC2 zout))
  let (cCp, nCp) ← pretty B (.flatConvStridedF (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}Wp" s!"%{p}bp" zk1 zc (.operand xName zinS))
  let (cNp, nNp) ← pretty B (.bnPerChannelF (oc := c) (h := hh) (w := ww) s!"%{p}gp" s!"%{p}btp" epsStr 0 zc zc (.operand nCp zout))
  let (cA,  nA)  ← pretty B (.addV (.operand nN2 zout) (.operand nNp zout))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zout))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cCp ++ cNp ++ cA ++ cO,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := nCp }

-- ════════════════════════════════════════════════════════════════
-- § Block backward + param SGD (the cotangent fans through, then sums at the skip)
-- ════════════════════════════════════════════════════════════════

/-- Identity block backward + 8 param SGD ops. `dyName` = cotangent of the block output;
    `xName` = block input. The skip is identity, so the merged dx sums (body dx) + (masked cot). -/
private def idBackSgd (B c hh : Nat) (epsStr lrStr p xName : String) (f : BFwd) (dyName : String) :
    StateM Nat BBack := do
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zT  : Tensor3 c hh ww := fun _ _ _ => 0
  let zin : Vec (c*hh*ww) := fun _ => 0
  -- backward chain
  let (cDa,  nDa)  ← pretty B (.selectPos f.a zin (.operand dyName zin))
  let (cDn2, nDn2) ← pretty B (.bnPerChannelBack (oc := c) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zc zin (.operand nDa zin))
  let (cDc2, nDc2) ← pretty B (.convBack (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" zk zc zin (.operand nDn2 zin))
  let (cDr1, nDr1) ← pretty B (.selectPos f.n1 zin (.operand nDc2 zin))
  let (cDn1, nDn1) ← pretty B (.bnPerChannelBack (oc := c) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zc zin (.operand nDr1 zin))
  let (cDc1, nDc1) ← pretty B (.convBack (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W1" zk zc zin (.operand nDn1 zin))
  let (cDx,  nDx)  ← pretty B (.addV (.operand nDc1 zin) (.operand nDa zin))
  -- param SGD (func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2)
  let (cW1, nW1) ← pretty B (.convWeightSgd xName s!"%{p}W1" lrStr zc zT zk 0 (.operand nDn1 zin))
  let (cb1, nb1) ← pretty B (.convBiasSgd s!"%{p}b1" lrStr zk zT zc 0 (.operand nDn1 zin))
  let (cg1, ng1) ← pretty B (.bnGammaSgd s!"%{p}g1" f.c1 epsStr lrStr 0 zc zin 0 (.operand nDr1 zin))
  let (ct1, nt1) ← pretty B (.bnBetaSgd s!"%{p}bt1" lrStr zc 0 (.operand nDr1 zin))
  let (cW2, nW2) ← pretty B (.convWeightSgd f.r1 s!"%{p}W2" lrStr zc zT zk 0 (.operand nDn2 zin))
  let (cb2, nb2) ← pretty B (.convBiasSgd s!"%{p}b2" lrStr zk zT zc 0 (.operand nDn2 zin))
  let (cg2, ng2) ← pretty B (.bnGammaSgd s!"%{p}g2" f.c2 epsStr lrStr 0 zc zin 0 (.operand nDa zin))
  let (ct2, nt2) ← pretty B (.bnBetaSgd s!"%{p}bt2" lrStr zc 0 (.operand nDa zin))
  pure { code := cDa ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDx ++
                 cW1 ++ cb1 ++ cg1 ++ ct1 ++ cW2 ++ cb2 ++ cg2 ++ ct2,
         dx := nDx, names := [nW1, nb1, ng1, nt1, nW2, nb2, ng2, nt2] }

/-- Downsample block backward + 12 param SGD ops. The skip is a strided projection conv+BN, so
    the merged dx (at the `2hh×2ww` input) sums (strided body dx) + (strided projection dx). -/
private def downBackSgd (B cin c hh : Nat) (epsStr lrStr p xName : String) (f : BFwd) (dyName : String) :
    StateM Nat BBack := do
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zT   : Tensor3 c hh ww := fun _ _ _ => 0
  let zinS : Vec (cin*(2*hh)*(2*ww)) := fun _ => 0
  let zout : Vec (c*hh*ww) := fun _ => 0
  -- backward chain
  let (cDa,  nDa)  ← pretty B (.selectPos f.a zout (.operand dyName zout))
  -- main path
  let (cDn2, nDn2) ← pretty B (.bnPerChannelBack (oc := c) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zc zout (.operand nDa zout))
  let (cDc2, nDc2) ← pretty B (.convBack (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" zk2 zc zout (.operand nDn2 zout))
  let (cDr1, nDr1) ← pretty B (.selectPos f.n1 zout (.operand nDc2 zout))
  let (cDn1, nDn1) ← pretty B (.bnPerChannelBack (oc := c) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zc zout (.operand nDr1 zout))
  let (cDc1, nDc1) ← pretty B (.convStridedBack (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}W1" zk1 zc zinS (.operand nDn1 zout))
  -- projection skip
  let (cDnp, nDnp) ← pretty B (.bnPerChannelBack (oc := c) (h := hh) (w := ww) s!"%{p}gp" f.cp epsStr 0 zc zout (.operand nDa zout))
  let (cDcp, nDcp) ← pretty B (.convStridedBack (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}Wp" zk1 zc zinS (.operand nDnp zout))
  let (cDx,  nDx)  ← pretty B (.addV (.operand nDc1 zinS) (.operand nDcp zinS))
  -- param SGD (func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2 Wp bp gp btp)
  let (cW1, nW1) ← pretty B (.convStridedWeightSgd xName s!"%{p}W1" lrStr zc zinS zk1 0 (.operand nDn1 zout))
  let (cb1, nb1) ← pretty B (.convStridedBiasSgd s!"%{p}b1" lrStr zk1 zinS zc 0 (.operand nDn1 zout))
  let (cg1, ng1) ← pretty B (.bnGammaSgd s!"%{p}g1" f.c1 epsStr lrStr 0 zc zout 0 (.operand nDr1 zout))
  let (ct1, nt1) ← pretty B (.bnBetaSgd s!"%{p}bt1" lrStr zc 0 (.operand nDr1 zout))
  let (cW2, nW2) ← pretty B (.convWeightSgd f.r1 s!"%{p}W2" lrStr zc zT zk2 0 (.operand nDn2 zout))
  let (cb2, nb2) ← pretty B (.convBiasSgd s!"%{p}b2" lrStr zk2 zT zc 0 (.operand nDn2 zout))
  let (cg2, ng2) ← pretty B (.bnGammaSgd s!"%{p}g2" f.c2 epsStr lrStr 0 zc zout 0 (.operand nDa zout))
  let (ct2, nt2) ← pretty B (.bnBetaSgd s!"%{p}bt2" lrStr zc 0 (.operand nDa zout))
  let (cWp, nWp) ← pretty B (.convStridedWeightSgd xName s!"%{p}Wp" lrStr zc zinS zk1 0 (.operand nDnp zout))
  let (cbp, nbp) ← pretty B (.convStridedBiasSgd s!"%{p}bp" lrStr zk1 zinS zc 0 (.operand nDnp zout))
  let (cgp, ngp) ← pretty B (.bnGammaSgd s!"%{p}gp" f.cp epsStr lrStr 0 zc zout 0 (.operand nDa zout))
  let (ctp, ntp) ← pretty B (.bnBetaSgd s!"%{p}btp" lrStr zc 0 (.operand nDa zout))
  pure { code := cDa ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDnp ++ cDcp ++ cDx ++
                 cW1 ++ cb1 ++ cg1 ++ ct1 ++ cW2 ++ cb2 ++ cg2 ++ ct2 ++ cWp ++ cbp ++ cgp ++ ctp,
         dx := nDx, names := [nW1, nb1, ng1, nt1, nW2, nb2, ng2, nt2, nWp, nbp, ngp, ntp] }

-- ════════════════════════════════════════════════════════════════
-- § Param signature lists (func-arg order — names + types, shared by sig + return types)
-- ════════════════════════════════════════════════════════════════

private def idSig (p : String) (c : Nat) : List (String × String) :=
  [(s!"%{p}W1", ty [c,c,3,3]), (s!"%{p}b1", ty [c]), (s!"%{p}g1", ty [c]), (s!"%{p}bt1", ty [c]),
   (s!"%{p}W2", ty [c,c,3,3]), (s!"%{p}b2", ty [c]), (s!"%{p}g2", ty [c]), (s!"%{p}bt2", ty [c])]

private def downSig (p : String) (cin c : Nat) : List (String × String) :=
  [(s!"%{p}W1", ty [c,cin,3,3]), (s!"%{p}b1", ty [c]), (s!"%{p}g1", ty [c]), (s!"%{p}bt1", ty [c]),
   (s!"%{p}W2", ty [c,c,3,3]), (s!"%{p}b2", ty [c]), (s!"%{p}g2", ty [c]), (s!"%{p}bt2", ty [c]),
   (s!"%{p}Wp", ty [c,cin,3,3]), (s!"%{p}bp", ty [c]), (s!"%{p}gp", ty [c]), (s!"%{p}btp", ty [c])]

-- ════════════════════════════════════════════════════════════════
-- § The whole-net renderer
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 1000000 in
/-- **Full ResNet-34 `[3,4,6,3]` train step rendered ENTIRELY from the verified AST** (146 params).
    `B` batch, `nClasses` outputs (=10 for the committed Imagenette trainer). Every emitted line is
    `pretty` of a verified `SHlo` node; `ResNet34FaithfulPoC` proves each param output `den` =
    certified. Stem 7×7/s2 (3→64, 224→112), maxpool→56, stages 64/128/256/512 at 56/28/14/7. -/
def resnet34TrainStepFaithfulV (B nClasses : Nat) (epsStr lrStr : String) : String :=
  let go : StateM Nat String := do
    -- ═══ stem: 7×7/s2 conv → BN → relu → maxpool ═══
    let zx   : Vec (3*224*224) := fun _ => 0
    let zSk  : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
    let z64  : Vec 64 := fun _ => 0
    let z112 : Vec (64*112*112) := fun _ => 0
    let z56  : Vec (64*56*56) := fun _ => 0
    let (cStc, nStc) ← pretty B (.flatConvStridedF (ic := 3) (oc := 64) (h := 112) (w := 112) "%sW" "%sbi" zSk z64 (.operand "%x" zx))
    let (cStn, nStn) ← pretty B (.bnPerChannelF (oc := 64) (h := 112) (w := 112) "%sg" "%sbt" epsStr 0 z64 z64 (.operand nStc z112))
    let (cStr, nStr) ← pretty B (.reluF (.operand nStn z112))
    let (cStp, nStp) ← pretty B (.maxPoolF (c := 64) (h := 56) (w := 56) (.operand nStr z112))
    -- ═══ forward: 16 blocks ═══
    let f1  ← idFwd   B 64 56 epsStr "s1b0" nStp
    let f2  ← idFwd   B 64 56 epsStr "s1b1" f1.o
    let f3  ← idFwd   B 64 56 epsStr "s1b2" f2.o
    let f4  ← downFwd B 64 128 28 epsStr "d2" f3.o
    let f5  ← idFwd   B 128 28 epsStr "s2b0" f4.o
    let f6  ← idFwd   B 128 28 epsStr "s2b1" f5.o
    let f7  ← idFwd   B 128 28 epsStr "s2b2" f6.o
    let f8  ← downFwd B 128 256 14 epsStr "d3" f7.o
    let f9  ← idFwd   B 256 14 epsStr "s3b0" f8.o
    let f10 ← idFwd   B 256 14 epsStr "s3b1" f9.o
    let f11 ← idFwd   B 256 14 epsStr "s3b2" f10.o
    let f12 ← idFwd   B 256 14 epsStr "s3b3" f11.o
    let f13 ← idFwd   B 256 14 epsStr "s3b4" f12.o
    let f14 ← downFwd B 256 512 7 epsStr "d4" f13.o
    let f15 ← idFwd   B 512 7 epsStr "s4b0" f14.o
    let f16 ← idFwd   B 512 7 epsStr "s4b1" f15.o
    -- ═══ head: GAP(7×7) → dense(512→nClasses) → softmax-CE cotangent ═══
    let zL   : Vec (512*7*7) := fun _ => 0
    let z512 : Vec 512 := fun _ => 0
    let zWd  : Mat 512 nClasses := fun _ _ => 0
    let zNC  : Vec nClasses := fun _ => 0
    let (cGap, nGap) ← pretty B (.gapF (c := 512) (h := 7) (w := 7) (.operand f16.o zL))
    let (cLog, nLog) ← pretty B (denseF "%Wd" "%bd" zWd zNC (.operand nGap z512))
    let (cDy,  nDy)  ← pretty B (.sub (.softmaxDiv (.expe (.operand nLog zNC))) (.operand "%onehot" zNC))
    -- ═══ head backward: dense input-grad → GAP-back, dense W/b SGD ═══
    let (cDg,  nDg)  ← pretty B (.dotOut "%Wd" zWd (.operand nDy zNC))
    let (cDgi, nDgi) ← pretty B (.gapBack (c := 512) (h := 7) (w := 7) (.operand nDg z512))
    let (cWd, nWd) ← pretty B (.weightSgd nGap "%Wd" lrStr z512 zWd 0 (.operand nDy zNC))
    let (cbd, nbd) ← pretty B (.biasSgd "%bd" lrStr zNC 0 (.operand nDy zNC))
    -- ═══ backward: 16 blocks reversed (cotangent threads from nDgi) ═══
    let b16 ← idBackSgd   B 512 7 epsStr lrStr "s4b1" f15.o f16 nDgi
    let b15 ← idBackSgd   B 512 7 epsStr lrStr "s4b0" f14.o f15 b16.dx
    let b14 ← downBackSgd B 256 512 7 epsStr lrStr "d4" f13.o f14 b15.dx
    let b13 ← idBackSgd   B 256 14 epsStr lrStr "s3b4" f12.o f13 b14.dx
    let b12 ← idBackSgd   B 256 14 epsStr lrStr "s3b3" f11.o f12 b13.dx
    let b11 ← idBackSgd   B 256 14 epsStr lrStr "s3b2" f10.o f11 b12.dx
    let b10 ← idBackSgd   B 256 14 epsStr lrStr "s3b1" f9.o f10 b11.dx
    let b9  ← idBackSgd   B 256 14 epsStr lrStr "s3b0" f8.o f9 b10.dx
    let b8  ← downBackSgd B 128 256 14 epsStr lrStr "d3" f7.o f8 b9.dx
    let b7  ← idBackSgd   B 128 28 epsStr lrStr "s2b2" f6.o f7 b8.dx
    let b6  ← idBackSgd   B 128 28 epsStr lrStr "s2b1" f5.o f6 b7.dx
    let b5  ← idBackSgd   B 128 28 epsStr lrStr "s2b0" f4.o f5 b6.dx
    let b4  ← downBackSgd B 64 128 28 epsStr lrStr "d2" f3.o f4 b5.dx
    let b3  ← idBackSgd   B 64 56 epsStr lrStr "s1b2" f2.o f3 b4.dx
    let b2  ← idBackSgd   B 64 56 epsStr lrStr "s1b1" f1.o f2 b3.dx
    let b1  ← idBackSgd   B 64 56 epsStr lrStr "s1b0" nStp f1 b2.dx
    -- ═══ stem backward: maxpool-back → relu-back → BN-back, then stem param SGD ═══
    let zSt112 : Vec (64*112*112) := fun _ => 0
    let (cDmp, nDmp) ← pretty B (.maxPoolBack (c := 64) (h := 56) (w := 56) nStr zSt112 (.operand b1.dx z56))
    let (cDsr, nDsr) ← pretty B (.selectPos nStn zSt112 (.operand nDmp zSt112))
    let (cDsn, nDsn) ← pretty B (.bnPerChannelBack (oc := 64) (h := 112) (w := 112) "%sg" nStc epsStr 0 z64 zSt112 (.operand nDsr zSt112))
    let (csW, nsW) ← pretty B (.convStridedWeightSgd "%x" "%sW" lrStr z64 zx zSk 0 (.operand nDsn zSt112))
    let (csb, nsb) ← pretty B (.convStridedBiasSgd "%sbi" lrStr zSk zx z64 0 (.operand nDsn zSt112))
    let (csg, nsg) ← pretty B (.bnGammaSgd "%sg" nStc epsStr lrStr 0 z64 zSt112 0 (.operand nDsr zSt112))
    let (cst, nst) ← pretty B (.bnBetaSgd "%sbt" lrStr z64 0 (.operand nDsr zSt112))
    -- ═══ assemble body + return (146 outputs in func-arg order: stem, blocks fwd-order, dense) ═══
    let fwdCode := cStc ++ cStn ++ cStr ++ cStp ++
      f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++ f8.code ++
      f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++ f15.code ++ f16.code ++
      cGap ++ cLog ++ cDy
    let bwdCode := cDg ++ cDgi ++ cWd ++ cbd ++
      b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++ b10.code ++ b9.code ++
      b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDmp ++ cDsr ++ cDsn ++ csW ++ csb ++ csg ++ cst
    let outNames : List String :=
      [nsW, nsb, nsg, nst] ++
      b1.names ++ b2.names ++ b3.names ++ b4.names ++ b5.names ++ b6.names ++ b7.names ++ b8.names ++
      b9.names ++ b10.names ++ b11.names ++ b12.names ++ b13.names ++ b14.names ++ b15.names ++ b16.names ++
      [nWd, nbd]
    let outTypes : List String :=
      (([("%sW", ty [64,3,7,7]), ("%sbi", ty [64]), ("%sg", ty [64]), ("%sbt", ty [64])] :
          List (String × String)) ++
        idSig "s1b0" 64 ++ idSig "s1b1" 64 ++ idSig "s1b2" 64 ++
        downSig "d2" 64 128 ++ idSig "s2b0" 128 ++ idSig "s2b1" 128 ++ idSig "s2b2" 128 ++
        downSig "d3" 128 256 ++ idSig "s3b0" 256 ++ idSig "s3b1" 256 ++ idSig "s3b2" 256 ++
          idSig "s3b3" 256 ++ idSig "s3b4" 256 ++
        downSig "d4" 256 512 ++ idSig "s4b0" 512 ++ idSig "s4b1" 512 ++
        [("%Wd", ty [512, nClasses]), ("%bd", ty [nClasses])]).map (·.2)
    pure <|
      "    // ── ResNet-34 train step: every line is pretty(verified AST node) ──\n" ++
      fwdCode ++ bwdCode ++
      s!"    return {String.intercalate ", " outNames} : {String.intercalate ", " outTypes}\n"
  -- func signature: %x, all 146 params, %onehot
  let sigList : List (String × String) :=
    [("%sW", ty [64,3,7,7]), ("%sbi", ty [64]), ("%sg", ty [64]), ("%sbt", ty [64])] ++
    idSig "s1b0" 64 ++ idSig "s1b1" 64 ++ idSig "s1b2" 64 ++
    downSig "d2" 64 128 ++ idSig "s2b0" 128 ++ idSig "s2b1" 128 ++ idSig "s2b2" 128 ++
    downSig "d3" 128 256 ++ idSig "s3b0" 256 ++ idSig "s3b1" 256 ++ idSig "s3b2" 256 ++
      idSig "s3b3" 256 ++ idSig "s3b4" 256 ++
    downSig "d4" 256 512 ++ idSig "s4b0" 512 ++ idSig "s4b1" 512 ++
    [("%Wd", ty [512, nClasses]), ("%bd", ty [nClasses])]
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}")) ++
    s!", %onehot: {ty [B, nClasses]}"
  let outSig := String.intercalate ", " (sigList.map (·.2))
  let inner : String := go.run' 0
  "module @m {\n" ++
  s!"  func.func @resnet34_train_step({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- Regenerate `verified_mlir/resnet34_train_step.mlir` (what MainResnet34Verified trains on) from
-- the faithful renderer; the den-certified proofs live in ResNet34FaithfulPoC.lean. B=32 (the
-- committed Imagenette batch), nClasses=10, ε=1e-5, lr = 0.1/32 = 0.003125 (mean-loss equiv).
#eval IO.FS.writeFile "verified_mlir/resnet34_train_step.mlir"
  (Proofs.StableHLO.resnet34TrainStepFaithfulV 32 10 "1.0e-05" "0.003125")
