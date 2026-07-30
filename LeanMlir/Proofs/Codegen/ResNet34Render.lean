import LeanMlir.Proofs.Codegen.StableHLO

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

/-- Saved forward SSA names a block's backward + SGD passes reference. `xin` is carried by the
    forward itself so the backward never has to re-derive which block fed which — the wiring the
    train step reads back is the wiring the forward emitted. -/
structure BFwd where
  code : String
  xin : String       -- block input (the merged dx flows back to this)
  o  : String        -- block output (post-relu)
  a  : String        -- pre-output-relu sum (the add result)
  c1 : String        -- conv1 output (= BN1 input)
  n1 : String        -- BN1 output (= relu1 pre-activation)
  r1 : String        -- relu1 output (= conv2 input activation)
  c2 : String        -- conv2 output (= BN2 input)
  cp : String        -- projection conv output (downsample only; "" for identity)
deriving Inhabited

/-- Backward result: code, the dx cotangent to the previous block, and the block's param-update
    output SSA names in func-arg order. -/
structure BBack where
  code : String
  dx : String
  names : List String

-- ════════════════════════════════════════════════════════════════
-- § Block forward
-- ════════════════════════════════════════════════════════════════

/-- Which BatchNorm a forward render emits. Everything else about the two forwards is identical,
    which is exactly why they share one chain rather than two hand-kept-in-sync copies. -/
inductive R34Bn where
  /-- **Training**: statistics reduced out of the activation (`bnPerChannelF` — per channel, per
      example, over `H·W`). What `resnet34_train_step.mlir` differentiates. -/
  | train
  /-- **Inference**: frozen per-channel running stats arriving as graph inputs `%{p}mu`/`%{p}var`
      (`bnPerChannelEvalF`). The eval partner of a *batch*-statistic train step, whose EMA'd
      batch mean/var are exactly these per-channel scalars. -/
  | eval
deriving DecidableEq, Repr

/-- One BN site. `statP` is the running-stat input prefix (`%{statP}mu` / `%{statP}var`), used
    only in `.eval` mode; in `.train` mode the stats are reduced out of `xin` and `statP` is
    ignored. Every R34 BN site is spatially square, so one `hh` suffices. -/
private def bnSite (B oc hh : Nat) (mode : R34Bn) (epsStr gName btName statP xin : String) :
    StateM Nat (String × String) := do
  let zc  : Vec oc := fun _ => 0
  let zin : Vec (oc*hh*hh) := fun _ => 0
  match mode with
  | .train => pretty B (.bnPerChannelF (oc := oc) (h := hh) (w := hh)
                          gName btName epsStr 0 zc zc (.operand xin zin))
  | .eval  => pretty B (.bnPerChannelEvalF (oc := oc) (h := hh) (w := hh)
                          gName btName s!"%{statP}mu" s!"%{statP}var" epsStr 0 zc zc zc zc
                          (.operand xin zin))

/-- Identity block forward: `conv1→BN1→relu1→conv2→BN2→(+x)→relu`. `c` channels, `hh×ww` spatial. -/
private def idFwd (B c hh : Nat) (mode : R34Bn) (epsStr p xName : String) : StateM Nat BFwd := do
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (c*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvF (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W1" s!"%{p}b1" zk zc (.operand xName zin))
  let (cN1, nN1) ← bnSite B c hh mode epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zin))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" s!"%{p}b2" zk zc (.operand nR1 zin))
  let (cN2, nN2) ← bnSite B c hh mode epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cA,  nA)  ← pretty B (.addV (.operand nN2 zin) (.operand xName zin))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zin))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cA ++ cO, xin := xName,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := "" }

/-- Downsample block forward: strided `conv1→BN1→relu1→conv2→BN2` body + strided projection
    `convp→BNp` skip, `add`, `relu`. `cin→c` channels, input `2hh×2ww`, output `hh×ww`. -/
private def downFwd (B cin c hh : Nat) (mode : R34Bn) (epsStr p xName : String) : StateM Nat BFwd := do
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zinS : Vec (cin*(2*hh)*(2*ww)) := fun _ => 0
  let zout : Vec (c*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvStridedF (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}W1" s!"%{p}b1" zk1 zc (.operand xName zinS))
  let (cN1, nN1) ← bnSite B c hh mode epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zout))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" s!"%{p}b2" zk2 zc (.operand nR1 zout))
  let (cN2, nN2) ← bnSite B c hh mode epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cCp, nCp) ← pretty B (.flatConvStridedF (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}Wp" s!"%{p}bp" zk1 zc (.operand xName zinS))
  let (cNp, nNp) ← bnSite B c hh mode epsStr s!"%{p}gp" s!"%{p}btp" s!"{p}np" nCp
  let (cA,  nA)  ← pretty B (.addV (.operand nN2 zout) (.operand nNp zout))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zout))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cCp ++ cNp ++ cA ++ cO, xin := xName,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := nCp }

-- ════════════════════════════════════════════════════════════════
-- § Block backward + param SGD (the cotangent fans through, then sums at the skip)
-- ════════════════════════════════════════════════════════════════

/-- Identity block backward + 8 param SGD ops. `dyName` = cotangent of the block output; the block
    input comes from `f.xin`. The skip is identity, so the merged dx sums (body dx) + (masked cot). -/
private def idBackSgd (B c hh : Nat) (epsStr lrStr p : String) (f : BFwd) (dyName : String) :
    StateM Nat BBack := do
  let xName := f.xin
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
private def downBackSgd (B cin c hh : Nat) (epsStr lrStr p : String) (f : BFwd) (dyName : String) :
    StateM Nat BBack := do
  let xName := f.xin
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

/-- **The 146 ResNet-34 parameters in `net.paramShapes` (= func-arg) order**, names + types.
    The forward, the eval forward and the train step all take their signature from here, so the
    arity/type/order contract the driver relies on cannot drift between renders. -/
def r34SigList (nClasses : Nat) : List (String × String) :=
  [("%sW", ty [64,3,7,7]), ("%sbi", ty [64]), ("%sg", ty [64]), ("%sbt", ty [64])] ++
  idSig "s1b0" 64 ++ idSig "s1b1" 64 ++ idSig "s1b2" 64 ++
  downSig "d2" 64 128 ++ idSig "s2b0" 128 ++ idSig "s2b1" 128 ++ idSig "s2b2" 128 ++
  downSig "d3" 128 256 ++ idSig "s3b0" 256 ++ idSig "s3b1" 256 ++ idSig "s3b2" 256 ++
    idSig "s3b3" 256 ++ idSig "s3b4" 256 ++
  downSig "d4" 256 512 ++ idSig "s4b0" 512 ++ idSig "s4b1" 512 ++
  [("%Wd", ty [512, nClasses]), ("%bd", ty [nClasses])]

/-- **The 72 running-stat inputs** — 36 BN layers × (μ, var), each `[oc]`, in BN-forward order:
    stem, then per identity block `n1 n2`, per downsample block `n1 n2 np`. This is exactly the
    order `VerifiedNet.bnChannels` is listed in, which is how the driver packs `runningBnStats`
    (`bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]])`) — μ and var interleaved per layer,
    NOT all-μ-then-all-var. Appended after the 146 params, so `@resnet34_fwd_eval` takes
    1 + 146 + 72 = 219 inputs. -/
def r34StatSigList : List (String × String) :=
  let bn (p : String) (oc : Nat) : List (String × String) :=
    [(s!"%{p}mu", ty [oc]), (s!"%{p}var", ty [oc])]
  let idB (p : String) (c : Nat) := bn s!"{p}n1" c ++ bn s!"{p}n2" c
  let downB (p : String) (c : Nat) := bn s!"{p}n1" c ++ bn s!"{p}n2" c ++ bn s!"{p}np" c
  bn "stn" 64 ++
  idB "s1b0" 64 ++ idB "s1b1" 64 ++ idB "s1b2" 64 ++
  downB "d2" 128 ++ idB "s2b0" 128 ++ idB "s2b1" 128 ++ idB "s2b2" 128 ++
  downB "d3" 256 ++ idB "s3b0" 256 ++ idB "s3b1" 256 ++ idB "s3b2" 256 ++
    idB "s3b3" 256 ++ idB "s3b4" 256 ++
  downB "d4" 512 ++ idB "s4b0" 512 ++ idB "s4b1" 512

-- 36 BN layers ⇒ 72 stat inputs, matching resnet34Verified.bnChannels.size.
#guard r34StatSigList.length == 72

-- ════════════════════════════════════════════════════════════════
-- § The shared forward chain (all three renders emit this, so they cannot disagree)
-- ════════════════════════════════════════════════════════════════

/-- Every SSA name the ResNet-34 forward produces. `resnet34FwdFaithfulV` returns just `logits`;
    the train step additionally consumes the stem and per-block names on the way back. -/
structure R34Fwd where
  code   : String        -- stem → 16 blocks → GAP → dense, in emission order
  stc    : String        -- stem conv output (= stem BN input)
  stn    : String        -- stem BN output (= stem relu pre-activation)
  str    : String        -- stem relu output (= maxpool input)
  stp    : String        -- maxpool output (= block-1 input)
  blocks : Array BFwd    -- the 16 block forwards, in forward order
  gap    : String        -- global-average-pool output
  logits : String        -- dense output

set_option maxRecDepth 1000000 in
/-- **The full ResNet-34 `[3,4,6,3]` forward as `pretty` of the verified AST.** 7×7/s2 stem
    (3→64, 224→112) → 2×2 maxpool (→56) → stages 64/128/256/512 at 56/28/14/7 (stages 2–4 open with
    a strided downsample block) → GAP(7×7) → dense(512→`nClasses`). Every emitted line is `pretty`
    of a verified `SHlo` node. BN is the **batch-statistic** `bnPerChannelF` — this is the training
    forward; the running-stats eval forward is a separate render. -/
private def r34FwdChain (B nClasses : Nat) (mode : R34Bn) (epsStr : String) : StateM Nat R34Fwd := do
  -- ═══ stem: 7×7/s2 conv → BN → relu → maxpool ═══
  let zx   : Vec (3*224*224) := fun _ => 0
  let zSk  : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
  let z64  : Vec 64 := fun _ => 0
  let z112 : Vec (64*112*112) := fun _ => 0
  let (cStc, nStc) ← pretty B (.flatConvStridedF (ic := 3) (oc := 64) (h := 112) (w := 112) "%sW" "%sbi" zSk z64 (.operand "%x" zx))
  let (cStn, nStn) ← bnSite B 64 112 mode epsStr "%sg" "%sbt" "stn" nStc
  let (cStr, nStr) ← pretty B (.reluF (.operand nStn z112))
  let (cStp, nStp) ← pretty B (.maxPoolF (c := 64) (h := 56) (w := 56) (.operand nStr z112))
  -- ═══ 16 blocks ═══
  let f1  ← idFwd   B 64 56 mode epsStr "s1b0" nStp
  let f2  ← idFwd   B 64 56 mode epsStr "s1b1" f1.o
  let f3  ← idFwd   B 64 56 mode epsStr "s1b2" f2.o
  let f4  ← downFwd B 64 128 28 mode epsStr "d2" f3.o
  let f5  ← idFwd   B 128 28 mode epsStr "s2b0" f4.o
  let f6  ← idFwd   B 128 28 mode epsStr "s2b1" f5.o
  let f7  ← idFwd   B 128 28 mode epsStr "s2b2" f6.o
  let f8  ← downFwd B 128 256 14 mode epsStr "d3" f7.o
  let f9  ← idFwd   B 256 14 mode epsStr "s3b0" f8.o
  let f10 ← idFwd   B 256 14 mode epsStr "s3b1" f9.o
  let f11 ← idFwd   B 256 14 mode epsStr "s3b2" f10.o
  let f12 ← idFwd   B 256 14 mode epsStr "s3b3" f11.o
  let f13 ← idFwd   B 256 14 mode epsStr "s3b4" f12.o
  let f14 ← downFwd B 256 512 7 mode epsStr "d4" f13.o
  let f15 ← idFwd   B 512 7 mode epsStr "s4b0" f14.o
  let f16 ← idFwd   B 512 7 mode epsStr "s4b1" f15.o
  -- ═══ head: GAP(7×7) → dense(512→nClasses) ═══
  let zL   : Vec (512*7*7) := fun _ => 0
  let z512 : Vec 512 := fun _ => 0
  let zWd  : Mat 512 nClasses := fun _ _ => 0
  let zNC  : Vec nClasses := fun _ => 0
  let (cGap, nGap) ← pretty B (.gapF (c := 512) (h := 7) (w := 7) (.operand f16.o zL))
  let (cLog, nLog) ← pretty B (denseF "%Wd" "%bd" zWd zNC (.operand nGap z512))
  pure { code := cStc ++ cStn ++ cStr ++ cStp ++
           f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++ f8.code ++
           f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++ f15.code ++
           f16.code ++ cGap ++ cLog,
         stc := nStc, stn := nStn, str := nStr, stp := nStp,
         blocks := #[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16],
         gap := nGap, logits := nLog }

set_option maxRecDepth 1000000 in
/-- **`@resnet34_fwd` rendered ENTIRELY from the verified AST** — the Chapter-5 peer of the
    train-step render, sharing its forward chain and its 146-parameter signature. Takes `%x` plus
    the 146 params in `net.paramShapes` order (147 inputs) and returns logits `[B, nClasses]`.

    This replaces the independent hand-written string emitter in `tests/TestResnet34Fwd.lean`:
    the forward the driver evals is now the same graph the train step differentiates, by
    construction rather than by inspection. -/
def resnet34FwdFaithfulV (B nClasses : Nat) (epsStr : String)
    (slug : String := "resnet34") : String :=
  let sigList := r34SigList nClasses
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let F : R34Fwd := (r34FwdChain B nClasses .train epsStr).run' 0
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── ResNet-34 forward: every line is pretty(verified AST node) ──\n" ++
  F.code ++
  s!"    return {F.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

set_option maxRecDepth 1000000 in
/-- **`@resnet34_fwd_eval` rendered ENTIRELY from the verified AST** — the inference forward, with
    every BN site consuming frozen per-channel running stats (`bnPerChannelEvalF`) instead of
    reducing statistics out of its activation. Same net, same 146 params in the same order, plus
    the 72 stat inputs of `r34StatSigList`: **219 inputs**, returning logits `[B, nClasses]`.

    This is the eval partner of a **batch**-statistic train step, whose EMA'd batch mean/var are
    exactly these per-channel scalars — i.e. of `resnet34_adam_train_step.mlir`, which is still a
    hand-written render in `tests/TestResnet34Train.lean`. So the eval forward is now certified
    while the train step it partners is not; that asymmetry is the remaining §2a work, not a
    property of this render. -/
def resnet34FwdEvalFaithfulV (B nClasses : Nat) (epsStr : String)
    (slug : String := "resnet34") : String :=
  let sigList := r34SigList nClasses ++ r34StatSigList
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let F : R34Fwd := (r34FwdChain B nClasses .eval epsStr).run' 0
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd_eval({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── ResNet-34 eval forward (running-stats BN): every line is pretty(verified AST node) ──\n" ++
  F.code ++
  s!"    return {F.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

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
    -- ═══ forward: stem → 16 blocks → GAP → dense (the SAME chain `resnet34FwdFaithfulV` emits) ═══
    let F ← r34FwdChain B nClasses .train epsStr
    let blk := F.blocks
    let zx   : Vec (3*224*224) := fun _ => 0
    let zSk  : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
    let z64  : Vec 64 := fun _ => 0
    let z56  : Vec (64*56*56) := fun _ => 0
    let z512 : Vec 512 := fun _ => 0
    let zWd  : Mat 512 nClasses := fun _ _ => 0
    let zNC  : Vec nClasses := fun _ => 0
    -- ═══ softmax-CE cotangent ═══
    let (cDy,  nDy)  ← pretty B (.sub (.softmaxDiv (.expe (.operand F.logits zNC))) (.operand "%onehot" zNC))
    -- ═══ head backward: dense input-grad → GAP-back, dense W/b SGD ═══
    let (cDg,  nDg)  ← pretty B (.dotOut "%Wd" zWd (.operand nDy zNC))
    let (cDgi, nDgi) ← pretty B (.gapBack (c := 512) (h := 7) (w := 7) (.operand nDg z512))
    let (cWd, nWd) ← pretty B (.weightSgd F.gap "%Wd" lrStr z512 zWd 0 (.operand nDy zNC))
    let (cbd, nbd) ← pretty B (.biasSgd "%bd" lrStr zNC 0 (.operand nDy zNC))
    -- ═══ backward: 16 blocks reversed (cotangent threads from nDgi); each block's input comes
    --     from its own `BFwd.xin`, so forward and backward cannot be mispaired ═══
    let b16 ← idBackSgd   B 512 7 epsStr lrStr "s4b1" blk[15]! nDgi
    let b15 ← idBackSgd   B 512 7 epsStr lrStr "s4b0" blk[14]! b16.dx
    let b14 ← downBackSgd B 256 512 7 epsStr lrStr "d4" blk[13]! b15.dx
    let b13 ← idBackSgd   B 256 14 epsStr lrStr "s3b4" blk[12]! b14.dx
    let b12 ← idBackSgd   B 256 14 epsStr lrStr "s3b3" blk[11]! b13.dx
    let b11 ← idBackSgd   B 256 14 epsStr lrStr "s3b2" blk[10]! b12.dx
    let b10 ← idBackSgd   B 256 14 epsStr lrStr "s3b1" blk[9]! b11.dx
    let b9  ← idBackSgd   B 256 14 epsStr lrStr "s3b0" blk[8]! b10.dx
    let b8  ← downBackSgd B 128 256 14 epsStr lrStr "d3" blk[7]! b9.dx
    let b7  ← idBackSgd   B 128 28 epsStr lrStr "s2b2" blk[6]! b8.dx
    let b6  ← idBackSgd   B 128 28 epsStr lrStr "s2b1" blk[5]! b7.dx
    let b5  ← idBackSgd   B 128 28 epsStr lrStr "s2b0" blk[4]! b6.dx
    let b4  ← downBackSgd B 64 128 28 epsStr lrStr "d2" blk[3]! b5.dx
    let b3  ← idBackSgd   B 64 56 epsStr lrStr "s1b2" blk[2]! b4.dx
    let b2  ← idBackSgd   B 64 56 epsStr lrStr "s1b1" blk[1]! b3.dx
    let b1  ← idBackSgd   B 64 56 epsStr lrStr "s1b0" blk[0]! b2.dx
    -- ═══ stem backward: maxpool-back → relu-back → BN-back, then stem param SGD ═══
    let zSt112 : Vec (64*112*112) := fun _ => 0
    let (cDmp, nDmp) ← pretty B (.maxPoolBack (c := 64) (h := 56) (w := 56) F.str zSt112 (.operand b1.dx z56))
    let (cDsr, nDsr) ← pretty B (.selectPos F.stn zSt112 (.operand nDmp zSt112))
    let (cDsn, nDsn) ← pretty B (.bnPerChannelBack (oc := 64) (h := 112) (w := 112) "%sg" F.stc epsStr 0 z64 zSt112 (.operand nDsr zSt112))
    let (csW, nsW) ← pretty B (.convStridedWeightSgd "%x" "%sW" lrStr z64 zx zSk 0 (.operand nDsn zSt112))
    let (csb, nsb) ← pretty B (.convStridedBiasSgd "%sbi" lrStr zSk zx z64 0 (.operand nDsn zSt112))
    let (csg, nsg) ← pretty B (.bnGammaSgd "%sg" F.stc epsStr lrStr 0 z64 zSt112 0 (.operand nDsr zSt112))
    let (cst, nst) ← pretty B (.bnBetaSgd "%sbt" lrStr z64 0 (.operand nDsr zSt112))
    -- ═══ assemble body + return (146 outputs in func-arg order: stem, blocks fwd-order, dense) ═══
    let fwdCode := F.code ++ cDy
    let bwdCode := cDg ++ cDgi ++ cWd ++ cbd ++
      b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++ b10.code ++ b9.code ++
      b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDmp ++ cDsr ++ cDsn ++ csW ++ csb ++ csg ++ cst
    let outNames : List String :=
      [nsW, nsb, nsg, nst] ++
      b1.names ++ b2.names ++ b3.names ++ b4.names ++ b5.names ++ b6.names ++ b7.names ++ b8.names ++
      b9.names ++ b10.names ++ b11.names ++ b12.names ++ b13.names ++ b14.names ++ b15.names ++ b16.names ++
      [nWd, nbd]
    let outTypes : List String := (r34SigList nClasses).map (·.2)
    pure <|
      "    // ── ResNet-34 train step: every line is pretty(verified AST node) ──\n" ++
      fwdCode ++ bwdCode ++
      s!"    return {String.intercalate ", " outNames} : {String.intercalate ", " outTypes}\n"
  -- func signature: %x, all 146 params, %onehot
  let sigList : List (String × String) := r34SigList nClasses
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

-- Regenerate `verified_mlir/resnet34_fwd.mlir` (what MainResnet34Verified evals with) from the
-- SAME forward chain, at the same B/nClasses/ε. Previously rendered by an independent hand-written
-- emitter in `tests/TestResnet34Fwd.lean`; that copy is retired.
#eval IO.FS.writeFile "verified_mlir/resnet34_fwd.mlir"
  (Proofs.StableHLO.resnet34FwdFaithfulV 32 10 "1.0e-05")

-- Regenerate `verified_mlir/resnet34_fwd_eval.mlir` (what the AdamW drivers eval with once the
-- running BN stats are threaded) from the same chain in `.eval` mode. Previously rendered by the
-- hand-written emitter in `tests/TestResnet34Fwd.lean`; that copy is retired.
#eval IO.FS.writeFile "verified_mlir/resnet34_fwd_eval.mlir"
  (Proofs.StableHLO.resnet34FwdEvalFaithfulV 32 10 "1.0e-05")
