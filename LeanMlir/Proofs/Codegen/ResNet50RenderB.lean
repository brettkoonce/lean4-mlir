import LeanMlir.Proofs.Codegen.ResNet34RenderB

/-! # ResNet-50 train step rendered from the verified AST, at the BATCHED index

R50 phase 2 (`planning/next_session_pipeline_then_r50.md` §3.2). The bottleneck peer of
`ResNet34RenderB.lean`, block for block: batch BN (`bnBatchF`, reduced over `[0,2,3]`), the whole
graph at `N := B`, the un-fused `*GradB` parameter gradients, and the proven AdamW /
heavy-ball tail — `optOne`/`optConstsB` are **imported, not copied**, so the optimizer has one
definition across both nets.

## The three block forms, and why the third exists

`Resnet50BlocksCertified.lean` carries the certified VJPs; this file renders them.

| renderer | block | where |
|---|---|---|
| `bnkIdFwdB` / `bnkIdBackGradB` | identity, `oc → mid → mid → oc` | 12 blocks |
| `bnkProjFwdB` / `bnkProjBackGradB` | ⭐ **stride-1** projection | **stage 1 block 0 only** |
| `bnkStridedFwdB` / `bnkStridedBackGradB` | strided projection | stages 2/3/4 block 0 |

⚠ **The stride is on the 3×3 (`W2`), not the leading 1×1** — v1.5 / torchvision, which is what
`jax/MainResnet50Imagenet.lean` trains. So in the strided block `conv1`, `bn1` and `relu1` all run
at the **input** resolution `2hh`, and only `conv2` decimates. That asymmetry is why the strided
renderer carries two sets of zero-vectors (`zIn`/`zMidIn` at `2hh`, `zMid`/`zOut` at `hh`) where
the other two carry one.

⚠ **`mid = oc / 4`**, so an identity block's `mid` is a QUARTER of its channel count, not a
multiple. `VerifiedSpec.bottleneckStageSpec` already computes it that way and already selects the
projecting form when `stride != 1 || ic != oc` — which is exactly how stage 1 (64→256 at stride 1)
gets a projection. This file's block sequence must agree with that spec's ORDER, because the
driver packs `[θ|m|v]` off `net.specs`; `r50SigList` below is the render's side of that contract
and `#guard`s pin the counts.

## ⚠ No conv biases

Every conv here is bias-free (`convBnNB` in the spec — 9 tensors per identity block, 12 per
projection block), matching R34's ImageNet render. The `convBias` plumbing R34's renderer carries
for its CIFAR-era artifacts is deliberately absent rather than threaded and passed `false`.

## ⚠⚠ There is no incumbent hand-written R50 render to tie against

§3.2. Every other net's swap onto the verified renderer was licensed by a bit-exact numeric tie
against the hand-written artifact it replaced. R50 has no such artifact, so that license does not
exist here and **must not be implied**. The substitutes are the layer-level VJP oracle
(`tests/vjp_oracle/run.sh`) and a keep-1 known-answer check, and whichever one is used has to be
named in the commit that ships a number off this render.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

/-- The zero constant bound to every conv's bias operand. R50 is bias-free (`convBnNB`), but the
    proven conv ops still TAKE a bias, so it binds to `%zb{c}` and `zeroBiasPrelude` emits those
    constants once at the top of the body. Routed through `biasName` rather than spelled inline so
    the naming has one definition shared with R34's renderer. -/
private def zb (c : Nat) : String := biasName false "" c

-- ════════════════════════════════════════════════════════════════
-- § Signature lists — the render's side of the driver contract
-- ════════════════════════════════════════════════════════════════

/-- One bottleneck block's parameters, in `bneckIdBlk` order: `W1 g1 bt1 W2 g2 bt2 W3 g3 bt3`,
    plus `Wp gp btp` LAST when it projects. ⚠ The projection going last is not cosmetic — it is
    `bneckDownBlk`'s order and therefore the order the driver's `[θ|m|v]` walk assumes. -/
private def bnkSig (p : String) (cin mid oc : Nat) (proj : Bool) : List (String × List Nat) :=
  [(s!"%{p}W1", [mid,cin,1,1]), (s!"%{p}g1", [mid]), (s!"%{p}bt1", [mid]),
   (s!"%{p}W2", [mid,mid,3,3]), (s!"%{p}g2", [mid]), (s!"%{p}bt2", [mid]),
   (s!"%{p}W3", [oc,mid,1,1]),  (s!"%{p}g3", [oc]),  (s!"%{p}bt3", [oc])] ++
  (if proj then
     [(s!"%{p}Wp", [oc,cin,1,1]), (s!"%{p}gp", [oc]), (s!"%{p}btp", [oc])]
   else [])

/-- One bottleneck STAGE's parameters: block 0 projects (channels change, and in stage 1 that is
    the only thing that changes), the rest are identity blocks reading `oc`. -/
private def bnkStageSig (s : String) (ic oc count : Nat) : List (String × List Nat) :=
  let mid := oc / 4
  (bnkSig s!"{s}b0" ic mid oc true) ++
  ((List.range (count - 1)).flatMap (fun i => bnkSig s!"{s}b{i+1}" oc mid oc false))

/-- **The 161 R50 parameter inputs**, in func-arg order: stem (3), the `[3,4,6,3]` bottleneck
    stages, then the head. Single source for `%p`, `%pm`, `%pv` and the return order. -/
def r50ShapeList (nClasses : Nat) : List (String × List Nat) :=
  [("%sW", [64,3,7,7]), ("%sg", [64]), ("%sbt", [64])] ++
  bnkStageSig "s1"   64  256 3 ++
  bnkStageSig "s2"  256  512 4 ++
  bnkStageSig "s3"  512 1024 6 ++
  bnkStageSig "s4" 1024 2048 3 ++
  [("%Wd", [2048, nClasses]), ("%bd", [nClasses])]

/-- The same list as MLIR types. Derived, so the shapes have one definition. -/
def r50SigList (nClasses : Nat) : List (String × String) :=
  (r50ShapeList nClasses).map (fun (n, ds) => (n, ty ds))

/-- One block's BN running-stat slots, in BN-forward order `n1 n2 n3 [np]`. -/
private def bnkStatSig (p : String) (mid oc : Nat) (proj : Bool) : List (String × String) :=
  [(s!"%{p}n1mu", ty [mid]), (s!"%{p}n1var", ty [mid]),
   (s!"%{p}n2mu", ty [mid]), (s!"%{p}n2var", ty [mid]),
   (s!"%{p}n3mu", ty [oc]),  (s!"%{p}n3var", ty [oc])] ++
  (if proj then [(s!"%{p}npmu", ty [oc]), (s!"%{p}npvar", ty [oc])] else [])

private def bnkStageStatSig (s : String) (oc count : Nat) : List (String × String) :=
  let mid := oc / 4
  (bnkStatSig s!"{s}b0" mid oc true) ++
  ((List.range (count - 1)).flatMap (fun i => bnkStatSig s!"{s}b{i+1}" mid oc false))

/-- **The 106 running-stat inputs** — 53 BN layers × (μ, var), μ and var interleaved per layer,
    which is how the driver packs `runningBnStats` off `bnChannels`. -/
def r50StatSigList : List (String × String) :=
  [("%stnmu", ty [64]), ("%stnvar", ty [64])] ++
  bnkStageStatSig "s1"  256 3 ++
  bnkStageStatSig "s2"  512 4 ++
  bnkStageStatSig "s3" 1024 6 ++
  bnkStageStatSig "s4" 2048 3

-- 3 stem + (12+9+9) + (12+27) + (12+45) + (12+18) + 2 head = 161 parameter tensors.
#guard (r50SigList 1000).length == 161
-- 53 BN layers ⇒ 106 stat slots. stem 1 + s1 (4+3+3=10) + s2 13 + s3 19 + s4 10 = 53.
#guard r50StatSigList.length == 106

-- ════════════════════════════════════════════════════════════════
-- § Bottleneck block forwards (batch BN)
-- ════════════════════════════════════════════════════════════════

/-- Saved forward SSA names the bottleneck's backward + gradient passes reference. The 3-conv peer
    of `BFwdB`: two interior ReLUs (`r1`, `r2`) where the basic block has one. -/
structure BNFwd where
  code : String
  xin : String
  o  : String        -- block output (post-relu)
  a  : String        -- pre-output-relu sum
  c1 : String        -- conv1 out (= BN1 in)
  n1 : String        -- BN1 out (= relu1 pre-activation)
  r1 : String        -- relu1 out (= conv2 in)
  c2 : String        -- conv2 out (= BN2 in)
  n2 : String        -- BN2 out (= relu2 pre-activation)
  r2 : String        -- relu2 out (= conv3 in)
  c3 : String        -- conv3 out (= BN3 in)
  cp : String        -- projection conv out ("" for identity)
deriving Inhabited

/-- Identity bottleneck forward: `1×1 → BN → relu → 3×3 → BN → relu → 1×1 → BN → (+x) → relu`,
    all at `hh`, channels `oc → mid → mid → oc`. -/
private def bnkIdFwdB (B mid oc hh : Nat) (epsStr p xName : String) : StateM Nat BNFwd := do
  let ww := hh
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid oc 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zOut : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zMid : Vec (B*(mid*hh*ww)) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W1" (zb mid) zk1 zm) (.operand xName zOut))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zm zm (.operand nC1 zMid))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN1 zMid))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm) (.operand nR1 zMid))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zm zm (.operand nC2 zMid))
  let (cR2, nR2) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo) (.operand nR2 zMid))
  let (cN3, nN3) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" s!"%{p}bt3" epsStr 0 zo zo (.operand nC3 zOut))
  let (cA,  nA)  ← pretty B (.addVB (.operand nN3 zOut) (.operand xName zOut))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := oc*hh*ww)) (.operand nA zOut))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cA ++ cO,
         xin := xName, o := nO, a := nA,
         c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, n2 := nN2, r2 := nR2, c3 := nC3, cp := "" }

/-- ⭐ **Stride-1 projection bottleneck forward** — R50 stage 1 block 0, and nowhere else.
    `cin → mid → mid → oc` with the resolution unchanged, so the projection is a plain `1×1`
    conv → BN, NOT the strided one `bnkStridedFwdB` uses. -/
private def bnkProjFwdB (B cin mid oc hh : Nat) (epsStr p xName : String) : StateM Nat BNFwd := do
  let ww := hh
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn  : Vec (B*(cin*hh*ww)) := fun _ => 0
  let zOut : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zMid : Vec (B*(mid*hh*ww)) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W1" (zb mid) zk1 zm) (.operand xName zIn))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zm zm (.operand nC1 zMid))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN1 zMid))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm) (.operand nR1 zMid))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zm zm (.operand nC2 zMid))
  let (cR2, nR2) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo) (.operand nR2 zMid))
  let (cN3, nN3) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" s!"%{p}bt3" epsStr 0 zo zo (.operand nC3 zOut))
  -- the projection: a STRIDE-1 1×1 conv. `.conv`, not `.convStrided` — the whole point of this form.
  let (cCp, nCp) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}Wp" (zb oc) zkp zo) (.operand xName zIn))
  let (cNp, nNp) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}gp" s!"%{p}btp" epsStr 0 zo zo (.operand nCp zOut))
  let (cA,  nA)  ← pretty B (.addVB (.operand nN3 zOut) (.operand nNp zOut))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := oc*hh*ww)) (.operand nA zOut))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cCp ++ cNp ++ cA ++ cO,
         xin := xName, o := nO, a := nA,
         c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, n2 := nN2, r2 := nR2, c3 := nC3, cp := nCp }

/-- **Strided projection bottleneck forward** — stages 2/3/4 block 0. `cin → mid → mid → oc`,
    `2hh → hh`.

    ⚠ `conv1`/`bn1`/`relu1` run at the **input** resolution `2hh`; only `conv2` (the 3×3) is
    strided. v1.5. -/
private def bnkStridedFwdB (B cin mid oc hh : Nat) (epsStr p xName : String) : StateM Nat BNFwd := do
  let ww := hh
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn    : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zMidIn : Vec (B*(mid*(2*hh)*(2*ww))) := fun _ => 0
  let zMid   : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zOut   : Vec (B*(oc*hh*ww)) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (.conv (h := 2*hh) (w := 2*ww) s!"%{p}W1" (zb mid) zk1 zm) (.operand xName zIn))
  let (cN1, nN1) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}g1" s!"%{p}bt1" epsStr 0 zm zm (.operand nC1 zMidIn))
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := mid*(2*hh)*(2*ww))) (.operand nN1 zMidIn))
  -- ⚠⚠ THE STRIDE LIVES HERE, on the 3×3. Moving it to `W1` above is ResNet v1, a different net.
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (.convStrided (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm) (.operand nR1 zMidIn))
  let (cN2, nN2) ← pretty B (.bnBatchF (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" s!"%{p}bt2" epsStr 0 zm zm (.operand nC2 zMid))
  let (cR2, nR2) ← pretty B (.batchOp (N := B) (.relu (n := mid*hh*ww)) (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.batchOp (N := B) (.conv (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo) (.operand nR2 zMid))
  let (cN3, nN3) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" s!"%{p}bt3" epsStr 0 zo zo (.operand nC3 zOut))
  let (cCp, nCp) ← pretty B (.batchOp (N := B) (.convStrided (h := hh) (w := ww) s!"%{p}Wp" (zb oc) zkp zo) (.operand xName zIn))
  let (cNp, nNp) ← pretty B (.bnBatchF (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}gp" s!"%{p}btp" epsStr 0 zo zo (.operand nCp zOut))
  let (cA,  nA)  ← pretty B (.addVB (.operand nN3 zOut) (.operand nNp zOut))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := oc*hh*ww)) (.operand nA zOut))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cCp ++ cNp ++ cA ++ cO,
         xin := xName, o := nO, a := nA,
         c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, n2 := nN2, r2 := nR2, c3 := nC3, cp := nCp }

-- ════════════════════════════════════════════════════════════════
-- § Bottleneck block backwards + UN-FUSED parameter gradients
-- ════════════════════════════════════════════════════════════════

/-- Identity bottleneck backward + its 9 parameter gradients.

    Cotangent chain: `da` (output relu) → `bn3` → `conv3` → `relu2` → `bn2` → `conv2` → `relu1`
    → `bn1` → `conv1`, then `dx = dc1 + da` (the identity skip carries `da` unchanged).
    ⚠ Each BN's γ/β gradient reads the cotangent at that BN's OUTPUT: `da` for bn3, `dr2` for
    bn2, `dr1` for bn1 — off by one and the gradient is silently wrong. -/
private def bnkIdBackGradB (B mid oc hh : Nat) (epsStr p : String) (f : BNFwd) (dyName : String) :
    StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  let zm    : Vec mid := fun _ => 0
  let zo    : Vec oc := fun _ => 0
  let zk1   : Kernel4 mid oc 1 1 := fun _ _ _ _ => 0
  let zk2   : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3   : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zOut  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zMid  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zbnO  : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zbnM  : Vec (B*(mid*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zOut (.operand dyName zOut))
  let (cDn3, nDn3) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" f.c3 epsStr 0 zo zbnO (.operand nDa zbnO))
  let (cDc3, nDc3) ← pretty B (.convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" zk3 zo (.operand nDn3 zOut))
  let (cDr2, nDr2) ← pretty B (.selectPosB f.n2 zMid (.operand nDc3 zMid))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zm zbnM (.operand nDr2 zbnM))
  let (cDc2, nDc2) ← pretty B (.convBackBatched (N := B) (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" zk2 zm (.operand nDn2 zMid))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zMid (.operand nDc2 zMid))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zm zbnM (.operand nDr1 zbnM))
  let (cDc1, nDc1) ← pretty B (.convBackBatched (N := B) (ic := oc) (oc := mid) (h := hh) (w := ww) s!"%{p}W1" zk1 zm (.operand nDn1 zMid))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zOut) (.operand nDa zOut))
  -- parameter gradients, func-arg order: W1 g1 bt1 W2 g2 bt2 W3 g3 bt3
  let (cW1, nW1) ← pretty B (.convWeightGradB (ic := oc) (oc := mid) (h := hh) (w := ww) xName zm zOut zk1 (.operand nDn1 zMid))
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbnM (.operand nDr1 zbnM))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr1 zbnM))
  let (cW2, nW2) ← pretty B (.convWeightGradB (ic := mid) (oc := mid) (h := hh) (w := ww) f.r1 zm zMid zk2 (.operand nDn2 zMid))
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbnM (.operand nDr2 zbnM))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr2 zbnM))
  let (cW3, nW3) ← pretty B (.convWeightGradB (ic := mid) (oc := oc) (h := hh) (w := ww) f.r2 zo zMid zk3 (.operand nDn3 zOut))
  let (cg3, ng3) ← pretty B (.bnGammaGradB f.c3 epsStr 0 zbnO (.operand nDa zbnO))
  let (ct3, nt3) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDa zbnO))
  pure { code := cDa ++ cDn3 ++ cDc3 ++ cDr2 ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDx ++
                 cW1 ++ cg1 ++ ct1 ++ cW2 ++ cg2 ++ ct2 ++ cW3 ++ cg3 ++ ct3,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [mid,oc,1,1]⟩, ⟨s!"{p}g1", ng1, [mid]⟩, ⟨s!"{p}bt1", nt1, [mid]⟩,
                ⟨s!"{p}W2", nW2, [mid,mid,3,3]⟩, ⟨s!"{p}g2", ng2, [mid]⟩, ⟨s!"{p}bt2", nt2, [mid]⟩,
                ⟨s!"{p}W3", nW3, [oc,mid,1,1]⟩, ⟨s!"{p}g3", ng3, [oc]⟩, ⟨s!"{p}bt3", nt3, [oc]⟩] }

/-- ⭐ **Stride-1 projection bottleneck backward** + its 12 parameter gradients — stage 1 block 0.
    Identical to the identity backward except the skip branch carries its own `bn→conv` backward
    (`dnp`/`dcp`) and `dx = dc1 + dcp`. -/
private def bnkProjBackGradB (B cin mid oc hh : Nat) (epsStr p : String) (f : BNFwd)
    (dyName : String) : StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  let zm    : Vec mid := fun _ => 0
  let zo    : Vec oc := fun _ => 0
  let zk1   : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2   : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3   : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp   : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn   : Vec (B*(cin*hh*ww)) := fun _ => 0
  let zOut  : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zMid  : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zbnO  : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zbnM  : Vec (B*(mid*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zOut (.operand dyName zOut))
  let (cDn3, nDn3) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" f.c3 epsStr 0 zo zbnO (.operand nDa zbnO))
  let (cDc3, nDc3) ← pretty B (.convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" zk3 zo (.operand nDn3 zOut))
  let (cDr2, nDr2) ← pretty B (.selectPosB f.n2 zMid (.operand nDc3 zMid))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zm zbnM (.operand nDr2 zbnM))
  let (cDc2, nDc2) ← pretty B (.convBackBatched (N := B) (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" zk2 zm (.operand nDn2 zMid))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zMid (.operand nDc2 zMid))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g1" f.c1 epsStr 0 zm zbnM (.operand nDr1 zbnM))
  let (cDc1, nDc1) ← pretty B (.convBackBatched (N := B) (ic := cin) (oc := mid) (h := hh) (w := ww) s!"%{p}W1" zk1 zm (.operand nDn1 zMid))
  let (cDnp, nDnp) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}gp" f.cp epsStr 0 zo zbnO (.operand nDa zbnO))
  let (cDcp, nDcp) ← pretty B (.convBackBatched (N := B) (ic := cin) (oc := oc) (h := hh) (w := ww) s!"%{p}Wp" zkp zo (.operand nDnp zOut))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zIn) (.operand nDcp zIn))
  let (cW1, nW1) ← pretty B (.convWeightGradB (ic := cin) (oc := mid) (h := hh) (w := ww) xName zm zIn zk1 (.operand nDn1 zMid))
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbnM (.operand nDr1 zbnM))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr1 zbnM))
  let (cW2, nW2) ← pretty B (.convWeightGradB (ic := mid) (oc := mid) (h := hh) (w := ww) f.r1 zm zMid zk2 (.operand nDn2 zMid))
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbnM (.operand nDr2 zbnM))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr2 zbnM))
  let (cW3, nW3) ← pretty B (.convWeightGradB (ic := mid) (oc := oc) (h := hh) (w := ww) f.r2 zo zMid zk3 (.operand nDn3 zOut))
  let (cg3, ng3) ← pretty B (.bnGammaGradB f.c3 epsStr 0 zbnO (.operand nDa zbnO))
  let (ct3, nt3) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDa zbnO))
  let (cWp, nWp) ← pretty B (.convWeightGradB (ic := cin) (oc := oc) (h := hh) (w := ww) xName zo zIn zkp (.operand nDnp zOut))
  let (cgp, ngp) ← pretty B (.bnGammaGradB f.cp epsStr 0 zbnO (.operand nDa zbnO))
  let (ctp, ntp) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDa zbnO))
  pure { code := cDa ++ cDn3 ++ cDc3 ++ cDr2 ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++
                 cDnp ++ cDcp ++ cDx ++
                 cW1 ++ cg1 ++ ct1 ++ cW2 ++ cg2 ++ ct2 ++ cW3 ++ cg3 ++ ct3 ++ cWp ++ cgp ++ ctp,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [mid,cin,1,1]⟩, ⟨s!"{p}g1", ng1, [mid]⟩, ⟨s!"{p}bt1", nt1, [mid]⟩,
                ⟨s!"{p}W2", nW2, [mid,mid,3,3]⟩, ⟨s!"{p}g2", ng2, [mid]⟩, ⟨s!"{p}bt2", nt2, [mid]⟩,
                ⟨s!"{p}W3", nW3, [oc,mid,1,1]⟩, ⟨s!"{p}g3", ng3, [oc]⟩, ⟨s!"{p}bt3", nt3, [oc]⟩,
                ⟨s!"{p}Wp", nWp, [oc,cin,1,1]⟩, ⟨s!"{p}gp", ngp, [oc]⟩, ⟨s!"{p}btp", ntp, [oc]⟩] }

/-- **Strided projection bottleneck backward** + its 12 parameter gradients — stages 2/3/4 block 0.

    ⚠ `dc2` is the STRIDED conv backward, so it takes the cotangent from `hh` back up to `2hh`;
    everything upstream of it (`dr1`, `dn1`, `dc1`, `W1`'s grad) lives at `2hh`. -/
private def bnkStridedBackGradB (B cin mid oc hh : Nat) (epsStr p : String) (f : BNFwd)
    (dyName : String) : StateM Nat BBackB := do
  let xName := f.xin
  let ww := hh
  let zm    : Vec mid := fun _ => 0
  let zo    : Vec oc := fun _ => 0
  let zk1   : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2   : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3   : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp   : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn     : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zMidIn  : Vec (B*(mid*(2*hh)*(2*ww))) := fun _ => 0
  let zbnMIn  : Vec (B*(mid*((2*hh)*(2*ww)))) := fun _ => 0
  let zMid    : Vec (B*(mid*hh*ww)) := fun _ => 0
  let zOut    : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zbnO    : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zbnM    : Vec (B*(mid*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zOut (.operand dyName zOut))
  let (cDn3, nDn3) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}g3" f.c3 epsStr 0 zo zbnO (.operand nDa zbnO))
  let (cDc3, nDc3) ← pretty B (.convBackBatched (N := B) (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" zk3 zo (.operand nDn3 zOut))
  let (cDr2, nDr2) ← pretty B (.selectPosB f.n2 zMid (.operand nDc3 zMid))
  let (cDn2, nDn2) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := hh) (w := ww) s!"%{p}g2" f.c2 epsStr 0 zm zbnM (.operand nDr2 zbnM))
  -- the strided conv backward: hh → 2hh
  let (cDc2, nDc2) ← pretty B (.convStridedBackBatched (N := B) (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" zk2 zm (.operand nDn2 zMid))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zMidIn (.operand nDc2 zMidIn))
  let (cDn1, nDn1) ← pretty B (.bnBatchBack (N := B) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}g1" f.c1 epsStr 0 zm zbnMIn (.operand nDr1 zbnMIn))
  let (cDc1, nDc1) ← pretty B (.convBackBatched (N := B) (ic := cin) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}W1" zk1 zm (.operand nDn1 zMidIn))
  let (cDnp, nDnp) ← pretty B (.bnBatchBack (N := B) (oc := oc) (h := hh) (w := ww) s!"%{p}gp" f.cp epsStr 0 zo zbnO (.operand nDa zbnO))
  let (cDcp, nDcp) ← pretty B (.convStridedBackBatched (N := B) (ic := cin) (oc := oc) (h := hh) (w := ww) s!"%{p}Wp" zkp zo (.operand nDnp zOut))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zIn) (.operand nDcp zIn))
  let (cW1, nW1) ← pretty B (.convWeightGradB (ic := cin) (oc := mid) (h := 2*hh) (w := 2*ww) xName zm zIn zk1 (.operand nDn1 zMidIn))
  let (cg1, ng1) ← pretty B (.bnGammaGradB f.c1 epsStr 0 zbnMIn (.operand nDr1 zbnMIn))
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := 2*hh) (w := 2*ww) (.operand nDr1 zbnMIn))
  let (cW2, nW2) ← pretty B (.convStridedWeightGradB (ic := mid) (oc := mid) (h := hh) (w := ww) f.r1 zm zMidIn zk2 (.operand nDn2 zMid))
  let (cg2, ng2) ← pretty B (.bnGammaGradB f.c2 epsStr 0 zbnM (.operand nDr2 zbnM))
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := mid) (h := hh) (w := ww) (.operand nDr2 zbnM))
  let (cW3, nW3) ← pretty B (.convWeightGradB (ic := mid) (oc := oc) (h := hh) (w := ww) f.r2 zo zMid zk3 (.operand nDn3 zOut))
  let (cg3, ng3) ← pretty B (.bnGammaGradB f.c3 epsStr 0 zbnO (.operand nDa zbnO))
  let (ct3, nt3) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDa zbnO))
  let (cWp, nWp) ← pretty B (.convStridedWeightGradB (ic := cin) (oc := oc) (h := hh) (w := ww) xName zo zIn zkp (.operand nDnp zOut))
  let (cgp, ngp) ← pretty B (.bnGammaGradB f.cp epsStr 0 zbnO (.operand nDa zbnO))
  let (ctp, ntp) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDa zbnO))
  pure { code := cDa ++ cDn3 ++ cDc3 ++ cDr2 ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++
                 cDnp ++ cDcp ++ cDx ++
                 cW1 ++ cg1 ++ ct1 ++ cW2 ++ cg2 ++ ct2 ++ cW3 ++ cg3 ++ ct3 ++ cWp ++ cgp ++ ctp,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [mid,cin,1,1]⟩, ⟨s!"{p}g1", ng1, [mid]⟩, ⟨s!"{p}bt1", nt1, [mid]⟩,
                ⟨s!"{p}W2", nW2, [mid,mid,3,3]⟩, ⟨s!"{p}g2", ng2, [mid]⟩, ⟨s!"{p}bt2", nt2, [mid]⟩,
                ⟨s!"{p}W3", nW3, [oc,mid,1,1]⟩, ⟨s!"{p}g3", ng3, [oc]⟩, ⟨s!"{p}bt3", nt3, [oc]⟩,
                ⟨s!"{p}Wp", nWp, [oc,cin,1,1]⟩, ⟨s!"{p}gp", ngp, [oc]⟩, ⟨s!"{p}btp", ntp, [oc]⟩] }

-- ════════════════════════════════════════════════════════════════
-- § The whole-net batched train step
-- ════════════════════════════════════════════════════════════════

set_option maxRecDepth 4000000 in
/-- Everything the whole-net render needs out of ONE forward traversal of ResNet-50: the emitted
    code, the logits, and every saved activation the backward reads.

    ⭐⭐ **This exists so `@resnet50_fwd` and `@resnet50_adam_train_step` cannot be different nets.**
    They were: `resnet50FwdFaithfulV` built its forward from the PER-EXAMPLE chain (`r50FwdChain`,
    which reaches `bnSite` and therefore `bnPerChannelF`, reduce `[2,3]`, divisor `H·W`) while the
    train step is batch BN (reduce `[0,2,3]`, divisor `B·H·W`). Its own docstring claimed "the same
    forward the train step differentiates", which was the invariant that did not hold.
    `scripts/regen_verified_mlir.sh check` could not see it: it only ever paired a forward with the
    SGD train step, and R50 has none (`planning/mnv4_verified.md` §3d(b)).

    ⚠ The eval forward is deliberately NOT moved onto this chain. `bnEval` reads frozen per-channel
    statistics as graph inputs, so it is the same arithmetic in either vocabulary and has no
    BN-world to disagree about — and it is the artifact that actually scores, so leaving its bytes
    untouched keeps 89.86% (`runs/r50_imagenette_adam_80ep.log`) exactly where it is. -/
structure R50FwdRecB where
  code : String
  logits : String
  stc : String            -- stem conv out (= stem-BN input)
  stn : String            -- stem BN out   (= stem-relu pre-activation)
  str : String            -- stem relu out (= maxpool input)
  stp : String            -- maxpool out
  gap : String            -- GAP out (= dense input)
  b : Array BNFwd         -- the 16 bottleneck blocks, in forward order
deriving Inhabited

set_option maxRecDepth 4000000 in
/-- **The ResNet-50 forward chain at the BATCHED index** — one traversal, consumed by both
    `@resnet50_fwd` and the train step that differentiates it. -/
def r50FwdChainB (B nClasses : Nat) (epsStr : String) (q : Nat := 7) :
    StateM Nat R50FwdRecB := do
  -- The ladder, bottom-up. At q = 7: 5→…  no — at q = 7 these are 7, 14, 28, 56, 112 and the
  -- input is 224; at q = 5 they are 5, 10, 20, 40, 80 and the input is 160.
  let q5 := q            -- stage 4 / the GAP window
  let q4 := 2 * q5       -- stage 3
  let q3 := 2 * q4       -- stage 2
  let q2 := 2 * q3       -- stage 1, and the max-pool's output
  let q1 := 2 * q2       -- the stem conv's output
  -- ═══ stem: 7×7/s2 conv → batch BN → relu → He et al.'s 3×3/s2 pool (img→img/2→img/4) ═══
  -- ⚠ `2*q1` rather than a name, because `convStrided (h := q1)` demands its operand at exactly
  -- `Vec (B*(3*(2*q1)*(2*q1)))` and any other spelling of the same number is a different TERM.
  let zx    : Vec (B*(3*(2*q1)*(2*q1))) := fun _ => 0
  let zSk   : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
  let z64   : Vec 64 := fun _ => 0
  let z112  : Vec (B*(64*q1*q1)) := fun _ => 0
  let z112b : Vec (B*(64*(q1*q1))) := fun _ => 0
  let z56   : Vec (B*(64*q2*q2)) := fun _ => 0
  let (cStc, nStc) ← pretty B (.batchOp (N := B) (.convStrided (h := q1) (w := q1) "%sW" (zb 64) zSk z64) (.operand "%x" zx))
  let (cStn, nStn) ← pretty B (.bnBatchF (N := B) (oc := 64) (h := q1) (w := q1) "%sg" "%sbt" epsStr 0 z64 z64 (.operand nStc z112))
  let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu (n := 64*q1*q1)) (.operand nStn z112))
  let (cStp, nStp) ← pretty B (.batchOp (N := B) (.maxPool3s2 (c := 64) (h := q2) (w := q2)) (.operand nStr z112))
  -- ═══ 16 bottleneck blocks, [3,4,6,3] ═══
  let f1  ← bnkProjFwdB    B   64  64  256 q2 epsStr "s1b0" nStp   -- ⭐ the stride-1 projection
  let f2  ← bnkIdFwdB      B       64  256 q2 epsStr "s1b1" f1.o
  let f3  ← bnkIdFwdB      B       64  256 q2 epsStr "s1b2" f2.o
  let f4  ← bnkStridedFwdB B  256 128  512 q3 epsStr "s2b0" f3.o
  let f5  ← bnkIdFwdB      B      128  512 q3 epsStr "s2b1" f4.o
  let f6  ← bnkIdFwdB      B      128  512 q3 epsStr "s2b2" f5.o
  let f7  ← bnkIdFwdB      B      128  512 q3 epsStr "s2b3" f6.o
  let f8  ← bnkStridedFwdB B  512 256 1024 q4 epsStr "s3b0" f7.o
  let f9  ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b1" f8.o
  let f10 ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b2" f9.o
  let f11 ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b3" f10.o
  let f12 ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b4" f11.o
  let f13 ← bnkIdFwdB      B      256 1024 q4 epsStr "s3b5" f12.o
  let f14 ← bnkStridedFwdB B 1024 512 2048  q5 epsStr "s4b0" f13.o
  let f15 ← bnkIdFwdB      B      512 2048  q5 epsStr "s4b1" f14.o
  let f16 ← bnkIdFwdB      B      512 2048  q5 epsStr "s4b2" f15.o
  -- ═══ head: GAP(7×7) → dense(2048→nClasses) ═══
  let zL    : Vec (B*(2048*q5*q5)) := fun _ => 0
  let z2048 : Vec (B*2048) := fun _ => 0
  let zWd   : Mat 2048 nClasses := fun _ _ => 0
  let zNC   : Vec nClasses := fun _ => 0
  let zNCb  : Vec (B*(1*nClasses)) := fun _ => 0
  let zNCp  : Vec (B*nClasses) := fun _ => 0
  let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 2048) (h := q5) (w := q5)) (.operand f16.o zL))
  let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC) (.operand nGap z2048))
  pure { code := cStc ++ cStn ++ cStr ++ cStp ++
                 f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++
                 f8.code ++ f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++
                 f15.code ++ f16.code ++ cGap ++ cLog,
         logits := nLog, stc := nStc, stn := nStn, str := nStr, stp := nStp, gap := nGap,
         b := #[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16] }

/-- **ResNet-50 `[3,4,6,3]` bottleneck train step, batch-BN, rendered at `N := B`.**

    161 θ / 161 m / 161 v, `%lr`/`%bc1`/`%bc2`, 106 running-stat slots and `%onehot` in;
    161 θ' / 161 m' / 161 v', `%loss`/`%bc1`/`%bc2` and 106 batch stats out. Parameter ORDER comes
    from `r50SigList`, the same single source the signature and the return list use, so the
    arity/order contract cannot drift within this file — and it is written to agree with
    `VerifiedSpec.bottleneckStageSpec`, which is what the DRIVER walks.

    ⚠ The block sequence is `[3,4,6,3]` with block 0 of every stage projecting. Stage 1's projects
    at **stride 1** (`bnkProjFwdB`); stages 2/3/4 project strided. -/
def resnet50TrainStepFaithfulB (B nClasses : Nat) (epsStr : String)
    (replicas : Nat := 1) (opt : R34Opt := .adamw) (slug : String := "resnet50in")
    -- ⚠ TRAILING, per §2m: a parameter inserted mid-list captures an existing positional argument.
    -- `bce` swaps the LOSS (and therefore the cotangent) for BCE-with-logits — RSB-A2/A3's, and the
    -- reason the recipe's lr is what it is. `vSuffix` appends to `r34AdamVariant`'s name so the
    -- artifact, the entry point and `LEAN_MLIR_VARIANT` stay one string.
    (bce : Bool := false) (vSuffix : String := "")
    -- ⭐⭐ RESOLUTION, and it is parameterised by the FINAL feature size rather than by the input.
    -- `q = 7` is 224 (every committed artifact, byte for byte); `q = 5` is RSB-A3's **160**.
    --
    -- ⚠⚠ **Deriving UPWARD by doubling is not a style choice, it is what makes the dependent types
    -- work.** Every size relation this graph needs is of the form "the input to a stride-2 op is
    -- twice its output", and `Vec (ic*(2*h)*(2*w)) → Vec (oc*h*w)` states that IN THE TYPE. Written
    -- downward (`imgSize/2`, `imgSize/4`, …) Lean would have to prove `2*(imgSize/4) = imgSize/2`,
    -- which is false for odd inputs and not definitional for a variable. Written as `q4 := 2*q5`,
    -- `q3 := 2*q4`, … every such equation holds by zeta-reduction and nothing needs a proof.
    (q : Nat := 7) : String :=
  let optLabel : String := match opt with
    | .adamw          => "AdamW"
    | .heavyBall      => "heavy-ball momentum + coupled L2"
    | .lamb           => "LAMB (per-tensor trust ratio)"
    | .adamwAccum k   => s!"AdamW over {k} ACCUMULATED micro-batches"
    | .lambAccum k    => s!"LAMB (per-tensor trust ratio) over {k} ACCUMULATED micro-batches"
  -- ⚠ TYPE-based, so unlike the driver's string predicate this one could not silently miss the new
  -- arm — adding a constructor without extending it is a non-exhaustive match, i.e. a build error.
  -- That asymmetry is exactly why the driver needs `TestVariantPredicates` and this line does not.
  let accOn := match opt with | .adamwAccum _ => true | .lambAccum _ => true | _ => false
  let go : StateM Nat String := do
    -- ═══ forward: THE SHARED CHAIN, not a second copy (see `r50FwdChainB`) ═══
    let fw ← r50FwdChainB B nClasses epsStr q
    let q5 := q
    let q4 := 2 * q5
    let q3 := 2 * q4
    let q2 := 2 * q3
    let q1 := 2 * q2
    let zx    : Vec (B*(3*(2*q1)*(2*q1))) := fun _ => 0
    let zSk   : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
    let z64   : Vec 64 := fun _ => 0
    let z112  : Vec (B*(64*q1*q1)) := fun _ => 0
    let z112b : Vec (B*(64*(q1*q1))) := fun _ => 0
    let z56   : Vec (B*(64*q2*q2)) := fun _ => 0
    let zL    : Vec (B*(2048*q5*q5)) := fun _ => 0
    let z2048 : Vec (B*2048) := fun _ => 0
    let zWd   : Mat 2048 nClasses := fun _ _ => 0
    let zNC   : Vec nClasses := fun _ => 0
    let zNCb  : Vec (B*(1*nClasses)) := fun _ => 0
    let zNCp  : Vec (B*nClasses) := fun _ => 0
    let nStc := fw.stc; let nStn := fw.stn; let nStr := fw.str; let nStp := fw.stp
    let nGap := fw.gap; let nLog := fw.logits
    let f1 := fw.b[0]!;  let f2 := fw.b[1]!;  let f3 := fw.b[2]!;  let f4 := fw.b[3]!
    let f5 := fw.b[4]!;  let f6 := fw.b[5]!;  let f7 := fw.b[6]!;  let f8 := fw.b[7]!
    let f9 := fw.b[8]!;  let f10 := fw.b[9]!; let f11 := fw.b[10]!; let f12 := fw.b[11]!
    let f13 := fw.b[12]!; let f14 := fw.b[13]!; let f15 := fw.b[14]!; let f16 := fw.b[15]!
    -- ═══ the loss cotangent ═══
    --
    -- ⭐⭐ **BCE-with-logits is the SAME SHAPE with one op swapped, and THREE ops instead of five.**
    -- Softmax-CE's cotangent is `(softmax(z) − t)/B` with `t` the smoothed target; BCE's is
    -- `(σ(z) − t)/(B·K)`. So `softmaxRow → sigmoidB`, and the two label-smoothing nodes disappear.
    --
    -- ⚠⚠ **THE DIVISOR IS `B·K`, NOT `B`, AND THAT IS NOT A DETAIL.** timm's `BinaryCrossEntropy`
    -- is `reduction='mean'` over B×C, not the mean of the per-example SUM over classes. The
    -- reference's own comment: *"that would be NC× larger and need an NC× smaller lr — RSB-A2's
    -- lr 5e-3 is tuned to this form."* At K = 1000 the two differ by 1000× on the effective step.
    --
    -- ⚠ **NO LABEL SMOOTHING on the BCE path**, and that is A3's recipe rather than an omission:
    -- timm's a3 arg string is `…-m0.1-sd0.0-d0.0-ls0.0-100`, i.e. `ls0.0`. The soft targets come
    -- from mixup/cutmix through `%onehot` (wire v2), not from a smoothing constant in the loss.
    let (cSm,  nSm)  ← if bce
      -- ⚠ `zNCp : Vec (B*nClasses)`, not `zNCb : Vec (B*(1*nClasses))`. The two indices are equal
      -- but NOT definitionally so for a variable `nClasses` (`Nat.mul` recurses on its second
      -- argument, so `1 * n` does not reduce), and `sigmoidB`'s `{N n}` unify against the bare
      -- product. The `if` still typechecks because both branches are `StateM Nat (String × String)`
      -- — the SHlo index never escapes the branch.
      then pretty B (.sigmoidB (N := B) (n := nClasses) (.operand nLog zNCp))
      else pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses)) (.operand nLog zNCb))
    let (cD0,  nD0)  ← pretty B (.subB (.operand nSm zNCb) (.operand "%onehot" zNCb))
    let (cLsa, nLsa) ← if bce then pure ("", nD0)
      else pretty B (.scaleB "0.100000" 0 (.operand "%onehot" zNCb))
    let (cD1,  nD1)  ← if bce then pure ("", nD0)
      else pretty B (.addVB (.operand nD0 zNCb) (.operand nLsa zNCb))
    let (cD2,  nD2)  ← if bce then pure ("", nD0)
      else pretty B (.shiftB s!"-{alphaOverK nClasses}" 0 (.operand nD1 zNCb))
    let (cDy,  nDy)  ← pretty B (.divConstB (if bce then s!"{B * nClasses}.0" else s!"{B}.0") 0
                                  (.operand nD2 zNCb))
    -- ═══ head backward + dense grads ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (N := B) (.denseRowBack (rows := 1) (a := 2048) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    let (cWd,  nWd)  ← pretty B (.denseWeightGradB (c := nClasses) nGap z2048 (.operand nDy zNCp))
    let (cbd,  nbd)  ← pretty B (.denseBiasGradB (N := B) (.operand nDy zNCp))
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 2048) (h := q5) (w := q5) (.operand nDgi z2048))
    -- ═══ 16 block backwards, in reverse ═══
    let b16 ← bnkIdBackGradB      B      512 2048  q5 epsStr "s4b2" f16 nDgp
    let b15 ← bnkIdBackGradB      B      512 2048  q5 epsStr "s4b1" f15 b16.dx
    let b14 ← bnkStridedBackGradB B 1024 512 2048  q5 epsStr "s4b0" f14 b15.dx
    let b13 ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b5" f13 b14.dx
    let b12 ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b4" f12 b13.dx
    let b11 ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b3" f11 b12.dx
    let b10 ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b2" f10 b11.dx
    let b9  ← bnkIdBackGradB      B      256 1024 q4 epsStr "s3b1" f9  b10.dx
    let b8  ← bnkStridedBackGradB B  512 256 1024 q4 epsStr "s3b0" f8  b9.dx
    let b7  ← bnkIdBackGradB      B      128  512 q3 epsStr "s2b3" f7  b8.dx
    let b6  ← bnkIdBackGradB      B      128  512 q3 epsStr "s2b2" f6  b7.dx
    let b5  ← bnkIdBackGradB      B      128  512 q3 epsStr "s2b1" f5  b6.dx
    let b4  ← bnkStridedBackGradB B  256 128  512 q3 epsStr "s2b0" f4  b5.dx
    let b3  ← bnkIdBackGradB      B       64  256 q2 epsStr "s1b2" f3  b4.dx
    let b2  ← bnkIdBackGradB      B       64  256 q2 epsStr "s1b1" f2  b3.dx
    let b1  ← bnkProjBackGradB    B   64  64  256 q2 epsStr "s1b0" f1  b2.dx
    -- ═══ stem backward ═══
    let (cDmp, nDmp) ← pretty B (.maxPool3s2BackB (N := B) (c := 64) (h := q2) (w := q2) nStr z112 (.operand b1.dx z56))
    let (cDsr, nDsr) ← pretty B (.selectPosB nStn z112 (.operand nDmp z112))
    let (cDsn, nDsn) ← pretty B (.bnBatchBack (N := B) (oc := 64) (h := q1) (w := q1) "%sg" nStc epsStr 0 z64 z112b (.operand nDsr z112b))
    let (csW, nsW) ← pretty B (.convStridedWeightGradB "%x" z64 zx zSk (.operand nDsn z112))
    let (csg, nsg) ← pretty B (.bnGammaGradB nStc epsStr 0 z112b (.operand nDsr z112b))
    let (cst, nst) ← pretty B (.bnBetaGradB (N := B) (oc := 64) (h := q1) (w := q1) (.operand nDsr z112b))
    -- ═══ BN running statistics, from each BN layer's INPUT ═══
    let bnStat (oc hh : Nat) (xn : String) : StateM Nat (String × String × String) := do
      let zbv : Vec (B*(oc*(hh*hh))) := fun _ => 0
      let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zbv))
      let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zbv))
      pure (cM ++ cV, nM, nV)
    -- identity + stride-1-projection blocks keep every BN at `hh`.
    let idStats (mid oc hh : Nat) (f : BNFwd) : StateM Nat (String × List String) := do
      let (c1, m1, v1) ← bnStat mid hh f.c1
      let (c2, m2, v2) ← bnStat mid hh f.c2
      let (c3, m3, v3) ← bnStat oc hh f.c3
      pure (c1 ++ c2 ++ c3, [m1, v1, m2, v2, m3, v3])
    let projStats (mid oc hh : Nat) (f : BNFwd) : StateM Nat (String × List String) := do
      let (cb, ns) ← idStats mid oc hh f
      let (cp, mp, vp) ← bnStat oc hh f.cp
      pure (cb ++ cp, ns ++ [mp, vp])
    -- ⚠ the STRIDED block's bn1 lives at `2*hh`; only bn2/bn3/bnp are at `hh`.
    let strStats (mid oc hh : Nat) (f : BNFwd) : StateM Nat (String × List String) := do
      let (c1, m1, v1) ← bnStat mid (2*hh) f.c1
      let (c2, m2, v2) ← bnStat mid hh f.c2
      let (c3, m3, v3) ← bnStat oc hh f.c3
      let (cp, mp, vp) ← bnStat oc hh f.cp
      pure (c1 ++ c2 ++ c3 ++ cp, [m1, v1, m2, v2, m3, v3, mp, vp])
    let (cSt0, st0) ← bnStat 64 q1 nStc
    let (cSt1,  st1)  ← projStats  64  256 q2 f1
    let (cSt2,  st2)  ← idStats    64  256 q2 f2
    let (cSt3,  st3)  ← idStats    64  256 q2 f3
    let (cSt4,  st4)  ← strStats  128  512 q3 f4
    let (cSt5,  st5)  ← idStats   128  512 q3 f5
    let (cSt6,  st6)  ← idStats   128  512 q3 f6
    let (cSt7,  st7)  ← idStats   128  512 q3 f7
    let (cSt8,  st8)  ← strStats  256 1024 q4 f8
    let (cSt9,  st9)  ← idStats   256 1024 q4 f9
    let (cSt10, st10) ← idStats   256 1024 q4 f10
    let (cSt11, st11) ← idStats   256 1024 q4 f11
    let (cSt12, st12) ← idStats   256 1024 q4 f12
    let (cSt13, st13) ← idStats   256 1024 q4 f13
    let (cSt14, st14) ← strStats  512 2048 q5 f14
    let (cSt15, st15) ← idStats   512 2048 q5 f15
    let (cSt16, st16) ← idStats   512 2048 q5 f16
    -- ═══ the 161 parameter gradients in func-arg order ═══
    let stemPs : List PGrad := [⟨"sW", nsW, [64,3,7,7]⟩, ⟨"sg", nsg, [64]⟩, ⟨"sbt", nst, [64]⟩]
    let headPs : List PGrad := [⟨"Wd", nWd, [2048, nClasses]⟩, ⟨"bd", nbd, [nClasses]⟩]
    let allPs : List PGrad := stemPs ++
      b1.ps ++ b2.ps ++ b3.ps ++ b4.ps ++ b5.ps ++ b6.ps ++ b7.ps ++ b8.ps ++
      b9.ps ++ b10.ps ++ b11.ps ++ b12.ps ++ b13.ps ++ b14.ps ++ b15.ps ++ b16.ps ++ headPs
    -- ═══ the optimizer: one proven triple per parameter, `optOne` shared with R34 ═══
    let mut adamCode := ""
    let mut thetaN : List String := []
    let mut mNames : List String := []
    let mut vNames : List String := []
    let mut aNames : List String := []
    for g in allPs do
      let (c, nT, nM, nV, nA) ← optOne opt B replicas g
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]
      mNames := mNames ++ [nM]
      vNames := vNames ++ [nV]
      -- ⭐ The accumulator's output name, present only under `.adamwAccum`. It becomes the FOURTH
      -- region of the packed blob — the same shape the EMA renders use, so the driver's
      -- `nRegions = 4` path is reused rather than a second one being written.
      match nA with | some a => aNames := aNames ++ [a] | none => pure ()
    let statCode := cSt0 ++ cSt1 ++ cSt2 ++ cSt3 ++ cSt4 ++ cSt5 ++ cSt6 ++ cSt7 ++ cSt8 ++
      cSt9 ++ cSt10 ++ cSt11 ++ cSt12 ++ cSt13 ++ cSt14 ++ cSt15 ++ cSt16
    let statNames := st0.1 :: st0.2 :: (st1 ++ st2 ++ st3 ++ st4 ++ st5 ++ st6 ++ st7 ++ st8 ++
      st9 ++ st10 ++ st11 ++ st12 ++ st13 ++ st14 ++ st15 ++ st16)
    -- `%loss` is REPORT-ONLY (logging), on no gradient path, and NOT `pretty` of an AST node — the
    -- same §5 carve-out R34's takes. SMOOTHED CE, matching the cotangent's soft target; a first cut
    -- on R34 computed plain CE here and only a numeric tie caught it.
    -- ⭐ **BCE-with-logits, in the stable form `softplus(z) − t·z`.** Expanding
    -- `t·softplus(−z) + (1−t)·softplus(z)` with the identity `softplus(−x) = softplus(x) − x`
    -- collapses the reference's two softplus calls to ONE, and the result never exponentiates a
    -- positive number: `softplus(z) = max(z,0) + log(1 + exp(−|z|))`. Report-only, like the CE
    -- loss beside it — the §5 carve-out — but the arithmetic still has to be right, because the
    -- epoch curve is how the run is judged against the reference (§2b's `%loss` bug).
    let lossCodeBce :=
      "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
      "    // BCE-with-logits, mean over B x K: softplus(z) - t*z, softplus stable as\n" ++
      "    // max(z,0) + log(1 + exp(-|z|)). ⚠ mean over B*K, NOT mean of the per-example sum.\n" ++
      s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %lzb = stablehlo.constant dense<0.0> : {ty [B, nClasses]}\n" ++
      s!"    %labs = stablehlo.abs {nLog} : {ty [B, nClasses]}\n" ++
      s!"    %lneg = stablehlo.negate %labs : {ty [B, nClasses]}\n" ++
      s!"    %lexp = stablehlo.exponential %lneg : {ty [B, nClasses]}\n" ++
      s!"    %lone = stablehlo.constant dense<1.0> : {ty [B, nClasses]}\n" ++
      s!"    %l1pe = stablehlo.add %lone, %lexp : {ty [B, nClasses]}\n" ++
      s!"    %llg = stablehlo.log %l1pe : {ty [B, nClasses]}\n" ++
      s!"    %lmax = stablehlo.maximum {nLog}, %lzb : {ty [B, nClasses]}\n" ++
      s!"    %lsp = stablehlo.add %lmax, %llg : {ty [B, nClasses]}\n" ++
      s!"    %ltz = stablehlo.multiply %onehot, {nLog} : {ty [B, nClasses]}\n" ++
      s!"    %lbce = stablehlo.subtract %lsp, %ltz : {ty [B, nClasses]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%lbce init: %lz) applies stablehlo.add across dimensions = [0, 1] : ({ty [B, nClasses]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{B * nClasses}.0> : tensor<f32>\n" ++
      s!"    %loss = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n"
    let lossCode :=
      "    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──\n" ++
      s!"    %lz = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %llog = stablehlo.log {nSm} : {ty [B, nClasses]}\n" ++
      s!"    %lohll = stablehlo.multiply %onehot, %llog : {ty [B, nClasses]}\n" ++
      s!"    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B, nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
      s!"    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : ({ty [B, nClasses]}, tensor<f32>) -> {ty [B]}\n" ++
      s!"    %lomac = stablehlo.constant dense<0.900000> : {ty [B]}\n" ++
      s!"    %laKc = stablehlo.constant dense<{alphaOverK nClasses}> : {ty [B]}\n" ++
      s!"    %llt1 = stablehlo.multiply %lomac, %lt1s : {ty [B]}\n" ++
      s!"    %llt2 = stablehlo.multiply %laKc, %llsr : {ty [B]}\n" ++
      s!"    %llpe = stablehlo.add %llt1, %llt2 : {ty [B]}\n" ++
      s!"    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : ({ty [B]}, tensor<f32>) -> tensor<f32>\n" ++
      s!"    %lbfc = stablehlo.constant dense<{B}.0> : tensor<f32>\n" ++
      s!"    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>\n" ++
      s!"    %loss = stablehlo.negate %lossm : tensor<f32>\n"
    let body := fw.code ++ cSm ++ cD0 ++ cLsa ++ cD1 ++ cD2 ++ cDy ++
      cDgi ++ cWd ++ cbd ++ cDgp ++
      b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++ b10.code ++ b9.code ++
      b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDmp ++ cDsr ++ cDsn ++ csW ++ csg ++ cst ++ statCode
    let pTypes : List String := allPs.map (fun g => ty g.ds)
    let statTypes : List String := (r50StatSigList.map (·.2))
    -- ⚠ THE PACKED LAYOUT IS `[θ|m|v|(G)| scalars | BN stats]`, and the driver indexes it off
    -- `nRegions`/`nScalars` rather than literals — so the accumulator has to sit where the EMA
    -- shadow sits (fourth region) and the two accum scalars where EMA's two sit (slots 4 and 5).
    -- `%aup`/`%akeep` ride out as passthroughs so `#out = #in − 2` holds exactly as for `.adamw`.
    let accScalars := if accOn then ["%aup", "%akeep"] else []
    let accScalarTys := accScalars.map (fun _ => "tensor<f32>")
    let retVals := thetaN ++ mNames ++ vNames ++ aNames ++
      ["%loss", "%bc1", "%bc2"] ++ accScalars ++ statNames
    let retTys  := pTypes ++ pTypes ++ pTypes ++ (if accOn then pTypes else []) ++
      ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ accScalarTys ++ statTypes
    pure <|
      (if replicas ≤ 1 then
        s!"    // ── ResNet-50 bottleneck batch-BN {optLabel} train step: every line is pretty(verified AST node) ──\n"
       else
        s!"    // ── ResNet-50 bottleneck batch-BN {optLabel} train step, DATA-PARALLEL over {replicas} replicas ──\n" ++
        "    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`\n" ++
        "    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5).\n") ++
      zeroBiasPrelude false [64, 128, 256, 512, 1024, 2048] ++ body ++ optConstsB opt ++ adamCode ++
      (if bce then lossCodeBce else lossCode) ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let sigList : List (String × String) := r50SigList nClasses
  let pSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let mSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}m: {t}"))
  let vSig := String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}v: {t}"))
  -- The accumulator region `G`, named `<p>a`, present only under `.adamwAccum`. At every other
  -- optimizer this is the empty string, so the three committed R50 artifacts re-render byte for byte.
  let aSig := if accOn then ", " ++ String.intercalate ", "
                              (sigList.map (fun (n, t) => s!"{n}a: {t}")) else ""
  let accSSig := if accOn then ", %aup: tensor<f32>, %akeep: tensor<f32>" else ""
  let statSig := String.intercalate ", " (r50StatSigList.map (fun (n, t) => s!"{n}i: {t}"))
  let inSig := s!"%x: {ty [B, 3*(32*q)*(32*q)]}, " ++ pSig ++ ", " ++ mSig ++ ", " ++ vSig ++ aSig ++
    ", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>" ++ accSSig ++ ", " ++ statSig ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (·.2)
  let accTy := if accOn then ["tensor<f32>", "tensor<f32>"] else []
  let outSig := String.intercalate ", "
    (pTy ++ pTy ++ pTy ++ (if accOn then pTy else []) ++
     ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ accTy ++ (r50StatSigList.map (·.2)))
  let inner : String := go.run' 0
  -- ⚠ Same `{slug}_{variant}_train_step` convention the shim checks; `r34AdamVariant` is reused as
  -- the single source for the variant name so R50's artifact names cannot drift from R34's rule.
  let fname := s!"{slug}_{r34AdamVariant B replicas opt}{vSuffix}_train_step"
  "module @m {\n" ++
  s!"  func.func @{fname}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The forward chain — `@resnet50in_fwd` and `@resnet50in_fwd_eval`
-- ════════════════════════════════════════════════════════════════

/-! ⚠ These live here rather than in a `ResNet50Render.lean` peer of R34's split. That split is
historical — R34 has a per-example-BN net AND a batch-BN net, and they are different functions, so
they get different files. R50 has only the batch-BN net, so one file keeps `r50SigList` and every
consumer of it together.

⚠⚠ **The train and eval forwards MUST be the same chain with one switch**, which is why `bnSite`
(shared with R34, now public) takes the mode rather than each render spelling its BN. §2g's
`mobilenetv2_fwd` defect was exactly this drifting: a net trained with one pool and scored with
another, logits rel 1.86, silent. -/

/-- Identity bottleneck forward, per-example vocabulary. -/
private def bnkIdFwdV (B mid oc hh : Nat) (mode : R34Bn) (epsStr p xName : String) :
    StateM Nat (String × String) := do
  let ww := hh
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid oc 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zOut : Vec (oc*hh*ww) := fun _ => 0
  let zMid : Vec (mid*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvF (ic := oc) (oc := mid) (h := hh) (w := ww) s!"%{p}W1" (zb mid) zk1 zm (.operand xName zOut))
  let (cN1, nN1) ← bnSite B mid hh mode epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zMid))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm (.operand nR1 zMid))
  let (cN2, nN2) ← bnSite B mid hh mode epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cR2, nR2) ← pretty B (.reluF (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo (.operand nR2 zMid))
  let (cN3, nN3) ← bnSite B oc hh mode epsStr s!"%{p}g3" s!"%{p}bt3" s!"{p}n3" nC3
  let (cA,  nA)  ← pretty B (.addV (.operand nN3 zOut) (.operand xName zOut))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zOut))
  pure (cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cA ++ cO, nO)

/-- ⭐ Stride-1 projection bottleneck forward — stage 1 block 0. The projection is `.flatConvF`,
    NOT `.flatConvStridedF`. -/
private def bnkProjFwdV (B cin mid oc hh : Nat) (mode : R34Bn) (epsStr p xName : String) :
    StateM Nat (String × String) := do
  let ww := hh
  let zm   : Vec mid := fun _ => 0
  let zo   : Vec oc := fun _ => 0
  let zk1  : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2  : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3  : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp  : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn  : Vec (cin*hh*ww) := fun _ => 0
  let zOut : Vec (oc*hh*ww) := fun _ => 0
  let zMid : Vec (mid*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvF (ic := cin) (oc := mid) (h := hh) (w := ww) s!"%{p}W1" (zb mid) zk1 zm (.operand xName zIn))
  let (cN1, nN1) ← bnSite B mid hh mode epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zMid))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm (.operand nR1 zMid))
  let (cN2, nN2) ← bnSite B mid hh mode epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cR2, nR2) ← pretty B (.reluF (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo (.operand nR2 zMid))
  let (cN3, nN3) ← bnSite B oc hh mode epsStr s!"%{p}g3" s!"%{p}bt3" s!"{p}n3" nC3
  let (cCp, nCp) ← pretty B (.flatConvF (ic := cin) (oc := oc) (h := hh) (w := ww) s!"%{p}Wp" (zb oc) zkp zo (.operand xName zIn))
  let (cNp, nNp) ← bnSite B oc hh mode epsStr s!"%{p}gp" s!"%{p}btp" s!"{p}np" nCp
  let (cA,  nA)  ← pretty B (.addV (.operand nN3 zOut) (.operand nNp zOut))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zOut))
  pure (cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cCp ++ cNp ++ cA ++ cO, nO)

/-- Strided projection bottleneck forward. ⚠ `conv1`/`bn1`/`relu1` at `2hh`; only `conv2` strides. -/
private def bnkStridedFwdV (B cin mid oc hh : Nat) (mode : R34Bn) (epsStr p xName : String) :
    StateM Nat (String × String) := do
  let ww := hh
  let zm     : Vec mid := fun _ => 0
  let zo     : Vec oc := fun _ => 0
  let zk1    : Kernel4 mid cin 1 1 := fun _ _ _ _ => 0
  let zk2    : Kernel4 mid mid 3 3 := fun _ _ _ _ => 0
  let zk3    : Kernel4 oc mid 1 1 := fun _ _ _ _ => 0
  let zkp    : Kernel4 oc cin 1 1 := fun _ _ _ _ => 0
  let zIn    : Vec (cin*(2*hh)*(2*ww)) := fun _ => 0
  let zMidIn : Vec (mid*(2*hh)*(2*ww)) := fun _ => 0
  let zMid   : Vec (mid*hh*ww) := fun _ => 0
  let zOut   : Vec (oc*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvF (ic := cin) (oc := mid) (h := 2*hh) (w := 2*ww) s!"%{p}W1" (zb mid) zk1 zm (.operand xName zIn))
  let (cN1, nN1) ← bnSite B mid (2*hh) mode epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zMidIn))
  let (cC2, nC2) ← pretty B (.flatConvStridedF (ic := mid) (oc := mid) (h := hh) (w := ww) s!"%{p}W2" (zb mid) zk2 zm (.operand nR1 zMidIn))
  let (cN2, nN2) ← bnSite B mid hh mode epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cR2, nR2) ← pretty B (.reluF (.operand nN2 zMid))
  let (cC3, nC3) ← pretty B (.flatConvF (ic := mid) (oc := oc) (h := hh) (w := ww) s!"%{p}W3" (zb oc) zk3 zo (.operand nR2 zMid))
  let (cN3, nN3) ← bnSite B oc hh mode epsStr s!"%{p}g3" s!"%{p}bt3" s!"{p}n3" nC3
  let (cCp, nCp) ← pretty B (.flatConvStridedF (ic := cin) (oc := oc) (h := hh) (w := ww) s!"%{p}Wp" (zb oc) zkp zo (.operand xName zIn))
  let (cNp, nNp) ← bnSite B oc hh mode epsStr s!"%{p}gp" s!"%{p}btp" s!"{p}np" nCp
  let (cA,  nA)  ← pretty B (.addV (.operand nN3 zOut) (.operand nNp zOut))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zOut))
  pure (cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cR2 ++ cC3 ++ cN3 ++ cCp ++ cNp ++ cA ++ cO, nO)

set_option maxRecDepth 4000000 in
/-- The full R50 forward: stem → `[3,4,6,3]` bottlenecks → GAP(7×7) → dense(2048→nClasses).
    `mode` picks batch statistics (`.train`) or frozen running stats (`.eval`); ONE chain, so the
    two renders cannot describe different nets. -/
private def r50FwdChain (B nClasses : Nat) (mode : R34Bn) (epsStr : String)
    -- ⚠ TRAILING, and the SAME ladder the train step uses (`q = 7` is 224, `q = 5` is 160). It has
    -- to be the same derivation, not merely the same numbers: the §2g prefix audit only means
    -- anything if the forward and the train step are one chain at one resolution.
    (q : Nat := 7) : StateM Nat (String × String) := do
  let q5 := q; let q4 := 2 * q5; let q3 := 2 * q4; let q2 := 2 * q3; let q1 := 2 * q2
  let zx   : Vec (3*(2*q1)*(2*q1)) := fun _ => 0
  let zSk  : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
  let z64  : Vec 64 := fun _ => 0
  let z112 : Vec (64*q1*q1) := fun _ => 0
  let (cStc, nStc) ← pretty B (.flatConvStridedF (ic := 3) (oc := 64) (h := q1) (w := q1) "%sW" (zb 64) zSk z64 (.operand "%x" zx))
  let (cStn, nStn) ← bnSite B 64 q1 mode epsStr "%sg" "%sbt" "stn" nStc
  let (cStr, nStr) ← pretty B (.reluF (.operand nStn z112))
  let (cStp, nStp) ← pretty B (.maxPool3s2F (c := 64) (h := q2) (w := q2) (.operand nStr z112))
  let (c1, n1)   ← bnkProjFwdV    B   64  64  256 q2 mode epsStr "s1b0" nStp
  let (c2, n2)   ← bnkIdFwdV      B       64  256 q2 mode epsStr "s1b1" n1
  let (c3, n3)   ← bnkIdFwdV      B       64  256 q2 mode epsStr "s1b2" n2
  let (c4, n4)   ← bnkStridedFwdV B  256 128  512 q3 mode epsStr "s2b0" n3
  let (c5, n5)   ← bnkIdFwdV      B      128  512 q3 mode epsStr "s2b1" n4
  let (c6, n6)   ← bnkIdFwdV      B      128  512 q3 mode epsStr "s2b2" n5
  let (c7, n7)   ← bnkIdFwdV      B      128  512 q3 mode epsStr "s2b3" n6
  let (c8, n8)   ← bnkStridedFwdV B  512 256 1024 q4 mode epsStr "s3b0" n7
  let (c9, n9)   ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b1" n8
  let (c10, n10) ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b2" n9
  let (c11, n11) ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b3" n10
  let (c12, n12) ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b4" n11
  let (c13, n13) ← bnkIdFwdV      B      256 1024 q4 mode epsStr "s3b5" n12
  let (c14, n14) ← bnkStridedFwdV B 1024 512 2048  q5 mode epsStr "s4b0" n13
  let (c15, n15) ← bnkIdFwdV      B      512 2048  q5 mode epsStr "s4b1" n14
  let (c16, n16) ← bnkIdFwdV      B      512 2048  q5 mode epsStr "s4b2" n15
  let zL    : Vec (2048*q5*q5) := fun _ => 0
  let z2048 : Vec 2048 := fun _ => 0
  let zWd   : Mat 2048 nClasses := fun _ _ => 0
  let zNC   : Vec nClasses := fun _ => 0
  let (cGap, nGap) ← pretty B (.gapF (c := 2048) (h := q5) (w := q5) (.operand n16 zL))
  let (cLog, nLog) ← pretty B (denseF "%Wd" "%bd" zWd zNC (.operand nGap z2048))
  pure (cStc ++ cStn ++ cStr ++ cStp ++
        c1 ++ c2 ++ c3 ++ c4 ++ c5 ++ c6 ++ c7 ++ c8 ++
        c9 ++ c10 ++ c11 ++ c12 ++ c13 ++ c14 ++ c15 ++ c16 ++ cGap ++ cLog, nLog)

set_option maxRecDepth 4000000 in
/-- **`@resnet50in_fwd`** — 162 inputs (`%x` + 161 params), logits `[B, nClasses]`. Batch-statistic
    BN, i.e. the same forward the train step differentiates.

    ⚠⚠ **That sentence was FALSE until 2026-08-10 and is now enforced.** This built its forward from
    `r50FwdChain .train`, which reaches R34's shared `bnSite` and therefore `bnPerChannelF` — reduce
    `[2,3]`, divisor `H·W`, PER-EXAMPLE — while `resnet50TrainStepFaithfulB` is `bnBatchF`, reduce
    `[0,2,3]`, divisor `B·H·W`. Two different functions under one net's name (§3d(b)), and
    `regen_verified_mlir.sh check` could not see it because it only ever paired a forward with the
    SGD train step and R50 has none. It now renders from `r50FwdChainB` — literally the traversal
    the train step differentiates — so `check_adam_prefix` holds it as a byte prefix.

    ⚠ `r50FwdChain`'s `.train` branch is now UNUSED by R50 and must stay that way; it survives only
    because `.eval` shares the function. Rendering a forward from it reopens the split. -/
def resnet50FwdFaithfulV (B nClasses : Nat) (epsStr : String)
    (slug : String := "resnet50in") (q : Nat := 7) (vSuffix : String := "") : String :=
  let sigList := r50SigList nClasses
  let inSig := s!"%x: {ty [B, 3*(32*q)*(32*q)]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let fwr := (r50FwdChainB B nClasses epsStr q).run' 0
  let (code, logits) := (fwr.code, fwr.logits)
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd{vSuffix}({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── ResNet-50 forward: every line is pretty(verified AST node) ──\n" ++
  zeroBiasPrelude false [64, 128, 256, 512, 1024, 2048] ++ code ++
  s!"    return {logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

set_option maxRecDepth 4000000 in
/-- **`@resnet50in_fwd_eval`** — the inference forward, every BN site reading frozen running stats.
    161 params + 106 stat inputs + `%x` = **268 inputs**. This is what the driver scores through,
    so its BN order must match `r50StatSigList`, which it does by sharing the chain. -/
def resnet50FwdEvalFaithfulV (B nClasses : Nat) (epsStr : String)
    (slug : String := "resnet50in") (q : Nat := 7) (vSuffix : String := "") : String :=
  let sigList := r50SigList nClasses ++ r50StatSigList
  let inSig := s!"%x: {ty [B, 3*(32*q)*(32*q)]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let (code, logits) := (r50FwdChain B nClasses .eval epsStr q).run' 0
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd_eval{vSuffix}({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── ResNet-50 eval forward (running-stats BN): every line is pretty(verified AST node) ──\n" ++
  zeroBiasPrelude false [64, 128, 256, 512, 1024, 2048] ++ code ++
  s!"    return {logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

end Proofs.StableHLO

-- ⚠ Slug `resnet50in`, distinct from any 10-class R50, for the reason §2k records for `resnet34in`:
-- the forwards carry no variant in their path, so a shared slug would silently overwrite a
-- different-arity artifact.
#eval IO.FS.writeFile "verified_mlir/resnet50in_adam64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.adamw "resnet50in")

-- The 4-GPU data-parallel peer at `B := 64` PER REPLICA ⇒ global batch 256, matching R34's
-- ImageNet recipe and the reference's. One `#eval`: `optOne` already takes `replicas`.
#eval IO.FS.writeFile "verified_mlir/resnet50in_adamdp64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    Proofs.StableHLO.R34Opt.adamw "resnet50in")

-- ⭐ THE 2018 RECIPE, so R50 and R34 can be compared with ONLY the network swapped.
-- Heavy-ball momentum + coupled L2 at global batch 256, which is exactly
-- `resnet34in_mom256`'s configuration (`ResNet34RenderB.lean`) with the bottleneck backbone
-- substituted. Everything the two runs differ by is the architecture, which is what makes the
-- 2018-vs-A3 comparison in the book a recipe comparison rather than a confounded one.
--
-- ⚠ `mom256` is a SINGLE-device render at batch 256, matching R34's. The data-parallel peer is
-- `momdp64` (4 replicas x 64) if a 4-GPU run is wanted; both are the same global batch.
#eval IO.FS.writeFile "verified_mlir/resnet50in_mom256_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 256 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.heavyBall "resnet50in")

#eval IO.FS.writeFile "verified_mlir/resnet50in_momdp64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    Proofs.StableHLO.R34Opt.heavyBall "resnet50in")

-- ⭐⭐ GRADIENT ACCUMULATION — `planning/next_session_pipeline_then_r50.md` §4's blocker.
--
-- A FOURTH parameter region `G` and two runtime scalars, so one graph is both phases. `k` is in the
-- artifact name because the driver must agree with the `1/k` baked into `%ob1`/`%ob2`, and a
-- disagreement is silent (a wrong effective learning rate, no error anywhere).
--
-- ⚠ THE TWO ARE FOR DIFFERENT JOBS AND ONLY ONE IS THE RECIPE:
--   * `acc4x64`   — 1 replica, micro-batch 64, k = 4 ⇒ effective **256**. This is what
--     `lake build r50-accum-tie` gates, because the tie needs a SINGLE-DEVICE peer to compare
--     against and the DP render's `%loss` is replica-local (`tests/r50_dp_render_tie.py`).
--   * `accdp8x64` — 4 replicas, micro-batch 64, k = 8 ⇒ effective **2048**, which is RSB-A3's
--     design batch and LAMB's. ⚠ It is the batch, NOT the recipe: `planning/rsb_a3_r50_verified.md`
--     §2.3's LAMB and BCE-with-logits are still absent, so this is AdamW at bs2048 and must not be
--     described as `rsb-faithful`.
#eval IO.FS.writeFile "verified_mlir/resnet50in_acc4x64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.adamwAccum 4) "resnet50in")

-- ⚠ `accdp4x64` exists for ONE reason: it is `acc4x64`'s data-parallel peer at the SAME k, which is
-- what `tests/r50_dp_render_tie.py` needs to carry `r50-accum-tie`'s verdict onto the DP
-- accumulation path. Without a matched-k pair that tie has nothing to diff. It is not a recipe.
#eval IO.FS.writeFile "verified_mlir/resnet50in_accdp4x64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.adamwAccum 4) "resnet50in")

#eval IO.FS.writeFile "verified_mlir/resnet50in_accdp8x64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.adamwAccum 8) "resnet50in")

-- ⭐⭐ BCE-WITH-LOGITS — RSB-A2/A3's loss (`planning/next_session_pipeline_then_r50.md` §4).
--
-- Same three regions and the same signature as the CE renders: the loss is not state, so nothing
-- in the driver moves. ⚠ `adam64bce` exists so `lake build r50-bce-tie` can recover the cotangent
-- from AdamW's `m' = 0.1·g` (LAMB's trust ratio is in the way); `lamb64bce` is the recipe-facing
-- pair. Both share ONE `lossCodeBce`, so a defect in the loss cannot be present in one and not the
-- other.
#eval IO.FS.writeFile "verified_mlir/resnet50in_adam64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.adamw "resnet50in" (bce := true) (vSuffix := "bce"))

#eval IO.FS.writeFile "verified_mlir/resnet50in_lamb64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.lamb "resnet50in" (bce := true) (vSuffix := "bce"))

-- ⭐⭐ LAMB — RSB-A3's optimizer (`planning/rsb_a3_r50_verified.md` §2.3). THREE regions, same
-- `[θ|m|v]` signature as `adam64`, because the trust ratio is computed inside the graph from θ and
-- the direction and needs no extra state. So the driver is byte-identical across `adam64` and
-- `lamb64` — the only per-net fact is which file it opens.
--
-- ⚠⚠ THIS IS LAMB, NOT `rsb-faithful`. That recipe is LAMB **at effective batch 2048** with
-- BCE-with-logits and a 160/224 resolution split; `planning/rsb_a2_resnet50.md` records LAMB at
-- bs512 giving 40.8% against 78.1%, so the batch is not a detail. Composing this with the
-- accumulation render (`.lambAccum`) is not built — see the handoff.
#eval IO.FS.writeFile "verified_mlir/resnet50in_lamb64_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.lamb "resnet50in")

-- Pin the literal artifact paths above against the name the renderer emits, so a rename fails at
-- `lake build` rather than at run time as a shim "entry mismatch".
#guard Proofs.StableHLO.r34AdamVariant 64 1 == "adam64"
#guard Proofs.StableHLO.r34AdamVariant 64 4 == "adamdp64"
#guard Proofs.StableHLO.r34AdamVariant 64 1 Proofs.StableHLO.R34Opt.lamb == "lamb64"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.adamwAccum 4) == "acc4x64"
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.adamwAccum 8) == "accdp8x64"
-- ⚠ The driver reads `k` back OUT of this string (`VerifiedTrain`'s `accK`). Pin the round trip
-- here, where the name is produced, rather than trusting two parsers to agree.
#guard ((Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.adamwAccum 8)).drop 5
          |>.takeWhile (· != 'x')) == "8"

-- Batch 256 on the forwards, matching the shim's val batch: 195 × 256 = 49,920 after tfds
-- drop_remainder, the count the reference reports scoring. ⚠ `evalBs` in the driver is read OFF
-- the artifact, so this need not equal the train batch.
-- ⭐⭐ RSB-A3's RESOLUTION SPLIT — train @160, eval @224 (`next_session_pipeline_then_r50.md` §4).
--
-- Slug `resnet50in160`, so the resolution is in the name the driver opens rather than in a suffix
-- some call site has to remember. ⚠ THE EVAL FORWARD IS DELIBERATELY AT A DIFFERENT RESOLUTION
-- FROM ITS OWN TRAIN STEP — that is the recipe, not an oversight: `q = 5` (160) for training,
-- `q = 7` (224) for scoring, one renderer, two instantiations.
--
-- ⚠ These are the RENDER half. They have no data path yet: `resnet50ImagenetVerified` is a 224 net
-- (`d0 = 3·224·224`) and the tfds shim produces 224 crops, so a 160 `VerifiedNetSpec` and a 160
-- shim are still owed before any of this can run. Rendering them now is what type-checks the
-- resolution parameterisation at `q = 5` — the part with the dependent-type risk.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lamb64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.lamb "resnet50in160" (bce := true) (vSuffix := "bce") (q := 5))

-- ═══════════════════════════════════════════════════════════════════════════════════════════════
-- ⭐⭐⭐ **RSB-A3 ITSELF — LAMB × BCE × ACCUMULATION × 4 REPLICAS, AT 160.** The composition
-- `planning/next_session_rsb_a3.md` was written to reach, rendered 2026-08-06.
--
-- `lambaccdp8x64bce` = LAMB, BCE-with-logits, k = 8 accumulated micro-batches, 4 replicas × bs64
-- ⇒ effective batch **2048** — RSB-A3's design batch and the one LAMB was built for
-- (`planning/rsb_a2_resnet50.md` records LAMB at bs512 giving **40.8% against 78.1%**, so the batch
-- is not a detail). At `q = 5`, i.e. A3's 160² train resolution, scoring through
-- `resnet50in160_fwd_eval` at 224².
--
-- ⚠⚠ **WHAT THIS STILL IS NOT.** `wdExcludeNormBias` remains absent (§0), so BN γ/β and biases are
-- decayed where timm excludes them; and the BN regime is Ghost-BN over 64-image micro-batches, not
-- a genuine bs2048 forward. Quote it against §4.1's delta list, never as "RSB-A3 reproduced".
--
-- ⚠ The 1-replica peer below is NOT a second recipe — it is what the accumulation gates need. Both
-- `r50-accum-tie` and `r50-accum-shard-tie` compare against a SINGLE-DEVICE peer, so a DP-only
-- render would be ungateable. Same k, same loss, same resolution; only `replicas` differs.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambaccdp8x64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in160" (bce := true) (vSuffix := "bce") (q := 5))

#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambacc8x64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 8) "resnet50in160" (bce := true) (vSuffix := "bce") (q := 5))

-- ⚠ `k = 4` at ONE replica, the peer `r50-accum-tie` actually runs against (its gate is written at
-- k = 4). Rendering only k = 8 would leave the composed optimizer's accumulation ungated at the k
-- the gate uses.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambacc4x64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 1
    (Proofs.StableHLO.R34Opt.lambAccum 4) "resnet50in160" (bce := true) (vSuffix := "bce") (q := 5))

-- ⚠ `lambdp64bce` — LAMB + BCE at 4 replicas, NO accumulation. It exists for ONE reason: it is the
-- peer `r50-accum-shard-tie` needs. That gate's identity is `acc(x1..xk) == DP([x1|..|xk])`, so the
-- right-hand side must be the DATA-PARALLEL render of the SAME optimizer and loss — `adamdp64` would
-- compare LAMB to AdamW and BCE to CE and fail for three reasons with one number to read.
-- ▶ §3's "needs a `lambdp64` peer rendered", at the composition's resolution.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_lambdp64bce_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 64 1000 "1.0e-05" 4
    Proofs.StableHLO.R34Opt.lamb "resnet50in160" (bce := true) (vSuffix := "bce") (q := 5))

-- ⚠⚠ THE NAME ROUND TRIP FOR THE COMPOSED VARIANT, pinned on the PRODUCING side — the driver reads
-- `k` back out of this exact string with a substring parse, and a disagreement is silent (a wrong
-- effective learning rate, no error anywhere). `tests/TestVariantPredicates.lean` pins the
-- consuming side, including the counterfactual that the OLD `startsWith "acc"` test MISSED this.
#guard Proofs.StableHLO.r34AdamVariant 64 4 (Proofs.StableHLO.R34Opt.lambAccum 8) == "lambaccdp8x64"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 8) == "lambacc8x64"
#guard Proofs.StableHLO.r34AdamVariant 64 1 (Proofs.StableHLO.R34Opt.lambAccum 4) == "lambacc4x64"
-- ⚠ and the marker is NOT leading here, which is the whole defect: pin that fact so the spelling
-- cannot drift back to something the prefix test would have accepted by accident.
#guard ("lambaccdp8x64bce".startsWith "acc") == false
#guard (("lambaccdp8x64bce".splitOn "acc").length > 1) == true

#eval IO.FS.writeFile "verified_mlir/resnet50in160_fwd.mlir"
  (Proofs.StableHLO.resnet50FwdFaithfulV 64 1000 "1.0e-05" "resnet50in160" (q := 5))

-- ⭐ eval at 224 (`q = 7`) and batch 256, exactly as the 224 net scores. This is A3's
-- `train@160 / test@224 (crop 0.95)` split, and the only thing that makes it expressible is that
-- BN in `.eval` mode reads FROZEN per-channel statistics — which are resolution-independent.
#eval IO.FS.writeFile "verified_mlir/resnet50in160_fwd_eval.mlir"
  (Proofs.StableHLO.resnet50FwdEvalFaithfulV 256 1000 "1.0e-05" "resnet50in160" (q := 7))

#eval IO.FS.writeFile "verified_mlir/resnet50in_fwd.mlir"
  (Proofs.StableHLO.resnet50FwdFaithfulV 256 1000 "1.0e-05" "resnet50in")

#eval IO.FS.writeFile "verified_mlir/resnet50in_fwd_eval.mlir"
  (Proofs.StableHLO.resnet50FwdEvalFaithfulV 256 1000 "1.0e-05" "resnet50in")

-- ═══════════════════════════════════════════════════════════════════════════
-- § `resnet50Verified` — the IMAGENETTE (10-class) renders.
--
-- `resnet50Verified` has existed as a layout skeleton since the R50 scoping work
-- ("no render, no proof chain, no artifact yet"); these are the artifacts. Same
-- renderer, same 224² backbone as `resnet50in` — the ONLY deltas are `nClasses`
-- 1000 → 10 and the slug, which is why this is three lines and not a new chain.
--
-- ⭐ B = 32 makes `r34AdamVariant` emit the bare name `"adam"` (the `if B == 32
-- then ""` branch), so the artifact, the `@resnet50_adam_train_step` entry point
-- and the driver's default `LEAN_MLIR_VARIANT=adam` are one string — matching
-- R34's Imagenette pairing exactly rather than inventing a second convention.
--
-- ⚠ The forwards are rendered at B = 32 too, because the driver scores eval
-- through them at the train batch. Any other batch needs re-rendered forwards or
-- `LEAN_MLIR_SKIP_EVAL=1`, and would otherwise be a shape error at first invoke.
#eval IO.FS.writeFile "verified_mlir/resnet50_adam_train_step.mlir"
  (Proofs.StableHLO.resnet50TrainStepFaithfulB 32 10 "1.0e-05" 1
    Proofs.StableHLO.R34Opt.adamw "resnet50")

#eval IO.FS.writeFile "verified_mlir/resnet50_fwd.mlir"
  (Proofs.StableHLO.resnet50FwdFaithfulV 32 10 "1.0e-05" "resnet50")

#eval IO.FS.writeFile "verified_mlir/resnet50_fwd_eval.mlir"
  (Proofs.StableHLO.resnet50FwdEvalFaithfulV 32 10 "1.0e-05" "resnet50")
