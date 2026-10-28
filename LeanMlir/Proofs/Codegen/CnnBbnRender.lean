module

public import LeanMlir.Proofs.Codegen.CnnRender
public import LeanMlir.Proofs.Codegen.SyncBnSites
meta import LeanMlir.Proofs.Codegen.CnnRender -- shake: keep
meta import LeanMlir.Proofs.Codegen.SyncBnSites -- shake: keep

@[expose] public section

/-! # The cifar8 BN net at BATCH statistics — `cifar8wb_bbn`

The eight-conv CIFAR net with per-channel BatchNorm, rendered on the batched op family with every
BN site reducing over `[0,2,3]` — Ioffe & Szegedy's layer, the one Chapter 5's ResNet-34 trains —
instead of the per-example `[2,3]` the `cifar8w_bn` / `cifar8wb_bn` artifacts normalise by. Same
38 parameters, same layer list; what moves is the statistic each BN divides by, and with it
everything a batch-statistics layer needs that a per-image one does not:

* **the train step hands back each BN's batch μ/var** (`bnBatchMeanB` / `bnBatchVarB`, 16
  `[c]` outputs after `%loss`/`%bc1`/`%bc2`) and takes 16 passthrough stat inputs before
  `%onehot`, exactly the `[θ|m|v]` + stats protocol `ResNet34RenderB` uses and
  `VerifiedNet.trainAdamSched` drives (`bnChannels` non-empty ⇒ EMA the returned stats);
* **`@cifar8wb_bbn_fwd_eval`** — the inference forward with frozen running statistics at every BN
  site (`bnEvalSite` → `bnPerChannelEvalF`), the artifact the driver scores through;
* **`@cifar8wb_bbn_fwd`** — the batch-BN training forward, emitted from the SAME chain the train
  step differentiates (`cifar8BbnFwdChainB`), so it is a byte prefix of the train step and the
  `LEAN_MLIR_EVAL_BATCHSTATS` diagnostic reads the net that trains.

Three BN emit sites — forward, input-cotangent, γ-gradient — are the shared
[`SyncBnSites`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Codegen/SyncBnSites.lean)
ones every batch-BN renderer uses, so at `replicas > 1` this render uses the shared sync sites; no
sync tie is stated for this net, and no data-parallel artifact is written.

Faithful by construction like every render here: the AST is built only from verified constructors,
so `pretty(provenGraph)` needs no new proof. Hand-written text: the signatures, the constants, the
report-only `%loss` and the `%bc1`/`%bc2` passthroughs — none of it feeds a parameter or a statistic.
The per-example ties (`Cifar8BnStepTie`, `Cifar8BnStepTieG`) are stated at `bnPerChannel*`; the
single-replica batch-statistics step tie at these nodes is
`Proofs.Cifar8BbnTieB.cifar8Bbn_train_step_tiedB` (proved, with `Cifar8BbnWholeBackB`). Not
stated: the `replicas > 1` sync-BN tie and the whole-net forward graph tie. The convolutions,
pools, dense head and optimizer tails are byte-for-byte the nodes `cifar8BnTrainStepBText`
emits — only the BN sites and the stat plumbing differ — so bf16 reaches the same 23
convolutions it reaches there, and BN stays f32.

The `#eval` writers are in `CnnArtifacts.lean`.
-/

namespace Proofs.StableHLO

open Proofs

/-- The 38-parameter signature of the cifar8 BN net, `cifar8BnVerified.toSpecs` order:
    (conv W, conv b, BN γ, BN β) ×8, then the three dense (W, b). ONE source for the order and the
    shapes — the arg signature, the packed `m`/`v` blocks and the return types all read it. -/
def cifar8BnSigList (ic c1 c2 c3 c4 flat d1 nClasses kH kW : Nat) : List (String × List Nat) :=
  [("W1",[c1,ic,kH,kW]), ("cb1",[c1]), ("g1",[c1]), ("bt1",[c1]),
   ("W2",[c1,c1,kH,kW]), ("cb2",[c1]), ("g2",[c1]), ("bt2",[c1]),
   ("W3",[c2,c1,kH,kW]), ("cb3",[c2]), ("g3",[c2]), ("bt3",[c2]),
   ("W4",[c2,c2,kH,kW]), ("cb4",[c2]), ("g4",[c2]), ("bt4",[c2]),
   ("W5",[c3,c2,kH,kW]), ("cb5",[c3]), ("g5",[c3]), ("bt5",[c3]),
   ("W6",[c3,c3,kH,kW]), ("cb6",[c3]), ("g6",[c3]), ("bt6",[c3]),
   ("W7",[c4,c3,kH,kW]), ("cb7",[c4]), ("g7",[c4]), ("bt7",[c4]),
   ("W8",[c4,c4,kH,kW]), ("cb8",[c4]), ("g8",[c4]), ("bt8",[c4]),
   ("W9",[flat,d1]), ("b9",[d1]), ("Wa",[d1,d1]), ("ba",[d1]),
   ("Wb",[d1,nClasses]), ("bb",[nClasses])]

/-- **The 16 running-stat slots** — 8 BN layers × (μ, var), each `[c]`, in BN-forward order
    `n1 … n8`, μ before var per layer (`bnStatSlots`). This is exactly the order
    `cifar8wbBbnVerified.bnChannels` lists (`#[16,16,16,16,32,32,32,32]`), which is how the driver
    packs `runningBnStats`. The train step takes them as passthrough inputs (`%n1mui`, …) and returns
    the batch statistics in the same order; `@cifar8wb_bbn_fwd_eval` reads them as `%n1mu`, …. -/
def cifar8BbnStatSigList (c1 c2 c3 c4 : Nat) : List (String × List Nat) :=
  bnStatSlots "n1" c1 ++ bnStatSlots "n2" c1 ++ bnStatSlots "n3" c2 ++ bnStatSlots "n4" c2 ++
  bnStatSlots "n5" c3 ++ bnStatSlots "n6" c3 ++ bnStatSlots "n7" c4 ++ bnStatSlots "n8" c4

-- 8 BN layers ⇒ 16 stat slots.
#guard (cifar8BbnStatSigList 16 16 32 32).length == 16

/-- One conv→BN→relu layer's saved SSA names: conv out (= BN input), BN out, the BN's packed sync
    statistics (`""` at one replica), relu out. -/
structure C8BbnLayer where
  code : String
  hc : String
  bn : String
  st : String
  ac : String

/-- **One conv→BN→relu layer at batch statistics**, batched op family: `batchOp (convAt bf16 …)`
    (the bf16 twin when `bf16`), then `bnFwdSite` (`bnBatchF` at one replica, the sync-BN
    composition above it), then `batchOp relu`. Parameters `%W{i}`/`%cb{i}`/`%g{i}`/`%bt{i}`;
    the sync-BN collectives are tagged `n{i}`, which is no parameter's name. -/
def c8bbnConvBnRelu (B ic oc hh ww kH kW : Nat) (bf16 sync : Bool) (replicas : Nat)
    (epsStr : String) (i : Nat) (xIn : String) : StateM EmitS C8BbnLayer := do
  let zx : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zW : Kernel4 oc ic kH kW := fun _ _ _ _ => 0
  let zb : Vec oc := fun _ => 0
  let zy : Vec (B*(oc*hh*ww)) := fun _ => 0
  let (cH, nH) ← pretty B (.batchOp (N := B)
    (.convAt bf16 (h := hh) (w := ww) zrnd s!"%W{i}" s!"%cb{i}" zW zb) (.operand xIn zx))
  let (cBn, nBn, st) ← bnFwdSite B oc hh ww sync replicas epsStr s!"%g{i}" s!"%bt{i}" s!"n{i}" nH
  let (cA, nA) ← pretty B (.batchOp (N := B) .relu (.operand nBn zy))
  pure ⟨cH ++ cBn ++ cA, nH, nBn, st, nA⟩

/-- **One conv→BN→relu layer at FROZEN running statistics** — `c8bbnConvBnRelu`'s eval twin, the
    per-example op family: `flatConvF`, then `bnEvalSite` (`bnPerChannelEvalF` at the `%n{i}mu` /
    `%n{i}var` graph inputs), then `reluF`. Same parameter names `%W{i}`/`%cb{i}`/`%g{i}`/`%bt{i}`,
    so the naming scheme and the shape-to-zero-operand pairing are written once per op family.
    Returns `(code, relu output)`. -/
def c8bbnEvalConvBnRelu (B ic oc hh ww kH kW : Nat) (epsStr : String) (i : Nat) (xIn : String) :
    StateM EmitS (String × String) := do
  let zx : Vec (ic*hh*ww) := fun _ => 0
  let zW : Kernel4 oc ic kH kW := fun _ _ _ _ => 0
  let zb : Vec oc := fun _ => 0
  let zy : Vec (oc*hh*ww) := fun _ => 0
  let (cH, nH) ← pretty B (.flatConvF (h := hh) (w := ww) s!"%W{i}" s!"%cb{i}" zW zb (.operand xIn zx))
  let (cBn, nBn) ← bnEvalSite B oc hh ww epsStr s!"%g{i}" s!"%bt{i}" s!"n{i}" nH
  let (cA, nA) ← pretty B (.reluF (.operand nBn zy))
  pure (cH ++ cBn ++ cA, nA)

/-- Saved forward SSA names the train step's backward and statistics read. -/
structure C8BbnFwdB where
  code : String
  l1 : C8BbnLayer
  l2 : C8BbnLayer
  l3 : C8BbnLayer
  l4 : C8BbnLayer
  l5 : C8BbnLayer
  l6 : C8BbnLayer
  l7 : C8BbnLayer
  l8 : C8BbnLayer
  pool1 : String
  pool2 : String
  pool3 : String
  pool4 : String
  h9 : String   -- dense 1 pre-activation
  a9 : String   -- its relu
  ha : String   -- dense 2 pre-activation
  aa : String   -- its relu
  log : String  -- logits

/-- **The batch-BN training forward, batched op family, at `N := B`** — the ONE chain both
    `@cifar8wb_bbn_fwd` and every `cifar8wb_bbn_*_train_step` emit, so the forward artifact is a
    byte prefix of the train step by construction. (conv→BN→relu)×2→pool, ×4, then
    (dense→relu)×2→dense. `h`,`w` are the final pooled spatials; stage spatials double from `s4 = 2h` back up to `s1` (the input). -/
def cifar8BbnFwdChainB (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat) (epsStr : String)
    (bf16 : Bool := false) (replicas : Nat := 1) : StateM EmitS C8BbnFwdB := do
  let sync := replicas > 1
  let s4h := 2*h; let s4w := 2*w
  let s3h := 2*s4h; let s3w := 2*s4w
  let s2h := 2*s3h; let s2w := 2*s3w
  let s1h := 2*s2h; let s1w := 2*s2w
  let bS1c1 : Vec (B*(c1*s1h*s1w)) := fun _ => 0
  let bS2c2 : Vec (B*(c2*s2h*s2w)) := fun _ => 0
  let bS3c3 : Vec (B*(c3*s3h*s3w)) := fun _ => 0
  let bS4c4 : Vec (B*(c4*s4h*s4w)) := fun _ => 0
  let bPc4  : Vec (B*(c4*h*w)) := fun _ => 0
  let bD1   : Vec (B*d1) := fun _ => 0
  let zW9 : Mat (c4*h*w) d1 := fun _ _ => 0
  let zWa : Mat d1 d1 := fun _ _ => 0
  let zWb : Mat d1 nClasses := fun _ _ => 0
  let zd1 : Vec d1 := fun _ => 0
  let znc : Vec nClasses := fun _ => 0
  -- ═══ (conv→BN→relu)×2 → pool, ×4 ═══
  let L1 ← c8bbnConvBnRelu B ic c1 s1h s1w kH kW bf16 sync replicas epsStr 1 "%x"
  let L2 ← c8bbnConvBnRelu B c1 c1 s1h s1w kH kW bf16 sync replicas epsStr 2 L1.ac
  let (cP1, nPool1) ← pretty B (.batchOp (N := B) (.maxPool (c := c1) (h := s2h) (w := s2w)) (.operand L2.ac bS1c1))
  let L3 ← c8bbnConvBnRelu B c1 c2 s2h s2w kH kW bf16 sync replicas epsStr 3 nPool1
  let L4 ← c8bbnConvBnRelu B c2 c2 s2h s2w kH kW bf16 sync replicas epsStr 4 L3.ac
  let (cP2, nPool2) ← pretty B (.batchOp (N := B) (.maxPool (c := c2) (h := s3h) (w := s3w)) (.operand L4.ac bS2c2))
  let L5 ← c8bbnConvBnRelu B c2 c3 s3h s3w kH kW bf16 sync replicas epsStr 5 nPool2
  let L6 ← c8bbnConvBnRelu B c3 c3 s3h s3w kH kW bf16 sync replicas epsStr 6 L5.ac
  let (cP3, nPool3) ← pretty B (.batchOp (N := B) (.maxPool (c := c3) (h := s4h) (w := s4w)) (.operand L6.ac bS3c3))
  let L7 ← c8bbnConvBnRelu B c3 c4 s4h s4w kH kW bf16 sync replicas epsStr 7 nPool3
  let L8 ← c8bbnConvBnRelu B c4 c4 s4h s4w kH kW bf16 sync replicas epsStr 8 L7.ac
  let (cP4, nPool4) ← pretty B (.batchOp (N := B) (.maxPool (c := c4) (h := h) (w := w)) (.operand L8.ac bS4c4))
  -- ═══ (dense→relu)×2 → dense ═══
  let (cH9, nH9) ← pretty B (.batchOp (N := B) (.dense "%W9" "%b9" zW9 zd1) (.operand nPool4 bPc4))
  let (cA9, nA9) ← pretty B (.batchOp (N := B) .relu (.operand nH9 bD1))
  let (cHa, nHa) ← pretty B (.batchOp (N := B) (.dense "%Wa" "%ba" zWa zd1) (.operand nA9 bD1))
  let (cAa, nAa) ← pretty B (.batchOp (N := B) .relu (.operand nHa bD1))
  let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wb" "%bb" zWb znc) (.operand nAa bD1))
  pure { code := L1.code ++ L2.code ++ cP1 ++ L3.code ++ L4.code ++ cP2 ++
                 L5.code ++ L6.code ++ cP3 ++ L7.code ++ L8.code ++ cP4 ++
                 cH9 ++ cA9 ++ cHa ++ cAa ++ cLog
         l1 := L1, l2 := L2, l3 := L3, l4 := L4, l5 := L5, l6 := L6, l7 := L7, l8 := L8
         pool1 := nPool1, pool2 := nPool2, pool3 := nPool3, pool4 := nPool4
         h9 := nH9, a9 := nA9, ha := nHa, aa := nAa, log := nLog }

/-- The parameter block of a signature: `%{nm}{suf}: {ty}` for each entry of `cifar8BnSigList`. -/
def c8bbnArgBlk (sig : List (String × List Nat)) (suf : String) : String :=
  String.intercalate ", " (sig.map (fun (nm, ds) => s!"%{nm}{suf}: {ty ds}"))

/-- **`@<slug>_fwd` — the batch-BN TRAINING forward** (BN over the batch's own statistics),
    `cifar8BbnFwdChainB` and a `return`. Not what the driver scores through — that is
    `@<slug>_fwd_eval` — but what `LEAN_MLIR_EVAL_BATCHSTATS` reads and what the BN-world check
    in `trainAdamSched` compares against the train step (`dimensions = [0, 2, 3]` in both). A byte
    prefix of every `<slug>_*_train_step` body from `%v0` on. -/
def cifar8BbnFwdText (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat) (epsStr slug : String)
    (bf16 : Bool := false) : String :=
  let flat := c4*h*w
  let sig := cifar8BnSigList ic c1 c2 c3 c4 flat d1 nClasses kH kW
  let F : C8BbnFwdB := (cifar8BbnFwdChainB B ic c1 c2 c3 c4 h w d1 nClasses kH kW epsStr bf16).run' (0, [])
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd(%x: {ty [B,ic*(2*(2*(2*(2*h))))*(2*(2*(2*(2*w))))]}, {c8bbnArgBlk sig ""}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── cifar8-BN batch-statistics forward (BATCHED op family, BN over [0,2,3]): every op is pretty(verified AST node) ──\n" ++
  F.code ++
  s!"    return {F.log} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

/-- **`@<slug>_fwd_eval` — the inference forward with FROZEN running statistics.** The
    per-example chain (`flatConvF` / `maxPoolF` / `denseF` / `reluF` under `pretty B`) with every
    BN site `bnEvalSite` (`bnPerChannelEvalF`: affine BN at the `%n{i}mu`/`%n{i}var` graph inputs,
    no reduction), so this forward is class-batch-independent. Inputs: `%x`, the 38 parameters,
    then the 16 stat slots of `cifar8BbnStatSigList` — the layout `trainAdamSched` packs
    (`paramShapes ++ bnStatShapes`). Returns logits `[B, nClasses]`. f32 only: BN is f32 in every
    arm, and the bf16 train steps score through this same artifact, as the ImageNet nets do. -/
def cifar8BbnFwdEvalText (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat) (epsStr slug : String) :
    String :=
  let s4h := 2*h; let s4w := 2*w
  let s3h := 2*s4h; let s3w := 2*s4w
  let s2h := 2*s3h; let s2w := 2*s3w
  let s1h := 2*s2h; let s1w := 2*s2w
  let flat := c4*h*w
  let sig := cifar8BnSigList ic c1 c2 c3 c4 flat d1 nClasses kH kW
  let statSig := cifar8BbnStatSigList c1 c2 c3 c4
  let zS1c1 : Vec (c1*s1h*s1w) := fun _ => 0
  let zS2c2 : Vec (c2*s2h*s2w) := fun _ => 0
  let zS3c3 : Vec (c3*s3h*s3w) := fun _ => 0
  let zS4c4 : Vec (c4*s4h*s4w) := fun _ => 0
  let zPc4 : Vec (c4*h*w) := fun _ => 0
  let zD1 : Vec d1 := fun _ => 0
  let zNC : Vec nClasses := fun _ => 0
  let zW9 : Mat (c4*h*w) d1 := fun _ _ => 0
  let zWa : Mat d1 d1 := fun _ _ => 0
  let zWb : Mat d1 nClasses := fun _ _ => 0
  let go : StateM EmitS String := do
    let (l1, a1) ← c8bbnEvalConvBnRelu B ic c1 s1h s1w kH kW epsStr 1 "%x"
    let (l2, a2) ← c8bbnEvalConvBnRelu B c1 c1 s1h s1w kH kW epsStr 2 a1
    let (cP1, nPool1) ← pretty B (.maxPoolF (c := c1) (h := s2h) (w := s2w) (.operand a2 zS1c1))
    let (l3, a3) ← c8bbnEvalConvBnRelu B c1 c2 s2h s2w kH kW epsStr 3 nPool1
    let (l4, a4) ← c8bbnEvalConvBnRelu B c2 c2 s2h s2w kH kW epsStr 4 a3
    let (cP2, nPool2) ← pretty B (.maxPoolF (c := c2) (h := s3h) (w := s3w) (.operand a4 zS2c2))
    let (l5, a5) ← c8bbnEvalConvBnRelu B c2 c3 s3h s3w kH kW epsStr 5 nPool2
    let (l6, a6) ← c8bbnEvalConvBnRelu B c3 c3 s3h s3w kH kW epsStr 6 a5
    let (cP3, nPool3) ← pretty B (.maxPoolF (c := c3) (h := s4h) (w := s4w) (.operand a6 zS3c3))
    let (l7, a7) ← c8bbnEvalConvBnRelu B c3 c4 s4h s4w kH kW epsStr 7 nPool3
    let (l8, a8) ← c8bbnEvalConvBnRelu B c4 c4 s4h s4w kH kW epsStr 8 a7
    let (cP4, nPool4) ← pretty B (.maxPoolF (c := c4) (h := h) (w := w) (.operand a8 zS4c4))
    let (cH9, nH9) ← pretty B (denseF "%W9" "%b9" zW9 zD1 (.operand nPool4 zPc4))
    let (cA9, nA9) ← pretty B (.reluF (.operand nH9 zD1))
    let (cHa, nHa) ← pretty B (denseF "%Wa" "%ba" zWa zD1 (.operand nA9 zD1))
    let (cAa, nAa) ← pretty B (.reluF (.operand nHa zD1))
    let (cLog, nLog) ← pretty B (denseF "%Wb" "%bb" zWb zNC (.operand nAa zD1))
    pure <|
      l1 ++ l2 ++ cP1 ++ l3 ++ l4 ++ cP2 ++ l5 ++ l6 ++ cP3 ++ l7 ++ l8 ++ cP4 ++
      cH9 ++ cA9 ++ cHa ++ cAa ++ cLog ++
      s!"    return {nLog} : {ty [B, nClasses]}\n"
  let inner : String := go.run' (0, [])
  let statArgs := String.intercalate ", " (statSig.map (fun (n, ds) => s!"{n}: {ty ds}"))
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd_eval(%x: {ty [B,ic*s1h*s1w]}, {c8bbnArgBlk sig ""}, {statArgs}) -> {ty [B, nClasses]} " ++ "{\n" ++
  "    // ── cifar8-BN eval forward (running-stats BN, frozen per-channel statistics): every op is pretty(verified AST node) ──\n" ++
  inner ++
  "  }\n}\n"

/-- One layer's backward at the BN: the relu mask on the BN output (`selectPosB`), then
    `bnBackSite` (`bnBatchBack` at one replica, the sync-BN backward above it) reading the layer's
    conv output and packed statistics. Returns `(code, dBn, dHc)`: the cotangent at the BN output
    (which the γ/β gradients read) and at the conv output (which the conv gradients read). -/
def c8bbnLayerBack (B oc hh ww : Nat) (sync : Bool) (replicas : Nat) (epsStr : String) (i : Nat)
    (L : C8BbnLayer) (dAc : String) : StateM EmitS (String × String × String) := do
  let zy : Vec (B*(oc*hh*ww)) := fun _ => 0
  let (cDbn, nDbn) ← pretty B (.selectPosB L.bn zy (.operand dAc zy))
  let (cDhc, nDhc) ← bnBackSite B oc hh ww sync replicas epsStr s!"%g{i}" L.hc s!"n{i}dst" nDbn L.st
  pure (cDbn ++ cDhc, nDbn, nDhc)

/-- One layer's four parameter gradients and optimizer tails, in `cifar8BnSigList` order
    (W, conv bias, γ, β): the batched conv weight gradient (bf16 twin when `bf16`) and bias
    gradient at the layer's INPUT `xIn`, `bnGammaSite` and `bnBetaGradB` at the BN-output cotangent.
    Returns `(code, θ' names, m' names, v' names)`. -/
def c8bbnLayerGrads (B ic oc hh ww kH kW : Nat) (bf16 sync : Bool) (replicas : Nat)
    (opt : CifarOpt) (epsStr : String) (i : Nat) (xIn : String) (L : C8BbnLayer)
    (nDbn nDhc : String) : StateM EmitS (String × List String × List String × List String) := do
  let zx : Vec (B*(ic*hh*ww)) := fun _ => 0
  let zW : Kernel4 oc ic kH kW := fun _ _ _ _ => 0
  let zb : Vec oc := fun _ => 0
  let zy : Vec (B*(oc*hh*ww)) := fun _ => 0
  let zbn : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let (gW, sW) ← pretty B ((SHlo.convWeightGradBAt bf16 (N := B) zrnd xIn zb zx zW) (.operand nDhc zy))
  let (aW, tW, mW, vW) ← optTail opt B replicas (oc*ic*kH*kW) s!"%W{i}" [oc,ic,kH,kW] sW
  let (gb, sb) ← pretty B (.convBiasGradB (N := B) zW zx zb (.operand nDhc zy))
  let (ab, tb, mb, vb) ← optTail opt B replicas oc s!"%cb{i}" [oc] sb
  let (gg, sg) ← bnGammaSite B oc hh ww sync epsStr L.hc nDbn L.st
  let (ag, tg, mg, vg) ← optTail opt B replicas oc s!"%g{i}" [oc] sg
  let (gt, stt) ← pretty B (.bnBetaGradB (N := B) (oc := oc) (h := hh) (w := ww) (.operand nDbn zbn))
  let (atc, tt, mt, vt) ← optTail opt B replicas oc s!"%bt{i}" [oc] stt
  pure (gW ++ aW ++ gb ++ ab ++ gg ++ ag ++ gt ++ atc,
        [tW, tb, tg, tt], [mW, mb, mg, mt], [vW, vb, vg, vt])

/-- One BN layer's returned batch statistics — μ then var, each `[oc]` — from that layer's BN
    INPUT (`bnBatchMeanB` / `bnBatchVarB`: the SAME `bnMean`/`bnVar` the forward normalised by).
    At `replicas > 1` read off the all-reduced packed vector (`bnStatsMeanB` / `bnStatsVarB`), so
    the host EMAs the GLOBAL batch statistics. -/
def c8bbnStat (B oc hh ww : Nat) (sync : Bool) (L : C8BbnLayer) :
    StateM EmitS (String × String × String) := do
  let zbn : Vec (B*(oc*(hh*ww))) := fun _ => 0
  let zst : Vec (oc+oc) := fun _ => 0
  if !sync then
    let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := ww) (.operand L.hc zbn))
    let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := ww) (.operand L.hc zbn))
    pure (cM ++ cV, nM, nV)
  else
    let (cM, nM) ← pretty B (.bnStatsMeanB (oc := oc) (.operand L.st zst))
    let (cV, nV) ← pretty B (.bnStatsVarB (oc := oc) (.operand L.st zst))
    pure (cM ++ cV, nM, nV)

set_option maxRecDepth 8000 in
/-- **`cifar8BbnTrainStepBText` — the cifar8 BN train step at BATCH statistics.**

    `cifar8BnTrainStepBText` with its eight `bnPerChannelF`/`bnPerChannelBack`/`bnGammaGrad` sites
    replaced by the batch-coupled `bnFwdSite`/`bnBackSite`/`bnGammaSite` (`bnBatchF` /
    `bnBatchBack` / `bnGammaGradB`, reduce over `[0,2,3]`) and `bnBetaGradB`, plus the
    running-statistics protocol: 16 passthrough stat inputs after `%bc2` and 16 batch-statistic
    outputs after `%bc2` (`cifar8BbnStatSigList` order). Signature otherwise the packed
    `[θ|m|v]` one every cifar8 optimizer render shares — `%x`, 38 θ, 38 m, 38 v, `%lr`/`%bc1`/`%bc2`,
    16 stats, `%onehot` → 38 θ', 38 m', 38 v', `%loss`/`%bc1`/`%bc2`, 16 stats — so
    `VerifiedNet.trainAdamSched` drives it unchanged, as it drives ResNet-34's.

    Three optimizers from one body (`optTail`); `bf16` routes the 23 convolutions through their
    bf16 twins and leaves BN in f32, as `cifar8BnTrainStepBText` and every ImageNet render do.
    Entry `@{slug}_{variant}_train_step`, the driver's convention. -/
def cifar8BbnTrainStepBText (B ic c1 c2 c3 c4 h w d1 nClasses kH kW : Nat)
    (bnEpsStr invBStr b1Str ob1Str b2Str ob2Str aEpsStr wdStr : String)
    (slug variant : String) (replicas : Nat := 1) (opt : CifarOpt := .adamw)
    (bf16 : Bool := false) : String :=
  let sync := replicas > 1
  let s4h := 2*h; let s4w := 2*w
  let s3h := 2*s4h; let s3w := 2*s4w
  let s2h := 2*s3h; let s2w := 2*s3w
  let s1h := 2*s2h; let s1w := 2*s2w
  let flat := c4*h*w
  let sig := cifar8BnSigList ic c1 c2 c3 c4 flat d1 nClasses kH kW
  let statSig := cifar8BbnStatSigList c1 c2 c3 c4
  let pTys : List String := sig.map (fun (_, ds) => ty ds)
  let statTys : List String := statSig.map (fun (_, ds) => ty ds)
  let bS1c1 : Vec (B*(c1*s1h*s1w)) := fun _ => 0
  let bS2c1 : Vec (B*(c1*s2h*s2w)) := fun _ => 0
  let bS2c2 : Vec (B*(c2*s2h*s2w)) := fun _ => 0
  let bS3c2 : Vec (B*(c2*s3h*s3w)) := fun _ => 0
  let bS3c3 : Vec (B*(c3*s3h*s3w)) := fun _ => 0
  let bS4c3 : Vec (B*(c3*s4h*s4w)) := fun _ => 0
  let bS4c4 : Vec (B*(c4*s4h*s4w)) := fun _ => 0
  let bPc4  : Vec (B*(c4*h*w)) := fun _ => 0
  let bD1   : Vec (B*d1) := fun _ => 0
  let bNC   : Vec (B*nClasses) := fun _ => 0
  -- `1 * n` is NOT defeq to `n` in Lean, so the `rows := 1` head ops (`softmaxRow`,
  -- `denseRowBack`) need operands declared at exactly their type. Confined to the head.
  let b1NC  : Vec (B*(1*nClasses)) := fun _ => 0
  let b1D1  : Vec (B*(1*d1)) := fun _ => 0
  let zW2 : Kernel4 c1 c1 kH kW := fun _ _ _ _ => 0
  let zW3 : Kernel4 c2 c1 kH kW := fun _ _ _ _ => 0
  let zW4 : Kernel4 c2 c2 kH kW := fun _ _ _ _ => 0
  let zW5 : Kernel4 c3 c2 kH kW := fun _ _ _ _ => 0
  let zW6 : Kernel4 c3 c3 kH kW := fun _ _ _ _ => 0
  let zW7 : Kernel4 c4 c3 kH kW := fun _ _ _ _ => 0
  let zW8 : Kernel4 c4 c4 kH kW := fun _ _ _ _ => 0
  let zc1 : Vec c1 := fun _ => 0
  let zc2 : Vec c2 := fun _ => 0
  let zc3 : Vec c3 := fun _ => 0
  let zc4 : Vec c4 := fun _ => 0
  let zW9 : Mat (c4*h*w) d1 := fun _ _ => 0
  let zWa : Mat d1 d1 := fun _ _ => 0
  let zWb : Mat d1 nClasses := fun _ _ => 0
  let go : StateM EmitS String := do
    -- ═══ forward: THE shared chain (`@<slug>_fwd` is this and a return) ═══
    let F ← cifar8BbnFwdChainB B ic c1 c2 c3 c4 h w d1 nClasses kH kW bnEpsStr bf16 replicas
    -- ═══ MEAN-loss cotangent: softmax split out so the report-only %loss can read it ═══
    let (cSm, nSm) ← pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses)) (.operand F.log b1NC))
    let (cD0, nD0) ← pretty B (.subB (.operand nSm bNC) (.operand "%onehot" bNC))
    let (cDy, nDy) ← pretty B (.scaleB invBStr 0 (.operand nD0 bNC))
    -- ═══ backward: dense head → (pool-back → relu-back → BN-back → conv-back) per stage ═══
    let (cDyA, nDyA) ← pretty B (.selectPosB F.ha b1D1 (.batchOp (N := B) (.denseRowBack (rows := 1) "%Wb" zWb) (.operand nDy b1NC)))
    let (cDy9, nDy9) ← pretty B (.selectPosB F.h9 b1D1 (.batchOp (N := B) (.denseRowBack (rows := 1) "%Wa" zWa) (.operand nDyA b1D1)))
    let (cDx9, nDx9) ← pretty B (.batchOp (N := B) (.denseRowBack (rows := 1) "%W9" zW9) (.operand nDy9 b1D1))
    -- stage 4
    let (cDac8, nDac8) ← pretty B (.maxPoolBackB (N := B) (c := c4) (h := h) (w := w) F.l8.ac bS4c4 (.operand nDx9 bPc4))
    let (cB8, nDbn8, nDhc8) ← c8bbnLayerBack B c4 s4h s4w sync replicas bnEpsStr 8 F.l8 nDac8
    let (cDac7, nDac7) ← pretty B ((SHlo.convBackBatchedAt bf16 (N := B) (h := s4h) (w := s4w) zrnd "%W8" zW8 zc4) (.operand nDhc8 bS4c4))
    let (cB7, nDbn7, nDhc7) ← c8bbnLayerBack B c4 s4h s4w sync replicas bnEpsStr 7 F.l7 nDac7
    let (cDpl3, nDpool3) ← pretty B ((SHlo.convBackBatchedAt bf16 (N := B) (h := s4h) (w := s4w) zrnd "%W7" zW7 zc4) (.operand nDhc7 bS4c4))
    -- stage 3
    let (cDac6, nDac6) ← pretty B (.maxPoolBackB (N := B) (c := c3) (h := s4h) (w := s4w) F.l6.ac bS3c3 (.operand nDpool3 bS4c3))
    let (cB6, nDbn6, nDhc6) ← c8bbnLayerBack B c3 s3h s3w sync replicas bnEpsStr 6 F.l6 nDac6
    let (cDac5, nDac5) ← pretty B ((SHlo.convBackBatchedAt bf16 (N := B) (h := s3h) (w := s3w) zrnd "%W6" zW6 zc3) (.operand nDhc6 bS3c3))
    let (cB5, nDbn5, nDhc5) ← c8bbnLayerBack B c3 s3h s3w sync replicas bnEpsStr 5 F.l5 nDac5
    let (cDpl2, nDpool2) ← pretty B ((SHlo.convBackBatchedAt bf16 (N := B) (h := s3h) (w := s3w) zrnd "%W5" zW5 zc3) (.operand nDhc5 bS3c3))
    -- stage 2
    let (cDac4, nDac4) ← pretty B (.maxPoolBackB (N := B) (c := c2) (h := s3h) (w := s3w) F.l4.ac bS2c2 (.operand nDpool2 bS3c2))
    let (cB4, nDbn4, nDhc4) ← c8bbnLayerBack B c2 s2h s2w sync replicas bnEpsStr 4 F.l4 nDac4
    let (cDac3, nDac3) ← pretty B ((SHlo.convBackBatchedAt bf16 (N := B) (h := s2h) (w := s2w) zrnd "%W4" zW4 zc2) (.operand nDhc4 bS2c2))
    let (cB3, nDbn3, nDhc3) ← c8bbnLayerBack B c2 s2h s2w sync replicas bnEpsStr 3 F.l3 nDac3
    let (cDpl1, nDpool1) ← pretty B ((SHlo.convBackBatchedAt bf16 (N := B) (h := s2h) (w := s2w) zrnd "%W3" zW3 zc2) (.operand nDhc3 bS2c2))
    -- stage 1
    let (cDac2, nDac2) ← pretty B (.maxPoolBackB (N := B) (c := c1) (h := s2h) (w := s2w) F.l2.ac bS1c1 (.operand nDpool1 bS2c1))
    let (cB2, nDbn2, nDhc2) ← c8bbnLayerBack B c1 s1h s1w sync replicas bnEpsStr 2 F.l2 nDac2
    let (cDac1, nDac1) ← pretty B ((SHlo.convBackBatchedAt bf16 (N := B) (h := s1h) (w := s1w) zrnd "%W2" zW2 zc1) (.operand nDhc2 bS1c1))
    let (cB1, nDbn1, nDhc1) ← c8bbnLayerBack B c1 s1h s1w sync replicas bnEpsStr 1 F.l1 nDac1
    -- ═══ per param: un-fused gradient, then the optimizer outputs. Order is `cifar8BnSigList`'s. ═══
    let (G1, t1, m1, v1) ← c8bbnLayerGrads B ic c1 s1h s1w kH kW bf16 sync replicas opt bnEpsStr 1 "%x"    F.l1 nDbn1 nDhc1
    let (G2, t2, m2, v2) ← c8bbnLayerGrads B c1 c1 s1h s1w kH kW bf16 sync replicas opt bnEpsStr 2 F.l1.ac F.l2 nDbn2 nDhc2
    let (G3, t3, m3, v3) ← c8bbnLayerGrads B c1 c2 s2h s2w kH kW bf16 sync replicas opt bnEpsStr 3 F.pool1 F.l3 nDbn3 nDhc3
    let (G4, t4, m4, v4) ← c8bbnLayerGrads B c2 c2 s2h s2w kH kW bf16 sync replicas opt bnEpsStr 4 F.l3.ac F.l4 nDbn4 nDhc4
    let (G5, t5, m5, v5) ← c8bbnLayerGrads B c2 c3 s3h s3w kH kW bf16 sync replicas opt bnEpsStr 5 F.pool2 F.l5 nDbn5 nDhc5
    let (G6, t6, m6, v6) ← c8bbnLayerGrads B c3 c3 s3h s3w kH kW bf16 sync replicas opt bnEpsStr 6 F.l5.ac F.l6 nDbn6 nDhc6
    let (G7, t7, m7, v7) ← c8bbnLayerGrads B c3 c4 s4h s4w kH kW bf16 sync replicas opt bnEpsStr 7 F.pool3 F.l7 nDbn7 nDhc7
    let (G8, t8, m8, v8) ← c8bbnLayerGrads B c4 c4 s4h s4w kH kW bf16 sync replicas opt bnEpsStr 8 F.l7.ac F.l8 nDbn8 nDhc8
    let (gW9, sW9) ← pretty B (.denseWeightGradB (N := B) F.pool4 bPc4 (.operand nDy9 bD1))
    let (aW9, tW9, mW9, vW9) ← optTail opt B replicas (flat*d1) "%W9" [flat,d1] sW9
    let (gb9, sb9) ← pretty B (.denseBiasGradB (N := B) (.operand nDy9 bD1))
    let (ab9, tb9, mb9, vb9) ← optTail opt B replicas d1 "%b9" [d1] sb9
    let (gWa, sWa) ← pretty B (.denseWeightGradB (N := B) F.a9 bD1 (.operand nDyA bD1))
    let (aWa, tWa, mWa, vWa) ← optTail opt B replicas (d1*d1) "%Wa" [d1,d1] sWa
    let (gba, sba) ← pretty B (.denseBiasGradB (N := B) (.operand nDyA bD1))
    let (aba, tba, mba, vba) ← optTail opt B replicas d1 "%ba" [d1] sba
    let (gWb, sWb) ← pretty B (.denseWeightGradB (N := B) F.aa bD1 (.operand nDy bNC))
    let (aWb, tWb, mWb, vWb) ← optTail opt B replicas (d1*nClasses) "%Wb" [d1,nClasses] sWb
    let (gbb, sbb) ← pretty B (.denseBiasGradB (N := B) (.operand nDy bNC))
    let (abb, tbb, mbb, vbb) ← optTail opt B replicas nClasses "%bb" [nClasses] sbb
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT ═══
    let (cS1, m1n, v1n) ← c8bbnStat B c1 s1h s1w sync F.l1
    let (cS2, m2n, v2n) ← c8bbnStat B c1 s1h s1w sync F.l2
    let (cS3, m3n, v3n) ← c8bbnStat B c2 s2h s2w sync F.l3
    let (cS4, m4n, v4n) ← c8bbnStat B c2 s2h s2w sync F.l4
    let (cS5, m5n, v5n) ← c8bbnStat B c3 s3h s3w sync F.l5
    let (cS6, m6n, v6n) ← c8bbnStat B c3 s3h s3w sync F.l6
    let (cS7, m7n, v7n) ← c8bbnStat B c4 s4h s4w sync F.l7
    let (cS8, m8n, v8n) ← c8bbnStat B c4 s4h s4w sync F.l8
    -- ═══ report-only scalar loss — OUTSIDE the proven surface, does not feed the update ═══
    let lossCode := reportCeLossOfSm B nClasses nSm
    let body := F.code ++ cSm ++ cD0 ++ cDy ++ lossCode ++
      cDyA ++ cDy9 ++ cDx9 ++
      cDac8 ++ cB8 ++ cDac7 ++ cB7 ++ cDpl3 ++
      cDac6 ++ cB6 ++ cDac5 ++ cB5 ++ cDpl2 ++
      cDac4 ++ cB4 ++ cDac3 ++ cB3 ++ cDpl1 ++
      cDac2 ++ cB2 ++ cDac1 ++ cB1 ++
      G1 ++ G2 ++ G3 ++ G4 ++ G5 ++ G6 ++ G7 ++ G8 ++
      gW9 ++ aW9 ++ gb9 ++ ab9 ++ gWa ++ aWa ++ gba ++ aba ++ gWb ++ aWb ++ gbb ++ abb ++
      cS1 ++ cS2 ++ cS3 ++ cS4 ++ cS5 ++ cS6 ++ cS7 ++ cS8
    let ths := t1 ++ t2 ++ t3 ++ t4 ++ t5 ++ t6 ++ t7 ++ t8 ++ [tW9, tb9, tWa, tba, tWb, tbb]
    let mns := m1 ++ m2 ++ m3 ++ m4 ++ m5 ++ m6 ++ m7 ++ m8 ++ [mW9, mb9, mWa, mba, mWb, mbb]
    let vns := v1 ++ v2 ++ v3 ++ v4 ++ v5 ++ v6 ++ v7 ++ v8 ++ [vW9, vb9, vWa, vba, vWb, vbb]
    let stats := [m1n, v1n, m2n, v2n, m3n, v3n, m4n, v4n, m5n, v5n, m6n, v6n, m7n, v7n, m8n, v8n]
    pure <|
      s!"    // ── cifar8-BN {opt.label} train step at BATCH statistics (BN over [0,2,3]), BATCHED op\n" ++
      "    //    family: every op is pretty(verified AST node) except the constants below, the marked\n" ++
      "    //    report-only loss and the %bc passthroughs; the 16 trailing outputs are each BN's\n" ++
      "    //    batch mean and variance, for the host's running-statistics EMA ──\n" ++
      -- NB the banner below says "proved as …", but neither name it is given exists: for this net
      -- the global-batch equivalence is not stated. No committed artifact carries it.
      (if sync then
        syncBnBanner "Cifar8BbnSyncTieB (not yet stated)" "cifar8BbnFwdGraphSync_shard (not yet stated)" "Small"
       else "") ++
      -- NB `%sc`/`%sa`/`%sb`/`%sd` are RESERVED: maxPoolBack's select_and_scatter emitter
      -- hardcodes them as region block arguments. Hence `%lzero`.
      "    %lzero = stablehlo.constant dense<0.0> : tensor<f32>\n" ++
      s!"    %b1 = stablehlo.constant dense<{b1Str}> : tensor<f32>\n" ++
      s!"    %ob1 = stablehlo.constant dense<{ob1Str}> : tensor<f32>\n" ++
      s!"    %b2 = stablehlo.constant dense<{b2Str}> : tensor<f32>\n" ++
      s!"    %ob2 = stablehlo.constant dense<{ob2Str}> : tensor<f32>\n" ++
      s!"    %eps = stablehlo.constant dense<{aEpsStr}> : tensor<f32>\n" ++
      s!"    %wd = stablehlo.constant dense<{wdStr}> : tensor<f32>\n" ++
      (if opt == .nesterov then "    %mu = stablehlo.constant dense<0.9> : tensor<f32>\n" else "") ++
      body ++
      s!"    return {String.intercalate ", " (ths ++ mns ++ vns ++ ["%loss", "%bc1", "%bc2"] ++ stats)} : " ++
      s!"{String.intercalate ", " (pTys ++ pTys ++ pTys ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statTys)}\n"
  let retTy := String.intercalate ", "
    (pTys ++ pTys ++ pTys ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statTys)
  let inner : String := go.run' (0, [])
  let fname := s!"{slug}_{variant}_train_step"
  let statIn := String.intercalate ", " (statSig.map (fun (n, ds) => s!"{n}i: {ty ds}"))
  let argSig := c8bbnArgBlk sig "" ++ ", " ++ c8bbnArgBlk sig "m" ++ ", " ++ c8bbnArgBlk sig "v" ++
                s!", %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, {statIn}, %onehot: {ty [B,nClasses]}"
  "module @m {\n" ++
  s!"  func.func @{fname}(%x: {ty [B,ic*s1h*s1w]}, {argSig}) -> ({retTy}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO
