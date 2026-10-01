import LeanMlir.Proofs.Codegen.SyncBnSites
import LeanMlir.Proofs.Codegen.RenderKit

/-! # ResNet-34 AdamW train step rendered from the verified AST, at the BATCHED index

**The sole writer of every ResNet-34 artifact.**

* **BatchNorm is `bnBatchF`** — μ/var reduced over `[0,2,3]`, coupling the batch — not
  `bnPerChannelF`'s per-example `[2,3]`. That is the semantics the AdamW trainer runs; a
  per-example-BN render would be a different function.
* **The whole graph sits at `N := B`**, so every batch-coupled `den` here is honest: `bnBatchF`,
  `bnBatchBack`, and the whole `*GradB` family reduce over the batch, and at `N = 1` they would
  each describe a one-example function while the emitted text reduces over all `B`.

The optimizer is the proven `adamMNextF`/`adamVNextF`/`adamWParamF` triple applied to the un-fused
`*GradB` gradients. β₁/β₂/ε/wd are baked; `%lr`/`%bc1`/`%bc2` arrive as runtime `tensor<f32>` args.

**The cotangent is composed from kit ops, not fused**:
`softmaxRow → subB → scaleB → addVB → shiftB → divConstB` (label smoothing α = 0.1, K =
`nClasses`), every line `pretty` of a verified node. `%loss` is report-only and stays outside the
AST, exactly as `cifar8_adam_train_step`'s does.

Render is value-independent (`skel` erases values), so placeholder zeros and `lr := 0`/`ε := 0` are
passed; the emitted literals carry the real values.
-/

open Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The per-example EVAL forward (`@resnet34_fwd_eval`) and the signature lists
--
-- What the inference forward needs, plus the signature lists other files read (`r34SigList` is the
-- single source for the arg order of every r34 artifact). Every BN site is `RenderKit.bnEvalSite`
-- (frozen running statistics); `ResNet50RenderB`'s eval chain uses the same site.
--
-- The eval forward does NOT move onto the batched chain, and could not meaningfully: frozen
-- per-channel statistics reduce nothing, so `bnPerChannelEvalF` is BatchNorm-world-agnostic and
-- `resnet34_fwd_eval` is correct against either. Same call R50 makes (`r50FwdChainB`'s docstring).
-- ════════════════════════════════════════════════════════════════

/-- A basic block's eval forward: its code and its output name (the next block's input). -/
structure BFwd where
  code : String
  o  : String        -- block output (post-relu)
deriving Inhabited

-- ════════════════════════════════════════════════════════════════
-- § Block forward
-- ════════════════════════════════════════════════════════════════

/-- Identity block forward: `conv1→BN1→relu1→conv2→BN2→(+x)→relu`. `c` channels, `hh×ww` spatial. -/
private def idFwd (B c hh : Nat) (epsStr p xName : String)
    (convBias : Bool) : StateM Proofs.StableHLO.EmitS BFwd := do
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (c*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvF (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk zc (.operand xName zin))
  let (cN1, nN1) ← bnEvalSite B c hh hh epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zin))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk zc (.operand nR1 zin))
  let (cN2, nN2) ← bnEvalSite B c hh hh epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cA,  nA)  ← pretty B (.addV (.operand nN2 zin) (.operand xName zin))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zin))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cA ++ cO, o := nO }

/-- Downsample block forward: strided `conv1→BN1→relu1→conv2→BN2` body + strided projection
    `convp→BNp` skip, `add`, `relu`. `cin→c` channels, input `2hh×2ww`, output `hh×ww`. -/
private def downFwd (B cin c hh : Nat) (epsStr p xName : String)
    (convBias : Bool) : StateM Proofs.StableHLO.EmitS BFwd := do
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  -- He et al.'s option-B shortcut is 1×1. A SEPARATE kernel from `zk1` — sharing that binding
  -- hides a 3×3 shortcut from everything but a param-count check.
  let zkp  : Kernel4 c cin 1 1 := fun _ _ _ _ => 0
  let zinS : Vec (cin*(2*hh)*(2*ww)) := fun _ => 0
  let zout : Vec (c*hh*ww) := fun _ => 0
  let (cC1, nC1) ← pretty B (.flatConvStridedF (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk1 zc (.operand xName zinS))
  let (cN1, nN1) ← bnEvalSite B c hh hh epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}n1" nC1
  let (cR1, nR1) ← pretty B (.reluF (.operand nN1 zout))
  let (cC2, nC2) ← pretty B (.flatConvF (ic := c) (oc := c) (h := hh) (w := ww) s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk2 zc (.operand nR1 zout))
  let (cN2, nN2) ← bnEvalSite B c hh hh epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}n2" nC2
  let (cCp, nCp) ← pretty B (.flatConvStridedF (ic := cin) (oc := c) (h := hh) (w := ww) s!"%{p}Wp" (biasName convBias s!"%{p}bp" c) zkp zc (.operand xName zinS))
  let (cNp, nNp) ← bnEvalSite B c hh hh epsStr s!"%{p}gp" s!"%{p}btp" s!"{p}np" nCp
  let (cA,  nA)  ← pretty B (.addV (.operand nN2 zout) (.operand nNp zout))
  let (cO,  nO)  ← pretty B (.reluF (.operand nA zout))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cCp ++ cNp ++ cA ++ cO, o := nO }

-- ════════════════════════════════════════════════════════════════
-- § Param signature lists (func-arg order — names + types, shared by sig + return types)
-- ════════════════════════════════════════════════════════════════

private def idSig (p : String) (c : Nat) (convBias : Bool) : List (String × String) :=
  let b (nm : String) : List (String × String) := if convBias then [(nm, ty [c])] else []
  [(s!"%{p}W1", ty [c,c,3,3])] ++ b s!"%{p}b1" ++ [(s!"%{p}g1", ty [c]), (s!"%{p}bt1", ty [c])] ++
  [(s!"%{p}W2", ty [c,c,3,3])] ++ b s!"%{p}b2" ++ [(s!"%{p}g2", ty [c]), (s!"%{p}bt2", ty [c])]

private def downSig (p : String) (cin c : Nat) (convBias : Bool) : List (String × String) :=
  let b (nm : String) : List (String × String) := if convBias then [(nm, ty [c])] else []
  [(s!"%{p}W1", ty [c,cin,3,3])] ++ b s!"%{p}b1" ++ [(s!"%{p}g1", ty [c]), (s!"%{p}bt1", ty [c])] ++
  [(s!"%{p}W2", ty [c,c,3,3])] ++ b s!"%{p}b2" ++ [(s!"%{p}g2", ty [c]), (s!"%{p}bt2", ty [c])] ++
  [(s!"%{p}Wp", ty [c,cin,1,1])] ++ b s!"%{p}bp" ++ [(s!"%{p}gp", ty [c]), (s!"%{p}btp", ty [c])]

/-- **The ResNet-34 parameters in `net.paramShapes` (= func-arg) order**, names + types.
    The forward, the eval forward and the train step all take their signature from here, so the
    arity/type/order contract the driver relies on cannot drift between renders.

    **110 at the shipped `convBias := false`** — stem 3 + 13 identity blocks × 6 + 3 downsample
    blocks × 9 + dense 2 — and 146 with the conv biases in. Every writer below omits the argument,
    so every committed artifact is the 110 one; the biases are `zeroBiasPrelude`'s zero constants. -/
def r34SigList (nClasses : Nat) (convBias : Bool := false) : List (String × String) :=
  [("%sW", ty [64,3,7,7])] ++ (if convBias then [("%sbi", ty [64])] else []) ++
  [("%sg", ty [64]), ("%sbt", ty [64])] ++
  idSig "s1b0" 64 convBias ++ idSig "s1b1" 64 convBias ++ idSig "s1b2" 64 convBias ++
  downSig "d2" 64 128 convBias ++ idSig "s2b0" 128 convBias ++ idSig "s2b1" 128 convBias ++ idSig "s2b2" 128 convBias ++
  downSig "d3" 128 256 convBias ++ idSig "s3b0" 256 convBias ++ idSig "s3b1" 256 convBias ++ idSig "s3b2" 256 convBias ++
    idSig "s3b3" 256 convBias ++ idSig "s3b4" 256 convBias ++
  downSig "d4" 256 512 convBias ++ idSig "s4b0" 512 convBias ++ idSig "s4b1" 512 convBias ++
  [("%Wd", ty [512, nClasses]), ("%bd", ty [nClasses])]

/-- **The 72 running-stat inputs** — 36 BN layers × (μ, var), each `[oc]`, in BN-forward order:
    stem, then per identity block `n1 n2`, per downsample block `n1 n2 np`. This is exactly the
    order `VerifiedNet.bnChannels` is listed in, which is how the driver packs `runningBnStats`
    (`bnChannels.foldl (fun acc c => acc ++ #[#[c], #[c]])`) — μ and var interleaved per layer,
    NOT all-μ-then-all-var. Appended after the parameters, so `@resnet34_fwd_eval` takes
    1 + 110 + 72 = **183** inputs as committed (1 + 146 + 72 = 219 at `convBias := true`). -/
def r34StatSigList : List (String × String) := List.map (fun (n, ds) => (n, ty ds)) <|
  let bn := bnStatSlots
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

/-- The ResNet-34 eval forward: its code and the `logits` name the eval function returns. (The train
    step walks the batched `r34FwdChainB` instead.) -/
structure R34Fwd where
  code   : String        -- stem → 16 blocks → GAP → dense, in emission order
  logits : String        -- dense output

/-- **The ResNet-34 `[3,4,6,3]` EVAL forward as `pretty` of the verified AST** (per-example index).
    7×7/s2 stem (3→64, 224→112) → 3×3/s2 max-pool (→56) → stages 64/128/256/512 at 56/28/14/7
    (stages 2–4 open with a strided downsample block) → GAP(7×7) → dense(512→`nClasses`). Every BN
    site is `bnEvalSite` — frozen running statistics — so this writes `@resnet34_fwd_eval` only; the
    training forward is the batched `r34FwdChainB`. -/
private def r34FwdChain (B nClasses : Nat) (epsStr : String)
    (convBias : Bool) : StateM Proofs.StableHLO.EmitS R34Fwd := do
  -- ═══ stem: 7×7/s2 conv → BN → relu → maxpool ═══
  let zx   : Vec (3*224*224) := fun _ => 0
  let zSk  : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
  let z64  : Vec 64 := fun _ => 0
  let z112 : Vec (64*112*112) := fun _ => 0
  let (cStc, nStc) ← pretty B (.flatConvStridedF (ic := 3) (oc := 64) (h := 112) (w := 112) "%sW" (biasName convBias "%sbi" 64) zSk z64 (.operand "%x" zx))
  let (cStn, nStn) ← bnEvalSite B 64 112 112 epsStr "%sg" "%sbt" "stn" nStc
  let (cStr, nStr) ← pretty B (.reluF (.operand nStn z112))
  -- He et al.'s 3×3/s2 stem pool — see the note on `ResNet34RenderB`'s peer. This renderer
  -- writes `resnet34_fwd{,_eval}` as well as the SGD train step, and the ADAMW trainer evals
  -- through `resnet34_fwd_eval`. So it must match `ResNet34RenderB`: a 2×2 pool here would
  -- train a 3×3-pool net and score it with a 2×2-pool forward (the `mobilenetv2_fwd` defect
  -- class, logits rel 1.86).
  let (cStp, nStp) ← pretty B (.maxPool3s2F (c := 64) (h := 56) (w := 56) (.operand nStr z112))
  -- ═══ 16 blocks ═══
  let f1  ← idFwd   B 64 56 epsStr "s1b0" nStp convBias
  let f2  ← idFwd   B 64 56 epsStr "s1b1" f1.o convBias
  let f3  ← idFwd   B 64 56 epsStr "s1b2" f2.o convBias
  let f4  ← downFwd B 64 128 28 epsStr "d2" f3.o convBias
  let f5  ← idFwd   B 128 28 epsStr "s2b0" f4.o convBias
  let f6  ← idFwd   B 128 28 epsStr "s2b1" f5.o convBias
  let f7  ← idFwd   B 128 28 epsStr "s2b2" f6.o convBias
  let f8  ← downFwd B 128 256 14 epsStr "d3" f7.o convBias
  let f9  ← idFwd   B 256 14 epsStr "s3b0" f8.o convBias
  let f10 ← idFwd   B 256 14 epsStr "s3b1" f9.o convBias
  let f11 ← idFwd   B 256 14 epsStr "s3b2" f10.o convBias
  let f12 ← idFwd   B 256 14 epsStr "s3b3" f11.o convBias
  let f13 ← idFwd   B 256 14 epsStr "s3b4" f12.o convBias
  let f14 ← downFwd B 256 512 7 epsStr "d4" f13.o convBias
  let f15 ← idFwd   B 512 7 epsStr "s4b0" f14.o convBias
  let f16 ← idFwd   B 512 7 epsStr "s4b1" f15.o convBias
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
         logits := nLog }

/-- **`@resnet34_fwd_eval` rendered ENTIRELY from the verified AST** — the inference forward, with
    every BN site consuming frozen per-channel running stats (`bnPerChannelEvalF`) instead of
    reducing statistics out of its activation. Same net, same parameters in the same order, plus
    the 72 stat inputs of `r34StatSigList`: **183 inputs** as committed, returning logits
    `[B, nClasses]`.

    This is the eval partner of `resnet34AdamTrainStepText` (this file), whose 72 returned
    batch statistics the driver EMAs into exactly these slots. -/
def resnet34FwdEvalText (B nClasses : Nat) (epsStr : String)
    (slug : String := "resnet34") (convBias : Bool := false) : String :=
  let sigList := r34SigList nClasses convBias ++ r34StatSigList
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let F : R34Fwd := (r34FwdChain B nClasses epsStr convBias).run' (0, [])
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd_eval({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  s!"    // ── ResNet-34 eval forward (running-stats BN): every op is pretty(verified AST node){if convBias then "" else " except the %zb zero-bias constants"} ──\n" ++
  zeroBiasPrelude convBias [64, 128, 256, 512] ++ F.code ++
  s!"    return {F.logits} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"


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
  -- SYNC-BN (`replicas > 1`): the all-reduced packed `[μ ‖ σ²]` of each BN site, which
  -- the backward, the γ gradient and the handed-back running stats all read. `""` at one replica.
  st1 : String
  st2 : String
  stp : String
deriving Inhabited

-- ════════════════════════════════════════════════════════════════
-- § Block forward (batch BN)
-- ════════════════════════════════════════════════════════════════

/-- Identity block forward: `conv1→BN1→relu1→conv2→BN2→(+x)→relu`, all at `N := B`. -/
def idFwdB (B c hh : Nat) (epsStr p xName : String)
    (convBias : Bool) (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS BFwdB := do
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (B*(c*hh*ww)) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (.convAt bf16 (h := hh) (w := ww) zrnd s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk zc) (.operand xName zin))
  let (cN1, nN1, st1) ← bnFwdSite B c hh hh sync replicas epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}g1" nC1
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nN1 zin))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (.convAt bf16 (h := hh) (w := ww) zrnd s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk zc) (.operand nR1 zin))
  let (cN2, nN2, st2) ← bnFwdSite B c hh hh sync replicas epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}g2" nC2
  let (cA,  nA)  ← pretty B (.addVB (.operand nN2 zin) (.operand xName zin))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nA zin))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cA ++ cO, xin := xName,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := "",
         st1 := st1, st2 := st2, stp := "" }

/-- Downsample block forward: strided body + strided projection skip. `cin→c`, `2hh→hh`. -/
def downFwdB (B cin c hh : Nat) (epsStr p xName : String)
    (convBias : Bool) (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS BFwdB := do
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  -- The projection shortcut is He et al.'s option-B **1×1**, not a 3×3. It is a SEPARATE kernel
  -- from the block's `zk1` — sharing that binding hides a 3×3 shortcut from everything but a
  -- param-count check.
  let zkp  : Kernel4 c cin 1 1 := fun _ _ _ _ => 0
  let zinS : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zout : Vec (B*(c*hh*ww)) := fun _ => 0
  let (cC1, nC1) ← pretty B (.batchOp (N := B) (.convStridedAt bf16 (h := hh) (w := ww) zrnd s!"%{p}W1" (biasName convBias s!"%{p}b1" c) zk1 zc) (.operand xName zinS))
  let (cN1, nN1, st1) ← bnFwdSite B c hh hh sync replicas epsStr s!"%{p}g1" s!"%{p}bt1" s!"{p}g1" nC1
  let (cR1, nR1) ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nN1 zout))
  let (cC2, nC2) ← pretty B (.batchOp (N := B) (.convAt bf16 (h := hh) (w := ww) zrnd s!"%{p}W2" (biasName convBias s!"%{p}b2" c) zk2 zc) (.operand nR1 zout))
  let (cN2, nN2, st2) ← bnFwdSite B c hh hh sync replicas epsStr s!"%{p}g2" s!"%{p}bt2" s!"{p}g2" nC2
  let (cCp, nCp) ← pretty B (.batchOp (N := B) (.convStridedAt bf16 (h := hh) (w := ww) zrnd s!"%{p}Wp" (biasName convBias s!"%{p}bp" c) zkp zc) (.operand xName zinS))
  let (cNp, nNp, stp) ← bnFwdSite B c hh hh sync replicas epsStr s!"%{p}gp" s!"%{p}btp" s!"{p}gp" nCp
  let (cA,  nA)  ← pretty B (.addVB (.operand nN2 zout) (.operand nNp zout))
  let (cO,  nO)  ← pretty B (.batchOp (N := B) (.relu (n := c*hh*ww)) (.operand nA zout))
  pure { code := cC1 ++ cN1 ++ cR1 ++ cC2 ++ cN2 ++ cCp ++ cNp ++ cA ++ cO, xin := xName,
         o := nO, a := nA, c1 := nC1, n1 := nN1, r1 := nR1, c2 := nC2, cp := nCp,
         st1 := st1, st2 := st2, stp := stp }

-- ════════════════════════════════════════════════════════════════
-- § Block backward + UN-FUSED parameter gradients
-- ════════════════════════════════════════════════════════════════

/-- Identity block backward + its 8 parameter gradients. -/
private def idBackGradB (B c hh : Nat) (epsStr p : String) (f : BFwdB) (dyName : String)
    (convBias : Bool) (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS BlockBack := do
  let xName := f.xin
  let ww := hh
  let zc  : Vec c := fun _ => 0
  let zk  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zin : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn : Vec (B*(c*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zin (.operand dyName zin))
  let (cDn2, nDn2) ← bnBackSite B c hh hh sync replicas epsStr s!"%{p}g2" f.c2 s!"{p}g2dst" nDa f.st2
  let (cDc2, nDc2) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := c) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}W2" zk zc (.operand nDn2 zin))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zin (.operand nDc2 zin))
  let (cDn1, nDn1) ← bnBackSite B c hh hh sync replicas epsStr s!"%{p}g1" f.c1 s!"{p}g1dst" nDr1 f.st1
  let (cDc1, nDc1) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := c) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}W1" zk zc (.operand nDn1 zin))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zin) (.operand nDa zin))
  -- parameter gradients, func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2
  let (cW1, nW1) ← pretty B (.convWeightGradBAt bf16 zrnd xName zc zin zk (.operand nDn1 zin))
  let (cb1, nb1) ← if convBias then pretty B (.convBiasGradB (h := hh) (w := ww) zk zin zc (.operand nDn1 zin)) else pure ("", "")
  let (cg1, ng1) ← bnGammaSite B c hh hh sync epsStr f.c1 nDr1 f.st1
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDr1 zbn))
  let (cW2, nW2) ← pretty B (.convWeightGradBAt bf16 zrnd f.r1 zc zin zk (.operand nDn2 zin))
  let (cb2, nb2) ← if convBias then pretty B (.convBiasGradB (h := hh) (w := ww) zk zin zc (.operand nDn2 zin)) else pure ("", "")
  let (cg2, ng2) ← bnGammaSite B c hh hh sync epsStr f.c2 nDa f.st2
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  pure { code := cDa ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDx ++
                 cW1 ++ cb1 ++ cg1 ++ ct1 ++ cW2 ++ cb2 ++ cg2 ++ ct2,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [c,c,3,3]⟩] ++
                (if convBias then [⟨s!"{p}b1", nb1, [c]⟩] else []) ++
                [⟨s!"{p}g1", ng1, [c]⟩, ⟨s!"{p}bt1", nt1, [c]⟩,
                 ⟨s!"{p}W2", nW2, [c,c,3,3]⟩] ++
                (if convBias then [⟨s!"{p}b2", nb2, [c]⟩] else []) ++
                [⟨s!"{p}g2", ng2, [c]⟩, ⟨s!"{p}bt2", nt2, [c]⟩] }

/-- Downsample block backward + its 12 parameter gradients. -/
private def downBackGradB (B cin c hh : Nat) (epsStr p : String) (f : BFwdB) (dyName : String)
    (convBias : Bool) (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS BlockBack := do
  let xName := f.xin
  let ww := hh
  let zc   : Vec c := fun _ => 0
  let zk1  : Kernel4 c cin 3 3 := fun _ _ _ _ => 0
  let zk2  : Kernel4 c c 3 3 := fun _ _ _ _ => 0
  let zkp  : Kernel4 c cin 1 1 := fun _ _ _ _ => 0      -- the 1×1 option-B shortcut
  let zinS : Vec (B*(cin*(2*hh)*(2*ww))) := fun _ => 0
  let zout : Vec (B*(c*hh*ww)) := fun _ => 0
  let zbn  : Vec (B*(c*(hh*ww))) := fun _ => 0
  let (cDa,  nDa)  ← pretty B (.selectPosB f.a zout (.operand dyName zout))
  let (cDn2, nDn2) ← bnBackSite B c hh hh sync replicas epsStr s!"%{p}g2" f.c2 s!"{p}g2dst" nDa f.st2
  let (cDc2, nDc2) ← pretty B (.convBackBatchedAt bf16 (N := B) (ic := c) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}W2" zk2 zc (.operand nDn2 zout))
  let (cDr1, nDr1) ← pretty B (.selectPosB f.n1 zout (.operand nDc2 zout))
  let (cDn1, nDn1) ← bnBackSite B c hh hh sync replicas epsStr s!"%{p}g1" f.c1 s!"{p}g1dst" nDr1 f.st1
  let (cDc1, nDc1) ← pretty B (.convStridedBackBatchedAt bf16 (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}W1" zk1 zc (.operand nDn1 zout))
  let (cDnp, nDnp) ← bnBackSite B c hh hh sync replicas epsStr s!"%{p}gp" f.cp s!"{p}gpdst" nDa f.stp
  let (cDcp, nDcp) ← pretty B (.convStridedBackBatchedAt bf16 (N := B) (ic := cin) (oc := c) (h := hh) (w := ww) zrnd s!"%{p}Wp" zkp zc (.operand nDnp zout))
  let (cDx,  nDx)  ← pretty B (.addVB (.operand nDc1 zinS) (.operand nDcp zinS))
  -- parameter gradients, func-arg order: W1 b1 g1 bt1 W2 b2 g2 bt2 Wp bp gp btp
  let (cW1, nW1) ← pretty B (.convStridedWeightGradBAt bf16 zrnd xName zc zinS zk1 (.operand nDn1 zout))
  let (cb1, nb1) ← if convBias then pretty B (.convStridedBiasGradB (h := hh) (w := ww) zk1 zinS zc (.operand nDn1 zout)) else pure ("", "")
  let (cg1, ng1) ← bnGammaSite B c hh hh sync epsStr f.c1 nDr1 f.st1
  let (ct1, nt1) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDr1 zbn))
  let (cW2, nW2) ← pretty B (.convWeightGradBAt bf16 zrnd f.r1 zc zout zk2 (.operand nDn2 zout))
  let (cb2, nb2) ← if convBias then pretty B (.convBiasGradB (h := hh) (w := ww) zk2 zout zc (.operand nDn2 zout)) else pure ("", "")
  let (cg2, ng2) ← bnGammaSite B c hh hh sync epsStr f.c2 nDa f.st2
  let (ct2, nt2) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  let (cWp, nWp) ← pretty B (.convStridedWeightGradBAt bf16 zrnd xName zc zinS zkp (.operand nDnp zout))
  let (cbp, nbp) ← if convBias then pretty B (.convStridedBiasGradB (h := hh) (w := ww) zkp zinS zc (.operand nDnp zout)) else pure ("", "")
  let (cgp, ngp) ← bnGammaSite B c hh hh sync epsStr f.cp nDa f.stp
  let (ctp, ntp) ← pretty B (.bnBetaGradB (N := B) (oc := c) (h := hh) (w := ww) (.operand nDa zbn))
  pure { code := cDa ++ cDn2 ++ cDc2 ++ cDr1 ++ cDn1 ++ cDc1 ++ cDnp ++ cDcp ++ cDx ++
                 cW1 ++ cb1 ++ cg1 ++ ct1 ++ cW2 ++ cb2 ++ cg2 ++ ct2 ++ cWp ++ cbp ++ cgp ++ ctp,
         dx := nDx,
         ps := [⟨s!"{p}W1", nW1, [c,cin,3,3]⟩] ++
                (if convBias then [⟨s!"{p}b1", nb1, [c]⟩] else []) ++
                [⟨s!"{p}g1", ng1, [c]⟩, ⟨s!"{p}bt1", nt1, [c]⟩,
                 ⟨s!"{p}W2", nW2, [c,c,3,3]⟩] ++
                (if convBias then [⟨s!"{p}b2", nb2, [c]⟩] else []) ++
                [⟨s!"{p}g2", ng2, [c]⟩, ⟨s!"{p}bt2", nt2, [c]⟩,
                 ⟨s!"{p}Wp", nWp, [c,cin,1,1]⟩] ++
                (if convBias then [⟨s!"{p}bp", nbp, [c]⟩] else []) ++
                [⟨s!"{p}gp", ngp, [c]⟩, ⟨s!"{p}btp", ntp, [c]⟩] }

end Proofs.StableHLO

namespace Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § The whole-net batched AdamW train step
-- ════════════════════════════════════════════════════════════════

/-- The driver's **variant slug** for a given `(B, replicas)`: the artifact is
    `verified_mlir/resnet34_<variant>_train_step.mlir`, the entry point is
    `@resnet34_<variant>_train_step`, and `LEAN_MLIR_VARIANT` selects it.

    All three must agree. The shim checks the entry name and refuses a mismatch outright ("entry
    mismatch") rather than running the wrong graph — which is exactly what it did the first time the
    DP render kept the single-device name. Deriving the name here, from the same two
    numbers the render is built from, is what stops it drifting from the `#eval` paths below; the
    `#guard`s at the bottom pin those literal paths against this function.

    `B = 32` is deliberately unsuffixed, so the two existing artifacts keep their names and bytes. -/
def r34AdamVariant (B replicas : Nat) (opt : OptRecipe := .adamw)
    -- `wx` = timm `no_weight_decay`. TRAILING and defaulted, so every existing spelling is
    -- unchanged. It must reach this function: R50 DERIVES its entry name from the variant, so a
    -- flag that reached the renderer but not here produces an artifact whose declared entry
    -- disagrees with its own path — the shim then refuses the call outright. The `#guard`s below
    -- pin every spelling.
    -- It needs NO driver predicate: excluding a parameter changes no arity, type or region.
    (wdExclude : Bool := false)
    -- `clip` = timm `Lamb.max_grad_norm`, the GLOBAL-norm gradient clip. TRAILING and
    -- defaulted, exactly as `wx` is, so every existing spelling is unchanged.
    -- It must reach THIS function and not merely the renderer — the same rule `wx` states above,
    -- and `cnxAdamVariant`'s docstring records ConvNeXt shipping that defect twice (once for `wx`,
    -- once for `clip`). R50 derives its entry name from this string, so a flag that stopped at the
    -- emission would produce an artifact whose declared entry disagrees with its own path and the
    -- shim would refuse the call outright.
    -- Like `wx` it needs NO driver predicate: the clip adds ops, not arity, types or regions.
    (gradClip : Bool := false)
    -- `bce` = BCE-with-logits instead of smoothed CE — RSB-A2/A3's loss, and the reason the
    -- recipe's lr is what it is. **IT IS A PARAMETER HERE, NOT A CALLER-SPELLED SUFFIX**: a
    -- `bce : Bool` that swaps the loss beside a separate name string the caller spells `"bce"` by
    -- hand is two writers for one fact, able to disagree with nothing noticing. Deriving the
    -- marker from the flag makes the disagreement unspellable rather than merely unobserved.
    -- It TRAILS `wx` and `clip` — `lambaccdp8x64wxclipbce`; the `#guard`s below pin it.
    (bce : Bool := false)
    -- A NON-DEFAULT weight decay, and it must reach the name for the reason `wdVariantMark` gives:
    -- `%wd` is BAKED, so two renders differing only in it would otherwise collide on one artifact
    -- path. Empty = the optimizer's own default = no marker = every committed name unchanged. It
    -- goes LAST because it is the newest axis and newest-axis-appends is unconditional.
    (wdStr : String := "")
    -- `bf16` LAST, after even the decay marker, for that marker's own reason: it is the newest axis
    -- and appending is the only placement that leaves every existing spelling untouched. It MUST
    -- reach this function and not merely the renderer — the `wx`/`clip` rule three parameters up.
    -- And it must not collide with the driver's variant predicates: `momdp64bf16` contains no
    -- "acc", no "ema", and no "do", so `accOn`/ `emaOn`/`cdOn` all stay false and the region/scalar
    -- counts are unchanged (checked).
    (bf16 : Bool := false)
    -- **`ema` — the model-EMA shadow, and it is the ONLY marker that LEADS**.
    -- **PREFIX, NOT SUFFIX, AND THAT IS FORCED BY THE DRIVER**: `VerifiedVariant.emaOn` is
    -- `startsWith "ema"` while `accOn` is a substring test, so `lambaccdp8x64wxclipbceema` would
    -- read as accumulation-only — four regions packed into a five-region graph, i.e. every
    -- parameter misaligned, with no error anywhere. `tests/TestVariantPredicates.lean` pins both
    -- directions.
    -- It is LAST in this signature and FIRST in the string, which is the one place in this
    -- function where those two orders disagree. Parameter position is "newest axis appends";
    -- string position is the driver's predicate. Defaulted, so every committed spelling is
    -- unchanged — `ema` prepends nothing at `false`.
    (ema : Bool := false)
    -- **`drop` — STOCHASTIC DEPTH.** It goes BETWEEN `clip` and `bce`, which is the variant grammar
    -- (`…[wx][clip][drop][do][bce][wd<d>][bf16]`) and `cnxAdamVariant`'s `wxclipdrop` order, rather
    -- than appended like the newer axes — one rule for both nets, so a reader need not know which
    -- net a slug came from. The marker is `"drop"` and not `"sd"`: `rms` ++ `dp` spells `rmsdp`,
    -- which CONTAINS "sd".
    -- Parameter position is trailing (so no call site moves); STRING position is the grammar's.
    (sd : Bool := false)
    -- **`ls<α>` — LABEL SMOOTHING**, §5.6's ablation axis. Same rule as `wd<d>` one marker over:
    -- α is baked, so it must reach the NAME or two renders collide on one path. Defaulted to the
    -- recipe's 0.1, so every committed spelling is unchanged.
    (alpha : Float := 0.1) : String :=
  (if ema then "ema" else "") ++
  opt.slug replicas ++
  (if B == 32 then "" else toString B) ++
  -- `wx` TRAILS THE BATCH, so it composes with every optimizer spelling and with the `clip` and
  -- `bce` markers appended after it — `lambaccdp8x64bcewx` would be wrong; the order below gives
  -- `lambaccdp8x64wxclipbce`. The order is a choice and
  -- the `#guard`s below are what make it a fixed one, because a marker's POSITION is as
  -- load-bearing as its presence.
  (if wdExclude then "wx" else "") ++
  -- `clip` TRAILS `wx`, which is `cnxAdamVariant`'s order (`wx` ++ `clip`) followed deliberately
  -- rather than re-chosen — the two nets spell the same two flags, and one rule for both is what
  -- keeps a reader from having to know which net a slug came from. With R50's `bce` appended after,
  -- the RSB-A3 composition reads `lambaccdp8x64wxclipbce`. The order is a CHOICE; the `#guard`s
  -- below are what make it a fixed one.
  (if gradClip then "clip" else "") ++
  -- `drop` TRAILS `clip`, matching `cnxAdamVariant`'s `wxclipdrop` and the variant grammar. Check the
  -- CONCATENATIONS rather than the marker: `clip` ++ `drop` spells `clipdrop` and `drop` ++ `bce`
  -- spells `dropbce`, and neither contains `"do"` (`dr`, not `do`) — the collision class that has
  -- already fired three times in this naming. Pinned in `tests/TestVariantPredicates.lean`.
  (if sd then "drop" else "") ++
  -- `bce` after that, which is where the hand-passed `vSuffix` put it. See this parameter's note.
  (if bce then "bce" else "") ++
  -- …and the decay marker after even that, because it is the newest axis and appending is the
  -- only placement that leaves all four existing spellings untouched. Empty at the default.
  lsVariantMark alpha ++
  wdVariantMark opt wdStr ++
  (if bf16 then "bf16" else "")

-- ════════════════════════════════════════════════════════════════
-- § The forward traversal, factored — ONE chain, two consumers
-- ════════════════════════════════════════════════════════════════

/-- Everything the whole-net render needs out of ONE forward traversal of ResNet-34 at the BATCHED
    index: the emitted code, the logits and GAP names, and every saved activation the backward
    reads.

    **This exists so `@resnet34_fwd` and the batch-BN train steps cannot be different nets**: a
    forward built from a PER-EXAMPLE chain (`bnPerChannelF`, reduce `[2,3]`, divisor `H·W`) next to
    batch-BN train steps (reduce `[0,2,3]`, divisor `B·H·W`) is a different function. This is
    `ResNet50RenderB.r50FwdChainB`'s shape, for R50's reason.

    The EVAL forward is deliberately NOT moved onto this chain, exactly as R50's is not:
    `bnPerChannelEvalF` reads frozen per-channel statistics and reduces nothing, so
    `resnet34_fwd_eval.mlir` is BatchNorm-world-agnostic and correct against both chains.

    Extracting the traversal is byte-neutral for the train step: `pretty`'s SSA counter follows
    the call SEQUENCE, and the sequence is unchanged. -/
structure R34FwdRecB where
  code : String
  stc : String            -- stem conv out (the stem BN's input)
  stn : String            -- stem BN out
  str : String            -- stem relu out (the pool's input)
  sst : String            -- the stem BN's all-reduced packed stats (sync-BN; "" at one replica)
  gap : String            -- GAP out (= dense input)
  log : String            -- logits
  b : Array BFwdB         -- the 16 basic blocks, in forward order
deriving Inhabited

/-- The stem's saved SSA names: conv, BN, BN stats (`""` at one replica), relu, pool output. -/
structure R34StemFwdB where
  code : String
  c : String
  n : String
  st : String
  r : String
  o : String

/-- Stem forward: 7×7/s2 conv → batch BN → relu → He et al.'s 3×3/s2 max-pool, on `%x`.
    A 2×2 non-overlapping `.maxPool` would give the identical 112→56 output shape and be a
    different function, so no shape check tells them apart. -/
def r34StemFwdB (B : Nat) (epsStr : String) (convBias : Bool) (bf16 : Bool := false)
    (replicas : Nat := 1) (sync : Bool := false) : StateM Proofs.StableHLO.EmitS R34StemFwdB := do
  let zx    : Vec (B*(3*224*224)) := fun _ => 0
  let zSk   : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
  let z64   : Vec 64 := fun _ => 0
  let z112  : Vec (B*(64*112*112)) := fun _ => 0
  let (cStc, nStc) ← pretty B (.batchOp (N := B) (.convStridedAt bf16 (h := 112) (w := 112) zrnd "%sW" (biasName convBias "%sbi" 64) zSk z64) (.operand "%x" zx))
  let (cStn, nStn, sst) ← bnFwdSite B 64 112 112 sync replicas epsStr "%sg" "%sbt" "sg" nStc
  let (cStr, nStr) ← pretty B (.batchOp (N := B) (.relu (n := 64*112*112)) (.operand nStn z112))
  let (cStp, nStp) ← pretty B (.batchOp (N := B) (.maxPool3s2 (c := 64) (h := 56) (w := 56)) (.operand nStr z112))
  pure { code := cStc ++ cStn ++ cStr ++ cStp, c := nStc, n := nStn, st := sst, r := nStr, o := nStp }

/-- Head forward: GAP(7×7) → dense(512→nClasses) on the last block's output `xName`. Returns the
    text and the GAP and logit names. -/
def r34HeadFwdB (B nClasses : Nat) (xName : String) :
    StateM Proofs.StableHLO.EmitS (String × String × String) := do
  let zL    : Vec (B*(512*7*7)) := fun _ => 0
  let z512  : Vec (B*512) := fun _ => 0
  let zWd   : Mat 512 nClasses := fun _ _ => 0
  let zNC   : Vec nClasses := fun _ => 0
  let (cGap, nGap) ← pretty B (.batchOp (N := B) (.gap (c := 512) (h := 7) (w := 7)) (.operand xName zL))
  let (cLog, nLog) ← pretty B (.batchOp (N := B) (.dense "%Wd" "%bd" zWd zNC) (.operand nGap z512))
  pure (cGap ++ cLog, nGap, nLog)

/-- **The ResNet-34 forward chain at the BATCHED index** — one traversal, consumed by both
    `@resnet34_fwd` and every train step that differentiates it. -/
def r34FwdChainB (B nClasses : Nat) (epsStr : String) (convBias : Bool := false)
    (bf16 : Bool := false) (replicas : Nat := 1) (sync : Bool := false) :
    StateM Proofs.StableHLO.EmitS R34FwdRecB := do
  -- ═══ stem: 7×7/s2 conv → batch BN → relu → 3×3/s2 maxpool ═══
  let st ← r34StemFwdB B epsStr convBias bf16 replicas sync
  let (nStc, nStn, sst, nStr, nStp) := (st.c, st.n, st.st, st.r, st.o)
  -- ═══ 16 blocks ═══
  let f1  ← idFwdB   B 64 56 epsStr "s1b0" nStp convBias bf16 replicas sync
  let f2  ← idFwdB   B 64 56 epsStr "s1b1" f1.o convBias bf16 replicas sync
  let f3  ← idFwdB   B 64 56 epsStr "s1b2" f2.o convBias bf16 replicas sync
  let f4  ← downFwdB B 64 128 28 epsStr "d2" f3.o convBias bf16 replicas sync
  let f5  ← idFwdB   B 128 28 epsStr "s2b0" f4.o convBias bf16 replicas sync
  let f6  ← idFwdB   B 128 28 epsStr "s2b1" f5.o convBias bf16 replicas sync
  let f7  ← idFwdB   B 128 28 epsStr "s2b2" f6.o convBias bf16 replicas sync
  let f8  ← downFwdB B 128 256 14 epsStr "d3" f7.o convBias bf16 replicas sync
  let f9  ← idFwdB   B 256 14 epsStr "s3b0" f8.o convBias bf16 replicas sync
  let f10 ← idFwdB   B 256 14 epsStr "s3b1" f9.o convBias bf16 replicas sync
  let f11 ← idFwdB   B 256 14 epsStr "s3b2" f10.o convBias bf16 replicas sync
  let f12 ← idFwdB   B 256 14 epsStr "s3b3" f11.o convBias bf16 replicas sync
  let f13 ← idFwdB   B 256 14 epsStr "s3b4" f12.o convBias bf16 replicas sync
  let f14 ← downFwdB B 256 512 7 epsStr "d4" f13.o convBias bf16 replicas sync
  let f15 ← idFwdB   B 512 7 epsStr "s4b0" f14.o convBias bf16 replicas sync
  let f16 ← idFwdB   B 512 7 epsStr "s4b1" f15.o convBias bf16 replicas sync
  -- ═══ head: GAP(7×7) → dense(512→nClasses) ═══
  let (cHead, nGap, nLog) ← r34HeadFwdB B nClasses f16.o
  pure { code := st.code ++
           f1.code ++ f2.code ++ f3.code ++ f4.code ++ f5.code ++ f6.code ++ f7.code ++ f8.code ++
           f9.code ++ f10.code ++ f11.code ++ f12.code ++ f13.code ++ f14.code ++ f15.code ++
           f16.code ++ cHead,
         stc := nStc, stn := nStn, str := nStr, sst := sst, gap := nGap, log := nLog,
         b := #[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16] }

/-- **`@resnet34_fwd` rendered from the BATCHED chain** — the same traversal every batch-BN train
    step in this file differentiates, so the net that scores and the net that trains are one graph
    by construction. The writer of `verified_mlir/resnet34_fwd.mlir`.
    Takes `%x` plus the parameters in `r34SigList` order — 111 inputs at the shipped
    `convBias := false` — and returns logits `[B, nClasses]`. -/
def resnet34FwdText (B nClasses : Nat) (epsStr : String)
    (slug : String := "resnet34") (convBias : Bool := false) (bf16 : Bool := false) : String :=
  let sigList := r34SigList nClasses convBias
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++
    String.intercalate ", " (sigList.map (fun (n, t) => s!"{n}: {t}"))
  let F : R34FwdRecB := (r34FwdChainB B nClasses epsStr convBias bf16).run' (0, [])
  "module @m {\n" ++
  s!"  func.func @{slug}_fwd({inSig}) -> {ty [B, nClasses]} " ++ "{\n" ++
  s!"    // ── ResNet-34 batch-BN forward: every op is pretty(verified AST node){if convBias then "" else " except the %zb zero-bias constants"} ──\n" ++
  zeroBiasPrelude convBias [64, 128, 256, 512] ++ F.code ++
  s!"    return {F.log} : {ty [B, nClasses]}\n" ++
  "  }\n}\n"

/-- **ResNet-34 `[3,4,6,3]` AdamW train step, batch-BN, rendered from the verified AST at `N := B`.**
    **407** inputs at the shipped `convBias := false` (`%x`, 110 θ, 110 m, 110 v,
    `%lr`/`%bc1`/`%bc2`, 72 running-stat slots, `%onehot`) and 405 outputs (110 θ', 110 m', 110 v',
    `%loss`/`%bc1`/`%bc2`, 72 batch stats); 146 θ at `convBias := true`. Parameter ORDER comes from
    `r34SigList`, the same single source both forwards use, so the arity/order contract cannot
    drift between them. -/
def resnet34AdamTrainStepText (B nClasses : Nat) (epsStr : String)
    (replicas : Nat := 1) (opt : OptRecipe := .adamw) (slug : String := "resnet34")
    (convBias : Bool := false)
    -- TRAILING and defaulted, so every existing render is byte-identical.
    (bf16 : Bool := false)
    -- The two ABLATION knobs (§5.6). Both are baked constants, which is why each needs its own
    -- render rather than a runtime flag: `wdStr := "0.0"` is the no-weight-decay arm and
    -- `alpha := 0.0` the no-label-smoothing one. Defaulted, so the committed renders do not move.
    -- `alpha := 0.0` does NOT change the graph's SHAPE -- the smoothing ops stay, at 1.0/0.0 --
    -- so the arm differs from the full recipe in constants only, which is what makes it an
    -- ablation of the recipe rather than of the network.
    (wdStr : String := "") (alpha : Float := 0.1)
    -- `forceSync`: the sync-BN graph at ONE replica (every collective empty), for the numeric
    -- gate only — it separates "the seven sync ops' text is right" from "the collective composes
    -- them right". Never a committed artifact; `resnet34-syncbn-check` renders it to `.lake/build/`.
    (forceSync : Bool := false) : String :=
  let sync : Bool := replicas > 1 || forceSync
  let optLabel : String := match opt with
    | .adamw     => "AdamW"
    | .heavyBall => "heavy-ball momentum + coupled L2"
    | .sgd       => "plain SGD + coupled L2 (no momentum)"
    -- R34 renders no accumulation artifact — `resnet34AdamTrainStepText` has no fourth region in
    -- its signature, so passing `.adamwAccum` here would emit an optimizer that reads `%<p>a` inputs
    -- the function does not declare. The renderer REFUSES rather than emitting invalid MLIR that
    -- `iree-compile` would report as an undefined-value error a hundred lines from the cause.
    | .lamb      => "LAMB"
    -- RMSProp's constants are a net's own `RmsHyper` (`optConstsB`'s `rms`), and ResNet-34 has
    -- no reference RMSProp recipe to take them from.
    | .rmsprop   => panic! "resnet34AdamTrainStepText: .rmsprop needs a ResNet-34 RmsHyper and \
none is defined"
    | .adamwAccum k => panic! s!"resnet34AdamTrainStepText: .adamwAccum {k} needs a fourth \
parameter region and R34's signature has three — render it from ResNet50RenderB, or add the region \
here first"
    -- Same refusal, same reason: the fourth region is a property of the SIGNATURE, not of which
    -- optimizer consumes the accumulator, so `.lambAccum` is no more renderable here than
    -- `.adamwAccum`. This arm exists because the match is exhaustive — adding the constructor
    -- without it is a build error, so the type catches this site rather than a run.
    | .lambAccum k => panic! s!"resnet34AdamTrainStepText: .lambAccum {k} needs a fourth \
parameter region and R34's signature has three — render it from ResNet50RenderB, or add the region \
here first"
  let go : StateM Proofs.StableHLO.EmitS String := do
    -- ═══ forward — the SAME traversal `@resnet34_fwd` renders, so the forward this differentiates
    --     and the forward the driver scores with are one graph by construction ═══
    let F : R34FwdRecB ← r34FwdChainB B nClasses epsStr convBias bf16 replicas sync
    let zx    : Vec (B*(3*224*224)) := fun _ => 0
    let zSk   : Kernel4 64 3 7 7 := fun _ _ _ _ => 0
    let z64   : Vec 64 := fun _ => 0
    let z112  : Vec (B*(64*112*112)) := fun _ => 0
    let z112b : Vec (B*(64*(112*112))) := fun _ => 0
    let z56   : Vec (B*(64*56*56)) := fun _ => 0
    let nStc := F.stc; let nStn := F.stn; let nStr := F.str
    let f1  := F.b[0]!;  let f2  := F.b[1]!;  let f3  := F.b[2]!;  let f4  := F.b[3]!
    let f5  := F.b[4]!;  let f6  := F.b[5]!;  let f7  := F.b[6]!;  let f8  := F.b[7]!
    let f9  := F.b[8]!;  let f10 := F.b[9]!;  let f11 := F.b[10]!; let f12 := F.b[11]!
    let f13 := F.b[12]!; let f14 := F.b[13]!; let f15 := F.b[14]!; let f16 := F.b[15]!
    let z512  : Vec (B*512) := fun _ => 0
    let zWd   : Mat 512 nClasses := fun _ _ => 0
    let zNCb  : Vec (B*(1*nClasses)) := fun _ => 0
    let zNCp  : Vec (B*nClasses) := fun _ => 0
    let nGap := F.gap; let nLog := F.log
    -- ═══ label-smoothed softmax-CE cotangent (α = 0.1, K = nClasses):
    --     dy = (softmax(logits) − onehot + α·onehot − α/K) / B. The softmax, then `smoothedCotB`:
    --     together `pretty` of `smoothedLossCotGraph`, the graph the step tie starts from. ═══
    let (cSm,  nSm)  ← pretty B (.batchOp (N := B) (.softmaxRow (m := 1) (n := nClasses)) (.operand nLog zNCb))
    let (cDy,  nDy)  ← smoothedCotB B (1 * nClasses) (fmt6 alpha) s!"-{alphaOverK nClasses alpha}"
      s!"{B}.0" nSm
    -- ═══ head backward + dense grads ═══
    let (cDgi, nDgi) ← pretty B (.batchOp (N := B) (.denseRowBack (rows := 1) (a := 512) (c := nClasses) "%Wd" zWd) (.operand nDy zNCb))
    let (cWd,  nWd)  ← pretty B (.denseWeightGradB (c := nClasses) nGap z512 (.operand nDy zNCp))
    let (cbd,  nbd)  ← pretty B (.denseBiasGradB (N := B) (.operand nDy zNCp))
    let (cDgp, nDgp) ← pretty B (.gapBackBatched (N := B) (c := 512) (h := 7) (w := 7) (.operand nDgi z512))
    -- ═══ 16 block backwards ═══
    let b16 ← idBackGradB   B 512 7 epsStr "s4b1" f16 nDgp convBias bf16 replicas sync
    let b15 ← idBackGradB   B 512 7 epsStr "s4b0" f15 b16.dx convBias bf16 replicas sync
    let b14 ← downBackGradB B 256 512 7 epsStr "d4" f14 b15.dx convBias bf16 replicas sync
    let b13 ← idBackGradB   B 256 14 epsStr "s3b4" f13 b14.dx convBias bf16 replicas sync
    let b12 ← idBackGradB   B 256 14 epsStr "s3b3" f12 b13.dx convBias bf16 replicas sync
    let b11 ← idBackGradB   B 256 14 epsStr "s3b2" f11 b12.dx convBias bf16 replicas sync
    let b10 ← idBackGradB   B 256 14 epsStr "s3b1" f10 b11.dx convBias bf16 replicas sync
    let b9  ← idBackGradB   B 256 14 epsStr "s3b0" f9  b10.dx convBias bf16 replicas sync
    let b8  ← downBackGradB B 128 256 14 epsStr "d3" f8 b9.dx convBias bf16 replicas sync
    let b7  ← idBackGradB   B 128 28 epsStr "s2b2" f7 b8.dx convBias bf16 replicas sync
    let b6  ← idBackGradB   B 128 28 epsStr "s2b1" f6 b7.dx convBias bf16 replicas sync
    let b5  ← idBackGradB   B 128 28 epsStr "s2b0" f5 b6.dx convBias bf16 replicas sync
    let b4  ← downBackGradB B 64 128 28 epsStr "d2" f4 b5.dx convBias bf16 replicas sync
    let b3  ← idBackGradB   B 64 56 epsStr "s1b2" f3 b4.dx convBias bf16 replicas sync
    let b2  ← idBackGradB   B 64 56 epsStr "s1b1" f2 b3.dx convBias bf16 replicas sync
    let b1  ← idBackGradB   B 64 56 epsStr "s1b0" f1 b2.dx convBias bf16 replicas sync
    -- ═══ stem backward: maxpool-back → relu mask → BN back, then the 4 stem grads ═══
    let (cDmp, nDmp) ← pretty B (.maxPool3s2BackB (N := B) (c := 64) (h := 56) (w := 56) nStr z112 (.operand b1.dx z56))
    let (cDsr, nDsr) ← pretty B (.selectPosB nStn z112 (.operand nDmp z112))
    let (cDsn, nDsn) ← bnBackSite B 64 112 112 sync replicas epsStr "%sg" nStc "sgdst" nDsr F.sst
    let (csW, nsW) ← pretty B (.convStridedWeightGradBAt bf16 zrnd "%x" z64 zx zSk (.operand nDsn z112))
    let (csb, nsb) ← if convBias then
        pretty B (.convStridedBiasGradB (h := 112) (w := 112) zSk zx z64 (.operand nDsn z112))
      else pure ("", "")
    let (csg, nsg) ← bnGammaSite B 64 112 112 sync epsStr nStc nDsr F.sst
    let (cst, nst) ← pretty B (.bnBetaGradB (N := B) (oc := 64) (h := 112) (w := 112) (.operand nDsr z112b))
    -- ═══ BN running statistics: batch μ/var per BN layer, from that layer's BN INPUT ═══
    -- At `replicas > 1` these are read off the all-reduced packed vector (`bnStatsMeanB` /
    -- `bnStatsVarB`), so the host EMAs the GLOBAL batch statistics — what the reference's `_bn`
    -- buffers hold — rather than replica 0's shard's.
    let bnStat (oc hh : Nat) (xn st : String) : StateM Proofs.StableHLO.EmitS (String × String × String) := do
      let zb : Vec (B*(oc*(hh*hh))) := fun _ => 0
      let zst : Vec (oc+oc) := fun _ => 0
      if !sync then
        let (cM, nM) ← pretty B (.bnBatchMeanB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
        let (cV, nV) ← pretty B (.bnBatchVarB (N := B) (oc := oc) (h := hh) (w := hh) (.operand xn zb))
        pure (cM ++ cV, nM, nV)
      else
        let (cM, nM) ← pretty B (.bnStatsMeanB (oc := oc) (.operand st zst))
        let (cV, nV) ← pretty B (.bnStatsVarB (oc := oc) (.operand st zst))
        pure (cM ++ cV, nM, nV)
    let idStats (oc hh : Nat) (f : BFwdB) : StateM Proofs.StableHLO.EmitS (String × List String) := do
      let (c1, m1, v1) ← bnStat oc hh f.c1 f.st1
      let (c2, m2, v2) ← bnStat oc hh f.c2 f.st2
      pure (c1 ++ c2, [m1, v1, m2, v2])
    let downStats (oc hh : Nat) (f : BFwdB) : StateM Proofs.StableHLO.EmitS (String × List String) := do
      let (c1, m1, v1) ← bnStat oc hh f.c1 f.st1
      let (c2, m2, v2) ← bnStat oc hh f.c2 f.st2
      let (cp, mp, vp) ← bnStat oc hh f.cp f.stp
      pure (c1 ++ c2 ++ cp, [m1, v1, m2, v2, mp, vp])
    let (cSt0, st0) ← bnStat 64 112 nStc F.sst
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
      [⟨"sW", nsW, [64,3,7,7]⟩] ++ (if convBias then [⟨"sbi", nsb, [64]⟩] else []) ++
      [⟨"sg", nsg, [64]⟩, ⟨"sbt", nst, [64]⟩]
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
      -- R34 renders no accumulation variant and no EMA one, so the fifth and sixth components
      -- (the accumulator's and the shadow's output names) are always `none` here. Dropping them
      -- rather than threading them is what keeps every committed R34 artifact byte-identical.
      let (c, nT, nM, nV, _, _) ← optOne opt B replicas g
      adamCode := adamCode ++ c
      thetaN := thetaN ++ [nT]
      mNames := mNames ++ [nM]
      vNames := vNames ++ [nV]
    -- ═══ assemble ═══
    let statCode := cSt0 ++ cSt1 ++ cSt2 ++ cSt3 ++ cSt4 ++ cSt5 ++ cSt6 ++ cSt7 ++ cSt8 ++
      cSt9 ++ cSt10 ++ cSt11 ++ cSt12 ++ cSt13 ++ cSt14 ++ cSt15 ++ cSt16
    let statNames := st0.1 :: st0.2 :: (st1 ++ st2 ++ st3 ++ st4 ++ st5 ++ st6 ++ st7 ++ st8 ++
      st9 ++ st10 ++ st11 ++ st12 ++ st13 ++ st14 ++ st15 ++ st16)
    -- `%loss`: the report-only smoothed CE, at the cotangent's α (`reportSmoothedCeLoss`).
    let lossCode := reportSmoothedCeLoss B nClasses nSm alpha
    let body := F.code ++ cSm ++ cDy ++
      cDgi ++ cWd ++ cbd ++ cDgp ++
      b16.code ++ b15.code ++ b14.code ++ b13.code ++ b12.code ++ b11.code ++ b10.code ++ b9.code ++
      b8.code ++ b7.code ++ b6.code ++ b5.code ++ b4.code ++ b3.code ++ b2.code ++ b1.code ++
      cDmp ++ cDsr ++ cDsn ++ csW ++ csb ++ csg ++ cst ++ statCode
    let pTypes : List String := allPs.map (fun g => ty g.ds)
    let statTypes : List String := (r34StatSigList.map (·.2))
    let retVals := thetaN ++ mNames ++ vNames ++ ["%loss", "%bc1", "%bc2"] ++ statNames
    let retTys  := pTypes ++ pTypes ++ pTypes ++ ["tensor<f32>", "tensor<f32>", "tensor<f32>"] ++ statTypes
    pure <|
      -- With `optLabel = "AdamW"` these are the committed byte sequences character for character;
      -- interpolating a constant changes the source, not the output (the inertness gate checks exactly that).
      (if replicas ≤ 1 then
        s!"    // ── ResNet-34 batch-BN {optLabel} train step: {trainStepHandNote} ──\n"
       else
        s!"    // ── ResNet-34 batch-BN {optLabel} train step, DATA-PARALLEL over {replicas} replicas ──\n" ++
        syncBnBanner "ResNet34SyncTieB.r34_net_syncTiedB"
          "StableHLO.resnet34FwdGraphSyncFull_shard" "ResNet" ++
        (if bf16 then syncBnBf16WgradNote else "")) ++
      zeroBiasPrelude convBias [64, 128, 256, 512] ++ body ++ optConstsB opt wdStr ++ adamCode ++ lossCode ++
      s!"    return {String.intercalate ", " retVals} : {String.intercalate ", " retTys}\n"
  let sigList : List (String × String) := r34SigList nClasses convBias
  let statSig := String.intercalate ", " (r34StatSigList.map (fun (n, t) => s!"{n}i: {t}"))
  let inSig := s!"%x: {ty [B, 3*224*224]}, " ++ packedTrainSig sigList ++ ", " ++ statSig ++
    s!", %onehot: {ty [B, nClasses]}"
  let pTy := sigList.map (·.2)
  let outSig := String.intercalate ", "
    (packedTrainRetTys pTy ++ (r34StatSigList.map (·.2)))
  let inner : String := go.run' (0, [])
  -- The entry name must track the driver's `{slug}_{variant}_train_step` convention, or the shim
  -- refuses the call ("entry mismatch"). `r34AdamVariant` is the single source for the name, the
  -- artifact path, and `LEAN_MLIR_VARIANT`.
  -- `bf16` MUST be passed here, not merely to the block renderers. This function's own
  -- docstring three hundred lines up says why, for `wx` and `clip`: the entry NAME is derived
  -- from the variant, so a flag that reaches the emission but not the name produces an artifact
  -- whose declared entry disagrees with its own path, and the driver refuses the call outright
  -- ("entry mismatch: session holds @…_momdp64_train_step, caller asked …_momdp64bf16_…").
  let fname := s!"{slug}_{r34AdamVariant B replicas opt (bf16 := bf16) (wdStr := wdStr) (alpha := alpha)}_train_step"
  "module @m {\n" ++
  s!"  func.func @{fname}({inSig}) -> ({outSig}) " ++ "{\n" ++
  inner ++
  "  }\n}\n"

end Proofs.StableHLO

-- Regenerate `verified_mlir/resnet34_adam_train_step.mlir` — the BATCHED (`N := B`) AdamW train
-- step as `pretty(provenGraph)`. B=32, nClasses=10, ε=1e-5. **This is the artifact
-- `resnet34-verified-adam{,-xla}` trains on**, and this `#eval` is its ONLY writer: two writers for
-- one artifact is a last-writer-wins race, so any other render of this step goes to its own path.
-- The gates on this render:
--
--   * the numeric tie (`resnet34-adam-tie`) — forward bit-exact, backward norm-rel ≤ 2e-6;
--   * the step bench (`resnet34-adam-bench`) — no cost, despite the extra emitted ops, because
--     XLA's CSE collapses the recomputes.
--
-- To run the tie against another render of this step, pass that render as the first argument.
#eval IO.FS.writeFile "verified_mlir/resnet34_adam_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05")

-- The two Imagenette forwards. `resnet34_fwd` comes from `r34FwdChainB` — the SAME traversal every
-- train step above differentiates — so the net that scores and the net that trains are one graph by
-- construction, and `check_adam_prefix` needs no `KNOWN_SPLIT` entry for this net.
--
-- There is no per-example SGD-inline `resnet34_train_step.mlir`: the batched SGD step
-- (`resnet34_sgd_train_step.mlir`, rendered above at `OptRecipe.sgd`) is what chapter 5's optimizer
-- ladder runs.
#eval IO.FS.writeFile "verified_mlir/resnet34_fwd.mlir"
  (Proofs.StableHLO.resnet34FwdText 32 10 "1.0e-05")

#eval IO.FS.writeFile "verified_mlir/resnet34_fwd_eval.mlir"
  (Proofs.StableHLO.resnet34FwdEvalText 32 10 "1.0e-05")

-- ── §5.6's two ABLATION renders. ────────────────────────────────────────────────────────────────
-- Weight decay and label smoothing are BAKED constants (`optConstsB`'s `%wd`, and the α/K pair in
-- the smoothed-CE cotangent), so unlike warmup, the schedule and augmentation they cannot be
-- ablated by a runtime flag. Each arm is its own artifact, named for what it removes.
-- They differ from `resnet34_adam_train_step.mlir` in CONSTANTS ONLY -- same ops, same shapes,
-- same arity -- which is the property that makes the ablation measure the recipe and not the net.
-- The PATHS are `r34AdamVariant`'s output, not hand-spelled: `wd00` and `ls0000` are what the
-- markers produce, and the entry name inside each file is built from the same call. Spelling the
-- path by hand is how a path and the entry declared inside it come to disagree.
#eval IO.FS.writeFile "verified_mlir/resnet34_adamwd00_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05" (wdStr := "0.0"))

#eval IO.FS.writeFile "verified_mlir/resnet34_adamls0_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05" (alpha := 0.0))

-- ── The bf16 half of §5.6, so every arm has a precision peer. ──────────────────────────────────
-- The point is NOT that bf16 is faster here — at batch 32 on Imagenette it is barely that. It is
-- that the recipe deltas should be the SAME SIZE in both precisions: an ablation that only holds in
-- fp32 is measuring the arithmetic, not the recipe. Chapter 4's Lever 3 makes the same argument on
-- the normalized CIFAR net, and this is its ResNet-scale peer.
-- `bf16` is LAST in the spelling, after even the decay and smoothing marks, which is what keeps
-- every fp32 name above unchanged.
#eval IO.FS.writeFile "verified_mlir/resnet34_adambf16_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05" (bf16 := true))

#eval IO.FS.writeFile "verified_mlir/resnet34_mombf16_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05"
    (opt := Proofs.StableHLO.OptRecipe.heavyBall) (bf16 := true))

#eval IO.FS.writeFile "verified_mlir/resnet34_adamwd00bf16_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05" (wdStr := "0.0") (bf16 := true))

#eval IO.FS.writeFile "verified_mlir/resnet34_adamls0bf16_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05" (alpha := 0.0) (bf16 := true))

-- ── §5.6's optimizer ladder needs a bottom rung. ──────────────────────────────────────────────
-- `.heavyBall` is NOT "AdamW removed" — momentum is most of what an adaptive optimizer buys at
-- this depth, so an arm that swaps one for the other measures the gap between two good optimizers
-- and calls it the optimizer's contribution. `.sgd` is the honest bottom: coupled decay and a step,
-- no velocity, no moments.
#eval IO.FS.writeFile "verified_mlir/resnet34_sgd_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05"
    (opt := Proofs.StableHLO.OptRecipe.sgd))

#eval IO.FS.writeFile "verified_mlir/resnet34_sgdbf16_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05"
    (opt := Proofs.StableHLO.OptRecipe.sgd) (bf16 := true))

-- ── The momentum-BASE ablation (§5.6). ────────────────────────────────────────────────────────
-- Same two baked knobs as the AdamW arms, on the heavy-ball render. The point of re-running the
-- recipe under a NON-adaptive base is that AdamW's per-parameter normalisation absorbs exactly the
-- tuning the other ingredients supply, so an ablation rooted at AdamW may understate every one of
-- them. Rooted at momentum there is nothing absorbing it.
#eval IO.FS.writeFile "verified_mlir/resnet34_momwd00_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05"
    (opt := Proofs.StableHLO.OptRecipe.heavyBall) (wdStr := "0.0"))

#eval IO.FS.writeFile "verified_mlir/resnet34_momls0_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05"
    (opt := Proofs.StableHLO.OptRecipe.heavyBall) (alpha := 0.0))

#guard Proofs.StableHLO.r34AdamVariant 32 1 .heavyBall (wdStr := "0.0") == "momwd00"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .heavyBall (alpha := 0.0) == "momls0"

#guard Proofs.StableHLO.r34AdamVariant 32 1 .sgd == "sgd"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .sgd (bf16 := true) == "sgdbf16"

-- The spellings, pinned the way every other marker here is.
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw (wdStr := "0.0") == "adamwd00"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw (alpha := 0.0) == "adamls0"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw == "adam"
-- bf16 COMPOSES with the two ablation marks and trails both. Pinned, because the composition is
-- exactly what a hand-spelled path would get wrong.
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw (bf16 := true) == "adambf16"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .heavyBall (bf16 := true) == "mombf16"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw (wdStr := "0.0") (bf16 := true) == "adamwd00bf16"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw (alpha := 0.0) (bf16 := true) == "adamls0bf16"

-- The DATA-PARALLEL render, selected at run time by `LEAN_MLIR_VARIANT=adamdp`. Same graph, plus
-- one `all_reduce(add)/N` per parameter gradient before its AdamW triple. The certified renderer is
-- the ONLY writer of both R34 AdamW artifacts.
--
-- SYNC-BN: every BatchNorm layer all-reduces its μ, then its Chan-corrected σ²
-- (`σ²_r + (μ_r − μ)²`), before normalising, and the two dy-reductions before its backward, and the
-- γ gradient reads the same global `x̂` — three more collectives per BN layer per step, each a
-- vector of ≤ 4·oc floats. So `adamdp` (2×32) computes the SAME function as `adam64` (1×64), a
-- known-answer identity `resnet34-syncbn-check` runs; with per-replica BN it would compute the mean
-- of two per-replica batch-32 losses (`dpMeanGrad_ne_globalBatchGrad`).
#eval IO.FS.writeFile "verified_mlir/resnet34_adamdp_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05" 2)

-- The **bs256** render, selected at run time by `LEAN_MLIR_VARIANT=adam256` with
-- `cfg.batchSize := 256`. bs256 is much faster per image on this net and fits on a 7900 XTX;
-- it is also the batch ImageNet wants. `B` is a true parameter of the renderer, so this is the
-- whole change — the graph structure is identical and only the tensor dimensions move.
--
-- It renders to its OWN path rather than re-pointing `resnet34_adam_train_step.mlir`, so the bs32
-- artifact and its tie/bench baselines are untouched. Note the eval forwards are still bs32: train
-- at 256 with `LEAN_MLIR_SKIP_EVAL=1`, or re-render them.
--
-- Gated by `resnet34-batch-check`: feeding 8 identical copies of one bs32 batch makes the
-- batch-BN statistics and the mean-CE cotangent identical to the bs32 render's, so the two must
-- produce the SAME step — an exact known-answer check on the re-render, not a tolerance argument.
#eval IO.FS.writeFile "verified_mlir/resnet34_adam256_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 256 10 "1.0e-05")

-- **bs128 × 2 replicas** — the data-parallel render at a real batch (global 256), the EfficientNet
-- `adamdp128` shape brought to R34. `B` and `replicas` are both true parameters, so
-- this composes the two `#eval`s above with no new renderer code.
--
-- Why this batch: bs256 is much faster per image than bs32 single-device, most of it from
-- amortising the `[θ|m|v]` host↔device round trip over 8× the images — and that transfer is
-- exactly what the DP path pays per replica per step. So 2×128 is where the batch win and the
-- replica win stack rather than fight.
--
-- **The eval forwards are still bs32**, so this variant needs `LEAN_MLIR_SKIP_EVAL=1` — it yields
-- descent and throughput, NOT a validation accuracy. Same caveat `adam256` carries.
#eval IO.FS.writeFile "verified_mlir/resnet34_adamdp128_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 128 10 "1.0e-05" 2)

-- **bs64, SINGLE device** — the single-device peer of `adamdp` (bs32 x 2 = global 64): same
-- global batch, same 147 steps/epoch, same schedule. The DP render is sync-BN, so `2 x 32` IS
-- `1 x 64` to float reduction order (`resnet34-syncbn-check`), and this render is the single-device
-- side of that identity. Under per-replica batch BN the pair would instead measure the
-- BN-splitting effect: with step count and global batch held fixed, the residual IS that effect.
#eval IO.FS.writeFile "verified_mlir/resnet34_adam64_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 64 10 "1.0e-05")

-- **bs128, SINGLE device** — the fourth point on the batch/step-count curve (global 128, 73
-- steps/epoch), and the single-device peer of `adamdp128` (bs128 x 2 = global 256). Together with
-- `adam`, `adam64` and `adamdp128` this brackets the step count 295 / 147 / 73 / 36 at a fixed
-- 80-epoch budget and unscaled LR, which is what isolates "fewer optimizer steps" from every other
-- moving part.
#eval IO.FS.writeFile "verified_mlir/resnet34_adam128_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 128 10 "1.0e-05")

-- **HEAVY-BALL MOMENTUM + coupled L2** — the optimizer `jax/MainResnetImagenet.lean` actually uses,
-- so the verified/XLA path and the Lean→JAX reference can be run as a matched pair rather than as
-- two different experiments. Selected by `LEAN_MLIR_VARIANT=mom`.
--
-- The reference rule, from `Jax/Codegen.lean`'s `.sgd` branch at `hasMomentum`:
--     grads    = g + WD * p          -- COUPLED L2, wd = 1e-4, every param (no wdExclude)
--     velocity = MOMENTUM * v + g    -- μ = 0.9
--     params   = p - lr * velocity   -- heavy-ball
--
-- **This is NOT `momParamF`, and that trap is the reason to read `optOne` before editing here.**
-- The `SgdMomentumStep` family is **Nesterov** (`θ − lr·(g + μ·v')`); the reference steps by `v'`
-- alone. `Proofs.momParam_heavyBall_diff` states the exact difference. Reaching for the
-- momentum-named op compiles, renders, trains, and produces a *different optimizer* than the thing
-- this artifact exists to be 1:1 with — silently.
--
-- Rendered at **bs32 / 10 classes**, i.e. the direct peer of `adam`, ON PURPOSE: that is the shape
-- the existing gates and the Imagenette trainer can exercise. The ImageNet shape is
-- `B := 256, nClasses := 1000` and is its own `#eval` below — `B` and `nClasses` are both true
-- renderer parameters, so no renderer change is involved.
--
-- **Gated by `r34-mom-tie`** (`tests/TestMomTie.lean`): from `m = v = 0` the `adam` render's
-- `m' = (1−β₁)·g` recovers the gradient exactly (`g = 10·m'`), so `v'` here must equal
-- `g + wd·θ` and `θ'` must equal `θ − lr·v'` on shared inputs — a cross-render known answer, not a
-- tolerance argument. Its control requires the Nesterov prediction to miss.
#eval IO.FS.writeFile "verified_mlir/resnet34_mom_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 32 10 "1.0e-05" 1 .heavyBall)

-- ══ ImageNet-1k: the three artifacts the reference-pairing run needs ══
--
-- `B := 256, nClasses := 1000`, heavy-ball — i.e. the `jax/MainResnetImagenet.lean` recipe, on the
-- certified renderer. **No renderer change is needed for any of this**: `B`, `nClasses`, `opt` and
-- `slug` are all parameters, which is the whole reason this is three `#eval`s and not a project.
--
-- **The slug is `resnet34in`, NOT `resnet34`, and that is load-bearing.** The forward artifacts
-- carry no variant in their path (`<slug>_fwd.mlir`), so rendering a 1000-class forward under the
-- `resnet34` slug would OVERWRITE the 10-class Imagenette one that five committed runs and the
-- prefix audit depend on — silently, and with a graph of a different arity. A distinct slug is what
-- keeps the two nets' artifacts disjoint; `slug` defaults to "resnet34" so every existing render is
-- byte-identical (checked).
--
-- Batch 256 on the forwards too, matching the shim's val batch: 195 batches × 256 = 49,920 images
-- after tfds `drop_remainder`, which is the count the JAX reference reports scoring.
#eval IO.FS.writeFile "verified_mlir/resnet34in_mom256_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 256 1000 "1.0e-05" 1
    Proofs.StableHLO.OptRecipe.heavyBall "resnet34in")

-- **The 4-GPU data-parallel peer**, at `B := 64` PER REPLICA so the GLOBAL batch is 64×4 = 256 —
-- the same global batch, the same 5004 steps/epoch and the same recipe as the single-device
-- `mom256` render above, and the same batch the JAX reference trains at (4×64, see
-- `jax/runs/r34_imagenet_bf16_90ep/RESULTS.md`). That is deliberate: rendering 256 per replica
-- would make the global batch 1024 and silently change the recipe, so the run would no longer be
-- comparable to either peer without an LR rescale. Matching the reference is what makes the
-- resulting wall-clock a like-for-like number rather than a new experiment.
--
-- Nothing in the renderer changes for this: `optOne` takes `replicas` and calls
-- `emitGradAllReduce`, exactly as for mnv2. This is one `#eval`.
#eval IO.FS.writeFile "verified_mlir/resnet34in_momdp64_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 64 1000 "1.0e-05" 4
    Proofs.StableHLO.OptRecipe.heavyBall "resnet34in")

-- **The bf16 peer of the render above** — `momdp64bf16`, byte-for-byte the same graph except
-- that every conv (stem, both block convs, the 1×1 projection, both dgrads, every wgrad) is its
-- bf16 twin: bf16 operands, a **bf16-TYPED** convolution result, then a convert back to f32.
-- Everything else — BN, the residual adds, the loss, the heavy-ball tail, the master weights —
-- stays f32, which is what `jax/MainResnetImagenet.lean` does and why its bf16 arm converges to
-- the same place as its f32 arm.
--
-- The bf16-typed RESULT is the load-bearing part and is not cosmetic. A conv with bf16 operands
-- and an f32 result has its converts deleted by XLA under excess precision — cuDNN then gets f32
-- parameters and the graph runs entirely in fp32 while still *reading* as mixed precision. This
-- is measured, not feared; `BatchableOp.convBf16` carries the note. Verify with the operand dtypes
-- in the OPTIMIZED HLO, never by grepping the op line, which shows only the result type.
#eval IO.FS.writeFile "verified_mlir/resnet34in_momdp64bf16_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 64 1000 "1.0e-05" 4
    Proofs.StableHLO.OptRecipe.heavyBall "resnet34in" false true)

-- **The 2-GPU data-parallel peer**, at `B := 128` PER REPLICA so the global batch is 128×2 = 256 —
-- the same global batch, the same 5004 steps/epoch and the same recipe as `mom256` and `momdp64`
-- above. That is the whole point: the phase-4 table in §5.7 compares wall-clock across boxes, so a
-- 2-card row is only readable if the recipe underneath it is the one the 4-card row ran. Rendering
-- 64 per replica would be the smaller change and the wrong number — global batch 128, 10,008
-- steps/epoch, and a run that needs an LR rescale before it means anything.
--
-- `B` and `replicas` are both true parameters, so this composes the two `#eval`s above and adds no
-- renderer code — the same "one `#eval`" as the 4-GPU peer. bs128/card fits a 24 GB 7900 XTX with
-- room: the single-device `mom256` render above is bs256 and fits on one.
#eval IO.FS.writeFile "verified_mlir/resnet34in_momdp128_train_step.mlir"
  (Proofs.StableHLO.resnet34AdamTrainStepText 128 1000 "1.0e-05" 2
    Proofs.StableHLO.OptRecipe.heavyBall "resnet34in")

-- Both ImageNet forwards. `resnet34in_fwd` comes from the batched chain, so it is batch BN like
-- every `resnet34in_*` train step above; a per-example-BN forward here would be the split
-- `check_adam_prefix` guards for the 10-class pair, at a scale that audit does not look at (its
-- PAIRS list has only the Imagenette names). The eval forward is world-agnostic.
#eval IO.FS.writeFile "verified_mlir/resnet34in_fwd.mlir"
  (Proofs.StableHLO.resnet34FwdText 256 1000 "1.0e-05" "resnet34in")

#eval IO.FS.writeFile "verified_mlir/resnet34in_fwd_eval.mlir"
  (Proofs.StableHLO.resnet34FwdEvalText 256 1000 "1.0e-05" "resnet34in")

-- Pin the seven literal artifact paths above against the name the renderer actually emits. If a
-- variant is renamed, this fails at `lake build` instead of at run time as an "entry mismatch".
#guard Proofs.StableHLO.r34AdamVariant 32 1 == "adam"
#guard Proofs.StableHLO.r34AdamVariant 32 2 == "adamdp"
#guard Proofs.StableHLO.r34AdamVariant 256 1 == "adam256"
#guard Proofs.StableHLO.r34AdamVariant 128 2 == "adamdp128"
#guard Proofs.StableHLO.r34AdamVariant 64 1 == "adam64"
#guard Proofs.StableHLO.r34AdamVariant 128 1 == "adam128"
-- The bf16 marker, pinned the way every other marker here is. These `#guard`s pin the SPELLING but
-- not the wiring: `resnet34AdamTrainStepText` derives its entry name from this function, and a
-- call WITHOUT the flag writes the artifact to `…momdp64bf16_train_step.mlir` while declaring
-- `@resnet34in_momdp64_train_step` inside, and the driver refuses at load with an entry mismatch.
-- The flag must reach both.
#guard Proofs.StableHLO.r34AdamVariant 64 4 Proofs.StableHLO.OptRecipe.heavyBall
         false false false "" true == "momdp64bf16"
#guard Proofs.StableHLO.r34AdamVariant 64 4 Proofs.StableHLO.OptRecipe.heavyBall == "momdp64"
-- And the marker must not collide with the DRIVER's variant predicates, which read the same
-- string to decide the blob layout. `cdOn` is the dangerous one: it is a substring test for "do",
-- and a slug that tripped it would silently add a dropout region to the checkpoint.
-- These are `cdOn`/`accOn`/`emaOn` (`Verified.Train`) evaluated on the bf16 slug: each is a
-- SUBSTRING test, and a false positive changes `nRegions`/`nScalars` — i.e. the checkpoint layout —
-- with no error anywhere.
#guard !"momdp64bf16".contains "do"
#guard !"momdp64bf16".contains "acc"
#guard !"momdp64bf16".startsWith "ema"
-- The optimizer axis. `.adamw` must keep every legacy name unchanged — that is what makes the
-- threading a no-op for the six artifacts above — and `.heavyBall` gets its own.
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw == "adam"
#guard Proofs.StableHLO.r34AdamVariant 32 1 .heavyBall == "mom"
#guard Proofs.StableHLO.r34AdamVariant 32 2 .heavyBall == "momdp"
#guard Proofs.StableHLO.r34AdamVariant 256 1 .heavyBall == "mom256"
-- The 4-GPU ImageNet render: per-replica 64, so the slug carries 64 and NOT the 256 global batch.
-- Pinned because the driver derives `LEAN_MLIR_VARIANT` from `(B, replicas)` while the `#eval`
-- above hardcodes the path, and an "entry mismatch" at run time is the failure they drift into.
#guard Proofs.StableHLO.r34AdamVariant 64 4 .heavyBall == "momdp64"
-- The 2-GPU ImageNet render. Same rule, and worth pinning separately: `momdp64` and `momdp128`
-- differ only in the per-replica batch, so a slug that dropped `B` would collide two artifacts
-- rendered at different replica counts onto one path — a last-writer-wins race.
#guard Proofs.StableHLO.r34AdamVariant 128 2 .heavyBall == "momdp128"
-- **THE TWO TRAILING FLAG AXES ARE INERT ON EVERY NAME ABOVE**, and that is what the defaults
-- buy: R34 renders no `wx` and no `clip` artifact, so passing them explicitly as `false` must
-- reproduce the legacy spellings character for character. `ResNet50RenderB.lean`'s guards pin the
-- ON spellings, since R50 is the net that renders them.
#guard Proofs.StableHLO.r34AdamVariant 32 1 .adamw false false == "adam"
#guard Proofs.StableHLO.r34AdamVariant 64 4 .heavyBall false false == "momdp64"
-- And the ORDER, pinned here beside the function that decides it rather than only at the call
-- site: `wx` then `clip`, both after the batch. Getting this backwards renders an artifact whose
-- declared entry disagrees with its path, which the shim reports only as "entry mismatch".
#guard Proofs.StableHLO.r34AdamVariant 64 1 .adamw true true == "adam64wxclip"

-- **THE CLIP'S TWO BAKED CONSTANTS, pinned against the theorem that licenses them.**
-- `Proofs.clipFactor_accum` says `min(1, kc/(√(k²s) + kε)) = min(1, c/(√s + ε))`, i.e. that folding
-- the norm on the accumulated SUM and scaling BOTH constants by `k` reproduces the reference's clip
-- of the MEAN exactly. These are that identity's `k = 8` instance as the render actually emits it,
-- and they are the line a reader checks when `dense<8.000000000000>` looks like a wrong threshold.
#guard Proofs.StableHLO.clipNormStr 1.0 8 == "8.000000000000"
#guard Proofs.StableHLO.clipEpsStr 8 == "0.000008000000"
-- `k = 1` — no accumulation — must leave both at the reference's own values, which is what makes
-- the scaling inert on a non-accumulating clip render.
#guard Proofs.StableHLO.clipNormStr 1.0 1 == "1.000000000000"
#guard Proofs.StableHLO.clipEpsStr 1 == "0.000001000000"
-- and `optAccumK` is the SINGLE source of the `k` those two read, so pin that it agrees with the
-- constructor rather than being a second parse of the variant string.
#guard Proofs.StableHLO.optAccumK (Proofs.StableHLO.OptRecipe.lambAccum 8) == 8
#guard Proofs.StableHLO.optAccumK (Proofs.StableHLO.OptRecipe.adamwAccum 4) == 4
#guard Proofs.StableHLO.optAccumK Proofs.StableHLO.OptRecipe.lamb == 1
#guard Proofs.StableHLO.optAccumK Proofs.StableHLO.OptRecipe.adamw == 1
