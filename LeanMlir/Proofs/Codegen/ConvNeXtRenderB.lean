import LeanMlir.Proofs.Codegen.ConvNeXtRender

/-! # ConvNeXt at the BATCHED index `N := B` — forward, backward and AdamW renders

The ConvNeXt peer of `ResNet34RenderB` / `MobileNetV2RenderB`, and the reason it exists is
stochastic depth: **the drop mask is per-EXAMPLE**, and in the per-example-indexed render
(`ConvNeXtRender`) a node denotes ONE example — `pretty B` lifts it across the batch, so the node
cannot see `j`. `dropPathB` needs its operand at index `B·n`, which is what this file's chain
produces.

**The trap this closes is that the wrong thing TYPECHECKS.** `pretty B` already emits
`tensor<B×n>`, so a `broadcast_in_dim %mask, dims = [0]` + multiply against a per-example node
compiles, trains and descends — with no faithful `den` behind it. Every node below is instead a
`batchOp`/`*B` form whose `den` is `batchMap N (…)` or `batchMapAux N (…)`, i.e. honest about which
index is the batch.

**What this file writes — every ConvNeXt artifact but one.** The forwards (drop-free and
stochastic-depth), and the AdamW/EMA train steps through `ConvNeXtRender.convNextAdamTrainStepFaithful`
with this file's traversal, at ConvNeXt-T/S/B, Imagenette and ImageNet, f32 and bf16. Only the
SGD-inline `convnext_train_step.mlir` is written by `ConvNeXtRender`: this traversal has no
fused-SGD arm, and `ConvNeXtStepTie`'s 182-parameter tie is stated at those bytes. The Proofs tier
for this chain is [`Nets/ConvNeXt/ConvNeXtFoldGB.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/LeanMlir/Proofs/Nets/ConvNeXt/ConvNeXtFoldGB.lean).

**The gate** (`lake build convnext-fwd-b-tie`): the per-example chain and this one must emit
**byte-identical** forwards, and train steps that differ on the conv-VJP `transpose`/`reverse` pair
and nothing else (commuting ops on disjoint axes). That is available *because* every batched form
was built to emit its per-example peer's text byte-for-byte
([`tests/TestBatchedEmitTie.lean`](https://github.com/brettkoonce/lean4-mlir/blob/main/tests/TestBatchedEmitTie.lean) checks each form). So the whole-net statement is the
per-form statement composed — and if it ever fails, the tie file localises which form did it in
one run. The committed bytes are this chain's, so the gate renders the per-example chain and
compares it against them.
-/

open Proofs Proofs.StableHLO

namespace Proofs.StableHLO

/-! ## The batched shape helpers

`reassoc`/`unassoc` in the per-example renderer cast `SHlo (c*h*h) ↔ SHlo (c*(h*h))`. At the
batched index the same cast has to happen UNDER `N * ·`, which is `bnBatchLA`'s existing move
(`congrArg (N * ·) (Nat.mul_assoc …)`). It is a reindex, not a function change — the emitted text
is unaffected, since `skel` never sees the index. -/
private def reassocB {N c h : Nat} (e : SHlo (N*(c*h*h))) : SHlo (N*(c*(h*h))) :=
  castIdx (la_assoc N c h h) e


/- **The per-replica batch — a PARAMETER, not a private constant.**

   See the long note at the top of `ConvNeXtRender` for WHY (the ImageNet job must run at its
   reference's global 256 for the pair to be a pair); this note records what is different on THIS
   side.

   **Here the batch is a TYPE INDEX, not just a serialization width.** In the per-example file
   `cBS` is only ever `pretty`'s first argument. Here it also appears in `Vec (bB*(c*h*h))`,
   `(N := bB)` and `Vec (bB*nClasses)` — the shapes the batched forms are INDEXED by. That is why
   this file is the one that actually has to be right: a wrong width in the per-example file is a
   wrong annotation, and a wrong width here is a different graph.

   Threading shape, same as the per-example file: private helpers take it FIRST with NO default
   (the compiler enumerates the sites); public entry points take it LAST, defaulted to 32, so
   every committed render, every `tests/` caller and both ConvNeXt-S/B forwards stay
   BYTE-IDENTICAL. Verified by regenerating: at the default the diff is empty.

   **The two files MUST MOVE TOGETHER and `convNextAdamTrainStepFaithfulB` is where they meet.**
   That function spells the batch ONCE and hands it to both halves — `convNextBackAllB` (the BODY,
   at `N := bB`) and `convNextAdamTrainStepFaithful` (the WRAPPER, via `cBS`: `%x`'s declared
   shape, the `%bsc` loss divisor, the drop-path signature). It is the same discipline `sd`, `V`
   and `bf16` get here, and for the same reason: two halves that a caller could set
   independently is the defect. A disagreement is LOUD —
   the wrapper declares `tensor<B₁×3×224×224>` over a body computing at `N := B₂` and the lowerer
   rejects the module — but loud at the lowerer is still later than loud at the type checker. -/
/-- **ConvNeXt-T / ImageNet's per-replica batch, spelled ONCE for the whole `convnextin_*`
    family.**

    **64, not the 32 every other ConvNeXt render uses**, and the reason is a PAIRING fact rather
    than a throughput one. This net's JAX reference trains at **global 256 = 4 × 64** (its own
    banner: `batch 256 (4x64) · SPE 5004`). At 4 × 32 the verified job would run half the batch,
    twice the updates, and an LR off the linear-scaling rule: a RECIPE difference, not a lowering
    difference, and it would forfeit the only thing this pair is in the book for: ConvNeXt has **no
    BatchNorm**, so a tie here says the LOWERER is clean and an offset here implicates the lowerer
    or the feed fleet-wide. Neither reading survives if the batch does not match.

    Spelled once because it has to reach every `convnextin_*` `#eval` AND both halves of every
    train step (body at `N := bB`, wrapper via `cBS`).

    ConvNeXt-S's and -B's EMA pair renders and eval forwards are at this batch too; their `adam*`
    train steps stay at the 32 default. A job's `LEAN_MLIR_BATCH` must match its render —
    the `PRECHECK` in `scripts/jobs/cnx-default-4gpu.conf` greps the artifact for its own baked
    batch and is the template for that. -/
def cnxInBS : Nat := 64

/-- Eps and the stage table — read from the per-example renderer's own constants where they
    are public, restated where they are `private`. Restated, not re-derived: if these drift the
    byte tie fails loudly, which is the point of tying against the committed artifact rather than
    against a second copy of the shapes. The batch is a parameter, not one of these constants —
    see the note above. -/
private def bEPS : String := "1.0e-6"
private def bSpats  : Array Nat := #[56, 28, 14, 7]

-- The three sizes, RESTATED (not imported) and `#guard`ed against `ConvNeXtRender`'s — the same
-- discipline `bB`/`bEPS` carry, kept because the byte tie is meant to test the RENDER rather than
-- the constants. Restating a RECORD is what makes that discipline work when a size is
-- two tables: an `#[3,3,27,3]` that drifted into the wrong `dims` would be caught by one `==`.
private def bTiny  : CnxDims := { depths := #v[3, 3,  9, 3], dims := #v[ 96, 192, 384,  768] }
private def bSmall : CnxDims := { depths := #v[3, 3, 27, 3], dims := #v[ 96, 192, 384,  768] }
private def bBase  : CnxDims := { depths := #v[3, 3, 27, 3], dims := #v[128, 256, 512, 1024] }
#guard bTiny  == cnxTiny
#guard bSmall == cnxSmall
#guard bBase  == cnxBase

-- The stochastic-depth site table (`cnxDropTotal`, `cnxBlockIdx`, `cnxDropSig`) lives in
-- `ConvNeXtRender.lean` — the train-step renderer there owns the signature and the variant name, and
-- this file imports it rather than the reverse. It is derived from THAT file's `cnxTiny`; this guard
-- is what says the two stage tables have not drifted, which the ramp indices below depend on.
#guard bTiny.depths.foldl (· + ·) 0 == cnxDropTotal

/-! ## The sites, one per per-example site in `ConvNeXtRender.lean` -/

/-- One **channel-LN forward** site, batched: transpose to `[h·w, c]`, normalise each spatial row
    over its channels at the scalar identities `%one`/`%zero`, apply the real `[c]` affine,
    transpose back. Five `batchOp`s where the per-example peer has five bare nodes.

    `m := h*h` is the SPATIAL row count PER EXAMPLE and `N := bB` is the batch. Collapsing those
    two into one index is exactly the defect this file exists to remove, and it is why the
    descriptors carry `(m, n)` internally instead of reading the SHlo index. -/
private def lnFwdSiteB (bB : Nat) (gN btN xin : String) (c h : Nat) :
    StateM Proofs.StableHLO.EmitS (String × String) := do
    let (k1, t)  ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                  (reassocB (.operand xin (0 : Vec (bB*(c*h*h))))))
    let (k2, n)  ← pretty bB (.batchOp (N := bB)
                                  (.lnRow (m := h*h) (n := c) "%one" "%zero" bEPS 0 1 0)
                                  (.operand t (0 : Vec (bB*(h*h*c)))))
    let (k3, sc) ← pretty bB (.batchOp (N := bB) (.rowScale (m := h*h) (n := c) gN (0 : Vec c))
                                  (.operand n (0 : Vec (bB*(h*h*c)))))
    let (k4, bi) ← pretty bB (.batchOp (N := bB) (.rowBias (m := h*h) (n := c) btN (0 : Vec c))
                                  (.operand sc (0 : Vec (bB*(h*h*c)))))
    let (k5, o)  ← pretty bB (.batchOp (N := bB) (.transpose (m := h*h) (n := c))
                                  (.operand bi (0 : Vec (bB*(h*h*c)))))
    pure (k1 ++ k2 ++ k3 ++ k4 ++ k5, o)

/-- **The HEAD LN, batched-index peer** — `lnFwdSiteB` with the transposes deleted, at `m = 1`:
    after GAP the tensor is one `[d]` row per example. Must stay op-for-op with
    `ConvNeXtRender.headLnFwdSite`, because `convnext-fwd-b-tie` asserts the two renderers emit the
    same bytes. -/
private def headLnFwdSiteB (bB : Nat) (gN btN xin : String) (d : Nat) :
    StateM Proofs.StableHLO.EmitS (String × String) := do
    let (k1, n)  ← pretty bB (.batchOp (N := bB)
                                (.lnRow (m := 1) (n := d) "%one" "%zero" bEPS 0 1 0)
                                (.operand xin (0 : Vec (bB*(1*d)))))
    let (k2, sc) ← pretty bB (.batchOp (N := bB) (.rowScale (m := 1) (n := d) gN (0 : Vec d))
                                (.operand n (0 : Vec (bB*(1*d)))))
    let (k3, o)  ← pretty bB (.batchOp (N := bB) (.rowBias (m := 1) (n := d) btN (0 : Vec d))
                                (.operand sc (0 : Vec (bB*(1*d)))))
    pure (k1 ++ k2 ++ k3, o)

/-- The head LN's **input-VJP**, batched peer. -/
private def headLnBackSiteB (bB : Nat) (gN xName cot : String) (d : Nat) :
    StateM Proofs.StableHLO.EmitS (String × String) := do
    let (k1, da) ← pretty bB (.batchOp (N := bB) (.rowScale (m := 1) (n := d) gN (0 : Vec d))
                                 (.operand cot (0 : Vec (bB*(1*d)))))
    let (k2, o)  ← pretty bB (.lnRowBackB (N := bB) (m := 1) (n := d) "%one" xName bEPS 0 1 0
                                 (.operand da (0 : Vec (bB*(1*d)))))
    pure (k1 ++ k2, o)

/-- The head LN's **γ / β tails**, batched peers — both contract the batch. -/
-- `_gN` is unused and that is correct rather than an oversight: the ADAM γ tail is a pure
-- GRADIENT (`veclnGammaGradB`), which does not read the current γ — only the SGD peer does, and
-- this batched renderer has no SGD arm. Named `_` so the linter says so instead of the reader
-- having to work it out.
private def headLnGammaTailB (bB : Nat) (_gN xName cot : String) (d : Nat) :
    StateM Proofs.StableHLO.EmitS (String × String) :=
  pretty bB (.veclnGammaGradB (N := bB) (R := 1) (D := d) xName bEPS 0
                (0 : Vec (bB*(1*d))) (.operand cot (0 : Vec (bB*(1*d)))))

private def headLnBetaTailB (bB : Nat) (cot : String) (d : Nat) :
    StateM Proofs.StableHLO.EmitS (String × String) :=
  pretty bB (.rowDenseBiasGradB (N := bB) (R := 1) (c := d)
                (.operand cot (0 : Vec (bB*(1*d)))))

/-- One **ConvNeXt block** forward, batched: depthwise 7×7 → channel-LN → 1×1 expand → GELU →
    1×1 project → LayerScale → [drop] → `+ skip`. The residual add is `addVB`, the binary batched
    form.

    `drop = some i` puts the stochastic-depth site at ramp index `i` **on the residual branch**, i.e.
    between `layerScaleCh` and the `addVB`. **That placement is the whole correctness question and
    the obvious gate is blind to it** — at an all-ones mask `1 ⊙ (branch + x) = branch + x` exactly,
    so a site on the block OUTPUT is the same function bit-for-bit and every endpoint gate passes on
    it (measured on EfficientNet). `scripts/probes/misplace_drop_sites.py` builds
    exactly that render — same SSA names, order, types and line count — and it is the control that
    licenses believing any green run here.

    At `drop = none` **no `pretty` call happens**, so the fresh-name counter does not move and the
    drop-free chain re-renders byte-identically. That is what keeps `convnext-fwd-b-tie` and the
    committed artifacts free of this feature. -/
private def fwdBlockB (bB : Nat) (pfx xin : String) (c e h : Nat) (drop : Option Nat := none)
    -- TRAILING and defaulted, the `wx`/`clip`/`sd` idiom: every existing call site re-renders
    -- byte-identically.
    (bf16 : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × FNames) := do
  let (k1, d) ← pretty bB (.batchOp (N := bB)
      (.depthwiseAt bf16 (h := h) (w := h) zrnd s!"%{pfx}dW" s!"%{pfx}db"
          (0 : DepthwiseKernel c 7 7) 0)
      (.operand xin 0))
  let (k2, n) ← lnFwdSiteB bB s!"%{pfx}ng" s!"%{pfx}nbt" d c h
  -- The block's 1×1s are `.conv` at `kH = kW = 1`, so they take `convBf16` — NOT a new matmul op.
  -- The only true matmul is the classifier head, which stays f32 like every other net's.
  let (k3, e') ← pretty bB (.batchOp (N := bB)
      (.convAt bf16 (h := h) (w := h) zrnd s!"%{pfx}eW" s!"%{pfx}eb" (0 : Kernel4 e c 1 1) 0)
      (.operand n 0))
  let (k4, g) ← pretty bB (.batchOp (N := bB) (.gelu (n := e*h*h))
      (.operand e' (0 : Vec (bB*(e*h*h)))))
  let (k5, p) ← pretty bB (.batchOp (N := bB)
      (.convAt bf16 (h := h) (w := h) zrnd s!"%{pfx}pW" s!"%{pfx}pb" (0 : Kernel4 c e 1 1) 0)
      (.operand g 0))
  let (k6, ls) ← pretty bB (.batchOp (N := bB)
      (.layerScaleCh (h := h) (w := h) s!"%{pfx}lg" (0 : Vec c)) (.operand p 0))
  let (kD, br) ← match drop with
    | some i => pretty bB (.dropPathB (N := bB) (n := c*h*h) (dpName i) (fun _ => 0 : Vec bB)
                             (.operand ls (0 : Vec (bB*(c*h*h)))))
    | none   => pure ("", ls)
  let (k7, bout) ← pretty bB (.addVB (.operand br (0 : Vec (bB*(c*h*h)))) (.operand xin 0))
  pure (k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ kD ++ k7, ⟨xin, d, n, e', g, p, bout⟩)

/-- One **downsample** forward, batched: channel-LN then 2×2/s2 conv. -/
private def fwdDownB (bB : Nat) (pfx xin : String) (ci co h2 : Nat) (bf16 : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × String × String) := do
  let (k1, n) ← lnFwdSiteB bB s!"%{pfx}ng" s!"%{pfx}nbt" xin ci (2*h2)
  -- SYMMETRIC pad (`convStrided`, not `convStridedXla`) — ConvNeXt is torchvision-origin. Both
  -- spellings give the same output size at every kernel, so only a forward tie separates them.
  let (k2, o) ← pretty bB (.batchOp (N := bB)
      (.convStridedAt bf16 (h := h2) (w := h2) zrnd s!"%{pfx}W" s!"%{pfx}b"
          (0 : Kernel4 co ci 2 2) 0)
      (.operand n 0))
  pure (k1 ++ k2, n, o)

/-- **The full ConvNeXt-T `[3,3,9,3]` forward at the batched index.** Node for node the same chain
    `convNextFwdChain` emits — 4×4/s4 patchify stem (3→96, 224→56) → stem channel-LN → 4 stages at
    56/28/14/7 with 2×2/s2 downsamples between → GAP(7×7) → dense(768→nClasses).

    Every node is a `batchOp`/`*B` form, so `den` is a `batchMap`/`batchMapAux` at `N := bB` and
    the batch is an index of the AST rather than a number only `pretty` knows. That is the entire
    content of the move; the emitted text is unchanged, which the tie checks.

    `sd := true` adds the 18 stochastic-depth sites, one per block, at ramp index `cnxBlockIdx si j`
    — and the sites are emitted in the FORWARD as well as the train step deliberately: at eval the driver supplies an all-ones mask, so they are the exact
    identity, and the `forward ⊂ train-step` prefix audit keeps a partner for the SD render instead
    of quietly not covering it. -/
def convNextFwdChainB (nClasses : Nat := 10) (sd : Bool := false)
    (V : CnxDims := bTiny) (bf16 : Bool := false)
    -- THE PER-REPLICA BATCH — trailing + defaulted, see the note at the top of this file.
    (bB : Nat := 32)
    -- `f`, the FINAL feature side: 7 at the 224 input every committed artifact uses, 9 at 288
    -- (timm's test size for `convnext_tiny.fb_in1k`). The stages run at 8f/4f/2f/f (`bSpats` · f/7),
    -- so the default is byte-identical. Eval forwards only; the train steps stay at 224.
    (f : Nat := 7) : StateM Proofs.StableHLO.EmitS CFwd := do
  -- **The 4×4/s4 patchify stem — one of ConvNeXt's two genuinely new bf16 ops.** Its emit keeps
  -- `convStride4`'s pad-one-less rule (`[[0,0]]` at k=4), which is NOT the symmetric pad every
  -- other forward conv uses; `BatchableOp.convStride4Bf16` carries the note.
  let (cS, stemC) ← pretty bB (.batchOp (N := bB)
      (.convStride4At bf16 (h := 8*f) (w := 8*f) zrnd "%psW" "%psb"
          (0 : Kernel4 (V.dims[0]!) 3 4 4) 0)
      (.operand "%x" (0 : Vec (bB*(3*(2*(2*(8*f)))*(2*(2*(8*f))))))))
  let (cSln, stem) ← lnFwdSiteB bB "%psng" "%psnbt" stemC V.dims[0]! (8*f)
  let mut fwd := cS ++ cSln
  let mut cur := stem
  let mut blksAll : Array (Array FNames) := #[]
  let mut downLn : Array String := #[]
  let mut downIn : Array String := #[]
  for si in [0:4] do
    let c := V.dims[si]!; let e := 4 * c; let h := bSpats[si]! * f / 7
    let mut blks : Array FNames := #[]
    for j in [0:V.depths[si]!] do
      -- `cnxBlockIdx si j V`, NOT `j`: the ramp counts blocks over the whole net (denominator 17
      -- at T, 35 at S). The backward calls the same function walking the stages in reverse, which is
      -- why it is a function of `(si, j)` rather than a counter either loop carries.
      -- AND `D` MUST BE PASSED. Dropping it takes the ConvNeXt-T default silently: at S that
      -- pairs stage 3's 27 blocks with T's stage-3 numbering and stage 4 with indices 15..17 that
      -- another stage already owns — duplicate mask sites on a graph that compiles and descends.
      let (code, bn) ← fwdBlockB bB s!"s{si}b{j}" cur c e h
                          (if sd then some (cnxBlockIdx si j V) else none) bf16
      fwd := fwd ++ code; cur := bn.bout; blks := blks.push bn
    blksAll := blksAll.push blks
    if si < 3 then
      downIn := downIn.push cur
      let (code, n, o) ← fwdDownB bB s!"d{si}" cur c V.dims[si+1]! (bSpats[si+1]! * f / 7) bf16
      fwd := fwd ++ code; downLn := downLn.push n; cur := o
  let (cG, gap) ← pretty bB (.batchOp (N := bB) (.gap (c := V.dims[3]!) (h := f) (w := f))
      (.operand cur 0))
  -- head LN — the per-example peer of `ConvNeXtRender`'s.
  let (cHn, hn) ← headLnFwdSiteB bB "%hng" "%hnbt" gap V.dims[3]!
  let (cLog, logits) ← pretty bB (.batchOp (N := bB)
      (.dense "%Wd" "%bd" (0 : Mat (V.dims[3]!) nClasses) 0) (.operand hn 0))
  pure { code := fwd ++ cG ++ cHn ++ cLog, blksAll := blksAll, downLn := downLn, downIn := downIn,
         gap := gap, stemC := stemC, hn := hn, logits := logits }

/-- **`@convnext_fwd_b`** — the batched-index peer of `convNextFwdFaithfulV`, same signature
    (182 parameters at ConvNeXt-T) and same `%x`. This WRITES `convnext_fwd`, `convnextin_fwd`,
    `convnextsin_fwd` and `convnextbin_fwd` (the `#eval`s at the bottom of this file);
    `convnext-fwd-b-tie` renders the per-example chain against these bytes. -/
def convNextFwdRenderB (funcName : String := "convnext_fwd_b") (nClasses : Nat := 10)
    (banner : String :=
      "    // ── ConvNeXt-T forward at the BATCHED index N := B: every op is pretty(batchOp …) except the %one/%zero LayerNorm constants ──\n")
    -- TRAILING: a parameter inserted mid-list captures an existing positional argument at every
    -- call site.
    (sd : Bool := false)
    (V : CnxDims := bTiny)
    -- bf16, TRAILING per the same rule. No bf16 FORWARD artifact is written — this
    -- parameter exists so the forward chain the train step differentiates is one function, not
    -- two. A bf16 eval forward would need its own prefix partner and its own gate; the train step
    -- is where the payoff is.
    (bf16 : Bool := false)
    -- THE PER-REPLICA BATCH — trailing + defaulted, see the note at the top of this file.
    (bB : Nat := 32)
    -- the input side (224, or timm's test size); a multiple of 32 (the final side is s/32)
    (s : Nat := 224)
    : String := Id.run do
  let F : CFwd := (convNextFwdChainB nClasses sd V bf16 (bB := bB) (f := s / 32)).run' (0, [])
  let body := F.code; let logits := F.logits
  let argSig := String.intercalate ", "
    (("%x: " ++ ty [bB, 3*s*s]) ::
      (cnxAllParams nClasses V).map (fun (nm, d) => s!"%{nm}: {ty d}"))
    -- The mask inputs go LAST, after every parameter, matching the train step's placement and the
    -- driver's blob layout. Anywhere else and they capture an existing positional slot, silently
    -- until the driver mis-walks the blob.
    ++ cnxDropSig bB sd V
  return "module @m {\n" ++ s!"  func.func @{funcName}({argSig}) -> {ty [bB,nClasses]} " ++ "{\n" ++
    banner ++
    chLnPrelude ++ body ++
    s!"    return {logits} : {ty [bB,nClasses]}\n" ++ "  }\n}\n"

/-! ## The backward

Every site below is the per-example site with its node swapped for the batched form. Two of them
are where increments 2 and 3's constructors earn their keep, and both are cases the emit tie alone
cannot judge:

* `lnRowBackB` takes the whole-batch saved LN input and hands example `k` `batchSlice k x`
  (`batchMapAux`). A descriptor here would hand example 0's activation to all `N` — same types,
  same bytes, different function (`den_lnRowBackB_per_example`).
* `veclnGammaGradB` / `rowDenseBiasGradB` contract the batch **and** the spatial rows. The
  per-example peers contract only the rows, and the two spellings agree at `N = 1`
  (`den_rowDenseBiasGradB_at_one`) — so a render that dropped the batch sum would pass a
  one-example check. -/

/-- One **channel-LN input-VJP** site, batched. -/
private def lnBackSiteB (bB : Nat) (gN xName cot : String) (c h : Nat) :
    StateM Proofs.StableHLO.EmitS (String × String) := do
    let (k1, xT)  ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                   (reassocB (.operand xName (0 : Vec (bB*(c*h*h))))))
    let (k2, dT)  ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                   (reassocB (.operand cot (0 : Vec (bB*(c*h*h))))))
    let (k3, da)  ← pretty bB (.batchOp (N := bB) (.rowScale (m := h*h) (n := c) gN (0 : Vec c))
                                   (.operand dT (0 : Vec (bB*(h*h*c)))))
    let (k4, dxT) ← pretty bB (.lnRowBackB (N := bB) (m := h*h) (n := c) "%one" xT bEPS 0 1 0
                                   (.operand da (0 : Vec (bB*(h*h*c)))))
    let (k5, o)   ← pretty bB (.batchOp (N := bB) (.transpose (m := h*h) (n := c))
                                   (.operand dxT (0 : Vec (bB*(h*h*c)))))
    pure (k1 ++ k2 ++ k3 ++ k4 ++ k5, o)

/-- The **γ tail** for one LN site — the two-level `veclnGammaGradB`. -/
private def lnGammaTailB (bB : Nat) (_gN xName cot : String) (c h : Nat) :
    StateM Proofs.StableHLO.EmitS (String × String) := do
    let (k1, xT) ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                  (reassocB (.operand xName (0 : Vec (bB*(c*h*h))))))
    let (k2, dT) ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                  (reassocB (.operand cot (0 : Vec (bB*(c*h*h))))))
    let (k3, o) ← pretty bB (.veclnGammaGradB (N := bB) (R := h*h) (D := c) xT bEPS 0
                                 (0 : Vec (bB*(h*h*c))) (.operand dT (0 : Vec (bB*(h*h*c)))))
    pure (k1 ++ k2 ++ k3, o)

/-- The **β tail** — the two-level `rowDenseBiasGradB`, contracting batch and spatial rows. -/
private def lnBetaTailB (bB : Nat) (cot : String) (c h : Nat) : StateM Proofs.StableHLO.EmitS (String × String) := do
    let (k1, dT) ← pretty bB (.batchOp (N := bB) (.transpose (m := c) (n := h*h))
                                  (reassocB (.operand cot (0 : Vec (bB*(c*h*h))))))
    let (k2, o) ← pretty bB (.rowDenseBiasGradB (N := bB) (R := h*h) (c := c)
                                 (.operand dT (0 : Vec (bB*(h*h*c)))))
    pure (k1 ++ k2, o)

/-- One **ConvNeXt block** backward: the cotangent chain only, param tails separate (the
    per-example renderer factors it the same way, which is what makes ConvNeXt the cheapest of the
    five to thread).

    **THE DROP'S BACKWARD IS THE SAME OP AT THE SAME MASK** (`Proofs.dropPath_vjp_is_self`): a
    diagonal linear map is its own transpose, so there is no `*Grad` peer to keep in step and no
    second emitter to drift.

    **IT APPLIES TO THE BRANCH ONLY, and that mirrors the forward's placement exactly.** The
    dropped cotangent `cotD` feeds the whole branch — LayerScale, project, GELU, expand, LN,
    depthwise — including every parameter gradient computed off it. The skip's fan-in at the bottom
    keeps the RAW `dy`. Dropping there too would attenuate the identity path, which is
    `s ⊙ (branch + x)` arriving by the other door.

    **AND `dyd` IS RETURNED BECAUSE ONE PARAMETER GRADIENT READS IT DIRECTLY.** LayerScale's γ
    gradient is `Σ (cot ⊙ p)` at the cotangent of the LayerScale OUTPUT — which is `s ⊙ dy` once a
    drop site sits between LayerScale and the add, not `dy`. Every other block gradient descends
    from `cot_p` and inherits the scale for free; `%…lg` is the one that would silently be computed
    against an undropped cotangent. It type-checks, trains and descends: 18 of 180 gradients wrong
    by a per-example factor, on the parameter stochastic depth is *about*. At `drop = none` this is
    `dy` itself, so nothing moves. -/
private def bwdBlockB (bB : Nat) (pfx dy : String) (b : FNames) (c e h : Nat) (drop : Option Nat := none)
    (bf16 : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × String × String × String × String × String × String) := do
  let (kD, dyd) ← match drop with
    | some i => pretty bB (.dropPathB (N := bB) (n := c*h*h) (dpName i) (fun _ => 0 : Vec bB)
                             (.operand dy (0 : Vec (bB*(c*h*h)))))
    | none   => pure ("", dy)
  let (k1, cot_p) ← pretty bB (.batchOp (N := bB)
      (.layerScaleCh (h := h) (w := h) s!"%{pfx}lg" (0 : Vec c)) (.operand dyd 0))
  let (k2, cot_g) ← pretty bB (.convBackBatchedAt bf16 (N := bB) (h := h) (w := h) zrnd s!"%{pfx}pW"
        (0 : Kernel4 c e 1 1) 0 (.operand cot_p 0))
  let (k3, cot_e) ← pretty bB (.geluBackB b.e (0 : Vec (bB*(e*h*h))) (.operand cot_g 0))
  let (k4, cot_n) ← pretty bB (.convBackBatchedAt bf16 (N := bB) (h := h) (w := h) zrnd s!"%{pfx}eW"
        (0 : Kernel4 e c 1 1) 0 (.operand cot_e 0))
  let (k5, cot_d) ← lnBackSiteB bB s!"%{pfx}ng" b.d cot_n c h
  let (k6, cot_main) ← pretty bB (.depthwiseBackBatchedAt bf16 (N := bB) (h := h) (w := h) zrnd
        s!"%{pfx}dW"
        (0 : DepthwiseKernel c 7 7) 0 (.operand cot_d 0))
  let (k7, cot_xin) ← pretty bB (.addVB (.operand cot_main (0 : Vec (bB*(c*h*h))))
      (.operand dy 0))
  pure (kD ++ k1 ++ k2 ++ k3 ++ k4 ++ k5 ++ k6 ++ k7, cot_xin, cot_p, cot_e, cot_n, cot_d, dyd)

/-- One **downsample** backward. -/
private def bwdDownB (bB : Nat) (pfx dy xin : String) (ci co h2 : Nat) (bf16 : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × String × String) := do
  -- **THE 2×2/s2 ASYMMETRIC-PAD TRAP.** `convStridedBackBatched`'s dgrad pad is
  -- `[[kH-1-pH, pH], …]`, which agrees with the symmetric spelling at every ODD kernel and is
  -- wrong at `k = 2` — and ConvNeXt's downsample is the repo's ONLY even strided kernel, so this
  -- is the only site in the repo where the difference is observable at all.
  -- `convStridedBackBatchedBf16` preserves it verbatim; do not "tidy" it.
  let (k1, cot_n) ← pretty bB (.convStridedBackBatchedAt bf16 (N := bB) (h := h2) (w := h2) zrnd
        s!"%{pfx}W"
        (0 : Kernel4 co ci 2 2) 0 (.operand dy (0 : Vec (bB*(co*h2*h2)))))
  let (k2, cot_x) ← lnBackSiteB bB s!"%{pfx}ng" xin cot_n ci (2*h2)
  pure (k1 ++ k2, cot_n, cot_x)

/-- The **parameter gradients of one block** — every one a `*GradB`, i.e. `Σ_n` over the batch of
    the per-example gradient on `batchSlice n`. AdamW only: the SGD-inline tail stays in the
    per-example renderer, since `%lr` is a runtime operand on the AdamW path and a baked literal
    on the SGD one — and `ConvNeXtStepTie.lean` is stated at those bytes. -/
private def blockParamGradB (bB : Nat) (pfx : String) (b : FNames)
    (cot_p cot_e cot_n cot_d dy : String) (c e h : Nat)
    -- `bf16` reaches the WEIGHT grads only. Every BIAS grad below stays f32 in every net:
    -- `Σ_{batch,spatial} dy` is a reduction, not a contraction, so there is nothing for a tensor
    -- core to do. Same for the two LN tails, which are reductions too.
    (bf16 : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × List (String × String)) := do
  let (cLg, nLg) ← pretty bB (.layerScaleChGammaGradB (N := bB) (c := c) (h := h) (w := h) b.p
      (0 : Vec (bB*(c*h*h))) (.operand dy 0))
  let (cPw, nPw) ← pretty bB (.convWeightGradBAt bf16 (N := bB) (ic := e) (oc := c) (h := h)
        (w := h)
        (kH := 1) (kW := 1) zrnd b.g (0 : Vec c) (0 : Vec (bB*(e*h*h))) (0 : Kernel4 c e 1 1)
        (.operand cot_p 0))
  let (cPb, nPb) ← pretty bB (.convBiasGradB (N := bB) (ic := e) (oc := c) (h := h) (w := h)
      (kH := 1) (kW := 1) (0 : Kernel4 c e 1 1) (0 : Vec (bB*(e*h*h))) (0 : Vec c)
      (.operand cot_p 0))
  let (cEw, nEw) ← pretty bB (.convWeightGradBAt bf16 (N := bB) (ic := c) (oc := e) (h := h)
        (w := h)
        (kH := 1) (kW := 1) zrnd b.n (0 : Vec e) (0 : Vec (bB*(c*h*h))) (0 : Kernel4 e c 1 1)
        (.operand cot_e 0))
  let (cEb, nEb) ← pretty bB (.convBiasGradB (N := bB) (ic := c) (oc := e) (h := h) (w := h)
      (kH := 1) (kW := 1) (0 : Kernel4 e c 1 1) (0 : Vec (bB*(c*h*h))) (0 : Vec e)
      (.operand cot_e 0))
  let (cNg, nNg) ← lnGammaTailB bB s!"%{pfx}ng" b.d cot_n c h
  let (cNb, nNb) ← lnBetaTailB bB cot_n c h
  let (cDw, nDw) ← pretty bB (.depthwiseWeightGradBAt bf16 (N := bB) (c := c) (h := h) (w := h)
        (kH := 7) (kW := 7) zrnd b.xin (0 : Vec c) (0 : Vec (bB*(c*h*h)))
        (0 : DepthwiseKernel c 7 7) (.operand cot_d 0))
  let (cDb, nDb) ← pretty bB (.depthwiseBiasGradB (N := bB) (c := c) (h := h) (w := h)
      (kH := 7) (kW := 7) (0 : DepthwiseKernel c 7 7) (0 : Vec (bB*(c*h*h))) (0 : Vec c)
      (.operand cot_d 0))
  pure (cLg ++ cPw ++ cPb ++ cEw ++ cEb ++ cNg ++ cNb ++ cDw ++ cDb,
    [(s!"{pfx}dW", nDw), (s!"{pfx}db", nDb), (s!"{pfx}ng", nNg), (s!"{pfx}nbt", nNb),
     (s!"{pfx}eW", nEw), (s!"{pfx}eb", nEb), (s!"{pfx}pW", nPw), (s!"{pfx}pb", nPb),
     (s!"{pfx}lg", nLg)])

/-- The **parameter gradients of one downsample**. -/
private def downParamGradB (bB : Nat) (pfx downLn downIn cot_n dy : String) (ci co h2 : Nat)
    (bf16 : Bool := false) :
    StateM Proofs.StableHLO.EmitS (String × List (String × String)) := do
  let (cB, nB) ← pretty bB (.convStridedBiasGradB (N := bB) (ic := ci) (oc := co) (h := h2)
      (w := h2) (kH := 2) (kW := 2) (0 : Kernel4 co ci 2 2)
      (0 : Vec (bB*(ci*(2*h2)*(2*h2)))) (0 : Vec co) (.operand dy 0))
  let (cNg, nNg) ← lnGammaTailB bB s!"%{pfx}ng" downIn cot_n ci (2*h2)
  let (cNb, nNb) ← lnBetaTailB bB cot_n ci (2*h2)
  -- The wgrad pad is `[[p-1, p+1], …]`, the OPPOSITE shift from the dgrad's `[[p+1, p-1], …]`
  -- in `bwdDownB`. `convStridedWeightGradBBf16` keeps its f32 peer's geometry verbatim;
  -- `scripts/gates/xla_pad_op_check.py` checks this pair; do not "fix" it by symmetry.
  let (wcode, nW) ← pretty bB (.convStridedWeightGradBAt bf16 (N := bB) (ic := ci) (oc := co)
        (h := h2)
        (w := h2) (kH := 2) (kW := 2) zrnd downLn (0 : Vec co)
        (0 : Vec (bB*(ci*(2*h2)*(2*h2)))) (0 : Kernel4 co ci 2 2)
        (.operand dy (0 : Vec (bB*(co*h2*h2)))))
  pure (cB ++ cNg ++ cNb ++ wcode,
    [(s!"{pfx}ng", nNg), (s!"{pfx}nbt", nNb), (s!"{pfx}W", nW), (s!"{pfx}b", nB)])

/-- **The whole-net batched traversal** — forward + cotangent + every parameter gradient, the
    batched peer of `convNextBackAll true (some …)`. Returns `(code, gradMap, softmaxSSA)` with the
    same shape, so the AdamW tail in `ConvNeXtRender.lean` can consume either.

    **AdamW only, deliberately.** The per-example traversal serves both renders off one `adam`
    flag; this one does not, because the SGD path bakes `lr` as a literal where AdamW takes it as a
    runtime `%lr` operand, and an SGD render at the batched index is not something any config asks
    for yet. Adding the flag later is cheap; an artifact nobody loads is a silent-hyperparameter
    hazard.

    The `%dgi`/`%dgb`/`%dgn`/`%dgd`/`%dgapf` GAP-backward block is **hand-written text on both
    sides**, carried over verbatim. It is one of the declared non-AST carve-outs, so the batched
    move neither improves nor degrades it — but note it is parameterised by `bB` and therefore
    already batch-correct, which is why it needs no peer. -/
def convNextBackAllB (smooth : Option (String × String × String) := none) (nClasses : Nat := 10)
    (sd : Bool := false) (V : CnxDims := bTiny)
    -- **bf16**, TRAILING and defaulted, so every existing render is byte-identical.
    -- It reaches every CONVOLUTION — the stem, the block 1×1s, the 7×7 depthwise, the 2×2/s2
    -- downsamples and their dgrads and wgrads — and NOTHING else. LayerNorm, GELU, LayerScale,
    -- every bias gradient, the classifier head and the whole AdamW tail stay f32, which is the
    -- carve-out every bf16 render in this repo makes. Measured: those carve-outs are
    -- NOT a fixed tax: they cost MobileNetV2 nothing and EfficientNet-B0 almost everything.
    (bf16 : Bool := false)
    -- THE PER-REPLICA BATCH — trailing + defaulted, see the note at the top of this file.
    (bB : Nat := 32) :
    StateM Proofs.StableHLO.EmitS (String × List (String × String) × String) := do
    -- ═══ forward — the SAME chain the byte-tied `convNextFwdChainB` emits ═══
    let F : CFwd ← convNextFwdChainB nClasses sd V bf16 (bB := bB)
    let (cSm, nSm) ← pretty bB (.batchOp (N := bB) (.softmaxDiv (n := nClasses))
        (.batchOp (N := bB) (.expe (n := nClasses))
          (.operand F.logits (0 : Vec (bB*nClasses)))))
    let (cSub, dyr) ← pretty bB (.subB (.operand nSm (0 : Vec (bB*nClasses)))
        (.operand "%onehot" 0))
    let fwd := F.code ++ cSm ++ cSub
    -- ═══ the cotangent ═══
    let (cDyC, dyName) ← match smooth with
      | none => pure (s!"    %dy = stablehlo.divide {dyr}, %bsc : {ty [bB, nClasses]}\n", "%dy")
      | some (aStr, negAK, bStr) => do
          let (c1, n1) ← pretty bB (.scaleB (N := bB) (n := nClasses) aStr 0
              (.operand "%onehot" (0 : Vec (bB*nClasses))))
          let (c2, n2) ← pretty bB (.addVB (.operand dyr (0 : Vec (bB*nClasses)))
              (.operand n1 (0 : Vec (bB*nClasses))))
          -- At the batched index these are `N := bB`, where the per-example render writes
          -- `N := 1` — the SAME emitted text (the tag's `n` is what the emitter reads), and the
          -- annotation trap that note warns about disappears, because `bB * nClasses` never has to
          -- reduce definitionally to anything.
          let (c3, n3) ← pretty bB (.shiftB (N := bB) (n := nClasses) negAK 0
              (.operand n2 (0 : Vec (bB*nClasses))))
          let (c4, n4) ← pretty bB (.divConstB (N := bB) (n := nClasses) bStr 0
              (.operand n3 (0 : Vec (bB*nClasses))))
          pure (c1 ++ c2 ++ c3 ++ c4, n4)
    -- ═══ head ═══
    let (cDd, cot_hn) ← pretty bB (.batchOp (N := bB) (.dotOut "%Wd" (0 : Mat (V.dims[3]!) nClasses))
        (.operand dyName 0))
    let (cHnB, cot_gap) ← headLnBackSiteB bB "%hng" F.gap cot_hn V.dims[3]!
    let (cWd, nWd) ← pretty bB (.weightGradB (N := bB) (m := V.dims[3]!) (n := nClasses) F.hn
        (0 : Vec (bB*V.dims[3]!)) (.operand dyName (0 : Vec (bB*nClasses))))
    let (cBd, nBd) ← pretty bB (.biasGradB (N := bB) (n := nClasses)
        (.operand dyName (0 : Vec (bB*nClasses))))
    -- THE CALL ORDER IS THE EMIT ORDER, and it must match `ConvNeXtRender`'s exactly —
    -- `pretty` allocates fresh SSA names from the state monad as it is CALLED, so a renderer that
    -- calls the γ/β tails before `Wd`/`bd` numbers the same graph differently and
    -- `convnext-fwd-b-tie` goes red on 24 lines that are otherwise character-for-character equal.
    -- (It also puts the names out of order against the concatenation below, i.e. use-before-def.)
    -- Order here: dotOut → LN input-VJP → Wd → bd → LN γ → LN β.
    let (cHg, nHg) ← headLnGammaTailB bB "%hng" F.gap cot_hn V.dims[3]!
    let (cHb, nHb) ← headLnBetaTailB bB cot_hn V.dims[3]!
    let mut updMap : List (String × String) :=
      [("hng", nHg), ("hnbt", nHb), ("Wd", nWd), ("bd", nBd)]
    let bD := V.dims[3]!
    let mut bwd := cDyC ++ cDd ++ cHnB ++ cWd ++ cBd ++ cHg ++ cHb ++
      -- HAND-WRITTEN TEXT (the carve-out this docstring declares), so `bD` is threaded by
      -- hand and nothing type-checks it. The `7`s and the `49.0` are SPATIAL and stay literals at
      -- every size; only the channel width moves.
      s!"    %dgi = stablehlo.reshape {cot_gap} : ({ty [bB,bD]}) -> {ty [bB,bD,1,1]}\n" ++
      s!"    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : ({ty [bB,bD,1,1]}) -> {ty [bB,bD,7,7]}\n" ++
      s!"    %dgn = stablehlo.constant dense<49.0> : {ty [bB,bD,7,7]}\n" ++
      s!"    %dgd = stablehlo.divide %dgb, %dgn : {ty [bB,bD,7,7]}\n" ++
      s!"    %dgapf = stablehlo.reshape %dgd : ({ty [bB,bD,7,7]}) -> {ty [bB, bD*7*7]}\n"
    let mut dy := "%dgapf"
    for si' in [0:4] do
      let si := 3 - si'
      let c := V.dims[si]!; let e := 4 * c; let h := bSpats[si]!
      for j' in [0:V.depths[si]!] do
        let j := V.depths[si]! - 1 - j'
        let b := (F.blksAll[si]!)[j]!
        -- `cnxBlockIdx si j` again, from the SAME function the forward called. This loop runs the
        -- stages and the blocks BACKWARDS, so a counter carried by either loop would have to be run
        -- in reverse to name the same sites — the mismatch would pair every backward site with the
        -- wrong forward mask, which typechecks and trains.
        let (code, cot_xin, cot_p, cot_e, cot_n, cot_d, dyd) ←
          bwdBlockB bB s!"s{si}b{j}" dy b c e h (if sd then some (cnxBlockIdx si j V) else none) bf16
        -- `dyd`, not `dy` — LayerScale's γ gradient reads the cotangent at the LayerScale OUTPUT,
        -- which the drop site scales. See `bwdBlockB`.
        let (pcode, pairs) ← blockParamGradB bB s!"s{si}b{j}" b cot_p cot_e cot_n cot_d dyd c e h bf16
        bwd := bwd ++ code ++ pcode; updMap := updMap ++ pairs; dy := cot_xin
      if si > 0 then
        let ci := V.dims[si-1]!; let h2 := bSpats[si]!
        let (code, cot_n, cot_x) ← bwdDownB bB s!"d{si-1}" dy (F.downIn[si-1]!) ci c h2 bf16
        let (pcode, pairs) ← downParamGradB bB s!"d{si-1}" (F.downLn[si-1]!) (F.downIn[si-1]!)
            cot_n dy ci c h2 bf16
        bwd := bwd ++ code ++ pcode; updMap := updMap ++ pairs; dy := cot_x
    -- ═══ stem: back through the stem LN, then the patchify conv's own gradients ═══
    let (cg, ng) ← lnGammaTailB bB "%psng" F.stemC dy V.dims[0]! 56
    let (cb, nb) ← lnBetaTailB bB dy V.dims[0]! 56
    let (cx, dx) ← lnBackSiteB bB "%psng" F.stemC dy V.dims[0]! 56
    bwd := bwd ++ cg ++ cb ++ cx
    updMap := updMap ++ [("psng", ng), ("psnbt", nb)]
    dy := dx
    let (cPsb, nPsb) ← pretty bB (.convBiasGradB (N := bB) (ic := 3) (oc := V.dims[0]!) (h := 56) (w := 56)
        (kH := 4) (kW := 4) (0 : Kernel4 (V.dims[0]!) 3 4 4) (0 : Vec (bB*(3*56*56))) (0 : Vec (V.dims[0]!))
        (.operand dy 0))
    -- **The stem weight grad — ConvNeXt's second and last new bf16 op.** There is no
    -- `convStride4BackBatched` and no bf16 twin of one: this is the patchify stem, its input is
    -- `%x`, and there is no input gradient to compute. TWO new ops for this net, not three.
    let (cPsW, nPsW) ← pretty bB (.convStride4WeightGradBAt bf16 (N := bB) (ic := 3)
          (oc := V.dims[0]!) (h := 56)
          (w := 56) (kH := 4) (kW := 4) zrnd "%x" (0 : Vec (V.dims[0]!))
          (0 : Vec (bB*(3*(2*(2*56))*(2*(2*56))))) (0 : Kernel4 (V.dims[0]!) 3 4 4)
          (.operand dy (0 : Vec (bB*(V.dims[0]!*56*56)))))
    bwd := bwd ++ cPsW ++ cPsb
    updMap := updMap ++ [("psW", nPsW), ("psb", nPsb)]
    pure (fwd ++ bwd, updMap, nSm)

/-- **The ConvNeXt-T AdamW train step at the batched index.** It is the SAME renderer the
    per-example path uses — `convNextAdamTrainStepFaithful` with `traversal` pointed at
    `convNextBackAllB` — not a copy.

    That is possible because the AdamW tail is entirely **parameter-space**: `adamMNextF`,
    `adamVNextF`, `adamWParamF`, `gradSumSqAccF` and `clipScaleF` are indexed by the parameter's
    own size and never see the batch. So "the AdamW tail at the batched index" is no
    work at all — the batch is factored out of it by the ops' own shapes. The only
    thing that moves is which traversal produced the gradients. -/
def convNextAdamTrainStepFaithfulB (alphaStr negAlphaKStr bStr : String)
    (replicas : Nat := 1) (nClasses : Nat := 10) (slug : String := "convnext")
    (ema : Bool := false) (wdExclude : Bool := false) (wdStr : String := "0.0001")
    (clip : Bool := false) (clipStr : String := "1.0") (sd : Bool := false)
    (V : CnxDims := bTiny)
    -- **bf16**, TRAILING and spelled ONCE HERE — exactly as `sd` and `V` are, and for the same
    -- reason. It has to reach TWO places that a caller must never be able to set independently:
    -- the TRAVERSAL (which decides the arithmetic) and `cnxAdamVariant` (which decides the entry
    -- NAME and therefore the artifact path). That second half is the entry-name defect: a flag that
    -- reaches the emission but not the name writes `…bf16_train_step.mlir` declaring
    -- `@convnextin_adamwxclipdrop_train_step`, and the driver refuses at load. The `#guard`s at the
    -- bottom of this file pin every spelling.
    (bf16 : Bool := false)
    -- **THE PER-REPLICA BATCH, AND THIS IS WHERE THE TWO RENDERERS MEET.** Spelled ONCE
    -- here and handed to BOTH halves below — `convNextBackAllB` (the BODY, at `N := bB`) and
    -- `convNextAdamTrainStepFaithful` (the WRAPPER, as `cBS`: `%x`'s declared shape, the `%bsc`
    -- loss divisor, the drop-path signature). Exactly the discipline `sd`, `V` and `bf16` get,
    -- and for the same reason: a caller able to set the two independently is the defect. At the
    -- default every committed artifact is byte-identical; the ImageNet `#eval`s below pass 64 so
    -- the verified job pairs with its JAX reference's global 256 (4 × 64).
    (bB : Nat := 32) : String :=
  let negAK := if negAlphaKStr.isEmpty then "-" ++ alphaOverK nClasses 0.1 else negAlphaKStr
  convNextAdamTrainStepFaithful alphaStr negAlphaKStr bStr replicas nClasses slug ema
    wdExclude wdStr clip clipStr
    -- `sd` IS SPELLED ONCE AND REACHES BOTH HALVES FROM HERE — the traversal (which places the
    -- 18 sites) and the wrapper (which declares the 18 inputs, the 18 pass-through outputs and the
    -- `drop` variant name). Letting a caller set them independently is the shape of defect this
    -- avoids; here there is nothing to keep in step.
    -- `D` IS THE SECOND SUCH PARAMETER and it is spelled once for the same reason: the traversal
    -- decides how many blocks are EMITTED, the wrapper decides how many parameters are DECLARED,
    -- and a disagreement between them is an arity mismatch the driver reports as a blob-walk
    -- failure rather than anything that names the depth table.
    (traversal := some (convNextBackAllB (some (alphaStr, negAK, bStr)) nClasses sd V bf16 (bB := bB)))
    -- AND HERE. `bf16` reaching the traversal above but not this line is the entry-name defect
    -- in its most convincing form — the graph really would be bf16, every other check would pass,
    -- and only the artifact's declared name would be wrong.
    (sd := sd) (V := V) (bf16 := bf16) (cBS := bB)

/-- The drop-free forward's banner — the line `ConvNeXtRender.convNextFwdFaithfulV` emits, restated
    here so the byte tie can demand **byte-identity** rather than "identical apart from a comment".
    Worth the parameter: a tie that compares modulo one line is a tie with a hole in it, and the
    hole is exactly where a renderer's own description of what it did would live. This is the banner
    the committed `convnext_fwd` / `convnextin_fwd` / `convnextsin_fwd` / `convnextbin_fwd` carry,
    which is why it takes the size: the model name is derived from the stage table, exactly as the
    per-example line derives it. -/
def cnxFwdBanner (V : CnxDims := bTiny) : String :=
  s!"    // ── {cnxModelName V} forward: every op is pretty(verified AST node) except the %one/%zero LayerNorm constants ──\n"

/-- The SD forward's banner. Its own, and not `cnxFwdBanner`, because these bytes ARE a
    different render and a banner claiming otherwise would misdescribe the artifact it heads. -/
def cnxDropFwdBanner (V : CnxDims := bTiny) : String :=
  -- Both the SIZE and the SITE COUNT are derived from `D`. A literal here is precisely the thing
  -- this docstring warns about one line up: at S the artifact would open by announcing itself as a
  -- net with half its blocks.
  s!"    // ── {cnxModelName V} forward at the BATCHED index N := B, with STOCHASTIC DEPTH ──\n" ++
  s!"    // {cnxDropTotal V} drop sites, one per block, on the RESIDUAL BRANCH (between LayerScale and the\n" ++
  "    // skip add). Emitted in the forward too, at an all-ones mask supplied by the driver:\n" ++
  "    // exactly the identity (Proofs.dropPath_ones_id), so this stays a byte-prefix of the\n" ++
  "    // SD train step and the forward-subset-train-step audit keeps a partner.\n"

end Proofs.StableHLO

-- ════════════════════════════════════════════════════════════════
-- § THE STOCHASTIC-DEPTH ARTIFACTS
-- ════════════════════════════════════════════════════════════════
--
-- The SD renders are built on the batched chain, where the per-example mask is expressible at all.
--
-- The SD render is NOT byte-comparable line-for-line with `convnext_adam`. Its keep = 1
-- gate is NUMERIC (`adamdrop` at every keep 1.0 must train what `adam` trains), under
-- `scripts/det_shim.sh` — cross-graph numeric comparison on CUDA has no resolution without it.

-- ── Imagenette, K=10, bs32 — the GATE VEHICLE ──────────────────────────────────────────────────
-- Not a matched pair: `convNextTinyConfig` sets no `dropPath`, so this render's accuracy is
-- comparable to nothing. It exists because every gate that has to run (keep = 1, the misplacement
-- control, the ones-mask forward) is seconds here and minutes at ImageNet scale.
#eval IO.FS.writeFile "verified_mlir/convnext_adamdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0" 1 10 "convnext"
    (ema := false) (wdExclude := false) (wdStr := "0.0001") (clip := false) (clipStr := "1.0")
    (sd := true))

-- Its prefix partner. The SD variant gets its OWN forward rather than reusing `convnext_fwd`:
-- letting the SD trainer eval through the drop-free forward is what the reference literally does,
-- but it would leave the SD train step with no prefix partner at all — i.e. SPEND one of the two
-- load-bearing structural gates in the repo rather than pay 18 dead multiplies at eval.
#eval IO.FS.writeFile "verified_mlir/convnext_drop_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnext_drop_fwd" 10
    Proofs.StableHLO.cnxDropFwdBanner (sd := true))

-- The **2-replica** peer, and it exists for one reason: `lake build drop-shard-check` is the gate
-- that says the per-example masks are SHARDED rather than replicated, and its known answer is exact
-- only at TWO replicas — f32 addition is COMMUTATIVE, so `(a+b)/2` is invariant under swapping the
-- halves, where above two the collective is a tree whose order a permutation changes and
-- associativity does not hold. So this is not "the DP render at a smaller replica count"; it is
-- the only replica count at which that gate is a bit-exactness claim.
--
-- The defect this gate targets lives in the SHIM (the shard flag, and the mask buffer sized at the
-- per-device batch), not in any render — so it is net-independent. Running the gate here is what
-- says so rather than assumes it: on an LN net the replica-0-local witness is `%loss` ALONE (there
-- are no batch statistics), which is a materially weaker anti-vacuity half.
#eval IO.FS.writeFile "verified_mlir/convnext_adamdpdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0" 2 10 "convnext"
    (ema := false) (wdExclude := false) (wdStr := "0.0001") (clip := false) (clipStr := "1.0")
    (sd := true))

-- ── Full 1000-class ImageNet, slug `convnextin` ─────────────────────────────────────────────────────
-- BOTH SCALES, deliberately: *a feature is not done when its Imagenette artifact renders*. Both
-- scales are one `#eval` apart, which is exactly why it is easy to stop at one.
--
-- AND BOTH THE SINGLE-DEVICE **AND THE DP** PEER: an ImageNet run loads the DP render, and DP
-- renders are the ones that silently fall behind their single-device peers. `wx` ++ `clip` ++
-- `drop` is `convNeXtTinyImagenetConfig` entire (`weightDecay := 0.05`,
-- `wdExcludeNormBias := true`, `gradClipNorm := 1.0`, `dropPath := 0.1`) — the first ConvNeXt
-- artifact that carries every optimizer-and-regulariser knob its reference sets.
#eval IO.FS.writeFile "verified_mlir/convnextin_adamwxclipdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 1 1000 "convnextin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (bB := cnxInBS))
#eval IO.FS.writeFile "verified_mlir/convnextin_adamdpwxclipdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 4 1000 "convnextin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (bB := cnxInBS))
#eval IO.FS.writeFile "verified_mlir/convnextin_drop_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnextin_drop_fwd" 1000
    Proofs.StableHLO.cnxDropFwdBanner (sd := true) (bB := cnxInBS))

-- ── THE bf16 PEERS — `adamwxclipdropbf16` ─────────────────────────────────────────────────────
-- ConvNeXt is the sixth net on the bf16 render path and it needs **exactly two new ops**:
-- `convStride4Bf16` (the 4×4/s4 patchify stem) and `convStride4WeightGradBBf16` (its weight grad).
-- Everything else — the 7×7 depthwise, the block 1×1s, the 2×2/s2 downsamples, and every dgrad and
-- wgrad among them — comes from the MobileNetV2 (8 ops) and MobileNetV4 (3 ops) sets unchanged.
--
-- **There is no third op**: `convStride4` is the STEM, so it has no input gradient to compute.
-- The block 1×1s are `.conv` at `kH = kW = 1` and `convBf16` already covers them. The only true
-- matmul is the classifier head, which stays f32.
--
-- Both new emit shapes are checked STANDALONE on the exact stem shapes: `bf16 operands → f32-typed
-- result` FOLDS to pure f32 at stride 4 exactly as it does at stride 1, stride 2 and grouped, and
-- `bf16 operands → bf16-TYPED result → convert` reaches the hardware. Stride buys no exemption.
--
-- What stays f32, here as in every other bf16 render: LayerNorm, GELU, LayerScale, the drop-path
-- masks, every BIAS gradient (`Σ_{batch,spatial} dy` is a reduction, not a contraction — nothing
-- for a tensor core to do), the classifier head, and the whole AdamW tail including the clip fold.
-- This carve-out is NOT a fixed tax: it cost MobileNetV2 nothing (1.92× verified
-- against a 1.94× JAX reference) and EfficientNet-B0 almost everything (1.09×). Which one ConvNeXt
-- resembles is a measurement, not a prediction.
--
-- SINGLE-DEVICE IS THE ARM THAT MEANS ANYTHING. MobileNetV2 measures 1.92× on one GPU
-- and 1.37× on four from the SAME GRAPH — the loss is the shim feed first and the f32 all-reduce
-- second, neither of which is a statement about the renderer. Read any 4-replica number here as a
-- SYSTEM result and check `SHIM_WORKERS` before ever blaming the emit.
#eval IO.FS.writeFile "verified_mlir/convnextin_adamwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 1 1000 "convnextin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (bf16 := true) (bB := cnxInBS))
-- The DP peer: an ImageNet run loads the DP render, and DP renders are exactly
-- the ones that silently fall behind their single-device peers. ConvNeXt's collectives are already
-- tied (unlike MNv4's, which is why THAT net rendered single-device only), so this inherits nothing
-- untied. The clip still sits AFTER the collective: 180 all_reduces, not 360, all before the norm
-- fold — and all of them f32, a tax this artifact pays.
#eval IO.FS.writeFile "verified_mlir/convnextin_adamdpwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 4 1000 "convnextin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (bf16 := true) (bB := cnxInBS))

-- **THE EMA PEERS OF THE SHIPPING RECIPE**. `convNeXtTinyImagenetConfig` sets
-- `useEMA := true` (0.9999) and its reference number is the SHADOW's, so the verified pair needs the
-- shadow on the recipe it actually trains — `convnextin_ema{,dp}` above carry it on the bare AdamW
-- graph, three features behind. One `adamMNextF` per parameter on the UPDATED weight (the same
-- reading `convnext_ema` uses), a fourth `[θ|m|v|ema]` region and the `%emad, %oemad` pair; the
-- clip, the drop masks and the bf16 twins are untouched. The shadow reads θ' AFTER the clipped
-- update, as the reference's `ema_update` follows `train_step`.
-- `vit-ema-drop-render convnextin` pins the artifact's arity against the driver's packing.
#eval IO.FS.writeFile "verified_mlir/convnextin_emawxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 1 1000 "convnextin"
    (ema := true) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (bf16 := true) (bB := cnxInBS))
#eval IO.FS.writeFile "verified_mlir/convnextin_emadpwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 4 1000 "convnextin"
    (ema := true) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (bf16 := true) (bB := cnxInBS))
#guard Proofs.StableHLO.cnxAdamVariant 4 true true true true true == "emadpwxclipdropbf16"
#guard Proofs.StableHLO.cnxAdamVariant 1 true true true true true == "emawxclipdropbf16"

-- ── ConvNeXt-**S** on ImageNet, slug `convnextsin` ────────────────────────────────────────────
-- Added by RESHAPING an existing renderer rather than writing a chain: ConvNeXt-S is **pure
-- depth** — `[3,3,9,3] → [3,3,27,3]` with the dims unchanged.
--
-- **The proof side needs nothing**, for the same reason ViT's does not: the certificates the
-- ops carry are per-SITE and stage-generic, so 18 more blocks is 18 more uses of theorems that
-- already quantify over `c`, `e` and `h`. Depth was never a hypothesis.
--
-- 342 parameter tensors and **50,222,152** scalars at K = 1000 — the published ConvNeXt-S figure,
-- and the count `jax/MainConvNeXtSImagenet.lean` emits from an independent implementation.
--
-- **BOTH the single-device and the DP peer**, as ConvNeXt-T carries both: an ImageNet run loads
-- the DP render, and DP renders are exactly the ones that silently fall behind. `wx` ++ `clip` ++ `drop` is `convNeXtTinyImagenetConfig` entire, and it is
-- the recipe ConvNeXt-S inherits — the paper changes the stochastic-depth RATE with the size (S is
-- 0.4 at 300 epochs against T's 0.1), and that rate is DATA (`dropKeeps` in the spec), not a
-- render knob, so both sizes render from this one call.
--
-- NOTHING HAS BEEN TRAINED. These render, the shapes tie, the counts are `#guard`ed. No
-- accuracy, and no wall clock beyond a step probe.
#eval IO.FS.writeFile "verified_mlir/convnextsin_adamwxclipdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 1 1000 "convnextsin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxSmall))
#eval IO.FS.writeFile "verified_mlir/convnextsin_adamdpwxclipdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 4 1000 "convnextsin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxSmall))

-- **The bf16 peers of S.** ConvNeXt-T, -S and -B each carry the full (precision × replicas)
-- square.
--
-- **No new proved operator.** Every bf16 op these need already exists — S is ConvNeXt-T's
-- depth table at T's dims, so it instantiates the SAME ops at the SAME widths. B is the stronger
-- precedent still: it uses four widths T and S do not, and it needs nothing new either.
--
-- The variant STRINGS are unchanged — `cnxAdamVariant` keys on replicas and flags, never on the
-- size — so `adamwxclipdropbf16` and `adamdpwxclipdropbf16` are already `#guard`ed below at every
-- concatenation. S reuses them at a different SLUG: the slug is the net and the variant is the
-- recipe.
--
-- **DO NOT ASSUME THIS IS FASTER.** A bf16 op can be SLOWER than its f32 peer with every gate
-- green — ViT's stem wgrad runs 0.19× — so the render existing says nothing about the wall clock.
-- Measure with `scripts/probes/bf16_device_step.py`, which times the GRAPH; a trainer's own ms/step is a
-- system number that also moves with `PJRT_FFI_RESIDENT` (off by default) and the shim feed.
#eval IO.FS.writeFile "verified_mlir/convnextsin_adamwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 1 1000 "convnextsin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxSmall) (bf16 := true))
#eval IO.FS.writeFile "verified_mlir/convnextsin_adamdpwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 4 1000 "convnextsin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxSmall) (bf16 := true))
-- The SD train step's PREFIX PARTNER — the same reason `convnextin_drop_fwd` exists. It is not the
-- forward the driver evals through (that is `convnextsin_fwd.mlir`, written by `ConvNeXtRender`);
-- it is what keeps the `forward ⊂ train-step` structural audit from having nothing to pair the SD
-- render with. Without that gate a forward can score a net it did not train.
#eval IO.FS.writeFile "verified_mlir/convnextsin_drop_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnextsin_drop_fwd" 1000
    (Proofs.StableHLO.cnxDropFwdBanner Proofs.StableHLO.cnxSmall)
    (sd := true) (V := Proofs.StableHLO.cnxSmall))

-- **S's PAIR RENDERS, at T's batch.** The EMA peers of the shipping recipe, as
-- `convnextin_ema{,dp}wxclipdropbf16`: 64 per replica × 4 = global 256, the reference's batch and
-- the LR's (2.5e-4 = 4e-3 @ 4096 scaled to 256). The 32-per-replica renders above stay as the
-- f32/bf16 siblings, with `convnextsin_drop_fwd` their prefix partner at 32. The eval forwards
-- (`convnextsin_fwd`, `_fwd_s288`) are at 64, and the driver reads the eval batch off the forward,
-- so either train batch scores through them.
-- `vit-ema-drop-render convnextsin` pins the arity.
#eval IO.FS.writeFile "verified_mlir/convnextsin_emawxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 1 1000 "convnextsin"
    (ema := true) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxSmall) (bf16 := true) (bB := cnxInBS))
#eval IO.FS.writeFile "verified_mlir/convnextsin_emadpwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 4 1000 "convnextsin"
    (ema := true) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxSmall) (bf16 := true) (bB := cnxInBS))

-- ── ConvNeXt-**B** on ImageNet, slug `convnextbin` ────────────────────────────────────────────
-- The size that made the DIMS a parameter. B is S's depth table at `[128,256,512,1024]`, so it
-- shares S's 36 drop sites and its 342 parameter tensors and differs only in every width —
-- 88,589,416 scalars at K = 1000, the published 88.59M and the count
-- `jax/MainConvNeXtBImagenet.lean` emits independently.
--
-- **The proof side needs nothing here either**, and B is the stronger evidence for that claim than
-- S: S reuses theorems at the SAME widths, where B instantiates them at four widths T and S do not
-- use. The certificates are generic in `c`/`e`/`h`, so this is arithmetic, not a new obligation.
--
-- **B IS THE SIZE THAT BREAKS ANYTHING KEYED ON BLOCK COUNT.** It has 36 blocks exactly like S,
-- so `cnxModelName` matches on the whole `CnxDims` record — keyed on `cnxDropTotal`, every B
-- artifact would open by calling itself a ConvNeXt-S. The guard beside that function checks it.
--
-- NOTHING HAS BEEN TRAINED, and see the app docstring before quoting any wall clock: B at bs32
-- fp32 is the first ConvNeXt size where fitting is a real question rather than a formality.
#eval IO.FS.writeFile "verified_mlir/convnextbin_adamwxclipdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 1 1000 "convnextbin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxBase))

-- **The bf16 peer of B** — rendered to answer the memory question the docstring above calls
-- "a real question rather than a formality" at this size with a measured `peak_memory_in_bytes`
-- rather than an extrapolation from T. Do not assume bf16 helps: on ConvNeXt-**T** it moves peak
-- memory by a few per cent, because this emit converts back to f32 after every op and so keeps the f32
-- activation alive anyway. bf16 here is a SPEED change, not a memory one.
#eval IO.FS.writeFile "verified_mlir/convnextbin_adamwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 1 1000 "convnextbin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxBase) (bf16 := true))
#eval IO.FS.writeFile "verified_mlir/convnextbin_adamdpwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 4 1000 "convnextbin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxBase) (bf16 := true))
#eval IO.FS.writeFile "verified_mlir/convnextbin_adamdpwxclipdrop_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" "32.0" 4 1000 "convnextbin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxBase))
-- **B's PAIR RENDERS, at T's batch**, as S's: 64 per replica × 4 = global 256, the EMA shadow, bf16.
-- The size question the app docstring raised is answered by a compile probe (2026-09-29): the DP
-- render peaks at **9.53 GiB of the plugin's 11.68 default**, so B needs no accumulation render
-- and no `LEAN_MLIR_MEM_FRACTION` (0.97 OOMs ConvNeXt's bf16 arms outside the pool).
-- `vit-ema-drop-render convnextbin` pins the arity.
#eval IO.FS.writeFile "verified_mlir/convnextbin_emawxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 1 1000 "convnextbin"
    (ema := true) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxBase) (bf16 := true) (bB := cnxInBS))
#eval IO.FS.writeFile "verified_mlir/convnextbin_emadpwxclipdropbf16_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 4 1000 "convnextbin"
    (ema := true) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (sd := true) (V := Proofs.StableHLO.cnxBase) (bf16 := true) (bB := cnxInBS))
#eval IO.FS.writeFile "verified_mlir/convnextbin_drop_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnextbin_drop_fwd" 1000
    (Proofs.StableHLO.cnxDropFwdBanner Proofs.StableHLO.cnxBase)
    (sd := true) (V := Proofs.StableHLO.cnxBase))

-- ════════════════════════════════════════════════════════════════
-- § THE DROP-FREE ARTIFACTS: ONE chain per net
-- ════════════════════════════════════════════════════════════════
--
-- Every ConvNeXt artifact but one renders from the batched traversal. The four forwards
-- (`convnext_fwd`, `convnextin_fwd`, `convnextsin_fwd`, `convnextbin_fwd`) render BYTE-IDENTICALLY
-- off this chain and the per-example one, and each of the thirteen AdamW/EMA train steps differs
-- from its per-example render on exactly 78 lines, every one the conv input-VJP's
-- `transpose`/`reverse` pair in the other order (commuting ops on disjoint axes;
-- `tests/TestConvNeXtFwdBTie.lean` allows that pair and nothing else). The numeric licence is the
-- keep = 1 gate, per-example against batched, 0 of 83,478,846 floats differing after three AdamW
-- steps with `scripts/probes/perturb_conv_vjp.py` as the negative control — re-run as
-- `convnext-adam-tie` on these bytes.
--
-- `Nets/ConvNeXt/ConvNeXtFoldGB.lean` folds every `*GradB` node this traversal emits, so no
-- committed artifact is `pretty` of an AST without a fold (byte-identity is not tier-identity).
--
-- `convnext_train_step.mlir` (the SGD-inline step) stays in `ConvNeXtRender.lean`: this traversal
-- has no fused-SGD arm, and `ConvNeXtStepTie.lean`'s 182-parameter tie is stated at those bytes.

-- Regenerate `verified_mlir/convnext_fwd.mlir` — what `convnext-smooth` certifies through, and the
-- eval forward for the ConvNeXt trainers — from the SAME `convNextFwdChain` the train steps
-- differentiate. `tests/TestConvNeXtFwd.lean` is an `iree-compile` smoke over the committed bytes.
--
-- **ConvNeXt needs no `_fwd_eval` peer and must not grow one.** LayerNorm reduces within one
-- example, never over the batch, so this forward is already class-batch-independent — the very
-- property `@resnet34_fwd_eval` / `@efficientnet_fwd_eval` exist to recover for the BN nets.
#eval IO.FS.writeFile "verified_mlir/convnext_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnext_fwd" 10 Proofs.StableHLO.cnxFwdBanner)

-- The **AdamW** train step — **the artifact `convnext-verified-adam` trains on**, and this `#eval`
-- is its ONLY writer. `tests/TestConvNeXtTrain.lean` only iree-compiles the committed bytes.
-- Literals: α = 0.1, −α/K = −0.01 (K = 10), batch 32.
--
-- `lake build convnext-adam-tie` ties it against the hand-written emitter's bytes (one AdamW step,
-- all 83,434,629 returned floats): `%loss` BIT-EXACT, 179 of 180 parameter gradients bit-exact, and the one that differs —
-- `s3b2lg`, the last block's layer-scale γ — agrees BETTER than this render does with itself under
-- a semantics-preserving batch reversal. That γ gradient is a cancelling reduce (|Σ|/Σ|·| ≈ 0.09)
-- and does not reproduce to 1e-4 against ANY reordering, so the gate is calibrated against that
-- control rather than an absolute bound, and gates the SPREAD as well as the magnitude — a
-- cotangent perturbation clears the magnitude gate while disturbing 178/180 params. To re-run:
--
--   git show b94e8e9:verified_mlir/convnext_adam_train_step.mlir > /tmp/retired.mlir
--   IREE_BACKEND=rocm .lake/build/bin/convnext-adam-tie /tmp/retired.mlir \
--     verified_mlir/convnext_adam_train_step.mlir
#eval IO.FS.writeFile "verified_mlir/convnext_adam_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0")

-- ── THE EMA VARIANT, selected by `LEAN_MLIR_VARIANT=ema` ────────────────
-- Same graph plus one `adamMNextF` per parameter on the UPDATED weight — `d·ema + (1−d)·θ'`, which
-- is `Proofs.adamMNext` at `(β₁ := d, m := ema, g := θ')`, so this costs **no new op, no new `den`,
-- no new faithfulness theorem and no new VJP**.
--
-- THE BLOB GAINS A FOURTH REGION: `[θ|m|v|ema]`, and the scalar tail goes 3 → 5 (`%emad`,
-- `%oemad`). That is why it renders to its OWN slug — a 4-region graph fed a 3-region blob is not a
-- subtle numeric wrong answer, it is every parameter misaligned, and the AdamW artifact must stay
-- exactly what it is. The driver's checkpoint SIZE GUARD is the other half of that:
-- checkpoints carry no header, so a 3-region file read as 4 resumes silent garbage.
--
-- `%emad`/`%oemad` are ARGS rather than constants because the reference's decay is time-varying,
-- `d = min(decay, (1+t)/(10+t))` — TF's warmup-corrected `ExponentialMovingAverage`. The
-- reference's own measurement of what dropping that correction costs: a shadow still
-- holding 12.8% of the random init at epoch 66, scoring **0.00% top-1** while the live weights
-- scored 70.48%. An 80-epoch Imagenette run is 2.4 τ, i.e. inside that regime.
#eval IO.FS.writeFile "verified_mlir/convnext_ema_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0"
    (ema := true))

-- The **DATA-PARALLEL** render, selected at run time by `LEAN_MLIR_VARIANT=adamdp`; `replicas` is
-- threaded through `adamOneEma`.
--
-- Same graph, plus one `all_reduce(add)/N` per parameter gradient between the certified gradient
-- and the certified AdamW triple: *certified gradient → trusted collective → certified AdamW*. The
-- collective is a DECLARED carve-out and the render says so in its own output banner at
-- `replicas > 1`, because an undeclared carve-out is how wrong things ship. The claim ceiling is
-- the single-device render's: the gradient averaging is a proven identity; the collective
-- implementing it is trusted, exactly like the lowerer.
--
-- Per-channel LayerNorm γ/β are `tensor<{c}xf32>`, so no collective here is rank-0. Nothing in the
-- repo exercises a rank-0 `all_reduce`.
--
-- It renders to its OWN path, which is what stops a race where producing a DP render means
-- editing a knob and clobbering the artifact the trainer runs. `2` is the replica count these are
-- rendered at and it must match `PJRT_REPLICAS` at run time, because the graph bakes
-- `replica_groups`. Re-render here to change it.
--
-- It needs the XLA build (`convnext-verified-adam`): collectives exist only on the PJRT
-- path, and the IREE shim refuses a DP entry point outright rather than silently running
-- single-device.
#eval IO.FS.writeFile "verified_mlir/convnext_adamdp_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0" 2)

-- The DATA-PARALLEL peer of the EMA render. One `#eval`: `replicas` and `ema` are both already
-- renderer parameters, so this is the cheap half exactly as `mobilenetv2in_rmsdp64` is.
--
-- The collective and the shadow do not interact, and that is worth stating because it is what
-- makes the gate meaningful rather than circular: `all_reduce` sits on the GRADIENT, upstream of
-- the AdamW triple, while the EMA reads θ' — the triple's OUTPUT. So the shadow inherits whatever
-- the collective produced and adds no new cross-replica coupling. What the duplicated-batch gate
-- then checks is that the 4th region is threaded identically on both paths, which an arity check
-- cannot see (both renders have the region; the question is whether it carries the same values).
#eval IO.FS.writeFile "verified_mlir/convnext_emadp_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0" 2
    (ema := true))

-- ── ConvNeXt-T on FULL 1000-class ImageNet, slug `convnextin` ──────────────────────────────────────
-- The ConvNeXt peer of `resnet34in_*` and `vitin_*`. `nClasses` is a renderer parameter, and these
-- render at `cnxInBS` (global 256 on four replicas).
--
-- `-α/K` is DERIVED here (empty string ⇒ `alphaOverK nClasses`), so the emitted constant is
-- -0.000100 at K=1000 rather than the K=10 literal the Imagenette renders carry. A hardcoded K
-- there sits on the gradient path and shows only as an implausible loss. Gated below by the
-- artifact check, not assumed.
-- `wdStr := "0.05"` ON BOTH: `convnextTinyImagenetConfig.weightDecay := 0.05`, where the file's
-- 1e-4 default is `convnextTinyConfig`'s IMAGENETTE value. **No config says "ImageNet ConvNeXt at
-- wd 1e-4"**. They stay short of the reference in the ways the docstring above lists (no `wx`, no
-- clip, one-hot targets); the decay is not one of those ways.
-- The variant that MATCHES the reference is `convnextin_adamdpwxclip` below.
#eval IO.FS.writeFile "verified_mlir/convnextin_adam_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 1 1000 "convnextin"
    (wdStr := "0.05") (bB := cnxInBS))
#eval IO.FS.writeFile "verified_mlir/convnextin_adamdp_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 4 1000 "convnextin"
    (wdStr := "0.05") (bB := cnxInBS))
#eval IO.FS.writeFile "verified_mlir/convnextin_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnextin_fwd" 1000 Proofs.StableHLO.cnxFwdBanner
    (bB := cnxInBS))
-- timm's TEST protocol for ConvNeXt-T (`convnext_tiny.fb_in1k`: 288px, crop 1.0,
-- jax/timm_eval_protocols.json): the same forward at a 288 input, stages 72/36/18/9, entry
-- `@convnextin_fwd_s288` (an artifact's entry is its file name — `regen_verified_mlir.sh check`).
-- Same operands, so `score-checkpoint` scores it under `LEAN_MLIR_EVAL_SIZE=288`.
#eval IO.FS.writeFile "verified_mlir/convnextin_fwd_s288.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnextin_fwd_s288" 1000 Proofs.StableHLO.cnxFwdBanner
    (bB := cnxInBS) (s := 288))
#guard Proofs.StableHLO.convNextFwdRenderB "convnextin_fwd" 1000 Proofs.StableHLO.cnxFwdBanner
    (bB := cnxInBS) (s := 224) ==
  Proofs.StableHLO.convNextFwdRenderB "convnextin_fwd" 1000 Proofs.StableHLO.cnxFwdBanner (bB := cnxInBS)

-- ── `wdExcludeNormBias` — timm/DeiT `no_weight_decay` ──────────────────────────────────────────
-- `convnextTinyImagenetConfig.wdExcludeNormBias := true`. 123 of the 182 params take `%wdz`: every
-- LN γ/β, every conv bias, and LayerScale γ — all 1-D, so the PLAIN RANK TEST covers them and
-- ConvNeXt needs no name carve-out (ViT's `pos` has no analogue here; the generated reference sets
-- `_WD_POS_SHAPE = None`). Same arity, same types, same regions.
--
-- The ImageNet render also takes wd = **0.05**, not the file's 1e-4 default: BOTH halves of the
-- reference's decay recipe — the magnitude and the mask — have to be right for the pair. `convnext_adamwx` is the Imagenette-shaped
-- peer that `wdx-tie convnext` drives, where the compile is seconds.
#eval IO.FS.writeFile "verified_mlir/convnext_adamwx_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0" 1 10 "convnext"
    (ema := false) (wdExclude := true))
#eval IO.FS.writeFile "verified_mlir/convnextin_adamwx_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 1 1000 "convnextin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (bB := cnxInBS))

-- ── GLOBAL-NORM GRADIENT CLIPPING ──────────────────────────────────────────────────────────────
-- `convnextTinyImagenetConfig.gradClipNorm := 1.0`. `convnextTinyConfig` sets nothing, so the
-- Imagenette artifacts keep their bytes and this is a variant, not a flipped default.
--
-- The Imagenette `clip` render is a GATE VEHICLE, not a matched pair — no Imagenette reference
-- run clips, so its accuracy is comparable to nothing. It exists so `clip-tie` can drive it at
-- bs32/K=10, where the compile is seconds.
--
-- THE BELOW-THRESHOLD RENDER IS NOT COMMITTED — `scripts/probes/perturb_clip.py hi` generates it,
-- because `cnxAdamVariant`'s `clip` is a **Bool**: a second render at a different threshold spells
-- the SAME variant, and `convnext_adamcliphi_train_step.mlir` would declare
-- `@convnext_adamclip_train_step` — an entry disagreeing with its own path. **A Bool-derived name cannot distinguish two renders that
-- differ only in a baked constant, and those two ARE different functions.** ViT's explicit
-- `funcName` hides this class of mistake; this net derives its name and does not.
#eval IO.FS.writeFile "verified_mlir/convnext_adamclip_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "-0.010000" "32.0" 1 10 "convnext"
    (ema := false) (wdExclude := false) (wdStr := "0.0001") (clip := true) (clipStr := "1.0"))
-- The ImageNet render — BOTH halves of the reference's recipe, `wx` ++ `clip`.
#eval IO.FS.writeFile "verified_mlir/convnextin_adamwxclip_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 1 1000 "convnextin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (bB := cnxInBS))
-- **THE DATA-PARALLEL PEER — the artifact an ImageNet run actually loads.** The clip sits AFTER the
-- collective: 180 all_reduces, not 360, all before the norm fold.
#eval IO.FS.writeFile "verified_mlir/convnextin_adamdpwxclip_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 4 1000 "convnextin"
    (ema := false) (wdExclude := true) (wdStr := "0.05") (clip := true) (clipStr := "1.0")
    (bB := cnxInBS))

-- ── THE IMAGENET EMA PEER ──────────────────────────────────────────────────────────────────────
-- ConvNeXt's reference number IS the EMA shadow's — **75.93%**, against a live best of 76.28% — so
-- without this render the `convnextin` pair is not comparable at all, whatever else it carries.
#eval IO.FS.writeFile "verified_mlir/convnextin_ema_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 1 1000 "convnextin" (ema := true) (bB := cnxInBS))
#eval IO.FS.writeFile "verified_mlir/convnextin_emadp_train_step.mlir"
  (Proofs.StableHLO.convNextAdamTrainStepFaithfulB "0.100000" "" s!"{cnxInBS}.0" 4 1000 "convnextin" (ema := true) (bB := cnxInBS))

-- ════════════════════════════════════════════════════════════════
-- § ConvNeXt-**S** on ImageNet, slug `convnextsin`
-- ════════════════════════════════════════════════════════════════
--
-- **The eval forward.** The `drop` variants are batched-only — the per-example render cannot
-- express a per-EXAMPLE mask at all.
--
-- IT IS LOAD-BEARING AND IT IS EASY TO OMIT. `Verified.Train` resolves the eval forward as
-- `<slug>_<variant>_fwd.mlir` if present else **`<slug>_fwd.mlir`**, BY NAME — so a net whose only
-- forward is `convnextsin_drop_fwd.mlir` trains fine and then dies at the first eval on a missing
-- file. No build-time check covers it because no build-time check reads a filename.
--
-- ConvNeXt needs no `_fwd_eval` peer and must not grow one, for S exactly as for T: LayerNorm
-- reduces within one example and never across the batch, so this forward is already
-- class-batch-independent.
#eval IO.FS.writeFile "verified_mlir/convnextsin_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnextsin_fwd" 1000
    (Proofs.StableHLO.cnxFwdBanner Proofs.StableHLO.cnxSmall) (V := Proofs.StableHLO.cnxSmall)
    (bB := cnxInBS))
-- timm's TEST protocol for ConvNeXt-S (`convnext_small.fb_in1k`: 288px, crop 1.0), as T's
-- `convnextin_fwd_s288`.
#eval IO.FS.writeFile "verified_mlir/convnextsin_fwd_s288.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnextsin_fwd_s288" 1000
    (Proofs.StableHLO.cnxFwdBanner Proofs.StableHLO.cnxSmall) (V := Proofs.StableHLO.cnxSmall)
    (bB := cnxInBS) (s := 288))

-- ── ConvNeXt-**B**, slug `convnextbin` — the eval forward ─────────────────────────────────────
-- B is S's depth at `[128,256,512,1024]`. Unlike S, it moves the STEM (96 → 128) and the HEAD
-- (768 → 1024) — see `CnxDims`.
--
-- **The `%dgi`/`%dgb`/`%dgn`/`%dgd`/`%dgapf` GAP backward is HAND-WRITTEN TEXT** (a declared
-- carve-out on both renderers), so its width is threaded by hand and NOTHING type-checks it. At T
-- and S the width is `768`; at B a missed `768` there would emit a graph whose GAP cotangent is
-- 768-wide against a 1024-wide stage — which the lowerer WOULD reject, but only after the artifact
-- was written and committed. The byte-identity gate at T and S says the threading does not disturb
-- those sizes; the shape check of the emitted B artifact says B's is right.
#eval IO.FS.writeFile "verified_mlir/convnextbin_fwd.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnextbin_fwd" 1000
    (Proofs.StableHLO.cnxFwdBanner Proofs.StableHLO.cnxBase) (V := Proofs.StableHLO.cnxBase)
    (bB := cnxInBS))
-- timm's TEST protocol for ConvNeXt-B (`convnext_base.fb_in1k`: 288px, crop 1.0).
#eval IO.FS.writeFile "verified_mlir/convnextbin_fwd_s288.mlir"
  (Proofs.StableHLO.convNextFwdRenderB "convnextbin_fwd_s288" 1000
    (Proofs.StableHLO.cnxFwdBanner Proofs.StableHLO.cnxBase) (V := Proofs.StableHLO.cnxBase)
    (bB := cnxInBS) (s := 288))

-- The entry name, the artifact path and `LEAN_MLIR_VARIANT` must agree or the shim refuses the call
-- ("entry mismatch"). These matter MORE for `drop` than for `wx` or `clip`, because `drop` also
-- changes the ARITY: a variant name that lost the marker would put an 18-input-wider graph behind
-- the plain `adam` path's artifact name, checkpoint and vmfb.
#guard Proofs.StableHLO.cnxAdamVariant 1 false false false true == "adamdrop"
#guard Proofs.StableHLO.cnxAdamVariant 4 false false false true == "adamdpdrop"
#guard Proofs.StableHLO.cnxAdamVariant 1 false true true true == "adamwxclipdrop"
#guard Proofs.StableHLO.cnxAdamVariant 4 false true true true == "adamdpwxclipdrop"
-- The marker must not LEAD: the driver keys its 4-region `[θ|m|v|ema]` blob off
-- `variant.startsWith "ema"`, so `emadrop` has to still start with "ema".
#guard (Proofs.StableHLO.cnxAdamVariant 1 true false false true).startsWith "ema"
-- And the driver's own predicate is a SUBSTRING test for `"drop"`; `"sd"` would fire on every
-- name containing `rmsdp`. ConvNeXt has no RMSProp variant, but the marker is shared with the nets
-- that do, so the property is checked here too rather than assumed to be EfficientNet's problem.
#guard !(Proofs.StableHLO.cnxAdamVariant 1).contains "drop"
#guard ((Proofs.StableHLO.cnxAdamVariant 4 false true true true).splitOn "drop").length == 2

-- The bf16 marker. ConvNeXt DERIVES its entry name from the variant, so `bf16` has to reach BOTH
-- `convNextAdamTrainStepFaithfulB`'s traversal AND `cnxAdamVariant`'s returned STRING — and the
-- second half is its own failure mode (a flag can reach the name function's SIGNATURE and the
-- function ignore it). These run the full concatenations, which is where a collision or a dropped marker appears.
#guard Proofs.StableHLO.cnxAdamVariant 1 false true true true true == "adamwxclipdropbf16"
#guard Proofs.StableHLO.cnxAdamVariant 4 false true true true true == "adamdpwxclipdropbf16"
-- And the marker must disturb NONE of the driver's substring predicates. `emaOn` is
-- `startsWith "ema"`, `cdOn` is `splitOn "do"`, `accOn` is `splitOn "acc"`. Note `drop` itself
-- is safe on `cdOn` for a reason that is easy to misread as luck: "drop" is d-r-o-p, so it does
-- not contain the substring "do". `bf16` adds no "do", no "acc", no "sd" and no "ema" prefix —
-- but it is checked rather than argued, because `rmsdp` containing "sd" is exactly the collision
-- between two OTHER markers that no placement rule would have predicted.
#guard !(Proofs.StableHLO.cnxAdamVariant 4 false true true true true).contains "do"
#guard !(Proofs.StableHLO.cnxAdamVariant 4 false true true true true).contains "acc"
#guard !(Proofs.StableHLO.cnxAdamVariant 4 false true true true true).contains "sd"
#guard !(Proofs.StableHLO.cnxAdamVariant 4 false true true true true).startsWith "ema"
-- And it must not BREAK `drop`'s own detection by appending after it.
#guard ((Proofs.StableHLO.cnxAdamVariant 4 false true true true true).splitOn "drop").length == 2
-- At `bf16 := false` every committed spelling is untouched — byte-identity, stated on the name as
-- well as on the bytes.
#guard Proofs.StableHLO.cnxAdamVariant 4 false true true true false == "adamdpwxclipdrop"
