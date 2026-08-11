import LeanMlir.VerifiedNets
import LeanMlir.Proofs.Codegen.EfficientNetRender

/-! # The stochastic-depth mask is SHARDED, not replicated — `stochastic_depth.md` §5b

The gate that document left open, in its own words: *"Any stochastic-depth DP render needs an
**asymmetric-batch** gate, not the duplicated-batch one … this is an open design question and it
should be settled before the render lands, not after."*

**The hole.** The drop mask is a **per-example** input (`%dp<i> : tensor<Bxf32>`, one Bernoulli per
example), so under data parallelism replica `r` must receive mask rows `[r·b, (r+1)·b)` — the same
split `x` gets. The masks ride in the PARAMETER blob (`VerifiedTrain`'s `dropShapes`), and the DP
shim's rule was *"x and the labels shard, everything between them replicates"*, so every replica got
replica 0's mask and applied it to its own rows. ⚠ **That was true of the shim before any DP drop
render existed to expose it** — §5b's prediction, found by building the render it predicted about.

**Why the existing gates cannot see it.**
* `efficientnet-dp-check` hands both replicas the **same rows**, so a sharded mask and a replicated
  one produce identical results and it passes bit-exact either way. That is §5's duplicated-batch
  hole one axis over.
* `shard-check` (`DP([A|B]) = mean(single(A), single(B))`) needs the gated slot **linear in the
  gradient** — true of AdamW's `m` at `m = 0`, **false of RMSProp's buffer**, and EfficientNet is
  the net that wants stochastic depth *and* RMSProp. §5b called that an open design question.

**▶ THE CONSTRUCTION, and it is optimizer-agnostic — which is what §5b said did not exist.**

Duplicate the DATA and make only the MASK asymmetric, then **swap the mask halves**:

> run 1: batch `[x|x]`, mask `[m₀|m₁]`   ·   run 2: batch `[x|x]`, mask `[m₁|m₀]`

| | run 1 computes | run 2 computes | swap-invariant? |
|---|---|---|---|
| mask **sharded** (correct) | `mean(g(x,m₀), g(x,m₁))` | `mean(g(x,m₁), g(x,m₀))` | **YES — bit-identical** |
| mask **replicated** (the defect) | `g(x,m₀)` | `g(x,m₁)` | **NO** |

**Bit-identical is the bar, and it is an argument rather than a hope.** At two replicas the
collective is `(a + b)/2`, and IEEE-754 addition is **commutative** — `a + b` and `b + a` are the
same float, exactly — so a correctly sharded run is invariant under the swap to the bit. ⚠ It is
commutativity, **not** associativity: at more than two replicas the reduction is a tree whose order
a permutation changes, and the known answer degrades to approximate. That is why the render is 2
replicas and why this harness refuses above 2.

**What it does and does not establish.** It proves the mask inputs are **split rather than copied**.
It does *not* pin the shard OFFSET (a reversed split is also swap-invariant) — but the splitting
code is `n_replicas`-generic C shared with `x`, already gated by `shard-check`, so the only new
question was whether the flag is set at all. Say "the masks are sharded", never "the masks are
sharded in the right order".

⚠ Nothing here needs the optimizer to be linear, so it transfers to `emarmsdrop` unchanged — it
compares two runs of the SAME graph rather than a device answer against a host one.

    lake build drop-shard-check
    scripts/det_shim.sh /tmp/detshim
    CUDA_VISIBLE_DEVICES=0,1 PJRT_REPLICAS=2 LD_LIBRARY_PATH=/tmp/detshim \
      .lake/build/bin/drop-shard-check

⚠ **The control is fault injection, not a perturbed render**, because the defect lives in the shim
rather than in the artifact. `PJRT_DP_NO_MASK_SHARD=1` restores the replicating behaviour and the
gate must go red:

    PJRT_DP_NO_MASK_SHARD=1 CUDA_VISIBLE_DEVICES=0,1 PJRT_REPLICAS=2 … drop-shard-check   # rc=1
-/

open Proofs.StableHLO

/-- The entry name read OUT OF THE FILE, never derived from the path. `TestViTDpCheck`'s helper,
    and its reason holds here too: the shim checks the entry and refuses a mismatch, so a control
    file named after something else would fail on the name rather than on the number. -/
private def entryOf (path : String) : IO String := do
  let txt ← IO.FS.readFile path
  match (txt.splitOn "func.func @")[1]? with
  | none      => throw (IO.userError s!"{path}: no `func.func @` line — not a render?")
  | some rest => pure ("m." ++ rest.takeWhile (· != '('))

def main (argv : List String) : IO Unit := do
  -- ⚠ ONE harness for both nets, per `rms-tie`/`wdx-tie`/`shard-check` — a second copy is the
  -- double-writer disease one level down, in code. `convnext` selects the LayerNorm net, and it is
  -- NOT a cosmetic switch: see `nBnStats == 0` below, where the anti-vacuity half of the gate
  -- changes shape because an LN net has no replica-0-local batch statistics.
  let cnx  := argv.contains "convnext"
  let vit  := argv.contains "vit"
  let spec := if cnx then convnextVerified else if vit then vitVerified else efficientnetVerified
  let net  := spec.toNet
  let bs   := 32
  let replicas := ((← IO.getEnv "DROP_REPLICAS").bind (·.toNat?)).getD 2
  if replicas != 2 then
    throw (IO.userError s!"DROP_REPLICAS={replicas}: this gate's known answer is EXACT only at 2. \
It rests on f32 addition being COMMUTATIVE (a+b == b+a to the bit); above two replicas the \
collective is a tree whose order a permutation of the halves changes, and associativity does not \
hold, so swap-invariance stops being a bit-exactness claim.")
  let gbs := bs * replicas
  let nDrop := net.dropKeeps.size
  if nDrop == 0 then
    throw (IO.userError "the selected net has no drop sites — nothing to gate")
  -- ⚠ The path is still argv[0] when it is a path, so every committed invocation is unchanged; the
  -- `convnext` selector only moves the DEFAULT (and the spec above, which is what actually matters).
  -- ⚠ Every SELECTOR must be filtered out here, not just the first one. `vit` was left in the list
  -- when it was added and became argv[0] — i.e. the artifact PATH — so the run died opening a file
  -- called "vit". Loud, but the shape of it is the §2m positional-argument hazard in argv form.
  let dpPath := match argv.filter (fun a => a != "convnext" && a != "vit") with
    | p :: _ => p
    | []     => if cnx then "verified_mlir/convnext_adamdpdrop_train_step.mlir"
                else if vit then "verified_mlir/vit_adamdpdrop_train_step.mlir"
                else "verified_mlir/efficientnet_adamdpdrop_train_step.mlir"
  IO.println "stochastic-depth SHARD gate — duplicated batch, ASYMMETRIC mask, halves swapped"
  IO.println s!"  DP     : {dpPath} ({replicas} replicas × bs {bs} = global {gbs})"
  IO.println s!"  {net.specs.size} params ({net.nParams} floats), {nDrop} drop sites, \
backend {← IreeSession.backendName}"
  if (← IO.getEnv "PJRT_DP_NO_MASK_SHARD") == some "1" then
    IO.println "  ⚠ FAULT INJECTED: PJRT_DP_NO_MASK_SHARD=1 — the masks are REPLICATED"
  -- ⚠⚠ TWO CONTROLS, AND THEY SHOW DIFFERENT THINGS.
  --   `PJRT_DP_NO_MASK_SHARD=1` clears the shard flag while the buffer stays GLOBAL, so each
  --     replica is handed 64 elements for a `tensor<32xf32>` input and the shim REFUSES on arity.
  --     That is a stronger result than a numeric miss: once the buffer is sized globally,
  --     replication is not expressible — it is a type error, not a wrong answer.
  --   `DROP_FAULT=replicate` reconstructs the PRE-FIX WORLD exactly — buffer at the PER-DEVICE
  --     batch, shapes `#[bs]`, shard flag off — which type-checks and silently hands every replica
  --     the same rows. That is the defect as it actually existed, and it is the one ① must catch
  --     NUMERICALLY. A control that only ever produces a refusal would not show that.
  let faultRep := (← IO.getEnv "DROP_FAULT") == some "replicate"
  if faultRep then
    IO.println "  ⚠ FAULT INJECTED: DROP_FAULT=replicate — the PRE-FIX world (mask buffer at the per-device batch, shard flag off). ① must fire."

  -- ── the shared inputs: θ, and a DUPLICATED batch so the only asymmetry is the mask ──
  -- ⚠ Duplicating the data is what makes this test the MASK and nothing else. With asymmetric data
  -- the two runs would differ for a second reason and a red result would not localise.
  let mut parts : Array ByteArray := #[]
  let mut sd := 909
  for i in [0:net.specs.size] do
    let (dims, kind) := net.specs[i]!
    let n := dims.foldl (· * ·) 1
    parts := parts.push (← match kind with
      | 1 => F32.const n.toUSize 1.0
      | 2 => F32.const n.toUSize 0.0
      | _ =>
        let fanIn := if dims.size == 4 then dims[1]! * dims[2]! * dims[3]! else dims[0]!
        F32.heInit sd.toUSize n.toUSize (Float.sqrt (2.0 / fanIn.toFloat)))
    sd := sd + 1
  let θ := F32.concat parts
  let z ← F32.const net.nParams.toUSize 0.0
  let mut tl ← F32.const 3 0.0
  tl ← F32.write3 tl 0 1.0 0.1 0.001
  let nBnStats := 2 * net.bnChannels.foldl (· + ·) 0
  let bnSlots ← F32.const nBnStats.toUSize 0.0
  let xHalf ← F32.heInit 555 (bs * net.d0).toUSize 1.0
  let x := F32.concat #[xHalf, xHalf]                        -- the SAME rows twice
  let mut y : ByteArray := .empty
  for i in [0:gbs] do
    y := y.push (UInt8.ofNat (i % bs % net.nClasses)); y := y.push 0; y := y.push 0; y := y.push 0
  let shapes := packShapes (net.paramShapes ++ net.paramShapes ++ net.paramShapes
                              ++ #[#[], #[], #[]]
                              ++ (net.bnChannels.flatMap (fun c => #[#[c], #[c]]))
                              ++ Array.replicate nDrop #[if faultRep then bs else gbs])

  -- ── the two masks, drawn at DIFFERENT seeds, then assembled both ways round ──
  -- ⚠ REFUSE IF THEY AGREE. Two identical halves make the swap a no-op and the gate vacuously
  -- green — the `shard-check` "refuses as VACUOUS" rule, which exists because §2d.1 shipped a
  -- reversed-batch control that produced no difference at all.
  let m0 ← F32.dropScales net.dropKeeps bs 11
  let m1 ← F32.dropScales net.dropKeeps bs 977
  let mut same := 0
  for i in [0:nDrop * bs] do
    if F32.read m0 i.toUSize == F32.read m1 i.toUSize then same := same + 1
  IO.println s!"  masks agree on {same}/{nDrop * bs} slots (both are Bernoulli, so overlap is expected)"
  if same == nDrop * bs then
    throw (IO.userError "VACUOUS: the two mask halves are identical — swapping them is a no-op and \
this gate cannot distinguish a sharded mask from a replicated one")

  -- Interleave site-major: each mask input is one contiguous `gbs` row block, so half A's `bs`
  -- rows must sit directly before half B's WITHIN each site, not one whole mask after the other.
  let build (a b : ByteArray) : IO ByteArray := do
    let mut acc : Array ByteArray := #[]
    for s in [0:nDrop] do
      acc := acc.push (F32.slice a (s * bs) bs)
      acc := acc.push (F32.slice b (s * bs) bs)
    pure (F32.concat acc)
  -- In the reconstructed pre-fix world there is only ONE half to give, so the two runs differ by
  -- the WHOLE mask — which is exactly what a replicated mask does to a global batch.
  let mAB ← if faultRep then pure m0 else build m0 m1
  let mBA ← if faultRep then pure m1 else build m1 m0

  let run (masks : ByteArray) (tag : String) : IO ByteArray := do
    let vmfb := s!".lake/build/drop_shard_{tag}.vmfb"
    let target := (← IO.getEnv "IREE_BACKEND").getD "cuda"
    for p in [vmfb, s!".lake/build/drop_shard_{tag}_{target}.vmfb"] do
      if ← System.FilePath.pathExists p then IO.FS.removeFile p
    let sess ← mkSession dpPath
    let buf := F32.concat #[θ, z, z, tl, bnSlots, masks]
    IreeSession.mlpTrainStepVDP sess (← entryOf dpPath) x buf shapes y
      gbs.toUSize net.d0.toUSize net.nClasses.toUSize replicas.toUSize 0
      (if faultRep then 0 else nDrop).toUSize

  let oAB ← run mAB "ab"
  let oBA ← run mBA "ba"

  -- ── the comparison, and it is TWO checks pulling in OPPOSITE directions ──
  -- ⚠⚠ THE BN STATISTICS ARE NOT SWAP-INVARIANT, AND THAT IS THE SHARPER HALF OF THE GATE.
  -- `θ'`/`m'`/`v'`/`%loss` are computed from the ALL-REDUCED gradient, so a sharded mask makes them
  -- swap-invariant to the bit. The batch statistics are different in kind: they are replica-0-LOCAL
  -- (never all-reduced, read back from replica 0 only) and every BN layer downstream of a drop site
  -- sees masked activations — so replica 0's stats depend on REPLICA 0's MASK, which the swap
  -- changes. They must MOVE.
  --
  -- Neither check alone is evidence. Invariance alone is satisfied by a mask that reaches nothing
  -- (all-ones, or a site wired to a dead branch) — the ones-mask blindness of §7b, one level up.
  -- The BN movement is what witnesses that replica 0 actually RECEIVED a different mask, and the
  -- invariance is what witnesses that the collective nevertheless saw both. Together they say the
  -- masks were split; either alone says much less.
  let P := net.nParams
  -- ⚠ `%loss` is NOT in the all-reduced group, and putting it there cost a run. It is the
  -- report-only forward scalar (§5's carve-out list: emitted text, on no gradient path, outside
  -- every faithfulness theorem) — computed on each replica from ITS OWN rows and mask, with only
  -- replica 0's returned. So it is replica-0-LOCAL exactly as the batch statistics are, and it must
  -- MOVE under the swap. The first version counted it as all-reduced and ① read
  -- `12061076/12061077` — one output, which is the tell: a wiring defect moves thousands.
  -- ⚠ The invariant set is NOT a contiguous prefix. The return layout is
  -- `θ' ++ m' ++ v' ++ [%loss, %bc1, %bc2] ++ bnstats`, so `%loss` sits at 3P — INSIDE any range
  -- that reaches the scalars. Taking `3P + 2` as "the all-reduced part" therefore still swept it
  -- up, and ① read one differing output at index 3P, which is `%loss` itself. Gate the three
  -- parameter regions, and check the two passthrough scalars separately.
  let nRed := 3 * P                          -- θ' | m' | v' — all-reduced
  let mut diff := 0
  let mut firstDiff := 0
  let mut worst : Float := 0.0
  let mut scale : Float := 0.0
  for i in [0:nRed] do
    let a := F32.read oAB i.toUSize
    let b := F32.read oBA i.toUSize
    if a != b then
      if diff == 0 then firstDiff := i
      diff := diff + 1
    if (a - b).abs > worst then worst := (a - b).abs
    if a.abs > scale then scale := a.abs
  let rel := if scale > 0.0 then worst / scale else 0.0
  -- `%loss` sits at index 3P, immediately after the three param regions; `%bc1`/`%bc2` follow it
  -- and are pure passthrough (the driver's own scalars echoed back), so they must not move.
  let lossAB := F32.read oAB (3 * P).toUSize
  let lossBA := F32.read oBA (3 * P).toUSize
  let bcSame := F32.read oAB (3*P+1).toUSize == F32.read oBA (3*P+1).toUSize
             && F32.read oAB (3*P+2).toUSize == F32.read oBA (3*P+2).toUSize
  let mut bnDiff := 0
  let mut bnWorst : Float := 0.0
  for i in [3 * P + 3:3 * P + 3 + nBnStats] do
    let a := F32.read oAB i.toUSize
    let b := F32.read oBA i.toUSize
    if a != b then bnDiff := bnDiff + 1
    if (a - b).abs > bnWorst then bnWorst := (a - b).abs
  IO.println s!"  ① all-reduced regions: {nRed - diff}/{nRed} BIT-IDENTICAL under the swap \
(max abs {worst}, rel {rel}{if diff == 0 then "" else s!", first at output {firstDiff}"})"
  IO.println s!"  ②a %loss (replica-0-local, report-only): {lossAB} vs {lossBA} — \
{if lossAB == lossBA then "IDENTICAL ⚠" else "MOVED"}"
  IO.println s!"  ② replica-0-local BN stats: {bnDiff}/{nBnStats} MOVED under the swap \
(max abs {bnWorst}) — they must, and a 0 here means replica 0 saw the same mask twice"

  if diff != 0 then
    throw (IO.userError s!"① FAILED — THE MASK IS NOT SHARDED. {diff} of {nRed} all-reduced \
outputs move when the two mask halves are swapped, rel {rel}.\n\
\n\
A correctly SHARDED mask makes the all-reduced regions swap-invariant to the BIT: replica 0 gets m₀ \
and replica 1 gets m₁ in one run and the other way round in the other, and the collective's (a+b)/2 \
is COMMUTATIVE in f32. A REPLICATED mask makes run 1 compute g(x,m₀) on both replicas and run 2 \
compute g(x,m₁) — a different function.\n\
\n\
Check that `nShardTail` reaches `pjrt_ffi_invoke_f32_dp2`, and that the mask buffer is sized at the \
GLOBAL batch (`dropShapes` must be `#[gbs]`, not `#[bs]`).")
  if !bcSame then
    throw (IO.userError "the %bc1/%bc2 passthrough scalars moved under the swap — they are the \
driver's own inputs echoed back and cannot depend on the mask at all. Something is misaligned in \
the output layout, not in the sharding.")
  if lossAB == lossBA then
    throw (IO.userError s!"②a FAILED — VACUOUS. `%loss` did not move when the mask halves were \
swapped. It is computed on replica 0 from replica 0's rows and replica 0's MASK, so a swap that \
reaches the device must change it. Identical means replica 0 saw the same mask twice.")
  -- ⚠⚠ ON A LAYERNORM NET THERE ARE NO BATCH STATISTICS, SO ②a CARRIES THE ANTI-VACUITY LOAD ALONE
  -- — and that is a real weakening, stated rather than papered over. EfficientNet witnesses "replica
  -- 0 received a different mask" with tens of thousands of replica-0-local floats; ConvNeXt has
  -- exactly ONE, `%loss`. It is still a genuine witness (report-only, computed on replica 0 from
  -- replica 0's rows and mask), and it is not the only thing standing between this gate and
  -- vacuity: the harness already REFUSES above if the two mask halves are equal. But a one-scalar
  -- anti-vacuity check would not survive a defect that happened to leave `%loss` fixed, and nothing
  -- here rules that out.
  if nBnStats == 0 then
    IO.println s!"  ⚠ {spec.name} normalises with LayerNorm — there are NO replica-0-local batch \
statistics, so ②a (`%loss`) is the WHOLE anti-vacuity half. Weaker than the BN nets' ②, and the \
mask-halves-differ refusal above is the other thing keeping ① from being vacuous."
  else if bnDiff == 0 then
    throw (IO.userError s!"② FAILED — VACUOUS. Not one of the {nBnStats} batch statistics moved \
when the mask halves were swapped. Those are replica-0-LOCAL, and every BN layer downstream of a \
drop site sees masked activations, so replica 0 receiving a different mask MUST move them. Zero \
movement means replica 0 saw the same mask both times — i.e. ① passed for the wrong reason (a mask \
that reaches nothing is trivially swap-invariant, which is §7b's ones-mask blindness one level up).")

  IO.println s!"✓ the drop masks are SHARDED, not replicated: the {nRed} all-reduced outputs are \
BIT-IDENTICAL under a swap of the two mask halves while {bnDiff}/{nBnStats} replica-0-local batch \
statistics MOVE, on halves that differ in {nDrop * bs - same} of {nDrop * bs} slots. ⚠ This pins \
split-vs-copied, NOT the shard offset — a reversed split is swap-invariant too, and the offset \
rides on the same `n_replicas`-generic C that `shard-check` already gates through `x`."
