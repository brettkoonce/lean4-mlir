/-! Standalone repro for the ImageNet prefetch retention (2026-09-11). No GPU, no data, no shim.

    `lake build && .lake/build/bin/leakrepro <mode> [sizeMB] [iters] [append] [conc] [sleepMs] [zero|pipe]`

    * `direct`    — read on the main thread. Flat.
    * `task`      — the trainer's old path: `Handle.read` allocates on a pool thread, main drops
                    it. RssAnon climbs and `LazyFree` climbs with it: mimalloc answers a
                    cross-thread free of a huge block with `madvise(MADV_FREE)`, not a release.
    * `dedicated` — same on a fresh OS thread per read: flat RSS but the whole buffer refaults
                    every iteration (the exiting thread abandons its heap).
    * `intoswap`  — the fix: main allocates, hands the buffer through an `IO.Ref`, the task
                    fills it with `readInto` (readinto.c). Flat, ~zero LazyFree, ~zero faults.
    * `into`      — the fix WITHOUT the ref: the buffer captured by the closure arrives with
                    rc = -2 and `readInto` refuses it. Kept because it is the obvious first try.
    `sleepMs` spaces iterations like a training step; `MIMALLOC_PURGE_DELAY=0` looks like a fix at
    0 ms and does nothing at 200 ms. -/

def procField (file key : String) : IO String := do
  let s ← IO.FS.readFile file
  pure <| ((s.splitOn "\n").filter (·.startsWith key)).headD s!"{key}: ?"

def minflt : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/stat"
  -- field 10 (1-based) after the ") " that ends comm
  let rest := (s.splitOn ") ").getD 1 ""
  pure <| ((rest.splitOn " ").getD 7 "0").toNat!   -- rest[0]=state ... rest[7]=minflt

def report (i : Nat) (t0 : Nat) (f0 : Nat) : IO Unit := do
  let a ← procField "/proc/self/status" "RssAnon"
  let l ← procField "/proc/self/smaps_rollup" "LazyFree"
  let t ← IO.monoMsNow
  let f ← minflt
  IO.println s!"iter {i}  {a.trim}  {l.trim}  minflt={f - f0}  ms={t - t0}"

@[extern "lean_mlir_read_into"]
opaque readInto (h : @& IO.FS.Handle) (buf : ByteArray) (len : USize) : IO ByteArray

/-- fill a caller-supplied buffer exactly: the candidate fix. -/
def readExactInto (h : IO.FS.Handle) (buf : ByteArray) (n : Nat) : IO ByteArray := do
  let target := buf.size + n
  let mut acc := buf
  while acc.size < target do
    let before := acc.size
    let want := USize.ofNat (target - before)
    acc ← readInto h acc want
    if acc.size == before then throw <| IO.userError s!"eof after {acc.size} of {target}"
  pure acc

/-- one "batch": read `n` zero bytes exactly as `readExact` does. -/
def readBatch (h : IO.FS.Handle) (n : Nat) (doAppend : Bool) : IO ByteArray := do
  let chunk ← h.read (USize.ofNat n)
  if doAppend then pure (ByteArray.empty ++ chunk) else pure chunk

def main (args : List String) : IO Unit := do
  -- args: mode sizeMB iters append(0/1) concurrency
  let mode := args.getD 0 "task"
  let mb := (args.getD 1 "150").toNat!
  let iters := (args.getD 2 "200").toNat!
  let doAppend := (args.getD 3 "1") != "0"
  let conc := (args.getD 4 "1").toNat!
  let sleepMs := (args.getD 5 "0").toNat!
  let n := mb * 1024 * 1024
  let src := args.getD 6 "zero"
  let h ← if src == "pipe" then do
      let child ← IO.Process.spawn { cmd := "cat", args := #["/dev/zero"], stdout := .piped, stdin := .null }
      pure child.stdout
    else IO.FS.Handle.mk "/dev/zero" .read
  IO.println s!"mode={mode} size={mb}MB iters={iters} append={doAppend} conc={conc}"
  let f0 ← minflt
  let t0 ← IO.monoMsNow
  report 0 t0 f0
  let mut inflight : Array (Task (Except IO.Error ByteArray)) := #[]
  let mut sink : Nat := 0
  for i in [0:iters] do
    let buf ← match mode with
      | "direct" => readBatch h n doAppend
      | "task" | "dedicated" =>
          let prio := if mode == "dedicated" then Task.Priority.dedicated else Task.Priority.default
          -- keep `conc` reads outstanding, consume the oldest
          while inflight.size < conc do
            inflight := inflight.push (← IO.asTask (readBatch h n doAppend) prio)
          let t := inflight[0]!
          inflight := inflight.eraseIdx! 0
          let r ← IO.wait t
          IO.ofExcept r
      | "into" =>
          -- main allocates, the pool thread fills, main frees
          while inflight.size < conc do
            let buf := ByteArray.emptyWithCapacity n
            inflight := inflight.push (← IO.asTask (readExactInto h buf n) Task.Priority.default)
          let t := inflight[0]!
          inflight := inflight.eraseIdx! 0
          IO.ofExcept (← IO.wait t)
      | "intodirect" =>
          readExactInto h (ByteArray.emptyWithCapacity n) n
      | "intoref" =>
          while inflight.size < conc do
            let ref ← IO.mkRef (ByteArray.emptyWithCapacity n)
            inflight := inflight.push (← IO.asTask (do
              let buf ← ref.modifyGet (fun b => (b, ByteArray.empty))
              readExactInto h buf n) Task.Priority.default)
          let t := inflight[0]!
          inflight := inflight.eraseIdx! 0
          IO.ofExcept (← IO.wait t)
      | "refonly" =>
          -- modifyGet alone, no read: does the handoff itself hang?
          let ref ← IO.mkRef (ByteArray.emptyWithCapacity n)
          let t ← IO.asTask (do
            let buf ← ref.modifyGet (fun b => (b, ByteArray.empty))
            pure buf) Task.Priority.default
          IO.ofExcept (← IO.wait t)
      | "swaponly" =>
          -- take via ST.Ref.swap? use get + set instead, to compare
          let ref ← IO.mkRef (ByteArray.emptyWithCapacity n)
          let t ← IO.asTask (do
            let buf ← ref.get
            ref.set ByteArray.empty
            pure buf) Task.Priority.default
          IO.ofExcept (← IO.wait t)
      | "intoswap" =>
          while inflight.size < conc do
            let ref ← IO.mkRef (ByteArray.emptyWithCapacity n)
            inflight := inflight.push (← IO.asTask (do
              let buf ← ref.swap ByteArray.empty
              readExactInto h buf n) Task.Priority.default)
          let t := inflight[0]!
          inflight := inflight.eraseIdx! 0
          IO.ofExcept (← IO.wait t)
      | _ => throw <| IO.userError "mode: direct | task | dedicated | into | intodirect | intoref | refonly | swaponly | intoswap"
    sink := sink + buf.size + (buf.get! 7).toNat
    if sleepMs > 0 then IO.sleep sleepMs.toUInt32
    if (i + 1) % 10 == 0 then report (i + 1) t0 f0
  IO.println s!"done sink={sink}"
