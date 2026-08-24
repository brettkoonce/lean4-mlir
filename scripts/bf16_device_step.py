#!/usr/bin/env python3
"""Time a rendered train-step artifact as a BARE DEVICE EXECUTABLE — no trainer, no shim, no
parameter round trip. This is what separates "the renderer" from "the system".

    .venv/bin/python scripts/bf16_device_step.py verified_mlir/<f32>.mlir verified_mlir/<bf16>.mlir

⚠⚠ WHY THIS EXISTS. `planning/bf16_renderer.md` has now recorded THREE ways a trainer's ms/step
can be a system result rather than a statement about the emitted graph:

  1. the shim feed          (§13.2 — MobileNetV2 is 1.92x on one GPU and 1.37x on four)
  2. the f32 all-reduce     (§13.2 — same measurement)
  3. the PARAMETER ROUND TRIP (§16.2 — ConvNeXt-T is 1.29x with `PJRT_FFI_RESIDENT=1` and
     1.11x without it, SAME GRAPH: its 540 parameter tensors are 327 MB and cross PCIe every
     step when residency is off, which it is BY DEFAULT)

⚠ `LEAN_MLIR_BENCH_SYNTH=1` controls for (1) ONLY. It removes the data feed and leaves the
parameter traffic untouched, so a synth-vs-real agreement does NOT rule out (3) — §14.1 used it
on EfficientNet-B0 and was read as ruling out more than it did.

This script has none of the three: it compiles the artifact, feeds zeros, and times the
executable. Whatever speedup it reports is the graph's.

▶ Peer tool: `scripts/bf16_gate2.py`, which answers the different question of whether the bf16
reached the hardware at all.
"""
import argparse, importlib.util, re, sys, time
import numpy as np

def _load_gate2():
    """Reuse bf16_gate2's compile path so both tools agree on how an artifact is loaded."""
    here = __file__.rsplit("/", 1)[0]
    spec = importlib.util.spec_from_file_location("bf16_gate2", f"{here}/bf16_gate2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def time_artifact(path, g2, ctxs, reps):
    jax, jex, _np, xc, jmlir, jcomp, ir = ctxs
    exe = g2.compile_mlir(path, ctxs)
    dev = jex.backend.get_backend().local_devices()[0]
    # Inputs come from the entry's declared parameter types. Zeros are fine: this is a wall-clock
    # measurement of a fixed graph, and no shape or kernel choice depends on the values.
    src = open(path).read()
    sig = re.search(r"func\.func\s+(?:public\s+)?@[\w.]+\((.*?)\)\s*->", src, re.S).group(1)
    args = [jax.device_put(np.zeros([int(d) for d in m.group(1).split("x") if d], np.float32), dev)
            for m in re.finditer(r"tensor<([0-9x]*)f32>", sig)]
    def once():
        # PJRT enqueues the whole step, so blocking on ONE result blocks on the step. Touching
        # all of them (561 on ConvNeXt) would add a host round trip per output and measure that.
        exe.execute_sharded(args).disassemble_into_single_device_arrays()[0][0].block_until_ready()
    once()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); once(); ts.append(time.perf_counter() - t0)
    return len(args), float(np.median(ts)) * 1e3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir", nargs="+", help="artifacts to time; with exactly two, a speedup is reported")
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()
    g2 = _load_gate2()
    ctxs = g2._lazy()
    out = []
    for p in a.mlir:
        n, t = time_artifact(p, g2, ctxs, a.reps)
        out.append(t)
        print(f"  {t:8.2f} ms/step (device only, {n} inputs, median of {a.reps})   {p.split('/')[-1]}")
    if len(out) == 2:
        print(f"\n  speedup {out[0] / out[1]:.2f}x  — the GRAPH's, with no trainer around it")
    return 0

if __name__ == "__main__":
    sys.exit(main())
