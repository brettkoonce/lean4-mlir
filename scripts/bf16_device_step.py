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

def compile_replicated(path, g2, ctxs, replicas):
    """`g2.compile_mlir` at N replicas.

    ⚠ It is a SEPARATE path, not an edit to `bf16_gate2.compile_mlir`, deliberately: that function
    is shared with the correctness gate, and widening a gate's compile path to add a benchmark
    feature is how a gate stops gating. At `--replicas 1` this is not called at all and the two
    tools still agree byte for byte on how an artifact is loaded.
    ⚠⚠ It exists because some nets have NO single-device render — ViT-S and ViT-B are 4-replica
    only — so without it their graphs cannot be timed at all, and "we could not measure it" would
    have been mistaken for "it was not worth measuring".
    """
    import re as _re
    jax, jex, np_, xc, jmlir, jcomp, ir = ctxs
    client = jex.backend.get_backend()
    devs = client.local_devices()[:replicas]
    if len(devs) < replicas:
        raise SystemExit(f"need {replicas} devices, have {len(devs)} — set CUDA_VISIBLE_DEVICES")
    src = open(path).read()
    src, n = _re.subn(r"func\.func\s+(public\s+)?@[\w.]+\(", "func.func public @main(", src, count=1)
    if n == 0:
        raise SystemExit(f"{path}: no `func.func @…(` found — not a rendered artifact?")
    opts = xc.CompileOptions()
    opts.executable_build_options.num_replicas = replicas
    ctx = jmlir.make_ir_context()
    with ctx, ir.Location.unknown(ctx):
        mod = ir.Module.parse(src)
        return jcomp.backend_compile_and_load(client, mod, xc.DeviceList(tuple(devs)), opts, [])

def time_artifact(path, g2, ctxs, reps, replicas=1):
    jax, jex, _np, xc, jmlir, jcomp, ir = ctxs
    exe = g2.compile_mlir(path, ctxs) if replicas == 1 else \
          compile_replicated(path, g2, ctxs, replicas)
    devs = jex.backend.get_backend().local_devices()[:replicas]
    dev = devs[0]
    # Inputs come from the entry's declared parameter types. Zeros are fine: this is a wall-clock
    # measurement of a fixed graph, and no shape or kernel choice depends on the values.
    src = open(path).read()
    sig = re.search(r"func\.func\s+(?:public\s+)?@[\w.]+\((.*?)\)\s*->", src, re.S).group(1)
    # ⚠ The declared shapes are PER REPLICA — a DP render is "DATA-PARALLEL over N replicas" with
    # its own vbB, not a global batch — so every replica takes a buffer of the DECLARED shape, and
    # a replicated array (one identical shard per device) is what `execute_sharded` wants for that.
    # Values are zeros, so replicating rather than sharding the data inputs costs the measurement
    # nothing: no shape and no kernel choice depends on them.
    shapes = [[int(d) for d in m.group(1).split("x") if d]
              for m in re.finditer(r"tensor<([0-9x]*)f32>", sig)]
    if replicas > 1:
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        mesh = Mesh(np.array(devs), ("d",))
        rep_sh = NamedSharding(mesh, P())
        args = [jax.make_array_from_single_device_arrays(
                    tuple(sh), rep_sh,
                    [jax.device_put(np.zeros(sh, np.float32), d) for d in devs])
                for sh in shapes]
    else:
        args = [jax.device_put(np.zeros(sh, np.float32), dev) for sh in shapes]
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
    ap.add_argument("--replicas", type=int, default=1,
                    help="compile for N replicas; required for DP-only renders (ViT-S/B, MNv4)")
    a = ap.parse_args()
    g2 = _load_gate2()
    ctxs = g2._lazy()
    out = []
    for p in a.mlir:
        n, t = time_artifact(p, g2, ctxs, a.reps, a.replicas)
        out.append(t)
        print(f"  {t:8.2f} ms/step (device only, {n} inputs, median of {a.reps})   {p.split('/')[-1]}")
    if len(out) == 2:
        tag = "" if a.replicas == 1 else f" at {a.replicas} replicas (all-reduce included)"
        print(f"\n  speedup {out[0] / out[1]:.2f}x  — the GRAPH's, with no trainer around it{tag}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
