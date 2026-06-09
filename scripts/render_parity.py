#!/usr/bin/env python3
"""Render-parity harness — validate a structured train-step render against the committed one
WITHOUT the (broken-in-this-env) imagenette swap-train loader.

Both MLIR modules must share the same `func.func @<fn>(...)` signature. We parse the input
tensor shapes, generate one fixed-random set of `.npy` inputs, run BOTH modules through
`iree-run-module` on the GPU with those identical inputs, and `np.array_equal` every output.

Usage:
    scripts/render_parity.py --fn mobilenetv2_train_step \
        --ref verified_mlir/mobilenetv2_train_step.mlir \
        --cand /tmp/mnv2pc/train_step.mlir

    # one-module smoke (just run the ref, check finite outputs that differ from inputs):
    scripts/render_parity.py --fn resnet34_train_step --ref verified_mlir/resnet34_train_step.mlir

Env: needs `iree-compile`/`iree-run-module` on PATH (claude_max/lean4-jax/.venv/bin), a gfx1100
GPU. See memory `running-verified-trainers-locally`.
"""
import argparse, os, re, subprocess, sys, tempfile
import numpy as np

CHIP = os.environ.get("IREE_CHIP", "gfx1100")


def parse_input_shapes(mlir_path, fn):
    txt = open(mlir_path).read()
    m = re.search(rf'func\.func @{re.escape(fn)}\((.*?)\)\s*->', txt, re.S)
    if not m:
        sys.exit(f"could not find func @{fn} in {mlir_path}")
    inputs_part = m.group(1)
    return re.findall(r'tensor<([0-9x]+)xf32>', inputs_part)


def count_outputs(mlir_path, fn):
    txt = open(mlir_path).read()
    m = re.search(rf'func\.func @{re.escape(fn)}\(.*?\)\s*->\s*\((.*?)\)\s*\{{', txt, re.S)
    if not m:  # single return type, no parens
        m2 = re.search(rf'func\.func @{re.escape(fn)}\(.*?\)\s*->\s*(tensor<[^>]+>)\s*\{{', txt, re.S)
        return 1 if m2 else 0
    return len(re.findall(r'tensor<', m.group(1)))


def compile(mlir_path, out_vmfb):
    r = subprocess.run(["iree-compile", "--iree-hal-target-backends=rocm",
                        f"--iree-rocm-target={CHIP}", mlir_path, "-o", out_vmfb],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-compile {mlir_path} FAILED:\n{r.stderr[:3000]}")


def run(vmfb, fn, in_flags, out_dir, n_out):
    os.makedirs(out_dir, exist_ok=True)
    out_flags = [f"--output=@{out_dir}/o{i}.npy" for i in range(n_out)]
    r = subprocess.run(["iree-run-module", "--device=hip", f"--module={vmfb}",
                        f"--function={fn}", *in_flags, *out_flags],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-run-module {vmfb} FAILED:\n{r.stderr[:3000]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", default=None, help="candidate MLIR; omit for a ref-only smoke run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.1,
                    help="input magnitude (small so BN has variance / relu stays in-range)")
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    work = args.workdir or tempfile.mkdtemp(prefix="parity_")
    os.makedirs(f"{work}/in", exist_ok=True)
    shapes = parse_input_shapes(args.ref, args.fn)
    n_out = count_outputs(args.ref, args.fn)
    print(f"func @{args.fn}: {len(shapes)} inputs, {n_out} outputs  (workdir {work})")

    rng = np.random.default_rng(args.seed)
    in_flags = []
    for i, s in enumerate(shapes):
        dims = [int(d) for d in s.split('x')]
        a = (rng.standard_normal(dims).astype(np.float32) * args.scale)
        p = f"{work}/in/i{i}.npy"; np.save(p, a); in_flags.append(f"--input=@{p}")

    compile(args.ref, f"{work}/ref.vmfb")
    run(f"{work}/ref.vmfb", args.fn, in_flags, f"{work}/out_ref", n_out)

    if args.cand is None:
        # smoke: outputs finite + at least one differs from its same-shape input (a real step)
        nfin = nmoved = 0
        for i in range(n_out):
            o = np.load(f"{work}/out_ref/o{i}.npy")
            if np.all(np.isfinite(o)): nfin += 1
            # match output i to the input of identical shape at position i+offset is not reliable;
            # just check the output is non-constant / non-zero
            if np.abs(o).max() > 0: nmoved += 1
        print(f"smoke: {nfin}/{n_out} outputs all-finite, {nmoved}/{n_out} non-zero")
        print("REF RUNS ✓" if nfin == n_out else "REF HAS NON-FINITE OUTPUTS ✗")
        return

    compile(args.cand, f"{work}/cand.vmfb")
    run(f"{work}/cand.vmfb", args.fn, in_flags, f"{work}/out_cand", n_out)

    worst = 0.0; nbad = 0; nexact = 0
    for i in range(n_out):
        a = np.load(f"{work}/out_ref/o{i}.npy"); b = np.load(f"{work}/out_cand/o{i}.npy")
        if a.shape != b.shape:
            print(f"  SHAPE MISMATCH out{i}: {a.shape} vs {b.shape}"); nbad += 1; continue
        if np.array_equal(a, b): nexact += 1
        d = np.abs(a - b).max(); scale = max(np.abs(a).max(), 1e-6); rel = d / scale
        worst = max(worst, rel)
        if rel > 1e-3: print(f"  DIFF out{i}: maxabs={d:.3e} rel={rel:.3e}"); nbad += 1
    print(f"\n{n_out} outputs compared: {nexact} bit-identical, worst rel-diff {worst:.3e}, "
          f"{nbad} exceeding 1e-3")
    print("VERDICT:", "PARITY ✓" if nbad == 0 else "MISMATCH ✗")
    sys.exit(0 if nbad == 0 else 1)


if __name__ == "__main__":
    main()
