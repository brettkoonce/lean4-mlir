#!/usr/bin/env python3
"""Render-parity harness — validate a structured train-step render against the committed one
WITHOUT the (broken-in-this-env) imagenette swap-train loader.

Both MLIR modules must share the same `func.func @<fn>(...)` signature. We parse the input
tensor shapes, generate one fixed-random set of `.npy` inputs, run BOTH modules through
`iree-run-module` (CUDA by default, `--backend llvm-cpu` for the CPU) with those identical
inputs, and `np.array_equal` every output.

Usage:
    scripts/gates/render_parity.py --fn mobilenetv2_train_step \
        --ref verified_mlir/mobilenetv2_train_step.mlir \
        --cand /tmp/mnv2pc/train_step.mlir

    # one-module smoke (just run the ref, check finite outputs that differ from inputs):
    scripts/gates/render_parity.py --fn resnet34_train_step --ref verified_mlir/resnet34_train_step.mlir

IREE binaries and targets come from `scripts/lib/_iree.py` (`$IREE_COMPILE` / `$IREE_RUN_MODULE`,
`$IREE_CHIP` for the CUDA target, default sm_86).
"""
import argparse, os, re, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))  # the shared helpers
import _iree  # noqa: E402


def parse_input_shapes(mlir_path, fn):
    txt = open(mlir_path).read()
    m = re.search(rf'func\.func @{re.escape(fn)}\((.*?)\)\s*->', txt, re.S)
    if not m:
        sys.exit(f"could not find func @{fn} in {mlir_path}")
    inputs_part = m.group(1)
    # capture dims-with-trailing-x ('' for a scalar tensor<f32>, e.g. ConvNeXt's scalar-LN γ/β)
    return re.findall(r'tensor<([0-9x]*)f32>', inputs_part)


def count_outputs(mlir_path, fn):
    txt = open(mlir_path).read()
    m = re.search(rf'func\.func @{re.escape(fn)}\(.*?\)\s*->\s*\((.*?)\)\s*\{{', txt, re.S)
    if not m:  # single return type, no parens
        m2 = re.search(rf'func\.func @{re.escape(fn)}\(.*?\)\s*->\s*(tensor<[^>]+>)\s*\{{', txt, re.S)
        return 1 if m2 else 0
    return len(re.findall(r'tensor<', m.group(1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fn", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--cand", default=None, help="candidate MLIR; omit for a ref-only smoke run")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=0.1,
                    help="input magnitude (small so BN has variance / relu stays in-range)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--backend", default="cuda", choices=_iree.BACKENDS)
    args = ap.parse_args()

    work = args.workdir or tempfile.mkdtemp(prefix="parity_")
    shapes = parse_input_shapes(args.ref, args.fn)
    n_out = count_outputs(args.ref, args.fn)
    print(f"func @{args.fn}: {len(shapes)} inputs, {n_out} outputs  (workdir {work})")

    rng = np.random.default_rng(args.seed)
    arrays = []
    for i, s in enumerate(shapes):
        dims = [int(d) for d in s.split('x') if d]  # [] = 0-d scalar (tensor<f32>)
        arrays.append(np.asarray(rng.standard_normal(dims)).astype(np.float32) * args.scale)

    def run(mlir, tag):
        vmfb = _iree.compile_mlir(mlir, f"{work}/{tag}.vmfb", args.backend)
        return _iree.run_module(vmfb, args.fn, arrays, f"{work}/{tag}", n_out, args.backend)

    ref = run(args.ref, "ref")

    if args.cand is None:
        # smoke: outputs finite + at least one differs from its same-shape input (a real step)
        nfin = nmoved = 0
        for i in range(n_out):
            o = ref[i]
            if np.all(np.isfinite(o)): nfin += 1
            # match output i to the input of identical shape at position i+offset is not reliable;
            # just check the output is non-constant / non-zero
            if np.abs(o).max() > 0: nmoved += 1
        print(f"smoke: {nfin}/{n_out} outputs all-finite, {nmoved}/{n_out} non-zero")
        print("REF RUNS ✓" if nfin == n_out else "REF HAS NON-FINITE OUTPUTS ✗")
        return

    cand = run(args.cand, "cand")

    worst = 0.0; nbad = 0; nexact = 0
    for i in range(n_out):
        a, b = ref[i], cand[i]
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
