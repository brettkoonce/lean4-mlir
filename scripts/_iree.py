"""Shared IREE harness for the scripts that run a verified render through IREE.

IREE is the second lowerer: the forward/grad ties and the `*_probe_check.py` gates compile a
render with `iree-compile` and execute it, independently of the XLA path the trainers use.

Binaries. `$IREE_COMPILE` / `$IREE_RUN_MODULE` win. Otherwise the first directory holding the
tool among: the running interpreter's `bin/` (so a script run under an IREE venv's python gets
the compiler that matches its `iree.runtime`), the repo `.venv/bin`, `$PATH`, and the sibling
`lean4-jax` checkout's venv, which is where both tools live on the 4060 Ti box. The repo
`.venv` is the pinned JAX environment and must not gain an IREE package.

⛔ Keep the compiler and the runtime from ONE install. A skewed pair (e.g. a pip compiler with
the `iree-build/` source runtime) fails with "hal.command_buffer.dispatch signature mismatch",
which reads like a bad module rather than a bad pairing.

Backends: `llvm-cpu` (default, portable) and `cuda` (target `$IREE_CHIP`, default `sm_86` —
RTX 40-series cards reject `sm_89` in IREE, so they take the `sm_86` target).
"""
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BACKENDS = ("llvm-cpu", "cuda")


def _find(tool, env):
    if os.environ.get(env):
        return os.environ[env]
    for d in (pathlib.Path(sys.executable).parent, ROOT / ".venv" / "bin"):
        if (d / tool).exists():
            return str(d / tool)
    if shutil.which(tool):
        return shutil.which(tool)
    sib = ROOT.parent / "lean4-jax" / ".venv" / "bin" / tool
    if sib.exists():
        return str(sib)
    sys.exit(f"{tool} not found: set ${env}, put it on PATH, or run under a python whose venv "
             f"has it (the sibling lean4-jax/.venv does)")


def compiler():
    return _find("iree-compile", "IREE_COMPILE")


def runner():
    return _find("iree-run-module", "IREE_RUN_MODULE")


def target_flags(backend="llvm-cpu"):
    if backend == "llvm-cpu":
        return ["--iree-hal-target-backends=llvm-cpu"]
    if backend == "cuda":
        return ["--iree-hal-target-backends=cuda",
                f"--iree-cuda-target={os.environ.get('IREE_CHIP', 'sm_86')}"]
    sys.exit(f"unknown IREE backend {backend!r} (have {', '.join(BACKENDS)})")


def devices(backend="llvm-cpu"):
    # ⚠ `local-task` dies on big modules with exit 245 / -11 and EMPTY stderr, while `local-sync`
    # runs the identical vmfb (planning/archive/mnv4_verified.md §3f). A silent 245 is a
    # threading problem, not a bad render — so the CPU backend falls back to `local-sync`.
    return ["cuda"] if backend == "cuda" else ["local-task", "local-sync"]


def compile_mlir(mlir, vmfb, backend="llvm-cpu", extra=(), what=None):
    """`iree-compile` MLIR → vmfb; exits with the compiler's stderr on failure."""
    r = subprocess.run([compiler(), *target_flags(backend), *extra, str(mlir), "-o", str(vmfb)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"iree-compile FAILED{f' for {what}' if what else ''} ({mlir}):\n{r.stderr[:3000]}")
    return vmfb


def run_module(vmfb, fn, arrays, work, n_out=1, backend="llvm-cpu", devs=None, verbose=True):
    """Run `fn` from `vmfb` on `arrays` via `iree-run-module`, trying each device in turn.

    Inputs and outputs travel as `.npy` files under `work`; returns the `n_out` outputs.
    """
    os.makedirs(f"{work}/in", exist_ok=True)
    in_flags = []
    for i, a in enumerate(arrays):
        q = f"{work}/in/i{i}.npy"
        np.save(q, a)
        in_flags.append(f"--input=@{q}")
    outs = [f"{work}/o{j}.npy" for j in range(n_out)]
    devs = devs or devices(backend)
    r = None
    for dev in devs:
        r = subprocess.run([runner(), f"--device={dev}", f"--module={vmfb}", f"--function={fn}",
                            *in_flags, *[f"--output=@{o}" for o in outs]],
                           capture_output=True, text=True)
        if r.returncode == 0:
            if verbose and dev != devs[0]:
                print(f"  ran on --device={dev}")
            break
        if verbose:
            print(f"  --device={dev}: rc {r.returncode}"
                  f"{', empty stderr' if not r.stderr.strip() else ''}; trying the next")
    if r.returncode != 0:
        # A NEGATIVE returncode is a signal (-9 = OOM-killed, -11 = segfault) and comes with EMPTY
        # stderr, which reads like a silent failure of the module. Name it.
        sig = f", signal {-r.returncode}" if r.returncode < 0 else ""
        sys.exit(f"iree-run-module FAILED rc={r.returncode}{sig} on every device tried "
                 f"({', '.join(devs)}):\nSTDERR {r.stderr[:3000] or '<empty>'}\n"
                 f"STDOUT {r.stdout[:2000]}")
    return [np.load(o) for o in outs]


def compile_and_run(mlir, fn, arrays, work, n_out=1, backend="llvm-cpu", devs=None):
    vmfb = compile_mlir(mlir, f"{work}/m.vmfb", backend)
    return run_module(vmfb, fn, arrays, work, n_out, backend, devs)


def load_function(vmfb, module, fn="main"):
    """Load a vmfb into the in-process `iree.runtime` (CPU driver) and return `module.fn`.

    Needs a python with `iree.runtime`; `compiler()` then resolves to that install's compiler.
    """
    import iree.runtime as rt
    ctx = rt.SystemContext(config=rt.Config("local-task"))
    with open(vmfb, "rb") as f:
        ctx.add_vm_module(rt.VmModule.copy_buffer(ctx.instance, f.read()))
    return getattr(ctx.modules, module)[fn]
