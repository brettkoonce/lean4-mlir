#!/usr/bin/env python3
"""Peak DEVICE memory of a rendered artifact, from XLA's own compiled-memory stats.

    .venv/bin/python scripts/bf16_peak_memory.py verified_mlir/<a>.mlir [<b>.mlir ...]

⚠⚠ WHY THIS EXISTS, AND WHY `nvidia-smi` IS THE WRONG TOOL. XLA's BFC allocator PREALLOCATES most
of the card (~73 %, i.e. ~11.68 GiB of a 16,380 MiB 4060 Ti — `planning/vit_convnext_sb_scaleup.md`).
So `nvidia-smi --query-gpu=memory.used` reports the POOL, not the graph, and two artifacts with very
different appetites read nearly the same. That mistake was made once in this repo: ConvNeXt-T was
reported at "12.2 GB f32 / 12.4 GB bf16" when its actual peaks are ~4.9 GiB and the 200 MiB
"difference" was pool noise (`bf16_renderer.md` §21.5).

`peak_memory_in_bytes` is what the compiler reserved for one execution: arguments + outputs +
temporaries, i.e. the number that decides whether a batch FITS. Compare it against the BFC budget,
not against the card.

⚠⚠ **AND THE BUDGET IS NOT 11.68 GiB UNLESS YOU LEAVE IT THERE.** That default is the CUDA
plugin's BFC `memory_fraction = 0.75`, not the card; `LEAN_MLIR_MEM_FRACTION` (verified path) and
`XLA_PYTHON_CLIENT_MEM_FRACTION` (this script's path) both raise it, and 0.97 gives **15.11 GiB**
on a 4060 Ti. A "does not fit" taken at the default is a statement about an unset environment
variable. Pass `--budget-gib 15.11` and set the fraction, or say which budget you meant.

⚠ **A compile-time peak is not independent of the allocator it was compiled against.** R50's fp32
`4×128` reads 11.52 G under the default budget and 11.91 G under the raised one, because XLA
rematerialises less when it has room. Quote the budget with the peak.

⚠ **`--replicas N` is not cosmetic for a DP render.** At the default `1`, a graph whose all-reduce
buffers only exist at N replicas is measured without them. ViT-S and ViT-B have no single-device
render at all, so for those the flag is the only way to compile the artifact as written.
"""
import argparse, importlib.util, os, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir", nargs="+")
    ap.add_argument("--budget-gib", type=float, default=11.68,
                    help="BFC budget per card (default 11.68 GiB, the plugin's 0.75 of a 4060 Ti; "
                         "15.11 at memory_fraction 0.97)")
    ap.add_argument("--replicas", type=int, default=1,
                    help="compile for N replicas; required for DP-only renders (ViT-S/B, MNv4), "
                         "and the only way the all-reduce buffers enter the peak")
    a = ap.parse_args()
    here = __file__.rsplit("/", 1)[0]
    spec = importlib.util.spec_from_file_location("bf16_gate2", f"{here}/bf16_gate2.py")
    g2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g2)
    # ⚠ The replicated compile path is BORROWED from `bf16_device_step.py`, not added to
    # `bf16_gate2.compile_mlir`: that function is shared with the correctness gate, and widening a
    # gate's compile path to serve a benchmark is how a gate stops gating. Same reason that file
    # gives for keeping it separate there; this is its second reader, not a second copy.
    spec2 = importlib.util.spec_from_file_location("bf16_device_step", f"{here}/bf16_device_step.py")
    ds = importlib.util.module_from_spec(spec2); spec2.loader.exec_module(ds)
    ctxs = g2._lazy()
    G = 1024 ** 3
    frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "unset (plugin default 0.75)")
    print(f"  budget {a.budget_gib:.2f} GiB/card · replicas {a.replicas} · "
          f"XLA_PYTHON_CLIENT_MEM_FRACTION={frac}")
    print(f"  {'artifact':52s} {'peak':>9s} {'args':>9s} {'temp':>9s}  {'of budget':>10s}")
    print("  " + "-" * 95)
    for p in a.mlir:
        exe = g2.compile_mlir(p, ctxs) if a.replicas == 1 else \
              ds.compile_replicated(p, g2, ctxs, a.replicas)
        st = exe.get_compiled_memory_stats()
        peak = st.peak_memory_in_bytes / G
        print(f"  {p.split('/')[-1]:52s} {peak:7.2f} G {st.argument_size_in_bytes/G:7.2f} G "
              f"{st.temp_size_in_bytes/G:7.2f} G  {100*peak/a.budget_gib:9.0f} %")
    return 0

if __name__ == "__main__":
    sys.exit(main())
