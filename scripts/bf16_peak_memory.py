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
temporaries, i.e. the number that decides whether a batch FITS. Compare it against the BFC budget
(~71 % of the card), not against the card.
"""
import argparse, importlib.util, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mlir", nargs="+")
    ap.add_argument("--budget-gib", type=float, default=11.68,
                    help="BFC budget per card (default 11.68 GiB, a 16 GB 4060 Ti)")
    a = ap.parse_args()
    here = __file__.rsplit("/", 1)[0]
    spec = importlib.util.spec_from_file_location("bf16_gate2", f"{here}/bf16_gate2.py")
    g2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(g2)
    ctxs = g2._lazy()
    G = 1024 ** 3
    print(f"  {'artifact':52s} {'peak':>9s} {'args':>9s} {'temp':>9s}  {'of budget':>10s}")
    print("  " + "-" * 95)
    for p in a.mlir:
        st = g2.compile_mlir(p, ctxs).get_compiled_memory_stats()
        peak = st.peak_memory_in_bytes / G
        print(f"  {p.split('/')[-1]:52s} {peak:7.2f} G {st.argument_size_in_bytes/G:7.2f} G "
              f"{st.temp_size_in_bytes/G:7.2f} G  {100*peak/a.budget_gib:9.0f} %")
    return 0

if __name__ == "__main__":
    sys.exit(main())
