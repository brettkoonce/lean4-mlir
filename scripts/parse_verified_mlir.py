#!/usr/bin/env python3
"""Every committed artifact must PARSE as StableHLO.

    .venv/bin/python scripts/parse_verified_mlir.py [verified_mlir/*.mlir]

⚠⚠ **THIS GATE EXISTS BECAUSE EIGHT UNPARSEABLE ARTIFACTS SHIPPED GREEN (2026-08-27).**
The R50 model-EMA renders named the shadow `{parameter}e`, and the stem BN gamma `%sg` therefore
produced **`%sge`** — which is the hardcoded block-local name inside `select_and_scatter`'s
comparator, the maxpool backward. MLIR's textual name scope is flat across nested regions, so the
files were rejected at parse with *"redefinition of SSA value '%sge'"*.

Every other gate in the repo passed on them:

  * `scripts/regen_verified_mlir.sh proofs` — byte-identity, and it only asks whether the OLD
    artifacts changed. A new one is new; there is nothing to diff it against.
  * `scripts/check_render_coverage.py` — reachability from the Proofs and Certs roots, and a
    renderer that emits garbage is still reachable.
  * `lake build` — the `#eval` writes a STRING. Lean has no opinion about its contents.
  * an arity check reading the signature with a regex — the signature was fine; the collision was
    2,000 lines further down, inside a region.

So the class of defect is: **a render can be well-typed in Lean, byte-stable, reachable, correctly
shaped, and still not be a program.** Nothing between the emitter and a GPU had ever read one.

⭐ It is CHEAP — 235 artifacts in well under a minute, no device, no execution. It should have
existed from the first artifact.
"""
import glob, sys, time

import jaxlib.mlir.ir as ir
from jax._src.interpreters import mlir as jmlir


def main(argv):
    files = sorted(argv[1:]) if len(argv) > 1 else sorted(glob.glob("verified_mlir/*.mlir"))
    if not files:
        sys.exit("no artifacts to parse")
    bad, t0 = [], time.monotonic()
    for p in files:
        # ⚠ A FRESH CONTEXT PER FILE. Reusing one leaks symbol names between modules, so a
        # collision in file N can surface as an error against file N+1 — which is a worse lie than
        # no gate at all.
        ctx = jmlir.make_ir_context()
        try:
            with ctx, ir.Location.unknown(ctx):
                ir.Module.parse(open(p).read())
        except Exception as e:
            first = next((l.strip() for l in str(e).split("\n") if "error:" in l), str(e)[:100])
            bad.append((p, first))
    dt = time.monotonic() - t0
    for p, m in bad:
        print(f"  ✗ {p}\n      {m[:140]}")
    if bad:
        print(f"\n✗ {len(bad)} of {len(files)} committed artifacts do not parse as StableHLO")
        return 1
    print(f"✓ {len(files)} committed artifacts parse as StableHLO ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
