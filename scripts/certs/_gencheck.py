"""Write-or-check for generators of committed files.

A generator calls `emit(path, text)` for every file it produces and `finish()` at the end. Run
plainly it writes them; run with `--check` it writes nothing and exits 1 naming every committed
file that differs from what the generator would write — the guard CI runs.
"""
import sys
from pathlib import Path

CHECK = "--check" in sys.argv
_drift = []
_seen = []


def emit(path, text):
    path = Path(path)
    _seen.append(path)
    if CHECK:
        if not path.exists() or path.read_text() != text:
            _drift.append(path)
    else:
        path.write_text(text)
        print(f"wrote {path}", flush=True)


def finish():
    if not CHECK:
        return
    if _drift:
        print(f"⛔ {len(_drift)} of {len(_seen)} generated file(s) differ from what "
              f"{Path(sys.argv[0]).name} emits — regenerate and commit:")
        for p in _drift:
            print(f"   {p}")
        sys.exit(1)
    print(f"✅ {len(_seen)} generated file(s) match {Path(sys.argv[0]).name}")
