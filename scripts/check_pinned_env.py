#!/usr/bin/env python3
"""check_pinned_env.py — assert the main `.venv` still IS the pinned environment.

    .venv/bin/python scripts/check_pinned_env.py      # exit 0 = pinned, 1 = drifted

⚠⚠ **THE FAILURE THIS EXISTS FOR IS SILENT-LOOKING, AND IT HAPPENED ON 2026-08-14.** Installing
`timm` into the main venv pulled `torch`, which requires `nvidia-cudnn-cu13`. That package installs
into the SAME `nvidia/cudnn/lib/` directory as the pinned `nvidia-cudnn-cu12` and **overwrites its
shared objects in place**. Every JAX convolution then died with

    Could not create cudnn handle: CUDNN_STATUS_NOT_INITIALIZED
    RET_CHECK failure (…/gpu_compiler.cc) dnn_support != nullptr

and the diagnosis was slow for one reason: **`pip list` still showed `nvidia-cudnn-cu12 9.23.2.1`.**
The metadata is per-package, the files are shared, so the pinned version can be "installed" and
absent at the same time. Nothing short of loading the library tells you which one is on disk.

▶ So this checks the LOADED versions, never the metadata — that is the whole design. It reads what
XLA reports and what the lockfile demands, and refuses on a difference.

⚠ It cannot be a `pip check` or a `pip freeze` diff for the same reason: both read metadata, and
metadata is exactly what was wrong.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOCK = REPO / "jax" / "requirements-cuda-lock.txt"


def want(pkg: str) -> str | None:
    """The version `requirements-cuda-lock.txt` pins for `pkg`."""
    if not LOCK.exists():
        sys.exit(f"no lockfile at {LOCK}")
    for line in LOCK.read_text().splitlines():
        m = re.match(rf"^{re.escape(pkg)}==([^\s;]+)", line.strip())
        if m:
            return m.group(1)
    return None


def main() -> int:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    bad: list[str] = []

    try:
        import jax
    except ImportError:
        sys.exit("jax is not importable from this interpreter — wrong venv?")

    want_jax = want("jax")
    if want_jax and jax.__version__ != want_jax:
        bad.append(f"jax {jax.__version__}, lockfile pins {want_jax}")

    # ── the cuDNN check, and it must come from a DEVICE, not from pip ──
    # `device_kind`/the client description is where XLA reports the cuDNN it actually dlopened.
    want_cudnn = want("nvidia-cudnn-cu12")          # e.g. "9.23.2.1" → banner reads "9.23.2"
    loaded = None
    try:
        devs = jax.devices()
    except Exception as e:                           # no GPU, or a broken plugin
        print(f"⚠ jax.devices() raised — {type(e).__name__}: {e}")
        devs = []
    if devs and devs[0].platform == "gpu":
        # The banner line XLA prints carries "DNN: X.Y.Z"; capture it by asking for the client's
        # own description rather than scraping stderr, which is not reliably ours to read.
        desc = getattr(devs[0].client, "platform_version", "") or ""
        m = re.search(r"[Dd][Nn][Nn][: ]+(\d+\.\d+\.\d+)", desc)
        if m:
            loaded = m.group(1)
        # ⭐ The load-bearing check regardless of what the banner says: RUN a convolution. This is
        # the operation the mismatch actually breaks, and it is cheap. A version string that agrees
        # while the conv fails is still a broken environment.
        try:
            import jax.numpy as jnp
            x = jnp.ones((1, 3, 8, 8), jnp.bfloat16)
            k = jnp.ones((4, 3, 3, 3), jnp.bfloat16)
            jax.lax.conv(x, k, (1, 1), "SAME").block_until_ready()
        except Exception as e:
            bad.append(
                f"a bf16 convolution FAILED on {devs[0]} — {type(e).__name__}: "
                f"{str(e).splitlines()[0][:160]}")
    if loaded and want_cudnn and not want_cudnn.startswith(loaded):
        bad.append(f"cuDNN {loaded} is loaded, lockfile pins {want_cudnn}")

    if bad:
        print("⛔ the pinned environment has DRIFTED:")
        for b in bad:
            print(f"   • {b}")
        print("\n   Most likely cause: something pulled torch into this venv, and torch requires\n"
              "   nvidia-cudnn-cu13, which overwrites the pinned cu12 cuDNN in the shared\n"
              "   nvidia/cudnn/lib/ directory. pip metadata will NOT show this.\n"
              "   Repair:\n"
              f"     .venv/bin/pip install --force-reinstall --no-deps nvidia-cudnn-cu12=={want_cudnn}\n"
              "   And keep timm/torch out of here — they have their own pinned env,\n"
              "   requirements-timm-lock.txt (CPU-only torch, zero nvidia-* packages).")
        return 1

    print(f"✅ pinned env OK — jax {jax.__version__}"
          + (f", cuDNN {loaded}" if loaded else "")
          + (", bf16 conv runs" if devs else " (no GPU visible; conv not exercised)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
