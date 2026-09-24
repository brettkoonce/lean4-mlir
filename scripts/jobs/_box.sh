# Sourced by the job confs (and two probe launchers) from the repo root: which box this is, and
# the PJRT plugin + python it trains with. Sets BOX, BOX_PLUG, BOX_PY, BOX_SHIMPY.
#   ares — the repo's pinned .venv with the xla_cuda12 plugin; no SHIM_PYTHON needed.
#   3060 — /home/skoonce/.venv-cuda with xla_cuda13 (driver 610.57.04 / CUDA 13.3), and
#          SHIM_PYTHON is REQUIRED: that box's .venv/bin/python3 execs a retired venv.
ARES_PLUG=".venv/lib/python3.12/site-packages/jax_plugins/xla_cuda12/xla_cuda_plugin.so"
if [ -f "$ARES_PLUG" ]; then
  BOX=ares;  BOX_PLUG="$ARES_PLUG"; BOX_PY=".venv/bin/python3"; BOX_SHIMPY=()
else
  BOX=3060;  BOX_PLUG="/home/skoonce/.venv-cuda/lib/python3.12/site-packages/jax_plugins/xla_cuda13/xla_cuda_plugin.so"
  BOX_PY="/home/skoonce/.venv-cuda/bin/python3"; BOX_SHIMPY=(SHIM_PYTHON="$BOX_PY")
fi
