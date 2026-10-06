# Sourced by the job confs (and two probe launchers) from the repo root: which box this is, and
# the PJRT plugin + python it trains with. Sets BOX, BOX_PLUG, BOX_PY, BOX_SHIMPY — the job confs'
# names for scripts/platform/env.sh's PLATFORM_HOST, _PLUGIN, _PY and _SHIMPY, which is where the
# per-box facts live (ares: the repo's pinned .venv with xla_cuda12; 3060: /home/skoonce/.venv-cuda
# with xla_cuda13 and SHIM_PYTHON required).
. scripts/platform/env.sh
BOX=$PLATFORM_HOST; BOX_PLUG=$PLATFORM_PLUGIN; BOX_PY=$PLATFORM_PY
BOX_SHIMPY=(${PLATFORM_SHIMPY[@]+"${PLATFORM_SHIMPY[@]}"})
