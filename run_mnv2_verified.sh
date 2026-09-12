#!/usr/bin/env bash
# Launch the phase-4 verified/PJRT MobileNetV2 350-epoch ImageNet run.
#
#   systemd-run --user --unit=mnv2-verified --working-directory="$PWD" ./run_mnv2_verified.sh
#
# ⛔ NOT a bare `&`. The agent harness SIGKILLs background processes under memory pressure and
# killed a supervisor mid-run on 2026-09-10. `Linger=yes` is set on this account, so a systemd
# user unit survives both that and logout.
#
# ⛔ RUNDIR is NOT left at supervise.sh's default `/tmp/supervise_<job>`. EfficientNet-B0's master
# log was lost to a power cut mid-run that way, and the phase-2 300ep ViT at /home/skoonce/vit/
# has NO log at all — its 300 .bin files are the only record of that curve. Both logs live in-repo
# here, under the run's own archive dir, from before the first step.
#
# ⚠ Identify the supervisor with `pgrep -f supervise.sh`, NOT the pid systemd reports for the unit
# or a `$!` — `setsid` forks, so those are transient wrappers.
#
# The arm, the workers and the box are all asserted by mnv2-default-4gpu.conf's PRECHECK, which
# refuses the launch rather than running the wrong graph for 55 hours. Measured on this box
# 2026-09-10: 96-104 ms/step at w4 bf16 with SHIM_DETERMINISM at its new OFF default.
export RUNDIR=runs/2026-09-10-mnv2-verified-350ep
exec scripts/supervise.sh mnv2-default-4gpu
