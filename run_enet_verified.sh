#!/usr/bin/env bash
# Launch the phase-4 verified/PJRT EfficientNet-B0 350-epoch ImageNet run
# (bf16, RMSProp + EMA + stochastic depth + classifier dropout: `emarmsdp64dropdobf16`).
#
#   systemd-run --user --unit=enet-verified --working-directory="$PWD" ./run_enet_verified.sh
#
# ⛔ NOT a bare `&` — see run_mnv2_verified.sh: the agent harness SIGKILLs background processes
# under memory pressure; a systemd user unit (`Linger=yes`) survives that and logout.
#
# ⛔ RUNDIR in-repo, from before the first step. This net's phase-2 master log was lost to a power
# cut when it lived under /tmp.
#
# ⚠ Identify the supervisor with `pgrep -f supervise.sh`, NOT the pid systemd reports for the unit
# or a `$!` — `setsid` forks, so those are transient wrappers.
#
# The arm, the worker count, the exe's freshness and the box are asserted by
# enet-default-4gpu.conf's PRECHECK, which refuses the launch rather than running the wrong graph
# for three days. Throughput measured on this box 2026-09-12: runs/2026-09-12-enet-bf16-sweep/.
export RUNDIR=runs/2026-09-12-enet-verified-350ep
exec scripts/supervise.sh enet-default-4gpu
