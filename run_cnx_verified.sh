#!/usr/bin/env bash
# Launch the phase-4 verified/PJRT ConvNeXt-T 300-epoch ImageNet run
# (bf16, AdamW + wd-exclude + grad clip + drop-path: `adamdpwxclipdropbf16`, batch 4x64).
#
#   systemd-run --user --unit=cnx-verified --working-directory="$PWD" ./run_cnx_verified.sh
#
# ⛔ NOT a bare `&` — the agent harness SIGKILLs background processes under memory pressure (it
# killed two watchers of the R34 bf16 run). A systemd user unit (`Linger=yes`) survives that and
# logout. ⚠ `| head` does not stop a trainer either.
#
# ⛔ RUNDIR in-repo, from before the first step: R34's phase-2 master log was lost to a power cut
# when it lived under /tmp.
#
# ⚠ Identify the supervisor with `pgrep -f supervise.sh`, NOT the pid systemd reports for the unit
# or a `$!` — `setsid` forks, so those are transient wrappers.
#
# ⭐ WHY THIS RUN MATTERS BEYOND ITS OWN NUMBER. ConvNeXt has no BatchNorm, so it is the net that
# isolates the LOWERER rather than a statistic group. EfficientNet sits a persistent -0.27 below its
# reference (0 of 300 epochs above it) with the BN group the named suspect, and MobileNetV2's tie
# shows BN *density* is not the explanation. If THIS pair ties, the BN group is the likely cause and
# the lowerer is clean; if it also carries a persistent offset, the lowerer or the feed is
# implicated fleet-wide. That reading only counts because the batch now matches (4x64 = global 256,
# the reference's own batch) — see the rescope note at the top of the conf.
#
# The arm, the batch (read off the ARTIFACT, not the env), the loss divisor, the exe's freshness,
# the shim's freshness, the LR and the box are all asserted by cnx-default-4gpu.conf's PRECHECK,
# which refuses the launch rather than running the wrong graph for four days.
# Throughput measured on this box 2026-09-17: runs/2026-09-17-cnx-sweep/.
export RUNDIR=runs/2026-09-18-cnx-verified-300ep
exec scripts/supervise.sh cnx-default-4gpu
