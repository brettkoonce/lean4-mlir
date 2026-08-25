#!/usr/bin/env python3
"""Sample nvidia-smi (all visible CUDA GPUs) to a CSV every INTERVAL seconds.

The CUDA peer of `log_gpu_temps.py`, which shells out to `rocm-smi` and therefore
does nothing on an NVIDIA box. Same purpose: correlate temps/clocks/power/util with
epochs and cooldowns over a long run.

Reading it: `power_W` near the cap with `util_pct` 100 = compute-bound; low power
despite util 100 = likely input/augmentation-bound. `sm_clk_MHz` sagging while
`temp_C` climbs is the throttle tell, and `throttle` names the reason NVML gives —
`SW Thermal Slowdown` / `HW Slowdown` are the ones that cost you wall clock, while
`Idle`/`None` are not throttling at all.

    nohup python3 scripts/log_gpu_temps_cuda.py runs/gpu_temps_cuda.csv 30 >/dev/null 2>&1 &
"""
import subprocess, sys, time, datetime, os

OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/gpu_temps_cuda.csv"
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 30
HDR = "ts,gpu,temp_C,power_W,power_cap_W,sm_clk_MHz,mem_clk_MHz,util_pct,mem_used_MiB,throttle\n"

FIELDS = ("index,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,"
          "utilization.gpu,memory.used,clocks_throttle_reasons.active")

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
if not os.path.exists(OUT) or os.path.getsize(OUT) == 0:
    with open(OUT, "w") as f:
        f.write(HDR)

while True:
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=20).decode()
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        rows = []
        for line in raw.strip().splitlines():
            c = [x.strip() for x in line.split(",")]
            if len(c) < 9:
                continue
            rows.append(f"{ts},{','.join(c)}\n")
        if rows:
            with open(OUT, "a") as f:
                f.writelines(rows)
    except Exception:
        # a transient nvidia-smi hiccup must not kill a logger that outlives a 44 h run
        pass
    time.sleep(INTERVAL)
