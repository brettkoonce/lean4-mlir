#!/usr/bin/env python3
"""Sample rocm-smi (both gfx1100s) to a CSV every INTERVAL seconds.

Runs alongside the paced ViT training so temps/clocks/power/util can be
correlated with epochs and the 30-min pauses. Junction temp + sclk are the
throttle tells; power near TBP + util 100% = compute-bound, low power despite
util 100% = likely input/augmentation-bound.

    nohup python3 log_gpu_temps.py runs/gpu_temps.csv 30 > /dev/null 2>&1 &
"""
import json, subprocess, sys, time, datetime, os

OUT = sys.argv[1] if len(sys.argv) > 1 else "runs/gpu_temps.csv"
INTERVAL = int(sys.argv[2]) if len(sys.argv) > 2 else 30
HDR = "ts,card,edge_C,junction_C,mem_C,sclk_MHz,power_W,use_pct\n"

def num(s):
    return "".join(c for c in str(s) if (c.isdigit() or c == "."))

os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
if not os.path.exists(OUT) or os.path.getsize(OUT) == 0:
    with open(OUT, "w") as f:
        f.write(HDR)

while True:
    try:
        raw = subprocess.check_output(
            ["rocm-smi", "--showtemp", "--showpower", "--showclocks", "--showuse", "--json"],
            stderr=subprocess.DEVNULL, timeout=20)
        j = json.loads(raw)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        rows = []
        for card in sorted(j):
            d = j[card]
            rows.append(",".join([
                ts, card,
                num(d.get("Temperature (Sensor edge) (C)", "")),
                num(d.get("Temperature (Sensor junction) (C)", "")),
                num(d.get("Temperature (Sensor memory) (C)", "")),
                num(d.get("sclk clock speed:", "")),
                num(d.get("Average Graphics Package Power (W)", "")),
                num(d.get("GPU use (%)", "")),
            ]))
        with open(OUT, "a") as f:
            f.write("\n".join(rows) + "\n")
    except Exception:
        # transient rocm-smi hiccup — skip this sample, keep going
        pass
    time.sleep(INTERVAL)
