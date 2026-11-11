#!/usr/bin/env python3
"""bench_run.py — run one ImageNet trainer for N epochs on real data, timestamp every line it
prints, and append one TSV row per epoch. Called by scripts/bench/bench.sh, which resolves the job
conf into the command and environment; this file knows the two trainers' log formats and nothing
about confs.

    bench_run.py --row-dir D --tsv T --job J --path jax|verified --net N --epochs 2 \
                 [--expect-env KEY=VAL ...] -- <cmd ...>

What a row measures (planning/next_session_bench_and_a3.md §0 is why each rule exists):
  * epoch wall-clock = consecutive epoch-end lines (val included, the previous epoch's checkpoint
    write included). Epoch 1 also carries startup and compile; epoch 2 onward is the number.
  * train / val split: the JAX trainer prints its own `[Xs train, Ys val]`; the verified driver
    prints no times, so its split comes from this file's timestamps (`Epoch k/E: loss=` ends
    training, `epoch k: test_acc` ends val).
  * steady step time: inside the epoch, from the first progress line at in-epoch step >= 200 to the
    last one. Never the trainer's printed average, which includes compile, and never the first
    steps after a val pass, which drain a queue the producers filled during it (burst rate).
  * img/s = the epoch's images / train seconds. GPU memory is what nvidia-smi reports, which for
    the JAX path is XLA's preallocated arena, not the need.
  * --images N: stop after N training images instead of after --epochs. One row, epoch 0, step time
    over the images after the first quarter (compile and queue fill stay out of it), img/s from the
    step time and the images a step consumes. Needs no knob in either trainer.
  * the live process's environment is read back from /proc and saved; an --expect-env mismatch
    (the checkpoint redirect, above all) kills the run before it writes a step.
"""
import argparse, json, os, re, signal, socket, subprocess, sys, threading, time

IMAGENET_TRAIN = 1281167

# val runs every few epochs on some trainers (ViT: 5); the rest end "[Xs train, val skipped]"
JAX_EPOCH = re.compile(r'^\[Epoch (\d+)\](?:.*?val_top1=\d+/\d+ \(([\d.]+)\))?.*\[([\d.]+)s train, (?:([\d.]+)s val|val skipped)\]')
JAX_STEP = re.compile(r'^\s+step (\d+)/(\d+)\s+loss=\S+.*\(([\d.]+)ms/step avg\)')
JAX_SPE = re.compile(r'steps_per_epoch=(\d+)')
VER_TRAIN_END = re.compile(r'^Epoch (\d+)/(\d+): loss=')
VER_EPOCH = re.compile(r'^\s+epoch (\d+): (?:test|val)_acc = \d+/\d+ = ([\d.]+)%')
VER_STEP = re.compile(r'^\s+step (\d+)/(\d+): loss=')
ENV_KEEP = re.compile(r'^(LEAN_MLIR_|PJRT_|SHIM_|XLA_|TF_|JAX_|TFDS_|NVIDIA_|CUDA_VISIBLE_DEVICES$|RUNDIR$)')
WARM_STEP = 200

COLS = ["date_utc", "host", "gpu", "n_gpu", "driver", "cpu", "nproc", "ram_gb", "git", "job", "path",
        "net", "epoch", "wall_s", "train_s", "val_s", "step_ms", "img_s", "gpu_util", "gpu_mem_mib",
        "cpu_busy", "top1", "status", "row_dir"]


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def host_info(devs):
    q = sh("nvidia-smi", "-i", devs, "--query-gpu=name,driver_version", "--format=csv,noheader")
    rows = [r.split(", ") for r in q.splitlines() if r]
    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")
    mem_kb = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal"))
    head = sh("git", "rev-parse", "--short", "HEAD")
    dirty = sh("git", "status", "--porcelain", "--untracked-files=no")
    return dict(host=socket.gethostname(), gpu=rows[0][0] if rows else "?", n_gpu=len(rows),
                driver=rows[0][1] if rows else "?", cpu=cpu, nproc=os.cpu_count(),
                ram_gb=round(mem_kb / 1048576), git=head + ("+dirty" if dirty else ""))


class Telemetry(threading.Thread):
    """nvidia-smi + /proc/stat every 5 s: (t, [mem MiB], [util %], cpu busy fraction since last)."""

    def __init__(self, devs, t0):
        super().__init__(daemon=True)
        self.devs, self.t0, self.samples, self.stop = devs, t0, [], threading.Event()

    @staticmethod
    def cpu():
        f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(f), f[3] + f[4]   # total, idle + iowait

    def run(self):
        last = self.cpu()
        while not self.stop.wait(5):
            q = sh("nvidia-smi", "-i", self.devs, "--query-gpu=memory.used,utilization.gpu",
                   "--format=csv,noheader,nounits")
            now = self.cpu()
            dt, di = now[0] - last[0], now[1] - last[1]
            last = now
            try:
                mu = [tuple(int(v) for v in r.split(", ")) for r in q.splitlines() if r]
            except ValueError:
                continue
            self.samples.append((time.monotonic() - self.t0, [m for m, _ in mu], [u for _, u in mu],
                                 1 - di / dt if dt else 0.0))

    def window(self, a, b):
        s = [x for x in self.samples if a <= x[0] <= b and x[2]]
        if not s:
            return "", "", ""
        util = sum(sum(x[2]) / len(x[2]) for x in s) / len(s)
        mem = max(max(x[1]) for x in s)
        cpu = sum(x[3] for x in s) / len(s)
        return f"{util:.0f}", str(mem), f"{cpu:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-dir", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--path", choices=["jax", "verified"], required=True)
    ap.add_argument("--net", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--timeout-min", type=float, default=180)
    ap.add_argument("--images", type=int, default=0, help="stop after this many images, not --epochs")
    ap.add_argument("--images-per-step", type=int, default=0,
                    help="verified: batch x replicas; JAX reads its steps_per_epoch line")
    ap.add_argument("--expect-env", action="append", default=[])
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    os.makedirs(a.row_dir, exist_ok=True)
    devs = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    info = host_info(devs)

    t0 = time.monotonic()
    log = open(os.path.join(a.row_dir, "train.log"), "w", buffering=1)
    log.write(f"# {time.strftime('%F %T', time.gmtime())} UTC  {' '.join(cmd)}\n# t_s\tline\n")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         errors="replace", bufsize=1, start_new_session=True)
    tel = Telemetry(devs, t0)
    tel.start()

    def kill(why):
        log.write(f"{time.monotonic() - t0:.3f}\t[bench] {why} — stopping the process group\n")
        for sig, wait in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                return
            try:
                p.wait(wait)
                return
            except subprocess.TimeoutExpired:
                pass

    # what the trainer actually got, read back from the live process
    status = "ok"

    def check_env():
        try:
            raw = open(f"/proc/{p.pid}/environ", "rb").read().split(b"\0")
        except OSError:
            return None
        env = dict(kv.decode(errors="replace").split("=", 1) for kv in raw if b"=" in kv)
        kept = {k: v for k, v in sorted(env.items()) if ENV_KEEP.match(k)}
        json.dump(kept, open(os.path.join(a.row_dir, "env.json"), "w"), indent=1)
        bad = [e for e in a.expect_env if env.get(e.split("=", 1)[0]) != e.split("=", 1)[1]]
        return bad

    ep_end, train_end, steps = {}, {}, []      # epoch -> t ; epoch -> t ; (t, epoch, in-epoch step)
    jax_train, jax_val, top1 = {}, {}, {}
    spe, done_at = None, None
    deadline = t0 + a.timeout_min * 60
    why = {}
    budget = {}

    def cum(rows):
        """(t, epoch, images-so-far in steps): the in-epoch counter restarts, the budget does not."""
        out, base, last = [], 0, 0
        for t, _, g in rows:
            if g < last:
                base += last
            last = g
            out.append((t, 0, base + g))
        return out

    # The line loop only wakes on output, so the environment check, the timeout and the stop after
    # the last epoch live here. After that epoch the JAX trainer is stopped at its state-save line
    # (below) and the verified driver, under LEAN_MLIR_MAX_EPOCHS=N, exits by itself; `grace` is the
    # backstop for both.
    grace = 60 if a.path == "jax" else 180

    def watchdog():
        while p.poll() is None:
            if "env" not in why and time.monotonic() - t0 > 2:
                bad = check_env()
                if bad is not None:
                    why["env"] = bad
                    if bad:
                        why["status"] = "FAIL:env " + ",".join(bad)
                        kill(f"environment mismatch {bad}")
                        return
            if time.monotonic() > deadline:
                why["status"] = f"FAIL:timeout {a.timeout_min:g} min"
                kill(why["status"])
                return
            if done_at and time.monotonic() - done_at > grace:
                kill(f"{a.epochs} epoch(s) measured; stopped {grace} s after the last")
                return
            time.sleep(1)
    threading.Thread(target=watchdog, daemon=True).start()

    for line in p.stdout:
        t = time.monotonic() - t0
        line = line.rstrip("\n")
        log.write(f"{t:.3f}\t{line}\n")
        cur = len(ep_end) + 1
        if a.path == "jax":
            if (m := JAX_SPE.search(line)) and spe is None:
                spe = int(m[1])
            if m := JAX_STEP.match(line):
                g = int(m[1])
                steps.append((t, cur, g - (cur - 1) * spe if spe else g))
            elif m := JAX_EPOCH.match(line):
                k = int(m[1])
                ep_end[k], top1[k] = t, f"{100 * float(m[2]):.3f}" if m[2] else ""
                jax_train[k], jax_val[k] = float(m[3]), float(m[4] or 0)
        else:
            if m := VER_STEP.match(line):
                steps.append((t, cur, int(m[1])))
            elif m := VER_TRAIN_END.match(line):
                train_end[int(m[1])] = t
            elif m := VER_EPOCH.match(line):
                ep_end[int(m[1])], top1[int(m[1])] = t, f"{float(m[2]):.3f}"
        if a.images:
            ips = a.images_per_step or (IMAGENET_TRAIN // spe if spe else 0)
            warm = [(tt, g) for tt, _, g in cum(steps) if ips and g * ips >= a.images // 4]
            if ips and steps and cum(steps)[-1][2] * ips >= a.images and len(warm) >= 3:
                budget.update(ips=ips, window=warm)
                kill(f"{a.images} images measured")
                break
            continue
        if len(ep_end) >= a.epochs and done_at is None:
            done_at = time.monotonic()
        # the JAX trainer: stop once the epoch's state write has finished (a kill mid-write leaves a
        # .tmp behind); it would otherwise train all of its EPOCHS
        if done_at and a.path == "jax" and "saved full train state" in line:
            kill(f"{a.epochs} epoch(s) measured")
            break
    try:
        p.wait(30)
    except subprocess.TimeoutExpired:
        kill("cleanup")
    if "status" in why:
        status = why["status"]
    elif a.images and not budget:
        status = f"FAIL:exit {p.returncode} before {a.images} images"
    elif not a.images and len(ep_end) < a.epochs:
        status = f"FAIL:exit {p.returncode} after {len(ep_end)} epoch(s)"
    elif "env" not in why:
        status = "FAIL:env unread"   # the process was gone before its environment could be read
    tel.stop.set()
    tel.join(10)

    if budget:
        w = budget["window"]
        ms = 1000 * (w[-1][0] - w[0][0]) / (w[-1][1] - w[0][1])
        new = not os.path.exists(a.tsv)
        util, mem, cpu = tel.window(w[0][0], w[-1][0])
        vals = dict(epoch=0, wall_s=f"{w[-1][0]:.1f}", train_s=f"{w[-1][0] - w[0][0]:.1f}", val_s="",
                    step_ms=f"{ms:.1f}", img_s=f"{budget['ips'] * 1000 / ms:.0f}", top1="")
        row = dict(info, date_utc=time.strftime("%F %T", time.gmtime()), job=a.job, path=a.path,
                   net=a.net, gpu_util=util, gpu_mem_mib=mem, cpu_busy=cpu, status=status,
                   row_dir=a.row_dir, **vals)
        with open(a.tsv, "a") as f:
            if new:
                f.write("\t".join(COLS) + "\n")
            f.write("\t".join(str(row[c]) for c in COLS) + "\n")
        print(f"[bench] {a.job}: {status}; {budget['ips'] * 1000 / ms:.0f} img/s -> {a.tsv}")
        return 0 if status == "ok" else 1

    new = not os.path.exists(a.tsv)
    with open(a.tsv, "a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        rows = [k for k in sorted(ep_end) if k <= a.epochs] or [0]
        for k in rows:
            if k == 0:
                vals = dict(epoch="", wall_s="", train_s="", val_s="", step_ms="", img_s="", top1="")
                util = mem = cpu = ""
            else:
                start = ep_end.get(k - 1, 0.0)
                wall = ep_end[k] - start
                if a.path == "jax":
                    tr, va = jax_train[k], jax_val[k]
                else:
                    tr = train_end.get(k, ep_end[k]) - start
                    va = ep_end[k] - train_end.get(k, ep_end[k])
                st = [(t, s) for t, e, s in steps if e == k and s >= WARM_STEP]
                ms = (f"{1000 * (st[-1][0] - st[0][0]) / (st[-1][1] - st[0][1]):.1f}"
                      if len(st) >= 2 and st[-1][1] > st[0][1] else "")
                vals = dict(epoch=k, wall_s=f"{wall:.1f}", train_s=f"{tr:.1f}", val_s=f"{va:.1f}",
                            step_ms=ms, img_s=f"{IMAGENET_TRAIN / tr:.0f}" if tr > 0 else "",
                            top1=top1.get(k, ""))
                util, mem, cpu = tel.window(start, ep_end[k])
            row = dict(info, date_utc=time.strftime("%F %T", time.gmtime()), job=a.job, path=a.path,
                       net=a.net, gpu_util=util, gpu_mem_mib=mem, cpu_busy=cpu, status=status,
                       row_dir=a.row_dir, **vals)
            f.write("\t".join(str(row[c]) for c in COLS) + "\n")
    print(f"[bench] {a.job}: {status}; {len(ep_end)} epoch(s) -> {a.tsv}")
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
