#!/usr/bin/env python3
"""Shim producer throughput at the job's own shape — CPU only, no trainer, no GPU.

Spawns N copies of a generated ImageNet shim exactly as `spawnShim` does (same env: SHIM_BATCH,
SHIM_SPLIT=train, SHIM_SEED=seed+i, SHIM_SHARD=i/N, SHIM_NCLASSES when soft), reads the wire
(preamble, then per batch: int32 rows, targets, float32 pixels) and reports images/s.

  --mode rr    read batch k from producer k % N, one at a time — the trainer's `readShimBatchRR`
               order, so one slow producer paces all of them (what a job actually sees)
  --mode free  one reader thread per producer, each draining as fast as it can — the producers'
               aggregate capacity with the round-robin coupling removed

The trainer additionally prefetches one batch per handle; `rr` without prefetch is the pessimistic
bound and `free` the optimistic one. Usage:

  python3 bench.py --shim generated_vit_tiny_imagenet_shim.py --batch 512 --n 4 --nclasses 1000
"""
import argparse, os, subprocess, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument('--shim', required=True)
ap.add_argument('--batch', type=int, default=512)
ap.add_argument('--n', type=int, default=4)
ap.add_argument('--nclasses', type=int, default=0)
ap.add_argument('--mix', default=None, help='SHIM_MIX override (off|mixup|cutmix|both)')
ap.add_argument('--mode', default='rr', choices=['rr', 'free'])
ap.add_argument('--warm', type=float, default=45.0, help='seconds before the clock starts')
ap.add_argument('--secs', type=float, default=90.0, help='seconds on the clock')
ap.add_argument('--label', default='')
ap.add_argument('--env', action='append', default=[], help='extra KEY=VAL for the producers')
a = ap.parse_args()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
script = os.path.join(ROOT, 'jax/.lake/build', a.shim)
py = os.path.join(ROOT, '.venv/bin/python3')
procs = []
for i in range(a.n):
    env = dict(os.environ, SHIM_BATCH=str(a.batch), SHIM_SPLIT='train', SHIM_SEED=str(1 + i),
               TFDS_DATA_DIR=os.environ.get('TFDS_DATA_DIR', '/home/skoonce/tensorflow_datasets'),
               CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='2')
    if a.n > 1:
        env['SHIM_SHARD'] = f'{i}/{a.n}'
    if a.nclasses:
        env['SHIM_NCLASSES'] = str(a.nclasses)
    else:
        env['SHIM_MIX'] = 'off'
    if a.mix is not None:
        env['SHIM_MIX'] = a.mix
    for kv in a.env:
        k, v = kv.split('=', 1); env[k] = v
    procs.append(subprocess.Popen([py, script], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  env=env, bufsize=0))


def read_exact(f, buf):
    mv = memoryview(buf); got = 0
    while got < len(buf):
        n = f.readinto(mv[got:])
        if not n:
            raise EOFError('producer closed the pipe')
        got += n


flat = None
for p in procs:
    pre = bytearray(16); read_exact(p.stdout, pre)
    assert pre[:4] == b'LMSH', pre[:4]
    ver, b, flat = (int.from_bytes(pre[o:o + 4], 'little') for o in (4, 8, 12))
    if ver == 4:
        read_exact(p.stdout, bytearray(4))
    assert b == a.batch
lab = a.batch * a.nclasses * 4 if a.nclasses else a.batch * 4
rec = 4 + lab + a.batch * flat * 4
t0 = time.time()
stamps = []          # (wall, producer) per batch received
lock = threading.Lock()

if a.mode == 'rr':
    buf = bytearray(rec); k = 0
    try:
        while time.time() - t0 < a.warm + a.secs:
            read_exact(procs[k % a.n].stdout, buf)
            stamps.append((time.time(), k % a.n)); k += 1
    except EOFError as e:
        print('EOF', e)
else:
    def drain(i):
        buf = bytearray(rec)
        try:
            while time.time() - t0 < a.warm + a.secs:
                read_exact(procs[i].stdout, buf)
                with lock:
                    stamps.append((time.time(), i))
        except EOFError:
            pass
    ts = [threading.Thread(target=drain, args=(i,)) for i in range(a.n)]
    for t in ts: t.start()
    for t in ts: t.join()

for p in procs:
    p.kill()
w = [(t, i) for t, i in stamps if t - t0 >= a.warm]
span = (w[-1][0] - w[0][0]) if len(w) > 1 else float('nan')
imgs = (len(w) - 1) * a.batch
per = [sum(1 for _, j in w if j == i) for i in range(a.n)]
ips = imgs / span if span == span and span > 0 else 0.0
print(f"{a.label or a.shim} batch={a.batch} n={a.n} mode={a.mode} mix={a.mix or ('default' if a.nclasses else 'off')} "
      f"extra={a.env}: {ips:,.0f} img/s ({len(w)} batches in {span:.1f}s; per producer {per}; "
      f"= {a.batch / ips * 1000 if ips else float('nan'):.0f} ms per {a.batch})", flush=True)
