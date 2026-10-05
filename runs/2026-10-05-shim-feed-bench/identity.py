#!/usr/bin/env python3
"""Old vs new shim: SHIM_HASH digests and the streamed bytes of the first N records must match.

Run with SHIM_DETERMINISM=1 (else tf.data's order differs run to run). Odd batch sizes exercise
`_mix_rows`'s self-paired middle row; N >= 2 covers both mixup (even steps) and cutmix (odd).
"""
import hashlib, os, subprocess, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PY = os.path.join(ROOT, '.venv/bin/python3')
OLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'old')
NEW = os.path.join(ROOT, 'jax/.lake/build')


def env(batch, nc, extra):
    e = dict(os.environ, SHIM_BATCH=str(batch), SHIM_SPLIT=extra.get('split', 'train'), SHIM_SEED='3',
             SHIM_DETERMINISM='1', CUDA_VISIBLE_DEVICES='', TF_CPP_MIN_LOG_LEVEL='3',
             TFDS_DATA_DIR='/home/skoonce/tensorflow_datasets')
    if nc: e['SHIM_NCLASSES'] = str(nc)
    else: e['SHIM_MIX'] = 'off'
    return e


def digest(script, batch, nc, n, **extra):
    e = env(batch, nc, extra); e['SHIM_HASH'] = str(n)
    r = subprocess.run([PY, script], env=e, capture_output=True, text=True)
    return [l for l in r.stderr.splitlines() if l.startswith('SHIM_HASH')][-1].split(': ')[-1]


def stream(script, batch, nc, n, **extra):
    p = subprocess.Popen([PY, script], env=env(batch, nc, extra), stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    f = p.stdout; h = hashlib.sha256()
    pre = f.read(16); h.update(pre)
    flat = int.from_bytes(pre[12:16], 'little')
    if nc: h.update(f.read(4))
    for _ in range(n):
        rb = f.read(4); h.update(rb); rows = int.from_bytes(rb, 'little')
        h.update(f.read(rows * (nc * 4 if nc else 4)))
        h.update(f.read(rows * flat * 4))
    p.kill(); return h.hexdigest()


cases = [  # shim, batch, nclasses, batches
    ('generated_vit_tiny_imagenet_shim.py', 63, 1000, 4),
    ('generated_vit_tiny_imagenet_shim.py', 64, 1000, 4),
    ('generated_convnext_s_imagenet_shim.py', 32, 1000, 4),
    ('generated_resnet50_imagenet_a2accum_shim.py', 33, 1000, 4),
    ('generated_mobilenet_v4_imagenet_full_shim.py', 32, 0, 3),
]
bad = 0
for s, b, nc, n in cases:
    for kind, fn in (('hash', digest), ('stream', stream)):
        o = fn(os.path.join(OLD, s), b, nc, n); w = fn(os.path.join(NEW, s), b, nc, n)
        ok = o == w; bad += not ok
        print(f"{'✓' if ok else '✗'} {kind:6} {s} B={b} nc={nc} n={n}: {o[:16]} {'==' if ok else '!='} {w[:16]}", flush=True)
print('✓ identical' if not bad else f'✗ {bad} mismatches'); sys.exit(1 if bad else 0)
