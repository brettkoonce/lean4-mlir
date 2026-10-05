import time, numpy as np
B, H = 128, 224; F = 3 * H * H
x = np.random.default_rng(0).standard_normal((B, F), dtype=np.float32)
def t(fn, n=10):
    fn(); s = time.perf_counter()
    for _ in range(n): r = fn()
    return (time.perf_counter() - s) / n * 1e3, r
lam = np.float32(0.3)
def mix_now():
    xm = lam * x + (np.float32(1.0) - lam) * np.flip(x, 0)
    return np.ascontiguousarray(xm, dtype=np.float32)
def mix_inplace():
    out = np.multiply(x[::-1], np.float32(1.0) - lam)
    out += lam * x   # one temp
    return out
def mix_inplace2():
    out = np.empty_like(x); np.multiply(x[::-1], np.float32(1.0) - lam, out=out)
    # fused a*x + out via a scratch-free loop over row blocks
    for i in range(0, B, 16):
        out[i:i+16] += lam * x[i:i+16]
    return out
mask = np.zeros((H, H), np.float32); mask[40:180, 30:200] = 1
x4 = x.reshape(B, 3, H, H)
def cut_now():
    return np.ascontiguousarray((x4 * (np.float32(1.0) - mask) + np.flip(x4, 0) * mask).reshape(B, -1), dtype=np.float32)
def cut_copy():
    out = x4.copy(); out[:, :, 40:180, 30:200] = x4[::-1, :, 40:180, 30:200]; return out.reshape(B, -1)
for nm, fn in (("mixup now", mix_now), ("mixup 1 temp", mix_inplace), ("mixup blocked", mix_inplace2),
               ("cutmix now", cut_now), ("cutmix box copy", cut_copy)):
    ms, r = t(fn); print(f"{nm:16s} {ms:6.1f} ms / batch of {B}")
a, b = mix_now(), mix_inplace2(); print("mixup max |Δ|", float(np.abs(a - b).max()))
print("cutmix max |Δ|", float(np.abs(cut_now() - cut_copy()).max()))
