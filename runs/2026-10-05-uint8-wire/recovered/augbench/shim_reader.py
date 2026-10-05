# Read a shim's stdout (wire v2), skip W batches, time the next K; print img/s.
import sys, struct, time
W, K = int(sys.argv[1]), int(sys.argv[2])
f = sys.stdin.buffer
pre = f.read(16); magic, ver, batch, flat = struct.unpack('<4siii', pre)
ncls = struct.unpack('<i', f.read(4))[0] if ver >= 4 else 0
def rd(n):
    b = bytearray(n); mv = memoryview(b); got = 0
    while got < n:
        k = f.readinto(mv[got:])
        if not k: raise SystemExit("eof")
        got += k
    return b
t0 = None; imgs = 0
for i in range(W + K):
    rows = struct.unpack('<i', rd(4))[0]
    rd(rows * (ncls * 4 if ncls else 4)); rd(rows * flat * 4)
    if i == W - 1: t0 = time.perf_counter()
    elif i >= W: imgs += rows
print(f"{imgs / (time.perf_counter() - t0):.0f}", flush=True)
