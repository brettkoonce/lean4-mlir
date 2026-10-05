# Copy bandwidth: k processes each copying a pre-faulted 1 GiB NumPy array 10x (09-14's E22 shape).
import numpy as np, sys, time, multiprocessing as mp
def w(q):
    a = np.ones(1 << 27); b = np.empty_like(a); b[:] = a
    t = time.time()
    for _ in range(10): b[:] = a
    q.put(10 * a.nbytes * 2 / (time.time() - t) / 1e9)   # read + write bytes
if __name__ == '__main__':
    for k in (1, 4, 8):
        q = mp.Queue(); ps = [mp.Process(target=w, args=(q,)) for _ in range(k)]
        [p.start() for p in ps]; r = [q.get() for _ in ps]; [p.join() for p in ps]
        print(f"{k} stream(s): {sum(r):.1f} GB/s total (read+write), {sum(r)/k:.1f} each")
