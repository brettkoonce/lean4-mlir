#!/usr/bin/env python3
"""F_GETPIPE_SZ for every pipe fd of a process (via /proc/PID/fd), plus the user's pipe census.
Usage: pipe_sizes.py <pid>   (run while the trainer trains)"""
import fcntl, os, sys, glob
F_GETPIPE_SZ = 1032
pid = sys.argv[1]
rows = []
for fdp in sorted(glob.glob(f"/proc/{pid}/fd/*"), key=lambda p: int(p.rsplit("/", 1)[1])):
    try:
        tgt = os.readlink(fdp)
    except OSError:
        continue
    if not tgt.startswith("pipe:"):
        continue
    try:
        fd = os.open(fdp, os.O_RDONLY | os.O_NONBLOCK)
        sz = fcntl.fcntl(fd, F_GETPIPE_SZ)
        os.close(fd)
    except OSError as e:
        sz = f"err:{e.errno}"
    rows.append((fdp.rsplit("/", 1)[1], tgt, sz))
for fd, tgt, sz in rows:
    print(f"fd {fd:>3}  {tgt:<16} pipe_size={sz}")
pipes = set()
for p in glob.glob("/proc/[0-9]*/fd/*"):
    try:
        t = os.readlink(p)
        if t.startswith("pipe:"): pipes.add(t)
    except OSError:
        pass
print(f"user pipe census: {len(pipes)} unique pipes visible ({len(pipes)*16} pages at 16/pipe; soft limit {open('/proc/sys/fs/pipe-user-pages-soft').read().strip()})")
