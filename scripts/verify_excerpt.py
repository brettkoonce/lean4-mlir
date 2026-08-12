#!/usr/bin/env python3
"""Verify every quoted training-log line in a .tex excerpt against the real log.

chapter_makeover.md §5: writing ch5's "Run it first" two log lines were
FABRICATED while hand-eliding the middle of a run, and were caught only by a
diff like this one. Elision is exactly where invented numbers get in.

Checks every line that looks like training output (`Epoch n/N:`, `  epoch n:`,
`[pjrt_ffi] ...`, `done (...)`) appears verbatim in the source log.

usage: verify_excerpt.py <content.tex> <logfile> <tex-start-line> <tex-end-line>
"""
import re, sys

tex, logfile, lo, hi = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

log = set(l.rstrip() for l in open(logfile))
# Also index the whitespace-collapsed form, since the book wraps long lines.
log_collapsed = set(re.sub(r'\s+', ' ', l).strip() for l in log)

SIGNAL = re.compile(r'^\s*(Epoch \d+/\d+:|epoch \d+:|\[pjrt_ffi\]|done \()')

# The book WRAPS long log lines across several .tex lines (ch6 says so in its
# lead-in). So build logical lines first: a SIGNAL line plus any following
# deeper-indented continuation lines, then compare whitespace-collapsed.
rows = [(i, l.rstrip('\n')) for i, l in enumerate(open(tex), 1)
        if lo <= i <= hi]

logical, n = [], 0
while n < len(rows):
    i, s = rows[n]
    if not SIGNAL.match(s):
        n += 1
        continue
    indent = len(s) - len(s.lstrip())
    buf, n = [s], n + 1
    while n < len(rows):
        j, t = rows[n]
        if (not t.strip() or SIGNAL.match(t) or t.startswith('\\')
                or (len(t) - len(t.lstrip())) <= indent):
            break
        buf.append(t.strip())
        n += 1
    logical.append((i, ' '.join(x.strip() for x in buf)))

bad, checked = [], 0
for i, s in logical:
    checked += 1
    if s in log or re.sub(r'\s+', ' ', s).strip() in log_collapsed:
        continue
    bad.append((i, s))

print(f'checked {checked} quoted log lines against {logfile}')
if bad:
    print(f'\n!!! {len(bad)} NOT FOUND IN THE LOG — do not commit:\n')
    for i, s in bad:
        print(f'  content.tex:{i}  {s.strip()}')
    sys.exit(1)
print('all quoted lines verified present in the source log.')
