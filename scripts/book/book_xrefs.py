#!/usr/bin/env python3
"""Every cross-reference in the book, grouped by what it points at.

    python3 scripts/book/book_xrefs.py            # one block per target, its citing sentences under it
    python3 scripts/book/book_xrefs.py --summary  # one line per target: how many sentences point at it

A cross-reference may NAME where something lives; it must not ASSERT what the target says
(a number, a result, "the pair that isolates it"), because the target gets rewritten and the
sentence in another chapter does not. planning/book_xrefs.md is the pass that removes those.
This script is the census that pass works from: it lists each referring sentence next to the
target's heading so the two can be read side by side. Theorem refs are listed only when they
cross a chapter boundary; same-chapter proof steps are stable by label.
"""
import bisect, collections, re, sys

TEX = 'blueprint/src/content.tex'
REF = re.compile(r'(\\S\\ref|Chapters?~\\ref|Section~\\ref|Appendix~\\ref|Figure~\\ref|Theorem~\\ref|\\bestiaryref)\{([^}]*)\}')
HEAD = re.compile(r'\\(chapter\*?|section|subsection|prosesection|prosesubsection)\{(.*)\}')
ENV = re.compile(r'\\begin\{(theorem|lemma|definition|axiom|corollary|proposition)\}(\[.*?\])?')
ABBR = re.compile(r'(e\.g|i\.e|vs|cf|Fig|Ch|Sec|No)\.$')


def main():
    lines = open(TEX).read().split('\n')
    text = '\n'.join(lines)
    starts = [0]
    for l in lines:
        starts.append(starts[-1] + len(l) + 1)
    line_of = lambda off: bisect.bisect_right(starts, off)

    chapters = [(i, m.group(2)) for i, l in enumerate(lines, 1) for m in [HEAD.match(l)] if m and m.group(1).startswith('chapter')]
    def chapter_of(i):
        t = '(front)'
        for ln, ti in chapters:
            if ln <= i: t = ti
            else: break
        return t
    labels = {m.group(1): i for i, l in enumerate(lines, 1) for m in re.finditer(r'\\label\{([^}]*)\}', l)}
    def heading_at(i):
        for j in range(i, max(0, i - 6), -1):
            m = HEAD.match(lines[j - 1])
            if m: return m.group(1) + ': ' + m.group(2)
        for j in range(i, max(0, i - 8), -1):
            m = ENV.match(lines[j - 1])
            if m: return m.group(1) + ' ' + (m.group(2) or '')
        return lines[i - 1][:80]

    def sentence(off):
        a = off
        while a > 0:
            if text[a - 1:a + 1] == '\n\n': break
            if text[a - 1] in '.!?' and text[a] in ' \n' and not text[max(0, a - 3):a].endswith('\\S') and not ABBR.search(text[max(0, a - 5):a]): break
            a -= 1
        b = off
        while b < len(text):
            if text[b:b + 2] == '\n\n': break
            if text[b] in '.!?' and (b + 1 == len(text) or text[b + 1] in ' \n') and not ABBR.search(text[max(0, b - 5):b + 1]):
                b += 1; break
            b += 1
        return ' '.join(text[a:b].split())

    by = collections.defaultdict(list)
    for m in REF.finditer(text):
        kind, lab = m.group(1), m.group(2)
        src, tgt = line_of(m.start()), labels.get(lab)
        if kind.startswith('Theorem') and tgt and chapter_of(src) == chapter_of(tgt): continue
        by[lab].append((src, chapter_of(src), sentence(m.start())))

    ordered = sorted(by.items(), key=lambda kv: labels.get(kv[0], 0))
    if '--summary' in sys.argv:
        for lab, rs in ordered:
            print('%3d  %-34s L%-6s %s' % (len(rs), lab, labels.get(lab, '?'), heading_at(labels[lab])[:70] if lab in labels else 'UNRESOLVED'))
        print('%d refs, %d targets' % (sum(len(rs) for _, rs in ordered), len(ordered)))
        return
    for lab, rs in ordered:
        tgt = labels.get(lab)
        print('\n=== %s  (content.tex:%s  %s | chapter: %s)  refs=%d' % (lab, tgt, heading_at(tgt) if tgt else 'UNRESOLVED', chapter_of(tgt) if tgt else '?', len(rs)))
        for src, ch, sent in rs:
            print('  [%d | %s] %s' % (src, ch, sent))


if __name__ == '__main__':
    main()
