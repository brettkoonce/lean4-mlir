#!/usr/bin/env python3
"""Generate sitemap.xml (+ robots.txt) for the GitHub Pages site.

Run against the assembled `public/` tree, not the repo: the two big subtrees
(`blueprint/`, `docs/`) are BUILD PRODUCTS, so a committed static sitemap would be
stale the moment leanblueprint or doc-gen4 emits a different set of pages.

    python3 scripts/gen_sitemap.py public https://brettkoonce.github.io/lean4-mlir

⭐ THE ONE DECISION THAT MATTERS: `public/docs/` is a doc-gen4 tree that contains a
LOCALLY REBUILT MATHLIB (doc-gen4 cannot yet reference hosted Mathlib docs — see the
blueprint workflow's doc-gen4 step). That is tens of thousands of pages of somebody
else's library. Listing it would

  * blow the sitemap spec's 50,000-URL / 50 MB cap, and
  * ask Google to index a duplicate of Mathlib's own docs under this domain,
    which is precisely the duplicate-content pattern that earns a ranking penalty.

So `docs/` is included ONLY under the project's own namespaces (DOC_KEEP below).

Excluded outright: build-logs/ (CI diagnostics, deliberately published for curl-ing
but not for indexing), 404.html, and the Search Console verification token.
"""
from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path
from xml.sax.saxutils import escape

# doc-gen4 subtrees that are OURS. Everything else under docs/ is a vendored library.
DOC_KEEP = ("LeanMlir",)

# Directories never worth indexing.
SKIP_DIRS = {"build-logs"}

# Files never worth indexing (404 is not content; the token must stay reachable but
# unindexed or it can outrank real pages for the domain).
SKIP_FILES = {"404.html"}

MAX_URLS = 50_000        # sitemap protocol hard cap
WARN_URLS = 45_000       # leave headroom before the cap bites

INDEXABLE = {".html", ".pdf"}


def priority_for(rel: str) -> tuple[str, str]:
    """(priority, changefreq) — a crawl hint, not a ranking lever."""
    if rel == "":                       # site root
        return "1.0", "weekly"
    if rel == "blueprint.pdf":
        return "0.9", "weekly"
    if rel.startswith("blueprint/"):
        return "0.8", "weekly"
    if rel.startswith("docs/"):
        return "0.4", "monthly"
    return "0.6", "monthly"


def collect(root: Path) -> list[str]:
    urls: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in INDEXABLE:
            continue
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        if parts[0] in SKIP_DIRS:
            continue
        if p.name in SKIP_FILES or p.name.startswith("google"):
            continue
        # docs/: keep only our own namespaces (see module docstring)
        if parts[0] == "docs":
            if len(parts) < 2 or not any(parts[1] == k or parts[1].startswith(k + ".")
                                         for k in DOC_KEEP):
                continue
        # index.html is served AS its directory, and the canonical form keeps the
        # trailing slash — `/blueprint` 301s to `/blueprint/` on GitHub Pages, so
        # emitting the bare form spends a redirect on every crawl.
        if p.name == "index.html":
            rel = rel[: -len("index.html")]          # "blueprint/", or "" at the root
        urls.append(rel)
    # dedupe (a dir with index.html can collide with nothing else, but be safe)
    return sorted(set(urls))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    root, base = Path(sys.argv[1]), sys.argv[2].rstrip("/")
    if not root.is_dir():
        print(f"::warning::{root} is not a directory; no sitemap written")
        return 0

    urls = collect(root)
    if len(urls) > WARN_URLS:
        print(f"::warning::sitemap has {len(urls)} URLs (cap {MAX_URLS}); "
              "consider splitting into a sitemap index")
    urls = urls[:MAX_URLS]

    today = _dt.date.today().isoformat()
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for rel in urls:
        loc = f"{base}/{rel}" if rel else f"{base}/"
        pri, freq = priority_for(rel)
        out += ["  <url>",
                f"    <loc>{escape(loc)}</loc>",
                f"    <lastmod>{today}</lastmod>",
                f"    <changefreq>{freq}</changefreq>",
                f"    <priority>{pri}</priority>",
                "  </url>"]
    out.append("</urlset>")
    (root / "sitemap.xml").write_text("\n".join(out) + "\n", encoding="utf-8")

    # robots.txt is how a crawler DISCOVERS the sitemap without Search Console.
    # Also keeps the CI diagnostics out of the index.
    (root / "robots.txt").write_text(
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /build-logs/\n"
        f"\nSitemap: {base}/sitemap.xml\n",
        encoding="utf-8")

    n_bp = sum(1 for u in urls if u.startswith("blueprint"))
    n_doc = sum(1 for u in urls if u.startswith("docs/"))
    print(f"sitemap.xml: {len(urls)} URLs "
          f"({n_bp} blueprint, {n_doc} docs, {len(urls)-n_bp-n_doc} other) -> {base}/")
    print("robots.txt: written (build-logs/ disallowed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
