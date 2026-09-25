#!/usr/bin/env python3
"""Generate sitemap.xml (+ robots.txt) for the GitHub Pages site.

Run against the assembled `public/` tree, not the repo: the two big subtrees
(`blueprint/`, `docs/`) are BUILD PRODUCTS, so a committed static sitemap would be
stale the moment leanblueprint or doc-gen4 emits a different set of pages.

    python3 scripts/book/gen_sitemap.py public https://lean.brettkoonce.com

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
    #
    # `docs/declarations/` is doc-gen4's search index: declaration-data.bmp is a
    # 34 MB JSON blob (named .bmp so the browser will `<link rel=prefetch as=image>`
    # it) that every one of the tens of thousands of doc pages references in its
    # <head>. Crawlers that follow that link pull ~3 MB compressed apiece -- 48 GB
    # and ~16,700 requests in one overnight window, 98% of it served from
    # Cloudflare's cache, i.e. pure crawler traffic for a file no human reads. It
    # is not a page and has no business being indexed; the search box fetches it
    # from the browser regardless of what this says.
    # Everything under docs/ except our own namespaces is a vendored library (a
    # locally rebuilt Mathlib and its deps — see the module docstring). The sitemap
    # already omits them, but that only means we do not ASK for them: doc-gen4 links
    # every type in every signature to its declaration page, so a crawler walking
    # docs/LeanMlir/ falls straight into docs/Mathlib/ and mirrors somebody else's
    # library on this domain. Invert it — block docs/ wholesale, re-Allow what is
    # ours. Google and Bing resolve conflicts by LONGEST match, so
    # `Allow: /docs/LeanMlir/` (18 chars) beats `Disallow: /docs/` (6).
    #
    # NB this blocks CRAWLING, not indexing. A URL already in the index stays there
    # (URL-only, no snippet) until re-crawled, which this now prevents. To actively
    # REMOVE pages you must serve `<meta name="robots" content="noindex">` and let
    # them be crawled first, then disallow. Nothing under docs/ is externally linked
    # enough for that to be worth the two-step today.
    # ORDER AND THE MISSING `Allow: /` ARE BOTH DELIBERATE.
    #
    # Two parser families disagree about conflicts. RFC 9309 (and Google, Bing)
    # take the MOST SPECIFIC match; older/simpler parsers — urllib.robotparser
    # among them, and plenty of bots — take the FIRST match in file order. A
    # leading `Allow: /` is a no-op under the first (a path with no Disallow is
    # allowed anyway) and catastrophic under the second: it matches everything
    # first, so every Disallow below it is dead. The file carried one until
    # 2026-09-14, which silently neutered the docs/declarations/ rule for exactly
    # the unsophisticated crawlers it was written for.
    #
    # So: no blanket Allow, and Allow lines BEFORE the Disallow they carve out of.
    # Longest-match parsers pick `/docs/LeanMlir/` (18 chars) over `/docs/` (6);
    # first-match parsers hit the Allow first. Both land in the same place.
    doc_allows = "".join(f"Allow: /docs/{k}/\n" for k in DOC_KEEP)
    (root / "robots.txt").write_text(
        "User-agent: *\n"
        "Disallow: /build-logs/\n"
        + doc_allows
        # doc-gen4's shared CSS/JS sit at the docs/ root; without these our OWN
        # pages report as blocked-resource in Search Console. Wildcards are an
        # RFC 9309 extension — parsers that lack them fall through to
        # `Disallow: /docs/`, which costs styling on a page we do not rank on.
        + "Allow: /docs/*.css\n"
        + "Allow: /docs/*.js\n"
        # Ahead of the blanket docs/ rule so first-match parsers still attribute
        # the block to this line. It is the 48 GB overnight; keep it visible.
        + "Disallow: /docs/declarations/\n"
        # Everything else under docs/ is a locally rebuilt Mathlib and its deps —
        # see the module docstring. The sitemap already omits them, but that only
        # means we do not ASK: doc-gen4 links every type in every signature to its
        # declaration page, so a crawler walking docs/LeanMlir/ falls straight
        # into docs/Mathlib/ and mirrors somebody else's library on this domain.
        #
        # NB this blocks CRAWLING, not indexing. A URL already indexed stays
        # (URL-only, no snippet) until re-crawled, which this now prevents. To
        # actively REMOVE such pages, serve `<meta name="robots" content=
        # "noindex">` and let them be crawled first, THEN disallow.
        + "Disallow: /docs/\n"
        + f"\nSitemap: {base}/sitemap.xml\n",
        encoding="utf-8")

    n_bp = sum(1 for u in urls if u.startswith("blueprint"))
    n_doc = sum(1 for u in urls if u.startswith("docs/"))
    print(f"sitemap.xml: {len(urls)} URLs "
          f"({n_bp} blueprint, {n_doc} docs, {len(urls)-n_bp-n_doc} other) -> {base}/")
    print(f"robots.txt: written (build-logs/ + docs/ disallowed; "
          f"allowed: {', '.join(DOC_KEEP)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
