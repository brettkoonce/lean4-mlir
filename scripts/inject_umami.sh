#!/bin/bash
# Inject the Umami analytics tag into every .html file under <dir>, right
# before </head>. Idempotent: skips files that already contain the tag.
#
# Used in CI (.github/workflows/blueprint.yml) to instrument the doc-gen4
# output AFTER it's been copied into public/docs/, so we don't have to
# patch the vendored doc-gen4 template. Same Umami site as the blueprint
# (cf. blueprint/src/umami_analytics.js).
#
# Usage:  scripts/inject_umami.sh public/docs
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "usage: $0 <html-tree-dir>" >&2
  exit 1
fi

DIR="$1"
WEBSITE_ID="df07a183-ef82-4758-8b6f-1d888ba3ec5b"
SNIPPET="        <script defer src=\"https://cloud.umami.is/script.js\" data-website-id=\"$WEBSITE_ID\"></script>"

if [ ! -d "$DIR" ]; then
  echo "ERROR: $DIR does not exist" >&2
  exit 1
fi

total=0
injected=0
skipped=0
while IFS= read -r -d '' f; do
  total=$((total + 1))
  if grep -q "cloud.umami.is" "$f"; then
    skipped=$((skipped + 1))
    continue
  fi
  # GNU sed: insert SNIPPET on the line BEFORE the first </head>.
  if grep -q "</head>" "$f"; then
    sed -i "0,|</head>|s||${SNIPPET}\n</head>|" "$f" 2>/dev/null || {
      # Fall back to a simpler replace-first-occurrence pattern if the
      # 0,|...| syntax above isn't supported.
      python3 - "$f" "$SNIPPET" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
snippet = sys.argv[2]
text = path.read_text(encoding="utf-8", errors="replace")
idx = text.find("</head>")
if idx >= 0:
    path.write_text(text[:idx] + snippet + "\n" + text[idx:], encoding="utf-8")
PY
    }
    injected=$((injected + 1))
  else
    skipped=$((skipped + 1))
  fi
done < <(find "$DIR" -name "*.html" -print0)

echo "umami inject: $injected/$total files instrumented ($skipped skipped — already-had-snippet or no </head>)"
