#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # the repo root: data/ lives there
mkdir -p data/shakespeare
cd data/shakespeare

if [ ! -f "tinyshakespeare.txt" ]; then
  echo "Downloading tinyshakespeare.txt (Karpathy / char-rnn corpus)..."
  curl -L -o tinyshakespeare.txt \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
fi

# Sanity report
bytes=$(wc -c < tinyshakespeare.txt)
chars=$(wc -m < tinyshakespeare.txt)
echo "tinyshakespeare.txt: ${bytes} bytes, ${chars} chars"
