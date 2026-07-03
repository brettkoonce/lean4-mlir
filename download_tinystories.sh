#!/bin/bash
set -e
mkdir -p data/tinystories
cd data/tinystories

# TinyStories (Eldan & Li 2023) — ~2.1M GPT-3.5/4-generated children's
# stories. train ~1.9GB, valid ~19MB. See planning/tinygpt_demo_v2.md
# Part II.
BASE=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main

if [ ! -f "TinyStories-valid.txt" ]; then
  echo "Downloading TinyStories-valid.txt (~19MB)..."
  curl -L -o TinyStories-valid.txt "$BASE/TinyStories-valid.txt"
fi

if [ ! -f "TinyStories-train.txt" ]; then
  echo "Downloading TinyStories-train.txt (~1.9GB)..."
  curl -L -o TinyStories-train.txt "$BASE/TinyStories-train.txt"
fi

echo "TinyStories-valid.txt: $(wc -c < TinyStories-valid.txt) bytes"
echo "TinyStories-train.txt: $(wc -c < TinyStories-train.txt) bytes"
