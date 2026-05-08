#!/usr/bin/env python3
"""Char-level tokenizer + binary packer for tinyshakespeare.

Reads `data/shakespeare/tinyshakespeare.txt`, produces:
  data/shakespeare/vocab.txt   — one byte per line (raw byte, decimal repr)
  data/shakespeare/train.bin   — flat int32 LE token IDs (90% of corpus)
  data/shakespeare/val.bin     — flat int32 LE token IDs (10% of corpus)
  data/shakespeare/meta.txt    — vocab_size + train_tokens + val_tokens summary

We treat the input as raw bytes (not str) to keep the tokenizer trivially
lossless and round-trippable — no Unicode normalization, no encoding edge
cases. Shakespeare's full vocab on this file is ~65 distinct bytes.
"""
import os
import struct
import sys

src = "data/shakespeare/tinyshakespeare.txt"
out_dir = "data/shakespeare"
if not os.path.exists(src):
    sys.exit(f"missing {src} — run download_shakespeare.sh first")

with open(src, "rb") as f:
    raw = f.read()

print(f"corpus: {len(raw):,} bytes")
vocab = sorted(set(raw))
print(f"unique bytes (vocab size): {len(vocab)}")

byte_to_id = {b: i for i, b in enumerate(vocab)}
ids = [byte_to_id[b] for b in raw]

# 90/10 split — the standard char-rnn split. Take the val tail so a
# held-out test always covers the same passage of the play (deterministic
# perplexity tracking across runs).
split = int(0.9 * len(ids))
train_ids = ids[:split]
val_ids = ids[split:]
print(f"train tokens: {len(train_ids):,}   val tokens: {len(val_ids):,}")

def write_int32_le(path, ids):
    with open(path, "wb") as f:
        for i in ids:
            f.write(struct.pack("<i", i))

write_int32_le(f"{out_dir}/train.bin", train_ids)
write_int32_le(f"{out_dir}/val.bin", val_ids)

with open(f"{out_dir}/vocab.txt", "w") as f:
    for i, b in enumerate(vocab):
        # Print id, decimal byte, and a printable form.
        ch = chr(b) if 32 <= b < 127 else f"\\x{b:02x}"
        f.write(f"{i}\t{b}\t{ch!r}\n")

with open(f"{out_dir}/meta.txt", "w") as f:
    f.write(f"vocab_size {len(vocab)}\n")
    f.write(f"train_tokens {len(train_ids)}\n")
    f.write(f"val_tokens {len(val_ids)}\n")

print("wrote train.bin val.bin vocab.txt meta.txt")
