#!/usr/bin/env python3
"""Byte-level BPE tokenizer + binary packer for TinyStories.

Trains a small BPE vocab on a slice of the corpus, then encodes a
(subset of the) corpus to the same flat int32-LE token-stream format
the Shakespeare loader already reads — so the Lean side reuses
`F32.loadTokenStream` / `F32.sampleChunks` unchanged
(planning/tinygpt_demo_v2.md Part II, Workstream D).

Produces in data/tinystories/:
  vocab.json / merges.txt   — the trained BPE tokenizer (HF format)
  tokenizer.json            — full serialized tokenizer (for re-encode)
  train.bin / val.bin       — flat int32 LE token IDs
  meta.txt                  — vocab_size + train/val token counts + eot id

Deliberately small vocab (default 4096): keeps the embedding/head
matmuls cheap for the in-graph one-hot path and keeps the tokenizer
auditable. `<|endoftext|>` is a real token (id printed to meta) so the
sampler can stop on it.

Usage: python3 preprocess_tinystories.py [vocab=4096] [train_chars=200_000_000]
"""
import os, sys, struct, json

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    import numpy as np
except ImportError:
    sys.exit("ERROR: `tokenizers` + `numpy` required (pip install tokenizers numpy)")

DATA = "data/tinystories"
TRAIN_TXT = f"{DATA}/TinyStories-train.txt"
VALID_TXT = f"{DATA}/TinyStories-valid.txt"
EOT = "<|endoftext|>"

VOCAB = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
# Cap on characters encoded into train.bin (the corpus is ~1.7GB; a
# few hundred M tokens is plenty for a 10M-param model — see the plan).
TRAIN_CHARS = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000_000

for p in (TRAIN_TXT, VALID_TXT):
    if not os.path.exists(p):
        sys.exit(f"missing {p} — run download_tinystories.sh first")


def bpe_train_iter(path, char_budget, chunk=1 << 20):
    """Yield text chunks from `path` up to `char_budget` characters."""
    seen = 0
    with open(path, "r", encoding="utf-8") as f:
        while seen < char_budget:
            block = f.read(chunk)
            if not block:
                break
            seen += len(block)
            yield block


# ── Train the BPE tokenizer on a 50M-char slice (enough to learn merges) ──
print(f"training byte-level BPE, vocab={VOCAB} ...")
tok = Tokenizer(models.BPE(unk_token=None))
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB,
    special_tokens=[EOT],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)
tok.train_from_iterator(bpe_train_iter(TRAIN_TXT, 50_000_000), trainer=trainer)
tok.save(f"{DATA}/tokenizer.json")
tok.model.save(DATA)  # vocab.json + merges.txt
eot_id = tok.token_to_id(EOT)
real_vocab = tok.get_vocab_size()
print(f"  trained: vocab_size={real_vocab}, eot id={eot_id}")


def encode_file(path, char_budget):
    """Stream-encode `path` (up to char_budget chars) to a flat int32
    numpy array. Splits on EOT so each story is encoded independently
    and the EOT id is inserted between stories."""
    ids = []
    for block in bpe_train_iter(path, char_budget):
        # Encode the raw block; EOT markers in-text become the special id
        # via the trained special token, so no manual splitting needed.
        enc = tok.encode(block)
        ids.extend(enc.ids)
    return np.asarray(ids, dtype=np.int32)


print(f"encoding train (≤{TRAIN_CHARS:,} chars) ...")
train_ids = encode_file(TRAIN_TXT, TRAIN_CHARS)
print(f"  train tokens: {len(train_ids):,}")
print("encoding val (full valid split) ...")
val_ids = encode_file(VALID_TXT, 1 << 62)
print(f"  val tokens: {len(val_ids):,}")

train_ids.tofile(f"{DATA}/train.bin")
val_ids.tofile(f"{DATA}/val.bin")

with open(f"{DATA}/meta.txt", "w") as f:
    f.write(f"vocab_size {real_vocab}\n")
    f.write(f"train_tokens {len(train_ids)}\n")
    f.write(f"val_tokens {len(val_ids)}\n")
    f.write(f"eot_id {eot_id}\n")

print("wrote train.bin val.bin tokenizer.json vocab.json merges.txt meta.txt")
print(f"  sample decode of first 40 train ids:\n  {tok.decode(train_ids[:40].tolist())!r}")
