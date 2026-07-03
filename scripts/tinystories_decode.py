#!/usr/bin/env python3
"""Encode a prompt to BPE ids / decode generated ids for the TinyStories
demo (the Lean sampler emits raw ids since BPE detok lives here).

  # encode a prompt to data/tinystories/prompt_ids.txt, then sample:
  python3 scripts/tinystories_decode.py encode "Once upon a time, there was a"
  .lake/build/bin/tinystories sample 200 80 40 95 1 > /tmp/gen.txt
  python3 scripts/tinystories_decode.py decode "Once upon a time, there was a" < /tmp/gen.txt
"""
import sys
from tokenizers import Tokenizer

DATA = "data/tinystories"
tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")

if len(sys.argv) < 2:
    sys.exit(__doc__)
mode = sys.argv[1]

if mode == "encode":
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Once upon a time"
    ids = tok.encode(prompt).ids
    with open(f"{DATA}/prompt_ids.txt", "w") as f:
        f.write(" ".join(map(str, ids)))
    print(f"wrote {len(ids)} prompt ids: {ids}")
elif mode == "decode":
    prompt = sys.argv[2] if len(sys.argv) > 2 else ""
    gen_ids = [int(x) for x in sys.stdin.read().split()]
    # Stop at the first end-of-text token.
    eot = tok.token_to_id("<|endoftext|>")
    if eot in gen_ids:
        gen_ids = gen_ids[: gen_ids.index(eot)]
    print(prompt + tok.decode(gen_ids))
else:
    sys.exit(f"unknown mode {mode!r} (encode|decode)")
