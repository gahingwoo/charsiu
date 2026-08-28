#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Diff charsiu's CLIP BPE against one driven by the REAL merges.txt.

⚠ WHY THIS TEST EXISTS. The gguf carries `tokenizer.ggml.tokens` and no merge
list -- clip.cpp never ran the text tower, so its converter had no reason to
write one. charsiu recovers the ranks from the vocabulary ORDER instead, on the
claim that CLIP's vocabulary IS the merge order: 256 byte symbols, the same 256
with the end of word marker, then every merge as it was learned.

That claim is the thing under test. The reference below merges by the rank in
openai/clip-vit-base-patch32's merges.txt; charsiu merges by the lowest index in
the vocabulary. If the claim is wrong they diverge on some word, and this prints
which one.

    tests/clip_tokenizer_cross.py build/charsiu_clip MODEL.gguf MERGES.txt VOCAB.json
"""

import json
import re
import subprocess
import sys


def bytes_to_unicode():
    bs = (list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


PAT = re.compile(
    r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"""
    r"""[a-z]+|[0-9]|[^\sa-z0-9]+""", re.IGNORECASE)


class Reference:
    """CLIP's BPE, driven by merges.txt ranks."""

    def __init__(self, merges_path, vocab_path):
        self.enc = json.load(open(vocab_path, encoding="utf-8"))
        lines = open(merges_path, encoding="utf-8").read().split("\n")
        lines = [l for l in lines[1:] if l and " " in l]
        self.ranks = {tuple(l.split()): i for i, l in enumerate(lines)}
        self.b2u = bytes_to_unicode()

    def bpe(self, word):
        syms = [self.b2u[b] for b in word.encode()]
        if not syms:
            return []
        syms[-1] += "</w>"
        while len(syms) > 1:
            best, at = None, -1
            for i in range(len(syms) - 1):
                r = self.ranks.get((syms[i], syms[i + 1]))
                if r is not None and (best is None or r < best):
                    best, at = r, i
            if at < 0:
                break
            syms[at:at + 2] = [syms[at] + syms[at + 1]]
        return [self.enc[s] for s in syms if s in self.enc]

    def encode(self, text):
        out = [self.enc["<|startoftext|>"]]
        for w in PAT.findall(text.lower()):
            out += self.bpe(w)
        out.append(self.enc["<|endoftext|>"])
        return out


TEXTS = [
    "a photo of a cat", "a dog", "a diagram", "a photograph of the eiffel tower",
    "an astronaut riding a horse in space", "hello, world!",
    "the quick brown fox jumps over the lazy dog", "2024",
    "a close-up of a bee on a pink flower", "screenshot of a web page",
    "it's a llama, isn't it? we're sure", "san francisco at night",
    "unbelievably photorealistic rendering", "x", "!!!", "a b c d e f g",
    "restaurant menu with prices", "the 3rd of may, 1808",
    "a man's hat and a woman's coat", "supercalifragilisticexpialidocious",
    "GPU, CPU and NPU benchmarks", "black-and-white photography",
    "a bird's-eye view of tokyo", "some text with    extra   spaces",
]


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    tool, model, merges, vocab = sys.argv[1:5]
    ref = Reference(merges, vocab)

    cmd = [tool, model, "--tokens", "--text"] + TEXTS
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr)
        return 1
    lines = [l for l in r.stdout.strip().split("\n") if l]
    if len(lines) != len(TEXTS):
        print(f"FAILED: {len(lines)} results for {len(TEXTS)} strings")
        return 1

    fail = 0
    for text, line in zip(TEXTS, lines):
        got = [int(v) for v in line.split()][1:]
        want = ref.encode(text)
        if got != want:
            fail = 1
            print(f"FAILED  {text!r}")
            print(f"  charsiu   {got}")
            print(f"  reference {want}")
    print(f"{len(TEXTS)} strings, {'FAILED' if fail else 'all identical'}")
    print("FAILED" if fail else "PASS")
    return fail


if __name__ == "__main__":
    sys.exit(main())
