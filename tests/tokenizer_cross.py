#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
"""
Cross check charsiu's byte level BPE against the reference tokenizer.

The pre-tokenizer is the part of a BPE most likely to be subtly wrong, and a
subtly wrong one still produces fluent text, so eyeballing an output proves
nothing. This diffs ids against huggingface's own implementation on a corpus
built to hit each branch of the alternation.

    tests/tokenizer_cross.py build/charsiu_run model.gguf tokenizer.json [N_FUZZ]
"""

import subprocess
import sys
import tempfile
import os

CASES = [
    # the plain path
    "The capital of France is Paris.",
    "Hello, world!",
    "a",
    "",
    " ",
    "  ",
    "   ",
    # contractions, the first branch of the alternation
    "don't", "DON'T", "I'll", "we've", "he's", "it'd", "they're", "o'clock",
    "'s", "'S", "'re", "''", "'''",
    # digits, which split into groups of at most three
    "1", "12", "123", "1234", "12345", "1234567890",
    "3.14159", "1,000,000", "v1.2.3", "0x1F", "2026-08-21",
    # a letter run with one lead-in character
    "-word", "(word)", "#hashtag", "@user", "_underscore", "a-b-c",
    # whitespace, where (?!\S) decides who gets the last space
    "a  b", "a   b", "a\tb", "a\nb", "a\n\nb", "a \n b", "\n", "\n\n",
    "  leading", "trailing  ", "\ttab", "line1\nline2\nline3",
    "a \r\n b", "\r\n\r\n",
    # punctuation runs and the trailing newline the fourth branch takes
    "!!!", "...", "?!", "-->", "**bold**", "a...\n", "!!\n\n",
    # code, which mixes every branch
    "int main(void) { return 0; }",
    "for (i = 0; i < n; i++)\n\tprintf(\"%d\\n\", i);",
    "x = {'a': 1, 'b': [2, 3]}",
    "SELECT * FROM t WHERE id=1;",
    "https://example.com/path?q=1&r=2",
    # unicode
    "café", "naïve", "Grüße", "Ελλάδα", "Привет", "שלום", "مرحبا",
    "日本語のテキスト", "中文测试", "한국어",
    "中文,标点。测试", "全角１２３",
    "emoji 🎉 test", "🇨🇳🇬🇧", "a🎉b",
    "→ ← ↑ ↓", "½ ¼ ¾", "° ± × ÷", "µ ª º",
    # mixed
    "Cost: $1,234.56 (as of 2026-08-21)",
    "北京は中国の首都です。Population: 21,893,000.",
    "  \n  mixed   whitespace \t\n\n  end  ",
    # the control tokens a chat prompt is built out of
    "<|begin_of_text|>hello",
    "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|>",
    "text <|eot_id|> more text",
    "<|end_of_text|>",
    "a<|eot_id|>b",
    # things that look like a control token but are not
    "<|not_a_token|>", "<|", "|>", "<||>",
]


def charsiu_tokens(exe, model, text):
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(text.encode())
        path = f.name
    try:
        r = subprocess.run([exe, model, "--tokens", "--no-bos", "-f", path],
                           capture_output=True, check=True)
        # a token can be half a utf8 sequence, so the listing is not valid utf8
        out = r.stdout.decode("utf-8", "replace")
    finally:
        os.unlink(path)
    ids = []
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) >= 2:
            ids.append(int(parts[1]))
    return ids


BLOCKS = [
    (0x20, 0x7e),        # ascii
    (0xa0, 0xff),        # latin-1, where the fractions live
    (0x100, 0x17f),      # latin extended
    (0x370, 0x3ff),      # greek
    (0x400, 0x4ff),      # cyrillic
    (0x590, 0x5ff),      # hebrew
    (0x600, 0x6ff),      # arabic
    (0x900, 0x97f),      # devanagari
    (0x2000, 0x206f),    # general punctuation
    (0x2100, 0x21ff),    # letterlike and arrows
    (0x2460, 0x24ff),    # circled numbers
    (0x3000, 0x30ff),    # cjk punctuation and kana
    (0x4e00, 0x4eff),    # cjk
    (0xac00, 0xacff),    # hangul
    (0xff00, 0xffef),    # fullwidth
    (0x1f300, 0x1f5ff),  # emoji
]


def fuzz_cases(n, seed):
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 12)):
            r = rng.random()
            if r < 0.30:
                parts.append(rng.choice([" ", "  ", "\n", "\t", " \n ", "\n\n"]))
            elif r < 0.45:
                parts.append(str(rng.randint(0, 10 ** rng.randint(1, 8))))
            elif r < 0.60:
                parts.append(rng.choice(["'s", "'t", "'re", "'ll", "'ve", "'d", "'"]))
            else:
                lo, hi = rng.choice(BLOCKS)
                k = rng.randint(1, 6)
                w = "".join(chr(rng.randint(lo, hi)) for _ in range(k))
                parts.append(w)
        t = "".join(parts)
        # surrogates and unassigned codepoints are not text; drop those draws
        try:
            t.encode()
        except UnicodeEncodeError:
            continue
        out.append(t)
    return out


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    exe, model, tokjson = sys.argv[1:4]
    n_fuzz = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    from tokenizers import Tokenizer
    ref = Tokenizer.from_file(tokjson)

    cases = list(CASES)
    if n_fuzz:
        cases += fuzz_cases(n_fuzz, 20260821)

    bad = 0
    for text in cases:
        want = ref.encode(text, add_special_tokens=False).ids
        got = charsiu_tokens(exe, model, text)
        if want != got:
            bad += 1
            print("MISMATCH %r" % text)
            print("   want %s" % want)
            print("   got  %s" % got)
            print("   want %s" % [ref.decode([t], skip_special_tokens=False) for t in want])
            print("   got  %s" % [ref.decode([t], skip_special_tokens=False) for t in got])

    print("%d cases, %d mismatches" % (len(cases), bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
