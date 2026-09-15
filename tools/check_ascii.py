#!/usr/bin/env python3
"""No non-ASCII character in C code, as opposed to in a comment or a string.

WHY THIS EXISTS. src/npuquant.c declared `double bestе = -1.0` where the final
letter is U+0435 CYRILLIC SMALL LETTER IE, and used it three more times with
the same spelling. It compiled, it ran, and it was correct -- and `grep beste`
found nothing, `grep best` showed a name that looked ASCII, and anyone who
retyped it would have got a different identifier. It sat next to `bestd`, so
the name it wanted was `beste`.

A non-ASCII character inside a comment or a string is often DATA and has to
stay: tokenizer.c explains SentencePiece's U+2581 meta-space and cannot do it
in ASCII, npuquant.c quotes a model's own garbled output, and
tests/tokenizer_cross.py feeds emoji to the tokenizer on purpose. So the rule
is not "no non-ASCII in the file". It is "none in the CODE", which needs the
comments and strings actually parsed rather than grepped around.

    tools/check_ascii.py [FILE ...]      default: src/*.c src/*.h tools/*.c
    tools/check_ascii.py --self-test
"""
import glob
import sys


def code_only(text):
    """Yield (line, col, ch) for every non-ASCII character that is not inside a
    comment, a string literal or a character literal."""
    i, n = 0, len(text)
    line, col = 1, 1
    state = "code"
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "*":
                state, i = "block", i + 2
                col += 2
                continue
            if c == "/" and nxt == "/":
                state, i = "line", i + 2
                col += 2
                continue
            if c == '"':
                state = "str"
            elif c == "'":
                state = "chr"
            elif ord(c) > 127:
                yield (line, col, c)
        elif state == "block":
            if c == "*" and nxt == "/":
                state, i = "code", i + 2
                col += 2
                continue
        elif state == "line":
            if c == "\n":
                state = "code"
        elif state in ("str", "chr"):
            if c == "\\":
                i += 2
                col += 2
                continue
            if (state == "str" and c == '"') or (state == "chr" and c == "'"):
                state = "code"
        if c == "\n":
            line += 1
            col = 1
        else:
            col += 1
        i += 1


def check(paths):
    bad = 0
    for p in paths:
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError) as e:
            print("  !!   %s: %s" % (p, e))
            bad += 1
            continue
        for line, col, ch in code_only(text):
            print("  !!   %s:%d:%d U+%04X %s is in CODE, not a comment or a "
                  "string" % (p, line, col, ord(ch), repr(ch)))
            bad += 1
    if bad:
        print("\n%d non-ASCII character(s) in code." % bad)
        return 1
    print("  ok    %d file(s): every non-ASCII character is in a comment or a "
          "string" % len(paths))
    return 0


def self_test():
    """A RULE THAT HAS NEVER FAILED HAS NOT BEEN TESTED."""
    cases = [
        ("double bestе = -1.0;", 1, "a Cyrillic ie in an identifier"),
        ("/* ▁ is the meta-space */", 0, "a block comment"),
        ("// ▁ is the meta-space", 0, "a line comment"),
        ('puts("▁the");', 0, "a string literal"),
        ("char c = 'µ';", 0, "a character literal"),
        ('s = "a\\"▁b"; int е = 1;', 1, "an escaped quote then code"),
        ("/* → */ int a; // →\n int b;", 0, "both comment kinds"),
        ('"/* not a comment ▁ */"', 0, "a comment opener inside a string"),
    ]
    ok = True
    for src, want, why in cases:
        got = len(list(code_only(src)))
        mark = "ok  " if got == want else "FAIL"
        if got != want:
            ok = False
        print("  %s  %-34s want %d, got %d" % (mark, why, want, got))
    return 0 if ok else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--self-test":
        print("== the parser, against text that must and must not trip it")
        sys.exit(self_test())
    files = args or sorted(glob.glob("src/*.c") + glob.glob("src/*.h")
                           + glob.glob("tools/*.c") + glob.glob("tests/*.c"))
    sys.exit(check(files))
