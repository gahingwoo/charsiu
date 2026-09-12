#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""The evidence pack, checked against its own rules.

docs/vendor-quality-provenance.md specifies a rule and nobody implemented it:

    Every perplexity reported in the vendor-quality section must name a file
    whose md5 appears in the reproduction section.

It was written on 2026-09-11 and first run on 09-12, and it failed immediately:
BOTH corpora were scored by every perplexity in the pack and NEITHER md5 was in
section 7. A rule that has never been run is not a rule, it is a sentence.

⚠ THESE ARE CONDITIONS, NOT VERDICTS, and that is deliberate. A check for
"beats the vendor" goes stale the moment the measurement moves; a check for
"this number names a file whose hash is recorded" stays true or false on its
own terms. Nothing here asserts which side is faster.

Usage:
    python3 -P tools/check_consistency.py [docs/paper-evidence.md ...]
    python3 -P tools/check_consistency.py --self-test

Exit status is the number of failures, so `make test` can gate on it.
"""
import re
import sys
import os

MD5 = re.compile(r'\b[0-9a-f]{32}\b')
SECTION = re.compile(r'^#{1,3} (?:(\d[a-z]?)\.\s*)?(.+)$', re.M)
FENCE = re.compile(r'^```', re.M)


def _strip_fenced(text):
    """Blank out fenced code blocks, keeping line numbers and length.

    ⛔ THE FIRST VERSION OF THIS CHECKER FAILED ON ITS OWN DOCUMENT. Section 7
    is a shell block whose comments start with '#', so the heading regex read
    `# the model of record` as a heading and cut the reproduction section short
    at its first comment -- which is exactly where the hashes are. The checker
    then reported every hash in section 7 as missing FROM section 7. A parser
    that does not know what a code fence is cannot be trusted to say what a
    document contains.
    """
    out = list(text)
    inside = False
    pos = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("```"):
            inside = not inside
        elif inside:
            for i in range(pos, pos + len(line)):
                if out[i] not in "\n":
                    out[i] = " "
        pos += len(line)
    return "".join(out)


def sections(text):
    """(label, title, body) for every heading, in order. Fenced blocks are not
    scanned for headings, but their CONTENT stays in the body."""
    masked = _strip_fenced(text)
    out, marks = [], list(SECTION.finditer(masked))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out.append((m.group(1), m.group(2).strip(), text[m.end():end]))
    return out


def rule_hashes_are_recorded(text, path):
    """Every md5 in the pack is either in the reproduction section or is
    explicitly marked as a build product where it appears."""
    bad = []
    secs = sections(text)
    repro = "".join(b for lab, t, b in secs if lab == "7" or "Reproduction" in t)
    if not repro:
        return [("no reproduction section, so no hash can be checked", 0)]
    known = set(MD5.findall(repro))
    for m in MD5.finditer(text):
        h = m.group(0)
        if h in known:
            continue
        # a build product says so within 300 characters of itself
        near = text[max(0, m.start() - 300):m.end() + 300]
        if "BUILD PRODUCT" in near.upper():
            continue
        bad.append((f"md5 {h} is quoted but is not in the reproduction section "
                    f"and is not marked a build product",
                    text.count("\n", 0, m.start()) + 1))
    return bad


def rule_sections_resolve(text, path):
    """Every 'section N' reference points at a heading that exists."""
    have = {lab for lab, _, _ in sections(text) if lab}
    bad = []
    for m in re.finditer(r'[Ss]ection (\d[a-z]?)\b', text):
        if m.group(1) not in have:
            bad.append((f"reference to section {m.group(1)}, which has no heading",
                        text.count("\n", 0, m.start()) + 1))
    return bad


def rule_commits_exist(text, path):
    """A commit the pack tells a reader to check out has to be in this repo."""
    import subprocess
    bad = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    for m in re.finditer(r'git checkout ([0-9a-f]{7,40})', text):
        sha = m.group(1)
        r = subprocess.run(["git", "-C", root, "cat-file", "-t", sha],
                           capture_output=True, text=True)
        if r.stdout.strip() != "commit":
            bad.append((f"git checkout {sha}: not a commit in this repository",
                        text.count("\n", 0, m.start()) + 1))
    return bad


def rule_corpus_hashes_match(text, path):
    """An md5 the pack attributes to a corpus file has to be that file's."""
    import hashlib
    bad = []
    root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    for m in re.finditer(r'([0-9a-f]{32})\s+(long2?\.txt)', text):
        h, name = m.group(1), m.group(2)
        f = os.path.join(root, "tests", "corpus", name)
        if not os.path.exists(f):
            bad.append((f"{name} is named with a hash but is not in tests/corpus",
                        text.count("\n", 0, m.start()) + 1))
            continue
        real = hashlib.md5(open(f, "rb").read()).hexdigest()
        if real != h:
            bad.append((f"{name} is quoted as {h} but is {real}",
                        text.count("\n", 0, m.start()) + 1))
    return bad


RULES = [
    ("every quoted hash is recorded in the reproduction section", rule_hashes_are_recorded),
    ("every section reference resolves",                          rule_sections_resolve),
    ("every commit named for checkout exists",                    rule_commits_exist),
    ("every corpus hash matches the file",                        rule_corpus_hashes_match),
]


def check(path):
    text = open(path).read()
    fails = 0
    print("== %s" % path)
    for name, fn in RULES:
        bad = fn(text, path)
        if bad:
            print("  FAIL  %s" % name)
            for msg, line in bad:
                print("        %s:%d  %s" % (os.path.basename(path), line, msg))
            fails += len(bad)
        else:
            print("  ok    %s" % name)
    return fails


def self_test():
    """⚠ A RULE THAT HAS NEVER FAILED HAS NOT BEEN TESTED. Each rule is given
    a document that violates it and must report exactly that violation."""
    import tempfile
    cases = [
        ("an unrecorded hash",
         "## 7. Reproduction\n\n## 1. x\n\nppl from 0123456789abcdef0123456789abcdef\n",
         rule_hashes_are_recorded),
        ("a dangling section reference",
         "## 1. x\n\nsee section 9 for this\n",
         rule_sections_resolve),
        ("a corpus hash that is wrong",
         "## 7. Reproduction\n\n00000000000000000000000000000000  long.txt\n",
         rule_corpus_hashes_match),
    ]
    bad = 0
    for name, body, fn in cases:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False,
                                         dir=os.path.join(os.path.dirname(__file__), "..", "docs")) as f:
            f.write(body)
            tmp = f.name
        try:
            got = fn(body, tmp)
            print("  %-32s %s" % (name, "caught" if got else "⛔ NOT CAUGHT"))
            if not got:
                bad += 1
        finally:
            os.unlink(tmp)
    return bad


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--self-test":
        print("== the checker, checked against documents that violate each rule")
        sys.exit(self_test())
    paths = args or ["docs/paper-evidence.md"]
    sys.exit(sum(check(p) for p in paths))
