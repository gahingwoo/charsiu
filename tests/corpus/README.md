# The corpora every recorded perplexity in this tree was measured on

`charsiu_ppl` is the one number here that is comparable across sessions: the
board drifts about 3% between boots and every timing with it, but perplexity
is deterministic, so a ppl measured today can be put next to one measured a
week ago. That property is worth exactly as much as the text it was measured
on, and until 2026-09-10 the text lived in a temp directory.

Every figure in `docs/lab-notebook.md` and in the README's quality table was
measured on these three files. A reboot would have taken them, and with them
the comparability of every one of those numbers -- 41.5289 would still be
written down and nothing would be able to produce it again.

⚠ **They are inputs to a measurement, so they do not get edited.** Not a
typo, not a wrapped line, not a trailing newline. Anything that changes a
byte changes every number this tree has ever recorded and silently makes the
old ones incomparable; a new question gets a new file next to these.

```
  4237c8fc3163a359fc21bde60c7b1d8b  long.txt    2264 B
  7fd405ffb641d25ddbed66f3c6414fbe  calib.txt    562 B
  96bd8dd96214fcc776d55bbacb643dc8  ppl.txt     1003 B
```

`ppl.txt` is a prefix of `long.txt` -- the first fourteen lines -- and is kept
as its own file because the runs that used it are recorded against it.

## What each one is for

`long.txt` is the evaluation text: two passages, a lighthouse keeper and a
branch railway, chosen for ordinary English at a steady register with no code,
no lists and no proper nouns a 1B model would have memorised. 300 tokens of it
is the length every quantiser comparison uses.

`calib.txt` is the AWQ calibration text, and it is deliberately about
something else -- ships and longitude. Calibrating and evaluating on the same
passage measures how well the statistics fit that passage, which is not the
question.

`ppl.txt` is the short arm, for smoke tests where 300 tokens is more wall
clock than the question deserves.

## The recipe the recorded numbers use

```sh
CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
    build/charsiu_ppl models/Llama-3.2-1B-Instruct-Q4_0.gguf \
                      tests/corpus/long.txt -n 300
```

`CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1` is the CPU reference against the SAME
quantised weights, so it isolates the quantiser from the hardware path.
Dropping `CHARSIU_NPU_W4V=1` gives the eight-bit arm. On 2026-09-10 that
recipe reproduced, to the last digit:

```
  int4, group 1024        41.5289
  INT8_LAYERS=0-1         26.0672
  int8                    17.9772
```
