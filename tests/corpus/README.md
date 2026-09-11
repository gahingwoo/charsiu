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
  9c1f92b423e3ab3d3f9279a8d70a4ae6  long2.txt   2983 B
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

⚠⚠ **`long2.txt` is the SECOND OPINION, and it exists because one passage was
not enough.** On 2026-09-10 a sweep of AWQ's exponent on Qwen3-0.6B came back
NOT MONOTONE at 300 tokens of `long.txt` -- 0.15 above both its neighbours by
eight to ten percent -- while `charsiu_ppl` is deterministic, so that is the
corpus's own sampling and not the measurement. **It puts a floor of roughly ten
percent on what a single passage at that length can order.** Anything closer
than that has to be checked against a different passage or a longer run before
it is believed, and until this file there was no different passage: `ppl.txt`
is a prefix of `long.txt` and `calib.txt` is the calibration text.

It is a different subject on purpose -- a mill leat, hand papermaking, bell
founding, winter bees -- and, like the others, ordinary English at a steady
register with no code, no lists and no proper nouns a 1B model would have
memorised. **It does not replace `long.txt`**: every number on record was
measured on that one, and a second opinion is a second number, not a new
baseline.

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
  int4, ONE SCALE A ROW   41.5289
  INT8_LAYERS=0-1         26.0672
  int8                    17.9772
```

⛔⛔ **THAT FIRST ROW SAID `int4, group 1024` UNTIL 2026-09-11 AND THE RECIPE
ABOVE IT DOES NOT SET A GROUP.** `llama_auto_kmax()` pins the group to 1024 and
is called only when the NPU is on, so `CHARSIU_NPU=0` never reaches it and
takes npuquant's own default of one absmax a row. The same file at group 1024
is **33.8071**, a fifth away. The number was right and the label was wrong, in
the file whose whole job is to say what the numbers mean.

To measure what the BOARD runs, set the group by hand:

```sh
CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
    CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
    build/charsiu_ppl models/Llama-3.2-1B-Instruct-Q4_0.gguf \
                      tests/corpus/long.txt -n 300      # 33.8071
```

The runtime says which of the two it is taking, once, on stderr.

## ⛔⛔ THE RECORDED NUMBERS ARE ON A FILE HUGGING FACE NO LONGER SERVES

2026-09-11. The board and the desk disagreed by 1.3% on the same nominal model.
It was neither the toolchain nor the thread count: **it was the file.**
`bartowski/Llama-3.2-1B-Instruct-GGUF` has been re-uploaded at some point, and
the copy baked into this project's rootfs-overlay is the older one.

```
                                              md5           bytes      a row   g1024
  rootfs-overlay, the whole record on it   c82c0340...   773025824   41.5289  33.8071
  what Hugging Face serves today           48ff0243...   773025920   41.3739  34.2425
```

🔑 **With the SAME file the two machines agree to the last digit.** Desk against
board, three arms, and the calibration pass writing the same 2647768 bytes:

```
              desk      board
  int4      34.2425   34.2425
  AWQ 0.20  23.6746   23.6746
  int8      18.3604   18.3604
```

The desk is an aarch64 VM on GCC 13.3 and glibc 2.39; the board is a ROCK 4D
on GCC 15.2 and glibc 2.43, a different kernel and a different thread count.
**Perplexity survives all of that. It does not survive a different file**, and
the file is the one thing nobody writes down.

⚠ So a reader reproducing the README's 33.8071 today will get **34.2425** and
have no way to see why. Every recorded figure stands, on the file named beside
it; the file of record going forward is the one Hugging Face serves, because
that is what a reader will get and what the board already runs.

## ⚠⚠ A perplexity needs THREE names, and the third is the one nobody writes

**A model, a corpus, and a FILE.** charsiu re-quantises whatever it loads, so
the source format survives into the answer. Three ggufs of the same
Llama-3.2-1B, these same 300 tokens, the same binary:

```
                                   one scale a row    group 1024
  Q8_0                                 34.6888         28.7072
  Q4_0  c82c0340 (rootfs-overlay)      41.5289         33.8071   <- the record
  Q4_0  48ff0243 (Hugging Face today)  41.3739         34.2425   <- what a reader gets
  Q4_0 "pure"                          41.8712         30.2425
```

⚠ **The two Q4_0 rows are the same URL at different times.** A file name and a
quantisation are not an identity; only the md5 is.

**Every Llama figure on record is the Q4_0 file**, and it reproduces to the
last digit. A number from one file put beside a number from another is a 20%
error with nothing on the page to show it.

⚠ `scripts/charsiu-get` annotates its **Q8_0** line "THE ONE EVERY BOARD ROUND
USES", and the board's own AWQ round read 33.4149 -- the Q4_0 figure. Both may
be true, the timing rounds on one file and the quality round on the other, and
that is worse than either being wrong: it means the two halves of the
scoreboard are not about the same weights. It is not resolved yet, and this
paragraph is here so it is not resolved silently.

🔑 **`tests/vendor_quality.sh` is the exception and says so**: its arms all
descend from the Q8_0 file, because there the question is which QUANTISER is
better and a source that is already four bits flatters whichever one is asked
to quantise it again.
