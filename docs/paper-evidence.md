# Evidence pack, 2026-09-11

Every number a paper could quote from this project, with the protocol that
produced it and the sentence that has to travel beside it. Assembled over a day
of board rounds on one ROCK 4D.

Tree state: `dev 4057a39`. Board: Armbian, kernel `7.2.0-rc5-next-20260730+`,
GCC 15.2.0, glibc 2.43. Desk: aarch64 VM, GCC 13.3.0, glibc 2.39.

---

## 0. Three things that must be said beside the numbers

**The vendor column in section 1 is a citation, not an arm.** It is copied
from `airockchip/rknn-llm/main/benchmark.md`, fetched 2026-08-28, and their
header says the figures were "collected based on the maximum CPU and NPU
frequencies of each platform". The column has no N, no spread, and no method
beyond that one sentence, so a margin over it cannot be inside anyone's noise
in either direction. Section 1 is our median of seven with its range, against
their published point.

⚠ **Section 1d is different: their runtime has now been RUN here**, on the
same board, the same kernel and the same clock as ours, and it is an arm.
Section 1 and section 1d are two different comparisons and must not be merged
into one table: section 1 is four models at maximum frequency against a
citation, section 1d is one model at 594 MHz against a measurement. The
citation and the arm do not even agree about which side is ahead on TTFT, for
a reason section 1d gives.

**A perplexity needs a model, a corpus, and a file.** charsiu re-quantises
whatever it loads, so the source format is inside every quality number. The
same Llama-3.2-1B reads 41.37, or 34.24, or 28.71, depending on which file and
which group. Every figure here names its md5.

**One board.** Everything measured on hardware is a single ROCK 4D. That
separates voltage from clock, and it does not separate this board from the
part, so a leakage bin cannot be ruled out. The ArmSoM CM5 is RK3588S and
cannot test this rail at all.

---

## 1. Speed against the vendor

One boot, `performance` governor, `board_record.sh` REPEAT=7, median of seven
with every reading kept. The binary pins its calling thread to the fast cluster
(a58086c). Their protocol is a 128-token prompt and 64 new tokens.

```
              decode t/s                      TTFT ms
            ours (median)   spread   theirs      ours (median)  range        theirs
  Qwen3 0.6B      26.26      1.9%    24.85           613      601..623      468.61
  TinyLLAMA       22.79      1.2%    19.71           890      876..903      543.68
  Phi3 3.8B        7.03      0.1%     6.58          2987     2942..3013    1829.12
  Gemma4 E2B       9.25      1.7%     9.23          2222     2193..2245    1219.25
```

Decode is ahead on all four: +5.7%, +15.6%, +6.8%, +0.2%.
TTFT is behind on all four: 1.31x, 1.64x, 1.63x, 1.82x theirs.

The prompt is where this runtime is still losing, and it is the half the vendor
spends 3328 M=1 dispatches on. Reporting only the decode column would be
choosing a column.

Their TTFT and ours are not the same quantity. Theirs is time to the first
token; ours is the prompt's forward passes, so the first token's own step is in
theirs and not in ours, which is one token's worth in our favour.

### 1a. Which of their three columns this is, and what the other two say

`benchmark.md` gives three quantisations a model, not one. For the same four:

```
                 w4a16            w4a16_g128         w8a8
              TTFT    t/s        TTFT    t/s      TTFT    t/s
  Qwen3 0.6B  468.61  24.85     506.41  23.48    461.54  17.17
  TinyLLAMA   543.68  19.71     672.61  18.08    534.13  12.18
  Phi3 3.8B  1829.12   6.58    2253.14   6.06   1615.97   3.74
  Gemma4 E2B 1219.25   9.23    1445.36   8.27   1166.94   5.56
```

**The comparison above uses w4a16, which is their FASTEST decode of the three.**
Against w8a8 the same charsiu medians would read +52.9%, +87.1%, +88.0% and
+66.4% instead of +5.7%, +15.6%, +6.8% and +0.2%. A paper that reported the
w8a8 row would be claiming roughly four times the margin for the same work.

⚠ **It is not uniformly the conservative choice.** Their w8a8 TTFT is slightly
FASTER than their w4a16 on three of the four, so picking w4a16 is conservative
on the column we lead and slightly generous on the column we trail. Both halves
of that belong in the sentence.

⚠ w8a8 also costs roughly twice the memory: 796 MB against 513 for Qwen3,
3767 against 1996 for Phi3. A reader choosing between them is not choosing on
speed alone, and neither column is "the vendor's number" on its own.

🔑 **`w4a16_g128` exists, and section 2's group caveat is about OUR FILE, not
about what they can do.** The `.rkllm` scored there keeps one scale per output
row; this column shows they ship a group-128 option as well. The asymmetry
section 2 states is between two particular files, not between two vendors'
capabilities.

### 1d. Their runtime, RUN

`board-logs/r389`. The vendor's rknpu driver built against this board's own
kernel from Kiln's mainline port, their `librkllmrt` 1.3.0 loading
`Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm`, md5 `2d3962468e2e7c0d0571157f8c9eae71`,
the same file section 2 scores. Timed with the runtime's own `RKLLMPerfStat`
rather than from outside, so a disagreement about where prefill ends cannot be
blamed on the harness.

Held equal: one Image, 594 MHz, identical prompt strings, `-n 16`, ignore-eos,
context 512. Both runtimes pin the big cluster, and the vendor's says so on
stdout: `Enabled cpus: [4, 5, 6, 7]`.

```
  decode tok/s     charsiu 18.56 (18.47..18.58)   vendor 12.88 (12.71..12.95)
                   charsiu 1.44x
```

⚠⚠ **The prefill answer depends on which question is asked, and the two
readings point in OPPOSITE directions.** The vendor wraps the prompt in its own
chat template: the same string is 50 tokens to them and 16 to us.

```
  wall-clock TTFT, same string    charsiu 265 ms      vendor 466 ms   charsiu 1.76x
  prefill throughput              charsiu 61 tok/s    vendor 107 tok/s vendor 1.76x
```

Both are true. Two prompts separate the rate from the amount: a second, longer
prompt is 79 tokens to us and 113 to them.

```
               short              long             ms/token   fixed cost
  charsiu   16 tok  265 ms     79 tok  751 ms        7.71       141.6 ms
  vendor    50 tok  466 ms    113 tok  860 ms        6.25       153.6 ms
```

🔑 **The slope is the prefill rate and the intercept is the overhead.** Per
prompt token the vendor is 1.23x faster, which is the real gap and is far
smaller than either single-prompt reading. The fixed costs are within 8% and
ours is the lower of the two. charsiu's wall-clock win on a short prompt is a
tokeniser difference and must not be quoted as a prefill result.

⚠ The rail differs: 800 mV for charsiu, 750 for the vendor, because their
driver sets it through `npu-supply` and its OPP table rather than taking the
device tree's floor. Both compute correctly at 594 MHz and voltage does not set
clock rate, so this is a caveat and not a confound.

⚠ Neither side scaled. Their device tree asks for devfreq over an OPP table
reaching 800 MHz; Kiln's patch pins the clock. So this says nothing about
either side at the frequency their published figures were taken at.

⛔ **Their runtime cannot be asked for perplexity, so section 2 cannot be
cross-checked this way.** `RKLLM_INFER_GET_LOGITS` is in their API and the
model refuses it: every sequence length tried, 64 through 544, returns
`meet unkown shape, op name: matmul_qk_rkllm_spilt_0` and no logits. The
model is exported for generation and the attention shapes for a logits pass
are not in its compiled set. Section 2's reconstruction therefore remains
unvalidated against their real runtime, and the pack still says the
112-matrix figure measures the reconstruction.

### 1b. The decode margin is younger than the round

Before `a58086c` the decode was bimodal and the table reported best-of-N:

```
  Qwen3      20.76 24.25 20.61 20.27 26.06 20.81 20.54     best 26.06, median 20.76
  TinyLLAMA  19.11 19.41 22.83 21.94 19.24 19.15 20.08     best 22.83, median 19.41
  Phi3        7.04  5.96  5.98  5.96  5.96  5.96  7.04     best  7.04, median  5.96
  Gemma4      7.25  7.34  7.28  9.33  7.28  7.32  7.28     best  9.33, median  7.28
```

Four threads (`-t 4`) on a part with four A72s and four A53s: when all four
landed on the A72s it was fast. `taskset -c 0-7` behaved exactly like no
taskset, so it was never about which cores were permitted, only about where the
scheduler put them.

```
  default median  20.92 -> 26.25   +25.5%       spread 28.5% -> 2.3%
```

The new medians are the old best-of-seven to within 1%. The claim on record was
therefore the right number reached the wrong way: the high mode was the
machine's real capability, and best-of-N was reporting something the runtime
could do but would not do reliably. A paper should give the margin as
+5.7%/+15.6% from a median, say that it required pinning, and not quote a
best-of-N.

### 1b-2. The TTFT gap is not attributed, and the obvious explanation does not fit

The runtime's own report from the same round:

```
              TTFT  theirs  ratio   submits  MB/submit  "dispatch rather than bytes"
  Qwen3        613  468.61  1.31x    14718     1.32          23%
  TinyLLAMA    890  543.68  1.64x    11702     2.87          14%
  Phi3        2987 1829.12  1.63x    16962     7.13           4%
  Gemma4      2222 1219.25  1.82x    25790     2.92          13%
```

Per-call overhead does not explain the gap. The model with the most dispatch
overhead has the smallest gap (Qwen3, 23% and 1.31x) and the one with the least
has nearly the largest (Phi3, 4% and 1.63x). The correlation runs backwards.

Those counters could not attribute TTFT even if they had pointed the right way.
They cover the whole process, and at 64 generated tokens a run the calls are
mostly decode's. Any attribution of prompt time from them is a category error,
including the paragraph above, which is written down to close the road rather
than to travel it.

The report's own GB/s figures are self-flagged in all four models: *"that
remainder is NEGATIVE, so the hardware path and the wall clock are counting
different calls, the rate above is not a fact about the hardware"*. Do not
quote them.

### 1b-3. Where the prompt's time actually goes

`tests/board_prefill_stages.sh` with `-n 1`, so the table is the prompt's, and
the protocol prompt, so the rows are the rows the TTFT column is measured over:

```
              ms/row   in the NPU entry     pack   fence    read   read/fence  read/total
  Qwen3         5.66    3.90   (69%)        0.85    1.28    1.39     1.09x        25%
  TinyLLAMA     7.79    6.12   (79%)        0.89    2.22    2.79     1.26x        36%
  Phi3         25.10   20.45   (81%)        2.72    8.72    8.63     0.99x        34%
  Gemma4       17.38   12.79   (74%)        2.17    4.59    5.16     1.12x        30%
```

Three results, and the first two close roads.

Zero rows fell back to the CPU on any model, so the silent-fallback
explanation, a projection the hardware refuses becoming a matvec a row at a
time, is dead. The batched path is entirely on the hardware.

Submitting costs 0.06 to 0.12 ms a row, 1 to 2% of the prompt. Per-call
dispatch is not the prefill story, which is the second independent way that
explanation has failed.

Reading the results back costs as much as computing them. `read` is 0.99 to
1.26 times `fence`, the hardware's own MAC time, and is 25 to 36% of the whole
prompt on every model.

The read-back is therefore the largest single lever in prefill, and it is set
by the quantisation group: the read volume is `m * n * ceil(K/KMAX) * 4`, so a
finer group is paid for here. Prefill speed and answer quality are traded
through one parameter, which is a tension a paper can state precisely.

This attributes charsiu's prompt time. It does not by itself explain the size
of the gap against the vendor, because there is no equivalent breakdown of
theirs, and `read/fence` does not track the gap across the four models (Phi3
has the lowest ratio and nearly the largest gap). What can be said is what
charsiu spends the prompt on, and that the vendor dispatches at M=1 where
charsiu batches, so the two are not paying the same costs in the same places.

The probe also prints an `in its wrapper` figure, 24.21 ms a row on Qwen3
against 3.90 inside the entry. That one includes staging, 4135 ms of it on
Qwen3, which is a once-per-process cost and is excluded from both the prompt
total and TTFT. It is not prompt time.

### 1c. The pinning default is safe across every architecture

`a58086c` changes a default that touches every workload on every model, and
what had been checked was one model's decode text. `board_text_all.sh` compares
each model's batched prompt against its own token loop, on the hardware:

```
  Phi-3.5-mini   Qwen2.5-1.5B   Qwen3-0.6B   SmolLM2-1.7B   SmolLM2-135M
  gemma-3-1b     gemma-4-E2B    tinyllama-1.1b   Llama-3.2-1B

  9 models compared, 0 differing, every one "prompt batched, text identical"
```

Every row says `prompt batched`, not `prompt a token`. A model that refuses to
batch would report "text identical" meaning only that the token loop agrees
with itself; none did. This is the check that caught gemma4 emitting
"31 32 1 2 3" on the card in 2026-08-30, after six architectures had passed on
a desktop.

---

## 2. Quality against the vendor's own int4

The comparison nothing in the literature has: the vendor's stored weights,
scored. It needs no board and no vendor install. `tools/rkllm_rebuild.py` reads
the `.rkllm` and `tests/vendor_quality.sh` scores it.

All arms are f16 files differing only in the swapped matrices, rebuilt from
`Llama-3.2-1B-Instruct-f16.gguf` (md5 `3ba43423d342673e26016ffe85268937`) so
that both sides quantise the ORIGINAL weights, at `-n 300`, with no quantiser
running at inference. Every rung is scored on both passages:

```
                            tests/corpus/long.txt      tests/corpus/long2.txt
  reference (f16)                  17.9023                     32.8163

  matrices                vendor  charsiu  ratio      vendor  charsiu  ratio
  43  (rho = 1)          +13.81%   +7.45%  1.85x     +10.26%   +3.92%  2.62x
  91  (layers 3..15)     +41.00%  +20.48%  2.00x     +26.01%  +18.38%  1.42x
  105 (all but layer 1)  +67.19%  +30.82%  2.18x     +54.30%  +32.93%  1.65x
```

**What is robust is the direction: six cells of six, the vendor's four-bit
excess is larger than charsiu's.** The size of it is an interval, 1.4x to 2.6x,
and the paper should quote the interval and nothing narrower.

⛔ **The ladder is NOT monotone and the earlier text saying so was one
passage.** On `long.txt` the ratio climbs with the subset, 1.85 to 2.00 to
2.18, which is what "monotone" was read from. On `long2.txt` it does not:
2.62, 1.42, 1.65, and the 43-matrix rung goes from the lowest of the three to
the highest. Nothing about subset size orders these; only the sign survives.

**The quantisation origin, which is why the numbers above are not the ones
first published.** The vendor quantised the ORIGINAL weights; the charsiu arm
quantised Q8_0, because that is what the reference pointed at, so "asked to
quantise the same thing" was not defensible. On the 43-matrix rung, both
origins:

```
                        charsiu     vendor      ratio
  from Q8_0   long        +6.65%    +13.32%     2.00x
              long2       +4.39%     +9.50%     2.16x
  from f16    long        +7.45%    +13.81%     1.85x
              long2       +3.92%    +10.26%     2.62x
```

The asymmetry is real and points no particular way: from f16 charsiu is worse
on one passage and better on the other. Report the f16 rows: they are the arm
whose precondition holds. The Q8_0 interval is narrower for no good reason.

⚠ `rho1` and `vendor43` are the SAME FILE, byte for byte, md5
`9d8e82952e96a6f14eeaa4b016704dde`. That is the point of the rho = 1 subset
rather than a second measurement of it: dividing the calibration out changes
not one f16 weight there, so the two code paths land on the same weights. Do
not report them as two agreeing arms.

**The group size.** The vendor keeps one fp32 scale and one integer zero point
per OUTPUT ROW, so its group is the whole of K against charsiu's 1024. Matching
the group is the obvious remedy and it is worse than the caveat: 2.00 and 2.16
at group 1024 on the two passages, against 3.76 and 1.97 matched, which swings
by a factor of two and cannot be quoted. State the difference; do not try to
remove it.

A third asymmetry runs the other way and no arm removes it: the vendor is
asymmetric with an integer zero point and charsiu is symmetric absmax, worth
about 1.4% at group 1024 by this tree's own measurement.

### What makes it evidence rather than a reconstruction of mine

The weight layout is solved and held out: fitted on blocks 0 to 47, scored on
rows 768 to 2047, 99.72% of codes away from a rounding boundary, with residuals
of plus or minus 1, which is the boundary signature.

The calibration is not recovered, it cancels. The vendor folds 1/c into the
RMSNorm ahead of each projection, so `corr(vendor_norm, ref/c)` runs 0.9905 to
0.9945 against 0.78 to 0.89 for `corr(vendor_norm, ref)`.

The noise control decides whether any of it means anything. Unstructured error
at the same per-tensor magnitude costs +60.00% where their actual quantisation
costs +13.32%, so the vendor row sits four and a half times further from noise
than from the reference.

An earlier version scored 1700.98 by recovering c through division, while the
same magnitude as Gaussian noise scored 32.10. That is how the fault was found,
and it is why the noise arm is inside the harness rather than beside it.

Layer 1 is excluded and the reason is named. All 112 matrices read 58.76
against 32.13 for the same set minus layer 1; `blk.1` carries the most extreme
row gauge in the model and `blk.1.ffn_down` is not reconstructed at all. That
number measures the reconstruction rather than their quality, and is not
quoted.

The unrecorded protocol gave 1.65 / 1.71 / 1.81 for the same three sets: same
shape, consistently lower. Neither ladder is quotable without its corpus and
length, and this one has them.

---

## 3. Quality of charsiu's own quantiser

CPU reference (`CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1`) at the group the board
runs, on `Llama-3.2-1B-Instruct-Q4_0.gguf` md5 `48ff0243`, 773025920 bytes,
`tests/corpus/long.txt` at `-n 300`:

```
  int4, group 1024        34.2425
  + AWQ alpha 0.20        23.6746     -30.9%
  int8                    18.3604
```

On a second architecture, Qwen3-0.6B (md5 `45f23a28`), same group, same corpus,
through `tests/host_awq.sh`:

```
  AWQ off                 95.5754
  AWQ on, no statistics   95.5754     must equal off: it declines, and does
  AWQ on, alpha 0.20      72.5107     -24.1%
```

So AWQ is worth 24 to 31% across two architectures at the configuration the
board runs. Qwen3's own optimum is 0.25 rather than 0.20, so its 24.1% is the
conservative reading of its own row.

The middle arm is a tell rather than decoration. With no calibration statistics
`CHARSIU_NPU_AWQ` declines and says so, and an AWQ arm that silently equals its
control is what a missing calibration looks like. It cost a board round.

Board and desk are bit-identical on all three arms, and the calibration pass
writes the same 2647768 bytes, across two compilers, two glibcs, two kernels
and two thread counts. Perplexity survives all of that.

It does not survive a different file.
`bartowski/Llama-3.2-1B-Instruct-GGUF` has been re-uploaded, and this project's
image carries the older copy:

```
                                     md5        bytes      a row   g1024
  in the rootfs-overlay          c82c0340   773025824   41.5289  33.8071
  what Hugging Face serves now   48ff0243   773025920   41.3739  34.2425
```

Every quality figure recorded in this tree before 2026-09-11 is the first row.
A reader reproducing today gets the second. Both are named by md5 in
`tests/corpus/README.md`, and the file of record going forward is the one a
reader will actually get.

"Board and host agree to 0.3%" in the earlier AWQ round was two different files
landing near each other. It is not evidence of anything.

---

## 4. The output head, and why 12.40 and 12.59 are not a disagreement

`tests/prefill_control.sh` runs batched, control, batched on one binary:

```
  control   65 tok / 4304 ms   66.22 ms a token   15.10 tok/s
  batched   65 tok / 3498 ms   53.82 ms a token   18.59 tok/s  (18.63, 18.54)
  saved                        12.40 ms a token
```

The head runs once instead of 65 times, so the saving is `H * 64/65` where H is
the head's own cost, 12.59 ms:

```
  12.59 * 64/65 = 12.396 ms   predicted
                  12.40  ms   measured
```

Independently, Llama-3.2-1B's head is 128256 x 2048, which at int4 is 131.3 MB:

```
  131.3 MB / 12.59 ms = 10.43 GB/s   against the 10.58 GB/s the NPU summary reports
```

That is 1.4% apart, from a shape and a rate that never saw the stopwatch. It is
the strongest triangulation in the project, and it is one measurement plus two
independent predictions rather than two estimates of one thing.

---

## 4b. The two cores, the overlap fault, and what the speed numbers ran under

The fault is real and it is attributed: it is the NPU's voltage margin, not the
overlap. `src/overlap.h` states it and the notebook carries the sweep, four
device trees, same probe, four passes of 5400 rows each:

```
  786 MHz, 750 mV (mainline as U-Boot leaves it)   11 to 25 wrong words a pass
  594 MHz, 750 mV                                   0, 0, 0, 0   (10% slower)
  786 MHz, 800 mV                                   0, 0, 0, 0   (full speed)
  786 MHz, 850 mV                                   0, 0, 0, 0
```

Cite that sweep. It is a controlled voltage series across four DTBs, which is
not the same kind of evidence as a probe that failed to fire, and this project
has both.

Every speed number in section 1 ran with the two cores overlapped. The board
reads 800 mV, `overlap_safe()` approves, and `batch_serial()` defaults to
`!overlap_safe()`, so the cores overlap by default.

What that costs was measured in the same boot as the round of record, with
`CHARSIU_NPU_BATCH_PARALLEL=0` the only thing changed. Same binary, same
governor, same rail, same model file, boot id `929e85e2` at both ends:

```
             decode              TTFT ms
           overlapped serial     overlapped serial
  Qwen3       26.26   26.11        613     679    +10.8%
  TinyLLAMA   22.79   22.82        890    1157    +30.0%
  Phi3         7.03    7.04       2987    3896    +30.4%
  Gemma4       9.25    9.20       2222    2687    +20.9%
```

**Serialising costs nothing on decode and 11 to 30% on TTFT.** All four decode
figures move by less than 0.6%, which is inside the run-to-run spread, and two
of the four move upward. That is what the mechanism predicts once stated:
overlap puts two cores in flight at once, a decode step has one row and nothing
to overlap, and the whole gain is on the batched prompt.

So a reader on a 750 mV device tree, where `overlap_safe()` refuses, gets the
decode numbers in section 1 unchanged and a TTFT 11 to 30% worse. The decode
margin over the vendor does not depend on the overlap.

The four serial figures this project quoted before today (24.28, 20.34, 6.82,
8.68 tok/s) are all BELOW the serial decode measured here, and they are from
2026-09-04 at a different governor and before the calling thread was pinned.
They measure the cost of not pinning, which section 1b measures directly at
+25.5%, rather than the cost of serialising, which is zero. Replace them rather
than dating them.

One boot on 2026-09-11 could not fire the fault at all: 68 runs at the old
map's worst cell (phi3, chunk 24, KMAX 2048) across `default`, `onedev`,
`serial`, `parallel` and `zero`, with the affinity pin on and off. That is
consistent with the sweep, since the board is at 800 mV, and it is worth
exactly one sentence. It confirms the guard end to end on the shipped binary
and establishes nothing on its own, because a probe that does not fire and a
fault that no longer happens are the same picture.

What those arms did establish is that the affinity default does not mask the
race: 14 of 14 clean with `CHARSIU_AFFINITY=0` and 14 of 14 with the pin. A
race that stops firing because the timing moved would be masked rather than
fixed, and that is worth checking whenever a scheduling default changes.

---

## 5. Bandwidth figures, and which are quotable

```
  5.2 GB/s   the read back        measured directly    quotable
  4.7 GB/s   the activation pack  measured directly    quotable
  11.9 GB/s  what 8 threads reach measured directly    quotable
  10.58 GB/s weight rate, NPU summary                  quotable (see section 4)
  --------------------------------------------------------------------------
  11.7 GB/s  npu_prep_cost cache walk at 65536 bytes   real, but it is BUFFER
                                                       MAINTENANCE, not weights
  9.9 GB/s   NOT A BANDWIDTH. gemma4's q/k/v stage from gguf shapes. The entry
             it comes from exists to argue that such a figure is not a roof:
             gate+up reaches 16.8 in the same table, the biggest stage being
             the fastest. Quoting it cites a number derived to refute it.
```

---

## 6. Variability, and what one passage can order

A single reading cannot see a change worth less than about 25%. TinyLLAMA has
read 12.64 and 17.39 tok/s on the same build minutes apart.

One passage of 300 tokens resolves about 10% of perplexity. A sweep of AWQ's
exponent on Qwen3 came back non-monotone at that length on the evaluation
corpus while `charsiu_ppl` is deterministic, so that is the corpus's own
sampling. `tests/corpus/long2.txt` is the second opinion.

gemma4's TTFT spread was 46% at whatever governor the board had, 9.9% at
`performance`, and 4.4% pinned. Two clusters at the middle step, none at the
last. It was the same scheduling lottery.

---

## 7. Reproduction

```sh
git clone <charsiu> && cd charsiu && git checkout 4057a39 && make
# the model of record
curl -L -o models/Llama-3.2-1B-Instruct-Q4_0.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
md5sum models/Llama-3.2-1B-Instruct-Q4_0.gguf   # 48ff0243978606fdba19d899b77802fc

# quality, the CPU reference at the group the board runs
CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
  CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
  build/charsiu_ppl models/Llama-3.2-1B-Instruct-Q4_0.gguf \
                    tests/corpus/long.txt -n 300          # 34.2425

# the vendor's own weights, scored (needs the .rkllm and a Q8_0 reference)
sh tests/vendor_quality.sh 43

# the board round of record: speed, quality, the gemma4 sweep, one boot
sh tests/board_record.sh
```

`tests/corpus_fixed.sh` locks the corpora by md5 and runs inside `make test`.
`board_record.sh` stamps the model's md5, the binary's md5, the thread count,
the governor, the NPU rail, both CPU clusters' clock, and the boot id at both
ends, and refuses to be read as one round if the boot id moved.

---

## 8. What is not supported

The vendor at the frequency their published figures were taken at. Their
runtime HAS now been run here, in section 1d, but at 594 MHz on both sides with
neither scaling; section 1's right column is still a citation at maximum
frequency and the two must not be combined.

A single number for the prefill gap. Section 1d: 1.76x in our favour on
wall-clock TTFT for one string, 1.76x against us on throughput for the same
string, and 1.23x against us per token once two prompt lengths separate the
slope from the intercept. The last is the one that answers "how fast is the
prefill"; the first answers "what does a user wait for" and is a tokeniser
difference.

That the overlap fault is gone. Section 4b: it is rail-conditioned, this board
is at 800 mV, and a probe that did not fire says nothing on its own.

Anything about a second RK3576. One board.

The 112-matrix vendor rebuild, 58.76. It measures the reconstruction.

A single figure for the vendor-to-charsiu ratio. Section 2: all three rungs are
now under the f16 origin on both passages, and the six ratios run 1.42x to
2.62x. The direction is unanimous; the magnitude is an interval.

That the ladder is monotone in subset size. It is on `long.txt` and it is not
on `long2.txt`, where the 43-matrix rung is the highest of the three rather
than the lowest. This was stated as a finding and it was one passage.

Cross-machine quality comparisons made before 2026-09-11. They compared two
different files.

The size of the TTFT gap against the vendor. Section 1b-3 attributes charsiu's
own prompt time, 69 to 81% inside the NPU entry, of which the read-back is 25
to 36% of the whole and matches the fence. There is no equivalent breakdown of
theirs, and read/fence does not track the gap across models.

`CHARSIU_NPU_INT8_LAYERS` on hardware. `npu_mixed_test` shows that one open
device alternates w8a8 and w4a16 correctly, 0 of 18 dispatches wrong over eight
alternations in both directions, at K=256 and N=64. That says the per-tensor
width refactor is justified. It does not say the knob works at the scale a real
model dispatches at, because the refactor is not written.
