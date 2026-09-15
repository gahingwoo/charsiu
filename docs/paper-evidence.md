# Evidence pack, 2026-09-12

Every number this paper may quote, the protocol that produced it, and the
sentence that has to travel beside it. Section 8 is the list of things that
are NOT supported; read it before quoting anything.

```
  0   what must be said beside the numbers
  1   speed against the vendor        1a  their three columns, and which this uses
                                      1b  their runtime, MEASURED
                                      1c  the decode margin is younger than the round
                                      1d  the TTFT gap is not attributed
                                      1e  where the prompt's time actually goes
                                      1f  the pinning default is safe everywhere
  2   quality against the vendor's own int4   2a  why it is evidence, not my reconstruction
  3   quality of charsiu's own quantiser
  4   the output head                 4a  the two cores and the overlap fault
  5   bandwidth figures
  6   variability, and what one passage can order
  7   reproduction                    7a  what the PAPER has to change
  8   what is NOT supported
```

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

**Section 1b is different: their runtime has now been RUN here**, on the
same board, the same kernel and the same clock as ours, and it is an arm.
Section 1 and section 1b are two different comparisons and must not be merged
into one table: section 1 is four models at maximum frequency against a
citation, section 1b is one model at 594 MHz against a measurement. The
citation and the arm do not even agree about which side is ahead on TTFT, for
a reason section 1b gives.

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

Decode is ahead on three: +5.7%, +15.6%, +6.8%. **Gemma4 is LEVEL, not ahead**
-- its +0.2% does not clear its own 1.7% spread, and their column has no
spread at all to clear, so there is no margin there to report. This pack's own
rule is that a lead smaller than the arm's dispersion is not a lead, and
writing "ahead on all four" applies the rule to three of them.

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

**It is not uniformly the conservative choice.** Their w8a8 TTFT is slightly
FASTER than their w4a16 on three of the four, so picking w4a16 is conservative
on the column we lead and slightly generous on the column we trail. Both halves
of that belong in the sentence.

w8a8 also costs roughly twice the memory: 796 MB against 513 for Qwen3,
3767 against 1996 for Phi3. A reader choosing between them is not choosing on
speed alone, and neither column is "the vendor's number" on its own.

**`w4a16_g128` exists, and section 2's group caveat is about OUR FILE, not
about what they can do.** The `.rkllm` scored there keeps one scale per output
row; this column shows they ship a group-128 option as well. The asymmetry
section 2 states is between two particular files, not between two vendors'
capabilities.

### 1b. Their runtime, measured

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
                   charsiu 1.44x          <- r389
```

**1.44 here and 1.39 in the README are the same condition in two rounds.**
r389 read 18.56 against 12.88; r390, below, read 17.85 against 12.85 at the same
594 MHz. The README quotes r390 because r390 is the round that has a second
clock beside it and so can say the ratio moves. Two rounds apart is the size of
the disagreement, and it is why the next line exists.

**1.44x IS NOT A CONSTANT AND MUST NOT BE QUOTED AS ONE.** `board-logs/r390`
measured both arms at a second clock, everything else held:

```
                594 MHz   786 MHz   change    spread at 786
  charsiu         17.85     18.60    +4.2%    18.51..18.63
  vendor          12.85     12.73    -0.9%    12.72..12.76
  ratio           1.389     1.461    +5.2%
```

The ratio moves 5.2%, far outside either arm's spread. Report an interval, the
way section 2 does, and say at which clock each end was taken.

**And their decode does not use the NPU clock at all.** 32.4% more clock
buys them -0.9%, with a 0.3% spread, so it is not noise; ours buys +4.2%.
Whatever bounds their decode, it is not this clock.

That weakens the fairness worry about section 1. Their published figures are
at maximum CPU and NPU frequency and none of our arms are, but on DECODE the
NPU clock is nearly inert for them.

**The obvious extension is NOT supported.** Only the NPU clock was varied.
"Maximum frequency" in their header is CPU *and* NPU, and a decode that ignores
the NPU clock is one that may well be bound by the CPU, which this round did
not touch. Nothing here says their published figures would be reproduced at
594 MHz.

The two metrics swap sides: the clock is inert for their decode and worth
6.5% of their prefill, while for charsiu it is worth 4.2% of decode and
almost nothing on prefill.

**The prefill answer depends on which question is asked, and the two
readings point in OPPOSITE directions.** The vendor wraps the prompt in its own
chat template: the same string is 50 tokens to them and 16 to us.

```
  wall-clock TTFT, same string    charsiu 265 ms      vendor 466 ms   charsiu 1.76x
  prefill throughput              charsiu 61 tok/s    vendor 107 tok/s vendor 1.76x
```

Both are true. Two prompts separate the rate from the amount: a second, longer
prompt is 79 tokens to us and 113 to them.

```
               short              long
  charsiu   16 tok  265 ms     79 tok  751 ms
  vendor    50 tok  466 ms    113 tok  860 ms
```

**THIS PAIR WAS FITTED TO A LINE AND THE FIT IS WITHDRAWN.** It read "7.71
against 6.25 ms/token, fixed costs 141.6 against 153.6, so the vendor is 1.23x
faster per prompt token, which is the real gap". None of those four numbers is
supported. See 1g for the measurement that removed them: TTFT is convex in
prompt length, charsiu's 79-token point carries a token-loop step its 16-token
point does not, and the chunking changes with the length as well.

And this pair has a fault the 1g pair does not: **the two secants are taken
over different token ranges**, 16 to 79 on our side and 50 to 113 on theirs.
On a convex curve a secant over a higher range is steeper, so the two slopes
are not comparable even if each were exact.

What stands from this section is the part above it: the same string reads 1.76x
in each direction depending on which question is asked. That is a tokeniser
difference, and r393 makes it exact: **their chat template costs 33 tokens, the
same 33 at all eight prompt lengths from 27 to 852.** It is not a different
tokenisation of the body, it is one tokenisation plus a fixed wrapper.

### 1b-ii. Both curves, measured, and what the fit says

**The charsiu column here is r393 and four rounds of work have landed since.
1k-ii is the current ladder** -- 852 tokens go 8081 ms here to 5905 there. What
this section is still the source for is the VENDOR column, which has not been
re-measured since, and the 2.2% boot-to-boot drift bound that every comparison
against it has to clear.

Eight prompt lengths through each runtime, one script so the TEXT is the same,
NPU 594 MHz on both sides, rail 800 mV, CPU pinned at maximum, 3 readings a
point. Least squares `a + b*n + c*n^2` on each -- three parameters, eight
points, which is a fit rather than the arithmetic 1g threw out:

```
                 fixed a      linear b        quadratic c
    charsiu     187.9 ms   5.456 ms/tok   0.004483 ms/tok^2
    vendor       45.8 ms   6.095 ms/tok   0.000743 ms/tok^2
                   4.1x          0.90x               6.0x
```

Residuals are inside 2% of each point beyond 200 tokens on both curves; the
5-11% at the short end is the chunk stepping, which a smooth curve cannot
follow (1h).

**THE 6.0x IS NOT SUPPORTED BY THIS DATA AND NOTHING SHOULD BE BUILT ON
IT.** A fit can describe a curve to 2% and still not determine its individual
coefficients, and this one does not. Two independent readings say so:

- **The same configuration, two boots.** 1j re-ran the `CHARSIU_ATTN_NPU=0`
  ladder, which is exactly this row, and fitted `196.4 / 5.278 / 0.004974`
  against the `187.9 / 5.456 / 0.004483` here -- an **11% swing in c** between
  curves that agree to 3% point by point. Over eight points the three
  parameters trade against each other; a lower c is bought with a higher b.
- **The marginal, which does not trade them.** `(T(n2)-T(n1))/(n2-n1)` is
  `b + 2cn`, so the slope of the marginal column is `2c` and it is local.
  Taken from 200 tokens up it puts the ratio at **1.6x**; taken from 100
  tokens up, at **19x**. One vendor point -- their 135 token reading, the
  worst residual in their own curve at +5.3% -- moves it by a factor of
  twelve.

**The direction survives and the magnitude does not.** Our prefill grows
faster with length than theirs, which is a fact about the measured points and
needs no fit at all. "Six times" is an artefact.

**"BEHIND AT 852" WAS TRUE OF THIS SECTION'S BINARY AND IS NOT TRUE OF THE
SHIPPING ONE.** That is what 1k-ii measures: ahead at every rung to 602 and
LEVEL at 852, +2.1% against a 2.5% floor. The direction is unchanged -- the
margin shrinks as the prompt grows -- and where it ends up is 1k-ii's to say,
not this section's.

**This matters beyond a number.** "The quadratic term is the whole deficit
and theirs is six times smaller" organised months of work, and 1i is named
after it. What that work produced stands -- the attention arm is real and 1j
measures it -- but the TARGET it was aimed at was never determined. The
comparison that is determined is the point-by-point one in 1j, and the plan
that follows from it is stage costs, not coefficients.

**Our per-token linear rate is 12% BETTER than theirs.** The withdrawn
"the vendor is 1.23x faster per prompt token" was not merely unsupported --
**it had the sign backwards.** Two points a side, taken over two different
token ranges on a curve with a large quadratic term, produced a number that
pointed the wrong way.

**They win on the other two terms.** And only one of the two is a number:
their fixed cost is a quarter of ours, 45.8 against 187.9 ms, which the short
end of the measured ladder shows directly. Their quadratic term is SMALLER and
by how much is not determined -- see the refutation above, which this
paragraph used to contradict by restating the 6.0x as a finding. The
quadratic term is the attention scaling. Per prompt token we are ahead; what
we lose is how that cost GROWS, and a fixed cost four times theirs on top of
it.

**So "who starts a prompt faster" is a crossover, not a ratio.** With their
33 tokens included the fits cross at 248 of our tokens; measured, we are ahead
at 202 and behind at 302.

```
  matched INPUT TEXT, what a user waits for
    ch tok   ch ms    vn tok   vn ms    ratio
       102   852.0       135   931.8    charsiu 1.094x
       202  1457.0       235  1528.5    charsiu 1.049x
       302  2266.0       335  2145.0    vendor  1.056x
       602  5149.0       635  4200.5    vendor  1.226x
       852  8081.0       885  6026.8    vendor  1.341x

  matched TOKEN COUNT, what a runtime does with a token
       160  vendor 1.121x     410  vendor 1.175x     710  vendor 1.351x
       210  vendor 1.078x     510  vendor 1.216x     810  vendor 1.387x
```

**The two runtimes cannot share a boot** -- one arm binds one driver -- so
this breaks the rule against comparing across sessions. The drift was measured
rather than assumed: the same charsiu ladder, one boot apart, is **2.2% at
worst and -0.64% on average** across the eight points. Everything above is far
outside that.

The matched-token rows start at 160 because below it our curve has two points
and one of them carries a token-loop leftover; dropping it moves the 60-token
row from 1.160x to 1.225x and leaves 160 upward unchanged.

The per-point marginal ms/token is not usable -- both curves step. What the
stepping does not hide is the trend: ours more than doubles across the range,
5.5 to 11.9 ms a token, and theirs moves 24%.

Not supported by this: why their quadratic term is six times smaller. That
is the next question and nothing here touches it.

The rail differs: 800 mV for charsiu, 750 for the vendor, because their
driver sets it through `npu-supply` and its OPP table rather than taking the
device tree's floor. Both compute correctly at 594 MHz and voltage does not set
clock rate, so this is a caveat and not a confound.

Neither side scaled its NPU. Their device tree asks for devfreq over an OPP
table reaching 800 MHz; Kiln's patch pins the clock. So this section says
nothing about either side at the frequency their published figures were taken
at -- section 1g does, for the CPU half, and finds the ratio unmoved by it.

**Their runtime cannot be asked for its accuracy at all, by any route this
project can find, and section 2's reconstruction is therefore the only way to
it rather than the second best.** Three refusals, r398:

1. **Logits.** `RKLLM_INFER_GET_LOGITS` is in their API and every sequence
   length tried, 64 through 544, returns
   `meet unkown shape, op name: matmul_qk_rkllm_spilt_0`.
2. **Raw tokens.** Accuracy does not actually need logits: teacher-forced
   top-1 agreement against the f16 origin has perplexity's structure and needs
   only a token back, and `RKLLM_INPUT_TOKEN` would bypass their chat template
   so both runtimes could be asked about the same sequence. It is accepted, it
   returns a value, and **the value is token 0**. Asked for ten tokens it says
   `"!!!!!!!!!!"`. Across 65 prefixes: 7232 `meet unkown shape, op name:
   matmul_qkv_rkllm_spilt_0` lines, a token back every time, and prefixes 32
   to 62 report no error at all and still answer 0.
3. **Text prompts.** These work, and they apply a chat template we cannot
   reproduce: our token count against theirs is +34 on one string and +33 on
   another, so the two templates are not the same and a comparison through
   them compares two framings as well as two runtimes.

**Which also corrects what (1) was taken to mean.** It was recorded as "the
model refuses logits". `matmul_qk` and `matmul_qkv` are sibling ops and the
error is the same class, so the constraint was never about logits: **their
export carries a fixed set of compiled shapes** and anything off that set
fails. And it fails by returning the null result rather than an error, which
is the same shape as the all-0x80 buffer of a cancelled rocket job and the
unwritten shmem BO that reads back a uniform 128 (1b, and Igor's 2026-09-12
correction on the v12 thread).

What the attempt did leave is arm A, which had never been run either: **every
quality number in this pack before 2026-09-13 came from the CPU path**, because
`charsiu_ppl` had no aarch64 target. Built now, our q4_0 through the board's NPU
agrees with the f16 origin's top-1 on **55.0% of 420 positions**, with its own
hit rate on the text 29.5% against the origin's 37.1%.

### 1c. The decode margin is younger than the round

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

### 1d. The TTFT gap is not attributed, and the obvious explanation does not fit

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

### 1e. Where the prompt's time actually goes

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

### 1f. The pinning default is safe across every architecture

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

### 1g. What the CPU clock is worth, and what it does to the ratio

Section 1b varied the NPU clock only, and said so: their header names maximum
CPU *and* NPU, and a decode that ignores the NPU clock is a decode to suspect
the CPU of. r391 is that measurement.

It is a stronger experiment than 1b's. The NPU rate lives in the device
tree, so 1b spent a reboot per point and had to price the boot-to-boot drift it
was warned about. cpufreq is sysfs: every point below is one boot, one load,
one binary, swept up and then back down, with the frequency read back from
`scaling_cur_freq` rather than assumed from the write to `scaling_setspeed`.

A CPU-only arm runs at every point. Without it a flat measured arm and a knob
that never took effect are the same reading; it moved 2.49x.

```
  decode t/s, -n 64        1008 MHz    1416 MHz    max (2016/2208)   1008 -> max
    vendor  (NPU 786)        12.68       13.84          15.44          x1.2177
    charsiu (NPU 594)        17.65       19.51          21.51          x1.2188
    control (CPU only)       15.76       23.14          39.17          x2.485
```

**The CPU clock is worth the same to both, so the decode ratio does not move
with it**: 1.392, 1.410, 1.393 across a clock that more than doubles. That is
the opposite of 1b, where the NPU clock moved it from 1.389 to 1.461.

The two runtimes are at different NPU clocks in that table, so the ratio
VALUES are not like-for-like; what is compared across them is each one's own
elasticity, measured inside one boot and one arm.

**What the governor was actually giving them.** Sampling `scaling_cur_freq` 111
times during one vendor run, `schedutil` held the A72 cluster at 1200 MHz for
66% of samples and scored 13.27 t/s. r389 read 12.85 and r390 read 12.73 on
that governor, so every round this tree has run against their runtime measured
them at about 1200 MHz, not at the maximum their benchmark.md names. The gap to
maximum is 16.4% -- and by the line above it is the same 16% on our side, so it
does not move the ratio.

**Prefill: what the CPU is worth depends on the prompt length, and that is as
far as this goes.** The raw numbers, with no model fitted to them:

```
                          TTFT ms                 gain at max CPU
                     schedutil   max CPU
  vendor    50 tok       437.3     352.3              -19.4%
           113 tok       772.8     692.6              -10.4%
  charsiu   16 tok         267       224              -16.1%
            79 tok         754       677              -10.2%
```

Both runtimes gain more at a short prompt than at a long one, which is what
"the part of prefill the CPU touches is more fixed than per-token" means
without assuming a shape. The two runtimes' prompt lengths differ because their
tokenisers do (1b), so the rows are not to be read across.

**AND THAT IS THE WHOLE CLAIM. The earlier version of this section fitted a
line through each pair and reported a per-token rate and a fixed cost -- 5.325
against 7.730 ms/token, 171.0 against 143.3 ms fixed, theirs halving and ours
falling 24%. Every one of those numbers is withdrawn.** Three separate reasons,
each on its own sufficient:

1. **TTFT is not linear in prompt length; it is convex, and measurably so.**
   `tests/board_ttft_curve.sh`, one boot, CPU pinned at maximum, Llama-3.2-1B:
   the marginal cost per token runs 6.200 ms between 102 and 202 tokens and
   12.100 ms between 602 and 852. Attention is O(n^2) and this is what that
   looks like. An intercept extrapolated to zero from two points on a curve is
   not a fixed cost, it is an artefact of where the two points were.
2. **The two points are not the same kind of prompt.** `prefill_width()` does
   `w &= ~1`, so 79 tokens run as a batched 78 **plus one token through
   `llama_forward`**, inside the prompt timer; 16 tokens have no leftover. The
   runtime prints *"prompt batched for 78 of 79 tokens, the rest a token at a
   time"* and the round was not reading the line.
3. **Prompt length changes the CHUNKING as well as the work.** The marginal is
   not even monotone -- 8.320 ms/token between 52 and 102 against 6.200 between
   102 and 202 -- because at 102 tokens `onechunk_on()` widens the prompt into
   a single chunk of 102, and that is 5.9% slower than letting it run as
   `1x80+1x22`. See 1h.

What survives is the table above, because chunking and the leftover token do
not depend on the CPU clock: the same prompt, the same chunking, under two
governors.

The first charsiu sweep of this round was thrown away rather than published.
It read 13.85 t/s at maximum CPU where r388 had recorded 17.85 at a lower
clock, which is impossible. Reproduced against the alternative on the same
boot, the variable was `CHARSIU_NPU_MAXN`: its C default of 8192 is below every
vocabulary, so the output head silently stayed on the CPU, and it was worth 55%
of decode at maximum CPU and 83% at `schedutil`. The default is now 262144,
which is what every board round in this tree had already been setting.

### 1h. The prompt's chunk width, and a default that is right at 128 and wrong at 102

Found while measuring 1g's TTFT curve: the marginal cost per token is not
monotone in prompt length -- 8.320 ms between 52 and 102 tokens against 6.200
between 102 and 202 -- which a convex function cannot do. The cause is that the
prompt length also decides the chunking.

`charsiu_run` runs a nominal chunk of 80, and `onechunk_on()` widens the whole
prompt into ONE chunk when it fits under the model's ceiling (160 tokens here).
A 102-token prompt therefore runs as a single chunk of 102 rather than as
`1x80+1x22`. One 102-token prompt, CPU pinned at maximum, chunk width the only
variable, 3 readings a cell:

```
  chunk   widths actually run     median TTFT
     48   2x48+1x6                    809 ms
     80   1x80+1x22                   809 ms
     52   1x52+1x50                   830 ms
     64   1x64+1x38                   849 ms
    102   1x102   <- the default      857 ms
     34   3x34                        870 ms
```

Within a cell the spread is at most 6 ms; between the best and the default it
is 48 ms, so the ordering is outside the noise. **The shipped default is 5.9%
slower than chunking at 80 on this prompt length.**

It is not "fewer chunks is better" -- one chunk of 102 is nearly the worst
row and three chunks of 34 is the worst. It is not a clean multiple-of-16 rule
either: `1x64+1x38` should then beat `1x52+1x50` and it does not.

**AND THE DEFAULT IS NOT SIMPLY WRONG.** `onechunk_on()` was turned on
against measurement: TTFT fell on all four vendor-protocol models, 5.1% and
5.7% on the two whose baselines repeat. Those are 128-token prompts. So the
knob wins at 128 and loses at 102, which makes this a non-monotone response to
prompt length and not a regression to revert.

**Six cells at one prompt length cannot choose a new default.** The comment
above the knob records what happened the last time a chunk rule was derived
rather than measured -- SmolLM2 came back 77% slower. What is supported here is
that the default is not optimal at every length, which is enough to stop
quoting a single TTFT figure as if chunking were settled, and not enough to
change anything.

### 1i. Why our quadratic term is larger than theirs: attention is on the CPU

**This section was called "six times theirs" and that factor is withdrawn --
see 1b-ii.** The fitted coefficients do not determine it; the direction is
sound and the magnitude is not. Everything below is about the direction, which
is what it actually establishes.

1b-ii left one question: the prefill curves differ mostly in the `n^2` term,
0.004483 against 0.000743 ms/tok^2 as fitted, and the `n^2` term is attention. The batched
stage table answers it directly. Llama-3.2-1B, 852 tokens, CPU pinned:

```
  attention      3.82 ms a row   42.1%    <- the largest row
  gate + up      2.28            25.1%
  down           1.43            15.7%
  q k v          0.59             6.5%
  matmul rows:   4.66 ms a row inside the NPU entry
```

4.66 is exactly the four projection rows summed, so attention's 3.82 ms is
OUTSIDE the NPU entry: **every projection is on the hardware and attention is
not.** 42.1% measured against 40.2% inferred from the fit is two methods
agreeing to two points.

**There is an NPU attention path, it is correct, and it is off.**
`attn_npu_want_for()` states the reason: *"head_dim >= 128 is only half the
rule and the other half (how long a prompt) has not been measured"*. Both
halves, measured -- ratio is NPU arm over CPU arm, so below 1.00 the NPU wins:

```
  model            head_dim   ~52 tok   ~202    ~452    ~852
  Llama-3.2-1B           64     1.269   1.259   1.250   1.257
  Qwen3-0.6B            128     1.622   1.354   1.163   1.018
  gemma-3-1b            256     1.121   1.021   0.923   0.877
```

Both halves are real, and they are **one quantity**: head_dim is the inner
dimension of both attention matmuls and prompt length is the outer, so both
feed arithmetic per dispatch. head_dim 256 wins from about 230 tokens and by
12.3% at 852.

**"head_dim 64 is flat at 1.25 because no length makes a dispatch worth
taking there" is WITHDRAWN — see 1j.** It was flat because the fp16 path spent
1550 ms of its 2924 on one core doing work proportional to the answer, not
because of anything about head_dim 64. With that gone the ratio falls with
length and crosses at about 376 tokens.

**So the paper must not cite Llama-3.2-1B for this.** It is the model every
speed round in this pack uses and the furthest below the threshold of anything
on the card -- head_dim 64 against a rule about 128.
`tools/gguf_head_dim.py` prints the census: 512, 256, 128, 128, 96, 64, 64, 64.

**gemma-4-E2B is not a head_dim 512 point** and was nearly reported as one.
It has two head_dims -- `key_length` 512 and `key_length_swa` 256 -- and at the
time the pool held one, so 28 of its 35 layers never reached the hardware. Its
0.936 at 852 tokens is what seven layers bought. **That is no longer true of
the tree**: the mirror holds a head per layer, and gemma-4 now runs 385 of 385
layer calls with zero refusals (1j). The paragraph is kept because the number
above was measured under the old behaviour and must not be read as current. The instrument said "fell back on 0"
throughout, because five refusal paths touched no counter; there is one per
reason now and it names `head_dim` directly.

**Not supported on this evidence: any change of default.** Three clean
models is not a rule, `CHARSIU_ATTN_NPU=auto` was already withdrawn once, and
the last chunk rule derived rather than measured cost SmolLM2 77% (1h). What is
established here is that the vendor's smaller quadratic term has a named cause
on our side and an existing, correct, measured lever.

**The default DID change, on different evidence: 1j.** Ten models rather than
three, the arm itself 2x faster than it is here, perplexity and peak memory
measured, and a rule that leaves short prompts on the identical code path.

---

### 1j. The attention arm made the default, and half the prefill gap

1i established that the vendor's smaller quadratic term is our attention being
on the CPU, and left the lever off because three models is not a rule. This is
what the lever is worth once the arm itself was made fast, and what that did
to the curve in 1b-ii.

**The arm, first.** At 852 tokens on Llama-3.2-1B the attention stage went
5.48 -> 2.75 ms a row against a CPU arm at 4.00, across five builds. Nothing
in it is a new algorithm; it is four things that were each a stage doing work
on one core:

```
  the softmax between the two matmuls   1305 -> 466 ms   it never used the pool
  the activation pack                   1132 -> 228 ms   scalar, then pooled
  poisoning the output buffer            668 ->  24 ms   every cell, then one
                                                         sentinel a row, then
                                                         written before release
  the scores readback                    257 ->  10 ms   copied out, then
                                                         reduced in place
```

**1i's own conclusion about head_dim 64 is what this overturns**, and the
shape of that matters: r400 measured the NPU arm at 1.19 to 1.23 times the CPU
arm over an eight-fold range of prompt length, FLAT, and concluded there is no
crossover at head_dim 64. That was a fact about the code. The ratio was flat
because the fp16 path's per-element costs grew with the prompt exactly as the
CPU arm's arithmetic does.

```
  tokens        52     102     202     302     452     852   crossover
  r400        1.229   1.208     -       -     1.220   1.186   none, flat
  final       1.207   1.153   1.087   1.035   0.964   0.864    ~376
```

**The default.** `CHARSIU_ATTN_NPU=auto` decides once per prompt, before the
mirror is built, on a length the caller supplies (`llama_prefill_hint`).
Ten models, 852 token prompt, clock pinned, leading with the one that lost:

```
  Phi-3.5-mini       -0.3%      SmolLM2-135M      +12.9%
  SmolLM2-1.7B       +2.3%      tinyllama-1.1B    +17.0%
  Llama-3.2-1B Q4_0 +13.2%      Qwen2.5-1.5B      +17.5%
  Llama-3.2-1B Q8_0 +13.5%      gemma-3-1b        +19.8%
                                gemma-4-E2B       +28.5%
                                Qwen3-0.6B        +28.6%
```

Every one is text-identical to the CPU arm and none refuses a single layer --
including gemma-4, which 1i describes as refusing 28 of 35. The gain tracks
attention's SHARE of the prompt, which is why the two that do not move are the
two largest models.

**The two questions a default has to answer that speed does not.**
Perplexity on `tests/corpus/long.txt`, batched, Llama-3.2-1B Q4_0: **40.9987
off, 40.9213 on**, reproduced exactly twice. Deterministic, so that is a
numerics difference and not an improvement to claim -- but it is not a cost
either, which is the question. Peak memory 1474 -> 1459 MB on Llama, 2671 ->
2670 on gemma-4.

**Below the threshold the two arms run the same code**, not merely at the
same speed: `attn_npu_get` returns NULL before the mirror exists. That is what
makes a default defensible from one board -- the change is confined to prompts
long enough to have been measured winning. Verified with an empty environment:
202 tokens 1477 against 1483 ms, 852 tokens 7004 against 8053.

**And the curve in 1b-ii.** Both charsiu arms measured in ONE boot, so
`CHARSIU_ATTN_NPU=0` is a drift control on the vendor curve being from
another: it reproduces that boot's charsiu points to within 3% at seven of
eight lengths.

```
  charsiu tok    CPU arm (the 1b-ii config)    the default now
       102          charsiu 1.146x               charsiu 1.153x
       202          charsiu 1.033x               charsiu 1.031x
       302          vendor  1.054x               vendor  1.064x
       452          vendor  1.126x               vendor  1.087x
       602          vendor  1.225x               vendor  1.167x
       852          vendor  1.380x               vendor  1.193x
```

**The vendor's lead at 852 tokens is halved, 1.380x to 1.193x.** The
crossover does not move -- it is still between 202 and 302 tokens -- and that
is by construction, because the arm did not turn on below 448 when this was
measured. **That threshold is 272 as of r413, and a model with no GQA has a
second one at 448**; the reading above is unaffected (it is a 852 token row,
above either threshold, on a GQA model) but a re-run at 302 or 352 would not
reproduce the "by construction" clause.

**Do not quote the fitted coefficients for this.** Fitting `a + bn + cn^2`
to both arms shows the quadratic more than halving, 0.004974 -> 0.002106, and
folding in the vendor's own fit puts the crossover at 195 tokens -- *worse*
than 1b-ii's 248, from a change that made the runtime faster at every length
it touched. The control says why: the same configuration fits 0.004483 in
1b-ii and 0.004974 here, an 11% swing between curves that agree to 3% point by
point. Over an eight-point ladder the three parameters are correlated, and a
lower `c` is bought with a higher `b`. `tools/ttft_compare.py` interpolates
inside each curve and never fits across them; that is why it exists.

**THE 250 TOKEN BOUND IN THIS PARAGRAPH IS SUPERSEDED BY 1k-ii AND WAS LEFT
STANDING.** It read "Still not supported: that charsiu beats the vendor on
prefill. It does not, above about 250 tokens." That was true of the binary
this section measured. r413 re-ran the whole ladder on the SHIPPING binary,
five repeats a point, one boot, clock and governor pinned, and charsiu is
ahead at every rung to 602 tokens and level at 852 -- +24.8% at 27 tokens,
+9.6% at 202, +4.9% at 302, +3.5% at 602, +2.1% and LEVEL at 852. The
crossover this paragraph names does not exist on that binary.

What survives is the direction and the cause, not the bound: the margin
SHRINKS with prompt length, because their attention remains roughly 4.3x
cheaper than ours -- our fence alone is 880 ms against their entire quadratic
term's 539 at 852 tokens. Extended far enough that still crosses; 852 is where
it reaches level, and nothing here measures beyond it.

Two paragraphs in one document disagreeing about who leads prefill is exactly
what section 0 warns about when it says the citation and the arm do not agree
about TTFT. Read 1k-ii, which names its binary, its boot and its spread.

**AND THE EXPLANATION THAT WAS ATTACHED TO THAT WAS NOT MEASURED.** This
paragraph said 0.119 TMAC/s against the int4 path's 0.45 to 0.70, and called
it structural. 0.119 was arithmetic done on a stage table in a write-up. The
measured figures are **0.026 (scores) and 0.048 (values)** -- wrong by four
times in the direction that flatters the claim -- and the parts are nothing
like the sentence implied:

```
  dispatch   10% of attention. A task costs 20 us, not the 90 the submit
             count suggests: forcing 1x/2x/4x/8x the tasks at constant
             arithmetic gives 2.77/3.05/3.60/4.67 ms a row, a straight line.
             Merging the four GQA heads that share a surface is worth 7%.
  cores      both of them is SLOWER, 2.91 against 2.76 ms a row, twice.
  shape      6 to 7 times, and measured INSIDE fp16 with dtype, m and group
             held still: k=1024 n=1024 runs at 0.190 TMAC/s against 0.026 for
             the scores shape -- 16x the arithmetic for 2.6x the fence.
  dtype      the rest, 2 to 3.5x, BOUNDED not measured: 0.190 for fp16 against
             0.36 to 0.69 for int4, at shapes that are not the same.
```

**What that leaves open is a road, not a wall.** Attention's weights ARE the
KV cache and charsiu packs them itself, so they can be int4 with an fp16
activation -- the w4a16 arrangement the projections already run at 0.36 to
0.69. Nothing about attention requires fp16 weights.

**The first measurement of that road does not exist**: int4 at the attention
shapes. `charsiu_int4` and `charsiu_matmul` take m, k and n and print no time;
`npu_fp16_test` is fp16 only. Until that probe exists the dtype factor above is
two shapes apart, and the projection that follows from it -- attention 2343 ms
becoming about 780, TTFT about 5630 against their 6027 -- is arithmetic on a
bound and **not a result**.

### 1k-ii. The ladder re-measured on the shipping binary: ahead to 602, level at 852

r413, commit 58d2d360d971, five repeats a point, one boot (0c9923fe), clock and
governor pinned. Supersedes 1k below, which is kept because its numbers are a
dated reading of a different binary.

```
    ch tok   ch ms       range   vn tok    vn ms   margin   spread   verdict
        27     331   330..332        60    413.2   +24.8%     0.6%   ahead
        52     409   407..416        85    529.6   +29.5%     2.2%   ahead
       102     750   745..767       135    931.8   +24.2%     2.9%   ahead
       202    1395  1390..1402      235   1528.5    +9.6%     0.9%   ahead
       302    2045  2030..2050      335   2145.0    +4.9%     1.0%   ahead
       452    2991  2974..3014      485   3194.7    +6.8%     1.3%   ahead
       602    4060  4055..4072      635   4200.5    +3.5%     0.4%   ahead
       852    5905  5892..5936      885   6026.8    +2.1%     0.7%   LEVEL
```

**The bound is measured now, not quoted.** r413 ran twenty readings of ONE
arm at 302 tokens on one boot with nothing changed: median 2130, range
2110..2164, **spread 2.5%**. That is the floor a margin has to clear, and it is
slightly LARGER than the 2.2% cross-boot figure this section used to quote. All
eight rows clear their own spread, so the floor is what separates them.

**852 does not clear it** (+2.1%) and stays level. **602 is the thin row**:
+3.5% clears 2.5% by one point and a slightly stricter bound puts it back.

**What moved 302 and 452 from level to ahead** is the attention threshold
coming down from 320 to 272, re-derived against the arm that ships. 320 itself
had the defect 448 had: it was measured in r412 section 2 on the r411 binary,
and section 7 of the same round made that arm faster.

**And the null control held.** The four rows below the threshold -- 27, 52,
102, 202 -- run the SAME code in both arms, because attn_npu_get returns NULL
before the mirror is built. They moved -0.3 to -1.4%, all inside the floor. If
any of them had moved, this whole re-read would be suspect.

**This table was overclaimed once**, when every point estimate favoured
charsiu and it was written up as "completely surpassed" and withdrawn the same
day. The difference is a measured floor, one row still level and one row thin.

### 1k. SUPERSEDED BY 1k-ii: the ladder with the overlap work in, on an older binary

**SUPERSEDED BY 1k-ii above**, which re-measured this on the shipping binary
with five repeats and a measured noise floor. Kept as a dated reading.


r411, shipping defaults, board to itself, three repeats a point, one warm-up
discarded, all eight points on one boot.

```
    ch tok   ch ms      range   vn tok    vn ms   winner   ratio
        27     332   327..332       60    413.2  charsiu   1.245
        52     415   413..417       85    529.6  charsiu   1.276
       102     761   750..762      135    931.8  charsiu   1.224
       202    1400  1398..1401     235   1528.5  charsiu   1.092
       302    2131  2124..2134     335   2145.0  charsiu   1.007
       452    3059  3037..3179     485   3194.7  charsiu   1.044
       602    4115  4111..4156     635   4200.5  charsiu   1.021
       852    5953  5948..5959     885   6026.8  charsiu   1.012
```

**What this supports: two regimes.** charsiu leads by 1.09x to 1.28x below
about 250 of its own tokens, and from 302 up the two are level. 852 goes 8081
(r393) -> 7016 (r408) -> 6818 (r410) -> 6328 -> 5953 -> 5905 (r413), so their 1.341x lead
there is gone.

**WHAT IT DOES NOT SUPPORT IS "AHEAD AT EVERY LENGTH", and this list said
that for an hour before it was corrected.** Every point estimate favours
charsiu, but a margin has to clear TWO things, not one:

  - **the cross-boot drift.** Their column needs their driver bound, so the two
    columns cannot share a boot; 1b-ii measured the boot-to-boot drift of this
    same charsiu ladder at **2.2% at worst**. 302 (+0.7%), 602 (+2.1%) and 852
    (+1.2%) are inside it.
  - **the arm's own spread AT THAT POINT.** 452 is +4.4% over them and its own
    three readings span 3037..3179, which is 142 ms and **4.6% of its median**.
    A margin smaller than the spread of the arm it came from is not a margin,
    and comparing only against the drift bound missed this one.

```
    clear    27 (+24.5%)  52 (+27.6%)  102 (+22.4%)  202 (+9.2%)
    level   302  452  602  852
```

The four changes behind it, each measured against its own control on this
board and each text identical to it: two head groups so the softmax runs
during the scores fence (322 ms of layer at 852), the kv ladder's ceiling and
its block copy (126), the deferred int4 accumulator gather (326), and the
fence poll (76). `tests/board_text_all.sh` is 9 models 0 differing with the
shipping defaults.

Decode did not pay for it: 18.15 tok/s against 18.06 before the round, peak
1448 MB against 1449.

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

**What is robust is the direction: in every cell the vendor's four-bit excess
is larger than charsiu's.** The size of it is an interval, 1.4x to 2.6x, and
the paper should quote the interval and nothing narrower.

**SIX CELLS ARE NOT SIX INDEPENDENT SAMPLES AND MUST NOT BE WRITTEN AS
STATISTICAL STRENGTH.** The three subsets are NESTED: the 43 matrices are
inside the 91, which are inside the 105, so the rungs share most of their
weights and cannot disagree freely. The two passages are independent of each
other; the three rungs are not. What the table supports is "the direction did
not reverse under either axis we varied", which is a robustness check, not six
trials.

**The ladder is NOT monotone and the earlier text saying so was one
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

`rho1` and `vendor43` are the SAME FILE, byte for byte, md5
`9d8e82952e96a6f14eeaa4b016704dde` (a BUILD PRODUCT, not an input, so it is
deliberately not in section 7's list). That is the point of the rho = 1 subset
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

### 2a. What makes it evidence rather than a reconstruction of mine

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

### The weight group, and the five models that do not get one

A tensor is grouped only when the group width divides its K and is strictly
narrower than it; npuquant falls back to one scale a row otherwise, and
`llama_auto_kmax` pins the width at 1024 and only ever considers WIDENING.
Counted with `CHARSIU_NPU_VERBOSE` on 2026-09-15:

```
  model             distinct K                       widest width dividing all
  Llama-3.2-1B      2048 8192                        1024   grouped as shipped
  Phi-3.5-mini      3072 8192                        1024   grouped as shipped
  SmolLM2-1.7B      2048 8192                        1024   grouped as shipped
  Qwen3-0.6B        1024 2048 3072                    512
  tinyllama-1.1b    2048 5632                         512
  Qwen2.5-1.5B      1536 8960                         256
  gemma-3-1b        1024 1152 6912                    128
  gemma-4-E2B       256 1536 2048 4096 6144 12288     128
```

Three of eight are fully grouped at the shipping width. The rest carry tensors
on the per-row path, and gemma-3-1b carries all of them there.

**What that costs gemma-3-1b, and what a narrower group returns.** Quality on
the desk over 511 scored positions of `tests/corpus/long.txt`; decode on the
board, boot afe55e04, 594 MHz, performance governor, three readings an arm,
medians, spreads under 0.2%.

```
  group  1152->      6912->     decode t/s        ppl
  1024   ungrouped   ungrouped   20.10         108.7063   ships
   576   2 groups    12 groups   20.70  +3.0%   79.2824   -27.1%
   384   3 groups    18 groups   18.69  -7.0%   65.4765   -39.8%
   192   6 groups    36 groups   14.68 -27.0%   57.1298   -47.4%
   128   9 groups    54 groups    -             47.0269   -56.7%
  the file's own q4_0 on the CPU, no charsiu requantisation:  49.0532
```

576 is not a trade. It is faster AND better than the shipping default, on a
model the shipping default groups nowhere.

**At 128, charsiu's four bits pass the gguf's own q4_0**: 47.0269 against
49.0532. q4_0 is blocks of 32 with an fp16 scale, finer than anything in that
table, and charsiu at 128 with an fp32 scale and the wsum correction still
scores better. **One model, one corpus, one length. It is not a general claim**
and the three sentences this pack requires of every quality figure apply to it
unchanged.

**It does not generalise as a constant.** The same 1024 against 512 costs
tinyllama 11.7% of decode and Qwen3-0.6B 5.5%, and both already have grouped
tensors at 1024. gemma-3-1b is the only one of the three with nothing to get
out of, and the only one that gains. What this asks for is a per model
decision, which is what `llama_auto_kmax` already is; it only ever looks
upward.

### Looking downward, tried on all eight, and why the default stays up

`CHARSIU_NPU_W4_GROUP_FIT=1`, default off, gives each tensor the widest PROPER
divisor of its own K that is no wider than the width asked for. It is the
narrowest form of "look downward": the requested width is still the ceiling,
and nothing changes for a tensor the ceiling already divides. Quality on the
desk, 511 scored positions of `tests/corpus/long.txt`; decode on the board,
boot afe55e04, 594 MHz, boot entry 1, performance governor, three readings an
arm, medians, binary 95e72869ba97.

```
  model            ppl FIT=0   FIT=1       ppl     decode 0   1     decode
  Qwen2.5-1.5B      37.6358   27.1185   -27.9%      14.96  15.29    +2.2%
  gemma-3-1b       108.7063   82.4102   -24.2%      21.61  21.71    +0.5%
  tinyllama-1.1b    29.9309   28.2998    -5.4%      22.87  22.80    -0.3%
  Llama-3.2-1B      41.2763   41.2763     0.0%      21.52  21.43    -0.4%
  Phi-3.5-mini      17.8023   17.8023     0.0%       7.25   7.22    -0.4%
  SmolLM2-1.7B      31.1293   31.1293     0.0%      15.30  15.24    -0.4%
  Qwen3-0.6B        87.8019   89.2738    +1.7%      30.64  29.22    -4.6%
  gemma-4-E2B       89.0370   99.8674   +12.2%       9.51   9.88    +3.9%
```

**Two models are strictly better on both axes**, and a third is nearly free:
Qwen2.5-1.5B gains 27.9% of perplexity and 2.2% of decode, gemma-3-1b 24.2%
and 0.5%, tinyllama 5.4% of quality for 0.3% of decode.

**Three are unchanged to the digit, which is the property that makes it
safe to leave on.** Llama, Phi-3.5 and SmolLM2-1.7B read the same perplexity
with the flag on as off, because their K values already have a proper divisor
at the requested width and the fit picks the same number. It is a no-op where
it is not needed, ON as well as off.

**Their -0.4% of decode is the flag's own cost, and it is not nothing.**
Nothing about those three models' weights changes, so that column is the
doubled slot capacity the flag allocates, measured on models where it buys
nothing.

**And two models get worse from a strictly finer quantisation.** Qwen3-0.6B's
only change is its 1024-wide tensors going from one scale a row to two groups
of 512; its 2048 and 3072 tensors are untouched and read the same rms either
way. gemma-4-E2B's are 1536 to three groups of 512 and 256 to two of 128.
Nothing else moves in either model, and both lose.

That is not float rounding from a different slice count. These are desk
numbers, where `CHARSIU_NPU=0` means there are no slices at all: `npu_matvec`
walks the groups and accumulates in double. The difference is the
quantisation itself. **Both signs hold on the second corpus**, which is what
this pack requires of a result it does not like. On `tests/corpus/long2.txt`,
same arms, same lengths:

```
  Qwen3-0.6B     long   87.8019 ->  89.2738   +1.7%
                 long2 120.4219 -> 124.7481   +3.6%
  Qwen2.5-1.5B   long   37.6358 ->  27.1185  -27.9%
                 long2  51.3478 ->  37.4671  -27.0%
```

Qwen2.5's gain reproduces to within a point of its own size; Qwen3's loss
reproduces in sign and grows. Neither is one passage.

**The weight error falls on every model and the perplexity goes both ways.**
`CHARSIU_NPU_RMS=1` makes the quantiser report each tensor's reconstruction
error:

```
  model          mean rms FIT=0   FIT=1             ppl
  gemma-3-1b        14.2882%   12.4938%   -12.6%   -24.2%   better
  gemma-4-E2B       14.2980%   13.1508%    -8.0%   +12.2%   worse
  Qwen3-0.6B        13.6892%   12.0744%   -11.8%    +1.7%   worse
```

It has to fall: a finer partition of the same row cannot have a worse absmax.
The reductions are all the same size, 8 to 12.6%, while the quality outcome
differs in SIGN. **So the Frobenius error cannot predict the direction of the
quality change, let alone its size**, and no ranking built on it can order
these models. This pack already has that statement from the other side, in the
reconstruction that scored ppl 1701 at 18.3% weight error where Gaussian noise
of the same magnitude scored 32.10. This is the same fact read backwards: a
genuine reduction in weight error, and two models of eight pay for it.

**What that leaves `llama_auto_kmax`.** A downward rule would need a
predictor, evaluated before any weights are quantised, for which side of zero
a model lands on. The candidates measured here do not supply one: it is not
the group count (gemma-3-1b's odd rungs are among its best), not the widths
involved (Qwen3 and Qwen2.5 both go 1024-wide tensors to 512 and move in
opposite directions), and not the weight error, which moves the same way for
everyone. Choosing per model is possible and costs one perplexity run per
model per candidate width; choosing per model **without running the model** is
not something this data supports.

So the flag ships default off. Two of eight models get worse, and a default
that changes output under a shipped model belongs to whoever ships it.

**And the trade cannot be engineered away.** A narrow group forces a narrow
slice, because one dispatch cannot cover K wider than one group: the hardware
returns one accumulator per output channel per slice (`fo[j]`,
`src/npudev.c:3818`) and the CPU multiplies it by that slice's single scale
afterwards, so the whole slice K is summed before any scale exists to apply.
The NPU's own per-channel multiplier is per OUTPUT channel, not per K range.
So the cost of a narrower group is exactly the cost of more slices, in every
version of this.

**The shipping arm IS grouping, and that was checked rather than assumed.** A
group width of 99991 divides nothing, so nothing groups, and Llama-3.2-1B reads
50.7838 there against 41.2763 at the shipping 1024.

**And it does not survive a different length either.** Every number in that
table is `-n 300`, which scores 299 positions. `charsiu_ppl` with no `-n`
scores 511, and on the same file, the same corpus and the same grouped arm that
reads 33.8071 at 300 it reads **41.2763** at 511. That is near enough to the
ungrouped 41.5289 above to be mistaken for it, and it was, for about an hour on
2026-09-15: a run of the board regression came back 41.2777, which looked like
the grouped arm having silently stopped grouping. It had not. The tool prints
the count on every line it emits, so the tell is always on screen -- "over 299
scored positions of 300 tokens" against "over 511 scored positions of 512" --
but two numbers that happen to land four tenths of a percent apart will not
announce that they are different measurements. A perplexity belongs to the
model file, the corpus file AND the length.

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

### 4a. The two cores, the overlap fault, and what the speed numbers ran under

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
They measure the cost of not pinning, which section 1c measures directly at
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

### 5a. The controller is not the binding constraint at decode

Both loads were run against each other on 2026-09-15, boot 8934c5ee, 594 MHz,
performance governor: `charsiu_membw` reading a 256 MB buffer for 8 s inside
the 12.3 s generation window of a 256 token Llama-3.2-1B decode.

```
                            readers alone   alongside a decode   decode t/s
  1 reader thread              8.63 GB/s          8.64             20.84
  8 reader threads            11.93 GB/s         11.92             20.82
  decode alone                     --               --             20.79
```

**Neither side loses anything.** Eight cores reading DRAM flat out move decode
by 0.1%, and the decode moves them by 0.1%. So a second engine taking part of
the work is not zero sum for want of bandwidth, which is what the tool was
written to find out.

That is consistent with the per-call floor being 38% of a token: the NPU pulls
its weights in bursts and waits between them, so its average demand is well
under the peak even though the total bytes a token are large. **It does not
show that decode is compute bound, and it does not measure the GPU.** The
readers here are an independent load, not a second engine taking part of the
same tensor, and they were not placed on the runtime's own cores.

**And the thread count belongs to the figure.** The reader sweep is not
monotone: 1 thread 8.63, 2 threads 8.09, 3 threads 7.68, 4 threads 7.54, 6
threads 10.59, 8 threads 11.92. Four threads are worse than one. That is the
A72 and A53 clusters, the same split that makes decode bimodal, so "the CPU
reaches X GB/s" is not a quantity without a thread count beside it.

### 5b. What one dispatch costs, and int4 against int8 measured rather than halved

`npu_fence_scan` holds k, m and the buffers still and moves only the output
width. Buffers are allocated once at the widest point, so no row pays for an
allocation. Least squares on all eight widths of each row, m=1, 20 repeats a
point, same boot and clock as 5a.

```
    k      int8 us/n    int4 us/n    ratio    int8 GB/s    int4 GB/s
    1024     0.1167       0.0595     0.51        8.83         8.32
    2048     0.2300       0.1168     0.51        8.94         8.84
    4096     0.4370       0.2224     0.51        9.40         9.24
```

**The slope doubles when k doubles**, over a 16x range: the int8 per-channel
cost runs 0.0328, 0.0618, 0.1167, 0.2300, 0.4370 for k of 256 to 4096, which
is x1.89, x1.89, x1.97, x1.90. So a dispatch at m=1 costs its weight BYTES,
k*n of them for int8 and half that for int4, at 8.3 to 9.4 GB/s either way.

**The int4 halving used to be an inference.** Every earlier sweep in this tree
dispatched int8 and the w4a16 cost model was reached by halving the weight
bytes on paper. Measured at three k, the ratio is 0.51 three times.

**That rate is the single-stream memory roof.** Section 5a gives one CPU core
8.63 GB/s of DRAM on the same boot, and a dispatch reads its weights at 8.3 to
9.4. So at m=1 what is left to win is bytes, not calls.

**The intercept is not quoted and should not be.** The fits put it between 13
and 104 us, and the same k measured twice in one round gave 71.4 and 52.0. The
line describes the wide end to 6.5% at k=4096 and to 31% at k=256. It
describes without determining its constant.

**The m axis, on the same surface.** At k=1024 int8, the per-channel cost runs
0.111 at m=1, 0.118 at m=8, 0.122 at m=20, 0.131 at m=40 and 0.202 at m=80:
eighty times the rows for 1.8 times the cost, because the weights are read
once for the whole batch. That is 0.0025 us a channel a row at m=80 against
0.111 at m=1, a factor of 44, and it is why prefill batches and decode cannot.
The slope is flat to m=40 and then jumps by half, so the widest chunk is not
free either.

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
git clone <charsiu> && cd charsiu && git checkout d97e212 && make
# the model of record
curl -L -o models/Llama-3.2-1B-Instruct-Q4_0.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
md5sum models/Llama-3.2-1B-Instruct-Q4_0.gguf   # 48ff0243978606fdba19d899b77802fc

# the corpora. Every perplexity in this pack names one of these two, and the
# checker rule is that a perplexity must name a file whose md5 is HERE.
md5sum tests/corpus/long.txt tests/corpus/long2.txt
#   4237c8fc3163a359fc21bde60c7b1d8b  long.txt
#   9c1f92b423e3ab3d3f9279a8d70a4ae6  long2.txt
# and the f16 original, which the vendor-quality round of record starts from
#   3ba43423d342673e26016ffe85268937  Llama-3.2-1B-Instruct-f16.gguf
# and the vendor's file, which section 1b times and section 2 scores
#   2d3962468e2e7c0d0571157f8c9eae71  Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm

# quality, the CPU reference at the group the board runs.
#
# CHECK THE md5 FIRST AND DO NOT SKIP IT. This number belongs to the file,
# not to the path. In a fresh clone the curl above puts 48ff0243 at that path
# and it reads 34.2425. In THIS project's working tree the same path is a
# SYMLINK into rootfs-overlay, which is the older c82c0340 the board image
# carries, and the identical command reads 33.8071 -- a 1.3% difference with
# nothing on screen to say the input changed. Both files are in the table in
# section 3; neither number is wrong and only one of them answers this command.
md5sum models/Llama-3.2-1B-Instruct-Q4_0.gguf   # must be 48ff0243...
CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
  CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
  build/charsiu_ppl models/Llama-3.2-1B-Instruct-Q4_0.gguf \
                    tests/corpus/long.txt -n 300
#   48ff0243  ->  34.2425     the model of record, what a reader gets
#   c82c0340  ->  33.8071     the board image's older copy, this tree's symlink

# the vendor's own weights, scored. CHARSIU_RKLLM_REF IS NOT OPTIONAL: the
# default reference is Q8_0, and the round of record is from the f16 original,
# because the vendor quantised the ORIGINAL weights. Without it every
# percentage in section 2 shifts. Each rung scores both passages from one
# build, so this is the whole ladder:
CHARSIU_RKLLM_REF=$PWD/models/Llama-3.2-1B-Instruct-f16.gguf \
  sh tests/vendor_quality.sh 43     # and L3, and No1

# the board round of record: speed, quality, the gemma4 sweep, one boot
sh tests/board_record.sh
```

`tests/corpus_fixed.sh` locks the corpora by md5 and runs inside `make test`.
`board_record.sh` stamps the model's md5, the binary's md5, the thread count,
the governor, the NPU rail, both CPU clusters' clock, and the boot id at both
ends, and refuses to be read as one round if the boot id moved.

---

## 7a. What the paper has to change because of these rounds

Not evidence; a list of places the prose is now wrong, kept here because the
`.tex` is not in this repository and nothing else tracks it.

**Section 2 lost its headline number.** It was "2.0 to 2.4x, monotone across
three nested subsets". It is now a direction with an interval, 1.4x to 2.6x.
Every place that writes 2.0-2.4 has to move: abstract, introduction, results.

**And "monotone" was load-bearing, not decorative.** It was one of the
arguments that the ladder measured a real difference between quantisers rather
than an artefact of the reconstruction: damage that grows with the subset looks
like a property of the weights. That argument needs a different support now.
What survives in its place is section 2a: the layout is fitted on blocks 0 to
47 and scored on rows 768 to 2047, 99.72% of codes away from a rounding
boundary, residuals of plus or minus one.

**Sections 1 and 1b disagree and both belong in the paper.** Section 1 is four
models at maximum frequency against a published point estimate; 1b is one model
at 594 MHz against a measurement of their runtime. They do not agree on which
side leads TTFT, for the tokeniser reason 1b gives. They must not be merged
into one table.

**The ordering argument for putting quality first no longer holds alone.**
The vendor-quality comparison led the results because it was the only one where
both sides ran here. Section 1b is now also that. If quality still leads it
needs a different reason -- the obvious one being that it is the comparison
nothing in the literature has, while a speed comparison against a vendor
runtime is ordinary.

**Anything saying their runtime has never been run here.** It has, in 1b.

**Section 2 can be stated more strongly than it has been.** It has read as a
fallback -- "we could not ask their runtime, so we reconstructed". r398 tried
all three routes into their runtime and all three refuse, one of them by
returning the null result without an error. So the reconstruction is not a
second best; it is the only route, and the paper can say why in four lines
instead of apologising for it in one.

**The fairness caveat about frequency can be narrowed, and one half of it
reversed.** The paper has been carrying "their published figures are at maximum
CPU and NPU and our arms are not" as an open concession. Section 1g measures
it: on decode the two runtimes have the same CPU elasticity to three digits, so
their condition moves both columns together and leaves the ratio alone. The
concession survives for prefill, where both runtimes gain more from the CPU at
a short prompt than at a long one, so any TTFT claim has to name the governor
AND the prompt length it was measured under.

**The paper's prefill claim has to be rebuilt from 1b-ii, and its sign
changes.** The 1.23x came from fitting a line through two prompt lengths a
side; the curves say our per-token linear rate is 12% BETTER than theirs, and
that we lose on the fixed cost (4x measured directly at the short end) and on
the attention scaling. The "(6x)" that stood here is the same withdrawn
factor 1b-ii refutes two pages earlier -- the fit does not determine it -- and
this sentence was restating it as a finding. Anywhere the prose says the
vendor's prefill is faster per token, or that the two fixed costs are within
8%, is wrong rather than unsupported.

**And there is a new claim worth making that the old framing could not
reach**: "who starts a prompt faster" has a measured answer that depends on the
prompt, not a ratio.

The shape of that answer has since moved and this paragraph used to give the
old one. When it was written the crossover sat at about 248 of our tokens with
the vendor ahead above it.

**1k-ii IS THE CURRENT LADDER, NOT 1k**, and this paragraph pointed at the
superseded one. On the shipping binary charsiu is ahead at every rung from 27
to 602 tokens -- +24.8%, +29.5%, +24.2%, +9.6%, +4.9%, +6.8%, +3.5% -- and
LEVEL at 852. So what depends on the prompt is the size of our lead and not
who has it, which is what this paragraph said; the numbers it said it with
were a ladder ago.

## 8. What is not supported

**"39% of a decode token is dispatch." WITHDRAWN r412, and it was never a
token share.** It is `(fix + tsk) / busy_us` = (11.5 + 7.4) / 48.0, a share of
the decode HARDWARE PATH; against the 58.4 ms token in the same paragraph the
same numerator is 32%. Four things are wrong with quoting it:

- the five stage times it was fitted from exist in **no board log in either
  repository**. There is no round record for 2026-08-31, so the run behind it
  cannot be reproduced or checked;
- its per task coefficient, 36.8 us, was refuted eight days later by round 152,
  which measured a job directly at 16.85 us + 4.81 us a task on a matmul with
  no arithmetic in it. At 4.81 the share is **26%**;
- the same fit was hand computed twice in one file on one day, from different
  assumed call and task counts, and written down as 39 in one place and 40 in
  another;
- the counter that replaced it reads **14.5% and 15.0%** on gemma4 and gemma3.
  The per call term fell from 128.7 us to 43-51 across the qos hold, the v11
  tree and the two rocket patches, so even the corrected 26% describes a
  runtime that no longer exists.

**Do not confuse it with a different number that is NOT withdrawn**: `130 us
x 150 calls = 19.5 ms of a 51.7 ms token, 38%` is Qwen3, 2026-09-02, and its
denominator is a real wall clock token. That one is dated rather than wrong,
and its per call term has also moved.

**And the ceiling on removing the IOMMU half of it is 100%, measured, with
no kernel change.** r413 §7 reasoned the other way -- an IOMMU domain belongs to
an open DRM file, charsiu opens accel0 twice, rocket gives each file one
scheduler entity spanning all cores, so "either domain can land on either core"
and attach-once would degenerate into detach plus attach on nearly every job.
The premise is right and the conclusion does not follow: `drm_sched_pick_best`
takes the first strict minimum, so a tie goes to core 0; `rocket_job_push` arms
and pushes under one per-device mutex, so two sequential submits are ordered;
and charsiu submits every device before waiting on any. Round 414 counts the
consequence in userspace, in `account_call`, where a device with no slices
contributes no megabytes:

```
  model          calls    both cores   dev 0 only   dev 1 only   core 0 flips
  gemma-4-E2B    13572        11332         2240            0        0 (0.00%)
  gemma-3-1b      6724         5060         1664            0        0 (0.00%)
  tinyllama       5700         5700            0            0        0 (0.00%)
```

No call in 26,000 sent device 1 alone, because `deal_pick` breaks its tie to
device 0 and so the first slice of any call always lands there. Core 0 therefore
always carries domain 0 and core 1 always carries domain 1, and neither ever
changes. Every per-job attach and detach in the run is removable.

**This is a MODEL of the scheduler, not a reading from it.** It is arithmetic
over charsiu's submit shape plus `pick_best`'s tie rule, and the both-devices
case is a race rather than a guarantee: if device 0's job retires between the
two ioctls, file 1 ties onto core 0 as well. Core 0's figure is a lower bound.
What it settles is r413's claim that the saving was not available -- it is.

**The withdrawn fit is no longer executing, as of round 414.**
`DEAL_US_TASK` in npudev.c decides which core every slice lands on, and it held
the refuted 36.8 for eight days because there was no way to try the other value
without rebuilding, and two binaries is the one thing a paired arm must not be.
`CHARSIU_NPU_DEAL_US_TASK` made it one environment variable. One binary, one
boot, four alternating pairs, performance governor:

```
  model          36.8       4.81      margin   balance 36.8 -> 4.81
  gemma-4-E2B    9.58 t/s   9.55 t/s  -0.31%   1.03x -> 1.02x
  gemma-3-1b    21.68      21.66      -0.09%   1.08x -> 1.08x
  Qwen3-0.6B    30.62      30.70      +0.26%   1.06x -> 1.06x
```

Every margin is inside its own arm's spread and the text is identical across
both arms and all four repeats. The per task term does not decide this deal on
these shapes; the megabyte term does. The default is now the measured 4.81,
which is what `npu_job_cost` and `charsiu_shapes` have used since round 155.

**And 4.81 was measured at 64 x 32, which section 5b says cannot order
anything.** `npu_job_cost`'s task table at that shape reads 70.06 us for one
task, 67.49 for four and 222.76 for sixteen, so a straight line through it has
no single slope to give. Re-run at k=1024 n=1024 the same table is FLAT in us
a task -- 137.51 for one, 131.39 for four, 124.01 for sixteen, and a second
pass moves in the other direction -- because at m=1 a task costs what its
weights cost to read. So the per task coefficient is not a per task cost at a
shape the runtime actually submits. The board A/B above is what makes it
harmless here: the term does not decide this deal either way.

**Three models, not nine.** That is what the table says and all it says.

The vendor at the frequency their published figures were taken at. Their
runtime HAS now been run here -- 1b at 594 MHz on both sides, 1g across the
whole CPU range including their published maximum -- but section 1's right
column is still a citation, with no N and no spread, and the two must not be
combined. What 1g removes is the WORRY that their condition would change the
comparison; it does not turn their published row into a measurement.

A single number for the decode ratio against their runtime. Section 1b: 1.389
at 594 MHz and 1.461 at 786, a 5.2% move against arm spreads of 0.3 and 0.6%.
It is an interval, and each end has a clock attached.

Their published figures reproduced at any clock measured here. Both clocks have
now been varied -- the NPU in 1b, the CPU in 1g -- and at maximum CPU with the
NPU at 786 MHz their decode reads 15.44 t/s, still short of the published
figure for this model class. Neither section claims to reproduce their table;
what 1g supports is that moving to their condition does not change the RATIO,
because it moves both columns by the same 22%.

ANY single number for the prefill gap, and in particular the 1.23x this list
used to give -- 1b-ii now has both curves and the answer is a CROSSOVER at
about 248 of our tokens, with our per-token linear rate 12% BETTER than theirs
and their fixed cost 4x smaller. Their quadratic term is smaller too and the
6x that used to stand here is withdrawn: the fit does not determine it. The 1.23x was two
points a side over two different token ranges and it pointed the wrong way.
What cannot be quoted as one number: the matched-text ratio changes sign with
prompt length, and the matched-token ratio runs 1.08x to 1.39x over the
measured span.

That the overlap fault is gone. Section 4a: it is rail-conditioned, this board
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

The size of the TTFT gap against the vendor. Section 1e attributes charsiu's
own prompt time, 69 to 81% inside the NPU entry, of which the read-back is 25
to 36% of the whole and matches the fence. There is no equivalent breakdown of
theirs, and read/fence does not track the gap across models.

`CHARSIU_NPU_INT8_LAYERS` on hardware. `npu_mixed_test` shows that one open
device alternates w8a8 and w4a16 correctly, 0 of 18 dispatches wrong over eight
alternations in both directions, at K=256 and N=64. That says the per-tensor
width refactor is justified, and the refactor IS written: `w4_for(g, t)` is
threaded through all 42 sites that ask, and the bmap cache key carries the
width. What is not written is the one line that makes it answer differently.
`w4_for()` still ignores its tensor and returns the device's width, which is
why the tree is bit identical. The version that did answer per tensor was
reverted on 2026-09-11 after nine of nine models disagreed with their own token
loop, so the knob's behaviour on a real model is not just unmeasured, it is
known to have been wrong once.
