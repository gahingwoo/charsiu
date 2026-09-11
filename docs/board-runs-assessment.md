# Review of "Board runs that would make the evidence hold", with the results

2026-09-11. A desk review of the checklist, plus the board rounds it asked for,
on one ROCK 4D at the `performance` governor. charsiu `dev 4057a39`, all
pushed.

---

# R1: the margin was made by a statistic, and then it was really fixed

## Step one: the margin did not exist

The first board round that printed every reading instead of only the best:

```
  model         decode readings (time order)                    best    median   theirs
  Qwen3      20.76 24.25 20.61 20.27 26.06 20.81 20.54          26.06   20.76   24.85
  TinyLLAMA  19.11 19.41 22.83 21.94 19.24 19.15 20.08          22.83   19.41   19.71
  Phi3        7.04  5.96  5.98  5.96  5.96  5.96  7.04           7.04    5.96    6.58
  Gemma4      7.25  7.34  7.28  9.33  7.28  7.32  7.28           9.33    7.28    9.23
```

All four models split 5 low and 2 high. Phi3 is the cleanest: `5.96` five times
and `7.04` twice, the same two values exactly, with a median gap of 0.0.

```
  best of 7       Qwen3 +4.9%    TinyLLAMA +15.8%
  single reading  Qwen3 +6.0%    TinyLLAMA +15.1%
  on record       Qwen3 +5.8%    TinyLLAMA +16.1%
  ------------------------------------------------
  median of 7     Qwen3 -16.5%   TinyLLAMA  -1.5%
```

Three different ways of taking one number out of a bimodal sample all land on
the high mode, and all reproduce the published figures.

## Step two: what switches is core placement

`board_bimodal.sh` runs `board_vendor.sh`'s invocation verbatim (`-n 64 -t 4`
plus `MAXN` and `COEF_ELEMS` and the protocol prompt), moves affinity only,
alternates four arms, and discards the cold first pass.

The first version of that probe changed the prompt, the `-n`, and two
environment variables, three things at once, and the levels it reported did not
match the observation. An arm meant to explain an observation has to start from
that observation's configuration.

```
  default     20.92 20.59 20.53 26.35 20.95 20.41 25.91 20.67 26.37   two clusters  28.5%
  all eight   20.65 20.11 20.94 20.55 26.31 20.42 20.39 20.54 26.40   two clusters  30.6%
  big four    26.20 26.53 26.55 26.60 26.18 26.40 26.52 26.33 26.49                  1.6%
  little4     18.35 18.22 18.16 18.26 18.33 18.27 18.29 18.33 18.34                  1.0%
```

The control passed first, in that `default` reproduced the bimodality, and then
three facts arrived together. The high mode of `default` is the big-four level,
the same population rather than a near one. Pinning to the big four removes the
bimodality at the fast value. And `taskset -c 0-7` behaves exactly like no
taskset, so the question is not which cores are permitted but where the
scheduler puts four threads given eight and no instruction.

## Step three: the code change, verified on the board

`a58086c` pins the calling thread to the fastest cluster by default. The cluster
is read from `cpuinfo_max_freq`, intersected with the inherited mask, and a
homogeneous machine or a kernel without cpufreq gets nothing done to it.

```
                    before                            after
  default    20.92 20.59 20.53 26.35 ... 26.37  |  26.07 26.39 25.89 26.46 25.86 26.32 26.25
  all eight  20.65 20.11 20.94 20.55 ... 26.40  |  26.62 25.62 26.36 26.24 26.10 26.49 26.35
  big four   26.20 26.53 26.55 26.60 ... 26.49  |  26.16 26.06 26.48 26.67 26.49 26.43 26.51
  little4    18.35 18.22 18.16 18.26 ... 18.34  |  18.26 18.32 18.31 18.18 18.30 18.32 18.26

  default median   20.92 -> 26.25    +25.5%
  spread           28.5% ->   2.3%
  against 24.85   -15.8% -> +5.6%
```

Three checks passed: the tell fired (`this thread on the fastest cluster, 4
CPUs`), the text was identical, and the arms took the shape the prediction had
been written down for.

`little4` does not move from 18.3, and that is the intersection working.
`taskset -c 0-3` confines the process to the slow cluster, charsiu looks for the
fastest cluster within the inherited mask, finds four equal CPUs, and declines.
An operator who has already answered this question is not overruled by the
runtime.

## Step four: all four models re-measured, and the published figures were right

One boot, median of seven, every reading printed. The quality table reproduces
the previous round bit for bit (34.2425 / 23.6746 / 18.3604), which is a free
regression check that pinning changes scheduling and not arithmetic.

```
             old median  old best   new median  spread   gain    vs vendor
  Qwen3         20.76      26.06      26.26      1.9%   +26.5%     +5.7%
  TinyLLAMA     19.41      22.83      22.79      1.2%   +17.4%    +15.6%
  Phi3           5.96       7.04       7.03      0.1%   +18.0%     +6.8%
  Gemma4         7.28       9.33       9.25      1.7%   +27.1%     +0.2%
```

All four are ahead on the median, with spreads of 0.1 to 1.9%. Phi3's seven
readings are `7.03 7.04 7.03 7.03 7.04 7.04 7.03`, a tenth of a percent.

### The new medians are the old best-of-seven, to within 1%

```
  Qwen3     26.06 -> 26.26   +0.8%
  TinyLLAMA 22.83 -> 22.79   -0.2%
  Phi3       7.04 ->  7.03   -0.1%
  Gemma4     9.33 ->  9.25   -0.9%
```

So the published numbers were right and the method for getting them was not.
The high mode was the machine's real capability all along, and best-of-N was
reporting something the runtime could reach but would not deliver reliably.

The finding is therefore not that the claim was inflated. It is that the claim
was unreproducible, and the fix was to make the runtime do reliably what it had
been doing by luck.

Against the +5.8% and +16.1% on record, the medians now give +5.7% and +15.6%.
The headline survives, from a statistic that can be defended.

TTFT did not move, and should not have. Qwen3 went 607 to 613, Phi3 2923 to
2987, gemma4 2269 to 2222, a couple of percent either way. A prompt's work is
on the pool, which still has the whole machine; decode is the one thread that
was losing the lottery. That is the shape of the 09-06 table exactly.

The vendor column is still a citation: benchmark.md, maximum frequency, no N
and no spread. A margin over a point estimate of unknown method is a
comparison, not a result. What is a result is the same binary on the same
board, four models, +17% to +27%.

## This was available four days earlier

A comment in `llama.c` dated 2026-09-06 already said, in those words, *"that is
the calling thread landing on an A53, and a scheduling lottery is worse than a
small steady loss"*, and the table beside it had already measured the best
column. Then neither knob was given a default, on the grounds that "a wrong
guess baked in is a regression nobody could see".

That reason was correct while nobody had measured what not choosing costs. It
is the same shape as the AWQ clamp: a reason not to act has to be rechecked
when the measurement behind it changes.

---

# R4: answered, then closed by the same fix

Twenty readings at `performance`, in time order. The verdict: a gap ratio of
15.0 with the gap in the middle, no drift, 12 low and 8 high, and 10 runs
against the 10.6 that independent coin flips would give, so cluster membership
is decided afresh each run.

Two clusters, 1890 and 2010, 6.3% apart, with a total spread of 9.9%.

## Then the same fix closed it

Twenty readings after pinning: median 1978, spread 4.4%, and the decision tool
refuses to call a shape at that width.

```
  46%    seven readings, whatever governor the board had    what R4 asked about
   9.9%  twenty readings, performance
   4.4%  twenty readings, performance, pinned
```

Two clusters at the second step and none at the third. gemma4's TTFT was the
same lottery, and the row can now be quoted as a median without qualification.

The seven readings that motivated R4 spanned 46% because they were taken at
whatever governor the board had, so the question was being asked of a
configuration nobody would quote from.

---

# R8 is worse than the checklist assumed: the two machines had different files

```
              md5                                bytes
  desk   c82c0340d974c3eca5b528236cf9f621       773025824
  board  48ff0243978606fdba19d899b77802fc       773025920
```

Thread count was already ruled out, since the desk gives bit-identical results
at 8, 4 and 1 threads. The toolchains differ too, but the file difference is
enough on its own.

`bartowski/Llama-3.2-1B-Instruct-GGUF` has been re-uploaded. Downloading what
Hugging Face serves today gives the board's md5 exactly, and with the same file
the two machines agree to the last digit on all three arms.

That rewrites something on record. "Board and host agree to 0.3%" in the AWQ
round, and "the board confirms the desk", were two different files landing near
each other. Cross-machine quality numbers were never comparable, and nobody had
checked a fingerprint.

`board_record.sh` now stamps the model's md5 and size, the binary's md5, and
the thread count on every round.

---

# The quality table: AWQ confirmed on the board at -30.9%

```
   calibration      2647768 bytes from calib.txt
   int4             34.2425
   int4+AWQ 0.20    23.6746        -30.9%
   int8             18.3604
```

Since the two sides were on different files when this was first compared, what
can be said is that the size of the gain agrees within a percentage point, not
that the numbers match.

---

# Five harness defects found in one day

None of them were the board. All five were a harness reporting a state it had
never verified.

| defect | symptom | cause |
|---|---|---|
| null AWQ arm | equal to its control to the last digit | no calibration, so AWQ declined, and the stderr saying so was discarded |
| "gemma4 is not on the board" | the whole sweep skipped | it was on the board, in the other models directory |
| false silence | none | `\| tail` gave zero output for a whole round, and 300 s of silence tripped a USB reset |
| silent death | section 3 printed its header only, and the boot check never ran | no `thermal_zone` on this board, so the glob stayed literal and `set -e` ended the script |
| the decision tool read backwards | `big4` at 1.6% spread was called "not quotable" | I had written "no structure" as "not quotable", and a scale-free gap ratio found structure in the last digit |

The `set -e` defect only became reachable once gemma4 was found: while the
sweep was skipping, that line never ran. Fixing one fault uncovered the one
beneath it.

The null arm was the most expensive. Without the 296 s smoke run, that number
would have reached a paper as "AWQ does nothing on the board".

---

# Final state of the checklist's eight items

| item | state |
|---|---|
| R1 | Diagnosed, fixed, and re-measured on four models. The margin was best-of-N picking the fast mode of a bimodal decode; the cause is core placement; `a58086c` pins the fast cluster by default, giving +17% to +27% across four models at spreads of 0.1 to 1.9%, all four ahead of the vendor on the median, with the new medians equal to the old best-of-seven to within 1% |
| R2 | Done, entirely on the desk. On the pinned protocol the vendor's excess is 2.00, 2.23 and 2.35 times charsiu's across three nested subsets |
| R3 | Done. One boot, boot id taken at both ends and unchanged |
| R4 | Closed. It was two clusters 6.3% apart; after pinning the spread is 4.4% with no shape callable. The same lottery |
| R5 | Not a run, a misreading. `12.59 * 64/65 = 12.40`, so the gate can come out |
| R6 | Settled. 9.9 GB/s retired, 11.7 GB/s annotated as a cache walk |
| R7 | One RK3576 only. State it as a limitation |
| R8 | Worse than the checklist assumed. It is the file, and the two machines had different ones |

---

# Next

The `npudev.c` per-tensor width refactor. `npu_mixed_test` on the board answers
the question that blocked it: one open device alternates w8a8 and w4a16
correctly, 0 of 18 dispatches wrong over eight alternations in both directions.
That was at K=256 and N=64, so it justifies the refactor rather than
demonstrating the knob.

The checker's ban on "beats the vendor" can now be lifted against a
measurement. The rule should record what was measured and when, not the verdict
it produced.

Every speed figure in a paper needs to say it is a median of seven with the
calling thread pinned, and the vendor column needs to say it is a citation at
maximum frequency with no N and no spread.

Two notes for whoever runs the next round: one script per UART opener, and
`charsiu update` is interactive so it needs `CTUI_ASSUME=yes ... </dev/null`.
The `stable` channel installs neither the probes nor the corpora, so a board
round needs `update dev`.
