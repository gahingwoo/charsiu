# Plan

The goal is the vendor's own number on the same board and the same model, with
nothing closed in the execution path. Whatever the NPU can carry goes on the NPU.

Each step below states what it produces and how it is known to be right. The rule
carried over from the driver work applies here too: **a board round states its
decision rule before the run and carries a control that can fail**, and a proxy
metric is not a correctness oracle.

## What the first five steps turned out to be

They are kept as one section rather than five because they are answered, and what
they were for is the answers.

- **The instrument.** `tools/rkllm_regcmd.py` reads the vendor's dispatch plan out of
  a `.rkllm` with no board: three precisions by role, projections split across two
  cores, and 3752 dispatches at M = 1. It is still how a question about the vendor
  gets settled, and it is how the M > 1 register geometry was read.
- **M = 1 at an LLM's width.** Exact, and M = 2, 3, 4 and 8 are exact too through
  Mesa's own delegate. The RK3588 height constraint is not on this silicon.
- **One vendor matmul, bit exact.** Done, in int4, and then every matmul in
  Llama-3.2-1B.
- **The measurement that decides the architecture.** Answered in the NPU's favour,
  and not marginally: **15.91 tok/s at int4 and 64 tokens against the vendor's
  roughly 13**, with the hardware moving weights at 10.8 GB/s, which is this board's
  DRAM roof. The RK3588 conclusion that decode belongs on the CPU is not this board's
  conclusion.
- **The runtime.** It exists, it is standalone rather than a ggml backend, and it
  reads llama, qwen2, qwen3, gemma3, gemma4, phi3 and smollm3.
- **The product layer.** `charsiu`, `charsiu-get`, `charsiu-config`, `charsiu-doctor`,
  `charsiu serve`, and an installer.

## 1. Prefill: batched, correct, 2.94x on hardware. DONE

The vendor dispatches every one of its 3328 int4 projections at M = 1, so it
re-streams all 487 MB of weights for every prompt token. charsiu does not.

A 64 token prompt on an int8 staged Llama-3.2-1B, on a ROCK 4D, same binary,
one flag apart:

```
  batched   prompt 64 tok in 2406 ms   26.60 tok/s    37.6 ms a token
  control   prompt 64 tok in 7078 ms    9.04 tok/s   110.6 ms a token
```

**2.94x, and the two runs generate the same text word for word.** Prefill used
to cost more per token than decode; it now costs a third of it.

`llama_prefill_batch` runs the layer loop with n rows, batching the feed
forward -- gate, up and down, 63% of the projection time -- and leaving q, k, v
and o a row at a time. Rows are walked in order inside a layer because row r's
attention reads the KV the rows before it wrote, and the head is not batched at
all: a prompt needs logits for its last token and no other, so the widest
projection in the model is skipped n - 1 times rather than made n times wider.
The prompt goes in chunks of 32, because the buffers scale with the batch and
the probe's sweep flattens after 16.

### The other three modalities, measured on the board 2026-08-28

Seeing, hearing and matching all give the right answer on a ROCK 4D -- jfk.wav
transcribed word for word, the llama.cpp logo called a llama by both the VLM and
CLIP -- and all three are SLOW, in the same way:

```
  whisper tiny.en    34.5 s   for 11 s of audio    ~20 G-mac    0.58 G-mac/s
  SmolVLM-256M      153.1 s   for one picture      ~90 G-mac    0.59 G-mac/s
  CLIP ViT-B/32       4.7 s   for one picture     ~4.2 G-mac    0.9  G-mac/s
  the decode under them, unchanged:  19.75 tok/s
```

⚠ **NONE OF THEM TOUCHES THE NPU.** vision.c, clip.c and whisper.c call
gguf_matvec directly; only llama.c's projections are routed. So 0.6 G-mac/s is
not the hardware being bad at this, it is the scalar CPU path, and three
independent graphs landing on the same number is the same bottleneck three
times rather than a coincidence.

⚠ **AND THE SHAPE IS THE ONE ALREADY SOLVED.** A picture is 1024 patches, a
thirty second clip is 1500 encoder positions, a CLIP image is 50 -- all present
at once, against weights that do not change. That is the batched matmul measured
at 2.94x on 2026-08-27, at m values twenty to fifty times larger than a prompt
chunk. The NPU moves weights at 10.8 GB/s, which at int8 and m = 1 is about
10 G-mac/s before batching buys anything.

The work is not the arithmetic, which is written and checked. It is that the
NPU tensor cache lives in struct llama_state and these three graphs are not
llama.

### The towers on the hardware, and where the batch actually stops

Board, 2026-08-28, after routing the vision tower and the whisper encoder:

```
                CPU      NPU     with the cache warm
  whisper      34.5 s   30.0 s        30.0 s
  SmolVLM     153.1 s   82.4 s        81.3 s
  CLIP          4.7 s    3.9 s         2.8 s
```

⚠⚠ **AND THAT SUBTRACTION WAS WRONG.** The paragraph here used to read: 75 of
the vision tower's 82 seconds is the quantiser, so take it out and the matmuls
are 17x. The next board round put the staging BEHIND A CACHE -- whisper's
quantising went from 19298 ms to 118 -- and the wall clock did not move at all:

```
  whisper   30.0 s before the cache,  30.1 s after it, 30.0 s warm
```

Nineteen seconds left the staging line and the total stayed put, so the two
numbers were never parts of one sum. Nothing was counting how much of the work
reached the hardware, and a subtraction cannot be checked against a quantity
nobody measured. `charsiu_pool_report` counts it now: tensors actually routed,
matmuls, rows, milliseconds, and how many fell back.

The towers do stage eagerly into a cache in `$XDG_CACHE_HOME/charsiu`, and that
part works -- ⚠ the cache is ONE SEQUENTIAL FILE with one static handle, so a
caller claims it for its own stretch rather than sharing it, because two graphs
staging at once interleave and neither can read the result back.

⚠ **AND THE BATCH STOPS AT 80.** Swept against the same tower at two rows,
which is the smallest verified batch:

```
  4 8 16 32 48 64 80    0.000000   identical
  96 and above          56 to 95   a different tower
```

Mesa's budget test fires above m = 320 and its split above m = 640, so **that
prediction is about something else**; this is a different limit and it is
measured rather than derived. The default is 64: inside the edge with a step to
spare, and the rate is FLAT from 4 rows to 1024 anyway.

⚠ 80 was measured on one tower at K = 768 and 3072. Whether the bound is m alone
or m against K is not known.

### And then the counter said the matmul was never the cost

`charsiu_pool_report`, added because the 17x above had to be withdrawn:

```
  whisper   24 of 24 tensors on the hardware; 24 matmuls of 36000 rows in 936 ms
  CLIP      73 of 73 tensors on the hardware; 72 matmuls of  3600 rows in 210 ms
```

**936 ms of a 30 second transcription.** So the stage table, which is the thing
that should have been written before any of the routing:

```
  attention              2181 ms   62.7%     O(T^2) scalar C, never routed
  feed forward            670 ms   19.3%     routed
  q k v                   179 ms    5.2%     routed
  mel spectrogram         189 ms    5.4%     3000 FFTs
  the two convolutions     70 ms    2.0%
  out proj                 59 ms    1.7%     routed
  the decoder             121 ms    3.5%
  layernorms                8 ms    0.2%
```

⚠⚠ **THE WORK THAT WAS ROUTED IS 26% OF THE TIME AND THE ATTENTION IS 63%.**
1500 positions against 1500, six heads, four layers, in three nested loops --
and it is not a matmul against a weight, so none of the machinery built for the
towers touches it. Everything above this heading is correct and was aimed at a
quarter of the problem.

The next thing is the attention, and the stage table has to come first from now
on. It cost three board rounds and a withdrawn number to write one.

### The vendor's own table, and what it says we lose

Rockchip publish RK3576 numbers for their runtime on their driver, w4a16, a 128
token prompt and 64 new tokens (airockchip/rknn-llm/benchmark.md, 2026-08-28).
`tests/board_vendor.sh` runs the same protocol and prints both columns.

```
                 ours    theirs      ours TTFT   theirs
  Qwen3 0.6B    10.48     24.85         7354 ms   469 ms
  Phi3 3.8B      4.89      6.58        23354 ms  1829 ms
  SmolVLM-256M image encoder            10000 ms   768 ms
```

⚠ **THEIR NUMBERS ARE AT MAXIMUM CPU AND NPU FREQUENCY** and ours are at
whatever the governor is doing. That is in their header and it is not a small
difference.

⚠⚠ **AND THE TTFT COLUMN WAS ONE BUG, NOT FIVE.** Of the five models in their
table this tree can run, FOUR were refused by `batch_ok` -- a bias, a query
norm, a fused K and V, per layer embeddings -- so their prompts went through the
token loop one token at a time. Only TinyLLAMA batched.

Four of those refusals were lifted on 2026-08-28 because they are the SAME PER
ROW OPERATION the token loop already does, in the same order: biases, the query
and key norms, the two post norms, and the logit softcap. Refusing them was
right while they were unwritten and became the dominant cost the moment there
was a scoreboard to read.

What is still refused is what would be a different computation: a fused K and V
needs a tensor split, shared KV needs another layer's cache, and a varying feed
forward width needs per layer buffers. (The sliding window left this list the
same day; the paragraph above is the state before that.)

### What the submit count settled

The device's own report, on Rockchip's protocol, with CHARSIU_NPU_W4V=1:

```
  Qwen3 0.6B    38608 submits   189 us each   5.97 GB/s   222 submits a token
  TinyLLAMA     47128           206           9.21
  Phi3 3.8B     68488           460          10.47
```

⚠ **THE PREFILL IS ON THE HARDWARE, ONE ROW AT A TIME.** Not a CPU fallback:
every row of the prompt streams the whole weight again, at 6 to 10.5 GB/s, which
is the board's roof. The hardware is not being lazy -- 222 submits a token is
what one row per submit means for 28 layers of 7 projections in 1.5 slices.

⚠ **AND THE VENDOR CANNOT BE DOING THAT.** They publish 469 ms to a first token
for a 128 token prompt on the same model, and 40 ms a token to decode. A prompt
token 11 times cheaper than a generated one is not one row per submit: Qwen3's
weights are 300 MB at int4, and 128 rows of that is 38 GB in 469 ms. They batch
the prefill, on w4a16.

That contradicts what this tree recorded as "w4a16 makes exactly one row". What
five rounds established is narrower: **charsiu's** w4a16 stream makes one row,
and no register in it changes that. Their prefill graph may not be the same
stream at all.

⚠ **ANSWERED, TWO SECTIONS DOWN.** It is the same stream with M on the other
axis. Way 3 below was the one that removes the problem and it is no longer a
question.

Three ways out, and only the first is free:

1. **the grouped submit**, which this loop had and lost. q, k and v read one
   RMSNorm output and matvec_pair sends the three in ONE submit; round 321
   measured the fence at 94% of the hardware path and that grouping took 113
   fences a token to 65. Batching them as three separate calls turns n grouped
   submits into 3n the moment the batch is refused -- which on int4 is always.
   Restored: matmul_rows says whether the hardware took the batch, and the
   caller groups when it did not.
2. **int8 for the prompt**, which batches and was measured at 2.94x. It costs a
   second copy of every projection -- 600 MB on a model whose whole int4 weight
   is 300 -- and this board's peak is already twice the vendor's.
3. **find what their prefill stream does that ours cannot.** The only one that
   removes the problem rather than paying for it.

### Asked their model, and read the wrong axis

`lyvivian/Qwen3-0.6B_rk3576_w4a16_4k.rkllm`, 720 MB -- the same model, the same
board and the same format as the row above -- read with `tools/rkllm_regcmd.py`:

```
  bits=4.0   M=1    9296     every int4 weight matmul is ONE ROW
  bits=16.0  M>1    4984     every batched op is fp16
  no 4-bit stream has M > 1
```

**That reading was wrong, and the next section is the correction.** It is left
here because the mistake is the useful part: M was taken from the register that
holds the row count.

### They batch int4 after all, up to 80, and 80 is our own ceiling

`Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm` -- a model this tree also has as a
gguf, so both sides of the comparison are on one desktop. Reading M from the
PIXEL count (`0x1034`) instead of the row count (`0x102c`):

```
  bits=4.0    3328 streams   M = 1:512  16:384  24:64  32:512
                                 40:320 48:384  64:384 80:768
              2816 of 3328 -- 85% -- are BATCHED
  bits=16.0   4940 streams   rows == pixels in 4940 of 4940
  largest int4 M anywhere in the file:  80
```

⚠⚠ **THE TWO AXES ARE THE WHOLE STORY.** The vendor's fp16 attention is emitted
as an M row image, so its row count and its pixel count are the same number --
all 4940 of them -- and a reader that takes rows is right about fp16 and never
finds out. Its int4 projections are emitted as a **one row image, M pixels
wide**, so their row count is 1 whatever M is. Every int4 stream in every
.rkllm read here therefore reported M = 1, and the file was made to say the
opposite of what it holds.

⚠ **AND THE LARGEST M THEY EMIT FOR int4 IS 80**, which is also where this
board's batched prefill stops being exact. ⚠⚠ **That is not the same fact twice
and it was written here as if it were.** Ours is int8 on the HEIGHT axis --
`board_rows_sweep.sh` sweeps the vision tower, whose matmuls are int8, and it
is exact to 80 and wrong from 96. Theirs is int4 on the WIDTH. Different
format, different arrangement; the shared 80 is so far a coincidence.

⚠ **AND THEIR int8 GOES TO 128.** All 40 int8 streams in the file are the LM
head, `ic=2048 oc=8160`, at M of 1, 32, 64, 96 and 128 -- and every one of them
is ONE ROW HIGH too. So the vendor puts M on the width for **both** weight
formats and uses the height axis for nothing but its fp16 attention.

That makes our own 80 suspicious rather than confirmed: the arrangement that
breaks at 96 is the one the vendor never uses for a weight matmul, and the one
it does use runs the same int8 at 128. `board_rows_sweep.sh` now sweeps both
axes, height first as the control.

So batching an int4 weight matmul **is** available, and the plan to beat their
TTFT by doing it was aimed at something they do every layer.

### The board: the width axis batches, and the read order is what is left

Both rounds ran on 2026-08-29. They disagree about nothing and they move the
problem from "int4 cannot do more than one row" to "the values come back
somewhere this tree does not look".

**int4, `board_w4_axis.sh`.** On the width axis the probe ran and the hardware
batched:

```
   blk.0.attn_q.weight at m=2: 2526 of 4096 wanted values are in the batch
     row0 batched   -0.002  0.120  0.182  0.265  0.100  0.317
     row0 one row   -0.002  0.120  0.182  0.265  0.100  0.317
     row1 batched   -0.341  0.141  0.052 -0.155 -0.620  0.251
     row1 one row   -0.104 -0.127 -0.237 -0.390  0.892 -0.450
     row0: 1239 of 2048 of its values are in the batch
     row1: 1287 of 2048

     m   worst rel   rows agree   one row   batched  speedup  GB/s
     2    1.13e+03    0 of 226      118 ms    75 ms   1.58x   16.55
     4    1.21e+03    0 of 452      230 ms    92 ms   2.51x   13.50
     8    3.16e+04    0 of 904      461 ms   174 ms   2.66x    7.12
    16    1.52e+03    0 of 1808    1018 ms   293 ms   3.47x    4.21
    32    2.00e+03    0 of 3616    2004 ms   581 ms   3.45x    2.13
```

⚠ **THE SPEEDUP IS NOT THE EVIDENCE.** A run that computes one row and returns
noise for the rest is also sub-linear in m. The discriminating fact is **row 1**:
1287 of its 2048 values are in the buffer. Five rounds on the height axis had
row 1 matching row 0 in 1 of 2048 -- absent, not misplaced. It is being
computed now.

⚠⚠ **AND THE HEIGHT ARM OF THAT ROUND WAS VACUOUS.** It was the arm that had to
fail and it failed by hitting `w4_batch_gate()` in npudev.c -- a decision in
software, which says nothing about silicon. A control that cannot reach the
thing it controls for is not a control. `CHARSIU_NPU_W4_BATCH=height` now lets
the wrong axis through on purpose and the script stops if the refusal appears.

**int8, `board_rows_sweep.sh`.** Its height arm is a real control and it passed
exactly: identical to the one row path at 4, 8, 16, 32, 48, 64 and 80 and
DIFFERS from 96 -- the known bound, reproduced. Its width arm **DIFFERS at
every row count from 4 up**.

So both formats say the same thing from opposite sides. `charsiu_acc_index()`
was solved on the HEIGHT axis, from maps printed at m = 2 and m = 4, and the
width axis swaps the two image axes underneath it. The arithmetic is right and
the read order is wrong, which is a smaller problem than the one before it and
a different one.

⚠ **AND 80 IS STILL int8-ON-THE-HEIGHT.** The width arm does not reach it,
being wrong at 4, so nothing yet says whether that ceiling is the arrangement.
That question survives this round.

**Next: `board_acc_map.sh`.** `npu_gemm_test --read` prints the permutation
itself rather than a candidate for it -- it is what solved the height axis --
at m = 2 and m = 4, both axes, K = 64 N = 64 where the reference has 112
distinct values in 128. The height arm is solved and must come back EXACT.

### The map: the width axis writes a SHORT surface, not a shuffled one

`board_acc_map.sh`, 2026-08-29, straight after the round above.

**The control is exact.** The height arm is the solved one and it came back
128 of 128 words at m = 2 and 256 of 256 at m = 4, **0 values absent**, the map
reproducing `charsiu_acc_index` position for position. So the shape, the build
and the tool are all where they were when that expression was fitted.

**The width arm is not a permutation.**

```
             words written      reference values absent
  height     128 of 128 (m=2)   0 of 128
             256 of 256 (m=4)   0 of 256
  width       92 of 128 (m=2)   31 of 128
             148 of 256 (m=4)   87 of 256
```

npu_gemm_test's verdict on the width arm is the opposite of the one it prints
on the height arm: *"31 values were never computed, so no read order recovers
them."*

⚠⚠ **AND THE LINE ABOVE IS A CORRECTION.** The round before this recorded "the
arithmetic is right and only the read order is wrong". That sentence is the
tool's, printed on the HEIGHT arm, and it was carried across to the width one on
the strength of row 1 coming back. Row 1 does come back -- that part stands and
it is still the first movement in five rounds -- but a third of the surface is
never written, and no permutation recovers a word that was not produced.

**The shortfall is regular, which is the useful part.** Inside every (row, 32
channel super group) the width axis writes the FIRST FIVE four word runs --
channels 0-3, 16-19, 4-7, 20-23, 8-11, in the same interleave the height axis
uses -- then moves to the next group, dropping 24-27, 12-15 and 28-31 every
time. Five runs of eight, at every group, at both m. Words written come to
**36 + 28m** against the height axis's **64m**.

A fixed fraction of every group is a size or a stride, not an address and not a
read order. `tests/board_width_short.sh` sweeps the candidates one field at a
time through `CHARSIU_OVERRIDE` -- 0x1090, 0x1094, 0x1098, 0x401c, 0x4028,
0x40b8 and RDMA 0x5010, which are ours at 8, 2, 4, 2, 0, 2 and 1 on this axis
at m = 2 -- and scores each by the number that cannot be argued with: how many
words the board wrote. The baseline runs first and must reproduce 92 of 128 or
the sweep refuses to print a table.

⚠ Mesa's generic encoder computes the first three from `inw` and `full_inh`,
which on this axis are M and 1 rather than 1 and M. The swap is what is under
suspicion, and the sweep is how it stops being a suspicion.

### 0x40b8 is 3 * M, and `rows` is not M on the width axis

`board_width_short.sh`, 2026-08-29. Seven candidate size registers, one field
at a time from the baseline, scored by the words the board wrote:

```
  0x40b8   value    1    2    3    4    6    8   16
           words   68   80   92  104  128  104   64
           absent  54   41   31   20    0   21   60
```

**6 writes the full surface with nothing absent, and 6 is 3 * M at m = 2.**
0x1090, 0x1098, 0x4028 and RDMA 0x5010 changed nothing at any value in range,
which is worth printing: four fields excluded with data. 0x1094 and 0x401c move
the number but never past 100 of 128.

⚠ **AND THE BASELINE WAS 3.** npu_gemm_test takes the acc_out arm, which
computes `3 * rows`, and `rows` is 1 on the width axis at every M. The
baseline's 92 and the sweep's `0x40b8 = 3` row are the same number twice, which
is what says the two are the same setting rather than a coincidence.

⚠⚠ **THIS IS NOT A ONE WIDTH FIT, AND THE RECORD ALREADY SAID SO.** Round 385
swept this register on the HEIGHT axis at three widths and found 3, 6, 12 at
m = 1, 2, 4 -- `3 * M` -- and recorded that each peak was worth about ONE ROW,
64 values whatever m was, so it was filed as "not the whole fix". It is the same
expression. What the axis decides is whether it completes one row of the surface
or all of it.

The fix is `3u * (wide ? ow : rows)`, which is M under either arrangement.
Checked rather than argued: the emitter before and after, across both acc_out
arms, 2 weight formats, 2 K, 2 N and 6 M -- **96 of 96 bit identical on the
height axis** -- and on the width axis 0x40b8 goes 3, 6, 12, 24, 96 at
m = 1, 2, 4, 8, 32 where it was 3 at all of them.

⚠ **A FULL SURFACE IS NECESSARY AND NOT SUFFICIENT.** Every value being
somewhere is not the same as this tree knowing where. `board_acc_map.sh` runs
again next -- now at m = 2, 4 **and 8**, a width 3 * M has never been asked at
-- and if the width arm comes back full then the map under it is the read
order, which is the last thing between here and a batched int4 prefill.

### The width map IS the height map, so the read order was never the problem

`board_acc_map.sh` again with 0x40b8 fixed, 2026-08-29.

**The width arm writes the full surface.** m = 2: 128 of 128 words, 0 absent.
m = 4: 256 of 256, 0 absent. The height arm, the control, is unchanged and
exact.

**And the two maps are the same map.** Diffed rather than eyeballed: the m = 2
tables are identical cell for cell, and so are the m = 4 ones. Then the
expression itself, against the board's own table:

```
  charsiu_acc_index agrees on 99 unique slots at m = 2, disagrees on 0
  (29 duplicates skipped -- locate() reports the FIRST slot holding a value,
   so a value the reference produces twice is not evidence)
```

⚠ **SO THERE IS NO WIDTH AXIS READ ORDER TO SOLVE.** The round before this
called the read order "the last thing between here and a batched int4 prefill".
It was not a thing at all: once 0x40b8 counts M, the width axis returns the
same surface in the same order the height axis does, and the expression solved
on one reads the other with nothing changed.

⚠⚠ **AND THIS ROUND IS ABOUT int8.** npu_gemm_test is the int8 accumulator.
What it settles is that on the width axis 3 * M is necessary and sufficient for
a full surface, and that the read order transfers. It says nothing directly
about w4a16, which is a different weight format and the one five rounds called
"exactly one row". The runtime takes the same arm -- `npudev.c` sets
`acc_out = 1` for int4 and int8 alike -- so the fix reaches it, and whether it
is enough is the next round rather than a conclusion of this one.

⚠ **m = 8 WAS NOT MEASURED AND THE SCRIPT IS WHY.** It printed the map with
`head -80`, and npu_gemm_test puts its counts AFTER the map, so at 512 words
the verdict was cut off on both arms. The three lines that decide the round are
grepped out before the map now. m = 8 is still owed on both axes.

**Next: run the two rounds that were wrong before the fix again** --
`board_w4_axis.sh` for int4 and `board_rows_sweep.sh` for the int8 ceiling.
Both last ran with 0x40b8 at 3, which is what made the surface short, so
neither of their width arms was asking its question.

### Both rounds again, with 0x40b8 counting M

**int8, the tower sweep. The axis moved the ceiling and the control says so.**

```
  height (the control)   identical to 80,  DIFFERS from 96
  width                  identical to 96,  DIFFERS from 112
```

The height arm reproduced its known bound exactly in the same run, so the +16
is the arrangement and not the board warming up or the build moving. The true
bound is now somewhere in (96, 112]; the vendor's int8 head runs 128 wide.

⚠ It buys nothing today. `CHARSIU_NPU_ROWS_MAX` defaults to 64, under both
bounds, and the sweep's own seconds column is flat at every width -- 9.8 to
12.8 s from 4 rows to 1024. The default stays on the height axis until
something needs the room.

**int4, the probe. The axis is NOT the blocker, and now the control says that
too.**

```
                    values present   row0     row1     rows agreeing
  height             3016 of 4096    1673     1343     0 at every m
  width              3354 of 4096    1677     1677     0 at every m
```

The height arm ran this time -- `CHARSIU_NPU_W4_BATCH=height` reaches the
hardware where the last round's control was refused in software -- and it is
wrong in the same way the width arm is. Row 0 is exact on both; row 1 batched
is the SAME wrong numbers on both and matches neither row of the reference.
The width arm's per row counts are symmetric where the height arm's are not,
which is worth something, and neither agrees on a single row at any m.

So w4a16 batching survives the axis and survives 0x40b8. Those were the two
things this fortnight had reason to suspect and both are now excluded on the
board.

⚠⚠ **AND w4a16's OUTPUT SURFACE HAS NEVER BEEN MAPPED ABOVE ONE ROW.**
npu_gemm_test has no int4 in it at all -- the tool that solved the accumulator
twice cannot ask this question -- and `charsiu_int4` runs w4a8, an int8
activation, which is not the runtime's format. Everything known about the read
order is from the int8 accumulator.

`llama_batch_probe` already held both arrays and counted how many wanted values
were present. It prints WHERE each one landed now: the whole of row 0 rather
than six channels of it, the distinct count first because a repeated value
answers a question it was not asked, and the residue after this tree's own read
order rather than a raw index -- Y comes back through `charsiu_acc_index`
already, so "landed at (r, c) itself" is what correct looks like.

That runs on the real path with the real weights, which the two dedicated tools
cannot do.

### The landing table, and the two things wrong with reading it

The round printed, and then my own script cut it off at eighteen lines -- which
is exactly where the landing table ends. Row 0 came back sampled to channel 384,
no row 1, and none of the timing table either. `board_acc_map.sh` lost m = 8 to
a `head -80` one round earlier. Twice is a pattern: **the deciding output is at
the bottom of what these tools print, and a cap is a silent cut.** The cap is
gone; the one left in `board_acc_map.sh` caps the map only, with the three
deciding lines grepped out ahead of it.

What did print is worth having:

```
  height   (0,320) landed at (0,63)      the other six sampled channels correct
  width    (0,320) landed at (0,320)     all seven correct
           row0 1677 of 2048, row1 1677  against height's 1673 and 1343
```

⚠⚠ **AND THE ORACLE IS WEAK, WHICH THE BOARD SAID ITSELF:** *the reference has
1456 unique values in 4096*. The search matches on 1e-3 RELATIVE, which for a
value near zero is 1e-3 absolute, so every small output matches every other
small output. Both the present count and the landed-at column are inflated by
that, and 2640 of 4096 reference values are not unique. A hits column reading
1 to 4 is saying so on every line.

So the probe now scans **position by position** as well, `Y[r][c]` against
`Yref[r][c]`, which has no such freedom: how many channels of each row agree
where they stand, and the first one that does not. It cannot say where a value
went, but it says exactly where a row stops being right -- which is the question
row 0 has been raising for three rounds by being exact in its first six channels
and agreeing on no row at all.

### Exactly half, and the first miss is channel 16

The in place scan, which is the instrument the fuzzy one should always have
been:

```
  height   row0  1025 of 2048 agree, first wrong at channel 16
           row1     4 of 2048
  width    row0  1024 of 2048 agree, first wrong at channel 16
           row1  1025 of 2048, first wrong at channel 0
```

**Two things at once.**

The axis, quantified at last: on the width axis **row 1 is as good as row 0**,
1025 against 4. That is what "the width axis batches" means in a number, and
the height arm in the same run is the control that makes it one.

And both arms agree on **exactly half** their channels with the first miss at
**16**. In `charsiu_acc_index` the channel splits as `a = (c % 32) / 16` and
`t = c % 16`; a = 0 is exactly half the channels and 16 is the first of the
other half. So the int8 expression is right about w4a16 everywhere except
**where the second sixteen channel half goes**, and `a * 4` is the only term
that places it.

⚠ The landing table sampled every 64 channels, so every one of its 32 samples
had a = 0 and nearly all were correct. It was looking only at the good half.
It walks 0 to 63 in full now -- the structure repeats every 32, so one pair of
super groups holds all of it -- with a coarse tail after.

**The family has two members.** Of `a * A` and the two halves swapped, only
A = 4 and the swap are permutations at all: 128 of 128 distinct slots against
68, 80, 96 and 64 for A of 8, 1, 2 and 0. A = 4 is the control, so this is a
two horse race. `CHARSIU_ACC_A` picks, `board_w4_axis.sh` runs both plus the
height control, and the deciding line is the in place count -- 1024 is the
a = 0 half and nothing else, 2048 is the read order solved.

⚠⚠ **AN OFFLINE SWEEP OF THIS WAS WRITTEN AND DELETED.** It reconstructed the
raw buffer from Y and scored candidates with no board round at all. Y is not
raw: the batched read is `yr[j] += fo[mp[j]] * sc[j]` and **sc is per output
channel**, so a value scored at a different channel carries the wrong scale and
the table is off by a ratio wherever a candidate crosses a group boundary.
Nearly sound is the kind of instrument this fortnight has already been burned
by twice.

### The wrong arm printed the answer: a and the row trade places

The swap arm was wrong everywhere -- 0 of 2048 channels in place -- and it is
the round that solved this, because it printed WHERE its values went:

```
  (0,16) -> (1,16)   (0,17) -> (1,17)   (0,21) -> (1,21)  ... 23 of them
  (1,0)  -> (0,0)    (1,1)  -> (0,1)    (1,15) -> (0,15)  ... 13 of them
```

Row 0's a = 1 channels at row 1's slot, row 1's a = 0 channels at row 0's, same
channel both ways. And the default arm's in place counts say the same thing
from the other side: of the four quadrants of (row, half), **(0, a=0) and
(1, a=1) are correct and the two off diagonal ones are not**, which is exactly
what 1024 and 1025 of 2048 are.

One expression covers both. Put `a` where `mi / P` was and `mi / P` where `a`
was:

```
  j     = (t/4)*(8P) + (mi%P)*8 + (mi/P)*4 + (t%4)
  index = G*m*32   + a*(32P)   + j
```

Checked, not fitted:

- it reproduces **all 36** of the swap arm's printed landings, none missed
- it agrees with the default on **exactly** the two quadrants the board calls
  correct and differs on **exactly** the two it calls wrong
- it is a permutation at m = 2, 4, 8, 32 and 80 -- 163840 distinct slots of
  163840 at the widest
- the C and the arithmetic that derived it agree index for index at m = 2, 4
  and 8

⚠ **ALL 36 LANDINGS ARE AT m = 2**, where P is 1 and `(mi % P) * 8` is inert.
Which half of the row takes the 4 and which keeps the 8 is a choice at wider m
rather than something the board has said. The probe sweeps m to 32 and its per
m row count is what would catch it.

`CHARSIU_ACC_A=roleswap`, and `board_w4_axis.sh` runs it as the third arm
against the height control and the default width. 2048 of 2048 on both rows is
the read order solved.

### Gemma4's whole list, in one line, because the refusal enumerates now

```
  not batched: per layer embeddings, fused or absent K and V projections,
               KV shared between layers, a feed forward width that varies by layer
```

Four, not one. The value norm that used to be reported first is lifted and was
never the whole story; returning the first reason would have cost a board round
for each of these in turn. 17844 ms to a first token against Rockchip's 1219 --
it is the worst number left on the table and it is four separate pieces of work.

## Best of three, and the numbers are trustworthy now

```
                  best     spread          theirs   gap
  Qwen3 0.6B      1792     1792..1809       469     3.8x
  TinyLLAMA       2162     2162..2265       544     4.0x
  Phi3 3.8B       6551     6551..6809      1829     3.6x
  Gemma4 E2B     17564    17564..17978     1219    14.4x   not batched
```

`prefill_control`: 50.59 and 52.50 tok/s against the control's 15.26, text
identical, decode unchanged. The gather change did land -- 47.88 before it.

⚠ **AND THE SPREAD ITSELF SAYS SOMETHING.** Qwen3 moves 1% inside one round and
moved 9% across three rounds of nearly the same build. The variance is BETWEEN
rounds -- thermal state at the start, page cache -- not inside one. Best of
three within a round is the right statistic and comparing single runs across
rounds was never going to work.

TTFT against where this line started: **7354 to 1792 on Qwen3, 4.1x**.

### What a submit costs, from four points that land on a known number

Per-submit time against the bytes each submit moves, off one round:

```
  us a submit = 102.7 * MB + 112       marginal 9.74 GB/s, fixed 112 us
    Qwen3      1.70 MB   357 us   fit 287   +70
    TinyLLAMA  2.78      390         398     -8
    Gemma4     2.93      329         413    -84
    Phi3       6.91      843         822    +21
```

Four points with visible scatter, so the slope is approximate -- **but it lands
on 9.74 GB/s, which is this board's known roof**, independently measured at 9.3
by decode. A fit that arrives at a number nobody fitted it to is worth more
than the residuals suggest.

The fixed part is what matters:

```
  Qwen3      11896 submits x 112 us = 1335 ms of 4245 ms hardware   31%
  TinyLLAMA  12632                   1418 ms of 4933              29%
  Phi3       18312                   2056 ms of 15432             13%
```

**A third of Qwen3's hardware time is per-submit overhead**, because it moves
only 1.70 MB each time. The pool's own report puts it where you would expect:
13129 ms of a 17956 ms hardware path is waiting on the fence.

⚠⚠ **AND THE OBVIOUS FIX IS NOT FREE.** Fewer submits means a bigger K slice,
and `CHARSIU_NPU_KMAX` is deliberately tied to `CHARSIU_NPU_W4_GROUP`: the
slice must BE the quantisation group for the group's scale to be applied on the
way in with nothing extra on the hardware. Raising it coarsens int4 -- per
channel RTN measures 0.1067 relative against group 32's 0.0666. Anyone reaching
for that knob to cut submits is trading the answer for the clock.

### The gather on the board: 28% off, and the vendor table cannot see it

Correctness first, because a faster gather that is wrong is the failure this
tree has already shipped once. Every width still exact: 226, 452, 1808, 3616,
5424, 7232 and 9040 rows agreeing, worst relative 5.10e-05 at all of them. And
m = 8 is now visibly refused rather than silently wrong -- `NOT on the NPU --
int4 at m=8 misses row 0 of the n=8192 tensors` -- and drops out of the table.

The probe's `read` column, which compares inside one run:

```
   m     before    after    change
   2        11        7      -36%
  16       132      104      -21%
  32       368      269      -27%
  48       521      370      -29%
  64       846      620      -27%
  80       933      645      -31%
```

About 28% off the gather, and the batched matmul at m = 80 went 1746 to 1440.
`read` is still 45% of it.

⚠⚠ **AND THE VENDOR TABLE READ WORSE, WHICH IS NOISE AND NOT A REGRESSION.**
Qwen3's TTFT came back 2055, 1867 and 2191 on three consecutive rounds of
builds that differ by this change and the one before it -- a spread of 9% on a
governor left at `ondemand`, with each model run once. A change worth less than
that cannot be seen in that column at all.

That is a fault in the instrument, not in the reading, and it is the headline
column. `CHARSIU_BENCH_REPEAT=3` runs each model three times and prints the
best with its spread beside it, so a number and its noise arrive together. The
default stays 1 because three times four models is a long round.

⚠ The script also still grepped for `int4 computes one row` to show the path,
and that string has not existed since the refusal was rewritten -- it matched
nothing and said nothing for a round. It looks for the m = 8 fallback and the
`batch_why_not` list now.

## THE GATHER IS NOW THE COST, AND IT IS FOUR AT A TIME

The probe's own breakdown of a batched matmul, which the hardware work has
turned inside out:

```
   m     batched     read      read as a share
   2       79 ms      11             14%
  16      308        132             43%
  32      702        368             52%
  64     1478        846             57%
  80     1746        933             53%
```

`read` is `yr[j] += fo[mp[j]] * sc[j]`, the gather through the read order
table, on the CPU. **More than half of a batched projection at every width a
prompt uses.** The hardware part is right now and this is what is left.

**And the read order is four consecutive slots.** `charsiu_acc_index(r, j+q)`
is `charsiu_acc_index(r, j) + q` for q of 1, 2 and 3 at every j that is a
multiple of four -- the `t % 4` term is the only one that moves inside a group
of four and it moves by one. Checked at **1503680 groups** over both formats,
m of 2 to 80 and n of 64 to 8192, none broken.

So the table was carrying a uint32 for every output channel: **four bytes of
index for every four bytes of data, half the gather's memory traffic being the
table itself.** It holds one entry per four channels now and the inner loop
moves four floats off one index.

Checked rather than asserted: the old loop against the new one on the same
buffers, **84 cases, 0 mismatched**, including slice widths of 61, 255 and 8191
that exercise the tail.

⚠ This has not been on the board. What it should move is the `read` column
above, and through it the TTFT column.

## 🏁🏁🏁 THE PROMPT IS 3.04x AND THE TEXT IS THE SAME. 2026-08-29

`prefill_control.sh`, Llama-3.2-1B int4, run twice against the token loop:

```
  batched1   65 tok in 1358 ms   47.88 tok/s
  control    65 tok in 4132 ms   15.73 tok/s
  batched2   65 tok in 1359 ms   47.83 tok/s
  decode     16.07 / 15.95 / 15.97 tok/s      unchanged
  text       IDENTICAL to the control
```

3.04x on the prompt, reproduced to within a millisecond, and the text is the
control's. **That is a batched w4a16 prefill, which five rounds said was not
available on this silicon.**

### The vendor's own table, and how far the gap moved

```
                    before      now     theirs      gap
  Qwen3 0.6B TTFT     7354      2055     469       15.7x -> 4.4x
  Phi3 3.8B          23354      7174    1829       12.8x -> 3.9x
  TinyLLAMA           ~7000     3441     544
  Gemma4 E2B         ~17600    18038    1219       unmoved, and it says why
```

⚠ Ours is the prompt's forward passes and theirs is time to the first token,
which includes that token's own step -- one token in our favour -- and their
numbers are at maximum CPU and NPU frequency while this ran on `ondemand`. The
script prints both caveats itself.

### Gemma4 did not move, and the log named the reason

`prompt a token at a time -- this model is not batched: a value norm`.

The value norm is `qk_norm` with a NULL gain, per row, in the same position the
token loop puts it -- **the same call the batched loop already makes twice**,
for q and for k. Refusing a model for a call the loop already makes was a
statement about the loop and not about the model, exactly like the four lifted
on 2026-08-28. It is in the batched loop now.

⚠ **THAT ALONE WILL NOT BATCH GEMMA4.** Its last layers carry no `wk` and
attend against an earlier layer's cache, so the next reason is waiting behind
this one. `llama_batch_why_not` returns **every** reason now rather than the
first: returning the first costs a board round for each one fixed, and one line
should say the whole distance.

⚠ **AND prefill_control.sh's OWN INVARIANT HAD INVERTED.** It said "int4
refusals in the matmul: batched must be >0" -- true while int4 refused every
batch, where a refusal proved the batched path had reached the int4 matmul.
int4 batches now, so zero is what a correct run looks like and the old sentence
would have condemned this one. It counts the m = 8 fallback and says so.

## 🏁🏁 EVERY WIDTH A PROMPT USES, EXACT. int4 batching is on by default

```
  w4a16, width axis, nothing set
     m   worst rel    rows that agree
     2   5.10e-05      226 of 226
     4   5.10e-05      452 of 452
     8   9.77e+04      871 of 904      <- the only one
    16   5.10e-05     1808 of 1808
    32   5.10e-05     3616 of 3616
    48   5.10e-05     5424 of 5424
    64   5.10e-05     7232 of 7232
    80   5.10e-05     9040 of 9040
```

113 tensors at every width, and 5.10e-05 is float summation order. The height
control in the same run still agrees on nothing, so the probe is discriminating.

### And m = 8 named itself

```
  MISS blk.0.ffn_gate.weight  k=2048 n=8192  row 0 of 8
  MISS blk.0.ffn_up.weight    k=2048 n=8192  row 0 of 8
  MISS blk.1.ffn_gate.weight  k=2048 n=8192  row 0 of 8
  ... eight shown, all of them the same shape and the same row
```

**Row 0 of the n = 8192 tensors, at m = 8 and no other width.** Not scattered,
not a row index across shapes: one shape and one row. 33 rows of 904 is every
ffn_gate and ffn_up in the model, once each. That is a small enough target to
find, and until it is found the batch refuses that one width and the caller
falls back to a row at a time for that chunk -- correct, and merely slower.

### What is now the default

`CHARSIU_M_AXIS` is three states rather than two: `w`, `h`, or unset, and unset
asks the format -- w4a16 on the width, int8 on the height, because that is what
each is right on. `charsiu_acc_index` takes the format for the same reason. The
int4 batch refusal is gone except at m = 8.

Checked rather than asserted: **int8's stream is bit identical at 54 shapes**
against the tree from before any of this session's work, and the host
architecture sanity still passes.

**Next is the number this whole line was for.** `board_vendor.sh` runs
Rockchip's own protocol -- 128 token prompt, 64 new -- and its TTFT column is
what a batched int4 prefill was supposed to move. `prefill_control.sh` is the
correctness half: it compares the batched prompt's TEXT against the token loop.

### The default reproduces it with nothing set, and 48/64/80 still were not asked

```
  M axis: w, no environment variable at all
     m   worst rel    rows that agree
     2   5.10e-05      226 of 226
     4   5.10e-05      452 of 452
     8   5.78e+04      871 of 904
    16   5.10e-05     1808 of 1808
    32   5.10e-05     3616 of 3616
```

`charsiu_acc_index` taking the format is enough: the width arm is exact with
nothing set and the height control is unchanged and still wrong. Same numbers
as the hand-set round, m = 8 included, so it is reproducible rather than a
reading.

⚠⚠ **AND THE THREE UNTESTED WIDTHS ARE STILL UNTESTED.** MS[] was widened to
2, 4, 8, 16, 32, 48, 64, 80 and the board script was still passing
`--batch-probe 32`, which is the cap. The round that existed to reach 48, 64
and 80 stopped at 32, and its own header said 32 while the reason for running
it said otherwise. Changed one half, left the other -- the same shape of
mistake as the `head` caps.

Both ends are fixed and instrumented rather than just corrected: the script
passes 80, and the probe now PRINTS the widths it is about to run next to the
cap it was given, so the two cannot disagree in silence again.

### m = 8 will name itself next time

871 of 904 is 33 rows. Not a whole tensor, not a whole row index across
tensors, and a count cannot tell those apart. The probe names the first eight
misses now -- tensor, k, n, which row of which m, and that row's own worst
relative error -- which separates "one shape", "one row index" and "scattered"
in one round.

## 🏁 THE READ ORDER IS SOLVED. w4a16 batches exactly, 2 to 32

`roleswap2`, on the board:

```
   m    worst rel    rows that agree
   2    5.10e-05      226 of 226      every tensor, every row
   4    5.10e-05      452 of 452
   8    3.08e+04      871 of 904      <- the one width that bends
  16    5.10e-05     1808 of 1808
  32    5.10e-05     3616 of 3616
```

Four widths exact to 5.1e-05, which is float summation order and nothing else.
The prediction that named it -- that roleswap is right only on rows 0 and m-1
if roleswap2 is the truth, giving 226 everywhere -- held at every width it was
made for.

### And the read order depends on the FORMAT, not only the axis

Both halves of that came off the board:

```
  int8,  height axis   a * 4       exact to m = 80, the tower sweep
  int8,  width axis    a * 4       its RAW surface is identical to the
                                   height one, mapped cell for cell
  w4a16, width axis    roleswap2   2048 of 2048 on every row
  w4a16, height axis   neither     row 1 is not written at all
```

So `charsiu_acc_index` takes the format now and w4a16-on-the-width is the one
case that reads differently. Every other caller gets exactly the expression it
had, checked on the desktop at m = 2, 4, 8 and 32. `CHARSIU_ACC_A` still
overrides, which is how the two readings were told apart.

The whole chain, and not one link of it shows at m = 1 -- which is why decode
ran for hundreds of rounds without meeting any of it:

1. M on the **width** axis, not the height
2. **0x40b8 = 3 * M**, so the surface is not short
3. **`a` and the row trade places**, and the row splits as `mi/2`, `mi%2`

⚠ **m = 8 IS THE ONE THAT BENDS AND NOTHING HERE EXPLAINS IT.** 871 of 904,
worst 3.1e+04, where 4 and 16 either side of it are exact. It has been four to
six orders out in every arm of every round since this began. The refusal stays
until something explains it.

⚠ **AND 48, 64 AND 80 HAVE NEVER BEEN ASKED.** The probe swept 2 to 32 and a
real prompt hands it chunks up to `CHARSIU_NPU_ROWS_MAX`, which defaults to 64.
Its widths are 2, 4, 8, 16, 32, 48, 64, 80 now. Flipping the default before
those are measured would be shipping three untested widths on the strength of
five tested ones.

## 🏁 w4a16 BATCHES. m = 2 is exact, and it was the read order all along

```
  M axis: w, CHARSIU_ACC_A=roleswap
    row0 in place: 2048 of 2048 channels agree, the whole row
    row1 in place: 2048 of 2048 channels agree, the whole row
    4096 of 4096 wanted values are somewhere in the batch

     m  worst rel   rows that agree   one row   batched  speedup
     2   5.10e-05     226 of 226       118 ms    77 ms    1.53x
```

**Five rounds established that w4a16 computes exactly one row. It computes as
many as it is asked for; this tree was reading the answer out of the wrong
slots.** Worst relative error 5.10e-05 on a float sum of two thousand terms in
two orders, every tensor, both rows.

The whole chain, in order: M on the width axis rather than the height, 0x40b8
counting M so the surface is not short, and then `a` and the row trading places
in the read order. None of the three shows at m = 1, which is why decode never
saw any of it.

### Above m = 2 it is 226 rows exactly, which names the last ambiguity

```
   m     rows that agree
   2     226 of 226      <- solved
   4     226 of 452
   8     194 of 904
  16     226 of 1808
  32     226 of 3616
```

226 is 113 tensors times TWO ROWS, at every width. That is the ambiguity this
was shipped with and it landed exactly where it was flagged: once `a` takes the
32P block the row has two slots left, one of stride 8 with P values and one of
stride 4 with 2, and **at m = 2, P is 1 and the stride 8 slot is a singleton, so
the two readings are the same function.**

`roleswap2` is the other reading -- `mi/2` at the 8 and `mi%2` at the 4, where
`roleswap` has `mi%P` and `mi/P`. If it is the truth then roleswap is right
exactly where they coincide, which is rows 0 and m-1 and nothing else:

```
   m      shared rows   predicts       board
   2      0, 1          226 of 226     226 of 226
   4      0, 3          226 of 452     226 of 452
  16      0, 15         226 of 1808    226 of 1808
  32      0, 31         226 of 3616    226 of 3616
   8      0, 7          226 of 904     194 of 904   <- the one miss
```

Four of five, and it is identical to roleswap at m = 2, so it cannot lose what
is already won. It is a permutation at m = 2, 4, 8, 16, 32 and 80.

⚠ **m = 8 IS A SEPARATE FAULT AND THIS DOES NOT EXPLAIN IT.** Its worst
relative error is four to six orders out in EVERY arm of every round -- 1.3e4,
3.2e4, 2.9e5, 9.7e3, 7.5e4, 1.7e5, 4.7e4 -- where m = 4 and m = 16 sit at 1e3.
Something is wrong at that one width, it has been wrong at it all along, and
194 of 904 is not what either reading predicts.

### What charsiu emits, against what they emit

`tools/cmp_vendor.py` now diffs the emitter that actually runs -- `job.c`, not
the geometry-only `regcmd.c` nothing else calls -- and merges the DPU stream
the vendor puts right behind each CNA one. Same shape, `ic=2048 oc=1024 int4`:

```
             M=1   M=16  M=32  M=48  M=64  M=80    registers differing
  height      3     14    12    14    12    16     (today's default)
  width       3      5     2     5     2     5     (CHARSIU_M_AXIS=w)
```

At M = 32 and 64 the width axis is **two registers** from the vendor's stream --
fewer than at M = 1, where this hardware is known to be right. The two are a
flag that alternates between the vendor's own streams at a fixed M, and one
DPU register that does the same.

Two differences were real and are fixed:

- **`0x118c` is M-1 in BOTH halves** on the width axis, not width and height.
  Exact on all 3328. Round 380 set this register from the vendor's file and the
  board said it changed nothing -- that round ran on the height axis, where M
  moves nothing at all, so it did not test this.
- **the split CBUF pair** when the input surface passes 4096 atoms. Read
  symmetrically across the whole file, "more than 4096 implies split" holds on
  all 8692 streams; the converse fails on 240 fp16 ones, and int8 has no
  evidence either way because its 40 streams never reach the threshold. So it
  is scoped to the width axis, and every stream that runs today is bit
  identical to before -- checked rather than argued: the emitter from before
  the change and the one after it were built side by side and asked for 1200
  shapes on the default axis (2 weight formats x 2 activations x 5 K x 5 N x
  12 M), and **0 differ**. On the width axis the same pair differs at all 40
  shapes with M > 1 and at none of the 8 with M = 1, which is the control --
  a change that touched nothing would have passed the first check too.

⚠ **NONE OF THIS HAS BEEN ON THE BOARD.** `tests/board_w4_axis.sh` is that
round: the height axis first as a control -- it must fail, or the probe is not
discriminating -- then the width one, checked row by row before anything is
timed. `CHARSIU_NPU_W4_BATCH=1` lifts the refusal and only together with
`CHARSIU_M_AXIS=w`, because an int4 batch on the height axis is the wrong
answer at 37 tok/s and this tree has already shipped that once.

### And then the board contradicted the fence

Grouping q, k and v into one submit per row is fewer fences -- round 321 put the
fence at 94% of the hardware path -- and it was SLOWER:

```
  tensor major (three calls)   38608 submits   TTFT 5239 ms
  grouped (matvec_pair)        35528 submits   TTFT 6302 ms
```

Eight percent fewer submits and twenty percent slower. The only reading that
fits is that **consecutive submits of the same weight do not pay for it twice**,
and grouping threw that locality away to save a fence. Tensor major is the
default now and `CHARSIU_PREFILL_GROUPED=1` restores the other, so the two can
be compared in one session on one board rather than across two.

⚠ Two runs on a warming board is a reading, not a result.

### Which weight format, answered

Four numbers off the board, same model, same prompt, same 16 generated tokens:

```
              int4     int8
  prefill    19.24    26.60 tok/s
  decode     15.46     9.16 tok/s
```

int8 is ahead on prefill and behind on decode, so it depends on the shape of
the work. For a prompt of P tokens and G generated,

  int8 wins  when  P * (1/19.24 - 1/26.60) > G * (1/9.16 - 1/15.46)
             i.e.  P > 3.1 * G

**So int4 stays the default.** Chat is a short prompt and a long answer and int4
wins it outright; int8 is for prompt-heavy work -- summarising a document,
retrieval -- where the prompt is several times the answer.

⚠ The batched prefill helps int4 too, and not because the matmul batches: it
refuses there. It is that a prompt needs logits for its last token only, so the
head is skipped n - 1 times whatever the format.

That was written here as an inference for a fortnight. It is now measured.
`tests/prefill_control.sh` runs batched -> control -> batched on the same
binary, one flag apart, and the two batched samples bracket the control so a
board that warms over a minute cannot be mistaken for the flag:

```
  control   65 tok / 4304 ms   66.22 ms a token   15.10 tok/s
  batched   65 tok / 3498 ms   53.82 ms a token   18.59 tok/s   (18.63, 18.54)
                               ---------
  saved                        12.40 ms a token
```

⚠ AND THE SAVING IS THE OUTPUT HEAD, TO WITHIN 1.4%, BY AN ARITHMETIC THAT
NEVER SAW THESE TIMINGS. The head runs once instead of 65 times, so the saving
per token is H * 64/65 and H is 12.59 ms. Llama-3.2-1B's head is 128256 x 2048,
which at int4 is 131.3 MB of weights, and

  131.3 MB / 12.59 ms = 10.43 GB/s

against the 10.58 GB/s this model's own NPU summary reports for weight
bandwidth. The time the batched prompt does not spend is exactly the time it
takes to stream the head's weights, once per token, at the rate this board
moves weights.

⚠ And the control says what the baseline was: 15.10 tok/s prefill against a
decode of 15.70. Without batching a prompt token costs what a generated one
costs, which is where this started.

⚠ It is a second copy of the layer loop and it is deliberately blind. It
handles the plain case and REFUSES the rest -- gemma3's window and two rope
bases, gemma4's per layer embeddings and shared KV, qwen3's q and k norms,
phi3's fused K and V, biases, post norms, softcaps -- and the caller falls back
to the token loop, which is correct for all seven architectures and merely
slower. Refusing is not an error path, it is the other half of the same
decision.

⚠ SO PHI-3.5 HAS NO BATCHED PREFILL, and that is a real gap rather than a note.
Its K and V arrive as one fused tensor, llama_prefill_batch refuses on the
first layer, and its prompt costs 4.96 tok/s -- what a generated token costs.
Splitting a fused qkv into three views is the whole of what it would take. It
has not been done because nothing had measured what it was worth until the
control above put a number on the head skip, and 12.4 ms a token is what phi3
is leaving.

⚠ And a refusal has to be AUDIBLE. batch_ok used to return 0 and the caller
fell back in silence, so "the flag did nothing" and "this model was never
batchable" looked identical from outside -- which is how the first attempt at
the control above spent four minutes on Phi-3.5 and produced 4.96, 5.00 and
5.10 tok/s, three numbers that agree and mean nothing. llama_batch_why_not
returns the reason as a phrase now and every run prints one line saying which
path its prompt took.

⚠ And it runs without the NPU, which is what makes the risky half checkable off
the board: matmul_rows falls back to matvec, so a host with no hardware
compares the two layer loops and nothing else.

### What it took, and the shape of every wrong turn

Five rounds went into establishing that the int4 path cannot do this at all.
w4a16 produces exactly ONE row: fed the same activation twice, row 1 comes back
matching row 0 in 1 of 2048. The DPU and RDMA blocks are identical to a stream
that does two rows, all 69 and all 22 registers, and every CNA word that
differs was put back one at a time with a liveness check between them.

⚠ **ALL FIVE RAN ON THE HEIGHT AXIS**, and the sentence that used to end this
paragraph -- "the vendor never batches a weight matmul, so there is no M > 1
int4 stream to copy" -- was false. There are 2816 of them in one file, one row
high and M pixels wide, and the largest is 80.

Two registers on that path were literals chosen at M = 1, the one width where a
value that follows the row count cannot show that it does: 0x40b8's 3, which is
3 * rows, and 0x301c's half, which is zero either way at m = 1.

The read order is `P = m/2`, super groups of 32, rows pairing P at a time with
the four word runs alternating. Solved from two printed maps and confirmed at a
width it was not fitted on.

## 2. The output head: routed, and it was worth 49 ms a token. DONE

gemma3's head is 262144 wide and was spending 49.8 ms of a 113 ms token on the CPU.
It is on the NPU now, and the tokens are identical:

```
                    before      after
  output head       49.81 ms    11.53 ms      44.2% of the token -> 15.6%
  a token          112.7  ms    73.9  ms
  generation         8.76 tok/s 13.07 tok/s
  tensors staged      182         183
  slices              468         532
  weights           6.16 GB/s   7.18 GB/s
```

**It was not the coefficient bound**, which is what this section said for four
rounds. The shipped runner has passed `CHARSIU_COEF_ELEMS=65536` all along, putting
the head's coefficients at 10.5 MB. What refused it was `maxn`, defaulted to 131072
to clear Llama 3.2's 128256 vocabulary against gemma3's 262144, and two things kept
that invisible: the gate was the only refusal in the runtime that did not say why,
and the round that set `CHARSIU_NPU_MAXN` on the command line did not set it, because
`charsiu run` built its environment from config.ini and handed it to `env NAME=VALUE`
in front of the caller's.

Prefill is also a real number now rather than a slander: staging is built lazily
inside the first forward pass, so all 15 seconds of it were charged to the prompt.
Reported separately, the same run reads `staging 15081 ms | prompt 6 tok in 829 ms,
7.24 tok/s` where it used to read 0.92.

⚠ One number to keep an eye on. 151 MB of int4 head in 11.5 ms is 13.1 GB/s, which is
above the 10.8 GB/s this board's DRAM roof was measured at. Either the roof is higher
for this access pattern or the stage timer is not charging the head everything it
costs. The tokens are right either way, so this is a measurement question.

The coefficient bound is still a guess and still unmeasured; `npu_gemm_test K N
--coef` walks it down. It is no longer in the way of anything.

## 2b. gemma3 runs the NPU at 6.16 GB/s where llama reaches 10.3

Same board, same int4 path, and the difference is the shape. n_embd 1152 does not
divide the 1024 K slice, so q, k, v, gate and up each split 1024 + 128 and the second
slice carries a ninth of a slice of work for a whole task's cost. 468 slices where
338 would do, five of the extras in every one of 26 layers.

`CHARSIU_NPU_KFIT=1` gives the last slice the remainder instead, so 1152 is one slice.
Ungrouped tensors only: a grouped tensor's scale gather reads one group per slice.
Off by default, unrun.

**Control**: the same prompt with and without it in one boot. Identical tokens is the
correctness bar, since acc_out sums int32 across K slices and any split of the same K
has to give the same accumulator; slices and GB/s in the NPU report are the win.

## 3. Upstreaming

- The driver is [PATCH v9](https://lore.kernel.org/all/cover.1787568658.git.gahing@gahingwoo.com/),
  with an Acked-by on both dt-bindings, a Reviewed-by on both pmdomain patches and a
  Tested-by on the two reset-race patches. v10 is tag collection and the git note that
  v9's cover letter promised and did not carry.
- The Mesa side is [mesa!43804](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/43804).

## What would make this stop

Stated in advance, so it is a decision rather than a slow fade:

- if prefill cannot be batched after a fair attempt, that is a written negative result
  and decode stands on its own -- it already beats the vendor;
- if the output head routes and gemma3 does not get faster, the head belongs on the
  CPU on this board and gemma-shaped models are simply expensive here, which is worth
  saying rather than working around.
