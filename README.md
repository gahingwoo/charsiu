# charsiu

An open LLM runtime for the **RK3576 NPU on a mainline Linux kernel**, driving the
hardware through the mainline `rocket` DRM-accel driver with no vendor userspace in
the execution path.

**Status.** Two halves, and only one of them touches the NPU.

The NPU half computes a signed int8 matmul, **byte exact** against a CPU reference. It
opens `/dev/accel/accel0` through the mainline `rocket` driver, packs the operands into
the hardware's tile layouts, builds the coefficient buffer, emits the register stream
and submits it. No Mesa, no vendor runtime, nothing borrowed at run time.

The other half is **a complete decode loop on the CPU, with no NPU in it at all**, and
it runs a model. `charsiu_run` reads a gguf, tokenizes, and generates text. It exists to
be the oracle: a run of it is a known-correct sequence of tokens, and every later version
that moves a matmul onto the NPU is diffed against it rather than against a guess.

```
$ charsiu_run Llama-3.2-1B-Instruct-Q4_0.gguf -p "The capital of France is" -n 24
The capital of France is Paris. The capital of Germany is Berlin. The capital of
Italy is Rome. The capital of Spain is Madrid.
```


Measured on a ROCK 4D, 2026-08-15, every value identical to the reference rather than
close to it:

| probe | shape | result |
|---|---|---|
| dense | M=1 K=64 N=64 | 64 of 64 bytes exact |
| negative MAC, output -66 to +63 | M=1 K=64 N=64 | 64 of 64 bytes exact |
| bias ramp | M=1 K=64 N=64 | 64 of 64 bytes exact |
| impulse | M=1 K=64 N=64 | 64 of 64 bytes exact |
| dense, a projection's shape | M=1 K=512 N=1024 | 1023 of 1024 exact, but see the note |
| dense, **many rows** | M=224 K=64 N=64 | 14313 of 14336, none off by more than 1 |
| dense, a whole 56x56 surface | **M=3136 K=33 N=64** | 200344 of 200704, 51 off by more than 1 |

**The rows above M = 1 were wrong for 36 rounds and the hardware was not.** The
output surface is `[n/atom][m][n%atom]`, the mirror of the input's, and at M = 1
that expression collapses to exactly `n`. So a row major reading was right at one
row and only there, and every correctness run this project had done was at one
row. Nothing about the job changed when this was found: same register stream,
same packing, same buffers, only which byte the checker reads.

**Which of these rows is evidence, and which is not.** A matching channel is not a
computed one: if the reference is flat, both sides can hold the same common value
without a single multiply having been right. Computed with the tool's own
cpu_reference() at M = 1, K = 64, N = 64:

| probe | distinct reference values | range | reads as evidence |
|---|---|---|---|
| negative MAC | 64 | -66 to 63 | yes |
| bias ramp | 64 | -48 to 53 | yes |
| dense | 5 | -2 to 2 | no |
| impulse | 5 | -2 to 2 | no, until the scale fix below |

The K=512 row is the same problem again, 7 distinct values across 1024 channels.

This file used to say the four 64 wide rows were all strong evidence because they
had 64 distinct reference values. Two of them do. The other two put the whole
reference inside four counts, which is under the tool's own TOO FLAT TO JUDGE A
MATCH threshold, so those two 64 of 64 results were never readable.

It matters which two. The two that hold are the ones that exercise the output
stage, one walking the MAC through zero and one holding it at zero and walking
the bias. The two that do not are the ones that exercise the WEIGHT LAYOUT. So
the layout evidence is the weaker half, which is the opposite of what this file
implied.

The impulse now sets its own weight scale, the way the int4 probes learned to in
rounds 170 and 176, and its reference spans 62 values across -115 to 115.
CHARSIU_I8_IMPULSE_SMALLSCALE restores the old setting as a control. The dense
probe has the same problem and has deliberately not been touched in the same
change, so that whichever result moves can be attributed.

The two things that had to be understood to get there are worth stating because both
were wrong in this file before:

- **both operands are signed bytes.** The weight is stored biased by `-0x80` and so is
  the input; storing the input raw makes a byte of 168, meaning +40 against a zero
  point of 128, arrive as -88.
- **the output stage is** `out = clamp(max(requant, 0) + offset, -128, 127)`, an int8
  with a floor under it. The floor is a real fused ReLU and it does not need to be
  switched off: lifting the accumulator clears it, and the offset takes the same
  amount back.

  ⚠ **The lift is the output zero point, not 128.** 128 is right here only because
  charsiu's own zero point is 0, which makes its offset `-128` already. Reading this
  as a constant is what took the driver project a board round to undo: it lifted by
  128 on a tensor whose zero point was 128, where the offset is 0 and nothing takes
  the lift back, and the whole surface railed at the top. `out_zp` is the smallest
  lift that clears the floor, and with it the offset is a constant `-128` and the
  expression collapses to `clamp(requant + out_zp, 0, 255)`, which is what the
  operation means.

It is not a runtime yet. What is left:

- **no model runs.** The matmul is correct; the KV cache, the sampler, the per token
  geometry and the CPU/NPU split are not written.
- the reference still requantises in float where the hardware uses an integer scale and
  shift. It agrees to the byte on everything measured so far, which does not mean it
  will at every scale.
- **int4 does not compute end to end yet**, though what it does is no longer a
  mystery. The layout is read, the `w4a16` stage is ported, and the output is
  `((int16)fp16bits(w) * (int16)fp16bits(a)) >> 16` exactly, 18 measured points with
  no error and a four point prediction written into the board script before the run.
  What is missing is the layout for `k = 16` and above, whose bytes are dead in the
  map with no data about them. See below.

The three tools that got it there are in the repository rather than in a shell history,
because every step that worked was a diff against something known to compute:
`tools/rkllm_regcmd.py` reads the vendor's own dispatches out of a `.rkllm`,
`tools/cmp_vendor.py` diffs charsiu's stream against them, and `tools/emit_job.c`
prints charsiu's stream on a desktop so a change to it costs no board round.

## The decode loop, and how it is known to be right

`src/gguf.c`, `src/tokenizer.c` and `src/llama.c` are a gguf reader, byte level BPE, and
a Llama forward pass: RMSNorm, RoPE, grouped query attention over a KV cache, SwiGLU,
and argmax. f32, f16, q4_0, q4_1, q8_0 and q6_K weights are read. Every matmul goes
through one call, `gguf_matvec()`, which is the single place a projection moves to the
NPU from.

Coherent output is not evidence. A tokenizer with a wrong pre-tokenizer and a forward
pass with a subtly wrong RoPE both produce fluent English. So both halves are diffed
against a reference implementation, and both diffs found real defects.

**The tokenizer**, against huggingface's own implementation: 94 hand written cases
covering each branch of the pre-tokenizer alternation, plus a fuzzer drawing from
sixteen Unicode blocks. **3094 cases, 0 mismatches** (`tests/tokenizer_cross.py`). Two
things it refuted:

- `\s*[\r\n]+` runs to the **last** newline of a whitespace run, not the first. `\s*` is
  greedy and gives back only what `[\r\n]+` needs, so `"\n \n"` is one token. Reading it
  as "spaces, then a newline" split it in two and every id after it moved.
- `\p{L}` is a category, not a range of blocks. U+05CF is Hebrew punctuation, and calling
  it a letter put a leading apostrophe on the wrong side of a cut. The classifiers now
  binary search real Unicode tables generated by `tools/gen_unicode_tables.py`.

**The forward pass**, against llama.cpp on the same file: top-8 logits and a 48 token
greedy continuation, over ten prompts covering prose, code, arithmetic, SQL and Chinese.
**10 prompts, 0 failures, worst logit delta 0.01027** on logits near 16, and all 480
greedy tokens identical (`tests/forward_cross.py`).

That comparison has to be run on an **f32** model, which is the part worth writing down.
On q8_0 the same test showed a delta of 0.14 and the text diverged after twelve tokens,
and neither side is wrong: llama.cpp quantises the **activation** to q8_0 before the dot
product, and charsiu keeps it in f32. Taking the quantisation off both sides collapsed
the disagreement by a factor of thirty, which is what makes that an explanation rather
than a story.

### The CPU baseline is meant to be honest

A slow CPU path would flatter the NPU, and the measurement this whole project turns on
is a comparison against four Cortex-A72 cores. So the dot products are NEON, and
deliberately **ARMv8.0 only**: no fp16 arithmetic, no SDOT. Writing them against this
development machine's feature bits would produce a kernel that runs here and not on the
board it is a baseline for. `CHARSIU_NO_NEON` builds the portable versions, and
`tests/neon_control.sh` requires the two to produce the same tokens.

One thing that cost a day and is worth stating plainly: **a q4_0 file is not all q4_0.**
llama.cpp quantises `token_embd` to q6_K even there, and Llama 3.2 ties the output head
to it, so at a 128256 vocabulary that one tensor is a fifth of the weights and the
biggest matmul in a decode step. It was also the one type whose vector path had been
removed to silence a spurious compiler warning, and a q4_0 model ran at **half** a q8_0
model's speed while reading half the bytes.

| | before | vector q6_K over a dequantised buffer | dot folded into the unpacking |
|---|---|---|---|
| q4_0, 6 threads | 9.21 tok/s | 12.06 | **18.70** |

The first guess had been register spilling in the q4_0 kernel. Rewriting it changed
nothing, which refuted it, and the answer turned out to be in the file rather than in the
code.

For scale, llama.cpp on the same host and file is 48.3 tok/s. It quantises the activation
to int8 once per matvec and then does integer dot products; charsiu converts each weight
to f32 instead. That gap is the next structural piece of work, and it is also exactly the
interface the NPU path needs, since the NPU consumes an int8 activation with a zero point.

## What it costs, and what the cost is OF

Measured on the same board, 2026-08-15. A submit carries jobs, a job carries tasks;
tasks in one job are chained on a single core with no further ioctl. Sweeping seven
shapes at 32 chained tasks and fitting all of them at once:

```
us per task = 26.3 + weight_MB * 84.3        i.e. 11.9 GB/s, plus 26 us per task
```

with every point inside 10% of that line and most inside 4%. **The cost is the weight
fetch.** Three things say so and each of them could have said otherwise:

- **M is nearly free.** The same 2.10 MB of weights costs 201.9 us at M = 1 and
  217.6 us at M = 32, which is 1.08 times the time for 32 times the arithmetic.
- **the same bytes in different shapes cost the same.** K=1024 N=1024 and K=2048 N=512
  are both 1.05 MB and came out 111.56 and 111.24 us, 0.3% apart. K=2048 N=1024 and
  K=1024 N=2048 are both 2.10 MB, 208.06 and 211.21 us, 1.5% apart. That test was run
  to break the reading above and did not.
- **a second core does not help.** Two jobs of eight tasks were about 5% *worse* than
  one job of sixteen at every shape, which is what a bandwidth bound workload does.

On top of the per task cost sits about 172 us per submit, which chaining removes. That
is why a 0.5 MB projection gains 4.4x from batching and a 2 MB one only 1.9x.

### What that means for a token

Llama-3.2-1B reads about 973 M projection weights per token, once each, so decode is
DRAM bound and not MAC bound:

| weights | bytes per token | time | tokens/s |
|---|---|---|---|
| int8 | 973 MB | 85 ms | **11.8** |
| int4 | 487 MB | 44 ms | **22.7** |

The vendor ships about 13 tokens a second on this board. **So int4 is not a
nice-to-have, it is the only 2x available**, which is why it moves to the front of the
queue despite its layout still being unconfirmed here.

Prefill is a different machine entirely: at M = 32 the same weights are amortised over
32 rows, 604 GOP/s and 6.9 us per row against 200 us per row at M = 1.

### The honest denominator

The same matmul on one CPU thread, a naive scalar loop, takes 5078 us against the
NPU's 201 us. That ratio is **not** 25x in any useful sense: the loop has no NEON in
it, and more importantly the CPU has to read the same 2.10 MB, so a tuned kernel would
run into the same wall from the other side. What the measurement does settle is the
RK3588 stacks' conclusion that a single row matmul belongs on the CPU. On RK3576 it
does not.

## Reading a weight layout off the hardware

A weight layout used to be inferred here from whether an output came out right.
It can be asked directly instead: put **one** live weight in the whole buffer, sweep
it, and record which output channel lights. `tools/charsiu_int4.c --map` does that,
and the point of it is that it was validated against int8, which is byte exact on
this silicon:

```
int8   K=64 N=64, 4096 bytes    512 of 512 probes light exactly one channel,
                                no dead region, n = byte / 32, k = byte % 32
```

which is exactly what `src/regcmd.c` packs. The instrument agrees with a case whose
answer is already known, which is what makes its answer on int4 worth anything.

`--kpair` goes further and is sparse on **both** sides: one live nibble, a one hot
input, sweeping k. Every nibble pairs with exactly one k, so nothing is broadcast and
the pairing can be read with no sum to unpick.

### What int4's layout turned out to be

```
channel n is read from byte (n / 32) * 512 + (n % 32) * 8, eight bytes
byte b nibble h is k = 2b + h
an activation element is TWO bytes wide
```

Each part measured, then confirmed by a second and different probe. The element width
was found the hard way: with an 8 bit input packing every nibble paired with
`k = 2 * k_ours + 1`, and with a 16 bit one it pairs with `k` exactly.

That last also retires a reading this project carried for eight rounds. The hardware
appeared to fetch only a **quarter** of the weight buffer; it never did. It fetches
half as many elements, each twice as wide.

### The w4a16 stage, and what the output actually is

The vendor never runs int4 against an int8 activation. It runs `w4a16`, and that path
has its own DPU output stage **and** its own RDMA coefficient fetch, both of which live
in vendor streams separate from the 3328 int4 convolution streams, which carry no DPU
and no RDMA registers at all. Both are ported. The line worth writing down is:

```
0x40ac = 0    0x40b0 = 1    0x40b4 = 0        an IDENTITY requant
```

`w4a16` does not requantise. This file used to say "so the output is a float", and that
was an addition rather than a measurement. An un-requantised output is the raw
accumulator, and an accumulator is an integer. Read as halves the outputs came back in
a period of four, with `0xffff` and `0x0000` filling every second slot; paired up
little endian they are 32 bit integers of alternating sign:

```
[94 f1][ff ff] -> 0xfffff194 = -3692
[a4 10][00 00] -> 0x000010a4 = +4260
```

### Reading the output layout the same way

One live nibble in the entire weight buffer, one byte at a time, all 2048 of them, the
activation held at a single constant so nothing in the output can track an activation
index, and the output bytes dumped with nothing assumed about how wide an element is:

```
byte b feeds output slot b / 8       eight consecutive bytes per slot
each slot is fed by 16 bytes         two runs of eight, 256 apart
640 bytes light something            1408 light nothing, and the entire
                                     second half of the buffer is dark
no byte lights more than one slot
```

The control, an all zero weight buffer, lights nothing.

⚠ Round 266 put the third line in doubt and it was wrong to. `--kpair` swept
bytes 1024, 1536 and 1920, predicted words 32, 48 and 56 for them, and all three
lit nothing. **The second half of the weight buffer really is dark.** What is
confirmed instead is that the boundary sits at 1024 rather than at 512: bytes
512, 640, 768 and 896 all light, at words 16, 24, 16 and 24.

### What the output actually is, exactly

`out = ((int16)fp16bits(w) * (int16)fp16bits(a)) >> 16`, **18 of 18 measured
points exact**, both negative nibbles included. The hardware multiplies the two
operands' fp16 **bit patterns as signed 16 bit integers** and shifts right 16.
charsiu packs a genuine fp16 and the hardware never reads it as a float.

Getting there needed a confound broken first. Doubling an input ADDS a constant
to the output rather than scaling it: +284 for the activation three times over,
+288 for the weight twice, exact each time. So the output is linear in log2 of
its inputs, which is why changing a nibble from 7 to 3 scaled the result by
0.930 instead of 3/7 and why no line ever fit two points of it.

Two readings then fit everything measured, and they separate only on values
that are not powers of two, by 20 to 24 counts. The predicted numbers for both
went into the board script before the run:

```
   a      fp16 bits   integer reading   true logarithm   measured
  1.5         15872              4402             4426       4402
  3.0         16896              4686             4710       4686
  5.0         17664              4899             4919       4899
  6.0         17920              4970             4994       4970
```

Four for four on the integer reading, zero error.

## What int4 buys, measured

```
                    int8 us/task   int4 us/task
K=224 N=160            22.7           18.4        -19%
K=128 N=160            16.0           14.9        - 7%
K=224 N= 64            14.7           13.3        -10%
K=2048 N=1024         201.5          cannot run this shape
```

⚠ **Not 2x, and the reason is that the envelope never reaches the regime where
halving the bytes would pay.** At 0.04 MB of weights the achieved bandwidth is
1.58 GB/s; at 2.10 MB it is 10.41. These shapes are latency bound, not bytes
bound, so int4's halved weight traffic buys almost nothing.

⚠ **The twelve-times figure that first followed from that was wrong, and the K
ceiling was charsiu's own guard.** The layout works at every K tried up to 2048:
288, 384, 512, 1024 and 2048 all give `wrote 40 of 64, exact 32` at N = 64, and
K = 512 at N = 16 is 16 of 16. `exact` is `N/2` at all of them, independent of K
— a working half-width job, not a failure. Benched at K = 512, N = 64: int8 24.6
us against int4 16.7, **int4 by 32%**.

So the arithmetic is per job. An int4 job at a declared `K` and `N` gives `K/2`
real k a channel and `N/2` usable channels:

```
int4   NK/4 MACs for NK/8 weight bytes fetched   2 MACs a byte
int8   NK   MACs for NK   bytes                  1 MAC  a byte
```

int4 really is twice the work per byte, which is what a 4-bit weight should buy.
It pays in **jobs**: four times as many for the same work, each with a fixed
cost.

⚠ **Measured at the shape that matters, it does not survive that.** At
`K = 2048, N = 256` a single job is **222.7 us for int4 against 224.2 us for
int8** — half the weight bytes, the same time. int8 is genuinely bytes-bound
there, 9.14 GB/s marginal; int4 moves 0.26 MB in the same 223 us, which is
1.18 GB/s. The 2x per byte is real on paper and none of it reaches the clock.
And chaining hangs: the second task of that shape times out and takes the block
with it.

The layout itself is fine that far out — `K = 2048` at `N` of 64, 128, 160 and
256 all give `wrote = N/2 + 8` and `exact = N/2`, so 128 usable channels a job at
`N = 256`. `N = 512` and above were recorded as failures and were **this packer's
own group-count guard**, the same mistake as the K whitelist that had `K = 192`
down as a layout fault for two rounds.

## An int4 matmul that computes

**Round 280: `int4 output: 64 of 64 words written, 64 EXACT`.** Every channel,
against `cpu_reference`, on a dense buffer. The pieces:

```
charsiu_pack_weights writes the measured layout   address table + k parity
CORE 0x3020 = 2*(n-8) - 1                          111 at n = 64
weights zeroed outside each channel's fed half     CHARSIU_W4_HALFK
```

The controls behaved. `CHARSIU_INT4_ORDER=1` scored 0 of 64, so the intra group
order is settled at `byte j holds k = 2j low and k = 2j+1 high`. The no-`HALFK`
control scored 0 of 64 **with npu values identical to the passing arm**, which is
the model confirming itself: the hardware computed the same thing and only the
reference moved. int8 stayed 64 of 64 byte exact throughout.

Both shift placements scored 64, and that is not a weak test. fp16 bits of an
integer have a zero low byte and `abits` is `(a-0x80) << 8`, so every product is
a multiple of 65536 and the shift loses nothing per element. **The two readings
are identically equal for this packing** rather than undecided.

### The base table is one expression at both K

```
K=64  0 128 512 640 1024 1152 1536 1600
K=32  0 128 256 384  512  640  768  832

base[g] = (g/2) * 8K + (g odd ? (g == 7 ? 64 : 128) : 0)
```

Sixteen points, one expression, and the last pair sitting 64 bytes in rather than
128 appears in **both** tables independently, so it is read and not fitted. It is
not a cap either: `3*512 + 128 + 56` is well inside 2048.

Both hold on hardware with **no override at all**: `K = 64` and `K = 32` at
`N = 64` are each 64 of 64 exact with `0x3020` emitted as `2*(n-8) - 1`.

⚠ `N = 32` and `N = 16` came back 24 of 32 and 8 of 16, short by exactly one
group of eight each. "The last pair" was written as `g == 7`, which is the last
group only at `N = 64`; at 32 it is 3 and at 16 it is 1, and those are the sizes
of the shortfalls. So the irregular term belongs to the highest group in use,
`g == (n-1)/8`.

**Confirmed.** Every `N` whose highest group is odd is now exact, with every
group at 8 of 8:

```
N = 16  16/16      N = 32  32/32      N = 48  48/48      N = 64  64/64
```

32 and 16 were 24 and 8 before the fix, and 48 had never been run and was exact
first time.

### The write quantises to sixteen channels

The `N` with an even highest group — 24, 40 and 56 — failed, and **not on the
packing**. Whole groups came back 0 of 8 at the top and the count of words
*written* was short by the same amount:

```
n      16  24  32  40  48  56  64
wrote  16  16  32  32  48  48  64      = floor(n/16) * 16, seven for seven
```

So `0x3020` asks for the next multiple of sixteen,
`2*(ALIGN_UP(n,16) - 8) - 1`, and the extra channels are computed and thrown
away. That fixed the write: 24, 40 and 56 now report every word written.

### The pair stride was only ever measured at N = 64

The same three `N` are **still wrong**, and with the write fixed that is now a
placement fault: whole groups at 0 of 8 at the top, `g2` at `N = 24`, `g3` and
`g4` at 40, `g4` to `g6` at 56.

The base expression used a pair stride of `8*K`. The weight buffer is `k*n/2`
bytes, so at `N = 64` it is `k*32`, and **at `N = 64` `8*K` and `wbytes/4` are the
same number**. Both measured tables were swept at `N = 64`, so no data in this
project can tell the two forms apart, and the `N` sweep is the first thing that
could. It says the map scales with the buffer: at `N = 24` the second group of
`g2` lands at byte 768, which is exactly the size of that buffer.

`wbytes/4` was tried and is **refuted**, and not narrowly. It did not fix the
three that were wrong and it broke the three that were right:

```
            8*K      wbytes/4
N = 16    16/16          0/16
N = 32    32/32          0/32
N = 48    48/48          8/48
N = 64    64/64         64/64     equal here by construction, so proves nothing
N = 24    16/24          0/24
N = 40    24/40          0/40
N = 56    32/56          0/56
```

So `8*K` is right everywhere there is data and the failures at 24, 40 and 56 are
something else. The reasoning that produced `wbytes/4` was tidy — the map must
scale with the buffer, and at `N = 24` the second group of `g2` lands at 768
which is exactly that buffer's size — and it did not survive one round.

### The map away from N = 64, and what it turned out to be

Sweeping `--map` at `N = 24` and `N = 40` — the first time the map was read
anywhere but 64 — disagreed with `8*K` at both:

```
N = 24  measured 0, 128, 192              8*K said 0, 128, 512
N = 40  measured 0, 128, 512, 576, 704    8*K said ... 640, 1024
```

The **skeleton** is regular and K independent. Groups of eight channels sit in
*blocks* of `8*K` bytes at *slots* of 64, with a channel's k+ half four slots on,
and the `K = 32` table decomposes exactly like the `K = 64` one — only the block
stride scales.

**Which slots** looked irregular until `G = 9` was swept. As flat slot indices,
block times 8 plus slot:

```
G=2  0 1              G=3  0 2 3
G=4  0 2 8 9          G=5  0 2 8 9 11
G=6  0 2 8 10 16 17   G=7  0 2 8 10 11 17 19
G=8  0 2 8 10 16 18 24 25
G=9  0 2 8 10 16 17 19 25 27

EVEN G   pairs at slots 0 and 2 in every block but the last, which takes 0 and 1
ODD G    one block of three at b3 = (G-1)/4, slots 0 and 2 before it and 1 and 3
         after it, the three itself {0,2,3} when G mod 4 is 3, {0,1,3} when it is 1
```

This is the third closed form written for this layout. The first two died in the
round after they were written — `wbytes/4`, which also broke three working
geometries, and "the last block takes the odd slots", which put `g4` of `N = 56`
at 1024 where `--map` found it at 704 — and both were fitted to a single point.

**This one predicted `G = 10` and `G = 11` before either was swept, and both
landed.** `N = 80` and `N = 88` came back 80 of 80 and 88 of 88 against the CPU
reference, and their `--map` grids reproduce all 21 predicted bases exactly:

```
N=80  0 128 512 640 1024 1152 1536 1664 2048 2112
N=88  0 128 512 640 1024 1152 1216 1600 1728 2112 2240
```

### Where int4 stands

```
works, no override, byte exact against a CPU reference
  K = 32, 64, 128, 160, 192 and 224 — K must be a MULTIPLE OF 32, which is the
    run count K/32 being whole: 144 and 176 give 8 of 64 where 160 and 192 give 64
  every N that is a multiple of 8 that has been tried, 16 through 160
  every channel, K/2 real k each
  out = ((int16)fp16bits(w) * (int16)abits) >> 16

open, and both are measured rather than untried
  M > 1 is int4 only — int8 at M = 4 is 256 of 256 byte exact. Two faults, now
    separated. The surface groups by SIXTEEN BYTES, read straight off a raw dump
    at M = 2: row 0's channels 0-3 are at words 0-3 and 4-7 at words 8-11, so an
    atom is 4 elements for w4a16 and 16 for int8, and at M = 1 every atom
    collapses to n. That fix landed: at M = 2, N = 16 row 0 now has no mismatches
    at all. What is left at M > 1 is SLOT 2: row 0 at N = 64 fails on channels
    8-15, 24-31 and 40-47, which are g1, g3 and g5, and those are every slot 2
    group in the packer's own placement, while slots 0 and 1 all pass. N = 16 has
    no slot 2 group, which is why its row 0 is clean — and at N = 24, which has
    a slot 2 AND a slot 3, both fail, so at M > 1 only slots 0 and 1 work.
    ⚠ The weight layout has NO M term — that reading came from a probe that
    drove every row at once and printed only the first lit word. Holding one row
    live at a time: row 0 alone has byte 8j lighting channel j in BOTH rows, at
    the same address M = 1 uses, and **row 1 alone lights nothing at all**. Row 1
    IS computed — in the row-0-only baseline its words come back nonzero with its
    own activation at the zero point — so both output rows are computed from
    row 0's activation slot and row 1's bytes are never fetched.
    Three registers wake row 1 up on the map — `0x1044`'s DATA_ENTRIES at
    `surf*m`, `0x103c` at `surf*m << 16`, and `0x1078` back at its M = 1 value —
    and **as matmuls all three are 16 of 32, identical to changing nothing.**
    `0x1094` breaks row 0 as well and `0x118c` is the baseline. Six registers,
    none of them a row count.
    The packing is not inert and no arrangement is right either. Sweeping the
    granularity at which rows interleave over 1, 2, 4, 8, 16, 32 and 64 elements,
    with the shipped value 8 and "rows outermost" 64 in the sweep as controls and
    both reproducing, **only 8 gives 16 of 32 and every other value gives 0**.
    ⚠ **M > 1 is closed with a negative**: six registers and nine packings, and
    the hardware computes M rows while feeding every one from row 0's activation.
    It does not block the project — decoding LLM tokens is M = 1, and int4 at
    M = 1 is exact across eleven geometries. M > 1 is a chaining problem.
  N not a multiple of 8: the hardware does not put the short group LAST. Its live
    channels at N = 20 are 0-11 and 16-23, and at N = 36 they are 0-19 and 24-39,
    so bytes 160-191 and 544-575 are dead and the packer writes logical channels
    12-15 and 20-23 into them. Both match the mismatch lists exactly. Measured at
    two N, no rule written.
  ⚠ The LAYOUT scales in K and the CHANNEL COUNT does not. K = 128 is exact at
    three N once the group count is right: a channel is fed by K/32 runs of eight
    bytes spaced a constant 256 — 1, 2 and 4 at K of 32, 64 and 128 — where the
    packer wrote two at an offset of 4*K, which equals 256 only when K is 64.
    That was the same trap as `8*K` against `wbytes/4`.
    At K = 256 thirty-two channels are correct whatever gets written: `0x3020`
    swept over six values gives 40, 40, 56, 56, 40, 40 written with `exact` at
    `wrote - 8` throughout, and `SIZE_E_2` moves what is written, 32/36/40, with
    `exact` stuck at 32. Beside the K that work, in bytes of weight actually
    fetched, which is `K/2` a channel:

    ```
    K =  64  N = 88 correct    88 x 32  = 2816
    K = 128  N = 64 correct    64 x 64  = 4096
    K = 256  N = 32 correct    32 x 128 = 4096
    ```

    Two land on 4096 exactly and the third is under it, which looked like a 4096
    byte weight fetch budget. ⚠ **Refuted, usefully.** `K = 64` at `N = 160` is
    5120 bytes and comes back 160 of 160, `K = 128` at `N = 96` is 6144 and comes
    back 96 of 96, and both were predicted to cap. There is no byte budget: **K
    up to 128 works at every N tried and K of 192 and 256 fail**, in two
    different ways. `N = 160` is also `G = 20`, well past the `G = 11` the slot
    form was read at, so that part generalises far.
  ⚠ K = 192 was never a fault. Rounds 300 and 301 recorded it as "writes every
    channel and computes none, the first non-power-of-two K"; it was the packer's
    K whitelist, which did not contain 192, so it returned without writing a
    byte, while `--map` lit anyway because it writes raw bytes. With the guard
    widened to multiples of 32 it is 64 of 64.
  ⚠ K = 256 is a count fault, and it is now exact: the channel count is `N/2`
    and `SIZE_E_2`'s additive 8 writes garbage on top, so `wrote` is `N/2 + 8`
    and `exact` is `N/2`. At `N = 16` those coincide with `N`, which is why
    `N = 16` alone comes back 16 of 16 there. K = 224 is full, so 256 is where
    it stops.
  charsiu_bench has an int4 path now, with the GB/s column counting the weight
    bytes a shape really moves — `k*n/2` for int4 against `k*n` for int8.
  charsiu_bench has no int4 path, so "what does int4 buy" still cannot be asked.
```

### "Half the k" is not half the weights

Only half the k reach any channel, but that is a statement about the reduction
depth and **not** about wasted weights. Every nibble the packer writes is
fetched: a channel gets two eight byte groups, 32 nibbles, and it is fed exactly
32 k, one per nibble. So a job declared at `K` computes a correct `K/2` deep
reduction, and what is wasted is buffer *space*, 2048 bytes allocated to hold
1024 bytes of live nibbles.

That also puts the two working configurations on a comparable footing:

```
shipped arithmetic, 0x3020 = 111    64 channels x 32 k  = 2048 MACs a job
                                    exact, and it runs today
PROC_PRECISION = 0                  40 channels x 64 k  = 2560 MACs a job
                                    but out = 127 * w unless the activation is
                                    one byte wide, and that halves the k back
```

The second is the larger job if its arithmetic can be made to hold, which is
what the `A8_STRIDE1` thread was chasing when it found `out = a * w` on one of
the two paired k.

### What is still open on int4

**The weights are read.** With no live nibble anywhere the output comes back all zero,
and the sign of the result follows the sign of the nibble: 7 gives +5112, and 15, which
is -1 as a signed nibble, gives -4896.

⚠ This section used to end here saying the magnitude did not follow, because changing
the nibble from 7 to 3 scales the output by 0.930 where 3/7 was expected. That is
answered by the section above and the text was left behind when the answer arrived.
The output is linear in log2 of its inputs, so 0.930 is what the formula predicts and
no line was ever going to fit two points of it. The coefficient buffer being unread on
this path is consistent rather than a defect, since `w4a16` does not requantise.

**What is genuinely left is the layout.** `k = 16` and above is no longer the
open part; rounds 262 to 265 read it. What replaced it is narrower and sharper.

Two geometries close on their own data, 40 of 40 points at `N = 16` and 48 of 48
at `N = 64`, with `nib` 0 for the low nibble and 1 for the high:

```
N=16, B<128:  word = B/8
              k    = 16*((B/8)&1) + 2*(B%8) + nib

N=64:         G = B/128, b = B%128,  b >= 64 fetches nothing
              word = b/8 + 8*(G&1) + 16*(G>>2)
              k    = 16*((b/8)&1) + 32*((G>>1)&1) + nib
```

Written as bits, one address bit does two jobs and one does none:

```
k    bit 0   = the nibble        word bit 0    = B bit 3     <- the same bit
k    bits1-3 = B bits 0,1,2      word bits 1,2 = B bits 4,5
k    bit 4   = B bit 3           word bit 3    = B bit 7
k    bit 5   = B bit 8           word bits 4+  = B bits 9+
                                 B bit 6 must be 0 at N = 64
```

So exactly half the `(channel, k)` pairs have no nibble feeding them, at both
`N`. `0x1020` says 32 weight bytes per channel and only 16 of them are ever
reached. That is one folded address bit, not a shape, and it is the whole of
what stands between here and an int4 projection.

Round 265 also settled what an output element is, which every reading before it
had guessed at: **four byte signed little endian, one per channel**. A live
nibble of 7 gives `00001bbc`, which is `+7100` and lights two bytes; a nibble of
15 gives `ffffe570`, which is `-6800` and lights four. The two arms agree on
every word and every k across 48 points.

### The whole weight map, read densely

`--map` needs one submit per byte where `--kpair` needs sixty four to sweep k,
and a nibble pairing with exactly one k is now measured at over a hundred
points, which retires the summation objection `--map` was written under. That
buys the whole buffer at a stride of 8 at three geometries in one boot:

```
N=64                         N=32                N=16
   0: w0-7    512: w16-23       0: w0-7             0: w0-7
  64: .       576: .           64: .               64: w8-15
 128: w8-15   640: w24-31     128: w8-15          128: .
 192: .       704: w32-39     192: w16-23         192: .
 256: w0-7'   768: w16-23'    256: w0-7'          256: w0-7'
 320: .       832: .          320: .              320: w8-15'
 384: w8-15'  896: w24-31'    384: w8-15'         384: .
 448: .       960: w32-39'    448: w16-23'        448: .
             1024+: dark      512+: dark          (' is the same channel, k+32)
```

Every word that gets written gets lit: 16 of 16, 24 of 24, 40 of 40. Nothing
writes a channel it does not compute.

Two numbers come out of that map and they are **different problems**:

```
channels reached = 8 * (N/16 + 1)    16, 24, 32, 40 and 72 at N of 16, 32,
                                     48, 64 and 128. 48 and 128 were derived
                                     from the other three, not fitted to.
k per channel    = 32                two eight byte groups, at B and B+256,
                                     sixteen nibbles each, while K is 64
```

The second is not the first in disguise. Every channel that exists at all gets
exactly half of its k, whatever `N` is. And `N/8` groups of eight would be every
channel, so the count is halved and then incremented.

### int8 is the oracle, and it computes everything it is asked for

The same probe, the same `K`, the same `N`, one flag:

```
int8:  channel = 32*(B/2048) + (B/32) mod 32
       k       = B mod 32 + 32*((B/1024) mod 2)
       64 bytes a channel, 64 k, no dead byte anywhere, highest byte written N-1
```

The two paths are the **same shape with int4's runs half as long**. int8 reads
32 bytes a channel a pass and takes two passes to cover 64 k. int4 reads 8 and
takes two, so it covers 32. To cover 64 it would have to read 16.

Diffing the two register streams at the same geometry leaves **fourteen**
registers, of which three are the requant and settled and one is the precision
itself:

```
CNA 0x100c  0 -> 0x20600120        DPU 0x4030  ..0710 -> ..0310
CNA 0x101c  0x1000 -> 0x800        DPU 0x4038  0x00120080 -> 0x53
CNA 0x1020  64 -> 32               DPU 0x4044  1 -> 2
CNA 0x1028  surf 1 -> 2            DPU 0x4050  0x80011111 -> 0x00023333
CNA 0x1030  128<<16 -> 32<<16      DPU 0x4010  0 -> 0xa0000002
CNA 0x103c, 0x1044  surf 1 -> 2    DPU 0x40ac, 0x40b0, 0x40b4  requant
```

`--stream` dumps that list from the tool and `CHARSIU_OVERRIDE` sets any single
register from the environment, so the ten that are left can be flipped to their
int8 value one at a time without a rebuild.

Nine of those ten arms came back clean and **not one moved either number**:

```
0x1020 0x1028 0x1030 0x103c 0x1044   five CNA size registers, excluded
0x4030 0x4044                        excluded
0x4038 0x4050                        both HANG when set to int8's value
0x100c                               wrote stays 40, every byte goes dark
```

`0x1020` set to int8's 64 and `0x1030` to int8's 128 left the eight byte run
exactly where it was, so bytes per kernel is not what bounds the fetch.

**The last line is the structural result.** The write extent does not depend on
the weights being read at all, so the two shortfalls are in different units:

```
k per channel is 32 and not 64      CNA. 0x100C CONV_CON1 is the only CNA
                                    register that did anything.
channels is 8*(N/16+1) not N/8      DPU, since no CNA arm touched it.
```

`CONV_CON1` reads `CONV_MODE 0, IN_PRECISION 2, PROC_PRECISION 2, RESERVED_1
48, GROUP_LINE_OFF 1`, and a field called RESERVED holding 48 is the same shape
round 260 found on `0x4050`, where four of five fields were load bearing and two
of them were in the reserved range.

### Two fields that move the two numbers

Sweeping those two registers by field found one each:

```
CNA 0x100C  RESERVED_1     48 -> 0    one group a channel becomes TWO
            PROC_PRECISION  2 -> 0    the same, from a different field
DPU 0x4050  SIZE_E_2        3 -> 1    channels 40 -> 32
```

Sixteen weight bytes a channel a pass instead of eight is 32 nibbles, and with
the pass at `B+256` that is 64 k, exactly the half that was missing. Both arms
left the channel count at 40, so the split holds.

That was a word pattern, not a k measurement, and `--kpair` separated the two
arms in one entry. **They are not the same thing.**

```
RESERVED_1 = 0        byte 0 low k0, byte 0 HIGH ALSO k0 with a different
                      value, byte 8 k8, k = byte mod 32
                      the byte is read as ONE weight. This does not fix int4,
                      it turns it off.

PROC_PRECISION = 0    byte 0 low k0 high k1, byte 8 low k16 high k17,
                      byte 16 -> w1 k0
                      nibble packing intact, 32 k a pass. This one is real.
```

The derived prediction was byte 8 at k 16 and the second arm did exactly that,
four of six points. The other two moved meaning: bytes 256 and 264 used to be
the same channel at `k+32` and are now channel 8 at k 0 and 16, so where k 32
through 63 lives is open again.

**`PROC_PRECISION = 0` closes the k side completely.** The full sweep under it:

```
   0- 127: w0..w7      512- 639: w0..w7  k+32      1024-1151: w16..w23
 128- 255: dark        640- 767: dark              1280-1407: w24..w31
 256- 383: w8..w15     768- 895: w8..w15 k+32      1408-1535: w32..w39
 384- 511: dark        896-1023: dark              1536-2047: the k+32 half
```

Four groups of eight bytes a channel: 32 bytes, 64 nibbles, **64 k**. `0x1020`
says 32 bytes a kernel and 32 is now what gets fetched. The dark 768 bytes are
exactly 24 channels times 32, so the buffer is the right size and 24 channels'
worth is never read. What is left is only the channel count.

The channel count scales with `M`: 40 words at `M = 1` and 80 at `M = 2`, so it
is `M * (N/2 + 8)` and the halving is not about `M`.

### The two arithmetic modes, measured

`PROC_PRECISION` is the arithmetic mode and the two are different operations,
not a working one and a broken one. Sweeping the live nibble under both:

```
nibble        1    2    3    4    5    7    15
shipped      60   64   66   68   69   71   -68
PP = 0      127  254  381  508  635  889  -127
```

The shipped column is the formula above, exact at all seven, once the activation
is read as the **raw 16 bit slot** rather than as an fp16 of the value. `--map`
packs an int8 into the high byte of a 2 byte slot, so `abits` is 256 for `a = 1`:

```
out = ((int16)fp16bits(w) * (int16)abits) >> 16
w=1  15360*256>>16 = 60      w=5  17664*256>>16 = 69
w=2  16384*256>>16 = 64      w=7  18176*256>>16 = 71
w=3  16896*256>>16 = 66      w=15 (signed -1) -17408*256>>16 = -68
w=4  17408*256>>16 = 68      and the high nibble, abits 512, is 2x each
```

`PP = 0` is `127 * w` with `w` a signed nibble, linear and exact at all seven.
With a **one byte activation** it becomes the real thing. Sweeping the one hot
amplitude against a nibble of 7:

```
amp     1    2     5    10   100
out     7   14    35    70   700        out = a * w, exact at five of five
```

A plain integer multiply, no fp16 bit pattern and no logarithm, which is the
operation an int4 matmul wants. The paired second k still returns `127 * w`
whatever the amplitude, so it reads something that is not in the activation
buffer, and that is a separate defect.

### The channel count has a source, and it is 0x3020

`CORE 0x3020` set to 127, claiming 128 channels at `N = 64`, gives 72 written.
`0x402c` and `0x5014` carry the same `n-1` and are inert, `CONV_CON2` is inert,
`SIZE_E_0` is inert or hangs, `CBUF_CON0` turns the fetch dark without moving the
count:

```
channels = ceil((v+1)/2) + extra(SIZE_E_2)     v = 0x3020, extra 0, 0, 4, 8

v =  31 -> 24    v =  79 -> 48    v = 111 -> 64
v =  47 -> 32    v =  95 -> 56    v = 127 -> 72
```

Six for six, derived from two measured points rather than fitted to six. **So
`0x3020 = 111` writes all 64 channels at `N = 64`**, and the channel shortfall
is not a wall, it is a value.

**Written and reachable.** The full sweep at `v = 111`: 256 groups, 128 live, 64
distinct words lit, range 0 to 63, exactly the derived prediction. The channel
shortfall is solved.

So without any other change that configuration is, today:

```
64 channels, each fed by 16 weight bytes = 32 nibbles = 32 k
the SHIPPED arithmetic, whose formula is known exactly and exact at seven points
```

Bytes per channel is `K/4`, measured at two `K`: 8 at `K = 32` and 16 at
`K = 64`. So half the k is missing at **every** `K` and it is not a fixed cap.
`K = 16` lights nothing at all.

### The fetched half is a known half

With `0x3020 = 111` at `N = 64, K = 64`, every channel gets 16 bytes in two runs,
its own and one 256 bytes later, which `--kpair` reads as k 0..15 and k 32..47:

```
fetched  <=>  (k mod 32) < 16
```

A weight the hardware never reads is only wrong if it matters. **Zero the ones
it does not read and the partial sum it computes is the full sum**, so a CPU
reference and the hardware answer the same question. That is a real 32 deep int4
matmul on all 64 channels with the arithmetic already known exact, at the cost of
half the weight buffer, and it needs no packer change and no `CONV_CON1` change.
`CHARSIU_W4_HALFK` does the zeroing.

⚠ **Round 278 ran that comparison and it was void, for two reasons that are both
instrument.** The matmul harness read the output as **bytes**, and the giveaway
is in its own log: every group of four reads `X Y 255 255` or `X Y 0 0`, which is
a little endian int32 pulled apart, `159 229 255 255` being `0xFFFFE59F`. And
`cpu_reference()` returns a requantised int8 where w4a16 returns a raw
accumulator, so the two were never in the same domain and the round could not
have passed whatever the hardware did.

That is the **fourth** place in this repo to read a four byte output as bytes,
after `--kpair`, `--map` and the matmul. Every one was correct for int8, which is
how each passed its int8 validation and kept the bug; 278's int8 arm was 64 of 64
byte exact in the same log.

⚠ **Where the shift goes has never been measured.** Every point behind the
formula had one live nibble, and with a single term a shift per element and a
shift on the sum are the same number. A dense buffer is the first thing that can
tell them apart.

### Why every int4 matmul arm returned the same numbers

Channels 0 to 7 came back as -6753, -4633, -7670, -9732, -3554, -12306, -706 and
-8100 in every arm of two rounds: `HALFK` on and off, `0x3020 = 111` and not, and
`N` of 64, 32 and 16. The output did not move by one count while `HALFK` zeroed
2048 of 4096 weights.

Not the hardware. `charsiu_pack_weights` **refuses to place k >= 16 for int4**,
deliberately since round 173, and `HALFK` only ever touched k the packer never
wrote, so both buffers were byte identical. The `N` invariance falls out of the
same thing: the row it used, `(n/32)*512 + (n%32)*8`, has no `N` in it — and it
came from `--map` before the byte width defect was fixed, so it was the stale map
as well as an incomplete one.

### The layout the packer writes now

```
ADDRESS   c 0..7  -> 0      c 16..23 -> 512    c 32..39 -> 1024   c 48..55 -> 1536
          c 8..15 -> 128    c 24..31 -> 640    c 40..47 -> 1152   c 56..63 -> 1600
          plus a second eight byte group 256 bytes later

k         an EVEN channel is fed k 0..15 and 32..47
          an ODD  channel is fed k 16..31 and 48..63
```

A table and not a formula on purpose: seven of the eight steps are 128 or 384 and
the last is 64, so a closed form would be fitted to one point. The parity rule is
read off byte 0 pairing with k 0 on channel 0, byte 8 with k 16 on channel 1 and
byte 16 with k 0 on channel 2, and confirmed independently at `K = 32`.

⚠ So the `HALFK` mask in rounds 278 and 279 was **backwards on half the
channels**: both zeroed `(k mod 32) >= 16` everywhere, which is what an even
channel is fed.

⚠ And the address map was read at `0x3020 = 111`. charsiu emits `n - 1` there,
which gives 40 channels, so this layout describes the hardware only when that
register is overridden.

Five rounds of sweeping and the answer was in a register excluded for being
**identical on both paths**. The sweep list came from diffing int8's stream
against int4's, and a register that is the same in both cannot cause the
difference, but it can be the bound one path reaches and the other does not.

⚠ **The two fixes do not combine, and the buffer explanation for it was wrong.**
The prediction was that `v = 111` would not fight, since 64 channels times 32
bytes is exactly the 2048 the buffer holds. It hung: 235 groups swept and 4
alive. The other route to 64, `v = 127` with `SIZE_E_2 = 1`, drops to 32 under
the k fix rather than hanging. Three points, no explanation:

```
v = 63,  extra 8, k fix  ->  40      unchanged from without the fix
v = 127, extra 0, k fix  ->  32      halved from 64
v = 71, 79, 87, 95, 103, 111  ->  all hang
```

`2^n - 1` looked like the rule on eight points and is **refuted**: 15 and 31 are
`2^n - 1` and both hang. The predicate that fits all twelve is `(v+1) mod 64 ==
0`. And under the k fix `v` stops driving the count entirely, since 63, 127 and
255 all give 40 and only `SIZE_E_2` moves it, so the k fix pins the channels at
32 plus `SIZE_E_2`'s term. **The two halves cannot be had together.**

### The activation packing

It was the packing, at least partly. charsiu packs the activation as a 2 byte
fp16 whenever the weight is int4, int8 value in the high byte and the low byte
zero. Switching to a 1 byte element with `CHARSIU_A8_STRIDE1` makes a nibble
pair with **two** k, and the two do different things:

```
byte 0 low  ->  k0 = 700 = 100 * 7    the one hot amplitude times the nibble.
                                      a * w, exactly.
            ->  k1 = 889 = 127 * 7    the constant again.
```

So the mode can multiply by the activation. Whatever the 127 is, it is not "this
mode ignores the activation", because one of the two k did not ignore it.

⚠ It costs half the k back: 80 live groups where `PROC_PRECISION = 0` alone gave
160, and byte 512 goes dark.

`SIZE_E_2` swept across all eight values gives `0 -> 32, 1 -> 32, 2 -> 36,
3 -> 40`, and 4 upward hang: an additive 0, 0, 4, 8 on top of `N/2`, so it is
the `+1` in `8*(N/16 + 1)`. **No value gives 64.** **It is not the halving**: int8 runs `SIZE_E_2 = 1` and gets all
64 channels where int4 runs the same value and gets 32. `RGP_CNTER`, `SIZE_E_1`
and `OD_BYPASS` are inert, `RESERVED_0` hangs, `0x4038`'s `NOTCH_ADDR_0` hangs
and `NOTCH_ADDR_1` is inert, and `0x4058` is inert in both its fields, which
settles that the register naming the output channel count is not what bounds
it.

### The hardware writes fewer channels than the job declares

Filling the output buffer with a sentinel rather than zero, and reading the
whole of it, says how far the write actually reaches. It is not `N`:

```
N = 16   ->  16 words written        N = 64   ->  40 words written
N = 32   ->  24 words written        which is N/2 + 8
```

Three geometries, and the fit is exact. At `N = 16` it happens to equal `N`,
which is why every closed result in this file is at `N = 16`. At `N = 32` eight
declared channels are never written and at `N = 64` it is twenty four. This is
an output side fact and none of the weight address arithmetic above is needed to
state it.

### The defect this project put there, and its fix

**A w4a16 job used to leave the NPU unable to start the next one.** The same int8 binary, the
same register stream, the same shape and the same boot: byte exact when it runs first,
`NPU job timed out` when it runs after w4a16 jobs, with the output still holding its
0xa5 sentinel and an `rk_iommu` reset error beside it. Mesa's own models run fine
afterwards, so the driver recovers and nothing is permanently broken.

It was not the stream: int8's register stream is byte identical before and after the
w4a16 port, at both shapes, checked offline. It took a one job repro to bisect, and two
earlier attempts were killed by their own written controls first, once by a recovery
step that was itself broken and once by a probe whose "zero jobs" row had five jobs
behind it. Both are recorded in the board scripts rather than quietly fixed, because
those controls are the only reason the wrong answers were not published.

**It is the RDMA coefficient fetch group, and within it `0x5034` and `0x5044`
each on their own.** One w4a16 job per mask, judged by a separate process running
the int8 path that has been byte exact since round 164:

```
mask 0xf  all four     int8 TIMED OUT        mask 2  0x5034 only  int8 TIMED OUT
mask 0    none         int8 byte exact       mask 4  0x5040 only  int8 byte exact
mask 1    0x501c only  int8 byte exact       mask 8  0x5044 only  int8 TIMED OUT
```

That group is out of the default stream now. It buys nothing: the byte map with it
off and the byte map with it forced on are identical line for line. It stays reachable
one bit per register through `CHARSIU_W4_RDMA_MASK`, because fetching the coefficient
surface will have to start there.

## On the name

Char siu is Cantonese barbecue pork, eaten across Guangdong, Hong Kong and Malaysia. A kiln is
the oven it is roasted in. [kiln](https://github.com/gahingwoo/kiln) runs the
**vendor** LLM stack on a mainline kernel and is that oven here: it is what makes a
vendor `.rkllm` readable, what a live dispatch is captured with, and the number this
project has to beat. charsiu is what comes out of it and what you actually eat, an
open runtime with nothing closed left in the path.

## The three repositories

| repo | what it is |
|---|---|
| [linux-rk3576-npu](https://github.com/gahingwoo/linux-rk3576-npu) | the open RK3576 NPU driver and Mesa work: `rocket` upstream, and where every register name used here was established |
| [kiln](https://github.com/gahingwoo/kiln) | the vendor RKLLM/RKNN stack on a mainline kernel. The measuring stick, and the capture harness |
| **charsiu** | this one: an open LLM runtime on top of the open driver |

## Why this exists, and why it is not a port

Two open stacks already drive a Rockchip NPU for LLM work, both on the **RK3588**:
[rocket-userspace](https://github.com/gregordinary/rocket-userspace) with
[ggml-rocket](https://github.com/gregordinary/ggml-rocket) on the mainline `rocket`
driver, and [iwagumi](https://github.com/fukumori/iwagumi) on the vendor `rknpu` ioctl.
Their measurements are worth reading before starting anything here, and they are
collected in [rockchip-npu-notes](https://github.com/gregordinary/rockchip-npu-notes).

Their central finding is that **the NPU is a prefill engine and decode belongs on the
CPU**: a single-row matmul is about 82 times slower on the NPU than the batched shape
it was built for, a feature height below four computes wrong output at all, and
quantisation does not speed prefill up because the pipeline sits at a dispatch and DMA
floor rather than a MAC one.

**On the RK3576 the vendor does the thing that finding says not to do.** Reading the
register command streams out of a vendor `.rkllm` for Llama-3.2-1B, 3752 of its
convolution dispatches are `M = 1`, one row and one output pixel, and they are the
model's own projections. The vendor ships that and it generates about 13 tokens a
second on this board.

So the question this project starts from is not "how do we port the RK3588 result"
but **"what does the RK3576 actually do, and can an open runtime match it"**. The
RK3576 is not the same machine: 1 MiB of CBUF against the RK3588's 384 KiB, two cores
rather than three, a 16 bit task number, and a different weight tile stride. A
negative result measured on the other chip is a hypothesis here, not a conclusion.

### Measured on hardware, 2026-08-14

Through the open driver and Mesa's own delegate, on a ROCK 4D, single 1x1
convolutions at an LLM projection's shape:

| shape | M | result |
|---|---|---|
| 512 to 1024 | 1 | 1024 of 1024 channels exact, 146 distinct values against the CPU's 146 |
| 512 to 1024 | 2 | 1024 of 1024, 357 of them computed |
| 512 to 1024 | 3 | 1024 of 1024, 416 computed |
| 512 to 1024 | 4 | 1024 of 1024, 465 computed |
| 512 to 1024 | 8 | 1024 of 1024, 518 computed |
| 512 to 512 | 1 | 512 of 512 |
| 256 to 1024 | 1 | 1024 of 1024 |

**Heights of one, two and three are exact.** The RK3588 constraint is not on this
silicon, which is what the vendor dispatching M = 1, 2 and 3 in its own `.rkllm`
already implied. A one row surface has nothing to vary within a channel, so the
evidence at M = 1 is the per channel match count and the distinct count; M = 2 and
up carry the computed confirmation.

What this does **not** yet say is anything about speed, or about int4. Both are the
next board round.

## What runs today

```
$ charsiu_probe
open /dev/accel/accel0 ok
bo handle 1  dma 0x1000  size 4096  in the regcmd window
write/readback through the mapping: ok (0 bytes differ)

$ charsiu_matmul 1 64 64          # M=1, the shape an LLM decodes with
matmul M=1 K=64 N=64 int8, feature atom 16, 1 entries per row
register stream: 143 entries
submit ok
output: 64 of 64 bytes written, 64 BYTE EXACT, 0 differ by more than 1
```

The stream those 143 entries make is identical to the one Mesa emits for the same
shape, entry for entry, values and order, addresses and quantisation aside, and it is
also identical to what the vendor's own compiler produces for that shape with no
activation. Both are checked on a desktop, not on the board:
`vendor-capture/cmp_charsiu.py` in the driver repository diffs the whole stream against
a vendor `.rknn` compiled at charsiu's exact geometry.

## The instrument

`tools/rkllm_regcmd.py` reads the NPU register command streams straight out of a
`.rkllm`, on a desktop, with no board and no vendor runtime running. The vendor's
model file carries the programs it submits, so what the closed stack asks the hardware
to do can be read directly.

```
$ tools/rkllm_regcmd.py Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm
  streams            21532, of which 8808 are convolutions and 12724 DPU only
  distinct shapes    1061
  weight bits        {16.0: 4940, 4.0: 3328, 8.0: 40}
  M (rows per op)    {1: 3752, 32: 1108, 64: 856, 96: 728, 128: 672}

  ic       oc       surf     M      count
  4096     1024     5120     1      320
  2048     1024     5120     1      256
  2048     256      5120     1      256
  2048     4096     5120     1      256
```

Read against Llama-3.2-1B's own dimensions, hidden 2048 and FFN 8192 with 8 KV heads,
every one of those output widths is **half** of a projection: 2048 to 1024 is half the
Q projection, 2048 to 256 half the K and V, 2048 to 4096 half the FFN up, and 4096 to
1024 half the FFN down. The vendor splits each projection across the two NPU cores by
output channel.

The rest of what the file says is in [docs/vendor-dispatch.md](docs/vendor-dispatch.md).

## Scope

- **Target**: RK3576, Radxa ROCK 4D, mainline kernel with `rocket`.
- **In**: the matmul and the ops an LLM needs, on the NPU, in the precisions the
  vendor uses; a frontend that runs a real model end to end.
- **Out**: no vendor `librkllmrt` or `librknnrt` in the execution path, and no claim
  about any SoC this has not been run on.
- **Honest about performance**: the target is the vendor's own number on the same
  board and model. The projection cost is measured above and the arithmetic to a token
  rate is written out; **there is still no end to end number here**, because no model
  runs yet, and a projected rate is not a measured one.

## Prerequisite

The RK3576 support in `rocket` is not upstream yet. It is on the list as
[PATCH v7](https://lore.kernel.org/all/20260812094106.1391698-1-gahing@gahingwoo.com/),
and the driver plus the Mesa work it comes from is
[linux-rk3576-npu](https://github.com/gahingwoo/linux-rk3576-npu), which is where the
RK3576 register knowledge in this repository was established.

kiln's `capture/rknpu-regcmd-dump.patch` is how a live vendor dispatch is read when
the model file is not enough.

## Licence

GPL-2.0-or-later. `LICENSE` carries the GPL version 2 text; the "or later" is what
lets this be combined with GPL-3 code if it ever links any.
