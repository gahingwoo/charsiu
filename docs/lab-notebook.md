# The lab notebook

Moved out of the README on 2026-09-02, verbatim and in the order it was written:
the record of how the int4 layout, the accumulator read order, the batching and the
rest were established on the board, with the numbers of the day each was written.
The README carries what is true now; this carries how it was found out.


---

<!-- from the README: the 2026-08-15 probe table and the M > 1 discussion -->

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

⚠ **The last two rows are the int8 output path, not the accumulator one.** They were
measured by `charsiu_matmul`, which requantises to int8 and reads the result as a
surface. The runtime's decode path sets `acc_out` for the raw int32 accumulator and
reads it flat, and on that path nothing above one row has ever been correct. Do not
read these two rows as "m > 1 works"; they say it works in the path they measured.

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

⚠ **The three paragraphs above are where this stood when the matmul was the whole
question, and they are kept because the reasoning in them is still the reasoning.**
What they say is left has since been done: models run, int4 computes end to end, and
the numbers at the top of this file are measured rather than projected. The two items
this paragraph used to name as open have closed since -- prefill batches above M = 1,
and the stale width cap that kept an output head on the CPU is gone.

One thing on that list has not moved: the reference still requantises in float where
the hardware uses an integer scale and shift. It agrees to the byte on everything
measured so far, which does not mean it will at every scale.


---

<!-- from the README: the CPU baseline's q6_K story -->

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


---

<!-- from the README: the activation is quantised -->

### The activation is quantised, not the weights converted

The first version converted every **weight** to a float in order to multiply it by a float
activation. That is the expensive way round: a matvec has `N*K` weights and only `K`
activations. So the activation is quantised to signed int8 once per matvec, in blocks of
32 with a scale each, and the dot product becomes **integer**, with one float multiply per
block at the end. Q, K and V all read one RMSNorm output and so do gate and up, so six of
the nine quantisations in a layer are of a vector already done and are skipped.

| host, q4_0, 4 threads | tok/s |
|---|---|
| exact f32 activation | 17.99 |
| quantised | 31.50 |
| plus the deduplication | **36.85** |

**The order flipped**: q4_0 36.85 now beats q8_0 34.80, where q8_0 used to win. Thread
scaling flattened as well (1 → 18.4, 2 → 30.2, 4 → 33.4, 6 → 30.8). Both are what a
kernel that has stopped being ALU bound looks like. llama.cpp on the same host and file
is 48.3 tok/s; charsiu was at 39% of it and is now at 76%.

This matters beyond speed. **An int8 activation with a scale is exactly what the NPU
consumes**, so the CPU fallback and an NPU job can be fed from the same buffer, and the
CPU path is the reference the NPU path gets diffed against rather than a separate design.

It is an approximation, so it has controls. `CHARSIU_NO_QACT` restores the exact path, and
that path is proven untouched: the logits it prints are byte identical to the values
recorded before the change. `tests/qact_control.py` measures what the approximation costs
without needing any reference implementation, q4_0 moves a logit by at most 0.055 with 48
greedy tokens unchanged, q8_0 by at most 0.103.


---

<!-- from the README: the int4 record -->

## The NPU's number format, answered on the CPU

Moving a projection onto the NPU means accepting the format the hardware imposes: signed
int8 weights with **one scale per output channel**, which is what the coefficient buffer
applies, against an activation with **one scale for the whole vector**, because a
`charsiu_job` carries a single `input_scale`. Both zero points are 128, so what the DPU
computes is exactly

```
acc[n] = sum_k a_q[k] * w_q[n][k]
y[n]   = acc[n] * a_scale * w_scale[n]
```

Whether that costs the model its output is not a question about register streams, and the
CPU decode loop is already the oracle for it. So the whole model runs in it:
`CHARSIU_NPU_QUANT=1` builds a second copy of every routed tensor in that format and
multiplies with a widening integer dot.

Llama-3.2-1B from a q8_0 file, **113 tensors routed**, every projection and the tied
output head, with a per tensor RMS error against the original of **0.85% to 1.57%**. The
text is coherent, and at this prompt it is the continuation llama.cpp itself produces:

```
$ CHARSIU_NPU_QUANT=1 charsiu_run Llama-3.2-1B-Instruct-Q8_0.gguf \
      -p "The capital of France is" -n 32
Paris. The Eiffel Tower is located in Paris. The Eiffel Tower is one of the most
famous landmarks in the world. It was built for
```

So the format is not the problem, and what is left of that step is plumbing rather than
accuracy. `npu_tensor_build()` in `src/npuquant.c` is also the converter's arithmetic
already written: it produces `q`, the per channel scale, and the per channel weight sum
the coefficient buffer wants, verified by the tokens it produces rather than by reading it.

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
- **a second core did not help, and the reason was not bandwidth.** Two jobs of eight
  tasks were about 5% *worse* than one job of sixteen at every shape. That was read as a
  bandwidth bound workload and it was the deal: slices went to devices by index,
  `(ki * ns + ni) & 1`, and ki and ni restart at zero for every tensor, so a
  single slice tensor landed entirely on device 0 and the other core sat out. Dealing to
  the less loaded core instead is worth 15% to 21% of the token hardware path.

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
K = 512 at N = 16 is 16 of 16. `exact` is `N/2` at all of them, independent of K, a working half-width job, not a failure. Benched at K = 512, N = 64: int8 24.6
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
int8**, half the weight bytes, the same time. int8 is genuinely bytes-bound
there, 9.14 GB/s marginal; int4 moves 0.26 MB in the same 223 us, which is
1.18 GB/s. The 2x per byte is real on paper and none of it reaches the clock.
And chaining hangs: the second task of that shape times out and takes the block
with it.

The layout itself is fine that far out, `K = 2048` at `N` of 64, 128, 160 and
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

The `N` with an even highest group, 24, 40 and 56, failed, and **not on the
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
something else. The reasoning that produced `wbytes/4` was tidy, the map must
scale with the buffer, and at `N = 24` the second group of `g2` lands at 768
which is exactly that buffer's size, and it did not survive one round.

### The map away from N = 64, and what it turned out to be

Sweeping `--map` at `N = 24` and `N = 40`, the first time the map was read
anywhere but 64, disagreed with `8*K` at both:

```
N = 24  measured 0, 128, 192              8*K said 0, 128, 512
N = 40  measured 0, 128, 512, 576, 704    8*K said ... 640, 1024
```

The **skeleton** is regular and K independent. Groups of eight channels sit in
*blocks* of `8*K` bytes at *slots* of 64, with a channel's k+ half four slots on,
and the `K = 32` table decomposes exactly like the `K = 64` one, only the block
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
round after they were written, `wbytes/4`, which also broke three working
geometries, and "the last block takes the odd slots", which put `g4` of `N = 56`
at 1024 where `--map` found it at 704, and both were fitted to a single point.

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
  K = 32, 64, 128, 160, 192 and 224. K must be a MULTIPLE OF 32, which is the
    run count K/32 being whole: 144 and 176 give 8 of 64 where 160 and 192 give 64
  every N that is a multiple of 8 that has been tried, 16 through 160
  every channel, K/2 real k each
  out = ((int16)fp16bits(w) * (int16)abits) >> 16

open, and both are measured rather than untried
  M > 1 is int4 only, int8 at M = 4 is 256 of 256 byte exact. Two faults, now
    separated. The surface groups by SIXTEEN BYTES, read straight off a raw dump
    at M = 2: row 0's channels 0-3 are at words 0-3 and 4-7 at words 8-11, so an
    atom is 4 elements for w4a16 and 16 for int8, and at M = 1 every atom
    collapses to n. That fix landed: at M = 2, N = 16 row 0 now has no mismatches
    at all. What is left at M > 1 is SLOT 2: row 0 at N = 64 fails on channels
    8-15, 24-31 and 40-47, which are g1, g3 and g5, and those are every slot 2
    group in the packer's own placement, while slots 0 and 1 all pass. N = 16 has
    no slot 2 group, which is why its row 0 is clean, and at N = 24, which has
    a slot 2 AND a slot 3, both fail, so at M > 1 only slots 0 and 1 work.
    ⚠ The weight layout has NO M term, that reading came from a probe that
    drove every row at once and printed only the first lit word. Holding one row
    live at a time: row 0 alone has byte 8j lighting channel j in BOTH rows, at
    the same address M = 1 uses, and **row 1 alone lights nothing at all**. Row 1
    IS computed, in the row-0-only baseline its words come back nonzero with its
    own activation at the zero point, so both output rows are computed from
    row 0's activation slot and row 1's bytes are never fetched.
    Three registers wake row 1 up on the map, `0x1044`'s DATA_ENTRIES at
    `surf*m`, `0x103c` at `surf*m << 16`, and `0x1078` back at its M = 1 value, and **as matmuls all three are 16 of 32, identical to changing nothing.**
    `0x1094` breaks row 0 as well and `0x118c` is the baseline. Six registers,
    none of them a row count.
    The packing is not inert and no arrangement is right either. Sweeping the
    granularity at which rows interleave over 1, 2, 4, 8, 16, 32 and 64 elements,
    with the shipped value 8 and "rows outermost" 64 in the sweep as controls and
    both reproducing, **only 8 gives 16 of 32 and every other value gives 0**.
    ⚠ **M > 1 is closed with a negative**: six registers and nine packings, and
    the hardware computes M rows while feeding every one from row 0's activation.
    It does not block the project, decoding LLM tokens is M = 1, and int4 at
    M = 1 is exact across eleven geometries. M > 1 is a chaining problem.
  N not a multiple of 8: the hardware does not put the short group LAST. Its live
    channels at N = 20 are 0-11 and 16-23, and at N = 36 they are 0-19 and 24-39,
    so bytes 160-191 and 544-575 are dead and the packer writes logical channels
    12-15 and 20-23 into them. Both match the mismatch lists exactly. Measured at
    two N, no rule written.
  ⚠ The LAYOUT scales in K and the CHANNEL COUNT does not. K = 128 is exact at
    three N once the group count is right: a channel is fed by K/32 runs of eight
    bytes spaced a constant 256, 1, 2 and 4 at K of 32, 64 and 128, where the
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
    bytes a shape really moves, `k*n/2` for int4 against `k*n` for int8.
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
same thing: the row it used, `(n/32)*512 + (n%32)*8`, has no `N` in it, and it
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


---

<!-- from the README: measured on hardware 2026-08-14, and what M > 1 turned out to be -->

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

### What charsiu's own M > 1 turned out to be

**Solved, and there was never a hardware wall.** Two defects of ours, both invisible at
the only width they had ever been exercised at:

- `0x40b8` was the literal `3` where it must be `3 * rows`. It was fitted at M = 1,
  which is the one width at which a value that follows the row count cannot show that
  it does.
- the accumulator's read order was unknown. It is `charsiu_acc_index()`: `P = m / 2`,
  super groups of 32, rows pairing P at a time, four word runs alternating. Confirmed
  at m = 8 and N = 2048, both widths it was not fitted on.

int8 batches correctly from m = 2 to 32 and a batched prompt is 2.94x on the board.
**w4a16 computes exactly one row with M on the HEIGHT axis**: fed the same activation
twice, row 1 matches row 0 in 1 of 2048; the DPU and RDMA blocks are identical to a
stream that does two rows; every CNA word that differs was put back one at a time.

⚠ That was called silicon rather than a literal, and it is not. The vendor's own
Llama-3.2-1B file carries 3328 int4 streams and 2816 of them are batched, at M up to
80 -- one row high and M PIXELS WIDE, which is why anything reading the row count sees
M = 1 and concludes there is nothing to copy. Their fp16 attention does use the height
axis, all 4940 streams of it, which is what made the two look like one. charsiu's
width axis form is now two registers from theirs at M = 32 and 64, fewer than at M = 1
where this board is known to be right; `tests/board_w4_axis.sh` is the round that has
not happened yet.

The same file settles something about the int8 path too. Its 40 int8 streams are the
LM head at M of 1, 32, 64, 96 and 128, and every one of them is **one row high** as
well -- so the vendor puts M on the width for both weight formats and uses the height
axis for nothing but attention. This tree's int8 batch, which is the one that works,
runs on the height and stops being exact at 96. `tests/board_rows_sweep.sh` sweeps
both axes now, height first as the control. If the width arm reaches 128 then the
ceiling was the arrangement, on the path that already carries the 2.94x.

**What follows is the record of the search**, kept because the reasoning in it is still
the reasoning and because most of a fortnight was spent fitting models to the symptom.

charsiu asks the hardware for one row. A batched prefill wants thirty two, where the
same weight bytes serve thirty two rows instead of one, and at a projection's K this
tree had never got a correct answer above M = 1.

Four rounds went into fitting an address function to the output, because the values
came back in the wrong places and that looks like a layout. Against a reference with
121 distinct values in 128 the fit is unique -- one function, 80 of 80 cells, no misses
-- and then it predicts 132 words at M = 4 where the board writes 148. A model that
fits one width perfectly and misses the next is a model of that width.

Counting instead of fitting says what it actually is:

```
N=64 m=2   92 of 128       N=64 m=4  148 of 256      N=64 m=8  260 of 512
N=32 m=2   52 of  64       N=16 m=2   32 of  32
```

`written = N + inc * (m - 1)`, with `inc = N/4 + 12` through all three widths. **The
first row gets all N words. Every row after it gets N/4 + 12, and needs N.** They meet
at sixteen, and at sixteen nothing is missing -- every value present, none of them
where a contiguous read expects it.

So it is not a stride. A stride puts values in the wrong place; this leaves them
uncomputed, and "one row whole and the rest a fixed share" is a budget being divided.

**The budget is not the CBUF, and the arithmetic settles that without a board.** An
LLM matmul reaches the encoder one column wide, `input_height = m`,
`input_channels = K`, so `entries_per_slice` is 16 at K = 1024. The RK3576 CBUF is 16
banks of 512 entries and Mesa's own budget is five banks usable, ten total: at m = 8
that is **128 entries against 2560**. Mesa's over-budget test needs m > 320, its split
needs m > 640, and its row-window path needs a surface at least 112 wide. Mesa does
not split these shapes either, so not splitting is not the difference.

**It is not `surf` either, and the control said so.** The reading before that was
that every shape ever correct above one row has `charsiu_entries_per_row() == 1`, so
the axis is entries per row rather than m. The rule was written down before the run:
K = 48 and K = 64 are surf 1 and have to be exact at m = 2 and m = 4, or the axis is
not surf. They were not.

```
     K  surf   m   exact of      all values present
    48     1   2       8 of 128       99 of 128
    64     1   2       8 of 128       97 of 128
   128     2   2       8 of 128       93 of 128
   256     4   2       8 of 128       92 of 128
  1024    16   2       8 of 128       95 of 128
```

Flat from surf 1 to surf 16, **8 words exact at m = 2 and 8 at m = 4 at every K**. The
`N + (N/4 + 12)(m - 1)` budget above fitted one K and does not survive the rest.

**What the record was comparing is two different output paths.** M = 224 and M = 3136,
the rows in the table near the top of this file, were measured by `charsiu_matmul`,
which takes the requantised **int8** output and reads it as `[n/atom][m][n%atom]` with
a 16 byte atom. Every m > 1 failure was measured by `npu_gemm_test`, which sets
`acc_out = 1` for the **raw int32 accumulator** and reads it flat. Those are not the
same experiment, and the runtime's decode path uses the accumulator, so the layout
that matters above one row has never been established.

⚠ The obvious guess is that the accumulator mirrors the same surface with a four word
atom. It predicts 8 exact at m = 2, which is what the board wrote, and 16 at m = 4,
where the board wrote 8. Right at one width and wrong at the next is what the last
four rounds kept producing, so it is recorded and not acted on.

That was right, and it is what settled it: both probes at one shape in one session,
`charsiu_matmul` against `npu_gemm_test --surf`, with `CHARSIU_OUT_ROWMAJOR=1` as the
control that had to fail. The accumulator does not mirror the int8 surface with a four
word atom. It is `charsiu_acc_index` above, and the reason every earlier fit died at
the next width is that `0x40b8` was writing a row budget for one row whatever m was --
so the counting was measuring a truncation, not a layout.

⚠ The test that measured all of this printed the opposite of its own data first: "1 of
5 widths exact at N=16, the budget reading does not hold", while its own increment line
three screens up said `at N=16, m=2: wrote 32 of 32`. `check()` compares position by
position and could not tell a wrong value from a wrong order. It says which now.


---

<!-- from the README: what runs today, at the time -->

## What runs today

```
$ charsiu_run models/Llama-3.2-1B-Instruct-Q8_0.gguf -p "The capital of France is" \
      -n 64 --ignore-eos -c 512 -t 4
 Paris. The Eiffel Tower is located in Paris. The Louvre Museum is also located in
 Paris. The famous painting "The Mona Lisa" is on display at the Louvre Museum. ...

[load 138 ms | gen 64 tok in 4008 ms, 15.97 tok/s | peak 2018 MB]
[first 32 tok 16.84 tok/s, last 32 tok 15.18 tok/s, board 53 -> 51 C, cpu 2208 MHz under load]
```

with the hardware taking every projection:

```
CHARSIU_NPU=1 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 CHARSIU_NPU_MAXN=262144
```

`CHARSIU_STAGES=1` prints where a token goes, and prints it once per half of the run,
because a table averaged over the whole of it cannot answer a question about how the
run changed. On a ROCK 4D at 64 tokens:

```
  stage            first 31   last 33
  q k v              9.02       9.21
  o proj             5.42       5.62
  gate + up         19.87      20.18
  down              14.72      15.14
  output head        8.53       8.41
  attention          1.18       2.94
  total             60.4       63.3
```

Five of those rows are the hardware; the fence inside them is 1661 ms of the 1792.

Attention is 1.76 ms of the 2.9 ms between the halves, which is sixty percent of it and
is what a cost linear in the context looks like. The rest is a fifth of a millisecond
here and there across the projections -- the KV cache growing into the same DRAM the
weights are streaming through. Nothing else moved, the board **cooled** 53 C to 51
across the run, and the A72 held 2208 MHz throughout. So 60.4 ms is 16.6 tok/s and 63.3
is 15.8, and a remembered 17.1 is a shorter window on the same curve.

### gemma4 costs what a 4.6B model costs, and the head is two fifths of it

Measured on this host, everything on the CPU, 24 tokens:

```
  Llama-3.2-1B    30.6 ms a token
  gemma3-1b       30.8
  gemma-4-E2B     63.8
```

Twice the time for about twice the weights, which is the model rather than the
runtime. Per token gemma-4-E2B moves

```
  projections and the per layer gates  1048 MB   the hardware's
  per_layer_model_proj, bf16              28     the CPU's, and now vectorised
  output head, q8_0 and tied             428     the CPU's, because it is not routed
```

The per layer embedding does what its name says: the 2.35B parameter table is
looked up one row at a time, so the 1048 MB really is the "2B active" and not
the 4.6B stored. What is left is the head, and it is **two fifths of the token
on a CPU that reads at 6.5 GB/s**. At the board's measured rates that is about
167 ms a token, or 6 tok/s, against Llama-3.2-1B's measured 60.4.

### The output head is routed now, and it was worth 49 ms a token

```
                    before      after
  output head       49.81 ms    11.53 ms      44.2% of the token -> 15.6%
  a token          112.7  ms    73.9  ms
  generation         8.76 tok/s 13.07 tok/s
  weights           6.16 GB/s   7.18 GB/s
```

Same prompt, same tokens, on a ROCK 4D. Staging is also out of the prompt figure
now, so the same run reads `staging 15081 ms | prompt 6 tok in 829 ms, 7.24 tok/s`
where it used to charge all fifteen seconds of quantising to prefill and report 0.92.

### Why it was not routed before, which was not what it looked like

The suspect was the coefficient buffer. `charsiu_coef_bytes` used to bound the
coefficient surface by `k*n`, which makes it four times the weight buffer and
would put this head at 1210 MB against 151 MB of weights. That bound is a guess
and its own comment has said so since it was written: the two walls it was sized
against were tens of KILObytes -- 4.7 KB allocated against 33 KB read, and 1280
bytes against 20800 -- and nothing has ever measured the read growing with
`k*n`.

**It was not the coefficient buffer.** The shipped runner has passed
`CHARSIU_COEF_ELEMS=65536` all along, which puts the same head at 10.5 MB. What
refused it was `maxn`: the config default was 131072, chosen to clear Llama
3.2's 128256 vocabulary, and gemma3's is 262144. Raising it is the whole fix, and
the table above is the result. The runtime clamps the cap to
the model's own vocabulary anyway, so a wider number costs nothing, and the one
reason to keep a wide head off -- it used to share one output buffer with every
other tensor, so all 113 matvecs a token paid for its size -- stopped being true
when each tensor got its own.

Two things kept that invisible for four board rounds:

- the runtime **declined without saying so**. Every other refusal in npudev
  whines once per reason; this gate only spoke under `CHARSIU_NPU_VERBOSE`. The
  three silent paths -- the cap, the tensor slots, and the quantised copy
  failing to allocate -- now each name the tensor, the reason and the setting.
- the round that put `CHARSIU_NPU_MAXN=262144` on the command line **did not set
  it**. `charsiu run` builds its environment from config.ini and hands it to
  `env NAME=VALUE`, which goes in front of the inherited environment, so the
  config's 131072 silently won. The config file is the default now and the
  environment is the override; `charsiu --show` names anything it deferred to.

The staging bar hid the arithmetic too: its denominator is a prediction and its
heartbeat fires every sixteenth tensor, so it drew `183/183` over a run that
staged 182. It reconciles against the runtime's own count now.

65536 elements is the default now, because it is what every board round has used;
`CHARSIU_COEF_ELEMS=0` asks for `k*n`. The bound itself is still unmeasured and
still worth measuring. `npu_gemm_test K N --coef` walks it downward and stops at
the first value that is not exact. ⚠ It walks DOWN because under-allocating does not return an error: the
RDMA reads past the buffer, the IOMMU faults, and the job times out with every
register correct. The last exact value is the floor; everything below it is
unexplored rather than known bad.

### gemma3-1b is slower here, and the reason is not the graph

113 ms a token against llama's 60, on the same board. On a host where everything runs
on the CPU the two are within ten percent of each other, so it is not the model.

`output head 49.8 ms, 44% of the token`. Gemma-3-1B's vocabulary is 262144 against
Llama-3.2-1B's 128256, and its head is **tied to `token_embd` at q8_0**, so it reads
**321 MB every token**. It is not routed -- the gate is `n <= CHARSIU_NPU_MAXN` -- so
that lands on a CPU whose sequential read is about 6.5 GB/s, and 321 MB at 6.5 GB/s is
49 ms. The arithmetic closes; there is no bug in it.

The rest is size, and part of it is a remainder. gemma3-1b's embedding is 1152 against
llama's 2048, so its submits carry **1.92 MB against llama's 4.75** for roughly the
same fixed cost per submit, and the hardware rate falls from 10.82 GB/s to 6.16.

1152 also does not divide the 1024 K slice. `ceil(k / KMAX)` leaves the remainder in a
slice of its own, so q, k, v, gate and up each split 1024 + 128 and the second slice
carries a ninth of a slice of work for a whole task's cost -- five extra tasks in every
one of 26 layers, and 468 slices where 338 would do. Every llama dimension is a power
of two and divides 1024 exactly, which is why this had never come up.
`CHARSIU_NPU_KFIT=1` gives the last slice the remainder instead. Ungrouped tensors
only: a grouped tensor carries one scale per (channel, K group) and a slice spanning
two groups would apply the first group's scale to both. Off by default. The first
run of it on the board overran three batched buffers; with that fixed the tokens
are identical on eight models.


---

<!-- from the README: more than one architecture: the details and the two bugs -->

qwen3 is the llama graph with the three QKV biases dropped and a norm added on Q and K
-- per head, over one head's head_dim, before rope. It is also the first architecture
here whose head is not `n_embd / n_head`: Qwen3-0.6B is 16 heads of 128 against an
embedding of 1024, so attention produces 2048 floats and hands them to `attn_output`,
and the buffer that held both was sized by `n_embd` alone.

gemma3 adds a sliding window over most of its attention, **two rope bases** -- the
window layers rotate at 10000 and the full ones at the model's own 1000000, and the
file carries no key for it -- a second norm on each branch before the residual add, the
embedding times sqrt(n_embd), GELU, and a fourth chat format whose assistant is spelled
`model` and which has no system role at all.

gemma4 is the one that needed the most, and almost none of it is the attention.
gemma-4-E2B is 4.6B parameters of which about two are active, and the mechanism
that decides which is a **per layer embedding**: a second table that every layer
gates itself against after its feed forward residual. Around that:
`feed_forward_length` is an **array**, 6144 for its first fifteen layers and
12288 after; a window layer's head is **256** where a full layer's is **512**;
its last twenty layers have no `attn_k` at all and read an earlier layer's KV --
and gemma4 keeps **two caches, one per kind of attention**, so E2B's first
shared layer (15, a window layer) reads layer 13 and not layer 14, which is the
last layer with a KV of its own and is a full one. Pointing it at 14 reads 256
floats out of a slot written as 512, the residual norms stay between 0.75 and 2
the whole way down, and the model answers in a different language every token.
Its attention scale is 1.0 rather than 1/sqrt(head), it RMS normalises V with no
gain at all, and one of its tensors is bf16.

Two bugs found on the way, both of which produced **fluent English** rather than
anything that looks like a fault:

**The RoPE pairing.** There are two, and no gguf key says which one a file wants;
llama.cpp carries it as a switch over the architecture enum. llama and smollm3 are the
permuted, interleaved ones. qwen2, qwen3, phi3 and every gemma are not. charsiu had only
the interleaved form and applied it to all of them. Asked for the capital of France,
Qwen2.5-0.5B answered *"a country in the world. The capital of the world."* before and
*"Paris. It is the largest city in France"* after; Phi-3.5-mini answered *"the most
important city in the country"* and then *"Paris."* Llama is unchanged, which is the
control.

**The SentencePiece tokenizer.** charsiu segmented with a Viterbi over the piece scores,
which is the unigram objective and is not what a gguf means -- llama.cpp does a greedy
bigram merge. The two agree only when the scores really are log probabilities. Gemma
stores what is effectively minus a rank: `▁The` scores -175 while `▁T` and `he` score
-64 and -5, so the additive objective prefers the two small pieces by more than a
hundred and the prompt segments as `▁T | he | ▁c | ap | it | al`.

A third, which is neither: **a tensor that is not found is simply not used.**
Three of gemma4's per layer names are SHORTER than the model wide ones they
belong to -- `per_layer_token_embd` and `per_layer_model_proj` are top level,
but a layer's three are `blk.N.inp_gate`, `blk.N.proj` and `blk.N.post_norm`.
Guessing `per_layer_*` found nothing three times, so the whole per layer path
was skipped, and the model loaded, ran, answered and was missing the half of
itself its name is about.


---

<!-- from the README: pictures: the guesses -->

Every tensor name here was a guess -- llama.cpp's clip naming as we read it -- so the
loader was written to be **loud** before it was written to be right: it binds what it
can and reports what it could not, by name, in the order it wanted them. Pointed at
SmolVLM-256M's real mmproj it printed one missing name out of about two hundred, and
the file then corrected two things:


---

<!-- from the README: the parts that are not the hardware -->

### The parts that are not the hardware

Each of these was a board round with a control that could fail, and each is in
`board-logs/` in the driver repository:

| | before | after |
|---|---|---|
| weight packing into the NPU's layout | 8.5 ms a token | 2.8 |
| summing the K slices | 8.6 | 3.5 |
| the activation quantised for consumers that never read it | 8.1 | 0.01 |
| attention, at 384 tokens | 37.9 | 16.2 |

The last one is two changes. Four query heads share every key/value row in this
model, and the loop was reading each row four times; and the cache was laid out
`[layer][position][head]` while attention walks one head across every position.
Neither changes a number (the tokens are byte identical either way) and both
were measured against a control that put the old behaviour back.

## fp16 on the NPU, and what it is worth

2026-09-05. The vendor runs attention on the NPU in fp16: 4940 of the 8808
convolutions in its own `Llama-3.2-1B-rk3576-w4a16.rkllm`, 56% of everything it
submits, against 3328 int4 projections. 2908 of those carry `oc = 64`, which is
that model's head_dim, and their `ic` walks in steps of 32 with M chosen so the
input surface lands just under 4096 every time. We run all of it on the CPU,
where the batched stage table puts it at 30 to 52% of a prompt.

So: can this hardware do an fp16 matmul at all, and is it worth moving?

Both answers are yes and no in a specific way. The matmul works and is bit
exact against a CPU fp16 reference at every width tried, on two independent
paths. Priced on that verified computation, with the governor pinned and arms
alternating, attention would cost 3.68 ms a row against the CPU's 6.67 at a
batch of 178 -- 1.81x on attention, about 1.31x on a Qwen3 prompt. That is a
step and not a finish: the vendor's TTFT is 2.2 to 3.0x ahead of ours, and this
closes perhaps a third of it.

### What had to be found

Three things came off the vendor's file and were exact over all 4940 of its
fp16 streams: `CNA 0x1030 = (ic*2) << 16`, `CNA 0x1090 = ic/8`,
`DPU 0x4028 = oc/4 - 1`, `DPU 0x4030 = ((oc-1) << 16) | 0x310`, and
`DPU 0x40b8 = (oc/4 + 3) - (M*oc)/4`. Two came off the board, each named by a
number it returned:

  - every output word `0x80808080`, the int8 zero point in all four bytes: the
    DPU had been asked for an int8 output. An fp16 job takes the w4a16 output
    stage, and every register that switches with it lands on the vendor's own
    fp16 value.
  - a weight of 1.0 against `A[0] = 1.0` returning **3600**, and `A[8] = 9.0`
    returning **4320**. 3600 is `0x3c * 0x3c` and 4320 is `0x48 * 0x3c`: the
    high bytes of the two fp16 patterns, multiplied as int8. The output stage
    had moved and the multiply had not. `CORE 0x3018` takes its `0x200` form.

And three layouts, none of them guessed. The weight buffer is `ngroup` 16 by
`kgroup` 32 with two byte elements, measured by holing a dense buffer one
element at a time: 1024 points, no exceptions. The output is flat, `m` by `n`
row major, measured by putting `2^c` through it. The activation is plain row
major, `[m][k]`, measured by writing one value into each packed input slot and
reading which row answered.

### The part worth reading twice

fp16 was exact at `m = 1` and wrong at every width above it, and six
register-level fixes were tried against that symptom without moving the error
by a digit. The fault was in buffer content: `charsiu_pack_input_f16` writes
`[k/8][m][8]`, and this path wants `[m][k]`. At `m = 1` the interleave is the
identity, which is why one width worked and no other did.

The register stream is where the previous four answers had been, so that is
where each new one was looked for. "Where the last answer was" is not "where
this answer is", and the thing that finally found it was the same slot sweep
that had already settled the weight layout, pointed at the side nobody had
measured.

## Attention, grouped: the fence is the job

The first pricing of the fp16 matmul was per dispatch, and the vendor's file
said that could not be the whole story. At M=64 it issues **24** int4
projection dispatches a layer against charsiu's 7, and is still 2.2 to 3.0x
ahead on time to first token. Dispatch count is not what it is winning on.

Splitting one call three ways said what is: fence and sync 0.345 to 0.454 ms,
the copy out 4 to 63 us, the cache maintenance 2 to 44 us. **98% of a job is
waiting for it.** `npudev.c` has had that written down since round 321 -- "the
fence at 94% of the hardware path" -- and answers it with
`charsiu_npu_matvec_group`, which puts q, k and v in one submit. `npufp16.c`
was one matmul, one submit, one fence.

So the unit takes a group. Tasks inside one job are chained by the program
counter on a single core, so N matmuls are one submit and one wait, and the two
cores are never in flight together -- which matters here, because they corrupt
single words when they are and the fix for that is a voltage, not this.

A layer of attention is exactly that shape: H independent scores matmuls, a
softmax on the CPU, H independent values matmuls. Two waits a layer where a
loop pays 2H.

### What the board said, four rounds

Every arm below was checked bit for bit against the same ops run one at a time
before any of it was timed, at MIXED shapes -- uniform ones cannot catch an
offset that is wrong by a whole region, because a wrong region is still a legal
one. All identical, nothing refused, dmesg clean throughout.

```
  ops   shape                one at a time   grouped
  16    k=64   n=1024 m=8    1.735 ms        0.211 ms     8.2x
  32    k=64   n=1024 m=8    3.214 ms        0.211 ms    15.2x
  16    k=1024 n=64   m=8    2.687 ms        0.300 ms     9.0x
  32    k=1024 n=64   m=8    7.232 ms        0.375 ms    19.3x
```

Each round then reported where the time had gone, and the answer moved three
times.

**The coefficients were 23% of a round.** This unit builds them from a zero
bias, zero weight sums and unit scales, so their content depends on n and
nothing else -- and `charsiu_build_coefs` begins by zeroing the whole buffer,
262 kB at the default element bound. Sixteen ops of one shape were sixteen
identical copies of that. Now the plan gives one region per distinct n, the
group builds each once, and a second group with the same shapes in a buffer
that has not moved builds none: `coefs 0.000`.

**The weight memcpy was 55%.** A KV cache is appended to a row a token and
never changes, so copying it into a device buffer is pure waste.
`charsiu_fp16_w` is a buffer the caller writes rows into and the hardware reads
where they lie. 1.29 to 1.36x at eight rows.

**Then the pack, and the copy out.** With the fence amortised and the copies
gone, the largest line was converting the activation to halves: a call into
another translation unit per element, a bounds test per element, two byte
stores per element, and a memset of the whole region first. And an op that is
about to have a softmax run over it does not need its answer copied anywhere: a
NULL `Y` leaves it in the device buffer.

### The appending law, and the half of the test that could fail

All of that rests on one claim: a GROUP offset is

    (n/16)*16*ke + (k/32)*32*ngsz + (n%16)*kgsz + (k%32)

and only `ngsz` carries n, and only for a partial last group -- so while every
group of 16 output channels is full, **an offset does not depend on n at all**,
and a cache can be appended to and read at whatever width it has reached.

That was a derivation, and this project has published derivations the board
then refused. So `tests/pack_f16w.c` checks it on every shape, and also checks
that a partial group really does move -- otherwise the law would be passing
because it cannot fail. The second half is what found the exact precondition:
`ngsz` reaches the offset only through `(k/32)*32*ngsz`, so with a single k
group it cancels and nothing moves at all. Above one k group, which is where a
cache lives, the difference is real. Then the board was asked the same question
end to end: buffers allocated at 2n, written through `charsiu_fp16_woffset` at
that width, run at n. Identical.

### Where it ended, per matmul

Owned weight, borrowed answer, back to back, against the same matmul alone on
the first round:

```
  shape                    alone      grouped
  k=64   n=1024 m=8  G=16  0.989 ms   0.073 ms
  k=1024 n=64   m=8  G=16  1.245 ms   0.121 ms
  k=64   n=1024 m=80 G=16  2.251 ms   0.470 ms
  k=1024 n=64   m=80 G=16  1.922 ms   0.277 ms
```

⚠ **That is not a speedup over the CPU and must not be read as one.** These
shapes hold a cache 1024 positions deep, and the 6.62 ms a row the CPU spends
on attention was measured on a 256 token prompt, where the cache averages a
fraction of that. The two numbers are not comparable and nothing in the model
calls the group yet. This project has already published one attention speedup
that was void; the honest statement is the cost of a matmul, and the comparison
belongs to the integration that makes it.

### Two things this did not settle

`CHARSIU_FP16_JOBS=split` sends N jobs instead of N chained tasks, so the
scheduler may use both cores. It was faster in one round and slower in the
next, at different group sizes, and both were correct. Nothing here decides it.

And nothing in the model calls any of this yet. The next step is the KV cache
written in `charsiu_fp16_woffset` layout as tokens are appended, and
`attn_block` calling the group.

## The integration, and the number it actually produces

The group went into the model behind `CHARSIU_ATTN_NPU`: a layer is H scores
matmuls in one submit, a softmax on the CPU, H values matmuls in a second.

**It is correct.** On a Radxa ROCK 4D, Llama-3.2-1B, a 513 token prompt, with
`CHARSIU_ATTN_NPU_CHECK` running both arms on the same rows: 112 layers on the
hardware, none fell back, and the two arms land 0.25 to 0.5% apart --

```
  layer  9, 32 rows: worst 0.003501 against a largest of 0.8552 (0.0041 relative)
  layer 13, 32 rows: worst 0.009572 against a largest of 2.161  (0.0044 relative)
  layer 15, 32 rows: worst 0.0196   against a largest of 4.702  (0.0042 relative)
```

which is what fp16 inputs against an fp32 CPU arm should give. A cache written
in the wrong order or a stride applied to the wrong axis would be order 1, not
0.004. Comparing the sentences could not have told those apart, which is why
the check exists.

**And it is slower.** Same prompt, alternating, spread under 2%:

```
  attention on the CPU   4889, 4889, 4970 ms
  attention on the NPU   6710, 6306 ms          1.36x SLOWER
```

### The explanation I had was wrong

The values matmul runs at k = the context length whatever the prompt has
reached, because its k cannot grow. That looked like the term: at a 1024
context and an average cache depth of 256 it is four times the reduction
anyone needs. So the context was cut to 576, which is 1.78x less values work
for the same prompt.

It moved 6%. **6710 to 6306.** The reduction width is not what this is
spending, and the fix I was about to build for it would have bought almost
nothing.

What is left is the per matmul cost, and it is exactly what the probe already
measured: about 0.3 to 0.5 ms at 80 rows, times 32 heads, times two halves,
times 16 layers, times seven chunks. The hardware is not slow at this. There
is simply more of it than the CPU needs to do, because the CPU'''s attention
grows with the cache depth and at 513 tokens the cache is shallow.

`CHARSIU_FP16_JOBS=split`, which lets the scheduler use both cores, made it
6394 against 6306. Not that either.

### And the deep cache does not save it either

The one shape left was a deep cache: the CPU's attention is linear in depth and
a dispatch cost is not, so a long prompt should have been where this turns. It
is not.

```
   prompt   cache      CPU               NPU
   513 tok  ctx 1024   4889, 4919 ms     6710, 6736 ms    1.36x slower
  2074 tok  ctx 2560  35686, 35925 ms   52558, 53029 ms   1.47x slower
```

Both depths repeated, and the spread is under 1% in every arm.

**The gap WIDENS with depth.** Which is obvious once the number exists and was
not before it: the per matmul cost is not fixed either. The scores matmul's n
is the number of positions and the values matmul's k is the context, so the
hardware's work grows with the cache exactly as the CPU's does, and it starts
from behind. There is no shape in this range where the fp16 attention path is
the faster one.

So for text on this model the path is finished, and it is worth being plain
about that rather than leaving it looking like an unfinished lever. What
remains true and useful: the group underneath is 8 to 19x over one matmul at a
time, every op it runs is bit exact, the caches are written where the hardware
reads them, and the whole thing is correct to fp16 in the model. It is the
right unit; attention on this hardware is not the right customer for it.

Off by default, and with the flag unset four models produce text identical to
the build from before any of this existed.

## The morning after: the CPU stages nobody had looked at

The fp16 attention round ended by closing a door, so the next one began by
pricing the room. The batched stage table on the board, Llama-3.2-1B, a 512
row prompt, governor pinned:

```
   matmuls (q k v, o, gate + up, down)   5.33 ms a row   56.7%
   attention                             2.55 ms a row   27.1%
   silu * up                             1.07 ms a row   11.4%
   everything else                       0.44 ms a row    4.7%
```

Two things fell out of it immediately. Attention is 27% and not the 50% the
README quotes -- that number is Qwen3's, which has 28 layers against this
model's 16. And `silu * up`, third largest, was a bare `for (r)` loop with
`charsiu_parallel_for` sitting beside it: the attention has used that pool for
3.5x since the block work and this stage never had.

Pooling it, and then the three smaller elementwise stages with it:

```
   CHARSIU_ROW_POOL=0   4919, 5002, 4932, 4859, 4890 ms
   pooled               4584, 4537, 4507, 4354, 4554 ms      8.6% faster
```

`silu * up` went 1.07 to 0.24 ms a row, 4.5x -- better than the 3.5x the
attention gets from the same pool.

### The softmax, and the trade that was already the default

What was left in the attention was the softmax: three scalar passes with an
`expf` AND a divide an element, between two passes that have been NEON since
round 370, with `charsiu_vexpq` in the header the file already includes.

Vectorised, it is 3.8% of the whole prompt on the board, and every one of five
runs beat every one of the five controls:

```
   CHARSIU_EXACT_SOFTMAX=1   4516, 4443, 4485, 4426, 4570 ms
   vectorised                4328, 4280, 4309, 4366, 4299 ms
```

⚠ It moves tokens. On the development host, on a prose prompt, two of four
models take a different branch. The two implementations were measured against
each other over 2000 rows at ten widths first -- worst relative difference
1.9e-06, which is fp32 rounding and not a defect -- and then the question that
decides it was asked: **does the tree already ship this trade?** It does. The
silu that has been the default for months uses the same `charsiu_vexpq`, and
switching it off with `CHARSIU_EXACT_SILU` moves three of the same four models.

On the board, both arms print the same sentence.

### Where the prompt stands

9.39 to 8.21 ms a row, 12.8% off a prompt, in two changes that touch no
arithmetic the hardware does. What is left is 5.3 ms a row inside the NPU
matmul entry -- 64% of the row now -- and the stage table has never said
whether that is the hardware or the pack in front of it. It says so from this
commit on.

## Where a batched prompt's time actually is

The stage table learned to split the NPU matmul entry, and then to split the
pack inside it. On the board, Llama-3.2-1B and Qwen3-0.6B, governor pinned:

```
   llama  entry 4.86  pack 1.20 (gather 0.33  packer 0.61)  fence 1.32  read 2.23
   qwen3  entry 4.50  pack 1.62 (gather 0.20  packer 1.11)  fence 1.18  read 0.98
```

The two models do not have the same bottleneck. That is the whole argument for
splitting a number rather than naming it: "the matmuls are 60%" would have sent
both of them at the same repair.

### What the morning moved

Three changes, each measured on its own alternating arms, then all three
together against all three off in one binary:

```
             all off      all on
   llama      4948 ms      4144 ms     16.2%
   SmolLM2    5590         4461        20.2%
   Qwen3     12625        12381         1.9%
```

Qwen3 gains almost nothing from the set, and taking each one away in turn says
each is still positive on it -- row pool 4.3%, softmax 2.6%, read threshold
14.3%. They do not add, because all three run on one thread pool and warm it
for each other. The set is the best configuration for all three models; the
decomposition is not additive and should not be quoted as if it were.

### And the packer, which is at the memory roof already

Splitting the pack said the packer call is its largest piece, so it went on the
pool too, by GROUPS of 8 k rather than by rows -- the function had already
measured why: the destination is a cold write-back mapping, PREP_BO invalidates
it immediately before, and rows outermost falls to 6.87 GB/s against 24.57.

```
   packer serial vs pooled     llama 2.9%    SmolLM2 6.4%    Qwen3 0.3%
   llama's packer              0.61 -> 0.40 ms a row
```

1.5x from four cores, not 4x, and the reason is in that same note: one thread
already reaches 24 GB/s on this, which is most of what the memory gives. Qwen3
issues 2352 packs a prompt against Llama's 784, and the extra barriers eat the
difference exactly.

### The one that was not a speedup at all

The norm and residual stages went on the pool with the silu, and rmsnorm keeps
its gain row in a plain function static. Qwen3 stopped being reproducible: ten
runs of one seed, eight of one sentence and two of another, where the commit
before gave ten of ten. The first reading was "that model is nondeterministic
anyway" -- true of the build with the bug and false of every build before it.

What settled it was running one binary ten times and then the previous commit
ten times. **A difference between two arms cannot be told from a difference
between two runs without running one arm twice.**

## Two ways to make the read cheaper, and the board refused both

The read is the largest piece of a batched matmul, and `read_rows`' own note
had already reasoned out the physics: it is bandwidth bound, a run is 16 bytes
against a 64 byte line so the DRAM sees four times the useful traffic, "it was
never instruction bound ... the only lever left is fewer BYTES." It named two
ways to get them down.

**Whole lines, four rows at a time** was tried before: 2.3x slower on eight
models, because four rows off one line means four write streams whose rows sit
n floats apart and the A72's store buffer does not merge them.

**One pass over Y for all the K slices a device holds** is the other, and it is
what a partial sum invites: with s slices Y is read and written s times. The
counters said 2240 slot reads over 1568 ranges on Llama, so 672 round trips
were available.

```
   Llama-3.2-1B   plain 4070, 4062, 4082 ms   fused 4195, 4121, 4179   2.3% SLOWER
   Qwen3-0.6B     plain 12200, 12313, 12195   fused 12376, 12218, 12175   flat
```

It trades s sequential Y round trips for s scattered source streams live at
once, and the scattered side costs more. Both levers are now measured. The read
is at its floor for this accumulator layout.

### The part that is worth more than the result

The commit that added the fused read said **"bit exact by construction"**. It
was not. The derivation was about the ORDER OF THE ADDITIONS -- `y = c0;
y += c1` against `v = c0; v += c1` -- and it never checked that the OPERANDS
were the same. They are not: a slot's scale is

```
   s->sc[j] = t->scale[(n0 + j) * ng + k0 / kgroup]
```

so it depends on the slice's own `k0` and every K slice has a different one.
The loop used the first slice's for all of them. Llama came back
"000alivherherher" and Qwen3 counted 251, 254, 256 where it should have counted
257, 258, 259. Neither faulted, because a wrong operand does not fault.

A proof about one half of an expression is not a proof about the expression.

## A day of instruments, and four numbers that were correct about the wrong thing

The prompt got 17 to 21% faster on 2026-09-06 and the gap to the vendor went
from 1.85-2.40x to 1.53-1.88x. Every one of the changes came out of splitting a
number that had been reported whole, and every one of the day's mistakes came
out of reading a number whose label did not describe it.

### The measurements that worked

The stage table learned to split the NPU matmul entry into pack, submit, fence
and read, and then the pack into its gather and its packer call. That is what
found the pooled read's threshold -- the largest single change of the day -- and
it is also what showed that **Llama and Qwen3 do not share a bottleneck**:
llama's read is 2.23 ms a row against qwen3's packer at 1.11. "The matmuls are
60%" would have sent both at the same repair.

### The four that were correct about the wrong thing

**1. "-2438 ms of them is neither hardware nor packing."** Printed on every run
for a whole morning. `busy_us` is incremented in three call paths and `call_us`
in two, so a batched prompt put all its hardware time in the numerator and none
in the denominator. Nobody read the minus sign -- including the person who then
quoted the "2.36 GB/s of weights" beside it as a fact about the silicon and
went looking for a hardware problem.

**2. "GB/s of weights."** The window it divides by runs from the first submit to
the last fini, and on a batched call the READ BACK is most of it. Same binary:
2.05 GB/s on a prompt, 9.73 on a decode-heavy run. The number moves because the
readback's share moves, not because the hardware does.

**3. Two reports, four counters, one reset.** The stage report passed
`reset = 1` and npudev's report, which prints after it, divided by the
leftovers: "57 ms submitting and waiting" on a run whose fence alone was 189.
The one that prints first is not the counter's owner.

**4. "Bit exact by construction."** The fused read's derivation was about the
ORDER OF THE ADDITIONS and never checked the OPERANDS. Every K slice carries
its own scale, `t->scale[(n0+j)*ng + k0/kgroup]`, and the loop used the first
slice's for all of them. Llama printed "000alivherherher"; Qwen3 counted 251,
254, 256 for 257, 258, 259. **Neither faulted.** A proof about half an
expression is not a proof about the expression.

### And two about experiments rather than instruments

**A data race that read as a property of the model.** rmsnorm keeps its gain row
in a plain function static, which was safe until the norm stages went on the
pool. Qwen3 stopped being reproducible and the first reading was "that model is
nondeterministic anyway" -- true of the build with the bug and false of every
build before it. What settled it was running one binary ten times and then the
previous commit ten times: 8/2 against 10/0. **A difference between two arms
cannot be told from a difference between two runs without running one arm
twice.**

**Two variables in one arm.** The chunk width experiment set
`CHARSIU_NPU_KMAX=1024` and `CHARSIU_PREFILL_CHUNK=160` together and the result
was read as a KMAX effect. Separated, KMAX alone changes nothing at all -- the
chunk default is a hardcoded 80 and the surface ceiling can only lower it. The
whole effect was the chunk. **Print the baseline before comparing anything to
it.**

## The chunk width: a suspicion three months old, and the cliff under it

The default chunk is 80 and the note beside it already said where that came
from: *"96 ALSO CAME BACK IDENTICAL, so 80 is where the VENDOR stops and not
where the hardware does. It was in the sweep for exactly that reason: a sweep
that stops where they stop cannot tell a ceiling from a choice."* And then
`board_chunk_sweep.sh` walked 32 31 30 29 24 16 4 2 -- every one of them BELOW
the default. The suspicion was written down and the sweep never went up.

Going up finds two things.

**Qwen3-0.6B is 8% faster at 160**, at every prompt length measured, with zero
rows falling back. **Llama-3.2-1B is 5% slower.** And at 224 both fall off a
cliff -- Qwen3's 404 token prompt goes 3272 to 13507 ms -- because the batched
path refuses and every token takes the token loop.

### The derived default that shipped for twenty minutes

The mechanism looked clean: npudev refuses a dispatch whose input surface
`(k_slice / 32) * m` exceeds 5120, so the widest chunk that forces no extra
slicing is `163840 / k`. Qwen3's n_embd is 1024 and gives 160; Llama's 2048
gives 80. Both matched.

SmolLM2-135M came back **77% slower**.

The stage table said what, in one line: `2.89 ms a row on the CPU (27480 rows
fell back)`, and `down` going 0.31 to 2.65 ms a row. Not a slowdown -- a
REFUSAL. And the reason is that **`down` reads n_ff, not n_embd**: SmolLM2's
n_ff is 1536, `auto_kmax` widens it to 2048 so the slice stays 1536, and
`(1536/32) * 160 = 7680`. The formula was right and it was applied to the wrong
K.

The right one is `163840 / min(widest K, effective KMAX)`, and effective KMAX is
itself per model because `auto_kmax` only widens where no tensor is grouped:

```
   Qwen3      1024 is a multiple of 1024, so KMAX stays 1024
              widest slice 1024   cap 160   measured 160: 0 fallbacks, fastest
   SmolLM2    576 and 1536 are not, so KMAX widens to 2048
              widest slice 1536   cap 106   measured 106: 0 fallbacks, fastest
   Llama      2048 is a multiple, KMAX 1024, widest slice 1024, cap 160
              but 160 is 5% SLOWER -- legal is not the same as good
```

### What shipped is the clamp, not the default

80 is still what a caller gets. `CHARSIU_PREFILL_CHUNK` is clamped to the cap,
so the knob is now safe to raise on any model:

```
                 ask 160        ask 400        default 80
   Qwen3      160  11216 ms   160  11094         12149     -8%
   SmolLM2   ->106  4154     ->106  4217          4283     -3%   (was 6550)
   Llama      160   4301      160   4282          4105     +5%
```

SmolLM2's 77% regression becomes a 3% gain, and asking for 400 lands safely
everywhere. Llama still wants 80, which is why the cap is a ceiling and not a
recommendation: **legal is not the same as good, and this file now has both
numbers for all three.**

## The fence was never only a wait

The fence was the last large item in a batched row that nobody had looked
inside. On Llama-3.2-1B it is 1.36 of 7.75 ms a row, and every reading of it in
this file -- *"the hardware is BUSY"*, *"a fence removed is worth more than a
submit removed"* -- had taken it for time spent waiting for the NPU.

`rocket_ioctl_prep_bo` is two things:

```c
ret = dma_resv_wait_timeout(gem_obj->resv, DMA_RESV_USAGE_WRITE, true, timeout);
...
dma_sync_sgtable_for_cpu(dev->dev, shmem_obj->sgt, DMA_BIDIRECTIONAL);
```

A wait, and then a cache invalidate over the WHOLE buffer. One ioctl, one
number, two things -- the same shape as the three instruments corrected the day
before: true about something other than its label.

### Asking for the second one twice

A second `prep` on a buffer whose fence has already signalled waits for nothing.
`dma_resv_wait_timeout` returns at once and what is left is the invalidate, over
the same bytes, through the same ioctl. `CHARSIU_FENCE_SPLIT=1` does that and
charges it separately.

The probe cannot be trusted on its own say-so, so the round ran the plain binary
twice first and read the split arm against that spread:

```
   llama, 512 rows       fence   invalidate   the wait
     plain 1              1.36        -           -
     plain 2              1.35        -           -
     split                1.39      0.30        1.09    1.99 GiB, 14.03 GB/s
```

1.36, 1.35, 1.39: the probe moves the number it measures by less than 3%, which
is what makes the 0.30 worth reading. Qwen3 and SmolLM2 came back at the same
ratio -- 0.24 of 1.15, 0.19 of 0.52 -- so **about a fifth of every fence on this
board is cache maintenance, not the hardware.**

The bytes are printed beside the microseconds on purpose. A rate whose
denominator holds something other than the transfer it names is exactly how
"2.36 GB/s of weights" got quoted here as a fact about the silicon.

### And the other four fifths are not the weight fetch

The obvious next story was that the wait is the weight fetch: the NPU reads
every weight once per chunk, so a prompt in 7 chunks reads them seven times, and
1.09 ms a row over 512 rows is 558 ms, which is about 4.2 GB at this board's
measured roof. It fits.

It is wrong, and one arm says so. Doubling the chunk halves the number of passes
over the weights:

```
   llama, 512 rows      chunks   fence/row   read/row   prompt
     chunk  80             7        1.37       2.20     4093 ms
     chunk  80             7        1.35       2.27     4107 ms
     chunk 160             4        1.42       2.68     4262 ms
```

Half the weight traffic, and the fence per row did not move -- it rose. **The
fence scales with rows, not with weight passes**, so whatever the NPU is doing
in those 356 us it is not waiting on DRAM for weights, and the "hardware is the
floor" reading that has stood since the int4 work does not extend from decode to
prefill.

What DID move is the read, up 22% on the wider chunk, on a model whose widest
tensor is 8192 -- while Qwen3, whose widest is 3072, held flat at 0.93. That is
a working set, not a bandwidth, and it is the first mechanism this file has for
why the best chunk is per model.

## The chunk curve, and a knob that had been sitting in the source

The chunk had been read twice before: once by a sweep that only went DOWN from
80, and once by a sweep that went up and found 160 good for Qwen3 and bad for
Llama. Neither had a stage table beside it, so neither could say what moved.

With one:

```
                     16      32      48      80      80     120     160
   llama ms a row   9.50    8.48    8.16    7.79    7.78   7.67    8.12
   qwen3 ms a row  15.86   14.89   13.90   13.19   13.30  12.65   12.25
   smol  ms a row                           4.66    4.60          4.54 (->106)
```

Llama has a minimum at 120 and turns up after it. Qwen3 is still falling at its
own ceiling. SmolLM2 is still falling at its own ceiling, which is 106.

And the entry says which column does it. Between 80 and 160, on Llama, `read`
goes 2.20 to 2.68 while everything else holds; on Qwen3 it goes 0.96 to 0.93.
**Llama's widest tensor is 8192 and Qwen3's is 3072**, and the read walks the
output buffer of one slot, which is `widest * m * 4` bytes. That is the first
mechanism this file has had for why the chunk is per model -- the earlier note
said "whatever hurts a small model at a wide chunk has not been found", and the
answer is that it is not the small model that is hurt, it is the WIDE one.

`min(cap, 120)` is faster than 80 on all three. It is not shipped as a default
yet, because 120 is Llama's minimum and Llama is the only one of the three whose
curve has a minimum below its ceiling -- a rule fitted at the single point that
exercises it is the mistake this file has already made once.

### read_rows2: measured, at last

Underneath `read_rows` sits `read_rows2`, with a comment ending *"whether two is
on the right side of the A72's store buffer is a board question, and
CHARSIU_NPU_READ4=2 asks it."* Nothing in this repository had ever asked it.
`read_rows4` -- four rows off one line, four write streams -- lost 2.3x, and the
two-row form went in beside it and was never run.

```
                 ms a row    read     text
   llama plain     7.81      2.23     6c2e11a4...
   llama plain     7.78      2.23     6c2e11a4...
   llama read2     7.69      2.06     6c2e11a4...   identical
   qwen3 plain     13.35     0.96     955de80d...
   qwen3 read2     13.21     0.92     955de80d...   identical
```

Both plain arms agree on the read to the hundredth, so 2.23 to 2.06 is 7.6% and
not weather. Llama gains more than Qwen3, which is the same width story again:
2.23 ms a row of gather has more to save than 0.96.

### And it was applying by accident

`pool_arm` gives each worker `n / (4 * threads)` rows. At m = 80 with eight
threads that is 2, and the pair form needs an even range, so it applied. At m =
120 it is 3 and the pair form would have returned 0 on every range and fallen
back to the scalar path -- silently, because falling back is what it is supposed
to do when it cannot help.

So the 7.6% was measured at the one width where an unrelated division happened
to come out even. `charsiu_parallel_for_grain` rounds the chunk up to a multiple
of the grain and the read asks for 2 when the pair form is on. Nothing that did
not ask for a grain changes.

## The whole table, and the stage nobody had been looking at

Three rounds read the five columns inside the NPU entry and none printed what
sits outside them. With `prep` added as the fifth segment the entry closes to
0.01 ms a row on Llama, and the rest of the table can finally be read:

```
   Llama-3.2-1B, 512 rows, chunk 80, 7.73 ms a row
     token embedding   0.00    0.0%
     attn rmsnorm      0.05    0.6%
     q k v             0.70    9.1%
     rope + kv copy    0.15    1.9%
     attention         2.34   30.2%     <-- CPU
     o proj            0.40    5.2%
     residual          0.07    0.9%
     gate + up         2.29   29.6%
     silu * up         0.25    3.2%
     down              1.45   18.7%
     residual          0.04    0.6%
   of the entry: pack 1.11  submit 0.08  fence 1.33  read 2.21  prep 0.03
                 unaccounted 0.01
```

**Attention is the largest single stage in a prefilled row, larger than any
matmul**, and all of it is on the CPU. On Qwen3 at a 916 token prompt it is
**8.36 of 13.15 ms a row, 63.6%** -- attention is linear in the context and
Qwen3's prompt is longer, so the same code reads as a third of one model and
two thirds of another.

It is not bandwidth bound. The block already walks the KV cache once per 8 query
rows, which puts Qwen3 at about 1.6 GB/s against this board's 9.4 roof, and the
scores kernel is eight NEON accumulators over four keys at a time. What it IS
running at is about a fifth of the CPU's fp32 peak, and **the block was swept on
Qwen3, SmolLM2 and gemma-3 and never on Llama** -- gemma-3 was still falling at
32 when 8 was chosen as the smallest inside the spread of the best.

### The vendor protocol's prompt fits in one chunk, and did not get one

Their benchmark is a 128 token prompt; ours tokenises to 110 to 116. The surface
ceiling on these models is 160. It ran as 80 and then 30.

```
                base TTFT   onechunk    gap was -> now
   Qwen3           721         643      1.53x -> 1.37x
   TinyLLAMA       980         918      1.80x -> 1.69x
   Phi3           3160        3028      1.73x -> 1.66x
   Gemma4         2399        2133      1.97x -> 1.75x
```

**Decode is unchanged in both arms**, which is the control: a chunking change
cannot touch it, and if it had moved, the arm would have been measuring the
board rather than the change.

That control is not decoration. The `read2` arm of the same round showed Qwen3's
TTFT at 1008 against 721 AND its decode at 20.71 against 24.74 -- and `read4` is
read only by the batched gather, so it cannot reach decode at all. The next
arm's decode came back to 24.74. Arms run in BLOCKS track the state of the
board; the alternating round is the one to believe.

## A door that was closed on an allocation

fp16 attention on the NPU is implemented, was measured slower at every cache
depth, and was shut. Attention has since turned out to be the largest stage in a
prefilled row, so the verdict was worth re-reading -- and the note above the
implementation contains its own refutation:

> For the V cache a position is part of the reduction, so its k cannot move: it
> is allocated at the context length and the matmul always runs there, with the
> probabilities past the last token left zero. **That costs a fetch of the whole
> V surface every call** and buys never repacking.

Every round that closed the door ran `-c 2048`. A 916 token prompt then pays
2.2x, a 110 token one 18x. And the cost is doubled by a second route: the input
surface ceiling is `(k / 32) * m <= 5120`, so a wider k also buys fewer rows a
pass -- 80 at a 2048 context, 160 at 1024.

So: the same two arms, at a context barely above the prompt.

```
                    CPU attn    NPU attn
   llama -c 2048      2.34        6.50
   llama -c 640       2.28        4.16
   qwen3 -c 2048      8.46        7.61
   qwen3 -c 1024      8.39        5.97
```

**The CPU arm does not move** -- 2.34 to 2.28, 8.46 to 8.39 -- which is the
control that makes the rest readable: its loop runs to `pos`, not to `n_ctx`, so
if it HAD moved the round would have been measuring something else. The NPU arm
moves 36% on Llama and 22% on Qwen3 for no reason but the size of a buffer.

And on Qwen3 the NPU path **already wins at -c 2048**, and wins by 29% at 1024,
where its V surface is only 1.12x oversized and there is little left to take.
Attention is 63% of a prefilled Qwen3 row, so that is 18% of the whole prompt.

The split is by head_dim: Qwen3's is 128 and the NPU wins, Llama's is 64 and the
NPU loses at every context tried. That is the shape you would expect -- a wider
head is a wider matmul -- and it means this is a per model choice rather than a
default.

### What the fix has to preserve

`kv` now climbs a doubling ladder instead of sitting at the context length, so a
run repacks at most log2 of the context many times. Three things had to survive
it:

- **the layout.** It would be easy to write the new V surface out by hand in the
  growth path and easy to get it wrong. Instead the repack walks the FLOAT V
  cache -- the source of truth, which both paths write -- and hands each
  position to the same `charsiu_fp16_pack_vcol` the append path uses. There is
  still exactly one copy of that layout in the file.
- **the zeros.** The values matmul is legal only because the probabilities past
  the last token are zero. Those live in a scratch whose rows are `kv` apart, so
  a changed `kv` reinterprets every byte of it. It is re-zeroed on every rung --
  a memset, not a reallocation, because `mmax * kv` is `5120 * 32` on every rung
  and the size does not change.
- **the refusal.** `attn_npu_layer` now refuses a T its surface does not cover.
  Falling back to the CPU is always safe, because the float cache is written
  either way; a stale surface is a wrong answer.

### And the values kernel was four positions wide

`attn_axpy4` is deliberately not an FMA -- `vmulq`, a barrier, `vaddq` -- so it
rounds twice, as the scalar reference does. That stays. What was free was the
WIDTH: per four output floats the loop does four multiplies, four adds, four V
loads and one load and store of the output, so it is the output traffic that
bounds it, and eight positions a call halves that per multiply. The additions
still go a0, a1, a2 in order; what disappears is a store and a reload of a
float32 in between, and a float32 that goes to memory and comes back is the same
float32. `tests/axpy8` checks one eight wide call against two four wide ones
over 520 shapes and finds no case differing in a single bit.

## Two hashes that agreed for the wrong reason

The onechunk knob had passed `board_text_all.sh` on all nine architectures, so
the next step was to make it the default. Two rounds nearly did it on evidence
that was not evidence.

**The first**: eight of those nine prompts are 86 to 88 tokens, which is above
the chunk of 80 and below the ceiling, so the knob engaged. The ninth --
Llama-3.2-1B -- runs a 64 token prompt. `n_ids > chunk` is false at 64, the knob
did nothing, and its "text identical" said only that the code without the knob
still works.

So a round went out with a 100 number prompt, three arms, and the token loop as
the reference. All three hashes matched. They matched because `seq 1 100`
tokenises to about 200, which is over Llama's 160 ceiling, so `n_ids <= cap` was
false and the knob did nothing again. **The round printed the widths beside the
hash and the widths said `2x80+1x40`** -- which is what a chunk of 80 does, and
not what one chunk looks like.

**The second**: two of that round's three models produced
`d41d8cd98f00b204e9800998ecf8427e` in every arm. That is the md5 of the empty
string. `/opt/charsiu/models` holds three models and the rest live in
`~/.charsiu/models`, so those runs found no file, printed nothing, and hashed
nothing -- identically, in all three arms.

A missing model reads as "text identical" unless something checks that the model
ran. A knob that does not engage reads as "text identical" unless something
checks that it engaged. Both are the same failure as the instruments corrected
the day before: **a true statement about something other than what the label
says**, and in both cases the thing that caught it was a second line printed
beside the first.

The prompt that actually engages it is 45 numbers, 91 tokens, and the widths it
currently runs are `1x80+1x6+1x4` -- a chunk of six and a chunk of four, each
paying a fence, a pack and a read on every tensor of every layer.

### And the rule has four points

With the paths fixed -- `/opt/charsiu/models` holds three models and the rest
live in `~/.charsiu/models`, which is what produced the empty hashes -- the two
missing head widths ran:

```
                              CPU attn   NPU attn   whole prefill
   hd 256  gemma-3-1b           6.29       1.50      12.43 -> 7.85   -37%
   hd 128  Qwen3-0.6B           8.13       5.51      13.15 -> 11.34  -14%
   hd  64  Llama-3.2-1B         2.16       3.72      a loss
   hd  64  SmolLM2-135M         2.59       3.13      a loss
```

gemma-3's CPU arm repeats to 0.9% either side of the NPU one, so the 76% is not
weather. Monotone in head_dim, with a mechanism that does not need fitting: a
wider head is a wider matmul and this hardware wants width. `CHARSIU_ATTN_NPU=
auto` puts the crossover at 128.

**It is still off by default, and not because of the clock.** Everything else
turned on today -- the pair read, the eight wide values, the one chunk prompt --
shipped on a text hash that did not move. This one computes attention in fp16
where the CPU computes it in fp32, so the answer can differ. That is a decision
about the output, not about the speed, and it is not this file's to make.

## What this table can and cannot say

`board_vendor.sh` has warned since it was written that one reading of its TTFT
column has a large spread. Today put numbers on that. The same build, the same
governor, minutes apart:

```
   Qwen3 TTFT     721   924   729   707..784
   Gemma4 TTFT   2133  2182  2185  2325  2408  2707  ...and 3221 inside one arm
   Phi3 TTFT     3155  3149  3004..3167
   TinyLLAMA      965   987   915..1033
```

Phi-3.5 repeats to 0.2% and TinyLLAMA to 2%. **Qwen3 swings 27% and Gemma4
swings 51%.** So a change worth 5% can be attributed on two of these four models
and cannot be attributed on the other two at any repeat count this harness runs.

That is not a reason to drop them from the table -- they are the vendor's rows
and the comparison is the point -- but it is a reason to stop reading their
column as a measurement of anything charsiu did. Every attribution in this file
today comes from either the stage table, which compares inside one run, or from
Phi-3.5 and TinyLLAMA.

And the decode column has its own shape: nearly every arm shows one low outlier
(Phi-3.5 reads 4.66 and 6.84 in the same three runs), which is why the script
reports the best and prints the range beside it.

## Half a rule, shipped as a whole one

`CHARSIU_ATTN_NPU=auto` went in on four points that were monotone in head_dim
and had a mechanism that needed no fitting. It was measured on the vendor's own
protocol the same hour and it lost:

```
                   shipping   auto    head_dim   the gate
   Qwen3              707      954      128       ON    decode 24.63 -> 20.84
   TinyLLAMA          915      919       64       off   unchanged
   Phi3              3004     3019       96       off   unchanged
   Gemma4            2408     2527     >=128      ON    decode  8.70 ->  7.57
```

The gate is not the error -- it fires on exactly the two models it was meant to
and the two it skips do not move at all, which is as clean a control as this
harness gives. The error is that **every number the rule was built on came from
a 916 token prompt**, and the vendor's protocol is 110 tokens and 64 generated.

**The decode column is what says so.** A prefill change cannot touch decode, so
decode falling 15% is not the attention being slower -- it is something else
being paid per token. It is: the decode path appends to the fp16 mirror
deliberately, because a prompt continued after a generation would otherwise read
a cache with a hole in it. A generation pays to fill a mirror it never reads.
And the other hidden cost is the mirror's construction itself, which on 110
positions outweighs what fp16 attention saves.

So the envelope is a wide head AND a long prompt. Where the second one starts is
not known: 110 loses, 916 wins, and nothing has been run in between. **A
threshold placed between two points eight times apart is the chunk formula
again**, which this file recorded going wrong this morning, so `auto` is
withdrawn rather than guessed at. The knob stays and the note above it now says
where it pays.

## Where fp16 attention starts paying, and a rule that has to be tested where it hurts

`auto` was withdrawn because 110 tokens lost and 916 won and nothing had been
run in between. This is in between.

```
   prompt tok      39     135     279     532     916
   Qwen3   CPU    335     915    2168    4896   11347   ms
   hd 128  NPU    647    1441    2780    5496   10476
           ratio 1.93    1.57    1.28    1.12    0.92   <- crosses in here

   prompt tok      41     137     281     534     918
   gemma-3 CPU    541    1112    2445    5648   11400   ms
   hd 256  NPU    472    1170    2242    4090    7261
           ratio 0.87    1.05    0.92    0.72    0.64
```

Qwen3 is monotone and crosses between 532 and 916. gemma-3 wins nearly
everywhere and by 36% at the top; its 137 point is the one that goes the wrong
way, and it is also where the absolute times are smallest.

### One quantity separates eleven of twelve arms

Across every fp16 attention arm measured -- four models, five lengths --
`head_dim * prompt_tokens` puts every loss at or below 68096 and every win at or
above 71936:

```
   qwen3   128 *  532 =  68096   loss      <- the largest loss
   gemma3  256 *  281 =  71936   WIN       <- the smallest win
   qwen3   128 *  916 = 117248   WIN
   llama    64 *  512 =  32768   loss
   smol     64 *  916 =  58624   loss
```

The exception is gemma-3 at 41 tokens, product 10496, which won -- and is the
smallest and noisiest measurement on the list.

**That is a fitted threshold, and this file has recorded two of those going wrong
today.** So it does not ship on the fit. The two candidate rules disagree
somewhere specific, and that is what to run: **head_dim alone says Llama, at 64,
never wins at any length. The product says Llama wins past about 1100 tokens.**
One of those is about to be false.

### And the product rule is dead

Llama, head_dim 64, at 1101 tokens. `64 * 1101 = 70464`, which is above the
threshold the fit produced, so the product rule predicted a win:

```
   llama, 1101 tok    CPU 10852 ms    NPU 13782 ms     27% WORSE
                      CPU 10978 ms    (the control, 1.2% apart)
                      attention 4.67           6.79
```

Not marginal, and not weather: the two CPU arms are 1.2% apart and the NPU one
is 27% outside them. **A quantity that separated eleven of twelve arms was wrong
the first time it was asked a question it had not already been fitted to.**

That is the whole reason to test a fitted rule where it CONTRADICTS the
alternative rather than where it agrees. Eleven points of agreement cost nothing
to collect and bought nothing; one point of disagreement settled it in a single
round.

And 1701 tokens says the same thing again -- product 108864, further above the
threshold than any win in the fitted set, and the NPU arm is 22% behind (21409
against 26164, attention 7.34 against 9.82). The rule is dead twice over.

head_dim survives: 64 does not win at any length tried, 39 to 1701 on Llama and
916 on SmolLM2. What it
still does not have is the length condition, which is real -- Qwen3 at 128 loses
below about 700 tokens and wins at 916 -- and which is therefore per head width
rather than a single number. Two thresholds fitted on two models is not a rule
either, so the knob stays a knob and this table is what a deployment reads
instead.

### Eight lengths on a 64-wide head, and none of them win

The product rule got three more chances and took none of them:

```
   Llama    hd 64   1101 tok   product  70464   CPU 10852 / 10978   NPU 13782   +27%
   Llama    hd 64   1701 tok   product 108864   CPU 21409 / 21416   NPU 26164   +22%
   SmolLM2  hd 64   2092 tok   product 133888   CPU 16328 / 16841   NPU 18156   +11%
   SmolLM2  hd 64   3292 tok   product 210688   CPU 37259           NPU 39657   +6.4%
```

Every one of those products is above the 71936 that the fit called the smallest
win, and the largest is three times it. The CPU controls sit 0.03% to 3% apart,
so none of this is drift.

What the last column does show is a narrowing: 27, 22, 11, 6.4. A 64 wide head
may cross somewhere, but not inside a prompt anybody sends, and "eventually" is
not a rule either.

So the shape that survives is: **head width decides whether the NPU can win at
all, and length decides whether it does.** 256 pays from about 280 tokens, 128
from somewhere between 532 and 916, and 64 has not paid at 3292. That is a table
a deployment can read. It is not yet a gate the runtime can apply, and writing
one from these points would be the third fitted rule in a day.

## The fence is paid by output channel, not by arithmetic

Three things the fence is not: the weight fetch (doubling the chunk halves the
weight passes and it does not move), the two cores being serialised (worth 2.7x,
and the default already takes it), and CPU idle exit (the PM QoS hold forbids
it). What was left was a suspicion from three models -- each one's MACs a row
over its fence gives 0.72 TMAC/s on Llama at 8192 wide, 0.37 on Qwen3 at 3072,
0.20 on SmolLM2 at 1536 -- which looks like a fixed per dispatch cost.

Three models that differ in layer count, K and KV heads is not a measurement of
that, and a rule fitted across four models died earlier the same day. So the
fence was bucketed by width and asked inside ONE pass of ONE model:

```
                  ms/call   GMAC/call   slots/call    ms per unit of n
   llama  n  512    0.090      0.038        0.9          1.76e-4
   llama  n 2048    0.271      0.307        1.8          1.32e-4
   llama  n 8192    1.070      0.614        0.9          1.31e-4
   qwen3  n 1024    0.135      0.070        0.8          1.32e-4
   qwen3  n 2048    0.294      0.080        0.5          1.44e-4
   qwen3  n 3072    0.423      0.120        0.5          1.38e-4
```

**The rate column is contaminated and is not quoted.** Slots a call runs from
0.5 to 1.8, because a device is charged a whole fence even on the calls where
the deal gave it no slot of that tensor, and because a tensor with eight K
slices puts four on a device where one with two puts one. MACs a call therefore
varies for reasons that have nothing to do with the width.

`ms/call` does not have that problem -- it is a clock reading either way -- and
divided by n it is **1.31 to 1.44e-4 across both models and five of the six
widths**. The sixth is the smallest dispatch there is, and it is high, which is
what a fixed cost looks like when the work is too small to hide it.

So: **the fence is proportional to the output channels of a dispatch, about
0.135 us each at m = 80, and not to its arithmetic.** That is why the apparent
TMAC/s moved between models -- MACs scale with k as well and the time does not.

⚠ It is still a fit over buckets whose composition this instrument does not
control. The discriminating form is a timing harness that varies n at fixed k
and m on one dispatch, which `npu_gemm_test` is not (it checks correctness and
does not time) and `charsiu_matmul` is not (one shape, once). That is the next
tool, and it is a small one.

## What one dispatch costs, asked without the model in the way

The fence bucketed inside a forward pass said the time went with the output
width, but its buckets mixed tensors with different K slice counts and charged a
device a whole fence even where the deal gave it no slot. `npu_fence_scan` asks
it with nothing else moving: one dispatch, one device, k and m fixed, n swept,
and the buffers allocated ONCE at the widest point so the sweep cannot measure
an allocation -- a mistake this tree has already made, at 652 ms in one round.

```
   k=1024 m=80      n    fence us   us per n
                   256      334.8     1.308
                   512      343.3     0.671
                  1024      454.4     0.444
                  2048      595.1     0.291
                  4096      911.6     0.223
                  8192     1543.2     0.188
```

The last column falls, so it is not paid by output channel alone. Every sweep
fits `a + b * n` to a few percent:

```
      k      m     a (us)    b (us a channel)
   1024     80      299          0.152
   1024     16      118          0.091
    512     80      192          0.141
   2048     80      305          0.208
```

⚠ **I read the first table off a `tail` while the next sweep was printing and
quoted the m = 16 row as m = 80**, which made the fixed cost 118 us instead of
299. The controls are the only reason that got caught: an m that changes nothing
would have been the strange result, not the ordinary one.

And they say something the single sweep could not: **the fixed cost is not a
per call constant.** It is 118 us at m = 16 and 299 at m = 80, so part of what
looks fixed in n scales with the rows. Only part -- 5x the rows for 2.5x the
cost.

### What it is worth, which is the point of measuring it

The batched path already submits ONE job per (tensor, device) with the K slices
chained as tasks inside it, so the fixed cost is paid per tensor and not per
slice. A device walks 7 tensors * 16 layers * 7 chunks = 784 jobs in Llama's 512
token prompt, and 784 * 299 us is 234 ms against a measured fence of 701: **the
per dispatch fixed cost is about a third of the fence, and 6.3% of the prompt.**

There is one structural way to spend less of it: q, k and v read the SAME input,
so the three could be one job instead of three. That is two jobs saved of every
seven, 67 ms, **1.8% of the prompt.** Real, and small.

So the fence is close to its floor for this dispatch structure, and that is the
useful part of the answer: **after today the largest remaining item in a
prefilled row is attention** -- 2.34 ms against read's 2.04 and the fence's 1.37
-- and the thing that addresses attention is the fp16 path, which is measured,
reopened, and waiting on a decision about fp16 rather than on more measurement.

## A column that cannot carry a conclusion

Two changes went in for the fp16 attention mirror: decode stopped filling a
surface only the batched path reads, and the fp16 unit stopped opening a second
file descriptor on an accel device the process already had open. Both are right
on their own terms. Neither could be priced, and the way that became clear is
worth keeping.

The round ran base, mirror, base, mirror, so every arm had a repeat:

```
   m57                Qwen3 decode        TinyLLAMA decode
     base            18.48 .. 20.58       13.34 .. 20.57
     mirror          24.74 .. 24.75       13.26 .. 13.33
     base again      18.35 .. 20.91       13.23 .. 20.58
     mirror again    24.72 .. 24.72       20.58 .. 20.62
```

Read Qwen3 alone and the mirror is 20% FASTER, twice, tightly. That is not
credible, and it is not what happened. **TinyLLAMA with the mirror on reads
13.33 in one arm and 20.62 in the other** -- the same binary, the same flags,
the same minute apart. The column is bimodal WITHIN a configuration, so Qwen3
landing high twice and low twice is two coin flips agreeing, not a measurement.

Across rounds it is worse. Between the round before the fix and this one, the
mirror arm went 21.70 to 24.7 and the base arm went 24.69 to 20.6 -- **both
moved, in opposite directions**, and the fix cannot touch the base path at all.
Something on the board moves the baseline between rounds fifteen minutes apart.

And I read it wrong once on the way: I compared this round's mirror arm against
the PREVIOUS round's mirror arm and called the 14% difference the fix. That is a
cross round comparison, which this file spent the afternoon establishing is
worthless here, written down at 15:00 and broken at 17:17.

**So decode is not to be priced with board_vendor.sh.** It runs a prompt before
it generates, and whatever the prefill leaves behind -- the NPU's own operating
point is the first suspect -- follows into the window being timed. What that
needs is a decode-only harness with many repeats and the arms alternating inside
one process, and it does not exist yet.

The prefill side of the same work needs none of that, because the stage table
compares inside one run: Qwen3 at 916 tokens goes 12.41 to 11.21 ms a row with
attention 7.69 to 5.52, and it repeats.

## The tree already knew, and the new tool repeats the mistake its neighbour warns about

`charsiu_bench.c` has carried this in its header since round 165:

> Sweeping tasks per job from 1 to 128 at three shapes, the marginal cost of one
> more task is ... M is nearly free. The same weights at M = 32 cost 1.08 times
> what they cost at M = 1 while doing 32 times the arithmetic, so the cost is not
> MAC and not the row count. **It tracks the WEIGHT BYTES, at something close to
> 10 GB/s** ... There is also a **fixed cost of 180 to 195 us per submit that
> chaining removes**.

Which is most of what today's fence work concluded, written down months ago. The
afternoon spent establishing that the fence is not the arithmetic re-derived a
result this repository already held, and the only genuinely new part is that the
"fixed" cost is not fixed in m: 118 us at m = 16 against 299 at m = 80, where
round 165 measured 180 to 195 at M = 1.

**And checking the new tool against the old result finds a fault in the new
one.** If the marginal cost tracks weight bytes, then at k bytes per output
channel the slope must double when k doubles. It does not:

```
      k = 512    0.141 us a channel    2.75x what 10 GB/s would cost
      k = 1024   0.152                 1.48x
      k = 2048   0.208                 1.02x
```

⚠ **That admits two readings and I asserted one.** Either the narrow k points
are being served warm -- `npu_fence_scan` loops ONE shape twenty times, and
`bench_batch`'s header two files away says what that does: *"the first version
looped on one tensor 200 times, which left it in cache and measured arithmetic
rather than memory"* -- OR there is a per output channel FLOOR of about 0.14 us
that dominates until the weight bytes catch up with it, which happens around
k = 1400. The second fits too, and it fits better in one respect: at k = 1024 the
warm slope 0.152 is HIGHER than the 0.102 that 10 GB/s predicts, and caching
cannot make something slower than its own bandwidth bound.

The discriminating form is the same one either way: walk a model's layers so the
weights are cold, as `bench_batch` does. A floor survives that; caching does not.

The intercept is not in question -- a fixed cost per submit is paid warm or
cold.

The header of `npu_fence_scan` now says so. The fix is to walk a model's layers
the way `bench_batch` does, and it is not written.

**Read the neighbouring file's comments before building the instrument.** Both
halves of today's fence answer were already in this tree, and the tool built to
find them reproduced the specific error the file next to it exists to warn about.

## The fix was repeats

Two changes to the fp16 mirror -- decode no longer fills it, and the unit no
longer opens a second file descriptor on an accel device the process already
has -- went in unpriced, because two rounds of `board_vendor.sh` disagreed with
each other and with themselves. The conclusion drawn at the time was that the
harness could not carry the comparison and a decode-only one had to be built.

That was one step too far. **`board_vendor.sh` already reports the best of its
runs**, and the bimodality is not symmetric: the low mode is contamination and
the high mode is the clean reading. Best-of-TWO simply misses the high mode
often enough to produce nonsense. Best-of-six does not.

```
   CHARSIU_BENCH_REPEAT=6, arms alternating, every arm repeated

                  Qwen3 decode      TinyLLAMA decode
     base a/b      24.76 / 24.73     20.68 / 20.63
     mirror a/b    24.79 / 24.77     20.68 / 20.66
```

**Identical, inside 0.2%, on both models, with both arms repeated.** Before the
two changes the same configuration cost 12 to 14% of decode. Together they take
it to nothing, and the fp16 attention mirror is now free to hold while it is not
being used.

What is left is TTFT -- 685/673 to 939/940 on Qwen3, 912/910 to 1108/1110 on
TinyLLAMA -- and that is the mirror being BUILT for a 110 token prompt, which is
far too short to pay it back. That is the envelope this file already measured,
not a defect.

The lesson is smaller than the one I reached for. A harness that already reports
a best does not need replacing when its distribution is bimodal; it needs enough
samples to find the mode that means something. **Count the samples before
building the instrument.**

## Cold weights, and a two term model killed by its own prediction

The warm sweep's slope did not scale with k, and that had two readings: the
weights were being served warm because the sweep loops one buffer, or there was
a per output channel floor. `npu_fence_scan` grew a cold ring -- eight weight
buffers, each with its OWN emitted register stream, because the weight address
lives inside the stream and one stream reused would have submitted the same
buffer however many were allocated.

```
       k      slope warm    slope cold     cold/warm at n = 8192
     512        0.139         0.127               1.00
    1024        0.158         0.154               1.02
    2048        0.213         0.212               0.99
```

**The weights were never being cached.** 32 to 128 MB of ring behaves exactly
like one reused buffer at the wide points that set the slope, so the warm number
was the right number and the worry was mine, not the tool's.

That leaves a real floor, and three cold slopes at m = 80 fit
`b = 0.0987 + 5.51e-5 * k` us an output channel to within a thousandth. The two
terms read beautifully: `m * 4` bytes of accumulator at 3.2 GB/s, and `k` bytes
of weight at 18.1 GB/s. The write and the read.

Three points and two free parameters is not a result. So: the model predicts
that the k term is INDEPENDENT of m, and that is not something the fit can
arrange.

```
   slope, cold ring          m=8      m=16     m=32     m=80
     k = 512                0.0466   0.0461   0.0716   0.1300
     k = 2048               0.1702   0.1719   0.1836   0.2073

   the k term, which must not move:
                            12.4     12.2     13.7     19.9  GB/s
```

**It moves, by 60%.** The two terms are not separable, and the model is wrong as
stated -- the fifth rule today to be tested where it could fail and to fail
there.

What survives is better than what died:

- **m = 8 and m = 16 cost the same per output channel**, within 1% at both k.
  The row count is nearly free below about 16, which is round 165's "M is nearly
  free" arrived at from the other end.
- **A wider m makes the weight fetch itself more efficient**, 12.4 GB/s at m = 8
  to 19.9 at m = 80. Batching does not only amortise a fetch over more rows, it
  makes the fetch go faster, and that is a quantified argument for the whole
  batched prefill rather than an assumed one.

## Reading the vendor's own file, and finding we are already past it in one place

Two things came out of `rkllm_regcmd.py` on their Llama-3.2-1B w4a16 that the
afternoon's profiling could not have found.

**Their input surface is pinned at exactly the ceiling.** Every batched shape in
the file reads `surf = 5120`:

```
   ic       oc       surf     M      count
   4096     1024     5120     40     320
   2048     1024     5120     80     256
   2048      256     5120     80     256
   2048     4096     5120     80     256
```

`(ic / 32) * M` is 5120 in all of them. The bound this tree found by bisection
and recorded as "measured, cause unknown" is a number the vendor's compiler
targets deliberately.

**And they dispatch ic = 2048 unsliced at M = 80, where we cut the same tensor
in two.** Our KMAX is pinned at 1024, and the note that pins it does not say 2048
is bad -- phase 2 caught two models disagreeing with their own token loop at
4096, and 1024 was kept as "the widest the board has always run". So the obvious
move was to take the vendor's slice.

It loses, and the reason is a coupling this file had not written down:

```
   Qwen3, 156 tokens
     KMAX 1024   1 chunk of 156    196 calls   6.10 ms a row
     KMAX 2048   2 chunks 80+76    392 calls   6.56 ms a row
```

**The chunk cap is `163840 / KMAX`**, so raising the slice width lowers the
widest legal chunk in exact proportion. The slices per tensor halve and the
chunks double, the call count does not fall -- here it doubled -- and the ragged
tail comes back. Both knobs sit on one constraint, `(k / 32) * m <= 5120`, and
the vendor's (2048, 80) and ours (1024, 156) are two points on the same line.

Theirs is forced: a runtime that always chunks at 80 has to widen K to fill the
surface. Ours is not, and at the prompt length their own benchmark uses, one
chunk of 156 at KMAX 1024 beats their arrangement. The text is identical to the
token loop at both.

That is worth knowing in both directions. It closes "should we copy their slice"
with a measurement instead of an assumption, and it says the 5120 ceiling is the
real object -- the thing to attack is the surface bound itself, not either of
the two knobs that trade against each other underneath it.

### And the read is proportional to the K slices

TinyLLAMA at the same 158 tokens, with the same text out of both:

```
                        calls   ms a row   pack   fence   read
   KMAX 1024, 1 chunk    154      7.77     1.11    2.02   2.43
   KMAX 2048, 2 chunks   308      7.92     1.33    3.14   1.20
```

The totals are a wash. The COMPOSITION is not: **the read halves.** That is the
mechanism for the largest item at the vendor's prompt length, and it was not
written down -- `read_rows` walks Y once per K SLICE, because each slice
contributes a partial sum that has to be accumulated, so the read cost is
proportional to the slice count and not to the output alone.

So the constraint has two sides and they pull opposite ways:

```
   read   falls with FEWER K slices  -> wants a WIDE slice
   fence  rises with MORE chunks     -> wants a WIDE chunk
   and (k / 32) * m <= 5120 makes slices * chunk a constant
```

Llama says the same thing at 110 tokens, 7.45 against 7.47 with the fence going
2.11 to 3.16. Three models, three washes, and in each one the read and the fence
swap places.

**That is why this has been stuck.** Every knob touched so far slides along the
5120 line, and the two costs cancel. The thing worth attacking is the line: 5120
times 32 is 163840 bytes, which is the shape of a hardware input buffer, and the
vendor's compiler targets it exactly -- so it is probably not ours to move. What
IS ours is the read's dependence on the slice count, which is a software choice
about where partial sums are accumulated, and `read_fused` was one attempt at it
that lost 2.3% on Llama at a profile measured before any of today's work.

## Our register stream against theirs, at the shape they actually run

`--find m=80` picks their canonical batched projection -- ic 2048, oc 1024,
M 80, surf 5120, int4 -- and `emit_job` builds ours for the same shape. The
comparison has to use `emit_job` and not `emit_dump`: `emit_dump` calls
`charsiu_emit_matmul`, whose only caller is `emit_dump`, and this file has
already recorded one fix verified by the thing it changed.

```
                ours   theirs
     CNA          46     45
     CORE          6      5
     DPU          69     68
     U28           5      5
     RDMA         22      0
```

**The DPU block differs in three registers and they are one difference.** Ours
writes `401c = M`, `4028 = 0`, `40b8 = 3M`; theirs writes 96, 16, 304 at M = 80.
Counted over every int4 stream in the file -- 512 at M = 1, 384 at 16, 512 at
32, 320 at 40, 384 at 48, 384 at 64, 768 at 80 -- two relations hold without
exception:

```
     4028 = 401c - M
     40b8 = 4 * 401c - M
```

and our values satisfy both at `401c = M`. So they are not doing something we
are not; they allocate output surfaces at pooled widths -- 32, 64, 96, 128 --
and declare the unused tail, and we allocate exactly M. **The stream is the
same law at a different stride, which closes "is the gap in how we program a
dispatch" with a count rather than an opinion.**

### Except for one block we write and they do not

Their int4 projection programs **no RDMA registers at all**. Ours writes 22, and
`0x5020` in them is the coefficient address. So we build and fetch a coefficient
buffer per dispatch on a path where the vendor fetches none.

That is not yet a defect. The w4a16 read applies its group scales on the CPU --
`fp[0] * cp[0]` in the gather -- so what the coefficient buffer is still needed
for on this path is a question, not an answer, and `charsiu_build_coefs` writes
bias and weight sums that the int8 requant genuinely needs. The probe is to
leave it unprogrammed on w4a16 and read the text.

### The RDMA block, counted and then priced out

`--dump` on one stream said the vendor writes no RDMA. Counted over the whole
file that generalises: **none of the 8808 convolution streams carries a single
RDMA register** -- not the 3328 int4, not the 4940 fp16, not the 40 int8, not
the 500 weightless. They program the coefficient DMA in their DPU-only streams
and never in a matmul.

⚠ The note above the emit says the opposite -- *"the vendor keeps that
configuration in its own stream as well, 21 RDMA registers"* -- and the count
says otherwise for this file. That note may be about the int8 capture rather
than the .rkllm; it is left standing with this beside it rather than edited on a
guess.

We write 22 there, and `0x5020` is the coefficient address. On the w4a16 path
the buffer is provably all zeros -- bias and weight sums are both `calloc`ed and
only the int8 branch fills them, because int4 has no input zero point to correct
-- so the natural next move was to stop programming it.

**It is not worth the risk.** The DMA is not the 270 KB allocation, it is the
span `0x5024 - 0x5020`, which the emitter sets to the table plus the scale
table:

```
   0x5020 = 0x7000, 0x5024 = 0x9800   ->  10240 bytes
   10 KB at 10 GB/s = 1.0 us of a ~300 us dispatch = 0.34%
```

Against that, the note two screens up records what happens when this unit is
half programmed: *"the job timed out: the unit that fetches the per channel
records was half configured, so the DPU waited for data that never arrived."*
A third of a percent is not worth a wedged NPU on an unattended round.

What IS left over is memory, not time: one coefficient buffer is allocated PER
SLOT, and Llama makes 320 of them at 270 KB, so **84 MB of identical zero
buffers** on the w4 path. Sharing them is safe by construction and saves no
time; it is a memory fix and belongs with the fp16 mirror's 2 GB cap rather than
with the prefill.

### ⛔⛔ And the conclusion above is worthless

"The register stream is closed" was drawn from **one** shape, and it is a shape
**this runtime never emits**. KMAX is pinned at 1024, so every dispatch charsiu
makes has k = 1024; the diff was our k = 2048 against their k = 2048.

Counted, their shapes are ic 2048 or 4096 and oc 1024, 256, 4096, 1 or 64.
Llama's tensors are k 2048 and 8192, n 2048, 512 and 8192. **Their oc is exactly
half our n and their ic exactly half our k** -- int4 packs two weights to a
byte, which is the obvious unit difference and has not been confirmed. Neither
side emits a shape the other emits.

So nothing was compared, and the same mistake is on the record three times now:
the wall was chased as a CBUF property for months and was a field layout; the
0x4050 rule was fitted where two expressions are indistinguishable. **A
comparison between one shape of theirs and one shape we do not run cannot close
anything.**

## 2026-09-06: two board results that were sitting on the card unread

Both of these ran days ago, wrote their files, and were never quoted in a
status line, a source comment or a memory. Reading them first was cheaper than
any round, and one of them moves the target.

### `CHARSIU_NPU_NMAX=4096` ran on 2026-08-29 and does not fix m = 8

`board_w4_m8.sh` has three arms and all three are on the card
(`~/charsiu-board/w4-m8-{baseline,onedev,nmax4096}.txt`, all 22:21 on 08-29).
Only the onedev arm was ever written down.

```
  baseline   MISS blk.0.ffn_gate .. blk.13.ffn_up, k=2048 n=8192 row 0 of 8
             row 0: 8192 of the first 8192 wanted values are somewhere in the
             batch, 0 slots came back exactly zero
  onedev     904 of 904, worst 0.00e+00
  nmax4096   STILL MISSES.  blk.0.ffn_up onward, same k=2048 n=8192 row 0
             row 0: 8189 of the first 8192 wanted values are somewhere in the
             batch, 0 slots exactly zero
```

So the arm npudev.c still describes as open -- *"if m = 8 comes back exact the
fault is the WIDTH"* -- has run and come back dirty. The width is not it, which
is the same verdict the onedev arm reached from the other side, now with an
independent second arm behind it.

⚠ One caveat kept honestly: the MISS line prints the TENSOR's n, which is 8192
either way, so the line itself cannot show that the knob engaged. What does show
it is the miss set changing -- `blk.0.ffn_gate` and `blk.5.ffn_up` miss in the
baseline and not under nmax4096 -- and the present-value count moving from
8192 of 8192 to 8189. A knob that changed nothing would have reproduced the
baseline exactly.

### 🏁 The overlap at width 24 produced 21600 exact rows

`board_overlap_slots.sh` -- the probe npudev.c calls *"what turns this into a
mechanism"* -- ran on 2026-09-04 at 21:37, on phi3, width 24, KMAX 2048, 225
tensors a pass:

```
  arm            rows            worst rel   MISS   speedup   fence
  serial 1       5400 of 5400    1.61e-04    0      4.24x     409 ms
  parallel 1     5400 of 5400    1.61e-04    0      5.69x     164 ms
  parallel 2     5400 of 5400    1.61e-04    0      5.65x     165 ms
  parallel 3     5400 of 5400    1.61e-04    0      5.82x     167 ms
  parallel 4     5400 of 5400    1.61e-04    0      -         -
```

**Zero misses in four overlapped passes at the width whose TEXT is wrong 3 to
15 times in 16.** And the overlap demonstrably engaged: the fence is what
overlapping two cores is supposed to cut, and it collapses 409 -> 164 ms while
the pass gets a third faster. That is a behavioural signature, not a flag being
set -- which matters, because a knob that quietly does nothing has cost this
tree a round before.

The consequence is that the width-24 fault is **not in the batched matmul**,
or is no longer there at all. npudev.c currently describes the whole residual
as "two cores stepping on row 0 of a wide output"; that sentence is earned for
m = 8 and m = 10, where the same probe DOES catch 33 misses, and it is not
earned for 22 and 24, where this probe looks straight at the numbers and finds
none.

⛔ What this does NOT say: that the text is now right. The probe runs
`--batch-probe`, which exercises the batched matmul against the m = 1 path
tensor by tensor and never runs a norm, a rope table or a cache offset. A fault
outside the matmul is invisible to it by construction. Round m65 asks the text
question directly, with the KMAX 1024 cell as the positive control, because a
round where nothing reproduces says the fault is gone rather than located.

### m65: the overlap is clean at width 24, and the control never ran

Phi-3.5, 87 token prompt, 16 runs an arm, on the board's current dev build.

```
  KMAX 2048, chunk 24   widths 3x24+1x14   parallel 0/16 wrong   serial 0/16
  KMAX 1024, chunk 24   widths 1x86        parallel 0/16 wrong   serial 0/16
                                           onedev   0/16 wrong
```

The KMAX 2048 cell is a real width-24 test and the overlap is clean sixteen
times, where the 08-30 map has this configuration wrong 3 to 15 times in 16.
Together with the slots probe's 21600 exact rows that is two independent reads
saying the same thing.

**⚠ The KMAX 1024 cell -- the one that exists to reproduce 13 of 16 -- did not
run width 24 at all.** Its widths line says `1x86`: the chunk cap at KMAX 1024
is 163840/1024 = 160, the prompt is 87 tokens, and `CHARSIU_PREFILL_ONECHUNK`
(default ON since this week) replaced the requested chunk of 24 with one chunk
of 86. A cell that was supposed to fail tested a width that has never failed.

That is the fourth time this month a knob has been set and not engaged, and the
only reason it was caught is that the script prints the binary's own `widths`
line beside the rate. **A number is worth what the line next to it says was
run.** m66 repeats the control with `CHARSIU_PREFILL_ONECHUNK=0`.

So the standing claim after m65 is narrow and deliberately so: at KMAX 2048,
chunk 24, phi3, on this kernel and this build, sixteen overlapped runs are
right. Nothing here has yet reproduced the fault, so nothing here has yet
proved it gone -- an experiment where the positive control does not fire
cannot tell "fixed" from "not looking".

### 🏁 m66: the overlap fault does not reproduce, control included

The same script, the same model, `CHARSIU_PREFILL_ONECHUNK=0` so the requested
chunk survives to the chunker:

```
  KMAX 1024, chunk 24   widths 3x24+1x14   parallel 0/16 wrong   serial 0/16
  KMAX 2048, chunk 22   widths 3x22+1x20   parallel 0/16 wrong   serial 0/16
```

The first line is the 2026-08-30 reading's own cell -- phi3, chunk 24, KMAX
1024, both cores overlapped -- which was **13 of 16 WRONG** when it was priced.
It is now 16 of 16 right, and this time the widths line proves the width ran.
Width 22, which failed 1 in 16, is clean 16 of 16 too.

Three independent reads now say the same thing: 21600 exact rows in the element
probe, 16 clean at width 24, 16 clean at width 22.

**What that supports:** the 13-in-16 fault is gone. A rate that large gives
sixteen consecutive clean runs about once in 10^11 tries, so this is not luck.
**What it does not support:** that nothing is left. Sixteen clean runs bound a
rate at roughly one in six, so a fault firing one prompt in fifty would sail
through all of this untouched. That is the whole reason the 28 default was held
back before, and the answer to it is more runs on more models, not a louder
adjective.

**Which change fixed it is not established and this round cannot say.** The
running kernel is `2ffc0913` -- attach-once-v11 plus the igorfix patches, #9,
built 09-05 -- against the August kernel that attached and detached the IOMMU
per job, and the runtime has moved a long way in the same window. Telling the
two apart needs the other kernel booted, which is a flash, which is the user's.

⚠ And the warning both board scripts print here -- *"/boot/Image is NEWER THAN
THIS BOOT"* -- is a false alarm. The Image's mtime is 1788580697 and the boot
was at `now - uptime` = 1788636058, so the Image is 15 hours OLDER than the
boot and the kernel running is the one on disk. The test is
`[ /boot/Image -nt /proc/1 ]` and /proc/1's mtime is not the boot instant.

### m67: 48 identical hashes across four models, and none of them proves the overlap ran

Four models, two prompt lengths, the token loop as the reference and three runs
each of the serial default and `CHARSIU_NPU_PARALLEL_MIN_M=28`:

```
  short  qwen3  1x156        tinyl 1x158       llama 1x110       gemma3 1x80+1x78
  long   qwen3  5x80+1x12    tinyl 5x80+1x14   llama 3x80+1x20   gemma3 5x80+1x14
```

Every one of the 48 batched runs hashes exactly to its own token loop, and the
widths line beside each says which chunks it batched -- including tails of 12,
14 and 20, and the 80s the chunker really emits.

**⚠ And that is not yet evidence the overlap engaged.** `PARALLEL_MIN_M` is
consulted only after `batch_serial()` has already said "serialise", and
`batch_serial()` defaults to `!overlap_safe()` -- so on a rail `overlap_safe()`
approves, the two arms would be the same run twice and all 48 hashes would
match for a reason that has nothing to do with the question. Identical output
across arms is exactly what a knob that never engaged produces, which this
month has already cost four rounds.

m68 reads the two things that cannot stay quiet if the arms really differ: the
line the runtime prints about what it decided, and the fence, which is the
stage overlapping the cores cuts -- 409 ms to 164 ms when the element probe
engaged it.

### ⛔ Correction: the overlap is already the default, and the fault was already solved

m68 asked the runtime what it had decided, and all four models, all three arms,
printed the same line:

```
  charsiu NPU: batched calls, the two cores overlapped: 786 MHz at 800 mV,
                                                        the vendor asks 800
```

**The two cores have been overlapping by default on this board.** The "serial"
arm of m67 and m68 was not serial: `batch_serial()` defaults to
`!overlap_safe()`, the rail reads 800 mV, `overlap_safe()` approves, and
`CHARSIU_NPU_PARALLEL_MIN_M` is never even consulted. m67's two arms were the
same run twice, exactly as the caveat written beside it feared.

And the mechanism was found on 2026-09-04 and is written at the top of
`src/overlap.h`: **the overlap fault was the NPU's voltage margin.** 786 MHz at
the 750 mV U-Boot leaves, against the 800 mV the vendor's OPP asks of its 800
MHz step. Four DTBs, same probe, 4 passes of 5400 rows:

```
  786 MHz, 750 mV (mainline as shipped)   11 to 25 wrong words a pass
  594 MHz, 750 mV                          0, 0, 0, 0   (10% slower)
  786 MHz, 800 mV                          0, 0, 0, 0   (full speed)
  786 MHz, 850 mV                          0, 0, 0, 0
```

So m66's "the fault does not reproduce" reproduces nothing because the rail it
needed is gone, and its closing line -- *"which change fixed it is not
established"* -- is answered in the tree it was written next to. The three
rounds today are still worth their UART time, but for a smaller claim than they
were run for: they are an end-to-end check of the 09-04 guard at the width the
old map called worst, on four models the guard was never exercised on, and it
holds -- 16 of 16 at width 24, 16 of 16 at width 22, 48 hashes equal to their
token loops.

**The one thing that follows and is new: the TTFT numbers already quoted are
overlapped numbers.** 673 / 910 / 3004 / 2408 ms against the vendor's 469 / 544
/ 1829 / 1219 were measured with both cores in flight, so the fifth off the
prompt that `PARALLEL_MIN_M=28` was priced at is **already taken** and is not
sitting in the gap waiting to be collected.

⚠ And it leaves one question that is genuinely open, because its evidence
predates the rail. `board_w4_m8.sh`'s map -- m = 8 and m = 10 missing ROW 0 of
the n = 8192 tensors, 33 of 904, one core clean and two cores dirty -- was
measured on 2026-08-29, six days before the voltage was found, so every arm of
it ran at 750 mV. That is the same signature the voltage produced. m69 re-runs
it at the rail the board now holds.

### 🏁 m69/m70: m = 8 was the rail too, and the refusal now names the envelope

`board_w4_m8.sh`, re-run unchanged, with `vdd_npu_s0` reading 800000 uV:

```
  arm         m=2          m=4          m=8          worst      MISS lines
  baseline    226 of 226   452 of 452   904 of 904   5.10e-05   0
  onedev      226 of 226   452 of 452   904 of 904   0.00e+00   0
  nmax4096    226 of 226   452 of 452   904 of 904   5.10e-05   0
```

against the same script's **871 of 904 and 33 MISS lines** on 2026-08-29, when
the rail was the 750 mV U-Boot leaves. The baseline is the arm the script says
must fail, and it is exact.

So the second fault was never a second fault. "Two cores stepping on ROW 0 of a
wide output at m = 8 and m = 10" and "the width-24 text fault" are one thing:
786 MHz outside the vendor's OPP envelope, which `src/overlap.h` found on 09-04
with four DTBs. Both maps were drawn at 750 mV; both dissolve at 800.

onedev being 0.00e+00 while the two-core arms sit at 5.10e-05 is not a
residue of the fault -- it is one float summation order against two, which the
probe's own bar (1e-3) is set for. And onedev is *slower*: 516 ms against 368
at m = 8, which is the two cores doing their half each.

**What shipped from it:** the m = 8 / m = 10 refusal in `npudev.c` is now gated
on `charsiu_npu_overlap_ok()` -- the same reading of the same envelope that
already decides whether the two cores may overlap -- instead of on the width.
Inside the envelope the widths run; outside it they still refuse.

⚠ Deliberately conservative, and here is the gap: nobody has measured m = 8 at
750 mV with the cores *serialised*. onedev was clean there, but onedev halves
the hardware and the draw, so it cannot separate "needs two cores" from "needs
the current". Off-envelope this keeps refusing, which costs a fallback nobody
will notice -- no prompt length measured yet makes the chunker emit 8 or 10.

### m72: 8, 10, 22 and 24 all exact at 800 mV, through the shipped gate

Llama-3.2-1B, 113 staged tensors, two passes a width, `CHARSIU_NPU_W4_M8`
deliberately **not** set -- so 8 and 10 had to get past npudev's new gate to
reach the hardware at all, and getting a row count back at all is the gate
working:

```
   width   pass 0            pass 1            worst
    8       904 of 904        904 of 904       5.10e-05
   10      1130 of 1130      1130 of 1130      5.10e-05   <- never measured
   22      2486 of 2486      2486 of 2486      5.10e-05      at 800 mV before
   24      2712 of 2712      2712 of 2712      5.10e-05
```

m = 10 is the one that had to be asked separately: `board_w4_m8.sh` caps at 8,
and the chunker splits **both** 8 and 10, so lifting its split on one width's
evidence would have been the layout-proof mistake again.

So `prefill_width()`'s `8 -> 4+4` and `10 -> 6+4` now happen only when
`charsiu_npu_overlap_ok()` says the rail is below the envelope. **The split
stays there for a good reason rather than a cautious one:** off-envelope
npudev refuses 8 and 10, a refused chunk runs a row at a time, and two batched
calls of 4 are much faster than that. The same reading of the same envelope
now picks between two correct paths instead of guarding a broken one.

⚠ m71 is why this round existed: asked for `CHARSIU_PREFILL_CHUNK=8` the board
answered `widths 20x4`. The chunker had already split it, so the gate under
test was never reached and the round proved nothing about it. The widths line
said so on its own output.

### m73: the gate on the card, and the price with six runs

`charsiu update dev`, then a chunk of 8 and a chunk of 10 asked for explicitly
on four models, each against its own token loop:

```
  model    token loop     chunk 8            chunk 10          default
  llama    c728d6bc2799   10x8      same     8x10     same     1x80      same
  qwen3    4f050ade006f   13x8+1x6  same     11x10    same     1x110     same
  tinyl    c7a8689f0a9d   14x8      same     11x10+1x2 same    1x112     same
  gemma3   efd3dfb40845   14x8      same     11x10+1x2 same    1x80+1x32 same
```

Where m71 got `20x4` for the same request, the chunker now emits the width it
was asked for, and every hash equals its token loop. That is the change working
end to end: the chunker stopped splitting and npudev let the width through.

And the standing price, `board_vendor.sh` at `CHARSIU_BENCH_REPEAT=6`,
governor performance:

```
                 TTFT ours   theirs   gap      decode ours   theirs
  Qwen3 0.6B      687        469      1.47x     24.75         24.85
  TinyLLAMA 1.1B  900        544      1.65x     20.64         19.71
  Phi3 3.8B      3057       1829      1.67x      6.87          6.58
  Gemma4 E2B     2352       1219      1.93x      8.67          9.23
```

Unchanged by the day's work, and that is the expected result rather than a
disappointment: the default chunker emits one wide chunk at these prompt
lengths and never asks for 8 or 10, so nothing shipped today is on this path.
What the day bought is that three "open" faults are closed and two gates now
name the real condition. The gap itself is still the NPU entry.

## 2026-09-06 evening: where the prompt actually goes at the scoreboard's length

`CHARSIU_STAGES=1 CHARSIU_FENCE_SPLIT=1`, prompts of 80 to 112 tokens -- the
lengths `board_vendor.sh` itself uses -- so this is the profile of the number
in the gap and not of the 916-token runs everything was tuned on.

```
  model   entry    prep   pack    sub    fence   read    other
  qwen3   444 ms   2.9%   25.5%   1.7%   27.5%   28.4%   14.1%
  tinyl   685 ms   2.0%   18.6%   1.2%   37.6%   35.1%    5.6%
  llama   485 ms   2.6%   17.1%   1.2%   39.8%   34.3%    4.9%
  gemma3  664 ms   1.9%   31.0%   1.6%   28.4%   20.6%   16.6%
```

**The hardware is busy for 28% to 40% of the matmul entry.** `read` and `pack`
are 45% to 55% between them and both are CPU work with the NPU idle. And of
the fence itself, the invalidate is 0.22 to 0.42 ms a row against a wait of
0.89 to 2.14 -- so "fence" really is mostly the hardware, not cache
maintenance.

### 🔑 The widest output is the slowest, inside one run of one model

The per-width fence buckets. Nothing changes between rows but n: same model,
same pass, same clock, same prompt.

```
  llama    n=512   0.40 TMAC/s    tinyl   n=256   0.39    gemma3  n=256   0.16
           n=2048  0.58                   n=2048  0.58            n=1024  0.34
           n=8192  0.33                   n=5632  0.34            n=1152  1.09
                                                                  n=6912  0.32
```

A narrow n being slow is the per-dispatch floor and is expected. **The widest
being slow is not**, and it is not arithmetic intensity: a dispatch does 2m
MACs per weight byte at every n, so equal intensity should give equal TMAC/s.
Turning the buckets into bandwidth on llama: the n=2048 bucket moves 201 MB of
weights in 56.0 ms of summed fence and the n=8192 bucket 269 MB in 130.6 ms --
3.6 against 2.1 GB/s. **The wide bucket fetches weights at about half the rate
of the middle one.**

That is the biggest single line in the fence on three of four models: 130.6 of
llama's 193 ms, 166.6 of tinyl's 257, 143.2 of gemma3's 188.

And the vendor's own Llama-3.2-1B `.rkllm` never dispatches more than **4096**
output channels -- ours is the only shape that asks for twice that. m75 caps
it (`CHARSIU_NPU_NMAX`) and reads the bucket back.

⚠ `CHARSIU_NPU_NMAX` has been run exactly once, as a correctness control for
m = 8, where it changed nothing. **Its speed has never been measured.** It can
also lose: splitting n doubles the dispatches for those tensors and re-reads
the same activation twice.

### m75/m76/m77: capping the output width, and two hypotheses it killed

`CHARSIU_NPU_NMAX` had been run exactly once, as a correctness control for
m = 8 where it changed nothing. Its speed had never been measured. Swept at
each model's OWN default K slice, prompts of 81 and 113 tokens:

```
  nmax     gemma-3-1b            llama-3.2-1b          tinyllama-1.1b
           prompt  wide bucket   prompt  wide bucket   prompt  wide bucket
  default   912 ms  6912 0.29     616 ms  8192 0.33     884 ms  5632 0.33
  6144      884     6144 0.40     634     6144 0.32     872     5632 0.33
  4096      838     4096 0.53     630     4096 0.34     903     4096 0.32
  3072      872     3072 0.44     637     3072 0.34     882     3072 0.32
  2048      857     2048 0.54     627     (merged) 0.43 881     (merged)
  1024      949     1024 0.57     683     1024 0.42     957     (merged)
```

**gemma-3-1b takes 8% off its prompt at nmax 4096 and the other two take
nothing at any width.** gemma3's text was verified identical to its token loop
at 8192, 4096 and 2048 in m75, and not one weight changes -- the model is
ungrouped at every candidate K, so this is purely a dispatch shape.

Two readings die here, and both were mine:

- **not the weight bytes a dispatch.** llama at n = 4096 with a 1024 slice is
  2.10 MB and runs 0.34; gemma3 at n = 4096 with a 1152 slice is 2.36 MB and
  runs 0.53. More bytes, nearly double the rate.
- **not n.** Same n, same prompt, same clock, two models, 0.34 against 0.53.

⚠⚠ **And m76's llama result is not shippable, for a reason that is not
performance.** `KMAX 2048 + NMAX 4096` took llama from 643 to 576 ms -- but
`llama.c`'s auto-widener already refuses KMAX 2048 on llama, because k = 2048
is divisible by both candidate widths so widening coarsens the quantiser from
group 1024 to group 2048. gemma-3-1b (1152, 6912) is ungrouped at both and IS
already at 2048; llama is not, and setting it by hand buys speed with weights.
llama.c's own note says the counting prompt cannot see that: an earlier sweep
called eight models identical on "1 2 3 ..." while two were degrading.

So: a per-model 8%, a knob that is measured rather than guessed for the first
time, and no rule -- **nothing ships from this until something explains why
two dispatches of the same width and nearly the same weight bytes differ by
1.6x.** The next suspect is the INPUT surface: at KMAX 1024 and m = 80 we fill
`(1024/32) * 80 = 2560` of the 5120 the block allows, and every int4 dispatch
in the vendor's file sits at exactly 5120. ⚠ Already dented, not dead: llama at
KMAX 2048 (surf 5120) ran its wide bucket at 0.24, slower, not faster.

⚠ A note on m77's hashes: `CHARSIU_STAGES=1` was in the shared environment, so
the md5 covers a stage table full of timings and differs run to run. Those
hashes say nothing. The text check that counts is m75's, which hashed a run
with no stage output.

## 2026-09-07: what a dispatch actually costs, in the dtype the prefill runs

`npu_fence_scan`, one dispatch, one device, k and m fixed and only n moving,
buffers allocated once at the widest point, **cold ring of 64** so no repeat
re-reads the buffer before it. Slopes fitted at the wide end (n 4096 → 8192),
microseconds per output channel:

```
             k=512    k=1024   k=2048        m=20     m=40     m=80  (k=1024)
  int8      0.1347   0.1556   0.2085        --       --      0.1556
  w4a16     0.1056   0.2063   0.4103       0.0825   0.1247   0.2063
```

⚠ The cold ring matters and it was checked: against the warm loop the slope
moves under 4%, so the k term is a real fetch and not a cache. And **argv[5]
is new** -- every sweep before today dispatched int8, and the model fitted to
it was carried to a w4a16 prefill by halving the weight bytes on paper.

### The two dtypes are bound by different things

**int8** has a large k-INDEPENDENT floor: 0.111 µs a channel at m = 80, which
is 1.39 ns an output element, about one element per clock at 786 MHz. It is
**output-stage bound**.

**w4a16 is not.** Its slope doubles cleanly with k (×2.0 per doubling, twice)
and its k-free part is 0.004 µs -- zero within the fit. Separating m and k:

```
  w4a16:  µs per output channel  =  k × (4.01e-5  +  2.02e-6 × m)
```

Both halves are physical:

- `4.01e-5 µs per channel·k` is **k/2 bytes at 12.5 GB/s** -- the weight fetch,
  and it lands on the ~10 GB/s this tree has measured three other ways.
- `2.02e-6 µs per channel·k·m` is **2.02 ps per MAC = 0.495 TMAC/s**, and
  `n·k·m` is exactly the MAC count. **w4a16 is MAC-throughput bound at half a
  TMAC/s**, which is a sixth of the part's 3 TMAC/s int8 rating.

And it predicts the live buckets it was not fitted to: qwen3's whole prompt at
110 tokens does ~66 GMAC in 122 ms of fence, which is 0.54 TMAC/s.

### 🔑 Which means the prefill gap is not throughput

The vendor's Qwen3-0.6B TTFT is 469 ms for 110 tokens. The same 66 GMAC at
their wall clock is **0.14 TMAC/s** -- our hardware time is already three to
four times better than theirs. Ours is 122 ms of fence inside a 444 ms entry
inside a 705 ms prompt.

```
  qwen3, 110 tokens     pack 113   fence 122   read 126   prep 13   sub 8   other 63
```

**72% of the matmul entry is CPU work with the NPU idle**, and that is the
whole distance to the vendor. Not the width, not the K slice, not the
quantiser group, not the two cores -- all four of which the last two days
spent rounds on.

⇒ The lever is a **pipeline**: pack chunk N+1 and read chunk N−1 while the
hardware runs N. The ceiling is `max(CPU, NPU)` instead of `CPU + NPU`, which
on qwen3 is 315 ms against 444 -- prompt 705 → ~576, gap 1.47x → 1.23x.

⚠ What blocks it, and it is structural rather than hard: `struct npu_outbuf`
is one buffer per GEOMETRY, not per tensor -- k and v share one, gate and up
share one -- so submitting ahead would have the second tensor's dispatch
overwrite the first's accumulators before the CPU has read them. `ob->busy`
guards exactly that today. A ping-pong pair on the wide geometry costs about
5 MB a device.

### And two thirds of the pipeline idea does not exist to be built

Two things checked before writing any of it, both by reading:

**The pack is not reuse misses.** `charsiu_npu_matmul_same` already skips the
pack for the second and third tensor of a group, so a layer packs four times
(qkv, o, gate+up, down) and not seven. The known miss -- *"Phi-3.5 reused its
packed input 0 times out of 2304 asks"* -- is diagnosed, fixed and priced:
`CHARSIU_NPU_EVEN_KS` takes gemma4's 528 misses to zero and its prompt from
30110 to 29884 ms, inside the spread, and Phi-3.5 does not move at all because
at its own KMAX the slices were already even. That door is shut.

**The read already overlaps the second core.** The loop is submit both, then
`for d: fence(d); read(d)` -- so device 0's read runs while device 1 is still
computing, and the fence counter is what is LEFT after that.

**And the transformer has no next tensor to pack ahead.** o needs attention,
gate needs o, down needs silu; the chain is serial by construction. The only
independence inside a layer is `{q, k, v}` (one input) and `{gate, up}` (one
input), and those already share their pack.

So the pipeline is not "hide 322 ms of CPU". It is exactly this: **submit a
whole group before reading any of it**, so q's read overlaps k's run and
gate's read overlaps up's run. That is 5 of a layer's 7 tensors.

⚠ And what blocks even that is the output-buffer pool: k and v share the 512
geometry, gate and up share the 8192 one, so the second submit of a group
would overwrite the first's accumulators. `ob->busy` guards it today. A
ping-pong pair on the two shared geometries is about 5 MB a device.

### 🏁 One input buffer object per K slice: 18 of 18 paired runs

`rocket_ioctl_fini_bo` is `dma_sync_sgtable_for_device` over the whole object
and the uapi has no range -- both `drm_rocket_prep_bo` and `drm_rocket_fini_bo`
carry a bare handle and a `reserved` field. The batched path kept every K slice
in one buffer sized for the widest tensor in the model, so a call packed ONE
slice and flushed all of them. Llama's buffer is sized for `ffn_down`'s eight,
and its other six tensors each flushed eight slices' worth.

The pool report names it, and nobody had printed that line before today:

```
  model    pack     of which:  emit   FINI ioctls   the packing
  qwen3    112.1 ms            2.6     28.3          81.2
  llama     77.6               2.3     17.3          58.0
  gemma3   193.9               4.4     71.4         118.1
```

Split per slice -- `done_ki` already tracks what was written, and the slot walk
gives what will be read, which is what the handle list needs since a reusing
call packs nothing and still reads:

```
  FINI ioctls   qwen3 28.3 -> 13.4    llama 17.3 -> 8.1    gemma3 71.4 -> 27.4
```

⚠ And the first read of that could not be trusted: the entry TOTAL moved the
wrong way by a few percent, but the two arms were two sessions and the board
was 3% slower in the second. So `CHARSIU_NPU_BIN_ONEBO=1` restores the old
single buffer, and both arms come out of one binary, interleaved, six repeats:

```
  prompt ms      split (6 runs)             onebo (6 runs)          delta
  gemma3   884 882 879 874 870 891     909 907 902 901 909 906     -26 ms  -2.9%
  llama    601 603 600 605 602 604     604 609 611 613 609 609     -6.7    -1.1%
  qwen3    727 721 715 722 720 721     730 725 734 729 738 731     -10.2   -1.4%
```

**18 of 18 paired runs favour the split and the two arms' ranges do not overlap
on any model.** Every run in both arms hashes to its own token loop.

Small, and worth having for a reason beyond the milliseconds: flushing bytes
the call did not write is not a tuning choice. The remaining FINI is now
roughly what the writes justify.

### ⛔ w4a8 costs exactly what w4a16 costs — and int8 WEIGHTS are 1.9x faster

The vendor's Llama-3.2-1B `.rkllm` runs two activation precisions per weight
and picks by batch width: of 3328 int4 dispatches, 1408 clear CNA 0x100c bit 29
(8 bit activations) and they are exactly `M ∈ {1, 32, 64}`, while
`M ∈ {16, 24, 40, 48, 80}` set it. charsiu sets it on every int4 dispatch,
because `charsiu_effective_adtype` returns FP16 for int4 unconditionally, and
round 178's "int4 consumes 16 bit activations whatever the stream asks" was
measured with bit 29 set in every stream it asked with.

`npu_fence_scan` grew a w4a8 mode, and the mode demonstrably engaged -- the two
streams differ on the host, and not only in the flag:

```
  0x100c   20600120 -> 00600120     bit 29 cleared
  0x103c   0a000000 -> 05000000     the SURFACE halves, as "surf follows the
  0x1028   0a0003ff -> 050003ff     activation precision" predicts
```

Cold ring, µs per output channel at the wide end, m = 80:

```
           k = 1024        k = 2048
  int8       0.1548          0.2144
  w4a16      0.2099          0.4089
  w4a8       0.2121          0.4120     <- within 1% of w4a16 at both k
```

**Clearing bit 29 changes nothing about the cost.** The activation precision is
not the 6x between our 0.495 TMAC/s and the part's int8 rating. What it does buy
is a half-size input surface, which doubles the `(k/32)·m ≤ 5120` ceiling -- not
cashable today, since the chunk cap it widens is not what binds and the K slice
it would widen is held by the quantiser group.

🔑 **The other half of that table is not a negative.** At k = 2048 an int8-weight
dispatch is `0.2144` against int4's `0.4089` -- **1.9x faster while reading
TWICE the weight bytes**, which is the MAC-rate model saying the fence is not
weight-bandwidth bound at this shape. At k = 1024 it is 1.35x.

So for PREFILL, which is MAC-bound, int8 weights are the faster arm and int4 is
the DRAM-bound choice that belongs to decode. The tree already has both paths
and a heuristic between them; what it has not had until now is the per-dispatch
model saying how much the trade is worth.

### ⛔ And int8 weights lose the whole prompt by 25 to 30%

The per-dispatch 1.9x does not survive contact with a run. Interleaved, six
repeats, one session, each arm against its OWN token loop (the two quantise
differently, so their text should differ and a cross-arm hash would read a
correct run as a regression):

```
            prompt ms          decode 16 tok      staging     peak
  llama w4  601..608  (604)    756..760   (758)   7.2 s      1544 MB
  llama w8  740..769  (752)   1381..1390 (1385)   9.6 s      2130 MB
  qwen3 w4  710..725  (718)    607..610   (608)   3.7 s      1129 MB
  qwen3 w8  903..958  (936)    923..942   (930)   4.7 s      1420 MB
```

**+25% and +30% on the prompt, 1.5x to 1.8x on decode, +40% staging, +30%
memory.** The int4 default is right and this measures why rather than assuming
it: an int8-weight run also quantises the ACTIVATION per row on the CPU inside
the pack loop -- a max pass and a quantise pass over every slice -- where w4a16
packs fp16 straight through with NEON. The dispatch is faster and the entry
around it is not.

⚠ The w8 batched hash differs from the w8 token loop, and that is expected
rather than a bug: npudev's own note says a multi-slice int8 tensor is
quantised FINER in the batch than in the row loop, on purpose, and cannot match
it to 0.1%. Worth writing down because a round that compared the two arms'
hashes instead of each arm against its own reference would have called this a
correctness failure.

### Where that leaves the prefill gap

Everything cheap is now measured and most of it is closed:

```
  w4a8 instead of w4a16          no change at all (within 1%)
  int8 weights                   25-30% WORSE end to end
  capping the output width       8% on gemma-3-1b, nothing on two others
  KMAX 2048 by hand              a quantiser change on any model whose K
                                   divides both widths; llama.c already
                                   refuses it there
  input reuse misses             diagnosed, fixed, worth nothing
  one input BO per K slice       SHIPPED, 1.1-2.9%
  the pack's FINI ioctls         halved by the above
  hiding pack behind the NPU     no next tensor to pack: the chain is serial
```

What is left, with its price:

1. **Submit a group before reading it** -- `{q,k,v}` and `{gate,up}` are the
   only independence inside a layer. Worth about 8%: the hideable reads are
   bounded by the group's own NPU time, which on qwen3 is 19 ms in the first
   group and 18 in the second against a 444 ms entry. Blocked by k and v
   sharing an output geometry and gate and up sharing another; a ping-pong pair
   costs about 5 MB a device.
2. **The read, which is 21-37% of the entry and has never been split** the way
   the pack now is. Its volume is `m·n·S·4` read plus `m·n·4` written -- on
   llama 304 MB in 164.6 ms, **1.85 GB/s**, against the 7.13 GB/s one thread of
   this board managed on a plain read. Three to four times off memory speed,
   and the index gather is the suspect that the pack's FINI was.
3. **Fewer K slices** would cut the read AND the fence's intercepts
   proportionally, and it is blocked by the quantiser group, not by the
   hardware. That is a quantiser question -- go where the vendor is, one scale
   a row, and pay for it with a calibrated quantiser instead of RTN -- and
   npudev.c has the offline price already.

### The read is at a memory ceiling, and it is not the little cores

`read` is 36-38% of llama's matmul entry, 166 ms, and had never been split. It
turns out not to need splitting so much as bounding.

**It is fully pooled already**: "read back 320 slots on the pool and 0 one
thread". And the pool buys exactly **2.0x**, no more:

```
  CHARSIU_NPU_POOL_READ=0   one thread    read 328.7 ms   prompt 765 ms
  default / =1              the pool      read 164.6 ms   prompt 603 ms
```

Two explanations, both killed:

**Not the memory walk.** Simulating `charsiu_acc_index`'s traversal on the host,
every 64-byte line of the accumulator is fetched **exactly once** -- 1.00x the
ideal at n = 512, 2048 and 8192, even with only 256 KB of cache. There is no
line amplification to remove.

**Not the little cores.** This board is 4x A53 (MIDR d03, CPUs 0-3) and 4x A72
(d08, CPUs 4-7) -- read off the board rather than assumed -- and
`charsiu_parallel_for` splits a range into equal chunks, so four A53s dragging
three A72s to a barrier was the obvious story. It is wrong:

```
  0-7, 8 threads   read 166.7 / 167.9 / 165.7 ms      prompt 603 / 606 / 603
  4-7, 4 threads   read 169.9 / 177.7                 prompt 613
  0-3, 4 threads   read 504.6                         prompt 1059   <- the control
```

Four big cores are no faster than eight mixed, and the A53-only control is 3x
worse, which is what makes the first line mean something. **The read saturates
at about 2.6 GB/s of traffic and more threads do not move it.**

### 🔑 So both remaining levers are the same quantity: S

The read's volume is `m·n·S·4` bytes in and `m·n·4` out, where **S is the number
of K slices**. It cannot be threaded faster and it has no layout to fix, so the
only way down is fewer slices. And the fence's other term -- 270 µs a dispatch
-- is also proportional to S.

`S = ceil(k / KMAX)`, and KMAX is pinned to the quantisation group because one
dispatch cannot span two groups. For llama at KMAX 1024 that is S = 2 for six
tensors of seven and S = 8 for `ffn_down`.

**S = 1 would take the read from 166 to about 83 ms, the entry from 453 to ~370
and the prompt from 603 to ~520 -- 14%.** It is not a dispatch change; it is a
quantiser change, and npudev.c already carries its offline price: one scale a
row against group 1024 costs attn_q 0.1427 -> 0.1518 and ffn_down 0.1402 ->
0.1707 relative Frobenius, while **the per-k AWQ factor takes the K = 2048
tensors BELOW the grouped number** (attn_q 0.1409) and does nothing for
ffn_down, which is the one with the most slices to save.

That is a model-quality decision and belongs to the user, not to a round.

### ⛔ The read fusion is worth nothing, and an existing knob said so for free

The plan was to deal both K slices of a tensor to ONE device so
`read_fused_rows` walks Y once instead of twice -- on the traffic count that is
a quarter of the read, since Y goes from a write plus a read-modify-write to a
single write on every two-slice tensor. It needs a pair-submit refactor of the
hottest path to arrange.

`CHARSIU_NPU_ONEDEV=1` already arranges it, for every tensor, for free:

```
                       ranges   read ms/row   read total   fence
  two cores, split      224      2.07 / 2.09   166.0/167.2  195 ms
  onedev,   fused       112      2.06 / 2.06   164.6/164.6  327 ms
```

**The (device, output range) pairs halve, exactly as the refactor would arrange
them, and the read does not move by more than the noise.** So either the fusion
does not fire or -- far more likely -- the read is not bound by the Y traffic
at all.

The arithmetic agrees once it is done properly: the accumulator side is
`S·m·n·4` and is **unchanged by any dealing**, because S is a property of k and
KMAX and not of which device holds a slice. Y is `m·n·4` and is written
sequentially. On `ffn_gate` that is 5.24 MB of scattered accumulator against
2.62 MB of streaming Y, and the scattered side is what costs.

So the read has exactly one lever and it is S. That is now established from two
directions -- more threads do not move it (2.0x is the ceiling), and neither
does removing a third of its traffic.

What survives of the pair submit is its **overlap only**: reading `gate` while
`up` still runs, and `q` and `k` while `v` runs. Priced off the per-layer MAC
and read shares that is about 3.3 ms a layer, 52 ms of a 453 ms entry, **8.6%
of the prompt** -- worth having, and worth knowing it is 8.6% and not the 15%
it looked like an hour ago.

### 🏁 The tail per channel scale: 14-17% of the entry, and it had no name

An UNGROUPED tensor takes its per channel scale once at the end, as a full
`m × n` pass over Y. That pass sat after the device loop, **outside every
counter, single threaded and scalar**, and landed in the report's `other` row.
Which is why `other` read 2.2 ms on Llama and 62.5 on Qwen3 and 110.0 on
gemma-3-1b: llama is grouped everywhere and never runs it.

"Ungrouped" is wider than it sounds. `tensor_grouped` wants `kgroup < k`, so a
tensor whose K **is** one group -- Qwen3's 1024-wide projections at group 1024
-- is on this path too, as is every tensor of a model npuquant put on one scale
a row because its K divides no candidate width.

Counted, vectorised four at a time, and put on the pool by the same size rule
the read uses. `CHARSIU_NPU_TAIL_PLAIN=1` is the old loop, so both arms come
out of one binary, interleaved, in one session:

```
  prompt ms       fast (4 runs)            plain (4 runs)          delta
  gemma3     860 847 852 859   (854.5)   875 883 879 885 (880.5)   -26 ms  -3.0%  4/4
  qwen3      714 723 708 703   (712)     719 716 727 726 (722)     -10     -1.4%  3/4
  llama      603 605 602 602   (603)     606 600 604 605 (603.8)    -0.8    noise  <- the control
```

**llama is the null control and it correctly shows nothing**, because it never
runs the loop. Every run in both arms hashes exactly to its own token loop --
the arms multiply the same elements by the same scales in the same order, so
equal is the only acceptable answer, not close.

And the row now exists, which is the durable part:

```
  gemma3   scale 39.0 ms  6.8%     other 2.7 ms  0.5%    (other was 110.0, 16.6%)
  qwen3    scale 21.5 ms  5.4%     other 2.4 ms  0.6%    (other was  62.5, 14.1%)
```

The scalar loop was 110 and 62.5 ms; vectorised and pooled it is 39.0 and 21.5
-- **2.8x** -- and the unnamed row collapses to under 1% on every model. The
prompt moves less than the line does, which is honest and worth saying: some of
the old cost was overlapping a stall elsewhere.

### The scoreboard after the day, best of six

Same kernel (`2ffc0913`), governor performance, `CHARSIU_BENCH_REPEAT=6`:

```
                 TTFT ours   theirs    gap      decode ours   theirs   ours/theirs
  Qwen3 0.6B       653        469     1.39x      24.85        24.85      100.0%
  TinyLLAMA 1.1B   890        544     1.64x      20.70        19.71      105.0%
  Phi3 3.8B       2994       1829     1.64x       6.88         6.58      104.6%
  Gemma4 E2B      2272       1219     1.86x       8.72         9.23       94.5%
```

Against yesterday evening's six-run reading, also on this kernel:

```
  Qwen3      687 -> 653   1.47 -> 1.39
  TinyLLAMA  900 -> 890   1.65 -> 1.64
  Phi3      3057 -> 2994  1.67 -> 1.64
  Gemma4    2352 -> 2272  1.93 -> 1.86
```

⚠ **Those two readings are different sessions and the board drifts about 3%**,
which is the size of most of that column. The numbers to trust are the paired
in-session ones the two changes were measured with -- per-slice input buffers
1.1 to 2.9%, the tail scale 1.4 to 3.0% on the models that run it -- and the
scoreboard is consistent with them rather than evidence on its own.

Qwen3's decode is now exactly the vendor's to three figures, and three of four
models are at or above it.

### NMAX on the two models the scoreboard loses on, and a prediction that failed

Phi3 and Gemma4 -- 1.64x and 1.86x, the two the scoreboard actually loses on --
had never been in the NMAX sweep. Both batch their prefill, checked first: the
binary says "prompt batched, 112 tokens in chunks of 112", so the gap is not a
fallback. And on gemma4 the **read is the bigger line than the fence**, 539.6 ms
against 419.1 of a 1400 ms entry; phi3's is 874.5 of 2170.

```
  gemma4   default 1986 1968 1991    4096 1974 1978 1972    2048 1971 1983 1958
           -- inside 1%, nothing
  phi3     default 2730 2692 2748    4096 2666 2653 2661    2048 2797 2799 2788
           -- 4096 is -2.3% on 3 of 3, and 2048 is +2.6%, so there is an optimum
```

Phi-3.5's attention is one fused `attn_qkv` of n = 9216 against nmax 8192, so
its n slices are **8192 + 1024** -- an eight to one split handed to two cores
that are then waited on together, and `q k v` is its largest stage at 8.09 ms a
row. That gives a mechanism and a number: the best nmax should be **4608**,
which halves 9216 exactly.

```
  8192  (8:1)     2706 2675 2670   mean 2684
  4608  (1:1)     2670 2635 2662   mean 2656   -1.0%   <- the prediction
  4096  (4:1)     2649 2632 2646   mean 2642   -1.6%
  3072  (1:1:1)   2650 2686 2687   mean 2674   -0.4%
```

**Wrong.** A perfectly balanced two-slice split loses to an unbalanced
three-slice one, and the other balanced point (3072) is the worst of the three.
So the n-slice balance is not the cost, and NMAX remains what it has been all
week: worth 8% on gemma-3-1b, 1.5 to 2.3% on phi3, nothing on llama, tinyllama
or gemma4, **with no mechanism after five models and four rounds**.

That is the shape of fitting, and it is where this stops. No default ships from
it.

⚠ One weakness worth recording about every hash in these two rounds: the prompt
is "1 2 3 ... 40" and the models continue the count, so **gemma3 and gemma4 hash
to the same twelve characters**. The check still does its job -- batched against
that model's own token loop -- but it cannot see a quantisation change, which is
exactly what llama.c's own note says about counting prompts.

### ⛔ The pair submit is a 12 to 28% LOSS, built and reverted

The last non-quality lever: submit `gate` and `up` before reading either, so
gate's read runs while up is still on the hardware. It was priced at about 8%
off the per-layer shares, and the read is known to translate one for one into
TTFT (`POOL_READ=0` moved llama's read 164.6 -> 328.7 ms and its prompt 603 ->
765).

Built in three steps so a failure could not be ambiguous. The middle one --
`npu_collect_side` extracted byte for byte, and `batch_outbuf` learning to skip
a buffer still in flight -- was verified on the board as a **no-op first**: four
models, two prompt lengths, eight hashes equal to their token loops and every
time on baseline. Then the pair itself, behind `CHARSIU_NPU_PAIR`.

It works, and it is slower on every model:

```
  prompt ms      pair (3 runs)          control (3 runs)       delta
  llama      695 700 695              604 610 604             +15%
  qwen3      801 799 806              712 719 717             +12%
  tinyl      985 990 988              867 866 868             +14%
  gemma3    1116 1139 1121            860 854 904             +28%
```

Every hash is still exactly its token loop, so this is a speed result and not a
correctness one. The split says where it went, on qwen3:

```
              pair    control
  prep        46.6      12.4    <- 3.8x, "buffers, output alloc, memset of Y"
  pack       124.9     100.1
  fence      141.9     132.1    <- the thing it was supposed to CUT
  read       150.2     126.4
  entry      503       403
```

**Every line got worse and the fence went up.** A second output buffer per
geometry doubles the working set of the hottest shape, so both the pack and the
read meet colder memory, and the pool pays to allocate and manage it. Whatever
overlap the hardware gave back did not cover that.

Reverted, both commits. What is worth keeping is the number: the last lever
that does not touch the quantiser was built, measured against its own control
in one session, and is **negative**. The 8% estimate was wrong in sign, which
is the fourth estimate in this area to be wrong -- the read fusion, the NMAX
mechanism, the n-slice balance, and now this. Estimating in this part of the
system does not work; only the board does.

### 🏁 The rope table was rebuilt once per row PER LAYER

A whole day went into the matmul entry, which is 57 to 74% of the prompt. The
other 26 to 43% had not been looked at once.

`rope_table` is `head_dim/2` iterations of `powf`, `cosf` and `sinf`. The
batched prefill is `for layer { for row { ... } }` with the table built inside,
and its arguments are the position and the rope base -- **not the layer**. So a
110 token prompt on a 28 layer model built 3080 tables where 110 would do, and
2970 of them recomputed the same transcendentals to the same bits.

Cached per (position, window variant) for the chunk. Two variants, because a
window layer rotates at its own base and its own head. Cleared once a chunk and
not once an allocation -- the buffers outlive a chunk and the positions do not,
and a stale flag would hand the next chunk the previous one's rotation.

⚠ **The host is a real oracle for this one and it went first.** Without a card
only the NPU matmul falls back; the rope path itself runs. Four models
including both window-layer ones, a **prose** prompt so the hashes differ
between models at all, three arms -- token loop, cached, and
`CHARSIU_ROPE_TAB=0` uncached -- all identical.

Then the board, interleaved, three pairs a model:

```
  prompt ms       cached (3)              rebuilt (3)            delta
  qwen3       698 691 687  (692)      712 707 719  (713)       -21 ms  -2.9%  3/3
  gemma3      854 852 835  (847)      870 855 866  (864)       -17     -2.0%  3/3
  tinyl       858 861 862  (860)      860 876 868  (868)        -8      -0.9%  3/3
  llama       605 604 610  (606)      604 609 607  (607)        -1      noise
```

and the stage line says it is the thing itself:

```
  rope + kv copy   qwen3  0.59 -> 0.52 ms a row      gemma3  0.36 -> 0.22
```

llama is nearly flat and that is the mechanism, not a null result: head_dim 64
over 16 layers is 512 table iterations a row against qwen3's 64 over 28 = 1792,
and its prompt is 81 tokens rather than 111.

**And there is more in this half.** The loop that survives is still, per row per
layer: three memcpys of q, k and v into per-row scratch, `add_bias`, `qk_norm`,
`rope`, and the cache write -- all scalar and all single threaded, because the
scratch is shared and pooling it would need the batched buffers operated on in
place. On qwen3 that stage is still 0.52 ms a row after this, 8.7% of the
prompt, and `attention` is another 16.9%.

### 🏁 Rope in place: the scratch round trip was moving data to where it was

The loop copied q, k and v out of the batched buffers into `s->q`, `s->k` and
`s->v`, transformed them there, and copied q back -- purely so the token loop's
own helpers could be reused unchanged. The batched buffers already hold one
contiguous row each. On Qwen3 that is 8 kB of q in, 8 kB back and 4 kB each of
k and v, **per row per layer**: 672 kB a row over 28 layers, **74 MB across a
110 token prompt**.

`attn_heads` read `s->q`, which made it the caller's job to have put the roped
row there. It now takes the query as a pointer, which is what it always was;
decode passes `s->q` and is unchanged.

Host first, and it covers more combinations than the board round could: four
models including both window-layer ones, a prose prompt, and every crossing of
the two attention paths with the new control -- token loop, in place, scratch,
and scratch with the per-row attention -- all identical.

Board, interleaved, three pairs a model:

```
  prompt ms       in place (3)            scratch (3)            delta
  qwen3       683 684 687  (685)      695 692 697  (695)       -10 ms  -1.5%  3/3
  gemma3      842 834 846  (841)      845 851 856  (851)       -10     -1.2%  3/3
  tinyl       851 838 846  (845)      859 852 853  (855)       -10     -1.2%  3/3
  llama       595 597 602  (598)      600 602 599  (600)        -2      noise
```

and `rope + kv copy` moves with it: gemma3 0.22 -> 0.19 ms a row, qwen3 0.52 ->
0.51 (its q is 16 heads of 128 against gemma3's 4 of 256, so the copy it sheds
is a smaller share of a bigger stage).

**Smaller than the 5.3% the traffic count suggested** -- 74 MB at the board's
2.6 GB/s should have been 28 ms and it is 10 -- which says those copies were
partly hitting cache. Worth writing down as another estimate that came in high;
the direction and the sign were right and the size was not.

Together with the rope table: **qwen3 713 -> 685 ms (-3.9%), gemma3 864 -> 841
(-2.7%), tinyllama 868 -> 845 (-2.6%)**, all from the half of the prompt that
two days of work had not touched.

### 🏁 The rope stage on the pool, and the race the host caught first

With q, k and v roped in place, the rows stopped sharing anything and the stage
could go on the pool. Bias, QK norm and rope for every row now run through
`charsiu_parallel_for`; the KV cache write stays serial beside
`attn_npu_append`, which mutates `a->dirty` and `a->packed` and can reallocate
the mirror.

⚠⚠ **The first attempt was wrong and the host said so before the board saw
it.** `qk_norm` keeps the dequantised gain in a `static` -- the buffer, its
length, and which tensor it holds -- and writes all three on the way through,
including the length check the no-gain case takes before it returns. Pooled,
three models of four came back with different text, and the fourth was Phi-3.5,
**which has no q_norm and never enters the function**. That is as clean a
fingerprint as a race gets. `qk_norm_gain` now takes the gain as a read-only
pointer, dequantised once a layer off the pool; decode keeps `qk_norm` and its
static untouched.

⚠ **And the first board round measured the wrong thing.** `CHARSIU_ROW_POOL=0`
turns off *every* row stage -- silu, the residuals, the norms -- so the arm read
qwen3 648 against 753 ms, which is mostly stages that were already pooled before
this change existed. `CHARSIU_ROPE_POOL=0` moves only the rope:

```
  prompt ms       pooled (3)              rope serial (3)        delta
  qwen3       658 657 651  (655)      687 691 690  (689)       -34 ms  -4.9%  3/3
  gemma3      826 835 822  (828)      830 840 846  (839)       -11     -1.3%  3/3
  tinyl       847 836 850  (844)      839 844 857  (847)        -3      noise
  llama       596 597 597  (597)      603 598 596  (599)        -2      noise
```

qwen3 gains most and that is the mechanism: 16 heads of 128 with a QK norm on
both q and k is the most per-row arithmetic of the four. llama and tinyllama
have no `q_norm` at all, so the pooled stage has almost nothing in it.

The stage line agrees: gemma3's `rope + kv copy` is 0.19 ms a row serial and
**0.07 pooled**.

### The three rope changes together

```
  qwen3     713 -> 655 ms   -8.1%
  gemma3    864 -> 828      -4.2%
  tinyl     868 -> 844      -2.8%
  llama     607 -> 597      -1.6%
```

All of it out of the 26 to 43% of the prompt that is not the matmul entry --
the half two days of work had never touched, and which was reached only because
"there must be a way out" turned out to be right.

### ⛔ The attention block width does nothing at this length, and the rule said stop

Attention is the last unmeasured piece of the non-matmul half: 16.9% of qwen3's
prompt at the scoreboard's own length. It is already blocked, pooled over heads
and vectorised -- `attn_dot4` takes four key rows at once and the R query rows
of a block share them -- so the only untried thing was its one number.
`CHARSIU_ATTN_BLOCK` is 4, chosen at 916 tokens where the cache does not fit in
L2 and the trade is different.

The round was written with its own stopping rule: R trades cache reuse against
the scores buffer, `n_head * R * n_ctx` floats, which on Qwen3 at R = 32 is
4 MB against this board's 1 MB of L2 -- **so the curve had to turn somewhere,
and if it did not, the sweep was fitting and would stop.**

```
  prompt ms      R=2    R=4    R=8    R=16   R=32
  qwen3          656    656    650    656    651
                 666    702    658    652    652
  gemma3         824    817    836    821    816
                 827    831    828    836    829
  tinyl          834    844    838    851    847
                 837    836    840    841    852
```

Flat, on all three, across a sixteenfold range. **It did not turn**, so R is not
what binds here and no default moves.

Which is itself the answer about attention: at 110 tokens the whole KV cache of
a layer is 0.9 MB and stays in L2, so reuse is not the constraint. Counting the
arithmetic -- 6105 query-key pairs, 16 heads, 128 wide, dot and axpy -- gives
about 1.4 GFLOP a prompt in 110 ms, **12.7 GFLOP/s across four A72s and four
A53s**, which is close to what this board can do. There is no factor sitting in
attention at this length.

### The scoreboard at the end of the day

Same kernel `2ffc0913`, governor performance, `CHARSIU_BENCH_REPEAT=6`:

```
                 TTFT ours   theirs    gap      decode ours   theirs
  Qwen3 0.6B       602        469     1.28x      24.90        24.85
  TinyLLAMA 1.1B   870        544     1.60x      20.76        19.71
  Phi3 3.8B       2963       1829     1.62x       6.89         6.58
  Gemma4 E2B      2225       1219     1.83x       8.71         9.23
```

Across the three readings of this kernel:

```
             yesterday    this morning    now
  Qwen3      687  1.47x   653  1.39x     602  1.28x
  TinyLLAMA  900  1.65    890  1.64      870  1.60
  Phi3      3057  1.67   2994  1.64     2963  1.62
  Gemma4    2352  1.93   2272  1.86     2225  1.83
```

⚠ Those are three different sessions and the board drifts about 3%. What makes
the column trustworthy is that it agrees with the paired in-session numbers the
five changes were each measured with. On qwen3 those were -1.4% (per-slice
input buffers), -1.4% (the tail scale), -2.9% (the rope table), -1.5% (rope in
place) and -4.9% (the rope stage pooled), which compound to **-11.6%** against a
measured 687 -> 602, or **-12.4%**.

**Qwen3's decode is now above the vendor's** and three of four models are at or
above it.

Everything shipped today came from two moves, used five times:

1. **Name the unnamed row before optimising a named one.** The pack's FINI
   ioctls and the tail per-channel scale were both found by printing an
   instrument that already existed and had never been read.
2. **Look at the half nobody has looked at.** Two days went into the matmul
   entry, which is 57 to 74% of the prompt. All three rope changes came out of
   the other half, in the last few hours, after "there must be a way out."

### 🏁 The decode attention pool, and gemma4 is still not there

I reported "three of four at or above the vendor's decode" and led with the
model that wins. Gemma4's decode was **8.71 against 9.23, 94.4%, behind** -- the
number was in the table and the framing walked past it. That is the wall's
mistake: a story fitted to the cases that agree, with the exception left
unchased.

Chasing it: gemma4 does **9.98 tok/s at an 8 token context and beats the
vendor**, and 8.78 at the scoreboard's 111. The deficit is attention growing
with the context, not weights.

```
  attention ms a token      ctx 8     ctx 113    ctx 694
  gemma4                     1.48      13.37      62.19
  gemma3                     0.57       3.50      15.81
  qwen3                      1.39       8.66        --
```

Decode attention was deliberately serial, and round 368's comment is why -- and
it wrote its own follow-up: *"the path stays, because attention grows with the
context and 38 positions is not where this question gets settled."*

Settled at 113, five models, text identical in every pair:

```
                pinned                   unpinned
  gemma4     8.78 -> 9.12  +3.9%      8.79 -> 9.11  +3.6%
  qwen3     25.75 -> 27.17 +5.5%     25.70 -> 27.18 +5.8%
  tinyllama 20.90 -> 22.80 +9.1%
  llama     20.98 -> 21.57 +2.8%
  gemma3    21.26 -> 21.04 -1.0%     14.24 -> 18.53 +30%
```

⚠ **Round 368's 15 ms unpinned penalty is gone** -- unpinned is now the best
case of all, from the QoS hold and affinity work that landed since. At eight
positions it still loses, -0.5 to -2.5%, exactly as that round said. So the
rule is the context and not the model: pool from `CHARSIU_ATTN_POOL_MIN`
positions, 64 by default.

Scoreboard, best of six:

```
             decode ours   before   theirs   ours/theirs
  Qwen3       26.43        24.90    24.85     106.4%   (was 100.2%)
  TinyLLAMA   22.89        20.76    19.71     116.1%   (was 105.3%)
  Phi3         7.01         6.89     6.58     106.5%   (was 104.7%)
  Gemma4       9.00         8.71     9.23      97.5%   (was  94.4%)
```

**Gemma4 is still behind, by 2.5%.** Most of the gap closed and none of it is
closed by adjective. What is left, from its own token at 100 ms: `gate + up`
35.06 ms and `down` 19.73 -- 55% of the token in two weight reads, at the
9.74 GB/s the hardware path reports against the 11.9 this board has been
measured reading at. That is where the last 2.5% is, and it is a bandwidth
question, not an attention one.

TTFT moved with it: 609 / 866 / 2934 / 2183, so 1.30 / 1.59 / 1.60 / 1.79.

## 2026-09-07 evening: the bandwidth question, answered by refusing it

The handoff above says gemma4's last 2.5% is bandwidth -- "gate + up 35.06 ms
and down 19.73, 55% of the token in two weight reads, at the 9.74 GB/s the
hardware path reports against the 11.9 this board has been measured reading
at. Ask why the weight fetch runs at 82%."

Every part of that is wrong, and the tree said so before the round started.

**npudev.c already refuses the number it quotes.** The 9.74 GB/s is
`weight_mb / busy_us`, and the comment beside it: *"The 550 MB a token over
58.4 ms that reads as '9.4 GB/s, the bandwidth roof' is an average over stages
that run from 6.67 GB/s (q k v) to 15.60 (the head): a roof does not have a
2.3x spread across shapes, a fixed cost does."*

**gemma4's own stage table says the same, from its gguf shapes:**

```
  stage                        MB a token     ms    GB/s
  q k v                              81.0    8.15    9.9
  o proj                             74.3    6.05   12.2
  gate + up                         583.9   34.77   16.8    the BIGGEST is the FASTEST
  down                              292.0   19.51   15.0
  residual (per-layer embedding)     15.5    6.37    2.4
```

`gate + up` is above every rate this board has ever been quoted at. There is
no 82% to recover there.

### What the board's own three-term fit says, once the grep stops eating it

`charsiu_npu_report` has fitted `us a call = A + B a task + C a MB` on every
run since it was written -- `llama_state_free` calls it ungated -- and every
round's grep has discarded it, m105's included. Rounds 410 and 411 truncated
it mid-word before 412 finally kept it:

```
  gemma4   us a call = 43 + 7.7 a task + 116.7 a MB   of 4277 ms: 435 per call,
                                                      185 per task, 3288 weights
                                                      at 16.98 GB/s across 2 cores
  gemma3   us a call = 51 + 3.2 a task + 114.3 a MB   of 1881 ms: 257 per call,
                                                       26 per task, 1496 weights
                                                      at 16.38 GB/s across 2 cores
```

**The two models stream at the same rate.** Dispatch is 10% of gemma4's
hardware path and 14% of gemma3's. Neither is bandwidth-starved and neither is
dispatch-bound. gemma4 is simply a bigger model a token: 211 calls where
gemma3 makes 105, and 1140 MB of weights where gemma3 reads 545.

### Slices are cheap, and that killed my own hypothesis

Predicting the slice count from the gguf shapes and checking it against the
board: gemma4 933 predicted / 937 measured, gemma3 292 / 292 exact. The reason
they differ is `llama_auto_kmax`, which leaves gemma4 at KMAX 1024 and gives
gemma3 2048 -- gemma4's `down` has K = 6144 and 12288, exact multiples of
1024, and the rule declines the WHOLE MODEL when ANY K would regroup.

So: cut the slices, cut the cost. Round 411 cut them 937 -> 486, a 37% drop in
tasks, and bought **1.2% of the token**. Round 412's `CHARSIU_NPU_KFIT` cut
them to 693 -- 691 predicted, so the model is right -- and bought 1.1%.

```
              slices   tasks   hardware   a token   core balance
  default        937   24080     4277 ms   101.3     1.03x
  KMAX 2048      486   15280     4141       99.7     1.13x
  KFIT           693   19216     4142      100.2     1.07x
```

⚠ And fewer slices makes the balance WORSE, visibly: `q k v` went 8.23 -> 9.05
ms when its tensors fell to one slice and one core sat out the call. That is
the hazard npudev.c already documents, arriving on cue.

So PLAN.md section 2b's open speed half is answered: **KFIT is worth about 1%**,
and it is not where gemma4's deficit is.

### The one stage that misses the line, and it is not on the NPU

Fitting gemma4's largest call against its smallest gives `79 us a call + 54.8
us a MB`. Every stage of both models lands on it within 27 us:

```
  stage                calls   MB a call   measured   on the line   excess
  q k v                   35       2.314        233           206      +27 us
  o proj                  35       2.123        173           195      -22
  gate + up               35      16.683        993           993       -0
  down                    35       8.343        557           536      +21
  residual = pl pair      70       0.221         91            91        0
  gemma3 head              1     169.900       9380          9390      -10
  gemma4 head              1     226.500      17710         12496   +5214
```

**5.21 ms above the line on one call, 5.2% of a 100.9 ms token**, against a
2.5% deficit -- and gemma3's head, same 262144 vocabulary, same int4, sits
dead on it.

The difference is one line of metadata. gemma4 declares
`final_logit_softcapping` and gemma3 does not, so llama.c ran **262144 scalar
`tanhf` calls a token on one core**, inside `output head`, where nothing
charged them separately. The NPU side is excluded twice over: giving that head
a single K slice moved it 17.71 -> 17.18 (KMAX 2048) and 17.09 (KFIT), a tenth
of the gap each.

`softcap_logits()` pools it. The squash stays exact -- the same `tanhf` in the
same order, only a different core -- and `CHARSIU_SOFTCAP_POOL=0` is the
control.

### ⚠⚠ Two traps this round walked into, both of which have bitten before

**Hashing `charsiu_run`'s stdout hashes its own timings, in TWO forms, and
the second one cost three board rounds.** The `[load ... | prompt ... tok/s]`
summary goes to stdout -- that one `grep -v '^\['` removes. But with
`CHARSIU_STAGES=1` **the whole stage table goes to stdout as well**, once per
report, interleaved with the generated text, milliseconds and all. Neither
`2>/dev/null` nor the bracket strip can reach it.

So round 413's two arms hashed differently, and I went and built three rounds
to find out why:

```
  414  gemma4, STAGES off, 4 runs (2 pooled, 2 serial)   ALL FOUR IDENTICAL
  415  gemma4, STAGES on,  4 runs (2 pooled, 2 serial)   all four differ
  416  gemma3, STAGES on,  3 runs                        differs 3 ways
       gemma4, STAGES on, CHARSIU_ATTN_POOL=0            differs 3 ways
       gemma4, STAGES on, CHARSIU_THREADS=1              differs 3 ways
```

⚠ **The single-threaded arm was the tell and I ran it last.** A run on one
thread that still "differs" is not computing anything differently. Extracting
the generated text alone on the host: four runs, two with the timer on, **one
distinct text**. There is no nondeterminism -- not in the model, not from the
stage timer, not from the change.

The same mistake also produced a claim earlier in this round: I wrote that KMAX
2048 changed gemma4's answer, on hashes that contained stage tables. Nothing
was shown either way. **Hash with the stage timer OFF, and strip `^\[`. If a
round needs both timings and a text check, run the model twice.**

**`d41d8cd98f00` turned up again**, from a model path that does not exist on
the host, and three runs "agreed" about nothing.

⚠ Round 412 arm 2 raised `CHARSIU_NPU_NMAX` to 16384 and **wedged both cores**:
job timeouts and `rk_iommu ... MMU_DTE_ADDR is not functioning`. It recovered
by the next arm. 8192 is a limit, not a default.

## 2026-09-07 late: the scoreboard, and decode is done

`board_vendor.sh`, best of six, kernel `2ffc0913`, rail 800 mV, governor
performance. The second pass is the same script with `CHARSIU_SOFTCAP_POOL=0`,
so the control ran on the same boot as the arm:

```
              decode ours   theirs   ours/theirs     this morning
  Qwen3        26.59        24.85     107.0%         106.4%
  TinyLLAMA    22.85        19.71     115.9%         116.1%
  Phi3          7.02         6.58     106.7%         106.5%
  Gemma4        9.26         9.23     100.3%          97.5%
  Gemma4 ctrl   8.99         9.23      97.4%    <- softcap serial, same boot
```

**All four models are at or above the vendor's decode.** Four things agree at
once, which is more than the arm alone would give:

- the arm beats the vendor and the control does not;
- **the control reproduces this morning's 9.00** to within 0.01, so the
  baseline is the baseline and not a drift;
- the three models that declare no softcapping did not move -- 107.0 / 115.9 /
  106.7 against 106.4 / 116.1 / 106.5, all inside the 3% band;
- **TTFT is 2192 in both arms**, which is what a change that runs once a
  prefill and 262144 times a decode has to look like.

TTFT: 610 / 866 / 2937 / 2192, so 1.30 / 1.59 / 1.61 / 1.79 against theirs.

### ⚠ And one change I got wrong on the way

I committed `act_mul` -- the same elementwise join, split across the pool --
default ON, reasoning by analogy from the rope and tail scale wins of the
morning, without measuring it. The host, gemma-3-1b, 26 layers of 6912:

```
  CHARSIU_ACT_POOL=0    silu * up   0.18 ms a token
  CHARSIU_ACT_POOL=1    silu * up   1.91 ms a token     10x WORSE
```

26 barriers a token against 7 us of work a layer. `llama.c`'s own note on the
batched version of that same stage says it in as many words: a pool call is a
barrier and a stage can be too small to pay for one. **An analogy is not a
measurement**, and the default is off until a board round says otherwise --
the board's serial loop is 10.4 ns an element against the host's 1.0, so the
break-even for an eight way split lands near 7250 elements, which would split
gemma4's twenty 12288 wide layers from its fifteen 6144 wide ones.

## 2026-09-07 late: KFIT was priced on the wrong half

Round 412 measured `CHARSIU_NPU_KFIT` at decode, got 1%, and I wrote it off.
That is the right answer to the wrong question. The read back is
`m * n * ceil(K / KMAX)`, so at m = 1 there is almost nothing to read and the
slice count moves almost nothing. **At a prompt it is the biggest stream in
the run.**

gemma4 is the only one of the four scoreboard models on KMAX 1024 --
`llama_auto_kmax` declines the whole model when ANY of its K values would
regroup, and its `down` has K = 6144 and 12288. So it reads back twice what
its shape needs, on every row of every prompt:

```
  KMAX 1024, what gemma4 runs         read back  998 MB a 111 row prompt
  KMAX 1024 + KFIT                               652        -35%
  KMAX 2048, what the others get                 511        -49%
  the weights themselves, read once             1273
```

Round 420, three paired reps, 93 token prompt, text identical over four runs:

```
                slices  submits  read a row   entry a row   fence a row
  KFIT off        937     2450   4.75 ms      12.50         4.17
  KFIT on         693     2054   3.10  -35%   11.29  -9.7%  4.82  +16%
```

**The read fell by exactly what the arithmetic said it would.** 998 -> 652 MB
is -35%, and 4.75 -> 3.10 ms a row is -35%.

⚠ And the fence went UP 16%, which is the same core-balance cost the decode
round saw: fewer slices means a coarser deal between the two cores. It is
paid back three times over here, but it is the reason this is 9.7% and not 35%.

### The control could not have been arranged better

qwen3 ran beside it and **KFIT removes not one slice from it**: 299 -> 299.
Every one of its K values is a multiple of its KMAX, so there is no remainder
for the last slice to absorb. Its entry does not move either -- 3.89 -> 4.05
ms a row, inside the noise of the arm.

So the gain tracks the slice count, on a model where the slice count moves,
and vanishes on a model where it does not.

### ⛔ And a premise I had to throw away first

I had worked out that gemma4's 111 token prompt splits into two chunks of 80
and 31, so the weights are read twice, about 102 ms. **It does not.**
`CHARSIU_PREFILL_ONECHUNK` has been on since August and runs the whole prompt
as one chunk whenever it fits under the model's cap; gemma4's cap is 160
(`163840 / min(12288, KMAX 1024)`), so 111 rows are one chunk. The widths line
in round 420 reads `1x92` for a 93 token prompt, which is the tree telling me
so directly, in a line I had already grepped for.

The chunk table in `charsiu_run.c` -- SmolLM2 catastrophic at 160, Qwen3 9%
better, Llama neutral -- has no gemma4 in it and still does. But it is a
question about prompts LONGER than 160 tokens, not about the scoreboard.

## 2026-09-07 late: KFIT is a per-model property, and one arm was measuring a bug

Round 421, four models, three paired reps, 93 token prompt, text identical on
every arm of every model:

```
  model    slices        prompt            decode           verdict
  gemma4   937 -> 693    1597 -> 1470 ms   9.69 -> 9.82     -8.0% TTFT, +1.4%
  gemma3   292 -> 266     719 -> 1398      21.43 -> 20.97   +94%, a catastrophe
  tinyl    404 -> 382     745 ->  754      22.82 -> 22.43   slightly worse
  phi3     844 -> 844    2629 -> 2629      7.21 -> 7.21     zero control, held
```

**phi3 is the control that could have refuted this and did not.** Every one of
its K values is a multiple of its KMAX, so KFIT has no remainder to absorb and
removes no slice -- and nothing about it moved. It was written down in advance
that if phi3 moved, gemma4's win would need another explanation.

### gemma3's arm was not measuring KFIT

`submits 5020 -> 9128`. Fewer slices, 82% MORE submits. That is not a slicing
result, it is a fallback.

KFIT takes gemma3's `down` (K = 6912, KMAX 2048) from four slices to three,
and `charsiu_slice_kw` gives the last one whatever is left: **2816 wide**, a
width npudev.c already records as `WRONG, surf 88` from an unrelated model.
The surface ceiling is `(slice / 32) * m <= 5120`, so at 93 rows it is
88 * 93 = 8184 -- every projection refused, every row of the prompt back on
the token loop, in silence, with the text still correct.

`llama_prefill_chunk_cap` could not see it: it took `min(widest K, KMAX)`, and
KFIT is the one thing that makes a slice **wider** than KMAX. Fixed, and the
caps move:

```
  model     KFIT   widest slice   cap
  gemma4      0        1024       160
  gemma4      1        1536       106
  gemma3      0        2048        80
  gemma3      1        2816        58
  qwen3     0/1        1024       160   (no K of it has a remainder)
```

⚠⚠ **And gemma4 cleared its new 106 cap by thirteen rows, by luck.** The round
used a 93 token prompt; the scoreboard's is 111. 93 passing says nothing about
111, and with KFIT on and the cap unfixed, the scoreboard prompt would have
walked into the same silent fallback gemma3 did. Round 422 uses a prompt over
110 so the clamp actually engages.

### What is still open

KFIT wins on one model of four. gemma4 is the model that needs it -- worst
TTFT ratio of the four -- but one win, one catastrophe caused by a bug now
fixed, one small loss and one no-op is not a default. Round 422 re-runs the
three that move, with the cap fix installed and a prompt that splits, and that
decides whether this is a default or stays an explicit switch.

## 2026-09-07 late: KFIT does not ship, and the reason is the chunk and not the slice

Round 422, the cap fix installed, a 109 token prompt so the clamp engages --
which is the scoreboard's regime and not the 93 that flattered round 421:

```
  model    widths off -> KFIT       prompt off -> KFIT   decode
  gemma4   1x108   -> 1x80+1x28     1836 -> 1815 ms      9.56 -> 9.68
  tinyl    1x116   -> 1x80+1x36      822 ->  897         (EOS, no decode)
  gemma3   1x80+1x28 already two chunks off
```

**-8.0% became -1.1%.** KFIT takes gemma4's widest slice from 1024 to 1536, so
its cap falls 160 -> 106, and a 108 row prompt no longer fits in one chunk. The
second chunk reads all 1273 MB of weights again, which is most of what the 35%
smaller read back had just saved.

TinyLLAMA is the same mechanism with none of the compensation: it ran `1x116`
in one chunk, KFIT put its cap at 106, and it lost 9.1%.

**So it is the chunk COUNT that decides, not the slice count.** The read back
is the biggest single stream in a prefill and KFIT genuinely cuts it by 35% --
and a single extra pass over the weights outweighs that. KFIT stays an
explicit switch.

⚠ Round 421 said -8.0% on a 93 token prompt and I wrote "93 passing says
nothing about 111" into the round that followed. It did not.

### ⚠ And auto_kmax has a cost nobody had priced

`llama_auto_kmax` widens gemma-3-1b to KMAX 2048 because that halves its
slices, 532 -> 292. It also doubles its widest slice, 1024 -> 2048, which
halves its chunk cap, 160 -> 80 -- and at 109 rows that is the difference
between one chunk and two.

```
  gemma3   KMAX 1024   532 slices   cap 160   one chunk
  gemma3   KMAX 2048   292 slices   cap  80   two chunks   <- what it runs
```

The function optimises the read back and does not know the chunk exists. Which
of the two is faster has never been measured, on any model. It is the same
trade KFIT just lost, run in the opposite direction.

## 2026-09-07 late: the quantiser decision, priced — and it is the user's

Where gemma4's prefill actually goes, 93 rows, warm, measured:

```
  prompt total                    1589 ms
  fence -- which IS the MAC time   372 ms   0.465 TMAC/s
  read back                        433      836 MB at 1.93 GB/s
  pack                             239
```

The vendor does the same prompt in about 1021 ms. If their MAC runs at our
rate, that is 372 ms of arithmetic and **649 ms for everything else** -- and
our read plus pack alone is 672. We are not slower at the maths. We move more
bytes around it.

**Every lever on the read is closed except one.** read fusion lost 2.3x on
this board on all eight models (one read stream, four write streams, and an
A72 store buffer that will not merge interleaved partial lines); read_rows4
and read_rows2 both lost; the NEON form was bit-identical and moved nothing;
KFIT trades the read for a chunk and loses; the fence is already the MAC.

What is left is the quantisation group, because the read back is
`m * n * ceil(K / KMAX) * 4` and the vendor's own .rkllm dispatches K =
2048/4096 and never 1024. Round 425, gemma4, three reps an arm, 76 token
prompt -- short enough that every arm still runs ONE chunk, so this is the
lever without the chunk cost:

```
  group   slices   prompt        decode
   1024      937   1263 ms       9.78 tok/s
   2048      486   1082  -14.3%  9.96  +1.8%
   4096      256   1222   -3.2%  9.65  -1.3%   <- capped to 40, two chunks
```

**2048 is the optimum and 4096 is worse**, exactly as the chunk arithmetic
predicted: a wider group widens the slice, which lowers the chunk cap, which
splits the prompt and reads every weight again.

⚠ At the scoreboard's 111 tokens group 2048's cap of 80 splits the prompt, so
the -14.3% becomes about -9.6%. 76 rows flatter it.

### It changes the weights, and here is what that looks like

Same prompt, temperature 0, 60 tokens:

```
  1024  ...a man of unyielding resilience and quiet fortitude. He was a man who
        faced relentless adversity with a stoic acceptance of his harsh
        circumstances, yet he possessed an inner strength that allowed him to
        endure even the most brutal conditions.

  2048  ...a man of profound resilience. He was a man who didn't succumb to
        despair when faced with overwhelming adversity. He was a man who found
        a way to persevere through seemingly insurmountable odds. He was a man
        who possessed an unwavering determination to keep going

  4096  ...a man of profound resilience and quiet fortitude. He was a man who
        faced adversity not with despair, but with a quiet, unwavering resolve.
        He was a man who understood that life is not always easy, and that even
        in the face of overwhelming
```

All three are usable. 2048 repeats its sentence frame four times where 1024
varies it, and 4096 reads closer to 1024 than 2048 does -- so the quality is
not monotonic in the group width, which is itself worth knowing.

⚠⚠ **One prompt, one sample, greedy. That is not a quality measurement** and
nothing here should be read as one. It is what the change looks like, put
beside what it costs, so that a person can decide. `llama_auto_kmax` declines
this widening on gemma4 by design and says why; overriding it is
`CHARSIU_NPU_KMAX=2048 CHARSIU_NPU_W4_GROUP=2048`.

## 2026-09-07 late: two arithmetic errors of my own, both in the flattering direction

### `packer` is per SLICE, not per call

The pack split, gemma4, 92 rows: `gather 0.98  packer 1.22  the rest 0.39`,
and `the rest` divides into `emit 0.07 + fini 0.31` -- which sums to 0.38
against the 0.39 printed above it, so the counters agree to the hundredth.

I first divided `packer` by the 140 matmul CALLS and got 0.80 ms a call
against npudev's note that the fp16 conversion costs about 7 us, and wrote
that up as a 121x anomaly. **The loop is per K SLICE.** 937 of them, so it is
0.12 ms a slice moving about 530 MB of fp32 in and fp16 out:

```
  packer   112 ms / 937 slices = 0.12 ms   ~530 MB   4.7 GB/s
```

⚠ I also shipped, for about ten minutes, a report line printing
`packer - emit - fini` as "the packer itself". `tpe` starts AFTER
`bpackcall_us` is banked, so those are disjoint intervals and the subtraction
was meaningless. The board is what caught it: emit + fini came to exactly
`the rest`, not to a share of `packer`.

### And the read is 5.2 GB/s, not the 1.93 I quoted all evening

I counted only the int32 accumulator. **Y is read and written once per K
slice** -- that is what "Y once per K slice" in the read_rows note means, and
I had read that line. Per output-slice element it is 4 bytes of accumulator
plus a read-modify-write of Y on every slice after the first:

```
  read   434 ms / 209 M slice-elements = 10.8 bytes each   5.2 GB/s
```

So `36 cycles an element is too many for index-load-add-scale` was wrong too:
36 cycles moving 10.8 bytes is bandwidth, not compute.

### What that changes

```
  stage    rate      of the board's 11.9 GB/s sequential
  read     5.2       44%
  packer   4.7       39%
  fence    --        the MAC, and 3-4x the vendor's own wall clock
```

**Nothing on the CPU side of gemma4's prefill is running at a fraction of
what this board can do.** 40 to 45% is what a permutation walk and a
gather-plus-convert get here, and that is why every attempt to speed them up
failed: read fusion, read_rows4, read_rows2, the NEON form, the pair submit.
They were all trying to make something faster that was already at rate.

⇒ The remaining lever is not the rate, it is the BYTES, and the byte count is
`m * n * ceil(K / KMAX)`. Which is the quantiser group, and it is the user's
call. That conclusion has not changed, but it now rests on every stage being
measured at rate rather than on a list of failed attempts.

## 2026-09-07 late: six weeks of comparing charsiu against itself

`charsiu_ppl` exists now, and the first thing it did was ask a question this
project has never asked: **are the answers as good as the file's own q4_0?**

Every correctness check here has been an INTERNAL one -- "tokens identical to
the token loop", "text unchanged", eight models byte for byte, `board_text_all`
across nine architectures. They all compare charsiu to charsiu. None of them
compares it to llama.cpp.

```
                              ppl, same 200 tokens, same tool
  host   gguf q4_0 + fp32     Qwen3  43.85     gemma4  54.43
  board  charsiu int4 w4a16   Qwen3  75.17     gemma4  82.26
                                    +71%              +51%
```

### ⛔⛔ And the int8 path is not slower. It is broken.

```
  qwen3   w8a8  272369.9386        gemma4  w8a8  218167.8116
```

PLAN.md and this notebook both carry *"for PREFILL, which is MAC-bound, int8
weights are the faster arm and int4 is the DRAM-bound choice that belongs to
decode"*, and the round that produced it measured **milliseconds**. It was
turned down for being 25 to 30% slower on the whole prompt -- which is the only
reason nobody shipped a configuration that emits noise.

⚠ "int8 weights" is a misleading name for the arm. The ACTIVATION is quantised
to int8 as well; the int4 path keeps fp16 activations. What collapses is the
activation, not the weight.

⚠⚠ **AND THAT LAST SENTENCE IS WRONG -- SEE "w8a8 was never broken", the next
day.** Nothing collapsed. The weights were quantised into 1024-wide groups and
read as one scale a row, because `tensor_grouped()` requires `g->w4` and the
quantiser did not know it. With the layouts agreeing, this arm measures 27.07
against llama.cpp's own q4_0 at 26.64 -- the BEST quality number in the tree.
The guess in this paragraph was made without asking the CPU reference, which
answers it in one command.

### ⚠⚠ How it was found: `CHARSIU_NPU=0` opened the NPU

The round wanted a CPU control. `if (getenv("CHARSIU_NPU"))` is an existence
test, so the one spelling anybody would reach for to disable the device
enabled it, with W4V unset -- which is w8a8.

**Two arms came back bit-for-bit identical, 272369.9386 twice, and that is the
only reason it was caught.** The arm labelled "CPU, gguf weights" was the arm
labelled "NPU int8 weights".

Tonight's `charsiu_env_flag` sweep converted the twenty `!= NULL` switches and
missed this shape entirely. There are 51 implicit existence tests in the tree;
the three on `CHARSIU_NPU` itself are fixed, because that is the switch every
board round and every installer sets.

### 🔑 And ppl is the one quantity here that survives a session boundary

Rounds 427 and 428 ran the four int4 arms on different boots, with other models
and a reinstall in between, and every figure repeated **to the last digit**:
75.1684, 106.6864, 82.2555, 76.2290. It is a greedy forward with no sampling
and no timing in it. Where "never A/B across sessions" is the rule here, this
is the exception -- and that is worth having, because a quality regression can
now be caught weeks after it lands.

### The two int4 groups disagree between models, and not in the flattering way

```
              group 1024   one scale a row
  qwen3          75.17        106.69      +42%   much worse
  gemma4         82.26         76.23       -7%   better
```

⚠ qwen3's two coarse arms are identical because neither of its K values is a
multiple of 2048, so `group 2048` already IS one scale a row for it -- the
tensor_grouped test needs `k % kgroup == 0`. That is a check on the harness,
not a result.

Offline Frobenius error said grouped is better on every tensor. Perplexity
says the opposite on gemma4. npudev.c's own caption -- "weight error is not
the objective" -- is now measured rather than asserted.

## 2026-09-07 late: the quality instrument, and three arms that were inert

`charsiu_ppl` on the board, 200 tokens, model md5 identical to the host's:

```
                                   qwen3     gemma4
  board CPU  gguf q4_0 + fp32       43.49     55.22    <- host 43.85 / 54.43
  board NPU  charsiu int4 w4a16     75.17     82.26       +72.8%  +49.0%
  board CPU  charsiu int4 + int8 a 113.23     79.38
  board NPU  w8a8                 272370    218168    <- noise
```

⚠⚠ **THE w8a8 ROW IS A BUG, NOT A PROPERTY -- see "w8a8 was never broken", the
next day.** Both figures are one tensor's scales read in the wrong layout. On
qwen3 the fixed arm measures 27.07, better than every other row in this table
including the gguf's own q4_0 on the same tokens. **gemma4's 218168 has NOT
been re-measured** and the qwen3 result says it should be.

The chain checks out: the board's CPU arm lands within 1.5% of the host's, on
a file whose md5 matches, so the whole +73% / +49% belongs to the NPU path.

**The cause is not exotic.** q4_0 carries one fp16 scale per 32 weights;
charsiu carries one per `CHARSIU_NPU_W4_GROUP`, which auto_kmax sets to 1024.
Thirty-two times coarser -- and the group cannot simply be narrowed, because
the K slice IS the group and the read back is `m*n*ceil(K/KMAX)*4`.

### ⚠⚠ Three arms in a row measured nothing, and each control caught it

1. **`CHARSIU_NPU=0` opened the NPU.** `if (getenv(...))` is an existence test.
   Two arms came back bit-for-bit identical, 272369.9386 twice, and that is the
   only reason it surfaced. Fixed in llama/vision/whisper; 51 implicit tests
   remain elsewhere.
2. **The calibration wrote zero tensors.** `npu_calib_note` is called from one
   place -- `npu_matvec`, the CPU reference -- so a calibration run with the NPU
   on records nothing. Through `CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1` it wrote 197
   tensors and 2.3 MB.
3. **AWQ was never on.** `CHARSIU_AWQ_STATS` names the statistics file;
   `CHARSIU_NPU_AWQ` is the exponent and defaults to 0, which gates the whole
   block. Two rounds set only the first and returned 75.1684 to the digit.

Every one of the three was caught by a control arm returning EXACTLY the same
number as its baseline. An arm that reproduces its control to seven figures is
not a null result, it is an inert arm.

### And the separation probe changed two variables, so it separated nothing

`CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1` was meant to isolate the quantiser by
running charsiu's own int4 weights through the CPU reference. It does -- but
`npu_matvec` takes `a->q1`, an int8 activation, where the NPU path keeps fp16.
So the arm is w4a8 against w4a16 and moves the activation as well as the
hardware. qwen3 got worse (113 against 75) and gemma4 got better (79 against
82), which is two variables and one number.

Isolating the weight quantiser needs a charsiu-weights + fp32-activation path,
and there is not one.

## 2026-09-07 late: measuring the denominator I had been quoting all evening

Every "at rate, no lever left" conclusion tonight divided by 11.9 GB/s, which
came from a memory note and had never been measured on this card.
`charsiu_membw` had a build rule since it was written and was in neither `all:`
nor PROBE_BINS, so it had never run here.

```
  1 thread    8.62 GB/s
  2           8.04
  4           7.53      <- more threads, less bandwidth
  8          11.91      <- this is the 11.9
```

**Non-monotonic.** 11.9 is the EIGHT-thread figure, and the read back is
pooled across eight, so the denominator was right -- but I did not know it was
an eight-thread number, and I did not know 1 to 4 threads sit lower. The
board is four A53 and four A72 and the probe does not pin, so one thread
probably lands on an A72 and four straddle both kinds.

The tool's own header puts the bus peak near 21.9 GB/s (LPDDR5 2736 MHz, two
16-bit channels), so 11.91 is 54% of theoretical, which is ordinary.

### What it changes

Nothing in the conclusion, and two things around it:

- `read` at 5.2 GB/s is 44% of what eight threads can reach sequentially, and
  it is a PERMUTATION walk. 44% of sequential for a permutation is high, not
  low. `pack` at 4.7 is 39%. Both stand.
- **any pool of four threads or fewer is capped at 7.5 to 8.6**, below what
  eight get. That is an argument for `CHARSIU_POOL_CPUS=0-7` that nobody had
  measured, and it explains the "2.0x ceiling" on threading the read: the
  controller does not scale with threads, it steps.

⚠ And the NPU reads weights at 16.98 GB/s -- ABOVE anything the CPU can reach
here. The two do not share a path in the way the CPU-side numbers assume, which
is why the fence can sit at the MAC rate while read and pack are held at 11.9.

## 2026-09-07 late: AWQ was two bugs deep, and the second one is the interesting one

### Bug one: the factor was never applied to the activation

npuquant folds the AWQ factor into the weights (`row[i] /= t->kscale[i]` in
quant_rows) and leaves `t->kscale` for the caller to undo on the input. Its
note: *"on the board this is one multiply a k before the pack"*. Nothing did
it -- `kscale` appeared twelve times in npuquant.c and zero times in npudev.c.

```
  CHARSIU_NPU_AWQ off                     75.17
  CHARSIU_NPU_AWQ=0.5, column means      10755.75
  CHARSIU_NPU_AWQ=0.5, real |x| stats  65233808
```

The better the statistic, the worse the output: the signature of a factor
nothing undoes, since a truer mean |x_k| gives a more extreme one.

`charsiu_npu_matvec` applies it now. `charsiu_npu_matvec_group` refuses a
tensor that carries one -- the whole point of that entry is that q, k and v
share one packed input, and a per-tensor factor means per-tensor inputs.

### Bug two: with the multiply in, it still collapsed -- and the CPU said it was not mine

```
  board, NPU, factor now applied   AWQ 0.5   2155726
                                   AWQ 0.25     1103.80
```

⚠ The CPU reference path applies the factor too (`a->f[i] * t->kscale[i]`) and
touches none of tonight's code. On the host:

```
  CPU reference, AWQ off        114.22
  CPU reference, AWQ 0.5   7252312.56
```

**So it was broken in the quantiser, on every path, since it was written.**
That is what made it debuggable: the host runs the same collapse in seconds.

The factor is clamped to `[0.125, 8]` -- sixty-four fold. It DIVIDES the
weights, so a k with f = 0.125 has its column multiplied by eight before
rounding, and one such column sets vmax for the whole row: the int4 step for
every other weight in that row goes eight times coarser. That is the shape of
a collapse, not of a trade. Published AWQ keeps the factor near 1; the point is
to protect a few salient channels, not to rescale the tensor.

```
  CPU reference, qwen3, 200 tokens, AWQ 0.5
    off                    114.22
    clamp [0.95, 1.05]     111.08     <- AWQ starts working
    clamp [0.125, 8]  7252312.56      <- what it shipped as
```

`CHARSIU_NPU_AWQ_CLAMP` sets the bound, default 2.0.

⚠ Both alpha and the clamp move the same quantity -- how far the factor may
stray from 1 -- which is why alpha 0.25 was a thousand and alpha 0.5 was two
million on the same clamp. The clamp is the direct control and the exponent
should not be doing that job.

## 2026-09-07 late: AWQ was three bugs deep, and each was found by the layer under it

### 1. The factor was never applied to the activation

npuquant folds it into the weights (`row[i] /= t->kscale[i]`) and leaves
`t->kscale` for the caller to undo on the input -- *"on the board this is one
multiply a k before the pack"*. `kscale` appeared twelve times in npuquant.c
and zero times in npudev.c.

```
  off                              75.17
  AWQ 0.5, column means         10755.75
  AWQ 0.5, real |x| stats     65233808
```

The better the statistic, the worse the output: a factor nothing undoes, and a
truer mean |x_k| makes it more extreme. `charsiu_npu_matvec` applies it now;
`charsiu_npu_matvec_group` refuses a tensor carrying one, because that entry
exists to share ONE packed input across q, k and v.

### 2. With the multiply in, it still collapsed -- and the CPU said it was not mine

```
  board, factor now applied     AWQ 0.5  2155726     AWQ 0.25  1103.80
  host, CPU reference           AWQ off   114.22     AWQ 0.5   7252312.56
```

The CPU reference applies the factor itself and touches none of the npudev
work. Both collapsed, so the bug was in the quantiser and had been since it
was written -- which also made it debuggable in seconds on the host instead of
minutes over a UART.

### 3. The exponent was positive, and that is the whole method

`quant_rows` DIVIDES the weights by the factor, so W' = W/f. AWQ asks for
W' = W·s with s = (mean|x|)^alpha -- the columns that meet large activations
get MORE of the int4 grid. With a positive exponent this file shrank exactly
those columns. f = 1/s, so the exponent is **-alpha**.

```
  host, CPU reference, qwen3, 200 tokens
    off                            114.22
    AWQ 0.5  clamp 1.25             78.64
    AWQ 0.5  clamp 2.0              73.88     -35.3%
    AWQ 0.5  clamp 4.0              76.88
    AWQ 0.5  clamp 8.0              91.67
    AWQ 0.5  clamp 2.0, old sign   851.09     <- the control
```

**A U with a minimum, where before it was monotone worse.** That shape is the
result: a factor is supposed to have a best size. Before the flip every step
away from 1 pushed the wrong way, which is why narrowing the clamp to
[0.95, 1.05] had looked like a fix -- it was only reducing the error to nearly
nothing (111.08 against 114.22).

⚠ **I shipped a wrong default on the way.** After seeing the single point at
clamp 1.05 I set the default to 2.0; the sweep then showed 2.0 giving 851.09
under the old sign, seven times worse than off. It is right under the new sign
by luck, not by measurement, and the commit says so.

Against the gguf's own q4_0 on the same tokens (43.85), this takes charsiu's
int4 from +160% to +68% -- about 57% of the gap, from a factor applied
backwards.

⚠ Still w4a8 on the CPU reference. The board runs w4a16 and the multiply lives
in `charsiu_npu_matvec`; round 438 is the first time the fix meets hardware.

### 🏁 And it holds on the hardware

Round 438, board, qwen3, 600 tokens, the NPU's own w4a16 path:

```
  q4_0 baseline (a scale per 32 weights)   26.64
  charsiu int4, group 1024, AWQ off        49.89   +87.3%
  charsiu int4, group 1024, AWQ 0.5 c2     40.80   +53.2%   -18.2%
  charsiu int4, group 1024, AWQ 0.5 c4     43.74
  charsiu int4, AWQ 0.5 c2, OLD SIGN      441.34   the control
```

**39% of the quality gap, without one extra K slice.** The U survives (clamp 2
beats clamp 4) and the old-sign control stays broken, so the gain is the sign
fix and not something else that moved.

The board's -18.2% is smaller than the host's -35.3% and the two are not
comparable: the host measures w4a8 through the CPU reference on 200 tokens,
the board measures w4a16 through the NPU on 600. What transfers is the shape.

⚠ AWQ still needs a calibration pass, and that pass only records through
`CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1` -- `npu_calib_note` is called from
`npu_matvec` and nowhere else. And it makes decode slower, because a tensor
carrying a factor cannot share a packed input, so grouped q/k/v drop to single
calls. Neither is a reason not to have it; both are reasons it is not a
default.

### What the evening's quality line adds up to

```
                                   qwen3 600 tokens
  gguf q4_0 + fp32 (the baseline)        26.64
  charsiu int4, as it shipped this morning 49.89
  charsiu int4 + AWQ, tonight             40.80
```

Six weeks of "tokens identical" never compared charsiu to anything but
charsiu. The first external measurement said +87%, the cause was a group 32x
coarser than q4_0's, the fix for that was a method the tree had already
implemented and never got to work, and it was three bugs deep -- each of which
looked exactly like a null result.

## 2026-09-08 — w8a8 was never broken

Last night's quality line ended with a refusal: *"⛔ w8a8 IS BROKEN AND UNFIXED,
ppl 272369 (qwen3) / 218168 (gemma4), and PLAN.md still calls it the faster arm
for prefill."* That was the right thing to write down and the wrong thing to
conclude.

### The desk decided it, in two commands

The first question a broken path deserves is *which half*. `charsiu_ppl` on the
host runs the CPU reference against the same staged, requantised weights, so it
prices the quantiser with the device taken out of the loop. qwen3, 200 tokens:

```
  CHARSIU_NPU_W4V=1   int4   ppl 91.6559
  CHARSIU_NPU_W4V=0   int8   ppl 44.8050
```

int8 is not noise on the host. It is twice as good as int4, which is what eight
bits ought to be. So the fault is not in the weights — it is between them and
the hardware.

### And then the tree said what it was, twice, in comments it had already written

`tensor_grouped()` in npudev.c decides whether a K slice may carry one group's
scale. It wants four things, and the first is `g->w4`. The quantiser did not
know that: `npu_tensor_build` sets `t->kgroup = grp` for eight bits exactly as
for four, writes `scale[row * ngrp + group]`, and the int8 consumer reads
`scale[row]`. Every row takes some other row's scale.

There is already a guard for this. It was added when a partial last group did
the same thing to Qwen2.5-1.5B, whose k of 1536 and 8960 do not divide the 1024
slice — it "decoded fluent nonsense on the board while the same file was correct
on the CPU". The guard reads:

```c
if (t->kgroup && t->kgroup < t->k && (t->k % t->kgroup)) {
        whine(g, "a partial weight group would be read as one scale a row", ...);
```

⚠⚠ **It tests one of tensor_grouped's four clauses.** An int8 tensor at k 2048
with a group of 1024 has no remainder, so it walks straight past a guard written
against its exact failure. Every board round exports
`CHARSIU_NPU_W4_GROUP=1024` whatever the format, which is why the accidental
w8a8 arm hit it and no deliberate one ever had.

### The fix, and what it costs

```c
  npuquant.c   if (bits != 4) grp = k;      /* one scale a row, which is what
                                               the int8 consumer applies */
  npudev.c     if (t->kgroup && t->kgroup < t->k && !tensor_grouped(g, t))
                       refuse;              /* ask the predicate, not a clause */
```

Host, paired, one knob apart:

```
                     before     after
  int8 group 1024   44.8050   43.6595     the row number, digit for digit
  int4 group 1024   91.6559   91.6559     untouched, digit for digit
```

**At eight bits the coarser group is not a cost — it is a small gain.** A row's
spread fits in eight bits on its own, so the finer scales only add their own
rounding. Four bits is the opposite, 91.66 grouped against 114.22 a row, which
is the whole reason the group exists.

### On the board (round 139)

The corpus lives in /tmp and `usb_reset` reboots the card, so all four ppl arms
read a missing file and returned nothing — four blank lines, which is what four
equal arms also look like. The arms that need no corpus did run:

```
  the widened guard fired on 0 tensors, int4 and int8 alike
  w8a8 text: "storm had been coming for a long time, but the lighthouse is now
              only 100 miles from the coast."
  int8 decode 19.45 tok/s against int4's 31.53, same 13 token prompt
```

ppl 272369 does not write that sentence. Round 140 rebuilds the corpus first and
refuses to run the arms if it is not there.

### Where the four-bit error actually lives

`CHARSIU_NPU_W4_ONLY` narrows int4 to tensors whose name contains a substring.
Host, qwen3, 200 tokens, group 1024:

```
  everything int8                       43.66
  attention int4, the rest int8         54.76
  ffn int4, the rest int8               62.49
  everything int4                       91.66
```

⚠ This is an attribution, not a saving: `npu_q_packed()` is off whenever
W4_ONLY is set, so the four-bit tensors still occupy a byte a code. The file
says so itself — "a diagnostic for WHERE the error lives".

Two things it says. Attention at four bits costs less than the feed forward at
four bits, 11.10 against 18.83 — and the feed forward is the larger of the two,
so per byte they are close to the same. And **the two costs compound rather than
add**: 11.10 + 18.83 = 29.93, while both together cost 48.00. There is no cheap
win from splitting the format by tensor class; the error is spread about evenly
over the bytes, and it gets worse than proportionally when they are all four
bits at once.

### What this does to the format decision

PLAN.md's rule — int8 when the prompt is more than 3.1x the generated text —
was priced entirely in tok/s. The quality column, now that one exists, points
the same way and harder:

```
  int4, one scale a row                       114.22
  int4, group 1024                             91.66
  int4, group 1024 + AWQ 0.5 clamp 2           72.36
  int8, group 1024                             44.81
  int8, one scale a row                        43.66
```

Eight bits with no group and no AWQ beats four bits with both. So int8 is not
the format you accept for prefill speed and pay for in quality; it is better on
both counts for prompt-heavy work, and only decode speed argues against it.

⚠ AWQ and the finer group do stack, 73.88 and 91.66 separately against 72.36
together — but barely, and both are still a long way behind eight bits.

⚠ Still unmeasured: quality through the BATCHED prefill path. Every number here
is `charsiu_ppl`, one position at a time on purpose. The recommendation is about
prefill and the arm that ships it has not been scored.

### 🏁 Round 141: w8a8 lands 1.6% off llama.cpp's own q4_0

Round 140 answered the question on a corpus I had rebuilt wrong -- lifting the
two appends and dropping the `cp /tmp/ppl.txt /tmp/long.txt` between them, so
long.txt was one passage twice, 571 tokens instead of 600. Internally paired
and therefore sound:

```
                NPU      CPU reference
  int4      11.8818        12.3193
  int8       7.8720         7.7962
```

Both formats within about 1% of their own CPU reference, and int4's device arm
is the BETTER of its pair because the board runs w4a16 where the CPU reference
runs w4a8. That is the localisation closed from the other end: the device is
faithful to the weights it is given, for both formats.

Round 141 on the right corpus. ⚠ **Arm Z is a fingerprint, not a baseline** --
the gguf's own q4_0 through charsiu's CPU loop touches nothing this week
changed and is deterministic, so it says whether the corpus is last night's:

```
  Z  q4_0 gguf, the fingerprint      26.6416     recorded 26.6413
  A  int4 NPU                        49.8930     recorded 49.8930
  B  int8 NPU                        27.0668     was 272369
  C  int4 NPU + AWQ clamp 2          40.8296     recorded 40.8006
  E  int8 NPU + AWQ clamp 2        2162.7320     never asked before
```

**27.07 against 26.64.** The quality gap that opened last night at +87% closes
to 1.6% by fixing a scale layout, with no calibration, no finer group and
nothing new on the hardware.

⚠ Arm C is 0.07% off its record and that is the calibration, not a drift: this
round's calib pass ran with `CHARSIU_NPU_W4V` unset, so the activations were
recorded through an int8 model where round 138 recorded them through an int4
one. Small, and in the direction that says so.

### ⛔ And arm E is the same bug one layer down

int8 with AWQ is 2162.73 where int8 alone is 27.07. On the host it is 44.76
against 43.66 -- 2.5% worse, not eighty times. The difference is where the
factor's cancelling multiply lives.

`W' = W/f` happens in the quantiser for any width. `x' = x*f` happens in
`charsiu_npu_matvec`, which scales the activation before packing it -- and that
is the w4a16 path, where the activation is still a float when it gets there.
The int8 path packs `a->q1`, one absmax quantisation of the whole vector, and
has nowhere to put a per column factor. So at eight bits on the board the
divide happens and the multiply does not.

⚠⚠ **That is AWQ bug #1 again, in the branch nobody had run.** The first one
was "the factor was never applied to the activation -- kscale appears twelve
times in npuquant.c and zero times in npudev.c". This is the same sentence with
"on the int8 path" appended, and it was reachable the whole time.

Declined rather than fixed, because fixing it buys nothing: the host applies
the multiply correctly at eight bits and AWQ still makes int8 worse there.
Eight bits does not have the dynamic-range problem the method exists to solve.
`npu_tensor_build` now zeroes alpha for `bits != 4` and says so once, naming
the width. Paired on the host: int8 with AWQ 44.7554 -> 43.6595, which is the
AWQ-off number digit for digit, and int4 with AWQ 72.3630 -> 72.3630.

### Where this leaves the format choice

```
                     ppl      bytes a weight     decode, 13 tok prompt
  int4 w4a16       49.89          0.5            31.53 tok/s
  int8 w8a8        27.07          1.0            19.45 tok/s
  q4_0 reference   26.64          0.5625         (not on the NPU)
```

int4 stays the default: decode is memory bound, int8 moves twice the bytes,
and chat is a short prompt and a long answer. But PLAN.md's rule already sends
prompt-heavy work to int8 on speed grounds, and for that work it is now better
answers as well -- not a trade.

⚠ The decode figures are from round 139's 13-token text arms and are not
scoreboard numbers. The scoreboard runs 110-token prompts and int8 has never
been through it.

### The `--batch` number is also the instrument for the K-slice fault

`llama_auto_kmax` stops its candidate list at 2048 and says why:

```
  slice 2816   WRONG   (Qwen2.5 at KMAX 3072, surf 88)
  slice 3072   right   (gemma-3-1b at KMAX 3072, surf 96)
  slice 4096   WRONG   (both, surf 128)
```

"Something else is wrong above 2048 and it has not been found." Every probe
that has ever asked this question asked it as **text identical or not** -- a
bit. A bit cannot say whether a slice is slightly wrong or catastrophically
wrong, cannot rank two widths that both fail, and cannot see a width that is
wrong by less than one token choice.

`charsiu_ppl --batch` is the continuous version of the same question, on the
same path, and it costs one run a width. That is the next probe: sweep
`CHARSIU_NPU_KMAX` over 1024 / 2048 / 3072 / 4096 with `--batch` and read the
SHAPE of the error, against the token loop at the same width as the control.

⚠ And note what the widening note already establishes and this does not
change: the fault is in the batched path, not the quantiser -- at 1024, 2048
and 4096 the three models whose every K misses every width came back byte
identical, so the weights are the same bytes across the sweep and only the
slicing moves.

⚠⚠ It also closes an idea worth not having twice. int8 now sets `grp = k`, so
`tensor_grouped()` is false for it and the "the K slice IS the quantisation
group" coupling does not bind -- which looks like int8 being free to take a
wider KMAX and halve its task count. It is not free: the constraint that stops
2048 is the batched path itself, and it has nothing to do with grouping.

### Why int8 is the answer here, as an argument rather than an observation

The four-bit group is bought in READ, because the K slice IS the group: cut the
group by a factor and every tensor is dispatched that many more times, and each
dispatch reads its own accumulator back. Put the measured quality against that
cost, on qwen3, 600 tokens:

```
                              bytes a weight   read     ppl
  int4, group 1024 (ships)         0.502        1x     49.89
  int4, group 256                  0.508        4x     38.97
  int4, group 128                  0.516        8x     36.33
  q4_0, group 32 (llama.cpp)       0.5625      32x     26.64
  int8, one scale a row            1.000        1x     27.07
```

⚠ **The int8 row is 1x and that is the whole point.** Its group is the row, so
`tensor_grouped()` is false and KMAX is not tied to it at all -- eight bits
steps outside the coupling rather than paying it.

And the read budget says the other rows are not reachable. The batched prefill
reads at **5.2 GB/s against the 11.9 the board's eight threads can do** -- 2.3x
of headroom. Group 256 wants 4x, group 128 wants 8x, and q4_0's group of 32
wants 32x. **Only the 1x column is affordable, and int8 is the only row in it
with a usable number.**

So the format story closes: two extra bits buy llama.cpp's own q4_0 quality
(27.07 against 26.64) at ONE times the read, where buying the same quality with
a finer four-bit group would cost thirty-two. The price is the weight bytes --
twice as many, and decode is memory bound, which is why int4 stays the default
for chat and int8 is the prompt-heavy arm PLAN.md already recommends.

⚠ What this does NOT say: that a finer group is impossible. It says it is
read-bound on THIS board with THIS accumulator. The note over `tensor_grouped`
in npudev.c has the register-level argument for why one dispatch cannot carry
more than one group, and that is the thing to reopen if the surface ceiling
ever lifts.

### ⛔ Round 143: int8 is slower on the PROMPT too, and my own correction was wrong

The scoreboard, both formats, best of 6, one session, same prompts:

```
                 decode tok/s                     TTFT ms
              int4    int8   int8/int4     int4   int8    int8
  Qwen3      26.30   17.16     65.2%        594    751   +26.4%
  TinyLLAMA  22.89   12.88     56.3%        869    971   +11.7%
  Phi3        7.06    3.85     54.5%       2818   3091    +9.7%
  Gemma4      9.42    5.55     58.9%       2190   2512   +14.7%
```

The int4 arm is the carried control and it holds: 26.30 / 22.89 / 7.06 / 9.42
against last night's 26.59 / 22.85 / 7.02 / 9.26, all four still above the
vendor, all within the board's ~3% drift.

**int8 is behind on both axes.** So `PLAN.md`'s crossover -- int8 above a
prompt 3.1x the generated text -- has no crossover left to compute. Its prefill
pair (19.24 int4, 26.60 int8) was measured when int8 batched and int4 did not,
which the same section says two paragraphs down; the 3.3x batched w4a16 prefill
landed on 08-27 and int4 went past.

⚠⚠ **AND I WROTE THE WRONG THING TWICE THIS MORNING, THE SECOND TIME WHILE
CORRECTING THE FIRST.** The README got "better answers AND a faster prompt
rather than a trade", and PLAN.md got "prompt-heavy work AND better answers,
against decode speed" -- both of them repeating `PLAN.md`'s prefill claim as an
input while the whole point of the edit was that its quality claims had never
been checked. **I audited one column of that section and inherited the other.**
A stale measurement is load-bearing until something weighs it, and being in the
middle of fixing a document is not the same as having weighed it.

What survives is the part that was measured today: int8 buys 49.89 -> 27.07,
llama.cpp's own q4_0 to within 1.6%, for about a third of decode and about 15%
of TTFT. A quality option, not a speed one.

⚠ Phi-3.5 at eight bits fits: peak 4299 -> 6025 MB. On a smaller board it
would not.

### 🏁 Round 144: int8 generalises, on three models

```
                q4_0      int4      int8     int4 vs q4_0   int8 vs q4_0
  qwen3        26.64     49.89     27.07        +87.3%          +1.6%
  gemma4       38.41     53.47     39.83        +39.2%          +3.7%
  tinyllama    18.89     22.78     19.12        +20.6%          +1.2%
```

The q4_0 arm runs first on each model as a corpus fingerprint. gemma4's w8a8,
written down on 09-07 as 218168, is 39.83.

⚠ **int4's damage is very model dependent and int8's closeness is not.** qwen3
loses 87% at four bits and tinyllama 21%, a factor of four between them; both
land within 4% of the reference at eight. So "int4 costs about 50%" was never
a property of the format, and a single-model reading of it would have been
wrong in either direction.

### 🏁 Round 145: the batched path, scored for the first time in six weeks

```
                    token loop     --batch      difference
  int4               49.8930      49.8877       -0.011%
  int8               27.0668      27.8654       +2.95%
```

**The batched prefill costs int4 nothing.** That is worth stating plainly: the
path that carries every prompt this runtime serves has been unmeasured since it
shipped, and for the default format it is free.

**It costs int8 2.95%,** which is the first sign that eight bits is not just
"int4 with more bits" as far as the batched path is concerned. The one segment
the formats do not share is the per channel tail multiply -- grouped int4 skips
it because its scale rides in with the K slice, int8 always takes it.

⚠⚠ **AND THAT NUMBER IS FOR A WIDTH THE PRODUCT DOES NOT USE.** `--batch` read
`llama_prefill_chunk_cap()`, which is the surface CEILING -- 160 on qwen3 --
while `charsiu_run`'s default chunk is 80. So the first number ever produced
for "the quality of the batched path" was the quality of a path nobody runs.
A probe that reads the cap has forgotten the product has a default of its own.
Fixed: `--batch` takes `CHARSIU_PREFILL_CHUNK` or 80, the same rule, and prints
both the chunk and the ceiling so a pasted line cannot be read as the other.

Round 146 sweeps m over 8 / 40 / 80 / 160 to say whether 2.95% is a property of
the width or of the code path, and reads the prefill table with `scale` named
for the first time.

### 🏁 Round 146: int8's prefill loss is the PACK, and it is not the scale

Part 1 -- the batched path's 2.95% on int8 does not move with the width:

```
  int8, token loop                 27.0668
  int8, --batch m=8                27.8654
  int8, --batch m=40               27.8654
  int8, --batch m=80  (shipped)    27.8654
  int8, --batch m=160              27.8654
  int4, --batch m=80               49.8877
```

Identical to four decimals across a twenty-fold range of m. So it is a fixed
property of the code path and not of the batch, which rules out anything that
scales with the number of rows -- a shared scale, an accumulator that grows, a
pooling grain.

Part 2 -- the prefill table with `scale` named for the first time, qwen3,
90 rows, ms a row:

```
          pack   submit  fence   read   scale  prep  unacc
  int4    0.86    0.08   1.29   1.30    0.19   0.12   0.03    5.33 total
  int8    2.66    0.09   0.93   0.97    0.25   0.10   0.03    6.41
```

⚠⚠ **int8's fence and read are BOTH SMALLER.** Its hardware path is genuinely
faster, which is what PLAN.md claimed all along and what round 143's TTFT made
look false. **pack alone is 3.1x wider and eats the advantage and 1.08 ms a row
more.** The tail scale everybody suspected -- this author loudest -- is 0.06 of
the gap, and naming it is how that was settled in one round instead of three.

The cause, once they are side by side: int4 packs through `pack_f16_pooled`,
pooled across groups; int8 ran two scalar passes over the slice, one thread, no
NEON. Both vectorise; the rows are independent because each carries its own d1.

Fixed, with `CHARSIU_NPU_QPACK_PLAIN=1` as the same-binary control, and proved
identical by `tools/npu_qpack_test` rather than by generated text: 56000 rows
over fourteen widths, on data chosen to break it -- codes exactly halfway
between two integers, values past the clamp, an all-zero row, near-denormals,
every remainder mod 16 and mod 4. **0 byte mismatches, 0 d1 mismatches.**

🔑 **And it ran on the desk, because this host is aarch64.** Every
`__ARM_NEON` block here compiles AND executes. A vector rewrite does not need
a board round to be proved identical, and I had assumed the opposite.

### 🏁 Round 147: decode is 88 to 91% matmul, and the rest is a per call floor

qwen3, 30.7 ms a token; tinyllama, 41.3.

```
  qwen3          ms     %        tinyllama        ms     %
  gate + up     7.29  23.8       gate + up     15.89  38.5
  q k v         6.06  19.8       down           9.96  24.1
  down          5.05  16.5       q k v          5.66  13.7
  output head   5.00  16.3       o proj         3.95   9.6
  o proj        3.54  11.5       output head    1.93   4.7
  -- matmul    26.94  87.8       -- matmul     37.39  90.6
  attention     2.35   7.7       attention      1.99   4.8
  silu * up     0.72   2.4       silu * up      1.04   2.5
```

**Everything that is not a matmul is 12% and 9%.** The elementwise work is
done; whatever is left in decode is in the matmuls.

And the matmuls have a fingerprint. Effective bandwidth by class, qwen3:

```
  o proj        1.05 MB a call    8.29 GB/s
  down          1.57              8.72
  q k v         2.10              9.69
  gate + up     3.15             12.08
  output head  77.79             15.56    <- one call, nothing to amortise
```

Monotone in the bytes ONE call moves, which is what a fixed per-call cost looks
like and very little else does. Fitting `t = a + b*MB` through the smallest and
largest gives **b = 15.7 GB/s marginal** and **a = 60 us a call**. At 113 calls
a token that is 6.8 ms of 30.7 -- **22%** -- and removing it entirely would be
32.6 -> 44.0 tok/s.

⚠ Two corrections this kills. The marginal bandwidth is 15.7 GB/s, ABOVE the
11.9 the CPU threads reach, so using the CPU membw ladder as the roof for NPU
weight reads was wrong. And the floor is 60 us a call now, not the 130 us the
August note quotes -- attach-once and the halved ioctls did land.

⚠⚠ A line through two points is not a law. Round 148 turns it into a
prediction instead: `CHARSIU_NPU_NOGROUP=1` stops q/k/v and gate/up sharing a
submit, taking the count from 113 to 197, which at 60 us is +5.0 ms a token.
If the table moves by much less, the floor is smaller than the fit says.

### 🏁 Round 148: the prediction held, and it killed the theory behind it

`CHARSIU_NPU_NOGROUP=1` takes the call count from 113 to 197. The fit said
+5.0 ms a token. The board said **+10.6** (30.8 -> 41.4), which is 126 us a
call, twice the 60 us two points had implied and close to the 130 us the August
note quotes.

```
                 group ON   group OFF
  ms a token        30.8       41.4
  q k v             6.07      11.26
  gate + up         7.30      11.78
```

⚠ Not all of that is dispatch: without grouping, q/k/v pack the same activation
three times instead of once. The arm confirms a floor exists and is large; it
does not measure it cleanly.

⚠⚠ **AND THE NPU'S OWN SPLIT SAYS THE FLOOR IS NOT WHERE I WAS LOOKING.**

```
                        group ON   group OFF
  hardware path (ms)       1699       2267
    of it, the fence       1343       1939
    of it, submitting       133        152
    of it, summing          140         92
  end to end               1832       2514
  neither hardware nor pack   7         70
  weights                11.58 GB/s  8.68 GB/s
  a submit's wall clock    114 us     135 us
```

**End to end is 1832 against 1699 in the hardware path: 7.3% of a decode
happens outside it, and 7 ms of that is neither hardware nor packing.** Every
ioctl, malloc and cache clean on the CPU side, all of it together, is noise. I
had spent an hour reading submit paths and counting syscalls, and the answer
was that the CPU side is already finished.

The floor is INSIDE the fence -- 1343 of 1699 ms -- and the report's own
parenthesis says what else lives there: *"waiting for the fence (the invalidate
is in there)"*. So that one number is the hardware computing, plus the wakeup
from a blocking ioctl, plus the output BO invalidate. 11.58 GB/s inside the
hardware path against a 15.7 GB/s marginal rate puts about 26% of it in a fixed
per-submit cost, and two of those three components are not the hardware.

🔑 **`CHARSIU_NPU_SPIN_US` splits it, and it has existed unpriced the whole
time.** device.c: *"Off unless asked, until phase 21 has priced it against the
arm that disables CPU_SLEEP outright."* Phase 21 priced the QoS hold and never
came back. Polling with `timeout_ns = 1` before falling back to the blocking
wait removes the wakeup and leaves the invalidate and the hardware, so the
sweep attributes the fence rather than arguing about it. Round 149.

## 2026-09-08 — the `=0` audit, because two catches are not two bugs

`CHARSIU_NPU=0` opened the NPU and `CHARSIU_NPU_ONEDEV=0` closed the second
core. Both were found by accident, one night apart, and both were written up
as incidents. They are not incidents. They are one **shape**, and the only
honest question is how much of this notebook was measured through it.

**This is instrument contamination, not a code bug.** It does not crash, does
not warn, and produces a result that reads exactly like a real one. It belongs
with the four rounds of npu_slice_test and the withdrawn map that stood for
thirteen days: the probe was wrong and nothing said so.

### The window

```
  CHARSIU_NPU, CHARSIU_NPU_VERBOSE, CHARSIU_NPU_A16   introduced 08-22
  CHARSIU_NPU_ONEDEV                                             08-23
  CHARSIU_SPM_DEBUG                                              08-26
  CHARSIU_NO_BATCH_PREFILL, CHARSIU_BATCH_SWEEP                  08-27
  CHARSIU_NO_WCACHE                                              08-28
  CHARSIU_NPU_W4_ANYM, CHARSIU_BATCH_FORCE                       08-30
```

Forty-six switches read as existence tests until 2026-09-07 20:02 (34 of them),
2026-09-08 08:06 (8) and 09:43 (ONEDEV). The exposure runs from 08-22, so
**seventeen days**.

### What was actually run through it

Every `XXX=0` in the surviving round scripts, checked against the commit that
gave that switch a value rule:

```
  switch                    used =0 in      switch became a value rule   verdict
  CHARSIU_ATTN_POOL         m116 09-07 19:22    09-07 20:02   ⛔ CONTAMINATED
  CHARSIU_NPU_KFIT          m121 20:16, m122 20:29  09-07 20:02   ok, after
  CHARSIU_NPU_NOGROUP       m148 09-08 09:32    09-07 20:02       ok, after
  CHARSIU_NPU_ONEDEV        m150 09-08 09:46    09-08 09:43       ok, by 3 min
  CHARSIU_NPU_POOL_READ     m7 09-06, m87 09-07  09-05 11:11      ok, after
  CHARSIU_ROW_POOL          m2 m3 m13 m14 m15 m100  value from birth 09-06 07:28
  CHARSIU_NPU_PACK_POOL     m16 09-06           value from birth   ok
  CHARSIU_NPU_PAIR          m96 09-07 10:57     value from birth   ok
  CHARSIU_ROPE_INPLACE      m99 09-07 17:06     value from birth   ok
```

### ⛔ Round 116 is downgraded to UNVERIFIED

m116 ran four arms looking for a text nondeterminism. Its third,
`gemma4 + no attn pool`, set `CHARSIU_ATTN_POOL=0` **forty minutes before that
switch stopped being an existence test**. It did not turn the pool off. It
turned it on, which is where it already was.

So that arm is the same arm as the one above it, and "the pool is not the
cause" was never tested by that round. **The conclusion happens to have
survived** -- the nondeterminism turned out to be `CHARSIU_STAGES` writing
milliseconds onto stdout, which the next round found -- but it survived for a
reason m116 did not supply. A right answer reached through a dead knob is not
evidence, and if the pool HAD been the cause this round would have said it was
not.

### ⚠⚠ And two weeks cannot be audited at all

The oldest round script that still exists is 09-04 22:36. The switches go back
to 08-22. **Every round between 08-22 and 09-04 is outside this audit** -- not
cleared, not condemned, unexaminable. Anything from that window that rests on
a `=0` control arm should be re-run before it is leaned on. The scripts are
gone; the status files are not, so a specific claim can still be checked by
hand if someone needs it.

### ⛔ The audit was incomplete, and the five it missed are worse

That sentence was written as "charsiu_env_flag is now the only way a boolean
switch is read in src/". It was wrong within the hour. The first sweep grepped
`if (getenv("X"))`, and five switches use a different spelling:

```c
  v = getenv("CHARSIU_EXACT_ATTN")    == NULL && !cpu_plain();
  v = getenv("CHARSIU_EXACT_SILU")    == NULL && !cpu_plain();
  v = getenv("CHARSIU_EXACT_GELU")    == NULL && !cpu_plain();
  v = getenv("CHARSIU_EXACT_SOFTMAX") == NULL && !cpu_plain();
  g->nofini = getenv("CHARSIU_NPU_FINI") == NULL;
```

Unset gives the fast path; **any value, "0" included, gives the exact one**.
So `CHARSIU_EXACT_SILU=0`, written to mean "not exact", selects exact -- the
mirror image of `CHARSIU_NPU=0` opening the NPU, and these four control
NUMERICAL PRECISION. Measured on the host after the fix: `=1` gives 32.44
tok/s and `=0` gives 37.99, so the wrong branch is a **17%** difference
silently attached to the arm that asked for the other one.

🔑 **Nothing was contaminated, and the reason is luck.** Every use of these in
the surviving rounds and in tests/ is `=1` -- nine of `CHARSIU_EXACT_SOFTMAX=1`
in board_verify.sh, one of `CHARSIU_EXACT_GELU=1`, none of `=0` anywhere. `=1`
selected exact before the fix and selects exact after it, so every arm that
ever ran got what it asked for. The bug was live for the whole window and
nobody happened to spell it the broken way.

⚠⚠ **Two incomplete audits in one day is the actual lesson.** The first missed
these because it grepped a spelling rather than a semantics; this one found
them by asking "what reads a variable and never looks at its value", which is
the property, not the syntax. A census that greps for a shape will keep missing
whatever is written differently.

### What stops it happening again

`charsiu_env_flag` is the only way a boolean switch is read in src/, and that
claim is now checked by property rather than by pattern. The remaining
`getenv()` calls there take a value (`atoi`, `atol`, `atof`), a name (a file, a
substring, a CPU list) or a `strcmp` -- checked one by one. ⚠ tools/ still has thirty-odd, and they are probes: a probe that
inverts its own arm is exactly the failure this section is about.

### The vendor's quality cell: reconnaissance, and it is not encrypted

The README's three-row table has one empty cell -- the vendor's perplexity --
and it is the cell the other two rows are read against. `rkllm_regcmd.py` says
why it cannot be filled from the register streams: *"Address registers in a
static file are unpatched placeholders and read 0"*, so the programs are
readable and the data they point at is not located by them.

The weights are still in the file. Twenty minutes on
`Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm`, 1240 MB:

**It is not encrypted.** Entropy runs 5.2 to 6.4 bits a byte, never near 8, and
the nibble histogram over a megabyte at the midpoint is the signature:

```
   0:17.8  1:14.3  2:9.6  3:5.2  4:2.4  5:0.9  6:0.3  7:0.1
   8: 0.1  9: 0.3 10:0.8 11:2.3 12:5.0 13:9.2 14:14.0 15:17.5
```

A bell centred on zero in four-bit two's complement, symmetric to within 0.3
points across the sign (0 against 15, 1 against 14). Gaussian weights, rounded
to four bits, +/-7 the rarest. Encrypted bytes are flat.

**And the file has two halves.** A 200-step profile of entropy and nibble
symmetry:

```
   0 -- 520 MB    not bimodal, entropy ~6.0
 520 -- 1240 MB   int4 signature throughout, 713 MB of it
```

Against the model: 128256 x 2048 token embeddings at fp16 are **525 MB**, which
is the first half; 1236 M weights at four bits are **618 MB**, which is most of
the second.

⚠⚠ **AND THE 95 MB LEFT OVER CONTRADICTS SOMETHING THIS TREE BELIEVES.** The
memory of the vendor's format says it carries *one scale and one zero point per
row* -- for this model that is 505088 rows, about 2 MB, and it does not fit.
A scale and zero point every 64 weights costs 77 MB and lands at 695 against
713 measured; every 32 costs 154 and lands at 772.

So the file size looked like it said the vendor quantises in groups of tens of
weights rather than one group a row.

⛔ **AND THAT WAS OVERREADING A RESIDUAL, WITHIN THE HOUR IT WAS WRITTEN.** The
95 MB came from subtracting a 618 MB estimate from a 713 MB profile band whose
edges are good to about 3%, which is +/-20 MB on its own. Walking the regions
properly (below) closes the file to 1215 of 1240 MB, leaving **25 MB** for
every scale in the model -- and 25 MB cannot tell a group of 32 from one scale
a row, because one scale a row is 2 MB and 2 MB fits inside the error bar.

The honest statement is that **the scale format is unknown and this file's size
does not constrain it.** The written finding it appeared to contradict is
neither confirmed nor refuted.

**What it would take, and a cheaper intermediate.** Filling the cell properly
needs the tensor boundaries, the scale format, the weight layout, then a
dequantise and a gguf write. That is days. But the question underneath it --
*is the vendor's four-bit quantisation better than ours, and by how much* --
can be answered by decoding **one tensor**: dequantise a single gate
projection, compare its error against the same gguf tensor, and put that beside
charsiu's own. No tokenizer, no forward pass, no board.

### 🏁 Round 150: the floor is the hardware, and the counters are why that is knowable

```
                        two cores   one core
  ms a token               30.8       39.6
  hardware path (ms)       1710       2310
  weights                 11.50      8.52 GB/s
  a submit's wall clock    114 us     306 us
    of it, the fence       1352       2002 ms
    of it, submitting       138         93 ms
```

And the spin counters, added this morning for exactly this reading:

```
  charsiu spin: 13899 polls, 13827 won without blocking (99%, mean 114 us),
                72 fell back (mean 312 us of polling before they did)
```

⚠⚠ **NINETY-NINE PERCENT OF THE POLLS WIN, AND THE MEAN IS 114 us.** The
blocking wait is essentially never reached once the poll is on, so the wakeup
that round 149's 2.7% was attributed to is not a large hidden cost -- it is
about 2%, and what is left of the 114 us is the hardware computing. One core
with the poll on is 2268 ms against 2310 blocking: 1.8%.

**So the per-call floor is not ours.** Round 148 showed the CPU side is 7.3% of
a decode. Round 149 and this one show the wakeup inside the fence is ~2%. What
remains is a job's start on the device and the output invalidate, and only the
second of those is in software.

🔑 **Without the counters this round reads the opposite way.** A 1.8%
improvement from a 300 us poll is exactly what "the poll never fired" looks
like, and the next move from that reading -- raise the spin, chase the wakeup --
would have been wrong. `99%, mean 114 us` closes it in one line. That is what
the instrument was for.

⚠ What is still not separated: the invalidate. The winning poll's own ioctl
does the wait AND the invalidate, so the 114 us mean contains both. A second
PREP on an already-finished BO would be the invalidate alone -- one probe,
still unwritten.

### The .rkllm's three regions, settled by a test that discriminates

The entropy profile could not tell packed int4 from int8 -- both put bytes near
zero. One statistic does: **in packed int4 the low and high nibble hold two
independent weights, so their histograms are the SAME; in int8 the low nibble
is mantissa noise and the high nibble carries sign and magnitude.** Total
variation between the two halves, over a megabyte:

```
   520 -- 990 MB     0.26%   identical halves      -> packed int4
   990 -- 1240 MB   52.19%   low flat, high a bell -> int8
     0 -- 520 MB    55.39%   fp16 exponent pattern -> fp16
```

The low nibble at 1100 MB is 6.2 to 6.3% across all sixteen values -- flat to
a tenth of a point -- and the high nibble is 21.7 / 15.4 / 8.1 / 3.4 / ...
symmetric about the sign. That is int8, not a coincidence.

```
    0 -- 520 MB    fp16   token embedding    128256 x 2048 x 2  =  501 MB
  520 -- 990 MB    int4   16 layers          973 M x 0.5        =  464 MB
  990 -- 1240 MB   int8   output head        263 M x 1          =  250 MB
                                                        total     1215 of 1240
```

🏁 **The vendor keeps its output head at eight bits.** charsiu quantises it to
four along with everything else.

### And charsiu can already do that, so it was measured rather than assumed

`CHARSIU_NPU_W4_ONLY=blk` gives four bits to every tensor whose name contains
`blk` -- the layers -- and leaves `output.weight` at eight, which is the
vendor's shape. Host, qwen3, 200 tokens:

```
  everything int4                    91.66
  layers int4, head int8 (vendor)    86.83     -5.3%
  everything int8                    43.66
```

**Only 5.3%.** The head is 26% of qwen3's weight bytes and 5% of the damage, so
the vendor's eight-bit head is not where a quality difference would come from
if it has one. It is consistent with this morning's W4_ONLY sweep: the four-bit
error is spread roughly with the bytes, and no single tensor class carries it.

⚠ So the interesting difference, if there is one, is the layers -- and those
are int4 on both sides. Which leaves the group size, and the file does not say
what it is.

### 🏁 Round 151: the 71 us floor, fully attributed, and decode's remaining room

`npu_prep_cost` times prep and fini on a buffer that was never submitted, so
there is no fence in the number:

```
     bytes   prep us   fini us    GB/s
      4096      1.16      1.00     3.5
      8192      1.46      1.33     5.6
     32768      3.26      3.12    10.1
     65536      5.60      5.46    11.7
    262144     19.72     19.55    13.3
   1048576     76.47     76.06    13.7
   4194304    302.04    301.39    13.9
```

**A syscall with no work costs 0.87 us and the cache walk runs at 13.9 GB/s.**
Both halves of a call's maintenance follow: two output preps at charsiu's 32 KB
out_stride are 6.5 us, two input finis at ~8 KB are 2.7, so **9.2 us**.

With round 149's spin sweep and round 150's counters, the 71 us fixed term is
now accounted for end to end:

```
  the device starting a job    ~46 us   65%    not ours
  the blocking wakeup           ~11 us   15%    CHARSIU_NPU_SPIN_US removes it
  cache maintenance             9.2 us   13%    out_stride is 32 KB for a 4 KB result
  ioctl syscalls                 ~5 us    7%
```

⚠⚠ **So decode's remaining room is about 6% of a token, not the 22% the first
fit implied.** 26% of a token is the per-call floor, and 65% of that floor is
the device starting a job -- which no amount of userspace work removes.

The two pieces that are ours were both confirmed the same round.
`CHARSIU_NPU_NMAX=2048` drops out_stride from 32 KB to 8 and moves the hardware
path 1709 -> 1663 ms, **2.7%**, which is the right size for 2 x (3.26 - 1.46)
us a call. ⚠ NMAX is the wrong lever for it though: it also re-slices every
tensor wider than 2048, and qwen3's 151936-wide head would take 75 slices
instead of 19. The right change is `out_stride` per entry -- `min(nmax, t->n)`
-- which buys the same cache and re-slices nothing. Priced, not yet written.

🔑 Three rounds took a "22% of a token is overhead" reading down to 6%.

⛔ **AND THEN I WROTE THAT THE LINE WAS FINISHED, ON A NUMBER I HAD
SUBTRACTED.** The 46 us is what was left after removing three measured terms
from 71. Nothing measured it. The paragraph above it on this same page had just
retracted a residual for exactly that reason, and one screen later I used
another one to close a line of work.

**A residual attributed to hardware is the shape of the wall.** Every
explanation of that one was a property of the CBUF sequencer -- for months,
with data behind each -- until it turned out to be `PC_TASK_CON`'s field layout
in a header copied from RK3588. What made it a wall was not the difficulty. It
was that "the hardware does this" was never itself a measurement.

So the 46 us gets measured. `tools/npu_job_cost` submits a matmul small enough
that the arithmetic is nothing and times four ways of getting two jobs onto the
device:

```
  A  one job, one ioctl                  the floor itself
  B  TWO jobs in ONE ioctl, one fd       never run
  C  two jobs, two ioctls, one fd        an ioctl's own share
  D  two jobs, two ioctls, two fds       what charsiu does today
```

⚠⚠ **And arm B exists because of something in charsiu's own submit loop.**
`charsiu_npu_matvec_group` runs `for (d = 0; d < g->ndev; d++)
charsiu_submit_jobs(g->dev[d], &jl, 1)` -- one ioctl per core, in sequence, so
the second core starts a whole syscall after the first. `charsiu_submit_jobs`
has taken a job LIST since it was written, and device.c says why that matters
in a comment nobody acted on: *"Jobs are what the scheduler can hand to
different cores."*

Round 152. If B beats D, the second core has been starting late for the life of
this runtime, and 46 us was never the device's.

### ⚠⚠ And the tree already had a three-term model that my fit erased

npudev.c, from a round that fitted TinyLLAMA's five decode stages against the
geometry this file cuts:

```
     q k v     1.311 MB  3 tasks   measured  392.7   fit  383.5
     o         1.049     1         measured  276.4   fit  280.9
     gate+up   5.767     2         measured  845.0   fit  836.9
     down      3.146     3         measured  573.6   fit  585.4
     head     16.777     4         measured 2100.0   fit 2122.1

   us a call = 128.7 + 36.8 * tasks + 110.0 * MB   (busier core)
```

Inside 2.4% at all five, and it agrees with a 2026-08-15 sweep of synthetic
matmuls at 32 chained tasks that never saw this model: 26.3 us a task, 172 a
submit, 84.3 a megabyte. Same three terms, same order, different shapes,
different day.

**My fit today has no task term.** Two points, then five, then `a = 71 us +
63.4 us a MB` -- and it predicted the output head to 3% from the small tensors,
which is why I trusted it. But task count and megabytes are collinear across
those five classes, so a two-term fit puts the task cost inside the MB slope
and reports a floor that is only the part left over. That is how 46 us appeared
and then got a name.

If a task really costs tens of microseconds, decode's budget is a different
shape. qwen3 presents **271 tasks a token** across 113 calls:

```
  task marginal   36.8 us  ->  9.97 ms  =  32% of a 30.7 ms token
                  26.3 us  ->  7.13 ms  =  23%
                  10.0 us  ->  2.71 ms  =   9%
```

🔑 **AND THE TASK COUNT HAS A LEVER THAT int4 DOES NOT HAVE.** Tasks are K
slices, and `ceil(K / KMAX)` is the count. int4 cannot raise KMAX -- the K
slice IS the quantisation group, and `tensor_grouped()` requires
`kgroup == kmax`. **int8's group is the whole row, so it is free of that
coupling entirely**:

```
  KMAX 1024   down takes 3 slices   271 tasks a token
  KMAX 2048                 2       243        -28
  KMAX 4096                 1       215        -56
```

⚠ This may be why int8's fence was already smaller in round 146 (0.93 against
int4's 1.29 ms a row) while its pack was worse -- fewer tasks for the same
bytes. Nobody has swept KMAX on int8, because until this morning int8 was
believed to emit noise.

Round 152's task sweep -- 1 to 16 chained in one job, on a matmul with no
arithmetic in it -- prices the term directly instead of fitting it.

### 🏁 Round 152: the 46 us measured, and both of my levers are dead

```
  one job, tasks chained inside it
     tasks   us a job    us a task
         1      21.66        21.66
         2      25.93        12.96
         4      35.83         8.96
         8      55.33         6.92
        16     216.16        13.51

  two jobs, four ways
    A  one job, one ioctl                72.59 us
    B  two jobs, ONE ioctl, one fd      131.12 us     ~ 2 x A
    C  two jobs, two ioctls, one fd     131.93 us     ~ B
    D  two jobs, two ioctls, two fds     57.84 us     <- what charsiu does
```

**B and C are both about twice A, and D is the fastest of the four.** Two jobs
submitted through one fd RUN SERIALLY, whether they arrive in one ioctl or two.
Two fds run in parallel. So the `for (d) submit(dev[d])` loop that looked like
a missed opportunity is the only arrangement of the four that uses both cores,
and the comment about the scheduler handing jobs to different cores does not
mean it will do so within one fd.

⛔ **The fd merge is dead, and it is dead by measurement rather than by
argument.** So is the reasoning that led to it: charsiu's second core is not
starting a syscall late, because the alternative starts it on the same core.

**And the task term is not 36.8 us.** Fitting 1 to 8 tasks:

```
  a job costs 16.85 us + 4.81 us a task
```

Eight times smaller than the three-term model in npudev.c, which got 36.8 from
five decode stages on TinyLLAMA. Both cannot be right about the same hardware.
What the stage fit actually has is five points where tasks and megabytes move
together, so its task coefficient is carrying weight-fetch time that this probe
-- where the matmul is 64x32 and the arithmetic is nothing -- does not have.

⚠⚠ So my own correction was half wrong too. I said the two-term fit erased a
task term worth up to 32% of a token. At 4.81 us it is **271 x 4.81 = 1.30 ms,
4.2%**, and int8's freedom to raise KMAX is worth about a percent, not ten.

🔑 **What the residual actually was.** D -- two jobs on two fds, submitted and
waited on, which is exactly what one decode call does -- is 57.84 us against
the 71 us fixed term fitted from the stage table. The 13 us of difference is
the pack, the fini and the slice sum. **The floor is the dispatch, it is about
58 us of it, and it is now a measurement rather than a subtraction.** The
answer did not change. What changed is that it is now evidence.

⚠ **And the probe found something nobody was looking for.** Sixteen tasks in
one job costs 216 us where the line through 1 to 8 predicts 94 -- 130% over,
after four points that sit on it to within a microsecond. Something has a
ceiling between 8 and 16 chained tasks. charsiu's largest decode chain is 4
(the head at NMAX 8192), so nothing hits it today, and the batched prefill
chains more. Unexplained, reproducible, and written down.

### 🏁 The qpack fix, measured: int8's prefill now beats int4's outright

Round 153's first two arms, qwen3, 90 rows, ms a row:

```
              pack  submit  fence  read  scale  prep   total
  int4 1024   0.95   0.08   1.24   1.30   0.20  0.11    5.36
  int8 1024   0.88   0.09   0.90   0.94   0.25  0.10    4.59
```

**int8 is 14.4% faster a row than int4, at the same K slice.** This morning
the same comparison was 6.41 against 5.33 the other way, and the whole
difference was pack: 2.66 then, **0.88 now**. Vectorising and pooling int8's
activation quantiser took its own prefill down 28% and turned the sign of the
comparison.

⚠ I had not measured that change on the board -- it went in on a host
bit-identity proof and a prediction. The prediction was that pack would fall
towards int4's 0.86. It landed at 0.88.

And the rest of the row was already int8's: fence 0.90 against 1.24, read 0.94
against 1.30. Eight-bit weights dispatch fewer, wider slices for the same
tensor because their group does not pin KMAX, and both of those numbers follow
from it before KMAX is even swept.

### 🏁 Round 154: KMAX 4096 does not disagree, it collapses

The batched path scored at each width, int8, qwen3, 400 tokens:

```
  KMAX 1024, token loop        43.9086
  KMAX 1024, --batch           44.7567     +1.9%
  KMAX 2048, --batch           43.6578     -0.6%   better than the loop
  KMAX 4096, --batch    4,932,524,413,828  ⛔
```

`llama_auto_kmax` stops its candidate list at 2048 because "Qwen2.5 and
gemma-3-1b DISAGREE at 4096" -- a text hash, a bit, no magnitude. **It is
eleven orders of magnitude.** The candidate list was right and now the reason
has a shape.

And 2048 is CLEAN on int8 -- marginally better than the token loop, which is
what a wider slice should be: fewer accumulator round trips, the same
quantiser, because int8's group does not widen with KMAX.

⚠ But the width is not the lever it looked like. Its whole value is a 25%
smaller read, and the chunk cap halves with it, so it only wins where the
prompt still fits one chunk:

```
  prompt   KMAX 1024 chunks   KMAX 2048 chunks   winner
      64          1                  1           2048, read -25%
      80          1                  1           2048
     110          1                  2           1024
     400          3                  5           1024
```

At 91 tokens round 153 measured exactly that: 4.59 ms a row at 1024 in one
chunk, 5.21 at 2048 in two. **The ceiling on this whole line is about 3%, and
only for prompts under 80.**

### ⚠⚠ But normalising round 153 found something the scoreboard is wrong about

```
  m153  int8 KMAX 1024   413 ms / 91 tok  =  4.54 ms a token
  vendor qwen3           469 ms / 110 tok =  4.26            6.5% behind
  scoreboard int4        594 ms / 110 tok =  5.40            27% behind
```

**int8 takes TTFT from 27% behind the vendor to about 6.5%,** and the reason is
this morning's pack fix: int8's row went 6.41 -> 4.59 ms while int4's stayed at
5.36, so the format that was slower on both axes is now faster on one.

⛔ **And the scoreboard's int8 column predates that commit.** Round 143 ran
`CHARSIU_BENCH_W4V=0` at 08:41 and reported Qwen3 TTFT 751 ms -- int8 slower on
the prompt. The qpack change landed at ~09:2x. That stale column is what the
README's three-row table quotes, and what "int8 is a quality option, not a
speed one" was written from. Round 155 re-runs it.

### 🏁 The .rkllm, mapped to the byte, and the scale array found

Binary search on the nibble-half statistic puts three of the four regions
exactly, and the arithmetic closes to within a tenth of a percent:

```
                      offset          size      expected      error
  metadata          0            7.3 MB       header + tokenizer
  fp16 embedding    7.3 MB     501.0 MB       128256 x 2048 x 2
  fp16 SCALES     514.6 MB      11.2 MB       ?
  int4 layers     525.8 MB     463.8 MB       973 M x 0.5 = 464.0   -0.04%
  int8 head       989.6 MB     250.7 MB       263 M x 1   = 250.5   +0.10%
```

**The int4 region is the weights and nothing else** -- 463.800 against 464.000
predicted -- so the scales are not interleaved with them, which the 0.26%
nibble symmetry already implied and this confirms by size.

**The scales are their own region and it is findable by what a scale IS.**
Embeddings are symmetric about zero; a scale is positive. Sweeping the fraction
of positive fp16 values walks from 49.9% to 99.4% at 514.6 MB, and the region
from there to the int4 boundary is 11.203 MB -- 5,873,759 halves, 90 to 99%
positive, magnitudes from 1e-7 to 1e-3, and about 300 distinct values in any
64 KB.

⚠ **Mostly powers of two.** `frexp` mantissas over 2000 samples: 0.5 in 80% of
them, 0.75 in 12%, 1.0 in 5%. A scale quantised to a shift, not a free fp16.

⚠⚠ **And it is not a flat array of one scale a group.** The tail has an
obvious period of four:

```
  0.000488758   2.10156   3.75509e-06   1.78814e-07
  0.000488758   2.10938   4.94719e-06   0
  0.000488758   2.11719   ...
```

First slot constant, second drifting slowly, third and fourth small. Dividing
the count by the weights gives group 165.7 or 210.4 depending on whether the
head is included -- **neither is an integer**, which is the arithmetic saying
the layout is a record and not an array.

**So: the file is fully readable and the scales are located. What is left is a
record format, not a search.** Whether the vendor's four-bit quantiser is
better than charsiu's can be answered by decoding ONE tensor -- no tokenizer,
no forward pass, no board -- and that is now a bounded piece of work rather
than an open question.

⚠ Everything above is structure and arithmetic. No vendor weight has been
dequantised yet, and until one is, nothing here says anything about quality.

### ⚠ Round 158: the predictor's held-out extremes, and it fails at both ends

Ten ggufs predicted from the desk, every coefficient from `npu_job_cost`, no
model in the fit. Seven now have a measurement on one protocol:

```
  model            t/MB     pred    meas    error   implied matmul share
  SmolLM2-135M     3.66     9.36    14.6   -35.9%          64%
  qwen3            0.91    27.13    30.9   -12.2%          88%
  tinyllama        0.78    41.44    41.3    +0.3%         100%
  gemma-3-1b       1.12    41.11    42.8    -3.9%          96%
  Qwen2.5-1.5B     0.81    61.36    63.0    -2.6%          97%
  SmolLM2-1.7B     0.58    65.54    63.3    +3.5%         104%
  Phi-3.5          0.45   138.36   130.5    +6.0%         106%
```

Two of the three new models land inside 4%, on shapes nobody calibrated
anything against. **tinyllama is +0.3% and took part in nothing.**

⚠⚠ **But three rows imply a matmul share above 100%,** which is the model
claiming more of a token than the token has. That is not a fit error. Phi-3.5's
gate+up is **19.6 MB in one call** and the MB sweep stopped at 8.39, so those
rows are extrapolated 2.3x past the data.

And the other end fails the other way: SmolLM2-135M's calls are 0.13 to 0.69 MB
and it is missed by 36% LOW, with an implied matmul share of 64% -- i.e. the
elementwise work this predictor does not model is a third of that token where
it is a ninth of qwen3's.

⛔ **I tried to fix the small end by adding a term and made it worse.** A
residual proportional to `layers * n_embd` is physically the right shape for
elementwise work, and fitting it took RMS from 14.7% to **16.4%** -- because
the residuals it was fitted to are +303, +132, -3, +56, +38, -46, -80 ns per
unit, sign-changing, which is two errors of opposite sign being averaged into
one parameter.

🔑 **So: widen the measurement, do not add a parameter.** The sweep now runs
0.002 to 67 MB, covering every call any of the ten models makes. The
elementwise term can be fitted after the matmul term stops being extrapolated,
or it may turn out not to be needed. Fitting it first was fitting the second
parameter on top of a wrong first one.

### ⛔ Round 159 killed the board, and the probe had no business asking

Widening npu_job_cost's sweep to 67 MB put `4096 x 16384` on the end. **n =
16384 is past the 8192 the device is opened for.** The job was submitted
anyway:

```
  rocket 27708000.npu: NPU job timed out
  rk_iommu 2770a000.iommu: Error during raw reset. MMU_DTE_ADDR is not functioning
```

which is the dead state this project already has a note about, and the clock
between that round and the next says the board sat in it for **five hours**.

Both bounds are written down elsewhere in this tree -- npudev refuses
`(k/32)*m > 5120` and the device is opened for a max n -- and the probe checked
neither. **A probe that walks an axis has to know where the axis ends,** and I
widened one by editing an array.

🔑 **usb_reset brought it back.** A power cycle does what the driver's own
reset path cannot, and round 160 confirmed it: `accel0` present, 197 tensors
staged, 10.74 GB/s, English out. That is worth knowing the next time this
happens -- the five hours were nobody watching, not an unrecoverable board.

### And the head was being extrapolated, which was a smaller version of the same

`charsiu_shapes` priced the output head as one call of its whole weight -- 38
MB on Phi-3.5, 157 on gemma4 -- and `call_us` answered by extending a straight
line at 89 us a megabyte, where round 147 measured the head itself at 15.56
GB/s, which is 64. Three models came back implying a matmul share above 100% of
their own token: the arithmetic refusing an extrapolation, not a poor fit.

The head is `ceil(n_vocab / NMAX)` slices of at most NMAX, so pricing it by
slice puts every piece inside the measured range:

```
                 whole    by slice    measured     was        now
  Qwen3-0.6B     27.13      27.77        30.9    -12.2%    -10.1%
  gemma-3-1b     41.11      42.11        42.8     -3.9%     -1.6%
  Phi-3.5       138.36     138.45       130.5     +6.0%     +6.1%
                                          RMS      8.2%      6.9%
```

⚠ A small correction, because the head is one call against a hundred and
twenty. It was still an extrapolation being reported as a measurement.

### ⛔ A regression I shipped, and it hid inside a judgement I then made

Round 164 read prefill at 8.29 ms a row where round 153 read 4.59 on the same
90 rows in one chunk. The difference was `read`: 0.94 -> **4.09**.

Cause: this afternoon's "four tuned constants become one rule" derived
`CHARSIU_NPU_POOL_READ_MIN` from `charsiu_pool_min(rate, charsiu_threads())`
**at charsiu_npu_open() time**, and there is no thread pool there --
`charsiu_run` never calls `charsiu_threads_start`, so `g_pool.n` is 0 and
`charsiu_threads()` returns 1. Below two threads that function returns "never
pool". The accumulator read back stopped pooling, the prompt got 1.8x slower,
and it was committed and pushed.

⚠⚠ **arch_sanity 6/6 and hostcheck 4/4 both passed.** With no NPU the read back
path is never taken at all. The only reason it surfaced is that round 164
printed a stage table next to one from round 153.

Fixed by resolving on first use. ⚠ And the fix converted two of four call
sites, because the other two wrap the condition across a line and a
single-line pattern walked past them -- the same shape as this morning's audit
missing five `getenv(...) == NULL` switches. Twice in one day a mechanical
replace declared itself done on a subset, and both times counting the call
sites caught it, not reading the diff.

**And the judgement it poisoned.** From round 164 I read attention at 11.5% of
a prefill row and concluded TTFT was "spread over a dozen items, no single
target, carrying-water work, not worth chasing". On the fixed binary
(round 165):

```
  q k v         1.22   23.0%        attention     0.97   18.2%
  gate + up     1.38   26.0%        rmsnorm+rope+silu+residual  0.50  9.2%
  down          0.76   14.3%
  o proj        0.49    9.2%        matmul total          72.5%
```

**attention is 18.2%, and 0.97 ms a row on its own is larger than the entire
0.67 ms a row gap to the vendor.** The matmul share had been inflated by my own
regression, which pushed everything else down. I gave a "not worth chasing"
verdict on numbers from a binary I had broken four hours earlier.

### 🔑 Round 166: acc_out bypasses the output convert, it does not select it

One matmul with a known product -- A and B all ones at zero point 128, so every
element is (1-128)^2 = 16129 and k = 1024 should give 16516096:

```
   raw 0004fc00    as int32 16516096    as fp32 2.3e-38    fp16 6.1e-05
```

**int32, exactly right.** And the same job's registers are the vendor's float
stage, because `job.c` sets `if (job->acc_out) wide8 = 0x3f` -- 0x4010
a0000002 (PROC_PRECISION 2, fp16), 0x4044 = 2, identity requant.

So the note that says *"w4a16 does not requantise at all, the output is a
float"* is about the DPU's PROCESSING precision. `acc_out` takes the raw
accumulator before the convert. A narrower read is not a cast, it needs the
requant and the convert to actually run.

### What a narrower output would be worth, and where it is legal

TTFT is 15.8% behind the vendor on int8: 74 ms over 110 tokens, **0.67 ms a
row**. Prefill reads 0.94 ms a row at four bytes an element. Two bytes is 0.47.

**And the legality is a static property of the weights.** With int8's group
being the whole row, the DPU's per-channel requant can apply exactly the scale
the CPU applies now, and the largest value it can emit is `127 * sum|w|`,
because every `|a_q| <= 127`:

```
  Qwen3-0.6B     worst tensor   18,334      fp16 max 65,504    ✓
  Phi-3.5-mini   ffn_down      139,399                         ⛔ 2.1x over
```

⚠ `job.c` already named that tensor -- *"the output magnitude of ffn_down
varies by up to 2971x between tokens"* -- without the arithmetic. The bound is
computable at staging from the weights alone, so this is a per-tensor decision
and not a global switch, which is the shape this project wants: a rule over
`(m, k, n)` rather than a constant.

⚠⚠ The bound is worst case: it assumes every `|a_q|` is 127 and all the signs
agree. Real activations run about a third of that, so Phi-3.5 would probably
not overflow -- but an overflow is an inf that destroys the token, so only the
worst case can gate it.

### And prefill's attention is the other half, with room in it

Round 165 on the fixed binary: attention is **0.97 ms a row, 18.2%** of a
prefill row, and 0.97 alone is larger than the whole 0.67 gap. Its rate:

```
  33.2 MMAC over 87 ms   =  0.38 GMAC/s
  66.4 MB of KV over 87  =  0.76 GB/s      against 11.9 the threads reach
```

**Neither bandwidth nor arithmetic bound.** And the pool splits it over HEADS,
so the thread count only divides evenly when the head count does:

```
  qwen3 16 -> 100%    tinyllama 32 -> 100%
  gemma-3-1b 4 -> 50%       SmolLM2-135M 9 -> 56%
```

⚠ Splitting over (head, row block) instead needs the per-head scratch
(`s->batt + h * R * n_ctx`) reindexed, and prefill's best axis is the row block
(12 of them at n=90, R=8) while decode's is the head (one row block). Worth
doing, not a safe last-hour change.

### 🏁 Round 167: the output width is controllable, and both ends were reached

acc_out off, the same known matmul, WIDE8 swept coarsely:

```
  acc_out=1                 16516096   raw 0004fc00    four bytes, exact
  acc_out=0  WIDE8=0x3f     16516096   raw 0004fc00    identical
  acc_out=0  WIDE8=0                   raw 80808080    ONE byte, repeated
  acc_out=0  WIDE8=0                   raw 80808080    (repeat, same)
```

**0x3f writes four bytes and 0 writes one.** The values under WIDE8=0 are
garbage -- the coefficient buffer is zeroed, so a requant that multiplies
produces nothing -- but the WIDTH moved, and nothing timed out in any arm.

That is the sweep `job.c` has been asking for since it was written: *"the bits
exist separately because 'the whole bundle changed something' does not say
which register the width lives in, and a single field sweep is the method that
worked on 0x4050 in round 260."* Nobody had run it.

```
  bit 0  0x4010   bit 1  0x4030 low   bit 2  0x4038
  bit 3  0x4044   bit 4  0x4050       bit 5  0x40ac/b0/b4 identity
```

**What is worth finding is two bytes.** Prefill reads 0.94 ms of a 4.59 ms row
at four bytes an element; two would be 0.47 and the entire gap to the vendor is
0.67. One byte would be 0.24 -- and is exactly what `job.c` refused, because
ffn_down's output range does not fit in one. So fp16 is the target and int8 is
not, which makes this a search for a bit rather than for a bundle.

### 🏁 Round 168: the width sweep job.c asked for, and it has only two values

```
  WIDE8=0x01  bit0  0x4010          ⛔ NPU timed out
  WIDE8=0x02  bit1  0x4030 low         80808080    one byte
  WIDE8=0x04  bit2  0x4038             7F7F7F7F    one byte, saturated
  WIDE8=0x08  bit3  0x4044             80808080    one byte
  WIDE8=0x10  bit4  0x4050          ⛔ NPU timed out
  WIDE8=0x20  bit5  identity requant   00000000    zero
  WIDE8=0x3e  0x3f minus bit0       ⛔ NPU timed out
  WIDE8=0x37  0x3f minus bit3          16516096 ✓  four bytes
  WIDE8=0x1f  0x3f minus bit5          16516096 ✓  four bytes
  WIDE8=0x0f  low four               ⛔ NPU timed out
```

**bit0 and bit4 have to be together** -- either alone wedges, which reproduces
round 311 -- and **bit3 and bit5 are not needed at all**: 0x37 and 0x1f both
give the exact four-byte accumulator.

⚠⚠ **And the width has exactly two values, 1 and 4. There is no 2.** I had
picked `0x4044` as the width register on the strength of "int8 writes 1, w4a16
writes 2", and 0x37 removes it and stays four bytes wide. Whatever selects fp16
output is not in this bundle.

### But one byte is worth more than fp16 would have been

```
  prefill read, qwen3        0.94 ms a row at four bytes
  at two bytes               0.47      saves 0.47
  at ONE byte                0.23      saves 0.70     the gap is 0.67
```

`job.c` refused a byte for a real reason -- *"the coefficient buffer's scale is
fixed at build time, and the output magnitude of ffn_down varies by up to
2971x between tokens"*. But that variation is **per token**, and the term that
varies is known before the dispatch: the output is
`sum(w_q a_q) * w_scale * a_scale`, `w_scale` is per output channel (which is
what a DPU coefficient IS, and what int8's whole-row group already is), and
`a_scale` is the activation's own `d1`, computed at pack time.

So the coefficient buffer would be rewritten per call with the current
`a_scale` folded in. The trade:

```
  write  1024 x 2 bytes  =   2 KB a call
  save   read 360 KB -> 90 KB = 270 KB a call
```

**135 to 1.** ⚠ And it costs precision: an int8 output is quantised. But the
result feeds the next layer, which quantises its activations to int8 anyway, so
the loss may be nothing at all -- which is a `charsiu_ppl` question and this
tree now has the instrument for it.

⚠ Four of ten arms timed out. Round 169 checks the board is alive; round 167's
four arms had zero timeouts, so wedging is a property of the bits asked for and
not of asking.

### 🏁 Round 170: attention over (head, row block), and the control moved too

```
                        HR=0                HR=1          attention
  gemma-3-1b   attn 0.74  row 11.63    0.40  row 7.14        -46%
  SmolLM2-135M attn 0.28  row  2.50    0.26  row 2.43         -7%
  qwen3        attn 0.99  row  5.32    0.90  row 5.33         -9%
```

Text identical across the knob on all three -- one hash printed per model, so
both arms agreed.

**gemma-3-1b's -46% is what "half of eight threads idle" predicts**, and it has
4 heads. But the informative row is **qwen3, the control**: its 16 heads
already divided by 8, and it still gained 9%. So the finer unit helps beyond
the divisibility -- a row block finishes sooner than a whole head, so the tail
of the pool is shorter.

⚠⚠ **AND gemma-3-1b's WHOLE ROW is not readable.** 11.63 -> 7.14 looks like a
38% win, and in the same pair `staging` went 8943 -> 4236 ms. Staging has
nothing to do with how attention is pooled. That arm was the first thing the
round ran.

**A first point being cold is the third time today.** Round 162 called two
cells "a shape property" and they did not reproduce; round 155 and 161's small
end disagreed by 2x and best-of-five removed it; and here it inflates a whole
row by 60%. Round 171 runs HR=1 first, HR=0 second, with a warm-up pass before
either: if the attention numbers hold with the order reversed they are the
split, and if they follow the order they were the cache.

### 🏁 Round 171: reversed and warmed, and it splits the result in two

```
                  m170 (HR=0 first)     m171 (HR=1 first, warmed)
                  HR=0    HR=1          HR=1    HR=0
  gemma-3-1b      0.74    0.40          0.40    0.67
  SmolLM2-135M    0.28    0.26          0.26    0.27
  qwen3           0.99    0.90          0.88    0.96
```

**The attention numbers follow the knob, not the order.** gemma-3-1b reads 0.40
under HR=1 in both rounds and 0.74/0.67 under HR=0; qwen3 reads 0.88/0.90
against 0.96/0.99. The split is real, reproducible, and the text is identical
across it on three architectures and two orders.

⛔ **And the whole-row win was the cache.** gemma-3-1b's row read 11.63 -> 7.14
in round 170 with `staging` 8943 -> 4236 in the same pair; warmed and reversed
it reads **7.50 -> 7.27**, with staging 4289 -> 4277. A 38% win was a 3% win
and a cold first arm.

So the honest table:

```
                attention          whole row
  gemma-3-1b    0.67 -> 0.40  -40%    7.50 -> 7.27   -3.1%
  SmolLM2-135M  0.27 -> 0.26   -4%    2.42 -> 2.38   -1.7%
  qwen3         0.96 -> 0.88   -8%    5.40 -> 5.45   +0.9%
```

**Attention falls 4 to 40% and the row falls 0 to 3%,** because attention is 5
to 18% of a row. On qwen3 that is 0.08 ms a row against a 0.67 gap to the
vendor -- 12% of the distance, real and small.

🔑 The reason to keep it is not the 3%. It is that the win is largest exactly
where the head count divides worst, so it removes a model-shaped cliff rather
than adding a speedup: gemma-3-1b was paying 67% more for attention than its
arithmetic needed, and nothing in the code said so.

### 🏁 The pre-merge regression, and the one signal it raised

```
  1. ppl, three carried controls, four decimals
       q4_0  26.6416  want 26.6416   int4 49.8930 want 49.8930
       int8  27.0668  want 27.0668
  2. prefill, int8 KMAX 1024
       4.48 ms a row, read 0.93      (the regression read 8.29 and 4.09)
  3. scoreboard int4        earlier   regression   vendor
       Qwen3                 26.31      26.55       24.85
       TinyLLAMA             22.83      22.89       19.71
       Phi3                   7.07       7.04        6.58
       Gemma4                 9.33       8.93        9.23   <- -4.3%
```

**Seventeen commits touched the product path today and not one moved the
answer.** Three ppl arms reproducing to four decimals is the strongest check
this board offers, because it is the only measurement here that survives a
session boundary.

⚠ Gemma4's 4.3% sat outside the board's ~3% drift, and gemma4 was the only
model the attention change had never been measured on -- rounds 170 and 171
took gemma-3-1b, SmolLM2-135M and qwen3. So round 173 alternated the knob four
times on it, warmed:

```
  HR=1  10.05    HR=0  10.05    HR=1  10.04    HR=0  10.03   text identical
```

**Neutral, and the 4.3% is drift.** Which is the answer that could only be had
by asking -- gemma4's decode spread over six runs is 7.08 to 8.93, so a
best-of-six can move 4% without anything changing.

⚠⚠ And the merge check found what the test suites did not, twice: the
zero-sentinel in `poolread_min` (=0 meaning "always pool" was being replaced by
the derived value) was caught by reading today's diffs before proposing this,
not by any run.

### ⛔ The 11.2 MB "fp16 SCALES" region was the REGISTER COMMAND STREAM

Yesterday's entry put a scale array at 514.6 MB and described it as fp16, 90 to
99% positive, "mostly powers of two", with "an obvious period of four". Every
one of those observations is real and every one of them is an artefact of
reading `u64` register writes as `fp16`:

```
  1c 40 01 00  00 00 01 10      reg 0x401c <- 0x00000001   target 0x1001 DPU
  20 40 03 00  00 00 01 10      reg 0x4020 <- 0x00000003
  24 40 3f 00  00 00 01 10      reg 0x4024 <- 0x0000003f       <- WIDE8
  28 40 c0 15  00 00 01 10      reg 0x4028 <- 0x000015c0
```

The period of four is `u64` seen as four halves. The 96% positive is the two
high halves being small. The "powers of two" mantissas are register numbers.
**And the decoder was already in this tree** -- `tools/rkllm_regcmd.py` has had
`[63:48] target [47:16] value [15:0] register` written at the top of it since
the day it was written, and the region matches that layout at 95 to 98% over
its whole 11.7 MB.

🔑 **The tell I had and did not use: the group size came out non-integral.**
Yesterday's own note says "dividing the count by the weights gives 165.7 or
210.4 -- neither is an integer" and reads that as "the layout is a record".
A non-integral count is also what "these are not scales" looks like, and the
second reading costs nothing to test.

### 🏁 The float region, and the vendor's int4 quantiser written down

Sweeping two statistics over 505 to 527 MB -- the fraction of `f32` that is
positive and smaller than 10, and the fraction of `u64` whose target field is
one of the seven `rkllm_regcmd` knows -- separates the file cleanly:

```
        0 ..  7.47 MB    header + tokenizer
     7.47 .. 508.47 MB   fp16 token embeddings, 128256 x 2048   (501.0 MB)
   508.47 .. 512.10 MB   fp32 SCALES AND ZERO POINTS            (  3.6 MB)
   512.10 .. 512.60 MB   fp32, all integral, signed
   512.60 .. 514.15 MB   a table, 60 to 79% zero bytes
   514.15 .. 525.90 MB   REGISTER COMMAND STREAMS               ( 11.7 MB)
   525.90 .. 989.6  MB   packed int4 weights
   989.6  .. 1240.4 MB   int8 output head
```

The embedding boundary is not fitted: 128256 x 2048 x 2 is 525336576 bytes and
it ends exactly where the first scale begins, `0x1FC77DA0`.

**The record is per tensor: `rows` fp32 scales, then `rows` fp32 zero points**,
and each layer closes with two 2048-slot fp32 arrays (its two norms). So a
layer is `2 * 23552 + 4096 = 51200` slots and the whole thing is a stride, not
a search:

```
  blk.0.attn_q      122728        blk.1.attn_q      173928     +51200
  blk.0.attn_k      126824        blk.2.attn_q      225128     +51200
  blk.0.attn_v      127848        blk.3.attn_q      276328     +51200
  blk.0.attn_output 128872        ...
  blk.0.ffn_gate    132968
  blk.0.ffn_up      149352
  blk.0.ffn_down    165736
  (two norms)       169832
```

Found by correlating each tensor's per-row `max|w|`, taken from the same model
in q8_0, against the region -- 112 tensors, best offsets tiling it with no gap
and no overlap, in model order. The zero points are exactly integral, range -5
to +3, mean -0.50, and about half of them are zero.

**The quantiser is `w = s * (q - z)` with `q` in `[-8, +7]`.** Both halves of
that come from the data: `max/s + z` is 7.19 +- 0.34 and `min/s + z` is
-8.20 +- 0.33, while the other sign convention, `w = s * (q + z)`, has four
times the spread. And on six tensors the scale is the formula exactly:

```
  blk.3.attn_q    (max-min)/scale = 14.9999 +- 0.0038
  blk.3.attn_k                      14.9998 +- 0.0030
  blk.10.attn_q                     14.9999 +- 0.0040
  blk.10.attn_k                     14.9999 +- 0.0032
  blk.15.attn_q                     15.0000 +- 0.0035
  blk.15.attn_k                     15.0000 +- 0.0035
```

Four digits, and the +-0.004 is q8_0's own rounding of `max` and `min`. So
`scale = (max - min) / 15`, sixteen levels, asymmetric, **one scale and one
integer zero point per output row**.

### ⛔ And the first error table was scored against weights the vendor never saw

The other 106 tensors do NOT satisfy that relation -- they come back at 3.5,
7.1, 9.6, 18.2 with several percent of scatter. A per-tensor factor with
per-row spread is what quantising **transformed** weights looks like, so the
vendor has a calibration step in front of its quantiser and the reference in
this tree is not what it quantised.

I had already printed a 112-tensor table reading "vendor 23.667% against
charsiu 13.788%" before checking that. It contained `blk.1.ffn_up 99.520%` --
a reconstruction with no correlation to its input, which no shipped quantiser
produces -- and that one cell is the whole table's retraction. **The number was
the vendor's scales applied to weights they were not computed from.**

⚠ The same run had a second thing wrong that the result did not show: the
`vendor` and `vendor-sym` columns agreed to three decimals, and the reason is
that `round(w/s + z) - z` is `round(w/s)` for integer `z`. The zero point is
mathematically inert in that expression except where it moves the clip window.
Two arms that cannot differ are not two arms.

### 🏁 The six tensors that CAN be scored, and what they say

Restricted to the six where the vendor's scale provably comes from these exact
weights, all four quantisers on one set of weights:

```
                                      weight error
  vendor    one (s, z) a row, asym        15.843%
  charsiu   group 1024, symmetric         14.935%     <- what ships
  charsiu   one scale a row, symmetric    16.209%
  q4_0      group 32, symmetric            8.980%
```

**At equal granularity the vendor is 2.3% better than charsiu, and charsiu as
it ships is 5.7% better than the vendor.** Their whole advantage at that
granularity is the zero point, which is a thing charsiu can have.

⚠ Six tensors of 112, weight error and not perplexity, and this tree already
owns the counter-example to reading weight error as quality: `CHARSIU_NPU_W4_CLIP`
minimises exactly this number and made KL worse, 0.0989 to 0.2084. It narrows
the empty cell. It does not fill it.

### ⚠ The int4 weights are not stored in the reference's order

A sign correlation of `blk.0.ffn_gate` row 0 against every byte and nibble
alignment of the 480 MB int4 region tops out at `|r| = 0.133`, which is the
noise floor for a 2048-long pattern. Row major is dead; the weights are in some
NPU-native order, which is what charsiu's own packer also has to produce.

### ⛔ A scratch file called bisect.py power-cycled the board four times

`scratchpad/bisect.py` is an old round script, and it shadows the stdlib
`bisect` module. Anything run from that directory that reaches `random` or
`tempfile` -- `from gguf import GGUFReader` does, through `gguf_writer` --
executes it, and line 11 of it is `uart.usb_reset()`.

So `import numpy; from gguf import ...` reset the board. The failure surfaced
as `FATAL: no shell` printed by a script that contains no such string, and I
spent four tool calls looking for a hook before reading the traceback, which
had named the file the whole time.

🔑 **Never name a scratch script after a stdlib module**, and `python3 -P`
keeps the script's own directory off `sys.path` when the directory is not
trusted. The scratchpad now has none: `sys.stdlib_module_names` is the check.

### ⛔ The zero point, priced over all 112 tensors, and it is not worth taking

The six-tensor table says the vendor's asymmetry is worth something, so the
next question is what it would be worth to charsiu. Symmetric against
asymmetric at every group size, same weights, no vendor data needed
(`tools/rkllm_scales.py zero`, and the row arm's scale bytes are counted per
tensor because its group is the tensor's own k -- 2048 on most of these and
8192 on ffn_down):

```
                 weight error   bytes a weight
  sym   row         15.297%         0.5015
  asym  row         15.182%         0.5031        -0.8%
  sym  1024         13.788%         0.5039
  asym 1024         13.592%         0.5078        -1.4%
  sym   512         12.800%         0.5078
  asym  512         12.538%         0.5156        -2.0%
  sym   128         10.848%         0.5312
  asym  128         10.423%         0.5625        -3.9%
  sym    32          8.826%         0.6250
  asym   32          8.211%         0.7500        -7.0%
```

**At charsiu's shipped group of 1024 an asymmetric zero point buys 1.4%,** and
it costs a second fp32 array -- double the scale memory -- plus a
`zero * sum(a)` correction a group a row in the accumulate. The value grows as
the group narrows and is 7% at 32, which is the group charsiu cannot have,
because the group **is** the K slice and the read back is `m*n*ceil(K/KMAX)*4`.

🔑 **So the vendor's asymmetry is a consequence of its granularity, not an
advantage over ours.** They have one scale a row, so the zero point is the only
cheap thing left to add; charsiu already spends those bytes on 2x the scales
and gets more for them. Two rows of the same table:

```
  vendor    asym, one a row     15.18% of the way to their number
  charsiu    sym, group 1024    13.79%
```

⚠ Weight error. The same caveat as everywhere above: this tree's own
`CHARSIU_NPU_W4_CLIP` minimises this number and made KL worse. What it settles
is the *cost side* -- 1.4% for double the scale bytes is not a trade worth
making blind -- not the quality side.

### 🏁 The formula survives six alternatives, and two calibrations do not

`scale = (max - min) / 15` was read off six tensors where it holds to four
digits. The question that leaves open is whether it is the *form* of the
vendor's rule or just a coincidence there, and the way to ask is to offer the
rule some competition. Relative spread of `scale / f(row)` over 42 tensors,
seven candidate `f`:

```
                 range/15  absmax/8    rms   mean|w|   p99.9    p99    p95
  blk.6.attn_q      2.05%     4.54%  10.96%   11.52%   7.08%  10.16%  11.31%
  blk.3.attn_q      0.03%     5.50%  13.79%   14.51%   8.81%  12.60%  14.23%
  blk.0.ffn_gate    4.08%     5.46%  18.34%   21.96%   8.33%  15.05%  21.25%

  wins: range/15 39, absmax/8 3, everything else 0
```

The three losses are `ffn_down` rows where `range/15` and `absmax/8` read 68%
and 65% and neither is a description of anything. **So the scale is
`(max - min) / 15` of something, and the something is a transformed weight.**

`rho = 15 * scale / (max - min)` is the size of that transform, and it has
structure worth having:

```
  layer      attn_q    attn_k    attn_v   attn_out  ffn_gate   ffn_up  ffn_down
    0         3.507     3.451     2.856     0.990     1.679     2.121     0.830
    1         1.549     1.580     1.493     1.043     4.300    22.317     0.298
    2         2.949     3.069     3.141     0.906     1.386     1.684     0.833
    3         1.000     1.000     1.132     0.889     1.561     1.888     0.847
   10         1.000     1.000     0.950     1.114     0.954     1.051     0.926
   15         1.000     1.000     1.158     0.864     0.912     1.436     0.802
```

**The two columns that sit below 1 everywhere are attn_output and ffn_down --
exactly the two whose input is not a normed activation.** Everything the vendor
widens is fed by a norm, and it widens layers 0 to 2 hardest, up to 22.3x on
`blk.1.ffn_up`. Layers 3 to 15 sit within a few percent of 1. That is the shape
of an activation-aware method spending its budget where the outliers are, and
it is a map of where charsiu's own AWQ would be worth turning on.

⛔ **Two readings of what the transform IS, both refuted the same afternoon.**

*A few outlier input channels scaled up.* If `w' = w * c` with a few large
`c_j`, the row range would be carried by those columns. On `blk.1.ffn_up`, the
most extreme tensor in the table, `corr(row range, |w[:,j]|)` tops out at
**0.23** and nothing passes 0.5 -- while the untransformed `blk.3.attn_q` has a
median of 0.36 from the trivial correlation alone. No column carries it.

*The RMSNorm folded into the columns.* Structurally the best candidate: the
tensors with `rho > 1` are precisely the ones with a norm in front. Folding it
makes the spread **worse in every case** -- `blk.3.attn_q` goes from
`1.000 / 0.0%` to `2.233 / 20.2%` -- and folding `1 + norm` instead gives
`0.706 / 7.8%`. A hypothesis that turns an exact identity into a 20% scatter is
answered.

🔑 What the negatives cost: one run each. What they buy is that the next
candidate is not proposed against the same evidence.

### ⚠ AWQ by layer: 70% of the win from 3 layers of 28, and the other 25 are not free

The vendor's `rho` puts its widening in layers 0 to 2 and leaves 3 to 15 within
a few percent of 1, so the obvious transfer is to stop paying for AWQ where the
vendor does not. `CHARSIU_NPU_AWQ_LAYERS` restricts the factor to a range of
blocks. Qwen3-0.6B, host CPU reference, 200 tokens, one binary and one corpus,
every arm naming the knob:

```
  AWQ off                          113.2310
  CHARSIU_NPU_AWQ_LAYERS=0-27       73.7671   -34.9%   (identity arm)
  (unset, every layer)              73.7671   -34.9%   <- must be, and is
  CHARSIU_NPU_AWQ_LAYERS=0-5        84.1331   -25.7%
  CHARSIU_NPU_AWQ_LAYERS=0-2        85.4975   -24.5%
  CHARSIU_NPU_AWQ_LAYERS=0-1        95.2883   -15.8%
  CHARSIU_NPU_AWQ_LAYERS=3-27       91.4078   -19.3%
```

**Layers 0 to 2 carry 70% of the whole win on 11% of the layers.** Per layer
that is 8.2 points against 0.77 for the rest, so the vendor's profile does
transfer. AWQ's cost is structural -- a tensor carrying a factor cannot share a
packed input, so grouped q/k/v drop to single calls -- and restricting it to
three blocks leaves the other twenty-five grouped.

⛔ **But the strong form is refuted: `3-27` is still worth 19.3%.** A third of
the benefit is spread thinly over the layers the vendor leaves alone, so
"switch it off above layer 2" is a trade, not a free lunch. And `0-5` beats
`0-2` by only 1.4 points, so layers 3 to 5 are nearly worthless and the rest of
the value is diffuse across 6 to 27.

⚠ **And this corrects my own reading of `rho` from an hour earlier.** `rho = 1`
means the row's RANGE is unchanged, not that the weights are: a transform that
preserves each row's max and min is invisible to it. Only the six tensors at
`1.000 +- 0.0003` are provably untransformed. For the rest of layers 3 to 15,
at 0.95 to 1.15 with 2 to 10% spread, "untransformed" was more than the
statistic says -- and this ppl sweep is what says so, because charsiu's own
calibration still finds 19.3% to take there.

🔑 The identity arm is why the rest is readable: `0-27` had to equal the
unrestricted run to the last digit and does, so the parse is not quietly
excluding a layer. The controls before it are the same shape -- AWQ off and
AWQ everywhere reproduce the recorded 114.2234 / 73.8760 to within 0.9% and
0.15%, the residual being a corpus that differs slightly from that session's.

### 🏁 The int4 payload, and the whole 1240 MB file closes with zero bytes left

The last valid register command word ends at `0x20DDA980`, and `0x20DDA9C4` is
the only offset near it for which the arithmetic is exact:

```
  0x20DDA9C4  +  112 matrices        486539264 B   = 464.0 MB
              +  128256 x 2048 head  262668288 B   = 250.5 MB
              =  1300605380                        = the file, to the byte
```

A layer is `2048*2048 + 512*2048*2 + 2048*2048 + 8192*2048*2 + 2048*8192`
halved = **30408704 bytes**, sixteen of them is 486539264, and the region ends
exactly where the head begins. Nothing is fitted here.

**Confirmed without knowing the order, which is the point.** A tensor's int4
CODE HISTOGRAM survives any permutation of its codes, and for the six tensors
where `rho == 1` the histogram is predictable from the reference and the
vendor's own `(s, z)`. `blk.3.attn_q`'s prediction against every 2 MB window of
the 464 MB region:

```
  tv 0.00025   block 1392   0x264DA9C4   = layer 3, offset 0   <- what the map predicts
  tv 0.00108   block 1395   ... 196608 into layer 3
  tv 0.00123   block 4721   ... layer 10
  median 0.01501, 1st percentile 0.00327
```

The best of 7393 windows is the predicted one, 13x better than the 1st
percentile. So the weight region holds the same 112 tensors in the same order
as the scale region, and the quantiser identity `w = s * (q - z)` is now
confirmed against the actual stored codes rather than against a scale alone.

### 🔑 The codes fill the grid the same way at rho 22.3 as at rho 1.000

`rho = 22.3` on `blk.1.ffn_up` was the one number in the rho table that looked
like a mistake: a scale 22x wider than the row's range would crush every code
into 0 and +-1. It does not:

```
  tensor              code sd   |code| >= 4     rho
  blk.3.attn_q          2.210      11.80%     1.000
  blk.10.attn_q         2.221      12.20%     1.000
  blk.1.ffn_up          2.153      10.91%    22.317
  blk.1.ffn_gate        2.143      10.64%     4.300
  blk.15.ffn_down       1.830       6.25%     0.802
```

**The vendor's scale fits whatever it quantised.** `rho` is the size of a real
transform of the weights, not a badly chosen scale, and the doubt that the map
might simply be wrong for the 106 is answered by the file itself.

### ⛔ The order inside a tensor is still not known, and it had a fair run

With the byte range confirmed, the haystack is 2 MB rather than 464, so a
candidate order can be scored at a KNOWN offset with a single dot product and
no search at all. Row major, column major, and four orderings across eight
output tiles and six input tiles: **142 candidates on signs, 194 on code
values, every one at the noise floor.** Best code correlation 0.023 against a
floor of 0.002.

charsiu hands the device row-major nibbles and the device computes correctly,
so this is not the matmul weight layout -- it is the CONVOLUTION one, which is
what the vendor's own register streams dispatch, and it is not a simple tiling
of (output, input).

⚠ And a bug the output caught rather than the number: `rkllm_layout.py` printed
every hit at `start + i // 2` while the streaming rewrite had already seeded
its cursor at `start * 2`, so the addresses landed outside the range that was
searched. The correlations were right the whole time. A wrong address next to a
right correlation survives a glance at the top line.

### 🏁 AWQ's exponent was never swept, and 0.5 is on the wrong side of the minimum

Every AWQ experiment in this tree pinned `CHARSIU_NPU_AWQ` at 0.5 -- the usual
square root balance -- and moved the clamp. Sweeping the exponent instead, with
the clamp at its default 2.0, qwen3 on the host CPU reference at 500 tokens:

```
  alpha   0.25    0.30    0.35    0.40    0.45    0.50
  ppl    68.86   65.84   65.12   68.02   75.52   76.36     (AWQ off: 113.56)
```

A clean single minimum at **0.35**, and **65.12 against 76.36 is 14.7%**. The
same ordering holds on the other corpus at 200 tokens -- 68.07 against 73.77 --
so it is not one length or one passage.

⚠ **And the first version of this measurement ranked the wrong cell.** The
opening sweep was a 6 x 3 grid of alpha against clamp at 200 tokens, and its
winner was `0.35 / 1.5` at 66.99. At 500 tokens that cell reads **75.44** and
`0.35 / 2.0` -- third in the grid -- reads 65.12. The grid was not smooth
either: `0.50 / 1.5` at 78.28 sat worse than `0.20 / 3.0` at 71.98, which is
the shape of a statistic that cannot rank what it is being asked to rank.

🔑 **199 scored positions cannot separate cells a few points apart, and the
tell was in the surface, not in the numbers.** Holding the clamp at its default
and sweeping one variable gave a curve with one minimum and no crossings, and
that curve reproduces across both lengths. The cell that survived is the one
that was never the winner of the noisy grid.

⚠ The board's 40.83 in the README was measured at 0.5 and has not been re-run.
If the host's 14.7% transfers it lands near 35, which would be inside 31% of
llama.cpp's own q4_0 rather than 53% -- but that is a prediction, not a result.

### 🏁 The weight layout, 89% of the way, and the 11% that is not phase

Guessing layouts was dead -- 336 candidates at the noise floor. Deriving one
works much better, and the derivation has three steps, each of which is a
measurement rather than a hypothesis.

**1. The block is 16 output channels.** A row's codes average about its own
zero point, so the spread of window means over a tensor says how many rows a
window mixes. The excess over the sampling noise is flat at 0.179 from 1024
codes up to 32768 and then falls as `1/sqrt(m)`:

```
  window     2048    8192   16384   32768   65536  131072
  excess    0.1778  0.1783  0.1784  0.1786  0.1221  0.0844
```

The knee is exactly 32768 codes = 16 KB, and `sd(z)/sqrt(16)` is 0.195 against
the 0.179 observed. Three tensors agree.

**And which 16 is settled without any within-block knowledge**: block means
survive any permutation inside the block, so the 128 observed block means can
be regressed on the predicted ones. Contiguous 16-row groups give **r =
1.0000**; a strided grouping gives −0.05 and a shuffled control −0.16.

**2. The cycle is 512 codes.** Autocorrelation of window means, computed inside
blocks so the boundary cannot manufacture a period: at a 64-code window the
peaks are at lags 8, 16, 24, 32 (r 0.77); at 32 codes, lags 16 and 32 (0.72);
at 16 codes, lag 32 (0.60). All three say 512.

**3. Which row each of the 512 positions holds, fitted over all 128 blocks at
once.** Every one of the 16 rows comes out used exactly 32 times, the margin
between the best and second-best assignment has a median of 15.5 and a minimum
of 9.07, and the residual is 0.2545 against a sampling noise of 0.2762. There
is nothing ambiguous in it.

⛔ **And the layout built from all that is 89% right, not right.** Rebuilt and
scored against the file it reads 88.46% exact, which is what I first called the
ceiling -- 12% of predicted codes sit within 0.06 of a rounding boundary, so
88% looked like agreement. **It is not the ceiling, and the test that says so
is stratifying by confidence:**

```
  |frac from a boundary|   0.00-0.05  0.05-0.15  0.15-0.25  0.25-0.35  0.45-0.50
  exact match               89.33%     89.27%     89.25%     89.29%     81.08%
```

A correct mapping has to approach 100% at the confident end. It is **flat at
89.3%**, so about one position in nine is mapped wrong for reasons that have
nothing to do with my rounding -- and the disagreements are spread to +-3 and
beyond, where a rounding flip can only ever be +-1.

Scanning all 512 cycle phases against all 16 row rotations -- 1024
combinations -- tops out at **89.28%**. So it is not the phase and it is not
the row order.

⛔ **The 53% "vendor effective weight error" this produced is retracted before
it was used.** It is dominated by the 11% of mis-mapped positions, not by the
vendor: on `blk.3.attn_q`, `s(P - z)` against the reference is 14.18% while
`s(Q - z)` through this mapping is 50.37%, and the gap is the mapping.

🔑 What is banked: the block is 16 output channels and contiguous, the cycle is
512 codes, the row of each position is known. What is not: the k index inside a
row's run.

### 🏁 The weight layout, solved — and the k index came out the same way the row did

The 11% that phase and rotation could not fix was the k index, and guessing it
was never going to work. Solving it does, and by the same move that gave the
row: **at each slot the 128 blocks hand you 128 observed codes, and the row is
already known, so matching that vector against the 2048 candidate columns is a
fit with 128 samples and one answer.**

```
  best score        48/48       (a right k)
  runner-up         16/48       (chance, with this code distribution)
  margin >= 10      32712 of 32768 slots
  distinct k        2048 of 2048, each chosen 14..18 times -- a bijection
```

**Held out properly**: solved on blocks 0..47, scored on rows 768..2047 which
took no part in the fit --

```
  all codes                     98.79%
  codes away from a boundary    99.72%
  Q - P disagreements           -2: 851   -1: 13329   +1: 13506   +2: 1067
```

which is the rounding-boundary signature and nothing else. My own earlier
"solved" claim read 88.46% and was flat at 89.3% across every confidence bin;
this one climbs to 99.72% at the confident end, which is what a correct mapping
has to do.

⚠ **And the map has to be solved on a tensor the vendor did NOT transform.**
Caching it by shape and letting the first tensor of that shape fill the cache
put `blk.0.attn_q` (rho 3.507) in charge of the 2048x2048 map, and the table
came back at 279%. A mapping fitted to predictions that are wrong fits nothing.
The anchors are `blk.3.attn_q`, `blk.3.attn_k`, `blk.6.ffn_gate`.

⛔ `ffn_down` (k = 8192) is not this layout: 16.5%, the noise floor.

### 🏁 So the vendor's own codes, scored

Over the 41 tensors whose `rho` is within 5% of 1 -- the ones where the
reference IS what the vendor quantised:

```
  vendor's own stored codes    17.577%
  charsiu group 1024           13.946%     <- what ships
  llama.cpp q4_0, group 32      8.856%
```

and `blk.3.attn_q` alone reads **15.859%** here against the **15.843%** that
came out of applying the vendor's `(scale, zero)` to charsiu's own rounding.
Two independent routes, one number.

### ⚠ AWQ's exponent: the minimum is per model, and 0.5 is past it on both

Yesterday's sweep was one model. Llama-3.2-1B, its own calibration file
(113 tensors), same corpus and length:

```
  off     0.20    0.25    0.30    0.35    0.40    0.50    0.65
 52.34   43.86   46.98   52.55   50.58   52.95   68.26   92.42
```

Llama's minimum is **0.20**, not qwen3's 0.35, and at **0.5 AWQ is worse than
not running it at all** -- 68.26 against 52.34. The curve is not clean in the
middle either (0.30 sits worse than 0.35), which is the same resolution limit
as the grid that ranked the wrong cell.

🔑 **So "use 0.35" was one model's answer and is withdrawn.** What both models
support is narrower and more useful: the exponent has to be swept per model,
and 0.5 -- the value every experiment in this tree used -- is on the wrong side
of the minimum on both models tested, badly so on one.

### 🏁 The calibration, identified — and it says the same thing the ppl sweep did

With the layout solved, the vendor's transformed weights are readable, so the
factor comes out by division: `c_j = median_i(w'_ij / w_ij)` over the larger
half of each column. The residual after dividing it back out is 15.9 to 25.9%,
which is int4 quantisation and not much else, so the per-column form is most of
the transform.

**And `c` is activation-aware.** Against `mean|x_k|` recorded by charsiu's own
calibration pass on a 562-byte passage -- a completely different calibration
set from theirs:

```
  corr(log c, log mean|x|)   blk.0.attn_q +0.846   blk.1.ffn_up +0.822
                             blk.0.ffn_up +0.792   blk.2.attn_v +0.775
                             blk.3.attn_q -0.071   <- the untransformed one
```

That is AWQ's direction: the columns meeting large activations are scaled UP,
so they get more of the int4 grid.

**The exponent is per tensor, and it is small.** Fitting
`log(c/gm) = alpha * log(x/gm)`:

```
  rho    1.000  0.975  1.018  1.549  1.684  2.121  2.949  3.507  22.317
  alpha -0.001  0.038  0.032  0.057  0.093  0.120  0.142  0.175   0.401
  r     -0.071  0.662  0.442  0.810  0.611  0.792  0.817  0.846   0.822
```

Three tensors -- `blk.3/10/15.attn_q` -- come out at alpha 0.000 and r ~ 0,
which is the same three the `(max-min)/scale = 15` identity picked out. The
factor's range is bounded: `c/gm` runs 0.41 to 2.22 across every tensor
measured, which is charsiu's own `CHARSIU_NPU_AWQ_CLAMP` default of [0.5, 2.0]
almost exactly.

🔑 **This corroborates today's ppl sweep from a completely different direction.**
charsiu has used `CHARSIU_NPU_AWQ=0.5` since the factor was written. The vendor
never exceeds **0.401**, is usually **0.03 to 0.15**, and switches the method
off entirely on some tensors. The sweep found 0.5 past the minimum on both
models and actively harmful on Llama; the vendor's own file says they never go
near it.

⚠ The exponents here are fitted against MY calibration corpus, not theirs, so
the numbers are the vendor's transform expressed in my activation statistics.
The ordering and the magnitude survive that; a third decimal would not.

### ⛔ A per-tensor alpha search does not reproduce the vendor's choices

The vendor picks alpha per tensor, which is what published AWQ does -- a grid
search minimising the output error on calibration data. charsiu records
`mean|x_k|` and nothing else, so the cheapest version of that objective is
diagonal: `sum_j mean|x_j|^2 * sum_i (w_ij - what_ij)^2`. Before writing that
into npuquant.c, the question is whether it picks what they picked.

It does not:

```
  corr(vendor alpha, activation-weighted search)  -0.0197
  corr(vendor alpha, unweighted search)           -0.0556
```

Eighteen tensors, no relationship at either. So the objective charsiu can
afford is not the one they used, and implementing the search would have been
building on an unvalidated premise.

🔑 **Two things the run does support, and they are the useful half.** The
unweighted search picks alpha 0.00 on thirteen of eighteen tensors -- weight
error alone always prefers no smoothing, which is `CHARSIU_NPU_AWQ_CLIP`'s
lesson arriving from a third direction. The activation-weighted one picks 0.05
to 0.35, which is the vendor's own range (0.00 to 0.40) even though the
per-tensor choices disagree. **So the objective has to be activation-weighted
and the magnitude is 0.1 to 0.3** -- which is what the ppl sweep said, and what
the vendor's file says, and now what an offline search says.

⚠ Why it probably fails per tensor: `mean|x_k|` is a diagonal statistic taken
from a 562-byte passage, and AWQ's real objective is the output error of the
whole matmul under the activation covariance. The magnitude survives that
approximation; the ranking does not.

### ⚠ ffn_down: the block is 32 KB and k is the outer loop, the rest is unresolved

`ffn_down` is 2048 x 8192 and scores 16.5% -- the noise floor -- under the
k = 2048 layout, so it is a different one. The same window-mean probe puts its
knee at **65536 codes = 32 KB**, and the excess there is 0.1852 against
`sd(z)/sqrt(16)` = 0.1780 and `sd(z)/sqrt(8)` = 0.2518. So a block holds
**16 rows and 4096 of the 8192 k**, not 8 rows and all of them.

**k is the outer loop.** Regressing the 256 block means on candidate groupings:

```
  rows 16*(b%128).., k-half b//128    r = +0.52     <- k outer
  rows 16*(b//2).., k-half b%2        r = -0.02     <- k inner
  rows 8b..8b+7, whole k              r = +0.02
  shuffled control                    r = +0.08
```

and every way of splitting k -- contiguous halves, even/odd k, even/odd 32-,
512-, 1024-, 2048-chunks -- gives the same +0.52, because a row group's mean
does not depend on which of its k are in the block. The statistic can see the
row grouping and is blind to the k split.

⚠ **+0.52 is not +1.0000, and I am not calling this solved.** Assigning each
block to its nearest (row group, k half) slot puts 50 of 256 on the identity
and uses only 177 distinct slots, with a mean error of 0.00039 against a slot
spread of 0.186 -- so where it lands it lands hard, and where two groups have
close means it cannot choose. The k = 2048 case had `r = 1.0000` and no
ambiguity at all; this needs the code-level fit, not the mean-level one.

What is banked for `ffn_down`: block 32 KB, 16 rows x 4096 k, k outer. What is
not: the block order and everything inside the block.

### 🏁 ffn_down solved too, and the whole file is readable

The mean-level fit could not choose between block orders for `ffn_down`, so the
code-level one did. Assuming its block is the same 16-row shape with k split in
two, each of the 256 blocks was searched over the 128 row groups and 2 k halves
by matching its codes:

```
  best match   median 93.43%      runner-up  median 18.07%
  all 256 blocks over 80%,  256 distinct slots used -- a bijection
```

⚠ **And the order is neither of the two I would have written down.** It is not
`(row group, k half)` and not `(k half, row group)`: it is **64 row groups at a
time**, each superblock doing k-half 0 for all 64 and then k-half 1 for all 64.

```
  block   0..63    ->  groups  0..63, k half 0
  block  64..127   ->  groups  0..63, k half 1
  block 128..191   ->  groups 64..127, k half 0
```

**The first twelve blocks agree with all three orders**, which is exactly why I
called it the identity from a twelve-row print and got 54.29% where the right
one gives **98.99% on confident codes**. Look at where the candidates diverge,
not at where they agree.

### 🏁 So the vendor's own codes, over every tensor type

```
                                    weight error
  vendor's own int4 codes              17.711%
  charsiu group 1024, symmetric        13.981%   <- ships
  llama.cpp q4_0, group 32              8.864%
```

43 tensors -- the ones whose scale still satisfies `(max - min)/15`, so the
reference is what they quantised. Adding `ffn_down` moved the vendor's figure
from 17.577% to 17.711%, which is the kind of agreement that says the new
layout is the same quantiser and not a new fit.

### 🏁 The empty cell, filled -- and the first attempt at it was wrong

With the layout solved the vendor's weights can be written into a gguf and
run, so the comparison stops being a weight norm and becomes a perplexity.
Three f16 files differing only in the 43 matrices whose scale still satisfies
`(max - min)/15` -- same tokenizer, embeddings, head and norms in all of them,
and no quantiser running at inference:

```
  A  the reference weights, untouched       19.8844
  D  llama.cpp q4_0, group 32               20.0010    +0.59%
  C  charsiu int4, group 1024               21.2288    +6.76%
  B  the vendor's own stored int4 codes     22.1006   +11.14%
```

**The vendor's four-bit weights cost 11.1% where charsiu's cost 6.8%**, and
that is the same ordering the weight error gave -- 17.58% against 13.98%
against 8.86%. Two independent metrics, one answer.

⛔ **The first version of this read 1701 and was my own mistake.** Replacing
all 112 matrices with `s(q - z) / c`, where `c` is the calibration recovered
by division, gives a perplexity of **1700.98**. The median per-tensor weight
error of what went into that file was 18.3%, and charsiu's own int4 at 13.9%
scores 41 -- so the number was not credible and the question was whether 18.3%
is simply that expensive.

🔑 **The control answered it in one run.** A fourth file, the reference plus
Gaussian noise scaled to the SAME per-tensor relative error, scores **32.10**.
So the magnitude is worth 32 and the structure is worth 1701: the recovered
`c` is wrong in a way a Frobenius norm barely charges for and a forward pass
charges enormously -- a column scaled by the wrong factor is a systematically
wrong channel, not a small perturbation.

That is this tree's own recurring lesson arriving again from a new direction:
**weight error and functional error are different things**, which is why
`CHARSIU_NPU_AWQ_CLIP` minimises the first and made the second worse.

⚠ So the cell is filled for 43 of 112 matrices. The other 69 need the vendor's
actual calibration, not one recovered by dividing by a reference.

### 🏁 The calibration is not stored as a vector -- it is folded into the norm

Their runtime has to know `c` to divide the activation by it, so `c` must be in
the file. It is not: correlating a recovered 2048-vector against every f32
alignment of the header and the whole float/table/register region tops out at
**|r| = 0.20** where a hit would be 0.9 and the noise floor is 0.022.

**Because AWQ does not divide at runtime -- it folds `1/c` into the preceding
RMSNorm**, and that is exactly what the file shows:

```
                corr(vendor norm, ref)   corr(vendor norm, ref / c)
  blk.1 ffn            0.7777                    0.9945
  blk.9 attn           0.8406                    0.9914
  blk.2 attn           0.8916                    0.9922
  blk.3 attn           1.0000                    0.9905   <- c = 1 here
```

with the ratio `vendor_norm * c / ref_norm` at 0.9995 to 1.0000 throughout. And
the other half of the same fact: **q, k and v of one layer share `c`**, which
they must if it lives in `attn_norm` -- `corr(c_q, c_k)` is +0.996, +0.993,
+0.994, +0.990 across four layers.

🔑 **So nothing has to be recovered.** Taking their norms with their weights
makes `c` cancel by construction. Using the reference norms with their weights
is what scored 1701.

### 🏁 And a second gauge: a row factor on v and up, undone by o and down

`gate` and `up` share `ffn_norm`, so they share `c` -- but their `rho` are 4.30
and 22.32 in layer 1. The difference is a per-OUTPUT-ROW factor, and a row
factor is absorbed by the per-row scale, so it is invisible in the codes and
visible only in `s`. It has to be undone downstream, and only `up` can carry
one: `gate` goes through SiLU, which a scalar cannot pass.

```
  corr(log(up/gate), -log(down))   +0.9933      up * down   median 1.009
  corr(log(v/q),     -log(o))      +0.8387    (v/q) * o     median 1.024, 6%
  corr(log q,         log k)       +0.9978    <- no extra gauge on q, k
```

**So the whole pipeline is described**: one activation-aware factor per input
channel folded into the norm ahead of it, plus a row gauge on `v` and `up`
cancelled by the columns of `attn_output` and `ffn_down`.

### ⚠ The full-model rebuild is 58.76, and five tensors own it

Their norms with their weights, all 112 matrices and 32 norms:

```
  reference                      19.8844
  vendor, their norms too        58.7642
```

which is a long way from the 1701 the reference norms gave, and still a long
way from the 22.10 the 43 clean matrices gave on their own. The effective
weight error with the folding applied has a median of 21.5% and five tensors
above 40%:

```
  blk.1.ffn_up      450.67%      blk.15.ffn_up    72.53%
  blk.1.ffn_down     85.44%      blk.14.attn_v    44.07%
  blk.15.ffn_down    43.96%
```

`blk.1` is where the row gauge is most extreme -- `up` at rho 22.3 against
`down` at 0.298 -- so the residual is the gauge not being reconstructed
exactly, not the quantiser. **The 11.14% figure stands on the 43 matrices that
carry no gauge; the full-model number is not a measurement of their quality.**

### 🏁 The comparison widened to 91 matrices, and the ordering holds

Folding removed the need to recover `c`, so the same three-way comparison runs
over layers 3 to 15 -- 91 of the 112 matrices, every tensor type, with the
vendor's own norms in the vendor arm and the reference's in the other two,
because that is what each side's model actually is:

```
  the reference weights, untouched       19.8844
  llama.cpp q4_0, group 32               20.2695    +1.94%
  charsiu int4, group 1024               23.8090   +19.74%
  the vendor's own stored int4 codes     26.6062   +33.81%
```

**The vendor's excess is 1.71x charsiu's**, against 1.65x on the 43-matrix
subset -- two disjoint measurements of the same ratio.

⚠ Layers 0 to 2 are excluded and the exclusion is attributed, not assumed:
the full sixteen-layer file reads **58.76** and dropping those three takes it
to **26.61**, so they carry the excess. They are also where the row gauge is
extreme -- `blk.1.ffn_up` at rho 22.3 against `blk.1.ffn_down` at 0.298.

🔑 **And the layout is not what fails there.** Within a row, `V/W` should be
`c_j * r_i`, so any two rows' column profiles are proportional if the layout is
right -- and a gauge is exactly what that divides out. The median pairwise
correlation of those profiles:

```
  blk.3.attn_q   rho  1.000    0.011   <- untransformed: V/W is 1 plus noise
  blk.6.ffn_gate rho  0.975    0.030
  blk.0.attn_q   rho  3.507    0.710
  blk.1.ffn_gate rho  4.300    0.654
  blk.1.ffn_up   rho 22.317    0.614
```

The transformed tensors share a column profile at 0.61 to 0.71 where the
untransformed ones sit at 0.01 to 0.03, which is the layout being right and
`c` being real. What is not accurate enough is my ESTIMATE of the row gauge:
dividing both gauges out by least squares takes `blk.1.ffn_up` from 450% to
1645%, which is a fit to a quantity the noise dominates.

### 🏁 The ratio, measured three times on nested subsets

Adding layers back one group at a time, with charsiu's quantiser run over
exactly the same matrices each time so the comparison never drifts:

```
  matrices                       ref    charsiu    vendor    ratio
  43   (rho = 1 only)         19.8844   21.2288   22.1006    1.65
  91   (layers 3..15)         19.8844   23.8090   26.6062    1.71
  105  (all but layer 1)      19.8844   26.6654   32.1275    1.81
  112  (everything)           19.8844      --     58.7642     --
```

**The vendor's excess is about 1.7x charsiu's**, three times, on three
different sets of tensors.

🔑 **And the charsiu column is what says the early layers are not my mistake.**
Going from 91 matrices to 105 adds layers 0 and 2, and it costs charsiu
23.81 -> 26.67 as well as costing the vendor 26.61 -> 32.13. Both arms pay, so
those layers are genuinely more sensitive to four-bit weights; the difference
between the arms stays a ratio.

⛔ **Layer 1 alone is the 58.76.** Everything except layer 1 reads 32.13; with
it, 58.76. It is the layer whose row gauge is most extreme -- `ffn_up` at rho
22.3 against `ffn_down` at 0.298 -- and the gauge cancels at inference only if
both halves are reconstructed exactly. It is left out and said so, rather than
averaged in.

### 🔑 What charsiu can take from them: the factor a group can share

charsiu's AWQ is off by default and the reason is speed, not quality:
`charsiu_npu_matvec_group` refuses any tensor carrying a factor, so q, k and v
drop from one submit to three. The comment there states the reason exactly --
"two tensors with DIFFERENT factors need two different inputs" -- and the
vendor's file is what points out that the premise does not hold here.

**The factors in a group are the same factor.** The factor is built from
`mean |x_k|` over a calibration run; q, k and v read one activation, so their
recorded statistics are byte-identical -- checked on this model's own
calibration file, all sixteen layers, `q == k` and `q == v` exactly, and
`gate == up` as well. So the group can pack once and apply one factor once.

That is charsiu's version of what the vendor gets for nothing by folding `1/c`
into the RMSNorm ahead of the projection instead of scaling the activation at
all.

⚠ **Shape is not the test.** `attn_output` has the same `k` as `attn_q` and a
completely different input. The comparison is on the factor's values, through a
hash computed once at staging, because memcmp of 32 KB a tensor a call is
16 MB a token.

⚠ **Two things reading the diff caught, both real.** The first version let a
group through when entry 0 carried a factor and entry 1 did not -- entry 1's
weights were never scaled, so the shared input would have been multiplied by a
factor that belongs to somebody else. And the check dereferenced `ids[0]`
before the `!n` and bounds tests that were already there, which the loop it
replaced could not do because a loop body does not run at `n == 0`.

⛔ **`CHARSIU_NPU_AWQ_SHARE` is DEFAULT OFF and untested on hardware.** The
group path exists only on the board and the host cannot exercise it. What the
host says is that the knob is in the binary, that AWQ's perplexity is identical
with it on and off (35.2041 both ways, so nothing leaked into the CPU path),
and that arch_sanity is 8/8.

**The round it needs**: `CHARSIU_NPU_AWQ=0.2` with `AWQ_SHARE` 0 and 1, on the
board, same prompt. The tokens must be identical -- the arithmetic is unchanged
either way -- and the stage table should show the group path taken instead of
three single calls. If they are, AWQ stops costing decode and can be considered
for default-on, which is worth 16% of perplexity on Llama and 35% on qwen3.

### ⛔ AWQ's factor was never applied on the batched path, and nothing refused

Chasing whether AWQ could be default-on turned up a wrong-answer path that has
been reachable since batching shipped.

`npuquant` scales the weights by `kscale[k]` and leaves the inverse for the
caller to put on the activation. `kscale` appears **twice** in `npudev.c`:
`charsiu_npu_matvec` applies it, and -- since today -- a group whose members
share one factor. `npu_matmul_inner`, which is the batched path and takes the
w4 route by default, does not, and there was no refusal anywhere else:
`kscale` does not appear in `llama.c` at all.

So with `CHARSIU_NPU_AWQ` set, **decode was right and prefill was not**: scaled
weights multiplied by an unscaled input, which `charsiu_npu_add`'s own note
already prices at ppl 75.17 off against 65233808 on.

⚠ **And no measurement in this tree would have caught it.** `charsiu_ppl`
scores one token at a time unless `--batch` is passed, so every AWQ perplexity
ever recorded here went through the single matvec. The board's 40.83 is a
decode number and is not affected; a `charsiu_run` prompt is.

🔑 The note at `charsiu_npu_add` says "paths that cannot, refuse" and prices
the failure in the same paragraph. **The sentence was true about the intent and
false about the code** -- the refusal it describes had never been written. That
is the same shape as the guard that tested one of `tensor_grouped`'s four
clauses: a comment describing a check that does not exist reads exactly like
one describing a check that does.

The refusal is now written. It costs the batch and keeps the answer. Applying
the factor on that path is the better fix and needs the board, because the path
does not exist on the host.
