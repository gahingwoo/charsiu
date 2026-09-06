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
