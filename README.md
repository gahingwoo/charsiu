# charsiu

An open LLM runtime for the **RK3576 NPU on a mainline Linux kernel**, driving the
hardware through the mainline `rocket` DRM-accel driver with no vendor userspace in
the execution path.

**Status.** charsiu computes a signed int8 matmul on the NPU, **byte exact** against a
CPU reference. It opens `/dev/accel/accel0` through the mainline `rocket` driver, packs
the operands into the hardware's tile layouts, builds the coefficient buffer, emits the
register stream and submits it. No Mesa, no vendor runtime, nothing borrowed at run
time.

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

⚠ The first two lines are confirmed by `--kpair`, which reaches word 0 from
bytes 0 to 7 and again from 256 to 263. **The third is in doubt.** Byte 384 and
byte 512 both light, and the layout read below predicts the second half is live
throughout, so either the count is wrong or `--bmap` was reading a window too
small to see it, which is a mistake this probe has now made twice.

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
