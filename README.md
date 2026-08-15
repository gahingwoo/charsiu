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

**A note on that last row, added when the tooling grew a way to see it.** Its
reference has only 7 distinct values across the 1024 channels, so most of those 1023
matches are channels where both sides hold the same common value rather than channels
that were computed. The 64 wide rows above are the strong evidence: 64 distinct
reference values, 64 of 64 exact. A matching channel is not a computed one, and this
table now says which is which.

The two things that had to be understood to get there are worth stating because both
were wrong in this file before:

- **both operands are signed bytes.** The weight is stored biased by `-0x80` and so is
  the input; storing the input raw makes a byte of 168, meaning +40 against a zero
  point of 128, arrive as -88.
- **the output stage is** `out = clamp(max(requant, 0) + offset, -128, 127)`, an int8
  with a floor under it. The floor is a real fused ReLU and it does not need to be
  switched off: lifting the accumulator by 128 in the requant domain puts every value a
  signed byte can hold above it, and the offset takes the same 128 back, which is
  exactly Mesa's `out_zp - 0x80`.

It is not a runtime yet. What is left:

- **no model runs.** The matmul is correct; the KV cache, the sampler, the per token
  geometry and the CPU/NPU split are not written.
- the reference still requantises in float where the hardware uses an integer scale and
  shift. It agrees to the byte on everything measured so far, which does not mean it
  will at every scale.
- **int4's weight layout is still unknown.** Its registers are confirmed, read out of
  a vendor `.rkllm` by diffing its 3328 four bit streams against its 40 eight bit
  ones, and its output stage is byte exact on a probe that holds the MAC at zero. The
  LAYOUT is not: two rounds of full buffer patterns appeared to decode it and were
  withdrawn, because a probe that fills every byte cannot see a permutation. It is
  being read again with a sparse probe, one live nibble at a time, which is how the
  int8 layout was read.
- the runtime work has not started: no KV cache, no sampler, no per token geometry.

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
