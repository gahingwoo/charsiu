# What the vendor LLM runtime asks the RK3576 NPU to do

Everything here is read out of one file, `Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm`,
with `tools/rkllm_regcmd.py`. No board, no vendor runtime, no tracing. A `.rkllm`
carries the register command streams its runtime submits, so the closed stack's
dispatch plan can be read on a desktop.

Register names are from the RK3576 convolution work in
[linux-rk3576-npu](https://github.com/gahingwoo/linux-rk3576-npu), which established
them against the same hardware.

**Addresses are placeholders in a static file.** Nothing here reports one, and no
claim below depends on one.

## The whole model, by the numbers

```
streams            13224, of which 8808 are convolutions and 4416 DPU only
distinct shapes    1061
weight bits        {16.0: 4940, 4.0: 3328, 0.0: 500, 8.0: 40}
M (pixels a dispatch) {32: 1628, 64: 1248, 80: 776, 96: 736, 128: 680}
```

⚠ **Both of those lines used to say something else, and both changes were bugs in
the reader rather than in the model.** The stream count was 21532 with 12724 DPU
only, because target `0x0401` was missing from the table and an unknown target ENDS
a run, cutting every op into three. And the M histogram used to be headed by
`{1: 3752, ...}` because it read the ROW count, `0x102c`, instead of the pixel
count, `0x1034`. Do not quote either old figure.

Two things stand out before any detail.

**The vendor batches, and it batches almost everything.** 2816 of its 3328 int4
projection dispatches are above M = 1, which is 85%, at widths of 16, 24, 32, 40,
48, 64 and 80. This document said the opposite for a long time, and the reason is
worth keeping: an int4 projection is emitted as a ONE ROW image M PIXELS WIDE, so
its row count is 1 whatever M is. A reader that takes the row count gets 1 every
time and concludes there is no batch. The fp16 attention is emitted as an M row
image instead, and all 4940 of those have rows == pixels, which is why the two
readings agreed everywhere the mistake was invisible.

**It never hands one dispatch a K of 1024.** Every int4 dispatch is K = 2048 (2688
of them, 81%) or K = 4096 (640, 19%). Llama-3.2-1B is 2048 wide with an 8192 FFN, so
K = 2048 is the whole hidden size in one dispatch and K = 4096 is half the FFN.

**The runtime is mixed precision, per role.** Not one datatype for the model.

## Three roles, three precisions

| role | precision | shapes (ic, oc, M) | ops |
|---|---|---|---|
| projections | **int4** | 2048x1024 (896), 2048x256 (896), 2048x4096 (896), 4096x1024 (640); M = 1 to 80 | 3328 |
| attention | **fp16** | 2688..4064 x 64, M = 32 to 48 | 4940 |
| LM head | **int8** | 2048 x 8160, M = 1 and above | 40 |
| no weights | none | oc=1 | 500 |

The int4 M ladder in full, over all 3328: M = 1 (512), 16 (384), 24 (64), 32 (512),
40 (320), 48 (384), 64 (384), 80 (768). The widest is 80 and it is also the most
common. charsiu's own batched prefill chunks at 32.

### The projections are split across the two cores

Llama-3.2-1B has a hidden size of 2048, an FFN of 8192, 32 query heads and 8 KV heads
at head dim 64, so its projections are 2048x2048, 2048x512 and 2048x8192 with an FFN
down of 8192x2048. Every output width above is exactly **half** of one of those:

| dispatch | half of |
|---|---|
| 2048 to 1024 | Q, 2048 to 2048 |
| 2048 to 256 | K and V, 2048 to 512 |
| 2048 to 4096 | FFN up and gate, 2048 to 8192 |
| 4096 to 1024 | FFN down, 8192 to 2048, split on **both** axes |

The RK3576 has two NPU cores. The vendor splits each projection by output channel and
gives half to each, which is the same fan-out axis the RK3588 open stack uses across
its three cores.

### Attention is the fp16 half, and its K is the KV length

The fp16 shapes have oc = 64, the head dimension, and ic running 2688, 3104, 3136,
4000, 4032, 4064: a sequence of KV cache lengths. M is 32 to 48, a block of query
rows. So attention runs as blocked matmuls on the NPU in fp16 while the projections
run in int4. The precision split follows the operand, not the layer.

### The LM head is int8

40 ops of 2048 x 8160, and 32 of the 40 batch. Llama-3.2-1B's vocabulary is 128256, so the head is
split into pieces of 8160 output channels each, in int8 rather than the int4 the
projections use.

## One dispatch in full

A 4096 to 1024 int4 projection, the FFN down half, at M = 1:

```
CNA 0x1004 0000000e   mode, the same value the conv path uses
CNA 0x1014 00000009   stride 1
CNA 0x1018 40000404   the NON-split CBUF pair: one row window
CNA 0x101c 00200000   2097152 weight bytes = 4096 * 1024 / 2, so 4 bits per weight
CNA 0x1020 00000800   2048 bytes per kernel = ic / 2
CNA 0x1024 000003ff   oc - 1 = 1023
CNA 0x1028 00800fff   surf * rows = 128, ic - 1 = 4095
CNA 0x102c 00000000   one row, one row window
CNA 0x1034 00000000   one output pixel
CNA 0x103c 00800000   surf = 128 = 4096 / 32, the int4 feature atom
CNA 0x1040 10000000   the non-split pair's partner
```

Two of those are worth keeping.

`0x1018` and `0x1040` carry the **non-split** pair, and the layer is one row window,
which is what the rule derived from 87 compiled `.rknn` in the driver work says they
mean. The vendor's LLM dispatches agree with it.

`surf = 128 = 4096 / 32` says the int4 feature atom is **32 channels** per 16 byte
unit, against 16 for int8 and 8 for fp16. That matches the RK3588 tile-layout table
exactly, which is a useful independent check that the IP-level layouts are shared and
only the machine parameters differ.

## The 4416 DPU-only streams, and why "six programs" was an artifact

⚠⚠ **THE TABLE BELOW IS SUPERSEDED AND IS KEPT AS A RECORD OF THE BUG, NOT OF THE
MODEL.** Its counts sum to 12724, which is the pre-fix DPU-only total: it was built
when target `0x0401` was missing from the reader's table, so every op was cut into a
CNA fragment, a lost middle and a DPU fragment. The "six kinds" were the fragments,
carrying whichever registers happened to fall on each side of the cut, and grouping
them looked like a taxonomy.

Re-run against the same file with `0x0401` known, there are **4416** DPU-only
streams and, grouped by `0x4010` and `0x4050`, exactly **one** kind -- both
registers read 0 in all of them, and their lengths are 31, 33, 35, 94 and 96 words.
Whatever these programs are, those two registers do not separate them, and no
replacement taxonomy is offered here because none has been established.

The superseded table:

| 0x4010 | 0x4050 | entries | count | shape (ow, oh, oc) |
|---|---|---:|---:|---|
| a0000002 | 00023333 | 71 | 8268 | 1 x 1 x 1024 |
| 40000002 | 00020000 | 87 | 4096 | 4 x 32 x 16 |
| a0000005 | 00020000 | 31/33/35 | 320 | 96 and 64 wide |
| 80000000 | 00027777 | 71 | 40 | 1 x 1 x 8160 |

The first is the decode path's elementwise work: one pixel, 1024 channels, which is
half the hidden size, matching the same two-core split the projections use. The last
is the LM head's 8160 channels. So the vendor keeps the norms, the residual adds and
the activations on the NPU rather than handing the vector back to the cores between
projections.

Note the high nibble of 0x4050 is 0 in all of them, where a regular convolution on
this hardware carries 8. The DPU is being driven in a different mode here, and which
mode is not decoded yet.

## Attention is precompiled per KV length, one program every 32 tokens

The 2908 fp16 attention dispatches have oc = 64, the head dimension, and their input
channel count is the KV cache length. Those lengths are:

```
32, 64, 96, 128, ... 4032, 4064, 4096      128 buckets, every one a multiple of 32
```

That is the whole design in one line. The vendor cannot build a register program at
run time, so it **ships one per 32 tokens of context**, up to 4096, per attention
matmul. It is why a 1.3 GB file holds 13224 dispatch programs for a 1.2 GB model.

An open runtime does not have to do this. Building the register stream is what the
Mesa driver already does per operation, so charsiu can emit the exact geometry for
the KV length it actually has, and neither pay for 128 buckets nor be capped at 4096.

**And the M ladder goes below four.** The attention dispatches use M = 2, 3, 4, 5, 7
and on up to 128. The RK3588 notes record that a feature height below four computes
output uncorrelated with the reference, on every datatype, and that it is a hardware
constraint rather than a stride bug. The RK3576 vendor dispatches heights of two and
three as a matter of course. Either the constraint is not present on this chip or the
vendor is configuring around it, and that is a question a single board round can
answer.

## What this does not say

- **Nothing about speed.** The file says what is dispatched, not how long it takes or
  how much of the wall clock is NPU versus CPU. Only the board says that.
- **Nothing about the CPU side.** The runtime may still do sampling, the KV cache,
  RoPE, the norms, or anything else on the cores; 4416 DPU-only streams suggest a lot
  of elementwise work is on the NPU, but which ops those are is not decoded yet.
- **Nothing about correctness at M = 1.** That the vendor dispatches it is not proof
  the hardware is exact there. It is proof the vendor believes it is, which is a
  different and much cheaper thing to check on a board.

## The 4940 fp16 streams are attention, and here is what they write

Read 2026-09-05. The fp16 half of the file was counted long before this and
called "its fp16 attention" on the strength of the shape of the thing; this
section is what identifies it and what it takes to emit one.

### It is attention

2908 of the 4940 fp16 dispatches carry `oc = 64`, which is this model's
`head_dim`. Their `ic` walks in steps of 32 -- the 2 byte feature atom -- and M
is chosen so the input surface lands just under 4096 every single time:

```
ic 2688 M 48 surf 4032    ic 2848 M 46 surf 4094
ic 2720 M 48 surf 4080    ic 2880 M 45 surf 4050
ic 2752 M 47 surf 4042    ic 2912 M 45 surf 4095
ic 2784 M 47 surf 4089    ic 4032 M 32 surf 4032
```

So `ic` is not a dimension of the model at all: it is a **context length**
rounded up to the atom, which is why `ic = 1312` (41 * 32) matches nothing in
the config. The remaining `oc` values -- 32, 96, 128, 160, 192, 224, 256, each
sixteen times, which is the layer count -- are the `q.K^T` half, where `oc` is
the growing context instead.

Both halves of attention are on the NPU. **4940 of 8808 convolutions, 56% of
everything the runtime submits, is attention** -- against 3328 int4 projections.

### What an fp16 stream writes that an int4 one does not

Every register written by an fp16 stream is also written by an int4 stream and
the reverse, over all 8268 of them. The **register set is identical**; only
values differ. Six are constant within fp16 and differ from int4:

| register | fp16 | int4 |
|---|---|---|
| `CNA 0x100c` | `0x20200120` | `0x20600120` x1920, `0x00600120` x1408 |
| `CNA 0x1094` | `1` | `0x60` x896, `0x80` x896, `1` x512 |
| `CNA 0x118c` | `0` | `0x004f004f` x768, `0` x512, `0x001f001f` x512 |
| `DPU 0x401c` | `1` | `0x60` x896, `0x80` x896, `1` x512 |
| `DPU 0x4020` | `0` | `0x4f` x768, `0` x512, `0x1f` x512 |

⚠ A sixth, `CNA 0x1110`, looked like a differing constant and is **not one**:
`job.c` emits `job->weight_addr` there, and an address in a static file is an
unpatched placeholder. It is listed here only so the next reader does not count
it again.

These four are the **surface geometry** registers, not anything about weights.
`job.c` emits them as `inw * rows`, `ow * rows`, `((inw - 1) << 16) | (inh - 1)`
and `ow - 1`. Holding them at 1, 0, 1, 0 therefore says the fp16 regime
describes its window as **1 x 1**, with the count carried elsewhere -- which is
consistent with the round 380 note's observation that the vendor writes M
exactly into `0x1098`.

⚠ An earlier reading of this called the four "the weight group count and its
minus one, collapsed because a 16 bit weight carries its own exponent". That is
WRONG and the emitter says so: they are the window, not the weights. The story
was invented to fit four numbers before anyone looked at what writes them.

⚠ This is what round 380 hit from the other side: it copied fp16's `0x1094`,
`0x1098` and `0x118c` onto an int4 op, the board said no, and the op the values
came from could not be named at the time. It can now, and the round's
conclusion stands.

### The weight buffer is dense

Closed forms, exact over all 4940:

```
CNA 0x101c = ic * oc * 2      total weight bytes
CNA 0x1020 = ic * 2           bytes per output channel
CNA 0x1090 = ic / 8           the 2 byte feature atom is 8
CNA 0x107c = ic - 1
CNA 0x1024 = CORE 0x3020 = DPU 0x402c = oc - 1
```

`0x101c` and `0x1020` are the **true byte count** in every regime, checked
across all 8308 dispatches that write them: 2.000 x `ic*oc` for fp16, 1.000 for
int8, 0.500 for int4. No unit-of-two anywhere.

So an fp16 weight buffer is exactly `oc` kernels of `ic` fp16 values, with no
padding and no tiling overhead -- which is what `charsiu_weight_bytes()`
already returns for `CHARSIU_FP16`.

### What is still unknown

The byte **order inside a kernel**. `charsiu_weight_kgroup()` returns 32 for
anything that is not int4, which is int8's number inherited by fp16 as a guess,
and `charsiu_weight_ngroup()` is in the same position. Nothing in a static
model file can settle it: the fp16 "weights" here are the runtime KV cache, and
address registers in a static file read 0.

⚠ And there is no shortcut on a desk. The 30 vendor `.rknn` in the driver
repository contain no fp16-weight convolution, and rknn-toolkit2 is not
installed on this host, so one cannot be compiled here either. The tile has to
be walked on the board with sparse maps, the way int4's was.

## And the answer to the one thing the file could not say

The section above ends "the tile has to be walked on the board with sparse
maps". It was, on 2026-09-05, and the answer is that there is no tile:

**`slot = n * k_eff + k`, which is plain dense, output channel major.**

`npu_fp16_test --slots` writes 1.0 into one slot of the weight buffer at a time
and reads back which output channel it lands in and which `A[k]` it multiplied,
with `A` a ramp. That is the hardware's own mapping read directly, with no
candidate layout in the way -- which matters, because `--map`, the obvious
instrument, places the weight at the (n, k) that a CHOSEN layout picks and so
can only ever light up where we were already right.

Three independent sweeps at K=16 N=8, eighteen firing slots between them, and
every one satisfies that formula exactly.

### Getting there took three register findings, each named by the board

  1. every output `0x80808080`, the int8 zero point in all four bytes: the
     OUTPUT STAGE was int8. `WIDE(bit)` is `w4_dpu || wide8` and `w4_dpu` was
     `w4a16` alone. An fp16 job wants the same stage for the same reason w4a16
     does, and every register it switches lands on the vendor's own fp16 value.
  2. a weight of 1.0 against `A[0] = 1.0` returning **3600**, and `A[8] = 9.0`
     returning **4320**: 3600 is `0x3c * 0x3c` and 4320 is `0x48 * 0x3c`, the
     HIGH BYTES of the two fp16 patterns multiplied as int8. The output stage
     had moved and the MULTIPLY had not -- `CORE 0x3018` takes the
     `0x10000200` form for fp16, which is what the vendor writes in all 4940.
  3. four more read straight off the file and exact over all 4940:
     `CNA 0x1030 = (ic*2) << 16`, `CNA 0x1090 = ic/8`,
     `DPU 0x4028 = oc/4 - 1`, `DPU 0x4030 = ((oc-1) << 16) | 0x310`.

### What is still wrong

Coverage, not layout. Only about 12 of the 128 products are accumulated -- six
channels of eight, two consecutive `k` each -- and which ones changes from run
to run with nothing else changed. The mapping is stable and correct in every
run; the set of products is not. Channel 0 fires at `k = 0, 1` every time.

⚠ That drift is not a reason to go back and try another layout. It looked like
one for an hour: the first slot sweep and an earlier `--map` run disagreed at
the same shape, which read as "one of these instruments is lying". Both were
telling the truth about different draws.

### The sparse probe was invalid, and the answer survived it

`--slots` leaves the weight buffer 99.99% zero, and this silicon has weight
sparsity, so a fetch that skips zero blocks would produce exactly the drift it
showed. `--bits` (every weight 1.0, `A[k] = 2^k`, so each channel's output is a
bitmask of which k reached it) returned **0xffff on every channel** in three
consecutive runs at K=16 N=8 and again at K=64 N=64 and K=256 N=64: coverage is
complete, and the drift was the probe.

`--bits` cannot answer the layout on its own -- with every weight 1.0 the sum
is identical under any permutation -- so `--holes` asks it on a dense buffer:
every weight 1.0 except one, and the channel that returns missing a bit names
both halves of the hole.

Across four runs and the two instruments, which have nothing in common:

```
holes2 (dense)   12/12 satisfy slot = 16n + k
slotsA (sparse)  12/12
slotsB (sparse)  12/12
first  (sparse)  12/12
TOTAL 48 points, 0 exceptions
```

Still open, and NOT a layout question: only 12 of 128 single-weight
perturbations register at all, on the dense buffer too, and the set moves
between runs. Every observation that carries layout information agrees.
