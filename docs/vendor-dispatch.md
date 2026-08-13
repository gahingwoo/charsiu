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
streams            21532, of which 8808 are convolutions and 12724 DPU only
distinct shapes    1061
weight bits        {16: 4940, 4: 3328, 8: 40}
M (rows per op)    {1: 3752, 32: 1108, 64: 856, 96: 728, 128: 672}
```

Two things stand out before any detail.

**M = 1 is on the NPU.** 3752 convolution dispatches are a single row and a single
output pixel. On the RK3588 the open stack measured that shape at about 82 times
slower than a batched one and gates it back to the CPU. The RK3576 vendor stack does
not: it runs the model's projections there and gets roughly 13 tokens a second on
Llama-3.2-1B.

**The runtime is mixed precision, per role.** Not one datatype for the model.

## Three roles, three precisions

| role | precision | shapes (ic, oc, M) | ops |
|---|---|---|---|
| projections | **int4** | 2048x1024, 2048x256, 2048x4096, 4096x1024, all M=1 | 3328 |
| attention | **fp16** | 2688..4064 x 64, M = 32 to 48 | 4940 |
| LM head | **int8** | 2048 x 8160, M=1 | 40 |
| no weights | none | oc=1, M=1..5 | 500 |

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

40 ops of 2048 x 8160 at M=1. Llama-3.2-1B's vocabulary is 128256, so the head is
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

## What this does not say

- **Nothing about speed.** The file says what is dispatched, not how long it takes or
  how much of the wall clock is NPU versus CPU. Only the board says that.
- **Nothing about the CPU side.** The runtime may still do sampling, the KV cache,
  RoPE, the norms, or anything else on the cores; 12724 DPU-only streams suggest a lot
  of elementwise work is on the NPU, but which ops those are is not decoded yet.
- **Nothing about correctness at M = 1.** That the vendor dispatches it is not proof
  the hardware is exact there. It is proof the vendor believes it is, which is a
  different and much cheaper thing to check on a board.
