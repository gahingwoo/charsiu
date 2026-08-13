# charsiu

An open LLM runtime for the **RK3576 NPU on a mainline Linux kernel**, driving the
hardware through the mainline `rocket` DRM-accel driver with no vendor userspace in
the execution path.

**Status: day one.** There is no runtime yet. What exists is the instrument this is
being built from and the first thing it measured, both below. This file will say what
runs the moment anything does.

## On the name

A 叉烧炉 is the oven; 叉烧 is what comes out of it. [kiln](https://github.com/gahingwoo/kiln)
runs the **vendor** LLM stack on a mainline kernel and is the oven this was cooked
in: it is what makes a vendor `.rkllm` readable, what a live dispatch is captured
with, and the number this project has to beat. charsiu is meant to be the thing you
actually eat, an open runtime with nothing closed left in the path.

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
  board and model, and this file will carry both figures side by side as soon as there
  is one to carry.

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
