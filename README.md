# charsiu

An open LLM runtime for the RK3576 NPU on a mainline Linux kernel. It drives the
hardware through the mainline `rocket` DRM-accel driver, with no vendor userspace
anywhere in the execution path.

## Against the vendor, on the same board

Both runtimes have been run on this board at the same NPU clock with the CPU
pinned. Llama-3.2-1B is the one model this project has for both, and it is the
only comparison here where nothing is quoted.

```
  Llama-3.2-1B, one board, NPU 594 MHz both sides, CPU pinned at maximum

  decode     charsiu 1.39x faster at 594 MHz. It is 1.46x if the NPU clock is
             raised to 786, because their decode ignores that clock and ours
             does not, so the multiple is a function of the condition and not
             a property of either runtime.

  prompt     a crossover, not a ratio. charsiu is faster below about 250 of its
             own tokens and the vendor is faster above it:

                 our tok    charsiu    their tok    vendor     winner
                     102        799          135     932       charsiu 1.17x
                     202       1465          235    1529       charsiu 1.04x
                     302       2226          335    2145       vendor  1.04x
                     602       4654          635    4201       vendor  1.11x
                     852       6818          885    6027       vendor  1.13x

             Their chat template costs a constant 33 tokens at every length, so
             each row is the same input text. Their lead at 852 was 1.34x before
             the fp16 attention arm became the default, 1.16x after it, and
             1.13x now.

  quality    charsiu's stored weights score 1.4x to 2.6x better in perplexity
             than theirs, against the same f16 original both were quantised
             from. A direction, not a single number: the six cells do not agree
             on an ordering.
```

One model, one board. The decode and prompt rows are `docs/paper-evidence.md`
sections 1b, 1g and 1b-ii. The quality row is section 2 and is a reconstruction,
because their runtime refuses every route to its own accuracy: the logits call
fails on an uncompiled shape, token input returns token 0, and text prompts apply
a chat template we cannot reproduce.

The two runtimes cannot share a boot, since one needs `rocket` and the other the
vendor `rknpu`. The vendor column is a different boot, and the drift between
boots was measured on the same charsiu ladder at worst 2.2%.

## int4 or int8

Neither of charsiu's two weight formats wins outright, and they lose in different
places. On Qwen3 0.6B, int4 decodes at 26.31 tok/s with a 602 ms TTFT and scores
87% worse in perplexity than a plain q4_0 gguf. int8 decodes at 17.23 with a
543 ms TTFT and stays within 1.6%. int4 is the default because chat is a short
prompt and a long answer; `CHARSIU_NPU_W4V=0` selects int8 for the other shape of
work.

Perplexity is the only number here that survives a reboot. `tools/charsiu_ppl` is
deterministic to the last digit across boots, while everything else on this board
drifts about 3% between sessions and may only be compared inside one run. Board
time is the binding constraint on this project, so every speed claim needs its
baseline re-run beside it, while a quality regression can be caught weeks after it
lands.

## Install

```
curl -fsSL https://raw.githubusercontent.com/gahingwoo/charsiu/stable/scripts/charsiu-install.sh | sh
```

It fetches the source, checks whether the running kernel can drive the NPU, and
offers a kernel if it cannot. RK3576 NPU support is not upstream yet, so no
distribution kernel anywhere will bind this hardware. A kernel it installs becomes
the default boot entry; on an extlinux board the one already on the card stays
selectable five seconds into the boot, and on Armbian the previous kernel is kept
as `/boot/Image.previous`. Then it builds charsiu, fetches a model charsiu can
actually run, and asks it something so you can see it work.

`sh install.sh --dry-run` prints every action as `would ...` and changes nothing.
The read-only checks still run, which is the point: it shows what the installer
sees on your machine.

There are two channels. `stable` installs the runtime and the other modalities.
`dev` adds the board probes, the test corpora and `charsiu_ppl`, which is what any
measurement round needs.

## What runs today

`charsiu` picks its own environment: int4 weights, the K slice width chosen per
model, both NPU cores, and the CPUs held out of deep idle while the NPU is open.
`CHARSIU_STAGES=1` prints where a token goes.

```
                    decode tok/s                 TTFT ms         our prompt
                 charsiu  range          vendor  charsiu vendor  in tokens
  Qwen3 0.6B      26.02   25.68..26.03    24.85      699  468.6      110
  TinyLLAMA 1.1B  21.88   21.87..21.88    19.71      933  543.7      116
  Phi3 3.8B        6.69    6.68..6.72      6.58     3244 1829.1      116
  Gemma4 E2B       8.92    8.92..8.93      9.23     2180 1219.3      111
```

Gemma4 decodes 3.4% BELOW the vendor's figure there, and it has been above it in
other sessions; the board drifts about 3% between boots, so that row is level
rather than either. The other three are 1.7%, 4.7% and 11.0% above.

`board_vendor.sh`, `CHARSIU_BENCH_REPEAT=3`, performance governor, median of
three with the range beside it. Four things make this a weaker comparison than
the head-to-head at the top of this file, and they do not all push the same way.

Two favour them. Their column is a published point estimate with no N and no
spread, taken at maximum CPU and NPU frequency, where the head-to-head ran their
runtime on this board at 594 MHz. It is also their w4a16 row, the fastest decode
of the three quantisations `benchmark.md` lists; against their w8a8 row the same
charsiu medians would read tens of percent ahead instead of a few.

One favours us: ours is the int4 arm, the one that costs 87% in perplexity.
Picking their fastest quantisation and our least accurate one is a choice in
both directions at once, and neither side of it is stated in the table.

The fourth just makes it uncheckable. These are four models this project does
not have their runtime for, so none of them can be verified the way
Llama-3.2-1B was.

So the multiples here, roughly 0.97x to 1.11x, are not the 1.39x at the top of
the file and neither number is "charsiu's decode multiple". The top one is one
model measured against their running code; this one is four models against a
number they published under a faster condition.

The two TTFT columns are not at the same prompt length either, and TTFT is
convex in length, so part of every gap there is the length rather than the
runtime.

One line of device tree matters. Mainline clocks the NPU at 786 MHz and leaves the
rail wherever U-Boot put it, 750 mV, where the vendor's own table asks 800 mV of
that speed. At 750 the two cores get about one row in a thousand wrong when they
run together, so the runtime has to use them one at a time. Give the rail 800 mV
and both cores run on every batched call. charsiu reads the rail and the clock out
of sysfs and overlaps the cores only inside the vendor's envelope; its NPU report
says which it chose and why, and the decision itself is `tests/overlap_guard.c`,
which runs on a desk.

It loads llama, qwen2, qwen3, gemma3, gemma4, phi3 and smollm3 gguf files.

## It also sees, hears, and matches pictures to words

A vision tower read out of llama.cpp's `mmproj` gguf, on the same primitives. A
patch embedding is a convolution whose stride equals its kernel, which is a gather
into rows and one matmul, so nothing outside `(m, k, n)` is needed.

```
$ charsiu --image llama-logo.png "what animal is this?"
Llama.
```

On SmolVLM-256M a 512x512 picture is 1024 patches and 64 tokens, and the control is
that without the picture the same prompt makes one up.

```
$ charsiu_whisper ggml-tiny.en.bin --transcribe --audio jfk.wav
 And so my fellow Americans ask not what your country can do for you, ask what you
 can do for your country.
```

Whisper comes out of whisper.cpp's own container: the mel spectrogram, the audio
encoder, and a decoder with the first cross attention in this tree. Every stage is
diffed against numpy on the real weights, the spectrogram at 1.7e-05, the encoder
at 1.8e-04 over 576000 values, the decoder's logits at 3.8e-05 with the same
argmax.

```
$ charsiu_clip clip-vit-base-patch32.gguf --image logo.png \
      --text "a drawing of a llama" "the statue of liberty" "a dog on grass"
  0.2537  a drawing of a llama
  0.1743  the statue of liberty
  0.1533  a dog on grass
```

## Reading the vendor's model file

`tools/rkllm_regcmd.py` reads the NPU register command streams straight out of a
`.rkllm`, on a desktop, with no board and no vendor runtime running. The file
carries the programs the closed stack submits, so what it asks the hardware to do
can be read directly. For Llama-3.2-1B it reports 21532 streams, 8808 of them
convolutions, and weight widths of fp16 for 4940, int4 for 3328 and int8 for 40.

Read against that model's own dimensions, every output width in the file is half
of a projection: 2048 to 1024 is half the Q projection, 2048 to 256 half the K and
V, 2048 to 4096 half the FFN up. The vendor splits each projection across the two
cores by output channel. The rest is in
[docs/vendor-dispatch.md](docs/vendor-dispatch.md).

## Scope

Target: RK3576, Radxa ROCK 4D, mainline kernel with `rocket`. In scope: the matmul
and the ops an LLM needs, on the NPU, in the precisions the vendor uses, plus a
frontend that runs a real model end to end. Out of scope: any vendor library in
the execution path, and any claim about an SoC this has not been run on.

The target is the vendor's own number on the same board and model. Decode meets
it, prefill does not yet, and the table at the top says by how much.

## Why this exists, and why it is not a port

Two open stacks already drive a Rockchip NPU for LLM work, both on the RK3588:
[rocket-userspace](https://github.com/gregordinary/rocket-userspace) with
[ggml-rocket](https://github.com/gregordinary/ggml-rocket) on the mainline `rocket`
driver, and [iwagumi](https://github.com/fukumori/iwagumi) on the vendor `rknpu`
ioctl. Their measurements are collected in
[rockchip-npu-notes](https://github.com/gregordinary/rockchip-npu-notes) and are
worth reading before starting anything here.

Their central finding is that the NPU is a prefill engine and decode belongs on
the CPU. ggml-rocket's own words are "Decode (M=1 GEMV) is forced to the CPU, ~82x
slower on the NPU". A feature height below four computes wrong output at every
dtype, which is the M=1 case, and quantisation does not speed prefill up because
the pipeline sits at a dispatch and DMA floor rather than a MAC one. Both
repositories qualify that last one, and this file used to drop the qualifier:
ggml-rocket says quantisation buys RAM rather than prefill speed "at this
operating point", and rockchip-npu-notes calls it "bottleneck-conditional rather
than a permanent silicon law".

On the RK3576 the vendor does the thing that finding says not to do. Of the
convolution dispatches in its Llama-3.2-1B model file, 3752 are M=1, one row and
one output pixel, and they are the model's own projections. The vendor ships that
and gets about 13 tokens a second on this board.

So the question here is not how to port the RK3588 result. It is what the RK3576
actually does and whether an open runtime can match it. The two chips are not the
same machine: 1 MiB of CBUF against 384 KiB, two cores rather than three, a 16 bit
task number, and a different weight tile stride. A negative result measured on the
other chip is a hypothesis here, not a conclusion.

## Prerequisite

RK3576 support in `rocket` is not upstream yet. It is on the list as
[PATCH v11](https://lore.kernel.org/all/20260831081956.84871-1-gahing@gahingwoo.com/),
with an Acked-by from Conor Dooley on both dt-bindings, a Reviewed-by from Abel
Vesa on both pmdomain patches, and a Tested-by from Igor Paunovic on each of the
three reset-race patches. His testing on RK3588 reproduced, once in 102 induced
resets, an inference that signalled success while its output buffer was never
written, and only on the arm without those patches.

The driver and the Mesa work it comes from are in
[linux-rk3576-npu](https://github.com/gahingwoo/linux-rk3576-npu), which is where
the RK3576 register knowledge in this repository was established.
[kiln](https://github.com/gahingwoo/kiln) runs the vendor RKLLM stack on a
mainline kernel; it is the measuring stick and the capture harness.

## On the name

Char siu is Cantonese barbecue pork, eaten across Guangdong, Hong Kong and
Malaysia. A kiln is the oven it is roasted in. kiln runs the vendor stack and is
that oven here: it makes a vendor `.rkllm` readable, captures a live dispatch, and
produces the number this project has to beat. charsiu is what comes out of it.

## Licence

GPL-2.0-or-later. `LICENSE` carries the GPL version 2 text; the "or later" is what
lets this be combined with GPL-3 code if it ever links any.

## The record

How the int4 layout, the output surface, the accumulator read order and the
batching were read off the hardware, round by round, is in
[docs/lab-notebook.md](docs/lab-notebook.md). It is long because it is the
evidence. What the vendor's own model file says is in
[docs/vendor-dispatch.md](docs/vendor-dispatch.md), and the numbered claims with
their protocols are in [docs/paper-evidence.md](docs/paper-evidence.md).
