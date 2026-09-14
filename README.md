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

  prompt     ahead from 27 through 602 of its own tokens, and level at 852.
             Five repeats a point, one boot, clock and governor pinned:

                 our tok    charsiu       range    their tok  vendor   verdict
                      27        331   330..332           60     413    1.25x
                      52        409   407..416           85     530    1.30x
                     102        750   745..767          135     932    1.24x
                     202       1395  1390..1402         235    1529    1.10x
                     302       2045  2030..2050         335    2145    +4.9%
                     452       2991  2974..3014         485    3195    +6.8%
                     602       4060  4055..4072         635    4201    +3.5%
                     852       5905  5892..5936         885    6027    level

             Their chat template costs a constant 33 tokens at every length, so
             each row is the same input text.

             A margin has to clear two things, and the first one is now
             MEASURED rather than quoted:

               the floor  their runtime needs their driver bound, so the two
                          columns cannot share a boot. Twenty readings of ONE
                          arm at 302 tokens on one boot, changing nothing,
                          spanned 2.5% of their median. That is the floor a
                          margin has to clear, and it is slightly larger than
                          the 2.2% this table used to quote.
               the spread  each row's own range, above. All eight clear their
                          own spread; the floor is what separates them.

             ⚠ 852 is +2.1% and does not clear the floor, so it stays level.
             ⚠ 602 is the thin row: +3.5% clears 2.5% by one point, and a
             slightly stricter bound puts it back to level. 302 and 452 clear
             comfortably.

             What changed: at 852 their lead was 1.34x before the fp16
             attention arm became the default, 1.16x after it, 1.13x once the
             int4 accumulator work was done, 1.05x once the softmax ran during
             the fence, and level once the int4 accumulator gather ran during
             it too. 302 and 452 moved from level to ahead when the attention
             threshold came down to 272, re-derived against the arm that ships.

             ⛔ This table has been overclaimed once. Every point estimate
             favoured charsiu then too, it was written up as "completely
             surpassed", and it was withdrawn the same day. What is different
             is that the floor is measured and that one row is still level and
             one is thin. Both of those are part of the result.

  quality    charsiu's stored weights score 1.4x to 2.6x better in perplexity
             than theirs, against the same f16 original both were quantised
             from. A direction, not a single number: the six cells do not agree
             on an ordering.
```

One model, one board. Every row above is a measurement with a round behind it:
decode is `docs/paper-evidence.md` sections 1b and 1g, the prompt ladder is
section 1k, and the quality row is section 2. The quality row alone is a
reconstruction, because their runtime refuses every route to its own accuracy:
the logits call fails on an uncompiled shape, token input returns token 0, and
text prompts apply a chat template we cannot reproduce.

### What the vendor column actually is

Their runtime was run here, not quoted from a table. What ran:

```
  Every md5 below was re-read off the board on 2026-09-14, not copied
  forward from an older note.

  runtime   librkllmrt 1.3.0, md5 78d6d4094a64ee7659bbafde6b07c408, 7.6 MB.
            The version is the library's own: it prints "rkllm-runtime
            version: 1.3.0, rknpu driver version: 0.9.8, platform: RK3576"
            at init.
  weights   Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm, 1.30 GB,
            md5 2d3962468e2e7c0d0571157f8c9eae71 -- the same file section 2
            scores. Its own last 29 bytes read "rkllm-toolkit version: 1.1.4",
            so it was produced by Rockchip's converter and is not a published
            artefact somebody else can diff against.
  harness   vendor_bench, md5 6350eaef99f472404b05bdb7a38af6e2, 14 kB, calling
            rkllm_init / rkllm_run / rkllm_destroy and reading the runtime's
            own RKLLMPerfStat rather than timing from outside.
  driver    their rknpu, built against this board's own mainline kernel.
  ladder    tests/board_ttft_curve.sh, vendor arm, board-logs r393: three
            readings a point, NPU 594 MHz on both sides, rail 800 mV, CPU
            pinned 2016/2208 with the userspace governor, context 1024 on both
            sides. Decode is r389 and r390.
```

⚠ **The harness's source is not in this repository.** `vendor_bench` was
written, built and deployed to the board, and only the binary survives. Its
identity is recorded above so the runs can be tied to it, but a reader cannot
rebuild it from here, and re-measuring the vendor column needs the board booted
into the vendor arm. That is a gap, not a result.

The two runtimes cannot share a boot, since one needs `rocket` and the other the
vendor `rknpu`. The vendor column is a different boot, and the drift between
boots was measured on the same charsiu ladder at worst 2.2%.

## int4 or int8

Neither of charsiu's two weight formats wins outright, and they lose in different
places. Qwen3 0.6B, one board, `-c 1024`, three runs a cell, perplexity on
`tests/corpus/long.txt` (md5 `4237c8fc3163a359fc21bde60c7b1d8b`), NPU 594 MHz:

```
                 TTFT ms            decode tok/s       perplexity
  int4           151 / 153 / 168    28.09 28.50 28.68    87.7503   +102%
  int8           210 / 215 / 226    16.09 16.37 16.37    44.1828     +1.8%
  the gguf       -                  -                    43.3981     --
```

The third row is charsiu's own CPU path on the same file with no NPU
requantisation, which is what the other two are a loss against. int4 decodes
1.74x faster than int8 and scores 1.99x worse; against the untouched gguf it is
2.02x. int4 is the default because chat is a short prompt and a long answer;
`CHARSIU_NPU_W4V=0` selects int8 for the other shape of work.

⚠ The figures this replaces -- 26.31 and 17.23 tok/s, 87% and 1.6% -- appear in
no board log and no evidence section in either repository. The two perplexity
percentages were close (1.6 against 1.8, 87 against 102) and the two decode
figures were not; all four are now measured here rather than carried forward.

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
at 1.7e-04 over 576000 values, and the decoder's logits with the same argmax.

⚠ Those three tolerances are the printed output of
`tests/whisper_encoder_cross.py` and `tests/whisper_decoder_cross.py`, which
parse the container independently of charsiu and are what anybody should re-run
to check them. They are reproducible but not recorded with their conditions in
either repository, unlike every speed number above, so treat them as "run the
script" rather than as a citation.

⚠ Two of the three have now been re-run and neither matched what was written
here. The encoder read 1.8e-04 when written and reads **1.699e-04** at HEAD,
which is real drift from the attention and gelu rewrites since; the figure
above is corrected. The decoder's **3.8e-05 has no audio beside it**, and that
is the whole problem: run against `jfk.wav` it reads 4.005e-05 with argmax 843
(`b' And'`), and against the script's own default it reads 5.430e-05 with
argmax 357 (`b' ('`). Three readings of one "tolerance". It is not quoted above
any more, because a worst-case difference over a decoder's logits belongs to
the audio that produced it, the same way a perplexity belongs to its corpus.
Both PASS their thresholds, which are 2e-3 and 3e-3, ten to eighty times
looser.

```
$ charsiu_clip clip-b32.gguf --image logo.png \
      --text "a drawing of a llama" "the statue of liberty" "a dog on grass"
  0.2745  a drawing of a llama
  0.1854  the statue of liberty
  0.1474  a dog on grass
```

⚠ This block used to print 0.2537 / 0.1743 / 0.1533 and was not one run. Rows
one and two are the **f16** CLIP file, reproduced here to the last digit; the
file `tests/board_modalities.sh` actually downloads is the q4_0 above, which
gives the numbers now shown. Row three reproduces from neither: f16 gives
0.1273 for this sentence and 0.1323 for board_modalities' wording, and ten
other phrasings ranged 0.1016 to 0.1850 without reaching 0.1533. Each sentence
scores independently, so rows one and two pin the image and the model exactly
and row three came from somewhere else.

A score belongs to its model file as much as a perplexity does. What the board
test checks is unaffected -- it asserts only that "llama" ranks first, which
holds in every arm.

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

The target is the vendor's own number on the same board and model. Decode beats
it; prefill leads below about 250 tokens and is level above, which the table at
the top gives row by row with the ranges that decide which is which.

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
