# charsiu

An open LLM runtime for the **RK3576 NPU on a mainline Linux kernel**, driving the
hardware through the mainline `rocket` DRM-accel driver with no vendor userspace in
the execution path.

**Status.** On a ROCK 4D, under the vendor's own measuring protocol, decode is at
the vendor's speed:

```
                     decode tok/s            time to first token, ms
                     charsiu   vendor        charsiu   vendor
  Qwen3 0.6B          24.28    24.85          1037      469
  TinyLLAMA 1.1B      20.34    19.71          1565      544
  Phi3 3.8B            6.82     6.58          5073     1829
  Gemma4 E2B           8.68     9.23          3604     1219
```

Every projection, including the output head, runs on the NPU at four bits, and the
text is identical to what the CPU decode loop writes. Prefill is the open front:
between a third and a half of the vendor's, because every projection still comes
back to the CPU between layers. The numbers are read off `board_verify.sh 7`, the
same prompt and protocol the vendor publishes, and the rest of this file says how
they are known to be right.

**The NPU rail.** The times above run the two NPU cores one at a time, because
run together they get about one row in a thousand wrong. The runtime is not
what does it. Mainline clocks the NPU at 786 MHz and leaves the rail at whatever
U-Boot set, 750 mV, where the vendor's own table asks 800 mV of that speed. Give
the rail 800 mV, one line of device tree, and both cores run on every batched
call:

```
                     decode tok/s            time to first token, ms
                     charsiu   vendor        charsiu   vendor
  Qwen3 0.6B          24.88    24.85           717      469
  TinyLLAMA 1.1B      20.67    19.71           960      544
  Phi3 3.8B            6.88     6.58          3127     1829
  Gemma4 E2B           8.71     9.23          2295     1219
```

Time to first token came down 17 to 21% on 2026-09-06, which took the gap from
1.85-2.40x to 1.53-1.88x. None of it was new hardware work: the batched
prompt's elementwise stages, its softmax and its activation packer went on the
thread pool, the pooled read's size threshold came down from 262144 elements to
32768, and the calling thread and the pool workers were given separate core
sets -- decode runs on the caller and wants an A72, the prompt runs on the pool
and wants all eight. `docs/lab-notebook.md` has the arms for each.

The text stays identical on all nine models. charsiu reads the rail and the clock
out of sysfs and overlaps the cores only inside the vendor's envelope, and its NPU
report says which it chose and why. A board that leaves the rail alone keeps the
one core numbers. The decision itself is `tests/overlap_guard.c`, which runs on a
desk.

It reads **llama, qwen2, qwen3, gemma3, gemma4, phi3 and smollm3** gguf files.

**And it sees.** A vision tower read out of llama.cpp's `mmproj` gguf, on the same
primitives -- a patch embedding is a convolution whose stride equals its kernel, which
is a gather into rows and one matmul, so nothing outside `(m, k, n)` is needed.

```
$ charsiu --image llama-logo.png "what animal is this?"
Llama.
```

On SmolVLM-256M, a 512x512 picture is 1024 patches and 64 tokens, and the control is
that without the picture the same prompt makes one up.

**And it hears.** Whisper, out of whisper.cpp's own container -- the mel spectrogram,
the audio encoder, and a decoder with the first cross attention in this tree.

```
$ charsiu_whisper ggml-tiny.en.bin --transcribe --audio jfk.wav
 And so my fellow Americans ask not what your country can do for you, ask what you
 can do for your country.
```

Every stage is diffed against numpy on the real weights: the spectrogram at 1.7e-05,
the encoder at 1.8e-04 over 576000 values, the decoder's logits at 3.8e-05 with the
same argmax.

**And it matches pictures to words.** CLIP's two towers land in one space:

```
$ charsiu_clip clip-vit-base-patch32.gguf --image logo.png \
      --text "a drawing of a llama" "the statue of liberty" "a dog on grass"
  0.2537  a drawing of a llama
  0.1743  the statue of liberty
  0.1533  a dog on grass
```

## Install

```
curl -fsSL https://raw.githubusercontent.com/gahingwoo/charsiu/stable/scripts/charsiu-install.sh | sh
```

It fetches the source, checks whether the running kernel can drive the NPU, and
**offers a kernel if it cannot**, because RK3576 NPU support is not upstream yet,
so no distribution kernel anywhere will bind this hardware. A kernel it installs
becomes the default boot entry. On an extlinux board **the one already on the card
stays selectable**, five seconds into the boot; on Armbian there is no menu, and the
previous kernel is kept as `/boot/Image.previous` to copy back by hand.

Then it builds charsiu, fetches a model that charsiu can actually run, and asks it
something so you can see it work.

Rehearse it first if you would rather:

```
curl -fsSL https://raw.githubusercontent.com/gahingwoo/charsiu/stable/scripts/charsiu-install.sh -o install.sh
sh install.sh --dry-run
```

`--dry-run` prints every action as `would ...` and changes nothing. The read-only
checks still run, which is the point: it is what the installer *sees* on your machine.

### Two channels

**stable** installs the runtime AND the other modalities: `charsiu_run`,
`charsiu_check`, `charsiu_serve`, `charsiu_vision`, `charsiu_clip`,
`charsiu_whisper`. None of those touches an NPU register that the decode path
does not.

**dev** adds the hardware probes -- `npu_gemm_test`, `charsiu_matmul`,
`bench_batch`, `prefill_control.sh` -- and tracks `dev`, where the work happens.
They exist to ask the silicon questions, and asking has wedged the block, timed out
and printed the opposite of its own data on the way to the answers. They are not
something to install on a board you want to rely on.

⚠ The channel decides what is **installed**, not what is in the tree, and the two
build from the same sources. It is not a fork.

⚠ And a conversation shows the conversation. This tree prints a running commentary
on stderr -- which CPUs it pinned, what the governor is reading, which tensors did
not reach the hardware and why, which path the prompt took -- and every one of those
lines exists because a board log could not be read without it. In front of somebody
who typed a question they are noise, so `charsiu` turns them off in interactive mode
and a one-shot run, which is what a board log is, keeps them. Errors, the model
banner and the staging progress are not commentary and always print.

```
sh install.sh --dev          the probes too, from the start
charsiu update dev           switch an existing install, no reinstall needed
charsiu update stable        go back
charsiu update               whichever it was last told, remembered
charsiu update --auto        no questions, no demo at the end (-y);
                             `auto = yes` under [update] in config.ini keeps it
```

`charsiu doctor` says which channel it is on, which matters when a log gets
pasted somewhere: "it hung" from a board with the probes on it and from one
without are not the same report.

After that:

```
charsiu                     a conversation (the model stays staged between turns)
charsiu -p "..." -n 64      one answer
charsiu bench               what this board does, in tok/s
charsiu-config              pick a model, threads, context
charsiu-get                 more models: llama, qwen2, qwen3, gemma3, gemma4, phi3, smollm3
charsiu-doctor              what works and what does not
charsiu-doctor --paste      the block to put in a bug report
```

⚠ **A gguf's name does not tell you whether charsiu can run it**, in either
direction, see [What a file's name does not tell you, twice](#what-a-files-name-does-not-tell-you-twice).
`charsiu-get` gates every download by reading the file, and deletes what it cannot run.

The CPU decode loop came first and is the oracle; the NPU computes the same sum, so the
two runs must agree exactly, and "the text looks fine" was never allowed to count.

Still on the CPU: the embedding lookup, RMSNorm, RoPE, the attention score, softmax and
weighted sum, SwiGLU, the residuals and sampling. Every projection, including the output
head, is on the hardware.

Two halves, and only one of them touches the NPU.

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

### What a file's name does not tell you, twice

The comparison "q4_0 against q8_0" was never a bytes-against-bytes comparison, and it took
two board rounds to see why. llama.cpp quantises `token_embd` to **q6_K** in a q4_0 file
and to **q8_0** in a q8_0 file. Llama 3.2 ties the output head to that tensor, and at a
128256 vocabulary it is a fifth of the weights, so the two files differ in the format of
their single largest matmul, and q6_K costs far more arithmetic per byte than q8_0 does.

A third file, built with `llama-quantize --pure`, holds one format all the way through.
Host, 4 threads:

| | q4_0 (q6_K head) | q4_0 `--pure` | q8_0 |
|---|---|---|---|
| tok/s | 35.29 | **40.70** | 34.43 |

The head alone was worth 15%. With it gone the crossover disappears, pure q4_0 leads at
every thread count, by more as threads are added (t=1 21.67/21.45, t=2 34.26/32.92,
t=4 40.70/34.74, t=8 31.73/30.41, pure against q8_0), which is what bytes mattering more
per core looks like.

The `--pure` file is **an instrument, not a good model**: it was requantised from the q8_0
file, so it is quantised twice, and its output head is 4 bit. It holds one variable still.
Its text is only ever compared against itself.

One detail worth writing down: ggml's **reference C** quantises with `roundf`, ties away
from zero, but its **ARM path** uses `vcvtnq_s32_f32`, which is ties to even, and the ARM
path is the one that runs. Matching `roundf` made the worst logit disagreement with
llama.cpp twice as large and cost 9% of the tokens per second.

## Speculative decoding: more tokens a weight read

A decode step at one row reads every weight once, and on this board that is
the whole of its cost: Qwen3-0.6B is 298 MB of int4 read at about 16 GB/s
across the two cores, which is the 52 ms the token takes. Rockchip's runtime
reads the same bytes in 40 ms. Nothing inside a matmul changes the bytes and
the NPU has no int2 or int3, so the only way to more tokens a second is more
tokens a read, which is what a batch is.

`--spec K` guesses K tokens by prompt lookup, the longest recent n-gram that
has occurred before and what followed it, and feeds the last committed token
plus the guesses as one batch of 1 + K rows. Every row's logits are read back;
row i's argmax is what greedy decoding would have produced after guess i-1.
Guesses are accepted while they match and the first row that disagrees
supplies the token greedy would have produced there instead. So every
committed token is the token the plain loop would have committed, and the
guesses only decide how many of them one weight read yields. The text is
bit-identical to greedy by construction, and `tests/spec_identity.sh` holds it
to that on three arms, one of which drafts junk and must be refused every
time.

On a machine with no NPU, on a prompt that asks for repetition, the pass
yields 2.29 tokens on Qwen3-0.6B, 2.45 on gemma4-E2B and 1.70 on Phi-3.5.
Those are properties of the model and the prompt; a repetition prompt is
prompt lookup's best case and open prose will be lower. What the host cannot
measure is what a pass at four rows costs against one decode step, and the
speed-up is tokens a pass divided by that. The argument that it is about one
step is an argument about bytes; the fence and the accumulator read back both
grow with the rows. `board_verify.sh 20` measures it, and `--spec` stays off
until it has.

Sampling at a temperature runs the plain loop: lossless speculative sampling
exists and is not written here.

## What runs today

`charsiu` picks the environment itself: int4 weights, the K slice width chosen per
model, both cores, and the CPUs held out of deep idle while the NPU is open (see
below). `CHARSIU_STAGES=1` prints where a token goes, once per half of the run.

Three things moved decode from 82% of the vendor to parity, each measured on the
board with a control:

- **the CPUs are held out of deep idle while the NPU is open.** rk3576's CPU_SLEEP
  costs 250 us to leave, a token is about 150 calls into the driver, and each call
  wakes several threads that had gone to sleep for the 300 us the hardware took.
  charsiu writes a 100 us bound to `/dev/cpu_dma_latency` and keeps it open: +24%
  on its own, with the sysfs switch as the control. `CHARSIU_NPU_IDLE=1` leaves the
  CPUs alone.
- **the kernel keeps the IOMMU domain attached across jobs and handles the
  completion in the hardirq**, two patches under review in the driver repository,
  about +4% together.
- **the batched paths have their ceilings written down.** An int4 batch is one row
  M pixels wide and the hardware stops computing past 5120 input entries; an int8
  batch is M rows high and stops past 8192. Both were walked on the board, both are
  refused above the last exact cell, and the tower chunks its rows under the second.

And three moved the prompt, on a 915 token prompt at chunk 80, measured the same way:

- **attention runs eight rows a pass over the cache, a head range a thread, four
  positions a pass.** It was 45 to 72 percent of a long prompt, one row at a time on
  one core; it is 3.5 to 3.8 times faster on every model of eight, and each row still
  scores the same positions in the same order with the same kernels the token loop
  uses, so the text does not move. `CHARSIU_ATTN_BLOCK=0` is the row loop.
- **the packed input is reused across q, k and v and across gate and up**, on the
  core that packed it; the key expires at every leader, which is what phase 22 of
  `board_verify.sh` found the first version did not do.
- **a batched call does half the ioctls it did**, and asks the environment once
  rather than fifteen times per register stream: a four row speculative pass went
  from 2.5 decode steps to 2.1.
- **the thread pool hands out chunks from a cursor**, so the four A72s take more
  than the four A53s instead of waiting for them.

### More than one architecture, and two things that were quietly wrong

`llama_load` reads **llama, qwen2, qwen3, gemma3, gemma4, phi3 and smollm3**. `charsiu_check`'s
gate has to match it exactly: it exists to save a two gigabyte download and is wrong in
both directions if it drifts.

⚠ All seven decode and all seven batch a prompt. The last two off that list, gemma4
and phi3, were refused for properties that turned out not to be the cause: the faults
were an odd batch width, which has no expression on the accumulator surface, and the
two NPU cores corrupting each other when their submits overlap. The chunker emits only
even widths now and the submits are serialised.

⚠ An empty refusal list means the next architecture will not be refused by this loop,
it will be MISSED by it. `tests/board_text_all.sh` is the check that caught phi3, on
its first run.

⚠ And a refusal has to be audible. It used to be a `return 0` the caller swallowed, so
"the flag did nothing" and "this architecture was never batchable" looked identical from
outside -- which is how a board round spent four minutes on Phi-3.5 and produced 4.96,
5.00 and 5.10 tok/s, three numbers that agree and mean nothing. Every run prints one
line now saying which path its prompt took, and the reason when it fell back:

```
charsiu: prompt batched, 64 tokens in chunks of 32
charsiu: prompt a token at a time (CHARSIU_NO_BATCH_PREFILL)
```

`tests/arch_sanity.sh` asks every gguf in a directory one factual
question and reports the ones that miss it -- no llama.cpp build, no f32 copy -- and it
makes a second pass with `CHARSIU_STAGES=1`, because a crash that needs an environment
variable is still a crash and one of them cost four board rounds.

### Pictures, and what a real mmproj said about the guesses

The tower is a ViT and a ViT is matmuls, which is why this fitted at all: a patch
embedding is a `k x k` convolution with **stride k**, so the patches do not overlap and
it is a gather into rows followed by one matmul. charsiu's job encoder only ever speaks
`(m, k, n)` and that is enough for the whole of it. The real 2D convolution stays in
Mesa.

⚠ **And a picture is exactly the batched case.** 1024 patches arrive AT ONCE, which is
the m > 1 matmul, so the tower is where the batched path pays most. That also settles
the format: **int8**, because w4a16 computes one row whatever it is asked for and an
int4 tower would dispatch those patches one at a time.

Every tensor name was a guess until a real file corrected two of them:

- ⚠ **`ffn_up` and `ffn_down` are backwards in a vision tower.** `ffn_down.weight` is
  `(768, 3072)` -- the FIRST matmul -- and `ffn_up` is `(3072, 768)`, the opposite of
  the same two words in the language model. The pair is bound **by shape** now:
  whichever contracts over `n_embd` is fc1, and a file naming them either way reads.
- ⚠ **A picture is not one token per patch.** The projector is idefics3, and
  `mm.model.fc.weight` is `(12288, 576)` because a **pixel shuffle** by 4 folds sixteen
  neighbouring patches into one embedding first. 1024 patches become **64 tokens**.

Binding checks shape as well as name, because a name that exists with the wrong shape
is the worse case: it contracts over the wrong axis and produces a fluent sentence
rather than an error.

Three tests, none of which needs a model or a board. `tests/mmproj_synth.py` writes a
synthetic tower and checks that a removed tensor is reported **by name**;
`tests/vision_cross.py` diffs both projector paths against numpy at 2.4e-07;
`tests/resize_cross.py` checks the image resize on its own, because half a pixel of
shift is invisible in a caption and produces a confident answer about a slightly
different crop.

⚠ **What those tests cannot see**, written into their own docstrings: the reference was
written by whoever wrote the C, so a convention they SHARE passes. The patch gather
order is the one to suspect first if a real model ever describes the wrong picture
fluently.

⚠ **A prompt with a picture in it takes the token loop.** The tower batches -- it is
our own graph -- but the batched language model builds its rows from the embedding
table by token id, and the picture's rows did not come from there. Sending them
through it would batch the text and drop the picture.

### What a tokens-per-second number needs before it can be compared

Four things, and every one of them has been the answer to a "why is this slower than I
remember" at least once:

| | how it is reported |
|---|---|
| the weight width | `charsiu NPU: weights are int4, 2 devices` |
| how many tokens | already in the summary; attention grows with the context |
| the CPU's clock | `cpu 2208 MHz under load`, read AFTER the work |
| the board's temperature | `board 53 -> 51 C`, at both ends of the generation |

⚠ The clock has to be read under load. Two consecutive runs of the same command
reported 2208 MHz and 1200 MHz from the same line, both with the governor at ondemand,
purely because it was read at startup before ondemand had seen any work.

⚠ And the second half of a run is slower anyway. On a host with no thermal zones and no
way to throttle, 32 tokens split 34.28 and 29.59 tok/s -- fourteen percent, entirely
from attention. A throttle is a gap wider than that alongside a temperature that
climbed, which is why the temperature is printed next to the halves and not somewhere
else.

### fp16, which is how the vendor runs attention

The vendor's own model file submits 4940 fp16 dispatches against 3328 int4
ones: 56% of what it asks the NPU to do is attention, in fp16, and 2908 of
those carry `oc = 64`, this model's head_dim. charsiu runs attention on the
CPU, where it is 30 to 52% of a prompt.

The fp16 matmul now works on this hardware and is bit exact against a CPU
reference at every width tried.

The first pricing of it was per dispatch, and per dispatch it is 98% waiting:
0.35 to 0.45 ms in the fence against 6 to 107 us for everything else. So the
unit takes a GROUP -- a layer of attention is H independent scores matmuls, a
softmax, and H independent values matmuls, and each half is one submit and one
wait. On the board, against the same ops one at a time and bit for bit
identical to them:

```
  ops   shape                   one at a time   grouped
  16    k=64  n=1024 m=8        1.735 ms        0.211 ms    8.2x
  32    k=64  n=1024 m=8        3.214 ms        0.211 ms   15.2x
  16    k=1024 n=64  m=8        2.687 ms        0.300 ms    9.0x
  32    k=1024 n=64  m=8        7.232 ms        0.375 ms   19.3x
```

Each round then said what was left, and three things went in turn. The
coefficients are the same bytes for every op of one shape, so they are built
once and then never (`coefs 0.000`). A KV cache is appended to a row a token
and never changes, so `charsiu_fp16_w` is a device buffer the caller writes
rows into at `charsiu_fp16_woffset` and the hardware reads where it lies -- the
board confirms the offsets do not move as the cache grows, which is what makes
that legal. And an op that hands over a NULL `Y` leaves its answer in the
device buffer for a caller that is about to reduce over it anyway.

`CHARSIU_ATTN_NPU` puts it in the model, and the honest result is two things.
It is **correct**: `CHARSIU_ATTN_NPU_CHECK` runs both arms on the same rows and
they land 0.25 to 0.5% apart over 112 layers with nothing falling back, which
is fp16 inputs against an fp32 CPU arm and not a cache written in the wrong
order. And it is **slower**: 6710 ms against the CPU arm's 4889 on a 513 token
prompt, and 52558 against 35686 on a 2074 token one. The gap widens with the
cache rather than closing, because the scores matmul's n is the number of
positions and the values matmul's k is the context -- the hardware's work grows
with the cache exactly as the CPU's does, and it starts from behind. It is off by default, and with the flag unset
four models produce text identical to the build from before any of this.

`docs/lab-notebook.md` has what had to be found, including the six
register-level fixes tried against a fault that was in a buffer, and the
explanation for the slowdown that the board refuted.

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
  board and model. Decode meets it; prefill does not yet, and the table at the top
  says by how much. Every number in this file is measured on the board.

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

## Prerequisite

The RK3576 support in `rocket` is not upstream yet. It is on the list as
[PATCH v11](https://lore.kernel.org/all/20260831081956.84871-1-gahing@gahingwoo.com/),
which has collected an Acked-by from Conor Dooley on both dt-bindings, a Reviewed-by
from Abel Vesa on both pmdomain patches, and a Tested-by from Igor Paunovic on each of
the three reset-race patches with a Reviewed-by on the refactor -- his testing on RK3588 reproduced, once in 102 induced resets, an
inference that signalled success while its output buffer was never written, and only
on the arm without them. The driver plus the Mesa work it comes from is
[linux-rk3576-npu](https://github.com/gahingwoo/linux-rk3576-npu), which is where the
RK3576 register knowledge in this repository was established.

kiln's `capture/rknpu-regcmd-dump.patch` is how a live vendor dispatch is read when
the model file is not enough.

## Licence

GPL-2.0-or-later. `LICENSE` carries the GPL version 2 text; the "or later" is what
lets this be combined with GPL-3 code if it ever links any.

## The record

How the int4 layout, the output surface, the accumulator read order and the batching
were read off the hardware, round by round, is in [docs/lab-notebook.md](docs/lab-notebook.md).
It is long because it is the evidence. What the vendor's own model file says about its
dispatches is in [docs/vendor-dispatch.md](docs/vendor-dispatch.md).
