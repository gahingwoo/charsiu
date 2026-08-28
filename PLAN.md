# Plan

The goal is the vendor's own number on the same board and the same model, with
nothing closed in the execution path. Whatever the NPU can carry goes on the NPU.

Each step below states what it produces and how it is known to be right. The rule
carried over from the driver work applies here too: **a board round states its
decision rule before the run and carries a control that can fail**, and a proxy
metric is not a correctness oracle.

## What the first five steps turned out to be

They are kept as one section rather than five because they are answered, and what
they were for is the answers.

- **The instrument.** `tools/rkllm_regcmd.py` reads the vendor's dispatch plan out of
  a `.rkllm` with no board: three precisions by role, projections split across two
  cores, and 3752 dispatches at M = 1. It is still how a question about the vendor
  gets settled, and it is how the M > 1 register geometry was read.
- **M = 1 at an LLM's width.** Exact, and M = 2, 3, 4 and 8 are exact too through
  Mesa's own delegate. The RK3588 height constraint is not on this silicon.
- **One vendor matmul, bit exact.** Done, in int4, and then every matmul in
  Llama-3.2-1B.
- **The measurement that decides the architecture.** Answered in the NPU's favour,
  and not marginally: **15.91 tok/s at int4 and 64 tokens against the vendor's
  roughly 13**, with the hardware moving weights at 10.8 GB/s, which is this board's
  DRAM roof. The RK3588 conclusion that decode belongs on the CPU is not this board's
  conclusion.
- **The runtime.** It exists, it is standalone rather than a ggml backend, and it
  reads llama, qwen2, qwen3, gemma3, gemma4, phi3 and smollm3.
- **The product layer.** `charsiu`, `charsiu-get`, `charsiu-config`, `charsiu-doctor`,
  `charsiu serve`, and an installer.

## 1. Prefill: batched, correct, 2.94x on hardware. DONE

The vendor dispatches every one of its 3328 int4 projections at M = 1, so it
re-streams all 487 MB of weights for every prompt token. charsiu does not.

A 64 token prompt on an int8 staged Llama-3.2-1B, on a ROCK 4D, same binary,
one flag apart:

```
  batched   prompt 64 tok in 2406 ms   26.60 tok/s    37.6 ms a token
  control   prompt 64 tok in 7078 ms    9.04 tok/s   110.6 ms a token
```

**2.94x, and the two runs generate the same text word for word.** Prefill used
to cost more per token than decode; it now costs a third of it.

`llama_prefill_batch` runs the layer loop with n rows, batching the feed
forward -- gate, up and down, 63% of the projection time -- and leaving q, k, v
and o a row at a time. Rows are walked in order inside a layer because row r's
attention reads the KV the rows before it wrote, and the head is not batched at
all: a prompt needs logits for its last token and no other, so the widest
projection in the model is skipped n - 1 times rather than made n times wider.
The prompt goes in chunks of 32, because the buffers scale with the batch and
the probe's sweep flattens after 16.

### The other three modalities, measured on the board 2026-08-28

Seeing, hearing and matching all give the right answer on a ROCK 4D -- jfk.wav
transcribed word for word, the llama.cpp logo called a llama by both the VLM and
CLIP -- and all three are SLOW, in the same way:

```
  whisper tiny.en    34.5 s   for 11 s of audio    ~20 G-mac    0.58 G-mac/s
  SmolVLM-256M      153.1 s   for one picture      ~90 G-mac    0.59 G-mac/s
  CLIP ViT-B/32       4.7 s   for one picture     ~4.2 G-mac    0.9  G-mac/s
  the decode under them, unchanged:  19.75 tok/s
```

⚠ **NONE OF THEM TOUCHES THE NPU.** vision.c, clip.c and whisper.c call
gguf_matvec directly; only llama.c's projections are routed. So 0.6 G-mac/s is
not the hardware being bad at this, it is the scalar CPU path, and three
independent graphs landing on the same number is the same bottleneck three
times rather than a coincidence.

⚠ **AND THE SHAPE IS THE ONE ALREADY SOLVED.** A picture is 1024 patches, a
thirty second clip is 1500 encoder positions, a CLIP image is 50 -- all present
at once, against weights that do not change. That is the batched matmul measured
at 2.94x on 2026-08-27, at m values twenty to fifty times larger than a prompt
chunk. The NPU moves weights at 10.8 GB/s, which at int8 and m = 1 is about
10 G-mac/s before batching buys anything.

The work is not the arithmetic, which is written and checked. It is that the
NPU tensor cache lives in struct llama_state and these three graphs are not
llama.

### The towers on the hardware, and where the batch actually stops

Board, 2026-08-28, after routing the vision tower and the whisper encoder:

```
                CPU      NPU     with the cache warm
  whisper      34.5 s   30.0 s        30.0 s
  SmolVLM     153.1 s   82.4 s        81.3 s
  CLIP          4.7 s    3.9 s         2.8 s
```

⚠⚠ **AND THAT SUBTRACTION WAS WRONG.** The paragraph here used to read: 75 of
the vision tower's 82 seconds is the quantiser, so take it out and the matmuls
are 17x. The next board round put the staging BEHIND A CACHE -- whisper's
quantising went from 19298 ms to 118 -- and the wall clock did not move at all:

```
  whisper   30.0 s before the cache,  30.1 s after it, 30.0 s warm
```

Nineteen seconds left the staging line and the total stayed put, so the two
numbers were never parts of one sum. Nothing was counting how much of the work
reached the hardware, and a subtraction cannot be checked against a quantity
nobody measured. `charsiu_pool_report` counts it now: tensors actually routed,
matmuls, rows, milliseconds, and how many fell back.

The towers do stage eagerly into a cache in `$XDG_CACHE_HOME/charsiu`, and that
part works -- ⚠ the cache is ONE SEQUENTIAL FILE with one static handle, so a
caller claims it for its own stretch rather than sharing it, because two graphs
staging at once interleave and neither can read the result back.

⚠ **AND THE BATCH STOPS AT 80.** Swept against the same tower at two rows,
which is the smallest verified batch:

```
  4 8 16 32 48 64 80    0.000000   identical
  96 and above          56 to 95   a different tower
```

Mesa's budget test fires above m = 320 and its split above m = 640, so **that
prediction is about something else**; this is a different limit and it is
measured rather than derived. The default is 64: inside the edge with a step to
spare, and the rate is FLAT from 4 rows to 1024 anyway.

⚠ 80 was measured on one tower at K = 768 and 3072. Whether the bound is m alone
or m against K is not known.

### And then the counter said the matmul was never the cost

`charsiu_pool_report`, added because the 17x above had to be withdrawn:

```
  whisper   24 of 24 tensors on the hardware; 24 matmuls of 36000 rows in 936 ms
  CLIP      73 of 73 tensors on the hardware; 72 matmuls of  3600 rows in 210 ms
```

**936 ms of a 30 second transcription.** So the stage table, which is the thing
that should have been written before any of the routing:

```
  attention              2181 ms   62.7%     O(T^2) scalar C, never routed
  feed forward            670 ms   19.3%     routed
  q k v                   179 ms    5.2%     routed
  mel spectrogram         189 ms    5.4%     3000 FFTs
  the two convolutions     70 ms    2.0%
  out proj                 59 ms    1.7%     routed
  the decoder             121 ms    3.5%
  layernorms                8 ms    0.2%
```

⚠⚠ **THE WORK THAT WAS ROUTED IS 26% OF THE TIME AND THE ATTENTION IS 63%.**
1500 positions against 1500, six heads, four layers, in three nested loops --
and it is not a matmul against a weight, so none of the machinery built for the
towers touches it. Everything above this heading is correct and was aimed at a
quarter of the problem.

The next thing is the attention, and the stage table has to come first from now
on. It cost three board rounds and a withdrawn number to write one.

### Which weight format, answered

Four numbers off the board, same model, same prompt, same 16 generated tokens:

```
              int4     int8
  prefill    19.24    26.60 tok/s
  decode     15.46     9.16 tok/s
```

int8 is ahead on prefill and behind on decode, so it depends on the shape of
the work. For a prompt of P tokens and G generated,

  int8 wins  when  P * (1/19.24 - 1/26.60) > G * (1/9.16 - 1/15.46)
             i.e.  P > 3.1 * G

**So int4 stays the default.** Chat is a short prompt and a long answer and int4
wins it outright; int8 is for prompt-heavy work -- summarising a document,
retrieval -- where the prompt is several times the answer.

⚠ The batched prefill helps int4 too, and not because the matmul batches: it
refuses there. It is that a prompt needs logits for its last token only, so the
head is skipped n - 1 times whatever the format.

That was written here as an inference for a fortnight. It is now measured.
`tests/prefill_control.sh` runs batched -> control -> batched on the same
binary, one flag apart, and the two batched samples bracket the control so a
board that warms over a minute cannot be mistaken for the flag:

```
  control   65 tok / 4304 ms   66.22 ms a token   15.10 tok/s
  batched   65 tok / 3498 ms   53.82 ms a token   18.59 tok/s   (18.63, 18.54)
                               ---------
  saved                        12.40 ms a token
```

⚠ AND THE SAVING IS THE OUTPUT HEAD, TO WITHIN 1.4%, BY AN ARITHMETIC THAT
NEVER SAW THESE TIMINGS. The head runs once instead of 65 times, so the saving
per token is H * 64/65 and H is 12.59 ms. Llama-3.2-1B's head is 128256 x 2048,
which at int4 is 131.3 MB of weights, and

  131.3 MB / 12.59 ms = 10.43 GB/s

against the 10.58 GB/s this model's own NPU summary reports for weight
bandwidth. The time the batched prompt does not spend is exactly the time it
takes to stream the head's weights, once per token, at the rate this board
moves weights.

⚠ And the control says what the baseline was: 15.10 tok/s prefill against a
decode of 15.70. Without batching a prompt token costs what a generated one
costs, which is where this started.

⚠ It is a second copy of the layer loop and it is deliberately blind. It
handles the plain case and REFUSES the rest -- gemma3's window and two rope
bases, gemma4's per layer embeddings and shared KV, qwen3's q and k norms,
phi3's fused K and V, biases, post norms, softcaps -- and the caller falls back
to the token loop, which is correct for all seven architectures and merely
slower. Refusing is not an error path, it is the other half of the same
decision.

⚠ SO PHI-3.5 HAS NO BATCHED PREFILL, and that is a real gap rather than a note.
Its K and V arrive as one fused tensor, llama_prefill_batch refuses on the
first layer, and its prompt costs 4.96 tok/s -- what a generated token costs.
Splitting a fused qkv into three views is the whole of what it would take. It
has not been done because nothing had measured what it was worth until the
control above put a number on the head skip, and 12.4 ms a token is what phi3
is leaving.

⚠ And a refusal has to be AUDIBLE. batch_ok used to return 0 and the caller
fell back in silence, so "the flag did nothing" and "this model was never
batchable" looked identical from outside -- which is how the first attempt at
the control above spent four minutes on Phi-3.5 and produced 4.96, 5.00 and
5.10 tok/s, three numbers that agree and mean nothing. llama_batch_why_not
returns the reason as a phrase now and every run prints one line saying which
path its prompt took.

⚠ And it runs without the NPU, which is what makes the risky half checkable off
the board: matmul_rows falls back to matvec, so a host with no hardware
compares the two layer loops and nothing else.

### What it took, and the shape of every wrong turn

Five rounds went into establishing that the int4 path cannot do this at all.
w4a16 produces exactly ONE row: fed the same activation twice, row 1 comes back
matching row 0 in 1 of 2048. The DPU and RDMA blocks are identical to a stream
that does two rows, all 69 and all 22 registers, and every CNA word that
differs was put back one at a time with a liveness check between them. The
vendor never batches a weight matmul, so there is no M > 1 int4 stream to copy.

Two registers on that path were literals chosen at M = 1, the one width where a
value that follows the row count cannot show that it does: 0x40b8's 3, which is
3 * rows, and 0x301c's half, which is zero either way at m = 1.

The read order is `P = m/2`, super groups of 32, rows pairing P at a time with
the four word runs alternating. Solved from two printed maps and confirmed at a
width it was not fitted on.

## 2. The output head: routed, and it was worth 49 ms a token. DONE

gemma3's head is 262144 wide and was spending 49.8 ms of a 113 ms token on the CPU.
It is on the NPU now, and the tokens are identical:

```
                    before      after
  output head       49.81 ms    11.53 ms      44.2% of the token -> 15.6%
  a token          112.7  ms    73.9  ms
  generation         8.76 tok/s 13.07 tok/s
  tensors staged      182         183
  slices              468         532
  weights           6.16 GB/s   7.18 GB/s
```

**It was not the coefficient bound**, which is what this section said for four
rounds. The shipped runner has passed `CHARSIU_COEF_ELEMS=65536` all along, putting
the head's coefficients at 10.5 MB. What refused it was `maxn`, defaulted to 131072
to clear Llama 3.2's 128256 vocabulary against gemma3's 262144, and two things kept
that invisible: the gate was the only refusal in the runtime that did not say why,
and the round that set `CHARSIU_NPU_MAXN` on the command line did not set it, because
`charsiu run` built its environment from config.ini and handed it to `env NAME=VALUE`
in front of the caller's.

Prefill is also a real number now rather than a slander: staging is built lazily
inside the first forward pass, so all 15 seconds of it were charged to the prompt.
Reported separately, the same run reads `staging 15081 ms | prompt 6 tok in 829 ms,
7.24 tok/s` where it used to read 0.92.

⚠ One number to keep an eye on. 151 MB of int4 head in 11.5 ms is 13.1 GB/s, which is
above the 10.8 GB/s this board's DRAM roof was measured at. Either the roof is higher
for this access pattern or the stage timer is not charging the head everything it
costs. The tokens are right either way, so this is a measurement question.

The coefficient bound is still a guess and still unmeasured; `npu_gemm_test K N
--coef` walks it down. It is no longer in the way of anything.

## 2b. gemma3 runs the NPU at 6.16 GB/s where llama reaches 10.3

Same board, same int4 path, and the difference is the shape. n_embd 1152 does not
divide the 1024 K slice, so q, k, v, gate and up each split 1024 + 128 and the second
slice carries a ninth of a slice of work for a whole task's cost. 468 slices where
338 would do, five of the extras in every one of 26 layers.

`CHARSIU_NPU_KFIT=1` gives the last slice the remainder instead, so 1152 is one slice.
Ungrouped tensors only: a grouped tensor's scale gather reads one group per slice.
Off by default, unrun.

**Control**: the same prompt with and without it in one boot. Identical tokens is the
correctness bar, since acc_out sums int32 across K slices and any split of the same K
has to give the same accumulator; slices and GB/s in the NPU report are the win.

## 3. Upstreaming

- The driver is [PATCH v9](https://lore.kernel.org/all/cover.1787568658.git.gahing@gahingwoo.com/),
  with an Acked-by on both dt-bindings, a Reviewed-by on both pmdomain patches and a
  Tested-by on the two reset-race patches. v10 is tag collection and the git note that
  v9's cover letter promised and did not carry.
- The Mesa side is [mesa!43804](https://gitlab.freedesktop.org/mesa/mesa/-/merge_requests/43804).

## What would make this stop

Stated in advance, so it is a decision rather than a slow fade:

- if prefill cannot be batched after a fair attempt, that is a written negative result
  and decode stands on its own -- it already beats the vendor;
- if the output head routes and gemma3 does not get faster, the head belongs on the
  CPU on this board and gemma-shaped models are simply expensive here, which is worth
  saying rather than working around.
