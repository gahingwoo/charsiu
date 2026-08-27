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

## 1. Prefill: the hardware was never the problem. CLOSED as a defect, open as work

Decode is closed: four fifths of a token is the hardware on the DRAM roof, the CPU
side is about twelve milliseconds, and the whole token is accounted for to within one.
Prefill is where the vendor is beatable rather than matched -- **every one of its 3328
int4 projection dispatches is M = 1**, so it re-streams all 487 MB of weights for every
prompt token. At M = 32 the same bytes serve thirty two rows, and the board measured
6.9 us a row against 200.

Five rounds looked for a hardware wall above one row. There is none. Two things were
wrong, both in this tree.

### One register was a literal where it should have followed the row count

```
emit(DPU, 0x40b8, acc_out ? 3 : ow * rows)
```

On the height axis `ow` is 1 and `rows` is m, so the int8 arm writes the row count.
The acc_out arm wrote 3 at every m, and round 312 chose that 3 by sweeping 0 to 15 at
**M = 1**, the one width where a value that should follow M cannot show that it does.
Swept again where it can move, the peak walks: 3 at m = 1, 6 at m = 2, 12 at m = 4.
It is `3 * rows`, and 3 * rows is 3 at m = 1, so decode cannot change.

With that, **every wanted value is in the buffer at every shape and every m measured**.
The board prints "0 are absent from the buffer altogether" at K = 64 N = 64,
K = 256 N = 128 and K = 1024 N = 32, at m = 1, 2, 4 and 8. The arithmetic was never
wrong above one row; the register was.

### And the read order was not the one anybody guessed

```
P = m/2
G = ni/32, c = ni%32, a = c/16, t = c%16
j     = (t/4)*8P + (mi%P)*8 + a*4 + t%4
index = G*(m*32) + (mi/P)*(32P) + j
```

Channels go in super groups of 32. Inside one the rows pair up P at a time, and inside
a pair the four word runs alternate between the rows and between the two sixteen
channel halves. At m = 2 that is P = 1, each row its own block of 32 with the halves
interleaved: channels 0..3, 16..19, 4..7, 20..23. At m = 4 it is P = 2, rows pairing
into blocks of 64 that alternate every eight words.

Solved from the printed maps at m = 2 and m = 4, then **confirmed at m = 8, which it
was not fitted on**, and at m = 1, which is flat and a separate case:

```
  K=1024 N=32     m=1  32/32    m=2   64/64    m=4  128/128   m=8   256/256   all EXACT
  K=256  N=128    m=1 128/128   m=2  256/256   m=4  512/512   m=8  1024/1024  all EXACT
```

`charsiu_acc_index()` is that expression, in the library rather than the probe, because
the runtime needs the same copy.

**Next, and it is engineering rather than investigation.** The runtime bakes
`mm.m = 1` into every slice's register stream at staging time, so a batched submit has
to re-emit the stream at the width it wants. The weights and the coefficients do not
depend on m and do not move; the regcmd, the activation packing and the output read do.
Then llama.c has to feed the prompt in chunks of M rather than one token at a time.

**Control**: decode is m = 1 and must stay byte identical. The same sentence, and the
same tok/s, before and after.

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
