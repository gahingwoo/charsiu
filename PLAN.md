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

## 1. Prefill: batched, correct, and 3.63x. The route is int8

The vendor dispatches every one of its 3328 int4 projections at M = 1, so it
re-streams all 487 MB of weights for every prompt token. charsiu now does not:

```
    m   tensors   worst rel    rows that agree    one row   batched   speedup
    2     113     5.03e-05      226 of 226         227 ms    218 ms    1.04x
    4     113     5.03e-05      452 of 452         451 ms    266 ms    1.70x
    8     113     5.03e-05      904 of 904         899 ms    365 ms    2.46x
   16     113     5.03e-05     1808 of 1808       1812 ms    570 ms    3.18x
   32     113     5.03e-05     3616 of 3616       3590 ms    990 ms    3.63x
```

Every row of every tensor, checked against the one row path before anything is
timed, on a real model's own staged weights. 5.03e-05 is two float sums of two
thousand terms in different orders.

Per token of prompt, the projections go from 112 ms to 31 ms.

### It took five rounds to find out that the int4 path cannot do it

w4a16 produces exactly ONE row and nothing makes it produce two. Fed the same
activation twice, row 1 comes back matching row 0 in 1 of 2048. The DPU and
RDMA blocks are identical to a stream that does two rows, all 69 and all 22
registers; the four CNA geometry words were each put back to the int8 value
with a liveness check between them and none produced row 1; CORE 0x301c is
inert in either half and 0x3018 is the arithmetic switch. job.c has said since
round 347 that the vendor runs w4a16 on the width axis, and the vendor never
batches a weight matmul at all, so there is no M > 1 int4 stream to copy.

Two registers on that path were literals chosen at M = 1, the one width where a
value that follows the row count cannot show that it does: 0x40b8's 3, which
is 3 * rows, and 0x301c's half, which is zero either way at m = 1. Finding the
first cost five rounds and it was not even the answer.

### ⚠ And the thing that is not settled: decode wants int4

Batching works on int8 and decode is int4 -- 14.70 tok/s against int8's 9.71,
because decode is one row at the DRAM roof where halving the bytes nearly
doubles the rate. Prefill is not at that roof, which is why int8's extra byte
costs it nothing: at m = 32 the same weights serve thirty two rows.

A model is staged once, in one format. Reconciling those is a design question
and not an implementation detail:

- **all int8**: prefill 3.6x, decode loses a third. Right for long prompts,
  wrong for chat.
- **both staged**: 620 MB of int4 plus 1240 MB of int8, which this board does
  not have to spare.
- **per run**: the format becomes a setting, which is honest but pushes the
  choice onto somebody who cannot make it.

**Next**, and it is the cheap half: the batched path submits one slice at a
time and waits a full fence on each. Decode chains a projection's slices into
one submit for exactly this reason, and the report says 5072 ms of 7448 is
fence. Chaining is worth more than any of the above and does not need the
format question answered.

**Control**: decode is m = 1 and goes through charsiu_npu_matvec, which the
batched path does not touch. Same sentence, same tok/s.

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
