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

## 1. Prefill, which is the only number left that can move

Decode is closed: four fifths of a token is the hardware on the DRAM roof, the CPU
side is about twelve milliseconds, and the whole token is accounted for to within one.
Prefill is not, and it is where the vendor is beatable rather than matched -- **every
one of its 3328 int4 projection dispatches is M = 1**, so it re-streams all 487 MB of
weights for every prompt token. At M = 32 the same bytes serve thirty two rows, and
the board measured 6.9 us a row against 200.

What is in the way is measured and is not a layout:

```
N=64 m=2   92 of 128      N=64 m=4  148 of 256      N=64 m=8  260 of 512
N=32 m=2   52 of  64      N=16 m=2   32 of  32
```

`written = N + inc * (m - 1)`, with `inc = N/4 + 12` through all three widths. **The
first row gets all N words; every row after it gets N/4 + 12 and needs N.** They meet
at sixteen, and at sixteen nothing is missing -- every value present, none of them
where a contiguous read expects it. That is a budget being divided, not a stride, and
the four rounds spent fitting an address function to the output were fitting the
symptom.

### The CBUF split is not it, and the arithmetic says so without a board

The step this section named next was: Mesa splits a task's staged rows when
`(cbuf_rows + staged) * entries_per_slice > total_entries` and charsiu never checks.
Put the numbers in and it cannot fire.

An LLM matmul reaches the encoder as `input_width = 1`, `input_height = m`,
`input_channels = K`, so `entries_per_slice` is 16 at K = 1024 and 32 at K = 2048.
The RK3576 CBUF is 16 banks of 512 entries; Mesa's own budget is five banks usable
and ten total. At m = 8 and K = 1024 that is **128 entries against 2560**. The
over-budget test needs m > 320 and the split test needs m > 640; the row-window path
needs `input_width >= 112`, and this shape is one wide. **Mesa does not split these
either.** Not splitting is not the difference.

### What the record actually partitions on: `surf`, not `m`

Every shape this tree has ever computed correctly above one row:

| shape | surf | result |
|---|---|---|
| M=224 K=64 N=64 | **1** | 14313 of 14336, none off by more than 1 |
| M=3136 K=33 N=64 | **1** | 200344 of 200704, 51 off by more than 1 |

Every shape that fails is K = 256 or K = 512, which are `surf` **4** and **8**. There
is no measurement at surf 2 anywhere, and **no measurement at surf > 1 that ever
worked at any m**.

So "M > 1 is broken" is contradicted by M = 3136 computing 99.8% of a 56x56 surface,
and the statement that survives the whole record is narrower: **entries per row above
one is broken once there is more than one row.** `surf` is the stride between staged
rows -- it goes into `0x1028` as `surf * rows` -- and at surf 1 a wrong multiplier is
invisible however many rows there are, which is exactly the shape of a fault that
hides at m = 1 and hides again at K = 64.

It also explains the one result that had no explanation: `CHARSIU_ENTRY_ATOMICS=8`
changed nothing because at K = 256 it takes surf from 4 to 2, and 2 is not 1.

**Next**: `npu_gemm_test K N --surf` walks K itself with N and m held (the K argument is ignored), so surf is the only
thing moving, and it does it in **one process and one build** -- the surf 1 successes
were measured on 2026-08-16 and the surf 4 failures ten days later, which is the one
way this partition could still be two different faults wearing one name.

**Control**: K = 48 and K = 64 are surf 1 and have to be exact at m = 2 and m = 4. If
they are not, the axis is not surf and the table means nothing. K = 80 and K = 128 are
surf 2, which nothing has ever measured, and they are what says whether the boundary
is "surf 1" or "surf small".

**If it holds**, a batched prefill has a cheap route that needs no new register: K is
already sliced by KMAX and `acc_out` sums int32 across slices exactly, so a KMAX that
keeps surf at 1 (64 for int8, 32 for fp16) makes every slice a working shape. That
buys many more tasks per projection, which at m = 32 is paid for thirty two times
over.

## 2. The output head, which is two fifths of a gemma token

262144 wide, 151 MB every token, and on the CPU: the stage table for gemma3 charges
it 49.8 ms of a 113 ms token, at 3 GB/s, where the NPU moves weights at 6 to 10.

**It was not the coefficient bound.** The shipped runner already passes
`CHARSIU_COEF_ELEMS=65536`, which puts the head's coefficients at 10.5 MB and well
inside what allocates. What refused it was `maxn`: the config default was 131072,
picked to clear llama 3.2's 128256 vocabulary, and gemma3's is 262144.

Two things made that cost four board rounds to see:

- the runtime declined **silently**. Every other refusal in npudev whines once per
  reason; this gate only spoke under `CHARSIU_NPU_VERBOSE`. All three silent paths
  now say which tensor, why, and which setting to change.
- the round that set `CHARSIU_NPU_MAXN=262144` on the command line **did not set it**.
  `charsiu run` builds its environment from config.ini and passes it through
  `env NAME=VALUE`, which is put in front of the inherited environment, so the
  config's 131072 overwrote the caller's 262144 without a word. The config is the
  default now and the environment is the override.

**Next**: a board round with `charsiu run gemma-3-1b-it-Q4_0.gguf` and nothing else,
on a config that has the new `maxn = 262144`. If the head routes, the token should
lose most of 49.8 ms and gemma3 should go from 8.76 tok/s to somewhere near 12.

The coefficient bound is still a guess and still worth measuring, but it is no longer
in the way of anything.

**Control**: `npu_gemm_test K N --coef` walks the bound downward and stops at the
first value that is not exact. ⚠ It walks DOWN because under-allocating does not
return an error -- the RDMA reads past the buffer, the IOMMU faults, and the job times
out with every register correct. The last exact value is the floor; everything below
it is unexplored rather than known bad.

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
