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

**Next**: find what makes the per-row allowance `N/4 + 12` rather than `N`. Mesa
bounds exactly this and charsiu does not -- `rkt_task.c` splits a task's staged rows
when `(cbuf_rows + staged) * entries_per_slice > total_entries`, and charsiu submits
all m rows as one task and never checks.

**Control**: `npu_gemm_test` sweeps M and both entry-atomic constants, prints the
whole output as the (row, channel) each word holds, and separates *wrong values* from
*right values in the wrong order*. It has been wrong in both directions and says so
now.

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
