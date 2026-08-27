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

262144 wide, tied at q8_0, 428 MB every token, and on the CPU because it will not
route:

```
  weights       151 MB    32 slices at N=8192, int4
  coefficients 1210 MB    <-- charsiu_coef_bytes bounds it by k*n
```

The bound is a guess and its own comment has said so since it was written: the two
walls it was sized against were tens of KILObytes, and nothing has ever measured the
read growing with `k*n`. At 65536 elements the same head wants 10.5 MB and fits.

**Control**: `npu_gemm_test K N --coef` walks the bound downward and stops at the
first value that is not exact. ⚠ It walks DOWN because under-allocating does not
return an error -- the RDMA reads past the buffer, the IOMMU faults, and the job times
out with every register correct. The last exact value is the floor; everything below
it is unexplored rather than known bad.

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
- if the coefficient bound turns out to be real rather than a guess, the output head
  stays on the CPU and gemma-shaped models are simply expensive on this board, which
  is worth saying rather than working around.
