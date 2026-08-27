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

The step this section once named next was: Mesa splits a task's staged rows when
`(cbuf_rows + staged) * entries_per_slice > total_entries` and charsiu never checks.
Put the numbers in and it cannot fire.

An LLM matmul reaches the encoder as `input_width = 1`, `input_height = m`,
`input_channels = K`, so `entries_per_slice` is 16 at K = 1024 and 32 at K = 2048.
The RK3576 CBUF is 16 banks of 512 entries; Mesa's own budget is five banks usable
and ten total. At m = 8 and K = 1024 that is **128 entries against 2560**. The
over-budget test needs m > 320 and the split test needs m > 640; the row-window path
needs `input_width >= 112`, and this shape is one wide. **Mesa does not split these
either.** That much still stands.

### And `surf` is not it either. The control failed.

The next step after that was: every shape ever correct above one row has
`charsiu_entries_per_row() == 1`, so the axis is entries per row rather than m. The
rule was stated before the run -- K = 48 and K = 64 are surf 1 and have to be exact
at m = 2 and m = 4, or the axis is not surf. **They were not.**

```
     K  surf   m   exact of      all values present
    48     1   2       8 of 128       99 of 128
    64     1   2       8 of 128       97 of 128
   128     2   2       8 of 128       93 of 128
   256     4   2       8 of 128       92 of 128
  1024    16   2       8 of 128       95 of 128
```

Flat across the whole sweep, surf 1 to surf 16. **8 words exact at m = 2 and 8 at
m = 4, at every K.** Whatever this is, it does not scale with K, and the `written =
N + (N/4 + 12)(m - 1)` budget that fitted one K does not survive the others either.

### What the record was actually comparing: two different output paths

The shapes that computed correctly above one row -- M = 224 at K = 64, M = 3136 at
K = 33 -- were measured by `tools/charsiu_matmul.c`, which takes the **requantised
int8 output** and reads it as `[n/atom][m][n%atom]` with a 16 byte atom. Every m > 1
failure was measured by `tools/npu_gemm_test.c`, which sets `acc_out = 1` for the
**raw int32 accumulator** and reads it flat.

They have never run the same experiment. The runtime's decode path uses `acc_out`,
so the layout that matters has never been established above one row, and the layout
that was established does not apply to it.

⚠ The obvious guess is that the accumulator mirrors the same surface with a four
word atom. It predicts 8 exact at m = 2, which is what the board wrote, and 16 at
m = 4, where the board wrote 8. Right at one width and wrong at the next is the same
shape of wrong answer the last four rounds produced, so it is written down here and
not acted on.

**Next**: run both probes at the same shape in one session, `charsiu_matmul 2 64 64`
against `npu_gemm_test 64 64 --surf`. If the int8 path is exact at m = 2 where the
accumulator path is not, then m > 1 is a reading problem in the accumulator path
rather than a hardware wall, and the search collapses to one address function with a
control that already works. If the int8 path fails too, the README's own table is
stale and that has to be said.

**Control**: `CHARSIU_OUT_ROWMAJOR=1 charsiu_matmul 2 64 64` reads the same run flat.
It has to fail where the surface reading passes, or the surface reading is not doing
any work.

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
