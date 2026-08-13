# Plan

The goal is the vendor's own number on the same board and the same model, with
nothing closed in the execution path. Whatever the NPU can carry goes on the NPU.

Each step below states what it produces and how it is known to be right. The rule
carried over from the driver work applies here too: **a board round states its
decision rule before the run and carries a control that can fail**, and a proxy
metric is not a correctness oracle.

## 0. The instrument (done)

`tools/rkllm_regcmd.py` reads the vendor's dispatch plan out of a `.rkllm` with no
board. First reading is in [docs/vendor-dispatch.md](docs/vendor-dispatch.md): three
precisions by role, projections split across the two cores, and 3752 dispatches at
M = 1.

## 1. Finish reading the vendor

Offline, and cheap.

- Decode the 12724 DPU-only streams. Those are the elementwise and activation work,
  and they say how much of RMSNorm, RoPE, softmax and the residual adds the vendor
  keeps on the NPU rather than the cores.
- Read a second model, ideally a different architecture, to separate what is
  Llama-shaped from what is runtime policy.
- Pull the same streams from a live run with kiln's `rknpu-regcmd-dump.patch` and
  diff them against the file, so the ordering and the sync/broadcast entries the file
  does not resolve are known too.

## 2. Reproduce one vendor matmul, bit exact

On the board, through mainline `rocket`.

Take one int4 projection from the file, 2048 to 1024 at M = 1, build the same
register stream from our own code, feed it operands we control, and compare against a
CPU reference. This is the whole project in miniature: if a single vendor-shaped
matmul comes out exact, everything after it is engineering rather than discovery.

The RK3576 pieces this needs are already established in the driver work: the weight
tile layout, the CBUF row cost, the coefficient buffer alignment, the output channel
pair rounding, and the split pair rule.

What is new and unproven here: **int4 weights** and **M = 1**. Neither has ever been
run on this board through the open path.

## 3. The measurement that decides the architecture

Time one projection on the NPU against the same matmul on four A72 cores, at M = 1
and at M = 32.

The RK3588 open stack concluded decode belongs on the CPU. The RK3576 vendor
dispatches decode to the NPU. Both cannot be the right answer for this board, and one
number settles which, before any runtime is designed around either.

## 4. The runtime

Only after 3. The shape it takes depends on what 3 says:

- if M = 1 is competitive on this hardware, charsiu can run a whole model on the NPU,
  which is what the vendor does;
- if it is not, charsiu runs prefill and attention on the NPU and decode on the cores,
  which is what the RK3588 stack does.

Either way it needs: tiled weight packing, the two-core fan-out, a KV cache, and a
frontend. The frontend question, whether that is a ggml backend or a standalone
runtime, is deliberately left open until 3.

## 5. The product layer

Reusable from [kiln](https://github.com/gahingwoo/kiln) with the NPU half replaced:
the installer, the OpenAI-compatible API server, the model conversion path, the
health check. None of that is NPU code and none of it needs rewriting.

## What would make this stop

Stated in advance, so it is a decision rather than a slow fade:

- if step 2 cannot make one vendor-shaped matmul exact after a fair attempt, the
  honest outcome is a written negative result and a return to the driver work;
- if step 3 says the NPU cannot beat four A72 cores at any shape an LLM needs, then
  the answer for this board is llama.cpp on the CPU, and saying so is worth more than
  a runtime nobody should use.
