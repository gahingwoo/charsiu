# SPDX-License-Identifier: GPL-2.0-or-later
#
# make            host tools (the emitter dump, for the offline vendor diff)
# make board      the same plus charsiu_probe, cross compiled for the board
#
# CROSS was buildroot's toolchain from the linux-rk3576-npu tree. That tree was
# cleared when the project moved to a debootstrap Debian image, so the default
# is now: use it if it is still there, otherwise build NATIVELY, which works
# because the development host is itself aarch64. Override CROSS= for anything
# else.
# ⚠ -Winfinite-recursion IS NOT IN -Wall OR -Wextra, and it would have caught
# the one that mattered: act_q1_timed called itself instead of charsiu_act_q1,
# which killed every NPU run on the board and was invisible in an -O2 build
# because the compiler is entitled to delete an infinite recursion. Named
# explicitly so a gcc that lacks it is not a silent downgrade; ?= means a
# caller can still override the whole line.
CFLAGS ?= -O2 -Wall -Wextra -Winfinite-recursion -std=c11 -Iinclude
BUILD  := build
BRCROSS := $(HOME)/Desktop/linux-rk3576-npu/buildroot/br-out/host/bin/aarch64-buildroot-linux-gnu-
CROSS  ?= $(if $(wildcard $(BRCROSS)gcc),$(BRCROSS),)

SRC    := src/regcmd.c src/device.c src/job.c
LLM    := src/gguf.c src/tokenizer.c src/llama.c src/npuquant.c \
          src/npudev.c src/npupool.c src/device.c src/job.c src/regcmd.c

all: $(BUILD)/emit_dump $(BUILD)/emit_job $(BUILD)/charsiu_run \
     $(BUILD)/charsiu_check $(BUILD)/charsiu_serve $(BUILD)/bench_batch \
     $(BUILD)/npu_gemm_test $(BUILD)/npu_slice_test $(BUILD)/npu_fp16_test \
     $(BUILD)/charsiu_matmul \
     $(BUILD)/charsiu_vision $(BUILD)/charsiu_clip \
     $(BUILD)/charsiu_whisper \
     $(BUILD)/vattn_bench \
     $(BUILD)/tokenizer_roundtrip $(BUILD)/acc_index_check

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/emit_dump: tools/emit_dump.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD)/emit_job: tools/emit_job.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# ⚠ THE OTHER m > 1 PROBE, AND IT HAD NO NATIVE TARGET AT ALL.
#
# npu_gemm_test asks the hardware for the raw int32 accumulator and reads it
# flat; this one takes the requantised int8 output and reads it as a surface,
# and it is the only thing that has ever computed more than one row correctly
# (M = 224 and M = 3136, August). Those two answers have to be comparable in one
# session on one build, and until now this could only be reached through
# `make board`, which cross compiles. A board round asking for it got
# "No such file or directory".
$(BUILD)/charsiu_matmul: tools/charsiu_matmul.c $(SRC) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# ⚠ WHAT AN mmproj ACTUALLY CONTAINS, against what this reads. Every vision
# tensor name in the tree is a guess until a real file says otherwise, and a
# guess that finds nothing has cost this project a model that answered while
# missing half of itself. This prints the misses by name.
$(BUILD)/charsiu_vision: tools/charsiu_vision.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -Ithird_party -o $@ $^ -lm -lpthread

# ⚠ THE ATTENTION IS HALF THE BOARD'S ENCODE AND 6% OF THE HOST'S, because the
# board's matmuls go to the NPU and the host's do not. Timing it through the
# whole tower on a host is reading the feed forward's noise; this drives the
# one stage, at whatever shape is asked for.
$(BUILD)/vattn_bench: tools/vattn_bench.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -Ithird_party -o $@ $^ -lm -lpthread

$(BUILD)/vattn_bench.aarch64: tools/vattn_bench.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -Ithird_party -static -o $@ $^ -lm -lpthread

# ⚠ CLIP IS TWO TOWERS AND ONE SPACE, and the text one is not the language
# model's: causal, pooled at the end of text token, its own BPE.
$(BUILD)/charsiu_clip: tools/charsiu_clip.c src/vision.c src/clip.c src/image.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -Ithird_party -o $@ $^ -lm -lpthread

# ⚠ WHISPER READS ITS OWN CONTAINER, not a gguf: whisper.cpp's format is what
# every model anybody has is in, and it carries the mel filterbank and the
# vocabulary as well as the weights.
$(BUILD)/charsiu_whisper: tools/charsiu_whisper.c src/whisper.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

# the CPU decode loop: the oracle every NPU version is diffed against
$(BUILD)/charsiu_run: tools/charsiu_run.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -Ithird_party -o $@ $^ -lm -lpthread

$(BUILD)/charsiu_run_scalar: tools/charsiu_run.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -DCHARSIU_NO_NEON -o $@ $^ -lm -lpthread

$(BUILD)/charsiu_run.aarch64: tools/charsiu_run.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm -lpthread

# The control: same code with the NEON kernels compiled out, and slower.
#
# ⚠ SINCE ROUND 372 IT NO LONGER MATCHES THE DEFAULT BUILD, and the invariant
# is written differently rather than quietly dropped. Two of the vector paths
# reorder arithmetic on purpose -- the q.k dot product sums in four lanes and
# the exponential is a polynomial rather than glibc's -- and neither has a
# scalar twin. So:
#
#   charsiu_run_scalar  ==  charsiu_run with CHARSIU_EXACT_ATTN and
#                           CHARSIU_EXACT_SILU set
#
# which is checked on the host and is still a NEON bug detector: every vector
# path that is meant to be bit identical still has to reproduce it.
$(BUILD)/charsiu_run_scalar.aarch64: tools/charsiu_run.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -DCHARSIU_NO_NEON -static -o $@ $^ -lm -lpthread

board: $(BUILD)/charsiu_probe.aarch64 $(BUILD)/charsiu_matmul.aarch64 \
       $(BUILD)/charsiu_bench.aarch64 $(BUILD)/charsiu_int4.aarch64 \
       $(BUILD)/charsiu_run.aarch64 $(BUILD)/charsiu_run_scalar.aarch64 \
       $(BUILD)/charsiu_wide.aarch64 $(BUILD)/charsiu_vendor.aarch64 \
       $(BUILD)/charsiu_membw.aarch64 $(BUILD)/charsiu_check.aarch64 \
       $(BUILD)/charsiu_serve.aarch64 $(BUILD)/charsiu_vision.aarch64 \
       $(BUILD)/charsiu_clip.aarch64 $(BUILD)/charsiu_whisper.aarch64 \
       $(BUILD)/vattn_bench.aarch64

# ⚠ THE OTHER MODALITIES CROSS COMPILE TOO. `make board` is the target a board
# round reaches for, and a tool that is only in the native build is one that has
# to be rebuilt on the card before it can be asked anything.
$(BUILD)/charsiu_vision.aarch64: tools/charsiu_vision.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -Ithird_party -static -o $@ $^ -lm -lpthread

$(BUILD)/charsiu_clip.aarch64: tools/charsiu_clip.c src/vision.c src/clip.c src/image.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -Ithird_party -static -o $@ $^ -lm -lpthread

$(BUILD)/charsiu_whisper.aarch64: tools/charsiu_whisper.c src/whisper.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm -lpthread

# what the memory controller has left, which is the only question that decides
# whether splitting work across the CPU, the GPU and the NPU can pay
$(BUILD)/charsiu_membw.aarch64: tools/charsiu_membw.c | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lpthread

$(BUILD)/charsiu_membw: tools/charsiu_membw.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lpthread

$(BUILD)/charsiu_probe.aarch64: tools/charsiu_probe.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^

$(BUILD)/charsiu_matmul.aarch64: tools/charsiu_matmul.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

$(BUILD)/charsiu_bench.aarch64: tools/charsiu_bench.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

$(BUILD)/charsiu_int4.aarch64: tools/charsiu_int4.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

$(BUILD)/charsiu_vendor.aarch64: tools/charsiu_vendor.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -o $@ $^ -lm

$(BUILD)/charsiu_wide.aarch64: tools/charsiu_wide.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

# The gate charsiu-get calls before it keeps a download. It uses the RUNTIME's
# own gguf parser, so what it accepts cannot drift from what will load.
$(BUILD)/charsiu_check: tools/charsiu_check.c src/gguf.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/charsiu_check.aarch64: tools/charsiu_check.c src/gguf.c | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

# An OpenAI compatible endpoint, so every chat front end that already exists
# works against this board without anyone writing a GUI first.
$(BUILD)/charsiu_serve: tools/charsiu_serve.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/bench_batch: tools/bench_batch.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/npu_gemm_test: tools/npu_gemm_test.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/npu_slice_test: tools/npu_slice_test.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/npu_fp16_test: tools/npu_fp16_test.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/acc_index_check: tools/acc_index_check.c $(SRC) | $(BUILD)
	$(CROSS)$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/bench_gather: tools/bench_gather.c $(SRC) | $(BUILD)
	$(CROSS)$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/tokenizer_roundtrip: tools/tokenizer_roundtrip.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/charsiu_serve.aarch64: tools/charsiu_serve.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm -lpthread

test: $(BUILD)/pack_int4 $(BUILD)/reuse_key $(BUILD)/overlap_guard $(BUILD)/pack_stride $(BUILD)/even_ks $(BUILD)/pack_f16w
	./$(BUILD)/pack_int4
	./$(BUILD)/reuse_key
	./$(BUILD)/overlap_guard
	./$(BUILD)/pack_stride
	CHARSIU_NPU_PLAIN=1 ./$(BUILD)/pack_stride
	./$(BUILD)/even_ks
	./$(BUILD)/pack_f16w

$(BUILD)/pack_int4: tests/pack_int4.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/pack_stride: tests/pack_stride.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/pack_f16w: tests/pack_f16w.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# the guard is its own unit for exactly this reason: the table is testable on a
# desk without linking the hardware path behind it
$(BUILD)/overlap_guard: tests/overlap_guard.c src/overlap.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD)/reuse_key: tests/reuse_key.c src/reusekey.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD)/even_ks: tests/even_ks.c src/kslice.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(BUILD)

.PHONY: all board test clean
