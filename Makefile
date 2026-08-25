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
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -Iinclude
BUILD  := build
BRCROSS := $(HOME)/Desktop/linux-rk3576-npu/buildroot/br-out/host/bin/aarch64-buildroot-linux-gnu-
CROSS  ?= $(if $(wildcard $(BRCROSS)gcc),$(BRCROSS),)

SRC    := src/regcmd.c src/device.c src/job.c
LLM    := src/gguf.c src/tokenizer.c src/llama.c src/npuquant.c \
          src/npudev.c src/device.c src/job.c src/regcmd.c

all: $(BUILD)/emit_dump $(BUILD)/emit_job $(BUILD)/charsiu_run \
     $(BUILD)/charsiu_check

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/emit_dump: tools/emit_dump.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD)/emit_job: tools/emit_job.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# the CPU decode loop: the oracle every NPU version is diffed against
$(BUILD)/charsiu_run: tools/charsiu_run.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

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
       $(BUILD)/charsiu_membw.aarch64 $(BUILD)/charsiu_check.aarch64

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

test: $(BUILD)/pack_int4
	./$(BUILD)/pack_int4

$(BUILD)/pack_int4: tests/pack_int4.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -rf $(BUILD)

.PHONY: all board test clean
