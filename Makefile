# SPDX-License-Identifier: GPL-2.0-or-later
#
# make            host tools (the emitter dump, for the offline vendor diff)
# make board      the same plus charsiu_probe, cross compiled for the board
#
# CROSS is buildroot's toolchain from the linux-rk3576-npu tree by default,
# because that is what builds the image this runs in.
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -Iinclude
BUILD  := build
CROSS  ?= $(HOME)/Desktop/linux-rk3576-npu/buildroot/br-out/host/bin/aarch64-buildroot-linux-gnu-

SRC    := src/regcmd.c src/device.c src/job.c
LLM    := src/gguf.c src/tokenizer.c src/llama.c src/npuquant.c \
          src/npudev.c src/device.c src/job.c src/regcmd.c

all: $(BUILD)/emit_dump $(BUILD)/emit_job $(BUILD)/charsiu_run

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

# the control: same code with the NEON kernels compiled out. It has to produce
# the same tokens as the one above and be slower, on the board as well as here.
$(BUILD)/charsiu_run_scalar.aarch64: tools/charsiu_run.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -DCHARSIU_NO_NEON -static -o $@ $^ -lm -lpthread

board: $(BUILD)/charsiu_probe.aarch64 $(BUILD)/charsiu_matmul.aarch64 \
       $(BUILD)/charsiu_bench.aarch64 $(BUILD)/charsiu_int4.aarch64 \
       $(BUILD)/charsiu_run.aarch64 $(BUILD)/charsiu_run_scalar.aarch64 \
       $(BUILD)/charsiu_wide.aarch64

$(BUILD)/charsiu_probe.aarch64: tools/charsiu_probe.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^

$(BUILD)/charsiu_matmul.aarch64: tools/charsiu_matmul.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

$(BUILD)/charsiu_bench.aarch64: tools/charsiu_bench.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

$(BUILD)/charsiu_int4.aarch64: tools/charsiu_int4.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

$(BUILD)/charsiu_wide.aarch64: tools/charsiu_wide.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

test: $(BUILD)/pack_int4
	./$(BUILD)/pack_int4

$(BUILD)/pack_int4: tests/pack_int4.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -rf $(BUILD)

.PHONY: all board test clean
