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

all: $(BUILD)/emit_dump $(BUILD)/emit_job

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/emit_dump: tools/emit_dump.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD)/emit_job: tools/emit_job.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

board: $(BUILD)/charsiu_probe.aarch64 $(BUILD)/charsiu_matmul.aarch64 \
       $(BUILD)/charsiu_bench.aarch64 $(BUILD)/charsiu_int4.aarch64

$(BUILD)/charsiu_probe.aarch64: tools/charsiu_probe.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^

$(BUILD)/charsiu_matmul.aarch64: tools/charsiu_matmul.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

$(BUILD)/charsiu_bench.aarch64: tools/charsiu_bench.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

$(BUILD)/charsiu_int4.aarch64: tools/charsiu_int4.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

test: $(BUILD)/pack_int4
	./$(BUILD)/pack_int4

$(BUILD)/pack_int4: tests/pack_int4.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -rf $(BUILD)

.PHONY: all board test clean
