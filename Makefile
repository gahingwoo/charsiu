# SPDX-License-Identifier: GPL-2.0-or-later
CFLAGS ?= -O2 -Wall -Wextra -std=c11 -Iinclude
BUILD  := build

all: $(BUILD)/emit_dump

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/emit_dump: tools/emit_dump.c src/regcmd.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^

clean:
	rm -rf $(BUILD)

.PHONY: all clean
