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
# -Winfinite-recursion IS NOT IN -Wall OR -Wextra, and it would have caught
# the one that mattered: act_q1_timed called itself instead of charsiu_act_q1,
# which killed every NPU run on the board and was invisible in an -O2 build
# because the compiler is entitled to delete an infinite recursion. Named
# explicitly so a gcc that lacks it is not a silent downgrade; ?= means a
# caller can still override the whole line.
CFLAGS ?= -O2 -Wall -Wextra -Winfinite-recursion -std=c11 -Iinclude
#
# THE BUILD STAMPS ITSELF, because /opt/charsiu is not a git checkout and
# neither is /root/charsiu_run_<whatever>. Every board round has recorded the
# machine, the clock and the wall time and NOT the commit, so a round's numbers
# have been tied to a version by somebody remembering which binary they copied.
# That is how "the numerator and the denominator came from different builds"
# happens without anybody noticing.
#
# -dirty is part of it. A binary built from an edited tree is not the commit it
# names, and saying so is the whole point.
#
# := AND NOT =, or every compile line re-runs git.
CHARSIU_BUILD := $(shell git -C $(CURDIR) describe --always --dirty --abbrev=12 2>/dev/null || echo unknown)
#
# override, NOT a plain +=. CFLAGS is `?=` above, and a value given on the
# COMMAND LINE beats both -- make then ignores every `+=` to it, the define
# never reaches the compiler, and charsiu.h's fallback makes --version answer
# "unknown". An environment CFLAGS is fine; only the command line does this.
# A stamp that silently disappears under `make CFLAGS=...` is worse than none.
override CFLAGS += -DCHARSIU_BUILD=\"$(CHARSIU_BUILD)\"
BUILD  := build
BRCROSS := $(HOME)/Desktop/linux-rk3576-npu/buildroot/br-out/host/bin/aarch64-buildroot-linux-gnu-
CROSS  ?= $(if $(wildcard $(BRCROSS)gcc),$(BRCROSS),)

SRC    := src/regcmd.c src/device.c src/job.c
LLM    := src/gguf.c src/tokenizer.c src/llama.c src/npuquant.c \
          src/npudev.c src/npupool.c src/npufp16.c src/device.c src/job.c src/regcmd.c
# THE SAME LIST WITHOUT llama.c, for one test that IS a llama.c translation
# unit. tests/attn_two_thresholds.c includes src/llama.c so it can reach the
# two static threshold functions and the gate that composes them, so llama.c
# must not also arrive as an object or every symbol in it is defined twice.
# Its RULE still depends on $(LLM), llama.c included, or an edit to the file
# it tests would not rebuild it.
LLMNOLL := src/gguf.c src/tokenizer.c src/npuquant.c \
          src/npudev.c src/npupool.c src/npufp16.c src/device.c src/job.c src/regcmd.c

all: $(BUILD)/emit_dump $(BUILD)/emit_job $(BUILD)/charsiu_run \
     $(BUILD)/charsiu_check $(BUILD)/charsiu_serve $(BUILD)/bench_batch \
     $(BUILD)/npu_gemm_test $(BUILD)/npu_slice_test $(BUILD)/npu_fp16_test \
     $(BUILD)/npu_mixed_test $(BUILD)/out16_bound \
     $(BUILD)/npu_qpack_test $(BUILD)/npu_prep_cost $(BUILD)/npu_job_cost $(BUILD)/charsiu_shapes $(BUILD)/npu_out_fmt \
     $(BUILD)/charsiu_matmul $(BUILD)/npu_fence_scan \
     $(BUILD)/charsiu_vision $(BUILD)/charsiu_clip \
     $(BUILD)/charsiu_whisper \
     $(BUILD)/vattn_bench \
     $(BUILD)/tokenizer_roundtrip $(BUILD)/acc_index_check \
     $(BUILD)/fp16_plan \
     $(BUILD)/fp16_regrow \
     $(BUILD)/charsiu_ppl \
     $(BUILD)/charsiu_membw

#
# AND THE STAMP HAS TO GO STALE NEVER. Nothing in a link line depends on the
# commit, so a pull that changes no source leaves the OLD commit inside a
# binary that is otherwise up to date -- and then it answers --version with a
# confident wrong number, which is worse than not answering. This file changes
# only when the commit does (cmp before write, so an unchanged commit does not
# relink the world), and the binaries that carry the stamp depend on it.
STAMP  := $(BUILD)/.commit

.PHONY: FORCE
FORCE:

$(STAMP): FORCE | $(BUILD)
	@printf '%s\n' '$(CHARSIU_BUILD)' | cmp -s - $@ 2>/dev/null \
		|| printf '%s\n' '$(CHARSIU_BUILD)' > $@

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/emit_dump: tools/emit_dump.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD)/emit_job: tools/emit_job.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# AND STATICALLY, FOR THE BOARD. tools/cmp_vendor.py needs this emitter and
# the vendor's .rkllm in the same place, and the .rkllm is 1.3 GB on a board
# whose desk has 2.5 GB free -- so the diff runs there, not here.
$(BUILD)/emit_job.aarch64: tools/emit_job.c src/regcmd.c src/job.c | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

# THE OTHER m > 1 PROBE, AND IT HAD NO NATIVE TARGET AT ALL.
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

# WHAT AN mmproj ACTUALLY CONTAINS, against what this reads. Every vision
# tensor name in the tree is a guess until a real file says otherwise, and a
# guess that finds nothing has cost this project a model that answered while
# missing half of itself. This prints the misses by name.
$(BUILD)/charsiu_vision: tools/charsiu_vision.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -Ithird_party -o $@ $^ -lm -lpthread

# THE ATTENTION IS HALF THE BOARD'S ENCODE AND 6% OF THE HOST'S, because the
# board's matmuls go to the NPU and the host's do not. Timing it through the
# whole tower on a host is reading the feed forward's noise; this drives the
# one stage, at whatever shape is asked for.
$(BUILD)/vattn_bench: tools/vattn_bench.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -Ithird_party -o $@ $^ -lm -lpthread

$(BUILD)/vattn_bench.aarch64: tools/vattn_bench.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -Ithird_party -static -o $@ $^ -lm -lpthread

# CLIP IS TWO TOWERS AND ONE SPACE, and the text one is not the language
# model's: causal, pooled at the end of text token, its own BPE.
$(BUILD)/charsiu_clip: tools/charsiu_clip.c src/vision.c src/clip.c src/image.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -Ithird_party -o $@ $^ -lm -lpthread

# WHISPER READS ITS OWN CONTAINER, not a gguf: whisper.cpp's format is what
# every model anybody has is in, and it carries the mel filterbank and the
# vocabulary as well as the weights.
$(BUILD)/charsiu_whisper: tools/charsiu_whisper.c src/whisper.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

# the CPU decode loop: the oracle every NPU version is diffed against
$(BUILD)/charsiu_run: tools/charsiu_run.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -Ithird_party -o $@ $^ -lm -lpthread

$(BUILD)/charsiu_ppl: tools/charsiu_ppl.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

#
# AND IT WENT STALE THE SAME WAY, IN BOTH SCALAR RULES. The paragraph
# below this one says "a target that is never built is a target that is
# already broken" about exactly this, and then the two charsiu_run_scalar
# rules kept the pre-vision source list anyway -- so tests/neon_control.sh,
# the tree's own control for "a vector kernel that is wrong still produces
# fluent text", has not linked since vision landed and therefore has not run.
# `make test` builds it now, which is the only thing that keeps a list honest.
#
$(BUILD)/charsiu_run_scalar: tools/charsiu_run.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -Ithird_party -DCHARSIU_NO_NEON -o $@ $^ -lm -lpthread

#
# THE SOURCE LIST HAS TO TRACK charsiu_run's. This rule went stale when
# vision landed: it kept the old list, so the board's static binary stopped
# linking (undefined charsiu_vision_open and three more) and nobody noticed,
# because nothing builds it by default. A target that is never built is a
# target that is already broken.
#
$(BUILD)/charsiu_run.aarch64: tools/charsiu_run.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -Ithird_party -static -o $@ $^ -lm -lpthread

# THE SHAPE PROBE, ON THE BOARD. npu_fp16_test only ever built for the host,
# and the host build is dynamically linked against a glibc the board does not
# have -- so the one tool that can price an fp16 matmul at a CHOSEN shape could
# not be run where the shapes matter. Same caveat as every rule here: this
# source list has to track npu_fp16_test's.
$(BUILD)/npu_fp16_test.aarch64: tools/npu_fp16_test.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm -lpthread

# THE QUALITY INSTRUMENT, ON THE BOARD. charsiu_ppl only ever built for the
# host, so every perplexity and every --top1 in this tree came from the CPU
# path. The board's own quantiser is the one that ships, and scoring it needed
# this target to exist. Same caveat as the rule above: this source list has to
# track charsiu_ppl's.
$(BUILD)/charsiu_ppl.aarch64: tools/charsiu_ppl.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -Ithird_party -static -o $@ $^ -lm -lpthread

# The control: same code with the NEON kernels compiled out, and slower.
#
# SINCE ROUND 372 IT NO LONGER MATCHES THE DEFAULT BUILD, and the invariant
# is written differently rather than quietly dropped. Some vector paths reorder
# arithmetic on purpose and have no scalar twin. So:
#
#   charsiu_run_scalar  ==  charsiu_run with CHARSIU_EXACT_ATTN,
#                           CHARSIU_EXACT_SILU and CHARSIU_EXACT_SOFTMAX set
#
# which is checked on the host and is still a NEON bug detector: every vector
# path that is meant to be bit identical still has to reproduce it.
#
# THIS LIST SAID TWO UNTIL ROUND 414 AND THERE WERE THREE. It was written
# when the reordering paths were the q.k dot product (four lanes) and the
# exponential (a polynomial rather than glibc's), and it did not grow when the
# softmax joined them. tests/neon_control.sh had never set ANY of them, so the
# gap was invisible; the moment it set the two named here, two models still
# differed, and adding CHARSIU_EXACT_SOFTMAX closed both:
#
#   ATTN + SILU             Llama-Q8_0 DIFFERS     Qwen3-Q4_0 DIFFERS
#   ATTN + SILU + GELU      DIFFERS                DIFFERS
#   ATTN + SILU + SOFTMAX   IDENTICAL              IDENTICAL
#
# CHARSIU_EXACT_GELU is NOT in the invariant: that path is meant to be bit
# identical and the regression checks it separately. An invariant that names a
# knob it does not need hides the regression that knob exists to catch.
$(BUILD)/charsiu_run_scalar.aarch64: tools/charsiu_run.c src/vision.c src/image.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -Ithird_party -DCHARSIU_NO_NEON -static -o $@ $^ -lm -lpthread

board: $(BUILD)/charsiu_probe.aarch64 $(BUILD)/charsiu_matmul.aarch64 \
       $(BUILD)/charsiu_bench.aarch64 $(BUILD)/charsiu_int4.aarch64 \
       $(BUILD)/charsiu_run.aarch64 $(BUILD)/charsiu_run_scalar.aarch64 \
       $(BUILD)/charsiu_wide.aarch64 $(BUILD)/charsiu_vendor.aarch64 \
       $(BUILD)/charsiu_membw.aarch64 $(BUILD)/charsiu_check.aarch64 \
       $(BUILD)/charsiu_serve.aarch64 $(BUILD)/charsiu_vision.aarch64 \
       $(BUILD)/charsiu_clip.aarch64 $(BUILD)/charsiu_whisper.aarch64 \
       $(BUILD)/vattn_bench.aarch64 $(BUILD)/npu_fp16_test.aarch64

# THE OTHER MODALITIES CROSS COMPILE TOO. `make board` is the target a board
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

# AND src/gguf.c, WHICH IS WHERE charsiu_env_flag LIVES. This rule is in
# `board:` and has not linked -- a third instance today of the sentence three
# rules above: a target that is never built is a target that is already broken.
# Nothing in `make test` builds it either, which is why nothing said so.
$(BUILD)/charsiu_int4.aarch64: tools/charsiu_int4.c $(SRC) src/gguf.c | $(BUILD)
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

# NO $(LLM). This one links nothing: it is the two arms of one loop lifted
# out of npudev.c so a rewrite of them can be proved identical without a
# model, a device, or a board that has to boot first.
$(BUILD)/npu_qpack_test: tools/npu_qpack_test.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $< -lm

$(BUILD)/npu_prep_cost: tools/npu_prep_cost.c src/device.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/npu_job_cost: tools/npu_job_cost.c src/device.c src/job.c src/regcmd.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/charsiu_shapes: tools/charsiu_shapes.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/npu_out_fmt: tools/npu_out_fmt.c src/device.c src/job.c src/regcmd.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/npu_fence_scan: tools/npu_fence_scan.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/npu_slice_test: tools/npu_slice_test.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/npu_fp16_test: tools/npu_fp16_test.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/npu_mixed_test: tools/npu_mixed_test.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/out16_bound: tools/out16_bound.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/acc_index_check: tools/acc_index_check.c $(SRC) | $(BUILD)
	$(CROSS)$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/bench_gather: tools/bench_gather.c $(SRC) | $(BUILD)
	$(CROSS)$(CC) $(CFLAGS) -o $@ $^ -lm

# AND THE BOARD BUILD, because this tool's OWN HEADER says the host cannot
# answer its question -- "the host is aarch64 with caches that dwarf the
# board's ... the ratio is the thing to carry". It had a host rule only, and
# no board log mentions it, so the one instrument written for the largest line
# in the prefill had never been run where the argument lives.
$(BUILD)/bench_gather.aarch64: tools/bench_gather.c $(SRC) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm

$(BUILD)/tokenizer_roundtrip: tools/tokenizer_roundtrip.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BUILD)/charsiu_serve.aarch64: tools/charsiu_serve.c $(LLM) | $(BUILD)
	$(CROSS)gcc $(CFLAGS) -static -o $@ $^ -lm -lpthread

test: $(BUILD)/pack_int4 $(BUILD)/reuse_key $(BUILD)/overlap_guard $(BUILD)/pack_stride $(BUILD)/even_ks $(BUILD)/pack_f16w $(BUILD)/pack_w8 $(BUILD)/patch_waddr $(BUILD)/softmax_half $(BUILD)/pack_f16run $(BUILD)/sentinel $(BUILD)/coef_scales $(BUILD)/fp16_plan $(BUILD)/fp16_regrow $(BUILD)/fp16_regrow_fuzz $(BUILD)/pack_groups $(BUILD)/axpy8 $(BUILD)/attn_two_thresholds $(BUILD)/acc_index_check $(BUILD)/vattn_bench $(BUILD)/charsiu_run_scalar
#
# THE WIDTH LAW, WHICH NOTHING RAN. tools/acc_index_check.c says of itself
# that it ASSERTS the law rather than only printing it, and that it is the
# check that should have existed before the first board round -- and it was
# built by `all` and invoked by no target, so four rounds' worth of proof sat
# in a binary nobody executed. 0.14 s.
#
	./$(BUILD)/acc_index_check >/dev/null
	./$(BUILD)/pack_int4
	./$(BUILD)/reuse_key
	./$(BUILD)/overlap_guard
	./$(BUILD)/pack_stride
	CHARSIU_NPU_PLAIN=1 ./$(BUILD)/pack_stride
	./$(BUILD)/even_ks
	./$(BUILD)/pack_f16w
	./$(BUILD)/pack_w8
	./$(BUILD)/patch_waddr
	./$(BUILD)/softmax_half
	./$(BUILD)/pack_f16run
	./$(BUILD)/sentinel
	./$(BUILD)/coef_scales
	./$(BUILD)/fp16_plan
	./$(BUILD)/fp16_regrow
	./$(BUILD)/fp16_regrow_fuzz
	./$(BUILD)/pack_groups
	./$(BUILD)/axpy8
	./$(BUILD)/attn_two_thresholds
	./tests/corpus_fixed.sh
	./tests/probe_list.sh
	./tests/arch_list.sh
#
# TWO MORE CHECKERS THAT NEEDED NEITHER A BOARD NOR A MODEL AND THAT NOTHING
# RAN. verify_selftest.sh asks whether every phase of board_verify.sh can run
# alone, and whether anything still pins the K slice width that the product
# now chooses -- the second of those already shipped a wrong answer once, a
# round reporting nine of nine models correct in a configuration that had
# stopped being the one that ships. vattn_edges.sh puts the fused vision
# attention against the exact one at the ragged shapes vision_cross never
# reaches. Both were installed to the board and invoked by no target: 1.1 s
# and 0.2 s. Watched failing, the second on a stub that reports 3.10e-02.
#
	./tests/verify_selftest.sh
	./tests/vattn_edges.sh $(BUILD)/vattn_bench
	./tests/version_all.sh $(BUILD)
	./tests/host_checks.sh
#
# THE PACK CHECKED AGAINST ITS OWN RULES. vendor-quality-provenance.md
# specified "every perplexity must name a file whose md5 appears in the
# reproduction section" on 09-11 and nobody implemented it. Run for the first
# time on 09-12 it failed at once: both corpora were scored by every
# perplexity in the pack and NEITHER md5 was recorded. A rule nothing runs is
# a sentence.
#
	python3 -P tools/check_consistency.py --self-test
	python3 -P tools/check_consistency.py docs/paper-evidence.md
	python3 -P tools/ttft_compare.py --self-test

$(BUILD)/pack_int4: tests/pack_int4.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/pack_stride: tests/pack_stride.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BUILD)/pack_f16w: tests/pack_f16w.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# charsiu_w8_offset against charsiu_pack_weights, cell by cell. Two pieces of
# code describing one permutation is the arrangement that lets a KV surface be
# written in place, and a disagreement between them prints nothing
$(BUILD)/pack_w8: tests/pack_w8.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# the weight address patched into a stream against emitting it that way. The
# patch is only safe while exactly one word depends on that address, and
# nothing but this holds the emitter there
$(BUILD)/patch_waddr: tests/patch_waddr.c src/job.c src/regcmd.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# the fused softmax against the separate one, to the last bit of the half. Two
# copies of one piece of arithmetic is the hazard; that they AGREE is what is
# held, and a tolerance would not hold it
$(BUILD)/softmax_half: tests/softmax_half.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

# the vector fp16 conversion against the per element definition, over all 2^32
# floats. It is header only, so this links nothing: the run and the definition
# both live in charsiu.h precisely so they cannot drift into two conversions
$(BUILD)/pack_f16run: tests/pack_f16run.c include/charsiu.h | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $<

# the output sentinel, driven through all three verdicts. It is a header rule
# for the same reason overlap.h and reusekey.h are: a check that has never
# been seen to fire is not a check, and the hardware will not fire this one
$(BUILD)/sentinel: tests/sentinel.c src/sentinel.h | $(BUILD)
	$(CC) $(CFLAGS) -Isrc -o $@ $<

# the per-output-channel scale table, read back entry by entry. A table off by
# one channel gives every output a plausible wrong magnitude, and this tree's
# own history says a plausible wrong number survives a text check
$(BUILD)/coef_scales: tests/coef_scales.c src/job.c src/regcmd.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# the packer split by groups against the whole buffer it replaces: a byte
# wrong here is a slightly wrong sentence and not a fault
$(BUILD)/axpy8: tests/axpy8.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD)/pack_groups: tests/pack_groups.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

# the group's four buffers are addressed by arithmetic, and arithmetic that
# overlaps two regions returns another op's answer rather than an error
$(BUILD)/fp16_plan: tests/fp16_plan.c src/fp16plan.h src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tests/fp16_plan.c src/regcmd.c src/job.c -lm

$(BUILD)/fp16_regrow: tests/fp16_regrow.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tests/fp16_regrow.c src/regcmd.c src/job.c -lm

# AND THE ADVERSARIAL ARM. fp16_regrow above sweeps a FIXED table into
# freshly zeroed destinations, one step at a time; this one randomises the
# shape, climbs the ladder IN ONE BUFFER with positions appended between
# rungs, poisons the destination tail, puts PROT_NONE pages on both ends, and
# holds nine deliberate mutations of the walk to the same comparison so that
# "all ok" means the comparison can see a bug. CHARSIU_FUZZ_SEED changes the
# draw; the default seed is fixed so a failure reproduces.
$(BUILD)/fp16_regrow_fuzz: tests/fp16_regrow_fuzz.c src/regcmd.c src/job.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tests/fp16_regrow_fuzz.c src/regcmd.c src/job.c -lm

# THE TWO ATTENTION THRESHOLDS, WHICH WERE WRONG TWICE IN ONE DAY AND HAD NO
# TEST. attn_npu_min_for picks 320 or 448 from the model's head counts, and the
# gate that uses it also has to refuse a caller that never said how long the
# prompt is. Neither needs an NPU: the inputs are two head counts and an
# integer. The test includes src/llama.c rather than asking for a shim, so the
# link line is $(LLMNOLL) and the dependency is $(LLM).
# ONE FORK PER CASE. Every knob it moves is cached in a function static on
# first read, so a second case in the same process would read the first one's
# environment.
$(BUILD)/attn_two_thresholds: tests/attn_two_thresholds.c $(LLM) | $(BUILD)
	$(CC) $(CFLAGS) -o $@ tests/attn_two_thresholds.c $(LLMNOLL) -lm -lpthread

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
