# Evidence pack, 2026-09-11

Every number a paper could quote from this project, with the protocol that
produced it and the sentence that has to travel beside it. Assembled after five
board rounds in one session on one ROCK 4D.

Tree state: `dev 150e592`. Board: Armbian, kernel `7.2.0-rc5-next-20260730+`,
GCC 15.2.0, glibc 2.43. Desk: aarch64 VM, GCC 13.3.0, glibc 2.39.

---

## 0. The three things that must be said beside the numbers

**(a) The vendor column is a CITATION, not an arm.** It is copied from
`airockchip/rknn-llm/main/benchmark.md`, fetched 2026-08-28, and their header
says the figures were "collected based on the maximum CPU and NPU frequencies
of each platform". Nothing in this project runs their runtime. It has no N, no
spread and no method beyond that sentence. **A margin over it cannot be
"inside the noise" in either direction, because it has none.** Every comparison
below is our median of seven with its range against their published point.

**(b) A perplexity needs a model, a corpus AND a file.** charsiu re-quantises
whatever it loads, so the source format is inside every quality number. The
same Llama-3.2-1B reads 41.37 or 34.24 or 28.71 depending on which file and
which group. Every figure here names its md5.

**(c) One board.** Everything hardware is a single ROCK 4D. That separates
voltage from clock and it does not separate this board from the part, so a
leakage bin cannot be ruled out. The ArmSoM CM5 is RK3588S and cannot test
this rail at all.

---

## 1. Speed against the vendor

One boot, `performance` governor, `board_record.sh` REPEAT=7, median of seven
with every reading kept. Binary pins its calling thread to the fast cluster
(a58086c). Their protocol: 128-token prompt, 64 new tokens.

```
              decode t/s                      TTFT ms
            ours (median)   spread   theirs      ours (median)  range        theirs
  Qwen3 0.6B      26.26      1.9%    24.85           613      601..623      468.61
  TinyLLAMA       22.79      1.2%    19.71           890      876..903      543.68
  Phi3 3.8B        7.03      0.1%     6.58          2987     2942..3013    1829.12
  Gemma4 E2B       9.25      1.7%     9.23          2222     2193..2245    1219.25
```

**Decode: ahead on all four** — +5.7%, +15.6%, +6.8%, +0.2%.
**TTFT: behind on all four** — 1.31x, 1.64x, 1.63x, 1.82x theirs.

⚠ **The prompt is where this runtime is still losing**, and it is the half the
vendor spends 3328 M=1 dispatches on. Saying only the decode half would be
choosing a column.

⚠ TTFT and their TTFT are not the same quantity: theirs is time to the first
token, ours is the prompt's forward passes, so the first token's own step is in
theirs and not in ours — one token's worth in our favour.

### 1b. And the decode margin is younger than the round

Before `a58086c` the decode was **bimodal** and the table reported best-of-N:

```
  Qwen3      20.76 24.25 20.61 20.27 26.06 20.81 20.54     best 26.06, median 20.76
  TinyLLAMA  19.11 19.41 22.83 21.94 19.24 19.15 20.08     best 22.83, median 19.41
  Phi3        7.04  5.96  5.98  5.96  5.96  5.96  7.04     best  7.04, median  5.96
  Gemma4      7.25  7.34  7.28  9.33  7.28  7.32  7.28     best  9.33, median  7.28
```

Four threads (`-t 4`) on a 4xA72 + 4xA53 part: when they all landed on the A72s
it was fast. `taskset -c 0-7` behaved exactly like no taskset, so it was never
about which cores were permitted but about where the scheduler put them.

```
  default median  20.92 -> 26.25   +25.5%       spread 28.5% -> 2.3%
```

🔑 **The new medians are the old best-of-seven to within 1%**, so the claim on
record was the right number reached the wrong way: the high mode was the
machine's real capability, and best-of-N was reporting something the runtime
could do but would not do reliably. **The paper should say the margin is
+5.7%/+15.6% from a median, and that this required pinning; it should not
quote a best-of-N.**

### 1c. The pinning default is safe across every architecture

`a58086c` changes a default that touches every workload on every model, and
what had been checked was one model's decode text. `board_text_all.sh` compares
each model's BATCHED prompt against its own token loop, on the hardware:

```
  Phi-3.5-mini   Qwen2.5-1.5B   Qwen3-0.6B   SmolLM2-1.7B   SmolLM2-135M
  gemma-3-1b     gemma-4-E2B    tinyllama-1.1b   Llama-3.2-1B

  9 models compared, 0 differing -- every one "prompt batched, text identical"
```

🔑 **Every row says `prompt batched`, not `prompt a token`.** A model that
refuses to batch would report "text identical" meaning only that the token loop
agrees with itself; none did. This is the check that caught gemma4 emitting
"31 32 1 2 3" on the card in 2026-08-30 after six architectures had passed on a
desktop.

---

## 2. Quality against the vendor's own int4 — the empty cell, filled

The comparison nothing in the literature has: the vendor's stored weights,
scored. No board and no vendor install — `tools/rkllm_rebuild.py` reads the
`.rkllm`, and `tests/vendor_quality.sh` scores it.

All arms are f16 files differing only in the swapped matrices, from the **Q8_0**
source, `tests/corpus/long.txt` at `-n 300`, no quantiser at inference:

```
  matrices                   ref     vendor   charsiu    q4_0    vendor  charsiu  ratio
  43  (rho = 1)          17.8719   20.2531   19.0597  17.9111  +13.32%   +6.65%  2.00x
  91  (layers 3..15)     17.8719   25.1141   21.1186  18.0910  +40.52%  +18.17%  2.23x
  105 (all but layer 1)  17.8719   29.9056   22.9824       --  +67.33%  +28.60%  2.35x
```

**The vendor's four-bit excess is 2.0 to 2.4x charsiu's**, on three nested
subsets, monotone.

### What makes it evidence rather than a reconstruction of mine

- the weight layout is **solved and held out**: fitted on blocks 0..47, scored
  on rows 768..2047, 99.72% of codes away from a rounding boundary, residuals
  ±1 which is the boundary signature
- the calibration is **not recovered, it cancels**: the vendor folds 1/c into
  the RMSNorm ahead of each projection, `corr(vendor_norm, ref/c)` = 0.9905 to
  0.9945 against 0.78 to 0.89 for `corr(vendor_norm, ref)`
- **the noise control**: unstructured error at the same per-tensor magnitude
  costs +60.00% where their actual quantisation costs +13.32%. The vendor row
  sits four and a half times further from noise than from the reference
- an earlier version scored **1700.98** by recovering c through division; the
  same magnitude as Gaussian noise scored 32.10. That is how the fault was
  found, and it is why the noise arm is in the harness rather than beside it

⚠ **Layer 1 is excluded and the reason is named.** All 112 matrices read 58.76
against 32.13 for the same set minus layer 1; `blk.1` carries the most extreme
row gauge in the model and `blk.1.ffn_down` is not reconstructed at all. That
number measures the reconstruction, not their quality, and is not quoted.

⚠ **The unrecorded protocol gave 1.65 / 1.71 / 1.81 for the same three sets.**
Same shape, consistently lower. Neither ladder is quotable without its corpus
and length; this one has them.

---

## 3. Quality of charsiu's own quantiser

CPU reference (`CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1`) at the group the board
runs, `Llama-3.2-1B-Instruct-Q4_0.gguf` md5 `48ff0243…`, 773025920 bytes,
`tests/corpus/long.txt` `-n 300`:

```
  int4, group 1024        34.2425
  + AWQ alpha 0.20        23.6746     -30.9%
  int8                    18.3604
```

🔑 **Board and desk are bit-identical on all three**, and the calibration pass
writes the same 2647768 bytes — across two compilers, two glibcs, two kernels
and two thread counts. Perplexity survives all of that.

⚠ **It does not survive a different file.** `bartowski/Llama-3.2-1B-Instruct-GGUF`
has been re-uploaded; this project's image carries the older copy:

```
                                     md5        bytes      a row   g1024
  in the rootfs-overlay          c82c0340   773025824   41.5289  33.8071
  what Hugging Face serves now   48ff0243   773025920   41.3739  34.2425
```

**Every quality figure recorded in this tree before 2026-09-11 is the first
row.** A reader reproducing today gets the second. Both are named by md5 in
`tests/corpus/README.md`; the file of record going forward is the one a reader
will actually get.

⚠ And "board and host agree to 0.3%" in the earlier AWQ round was two
different files landing near each other. It is not evidence of anything.

---

## 4. The output head, and why 12.40 and 12.59 are not a disagreement

`tests/prefill_control.sh`, batched -> control -> batched on one binary:

```
  control   65 tok / 4304 ms   66.22 ms a token   15.10 tok/s
  batched   65 tok / 3498 ms   53.82 ms a token   18.59 tok/s  (18.63, 18.54)
  saved                        12.40 ms a token
```

The head runs once instead of 65 times, so the saving is `H * 64/65` where H is
the head's own cost, 12.59 ms:

```
  12.59 * 64/65 = 12.396 ms   predicted
                  12.40  ms   measured
```

and independently, Llama-3.2-1B's head is 128256 x 2048 = 131.3 MB at int4:

```
  131.3 MB / 12.59 ms = 10.43 GB/s   against the 10.58 GB/s the NPU summary reports
```

**1.4% apart, from a shape and a rate that never saw the stopwatch.** This is
the strongest triangulation in the project and it is one measurement plus two
independent predictions, not two estimates of one thing.

---

## 5. Bandwidth figures — which are quotable

```
  5.2 GB/s   the read back        measured directly    quotable
  4.7 GB/s   the activation pack  measured directly    quotable
  11.9 GB/s  what 8 threads reach measured directly    quotable
  10.58 GB/s weight rate, NPU summary                  quotable (see §4)
  --------------------------------------------------------------------
  11.7 GB/s  npu_prep_cost cache walk at 65536 bytes   real, but it is BUFFER
                                                       MAINTENANCE, not weights
  9.9 GB/s   ⛔ NOT A BANDWIDTH. gemma4's q/k/v stage from gguf shapes; the
             entry it comes from exists to argue such a figure is not a roof --
             gate+up reaches 16.8 in the same table, the biggest stage being
             the fastest. Quoting it cites a number derived to refute it.
```

---

## 6. Variability, and what one passage can order

- **A single reading cannot see a change worth less than ~25%.** TinyLLAMA has
  read 12.64 and 17.39 tok/s on the same build minutes apart.
- **One passage of 300 tokens resolves about 10% of perplexity.** A sweep of
  AWQ's exponent on Qwen3 came back non-monotone at that length on the
  evaluation corpus while `charsiu_ppl` is deterministic — so that is the
  corpus's own sampling. `tests/corpus/long2.txt` is the second opinion.
- **gemma4's TTFT**: 46% spread at whatever governor the board had, 9.9% at
  `performance`, **4.4%** pinned. Two clusters at the middle step, none at the
  last. It was the same scheduling lottery.

---

## 7. Reproduction

```sh
git clone <charsiu> && cd charsiu && git checkout 150e592 && make
# the model of record
curl -L -o models/Llama-3.2-1B-Instruct-Q4_0.gguf \
  https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_0.gguf
md5sum models/Llama-3.2-1B-Instruct-Q4_0.gguf   # 48ff0243978606fdba19d899b77802fc

# quality, the CPU reference at the group the board runs
CHARSIU_NPU=0 CHARSIU_NPU_QUANT=1 CHARSIU_NPU_W4V=1 \
  CHARSIU_NPU_KMAX=1024 CHARSIU_NPU_W4_GROUP=1024 \
  build/charsiu_ppl models/Llama-3.2-1B-Instruct-Q4_0.gguf \
                    tests/corpus/long.txt -n 300          # 34.2425

# the vendor's own weights, scored (needs the .rkllm and a Q8_0 reference)
sh tests/vendor_quality.sh 43

# the board round of record: speed, quality, the gemma4 sweep, one boot
sh tests/board_record.sh
```

`tests/corpus_fixed.sh` locks the corpora by md5 and runs in `make test`;
`board_record.sh` stamps the model's md5, the binary's md5, the thread count,
the governor, the NPU rail, both CPU clusters' clock, and the boot id at both
ends — and refuses to be read as one round if the boot id moved.

---

## 8. What is NOT supported

- **Any claim about the vendor's runtime speed measured here.** It has never
  been run in this project. §1's right column is a citation.
- **Anything about a second RK3576.** One board.
- **The 112-matrix vendor rebuild (58.76).** It measures the reconstruction.
- **Cross-machine quality comparisons made before 2026-09-11.** They compared
  two different files.
- **`CHARSIU_NPU_INT8_LAYERS` on hardware.** `npu_mixed_test` shows one open
  device alternates w8a8 and w4a16 correctly (0 of 18 dispatches wrong over
  eight alternations both ways) at K=256 N=64, which says the per-tensor width
  refactor is justified. It does not say the knob works at the scale a real
  model dispatches at, because the refactor is not written.
