# charsiu

An open LLM runtime for the RK3576 NPU on a mainline Linux kernel. It drives the
NPU through the mainline `rocket` DRM-accel driver. No vendor library is in the
execution path.

## Against the vendor

ArmSoM CM5-IO, Qwen3-0.6B, the same token ids on both sides. The vendor is
rkllm v1.3.1 on its own kernel (Armbian vendor 6.1, rknpu 0.9.8) with its own
w4a16 model. charsiu runs the NPU at 594 MHz, the mainline clock. The vendor is
shown at 600 MHz and at 950 MHz, its maximum.

```
  time to first token, ms        115 tok  124 tok  128 tok  131 tok  139 tok
  charsiu        594 MHz           405      441      443      459      462
  vendor         600 MHz           556      584      597      682      696
  vendor         950 MHz           510      494      502      564      585

  decode, tok/s, speculation off
  charsiu        594 MHz         27.0 - 28.0
  vendor         600 MHz         18.2 - 18.8
  vendor         950 MHz         23.2 - 23.75

  peak memory    charsiu 0.73 GB, vendor 1.12 GB
```

charsiu is the median of three runs, a79ff74, 2026-11-26; every single run was
ahead of the vendor at 950 MHz. The vendor rows were measured on the same board
in another boot (2026-11-25), and the board drifts about 3% between boots.
Against the vendor at 950 MHz, charsiu takes 10-21% less time to the first
token and decodes 14-21% faster.

Speculative decoding is on by default and gives the same text as with it off.

## Quality

The NPU runs int4 weights, one scale per 1024 inputs. On long.txt this costs
perplexity against llama.cpp's q4_0 on the CPU (Qwen3-0.6B 81.5 vs 43.7,
Llama-3.2-1B 33.3 vs 18.0, Qwen3-1.7B 36.8 vs 21.3). Almost half of the loss is in
the first two layers. `CHARSIU_NPU_INT8_LAYERS=0-1` keeps those two in int8: 76.8,
22.6 and 34.9, for 1-6% more prompt time and 4-10% less decode. It is off by
default because it changes the text.

## Install

```
curl -fsSL https://raw.githubusercontent.com/gahingwoo/charsiu/stable/scripts/charsiu-install.sh | sh
```

On Armbian it is in the Armbian repository:

```
sudo apt install charsiu
charsiu pull
```

On another arm64 Debian or Ubuntu, take the `.deb` from the releases page and
`sudo apt install ./charsiu_*_arm64.deb`.

The installer builds charsiu, checks that the running kernel can drive the NPU,
fetches a model and asks it something. On a kernel without NPU support it
offers Armbian's edge kernel; on Armbian it leaves the kernel alone.
`--dry-run` prints what it would do and changes nothing.

Two channels: `stable` is the runtime, `dev` adds the board probes, the test
corpora and `charsiu_ppl`. `charsiu update dev` or `charsiu update stable`
switches.

## Use

```
charsiu                       a conversation, or the guided setup on a new board
charsiu "what is a bandgap"   ask once
charsiu --image FILE "..."    ask about a picture
charsiu pull | list | rm | show | ps | bench | doctor
charsiu serve [--port N]      an OpenAI and Anthropic compatible endpoint
charsiu_whisper MODEL --transcribe --audio clip.wav
charsiu_clip MODEL --image FILE --text "a cat" "a dog"
```

It loads llama, qwen2, qwen3, gemma3, gemma4, phi3 and smollm3 gguf files.
smollm3 is accepted by the loader but has never been run. Vision comes from
llama.cpp `mmproj` files, speech from whisper.cpp files.

Useful switches: `CHARSIU_NPU=0` runs on the CPU only, `CHARSIU_SPEC=0` turns
speculation off, `CHARSIU_STAGES=1` prints where a token's time goes.

## Requirements

An RK3576 board (tested on the ArmSoM CM5-IO and the Radxa ROCK 4D) and a
kernel with RK3576 support in `rocket`. That support is not in mainline yet; it
is on the list as
[PATCH v18](https://lore.kernel.org/all/cover.1795478392.git.gahing@gahingwoo.com/),
and Armbian's edge kernel carries it. The kernel and Mesa work is in
[linux-rk3576-npu](https://github.com/gahingwoo/linux-rk3576-npu).

RK3588 support is being written on the `rk3588/llm` branch and has not run on a
board.

## Evidence

Every number a paper may cite, with how it was measured, is in
[docs/paper-evidence.md](docs/paper-evidence.md). The lab notebook and the
older planning documents were removed from the tree on 2026-11-26 and are in
the git history.

## Name and licence

Char siu is Cantonese barbecue pork; [kiln](https://github.com/gahingwoo/kiln),
the harness that runs the vendor stack for comparison, is the oven.

GPL-2.0-or-later.
