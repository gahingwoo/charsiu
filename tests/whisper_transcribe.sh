#!/bin/sh
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
#
# The end to end regression: a known clip has to come out as the known words.
#
# ⚠ THIS IS THE ONE TEST THAT WOULD SURVIVE EVERY OTHER ONE PASSING. The mel,
# the encoder and the decoder are each checked against numpy, and a wiring
# mistake BETWEEN them -- a window fed to the encoder before the clamp, the
# prompt built at the wrong positions -- passes all three and produces confident
# English about nothing.
#
#   tests/whisper_transcribe.sh build/charsiu_whisper MODEL.bin jfk.wav
#
# jfk.wav is whisper.cpp's own sample:
#   https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav
set -eu

TOOL=${1:?usage: whisper_transcribe.sh TOOL MODEL.bin AUDIO.wav}
MODEL=${2:?}
AUDIO=${3:?}

WANT="ask not what your country can do for you"
GOT=$("$TOOL" "$MODEL" --transcribe --audio "$AUDIO")

echo "$GOT"
case "$GOT" in
*"$WANT"*) echo "PASS" ;;
*)  echo "FAILED: the transcript does not contain \"$WANT\"" >&2
    exit 1 ;;
esac
