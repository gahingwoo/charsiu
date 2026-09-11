#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0-or-later
"""Rebuild the vendor's model as an f16 gguf, so its QUALITY can be scored.

The speed comparison against the vendor has always had an empty cell beside
it: their runtime's output has never been scored, so neither charsiu row could
be read as better. This fills it, on a desk, with no board and no vendor
install -- their .rkllm carries the weights and they are not encrypted.

WHAT IT WRITES. An f16 gguf that is the reference model with SOME of its 112
matmul weights replaced. Everything else -- embeddings, norms, the output head,
the tokenizer -- is the reference's, so the only thing a perplexity difference
can be is the matrices that were swapped. No quantiser runs at inference in any
arm; the quantisation has already happened, into f16.

    ref        the reference, q8_0 dequantised to f16          the control
    rho1       ONLY the 43 tensors whose scale still satisfies (max-min)/15,
               with no calibration divided out, because there is none
    vendor43   the same 43, through the full path               } compare
    chr43      the same 43, charsiu's int4 at group 1024        } these
    q4043      the same 43, llama.cpp q4_0 at group 32          } three
    vendorL3   layers 3..15 -- 91 matrices -- their norms too   } and the
    chrL3      the same 91, charsiu                             } same
    q40L3      the same 91, q4_0                                } three
    vendorNo1  everything except layer 1 -- 105 matrices
    chrNo1     the same 105, charsiu
    vendorfull all 112, their norms                             see LAYER 1
    noise      THE CONTROL: same per-tensor relative error, unstructured

🔑 THE CALIBRATION IS NOT RECOVERED, IT CANCELS. The vendor folds 1/c into the
RMSNorm ahead of each projection instead of dividing the activation at runtime,
so taking THEIR norms with THEIR weights makes c cancel by construction --
corr(vendor_norm, ref/c) is 0.9905 to 0.9945 where corr(vendor_norm, ref) is
0.78 to 0.89. Recovering c by division instead is what scored 1700.98.

⚠⚠ AND THAT 1701 IS WHY `noise` EXISTS. The median per-tensor weight error of
that file was 18.3%, and charsiu's own int4 at 13.9% scores 41 -- so either
18.3% is simply that expensive, or the reconstruction was wrong in a way a
Frobenius norm does not charge for. A fourth file, the reference plus Gaussian
noise at the SAME per-tensor relative error, scores 32.10. The magnitude is
worth 32 and the structure is worth 1701. Run `noise` beside any new arm.

⛔ LAYER 1 IS EXCLUDED AND THE REASON IS NAMED. Everything except layer 1 reads
32.13; with it, 58.76. It carries the most extreme row gauge in the model --
ffn_up at rho 22.3 against ffn_down at 0.298 -- and a gauge cancels at inference
only if both halves are reconstructed exactly. `blk.1.ffn_down` is not
reconstructed at all: the per-column model leaves 147% of it, so it is KEPT AS
THE REFERENCE and the run says so. The 112-matrix number is not a measurement
of the vendor's quality and must not be quoted as one.

⚠ THE SCRIPT LIVED IN A SCRATCHPAD UNTIL 2026-09-11. The project's most
valuable measurement depended on a file in /tmp for two days.

Usage:
    python3 -P tools/rkllm_rebuild.py <which>

⚠ -P, AND IT IS NOT OPTIONAL. This imports `gguf`, and a stray gguf.py in the
working directory is imported instead -- which has happened in this tree, with
a module that reset the board's USB at import time. -P drops the script's own
directory from sys.path.

Environment:
    CHARSIU_REBUILD_OUT    where to write        (default charsiu/models)
    CHARSIU_RKLLM_REF      the reference to quantise and to carry everything
                           BUT the swapped matrices; its md5 goes in the
                           output filename, see outpath()
    CHARSIU_GGUF_PY        gguf-py checkout      (default llama.cpp-ref/gguf-py)
"""
import sys, os
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.environ.get(
    "CHARSIU_GGUF_PY",
    os.path.join(os.path.dirname(_ROOT), "llama.cpp-ref", "gguf-py")))
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType    # noqa: E402
from gguf.constants import GGUFValueType                          # noqa: E402
sys.path.insert(0, _HERE)
import rkllm_codes as RC                                          # noqa: E402

OUT = os.environ.get("CHARSIU_REBUILD_OUT", os.path.join(_ROOT, "models"))


def outpath(which):
    """⛔ THE ORIGIN IS IN THE FILENAME BECAUSE IT IS PART OF THE ANSWER.

    An arm rebuilt from Q8_0 and the same arm rebuilt from the f16 original
    are different files that answer differently, and the arm name alone does
    not distinguish them. Anything that skips a rebuild when the path already
    exists -- tests/vendor_quality.sh does -- then scores the older origin
    under the newer origin's heading. See rkllm_codes.ref_tag.
    """
    return f"{OUT}/Llama-3.2-1B-{which}-{RC.ref_tag()}-F16.gguf"


def _partpath(which):
    """⛔ A HALF-WRITTEN FILE MUST NOT WEAR THE FINISHED NAME.

    tests/vendor_quality.sh skips the rebuild when the path exists, and
    `exists` is not `complete`: a build killed mid-write on 2026-09-11 left
    1046478848 bytes of a 2.48 GB arm under the final name, and the next round
    accepted it and tried to score it. It failed loudly that time. A truncated
    gguf that still loads would have been scored instead.

    So the writer writes here and os.replace()s into place only after close.
    """
    return outpath(which) + ".part"
S, f32, deq, codes, solve = RC.S, RC.f32, RC.deq, RC.codes, RC.solve


def maps_and_offsets():
    SLOT, a = {}, 122728
    NORM = {}
    for L in range(16):
        for nm, n, _ in S:
            SLOT[f"blk.{L}.{nm}.weight"] = (a, n)
            a += 2 * n
        NORM[f"blk.{L}.attn_norm.weight"] = a
        NORM[f"blk.{L}.ffn_norm.weight"] = a + 2048
        a += 4096
    SLOT["__norm__"] = NORM
    WOFF, a = {}, RC.INT4
    for L in range(16):
        for nm, n, k in S:
            WOFF[f"blk.{L}.{nm}.weight"] = (a, n, k)
            a += n * k // 2
    return SLOT, WOFF


RAW = False


def main():
    global RAW
    r = GGUFReader(RC.REF)
    G = {t.name: t for t in r.tensors}
    SLOT, WOFF = maps_and_offsets()
    NORM = SLOT["__norm__"]
    anchors = {(2048, 2048): "blk.3.attn_q.weight",
               (512, 2048): "blk.3.attn_k.weight",
               (8192, 2048): "blk.6.ffn_gate.weight"}
    maps = {}
    for shape, anch in anchors.items():
        off, rows, kd = WOFF[anch]
        slot, _ = SLOT[anch]
        W = deq(G[anch]); s = f32(slot, rows); z = f32(slot + rows, rows)
        P = np.clip(np.rint(W / s[:, None] + z[:, None]), -8, 7).astype(np.int8)
        maps[shape] = solve(P, codes(off, rows * kd // 2).reshape(rows // 16, -1),
                            min(48, rows // 16))
    rho2, kk2 = maps[(2048, 2048)]
    maps[(2048, 8192)] = (rho2, np.concatenate([kk2, kk2 + 2048]))

    def vendor_raw(name):
        """s(q - z) with the calibration left alone."""
        off, rows, kd = WOFF[name]
        slot, _ = SLOT[name]
        s = f32(slot, rows); z = f32(slot + rows, rows)
        rho, kk = maps[(rows, kd)]
        if kd == 8192:
            C = codes(off, rows * kd // 2).reshape(-1, 65536)
            nb = C.shape[0]
            ri = rho[np.arange(65536) % 512]
            b = np.arange(nb)
            gid = 64 * (b // 128) + (b % 128) % 64
            hid = (b % 128) // 64
            R = (gid[:, None] * 16 + ri[None, :]).reshape(-1)
            K = (kk[None, :] + 4096 * hid[:, None]).reshape(-1)
        else:
            C = codes(off, rows * kd // 2).reshape(rows // 16, -1)
            slots = C.shape[1]
            ri = rho[np.arange(slots) % 512]
            R = (np.arange(rows // 16)[:, None] * 16 + ri[None, :]).reshape(-1)
            K = np.broadcast_to(kk[None, :], (rows // 16, slots)).reshape(-1)
        Q = np.zeros((rows, kd))
        Q[R, K] = C.reshape(-1)
        return (Q - z[:, None]) * s[:, None]

    def vendor_weight(name):
        off, rows, kd = WOFF[name]
        slot, _ = SLOT[name]
        W = deq(G[name]); s = f32(slot, rows); z = f32(slot + rows, rows)
        rho, kk = maps[(rows, kd)]
        if kd == 8192:
            C = codes(off, rows * kd // 2).reshape(-1, 65536)
            nb = C.shape[0]
            ri = rho[np.arange(65536) % 512]
            b = np.arange(nb)
            gid = 64 * (b // 128) + (b % 128) % 64
            hid = (b % 128) // 64
            R = (gid[:, None] * 16 + ri[None, :]).reshape(-1)
            K = (kk[None, :] + 4096 * hid[:, None]).reshape(-1)
        else:
            C = codes(off, rows * kd // 2).reshape(rows // 16, -1)
            slots = C.shape[1]
            ri = rho[np.arange(slots) % 512]
            R = (np.arange(rows // 16)[:, None] * 16 + ri[None, :]).reshape(-1)
            K = np.broadcast_to(kk[None, :], (rows // 16, slots)).reshape(-1)
        Q = np.zeros((rows, kd))
        Q[R, K] = C.reshape(-1)
        V = (Q - z[:, None]) * s[:, None]
        if RAW:
            return V, float((15.0 * s / (W.max(axis=1) - W.min(axis=1))).mean())
        A = np.abs(W)
        M = A > np.percentile(A, 60, axis=0)[None, :]
        Rt = np.where(M, V / np.where(A > 1e-12, W, 1e-12), np.nan)
        c = np.nanmedian(Rt, axis=0)
        c = np.where(np.isfinite(c) & (np.abs(c) > 1e-9), c, 1.0)
        return V / c[None, :], float(
            (15.0 * s / (W.max(axis=1) - W.min(axis=1))).mean())

    which = sys.argv[1]        # ref | vendor | noise | rho1
    RAW = which == "rho1"      # rho1: only the untransformed 43, no c at all
    rng = np.random.default_rng(1)
    arch = r.fields["general.architecture"].parts[-1].tobytes().decode()
    w = GGUFWriter(_partpath(which), arch)
    for key, fld in r.fields.items():
        if key == "GGUF.version" or key.startswith("GGUF."):
            continue
        t = fld.types[0]
        try:
            if t == GGUFValueType.STRING:
                w.add_string(key, fld.parts[-1].tobytes().decode())
            elif t == GGUFValueType.ARRAY:
                sub = fld.types[1]
                if sub == GGUFValueType.STRING:
                    w.add_array(key, [fld.parts[i].tobytes().decode() for i in fld.data])
                else:
                    w.add_array(key, [fld.parts[i].tolist()[0] for i in fld.data])
            else:
                v = fld.parts[-1].tolist()[0]
                w.add_key_value(key, v, t)
        except Exception as e:                 # noqa: BLE001
            print(f"  skip {key}: {e}")
    n = 0
    nn = [0]
    for t in r.tensors:
        name = t.name
        #
        # ⚠⚠ 8 IS Q8_0'S TYPE ID AND THIS USED TO BE THE WHOLE TEST. Handed an
        # f16 reference every tensor fell through to a plain copy and the run
        # reported "0 matrices replaced" -- a complete, plausible, empty
        # result. The reference became an environment variable so that the
        # comparison could be rebuilt from the ORIGINAL weights, which is
        # exactly the file this gate refused.
        #
        # 1 is F16, 0 is F32. What the gate means is "a 2-D weight tensor
        # this script might replace", so it says that instead.
        #
        # ⚠ THE RANK IS PART OF IT. Widening the type alone let the 1-D norms
        # through, which are F32 in an f16 file and were never Q8_0, and the
        # shape assertion below indexes shp[1]. The norms have their own
        # branches further down and must reach them.
        #
        if int(t.tensor_type) in (0, 1, 8) and len(t.shape) >= 2:
            if which == "rho1" and name in WOFF:
                #
                # ONLY the tensors whose scale still satisfies (max-min)/15, and
                # NO calibration factor divided out, because there is none to
                # divide. This is the layout and the codes alone.
                #
                v, rho_t = vendor_weight(name)
                if abs(rho_t - 1.0) <= 0.05:
                    d = v.astype(np.float16); n += 1
                else:
                    d = deq(t).astype(np.float16)
            elif which == "noise" and name in WOFF:
                #
                # THE CONTROL. Same per-tensor relative error as the vendor
                # reconstruction, but unstructured. If this also explodes, the
                # magnitude is the whole story; if it does not, the vendor
                # column of my table is measuring my own mistake.
                #
                ref = deq(t)
                v, _ = vendor_weight(name)
                e = float(np.linalg.norm(ref - v) / np.linalg.norm(ref))
                if e > 0.5:
                    e = 0.0
                g = rng.standard_normal(ref.shape)
                g *= e * np.linalg.norm(ref) / max(np.linalg.norm(g), 1e-30)
                d = (ref + g).astype(np.float16)
                n += 1
            elif which in ("chr43", "q4043", "chr43row",
                           "chr43g") and name in WOFF:
                #
                # The SAME 43 tensors, quantised by us instead. Everything
                # else in the file is the reference, so the perplexity
                # difference against vendor43 is three quantisers on one
                # subset and nothing else.
                #
                off_, rows_, kd_ = WOFF[name]
                slot_, _ = SLOT[name]
                Wr = deq(t)
                sv = f32(slot_, rows_)
                rho_t = float((15.0 * sv
                               / (Wr.max(axis=1) - Wr.min(axis=1))).mean())
                if abs(rho_t - 1.0) <= 0.05:
                    #
                    # ⚠⚠ THE GROUPS ARE NOT THE SAME SIZE AND THE PAPER DOES
                    # NOT SAY SO. The vendor keeps ONE fp32 scale and one
                    # integer zero point per OUTPUT ROW -- confirmed from the
                    # slot offsets, 4096 floats between two 2048-row tensors --
                    # so its group is the whole of K. charsiu at 1024 has two
                    # groups on a 2048-wide row and eight on an 8192-wide one.
                    #
                    # So the 2.00 / 2.23 / 2.35 ladder is not one difference,
                    # it is at least two: a different algorithm AND a finer
                    # group, and the finer group flatters us. `chr43row` makes
                    # the group the whole row, which is the vendor's own, so
                    # the ratio can be read with that difference removed.
                    #
                    # ⚠ A third difference runs the OTHER way and is not
                    # removed by this: the vendor is asymmetric with an integer
                    # zero point and charsiu here is symmetric absmax. The zero
                    # point is worth about 1.4% at group 1024 by this tree's
                    # own measurement, so it is the small one.
                    #
                    #
                    # ⚠ AND THE THREE ARMS ARE ONE QUANTISER AT THREE GROUPS.
                    # llama.cpp's q4_0 is `d = max / -8` with max the signed
                    # value at max|x| (ggml-quants.c:132); charsiu's is
                    # `d = vmax / -8.0f` (npuquant.c:606); this is
                    # `dd = vm / -8.0`. Identical. So "charsiu against q4_0"
                    # compares GROUP SIZES, not algorithms, and only the
                    # vendor's arm is a different quantiser.
                    #
                    if which == "chr43row":
                        g = Wr.shape[1]
                    elif which == "chr43g":
                        g = int(os.environ.get("CHARSIU_CHR_GROUP", "1024"))
                    else:
                        g = 1024 if which == "chr43" else 32
                    g = min(g, Wr.shape[1])
                    ww = Wr.reshape(Wr.shape[0], -1, g)
                    i = np.abs(ww).argmax(axis=2)
                    vm = np.take_along_axis(ww, i[:, :, None], axis=2)
                    dd = np.where(vm == 0, 1.0, vm / -8.0)
                    d = (np.clip(np.rint(ww / dd), -8, 7) * dd
                         ).reshape(Wr.shape).astype(np.float16)
                    n += 1
                else:
                    d = Wr.astype(np.float16)
            elif which.startswith("i8L") and name in WOFF:
                #
                # The other route to the same place: eight bits on the early
                # layers instead of a finer four-bit group.  charsiu's int8 is
                # one scale a row, amax/127 -- see npuquant, where grp becomes
                # k for anything that is not four bits.  It costs BYTES where
                # the finer group costs READ BACK, so pricing both lets the
                # board round choose rather than guess.
                #
                E = int(which[3:])
                L = int(name.split(".")[1])
                Wr = deq(t)
                if L < E:
                    a = np.abs(Wr).max(axis=1, keepdims=True)
                    dd = np.where(a == 0, 1.0, a / 127.0)
                    d = (np.clip(np.rint(Wr / dd), -127, 127) * dd
                         ).astype(np.float16)
                else:
                    g = min(1024, Wr.shape[1])
                    ww = Wr.reshape(Wr.shape[0], -1, g)
                    i = np.abs(ww).argmax(axis=2)
                    vm = np.take_along_axis(ww, i[:, :, None], axis=2)
                    q = np.where(vm == 0, 1.0, vm / -8.0)
                    d = (np.clip(np.rint(ww / q), -8, 7) * q
                         ).reshape(Wr.shape).astype(np.float16)
                n += 1
            elif which.startswith("mix") and name in WOFF:
                #
                # "mixG_H_early": group G on layers 0..H-1, group H... no --
                # spelled mixEARLY_GFINE_GCOARSE, e.g. mix2_128_1024 means
                # layers 0 and 1 at group 128 and the rest at 1024.  The point
                # is to price a finer group WHERE IT MATTERS before building
                # any machinery for it: layers 0 and 1 carry 44% of the damage.
                #
                # "mix<EARLY>_<FINE>_<COARSE>": mix2_128_1024 is layers 0 and
                # 1 at group 128 and the rest at 1024.  ⚠ split("_") on that
                # gives THREE parts, not four -- the prefix and the count are
                # one token -- and the first version unpacked four, which
                # produced five blank arms and no error anyone saw.
                nearly, gfine, gcoarse = which[3:].split("_")
                L = int(name.split(".")[1])
                Wr = deq(t)
                g = int(gfine) if L < int(nearly) else int(gcoarse)
                g = min(g, Wr.shape[1])
                ww = Wr.reshape(Wr.shape[0], -1, g)
                i = np.abs(ww).argmax(axis=2)
                vm = np.take_along_axis(ww, i[:, :, None], axis=2)
                dd = np.where(vm == 0, 1.0, vm / -8.0)
                d = (np.clip(np.rint(ww / dd), -8, 7) * dd
                     ).reshape(Wr.shape).astype(np.float16)
                n += 1
            elif which in ("chrL3", "q40L3") and name in WOFF:
                # the SAME 91 matrices, quantised by us, reference norms
                L = int(name.split(".")[1])
                Wr = deq(t)
                if L >= 3:
                    g = min(1024 if which == "chrL3" else 32, Wr.shape[1])
                    ww = Wr.reshape(Wr.shape[0], -1, g)
                    i = np.abs(ww).argmax(axis=2)
                    vm = np.take_along_axis(ww, i[:, :, None], axis=2)
                    dd = np.where(vm == 0, 1.0, vm / -8.0)
                    d = (np.clip(np.rint(ww / dd), -8, 7) * dd
                         ).reshape(Wr.shape).astype(np.float16)
                    n += 1
                else:
                    d = Wr.astype(np.float16)
            elif which == "chrNo1" and name in WOFF:
                # charsiu's quantiser on the SAME 105 matrices, so the step
                # from 91 to 105 can be attributed to the layers rather than
                # to the reconstruction.
                L = int(name.split(".")[1])
                Wr = deq(t)
                if L != 1:
                    g = min(1024, Wr.shape[1])
                    ww = Wr.reshape(Wr.shape[0], -1, g)
                    i = np.abs(ww).argmax(axis=2)
                    vm = np.take_along_axis(ww, i[:, :, None], axis=2)
                    dd = np.where(vm == 0, 1.0, vm / -8.0)
                    d = (np.clip(np.rint(ww / dd), -8, 7) * dd
                         ).reshape(Wr.shape).astype(np.float16)
                    n += 1
                else:
                    d = Wr.astype(np.float16)
            elif which in ("l1ffn", "l1attn") and name in WOFF:
                # layer 1, one half at a time.  The gauge lives in the FFN
                # (ffn_up's rows undone by ffn_down's columns), so if the
                # damage follows it the mechanism is named.
                L = int(name.split(".")[1])
                part = name.split(".")[2]
                isffn = part.startswith("ffn")
                take = (L == 1 and (isffn if which == "l1ffn" else not isffn))
                d = (vendor_raw(name) if take else deq(t)).astype(np.float16)
                n += take
            elif which.startswith("chronly") and name in WOFF:
                want = int(which[7:])
                L = int(name.split(".")[1])
                Wr = deq(t)
                if L == want:
                    g = min(1024, Wr.shape[1])
                    ww = Wr.reshape(Wr.shape[0], -1, g)
                    i = np.abs(ww).argmax(axis=2)
                    vm = np.take_along_axis(ww, i[:, :, None], axis=2)
                    dd = np.where(vm == 0, 1.0, vm / -8.0)
                    d = (np.clip(np.rint(ww / dd), -8, 7) * dd
                         ).reshape(Wr.shape).astype(np.float16)
                    n += 1
                else:
                    d = Wr.astype(np.float16)
            elif which.startswith("only") and name in WOFF:
                want = int(which[4:])
                L = int(name.split(".")[1])
                if L == want:
                    d = vendor_raw(name).astype(np.float16); n += 1
                else:
                    d = deq(t).astype(np.float16)
            elif which == "vendorNo1" and name in WOFF:
                # everything of theirs EXCEPT layer 1, the one whose row gauge
                # is extreme.  If this lands near the 91-matrix 26.61 then that
                # single layer owns the 58.76.
                L = int(name.split(".")[1])
                if L != 1:
                    d = vendor_raw(name).astype(np.float16); n += 1
                else:
                    d = deq(t).astype(np.float16)
            elif which == "vendorL3" and name in WOFF:
                #
                # Layers 3..15 only. Layers 0 to 2 carry the extreme gauges
                # (blk.1.ffn_up at rho 22.3 against ffn_down at 0.298), so if
                # the full-model 58.76 is theirs this lands near the 43-matrix
                # 22.10 and if it is not, it does not.
                #
                L = int(name.split(".")[1])
                if L >= 3:
                    d = vendor_raw(name).astype(np.float16); n += 1
                else:
                    d = deq(t).astype(np.float16)
            elif which == "vendorfull" and name in WOFF:
                d = vendor_raw(name).astype(np.float16)
                n += 1
            elif which == "vendor43" and name in WOFF:
                #
                # ONLY the tensors the vendor did not transform, and NO
                # calibration division: rho within 5% of 1 means c is 1, so
                # this is their layout and their codes with nothing of mine
                # on top. If the full "vendor" file explodes and this one does
                # not, the explosion is my recovered c, not their quantiser.
                #
                off_, rows_, kd_ = WOFF[name]
                slot_, _ = SLOT[name]
                Wr = deq(t)
                sv = f32(slot_, rows_)
                rho_t = float((15.0 * sv
                               / (Wr.max(axis=1) - Wr.min(axis=1))).mean())
                if abs(rho_t - 1.0) <= 0.05:
                    d = vendor_raw(name).astype(np.float16)
                    n += 1
                else:
                    d = Wr.astype(np.float16)
            elif which == "vendor" and name in WOFF:
                v, _ = vendor_weight(name)
                ref = deq(t)
                e = float(np.linalg.norm(ref - v) / np.linalg.norm(ref))
                #
                # ⚠ ONE TENSOR IS NOT RECOVERED. blk.1.ffn_down has rho 0.298,
                # the only value in the whole table well below 0.8, and the
                # per-column model leaves 147% of it -- a reconstruction with
                # no relation to its input. It is kept as the reference rather
                # than shipped into the comparison, and the count says so.
                #
                if e > 0.5:
                    print(f"  KEEPING REFERENCE for {name}: {e*100:.1f}%")
                    d = ref.astype(np.float16)
                else:
                    d = v.astype(np.float16)
                    n += 1
            else:
                d = deq(t).astype(np.float16)
            shp = tuple(int(x) for x in t.shape)
            assert d.shape == (shp[1], shp[0]), (name, d.shape, shp)
        elif which in ("l1ffn", "l1attn") and name in NORM:
            L = int(name.split(".")[1])
            isffn = "ffn_norm" in name
            take = (L == 1 and (isffn if which == "l1ffn" else not isffn))
            d = (f32(NORM[name], 2048).astype(np.float32) if take
                 else np.array(t.data))
            nn[0] += take
        elif which.startswith("only") and name in NORM:
            want = int(which[4:])
            L = int(name.split(".")[1])
            d = (f32(NORM[name], 2048).astype(np.float32) if L == want
                 else np.array(t.data))
            nn[0] += L == want
        elif which == "vendorNo1" and name in NORM:
            L = int(name.split(".")[1])
            d = (f32(NORM[name], 2048).astype(np.float32) if L != 1
                 else np.array(t.data))
            nn[0] += L != 1
        elif which == "vendorL3" and name in NORM:
            L = int(name.split(".")[1])
            d = (f32(NORM[name], 2048).astype(np.float32) if L >= 3
                 else np.array(t.data))
            nn[0] += L >= 3
        elif which == "vendorfull" and name in NORM:
            #
            # 🔑 c IS STORED, AS THE FOLDED NORM. The vendor divides the
            # RMSNorm weight by its per-channel factor instead of dividing the
            # activation at runtime, so taking THEIR norms with THEIR weights
            # makes c cancel and nothing has to be recovered. Using the
            # reference norms with their weights is what scored 1701.
            #
            d = f32(NORM[name], 2048).astype(np.float32)
            nn[0] += 1
        else:
            d = np.array(t.data)
        w.add_tensor(name, d)
    print(f"{which}: {n} matrices and {nn[0]} norms replaced")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(_partpath(which), outpath(which))
    p = outpath(which)
    print(f"  wrote {p}  {os.path.getsize(p)/2**20:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
