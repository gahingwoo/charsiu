# Provenance for the vendor-quality comparison

Everything the C18 section and the reproduction section need in order to name
its inputs. Collected 2026-09-11.

## The files

```
  vendor weights   Llama-3.2-1B-Instruct-rk3576-w4a16.rkllm
                   md5  2d3962468e2e7c0d0571157f8c9eae71
                   1300605380 bytes
                   local path /home/parallels/Documents/kiln/model/

  reference        Llama-3.2-1B-Instruct-Q8_0.gguf   (bartowski)
                   md5  0cb5367ee9a555b8fad00db326ba1604
                   1321082528 bytes

  corpus           tests/corpus/long.txt
                   md5  4237c8fc3163a359fc21bde60c7b1d8b
                   scored at -n 300
```

⚠ The .rkllm's download URL is not recorded anywhere in this tree and I cannot
reconstruct it. It has to come from whoever fetched it. Without it the md5 is a
fingerprint of a file nobody else can obtain, which is half a provenance.

## The vendor's quantiser, read out of the file

`w = scale * (q - zero)`, `q` in [-8, +7]: **asymmetric int4 with an integer
zero point, one fp32 scale and one fp32 zero point per OUTPUT ROW.**

Confirmed by arithmetic on the slot offsets rather than by reading a comment:
`blk.14.attn_q` (2048 rows) to `blk.14.attn_k` is 4096 floats, which is 2048
scales plus 2048 zero points; `blk.15.ffn_gate` (8192 rows) to `ffn_up` is
16384. One of each, per row, on both shapes.

`tools/rkllm_scales.py verify` reports **6 of 112 tensors** satisfying
`scale = (max - min) / 15` exactly against this reference. The 43-tensor subset
the comparison uses is the wider band, rho within 5% of 1.

## ⚠⚠ THE GROUP SIZES ARE NOT THE SAME, AND THIS IS NOT IN THE PAPER

One scale per row means the vendor's group is the whole of K.

```
                   scales per 2048-wide row   scales per 8192-wide row
  vendor                    1                          1
  charsiu, group 1024       2                          8
  llama.cpp q4_0            64                       256
```

**charsiu's scales are 2 to 8 times finer than the vendor's on the very
tensors the comparison scores.** So the 2.00 / 2.23 / 2.35 ratio is not "the
same quantiser configuration, different algorithm". It is at least two
differences at once: a different algorithm and a finer group.

That is a larger asymmetry than the bf16-against-Q8_0 one, and it points the
same way: it flatters charsiu. Both belong in "How this could be wrong", and
"asked to quantise the same thing" is not defensible while either stands.

### Measured, and the remedy is worse than the caveat

`chr43g` takes the group from `CHARSIU_CHR_GROUP`. 43 matrices, `-n 300`, on
both passages:

```
                      long.txt   long2.txt
  charsiu group 1024     2.00x      2.16x      <- what the paper reports
  charsiu whole row      3.76x      1.97x      <- the "matched" variant
```

The configuration already reported is the stable one, on two independent
passages. The matched-group variant swings by a factor of two between them and
cannot be quoted.

So the caveat stands and stays a caveat: the groups differ, the paper should
say so, and it should NOT try to fix it by matching them.

⚠ Three readings taken from `long.txt` alone were all wrong, including two I
stated confidently. The finer group does not flatter charsiu (on that passage
group 1024 is worse than the whole row). There is no bad point at 1024
(`long.txt` spikes, `long2.txt` is monotone). And matching the group does not
harden the section. All three came from differences of 1 to 7%, and one passage
of 300 tokens cannot order anything closer than about 10%.

⚠ A third asymmetry runs the OTHER way and no arm removes it: the vendor is
asymmetric with an integer zero point, charsiu here is symmetric absmax. This
tree measures the zero point at about 1.4% at group 1024, so it is the small
one, and it favours the vendor.

## The checker rule, specified

> Every perplexity reported in the vendor-quality section must name a file
> whose md5 appears in the reproduction section.

Per trap 2's method: inject a violation first (change one md5 digit, or add a
perplexity whose file is not listed), confirm the checker FAILS, revert,
confirm it PASSES. A rule that has never failed has not been tested.

⚠ The rule as stated encodes a CONDITION, not a verdict, so it will not go
stale the way `BANNED: beats the vendor` did. Conditions stay true or false;
verdicts change when the measurement behind them changes.
