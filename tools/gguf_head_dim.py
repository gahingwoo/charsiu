#!/usr/bin/env python3
# Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com>
# SPDX-License-Identifier: GPL-2.0
"""A gguf's head_dim and layer count, which is what decides whether a rule
about head_dim applies to a model.

⚠ WHY THIS EXISTS. attn_npu_want_for() carries a withdrawn rule -- "head_dim
>= 128 is only half the rule" -- and nothing in this tree could say which of
the models on the card were on which side of that line. Llama-3.2-1B, the
model every speed round uses, turns out to be 64: the FURTHEST BELOW the
threshold of anything here. A negative result measured there is consistent
with the rule and does not test it.

  head_dim is attention.key_length when the file states it, and
  embedding_length / attention.head_count when it does not.

  python3 -P tools/gguf_head_dim.py models/*.gguf
"""
import struct, sys
T_U8,T_I8,T_U16,T_I16,T_U32,T_I32,T_F32,T_BOOL,T_STR,T_ARR,T_U64,T_I64,T_F64=range(13)
def rd(f,fmt): n=struct.calcsize(fmt); return struct.unpack(fmt,f.read(n))[0]
def rstr(f): return f.read(rd(f,'<Q')).decode('utf-8','replace')
def rval(f,t):
    if t==T_U8: return rd(f,'<B')
    if t==T_I8: return rd(f,'<b')
    if t==T_U16: return rd(f,'<H')
    if t==T_I16: return rd(f,'<h')
    if t==T_U32: return rd(f,'<I')
    if t==T_I32: return rd(f,'<i')
    if t==T_F32: return rd(f,'<f')
    if t==T_BOOL: return rd(f,'<B')
    if t==T_STR: return rstr(f)
    if t==T_U64: return rd(f,'<Q')
    if t==T_I64: return rd(f,'<q')
    if t==T_F64: return rd(f,'<d')
    if t==T_ARR:
        # ⚠ EVERY ELEMENT MUST BE CONSUMED even when only a few are kept, or
        # the file position is wrong for every key after this one -- which is
        # how a 128k-entry tokeniser array turned into a MemoryError three
        # keys later rather than an error here.
        et=rd(f,'<I'); n=rd(f,'<Q')
        out=[]
        for i in range(n):
            v=rval(f,et)
            if i<4: out.append(v)
        return out if n<=4 else '<%d items>'%n
    raise ValueError(t)
for path in sys.argv[1:]:
    with open(path,'rb') as f:
        assert f.read(4)==b'GGUF'
        rd(f,'<I'); rd(f,'<Q'); nkv=rd(f,'<Q')
        kv={}
        for _ in range(nkv):
            k=rstr(f); t=rd(f,'<I'); kv[k]=rval(f,t)
    g=lambda suf: next((v for k,v in kv.items() if k.endswith(suf)), None)
    emb=g('.embedding_length'); nh=g('.attention.head_count')
    kl=g('.attention.key_length')
    hd = kl if kl else (emb//nh if emb and nh else None)
    print("%-40s head_dim %-5s  (n_embd %s / n_head %s, key_length %s)  layers %s"
          % (path.split('/')[-1], hd, emb, nh, kl, g('.block_count')))
