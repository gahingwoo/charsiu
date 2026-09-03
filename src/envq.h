/* SPDX-License-Identifier: GPL-2.0-or-later */
/* Copyright (c) 2026 Jiaxing Hu <gahing@gahingwoo.com> */
/*
 * getenv, once per name.
 *
 * ⚠ THE REGISTER STREAM IS EMITTED PER CALL ON THE BATCHED PATH, and
 * charsiu_emit_job asked the environment fifteen times per emission --
 * CHARSIU_DPU_4038, CHARSIU_W4_BITPAT, CHARSIU_CNA_1098 and the rest, every
 * one a linear scan of environ with a strcmp per entry. Phase 20 priced the
 * batched call's "pack" at 110 us a call for four rows of activation; a
 * dozen environment scans per slot, two to six slots a call, is a real share
 * of that. The environment does not change while the process runs, so each
 * name is looked up once and the answer kept, keyed on the literal's address,
 * which is what every call site passes.
 */
#ifndef CHARSIU_ENVQ_H
#define CHARSIU_ENVQ_H
#include <stdlib.h>

#define ENVQ_SLOTS 64

static inline const char *envq(const char *name)
{
	static const char *keys[ENVQ_SLOTS];
	static const char *vals[ENVQ_SLOTS];
	static int n;

	for (int i = 0; i < n; i++)
		if (keys[i] == name)
			return vals[i];
	if (n < ENVQ_SLOTS) {
		keys[n] = name;
		vals[n] = getenv(name);
		return vals[n++];
	}
	return getenv(name);   /* the table is full; correct, only slower */
}
#endif
